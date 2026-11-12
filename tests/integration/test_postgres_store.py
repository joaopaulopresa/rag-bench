"""Search on a tiny index with exact expected results, against a real PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ragbench.components.postgres import Database
from ragbench.components.stores import PostgresStore
from ragbench.core.errors import StoreError
from ragbench.core.resources import Resources
from ragbench.core.types import Chunk, Question, RetrievedChunk, Span

pytestmark = pytest.mark.integration

# Four chunks on two axes: (pets, money). The vectors make the dense ranking obvious.
CHUNKS = [
    Chunk("pets.txt", Span(0, 30), "Cats purr when they are happy."),
    Chunk("pets.txt", Span(31, 62), "Dogs bark at the postman daily."),
    Chunk("tax.txt", Span(0, 34), "Income tax is due in April each year."),
    Chunk("tax.txt", Span(35, 70), "The cat owner pays tax on pet income."),
]
VECTORS = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.8]]
INDEX = "tiny"


@pytest.fixture
async def store(database: Database, resources: Resources) -> AsyncIterator[PostgresStore]:
    store = PostgresStore(resources, database.url, database.schema)
    await store.create_index(INDEX, dimensions=2)
    await store.add_chunks(INDEX, CHUNKS, VECTORS)
    await store.finish_index(INDEX)
    yield store
    for shared in resources.shared_objects():
        await shared.aclose()  # type: ignore[attr-defined]


def spans(results: list[RetrievedChunk]) -> list[tuple[str, int]]:
    return [(result.document_id, result.span.start) for result in results]


async def test_dense_search_ranks_by_cosine_similarity(store: PostgresStore) -> None:
    results = await store.search(INDEX, mode="dense", query="", vector=[1.0, 0.0], top_k=3)
    assert spans(results) == [("pets.txt", 0), ("pets.txt", 31), ("tax.txt", 35)]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1.0)
    assert results[2].score == pytest.approx(0.6)
    assert results[0].text == "Cats purr when they are happy."


async def test_lexical_search_matches_any_query_word_with_stemming(store: PostgresStore) -> None:
    results = await store.search(
        INDEX, mode="lexical", query="Which taxes do cats pay?", vector=None, top_k=10
    )
    # "taxes" matches "tax", "cats" matches "cat"/"Cats", "pay" matches "pays".
    assert spans(results)[0] == ("tax.txt", 35)
    assert set(spans(results)) == {("tax.txt", 35), ("tax.txt", 0), ("pets.txt", 0)}


async def test_lexical_search_without_usable_words_returns_nothing(store: PostgresStore) -> None:
    assert await store.search(INDEX, mode="lexical", query="the of and", vector=None, top_k=5) == []


async def test_hybrid_search_fuses_both_rankings_with_rrf(store: PostgresStore) -> None:
    results = await store.search(
        INDEX, mode="hybrid", query="income tax", vector=[1.0, 0.0], top_k=4
    )
    # dense ranking: cats(1) dogs(2) cat-owner(3) income-tax(4)
    # lexical ranking for "income tax": only the two tax chunks match, in some order (a, b)
    by_start = {(r.document_id, r.span.start): r.score for r in results}
    lexical = await store.search(INDEX, mode="lexical", query="income tax", vector=None, top_k=4)
    lexical_rank = {(r.document_id, r.span.start): r.rank for r in lexical}
    dense_rank = {("pets.txt", 0): 1, ("pets.txt", 31): 2, ("tax.txt", 35): 3, ("tax.txt", 0): 4}
    for key, score in by_start.items():
        expected = 1 / (60 + dense_rank[key])
        if key in lexical_rank:
            expected += 1 / (60 + lexical_rank[key])
        assert score == pytest.approx(expected)
    assert [r.score for r in results] == sorted(by_start.values(), reverse=True)
    assert {spans(results)[0], spans(results)[1]} == {("tax.txt", 0), ("tax.txt", 35)}


async def test_children_of_one_parent_are_returned_once(
    database: Database, resources: Resources
) -> None:
    store = PostgresStore(resources, database.url, database.schema)
    parent = Span(0, 100)
    children = [
        Chunk("doc", Span(0, 10), "alpha", parent=parent, parent_text="the whole section"),
        Chunk("doc", Span(11, 20), "alpha beta", parent=parent, parent_text="the whole section"),
        Chunk("doc", Span(200, 210), "gamma"),
    ]
    await store.create_index("family", dimensions=None)
    await store.add_chunks("family", children, None)
    await store.finish_index("family")

    results = await store.search("family", mode="lexical", query="alpha", vector=None, top_k=5)

    assert [(r.span, r.text) for r in results] == [(parent, "the whole section")]
    question = Question("q", "?", "doc", (Span(5, 8),))
    assert await store.relevant_keys("family", question) == frozenset({"doc:0-100"})
    assert await store.count("family") == 3
    for shared in resources.shared_objects():
        await shared.aclose()  # type: ignore[attr-defined]


async def test_relevant_keys_are_the_passages_that_overlap_gold(store: PostgresStore) -> None:
    question = Question("q", "?", "tax.txt", (Span(30, 36),))
    assert await store.relevant_keys(INDEX, question) == frozenset(
        {"tax.txt:0-34", "tax.txt:35-70"}
    )
    other = Question("q", "?", "pets.txt", (Span(30, 31),))
    assert await store.relevant_keys(INDEX, other) == frozenset()


async def test_index_lifecycle(store: PostgresStore) -> None:
    assert await store.has_index(INDEX)
    assert not await store.has_index("missing")
    await store.create_index("half", dimensions=2)
    assert not await store.has_index("half"), "an index is not usable until it is finished"
    await store.drop_index(INDEX)
    assert not await store.has_index(INDEX)
    with pytest.raises(StoreError, match="does not exist"):
        await store.search(INDEX, mode="lexical", query="x", vector=None, top_k=1)


async def test_misuse_is_reported(store: PostgresStore) -> None:
    with pytest.raises(StoreError, match="must have 2 dimensions"):
        await store.add_chunks(INDEX, CHUNKS[:1], [[1.0, 0.0, 0.0]])
    with pytest.raises(StoreError, match="1 chunks, 2 vectors"):
        await store.add_chunks(INDEX, CHUNKS[:1], VECTORS[:2])
    with pytest.raises(StoreError, match="needs a dense index and a vector"):
        await store.search(INDEX, mode="dense", query="x", vector=None, top_k=1)


async def test_wide_vectors_use_a_half_precision_index(
    database: Database, resources: Resources
) -> None:
    store = PostgresStore(resources, database.url, database.schema)
    wide = [[1.0] + [0.0] * 3071, [0.0, 1.0] + [0.0] * 3070]
    await store.create_index("wide", dimensions=3072)
    await store.add_chunks("wide", CHUNKS[:2], wide)
    await store.finish_index("wide")
    results = await store.search("wide", mode="dense", query="", vector=wide[1], top_k=1)
    assert spans(results) == [("pets.txt", 31)]
    for shared in resources.shared_objects():
        await shared.aclose()  # type: ignore[attr-defined]
