"""The LangGraph pipeline end to end on a tiny corpus, with offline components."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ragbench.components.postgres import Database
from ragbench.components.stores import PostgresStore
from ragbench.core.config import PipelineConfig
from ragbench.core.errors import ConfigError
from ragbench.core.resources import Resources
from ragbench.core.types import Document
from ragbench.pipeline import IndexRetriever, RerankingRetriever, build_pipeline
from ragbench.pipeline.prompts import build_answer_messages, cited_passages

pytestmark = pytest.mark.integration

CONTRACT = (Path(__file__).parent.parent / "fixtures" / "contract.txt").read_text()
DOCUMENTS = [
    Document(id="nda.txt", text=CONTRACT, title="Acme NDA"),
    Document(
        id="lease.txt",
        title="Office lease",
        text="1. Rent\n\nThe Tenant shall pay rent of 4,000 euros on the first day of each month.\n\n"
        "2. Deposit\n\nThe Tenant shall pay a deposit equal to three months of rent.",
    ),
]


@pytest.fixture
async def store(database: Database, resources: Resources) -> AsyncIterator[PostgresStore]:
    yield PostgresStore(resources, database.url, database.schema)
    for shared in resources.shared_objects():
        await shared.aclose()  # type: ignore[attr-defined]


def config(**changes: object) -> PipelineConfig:
    base: dict[str, object] = {
        "chunker": {"name": "recursive", "params": {"max_tokens": 30, "tokenizer": "words"}},
        "embedder": {"name": "hashing", "params": {"dimensions": 128}},
        "search_mode": "hybrid",
        "generator": {"name": "extractive"},
        "top_k": 3,
    }
    return PipelineConfig.model_validate({**base, **changes})


async def test_answer_runs_every_node_and_checks_citations(
    store: PostgresStore, resources: Resources
) -> None:
    pipeline = build_pipeline(config(), resources=resources, store=store, collection="tiny")
    report = await pipeline.index(DOCUMENTS)
    again = await pipeline.index(DOCUMENTS)

    answer = await pipeline.answer("How much is the deposit the Tenant shall pay?")

    assert report.built
    assert report.chunks > 4
    assert not again.built
    assert again.chunks == report.chunks
    assert len(answer.chunks) == 3
    assert answer.chunks[0].document_id == "lease.txt"
    assert "deposit equal to three months of rent" in answer.chunks[0].text
    assert answer.text.endswith("[1]")
    assert answer.citations == (1,)
    assert answer.invalid_citations == ()
    assert answer.usage.paid_usd == 0.0


async def test_reranker_node_reorders_candidates(
    store: PostgresStore, resources: Resources
) -> None:
    plain = build_pipeline(
        config(search_mode="dense", generator=None),
        resources=resources,
        store=store,
        collection="tiny",
    )
    reranked = build_pipeline(
        config(search_mode="dense", generator=None, reranker={"name": "overlap"}, candidates=20),
        resources=resources,
        store=store,
        collection="tiny",
    )
    await plain.index(DOCUMENTS)
    question = "What must the Receiving Party return or destroy on written request?"

    result = await reranked.answer(question)
    retrieval_only = await reranked.retrieve(question)

    assert result.text == ""
    assert result.chunks == retrieval_only.chunks
    assert [chunk.rank for chunk in result.chunks] == [1, 2, 3]
    scores = [chunk.score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert "return or destroy" in result.chunks[0].text
    assert reranked.index_name == plain.index_name


async def test_lexical_pipeline_needs_no_embedder(
    store: PostgresStore, resources: Resources
) -> None:
    pipeline = build_pipeline(
        config(search_mode="lexical", embedder=None),
        resources=resources,
        store=store,
        collection="tiny",
    )
    await pipeline.index(DOCUMENTS)
    answer = await pipeline.answer("ninety days notice to terminate")
    assert answer.chunks[0].document_id == "nda.txt"
    assert "ninety days notice" in answer.chunks[0].text


async def test_reranking_retriever_matches_the_graph(
    store: PostgresStore, resources: Resources
) -> None:
    pipeline = build_pipeline(
        config(generator=None, reranker={"name": "overlap"}, candidates=20),
        resources=resources,
        store=store,
        collection="tiny",
    )
    await pipeline.index(DOCUMENTS)
    from ragbench.components.embedders import HashingEmbedder
    from ragbench.components.rerankers import OverlapReranker

    retriever = RerankingRetriever(
        IndexRetriever(store, pipeline.index_name, "hybrid", HashingEmbedder(None, 128)),
        OverlapReranker(),
        candidates=20,
    )
    question = "rent on the first day of each month"
    assert (await retriever.retrieve(question, top_k=3)).chunks == (
        await pipeline.retrieve(question)
    ).chunks


def test_invalid_pipeline_configurations_are_rejected(resources: Resources) -> None:
    with pytest.raises(ValueError, match="embedder is required when search_mode is 'dense'"):
        config(search_mode="dense", embedder=None)
    with pytest.raises(ValueError, match="candidates must be >= top_k"):
        config(reranker={"name": "overlap"}, candidates=2)
    with pytest.raises(ConfigError, match="unknown chunker 'nope'"):
        build_pipeline(
            config(chunker={"name": "nope"}),
            resources=resources,
            store=None,  # type: ignore[arg-type]
            collection="x",
        )
    with pytest.raises(ConfigError, match="needs an embedder"):
        IndexRetriever(None, "i", "hybrid")  # type: ignore[arg-type]


def test_prompt_numbers_passages_and_citations_are_parsed() -> None:
    messages = build_answer_messages("Who pays?", ["The Tenant pays.", "The Landlord insures."])
    assert messages[0].role == "system"
    assert messages[1].content == (
        "Passages:\n\n[1] The Tenant pays.\n\n[2] The Landlord insures.\n\nQuestion: Who pays?"
    )
    assert cited_passages("The Tenant [1][2]. See [3] and [0].", 2) == ([1, 2], [0, 3])
    assert cited_passages("No markers here.", 2) == ([], [])


async def test_evaluate_retrieval_and_judge_answers(
    store: PostgresStore, resources: Resources
) -> None:
    from ragbench.components.judges import LexicalJudge
    from ragbench.core.types import Question, Span
    from ragbench.pipeline import JudgeItem, evaluate_retrieval, judge_answers

    pipeline = build_pipeline(
        config(search_mode="lexical", embedder=None, generator=None),
        resources=resources,
        store=store,
        collection="tiny",
    )
    await pipeline.index(DOCUMENTS)
    lease = DOCUMENTS[1].text
    deposit = lease.index("a deposit equal to three months")
    questions = [
        Question(
            "q1",
            "deposit equal to three months of rent",
            "lease.txt",
            (Span(deposit, deposit + 31),),
        ),
        Question("q2", "zebra giraffe elephant", "lease.txt", (Span(0, 7),)),
    ]

    evaluation = await evaluate_retrieval(
        pipeline, questions, metrics=["hit@3", "mrr@3", "char_recall@3"]
    )

    assert [score.question_id for score in evaluation.scores] == ["q1", "q2"]
    assert evaluation.scores[0].metrics == {"hit@3": 1.0, "mrr@3": 1.0, "char_recall@3": 1.0}
    assert evaluation.scores[0].retrieved[0].startswith("lease.txt:")
    assert evaluation.scores[1].metrics == {"hit@3": 0.0, "mrr@3": 0.0, "char_recall@3": 0.0}
    assert evaluation.means() == {"hit@3": 0.5, "mrr@3": 0.5, "char_recall@3": 0.5}
    assert (await evaluate_retrieval(pipeline, [])).means() == {}

    passage = "The Tenant shall pay a deposit equal to three months of rent."
    verdicts = await judge_answers(
        LexicalJudge(),
        [
            JudgeItem("How much is the deposit?", f"{passage} [1]", (passage,), (passage,)),
            JudgeItem(
                "How much is the deposit?", "Seventeen purple dragons.", (passage,), (passage,)
            ),
        ],
    )
    assert [verdict.scores() for verdict in verdicts] == [
        {"correctness": 2, "faithfulness": 2, "citation": 2},
        {"correctness": 0, "faithfulness": 0, "citation": 0},
    ]
