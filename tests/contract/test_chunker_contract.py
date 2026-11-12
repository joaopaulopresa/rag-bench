"""Rules every registered chunker must obey, whatever its strategy."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.core.interfaces import Chunker
from ragbench.core.registries import chunkers
from ragbench.core.resources import Resources
from ragbench.core.types import Document

CONTRACT = (Path(__file__).parent.parent / "fixtures" / "contract.txt").read_text()
PARAMS: dict[str, dict[str, object]] = {
    "fixed": {"max_tokens": 40, "tokenizer": "words"},
    "recursive": {"max_tokens": 40, "tokenizer": "words"},
    "parent_child": {"parent_tokens": 60, "child_tokens": 15, "tokenizer": "words"},
    "context_header": {"max_tokens": 40, "tokenizer": "words"},
    "semantic": {
        "embedder": {"name": "hashing", "params": {"dimensions": 64}},
        "max_tokens": 40,
        "min_tokens": 5,
        "tokenizer": "words",
    },
}
DOCUMENTS = [
    Document(id="contract", text=CONTRACT, title="Acme NDA"),
    Document(id="one-line", text="A single short sentence."),
    Document(id="unicode", text="﻿Café – naïve “quotes”.\r\n\r\nSecond § paragraph."),
]


def test_every_builtin_chunker_is_covered() -> None:
    assert set(PARAMS) == {"fixed", "recursive", "parent_child", "context_header", "semantic"}
    assert set(PARAMS) <= set(chunkers.names())


@pytest.fixture(params=sorted(PARAMS))
def chunker(request: pytest.FixtureRequest, resources: Resources) -> Chunker:
    instance = chunkers.create(request.param, resources, PARAMS[request.param])
    assert isinstance(instance, Chunker)
    return instance


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda document: document.id)
async def test_chunks_are_valid_ordered_and_complete(chunker: Chunker, document: Document) -> None:
    chunks = await chunker.chunk(document)
    text = document.text

    assert chunks, "a non-empty document gives at least one chunk"
    covered: set[int] = set()
    for chunk in chunks:
        assert chunk.document_id == document.id
        assert 0 <= chunk.span.start < chunk.span.end <= len(text)
        body = text[chunk.span.start : chunk.span.end]
        assert body == body.strip(), "chunks neither start nor end with whitespace"
        assert chunk.text.endswith(body), "the searched text ends with the document text"
        if chunk.parent is not None:
            assert chunk.parent.start <= chunk.span.start and chunk.span.end <= chunk.parent.end
            assert chunk.parent_text == text[chunk.parent.start : chunk.parent.end]
        assert chunk.returned_text
        covered.update(range(chunk.span.start, chunk.span.end))
    starts = [chunk.span.start for chunk in chunks]
    assert starts == sorted(starts), "chunks come in reading order"
    missing = [
        position for position, char in enumerate(text) if char.strip() and position not in covered
    ]
    assert not missing, "every visible character belongs to a chunk"


async def test_chunking_is_deterministic(chunker: Chunker) -> None:
    document = DOCUMENTS[0]
    assert await chunker.chunk(document) == await chunker.chunk(document)


async def test_empty_document_gives_no_chunks(chunker: Chunker) -> None:
    assert await chunker.chunk(Document(id="empty", text="  \n\n ")) == []
