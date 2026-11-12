"""Chunkers on tiny texts with exact expected output."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.components.chunkers import split_fixed, split_parent_child, split_recursive
from ragbench.components.chunkers._text import (
    Heading,
    TokenIndex,
    find_headings,
    section_path,
    sentence_spans,
    token_windows,
)
from ragbench.components.chunkers.context_header import ContextHeaderChunker
from ragbench.components.chunkers.semantic import (
    SemanticChunker,
    cosine_distances,
    group_sentences,
    semantic_breakpoints,
)
from ragbench.core.errors import ConfigError
from ragbench.core.resources import Resources
from ragbench.core.tokens import TiktokenTokenizer, WordTokenizer
from ragbench.core.types import Document, Embeddings, Span

WORDS = WordTokenizer()
CONTRACT = (Path(__file__).parent.parent / "fixtures" / "contract.txt").read_text()


def texts(text: str, spans: list[Span]) -> list[str]:
    return [text[span.start : span.end] for span in spans]


def test_token_windows_overlap_and_end() -> None:
    assert token_windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert token_windows(4, 4, 1) == [(0, 4)]
    assert token_windows(0, 4, 1) == []
    with pytest.raises(ValueError, match="overlap"):
        token_windows(10, 4, 4)


def test_fixed_windows_of_four_words_with_one_shared() -> None:
    text = "one two three four five six seven eight nine ten"
    spans = split_fixed(text, WORDS, max_tokens=4, overlap=0.25)
    assert texts(text, spans) == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]


def test_fixed_512_tokens_with_15_percent_overlap_shares_77_tokens() -> None:
    text = " ".join(f"w{number}" for number in range(1200))
    spans = split_fixed(text, WORDS, max_tokens=512, overlap=0.15)
    index = TokenIndex(text, WORDS)
    assert [index.count(span) for span in spans] == [512, 512, 330]
    assert texts(text, spans)[1].split()[0] == "w435"  # 512 - round(512 * 0.15)


def test_recursive_prefers_headings_then_paragraphs_then_sentences() -> None:
    text = (
        "# Intro\n\nAlpha beta gamma. Delta epsilon.\n\n"
        "# Body\n\nOne two three four five six. Seven eight nine ten eleven twelve.\n\nTail end."
    )
    assert texts(text, split_recursive(text, WORDS, max_tokens=8)) == [
        "# Intro\n\nAlpha beta gamma. Delta epsilon.",
        "# Body\n\nOne two three four five six.",
        "Seven eight nine ten eleven twelve.\n\nTail end.",
    ]


def test_recursive_falls_back_to_token_windows_without_any_boundary() -> None:
    text = "a b c d e f g"
    assert texts(text, split_recursive(text, WORDS, max_tokens=3)) == ["a b c", "d e f", "g"]


def test_recursive_keeps_small_documents_whole() -> None:
    assert texts("short text", split_recursive("short text", WORDS, max_tokens=50)) == [
        "short text"
    ]
    assert split_recursive("  \n ", WORDS, max_tokens=50) == []


def test_headings_and_section_paths_of_a_contract() -> None:
    headings = find_headings(CONTRACT)
    assert [(heading.title, heading.level) for heading in headings] == [
        ("MUTUAL NON-DISCLOSURE AGREEMENT", 1),
        ("1. Definitions", 1),
        ("1.1 Confidential Information", 2),
        ("1.2 Representatives", 2),
        ("2. Obligations", 1),
        ("2.1 Use", 2),
        ("2.2 Return", 2),
        ("3. Term", 1),
    ]
    assert section_path(headings, CONTRACT.index("reverse engineer")) == [
        "2. Obligations",
        "2.1 Use",
    ]
    assert section_path(headings, CONTRACT.index("five years")) == ["3. Term"]
    assert section_path([Heading(5, "A", 1)], 0) == []


def test_long_numbered_sentences_are_not_headings() -> None:
    text = "1. The Receiving Party shall keep all Confidential Information strictly " + "x " * 30
    assert find_headings(text) == []


def test_sentences_cover_the_text_without_gaps() -> None:
    text = "First one. Second one!\nThird; Fourth? yes."
    spans = sentence_spans(text)
    assert texts(text, spans) == ["First one. ", "Second one!\n", "Third; ", "Fourth? yes."]
    assert "".join(texts(text, spans)) == text


def test_parent_child_pairs_children_with_their_section() -> None:
    text = "# A\n\na1 a2 a3 a4 a5\n\n# B\n\nb1 b2 b3"
    pairs = split_parent_child(text, WORDS, parent_tokens=7, child_tokens=3)
    assert [(text[c.start : c.end], text[p.start : p.end]) for c, p in pairs] == [
        ("# A\n\na1", "# A\n\na1 a2 a3 a4 a5"),
        ("a2 a3 a4", "# A\n\na1 a2 a3 a4 a5"),
        ("a5", "# A\n\na1 a2 a3 a4 a5"),
        ("# B\n\nb1", "# B\n\nb1 b2 b3"),
        ("b2 b3", "# B\n\nb1 b2 b3"),
    ]


async def test_context_header_prefixes_title_and_section_path() -> None:
    chunker = ContextHeaderChunker(None, max_tokens=30, tokenizer="words")
    chunks = await chunker.chunk(Document(id="nda", text=CONTRACT, title="Acme NDA"))
    by_body = {CONTRACT[chunk.span.start : chunk.span.end][:12]: chunk for chunk in chunks}
    chunk = by_body["2.1 Use\nThe "]
    assert chunk.text.startswith(
        "Document: Acme NDA\nSection: 2. Obligations > 2.1 Use\n\n2.1 Use\n"
    )
    assert chunk.text.endswith(CONTRACT[chunk.span.start : chunk.span.end])
    assert all(chunk.text.startswith("Document: Acme NDA\n") for chunk in chunks)


def test_cosine_distances_and_breakpoints() -> None:
    vectors = [[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 1.0], [-1.0, 0.0]]
    assert cosine_distances(vectors) == [0.0, 1.0, 0.0, 1.0]
    assert cosine_distances([[1.0, 0.0]]) == []
    assert semantic_breakpoints([0.0, 1.0, 0.0, 0.4], 70) == [2]
    assert semantic_breakpoints([0.1, 0.9, 0.1, 0.8], 50) == [2, 4]
    assert semantic_breakpoints([], 90) == []


def test_small_semantic_groups_join_a_neighbour() -> None:
    text = "A b c d. E. F g h i."
    index = TokenIndex(text, WORDS)
    sentences = sentence_spans(text)
    assert texts(text, sentences) == ["A b c d. ", "E. ", "F g h i."]
    assert texts(text, group_sentences(index, sentences, [1, 2], min_tokens=0)) == [
        "A b c d. ",
        "E. ",
        "F g h i.",
    ]
    assert texts(text, group_sentences(index, sentences, [1, 2], min_tokens=2)) == [
        "A b c d. E. ",
        "F g h i.",
    ]
    assert texts(text, group_sentences(index, sentences, [1], min_tokens=5)) == [text]


class TopicEmbedder:
    """Embeds a sentence by the topic word it contains: cats on one axis, tax on the other."""

    key = "topic"
    dimensions = 2

    async def embed(self, texts: list[str], *, kind: str = "document") -> Embeddings:
        return Embeddings(
            vectors=tuple((1.0, 0.0) if "cat" in text else (0.0, 1.0) for text in texts)
        )


async def test_semantic_chunker_splits_where_the_topic_changes(resources: Resources) -> None:
    text = "The cat sleeps. A cat purrs. My cat eats. The tax is due. A tax rises. Pay tax now."
    chunker = SemanticChunker(
        resources,
        embedder={"name": "hashing"},
        percentile=80,
        buffer=0,
        min_tokens=0,
        tokenizer="words",
    )
    chunker._embedder = TopicEmbedder()  # type: ignore[assignment]
    chunks = await chunker.chunk(Document(id="d", text=text))
    assert [chunk.text for chunk in chunks] == [
        "The cat sleeps. A cat purrs. My cat eats.",
        "The tax is due. A tax rises. Pay tax now.",
    ]


def test_semantic_chunker_rejects_a_malformed_embedder(resources: Resources) -> None:
    with pytest.raises(ConfigError, match="embedder must be"):
        SemanticChunker(resources, embedder={"model": "x"})


def test_tiktoken_spans_rebuild_the_text() -> None:
    text = "Confidential Information means any technical data — naïve café, 100% sure."
    spans = TiktokenTokenizer().token_spans(text)
    assert "".join(text[span.start : span.end] for span in spans) == text
    assert 10 < len(spans) < len(text)
    assert TiktokenTokenizer().token_spans("") == []
