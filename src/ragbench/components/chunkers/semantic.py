"""Semantic chunking: split where the meaning of consecutive sentences diverges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ragbench.components.chunkers._text import (
    TokenIndex,
    boundary_levels,
    find_headings,
    recursive_split,
    sentence_spans,
    strip_span,
)
from ragbench.core.errors import ConfigError
from ragbench.core.interfaces import Embedder
from ragbench.core.registries import embedders, register_chunker, tokenizers
from ragbench.core.types import Chunk, Document, Span


def cosine_distances(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Return the cosine distance between each vector and the next one.

    Example:
        >>> cosine_distances([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        [0.0, 1.0]
    """
    if len(vectors) < 2:
        return []
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    unit = matrix / norms[:, None]
    similarities = np.sum(unit[:-1] * unit[1:], axis=1)
    return [float(value) for value in np.round(1.0 - similarities, 12)]


def semantic_breakpoints(distances: Sequence[float], percentile: float) -> list[int]:
    """Return the sentence indexes where a new chunk starts.

    A chunk starts after every gap whose distance is strictly above the given percentile of all
    distances, so about ``100 - percentile`` percent of the gaps become boundaries.

    Example:
        >>> semantic_breakpoints([0.1, 0.9, 0.1, 0.2], 75)
        [2]
    """
    if not distances:
        return []
    threshold = float(np.percentile(np.asarray(distances), percentile))
    return [position + 1 for position, value in enumerate(distances) if value > threshold]


def group_sentences(
    index: TokenIndex,
    sentences: Sequence[Span],
    breakpoints: Sequence[int],
    min_tokens: int,
) -> list[Span]:
    """Join sentences into groups that start at ``breakpoints``.

    A group smaller than ``min_tokens`` is joined to the group before it (the first group is
    joined to the one after it), so that no fragment stands alone.
    """
    starts = [0, *breakpoints]
    groups = [
        Span(sentences[first].start, sentences[last - 1].end)
        for first, last in zip(starts, [*breakpoints, len(sentences)], strict=True)
    ]
    merged: list[Span] = []
    for group in groups:
        if merged and (index.count(group) < min_tokens or index.count(merged[-1]) < min_tokens):
            merged[-1] = Span(merged[-1].start, group.end)
        else:
            merged.append(group)
    return merged


@register_chunker("semantic")
class SemanticChunker:
    """Starts a new chunk where consecutive sentences stop being about the same thing.

    Each sentence is embedded together with ``buffer`` sentences on either side; a boundary is
    placed where the distance between neighbours is above ``percentile``. Groups larger than
    ``max_tokens`` are then split by structure.

    Args:
        resources: Shared dependencies, used to build the tokenizer and the embedder.
        embedder: ``{"name": ..., "params": {...}}`` of the embedder for sentences.
        percentile: Percentile of neighbour distances above which a boundary is placed (0-100).
        buffer: Sentences added on each side of a sentence before embedding it.
        max_tokens: Largest chunk, in tokens.
        min_tokens: Smaller groups are joined to a neighbour.
        tokenizer: Registered tokenizer name.

    Raises:
        ConfigError: If ``embedder`` is not a mapping with a ``name``.
    """

    def __init__(
        self,
        resources: object,
        embedder: Mapping[str, object],
        percentile: float = 90.0,
        buffer: int = 1,
        max_tokens: int = 512,
        min_tokens: int = 48,
        tokenizer: str = "tiktoken",
    ) -> None:
        name = embedder.get("name") if isinstance(embedder, Mapping) else None
        params = embedder.get("params", {}) if isinstance(embedder, Mapping) else None
        if not isinstance(name, str) or not isinstance(params, Mapping):
            raise ConfigError("semantic chunker: embedder must be {name: ..., params: {...}}")
        if not 0 <= percentile <= 100 or buffer < 0 or not 0 <= min_tokens <= max_tokens:
            raise ValueError("need 0 <= percentile <= 100, buffer >= 0, min_tokens <= max_tokens")
        self._embedder: Embedder = embedders.create(name, resources, params)
        self._percentile = percentile
        self._buffer = buffer
        self._max_tokens = max_tokens
        self._min_tokens = min_tokens
        self._tokenizer = tokenizers.create(tokenizer, resources, {})

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the semantic chunks of ``document``."""
        text = document.text
        sentences = sentence_spans(text)
        if not sentences:
            return []
        windows = [
            text[
                sentences[max(0, position - self._buffer)].start : sentences[
                    min(len(sentences) - 1, position + self._buffer)
                ].end
            ]
            for position in range(len(sentences))
        ]
        vectors = (await self._embedder.embed(windows)).vectors
        index = TokenIndex(text, self._tokenizer)
        breakpoints = semantic_breakpoints(cosine_distances(vectors), self._percentile)
        levels = boundary_levels(text, find_headings(text))
        chunks = []
        for group in group_sentences(index, sentences, breakpoints, self._min_tokens):
            for piece in recursive_split(index, group, levels, self._max_tokens):
                span = strip_span(text, piece)
                if span is not None:
                    chunks.append(
                        Chunk(document_id=document.id, span=span, text=text[span.start : span.end])
                    )
        return chunks
