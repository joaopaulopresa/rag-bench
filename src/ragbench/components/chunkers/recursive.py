"""Recursive chunking by structure: headings, then paragraphs, lines, sentences, tokens."""

from __future__ import annotations

from ragbench.components.chunkers._text import (
    TokenIndex,
    boundary_levels,
    find_headings,
    recursive_split,
    strip_span,
)
from ragbench.core.interfaces import Tokenizer
from ragbench.core.registries import register_chunker, tokenizers
from ragbench.core.types import Chunk, Document, Span


def split_recursive(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[Span]:
    """Return spans of at most ``max_tokens`` tokens that respect the text's structure.

    Chunks never overlap and never start or end with whitespace.

    Example:
        >>> from ragbench.core.tokens import WordTokenizer
        >>> text = "one two.\\n\\nthree four five."
        >>> [text[s.start:s.end] for s in split_recursive(text, WordTokenizer(), 3)]
        ['one two.', 'three four five.']
    """
    index = TokenIndex(text, tokenizer)
    levels = boundary_levels(text, find_headings(text))
    pieces = recursive_split(index, Span(0, len(text)), levels, max_tokens)
    return [span for piece in pieces if (span := strip_span(text, piece)) is not None]


@register_chunker("recursive")
class RecursiveChunker:
    """Splits at the strongest boundary that keeps chunks under ``max_tokens``.

    Args:
        resources: Shared dependencies, used to build the tokenizer.
        max_tokens: Largest chunk, in tokens.
        tokenizer: Registered tokenizer name.
    """

    def __init__(
        self, resources: object, max_tokens: int = 512, tokenizer: str = "tiktoken"
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens
        self._tokenizer = tokenizers.create(tokenizer, resources, {})

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the structure-aware chunks of ``document``."""
        return [
            Chunk(document_id=document.id, span=span, text=document.text[span.start : span.end])
            for span in split_recursive(document.text, self._tokenizer, self._max_tokens)
        ]
