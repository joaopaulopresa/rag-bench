"""Fixed-size chunking: windows of N tokens with overlap. The baseline."""

from __future__ import annotations

from ragbench.components.chunkers._text import TokenIndex, token_windows
from ragbench.core.interfaces import Tokenizer
from ragbench.core.registries import register_chunker, tokenizers
from ragbench.core.types import Chunk, Document, Span


def split_fixed(text: str, tokenizer: Tokenizer, max_tokens: int, overlap: float) -> list[Span]:
    """Return spans of ``max_tokens`` tokens; consecutive spans share ``overlap`` of a window.

    Args:
        text: The text to split.
        tokenizer: Defines what a token is.
        max_tokens: Tokens per chunk.
        overlap: Fraction of ``max_tokens`` repeated at the start of the next chunk.

    Example:
        >>> from ragbench.core.tokens import WordTokenizer
        >>> [(s.start, s.end) for s in split_fixed("a b c d e", WordTokenizer(), 2, 0.5)]
        [(0, 3), (2, 5), (4, 7), (6, 9)]
    """
    index = TokenIndex(text, tokenizer)
    shared = round(max_tokens * overlap)
    return [
        Span(index.spans[first].start, index.spans[last - 1].end)
        for first, last in token_windows(len(index.spans), max_tokens, shared)
    ]


@register_chunker("fixed")
class FixedChunker:
    """Cuts every ``max_tokens`` tokens, ignoring the structure of the text.

    Args:
        resources: Shared dependencies, used to build the tokenizer.
        max_tokens: Tokens per chunk.
        overlap: Fraction of a chunk repeated at the start of the next one (0 to 0.9).
        tokenizer: Registered tokenizer name.
    """

    def __init__(
        self,
        resources: object,
        max_tokens: int = 512,
        overlap: float = 0.15,
        tokenizer: str = "tiktoken",
    ) -> None:
        if max_tokens < 1 or not 0 <= overlap <= 0.9:
            raise ValueError("max_tokens must be positive and overlap between 0 and 0.9")
        self._max_tokens = max_tokens
        self._overlap = overlap
        self._tokenizer = tokenizers.create(tokenizer, resources, {})

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the fixed-size chunks of ``document``."""
        spans = split_fixed(document.text, self._tokenizer, self._max_tokens, self._overlap)
        return [
            Chunk(document_id=document.id, span=span, text=document.text[span.start : span.end])
            for span in spans
        ]
