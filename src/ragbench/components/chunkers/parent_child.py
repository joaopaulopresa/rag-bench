"""Parent-child chunking: search small chunks, return the section that contains them."""

from __future__ import annotations

from ragbench.components.chunkers._text import (
    TokenIndex,
    boundary_levels,
    find_headings,
    recursive_split,
    strip_span,
    token_windows,
)
from ragbench.core.interfaces import Tokenizer
from ragbench.core.registries import register_chunker, tokenizers
from ragbench.core.types import Chunk, Document, Span


def split_parent_child(
    text: str, tokenizer: Tokenizer, parent_tokens: int, child_tokens: int, overlap: float = 0.0
) -> list[tuple[Span, Span]]:
    """Return ``(child, parent)`` span pairs.

    Parents are structure-aware sections of at most ``parent_tokens``. Children are windows of
    ``child_tokens`` inside one parent.

    Example:
        >>> from ragbench.core.tokens import WordTokenizer
        >>> text = "a b c d.\\n\\ne f."
        >>> pairs = split_parent_child(text, WordTokenizer(), 4, 2)
        >>> [(text[c.start:c.end], text[p.start:p.end]) for c, p in pairs]
        [('a b', 'a b c d.'), ('c d.', 'a b c d.'), ('e f.', 'e f.')]
    """
    index = TokenIndex(text, tokenizer)
    levels = boundary_levels(text, find_headings(text))
    pairs = []
    for piece in recursive_split(index, Span(0, len(text)), levels, parent_tokens):
        parent = strip_span(text, piece)
        if parent is None:
            continue
        first = index.first_token_at(parent.start)
        count = index.count(parent)
        for window_first, window_last in token_windows(
            count, child_tokens, round(child_tokens * overlap)
        ):
            start = index.spans[first + window_first].start
            end = index.spans[first + window_last - 1].end
            pairs.append((Span(start, min(end, parent.end)), parent))
    return pairs


@register_chunker("parent_child")
class ParentChildChunker:
    """Indexes small chunks for precise matching and returns their parent for context.

    Args:
        resources: Shared dependencies, used to build the tokenizer.
        parent_tokens: Largest parent section, in tokens.
        child_tokens: Size of the searched chunks, in tokens.
        overlap: Fraction of a child repeated at the start of the next one (0 to 0.9).
        tokenizer: Registered tokenizer name.
    """

    def __init__(
        self,
        resources: object,
        parent_tokens: int = 1024,
        child_tokens: int = 128,
        overlap: float = 0.0,
        tokenizer: str = "tiktoken",
    ) -> None:
        if not 1 <= child_tokens <= parent_tokens or not 0 <= overlap <= 0.9:
            raise ValueError("need 1 <= child_tokens <= parent_tokens and overlap in [0, 0.9]")
        self._parent_tokens = parent_tokens
        self._child_tokens = child_tokens
        self._overlap = overlap
        self._tokenizer = tokenizers.create(tokenizer, resources, {})

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the child chunks of ``document``, each carrying its parent."""
        text = document.text
        pairs = split_parent_child(
            text, self._tokenizer, self._parent_tokens, self._child_tokens, self._overlap
        )
        return [
            Chunk(
                document_id=document.id,
                span=child,
                text=text[child.start : child.end],
                parent=parent,
                parent_text=text[parent.start : parent.end],
            )
            for child, parent in pairs
        ]
