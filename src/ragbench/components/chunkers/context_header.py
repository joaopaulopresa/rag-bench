"""Context-header chunking: every chunk is prefixed with its document title and section path."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.components.chunkers._text import find_headings, section_path
from ragbench.components.chunkers.recursive import split_recursive
from ragbench.core.registries import register_chunker, tokenizers
from ragbench.core.types import Chunk, Document


def context_header(title: str, path: Sequence[str]) -> str:
    """Return the header put before a chunk.

    Example:
        >>> context_header("Acme NDA", ["ARTICLE 2", "2.1 Scope"])
        'Document: Acme NDA\\nSection: ARTICLE 2 > 2.1 Scope'
    """
    lines = []
    if title:
        lines.append(f"Document: {title}")
    if path:
        lines.append("Section: " + " > ".join(path))
    return "\n".join(lines)


@register_chunker("context_header")
class ContextHeaderChunker:
    """Structure-aware chunks whose searched text starts with where they come from.

    The header gives the embedder and the lexical index the document title and the headings
    above the chunk, which a chunk in the middle of a contract does not contain by itself. The
    chunk's span still covers only the document text.

    Args:
        resources: Shared dependencies, used to build the tokenizer.
        max_tokens: Largest chunk body, in tokens (the header is not counted).
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
        """Return the chunks of ``document``, each with its context header."""
        text = document.text
        headings = find_headings(text)
        chunks = []
        for span in split_recursive(text, self._tokenizer, self._max_tokens):
            header = context_header(document.title, section_path(headings, span.start))
            body = text[span.start : span.end]
            chunks.append(
                Chunk(
                    document_id=document.id,
                    span=span,
                    text=f"{header}\n\n{body}" if header else body,
                )
            )
        return chunks
