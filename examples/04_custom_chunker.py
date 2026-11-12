"""Add your own chunker: one class, one decorator, no change to ragbench.

A package can also publish it through the `ragbench.chunkers` entry point, after which
experiment files can name it like any built-in strategy.

    uv run python examples/04_custom_chunker.py
"""

from __future__ import annotations

import asyncio
import re

from ragbench import Chunk, Document, Span, get_chunker, register_chunker

_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\s*\n\s*\n|\s*\Z)", re.DOTALL)


@register_chunker("paragraph")
class ParagraphChunker:
    """One chunk per paragraph, skipping paragraphs shorter than `min_chars`."""

    def __init__(self, resources: object = None, min_chars: int = 1) -> None:
        self._min_chars = min_chars

    async def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(document.id, Span(match.start(), match.end()), match.group(0))
            for match in _PARAGRAPH.finditer(document.text)
            if len(match.group(0)) >= self._min_chars
        ]


async def main() -> None:
    document = Document(id="note", text="Title\n\nFirst paragraph, long enough.\n\nSecond one.\n")
    chunker = get_chunker("paragraph", min_chars=8)
    for chunk in await chunker.chunk(document):
        print(f"{chunk.span.start:>3}-{chunk.span.end:<3} {chunk.text!r}")


if __name__ == "__main__":
    asyncio.run(main())
