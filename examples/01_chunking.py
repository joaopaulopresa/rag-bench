"""Compare how the built-in chunkers cut the same document. Needs nothing but the package.

uv run python examples/01_chunking.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ragbench import Document, get_chunker

HANDBOOK = Path(__file__).parent / "handbook"


async def main() -> None:
    text = (HANDBOOK / "expenses.txt").read_text(encoding="utf-8")
    document = Document(id="expenses.txt", text=text, title="Expense Policy")
    strategies = {
        "fixed": get_chunker("fixed", max_tokens=60, overlap=0.15, tokenizer="words"),
        "recursive": get_chunker("recursive", max_tokens=60, tokenizer="words"),
        "parent_child": get_chunker(
            "parent_child", parent_tokens=120, child_tokens=30, tokenizer="words"
        ),
        "context_header": get_chunker("context_header", max_tokens=60, tokenizer="words"),
    }
    for name, chunker in strategies.items():
        pieces = await chunker.chunk(document)
        returned = {piece.returned_span for piece in pieces}
        print(f"\n{name}: {len(pieces)} chunks searched, {len(returned)} passages returned")
        second = pieces[1]
        print(f"  second chunk, characters {second.span.start}-{second.span.end}:")
        for line in second.text.splitlines()[:4]:
            print(f"    | {line[:90]}")


if __name__ == "__main__":
    asyncio.run(main())
