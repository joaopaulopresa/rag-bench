"""Turns documents into a searchable index: chunk, embed, store."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.core.interfaces import Chunker, Embedder, Store
from ragbench.core.types import NO_USAGE, Chunk, Document, Usage

_EMBED_BATCH = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What building (or finding) an index involved.

    Attributes:
        index: Index name.
        chunks: Number of chunks in the index.
        built: ``False`` when the index already existed and nothing was done.
        usage: Embedding cost and time; zero when ``built`` is ``False``.
    """

    index: str
    chunks: int
    built: bool
    usage: Usage = NO_USAGE


def index_name(collection: str, chunker_key: str, embedder_key: str | None) -> str:
    """Return the name of the index for one collection, chunker and embedding space."""
    return f"{collection}|{chunker_key}|{embedder_key or 'lexical'}"


async def build_index(
    store: Store,
    index: str,
    documents: Sequence[Document],
    chunker: Chunker,
    embedder: Embedder | None,
) -> IndexReport:
    """Build ``index`` from ``documents`` unless a complete one already exists.

    Args:
        store: Where the index lives.
        index: Index name, usually from `index_name`.
        documents: The documents to index.
        chunker: How to cut them.
        embedder: How to embed the chunks; ``None`` builds a lexical-only index.

    Returns:
        A report with the number of chunks and the embedding cost.
    """
    if await store.has_index(index):
        return IndexReport(index=index, chunks=await store.count(index), built=False)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(await chunker.chunk(document))
    await store.create_index(index, dimensions=embedder.dimensions if embedder else None)
    usage = NO_USAGE
    for start in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[start : start + _EMBED_BATCH]
        if embedder is None:
            await store.add_chunks(index, batch, None)
            continue
        embeddings = await embedder.embed([chunk.text for chunk in batch])
        usage += embeddings.usage
        await store.add_chunks(index, batch, embeddings.vectors)
    await store.finish_index(index)
    logger.info("built index %s: %d chunks, $%.5f", index, len(chunks), usage.paid_usd)
    return IndexReport(index=index, chunks=len(chunks), built=True, usage=usage)
