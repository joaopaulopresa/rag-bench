"""Retrievers: search an index, and optionally rerank what it returns."""

from __future__ import annotations

from dataclasses import replace

from ragbench.core.errors import ConfigError
from ragbench.core.interfaces import Embedder, Reranker, Retriever, Store
from ragbench.core.types import NO_USAGE, Retrieval, SearchMode


class IndexRetriever:
    """Searches one index of a store in dense, lexical or hybrid mode.

    Args:
        store: The store that holds the index.
        index: Index name.
        mode: Search mode.
        embedder: Embeds the query; required unless ``mode`` is ``lexical``.

    Raises:
        ConfigError: If ``mode`` needs an embedder and none is given.
    """

    def __init__(
        self, store: Store, index: str, mode: SearchMode, embedder: Embedder | None = None
    ) -> None:
        if mode != "lexical" and embedder is None:
            raise ConfigError(f"{mode} search needs an embedder")
        self._store = store
        self._index = index
        self._mode: SearchMode = mode
        self._embedder = embedder

    async def retrieve(self, query: str, *, top_k: int) -> Retrieval:
        """Return the ``top_k`` best passages for ``query``."""
        vector = None
        usage = NO_USAGE
        if self._mode != "lexical" and self._embedder is not None:
            embeddings = await self._embedder.embed([query], kind="query")
            vector = embeddings.vectors[0]
            usage = embeddings.usage
        chunks = await self._store.search(
            self._index, mode=self._mode, query=query, vector=vector, top_k=top_k
        )
        return Retrieval(chunks=tuple(chunks), usage=usage)


class RerankingRetriever:
    """Retrieves ``candidates`` passages from another retriever and reorders them.

    Args:
        base: The retriever that provides candidates.
        reranker: Scores each candidate against the query.
        candidates: How many candidates to ask ``base`` for.
    """

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 30) -> None:
        if candidates < 1:
            raise ConfigError("candidates must be positive")
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    async def retrieve(self, query: str, *, top_k: int) -> Retrieval:
        """Return the ``top_k`` best candidates according to the reranker."""
        first = await self._base.retrieve(query, top_k=max(self._candidates, top_k))
        if not first.chunks:
            return first
        reranking = await self._reranker.rerank(query, [chunk.text for chunk in first.chunks])
        chunks = tuple(
            replace(first.chunks[item.index], score=item.score, rank=rank)
            for rank, item in enumerate(reranking.scores[:top_k], start=1)
        )
        return Retrieval(chunks=chunks, usage=first.usage + reranking.usage)
