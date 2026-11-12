"""The extension points. Core code depends on these protocols, never on an implementation.

Anything that may touch the network or the database is ``async``. Loading a corpus is the one
synchronous I/O operation: it runs once, before any event loop is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragbench.core.types import (
    CacheEntry,
    Chunk,
    Corpus,
    Document,
    Embeddings,
    EmbedKind,
    Generation,
    Message,
    Question,
    Reranking,
    RetrievalSample,
    RetrievedChunk,
    SearchMode,
    Span,
    Verdict,
)


@runtime_checkable
class Tokenizer(Protocol):
    """Splits text into tokens and reports where each token lies."""

    def token_spans(self, text: str) -> list[Span]:
        """Return the character span of every token, in order.

        Consecutive spans may leave gaps (whitespace) but never overlap.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Cuts a document into chunks."""

    async def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks of ``document`` in reading order.

        Every chunk's ``span`` must lie inside the document, and the same document must always
        give the same chunks.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors."""

    @property
    def key(self) -> str:
        """Identity of the embedding space (backend, model, dimensions). Names indexes."""
        ...

    @property
    def dimensions(self) -> int:
        """Length of the vectors returned by `embed`."""
        ...

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> Embeddings:
        """Embed ``texts`` and return one unit-length vector per text, in input order."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Scores passages against a query with a cross-encoder style model."""

    @property
    def key(self) -> str:
        """Identity of the reranker (backend and model)."""
        ...

    async def rerank(self, query: str, texts: Sequence[str]) -> Reranking:
        """Score every text against ``query``; return positions from best to worst."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Produces text from chat messages."""

    @property
    def key(self) -> str:
        """Identity of the generator (backend and model)."""
        ...

    async def generate(self, messages: Sequence[Message]) -> Generation:
        """Return the model's reply to ``messages``."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Key-value store placed in front of every paid model call."""

    async def get_many(self, keys: Sequence[str]) -> dict[str, CacheEntry]:
        """Return the stored entries among ``keys``."""
        ...

    async def set_many(self, entries: Sequence[CacheEntry]) -> None:
        """Store ``entries``, replacing entries with the same key."""
        ...


@runtime_checkable
class Store(Protocol):
    """Chunk index with dense, lexical and hybrid search."""

    async def has_index(self, index: str) -> bool:
        """Return whether ``index`` exists and was completely written."""
        ...

    async def create_index(self, index: str, *, dimensions: int | None) -> None:
        """Create ``index`` empty, dropping any earlier incomplete version of it.

        Args:
            index: Index name.
            dimensions: Vector length, or ``None`` for a lexical-only index.
        """
        ...

    async def add_chunks(
        self, index: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]] | None
    ) -> None:
        """Append ``chunks`` (and their ``vectors``, in the same order) to ``index``."""
        ...

    async def finish_index(self, index: str) -> None:
        """Build the search structures of ``index`` and mark it complete."""
        ...

    async def search(
        self,
        index: str,
        *,
        mode: SearchMode,
        query: str,
        vector: Sequence[float] | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` best passages of ``index`` for the query, best first.

        Chunks that share a returned span (children of one parent) appear once.
        """
        ...

    async def relevant_keys(self, index: str, question: Question) -> frozenset[str]:
        """Return the keys of every passage of ``index`` that overlaps a gold span."""
        ...

    async def count(self, index: str) -> int:
        """Return the number of chunks stored in ``index``."""
        ...

    async def drop_index(self, index: str) -> None:
        """Delete ``index`` if it exists."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Answers a query with ranked passages."""

    async def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` best passages for ``query``, best first."""
        ...


@runtime_checkable
class Metric(Protocol):
    """A retrieval metric computed for one question."""

    @property
    def name(self) -> str:
        """Display name including parameters, for example ``recall@5``."""
        ...

    def score(self, sample: RetrievalSample) -> float:
        """Return the metric value for ``sample``, between 0 and 1."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Scores a generated answer against the gold evidence and the retrieved context."""

    @property
    def key(self) -> str:
        """Identity of the judge (backend, model and rubric version)."""
        ...

    async def judge(
        self, *, question: str, answer: str, gold: Sequence[str], context: Sequence[str]
    ) -> Verdict:
        """Score ``answer``.

        Args:
            question: The question that was asked.
            answer: The generated answer, with its ``[n]`` citations.
            gold: Text of the gold evidence spans.
            context: The passages the generator saw; passage ``n`` is ``context[n - 1]``.
        """
        ...


@runtime_checkable
class CorpusLoader(Protocol):
    """Loads documents and gold questions."""

    def load(self) -> Corpus:
        """Return the corpus, downloading and verifying it first if needed."""
        ...
