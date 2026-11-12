"""The RAG pipeline under test, as a LangGraph graph.

``retrieve -> rerank -> generate -> check_citations``. The ``rerank`` node is skipped when the
pipeline has no reranker and ``generate`` and ``check_citations`` when it has no generator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ragbench.core.config import PipelineConfig
from ragbench.core.errors import ConfigError
from ragbench.core.interfaces import Chunker, Embedder, Generator, Reranker, Store
from ragbench.core.registries import chunkers, embedders, generators, rerankers
from ragbench.core.resources import Resources
from ragbench.core.types import NO_USAGE, Document, RetrievedChunk, Usage
from ragbench.pipeline.indexing import IndexReport, build_index, index_name
from ragbench.pipeline.prompts import build_answer_messages, cited_passages
from ragbench.pipeline.retrievers import IndexRetriever


class _State(TypedDict, total=False):
    question: str
    chunks: tuple[RetrievedChunk, ...]
    retrieval_usage: Usage
    rerank_usage: Usage
    answer: str
    generation_usage: Usage
    citations: list[int]
    invalid_citations: list[int]


@dataclass(frozen=True, slots=True)
class Answer:
    """What the pipeline produced for one question.

    Attributes:
        question: The question asked.
        chunks: The passages given to the generator; passage ``n`` is ``chunks[n - 1]``.
        text: The generated answer; empty for a retrieval-only pipeline.
        citations: Numbers of the passages the answer cites.
        invalid_citations: Cited numbers that match no passage.
        retrieval_usage: Cost and time of embedding the query.
        rerank_usage: Cost and time of reranking.
        generation_usage: Cost and time of generating the answer.
    """

    question: str
    chunks: tuple[RetrievedChunk, ...]
    text: str = ""
    citations: tuple[int, ...] = ()
    invalid_citations: tuple[int, ...] = ()
    retrieval_usage: Usage = NO_USAGE
    rerank_usage: Usage = NO_USAGE
    generation_usage: Usage = NO_USAGE

    @property
    def usage(self) -> Usage:
        """Total cost and time of the answer."""
        return self.retrieval_usage + self.rerank_usage + self.generation_usage


class Pipeline:
    """A configured RAG pipeline bound to one collection of documents.

    Build it with `build_pipeline`, call `index` once, then `retrieve` or `answer`.

    Args:
        config: What the pipeline is made of.
        store: Where the index lives.
        collection: Name of the document collection; part of the index name.
        chunker: Cuts documents.
        embedder: Embeds chunks and queries; ``None`` for a lexical pipeline.
        reranker: Optional second-stage ranker.
        generator: Optional answer generator.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        store: Store,
        collection: str,
        chunker: Chunker,
        embedder: Embedder | None,
        reranker: Reranker | None,
        generator: Generator | None,
    ) -> None:
        self.config = config
        self._store = store
        self._chunker = chunker
        self._embedder = embedder
        self._reranker = reranker
        self._generator = generator
        self.index_name = index_name(
            collection, config.chunker.key, embedder.key if embedder is not None else None
        )
        self._retriever = IndexRetriever(store, self.index_name, config.search_mode, embedder)
        self._graph = self._compile()

    def _compile(self) -> Any:
        graph = StateGraph(_State)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("generate", self._generate)
        graph.add_node("check_citations", self._check_citations)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            lambda state: "rerank" if self._reranker is not None else self._after_ranking(),
        )
        graph.add_conditional_edges("rerank", lambda state: self._after_ranking())
        graph.add_edge("generate", "check_citations")
        graph.add_edge("check_citations", END)
        return graph.compile()

    def _after_ranking(self) -> str:
        return "generate" if self._generator is not None else END

    async def _retrieve(self, state: _State) -> _State:
        wanted = self.config.candidates if self._reranker is not None else self.config.top_k
        retrieval = await self._retriever.retrieve(state["question"], top_k=wanted)
        return {"chunks": retrieval.chunks, "retrieval_usage": retrieval.usage}

    async def _rerank(self, state: _State) -> _State:
        assert self._reranker is not None
        chunks = state["chunks"]
        if not chunks:
            return {}
        reranking = await self._reranker.rerank(state["question"], [chunk.text for chunk in chunks])
        reordered = tuple(
            replace(chunks[item.index], score=item.score, rank=rank)
            for rank, item in enumerate(reranking.scores[: self.config.top_k], start=1)
        )
        return {"chunks": reordered, "rerank_usage": reranking.usage}

    async def _generate(self, state: _State) -> _State:
        assert self._generator is not None
        messages = build_answer_messages(
            state["question"], [chunk.text for chunk in state["chunks"]]
        )
        generation = await self._generator.generate(messages)
        return {"answer": generation.text, "generation_usage": generation.usage}

    async def _check_citations(self, state: _State) -> _State:
        valid, invalid = cited_passages(state["answer"], len(state["chunks"]))
        return {"citations": valid, "invalid_citations": invalid}

    async def index(self, documents: Sequence[Document]) -> IndexReport:
        """Build the pipeline's index from ``documents`` unless it already exists."""
        return await build_index(
            self._store, self.index_name, documents, self._chunker, self._embedder
        )

    async def answer(self, question: str) -> Answer:
        """Run the graph for ``question``.

        Returns:
            The passages and, when the pipeline has a generator, the answer with its checked
            citations.
        """
        state: _State = await self._graph.ainvoke({"question": question})
        empty = NO_USAGE
        return Answer(
            question=question,
            chunks=state.get("chunks", ()),
            text=state.get("answer", ""),
            citations=tuple(state.get("citations", [])),
            invalid_citations=tuple(state.get("invalid_citations", [])),
            retrieval_usage=state.get("retrieval_usage", empty),
            rerank_usage=state.get("rerank_usage", empty),
            generation_usage=state.get("generation_usage", empty),
        )

    async def retrieve(self, question: str) -> Answer:
        """Run only retrieval and reranking for ``question``; no answer is generated."""
        state: _State = {"question": question}
        state.update(await self._retrieve(state))
        if self._reranker is not None:
            state.update(await self._rerank(state))
        empty = NO_USAGE
        return Answer(
            question=question,
            chunks=state.get("chunks", ()),
            retrieval_usage=state.get("retrieval_usage", empty),
            rerank_usage=state.get("rerank_usage", empty),
        )


def build_pipeline(
    config: PipelineConfig, *, resources: Resources, store: Store, collection: str
) -> Pipeline:
    """Build a RAG pipeline from its configuration.

    Args:
        config: Chunker, embedder, search mode, reranker and generator, by registered name.
        resources: Shared dependencies handed to every component.
        store: Where the index lives.
        collection: Name of the document collection the pipeline searches.

    Returns:
        The pipeline. Call `Pipeline.index` before the first question.

    Raises:
        ConfigError: If a component name is unknown or its parameters do not fit.

    Example:
        >>> config = PipelineConfig.model_validate({
        ...     "chunker": {"name": "recursive"},
        ...     "embedder": {"name": "hashing"},
        ...     "generator": {"name": "extractive"},
        ... })
        >>> pipeline = build_pipeline(config, resources=resources, store=store,
        ...                           collection="handbook")  # doctest: +SKIP
    """
    if not collection:
        raise ConfigError("collection must not be empty")
    return Pipeline(
        config,
        store=store,
        collection=collection,
        chunker=chunkers.create(config.chunker.name, resources, config.chunker.params),
        embedder=(
            embedders.create(config.embedder.name, resources, config.embedder.params)
            if config.embedder is not None
            else None
        ),
        reranker=(
            rerankers.create(config.reranker.name, resources, config.reranker.params)
            if config.reranker is not None
            else None
        ),
        generator=(
            generators.create(config.generator.name, resources, config.generator.params)
            if config.generator is not None
            else None
        ),
    )
