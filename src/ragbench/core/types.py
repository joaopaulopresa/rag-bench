"""Value types shared by every layer. All of them are immutable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ragbench.core.errors import DatasetError

Role = Literal["system", "user", "assistant"]
SearchMode = Literal["dense", "lexical", "hybrid"]
EmbedKind = Literal["document", "query"]


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """Half-open character range ``[start, end)`` inside a document."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def overlap(self, other: Span) -> int:
        """Return the number of characters shared with ``other``."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def overlaps(self, other: Span) -> bool:
        """Return whether at least one character is shared with ``other``."""
        return self.overlap(other) > 0


@dataclass(frozen=True, slots=True)
class Document:
    """A source document.

    Attributes:
        id: Stable identifier, unique inside a corpus.
        text: Full text. Gold spans and chunk spans index into it.
        title: Human-readable title, used by chunkers that add context headers.
        source: Name of the collection the document belongs to.
    """

    id: str
    text: str
    title: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class Question:
    """A question with its gold evidence.

    Attributes:
        id: Stable identifier, unique inside a corpus.
        text: The question as asked.
        document_id: The document that holds the evidence.
        spans: Gold evidence as character spans of that document.
        source: Name of the collection, used for stratified sampling and breakdowns.
    """

    id: str
    text: str
    document_id: str
    spans: tuple[Span, ...]
    source: str = ""


@dataclass(frozen=True, slots=True)
class Corpus:
    """Documents plus gold questions.

    Attributes:
        name: Corpus name.
        documents: The documents to index.
        questions: The gold questions. Each one points to a document in ``documents``.
        revision: Version of the upstream data (for example a dataset commit hash).
        licences: Licence of each source collection, keyed by source name.
    """

    name: str
    documents: tuple[Document, ...]
    questions: tuple[Question, ...]
    revision: str = ""
    licences: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = {document.id for document in self.documents}
        if len(known) != len(self.documents):
            raise DatasetError(f"corpus {self.name!r} has duplicate document ids")
        for question in self.questions:
            if question.document_id not in known:
                raise DatasetError(
                    f"question {question.id!r} points to unknown document {question.document_id!r}"
                )

    def document(self, document_id: str) -> Document:
        """Return the document with this id.

        Raises:
            DatasetError: If the corpus has no such document.
        """
        for document in self.documents:
            if document.id == document_id:
                return document
        raise DatasetError(f"corpus {self.name!r} has no document {document_id!r}")

    def gold_text(self, question: Question) -> list[str]:
        """Return the text of each gold span of ``question``."""
        text = self.document(question.document_id).text
        return [text[span.start : span.end] for span in question.spans]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document produced by a chunker.

    Attributes:
        document_id: The document the chunk was cut from.
        span: Where the chunk body lies in the document.
        text: The text that is embedded and searched. It may carry a prefix (for example a
            context header) that is not part of the document.
        parent: The span returned to the reader when this chunk matches. ``None`` means the
            chunk itself. Parent-child chunking sets it to the enclosing section.
        parent_text: Text of ``parent``. Empty when there is no parent.
    """

    document_id: str
    span: Span
    text: str
    parent: Span | None = None
    parent_text: str = ""

    @property
    def returned_span(self) -> Span:
        """The span a retriever returns for this chunk: the parent if there is one."""
        return self.parent if self.parent is not None else self.span

    @property
    def returned_text(self) -> str:
        """The text a retriever returns for this chunk: the parent text if there is one."""
        return self.parent_text if self.parent is not None else self.text


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One search result.

    Attributes:
        document_id: The document the passage comes from.
        span: The passage as a character span of that document.
        text: The passage text handed to rerankers and generators.
        score: Score given by the last stage that ranked it (higher is better).
        rank: 1-based position in the result list.
    """

    document_id: str
    span: Span
    text: str
    score: float
    rank: int

    @property
    def key(self) -> str:
        """Identifier of the passage: document id plus span."""
        return f"{self.document_id}:{self.span.start}-{self.span.end}"


@dataclass(frozen=True, slots=True)
class Usage:
    """Cost and timing of one model call, or of several added together.

    Attributes:
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
        cost_usd: Price of the calls when they were first made. For a result served from the
            cache this is what the original call cost.
        latency_s: Wall-clock seconds of the original calls.
        paid_usd: Money actually spent now. Calls served from the cache add nothing to it, so
            a sum of a cached and a fresh call pays only for the fresh one.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    paid_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
            paid_usd=self.paid_usd + other.paid_usd,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], *, cached: bool) -> Usage:
        """Read the usage fields stored in a cache payload.

        Args:
            payload: The stored response.
            cached: Whether the response was read from the cache (``True``) or has just been
                paid for (``False``).
        """

        def number(name: str) -> float:
            value = payload.get(name, 0)
            return float(value) if isinstance(value, int | float) else 0.0

        return cls(
            prompt_tokens=int(number("prompt_tokens")),
            completion_tokens=int(number("completion_tokens")),
            cost_usd=number("cost_usd"),
            latency_s=number("latency_s"),
            paid_usd=0.0 if cached else number("cost_usd"),
        )

    @property
    def cached(self) -> bool:
        """Whether nothing was paid now: every call behind this usage came from the cache."""
        return self.paid_usd == 0.0


#: Usage of something that cost nothing and called no model.
NO_USAGE = Usage()


@dataclass(frozen=True, slots=True)
class Retrieval:
    """Result of one retrieval: ranked passages plus what finding them cost."""

    chunks: tuple[RetrievedChunk, ...]
    usage: Usage = NO_USAGE


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Embeddings:
    """Vectors for a batch of texts, in input order, with the usage of the batch."""

    vectors: tuple[tuple[float, ...], ...]
    usage: Usage = Usage()


@dataclass(frozen=True, slots=True)
class Generation:
    """Text produced by a generator."""

    text: str
    model: str
    usage: Usage = Usage()


@dataclass(frozen=True, slots=True)
class RerankScore:
    """Relevance score of the input passage at position ``index``."""

    index: int
    score: float


@dataclass(frozen=True, slots=True)
class Reranking:
    """Reranker output: input positions ordered from most to least relevant."""

    scores: tuple[RerankScore, ...]
    usage: Usage = Usage()


@dataclass(frozen=True, slots=True)
class Verdict:
    """A judge's scores for one answer. Each score is 0 (bad), 1 (partial) or 2 (good).

    Attributes:
        correctness: Does the answer state what the gold evidence states?
        faithfulness: Is every claim supported by the retrieved context?
        citation: Do the citations point to the passages that support the claims?
        rationale: One or two sentences explaining the scores.
    """

    correctness: int
    faithfulness: int
    citation: int
    rationale: str = ""
    usage: Usage = Usage()

    def scores(self) -> dict[str, int]:
        """Return the three scores keyed by criterion name."""
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "citation": self.citation,
        }


@dataclass(frozen=True, slots=True)
class RetrievalSample:
    """Everything a retrieval metric needs for one question.

    Attributes:
        question: The gold question.
        retrieved: Ranked results, best first.
        relevant: Keys (`RetrievedChunk.key`) of every passage in the index that overlaps a
            gold span. Rank metrics need it to know how many relevant passages exist.
    """

    question: Question
    retrieved: Sequence[RetrievedChunk]
    relevant: frozenset[str]

    def is_relevant(self, chunk: RetrievedChunk) -> bool:
        """Return whether ``chunk`` overlaps a gold span of the question."""
        return chunk.document_id == self.question.document_id and any(
            chunk.span.overlaps(span) for span in self.question.spans
        )


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A stored model response.

    Attributes:
        key: Hash of provider, model, parameters and input.
        payload: JSON-serialisable response body.
        blob: Optional binary part (embedding vectors are stored as float32 bytes).
    """

    key: str
    payload: Mapping[str, object]
    blob: bytes | None = None
