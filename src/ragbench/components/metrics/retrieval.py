"""Retrieval metrics as pure functions of one question's ranked results.

Two families:

* Rank metrics (``recall``, ``precision``, ``ndcg``, ``mrr``, ``hit``) treat a passage as
  relevant when it overlaps a gold span. They follow the standard definitions and are checked
  against ``ranx`` in the test suite.
* Character metrics (``char_recall``, ``char_precision``) are the LegalBench-RAG measures: the
  share of gold characters that were retrieved, and the share of retrieved characters that are
  gold. They are comparable across chunking strategies because they do not depend on how the
  text was cut.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ragbench.core.errors import ConfigError
from ragbench.core.interfaces import Metric
from ragbench.core.registries import metrics, register_metric
from ragbench.core.types import RetrievalSample, RetrievedChunk, Span


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Return the union of ``spans`` as sorted, non-overlapping spans.

    Example:
        >>> merge_spans([Span(5, 9), Span(0, 3), Span(2, 6)])
        [Span(start=0, end=9)]
    """
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def overlap_size(first: Sequence[Span], second: Sequence[Span]) -> int:
    """Return how many characters two sets of spans share. Each set is merged first."""
    return sum(a.overlap(b) for a in merge_spans(first) for b in merge_spans(second))


class _AtK:
    """Base of metrics computed on the first ``k`` results."""

    label = ""

    def __init__(self, resources: object = None, k: int = 10) -> None:
        del resources
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    @property
    def name(self) -> str:
        """Display name, for example ``recall@5``."""
        return f"{self.label}@{self.k}"

    def _top(self, sample: RetrievalSample) -> Sequence[RetrievedChunk]:
        return sample.retrieved[: self.k]

    def _hits(self, sample: RetrievalSample) -> list[bool]:
        return [sample.is_relevant(chunk) for chunk in self._top(sample)]


@register_metric("recall")
class Recall(_AtK):
    """Share of the relevant passages of the index that appear in the first ``k`` results."""

    label = "recall"

    def score(self, sample: RetrievalSample) -> float:
        """Return relevant retrieved / relevant in the index (0 when none is relevant)."""
        if not sample.relevant:
            return 0.0
        return sum(self._hits(sample)) / len(sample.relevant)


@register_metric("precision")
class Precision(_AtK):
    """Share of the first ``k`` results that are relevant (missing results count as misses)."""

    label = "precision"

    def score(self, sample: RetrievalSample) -> float:
        """Return relevant retrieved / k."""
        return sum(self._hits(sample)) / self.k


@register_metric("hit")
class HitRate(_AtK):
    """Whether at least one of the first ``k`` results is relevant."""

    label = "hit"

    def score(self, sample: RetrievalSample) -> float:
        """Return 1.0 if a relevant passage is in the first ``k`` results, else 0.0."""
        return 1.0 if any(self._hits(sample)) else 0.0


@register_metric("mrr")
class Mrr(_AtK):
    """Reciprocal rank of the first relevant result among the first ``k``."""

    label = "mrr"

    def score(self, sample: RetrievalSample) -> float:
        """Return 1 / rank of the first relevant result, or 0.0 if there is none."""
        for position, hit in enumerate(self._hits(sample), start=1):
            if hit:
                return 1.0 / position
        return 0.0


@register_metric("ndcg")
class Ndcg(_AtK):
    """Normalised discounted cumulative gain with binary relevance."""

    label = "ndcg"

    def score(self, sample: RetrievalSample) -> float:
        """Return DCG of the first ``k`` results divided by the best possible DCG."""
        gain = sum(
            1.0 / math.log2(position + 1)
            for position, hit in enumerate(self._hits(sample), start=1)
            if hit
        )
        ideal = sum(
            1.0 / math.log2(position + 1)
            for position in range(1, min(len(sample.relevant), self.k) + 1)
        )
        return gain / ideal if ideal else 0.0


@register_metric("char_recall")
class CharRecall(_AtK):
    """Share of the gold characters covered by the first ``k`` results."""

    label = "char_recall"

    def score(self, sample: RetrievalSample) -> float:
        """Return retrieved gold characters / gold characters."""
        gold = merge_spans(sample.question.spans)
        total = sum(len(span) for span in gold)
        if not total:
            return 0.0
        retrieved = [
            chunk.span
            for chunk in self._top(sample)
            if chunk.document_id == sample.question.document_id
        ]
        return overlap_size(retrieved, gold) / total


@register_metric("char_precision")
class CharPrecision(_AtK):
    """Share of the characters in the first ``k`` results that are gold."""

    label = "char_precision"

    def score(self, sample: RetrievalSample) -> float:
        """Return retrieved gold characters / retrieved characters, over all documents."""
        by_document: dict[str, list[Span]] = {}
        for chunk in self._top(sample):
            by_document.setdefault(chunk.document_id, []).append(chunk.span)
        total = sum(len(span) for spans in by_document.values() for span in merge_spans(spans))
        if not total:
            return 0.0
        retrieved = by_document.get(sample.question.document_id, [])
        return overlap_size(retrieved, sample.question.spans) / total


def get_metric(spec: str) -> Metric:
    """Build a metric from a specification such as ``recall@5`` or ``ndcg``.

    Args:
        spec: Registered metric name, optionally followed by ``@k``.

    Raises:
        ConfigError: If the name is not registered or ``k`` is not a positive integer.

    Example:
        >>> get_metric("char_recall@5").name
        'char_recall@5'
    """
    name, separator, cutoff = spec.partition("@")
    if separator and not (cutoff.isdigit() and int(cutoff) > 0):
        raise ConfigError(f"metric {spec!r}: k must be a positive integer")
    return metrics.create(name, None, {"k": int(cutoff)} if separator else {})
