"""Retrieval metrics: exact values on a hand-made ranking, and agreement with ranx."""

from __future__ import annotations

import math
import random

import pytest

from ragbench.components.metrics import get_metric
from ragbench.components.metrics.retrieval import merge_spans, overlap_size
from ragbench.core.errors import ConfigError
from ragbench.core.registries import metrics
from ragbench.core.types import Question, RetrievalSample, RetrievedChunk, Span


def chunk(document: str, start: int, end: int, rank: int) -> RetrievedChunk:
    return RetrievedChunk(document, Span(start, end), "", 1.0 / rank, rank)


# Gold: characters 100-200 and 500-520 of document "a" (120 gold characters).
QUESTION = Question("q1", "?", "a", (Span(100, 200), Span(500, 520)))
RANKING = [
    chunk("b", 0, 300, 1),  # wrong document
    chunk("a", 150, 250, 2),  # covers 150-200: 50 gold characters
    chunk("a", 300, 400, 3),  # right document, no gold
    chunk("a", 180, 260, 4),  # overlaps the second result; adds nothing new (180-200)
    chunk("a", 490, 530, 5),  # covers 500-520: 20 gold characters
]
# Passages of the index that overlap gold: three retrieved ones plus one never retrieved.
RELEVANT = frozenset({"a:150-250", "a:180-260", "a:490-530", "a:90-160"})
SAMPLE = RetrievalSample(QUESTION, RANKING, RELEVANT)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("recall@1", 0.0),
        ("recall@2", 1 / 4),
        ("recall@5", 3 / 4),
        ("precision@2", 1 / 2),
        ("precision@5", 3 / 5),
        ("precision@10", 3 / 10),
        ("hit@1", 0.0),
        ("hit@2", 1.0),
        ("mrr@1", 0.0),
        ("mrr@5", 1 / 2),
        (
            "ndcg@5",
            (1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)),
        ),
        ("ndcg@1", 0.0),
        ("char_recall@1", 0.0),
        ("char_recall@2", 50 / 120),
        ("char_recall@4", 50 / 120),
        ("char_recall@5", 70 / 120),
        # retrieved characters: b 300; a 150-250 -> 100; gold among them 50
        ("char_precision@2", 50 / 400),
        # a: 150-260 (110) + 300-400 (100) + 490-530 (40) = 250, plus b 300; gold 70
        ("char_precision@5", 70 / 550),
    ],
)
def test_exact_values(spec: str, expected: float) -> None:
    metric = get_metric(spec)
    assert metric.name == spec
    assert metric.score(SAMPLE) == pytest.approx(expected)


def test_no_results_and_no_relevant_passages_score_zero() -> None:
    empty = RetrievalSample(QUESTION, [], RELEVANT)
    nothing_relevant = RetrievalSample(QUESTION, RANKING[:1], frozenset())
    for name in metrics.names():
        assert get_metric(f"{name}@5").score(empty) == 0.0
        assert get_metric(f"{name}@5").score(nothing_relevant) == 0.0


def test_default_k_and_bad_specifications() -> None:
    assert get_metric("recall").name == "recall@10"
    with pytest.raises(ConfigError, match="k must be a positive integer"):
        get_metric("recall@0")
    with pytest.raises(ConfigError, match="unknown metric 'bleu'; allowed: char_precision"):
        get_metric("bleu@5")


def test_span_helpers() -> None:
    assert merge_spans([Span(5, 9), Span(0, 3), Span(3, 4), Span(20, 21)]) == [
        Span(0, 4),
        Span(5, 9),
        Span(20, 21),
    ]
    assert overlap_size([Span(0, 10), Span(5, 15)], [Span(8, 12), Span(100, 110)]) == 4


def test_rank_metrics_agree_with_ranx() -> None:
    ranx = pytest.importorskip("ranx")
    rng = random.Random(7)
    samples = []
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for number in range(40):
        gold = (Span(100, 160),)
        question = Question(f"q{number}", "?", "a", gold)
        candidates = [chunk("a", start, start + 50, 1) for start in range(0, 400, 25)]
        candidates += [chunk("b", start, start + 50, 1) for start in range(0, 200, 50)]
        rng.shuffle(candidates)
        ranking = [
            RetrievedChunk(c.document_id, c.span, "", 1.0 / position, position)
            for position, c in enumerate(candidates[: rng.randint(1, 12)], start=1)
        ]
        relevant = frozenset(
            c.key for c in candidates if c.document_id == "a" and c.span.overlaps(gold[0])
        )
        samples.append(RetrievalSample(question, ranking, relevant))
        qrels[question.id] = dict.fromkeys(relevant, 1)
        run[question.id] = {c.key: c.score for c in ranking}

    names = {
        "recall": "recall",
        "precision": "precision",
        "ndcg": "ndcg",
        "mrr": "mrr",
        "hit": "hit_rate",
    }
    for ours, theirs in names.items():
        for k in (1, 3, 10):
            mean = sum(get_metric(f"{ours}@{k}").score(s) for s in samples) / len(samples)
            reference = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), f"{theirs}@{k}")
            assert mean == pytest.approx(float(reference), abs=1e-9), f"{ours}@{k}"
