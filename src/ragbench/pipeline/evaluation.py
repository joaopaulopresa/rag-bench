"""Scores what a pipeline retrieves and what it answers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.components.metrics import get_metric
from ragbench.core.interfaces import Judge, Metric
from ragbench.core.types import Question, RetrievalSample, Usage, Verdict
from ragbench.pipeline.graph import Pipeline

DEFAULT_METRICS = ("char_recall@5", "char_precision@5", "recall@5", "ndcg@10", "mrr@10")


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Retrieval metrics of one question.

    Attributes:
        question_id: The question.
        metrics: Value of each metric, keyed by its name (for example ``recall@5``).
        retrieved: Keys of the retrieved passages, best first.
        usage: Cost and time of the retrieval (query embedding and reranking).
    """

    question_id: str
    metrics: dict[str, float]
    retrieved: tuple[str, ...]
    usage: Usage


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Retrieval metrics of a set of questions.

    Attributes:
        scores: One entry per question, in input order.
    """

    scores: tuple[RetrievalScore, ...]

    def mean(self, metric: str) -> float:
        """Return the mean of ``metric`` over the questions (0.0 for no questions)."""
        if not self.scores:
            return 0.0
        return sum(score.metrics[metric] for score in self.scores) / len(self.scores)

    def means(self) -> dict[str, float]:
        """Return the mean of every metric."""
        names = list(self.scores[0].metrics) if self.scores else []
        return {name: self.mean(name) for name in names}


async def score_question(
    pipeline: Pipeline, question: Question, metrics: Sequence[Metric]
) -> RetrievalScore:
    """Retrieve for one question and compute ``metrics`` against its gold spans."""
    answer = await pipeline.retrieve(question.text)
    sample = RetrievalSample(question, answer.chunks, await pipeline.relevant_keys(question))
    return RetrievalScore(
        question_id=question.id,
        metrics={metric.name: metric.score(sample) for metric in metrics},
        retrieved=tuple(chunk.key for chunk in answer.chunks),
        usage=answer.usage,
    )


async def evaluate_retrieval(
    pipeline: Pipeline,
    questions: Sequence[Question],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    concurrency: int = 8,
) -> RetrievalEvaluation:
    """Measure how well ``pipeline`` retrieves the gold evidence of ``questions``.

    The pipeline's index must exist (call `Pipeline.index` first). Its ``top_k`` must be at
    least the largest ``k`` among the metrics.

    Args:
        pipeline: The pipeline to evaluate; its generator, if any, is not used.
        questions: Gold questions.
        metrics: Metric specifications such as ``recall@5``.
        concurrency: Questions evaluated at the same time.

    Raises:
        ConfigError: If a metric specification is invalid.

    Example:
        >>> evaluation = await evaluate_retrieval(pipeline, corpus.questions)  # doctest: +SKIP
        >>> evaluation.mean("char_recall@5")  # doctest: +SKIP
        0.82
    """
    built = [get_metric(spec) for spec in metrics]
    limit = asyncio.Semaphore(concurrency)

    async def one(question: Question) -> RetrievalScore:
        async with limit:
            return await score_question(pipeline, question, built)

    return RetrievalEvaluation(scores=tuple(await asyncio.gather(*(one(q) for q in questions))))


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """One answer to be judged.

    Attributes:
        question: The question asked.
        answer: The generated answer with its ``[n]`` citations.
        gold: Text of the gold evidence.
        context: The passages the generator saw, in citation order.
    """

    question: str
    answer: str
    gold: tuple[str, ...]
    context: tuple[str, ...]


async def judge_answers(
    judge: Judge, items: Sequence[JudgeItem], *, concurrency: int = 8
) -> list[Verdict]:
    """Score ``items`` with ``judge`` and return one verdict per item, in input order.

    Raises:
        ProviderError: If the judge fails on an item.

    Example:
        >>> passage = "The Tenant shall pay rent."
        >>> item = JudgeItem("Who pays?", "The Tenant [1].", (passage,), (passage,))
        >>> verdicts = await judge_answers(judge, [item])  # doctest: +SKIP
    """
    limit = asyncio.Semaphore(concurrency)

    async def one(item: JudgeItem) -> Verdict:
        async with limit:
            return await judge.judge(
                question=item.question, answer=item.answer, gold=item.gold, context=item.context
            )

    return list(await asyncio.gather(*(one(item) for item in items)))
