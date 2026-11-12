"""Runs an experiment in three stages, each keeping only the winners of the one before.

1. Retrieval: every chunker x embedder x search mode, without a reranker.
2. Reranking: the best stage-1 configurations, each with every reranker.
3. Answers: the best retrieval configurations, each with every generator, scored by the judge.

Every result is stored per question as soon as it exists, so an interrupted run continues where
it stopped, and a finished run started again does nothing and costs nothing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ragbench.components.metrics import get_metric
from ragbench.core.config import ComponentConfig, ExperimentConfig, PipelineConfig
from ragbench.core.errors import ExperimentError, ProviderError
from ragbench.core.hashing import stable_hash
from ragbench.core.interfaces import Judge, Metric
from ragbench.core.registries import (
    chunkers,
    embedders,
    generators,
    judges,
    loaders,
    rerankers,
)
from ragbench.core.resources import Resources
from ragbench.core.types import Corpus, Question, RetrievalSample, Usage
from ragbench.experiment.runtime import Runtime
from ragbench.experiment.stats import stratified_sample
from ragbench.pipeline.graph import Pipeline, build_pipeline

RETRIEVAL, RERANKING, ANSWERS = 1, 2, 3
STAGE_TITLES = {RETRIEVAL: "Retrieval", RERANKING: "Reranking", ANSWERS: "Answers"}

logger = logging.getLogger(__name__)


class Progress(Protocol):
    """Receives progress events of a run. The command line renders them; tests record them."""

    def start(self, title: str, total: int) -> None:
        """A unit of work with ``total`` steps begins."""
        ...

    def advance(self, steps: int = 1) -> None:
        """``steps`` steps of the current unit are done."""
        ...

    def note(self, message: str) -> None:
        """Something worth telling the user happened."""
        ...


class _Silent:
    def start(self, title: str, total: int) -> None:
        del title, total

    def advance(self, steps: int = 1) -> None:
        del steps

    def note(self, message: str) -> None:
        del message


@dataclass(slots=True)
class StageSummary:
    """Totals of one stage of a run.

    Attributes:
        configurations: Configurations evaluated in the stage.
        evaluated: Question results computed now.
        reused: Question results that were already stored.
        failed: Questions whose provider call failed; they are retried by the next run.
        cost_usd: What the stage's model calls cost when they were first made.
        paid_usd: What was actually paid now (zero for cached calls).
    """

    configurations: int = 0
    evaluated: int = 0
    reused: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    paid_usd: float = 0.0

    def add(self, usage: Usage) -> None:
        """Add the cost of one call or one group of calls."""
        self.cost_usd += usage.cost_usd
        self.paid_usd += usage.paid_usd


@dataclass(slots=True)
class RunSummary:
    """Outcome of `run_experiment`.

    Attributes:
        run_id: Identifier under which results are stored; pass it to the report.
        fingerprint: Hash of the settings that decide the results.
        questions: Questions used in the retrieval stages.
        seconds: Wall-clock duration of this invocation.
        stages: Totals per stage number (1, 2, 3).
    """

    run_id: str
    fingerprint: str
    questions: int = 0
    seconds: float = 0.0
    stages: dict[int, StageSummary] = field(default_factory=dict)

    @property
    def paid_usd(self) -> float:
        """Money spent by this invocation."""
        return sum(stage.paid_usd for stage in self.stages.values())

    @property
    def cost_usd(self) -> float:
        """What all model calls of the run cost when first made."""
        return sum(stage.cost_usd for stage in self.stages.values())

    @property
    def failed(self) -> int:
        """Questions left without a result because a provider call failed."""
        return sum(stage.failed for stage in self.stages.values())


def validate_components(config: ExperimentConfig) -> None:
    """Check every component name and parameter of ``config`` against the registries.

    Nothing is built and nothing is called, so a mistake is found before any money is spent.

    Raises:
        ConfigError: If a name is not registered or a parameter does not fit. The message names
            the component and lists what is allowed.
    """
    import ragbench.components  # noqa: F401, PLC0415  (registers the built-ins)

    loaders.validate(config.corpus.name, config.corpus.params)
    for component in config.retrieval.chunkers:
        chunkers.validate(component.name, component.params)
    for component in config.retrieval.embedders:
        embedders.validate(component.name, component.params)
    for metric in config.retrieval.metrics:
        get_metric(metric)
    for component in config.rerank.rerankers:
        rerankers.validate(component.name, component.params)
    for component in config.answer.generators:
        generators.validate(component.name, component.params)
    if config.answer.judge is not None:
        judges.validate(config.answer.judge.name, config.answer.judge.params)


def load_corpus(component: ComponentConfig, resources: Resources) -> Corpus:
    """Load the corpus described by ``component`` (a registered loader plus parameters).

    Raises:
        ConfigError: If the loader name or its parameters are invalid.
        DatasetError: If the data cannot be downloaded, read or verified.

    Example:
        >>> corpus = load_corpus(
        ...     ComponentConfig(name="folder", params={"path": "docs/"}), resources
        ... )  # doctest: +SKIP
    """
    import ragbench.components  # noqa: F401, PLC0415  (registers the built-ins)

    return loaders.create(component.name, resources, component.params).load()


def select_questions(corpus: Corpus, limit: int | None, seed: int) -> list[Question]:
    """Return the questions of a run: all of them, or ``limit`` spread evenly over sources."""
    questions = list(corpus.questions)
    if limit is None or limit >= len(questions):
        return questions
    by_source: dict[str, list[Question]] = {}
    for question in questions:
        by_source.setdefault(question.source, []).append(question)
    return stratified_sample(by_source, limit, seed=seed)


def collection_name(corpus: Corpus) -> str:
    """Return a name that identifies the corpus content, used to name indexes."""
    content = stable_hash(sorted((d.id, len(d.text)) for d in corpus.documents), length=8)
    revision = f"@{corpus.revision[:12]}" if corpus.revision else ""
    return f"{corpus.name}{revision}#{content}"


def rank_configurations(
    rows: Mapping[str, Mapping[str, float]], questions: Sequence[str]
) -> list[tuple[str, float]]:
    """Order configurations by their mean score over ``questions``, best first.

    Args:
        rows: Score per question id, by configuration key.
        questions: The question ids every configuration is averaged over; a missing score
            counts as zero.

    Returns:
        ``(config_key, mean)`` pairs. Ties keep the alphabetical order of the keys.
    """
    if not questions:
        return []
    means = {
        key: sum(scores.get(question, 0.0) for question in questions) / len(questions)
        for key, scores in rows.items()
    }
    return sorted(means.items(), key=lambda item: (-item[1], item[0]))


class Experiment:
    """One experiment: a configuration plus the live dependencies to run it.

    Args:
        runtime: Open cache, store and result tables.
        progress: Receiver of progress events.
    """

    def __init__(self, runtime: Runtime, progress: Progress | None = None) -> None:
        self._runtime = runtime
        self._config = runtime.config
        self._progress: Progress = progress or _Silent()
        self._metrics: list[Metric] = [get_metric(spec) for spec in self._config.retrieval.metrics]
        self._limit = asyncio.Semaphore(self._config.provider.max_concurrency)
        self.run_id = self._config.run_id

    async def run(self) -> RunSummary:
        """Run every stage, skipping what is already stored.

        Raises:
            ConfigError: If a component name or parameter is invalid.
            DatasetError: If the corpus cannot be loaded.
            ExperimentError: If the corpus has no questions.
        """
        started = time.perf_counter()
        config = self._config
        validate_components(config)
        results = self._runtime.results
        await results.start_run(self.run_id, config.name, config.fingerprint(), config.resolved())

        corpus = load_corpus(config.corpus, self._runtime.resources)
        questions = select_questions(corpus, config.limit, config.seed)
        if not questions:
            raise ExperimentError(f"corpus {corpus.name!r} has no gold questions to evaluate")
        sources: dict[str, int] = {}
        for question in questions:
            sources[question.source] = sources.get(question.source, 0) + 1
        await results.set_corpus(
            self.run_id,
            {
                "name": corpus.name,
                "revision": corpus.revision,
                "licences": dict(corpus.licences),
                "documents": len(corpus.documents),
                "questions": len(questions),
                "questions_by_source": sources,
            },
        )
        summary = RunSummary(self.run_id, config.fingerprint(), questions=len(questions))
        self._progress.note(
            f"run {self.run_id}: {len(questions)} questions, {len(corpus.documents)} documents"
        )

        retrieval_specs = self._retrieval_specs()
        summary.stages[RETRIEVAL] = await self._retrieval_stage(
            RETRIEVAL, retrieval_specs, corpus, questions
        )
        ranking = await self._ranking([RETRIEVAL], questions)

        candidates = list(retrieval_specs)
        if config.rerank.rerankers:
            winners = [key for key, _ in ranking[: config.rerank.top_configs]]
            by_key = {spec.key: spec for spec in retrieval_specs}
            rerank_specs = [
                by_key[key].model_copy(
                    update={"reranker": reranker, "candidates": config.rerank.candidates}
                )
                for key in winners
                for reranker in config.rerank.rerankers
            ]
            summary.stages[RERANKING] = await self._retrieval_stage(
                RERANKING, rerank_specs, corpus, questions
            )
            candidates += rerank_specs
            ranking = await self._ranking([RETRIEVAL, RERANKING], questions)

        if config.answer.generators:
            by_key = {spec.key: spec for spec in candidates}
            best = [by_key[key] for key, _ in ranking[: config.answer.top_configs]]
            answer_specs = [
                spec.model_copy(update={"generator": generator, "top_k": config.answer.context_k})
                for spec in best
                for generator in config.answer.generators
            ]
            by_source: dict[str, list[Question]] = {}
            for question in questions:
                by_source.setdefault(question.source, []).append(question)
            sample = stratified_sample(by_source, config.answer.questions, seed=config.seed)
            summary.stages[ANSWERS] = await self._answer_stage(answer_specs, corpus, sample)

        if not summary.failed:
            await results.finish_run(self.run_id)
        summary.seconds = time.perf_counter() - started
        return summary

    def _retrieval_specs(self) -> list[PipelineConfig]:
        """Return the stage-1 grid.

        Lexical search ignores embeddings, so it runs once per chunker, not once per embedder.
        """
        config = self._config.retrieval
        specs = []
        for chunker in config.chunkers:
            for mode in config.search_modes:
                for embedder in [None] if mode == "lexical" else config.embedders:
                    specs.append(
                        PipelineConfig(
                            chunker=chunker,
                            embedder=embedder,
                            search_mode=mode,
                            top_k=config.top_k,
                        )
                    )
        return specs

    async def _ranking(
        self, stages: Sequence[int], questions: Sequence[Question]
    ) -> list[tuple[str, float]]:
        primary = self._config.retrieval.primary_metric
        rows: dict[str, dict[str, float]] = {}
        for stage in stages:
            for row in await self._runtime.results.results(self.run_id, stage):
                metrics = row.data.get("metrics")
                if isinstance(metrics, dict) and primary in metrics:
                    rows.setdefault(row.config_key, {})[row.question_id] = float(metrics[primary])
        return rank_configurations(rows, [question.id for question in questions])

    async def _prepare(
        self, stage: int, position: int, spec: PipelineConfig, corpus: Corpus, summary: StageSummary
    ) -> Pipeline:
        pipeline = build_pipeline(
            spec,
            resources=self._runtime.resources,
            store=self._runtime.store,
            collection=collection_name(corpus),
        )
        report = await pipeline.index(corpus.documents)
        summary.add(report.usage)
        info: dict[str, object] = {"index": pipeline.index_name, "chunks": report.chunks}
        if report.built:
            info["index_cost_usd"] = report.usage.cost_usd
        await self._runtime.results.save_stage_config(
            self.run_id, stage, spec.key, position, spec.model_dump(mode="json"), info
        )
        return pipeline

    async def _evaluate(
        self,
        stage: int,
        spec: PipelineConfig,
        questions: Sequence[Question],
        summary: StageSummary,
        work: Callable[[Question], Awaitable[tuple[dict[str, object], Usage]]],
    ) -> None:
        results = self._runtime.results
        done = await results.done_questions(self.run_id, stage, spec.key)
        pending = [question for question in questions if question.id not in done]
        reused = len(questions) - len(pending)
        summary.reused += reused
        self._progress.advance(reused)

        async def one(question: Question) -> None:
            async with self._limit:
                try:
                    data, usage = await work(question)
                except ProviderError as error:
                    summary.failed += 1
                    logger.warning("%s on %s failed: %s", spec.display, question.id, error)
                    self._progress.advance()
                    return
                await results.save_result(self.run_id, stage, spec.key, question.id, data)
                summary.add(usage)
                summary.evaluated += 1
                self._progress.advance()

        await asyncio.gather(*(one(question) for question in pending))

    async def _retrieval_stage(
        self,
        stage: int,
        specs: Sequence[PipelineConfig],
        corpus: Corpus,
        questions: Sequence[Question],
    ) -> StageSummary:
        summary = StageSummary(configurations=len(specs))
        self._progress.start(
            f"Stage {stage} - {STAGE_TITLES[stage]}: {len(specs)} configurations",
            len(specs) * len(questions),
        )
        for position, spec in enumerate(specs):
            pipeline = await self._prepare(stage, position, spec, corpus, summary)

            async def work(
                question: Question, pipeline: Pipeline = pipeline
            ) -> tuple[dict[str, object], Usage]:
                answer = await pipeline.retrieve(question.text)
                relevant = await self._runtime.store.relevant_keys(pipeline.index_name, question)
                sample = RetrievalSample(question, answer.chunks, relevant)
                data: dict[str, object] = {
                    "metrics": {metric.name: metric.score(sample) for metric in self._metrics},
                    "retrieved": [chunk.key for chunk in answer.chunks],
                    "cost_usd": answer.usage.cost_usd,
                    "latency_s": answer.usage.latency_s,
                }
                return data, answer.usage

            await self._evaluate(stage, spec, questions, summary, work)
        return summary

    async def _answer_stage(
        self, specs: Sequence[PipelineConfig], corpus: Corpus, questions: Sequence[Question]
    ) -> StageSummary:
        judge_config = self._config.answer.judge
        assert judge_config is not None
        judge: Judge = judges.create(
            judge_config.name, self._runtime.resources, judge_config.params
        )
        summary = StageSummary(configurations=len(specs))
        self._progress.start(
            f"Stage {ANSWERS} - {STAGE_TITLES[ANSWERS]}: {len(specs)} configurations",
            len(specs) * len(questions),
        )
        for position, spec in enumerate(specs):
            pipeline = await self._prepare(ANSWERS, position, spec, corpus, summary)

            async def work(
                question: Question, pipeline: Pipeline = pipeline
            ) -> tuple[dict[str, object], Usage]:
                answer = await pipeline.answer(question.text)
                context = [chunk.text for chunk in answer.chunks]
                verdict = await judge.judge(
                    question=question.text,
                    answer=answer.text,
                    gold=corpus.gold_text(question),
                    context=context,
                )
                gold_passages = [
                    number
                    for number, chunk in enumerate(answer.chunks, start=1)
                    if chunk.document_id == question.document_id
                    and any(chunk.span.overlaps(span) for span in question.spans)
                ]
                cited = answer.citations
                generation = answer.generation_usage
                data: dict[str, object] = {
                    "answer": answer.text,
                    "context": context,
                    "context_keys": [chunk.key for chunk in answer.chunks],
                    "citations": list(cited),
                    "invalid_citations": list(answer.invalid_citations),
                    "gold_passages": gold_passages,
                    "gold_in_context": bool(gold_passages),
                    "cited_gold": (
                        sum(number in gold_passages for number in cited) / len(cited)
                        if cited
                        else None
                    ),
                    "scores": verdict.scores(),
                    "rationale": verdict.rationale,
                    "generation_cost_usd": generation.cost_usd,
                    "generation_latency_s": generation.latency_s,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "retrieval_cost_usd": (answer.retrieval_usage + answer.rerank_usage).cost_usd,
                    "retrieval_latency_s": (answer.retrieval_usage + answer.rerank_usage).latency_s,
                    "judge_cost_usd": verdict.usage.cost_usd,
                }
                return data, answer.usage + verdict.usage

            await self._evaluate(ANSWERS, spec, questions, summary, work)
        return summary


async def run_experiment(runtime: Runtime, progress: Progress | None = None) -> RunSummary:
    """Run the experiment configured in ``runtime`` and return its totals.

    The run is identified by its configuration: running the same configuration again continues
    or reuses the stored run, and every model call already made is served from the cache.

    Args:
        runtime: Open dependencies, from `open_runtime`.
        progress: Optional receiver of progress events.

    Returns:
        The run id (for `Report`) and the cost and counts per stage.

    Example:
        >>> config = load_config("configs/demo.yaml", env=os.environ)  # doctest: +SKIP
        >>> async with open_runtime(config, env=os.environ) as runtime:  # doctest: +SKIP
        ...     summary = await run_experiment(runtime)
    """
    return await Experiment(runtime, progress).run()
