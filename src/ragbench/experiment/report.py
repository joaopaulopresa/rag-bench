"""Turns the stored results of a run into leaderboards, intervals and a recommendation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ragbench.core.config import BootstrapConfig, PipelineConfig
from ragbench.core.errors import ExperimentError
from ragbench.experiment.results import ResultRow, ResultsRepository, RunRecord, StageConfig
from ragbench.experiment.runner import ANSWERS, RERANKING, RETRIEVAL
from ragbench.experiment.stats import (
    Interval,
    bootstrap_mean,
    cohen_kappa,
    paired_bootstrap,
    raw_agreement,
)

CRITERIA = ("correctness", "faithfulness", "citation")
MAX_SCORE = 2


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The configuration the report recommends, and why.

    Attributes:
        config_key: Identifier of the configuration.
        display: Its one-line description.
        stage: The stage whose leaderboard it comes from.
        reason: The rule that selected it, in one or two sentences.
    """

    config_key: str
    display: str
    stage: int
    reason: str


@dataclass(frozen=True, slots=True)
class Agreement:
    """Agreement between the judge and human labels on one criterion.

    Attributes:
        criterion: ``correctness``, ``faithfulness`` or ``citation``.
        items: Number of answers labelled by a person.
        raw: Share of answers on which judge and person gave the same score.
        kappa: Cohen's kappa.
        weighted_kappa: Cohen's kappa with quadratic weights (scores are ordered).
    """

    criterion: str
    items: int
    raw: float
    kappa: float
    weighted_kappa: float


@dataclass(frozen=True, slots=True)
class Report:
    """Everything shown in the report of one run.

    Build it with `Report.load`; write it with `Report.write`.

    Attributes:
        run: The stored run (id, fingerprint, configuration, corpus).
        primary_metric: The retrieval metric that ranks configurations.
        retrieval: Stage-1 leaderboard, best first; one row per configuration.
        reranking: Stage-2 leaderboard with the difference to each base configuration.
        answers: Stage-3 leaderboard, best correctness first.
        by_source: Primary metric of the best retrieval configurations per source collection.
        agreement: Judge-versus-human agreement; empty until labels are imported.
        recommendation: The recommended configuration, or ``None`` if the run has no results.
        generated_at: When the report was computed.
    """

    run: RunRecord
    primary_metric: str
    retrieval: pl.DataFrame
    reranking: pl.DataFrame
    answers: pl.DataFrame
    by_source: pl.DataFrame
    agreement: tuple[Agreement, ...]
    recommendation: Recommendation | None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    async def load(cls, results: ResultsRepository, run_id: str) -> Report:
        """Compute the report of ``run_id`` from the stored results.

        Raises:
            ExperimentError: If the run does not exist or has no results yet.

        Example:
            >>> async with open_runtime(config) as runtime:  # doctest: +SKIP
            ...     report = await Report.load(runtime.results, "demo-1a2b3c4d")
            ...     report.write("reports/demo-1a2b3c4d")
        """
        run = await results.get_run(run_id)
        configs = await results.stage_configs(run_id)
        rows = {
            stage: await results.results(run_id, stage) for stage in (RETRIEVAL, RERANKING, ANSWERS)
        }
        if not rows[RETRIEVAL]:
            raise ExperimentError(f"run {run_id} has no results yet; run it first")
        settings = _settings(run.config)
        primary = settings.primary_metric
        retrieval = _retrieval_table(configs, rows[RETRIEVAL], RETRIEVAL, primary, settings)
        reranking = _reranking_table(configs, rows, primary, settings)
        answers = _answers_table(configs, rows[ANSWERS], settings)
        labels = await results.labels(run_id)
        return cls(
            run=run,
            primary_metric=primary,
            retrieval=retrieval,
            reranking=reranking,
            answers=answers,
            by_source=_by_source(retrieval, reranking, rows, primary),
            agreement=_agreement(rows[ANSWERS], labels),
            recommendation=_recommend(retrieval, reranking, answers, primary),
        )

    def write(self, directory: str | Path) -> Path:
        """Write ``index.html``, one CSV per leaderboard and ``report.json`` to ``directory``.

        Returns:
            The path of ``index.html``.
        """
        from ragbench.experiment.report_html import render_html  # noqa: PLC0415

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in (
            ("retrieval", self.retrieval),
            ("reranking", self.reranking),
            ("answers", self.answers),
            ("by_source", self.by_source),
        ):
            if frame.height:
                frame.write_csv(target / f"{name}.csv")
        (target / "report.json").write_text(
            json.dumps(self.summary(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        page = target / "index.html"
        page.write_text(render_html(self), encoding="utf-8")
        return page

    def summary(self) -> dict[str, object]:
        """Return the headline facts of the report as plain data."""
        recommendation = self.recommendation
        return {
            "run_id": self.run.run_id,
            "fingerprint": self.run.fingerprint,
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "corpus": dict(self.run.corpus),
            "primary_metric": self.primary_metric,
            "configurations": {
                "retrieval": self.retrieval.height,
                "reranking": self.reranking.height,
                "answers": self.answers.height,
            },
            "recommendation": (
                {
                    "config_key": recommendation.config_key,
                    "display": recommendation.display,
                    "stage": recommendation.stage,
                    "reason": recommendation.reason,
                }
                if recommendation
                else None
            ),
            "judge_agreement": [
                {
                    "criterion": item.criterion,
                    "items": item.items,
                    "raw": item.raw,
                    "kappa": item.kappa,
                    "weighted_kappa": item.weighted_kappa,
                }
                for item in self.agreement
            ],
        }


@dataclass(frozen=True, slots=True)
class _Settings:
    primary_metric: str
    bootstrap: BootstrapConfig
    seed: int


def _settings(config: Mapping[str, object]) -> _Settings:
    retrieval = config.get("retrieval")
    bootstrap = config.get("bootstrap")
    seed = config.get("seed")
    return _Settings(
        primary_metric=str(retrieval["primary_metric"]) if isinstance(retrieval, dict) else "",
        bootstrap=BootstrapConfig.model_validate(bootstrap if isinstance(bootstrap, dict) else {}),
        seed=seed if isinstance(seed, int) else 0,
    )


def _interval(values: Sequence[float], settings: _Settings) -> Interval:
    return bootstrap_mean(
        values,
        samples=settings.bootstrap.samples,
        confidence=settings.bootstrap.confidence,
        seed=settings.seed,
    )


def _difference(first: Sequence[float], second: Sequence[float], settings: _Settings) -> Interval:
    return paired_bootstrap(
        first,
        second,
        samples=settings.bootstrap.samples,
        confidence=settings.bootstrap.confidence,
        seed=settings.seed,
    )


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _group(rows: Sequence[ResultRow]) -> dict[str, dict[str, Mapping[str, object]]]:
    grouped: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row.config_key, {})[row.question_id] = row.data
    return grouped


def _spec_columns(config: StageConfig) -> dict[str, object]:
    spec = PipelineConfig.model_validate(config.spec)
    return {
        "config_key": config.config_key,
        "configuration": spec.display,
        "chunker": spec.chunker.display,
        "embedder": spec.embedder.display if spec.embedder else "",
        "search_mode": spec.search_mode,
        "reranker": spec.reranker.display if spec.reranker else "",
        "generator": spec.generator.display if spec.generator else "",
        "chunks": int(_number(config.info.get("chunks"))),
    }


def _metric_scores(data: Mapping[str, Mapping[str, object]], metric: str) -> dict[str, float]:
    scores = {}
    for question, item in data.items():
        metrics = item.get("metrics")
        if isinstance(metrics, dict) and metric in metrics:
            scores[question] = float(metrics[metric])
    return scores


def _retrieval_rows(
    configs: Sequence[StageConfig],
    rows: Sequence[ResultRow],
    stage: int,
    primary: str,
    settings: _Settings,
) -> list[dict[str, object]]:
    grouped = _group(rows)
    table = []
    for config in configs:
        data = grouped.get(config.config_key)
        if config.stage != stage or not data:
            continue
        items = list(data.values())
        names: list[str] = []
        for item in items:
            metrics = item.get("metrics")
            if isinstance(metrics, dict):
                names = list(metrics)
                break
        record = _spec_columns(config)
        record["questions"] = len(items)
        for name in names:
            record[name] = float(np.mean(list(_metric_scores(data, name).values())))
        interval = _interval(list(_metric_scores(data, primary).values()), settings)
        record["primary_low"] = interval.low
        record["primary_high"] = interval.high
        record["cost_per_1k_queries_usd"] = 1000 * float(
            np.mean([_number(item.get("cost_usd")) for item in items])
        )
        record["latency_ms"] = 1000 * float(
            np.median([_number(item.get("latency_s")) for item in items])
        )
        table.append(record)
    return table


def _retrieval_table(
    configs: Sequence[StageConfig],
    rows: Sequence[ResultRow],
    stage: int,
    primary: str,
    settings: _Settings,
) -> pl.DataFrame:
    table = _retrieval_rows(configs, rows, stage, primary, settings)
    if not table:
        return pl.DataFrame()
    return pl.DataFrame(table).sort([primary, "config_key"], descending=[True, False])


def _base_key(config: StageConfig, stage_one: Sequence[StageConfig]) -> str | None:
    def identity(spec: Mapping[str, object]) -> tuple[str, str, str]:
        return (
            json.dumps(spec.get("chunker"), sort_keys=True),
            json.dumps(spec.get("embedder"), sort_keys=True),
            str(spec.get("search_mode")),
        )

    for candidate in stage_one:
        if identity(candidate.spec) == identity(config.spec):
            return candidate.config_key
    return None


def _reranking_table(
    configs: Sequence[StageConfig],
    rows: Mapping[int, Sequence[ResultRow]],
    primary: str,
    settings: _Settings,
) -> pl.DataFrame:
    table = _retrieval_rows(configs, rows[RERANKING], RERANKING, primary, settings)
    if not table:
        return pl.DataFrame()
    stage_one = [config for config in configs if config.stage == RETRIEVAL]
    by_key = {config.config_key: config for config in configs if config.stage == RERANKING}
    base_scores = _group(rows[RETRIEVAL])
    own_scores = _group(rows[RERANKING])
    for record in table:
        key = str(record["config_key"])
        base = _base_key(by_key[key], stage_one)
        record["base_key"] = base or ""
        record["base"] = float("nan")
        record["gain"] = float("nan")
        record["gain_low"] = float("nan")
        record["gain_high"] = float("nan")
        record["significant"] = False
        if base is None or base not in base_scores:
            continue
        mine = _metric_scores(own_scores[key], primary)
        theirs = _metric_scores(base_scores[base], primary)
        common = sorted(set(mine) & set(theirs))
        if not common:
            continue
        gain = _difference([mine[q] for q in common], [theirs[q] for q in common], settings)
        record["base"] = float(np.mean([theirs[q] for q in common]))
        record["gain"] = gain.mean
        record["gain_low"] = gain.low
        record["gain_high"] = gain.high
        record["significant"] = gain.excludes_zero
    return pl.DataFrame(table).sort([primary, "config_key"], descending=[True, False])


def _criterion_scores(data: Mapping[str, Mapping[str, object]], criterion: str) -> dict[str, float]:
    scores = {}
    for question, item in data.items():
        values = item.get("scores")
        if isinstance(values, dict) and criterion in values:
            scores[question] = float(values[criterion]) / MAX_SCORE
    return scores


def _answers_table(
    configs: Sequence[StageConfig], rows: Sequence[ResultRow], settings: _Settings
) -> pl.DataFrame:
    grouped = _group(rows)
    table: list[dict[str, object]] = []
    for config in configs:
        data = grouped.get(config.config_key)
        if config.stage != ANSWERS or not data:
            continue
        items = list(data.values())
        record = _spec_columns(config)
        spec = PipelineConfig.model_validate(config.spec)
        record["retrieval"] = spec.model_copy(update={"generator": None}).display
        record["questions"] = len(items)
        for criterion in CRITERIA:
            record[criterion] = float(np.mean(list(_criterion_scores(data, criterion).values())))
        interval = _interval(list(_criterion_scores(data, "correctness").values()), settings)
        record["correctness_low"] = interval.low
        record["correctness_high"] = interval.high
        record["gold_in_context"] = float(np.mean([bool(i.get("gold_in_context")) for i in items]))
        cited = [_number(i.get("cited_gold")) for i in items if i.get("cited_gold") is not None]
        record["cited_gold"] = float(np.mean(cited)) if cited else float("nan")
        record["answers_without_citation"] = float(np.mean([not i.get("citations") for i in items]))
        costs = [
            _number(i.get("generation_cost_usd")) + _number(i.get("retrieval_cost_usd"))
            for i in items
        ]
        record["cost_per_1k_answers_usd"] = 1000 * float(np.mean(costs))
        latencies = [
            _number(i.get("generation_latency_s")) + _number(i.get("retrieval_latency_s"))
            for i in items
        ]
        record["latency_p50_s"] = float(np.median(latencies))
        record["latency_p95_s"] = float(np.quantile(latencies, 0.95))
        record["prompt_tokens"] = float(np.mean([_number(i.get("prompt_tokens")) for i in items]))
        record["completion_tokens"] = float(
            np.mean([_number(i.get("completion_tokens")) for i in items])
        )
        record["judge_cost_usd"] = float(np.sum([_number(i.get("judge_cost_usd")) for i in items]))
        table.append(record)
    if not table:
        return pl.DataFrame()
    table.sort(key=lambda r: (-_number(r["correctness"]), str(r["config_key"])))
    best = _criterion_scores(grouped[str(table[0]["config_key"])], "correctness")
    for record in table:
        mine = _criterion_scores(grouped[str(record["config_key"])], "correctness")
        common = sorted(set(mine) & set(best))
        gap = _difference([mine[q] for q in common], [best[q] for q in common], settings)
        record["gap_to_best"] = gap.mean
        record["gap_low"] = gap.low
        record["gap_high"] = gap.high
        record["tied_with_best"] = not gap.excludes_zero
    return pl.DataFrame(table)


def _by_source(
    retrieval: pl.DataFrame,
    reranking: pl.DataFrame,
    rows: Mapping[int, Sequence[ResultRow]],
    primary: str,
    top: int = 5,
) -> pl.DataFrame:
    frames = [frame for frame in (retrieval, reranking) if frame.height]
    if not frames:
        return pl.DataFrame()
    columns = ["config_key", "configuration", primary]
    ranked = pl.concat([frame.select(columns) for frame in frames]).sort(primary, descending=True)
    grouped = {**_group(rows[RETRIEVAL]), **_group(rows[RERANKING])}
    table = []
    for key, configuration, overall in ranked.head(top).iter_rows():
        record: dict[str, object] = {"configuration": configuration, "all": overall}
        per_source: dict[str, list[float]] = {}
        for question, score in _metric_scores(grouped[key], primary).items():
            per_source.setdefault(question.rsplit("-", 1)[0], []).append(score)
        for source in sorted(per_source):
            record[source] = float(np.mean(per_source[source]))
        table.append(record)
    return pl.DataFrame(table)


def _agreement(
    rows: Sequence[ResultRow], labels: Mapping[tuple[str, str], Mapping[str, object]]
) -> tuple[Agreement, ...]:
    judged = {(row.config_key, row.question_id): row.data.get("scores") for row in rows}
    result = []
    for criterion in CRITERIA:
        human: list[int] = []
        judge: list[int] = []
        for item, scores in labels.items():
            theirs = judged.get(item)
            if isinstance(theirs, dict) and criterion in theirs and criterion in scores:
                human.append(int(_number(scores[criterion])))
                judge.append(int(theirs[criterion]))
        if human:
            result.append(
                Agreement(
                    criterion=criterion,
                    items=len(human),
                    raw=raw_agreement(judge, human),
                    kappa=cohen_kappa(judge, human),
                    weighted_kappa=cohen_kappa(judge, human, weighted=True),
                )
            )
    return tuple(result)


def _recommend(
    retrieval: pl.DataFrame, reranking: pl.DataFrame, answers: pl.DataFrame, primary: str
) -> Recommendation | None:
    if answers.height:
        tied = answers.filter(pl.col("tied_with_best")).sort(
            ["cost_per_1k_answers_usd", "latency_p50_s"]
        )
        chosen = tied.row(0, named=True)
        best = answers.row(0, named=True)
        if chosen["config_key"] == best["config_key"]:
            reason = (
                f"Highest correctness ({best['correctness']:.3f}), and no configuration that is "
                "statistically tied with it is cheaper."
            )
        else:
            reason = (
                f"Cheapest of the {tied.height} configurations whose correctness is not "
                f"significantly below the best. It scores {chosen['correctness']:.3f} at "
                f"${chosen['cost_per_1k_answers_usd']:.3f} per 1,000 answers; the best scores "
                f"{best['correctness']:.3f} at ${best['cost_per_1k_answers_usd']:.3f}."
            )
        return Recommendation(chosen["config_key"], chosen["configuration"], ANSWERS, reason)
    frames = [
        frame.select(["config_key", "configuration", primary]).with_columns(
            pl.lit(stage).alias("stage")
        )
        for stage, frame in ((RETRIEVAL, retrieval), (RERANKING, reranking))
        if frame.height
    ]
    if not frames:
        return None
    best = pl.concat(frames).sort(primary, descending=True).row(0, named=True)
    return Recommendation(
        best["config_key"],
        best["configuration"],
        best["stage"],
        f"Highest {primary} ({best[primary]:.3f}); the run has no answer stage.",
    )
