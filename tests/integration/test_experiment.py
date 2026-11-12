"""The whole experiment offline: three stages, resume, report, judge validation, CLI."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli.app import app
from ragbench.components.postgres import Database
from ragbench.core.config import ExperimentConfig, load_config
from ragbench.core.errors import ExperimentError, ProviderError
from ragbench.core.registries import generators
from ragbench.core.types import Generation, Message
from ragbench.experiment.report import Report
from ragbench.experiment.runner import ANSWERS, RERANKING, RETRIEVAL, run_experiment
from ragbench.experiment.runtime import open_runtime
from ragbench.experiment.validation import import_labels, read_labels, write_validation_sheet

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parent.parent.parent
OFFLINE = ROOT / "configs" / "offline.yaml"


class FlakyGenerator:
    """Fails on questions that mention 'leave' until `healthy` is set."""

    healthy = False

    def __init__(self, resources: object = None) -> None:
        del resources

    @property
    def key(self) -> str:
        return "flaky"

    async def generate(self, messages: Sequence[Message]) -> Generation:
        if (
            not FlakyGenerator.healthy
            and "leave" in messages[-1].content.rsplit("Question:", 1)[-1]
        ):
            raise ProviderError("simulated outage", status_code=503, retryable=True)
        return Generation(text="Thirty days [1].", model="flaky")


generators.register("flaky")(FlakyGenerator)


def offline_config(database: Database, *overrides: str) -> ExperimentConfig:
    return load_config(
        OFFLINE,
        overrides=[
            f"database.url={database.url}",
            f"database.schema={database.schema}",
            f"corpus.params.path={ROOT / 'examples' / 'handbook'}",
            *overrides,
        ],
    )


async def test_three_stages_then_a_second_run_reuses_everything(database: Database) -> None:
    config = offline_config(database)
    async with open_runtime(config) as runtime:
        first = await run_experiment(runtime)
        second = await run_experiment(runtime)
        record = await runtime.results.get_run(first.run_id)
        stage_one = await runtime.results.stage_configs(first.run_id, RETRIEVAL)

    assert first.run_id == config.run_id == second.run_id
    assert first.questions == 12
    # 3 chunkers x (dense + hybrid with one embedder, lexical once) = 9 configurations
    assert {stage: s.configurations for stage, s in first.stages.items()} == {
        RETRIEVAL: 9,
        RERANKING: 2,
        ANSWERS: 2,
    }
    assert {stage: s.evaluated for stage, s in first.stages.items()} == {
        RETRIEVAL: 108,
        RERANKING: 24,
        ANSWERS: 24,
    }
    assert first.failed == 0
    assert all(s.evaluated == 0 for s in second.stages.values())
    assert {stage: s.reused for stage, s in second.stages.items()} == {
        RETRIEVAL: 108,
        RERANKING: 24,
        ANSWERS: 24,
    }
    assert second.paid_usd == 0.0
    assert second.cost_usd == pytest.approx(first.cost_usd), "stored costs are reported again"
    assert record.finished_at is not None
    assert record.corpus["questions_by_source"] == {"handbook": 12}
    assert record.config["name"] == "offline"
    assert len({config.config_key for config in stage_one}) == 9
    assert all(int(str(config.info["chunks"])) > 0 for config in stage_one)


async def test_an_interrupted_run_continues_where_it_stopped(database: Database) -> None:
    config = offline_config(database)
    async with open_runtime(config) as runtime:
        await run_experiment(runtime)
        async with database.connection() as connection:
            await connection.execute(
                f'DELETE FROM "{database.schema}".results '
                "WHERE stage = 1 AND question_id IN ('handbook-0003', 'handbook-0007')"
            )
        resumed = await run_experiment(runtime)

    assert resumed.stages[RETRIEVAL].evaluated == 18  # 2 questions x 9 configurations
    assert resumed.stages[RETRIEVAL].reused == 90
    assert resumed.stages[ANSWERS].evaluated == 0


async def test_limit_takes_a_stratified_subset_and_gets_its_own_run(database: Database) -> None:
    full = offline_config(database)
    limited = offline_config(database, "limit=5", "answer.generators=[]", "rerank.rerankers=[]")
    assert limited.run_id != full.run_id
    async with open_runtime(limited) as runtime:
        summary = await run_experiment(runtime)
    assert summary.questions == 5
    assert set(summary.stages) == {RETRIEVAL}
    assert summary.stages[RETRIEVAL].evaluated == 45


async def test_provider_failures_are_counted_and_retried_by_the_next_run(
    database: Database,
) -> None:
    config = offline_config(database, "answer.generators=[{name: flaky}]", "answer.top_configs=1")
    FlakyGenerator.healthy = False
    async with open_runtime(config) as runtime:
        broken = await run_experiment(runtime)
        unfinished = await runtime.results.get_run(broken.run_id)
        FlakyGenerator.healthy = True
        repaired = await run_experiment(runtime)
        finished = await runtime.results.get_run(broken.run_id)

    assert broken.stages[ANSWERS].failed == 2  # the two questions that mention leave
    assert broken.stages[ANSWERS].evaluated == 10
    assert unfinished.finished_at is None
    assert repaired.stages[ANSWERS].failed == 0
    assert repaired.stages[ANSWERS].evaluated == 2
    assert repaired.stages[ANSWERS].reused == 10
    assert finished.finished_at is not None


async def test_report_tables_recommendation_and_files(database: Database, tmp_path: Path) -> None:
    config = offline_config(database)
    async with open_runtime(config) as runtime:
        summary = await run_experiment(runtime)
        report = await Report.load(runtime.results, summary.run_id)
        with pytest.raises(
            ExperimentError, match="unknown run 'nope'; runs in this database: offline-"
        ):
            await Report.load(runtime.results, "nope")

    assert report.primary_metric == "char_recall@3"
    assert (report.retrieval.height, report.reranking.height, report.answers.height) == (9, 2, 2)
    scores = report.retrieval["char_recall@3"].to_list()
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    top = report.retrieval.row(0, named=True)
    assert top["primary_low"] <= top["char_recall@3"] <= top["primary_high"]
    assert top["questions"] == 12
    assert set(report.reranking["base_key"].to_list()) <= set(
        report.retrieval["config_key"].to_list()
    )
    gain = report.reranking.row(0, named=True)
    assert gain["gain"] == pytest.approx(gain["char_recall@3"] - gain["base"])
    assert report.answers["gap_to_best"][0] == 0.0
    assert report.answers["tied_with_best"][0]
    assert report.agreement == ()
    assert report.recommendation is not None
    assert report.recommendation.stage == ANSWERS
    assert report.recommendation.config_key in report.answers["config_key"].to_list()
    assert report.by_source.columns == ["configuration", "all", "handbook"]

    page = report.write(tmp_path / "report")
    html = page.read_text()
    assert html.startswith("<!doctype html>")
    assert summary.run_id in html
    assert "Recommended configuration" in html
    assert "Judge validation" in html
    assert "Not done yet" in html
    assert "nan" not in html.lower()
    assert "<script" not in html
    assert {path.name for path in page.parent.iterdir()} == {
        "index.html",
        "report.json",
        "retrieval.csv",
        "reranking.csv",
        "answers.csv",
        "by_source.csv",
    }
    facts = json.loads((page.parent / "report.json").read_text())
    assert facts["run_id"] == summary.run_id
    assert facts["configurations"] == {"retrieval": 9, "reranking": 2, "answers": 2}
    assert facts["recommendation"]["config_key"] == report.recommendation.config_key


async def test_judge_validation_sheet_and_agreement(database: Database, tmp_path: Path) -> None:
    config = offline_config(database, "answer.validation_sample=6")
    sheet = tmp_path / "sheet.csv"
    async with open_runtime(config) as runtime:
        summary = await run_experiment(runtime)
        await write_validation_sheet(runtime, summary.run_id, sheet)
        with pytest.raises(ExperimentError, match="already exists; it may hold labels"):
            await write_validation_sheet(runtime, summary.run_id, sheet)
        with pytest.raises(ExperimentError, match="holds no labels yet"):
            await import_labels(runtime.results, summary.run_id, sheet)

        with sheet.open(newline="") as file:
            rows = list(csv.DictReader(file))
        judged = {
            (row.config_key, row.question_id): row.data["scores"]
            for row in await runtime.results.results(summary.run_id, ANSWERS)
        }
        assert len(rows) == 6
        assert len({row["config_key"] for row in rows}) == 2, "spread over both configurations"
        assert all(row["passages"].startswith("[1] ") and row["answer"] for row in rows)
        assert "scores" not in rows[0] and "generator" not in rows[0], "the person labels blind"
        assert "correctness - compare the ANSWER" in (tmp_path / "judge_rubric.txt").read_text()

        # A person who agrees with the judge on every answer but one.
        for position, row in enumerate(rows[:5]):
            scores = judged[(row["config_key"], row["question_id"])]
            assert isinstance(scores, dict)
            for criterion, score in scores.items():
                row[f"human_{criterion}"] = str(score)
            if position == 0:
                row["human_correctness"] = str((scores["correctness"] + 1) % 3)
        with sheet.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        agreement = {
            item.criterion: item
            for item in await import_labels(runtime.results, summary.run_id, sheet)
        }
        report = await Report.load(runtime.results, summary.run_id)

    assert len(read_labels(sheet)) == 5, "the unlabelled sixth row is skipped"
    assert agreement["correctness"].items == 5
    assert agreement["correctness"].raw == pytest.approx(0.8)
    assert agreement["correctness"].kappa < 1.0
    assert agreement["faithfulness"].raw == 1.0
    assert agreement["faithfulness"].kappa == 1.0
    assert report.agreement == tuple(agreement.values())
    html = report.write(tmp_path / "report").read_text()
    assert "Weighted kappa" in html and "Not done yet" not in html

    rows[0]["human_citation"] = "5"
    with sheet.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ExperimentError, match="line 2: every human_\\* column must be 0, 1 or 2"):
        read_labels(sheet)


def test_command_line_runs_reports_and_lists(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAGBENCH_DATABASE__URL", database.url)
    monkeypatch.setenv("RAGBENCH_DATABASE__SCHEMA", database.schema)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(ROOT)
    runner = CliRunner()

    listed = runner.invoke(app, ["components"])
    assert listed.exit_code == 0
    assert "ragbench.chunkers" in listed.output and "parent_child" in listed.output

    checked = runner.invoke(app, ["config", str(OFFLINE), "--limit", "6"])
    assert checked.exit_code == 0
    assert "limit: 6" in checked.output and "run id: offline-" in checked.output

    wrong = runner.invoke(app, ["config", str(OFFLINE), "--set", "retrieval.chunkers.0.name=nope"])
    assert wrong.exit_code == 1
    assert "unknown chunker 'nope'" in wrong.output

    ran = runner.invoke(app, ["run", str(OFFLINE), "--limit", "6"])
    assert ran.exit_code == 0, ran.output
    run_id = load_config(OFFLINE, overrides=["limit=6"]).run_id
    assert f"ragbench report {run_id}" in ran.output

    out = tmp_path / "report"
    reported = runner.invoke(app, ["report", run_id, "--out", str(out)])
    assert reported.exit_code == 0, reported.output
    assert "Recommended:" in reported.output
    assert (out / "index.html").exists()

    missing = runner.invoke(app, ["report", "nope"])
    assert missing.exit_code == 1
    assert "unknown run 'nope'" in missing.output

    runs = runner.invoke(app, ["runs"])
    assert run_id in runs.output


async def test_reranking_can_run_on_a_subsample_with_one_winner_per_chunker(
    database: Database,
) -> None:
    config = offline_config(
        database,
        "rerank.questions=6",
        "rerank.top_configs=3",
        "rerank.diversify=chunker",
        "answer.diversify=chunker",
        "answer.top_configs=2",
    )
    async with open_runtime(config) as runtime:
        summary = await run_experiment(runtime)
        report = await Report.load(runtime.results, summary.run_id)

    assert summary.stages[RERANKING].configurations == 3
    assert summary.stages[RERANKING].evaluated == 18  # 3 configurations x 6 questions
    assert sorted(report.reranking["chunker"].to_list()) == ["context-header", "fixed", "recursive"]
    assert report.reranking["questions"].to_list() == [6, 6, 6]
    assert report.answers["chunker"].n_unique() == 2
    gain = report.reranking.row(0, named=True)
    assert gain["gain"] == pytest.approx(gain["char_recall@3"] - gain["base"]), "same 6 questions"
