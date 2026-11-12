"""The ``ragbench`` command."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, TypeVar

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ragbench.components.caches import MemoryCache
from ragbench.components.generators import OpenRouterGenerator
from ragbench.components.loaders import FolderLoader
from ragbench.components.postgres import Database
from ragbench.core import registries
from ragbench.core.config import DatabaseConfig, ExperimentConfig, load_config
from ragbench.core.errors import RagBenchError
from ragbench.core.registry import Registry
from ragbench.core.resources import Resources
from ragbench.core.types import Usage
from ragbench.experiment.questions import generate_questions, write_question_file
from ragbench.experiment.report import Agreement, Report
from ragbench.experiment.results import ResultsRepository, RunRecord
from ragbench.experiment.runner import (
    STAGE_TITLES,
    RunSummary,
    run_experiment,
    validate_components,
)
from ragbench.experiment.runtime import open_runtime
from ragbench.experiment.validation import RUBRIC_FILE, import_labels, write_validation_sheet

T = TypeVar("T")

app = typer.Typer(
    help="Reproducible RAG benchmark on PostgreSQL and OpenRouter.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()

ConfigPath = Annotated[Path, typer.Argument(help="Experiment YAML file.")]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a value: --set answer.questions=50. Repeatable."),
]
Limit = Annotated[int | None, typer.Option("--limit", min=1, help="Use only this many questions.")]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Show log messages.")]


def read_env_file(path: Path = Path(".env")) -> dict[str, str]:
    """Return the ``NAME=value`` pairs of a ``.env`` file; an absent file gives no pairs."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("'\"")
    return values


def environment() -> dict[str, str]:
    """Return the process environment on top of ``.env`` (real variables win)."""
    return {**read_env_file(), **os.environ}


def _run(coroutine: Coroutine[object, object, T]) -> T:
    try:
        return asyncio.run(coroutine)
    except RagBenchError as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error


def _load(path: Path, overrides: list[str] | None, limit: int | None) -> ExperimentConfig:
    items = list(overrides or [])
    if limit is not None:
        items.append(f"limit={limit}")
    try:
        config = load_config(path, env=environment(), overrides=items)
        validate_components(config)
    except RagBenchError as error:
        console.print(f"[red]error:[/red] {error}")
        raise typer.Exit(code=1) from error
    return config


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


class _RichProgress:
    """Renders run progress as one bar per stage."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task: TaskID | None = None

    def start(self, title: str, total: int) -> None:
        if self._task is not None:
            self._progress.update(self._task, completed=self._progress.tasks[self._task].total)
        self._task = self._progress.add_task(title, total=total)

    def advance(self, steps: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, steps)

    def note(self, message: str) -> None:
        self._progress.console.print(message)


def _summary_table(summary: RunSummary) -> Table:
    table = Table(title=f"Run {summary.run_id}", title_justify="left")
    for column in ("Stage", "Configurations", "Evaluated now", "Reused", "Failed"):
        table.add_column(column, justify="left" if column == "Stage" else "right")
    table.add_column("Cost when first made", justify="right")
    table.add_column("Paid now", justify="right")
    for stage, totals in sorted(summary.stages.items()):
        table.add_row(
            f"{stage} {STAGE_TITLES[stage]}",
            str(totals.configurations),
            str(totals.evaluated),
            str(totals.reused),
            str(totals.failed),
            f"${totals.cost_usd:.4f}",
            f"${totals.paid_usd:.4f}",
        )
    table.add_section()
    table.add_row(
        "Total",
        "",
        "",
        "",
        str(summary.failed),
        f"${summary.cost_usd:.4f}",
        f"${summary.paid_usd:.4f}",
    )
    return table


@app.command()
def run(
    config_path: ConfigPath,
    limit: Limit = None,
    overrides: Overrides = None,
    verbose: Verbose = False,
) -> None:
    """Run an experiment. Interrupted runs continue; finished runs cost nothing to repeat."""
    _logging(verbose)
    config = _load(config_path, overrides, limit)

    async def main() -> RunSummary:
        columns = (
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        async with open_runtime(config, env=environment()) as runtime:
            with Progress(*columns, console=console) as progress:
                return await run_experiment(runtime, _RichProgress(progress))

    summary = _run(main())
    console.print(_summary_table(summary))
    console.print(f"Finished in {summary.seconds:.1f}s.")
    if summary.failed:
        console.print(
            f"[yellow]{summary.failed} questions failed at the provider; "
            "run the same command again to retry them.[/yellow]"
        )
        raise typer.Exit(code=2)
    console.print(f"Report: [bold]ragbench report {summary.run_id}[/bold]")


@app.command()
def config(
    config_path: ConfigPath,
    limit: Limit = None,
    overrides: Overrides = None,
) -> None:
    """Validate a configuration and print it fully resolved, with its run id. Costs nothing."""
    loaded = _load(config_path, overrides, limit)
    console.print(yaml.safe_dump(loaded.resolved(), sort_keys=False), end="", markup=False)
    console.print(f"run id: [bold]{loaded.run_id}[/bold]  fingerprint: {loaded.fingerprint()}")


@app.command()
def components() -> None:
    """List the registered implementations of every extension point."""
    import ragbench.components  # noqa: F401, PLC0415  (registers the built-ins)

    table = Table(title="Registered components", title_justify="left")
    table.add_column("Extension point")
    table.add_column("Entry-point group")
    table.add_column("Names")
    for value in vars(registries).values():
        if isinstance(value, Registry):
            table.add_row(value.kind, value.group, ", ".join(value.names()))
    console.print(table)


def _stored_config(run_config: object) -> ExperimentConfig:
    """Rebuild the configuration saved with a run, pointing at the current database."""
    data = dict(run_config) if isinstance(run_config, dict) else {}
    data["database"] = _database().model_dump(by_alias=True)
    return ExperimentConfig.model_validate(data)


_DATABASE_VARIABLES = (("url", "RAGBENCH_DATABASE__URL"), ("schema", "RAGBENCH_DATABASE__SCHEMA"))


def _database() -> DatabaseConfig:
    """Return the database settings from ``RAGBENCH_DATABASE__*`` or the defaults."""
    env = environment()
    values = {
        key: env[name]
        for key, name in (
            ("url", "RAGBENCH_DATABASE__URL"),
            ("schema", "RAGBENCH_DATABASE__SCHEMA"),
        )
        if env.get(name)
    }
    return DatabaseConfig.model_validate(values)


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Run id printed by `ragbench run`.")],
    out: Annotated[Path | None, typer.Option(help="Output folder.")] = None,
) -> None:
    """Write the HTML report and the leaderboard CSVs of a run from its stored results."""

    async def main() -> tuple[Report, Path]:
        database = Database(_database().url, _database().schema_name)
        try:
            loaded = await Report.load(ResultsRepository(database), run_id)
        finally:
            await database.aclose()
        return loaded, loaded.write(out or Path("reports") / run_id)

    loaded, page = _run(main())
    for title, frame, columns in (
        ("Stage 1 - Retrieval (top 5)", loaded.retrieval, [loaded.primary_metric]),
        ("Stage 2 - Reranking (top 5)", loaded.reranking, [loaded.primary_metric, "gain"]),
        (
            "Stage 3 - Answers (top 5)",
            loaded.answers,
            ["correctness", "faithfulness", "citation", "cost_per_1k_answers_usd"],
        ),
    ):
        if not frame.height:
            continue
        table = Table(title=title, title_justify="left")
        table.add_column("Configuration")
        for column in columns:
            table.add_column(column, justify="right")
        for row in frame.head(5).iter_rows(named=True):
            table.add_row(row["configuration"], *[f"{row[column]:.3f}" for column in columns])
        console.print(table)
    if loaded.recommendation:
        console.print(f"Recommended: [bold]{loaded.recommendation.display}[/bold]")
        console.print(loaded.recommendation.reason)
    console.print(f"Report written to [bold]{page}[/bold]")


@app.command()
def runs() -> None:
    """List the runs stored in the database."""

    async def main() -> list[RunRecord]:
        database = Database(_database().url, _database().schema_name)
        try:
            return await ResultsRepository(database).list_runs()
        finally:
            await database.aclose()

    table = Table(title="Runs", title_justify="left")
    for column in ("Run id", "Started (UTC)", "Finished (UTC)", "Questions", "Corpus"):
        table.add_column(column)
    for record in _run(main()):
        table.add_row(
            record.run_id,
            f"{record.created_at:%Y-%m-%d %H:%M}",
            f"{record.finished_at:%Y-%m-%d %H:%M}" if record.finished_at else "not finished",
            str(record.corpus.get("questions", "")),
            str(record.corpus.get("name", "")),
        )
    console.print(table)


judge_app = typer.Typer(help="Validate the judge against human labels.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")


@judge_app.command("sheet")
def judge_sheet(
    run_id: Annotated[str, typer.Argument(help="Run to sample answers from.")],
    out: Annotated[Path | None, typer.Option(help="CSV to write.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing sheet.")] = False,
) -> None:
    """Write a CSV of answers for a person to score blind with the judge's rubric."""

    async def main() -> Path:
        database = Database(_database().url, _database().schema_name)
        try:
            record = await ResultsRepository(database).get_run(run_id)
        finally:
            await database.aclose()
        async with open_runtime(_stored_config(record.config), env=environment()) as runtime:
            target = out or Path("runs") / run_id / "judge_validation.csv"
            return await write_validation_sheet(runtime, run_id, target, overwrite=overwrite)

    path = _run(main())
    console.print(f"Sheet written to [bold]{path}[/bold] (rubric: {path.with_name(RUBRIC_FILE)})")
    console.print(
        "Fill human_correctness, human_faithfulness and human_citation with 0, 1 or 2, then run "
        f"[bold]ragbench judge agreement {run_id} {path}[/bold]"
    )


@judge_app.command("agreement")
def judge_agreement(
    run_id: Annotated[str, typer.Argument(help="Run the sheet was made from.")],
    sheet: Annotated[Path, typer.Argument(help="The labelled CSV.")],
) -> None:
    """Import human labels and print judge-versus-human agreement per criterion."""

    async def main() -> tuple[Agreement, ...]:
        database = Database(_database().url, _database().schema_name)
        try:
            return await import_labels(ResultsRepository(database), run_id, sheet)
        finally:
            await database.aclose()

    table = Table(title=f"Judge versus human - {run_id}", title_justify="left")
    table.add_column("Criterion")
    for column in ("Answers", "Raw agreement", "Cohen's kappa", "Weighted kappa"):
        table.add_column(column, justify="right")
    for item in _run(main()):
        table.add_row(
            item.criterion,
            str(item.items),
            f"{item.raw:.1%}",
            f"{item.kappa:.2f}",
            f"{item.weighted_kappa:.2f}",
        )
    console.print(table)
    console.print(f"Labels stored. Regenerate the report: [bold]ragbench report {run_id}[/bold]")


questions_app = typer.Typer(help="Gold questions for your own documents.", no_args_is_help=True)
app.add_typer(questions_app, name="questions")


@questions_app.command("generate")
def questions_generate(
    folder: Annotated[Path, typer.Argument(help="Folder of .txt, .md or .pdf documents.")],
    count: Annotated[int, typer.Option(min=1, help="Passages to write a question for.")] = 20,
    model: Annotated[
        str, typer.Option(help="OpenRouter model that writes the questions.")
    ] = "deepseek/deepseek-v4.1-flash-1017",
    seed: Annotated[int, typer.Option(help="Seed of the passage choice.")] = 0,
    out: Annotated[Path | None, typer.Option(help="File to write.")] = None,
) -> None:
    """Draft questions with their evidence spans, for a person to review before they count."""
    env = environment()
    model = env.get("JUDGE_MODEL") or model
    target = out or folder / "questions.draft.json"

    async def main() -> tuple[int, Usage]:
        resources = Resources(
            cache=MemoryCache(),
            api_keys={name: value for name, value in env.items() if name.endswith("_API_KEY")},
        )
        try:
            corpus = FolderLoader(resources, str(folder)).load()
            generator = OpenRouterGenerator(
                resources,
                model=model,
                max_tokens=400,
                extra_body={"reasoning": {"enabled": False}},
            )
            drafts, usage = await generate_questions(corpus, generator, count=count, seed=seed)
            write_question_file(drafts, target)
        finally:
            for shared in resources.shared_objects():
                close = getattr(shared, "aclose", None)
                if close is not None:
                    await close()
        return len(drafts), usage

    written, usage = _run(main())
    console.print(
        f"{written} draft questions written to [bold]{target}[/bold] (${usage.paid_usd:.4f})."
    )
    console.print(
        'Review each one, set "reviewed": true on those you keep, and save the file as '
        f"[bold]{folder / 'questions.json'}[/bold]."
    )


def main() -> None:
    """Entry point of the ``ragbench`` script."""
    app()
