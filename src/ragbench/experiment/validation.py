"""Judge validation: export answers for a person to label, then measure agreement."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragbench.components.judges.llm import RUBRIC
from ragbench.core.errors import ExperimentError
from ragbench.experiment.report import CRITERIA, Agreement, Report
from ragbench.experiment.results import ResultRow, ResultsRepository
from ragbench.experiment.runner import ANSWERS, load_corpus
from ragbench.experiment.runtime import Runtime
from ragbench.experiment.stats import stratified_sample

SHEET_COLUMNS = (
    "config_key",
    "question_id",
    "question",
    "gold_evidence",
    "passages",
    "answer",
    *(f"human_{criterion}" for criterion in CRITERIA),
    "notes",
)
RUBRIC_FILE = "judge_rubric.txt"


def sample_for_labelling(rows: Sequence[ResultRow], size: int, seed: int) -> list[ResultRow]:
    """Pick ``size`` stage-3 results spread evenly over configurations.

    The draw is seeded, so the same run always gives the same sheet.
    """
    by_config: dict[str, list[ResultRow]] = {}
    for row in sorted(rows, key=lambda item: (item.config_key, item.question_id)):
        by_config.setdefault(row.config_key, []).append(row)
    return stratified_sample(by_config, size, seed=seed)


async def write_validation_sheet(
    runtime: Runtime, run_id: str, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a CSV of answers for a person to score, plus the rubric next to it.

    The sheet shows the question, the gold evidence, the passages the generator saw and its
    answer. It does not show the judge's scores or which model wrote the answer, so the person
    labels blind. They fill ``human_correctness``, ``human_faithfulness`` and
    ``human_citation`` with 0, 1 or 2.

    Args:
        runtime: Open dependencies of the run's configuration.
        run_id: The run to sample from.
        path: Where to write the CSV.
        overwrite: Replace an existing file. Off by default so that labels are never lost.

    Returns:
        The path of the CSV.

    Raises:
        ExperimentError: If the run has no answers, the configuration asks for no sample, or
            the file exists and ``overwrite`` is false.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise ExperimentError(f"{target} already exists; it may hold labels, so it is kept")
    config = runtime.config
    rows = await runtime.results.results(run_id, ANSWERS)
    if not rows:
        raise ExperimentError(f"run {run_id} has no stage-3 answers to label")
    if config.answer.validation_sample < 1:
        raise ExperimentError("answer.validation_sample is 0; set it to the number of answers")
    corpus = load_corpus(config.corpus, runtime.resources)
    questions = {question.id: question for question in corpus.questions}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for row in sample_for_labelling(rows, config.answer.validation_sample, config.seed):
            question = questions[row.question_id]
            context = row.data.get("context")
            passages = context if isinstance(context, list) else []
            writer.writerow(
                {
                    "config_key": row.config_key,
                    "question_id": row.question_id,
                    "question": question.text,
                    "gold_evidence": "\n---\n".join(corpus.gold_text(question)),
                    "passages": "\n\n".join(
                        f"[{number}] {text}" for number, text in enumerate(passages, start=1)
                    ),
                    "answer": row.data.get("answer", ""),
                }
            )
    target.with_name(RUBRIC_FILE).write_text(RUBRIC + "\n", encoding="utf-8")
    return target


def read_labels(path: str | Path) -> list[tuple[str, str, Mapping[str, int]]]:
    """Read a labelled sheet.

    Rows with all three human columns empty are skipped, so a partly labelled sheet works.

    Returns:
        ``(config_key, question_id, {criterion: score})`` for every labelled row.

    Raises:
        ExperimentError: If the file cannot be read, lacks a column, or a row is labelled
            partly or with something other than 0, 1 or 2. The message names the row.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError as error:
        raise ExperimentError(f"cannot read {path}: {error}") from error
    labels: list[tuple[str, str, Mapping[str, int]]] = []
    for number, row in enumerate(rows, start=2):
        try:
            values = [(row[f"human_{criterion}"] or "").strip() for criterion in CRITERIA]
            item = (row["config_key"], row["question_id"])
        except KeyError as error:
            raise ExperimentError(f"{path} has no column {error}") from None
        if not any(values):
            continue
        if not all(value in {"0", "1", "2"} for value in values):
            raise ExperimentError(
                f"{path} line {number}: every human_* column must be 0, 1 or 2, got {values}"
            )
        scores = {criterion: int(value) for criterion, value in zip(CRITERIA, values, strict=True)}
        labels.append((item[0], item[1], scores))
    return labels


async def import_labels(
    results: ResultsRepository, run_id: str, path: str | Path
) -> tuple[Agreement, ...]:
    """Store the human labels of a sheet and return judge-versus-human agreement.

    Raises:
        ExperimentError: If the sheet is malformed, holds no labels, or refers to answers that
            are not part of the run.
    """
    labels = read_labels(path)
    if not labels:
        raise ExperimentError(f"{path} holds no labels yet")
    known = {(row.config_key, row.question_id) for row in await results.results(run_id, ANSWERS)}
    unknown = [(key, question) for key, question, _ in labels if (key, question) not in known]
    if unknown:
        raise ExperimentError(f"{path}: {unknown[0]} is not an answer of run {run_id}")
    await results.save_labels(run_id, labels)
    return (await Report.load(results, run_id)).agreement
