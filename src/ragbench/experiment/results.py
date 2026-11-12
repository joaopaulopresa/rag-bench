"""Run metadata and per-question results in PostgreSQL. This is what makes runs resumable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql
from psycopg.types.json import Jsonb

from ragbench.components.postgres import Database
from ragbench.core.errors import ExperimentError

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS {runs} ("
    "run_id text PRIMARY KEY, name text NOT NULL, fingerprint text NOT NULL, "
    "config jsonb NOT NULL, corpus jsonb NOT NULL DEFAULT '{{}}', "
    "created_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz)",
    "CREATE TABLE IF NOT EXISTS {stage_configs} ("
    "run_id text NOT NULL, stage integer NOT NULL, config_key text NOT NULL, "
    "position integer NOT NULL, spec jsonb NOT NULL, info jsonb NOT NULL DEFAULT '{{}}', "
    "PRIMARY KEY (run_id, stage, config_key))",
    "CREATE TABLE IF NOT EXISTS {results} ("
    "run_id text NOT NULL, stage integer NOT NULL, config_key text NOT NULL, "
    "question_id text NOT NULL, data jsonb NOT NULL, "
    "PRIMARY KEY (run_id, stage, config_key, question_id))",
    "CREATE TABLE IF NOT EXISTS {human_labels} ("
    "run_id text NOT NULL, config_key text NOT NULL, question_id text NOT NULL, "
    "labels jsonb NOT NULL, PRIMARY KEY (run_id, config_key, question_id))",
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A stored run.

    Attributes:
        run_id: Identifier: configuration name plus fingerprint prefix.
        fingerprint: Hash of the settings that decide the results.
        config: The fully resolved configuration.
        corpus: Name, revision, licences and sizes of the data used.
        created_at: When the run was first started.
        finished_at: When every stage last completed, or ``None``.
    """

    run_id: str
    fingerprint: str
    config: Mapping[str, object]
    corpus: Mapping[str, object]
    created_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class StageConfig:
    """One configuration evaluated in a stage.

    Attributes:
        stage: 1 (retrieval), 2 (reranking) or 3 (answers).
        config_key: Identifier of the pipeline configuration.
        position: Order in which the configuration was scheduled.
        spec: The `PipelineConfig` as plain data.
        info: Facts gathered while running it (index size, indexing cost).
    """

    stage: int
    config_key: str
    position: int
    spec: Mapping[str, object]
    info: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResultRow:
    """The stored outcome of one configuration on one question."""

    stage: int
    config_key: str
    question_id: str
    data: Mapping[str, object]


class ResultsRepository:
    """Reads and writes runs, stage configurations, results and human labels.

    Args:
        database: The connection pool and schema to use.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._tables = {
            name: database.table(name)
            for name in ("runs", "stage_configs", "results", "human_labels")
        }
        self._ready = False

    def _sql(self, statement: str) -> sql.Composed:
        return sql.SQL(statement).format(**self._tables)

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        async with self._database.connection() as connection:
            for statement in _SCHEMA:
                await connection.execute(self._sql(statement))
        self._ready = True

    async def start_run(
        self, run_id: str, name: str, fingerprint: str, config: Mapping[str, object]
    ) -> None:
        """Create the run, or reopen it if it already exists.

        Raises:
            ExperimentError: If a run with this id exists for a different configuration.
        """
        await self._ensure_schema()
        async with self._database.connection() as connection:
            await connection.execute(
                self._sql(
                    "INSERT INTO {runs} (run_id, name, fingerprint, config) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING"
                ),
                (run_id, name, fingerprint, Jsonb(dict(config))),
            )
        stored = await self.get_run(run_id)
        if stored.fingerprint != fingerprint:
            raise ExperimentError(f"run {run_id} exists with a different configuration")

    async def set_corpus(self, run_id: str, corpus: Mapping[str, object]) -> None:
        """Record which data the run used."""
        async with self._database.connection() as connection:
            await connection.execute(
                self._sql("UPDATE {runs} SET corpus = %s WHERE run_id = %s"),
                (Jsonb(dict(corpus)), run_id),
            )

    async def finish_run(self, run_id: str) -> None:
        """Mark the run as complete now."""
        async with self._database.connection() as connection:
            await connection.execute(
                self._sql("UPDATE {runs} SET finished_at = now() WHERE run_id = %s"), (run_id,)
            )

    async def get_run(self, run_id: str) -> RunRecord:
        """Return the stored run.

        Raises:
            ExperimentError: If there is no such run. The message lists the runs that exist.
        """
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT run_id, fingerprint, config, corpus, created_at, finished_at "
                    "FROM {runs} WHERE run_id = %s"
                ),
                (run_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            known = ", ".join(run.run_id for run in await self.list_runs()) or "(none)"
            raise ExperimentError(f"unknown run {run_id!r}; runs in this database: {known}")
        return _run_record(row)

    async def list_runs(self) -> list[RunRecord]:
        """Return every stored run, newest first."""
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT run_id, fingerprint, config, corpus, created_at, finished_at "
                    "FROM {runs} ORDER BY created_at DESC"
                )
            )
            rows = await cursor.fetchall()
        return [_run_record(row) for row in rows]

    async def save_stage_config(
        self,
        run_id: str,
        stage: int,
        config_key: str,
        position: int,
        spec: Mapping[str, object],
        info: Mapping[str, object],
    ) -> None:
        """Record a configuration of a stage, merging ``info`` into what is stored."""
        async with self._database.connection() as connection:
            await connection.execute(
                self._sql(
                    "INSERT INTO {stage_configs} (run_id, stage, config_key, position, spec, info)"
                    " VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_id, stage, config_key) DO UPDATE SET "
                    "position = EXCLUDED.position, info = {stage_configs}.info || EXCLUDED.info"
                ),
                (run_id, stage, config_key, position, Jsonb(dict(spec)), Jsonb(dict(info))),
            )

    async def stage_configs(self, run_id: str, stage: int | None = None) -> list[StageConfig]:
        """Return the configurations of a run, optionally of one stage, in scheduling order."""
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT stage, config_key, position, spec, info FROM {stage_configs} "
                    "WHERE run_id = %s AND (%s::integer IS NULL OR stage = %s) "
                    "ORDER BY stage, position"
                ),
                (run_id, stage, stage),
            )
            rows = await cursor.fetchall()
        return [
            StageConfig(
                int(str(row[0])), str(row[1]), int(str(row[2])), _mapping(row[3]), _mapping(row[4])
            )
            for row in rows
        ]

    async def done_questions(self, run_id: str, stage: int, config_key: str) -> set[str]:
        """Return the ids of the questions that already have a result."""
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT question_id FROM {results} "
                    "WHERE run_id = %s AND stage = %s AND config_key = %s"
                ),
                (run_id, stage, config_key),
            )
            rows = await cursor.fetchall()
        return {str(row[0]) for row in rows}

    async def save_result(
        self, run_id: str, stage: int, config_key: str, question_id: str, data: Mapping[str, object]
    ) -> None:
        """Store the outcome of one configuration on one question."""
        async with self._database.connection() as connection:
            await connection.execute(
                self._sql(
                    "INSERT INTO {results} (run_id, stage, config_key, question_id, data) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_id, stage, config_key, question_id) "
                    "DO UPDATE SET data = EXCLUDED.data"
                ),
                (run_id, stage, config_key, question_id, Jsonb(dict(data))),
            )

    async def results(self, run_id: str, stage: int) -> list[ResultRow]:
        """Return every result of one stage of a run."""
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT stage, config_key, question_id, data FROM {results} "
                    "WHERE run_id = %s AND stage = %s ORDER BY config_key, question_id"
                ),
                (run_id, stage),
            )
            rows = await cursor.fetchall()
        return [
            ResultRow(int(str(row[0])), str(row[1]), str(row[2]), _mapping(row[3])) for row in rows
        ]

    async def save_labels(
        self, run_id: str, labels: Sequence[tuple[str, str, Mapping[str, int]]]
    ) -> None:
        """Store human labels as ``(config_key, question_id, {criterion: score})`` items."""
        await self._ensure_schema()
        async with self._database.connection() as connection, connection.cursor() as cursor:
            await cursor.executemany(
                self._sql(
                    "INSERT INTO {human_labels} (run_id, config_key, question_id, labels) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (run_id, config_key, question_id) "
                    "DO UPDATE SET labels = EXCLUDED.labels"
                ),
                [(run_id, key, question, Jsonb(dict(scores))) for key, question, scores in labels],
            )

    async def labels(self, run_id: str) -> dict[tuple[str, str], Mapping[str, object]]:
        """Return the stored human labels keyed by ``(config_key, question_id)``."""
        await self._ensure_schema()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                self._sql(
                    "SELECT config_key, question_id, labels FROM {human_labels} WHERE run_id = %s"
                ),
                (run_id,),
            )
            rows = await cursor.fetchall()
        return {(str(row[0]), str(row[1])): _mapping(row[2]) for row in rows}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _run_record(row: Sequence[object]) -> RunRecord:
    created, finished = row[4], row[5]
    assert isinstance(created, datetime)
    return RunRecord(
        run_id=str(row[0]),
        fingerprint=str(row[1]),
        config=_mapping(row[2]),
        corpus=_mapping(row[3]),
        created_at=created,
        finished_at=finished if isinstance(finished, datetime) else None,
    )
