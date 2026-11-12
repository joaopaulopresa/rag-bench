"""Shared fixtures. No test needs an API key or the network."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path

import httpx
import psycopg
import pytest

import ragbench.components  # noqa: F401  (registers the built-in implementations)
from ragbench.components.caches import MemoryCache
from ragbench.components.postgres import Database
from ragbench.core.config import ProviderConfig
from ragbench.core.resources import Resources

FIXTURES = Path(__file__).parent / "fixtures"
DATABASE_URL = os.environ.get(
    "RAGBENCH_TEST_DATABASE_URL", "postgresql://ragbench:ragbench@localhost:54329/ragbench"
)

Handler = Callable[[httpx.Request], httpx.Response]


class Replay:
    """HTTP transport that answers from recorded OpenRouter responses and counts requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, list[httpx.Response]] = {}

    def add(self, path: str, fixture: str | Mapping[str, object], *, status: int = 200) -> None:
        body = (
            json.loads((FIXTURES / "openrouter" / fixture).read_text())
            if isinstance(fixture, str)
            else fixture
        )
        self.routes.setdefault(path, []).append(httpx.Response(status, json=body))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1/")
        queue = self.routes.get(path)
        if not queue:
            return httpx.Response(404, json={"error": {"message": f"no fixture for {path}"}})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture
def replay() -> Replay:
    return Replay()


@pytest.fixture
def resources(replay: Replay) -> Resources:
    return Resources(
        cache=MemoryCache(),
        provider=ProviderConfig(max_retries=2, backoff_s=0.0),
        api_keys={"OPENROUTER_API_KEY": "test-key"},
        transport=httpx.MockTransport(replay),
    )


def _database_available() -> bool:
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


@pytest.fixture(scope="session")
def database_url() -> str:
    if not _database_available():
        pytest.skip("PostgreSQL is not reachable; run `docker compose up -d db`")
    return DATABASE_URL


@pytest.fixture
async def database(database_url: str) -> AsyncIterator[Database]:
    """A `Database` on a schema of its own, dropped after the test."""
    schema = f"test_{uuid.uuid4().hex[:12]}"
    database = Database(database_url, schema)
    try:
        yield database
    finally:
        async with database.connection() as connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await database.aclose()
