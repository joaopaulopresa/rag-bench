"""Connection pool shared by the PostgreSQL cache, store and result tables."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from ragbench.core.errors import ConfigError, StoreError
from ragbench.core.resources import Resources

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class Database:
    """A lazily opened connection pool bound to one schema.

    Args:
        url: PostgreSQL connection URL.
        schema: Schema that holds the package's tables. Created on first use.
        max_connections: Upper bound of pooled connections.

    Raises:
        ConfigError: If ``schema`` is not a plain lower-case identifier.
    """

    def __init__(self, url: str, schema: str = "ragbench", *, max_connections: int = 8) -> None:
        if not _IDENTIFIER.match(schema):
            raise ConfigError(f"schema {schema!r} must match {_IDENTIFIER.pattern}")
        self.url = url
        self.schema = schema
        self._pool = AsyncConnectionPool(
            url, min_size=1, max_size=max_connections, open=False, kwargs={"autocommit": True}
        )
        self._opened = False
        self._lock = asyncio.Lock()

    def table(self, name: str) -> sql.Identifier:
        """Return the schema-qualified identifier of table ``name``."""
        return sql.Identifier(self.schema, name)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection[tuple[object, ...]]]:
        """Yield an autocommit connection from the pool, opening the pool on first use.

        Raises:
            StoreError: If the database cannot be reached.
        """
        await self._open()
        async with self._pool.connection() as connection:
            yield connection

    async def _open(self) -> None:
        if self._opened:
            return
        async with self._lock:
            if self._opened:
                return
            try:
                await self._pool.open(wait=True, timeout=15.0)
                async with self._pool.connection() as connection:
                    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await connection.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                            sql.Identifier(self.schema)
                        )
                    )
            except (psycopg.Error, TimeoutError) as error:
                await self._pool.close()
                raise StoreError(
                    f"cannot connect to PostgreSQL ({error}); is `docker compose up -d db` running?"
                ) from error
            self._opened = True

    async def aclose(self) -> None:
        """Close the pool."""
        await self._pool.close()
        self._opened = False


def shared_database(resources: Resources, url: str, schema: str) -> Database:
    """Return the `Database` for ``url`` and ``schema``, one per `Resources`."""
    return resources.shared(f"postgres:{url}:{schema}", lambda: Database(url, schema))
