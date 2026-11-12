"""Durable cache in PostgreSQL: a second run of the same experiment costs nothing."""

from __future__ import annotations

from collections.abc import Sequence

from psycopg import sql
from psycopg.types.json import Jsonb

from ragbench.components.postgres import Database, shared_database
from ragbench.core.registries import register_cache
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry


@register_cache("postgres")
class PostgresCache:
    """Stores model responses in the table ``model_cache``.

    Args:
        resources: Shared dependencies; provides the connection pool.
        url: PostgreSQL connection URL.
        schema: Schema of the cache table.
    """

    def __init__(self, resources: Resources, url: str, schema: str = "ragbench") -> None:
        self._database: Database = shared_database(resources, url, schema)
        self._table = self._database.table("model_cache")
        self._ready = False

    async def _ensure_table(self) -> None:
        if self._ready:
            return
        async with self._database.connection() as connection:
            await connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} ("
                    "key text PRIMARY KEY, payload jsonb NOT NULL, blob bytea, "
                    "created_at timestamptz NOT NULL DEFAULT now())"
                ).format(self._table)
            )
        self._ready = True

    async def get_many(self, keys: Sequence[str]) -> dict[str, CacheEntry]:
        """Return the stored entries among ``keys``."""
        if not keys:
            return {}
        await self._ensure_table()
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT key, payload, blob FROM {} WHERE key = ANY(%s)").format(
                    self._table
                ),
                (list(keys),),
            )
            rows = await cursor.fetchall()
        found: dict[str, CacheEntry] = {}
        for key, payload, blob in rows:
            assert isinstance(key, str)
            assert isinstance(payload, dict)
            found[key] = CacheEntry(
                key=key,
                payload=payload,
                blob=bytes(blob) if isinstance(blob, bytes | memoryview) else None,
            )
        return found

    async def set_many(self, entries: Sequence[CacheEntry]) -> None:
        """Store ``entries``, replacing entries with the same key."""
        if not entries:
            return
        await self._ensure_table()
        statement = sql.SQL(
            "INSERT INTO {} (key, payload, blob) VALUES (%s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, blob = EXCLUDED.blob"
        ).format(self._table)
        async with self._database.connection() as connection, connection.cursor() as cursor:
            await cursor.executemany(
                statement,
                [(entry.key, Jsonb(dict(entry.payload)), entry.blob) for entry in entries],
            )
