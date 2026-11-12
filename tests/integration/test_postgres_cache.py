from __future__ import annotations

import pytest

from ragbench.components.caches import PostgresCache
from ragbench.components.postgres import Database
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry

pytestmark = pytest.mark.integration


async def test_entries_survive_a_new_cache_object(database: Database, resources: Resources) -> None:
    cache = PostgresCache(resources, database.url, database.schema)
    await cache.set_many(
        [
            CacheEntry(key="a", payload={"text": "one", "cost_usd": 0.5}),
            CacheEntry(key="b", payload={"n": 2}, blob=b"\x00\x01\x02"),
        ]
    )
    await cache.set_many([CacheEntry(key="a", payload={"text": "replaced"})])

    reopened = PostgresCache(resources, database.url, database.schema)
    found = await reopened.get_many(["a", "b", "missing"])

    assert set(found) == {"a", "b"}
    assert found["a"].payload == {"text": "replaced"}
    assert found["b"].blob == b"\x00\x01\x02"
    assert await reopened.get_many([]) == {}
    for shared in resources.shared_objects():
        await shared.aclose()  # type: ignore[attr-defined]
