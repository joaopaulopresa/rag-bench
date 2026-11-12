"""In-process cache, for tests and one-off scripts."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.core.registries import register_cache
from ragbench.core.types import CacheEntry


@register_cache("memory")
class MemoryCache:
    """Keeps entries in a dictionary. Nothing survives the process.

    Args:
        resources: Unused; present so the class fits the factory signature.
    """

    def __init__(self, resources: object = None) -> None:
        del resources
        self._entries: dict[str, CacheEntry] = {}

    async def get_many(self, keys: Sequence[str]) -> dict[str, CacheEntry]:
        """Return the stored entries among ``keys``."""
        return {key: self._entries[key] for key in keys if key in self._entries}

    async def set_many(self, entries: Sequence[CacheEntry]) -> None:
        """Store ``entries``, replacing entries with the same key."""
        for entry in entries:
            self._entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self._entries)
