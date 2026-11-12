"""Shared dependencies handed to every component factory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from ragbench.core.config import ProviderConfig
from ragbench.core.errors import ConfigError
from ragbench.core.interfaces import Cache

if TYPE_CHECKING:
    import httpx

T = TypeVar("T")


@dataclass
class Resources:
    """Dependencies that components receive explicitly instead of reaching for globals.

    Attributes:
        cache: Cache placed in front of every paid call.
        provider: Network settings (timeouts, concurrency, retries) for provider clients.
        api_keys: Secrets by environment-variable name, for example ``OPENROUTER_API_KEY``.
        data_dir: Directory where corpus loaders keep downloaded files.
        transport: Optional HTTP transport injected by tests to replay recorded responses.
    """

    cache: Cache
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    api_keys: Mapping[str, str] = field(default_factory=dict)
    data_dir: Path = Path(".data")
    transport: httpx.AsyncBaseTransport | None = None
    _shared: dict[str, object] = field(default_factory=dict, repr=False)

    def api_key(self, name: str) -> str:
        """Return the secret called ``name``.

        Raises:
            ConfigError: If it was not provided.
        """
        value = self.api_keys.get(name, "")
        if not value:
            raise ConfigError(f"{name} is not set; add it to the environment or to .env")
        return value

    def shared(self, key: str, factory: Callable[[], T]) -> T:
        """Return the object stored under ``key``, creating it with ``factory`` once.

        Components of one backend use it to share a client (one connection pool, one
        concurrency limit) without module-level state.
        """
        if key not in self._shared:
            self._shared[key] = factory()
        return cast(T, self._shared[key])

    def shared_objects(self) -> list[object]:
        """Return every object created through `shared`, so the owner can close them."""
        return list(self._shared.values())
