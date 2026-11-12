"""Opens everything an experiment needs and closes it again."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ragbench.components  # noqa: F401  (registers the built-in implementations)
from ragbench.components.caches import MemoryCache
from ragbench.components.postgres import shared_database
from ragbench.core.config import ExperimentConfig, ProviderConfig
from ragbench.core.interfaces import Store
from ragbench.core.registries import caches, stores
from ragbench.core.resources import Resources
from ragbench.experiment.results import ResultsRepository

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True, slots=True)
class Session:
    """A cache and a store on one PostgreSQL database, for use without an experiment.

    Attributes:
        resources: Shared dependencies handed to every component (cache, keys, client limits).
        store: The chunk index store.
    """

    resources: Resources
    store: Store


@asynccontextmanager
async def open_session(
    database_url: str,
    *,
    schema: str = "ragbench",
    env: Mapping[str, str] | None = None,
    provider: ProviderConfig | None = None,
    data_dir: Path = Path(".data"),
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Session]:
    """Open a PostgreSQL-backed cache and store; close every connection on exit.

    This is the entry point for applications that build pipelines directly instead of running
    an experiment.

    Args:
        database_url: PostgreSQL connection URL.
        schema: Schema that holds the package's tables.
        env: Environment from which secrets are read: every variable whose name ends in
            ``_API_KEY``. ``None`` means no secrets, which is enough for offline components.
        provider: Timeout, concurrency and retry settings of provider clients.
        data_dir: Directory where corpus loaders keep downloaded files.
        transport: Optional HTTP transport for provider clients (tests replay recordings).

    Example:
        >>> async with open_session(url, env=os.environ) as session:  # doctest: +SKIP
        ...     pipeline = build_pipeline(config, resources=session.resources,
        ...                               store=session.store, collection="handbook")
    """
    database = {"url": database_url, "schema": schema}
    resources = Resources(
        cache=MemoryCache(),
        provider=provider or ProviderConfig(),
        api_keys={
            name: value
            for name, value in (env or {}).items()
            if name.endswith("_API_KEY") and value
        },
        data_dir=data_dir,
        transport=transport,
    )
    resources.cache = caches.create("postgres", resources, database)
    try:
        yield Session(resources=resources, store=stores.create("postgres", resources, database))
    finally:
        for shared in resources.shared_objects():
            close = getattr(shared, "aclose", None)
            if close is not None:
                await close()


@dataclass(frozen=True, slots=True)
class Runtime:
    """Live dependencies of an experiment.

    Attributes:
        config: The experiment configuration.
        resources: Shared dependencies handed to every component (cache, keys, client limits).
        store: The chunk index store.
        results: Where runs and results are kept.
    """

    config: ExperimentConfig
    resources: Resources
    store: Store
    results: ResultsRepository


@asynccontextmanager
async def open_runtime(
    config: ExperimentConfig,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Runtime]:
    """Open the cache, the store and the result tables of ``config``; close them on exit.

    Args:
        config: The experiment configuration.
        env: Environment from which secrets are read: every variable whose name ends in
            ``_API_KEY``. ``None`` means no secrets, which is enough for offline components.
        transport: Optional HTTP transport for provider clients (tests replay recordings).

    Example:
        >>> async with open_runtime(config, env=os.environ) as runtime:  # doctest: +SKIP
        ...     corpus = load_corpus(config.corpus, runtime.resources)
    """
    async with open_session(
        config.database.url,
        schema=config.database.schema_name,
        env=env,
        provider=config.provider,
        data_dir=config.data_dir,
        transport=transport,
    ) as session:
        results = ResultsRepository(
            shared_database(session.resources, config.database.url, config.database.schema_name)
        )
        yield Runtime(
            config=config, resources=session.resources, store=session.store, results=results
        )
