"""Named registries of component factories, extensible through entry points."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from ragbench.core.errors import ConfigError, RegistryError

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., object])

logger = logging.getLogger(__name__)


class Registry(Generic[T]):
    """Maps names to factories of one kind of component.

    A factory is any callable (usually the class itself) whose first parameter is the
    `Resources` container and whose remaining parameters are keyword configuration values.

    Third-party packages add implementations without touching ragbench by declaring an entry
    point in the registry's group, for example::

        [project.entry-points."ragbench.chunkers"]
        markdown = "my_package.chunkers:MarkdownChunker"

    Args:
        kind: Singular name of the component kind, used in error messages ("chunker").
        group: Entry-point group searched for third-party implementations.
    """

    def __init__(self, kind: str, group: str) -> None:
        self.kind = kind
        self.group = group
        self._factories: dict[str, Callable[..., T]] = {}
        self._entry_points_loaded = False

    def register(self, name: str) -> Callable[[F], F]:
        """Return a decorator that registers a factory under ``name``.

        Raises:
            RegistryError: If ``name`` is already taken by a different factory.
        """

        def decorator(factory: F) -> F:
            existing = self._factories.get(name)
            if existing is not None and existing is not factory:
                raise RegistryError(f"{self.kind} {name!r} is already registered")
            self._factories[name] = factory  # type: ignore[assignment]
            return factory

        return decorator

    def names(self) -> list[str]:
        """Return every registered name, sorted, including entry-point implementations."""
        self._load_entry_points()
        return sorted(self._factories)

    def get(self, name: str) -> Callable[..., T]:
        """Return the factory registered under ``name``.

        Raises:
            RegistryError: If nothing is registered under ``name``. The message lists the
                names that are.
        """
        if name not in self._factories:
            self._load_entry_points()
        try:
            return self._factories[name]
        except KeyError:
            allowed = ", ".join(self.names()) or "(none)"
            raise RegistryError(f"unknown {self.kind} {name!r}; allowed: {allowed}") from None

    def validate(self, name: str, params: Mapping[str, object]) -> None:
        """Check that ``name`` exists and accepts ``params``, without building anything.

        Raises:
            RegistryError: If ``name`` is not registered.
            ConfigError: If a parameter is unknown or a required one is missing.
        """
        factory = self.get(name)
        signature = inspect.signature(factory)
        accepted = list(signature.parameters.values())[1:]
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted):
            return
        allowed = [p.name for p in accepted]
        unknown = sorted(set(params) - set(allowed))
        if unknown:
            raise ConfigError(
                f"{self.kind} {name!r} does not accept {', '.join(unknown)}; "
                f"allowed parameters: {', '.join(allowed) or '(none)'}"
            )
        missing = [
            p.name
            for p in accepted
            if p.default is inspect.Parameter.empty
            and p.kind is not inspect.Parameter.VAR_POSITIONAL
            and p.name not in params
        ]
        if missing:
            raise ConfigError(f"{self.kind} {name!r} needs parameters: {', '.join(missing)}")

    def create(self, name: str, resources: object, params: Mapping[str, object]) -> T:
        """Build the component ``name`` from ``params``.

        Args:
            name: Registered name.
            resources: The `Resources` container passed as the factory's first argument.
            params: Keyword configuration for the factory.

        Raises:
            RegistryError: If ``name`` is not registered.
            ConfigError: If ``params`` do not fit the factory.
        """
        self.validate(name, params)
        return self.get(name)(resources, **params)

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for entry_point in entry_points(group=self.group):
            if entry_point.name in self._factories:
                continue
            try:
                self._factories[entry_point.name] = entry_point.load()
            except Exception as error:
                # A broken third-party plugin must not take the whole registry down.
                logger.warning(
                    "could not load %s %r from entry point %s: %s",
                    self.kind,
                    entry_point.name,
                    entry_point.value,
                    error,
                )
