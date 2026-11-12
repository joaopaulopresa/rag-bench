"""Exception hierarchy. Every error raised by the package derives from `RagBenchError`."""

from __future__ import annotations


class RagBenchError(Exception):
    """Base class of every error raised by ragbench."""


class ConfigError(RagBenchError):
    """A configuration value is missing, unknown or invalid."""


class RegistryError(ConfigError):
    """A component name is not registered, or is registered twice."""


class ProviderError(RagBenchError):
    """A model provider returned an error or an unusable response.

    Attributes:
        status_code: HTTP status of the failing response, when there was one.
        retryable: Whether the same request may succeed if sent again.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class DatasetError(RagBenchError):
    """A corpus could not be downloaded, read or validated."""


class StoreError(RagBenchError):
    """The index store rejected an operation or is not reachable."""


class ExperimentError(RagBenchError):
    """An experiment cannot run or its stored results are inconsistent."""
