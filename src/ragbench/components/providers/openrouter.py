"""Async HTTP client for the OpenRouter API.

It knows nothing about chat, embeddings or reranking: it sends JSON, bounds concurrency, retries
transient failures with exponential backoff and turns every failure into a `ProviderError`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping

import httpx

from ragbench.core.config import ProviderConfig
from ragbench.core.errors import ProviderError
from ragbench.core.resources import Resources

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})

_MAX_DELAY_S = 60.0
_MAX_DETAIL_CHARS = 240

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Sends JSON requests to OpenRouter.

    Args:
        api_key: OpenRouter API key.
        base_url: API root.
        settings: Timeout, concurrency and retry settings.
        transport: Optional transport, used by tests to replay recorded responses.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        settings: ProviderConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or ProviderConfig()
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "ragbench"},
            timeout=self._settings.timeout_s,
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrency)

    async def post(self, path: str, body: Mapping[str, object]) -> tuple[dict[str, object], float]:
        """POST ``body`` to ``path`` and return the JSON reply and the seconds it took.

        The time is that of the successful attempt only; waiting and failed attempts are left
        out, so it measures the provider and not the retry policy.

        Raises:
            ProviderError: On a non-retryable error, or when retries are exhausted.
        """
        attempts = self._settings.max_retries + 1
        last_error: _TransientError | None = None
        for attempt in range(attempts):
            if attempt:
                delay = min(self._settings.backoff_s * 2 ** (attempt - 1), _MAX_DELAY_S)
                if last_error is not None and last_error.retry_after is not None:
                    delay = max(delay, last_error.retry_after)
                logger.warning(
                    "retrying %s in %.1fs (attempt %d/%d): %s",
                    path,
                    delay,
                    attempt + 1,
                    attempts,
                    last_error,
                )
                await asyncio.sleep(delay)
            try:
                async with self._semaphore:
                    return await self._send(path, body)
            except _TransientError as error:
                last_error = error
        assert last_error is not None
        raise ProviderError(
            f"{path} failed after {attempts} attempts: {last_error}",
            status_code=last_error.status_code,
            retryable=True,
        )

    async def _send(self, path: str, body: Mapping[str, object]) -> tuple[dict[str, object], float]:
        started = time.perf_counter()
        try:
            response = await self._http.post(path.lstrip("/"), json=dict(body))
        except httpx.TransportError as error:
            raise _TransientError(f"network error: {error!r}") from error
        elapsed = time.perf_counter() - started
        status = response.status_code
        try:
            data = response.json()
        except ValueError:
            data = None
        message = _error_message(data)
        if status >= 400 or message is not None:
            detail = (message or response.text)[:_MAX_DETAIL_CHARS]
            code = _error_code(data) or status
            if code in _RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after", "")
                raise _TransientError(
                    f"HTTP {code}: {detail}",
                    status_code=code,
                    retry_after=float(retry_after) if retry_after.isdigit() else None,
                )
            raise ProviderError(f"{path} returned HTTP {code}: {detail}", status_code=code)
        if not isinstance(data, dict):
            raise _TransientError(f"{path} returned a body that is not a JSON object")
        return data, elapsed

    async def aclose(self) -> None:
        """Close the underlying HTTP connections."""
        await self._http.aclose()


class _TransientError(ProviderError):
    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=status_code, retryable=True)
        self.retry_after = retry_after


def _error_message(data: object) -> str | None:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return str(data["error"].get("message", data["error"]))
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return str(data["error"])
    return None


def _error_code(data: object) -> int | None:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        if isinstance(code, int):
            return code
    return None


def shared_client(resources: Resources, base_url: str, api_key_env: str) -> OpenRouterClient:
    """Return the client for ``base_url``, one per `Resources`, so the limits are shared."""
    return resources.shared(
        f"openrouter:{base_url}:{api_key_env}",
        lambda: OpenRouterClient(
            api_key=resources.api_key(api_key_env),
            base_url=base_url,
            settings=resources.provider,
            transport=resources.transport,
        ),
    )


def usage_from(data: Mapping[str, object], latency_s: float) -> tuple[int, int, float, float]:
    """Read ``(prompt_tokens, completion_tokens, cost_usd, latency_s)`` from a reply."""
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0.0, latency_s
    prompt = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    cost = usage.get("cost") or 0.0
    return int(prompt), int(completion), float(cost), latency_s
