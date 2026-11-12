"""Chat completions through OpenRouter, cached per request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragbench.components.providers.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_ENV,
    shared_client,
    usage_from,
)
from ragbench.core.errors import ProviderError
from ragbench.core.hashing import stable_hash
from ragbench.core.registries import register_generator
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry, Generation, Message, Usage


@register_generator("openrouter")
class OpenRouterGenerator:
    """Generates text with any chat model served by OpenRouter.

    Args:
        resources: Shared dependencies: cache, API key and network settings.
        model: OpenRouter model id. Use the paid id, not the ``:free`` variant, for stable
            latency and rate limits.
        temperature: Sampling temperature. Zero keeps answers as repeatable as the model allows.
        max_tokens: Upper bound of generated tokens, reasoning included.
        seed: Sampling seed, sent to providers that honour it.
        extra_body: Extra request fields, for example ``{"reasoning": {"enabled": false}}`` or
            ``{"response_format": {"type": "json_object"}}``.
        base_url: API root.
        api_key_env: Name of the secret that holds the API key.
    """

    def __init__(
        self,
        resources: Resources,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        extra_body: Mapping[str, object] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_KEY_ENV,
    ) -> None:
        self._resources = resources
        self._model = model
        self._body: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **(extra_body or {}),
        }
        if seed is not None:
            self._body["seed"] = seed
        self._base_url = base_url
        self._api_key_env = api_key_env

    @property
    def key(self) -> str:
        """Identity of the generator."""
        return f"openrouter:{self._model}"

    async def generate(self, messages: Sequence[Message]) -> Generation:
        """Return the model's reply to ``messages``.

        Raises:
            ProviderError: If the API fails or the reply has no text (for example when a
                reasoning model spends all of ``max_tokens`` thinking).
        """
        body = {
            **self._body,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }
        key = stable_hash({"kind": "chat", "backend": "openrouter", "body": body})
        cache = self._resources.cache
        entry = (await cache.get_many([key])).get(key)
        cached = entry is not None
        if entry is None:
            client = shared_client(self._resources, self._base_url, self._api_key_env)
            data, latency = await client.post("chat/completions", body)
            text = _reply_text(data)
            if not text:
                raise ProviderError(
                    f"{self._model} returned no text (finish reason: {_finish_reason(data)}); "
                    "raise max_tokens or turn reasoning off in extra_body"
                )
            prompt_tokens, completion_tokens, cost, _ = usage_from(data, latency)
            entry = CacheEntry(
                key=key,
                payload={
                    "kind": "chat",
                    "model": self._model,
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost,
                    "latency_s": latency,
                },
            )
            await cache.set_many([entry])
        payload = entry.payload
        usage = Usage.from_payload(payload, cached=cached)
        return Generation(text=str(payload["text"]), model=self._model, usage=usage)


def _first_choice(data: Mapping[str, object]) -> Mapping[str, object]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _reply_text(data: Mapping[str, object]) -> str:
    message = _first_choice(data).get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"]).strip()
    return ""


def _finish_reason(data: Mapping[str, object]) -> str:
    return str(_first_choice(data).get("finish_reason", "unknown"))
