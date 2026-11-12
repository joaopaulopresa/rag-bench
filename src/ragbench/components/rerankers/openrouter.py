"""Reranking through the OpenRouter ``/rerank`` endpoint, cached per request."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.components.providers.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_ENV,
    shared_client,
    usage_from,
)
from ragbench.core.errors import ProviderError
from ragbench.core.hashing import stable_hash
from ragbench.core.registries import register_reranker
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry, Reranking, RerankScore, Usage


@register_reranker("openrouter")
class OpenRouterReranker:
    """Reranks passages with any rerank model served by OpenRouter.

    The request is ``{"model", "query", "documents"}`` and the reply lists
    ``{"index", "relevance_score"}`` items; both were confirmed against the live endpoint.

    Args:
        resources: Shared dependencies: cache, API key and network settings.
        model: OpenRouter model id.
        max_chars: Longer passages are cut to this many characters before scoring.
        base_url: API root.
        api_key_env: Name of the secret that holds the API key.
    """

    def __init__(
        self,
        resources: Resources,
        model: str,
        max_chars: int = 8000,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_KEY_ENV,
    ) -> None:
        self._resources = resources
        self._model = model
        self._max_chars = max_chars
        self._base_url = base_url
        self._api_key_env = api_key_env

    @property
    def key(self) -> str:
        """Identity of the reranker."""
        return f"openrouter:{self._model}"

    async def rerank(self, query: str, texts: Sequence[str]) -> Reranking:
        """Score every text against ``query``.

        Raises:
            ProviderError: If the API fails or does not score every passage.
        """
        if not texts:
            return Reranking(scores=(), usage=Usage(cached=True))
        documents = [text[: self._max_chars] for text in texts]
        key = stable_hash(
            {
                "kind": "rerank",
                "backend": "openrouter",
                "model": self._model,
                "query": query,
                "documents": documents,
            }
        )
        cache = self._resources.cache
        entry = (await cache.get_many([key])).get(key)
        cached = entry is not None
        if entry is None:
            client = shared_client(self._resources, self._base_url, self._api_key_env)
            data, latency = await client.post(
                "rerank", {"model": self._model, "query": query, "documents": documents}
            )
            results = data.get("results")
            if not isinstance(results, list):
                raise ProviderError(f"{self._model}: the rerank reply has no results")
            tokens, _, cost, _ = usage_from(data, latency)
            entry = CacheEntry(
                key=key,
                payload={
                    "kind": "rerank",
                    "model": self._model,
                    "scores": [
                        [int(item["index"]), float(item["relevance_score"])] for item in results
                    ],
                    "prompt_tokens": tokens,
                    "cost_usd": cost,
                    "latency_s": latency,
                },
            )
            await cache.set_many([entry])
        payload = entry.payload
        pairs = payload["scores"]
        assert isinstance(pairs, list)
        scores = sorted(
            (RerankScore(index=int(pair[0]), score=float(pair[1])) for pair in pairs),
            key=lambda item: (-item.score, item.index),
        )
        if {item.index for item in scores} != set(range(len(documents))):
            raise ProviderError(f"{self._model}: the rerank reply does not score every passage")
        usage = Usage.from_payload(payload, cached=cached)
        return Reranking(scores=tuple(scores), usage=usage)
