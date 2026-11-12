"""Embeddings through the OpenRouter ``/embeddings`` endpoint, cached per text."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np

from ragbench.components.providers.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_KEY_ENV,
    shared_client,
    usage_from,
)
from ragbench.core.errors import ProviderError
from ragbench.core.hashing import stable_hash
from ragbench.core.registries import register_embedder
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry, Embeddings, EmbedKind, Usage


@register_embedder("openrouter")
class OpenRouterEmbedder:
    """Embeds texts with any embedding model served by OpenRouter.

    Vectors are cut to ``dimensions`` and scaled to unit length on the client. Cutting is sound
    for models trained with Matryoshka representation learning (for example
    ``openai/text-embedding-3-large`` and ``qwen/qwen3-embedding-8b``); for other models set
    ``dimensions`` to the native size.

    Args:
        resources: Shared dependencies: cache, API key and network settings.
        model: OpenRouter model id.
        dimensions: Length of the returned vectors. Must not exceed the model's native size.
        query_prefix: Text put before queries (some models expect an instruction).
        batch_size: Texts per request.
        max_chars: Longer texts are cut to this many characters before embedding.
        base_url: API root.
        api_key_env: Name of the secret that holds the API key.
    """

    def __init__(
        self,
        resources: Resources,
        model: str,
        dimensions: int,
        query_prefix: str = "",
        batch_size: int = 64,
        max_chars: int = 24000,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_KEY_ENV,
    ) -> None:
        self._resources = resources
        self._model = model
        self._dimensions = dimensions
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._base_url = base_url
        self._api_key_env = api_key_env

    @property
    def key(self) -> str:
        """Identity of the embedding space."""
        return f"openrouter:{self._model}:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """Length of the vectors."""
        return self._dimensions

    def _cache_key(self, text: str) -> str:
        return stable_hash(
            {
                "kind": "embedding",
                "backend": "openrouter",
                "model": self._model,
                "dimensions": self._dimensions,
                "text": text,
            }
        )

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> Embeddings:
        """Embed ``texts``, calling the API only for texts that are not in the cache.

        Raises:
            ProviderError: If the API fails or returns vectors shorter than ``dimensions``.
        """
        prefix = self._query_prefix if kind == "query" else ""
        inputs = [(prefix + text)[: self._max_chars] or " " for text in texts]
        keys = [self._cache_key(text) for text in inputs]
        cache = self._resources.cache
        entries = await cache.get_many(list(dict.fromkeys(keys)))
        usage = Usage(cached=True)
        for key in keys:
            if key in entries:
                usage += Usage.from_payload(entries[key].payload, cached=True)

        missing: dict[str, str] = {}
        for key, text in zip(keys, inputs, strict=True):
            if key not in entries:
                missing.setdefault(key, text)
        if missing:
            items = list(missing.items())
            batches = [
                items[start : start + self._batch_size]
                for start in range(0, len(items), self._batch_size)
            ]
            for fresh in await asyncio.gather(*(self._embed_batch(batch) for batch in batches)):
                await cache.set_many(fresh)
                for entry in fresh:
                    entries[entry.key] = entry
                    usage += Usage.from_payload(entry.payload, cached=False)

        vectors = tuple(
            tuple(np.frombuffer(entries[key].blob or b"", dtype=np.float32).tolist())
            for key in keys
        )
        return Embeddings(vectors=vectors, usage=usage)

    async def _embed_batch(self, batch: Sequence[tuple[str, str]]) -> list[CacheEntry]:
        client = shared_client(self._resources, self._base_url, self._api_key_env)
        data, latency = await client.post(
            "embeddings", {"model": self._model, "input": [text for _, text in batch]}
        )
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise ProviderError(f"{self._model}: expected {len(batch)} embeddings in the reply")
        tokens, _, cost, _ = usage_from(data, latency)
        total_chars = sum(len(text) for _, text in batch) or 1
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        fresh = []
        for (key, text), row in zip(batch, ordered, strict=True):
            vector = np.asarray(row["embedding"], dtype=np.float32)
            if vector.shape[0] < self._dimensions:
                raise ProviderError(
                    f"{self._model} returns {vector.shape[0]} dimensions, "
                    f"fewer than the {self._dimensions} configured"
                )
            vector = vector[: self._dimensions]
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            share = len(text) / total_chars
            fresh.append(
                CacheEntry(
                    key=key,
                    payload={
                        "kind": "embedding",
                        "model": self._model,
                        "prompt_tokens": round(tokens * share),
                        "cost_usd": cost * share,
                        "latency_s": latency / len(batch),
                    },
                    blob=vector.astype(np.float32).tobytes(),
                )
            )
        return fresh
