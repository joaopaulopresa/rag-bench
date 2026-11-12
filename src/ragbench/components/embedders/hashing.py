"""Offline embedder based on feature hashing. Free, deterministic, no network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ragbench.core.registries import register_embedder
from ragbench.core.types import Embeddings, EmbedKind

_WORD = re.compile(r"[a-z0-9]+")


@register_embedder("hashing")
class HashingEmbedder:
    """Bag-of-words vectors: each word adds +1 or -1 to a position chosen by its hash.

    It is a weak lexical embedder, useful for tests, examples and continuous integration, where
    no API key is available. Texts that share words get similar vectors.

    Args:
        resources: Unused; present so the class fits the factory signature.
        dimensions: Vector length.
    """

    def __init__(self, resources: object = None, dimensions: int = 256) -> None:
        del resources
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def key(self) -> str:
        """Identity of the embedding space."""
        return f"hashing-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """Length of the vectors."""
        return self._dimensions

    async def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> Embeddings:
        """Embed ``texts``; ``kind`` makes no difference for this embedder."""
        del kind
        return Embeddings(vectors=tuple(self._vector(text) for text in texts))

    def _vector(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            values[number % self._dimensions] += 1.0 if (number >> 63) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)
