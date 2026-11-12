"""Offline reranker that scores passages by the share of query words they contain."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.core.registries import register_reranker
from ragbench.core.types import Reranking, RerankScore

_WORD = re.compile(r"[a-z0-9]+")


@register_reranker("overlap")
class OverlapReranker:
    """Scores a passage by the fraction of distinct query words that occur in it.

    Free and deterministic; meant for tests, examples and continuous integration.

    Args:
        resources: Unused; present so the class fits the factory signature.
    """

    def __init__(self, resources: object = None) -> None:
        del resources

    @property
    def key(self) -> str:
        """Identity of the reranker."""
        return "overlap"

    async def rerank(self, query: str, texts: Sequence[str]) -> Reranking:
        """Score every text; ties keep the input order."""
        words = set(_WORD.findall(query.lower()))
        scored = []
        for index, text in enumerate(texts):
            present = words & set(_WORD.findall(text.lower()))
            scored.append(
                RerankScore(index=index, score=len(present) / len(words) if words else 0.0)
            )
        scored.sort(key=lambda item: (-item.score, item.index))
        return Reranking(scores=tuple(scored))
