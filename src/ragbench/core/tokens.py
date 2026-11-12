"""Tokenizers. Chunk sizes are measured in tokens of one of these."""

from __future__ import annotations

import re
from functools import lru_cache

from ragbench.core.registries import register_tokenizer
from ragbench.core.types import Span

_WORD = re.compile(r"\S+")


@register_tokenizer("words")
class WordTokenizer:
    """Whitespace-separated words. Needs no download; exact and easy to reason about."""

    def __init__(self, resources: object = None) -> None:
        del resources

    def token_spans(self, text: str) -> list[Span]:
        """Return the span of every run of non-whitespace characters."""
        return [Span(match.start(), match.end()) for match in _WORD.finditer(text)]


@lru_cache(maxsize=4)
def _encoding(name: str):  # type: ignore[no-untyped-def]  # tiktoken.Encoding, imported lazily
    import tiktoken  # noqa: PLC0415

    return tiktoken.get_encoding(name)


@register_tokenizer("tiktoken")
class TiktokenTokenizer:
    """Byte-pair tokens of an OpenAI encoding, a good proxy for model token counts.

    Args:
        resources: Unused; present so the class fits the factory signature.
        encoding: Name of the tiktoken encoding.
    """

    def __init__(self, resources: object = None, encoding: str = "cl100k_base") -> None:
        del resources
        self._encoding_name = encoding

    def token_spans(self, text: str) -> list[Span]:
        """Return the character span of every token."""
        if not text:
            return []
        encoding = _encoding(self._encoding_name)
        tokens = encoding.encode(text, disallowed_special=())
        _, starts = encoding.decode_with_offsets(tokens)
        spans = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(text)
            if end > start:
                spans.append(Span(start, end))
        return spans
