"""Offline generator that answers with the first sentence of the first passage."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.core.registries import register_generator
from ragbench.core.types import Generation, Message

NO_ANSWER = "The provided passages do not answer the question."
_PASSAGE = re.compile(r"^\[1\]\s*(.+?)(?=^\[2\]|\Z)", re.MULTILINE | re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


@register_generator("extractive")
class ExtractiveGenerator:
    """Quotes the first sentence of passage ``[1]`` and cites it.

    It calls no model. It exists so that the pipeline, the examples and continuous integration
    run without an API key; its answers are only as good as the top passage.

    Args:
        resources: Unused; present so the class fits the factory signature.
    """

    def __init__(self, resources: object = None) -> None:
        del resources

    @property
    def key(self) -> str:
        """Identity of the generator."""
        return "extractive"

    async def generate(self, messages: Sequence[Message]) -> Generation:
        """Return the first sentence of passage ``[1]`` found in the last message."""
        prompt = messages[-1].content if messages else ""
        match = _PASSAGE.search(prompt)
        if match is None:
            return Generation(
                text="The provided passages do not answer the question.", model=self.key
            )
        passage = " ".join(match.group(1).split())
        sentence = _SENTENCE_END.split(passage, maxsplit=1)[0]
        return Generation(text=f"{sentence} [1]", model=self.key)
