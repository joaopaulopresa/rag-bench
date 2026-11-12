"""The answer prompt and the citation syntax it asks for."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.core.types import Message

SYSTEM_PROMPT = (
    "You answer questions about legal documents using only the numbered passages you are given. "
    "Write a short, direct answer of one to three sentences. "
    "After every claim, cite the passage that supports it with its number in square brackets, "
    "for example [2]. Cite only passages you used. "
    "If the passages do not contain the answer, reply exactly: "
    "The provided passages do not answer the question."
)
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def build_answer_messages(question: str, passages: Sequence[str]) -> list[Message]:
    """Return the chat messages that ask a generator to answer from ``passages``.

    Passage ``n`` is shown as ``[n]``, which is also how the answer must cite it.
    """
    numbered = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(passages, start=1))
    return [
        Message("system", SYSTEM_PROMPT),
        Message("user", f"Passages:\n\n{numbered}\n\nQuestion: {question}"),
    ]


def cited_passages(answer: str, passage_count: int) -> tuple[list[int], list[int]]:
    """Read the ``[n]`` citations of an answer.

    Args:
        answer: The generated answer.
        passage_count: How many passages the generator was given.

    Returns:
        Two sorted lists of distinct numbers: citations that point to a passage that exists,
        and citations that do not.

    Example:
        >>> cited_passages("Yes [1]. Also [2, 7] and again [1].", 5)
        ([1, 2], [7])
    """
    numbers = {
        int(number) for group in _CITATION.findall(answer) for number in re.split(r"\s*,\s*", group)
    }
    valid = sorted(number for number in numbers if 1 <= number <= passage_count)
    invalid = sorted(number for number in numbers if not 1 <= number <= passage_count)
    return valid, invalid
