"""Offline judge based on word overlap. A stand-in for a model, not a substitute for one."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.core.registries import register_judge
from ragbench.core.types import NO_USAGE, Verdict

_WORD = re.compile(r"[a-z0-9]+")
_CITATION = re.compile(r"\[(\d+)\]")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(_CITATION.sub(" ", text).lower()))


def _grade(share: float, good: float, partial: float) -> int:
    if share >= good:
        return 2
    return 1 if share >= partial else 0


@register_judge("lexical")
class LexicalJudge:
    """Scores an answer by the words it shares with the gold evidence and the passages.

    * correctness: share of the answer's words found in the gold evidence.
    * faithfulness: share of the answer's words found in the passages.
    * citation: 2 if every cited passage shares words with the answer, 1 if some do, 0 if the
      answer has no valid citation.

    It calls no model, so pipelines can be exercised end to end without a key. Its scores are
    crude and must not be reported as answer quality.

    Args:
        resources: Unused; present so the class fits the factory signature.
    """

    def __init__(self, resources: object = None) -> None:
        del resources

    @property
    def key(self) -> str:
        """Identity of the judge."""
        return "lexical"

    async def judge(
        self, *, question: str, answer: str, gold: Sequence[str], context: Sequence[str]
    ) -> Verdict:
        """Score ``answer`` by word overlap."""
        del question
        words = _words(answer)
        if not words:
            return Verdict(0, 0, 0, "empty answer", NO_USAGE)
        in_gold = len(words & _words(" ".join(gold))) / len(words)
        in_context = len(words & _words(" ".join(context))) / len(words)
        cited = [int(number) for number in _CITATION.findall(answer)]
        valid = [number for number in cited if 1 <= number <= len(context)]
        supported = [
            number
            for number in valid
            if len(words & _words(context[number - 1])) / len(words) >= 0.5
        ]
        if valid and len(supported) == len(valid) == len(cited):
            citation = 2
        else:
            citation = 1 if supported else 0
        return Verdict(
            correctness=_grade(in_gold, 0.8, 0.4),
            faithfulness=_grade(in_context, 0.9, 0.6),
            citation=citation,
            rationale=f"{in_gold:.0%} of answer words in gold, {in_context:.0%} in passages",
            usage=NO_USAGE,
        )
