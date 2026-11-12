"""Judge that asks a language model to apply a fixed rubric and reply in JSON."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragbench.core.errors import ConfigError, ProviderError
from ragbench.core.interfaces import Generator
from ragbench.core.registries import generators, register_judge
from ragbench.core.types import Message, Usage, Verdict

RUBRIC_VERSION = "1"

#: The rubric given to the judge model and, word for word, to human annotators.
RUBRIC = """\
Score the ANSWER on three criteria. Each score is 0, 1 or 2.

correctness - compare the ANSWER with the GOLD EVIDENCE (the text that truly answers the question).
  2: the answer states what the gold evidence states, and nothing that contradicts it.
  1: the answer is partly right: it misses a key point of the gold evidence or adds a minor error.
  0: the answer is wrong, contradicts the gold evidence, or says it cannot answer.

faithfulness - compare the ANSWER with the PASSAGES (the only text the writer was given).
  2: every claim in the answer is supported by the passages.
  1: the main claim is supported, but at least one detail is not found in the passages.
  0: the main claim is not supported by the passages or contradicts them.
  An answer that only says the passages do not contain the answer scores 2.

citation - check the [n] markers in the ANSWER against the PASSAGES.
  2: every claim carries a marker, and each cited passage supports the claim it is attached to.
  1: markers are present, but a claim lacks one or a cited passage does not support its claim.
  0: there are no markers, or the cited passages do not support the claims.
  An answer that makes no claim (it only says it cannot answer) scores 2."""

_SYSTEM = (
    "You are a careful grader of answers to questions about legal documents. "
    "Apply the rubric exactly. Judge only what is written. "
    'Reply with one JSON object and nothing else: {"rationale": "<one or two sentences>", '
    '"correctness": 0|1|2, "faithfulness": 0|1|2, "citation": 0|1|2}'
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ATTEMPTS = 2


class _Scores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    correctness: int = Field(ge=0, le=2)
    faithfulness: int = Field(ge=0, le=2)
    citation: int = Field(ge=0, le=2)
    rationale: str = ""


def build_judge_prompt(
    *, question: str, answer: str, gold: Sequence[str], context: Sequence[str]
) -> str:
    """Return the user message shown to the judge model for one answer."""
    passages = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(context, start=1))
    evidence = "\n\n".join(f"- {text}" for text in gold)
    return (
        f"{RUBRIC}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD EVIDENCE:\n{evidence}\n\n"
        f"PASSAGES:\n{passages or '(none)'}\n\n"
        f"ANSWER:\n{answer}"
    )


def parse_scores(reply: str) -> _Scores | None:
    """Read the scores from a model reply; ``None`` if it holds no valid JSON object."""
    match = _JSON_OBJECT.search(reply)
    if match is None:
        return None
    try:
        return _Scores.model_validate(json.loads(match.group(0)))
    except (ValueError, ValidationError):
        return None


@register_judge("llm")
class LlmJudge:
    """Scores answers with a language model and the rubric in `RUBRIC`.

    The judge model should not be one of the evaluated generators. Its scores are only
    trustworthy once compared with human labels (see ``ragbench judge-agreement``).

    Args:
        resources: Shared dependencies, used to build the generator.
        generator: ``{"name": ..., "params": {...}}`` of the generator that plays the judge.
        max_passage_chars: Longer passages are cut before they are shown to the judge.

    Raises:
        ConfigError: If ``generator`` is not a mapping with a ``name``.
    """

    def __init__(
        self,
        resources: object,
        generator: Mapping[str, object],
        max_passage_chars: int = 6000,
    ) -> None:
        name = generator.get("name") if isinstance(generator, Mapping) else None
        params = generator.get("params", {}) if isinstance(generator, Mapping) else None
        if not isinstance(name, str) or not isinstance(params, Mapping):
            raise ConfigError("llm judge: generator must be {name: ..., params: {...}}")
        self._generator: Generator = generators.create(name, resources, params)
        self._max_passage_chars = max_passage_chars

    @property
    def key(self) -> str:
        """Identity of the judge."""
        return f"llm:{self._generator.key}:rubric-{RUBRIC_VERSION}"

    async def judge(
        self, *, question: str, answer: str, gold: Sequence[str], context: Sequence[str]
    ) -> Verdict:
        """Score ``answer`` against the gold evidence and the passages.

        Raises:
            ProviderError: If the model does not return valid scores after two attempts.
        """
        prompt = build_judge_prompt(
            question=question,
            answer=answer,
            gold=gold,
            context=[text[: self._max_passage_chars] for text in context],
        )
        usage = Usage()
        reply = ""
        for attempt in range(_MAX_ATTEMPTS):
            # A different reminder gives a different cache key, so a malformed cached reply
            # does not block the retry.
            reminder = "" if attempt == 0 else "\n\nReturn only the JSON object."
            generation = await self._generator.generate(
                [Message("system", _SYSTEM), Message("user", prompt + reminder)]
            )
            usage += generation.usage
            reply = generation.text
            scores = parse_scores(reply)
            if scores is not None:
                return Verdict(
                    correctness=scores.correctness,
                    faithfulness=scores.faithfulness,
                    citation=scores.citation,
                    rationale=scores.rationale,
                    usage=usage,
                )
        raise ProviderError(f"judge {self.key} returned no valid scores: {reply[:200]!r}")
