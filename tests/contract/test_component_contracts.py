"""Rules every registered embedder, reranker, generator, judge and cache must obey.

OpenRouter-backed implementations are exercised through recorded responses.
"""

from __future__ import annotations

import math

import pytest

from ragbench.core.interfaces import Cache, Embedder, Generator, Judge, Reranker
from ragbench.core.registries import caches, embedders, generators, judges, rerankers
from ragbench.core.resources import Resources
from ragbench.core.types import CacheEntry, Message
from tests.conftest import Replay

TEXTS = ["hello world", "second text"]
EMBEDDERS: dict[str, dict[str, object]] = {
    "hashing": {"dimensions": 16},
    "openrouter": {"model": "baai/bge-m3", "dimensions": 8},
}
RERANKERS: dict[str, dict[str, object]] = {
    "overlap": {},
    "openrouter": {"model": "cohere/rerank-5-fast"},
}
GENERATORS: dict[str, dict[str, object]] = {
    "extractive": {},
    "openrouter": {"model": "ibm-granite/granite-4.2-8b"},
}
JUDGE_REPLY = {
    "choices": [
        {
            "message": {
                "content": 'Scores: {"rationale": "ok", "correctness": 2, "faithfulness": 1, "citation": 0}'
            }
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001},
}
JUDGES: dict[str, dict[str, object]] = {
    "lexical": {},
    "llm": {"generator": {"name": "openrouter", "params": {"model": "judge-model"}}},
}


@pytest.fixture
def recorded(replay: Replay) -> Replay:
    replay.add("embeddings", "embeddings.json")
    replay.add("rerank", "rerank.json")
    replay.add("chat/completions", "chat.json")
    return replay


def test_every_builtin_implementation_is_covered() -> None:
    assert set(EMBEDDERS) == set(embedders.names())
    assert set(RERANKERS) == set(rerankers.names())
    assert set(GENERATORS) == set(generators.names()) - {"flaky"}
    assert set(JUDGES) == set(judges.names())


@pytest.mark.parametrize("name", sorted(EMBEDDERS))
async def test_embedder_contract(name: str, resources: Resources, recorded: Replay) -> None:
    embedder = embedders.create(name, resources, EMBEDDERS[name])
    assert isinstance(embedder, Embedder)

    first = await embedder.embed(TEXTS)
    again = await embedder.embed(TEXTS)
    as_query = await embedder.embed(TEXTS[:1], kind="query")

    assert embedder.key
    assert len(first.vectors) == len(TEXTS), "one vector per text, in order"
    for vector in first.vectors:
        assert len(vector) == embedder.dimensions
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4), "unit length"
    assert again.vectors == first.vectors, "the same text always gets the same vector"
    assert first.vectors[0] != first.vectors[1]
    assert len(as_query.vectors) == 1
    assert (await embedder.embed([])).vectors == ()
    assert again.usage.paid_usd == 0.0 or name == "hashing"


@pytest.mark.parametrize("name", sorted(RERANKERS))
async def test_reranker_contract(name: str, resources: Resources, recorded: Replay) -> None:
    reranker = rerankers.create(name, resources, RERANKERS[name])
    assert isinstance(reranker, Reranker)
    texts = ["Berlin is the capital of Germany.", "Paris is the capital of France.", "Bananas."]

    result = await reranker.rerank("What is the capital of France?", texts)

    assert reranker.key
    assert sorted(item.index for item in result.scores) == [0, 1, 2], "every passage is scored once"
    scores = [item.score for item in result.scores]
    assert scores == sorted(scores, reverse=True), "best first"
    assert result.scores[0].index == 1
    assert (await reranker.rerank("anything", [])).scores == ()


@pytest.mark.parametrize("name", sorted(GENERATORS))
async def test_generator_contract(name: str, resources: Resources, recorded: Replay) -> None:
    generator = generators.create(name, resources, GENERATORS[name])
    assert isinstance(generator, Generator)
    messages = [
        Message("system", "Be brief."),
        Message("user", "[1] Pong is a game.\n\nQuestion: ?"),
    ]

    first = await generator.generate(messages)
    second = await generator.generate(messages)

    assert generator.key
    assert first.text.strip()
    assert first.model
    assert second.text == first.text, "the same messages give the same text"
    assert second.usage.paid_usd == 0.0


@pytest.mark.parametrize("name", sorted(JUDGES))
async def test_judge_contract(name: str, resources: Resources, replay: Replay) -> None:
    replay.add("chat/completions", JUDGE_REPLY)
    judge = judges.create(name, resources, JUDGES[name])
    assert isinstance(judge, Judge)

    verdict = await judge.judge(
        question="When is rent due?",
        answer="Rent is due on the first day of each month [1].",
        gold=["The Tenant shall pay rent on the first day of each month."],
        context=[
            "The Tenant shall pay rent on the first day of each month.",
            "Deposit: three months.",
        ],
    )

    assert judge.key
    assert set(verdict.scores()) == {"correctness", "faithfulness", "citation"}
    assert all(score in (0, 1, 2) for score in verdict.scores().values())
    assert isinstance(verdict.rationale, str)


async def test_llm_judge_reads_scores_asks_again_once_and_then_gives_up(
    resources: Resources, replay: Replay
) -> None:
    from ragbench.core.errors import ProviderError

    judge = judges.create("llm", resources, JUDGES["llm"])
    arguments = {"question": "q", "answer": "a [1]", "gold": ["g"], "context": ["c"]}
    garbage = {
        "choices": [{"message": {"content": "I think it is fine."}}],
        "usage": {"cost": 0.0001},
    }
    out_of_range = {
        "choices": [
            {"message": {"content": '{"correctness": 5, "faithfulness": 1, "citation": 1}'}}
        ]
    }

    replay.add("chat/completions", garbage)
    replay.add("chat/completions", JUDGE_REPLY)
    verdict = await judge.judge(**arguments)  # type: ignore[arg-type]
    assert (verdict.correctness, verdict.faithfulness, verdict.citation) == (2, 1, 0)
    assert verdict.usage.cost_usd == pytest.approx(0.0002), "both attempts are paid for"
    assert "Return only the JSON object." in str(replay.bodies()[1]["messages"])
    user_prompt = replay.bodies()[0]["messages"][1]["content"]  # type: ignore[index]
    assert "GOLD EVIDENCE:\n- g" in user_prompt and "PASSAGES:\n[1] c" in user_prompt

    replay.routes["chat/completions"] = []
    replay.add("chat/completions", out_of_range)
    with pytest.raises(ProviderError, match="returned no valid scores"):
        await judge.judge(question="other", answer="a", gold=["g"], context=["c"])


@pytest.mark.parametrize("name", ["memory"])
async def test_cache_contract(name: str, resources: Resources) -> None:
    cache = caches.create(name, resources, {})
    assert isinstance(cache, Cache)
    assert await cache.get_many(["a"]) == {}
    await cache.set_many([CacheEntry("a", {"n": 1}), CacheEntry("b", {"n": 2}, b"\x01")])
    await cache.set_many([CacheEntry("a", {"n": 3})])
    found = await cache.get_many(["a", "b", "c"])
    assert {key: entry.payload for key, entry in found.items()} == {"a": {"n": 3}, "b": {"n": 2}}
    assert found["b"].blob == b"\x01"
