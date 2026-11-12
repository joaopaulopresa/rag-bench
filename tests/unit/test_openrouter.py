"""OpenRouter components against recorded responses: parsing, caching, retries, errors."""

from __future__ import annotations

import math

import httpx
import pytest

from ragbench.components.embedders import OpenRouterEmbedder
from ragbench.components.generators import OpenRouterGenerator
from ragbench.components.rerankers import OpenRouterReranker
from ragbench.core.errors import ConfigError, ProviderError
from ragbench.core.resources import Resources
from ragbench.core.types import Message
from tests.conftest import Replay


async def test_chat_is_parsed_and_the_second_call_comes_from_the_cache(
    resources: Resources, replay: Replay
) -> None:
    replay.add("chat/completions", "chat.json")
    generator = OpenRouterGenerator(resources, model="ibm-granite/granite-4.2-8b", max_tokens=200)
    messages = [Message("user", "Reply with the single word: pong")]

    first = await generator.generate(messages)
    second = await generator.generate(messages)

    assert first.text == "pong"
    assert first.usage.prompt_tokens > 0
    assert first.usage.cost_usd > 0
    assert first.usage.paid_usd == first.usage.cost_usd
    assert not first.usage.cached
    assert second.text == "pong"
    assert second.usage.cached
    assert second.usage.paid_usd == 0.0
    assert second.usage.cost_usd == first.usage.cost_usd
    assert len(replay.requests) == 1
    body = replay.bodies()[0]
    assert body["model"] == "ibm-granite/granite-4.2-8b"
    assert body["temperature"] == 0.0
    assert replay.requests[0].headers["authorization"] == "Bearer test-key"


async def test_changing_a_parameter_misses_the_cache(resources: Resources, replay: Replay) -> None:
    replay.add("chat/completions", "chat.json")
    messages = [Message("user", "hi")]
    await OpenRouterGenerator(resources, model="m", max_tokens=100).generate(messages)
    await OpenRouterGenerator(resources, model="m", max_tokens=101).generate(messages)
    assert len(replay.requests) == 2


async def test_chat_without_text_is_an_error_that_says_what_to_do(
    resources: Resources, replay: Replay
) -> None:
    replay.add("chat/completions", "chat_no_text.json")
    generator = OpenRouterGenerator(resources, model="ibm-granite/granite-4.2-8b")
    with pytest.raises(ProviderError, match="returned no text.*length.*reasoning"):
        await generator.generate([Message("user", "hi")])


async def test_embeddings_are_cut_normalised_and_cached_per_text(
    resources: Resources, replay: Replay
) -> None:
    replay.add("embeddings", "embeddings.json")
    embedder = OpenRouterEmbedder(resources, model="baai/bge-m3", dimensions=4)

    first = await embedder.embed(["hello world", "second text"])
    again = await embedder.embed(["second text", "hello world", "second text"])

    assert embedder.key == "openrouter:baai/bge-m3:4"
    assert [len(vector) for vector in first.vectors] == [4, 4]
    for vector in first.vectors:
        assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-5)
    assert first.usage.paid_usd > 0
    assert again.vectors == (first.vectors[1], first.vectors[0], first.vectors[1])
    assert again.usage.cached
    assert again.usage.paid_usd == 0.0
    assert len(replay.requests) == 1
    assert replay.bodies()[0] == {"model": "baai/bge-m3", "input": ["hello world", "second text"]}


async def test_query_prefix_applies_to_queries_only(resources: Resources, replay: Replay) -> None:
    replay.add("embeddings", "embeddings.json")
    embedder = OpenRouterEmbedder(resources, model="m", dimensions=4, query_prefix="Query: ")
    await embedder.embed(["a", "b"], kind="query")
    assert replay.bodies()[0]["input"] == ["Query: a", "Query: b"]


async def test_embedding_shorter_than_configured_is_an_error(
    resources: Resources, replay: Replay
) -> None:
    replay.add("embeddings", "embeddings.json")
    embedder = OpenRouterEmbedder(resources, model="baai/bge-m3", dimensions=4096)
    with pytest.raises(ProviderError, match="returns 8 dimensions, fewer than the 4096"):
        await embedder.embed(["hello world", "second text"])


async def test_rerank_orders_positions_and_is_cached(resources: Resources, replay: Replay) -> None:
    replay.add("rerank", "rerank.json")
    reranker = OpenRouterReranker(resources, model="cohere/rerank-5-fast")
    query = "What is the capital of France?"
    texts = ["Berlin is the capital of Germany.", "Paris is the capital of France.", "Bananas."]

    first = await reranker.rerank(query, texts)
    second = await reranker.rerank(query, texts)

    assert [item.index for item in first.scores] == [1, 0, 2]
    assert first.scores[0].score == pytest.approx(0.8430725)
    assert first.usage.cost_usd == pytest.approx(8e-7)
    assert second.scores == first.scores
    assert second.usage.cached
    assert len(replay.requests) == 1
    assert replay.bodies()[0] == {
        "model": "cohere/rerank-5-fast",
        "query": query,
        "documents": texts,
    }


async def test_rerank_of_nothing_makes_no_request(resources: Resources, replay: Replay) -> None:
    result = await OpenRouterReranker(resources, model="m").rerank("q", [])
    assert result.scores == ()
    assert replay.requests == []


async def test_transient_errors_are_retried(resources: Resources, replay: Replay) -> None:
    replay.add("chat/completions", {"error": {"message": "slow down", "code": 429}}, status=429)
    replay.add("chat/completions", "chat.json")
    result = await OpenRouterGenerator(resources, model="m").generate([Message("user", "hi")])
    assert result.text == "pong"
    assert len(replay.requests) == 2


async def test_error_inside_a_200_reply_is_detected(resources: Resources, replay: Replay) -> None:
    replay.add("chat/completions", {"error": {"message": "provider overloaded", "code": 502}})
    with pytest.raises(ProviderError, match="failed after 3 attempts.*provider overloaded"):
        await OpenRouterGenerator(resources, model="m").generate([Message("user", "hi")])
    assert len(replay.requests) == 3


async def test_client_errors_are_not_retried(resources: Resources, replay: Replay) -> None:
    replay.add("chat/completions", {"error": {"message": "no such model", "code": 404}}, status=404)
    with pytest.raises(ProviderError, match="HTTP 404: no such model") as raised:
        await OpenRouterGenerator(resources, model="m").generate([Message("user", "hi")])
    assert raised.value.status_code == 404
    assert not raised.value.retryable
    assert len(replay.requests) == 1


async def test_network_errors_are_retried_then_reported(resources: Resources) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    resources.transport = httpx.MockTransport(broken)
    with pytest.raises(ProviderError, match="network error"):
        await OpenRouterGenerator(resources, model="m").generate([Message("user", "hi")])


async def test_missing_api_key_is_a_config_error(resources: Resources) -> None:
    resources.api_keys = {}
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY is not set"):
        await OpenRouterGenerator(resources, model="m").generate([Message("user", "hi")])


def test_adding_a_cached_and_a_fresh_call_pays_only_for_the_fresh_one() -> None:
    from ragbench.core.types import NO_USAGE, Usage

    cached_rerank = Usage.from_payload({"cost_usd": 0.003, "latency_s": 1.0}, cached=True)
    fresh_generation = Usage.from_payload({"cost_usd": 0.0005, "latency_s": 2.0}, cached=False)
    total = cached_rerank + fresh_generation + NO_USAGE

    assert total.cost_usd == pytest.approx(0.0035), "what the answer costs with nothing cached"
    assert total.paid_usd == pytest.approx(0.0005), "what was spent now"
    assert total.latency_s == pytest.approx(3.0)
    assert not total.cached
    assert (cached_rerank + NO_USAGE).cached
    assert NO_USAGE.cached
    assert NO_USAGE.paid_usd == 0.0


async def test_a_partly_cached_embedding_batch_pays_only_for_the_new_texts(
    resources: Resources, replay: Replay
) -> None:
    for vector in ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]):
        reply = {
            "data": [{"index": 0, "embedding": vector}],
            "usage": {"prompt_tokens": 3, "cost": 3e-8},
        }
        replay.add("embeddings", reply)
    embedder = OpenRouterEmbedder(resources, model="baai/bge-m3", dimensions=4, batch_size=1)
    first = await embedder.embed(["hello world"])
    both = await embedder.embed(["hello world", "second text"])

    assert both.usage.cost_usd == pytest.approx(first.usage.cost_usd + both.usage.paid_usd)
    assert both.usage.paid_usd == pytest.approx(3e-8)
    assert both.usage.cost_usd == pytest.approx(6e-8)
    assert len(replay.requests) == 2
