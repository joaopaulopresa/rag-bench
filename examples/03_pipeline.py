"""Ask a question through the LangGraph pipeline: retrieve, rerank, generate, check citations.

Runs offline with stand-in components. To use real models, set OPENROUTER_API_KEY and swap the
embedder, reranker and generator for `{"name": "openrouter", "params": {"model": ...}}`.

    docker compose up -d db
    uv run python examples/03_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ragbench import ComponentConfig, PipelineConfig, build_pipeline, load_corpus, open_session

HANDBOOK = Path(__file__).parent / "handbook"
DATABASE_URL = os.environ.get(
    "RAGBENCH_DATABASE__URL", "postgresql://ragbench:ragbench@localhost:54329/ragbench"
)
CONFIG = PipelineConfig.model_validate(
    {
        "chunker": {"name": "recursive", "params": {"max_tokens": 40, "tokenizer": "words"}},
        "embedder": {"name": "hashing", "params": {"dimensions": 256}},
        "search_mode": "hybrid",
        "reranker": {"name": "overlap"},
        "generator": {"name": "extractive"},
        "top_k": 3,
        "candidates": 10,
    }
)


async def main() -> None:
    async with open_session(DATABASE_URL, schema="ragbench_examples", env=os.environ) as session:
        corpus = load_corpus(
            ComponentConfig(name="folder", params={"path": str(HANDBOOK)}), session.resources
        )
        pipeline = build_pipeline(
            CONFIG, resources=session.resources, store=session.store, collection="handbook"
        )
        report = await pipeline.index(corpus.documents)
        print(f"index: {report.chunks} chunks ({'built' if report.built else 'already there'})")

        answer = await pipeline.answer("What is the deadline for submitting an expense claim?")
        print(f"\nQ: {answer.question}\nA: {answer.text}")
        print(f"cited passages: {list(answer.citations)}  cost: ${answer.usage.cost_usd:.4f}\n")
        for number, chunk in enumerate(answer.chunks, start=1):
            body = " ".join(chunk.text.split())
            print(f"[{number}] {chunk.document_id} score={chunk.score:.2f}  {body[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
