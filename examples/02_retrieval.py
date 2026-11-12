"""Index a folder of documents in PostgreSQL and measure retrieval in three search modes.

Uses the offline hashing embedder, so it needs no API key:

    docker compose up -d db
    uv run python examples/02_retrieval.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ragbench import (
    ComponentConfig,
    PipelineConfig,
    build_pipeline,
    evaluate_retrieval,
    load_corpus,
    open_session,
)

HANDBOOK = Path(__file__).parent / "handbook"
DATABASE_URL = os.environ.get(
    "RAGBENCH_DATABASE__URL", "postgresql://ragbench:ragbench@localhost:54329/ragbench"
)
METRICS = ("char_recall@3", "char_precision@3", "mrr@5")


async def main() -> None:
    async with open_session(DATABASE_URL, schema="ragbench_examples") as session:
        corpus = load_corpus(
            ComponentConfig(name="folder", params={"path": str(HANDBOOK)}), session.resources
        )
        print(f"{len(corpus.documents)} documents, {len(corpus.questions)} gold questions\n")
        print(f"{'search mode':<12}" + "".join(f"{metric:>18}" for metric in METRICS))
        for mode in ("dense", "lexical", "hybrid"):
            config = PipelineConfig.model_validate(
                {
                    "chunker": {
                        "name": "recursive",
                        "params": {"max_tokens": 60, "tokenizer": "words"},
                    },
                    "embedder": {"name": "hashing", "params": {"dimensions": 256}},
                    "search_mode": mode,
                    "top_k": 5,
                }
            )
            pipeline = build_pipeline(
                config, resources=session.resources, store=session.store, collection="handbook"
            )
            await pipeline.index(corpus.documents)
            evaluation = await evaluate_retrieval(pipeline, corpus.questions, metrics=METRICS)
            print(
                f"{mode:<12}" + "".join(f"{evaluation.mean(metric):>18.3f}" for metric in METRICS)
            )


if __name__ == "__main__":
    asyncio.run(main())
