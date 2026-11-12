"""One registry per extension point.

Built-in implementations register themselves when `ragbench.components` is imported.
"""

from __future__ import annotations

from ragbench.core.interfaces import (
    Cache,
    Chunker,
    CorpusLoader,
    Embedder,
    Generator,
    Judge,
    Metric,
    Reranker,
    Store,
    Tokenizer,
)
from ragbench.core.registry import Registry

tokenizers: Registry[Tokenizer] = Registry("tokenizer", "ragbench.tokenizers")
chunkers: Registry[Chunker] = Registry("chunker", "ragbench.chunkers")
embedders: Registry[Embedder] = Registry("embedder", "ragbench.embedders")
rerankers: Registry[Reranker] = Registry("reranker", "ragbench.rerankers")
generators: Registry[Generator] = Registry("generator", "ragbench.generators")
judges: Registry[Judge] = Registry("judge", "ragbench.judges")
metrics: Registry[Metric] = Registry("metric", "ragbench.metrics")
loaders: Registry[CorpusLoader] = Registry("corpus loader", "ragbench.loaders")
stores: Registry[Store] = Registry("store", "ragbench.stores")
caches: Registry[Cache] = Registry("cache", "ragbench.caches")

register_tokenizer = tokenizers.register
register_chunker = chunkers.register
register_embedder = embedders.register
register_reranker = rerankers.register
register_generator = generators.register
register_judge = judges.register
register_metric = metrics.register
register_loader = loaders.register
register_store = stores.register
register_cache = caches.register
