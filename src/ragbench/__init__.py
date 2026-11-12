"""ragbench: a reproducible RAG benchmark on PostgreSQL and OpenRouter.

Everything a caller needs is exported here; anything else is internal and may change. The
three ways in:

* Run an experiment: `load_config`, `open_runtime`, `run_experiment`, `Report`.
* Build one pipeline: `open_session`, `PipelineConfig`, `build_pipeline`,
  `evaluate_retrieval`, `judge_answers`.
* Extend it: the protocols (`Chunker`, `Embedder`, ...) and the ``register_*`` decorators.

Example:
    >>> import asyncio, os
    >>> from ragbench import load_config, open_runtime, run_experiment, Report
    >>> async def main() -> None:
    ...     config = load_config("configs/offline.yaml")
    ...     async with open_runtime(config, env=os.environ) as runtime:
    ...         summary = await run_experiment(runtime)
    ...         report = await Report.load(runtime.results, summary.run_id)
    ...         report.write(f"reports/{summary.run_id}")
    >>> asyncio.run(main())  # doctest: +SKIP
"""

from importlib.metadata import version

import ragbench.components  # noqa: F401  (registers the built-in implementations)
from ragbench.core.config import (
    ComponentConfig,
    ExperimentConfig,
    PipelineConfig,
    ProviderConfig,
    load_config,
)
from ragbench.core.errors import (
    ConfigError,
    DatasetError,
    ExperimentError,
    ProviderError,
    RagBenchError,
    RegistryError,
    StoreError,
)
from ragbench.core.interfaces import (
    Cache,
    Chunker,
    CorpusLoader,
    Embedder,
    Generator,
    Judge,
    Metric,
    Reranker,
    Retriever,
    Store,
    Tokenizer,
)
from ragbench.core.registries import (
    chunkers,
    register_cache,
    register_chunker,
    register_embedder,
    register_generator,
    register_judge,
    register_loader,
    register_metric,
    register_reranker,
    register_store,
    register_tokenizer,
)
from ragbench.core.resources import Resources
from ragbench.core.types import (
    Chunk,
    Corpus,
    Document,
    Generation,
    Message,
    Question,
    Retrieval,
    RetrievedChunk,
    Span,
    Usage,
    Verdict,
)
from ragbench.experiment.report import Recommendation, Report
from ragbench.experiment.runner import (
    Experiment,
    Progress,
    RunSummary,
    load_corpus,
    run_experiment,
    validate_components,
)
from ragbench.experiment.runtime import Runtime, Session, open_runtime, open_session
from ragbench.pipeline import (
    Answer,
    JudgeItem,
    Pipeline,
    RetrievalEvaluation,
    build_pipeline,
    evaluate_retrieval,
    judge_answers,
)

__version__ = version("ragbench")


def get_chunker(name: str, resources: Resources | None = None, **params: object) -> Chunker:
    """Build a registered chunker by name.

    Args:
        name: Registered name: ``fixed``, ``recursive``, ``parent_child``, ``semantic``,
            ``context_header``, or one added through the ``ragbench.chunkers`` entry point.
        resources: Shared dependencies. Only chunkers that call a model (``semantic``) need
            them; the others work without.
        **params: Parameters of the chunker, for example ``max_tokens=256``.

    Raises:
        ConfigError: If the name is unknown or a parameter does not fit.

    Example:
        >>> import asyncio
        >>> chunker = get_chunker("recursive", max_tokens=4, tokenizer="words")
        >>> chunks = asyncio.run(chunker.chunk(Document(id="d", text="One two. Three four five.")))
        >>> [chunk.text for chunk in chunks]
        ['One two.', 'Three four five.']
    """
    return chunkers.create(name, resources, params)


__all__ = [
    "Answer",
    "Cache",
    "Chunk",
    "Chunker",
    "ComponentConfig",
    "ConfigError",
    "Corpus",
    "CorpusLoader",
    "DatasetError",
    "Document",
    "Embedder",
    "Experiment",
    "ExperimentConfig",
    "ExperimentError",
    "Generation",
    "Generator",
    "Judge",
    "JudgeItem",
    "Message",
    "Metric",
    "Pipeline",
    "PipelineConfig",
    "Progress",
    "ProviderConfig",
    "ProviderError",
    "Question",
    "RagBenchError",
    "Recommendation",
    "RegistryError",
    "Report",
    "Reranker",
    "Resources",
    "Retrieval",
    "RetrievalEvaluation",
    "RetrievedChunk",
    "Retriever",
    "RunSummary",
    "Runtime",
    "Session",
    "Span",
    "Store",
    "StoreError",
    "Tokenizer",
    "Usage",
    "Verdict",
    "__version__",
    "build_pipeline",
    "evaluate_retrieval",
    "get_chunker",
    "judge_answers",
    "load_config",
    "load_corpus",
    "open_runtime",
    "open_session",
    "register_cache",
    "register_chunker",
    "register_embedder",
    "register_generator",
    "register_judge",
    "register_loader",
    "register_metric",
    "register_reranker",
    "register_store",
    "register_tokenizer",
    "run_experiment",
    "validate_components",
]
