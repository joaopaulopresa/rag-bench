"""The typed configuration tree of an experiment.

Values come from a YAML file, then environment variables, then command-line overrides; a later
source wins. Unknown keys are rejected everywhere.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from ragbench.core.errors import ConfigError
from ragbench.core.hashing import stable_hash
from ragbench.core.types import SearchMode

ENV_PREFIX = "RAGBENCH_"
Diversify = Literal["none", "chunker"]
_METRIC_PATTERN = re.compile(r"^[a-z_]+(@[1-9][0-9]*)?$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentConfig(_Model):
    """A component chosen by registered name, with its parameters."""

    name: str = Field(description="Name the implementation is registered under.")
    params: dict[str, JsonValue] = Field(
        default_factory=dict, description="Keyword parameters of the implementation."
    )
    label: str | None = Field(
        default=None, description="Name shown in reports; does not affect results."
    )

    @property
    def key(self) -> str:
        """Short stable identifier of this choice: the name plus a hash of the parameters."""
        return f"{self.name}-{stable_hash(self.params, length=8)}" if self.params else self.name

    @property
    def display(self) -> str:
        """Name shown in tables: ``label`` if set, else the model's short name, else the name."""
        if self.label:
            return self.label
        model = self.params.get("model")
        return model.rsplit("/", 1)[-1] if isinstance(model, str) else self.name


class ProviderConfig(_Model):
    """Network behaviour of model provider clients."""

    timeout_s: float = Field(default=120.0, gt=0, description="Timeout of one request.")
    max_concurrency: int = Field(default=8, ge=1, description="Requests in flight at once.")
    max_retries: int = Field(default=8, ge=0, description="Retries after a transient error.")
    backoff_s: float = Field(
        default=2.0,
        ge=0,
        description="First retry delay; doubles on each further retry, up to 60 seconds.",
    )


class DatabaseConfig(_Model):
    """Where PostgreSQL is."""

    url: str = Field(
        default="postgresql://ragbench:ragbench@localhost:54329/ragbench",
        description="Connection URL. The default matches docker-compose.yml.",
    )
    schema_name: str = Field(
        default="ragbench",
        alias="schema",
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Schema that holds every table of the package.",
    )


class RetrievalConfig(_Model):
    """Stage 1: every chunker x embedder x search mode, without a reranker."""

    chunkers: list[ComponentConfig] = Field(min_length=1)
    embedders: list[ComponentConfig] = Field(min_length=1)
    search_modes: list[SearchMode] = Field(default=["dense", "lexical", "hybrid"], min_length=1)
    top_k: int = Field(default=10, ge=1, description="Passages retrieved per question.")
    metrics: list[str] = Field(
        default=[
            "char_recall@5",
            "char_precision@5",
            "recall@5",
            "recall@10",
            "ndcg@10",
            "mrr@10",
        ],
        min_length=1,
        description="Metric specifications, as ``name`` or ``name@k``.",
    )
    primary_metric: str = Field(
        default="char_recall@5", description="Metric that ranks configurations."
    )

    @model_validator(mode="after")
    def _check_metrics(self) -> RetrievalConfig:
        for spec in self.metrics:
            if not _METRIC_PATTERN.match(spec):
                raise ValueError(f"metric {spec!r} must look like 'name' or 'name@k'")
            if "@" in spec and int(spec.split("@")[1]) > self.top_k:
                raise ValueError(f"metric {spec!r} needs k <= top_k ({self.top_k})")
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} must be one of metrics: "
                f"{', '.join(self.metrics)}"
            )
        return self


class RerankConfig(_Model):
    """Stage 2: the best stage-1 configurations, each with every reranker."""

    rerankers: list[ComponentConfig] = Field(
        default_factory=list, description="Empty list skips the stage."
    )
    top_configs: int = Field(default=5, ge=1, description="Stage-1 winners carried over.")
    candidates: int = Field(default=30, ge=1, description="Passages sent to the reranker.")
    questions: int | None = Field(
        default=None,
        ge=1,
        description="Run the stage on a stratified subsample of this size; null uses every "
        "question. Rerankers are billed per token, so this bounds the cost of the stage.",
    )
    diversify: Diversify = Field(
        default="none",
        description="'chunker' carries over the best configuration of each chunker instead of "
        "the overall best ones, which are often near-duplicates.",
    )


class AnswerConfig(_Model):
    """Stage 3: the best retrieval configurations, each with every generator, judged."""

    generators: list[ComponentConfig] = Field(
        default_factory=list, description="Empty list skips the stage."
    )
    judge: ComponentConfig | None = Field(
        default=None, description="Required when generators are set."
    )
    top_configs: int = Field(default=3, ge=1, description="Retrieval winners carried over.")
    diversify: Diversify = Field(
        default="none",
        description="'chunker' carries over the best configuration of each chunker instead of "
        "the overall best ones, which often give the generator the same passages.",
    )
    questions: int = Field(default=200, ge=1, description="Stratified subsample size.")
    context_k: int = Field(default=5, ge=1, description="Passages given to the generator.")
    validation_sample: int = Field(
        default=50, ge=0, description="Answers exported for human labelling."
    )

    @model_validator(mode="after")
    def _check_judge(self) -> AnswerConfig:
        if self.generators and self.judge is None:
            raise ValueError("judge is required when generators are set")
        return self


class BootstrapConfig(_Model):
    """Paired bootstrap used for confidence intervals."""

    samples: int = Field(default=2000, ge=100)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)


class ExperimentConfig(_Model):
    """Everything about an experiment. Nothing about it lives in code."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", description="Used in the run id.")
    seed: int = Field(default=42, description="Seed of every random choice.")
    limit: int | None = Field(
        default=None, ge=1, description="Use only this many questions (development runs)."
    )
    data_dir: Path = Field(default=Path(".data"), description="Downloaded corpora live here.")
    corpus: ComponentConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    retrieval: RetrievalConfig
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @model_validator(mode="after")
    def _check_stages(self) -> ExperimentConfig:
        if self.answer.context_k > self.retrieval.top_k:
            raise ValueError(
                f"answer.context_k ({self.answer.context_k}) must be <= retrieval.top_k "
                f"({self.retrieval.top_k})"
            )
        if self.rerank.rerankers and self.rerank.candidates < self.retrieval.top_k:
            raise ValueError("rerank.candidates must be >= retrieval.top_k")
        return self

    def resolved(self) -> dict[str, JsonValue]:
        """Return the full configuration as plain data, with the database password masked."""
        data: dict[str, JsonValue] = self.model_dump(mode="json", by_alias=True)
        database = data["database"]
        assert isinstance(database, dict)
        database["url"] = re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", str(database["url"]))
        return data

    def fingerprint(self) -> str:
        """Return a hash of the settings that decide the results.

        Where the database is, where files are downloaded and how the network client behaves do
        not change results, so they are left out.
        """
        data = self.model_dump(mode="json", by_alias=True)
        for irrelevant in ("database", "provider", "data_dir"):
            data.pop(irrelevant)
        return stable_hash(data)

    @property
    def run_id(self) -> str:
        """Identifier of the run this configuration produces: name plus fingerprint prefix.

        The same configuration always maps to the same run, which is what makes a run resumable.
        """
        return f"{self.name}-{self.fingerprint()[:8]}"


class PipelineConfig(_Model):
    """One RAG pipeline: how to chunk, embed, search, rerank and answer."""

    chunker: ComponentConfig
    embedder: ComponentConfig | None = Field(
        default=None, description="Not needed when search_mode is lexical."
    )
    search_mode: SearchMode = "hybrid"
    reranker: ComponentConfig | None = None
    generator: ComponentConfig | None = Field(
        default=None, description="Leave out for a retrieval-only pipeline."
    )
    top_k: int = Field(default=5, ge=1, description="Passages kept for the generator.")
    candidates: int = Field(default=30, ge=1, description="Passages sent to the reranker.")

    @model_validator(mode="after")
    def _check(self) -> PipelineConfig:
        if self.search_mode != "lexical" and self.embedder is None:
            raise ValueError(f"embedder is required when search_mode is {self.search_mode!r}")
        if self.reranker is not None and self.candidates < self.top_k:
            raise ValueError("candidates must be >= top_k")
        return self

    @property
    def key(self) -> str:
        """Stable identifier of the whole pipeline configuration. Labels do not change it."""
        data = self.model_dump(mode="json")
        for part in ("chunker", "embedder", "reranker", "generator"):
            if data[part] is not None:
                data[part].pop("label")
        return stable_hash(data, length=12)

    @property
    def display(self) -> str:
        """One-line description for tables, for example ``recursive | bge-m3 | hybrid``."""
        parts = [self.chunker.display]
        if self.embedder is not None:
            parts.append(self.embedder.display)
        parts.append(self.search_mode)
        if self.reranker is not None:
            parts.append(f"+ {self.reranker.display}")
        if self.generator is not None:
            parts.append(f"-> {self.generator.display}")
        return " | ".join(parts)


def format_validation_error(error: ValidationError) -> str:
    """Render a pydantic error as one line per problem: ``field.path: message``."""
    lines = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "(root)"
        lines.append(f"{location}: {problem['msg']}")
    return "; ".join(lines)


def _set_path(data: dict[str, object], path: Sequence[str], value: object, origin: str) -> None:
    node: object = data
    for depth, part in enumerate(path):
        last = depth == len(path) - 1
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                raise ConfigError(f"{origin}: {'.'.join(path[: depth + 1])} is not a list index")
            if last:
                node[int(part)] = value
            else:
                node = node[int(part)]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
        else:
            raise ConfigError(f"{origin}: {'.'.join(path[:depth])} is not a mapping or a list")


def apply_overrides(
    data: dict[str, object],
    *,
    env: Mapping[str, str] | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, object]:
    """Apply environment and command-line overrides to raw configuration data, in that order.

    Args:
        data: Parsed YAML. It is modified in place and returned.
        env: Environment. ``RAGBENCH_ANSWER__QUESTIONS=50`` sets ``answer.questions``.
        overrides: Command-line items such as ``answer.questions=50`` or
            ``retrieval.chunkers.0.params.max_tokens=256``. They win over the environment.

    Raises:
        ConfigError: If an item is malformed or addresses something that is not a container.
    """
    for name, raw in sorted((env or {}).items()):
        if name.startswith(ENV_PREFIX) and name != ENV_PREFIX:
            path = name[len(ENV_PREFIX) :].lower().split("__")
            _set_path(data, path, yaml.safe_load(raw), f"environment variable {name}")
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        _set_path(data, key.split("."), yaml.safe_load(raw), f"override {item!r}")
    return data


def load_config(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load and validate an experiment configuration.

    Args:
        path: YAML file.
        env: Environment used for ``RAGBENCH_*`` overrides. ``None`` applies none; pass
            ``os.environ`` to honour the process environment.
        overrides: ``key.path=value`` items, which win over the environment.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, is not a YAML mapping, or a value is invalid. The
            message names each offending field.

    Example:
        >>> config = load_config("configs/demo.yaml", overrides=["limit=20"])  # doctest: +SKIP
    """
    file = Path(path)
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read configuration {file}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{file} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{file} must contain a YAML mapping at the top level")
    data = apply_overrides(raw, env=env, overrides=overrides)
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(
            f"invalid configuration {file}: {format_validation_error(error)}"
        ) from error
