from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.core.config import ComponentConfig, PipelineConfig, apply_overrides, load_config
from ragbench.core.errors import ConfigError
from ragbench.experiment.runner import validate_components

MINIMAL = """
name: tiny
corpus: {name: folder, params: {path: docs}}
retrieval:
  chunkers: [{name: fixed, params: {max_tokens: 64}}]
  embedders: [{name: hashing}]
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(MINIMAL)
    return path


def test_defaults_are_explicit(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.seed == 42
    assert config.limit is None
    assert config.retrieval.search_modes == ["dense", "lexical", "hybrid"]
    assert config.retrieval.top_k == 10
    assert config.retrieval.primary_metric == "char_recall@5"
    assert config.rerank.rerankers == []
    assert config.answer.generators == []
    assert config.provider.max_concurrency == 8
    assert config.database.schema_name == "ragbench"
    validate_components(config)


def test_precedence_is_yaml_then_environment_then_command_line(config_file: Path) -> None:
    env = {"RAGBENCH_SEED": "7", "RAGBENCH_RETRIEVAL__TOP_K": "20", "UNRELATED": "x"}
    from_env = load_config(config_file, env=env)
    assert (from_env.seed, from_env.retrieval.top_k) == (7, 20)

    both = load_config(
        config_file, env=env, overrides=["seed=9", "retrieval.chunkers.0.params.max_tokens=32"]
    )
    assert both.seed == 9
    assert both.retrieval.top_k == 20
    assert both.retrieval.chunkers[0].params == {"max_tokens": 32}


def test_run_id_follows_the_settings_that_decide_results(config_file: Path) -> None:
    base = load_config(config_file)
    assert base.run_id == f"tiny-{base.fingerprint()[:8]}"
    assert load_config(config_file).fingerprint() == base.fingerprint()
    elsewhere = load_config(
        config_file,
        overrides=[
            "database.url=postgresql://u:secret@db/x",
            "provider.max_concurrency=2",
            "data_dir=/tmp/d",
        ],
    )
    assert elsewhere.fingerprint() == base.fingerprint(), "where things run does not change results"
    assert load_config(config_file, overrides=["limit=5"]).fingerprint() != base.fingerprint()
    assert load_config(config_file, overrides=["seed=1"]).fingerprint() != base.fingerprint()
    assert "secret" not in str(elsewhere.resolved())
    assert elsewhere.resolved()["database"] == {
        "url": "postgresql://u:***@db/x",
        "schema": "ragbench",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "retrieval.search_modes=[fuzzy]",
            "retrieval.search_modes.0: Input should be 'dense', 'lexical' or 'hybrid'",
        ),
        ("retrieval.top_k=0", "retrieval.top_k: Input should be greater than or equal to 1"),
        ("unknown_key=1", "unknown_key: Extra inputs are not permitted"),
        ("retrieval.primary_metric=ndcg@3", "primary_metric 'ndcg@3' must be one of metrics"),
        ("retrieval.metrics=[recall@50]", "metric 'recall@50' needs k <= top_k (10)"),
        ("answer.generators=[{name: extractive}]", "judge is required when generators are set"),
        ("answer.context_k=11", "answer.context_k (11) must be <= retrieval.top_k (10)"),
        ("name=Has Spaces", "name: String should match pattern"),
        ("database.schema=Bad-Name", "database.schema: String should match pattern"),
    ],
)
def test_invalid_values_fail_at_load_time_naming_the_field(
    config_file: Path, override: str, message: str
) -> None:
    with pytest.raises(ConfigError) as raised:
        load_config(config_file, overrides=[override])
    assert message in str(raised.value)
    assert str(config_file) in str(raised.value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "retrieval.chunkers.0.name=nope",
            "unknown chunker 'nope'; allowed: context_header, fixed",
        ),
        (
            "retrieval.chunkers.0.params.size=3",
            "chunker 'fixed' does not accept size; allowed parameters: max_tokens, overlap, tokenizer",
        ),
        (
            "retrieval.embedders.0.name=openrouter",
            "embedder 'openrouter' needs parameters: model, dimensions",
        ),
        (
            "corpus.name=wikipedia",
            "unknown corpus loader 'wikipedia'; allowed: folder, legalbench_rag",
        ),
        ("retrieval.metrics=[bleu@5]", "unknown metric 'bleu'"),
    ],
)
def test_unknown_components_and_parameters_are_caught_before_running(
    config_file: Path, override: str, message: str
) -> None:
    extra = ["retrieval.primary_metric=bleu@5"] if "bleu" in override else []
    config = load_config(config_file, overrides=[override, *extra])
    with pytest.raises(ConfigError) as raised:
        validate_components(config)
    assert message in str(raised.value)


def test_unreadable_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read configuration"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(tmp_path / "list.yaml")
    (tmp_path / "broken.yaml").write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        load_config(tmp_path / "broken.yaml")


def test_malformed_overrides() -> None:
    with pytest.raises(ConfigError, match="must look like key.path=value"):
        apply_overrides({}, overrides=["novalue"])
    with pytest.raises(ConfigError, match="is not a list index"):
        apply_overrides({"items": [1]}, overrides=["items.5=2"])
    with pytest.raises(ConfigError, match="is not a mapping or a list"):
        apply_overrides({"a": 1}, overrides=["a.b=2"])
    assert apply_overrides({"a": {"b": [1, 2]}}, overrides=["a.b.1=9", "c.d=x"]) == {
        "a": {"b": [1, 9]},
        "c": {"d": "x"},
    }


def test_component_and_pipeline_identity() -> None:
    plain = ComponentConfig(name="fixed")
    sized = ComponentConfig(name="fixed", params={"max_tokens": 64})
    labelled = ComponentConfig(name="fixed", params={"max_tokens": 64}, label="small")
    model = ComponentConfig(name="openrouter", params={"model": "baai/bge-m3", "dimensions": 8})
    assert plain.key == "fixed"
    assert sized.key == labelled.key != plain.key
    assert (plain.display, labelled.display, model.display) == ("fixed", "small", "bge-m3")

    first = PipelineConfig(chunker=sized, embedder=model, search_mode="hybrid")
    second = PipelineConfig(chunker=labelled, embedder=model, search_mode="hybrid")
    assert first.key == second.key, "a label does not change what is computed"
    assert first.key != first.model_copy(update={"top_k": 9}).key
    assert second.display == "small | bge-m3 | hybrid"
    full = second.model_copy(
        update={
            "reranker": ComponentConfig(name="overlap"),
            "generator": ComponentConfig(name="extractive"),
        }
    )
    assert full.display == "small | bge-m3 | hybrid | + overlap | -> extractive"
