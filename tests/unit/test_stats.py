from __future__ import annotations

import pytest

from ragbench.core.errors import ExperimentError
from ragbench.experiment.runner import rank_configurations
from ragbench.experiment.stats import (
    bootstrap_mean,
    cohen_kappa,
    paired_bootstrap,
    raw_agreement,
    stratified_sample,
)


def test_bootstrap_mean_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 50
    first = bootstrap_mean(values, samples=1000, seed=3)
    assert first == bootstrap_mean(values, samples=1000, seed=3)
    assert first != bootstrap_mean(values, samples=1000, seed=4)
    assert first.mean == 0.5
    assert 0.38 < first.low < 0.5 < first.high < 0.62
    constant = bootstrap_mean([0.7] * 10)
    assert constant.low == pytest.approx(0.7)
    assert constant.high == pytest.approx(0.7)
    with pytest.raises(ExperimentError, match="empty"):
        bootstrap_mean([])


def test_paired_bootstrap_sees_a_small_consistent_gain_that_unpaired_noise_hides() -> None:
    # Questions differ a lot in difficulty, but the second system is 0.05 better on every one.
    base = [(number % 10) / 10 for number in range(200)]
    better = [value + 0.05 for value in base]
    gain = paired_bootstrap(better, base, samples=1000, seed=1)
    assert gain.mean == pytest.approx(0.05)
    assert gain.low == pytest.approx(0.05)
    assert gain.excludes_zero
    # The two unpaired intervals overlap heavily, which is why the comparison must be paired.
    assert bootstrap_mean(better, seed=1).low < bootstrap_mean(base, seed=1).high


def test_paired_bootstrap_reports_no_difference_for_noise() -> None:
    first = [1.0, 0.0] * 40
    second = [0.0, 1.0] * 40
    difference = paired_bootstrap(first, second, samples=1000, seed=1)
    assert difference.mean == 0.0
    assert not difference.excludes_zero
    with pytest.raises(ExperimentError, match="same length"):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_cohen_kappa_known_values() -> None:
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4.
    first = [1] * 25 + [0] * 25
    second = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(first, second) == pytest.approx(0.4)
    assert raw_agreement(first, second) == pytest.approx(0.7)
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0
    assert cohen_kappa([2, 2, 2], [2, 2, 2]) == 1.0
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)
    with pytest.raises(ExperimentError):
        cohen_kappa([], [])


def test_weighted_kappa_forgives_near_misses() -> None:
    human = [0, 1, 2, 0, 1, 2, 0, 1, 2, 2]
    near = [1, 1, 2, 0, 2, 2, 0, 1, 1, 2]  # three disagreements, each by one step
    far = [2, 1, 2, 0, 1, 0, 0, 1, 0, 2]  # three disagreements, each by two steps
    assert raw_agreement(human, near) == raw_agreement(human, far) == 0.7
    assert cohen_kappa(human, near, weighted=True) > cohen_kappa(human, far, weighted=True)
    assert cohen_kappa(human, near, weighted=True) > cohen_kappa(human, near)
    # Hand computation for `far`: observed weighted disagreement 0.3, expected 0.355.
    assert cohen_kappa(human, far, weighted=True) == pytest.approx(1 - 0.3 / 0.355)


def test_stratified_sample_is_proportional_seeded_and_ordered() -> None:
    groups = {"a": list(range(100)), "b": list(range(100, 150)), "c": list(range(150, 160))}
    sample = stratified_sample(groups, 16, seed=5)
    assert sample == stratified_sample(groups, 16, seed=5)
    assert sample != stratified_sample(groups, 16, seed=6)
    assert [sum(item in groups[key] for item in sample) for key in groups] == [10, 5, 1]
    assert sample == sorted(sample)
    assert len(set(sample)) == 16
    assert stratified_sample(groups, 500) == list(range(160))
    thirds = stratified_sample({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]}, 2, seed=0)
    assert len(thirds) == 2


def test_rank_configurations_averages_over_all_questions() -> None:
    rows = {
        "b": {"q1": 1.0, "q2": 0.0},
        "a": {"q1": 0.5, "q2": 0.5},
        "c": {"q1": 1.0},  # q2 missing counts as zero
        "d": {"q1": 1.0, "q2": 1.0},
    }
    assert rank_configurations(rows, ["q1", "q2"]) == [
        ("d", 1.0),
        ("a", 0.5),
        ("b", 0.5),
        ("c", 0.5),
    ]
    assert rank_configurations(rows, []) == []


def test_pick_winners_keeps_the_best_overall_or_the_best_of_each_chunker() -> None:
    from ragbench.core.config import ComponentConfig, PipelineConfig
    from ragbench.core.types import SearchMode
    from ragbench.experiment.runner import pick_winners

    def spec(chunker: str, mode: SearchMode) -> PipelineConfig:
        return PipelineConfig(
            chunker=ComponentConfig(name=chunker),
            embedder=ComponentConfig(name="hashing"),
            search_mode=mode,
        )

    specs = {
        "a": spec("fixed", "dense"),
        "b": spec("fixed", "hybrid"),
        "c": spec("recursive", "dense"),
        "d": spec("semantic", "dense"),
    }
    ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
    assert pick_winners(ranking, specs, 2, "none") == [specs["a"], specs["b"]]
    assert pick_winners(ranking, specs, 2, "chunker") == [specs["a"], specs["c"]]
    assert pick_winners(ranking, specs, 9, "chunker") == [specs["a"], specs["c"], specs["d"]]
