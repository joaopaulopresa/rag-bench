"""Statistics for comparing configurations. Pure functions; every random choice is seeded."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ragbench.core.errors import ExperimentError

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        mean: The estimate.
        low: Lower bound of the interval.
        high: Upper bound of the interval.
    """

    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval lies on one side of zero (a significant difference)."""
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: Sequence[float], *, samples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Return the mean of ``values`` with a percentile bootstrap confidence interval.

    Args:
        values: One value per question.
        samples: Number of resamples.
        confidence: Coverage of the interval, for example 0.95.
        seed: Seed of the resampling.

    Raises:
        ExperimentError: If ``values`` is empty.
    """
    if not values:
        raise ExperimentError("cannot bootstrap an empty list of values")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[picks].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Interval(mean=float(data.mean()), low=float(low), high=float(high))


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Return the mean difference ``first - second`` with a paired bootstrap interval.

    Both lists must hold the scores of the same questions in the same order. Resampling
    questions (not scores) keeps each pair together, which removes the noise that comes from
    some questions being harder than others.

    Raises:
        ExperimentError: If the lists are empty or differ in length.
    """
    if len(first) != len(second):
        raise ExperimentError("paired bootstrap needs two lists of the same length")
    differences = [a - b for a, b in zip(first, second, strict=True)]
    return bootstrap_mean(differences, samples=samples, confidence=confidence, seed=seed)


def cohen_kappa(first: Sequence[int], second: Sequence[int], *, weighted: bool = False) -> float:
    """Return Cohen's kappa between two raters.

    Args:
        first: Labels of the first rater (integers).
        second: Labels of the second rater, for the same items in the same order.
        weighted: Use quadratic weights, which count a 0-versus-2 disagreement as worse than a
            0-versus-1 one. Suited to ordered scores.

    Returns:
        Kappa: 1.0 for perfect agreement, 0.0 for agreement no better than chance. When both
        raters use a single identical label everywhere the result is 1.0.

    Raises:
        ExperimentError: If the lists are empty or differ in length.
    """
    if not first or len(first) != len(second):
        raise ExperimentError("kappa needs two non-empty label lists of the same length")
    labels = sorted(set(first) | set(second))
    position = {label: index for index, label in enumerate(labels)}
    size = len(labels)
    observed = np.zeros((size, size), dtype=np.float64)
    for a, b in zip(first, second, strict=True):
        observed[position[a], position[b]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weighted and size > 1:
        values = np.asarray(labels, dtype=np.float64)
        span = float(max(labels) - min(labels))
        weights = ((values[:, None] - values[None, :]) / span) ** 2
    else:
        weights = 1.0 - np.eye(size)
    disagreement_expected = float((weights * expected).sum())
    if disagreement_expected == 0.0:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / disagreement_expected


def raw_agreement(first: Sequence[int], second: Sequence[int]) -> float:
    """Return the share of items on which two raters gave the same label.

    Raises:
        ExperimentError: If the lists are empty or differ in length.
    """
    if not first or len(first) != len(second):
        raise ExperimentError("agreement needs two non-empty label lists of the same length")
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first)


def stratified_sample(groups: Mapping[K, Sequence[T]], size: int, *, seed: int = 0) -> list[T]:
    """Draw ``size`` items spread over ``groups`` in proportion to their sizes.

    Each group gets its proportional share rounded down; the remaining places go to the groups
    with the largest remainders (ties to the earlier group). Items inside a group are drawn
    without replacement and keep their original order in the result.

    Args:
        groups: Items by stratum. Iteration order decides the order of the result.
        size: Number of items wanted. If it is at least the total, everything is returned.
        seed: Seed of the draw.

    Example:
        >>> stratified_sample({"a": [1, 2, 3, 4], "b": [5, 6]}, 3, seed=1)
        [2, 3, 6]
    """
    total = sum(len(items) for items in groups.values())
    if size >= total:
        return [item for items in groups.values() for item in items]
    shares = {key: size * len(items) / total for key, items in groups.items()}
    quotas = {key: int(share) for key, share in shares.items()}
    order = sorted(groups, key=lambda key: -(shares[key] - quotas[key]))
    for key in order[: size - sum(quotas.values())]:
        quotas[key] += 1
    rng = np.random.default_rng(seed)
    chosen: list[T] = []
    for key, items in groups.items():
        picks = sorted(rng.choice(len(items), size=quotas[key], replace=False).tolist())
        chosen.extend(items[pick] for pick in picks)
    return chosen
