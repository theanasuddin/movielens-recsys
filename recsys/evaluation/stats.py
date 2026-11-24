"""Uncertainty of per-user metric means (Lecture 3.1, slide 10).

The unit of resampling is the user: every evaluated user contributes one
metric value, and the confidence interval describes how much the mean would
move with a different sample of users from the same period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    confidence: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_ci(values: np.ndarray, n_samples: int, confidence: float, seed: int) -> Interval:
    """Percentile bootstrap interval for the mean of values."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_samples, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return Interval(float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), confidence)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_samples: int, confidence: float, seed: int) -> Interval:
    """Interval for mean(a - b) where a[i] and b[i] belong to the same user."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired comparison needs one value per user for both models")
    return bootstrap_ci(a - b, n_samples, confidence, seed)


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, n_samples: int, confidence: float,
                    seed: int) -> Interval:
    """Interval for sum(numerator) / sum(denominator) with users as the unit of
    resampling, such as the share of swapped-in movies that were relevant."""
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, numerator.size, size=(n_samples, numerator.size))
    totals = denominator[idx].sum(axis=1)
    ratios = np.divide(numerator[idx].sum(axis=1), totals, out=np.full(n_samples, np.nan), where=totals > 0)
    alpha = (1 - confidence) / 2
    return Interval(float(numerator.sum() / denominator.sum()), float(np.nanquantile(ratios, alpha)),
                    float(np.nanquantile(ratios, 1 - alpha)), confidence)


def bootstrap_system_diff(lists_a: np.ndarray, lists_b: np.ndarray, statistic, n_samples: int,
                          confidence: float, seed: int) -> Interval:
    """Interval for statistic(a) - statistic(b) when the statistic belongs to
    the whole set of lists, such as coverage or the Gini index.

    Row i of lists_a and lists_b belongs to the same user; each resample draws
    users with replacement and recomputes both statistics on the same users.
    The mean of the Interval is the difference on the full set of users.
    """
    lists_a, lists_b = np.asarray(lists_a), np.asarray(lists_b)
    if len(lists_a) != len(lists_b):
        raise ValueError("Paired comparison needs one list per user for both settings")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_samples)
    for s in range(n_samples):
        idx = rng.integers(0, len(lists_a), size=len(lists_a))
        diffs[s] = statistic(lists_a[idx]) - statistic(lists_b[idx])
    alpha = (1 - confidence) / 2
    return Interval(float(statistic(lists_a) - statistic(lists_b)), float(np.quantile(diffs, alpha)),
                    float(np.quantile(diffs, 1 - alpha)), confidence)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired per-user values. Returns (statistic, p-value).

    Users with identical values for both models are dropped (zero_method="wilcox").
    If every user is tied there is nothing to test and the p-value is 1.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.allclose(a, b):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    The smallest p-value is multiplied by m, the next by m - 1, and so on,
    keeping the adjusted values non-decreasing. Comparing each adjusted value
    with 0.05 controls the chance of any false positive in the family.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
