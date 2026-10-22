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


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired per-user values. Returns (statistic, p-value).

    Users with identical values for both models are dropped (zero_method="wilcox").
    """
    result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)
