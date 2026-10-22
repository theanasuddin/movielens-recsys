"""Ranking and beyond-accuracy metrics (Lecture 3.1, slides 7-9; Lecture 5.1, slide 7).

The accuracy metrics take a boolean hit matrix: hits[u, r] is True when the
item at rank r+1 of user u's list is relevant. n_relevant[u] counts all of the
user's relevant held-out items, including any the model could not recommend.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(hits: np.ndarray) -> np.ndarray:
    """Share of the k recommended items that are relevant."""
    return hits.sum(axis=1) / hits.shape[1]


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Share of the user's relevant items that appear in the top k."""
    return hits.sum(axis=1) / n_relevant


def dcg_weights(k: int) -> np.ndarray:
    """Position discounts 1 / log2(rank + 1) for ranks 1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Binary-relevance NDCG@k. The ideal list puts min(n_relevant, k) hits first."""
    k = hits.shape[1]
    weights = dcg_weights(k)
    dcg = hits @ weights
    ideal = np.cumsum(weights)[np.minimum(n_relevant, k) - 1]
    return dcg / ideal


def catalogue_coverage(recommendations: np.ndarray, catalogue_size: int) -> float:
    """Share of the candidate catalogue that appears in at least one top-k list."""
    return np.unique(recommendations).size / catalogue_size


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector.

    0 means every entry is equal; values near 1 mean a few entries hold
    almost everything. Used for item popularity and recommendation exposure.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(((2 * ranks - n - 1) @ x) / (n * x.sum()))
