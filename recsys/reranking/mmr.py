"""Maximal marginal relevance (MMR) re-ranking (Lecture 5.1, slide 9).

Each step picks the candidate maximising

    lambda * relevance(i) - (1 - lambda) * max similarity(i, items already picked)

so lambda = 1 keeps the relevance order and lambda = 0 picks, after the most
relevant item, whatever is least similar to the list so far. Similarity is the
cosine of the item vectors (here the 18 genre indicators), so the item
distance is 1 - cosine.

Relevance is min-max normalised within each candidate list by default.
EASE scores have no fixed scale, and without the rescaling the same lambda
would mean a different exchange rate for every user. The rescaling stretches
relevance differences, so a given lambda is not the slide's lambda: on the
worked example of slides 8 and 9 the raw scores give {i1, i2, i3} and the
normalised ones {i1, i3, i4}. normalise=False keeps the scores as they are.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank_batch(scores: np.ndarray, vectors: np.ndarray, k: int, lambda_: float,
                     normalise: bool = True) -> np.ndarray:
    """MMR for many candidate lists at once.

    scores   (n_lists, n) relevance of each candidate; -inf marks an empty slot
             (a user with fewer eligible movies than the pool size)
    vectors  (n_lists, n, d) non-negative item vectors of the candidates
    Returns  (n_lists, k) positions into the candidate lists, in picking order.
    Ties go to the earlier candidate, so candidates sorted by relevance with
    the evaluator's tie order give back exactly that order at lambda = 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if scores.ndim != 2 or vectors.shape[:2] != scores.shape:
        raise ValueError("vectors must have shape (n_lists, n_candidates, d) matching scores")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    if (vectors < 0).any():
        raise ValueError("item vectors must be non-negative")
    available = np.isfinite(scores)
    if (available.sum(axis=1) < k).any():
        raise ValueError(f"every list needs at least k={k} candidates")

    relevance = np.where(available, scores, 0.0)
    if normalise:
        low = np.where(available, scores, np.inf).min(axis=1, keepdims=True)
        high = np.where(available, scores, -np.inf).max(axis=1, keepdims=True)
        span = high - low
        relevance = np.where(span > 0, (relevance - low) / np.where(span > 0, span, 1.0), 0.0)
        relevance[~available] = 0.0

    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    rows = np.arange(len(scores))
    picked = np.empty((len(scores), k), dtype=np.int64)
    max_similarity = np.zeros_like(relevance)
    for step in range(k):
        # The first pick has nothing to be similar to: it is the most relevant item.
        value = relevance.copy() if step == 0 else lambda_ * relevance - (1.0 - lambda_) * max_similarity
        value[~available] = -np.inf
        best = value.argmax(axis=1)
        picked[:, step] = best
        available[rows, best] = False
        similarity = np.einsum("bnd,bd->bn", unit, unit[rows, best])
        max_similarity = np.maximum(max_similarity, np.clip(similarity, 0.0, 1.0))
    return picked


def mmr_rerank(item_indices: np.ndarray, relevance_scores: np.ndarray, vectors: np.ndarray, k: int = 10,
               lambda_: float = 0.5, normalise: bool = True) -> np.ndarray:
    """MMR for one candidate list; returns the chosen item indices in order."""
    items = np.asarray(item_indices)
    scores = np.asarray(relevance_scores, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if items.ndim != 1 or scores.shape != items.shape:
        raise ValueError("item_indices and relevance_scores must be 1-D and of equal length")
    if len(np.unique(items)) != len(items):
        raise ValueError("item_indices must not contain duplicates")
    if vectors.ndim != 2 or len(vectors) != len(items):
        raise ValueError("vectors must have one row per candidate")
    if not np.isfinite(scores).all():
        raise ValueError("relevance_scores must be finite")
    positions = mmr_rerank_batch(scores[None, :], vectors[None, :, :], min(k, len(items)), lambda_, normalise)
    return items[positions[0]]
