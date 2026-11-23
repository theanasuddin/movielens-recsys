
"""Maximal Marginal Relevance (MMR) re-ranking for Part 3, Track A.

Input movie IDs are *internal item indices*, not raw MovieLens IDs.
The three inputs must be aligned and describe the same candidate movies.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(
    item_indices: np.ndarray,
    relevance_scores: np.ndarray,
    genre_vectors: np.ndarray,
    k: int = 10,
    lambda_: float = 0.5,
) -> np.ndarray:
    """Re-rank candidate movies by relevance and genre dissimilarity.

    Args:
        item_indices: Internal item indices (e.g. EASE Top-100), length n.
        relevance_scores: EASE scores in the same order, length n.
        genre_vectors: Binary genre vectors in the same order, shape (n, 18).
        k: Number of movies to return.
        lambda_: Relevance weight in [0, 1]. A value of 1 preserves
            relevance ranking; 0 emphasizes genre dissimilarity.

    Returns:
        A 1-D array of selected *internal item indices*, in ranked order.

    Notes:
        Scores are min-max normalized within the supplied candidate list.
        Ties preserve input candidate order. To reproduce the original EASE
        order with lambda_=1, supply candidates already sorted by EASE's
        score with the project's existing deterministic tie-breaking.
        The first item is always chosen by relevance (also when lambda_=0).
    """
    items = np.asarray(item_indices)
    scores = np.asarray(relevance_scores, dtype=np.float64)
    genres = np.asarray(genre_vectors, dtype=np.float64)

    if items.ndim != 1 or scores.ndim != 1:
        raise ValueError("item_indices and relevance_scores must be 1-D")
    if genres.ndim != 2 or genres.shape[0] != len(items):
        raise ValueError("genre_vectors must have shape (n_candidates, n_genres)")
    if len(items) != len(scores):
        raise ValueError("item_indices and relevance_scores must have equal lengths")
    if len(np.unique(items)) != len(items):
        raise ValueError("item_indices must not contain duplicates")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    if not np.isfinite(scores).all() or not np.isfinite(genres).all():
        raise ValueError("relevance_scores and genre_vectors must be finite")
    if (genres < 0).any():
        raise ValueError("genre_vectors cannot contain negative values")

    n = len(items)
    if n == 0 or k == 0:
        return items[:0].copy()

    # Normalize EASE scores into [0, 1] within the candidate set.
    score_min = scores.min()
    score_range = scores.max() - score_min
    relevance = ((scores - score_min) / score_range
                 if score_range > 0 else np.zeros(n, dtype=np.float64))

    # Normalize genre vectors so dot products are cosine similarities.
    norms = np.linalg.norm(genres, axis=1)
    unit_genres = np.divide(
        genres,
        norms[:, None],
        out=np.zeros_like(genres),
        where=norms[:, None] > 0,
    )

    selected_positions: list[int] = []
    available = np.ones(n, dtype=bool)
    max_similarity = np.zeros(n, dtype=np.float64)

    for _ in range(min(k, n)):
        if not selected_positions:
            # No selected items yet: choose the most relevant item.
            mmr_scores = relevance.copy()
        else:
            mmr_scores = (
                lambda_ * relevance - (1.0 - lambda_) * max_similarity
            )

        mmr_scores[~available] = -np.inf
        best = int(np.argmax(mmr_scores))
        selected_positions.append(best)
        available[best] = False

        # Track the greatest similarity to any previously selected item.
        similarity = unit_genres @ unit_genres[best]
        max_similarity = np.maximum(
            max_similarity, np.clip(similarity, 0.0, 1.0)
        )

    return items[np.asarray(selected_positions)]
