
"""Part 3: Genre diversity and calibration metrics."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import rel_entr


def _check_genre_vectors(genre_vectors):
    genres = np.asarray(genre_vectors, dtype=np.float64)

    if genres.ndim != 2 or genres.shape[1] == 0:
        raise ValueError("genre_vectors must be a 2-D array")

    if not np.isfinite(genres).all() or np.any(genres < 0):
        raise ValueError("genre vectors must be finite and nonnegative")

    if np.any(genres.sum(axis=1) == 0):
        raise ValueError("every movie must have at least one genre")

    return genres


def _check_recommendations(recommendations, n_items):
    recs = np.asarray(recommendations)

    if recs.ndim != 2 or recs.dtype.kind not in "iu":
        raise ValueError("recommendations must be a 2-D integer array")

    if np.any(recs < 0) or np.any(recs >= n_items):
        raise ValueError("recommendations contain invalid item indices")

    return recs


def intra_list_diversity(
    recommendations: np.ndarray,
    genre_vectors: np.ndarray,
) -> np.ndarray:
    """Calculate average pairwise cosine distance for each user.

    recommendations: shape (n_users, k)
    genre_vectors: shape (n_items, n_genres)

    Returns one ILD value per user.
    """
    genres = _check_genre_vectors(genre_vectors)
    recs = _check_recommendations(recommendations, len(genres))

    n_users, k = recs.shape

    if k < 2:
        return np.zeros(n_users, dtype=np.float64)

    selected = genres[recs]

    norms = np.linalg.norm(selected, axis=2, keepdims=True)
    normalized = selected / norms

    cosine = normalized @ np.swapaxes(normalized, 1, 2)
    distances = 1.0 - np.clip(cosine, 0.0, 1.0)

    # Only use each unordered movie pair once.
    row, col = np.triu_indices(k, k=1)

    return distances[:, row, col].mean(axis=1)


def genre_calibration_error(
    recommendations: np.ndarray,
    genre_vectors: np.ndarray,
    positive_history: sp.spmatrix | np.ndarray,
    user_indices: np.ndarray,
) -> np.ndarray:
    """Compare historical and recommended genre distributions.

    Uses Jensen-Shannon divergence (base 2).

    positive_history is a user-item matrix of positive ratings
    recorded before the evaluation cutoff.

    Returns one calibration error per user.
    Users without positive history receive NaN.
    """
    genres = _check_genre_vectors(genre_vectors)
    recs = _check_recommendations(recommendations, len(genres))

    users = np.asarray(user_indices)

    if users.ndim != 1 or users.dtype.kind not in "iu":
        raise ValueError("user_indices must be a 1-D integer array")

    if len(users) != len(recs):
        raise ValueError("users and recommendations must match")

    if sp.issparse(positive_history):
        history = positive_history.tocsr()
    else:
        history = np.asarray(positive_history)

    if history.ndim != 2 or history.shape[1] != len(genres):
        raise ValueError("positive_history has incorrect shape")

    if np.any(users < 0) or np.any(users >= history.shape[0]):
        raise ValueError("user_indices contain invalid indices")

    # A movie contributes equally to each of its genres.
    genre_shares = genres / genres.sum(axis=1, keepdims=True)

    # Count historical positive interactions as binary feedback.
    if sp.issparse(history):
        selected_history = history[users].copy()
        selected_history.data = np.ones_like(
            selected_history.data, dtype=np.float64
        )
        historical_counts = np.asarray(
            selected_history @ genre_shares
        )
    else:
        historical_counts = (
            (history[users] > 0).astype(np.float64) @ genre_shares
        )

    recommended_counts = genre_shares[recs].sum(axis=1)

    historical_total = historical_counts.sum(axis=1)
    recommended_total = recommended_counts.sum(axis=1)

    errors = np.full(len(recs), np.nan)

    valid = (historical_total > 0) & (recommended_total > 0)

    if not valid.any():
        return errors

    p = historical_counts[valid] / historical_total[valid, None]
    q = recommended_counts[valid] / recommended_total[valid, None]

    midpoint = (p + q) / 2.0

    # Jensen-Shannon divergence in bits.
    jsd = 0.5 * (
        rel_entr(p, midpoint).sum(axis=1)
        + rel_entr(q, midpoint).sum(axis=1)
    ) / np.log(2.0)

    errors[valid] = np.clip(jsd, 0.0, 1.0)

    return errors
