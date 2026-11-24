"""Measures beyond accuracy for Part 3 (Lecture 5.1, slide 7).

Per list: intra-list diversity, calibration error and novelty. Per system:
catalogue coverage and the Gini index of exposure (metrics.py). Every
per-list measure returns one value per user, so differences between two
settings can be compared user by user.

Genre vectors are the 18 MovieLens genre indicators. Where a distribution
over genres is needed, a movie with several genres gives each of them an
equal share (a Drama|Romance movie counts half to each).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import rel_entr


def _check_genres(genre_vectors: np.ndarray) -> np.ndarray:
    genres = np.asarray(genre_vectors, dtype=np.float64)
    if genres.ndim != 2 or genres.shape[1] == 0:
        raise ValueError("genre_vectors must be a 2-D array")
    if not np.isfinite(genres).all() or (genres < 0).any():
        raise ValueError("genre vectors must be finite and non-negative")
    if (genres.sum(axis=1) == 0).any():
        raise ValueError("every movie must have at least one genre")
    return genres


def _check_lists(recommendations: np.ndarray, n_items: int) -> np.ndarray:
    recs = np.asarray(recommendations)
    if recs.ndim != 2 or recs.dtype.kind not in "iu":
        raise ValueError("recommendations must be a 2-D integer array")
    if (recs < 0).any() or (recs >= n_items).any():
        raise ValueError("recommendations contain invalid item indices")
    return recs


def _pair_mean(distances: np.ndarray) -> np.ndarray:
    """Mean over the unordered pairs of a (n_lists, k, k) distance array."""
    k = distances.shape[1]
    if k < 2:
        return np.zeros(len(distances))
    row, col = np.triu_indices(k, k=1)
    return distances[:, row, col].mean(axis=1)


def intra_list_diversity(recommendations: np.ndarray, genre_vectors: np.ndarray) -> np.ndarray:
    """Mean pairwise genre distance (1 - cosine) of each list (slide 7)."""
    genres = _check_genres(genre_vectors)
    recs = _check_lists(recommendations, len(genres))
    unit = genres / np.linalg.norm(genres, axis=1, keepdims=True)
    selected = unit[recs]
    cosine = selected @ np.swapaxes(selected, 1, 2)
    return _pair_mean(1.0 - np.clip(cosine, 0.0, 1.0))


def item_cosine(matrix: sp.spmatrix, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity between the columns `items` of a user x item matrix.

    Returns the (m, m) similarity of the given items and a lookup from item
    index to row, -1 for items not included. Two movies are similar here when
    the same people rated them, the collaborative view of distance that slide
    13 warns can disagree with genre distance.
    """
    items = np.asarray(items)
    columns = sp.csc_matrix(matrix, dtype=np.float64)[:, items]
    gram = (columns.T @ columns).toarray()
    norms = np.sqrt(np.diag(gram))
    similarity = np.divide(gram, np.outer(norms, norms), out=np.zeros_like(gram), where=np.outer(norms, norms) > 0)
    lookup = np.full(matrix.shape[1], -1, dtype=np.int64)
    lookup[items] = np.arange(len(items))
    return similarity, lookup


def intra_list_distance(recommendations: np.ndarray, similarity: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Mean pairwise distance (1 - similarity) of each list under a precomputed similarity."""
    rows = lookup[np.asarray(recommendations)]
    if (rows < 0).any():
        raise ValueError("a recommended item is missing from the similarity lookup")
    return _pair_mean(1.0 - np.clip(similarity[rows[:, :, None], rows[:, None, :]], 0.0, 1.0))


def genre_shares(genre_vectors: np.ndarray) -> np.ndarray:
    """Each movie's genres as a distribution: equal shares that sum to one."""
    genres = _check_genres(genre_vectors)
    return genres / genres.sum(axis=1, keepdims=True)


def history_distribution(positive_history: sp.spmatrix, users: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Genre distribution of each user's positive history; NaN rows for users without one."""
    history = sp.csr_matrix(positive_history[np.asarray(users)], dtype=np.float64)
    history.data[:] = 1.0
    counts = np.asarray(history @ shares)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)


def list_distribution(recommendations: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Genre distribution of each recommended list."""
    recs = _check_lists(recommendations, len(shares))
    counts = shares[recs].sum(axis=1)
    return counts / counts.sum(axis=1, keepdims=True)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits, row by row: 0 for equal rows, 1 for disjoint ones."""
    middle = (p + q) / 2.0
    jsd = 0.5 * (rel_entr(p, middle).sum(axis=1) + rel_entr(q, middle).sum(axis=1)) / np.log(2.0)
    return np.clip(jsd, 0.0, 1.0)


def jensen_shannon_parts(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Jensen-Shannon divergence split by genre: the part from genres the
    list under-serves (q < p) and the part from genres it over-serves (q > p).

    Each genre's term is non-negative, so the two parts add up to the
    divergence. They show which way a list is off: missing what the user
    likes, or spending slots on what they rarely watch.
    """
    middle = (p + q) / 2.0
    terms = 0.5 * (rel_entr(p, middle) + rel_entr(q, middle)) / np.log(2.0)
    return np.where(q < p, terms, 0.0).sum(axis=1), np.where(q > p, terms, 0.0).sum(axis=1)


def genre_calibration_error(recommendations: np.ndarray, genre_vectors: np.ndarray,
                            positive_history: sp.spmatrix | np.ndarray, user_indices: np.ndarray) -> np.ndarray:
    """Calibration error per user (slides 7 and 11).

    The divergence between the genre distribution of the user's positive
    history and that of the list, measured with Jensen-Shannon divergence in
    bits. It is symmetric and bounded, and needs no smoothing when the list
    misses a genre of the history (KL divergence would be infinite there).
    Users without positive history get NaN.
    """
    shares = genre_shares(genre_vectors)
    recs = _check_lists(recommendations, len(shares))
    users = np.asarray(user_indices)
    if users.ndim != 1 or users.dtype.kind not in "iu" or len(users) != len(recs):
        raise ValueError("user_indices must be a 1-D integer array with one entry per list")
    history = sp.csr_matrix(positive_history)
    if history.shape[1] != len(shares):
        raise ValueError("positive_history has the wrong number of items")
    p = history_distribution(history, users, shares)
    q = list_distribution(recs, shares)
    errors = np.full(len(recs), np.nan)
    valid = ~np.isnan(p).any(axis=1)
    errors[valid] = jensen_shannon(p[valid], q[valid])
    return errors


def top_genre_gap(history: np.ndarray, lists: np.ndarray) -> np.ndarray:
    """Share of the list given to the user's favourite genre minus its share in the history.

    Positive values mean the list over-serves the favourite genre (slide 11:
    sorting by score turns 70/30 into 100/0), negative values that it spreads
    away from it. NaN for users without history.
    """
    valid = ~np.isnan(history).any(axis=1)
    gap = np.full(len(history), np.nan)
    favourite = np.argmax(np.where(valid[:, None], history, 0.0), axis=1)
    rows = np.flatnonzero(valid)
    gap[rows] = lists[rows, favourite[rows]] - history[rows, favourite[rows]]
    return gap


def self_information(recommendations: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Mean novelty of each list, -log2(popularity) of its items (slide 7).

    popularity[i] is the share of training users who rated movie i; a movie
    every user rated has novelty 0, one rated by 1 user in 1,024 has 10 bits.
    """
    recs = np.asarray(recommendations)
    p = np.asarray(popularity, dtype=np.float64)[recs]
    if (p <= 0).any():
        raise ValueError("recommended movies need a positive popularity")
    return (-np.log2(p)).mean(axis=1)
