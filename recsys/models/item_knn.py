"""Item-based k-nearest-neighbour recommender (Lecture 1.2, slides 5-11).

Fitting computes an item x item similarity matrix from the training data and
keeps, for every item i, only its k most similar items (its neighbourhood).
A user's score for item i is the sum of the similarities between i and the
neighbours of i that the user has in their history:

    score(u, i) = sum of sim(i, j) over j in N_k(i) that u rated

This is the top-N form of item-based kNN (Deshpande and Karypis, 2004). The
score is used only for ranking, not as a predicted rating (slide 11).
Negative similarities are dropped, so only items that resemble i can vote.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender

SIMILARITIES = ("cosine", "jaccard", "adjusted_cosine")


def item_similarity(train: TrainData, similarity: str, min_rating: int | None = None) -> np.ndarray:
    """Dense item x item similarity with a zero diagonal.

    cosine           cosine between binary item columns (who rated the item)
    jaccard          |users of i and j| / |users of i or j|
    adjusted_cosine  cosine between item columns after subtracting each
                     user's mean rating (Sarwar et al., 2001); uses the stars
    min_rating       if set, only ratings >= min_rating enter the matrix
    """
    if similarity not in SIMILARITIES:
        raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")

    if similarity == "adjusted_cosine":
        x = train.matrix("rating", min_rating)
        counts = np.diff(x.indptr)
        means = np.divide(np.asarray(x.sum(axis=1)).ravel(), counts, out=np.zeros(x.shape[0]), where=counts > 0)
        x.data -= np.repeat(means, counts).astype(x.data.dtype)
    else:
        x = train.matrix("binary", min_rating)

    co = (x.T @ x).toarray().astype(np.float64)
    diag = np.diag(co).copy()
    if similarity == "jaccard":
        denom = diag[:, None] + diag[None, :] - co
    else:
        norms = np.sqrt(diag)
        denom = np.outer(norms, norms)
    sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def keep_top_k(sim: np.ndarray, k: int) -> sp.csr_matrix:
    """Keep the k largest positive entries of every column.

    Column i holds the neighbourhood of target item i, so after pruning
    history @ result sums, for each item, the similarities of its neighbours
    that the user rated.
    """
    n = sim.shape[0]
    if k >= n:
        rows, cols = np.nonzero(sim > 0)
        return sp.csr_matrix((sim[rows, cols], (rows, cols)), shape=sim.shape)
    top = np.argpartition(-sim, k - 1, axis=0)[:k]          # (k, n): neighbour rows per column
    values = np.take_along_axis(sim, top, axis=0)
    cols = np.broadcast_to(np.arange(n), top.shape)
    keep = values > 0
    return sp.csr_matrix((values[keep], (top[keep], cols[keep])), shape=sim.shape)


class ItemKNN(Recommender):
    """Item-based kNN for top-N ranking.

    k           neighbourhood size per item
    similarity  "cosine", "jaccard" or "adjusted_cosine"
    min_rating  training signal: None uses every rating, 4 uses only ratings
                >= 4, both for the similarities and for the user's history
    """

    name = "ItemKNN"

    def __init__(self, k: int = 50, similarity: str = "cosine", min_rating: int | None = None) -> None:
        super().__init__(k=k, similarity=similarity, min_rating=min_rating)
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")
        self.k = k
        self.similarity = similarity
        self.min_rating = min_rating
        self.neighbours: sp.csr_matrix | None = None
        self.history: sp.csr_matrix | None = None

    def _fit(self, train: TrainData) -> None:
        self._prune(train, item_similarity(train, self.similarity, self.min_rating))

    def fit_from_similarity(self, train: TrainData, sim: np.ndarray) -> ItemKNN:
        """Fit with a precomputed full similarity matrix. Tuning uses this to
        compute the similarity once and try every k on it."""
        self._train = train
        self._prune(train, sim)
        return self

    def _prune(self, train: TrainData, sim: np.ndarray) -> None:
        self.neighbours = keep_top_k(sim, self.k)
        self.history = train.matrix("binary", self.min_rating)

    def describe(self) -> str:
        signal = "all ratings" if self.min_rating is None else f"ratings >= {self.min_rating}"
        return f"k={self.k}, {self.similarity}, training signal: {signal}"

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.history[users] @ self.neighbours).toarray()
