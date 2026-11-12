"""User-based k-nearest-neighbour recommender (Lecture 1.2, slides 5-8).

Users are compared on the binary training matrix. Each user keeps the k most
similar other users with a positive similarity, and a movie's score is the
similarity-weighted number of those neighbours who rated it:

    score(u, i) = sum of sim(u, v) over the neighbours v of u that rated i

Slide 5 picks, for every target movie, the k most similar users among those
who rated it. Here every user has one fixed neighbourhood instead, which turns
scoring into a single sparse matrix product; with the large k that validation
prefers, the two give nearly the same neighbours. Dividing by the summed
similarity, as in the rating-prediction formula, would scale all of a user's
scores by the same number and leave the ranking unchanged, so it is skipped.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender
from recsys.models.item_knn import keep_top_k

SIMILARITIES = ("cosine", "jaccard")


def user_similarity(x: sp.csr_matrix, similarity: str) -> np.ndarray:
    """Dense user x user similarity of a binary matrix, with a zero diagonal."""
    co = (x @ x.T).toarray().astype(np.float32)
    sizes = np.diag(co).copy()
    if similarity == "jaccard":
        denom = sizes[:, None] + sizes[None, :] - co
    else:
        denom = np.sqrt(np.outer(sizes, sizes))
    sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


class UserKNN(Recommender):
    """User-based kNN for top-N ranking.

    k           number of neighbours per user
    similarity  "cosine" or "jaccard" on the binary matrix
    min_rating  training signal: 4 uses only ratings >= 4, None every rating
    """

    name = "UserKNN"

    def __init__(self, k: int = 100, similarity: str = "cosine", min_rating: int | None = 4) -> None:
        super().__init__(k=k, similarity=similarity, min_rating=min_rating)
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.similarity = similarity
        self.min_rating = min_rating
        self.history: sp.csr_matrix | None = None
        self.neighbours: sp.csr_matrix | None = None

    def _fit(self, train: TrainData) -> None:
        self.history = train.matrix("binary", self.min_rating)
        sim = user_similarity(self.history, self.similarity)
        # keep_top_k keeps the k largest entries of each column. The matrix is
        # symmetric, so column u holds u's neighbours; transposing moves them to row u.
        self.neighbours = keep_top_k(sim, self.k).T.tocsr()

    def describe(self) -> str:
        signal = "all ratings" if self.min_rating is None else f"ratings >= {self.min_rating}"
        return f"k={self.k}, {self.similarity}, training signal: {signal}"

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.neighbours[users] @ self.history).toarray()
