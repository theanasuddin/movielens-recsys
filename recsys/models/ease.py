"""EASE, a linear item-item autoencoder (Steck, 2019; Lecture 3.2, slides 9 and 11).

Each movie's column of the binary training matrix X is predicted from all
the other columns by ridge regression. The item x item weight matrix B solves

    min |X - X B|^2 + lam |B|^2    subject to diag(B) = 0

The zero diagonal stops a movie from predicting itself (the identity
solution an autoencoder would otherwise learn). The constrained problem has
a closed form (Steck, 2019, Section 3):

    P = (X^T X + lam I)^-1,    B_ij = -P_ij / P_jj  for i != j,    B_jj = 0

A user's scores are x_u B: the movies they rated vote for every movie with
the learned weights. This is the "well-tuned linear method" of the slides,
fitted with one matrix inverse and no gradient descent.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender


def ease_weights(x: sp.csr_matrix, lam: float) -> np.ndarray:
    """The closed-form EASE weight matrix for a binary user x item matrix."""
    gram = (x.T @ x).toarray().astype(np.float64)
    gram[np.diag_indices_from(gram)] += lam
    p = np.linalg.inv(gram)
    b = p / -np.diag(p)        # divides column j by -P_jj
    np.fill_diagonal(b, 0.0)
    return b


class EASE(Recommender):
    """EASE for top-N ranking.

    lam         ridge penalty; larger values shrink the weights toward zero
    min_rating  training signal: 4 uses only ratings >= 4, None every rating
    """

    name = "EASE"

    def __init__(self, lam: float = 500.0, min_rating: int | None = 4) -> None:
        super().__init__(lam=lam, min_rating=min_rating)
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        self.min_rating = min_rating
        self.history: sp.csr_matrix | None = None
        self.weights: np.ndarray | None = None

    def _fit(self, train: TrainData) -> None:
        self.history = train.matrix("binary", self.min_rating)
        self.weights = ease_weights(self.history, self.lam)

    def describe(self) -> str:
        signal = "all ratings" if self.min_rating is None else f"ratings >= {self.min_rating}"
        return f"lambda={self.lam:g}, training signal: {signal}"

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.history[users] @ self.weights)
