"""BPR matrix factorisation (Rendle et al., 2009; Lecture 2.1, slide 12).

One of the two models we implemented ourselves for the Part 2 requirement.

The score of movie i for user u is  s(u, i) = p_u . q_i + b_i. Training uses
triples (u, i, j): i is a movie in u's training signal and j a movie that is
not. For each triple it takes a gradient ascent step on

    ln sigmoid(x) - reg/2 (|p_u|^2 + |q_i|^2 + |q_j|^2 + b_i^2 + b_j^2),   x = s(u, i) - s(u, j)

so the chosen movie should outscore the other one, and only the difference
counts. With g = sigmoid(-x), the derivative of ln sigmoid(x) is g times the
derivative of x, which gives the updates in triple_gradients().

One epoch draws one triple per training interaction, in mini-batches.
j is drawn uniformly from the candidate movies (Lecture 2.1 calls this the
cheap, popularity-biased default). If it lands on a movie in u's signal it is
drawn once more, and triples that still collide are skipped. The gradients
of a user or movie that appears several times in a batch are added up.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender
from recsys.models.sgd import group_sum


def triple_gradients(pu, qi, qj, bi, bj, reg):
    """Gradients of the per-triple objective above (to be added, not subtracted).

    Returns (d/dp_u, d/dq_i, d/dq_j, d/db_i, d/db_j, x).
    """
    x = np.sum(pu * (qi - qj), axis=1) + bi - bj
    g = expit(-x)
    return (
        g[:, None] * (qi - qj) - reg * pu,
        g[:, None] * pu - reg * qi,
        -g[:, None] * pu - reg * qj,
        g - reg * bi,
        -g - reg * bj,
        x,
    )


class BPRMF(Recommender):
    """Matrix factorisation trained with the BPR pairwise ranking loss.

    factors     length of p_u and q_i
    epochs      passes over the training interactions
    lr          step size
    reg         L2 penalty
    min_rating  training signal: 4 uses only ratings >= 4, None every rating
    """

    name = "BPR-MF"
    stochastic = True

    def __init__(
        self,
        factors: int = 64,
        epochs: int = 30,
        lr: float = 0.05,
        reg: float = 0.001,
        min_rating: int | None = 4,
        batch_size: int = 4096,
        seed: int = 42,
    ) -> None:
        super().__init__(factors=factors, epochs=epochs, lr=lr, reg=reg, min_rating=min_rating, seed=seed)
        if factors < 1 or epochs < 0 or batch_size < 1 or lr <= 0 or reg < 0:
            raise ValueError("Invalid BPR-MF hyperparameters")
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.min_rating = min_rating
        self.batch_size = batch_size
        self.seed = seed
        self.log_likelihood: list[float] = []

    def _fit(self, train: TrainData) -> None:
        self.initialise(train)
        for _ in range(self.epochs):
            self.run_epoch()

    def initialise(self, train: TrainData) -> None:
        """Set up the parameters without training; run_epoch() then trains one pass."""
        self._train = train
        self.rng = np.random.default_rng(self.seed)
        signal = train.matrix("binary", self.min_rating).tocoo()
        self.users = signal.row.astype(np.int64)
        self.items = signal.col.astype(np.int64)
        self.n_items = train.n_items
        # Sorted keys of the (user, movie) pairs in the signal, for fast membership tests.
        self.pair_keys = np.sort(self.users * self.n_items + self.items)
        self.candidates = np.flatnonzero(train.catalogue)
        # float32 halves the memory traffic of the updates; the precision is plenty for ranking.
        self.p = self.rng.normal(0.0, 0.1, (train.n_users, self.factors)).astype(np.float32)
        self.q = self.rng.normal(0.0, 0.1, (train.n_items, self.factors)).astype(np.float32)
        self.b = np.zeros(train.n_items, dtype=np.float32)
        self.log_likelihood = []

    def _in_signal(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        if len(self.pair_keys) == 0:
            return np.zeros(len(u), dtype=bool)
        keys = u * self.n_items + j
        pos = np.minimum(np.searchsorted(self.pair_keys, keys), len(self.pair_keys) - 1)
        return self.pair_keys[pos] == keys

    def _draw_negatives(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        j = self.candidates[self.rng.integers(0, len(self.candidates), len(u))]
        clash = self._in_signal(u, j)
        j[clash] = self.candidates[self.rng.integers(0, len(self.candidates), int(clash.sum()))]
        return j, ~self._in_signal(u, j)

    def run_epoch(self) -> None:
        order = self.rng.permutation(len(self.users))
        total, count = 0.0, 0
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            u, i = self.users[batch], self.items[batch]
            j, keep = self._draw_negatives(u)
            u, i, j = u[keep], i[keep], j[keep]
            g_pu, g_qi, g_qj, g_bi, g_bj, x = triple_gradients(
                self.p[u], self.q[i], self.q[j], self.b[i], self.b[j], self.reg)
            # Movie i and movie j rows are updated together: one movie can be both in a batch.
            movies = np.concatenate([i, j])
            for param, index, grad in ((self.p, u, g_pu), (self.q, movies, np.concatenate([g_qi, g_qj]))):
                unique, summed, _ = group_sum(index, grad)
                param[unique] += self.lr * summed
            self.b += self.lr * np.bincount(movies, np.concatenate([g_bi, g_bj]),
                                            minlength=self.n_items).astype(np.float32)
            total += float(log_expit(x).sum())
            count += len(x)
        self.log_likelihood.append(total / max(count, 1))

    def describe(self) -> str:
        signal = "all ratings" if self.min_rating is None else f"ratings >= {self.min_rating}"
        return (f"factors={self.factors}, epochs={self.epochs}, lr={self.lr:.4g}, reg={self.reg:.4g}, "
                f"training signal: {signal}")

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.p[users] @ self.q.T + self.b
