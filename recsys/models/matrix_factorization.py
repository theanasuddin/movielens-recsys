"""Biased matrix factorisation fitted with mini-batch SGD (Lecture 2.1, slides 4 and 7-9).

One of the two models we implemented ourselves for the Part 2 requirement.

A rating is predicted as  r_hat(u, i) = mu + b_u + b_i + p_u . q_i, where mu is
the mean training rating. For every observed training rating the model takes
a gradient step on

    1/2 (r_hat(u, i) - r_ui)^2 + 1/2 reg (|p_u|^2 + |q_i|^2) + 1/2 reg_bias (b_u^2 + b_i^2)

so missing cells are never touched (slide 6). Each epoch visits the ratings
in a random order in mini-batches. Within a batch, the gradients of a user
or movie that appears several times are averaged, and every user and movie
in the batch then takes one step of size lr.

Movies are ranked by predicted rating. This is the objective the slides call
"the wrong thing" for top-N (slide 11): a movie with a few enthusiastic
ratings can get a large b_i and float to the top of every list.
"""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender
from recsys.models.sgd import group_sum


def rating_gradients(err, pu, qi, bu, bi, reg, reg_bias):
    """Gradients of the per-rating loss above with respect to p_u, q_i, b_u, b_i.

    err = r_hat(u, i) - r_ui for each rating in the batch; pu, qi, bu, bi are
    the current parameter rows of the batch.
    """
    return (
        err[:, None] * qi + reg * pu,
        err[:, None] * pu + reg * qi,
        err + reg_bias * bu,
        err + reg_bias * bi,
    )


class BiasedMF(Recommender):
    """Biased matrix factorisation for explicit ratings.

    factors     length of p_u and q_i
    epochs      passes over the training ratings
    lr          step size
    reg         L2 penalty on the factors
    reg_bias    L2 penalty on b_u and b_i (defaults to reg)
    batch_size  ratings per mini-batch
    """

    name = "BiasedMF"
    stochastic = True

    def __init__(
        self,
        factors: int = 32,
        epochs: int = 20,
        lr: float = 0.02,
        reg: float = 0.05,
        reg_bias: float | None = None,
        batch_size: int = 8192,
        seed: int = 42,
    ) -> None:
        reg_bias = reg if reg_bias is None else reg_bias
        super().__init__(factors=factors, epochs=epochs, lr=lr, reg=reg, reg_bias=reg_bias, seed=seed)
        if factors < 1 or epochs < 0 or batch_size < 1 or lr <= 0 or reg < 0 or reg_bias < 0:
            raise ValueError("Invalid BiasedMF hyperparameters")
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.reg_bias = reg_bias
        self.batch_size = batch_size
        self.seed = seed
        self.train_rmse: list[float] = []

    def _fit(self, train: TrainData) -> None:
        self.initialise(train)
        for _ in range(self.epochs):
            self.run_epoch()

    def initialise(self, train: TrainData) -> None:
        """Set up the parameters without training; run_epoch() then trains one pass."""
        self._train = train
        self.rng = np.random.default_rng(self.seed)
        rows = train.ratings
        self.u = train.ids.user_index(rows["user_id"]).astype(np.int64)
        self.i = train.ids.item_index(rows["item_id"]).astype(np.int64)
        self.y = rows["rating"].to_numpy(dtype=np.float32)
        self.mu = float(self.y.mean()) if len(self.y) else 0.0
        # float32 halves the memory traffic of the updates; the precision is plenty here.
        self.p = self.rng.normal(0.0, 0.08, (train.n_users, self.factors)).astype(np.float32)
        self.q = self.rng.normal(0.0, 0.08, (train.n_items, self.factors)).astype(np.float32)
        self.bu = np.zeros(train.n_users, dtype=np.float32)
        self.bi = np.zeros(train.n_items, dtype=np.float32)
        # The training error after each epoch is measured on a fixed sample of
        # ratings; it is only a diagnostic and the full set would cost an epoch's time.
        self.monitor = self.rng.choice(len(self.y), size=min(len(self.y), 50_000), replace=False)
        self.train_rmse = []

    def run_epoch(self) -> None:
        order = self.rng.permutation(len(self.y))
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            u, i = self.u[batch], self.i[batch]
            pu, qi = self.p[u], self.q[i]
            err = self.mu + self.bu[u] + self.bi[i] + np.sum(pu * qi, axis=1) - self.y[batch]
            g_pu, g_qi, g_bu, g_bi = rating_gradients(err, pu, qi, self.bu[u], self.bi[i], self.reg, self.reg_bias)
            for param, index, grad in ((self.p, u, g_pu), (self.q, i, g_qi)):
                unique, total, counts = group_sum(index, grad)
                param[unique] -= self.lr * total / counts[:, None]
            for param, index, grad in ((self.bu, u, g_bu), (self.bi, i, g_bi)):
                total = np.bincount(index, grad, minlength=len(param))
                counts = np.bincount(index, minlength=len(param))
                touched = counts > 0
                param[touched] -= (self.lr * total[touched] / counts[touched]).astype(np.float32)
        sample = self.monitor
        self.train_rmse.append(float(np.sqrt(np.mean((self.predict(self.u[sample], self.i[sample]) - self.y[sample]) ** 2))))

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predicted ratings for (user index, item index) pairs."""
        return self.mu + self.bu[users] + self.bi[items] + np.sum(self.p[users] * self.q[items], axis=1)

    def describe(self) -> str:
        return (f"factors={self.factors}, epochs={self.epochs}, lr={self.lr:.4g}, reg={self.reg:.4g}, "
                f"reg_bias={self.reg_bias:.4g}")

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[users, None] + self.bi[None, :] + self.p[users] @ self.q.T
