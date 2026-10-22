"""Non-personalised baselines (Lecture 1.1, slide 9)."""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender

SECONDS_PER_DAY = 86_400


class MostPopular(Recommender):
    """Rank items by how many training ratings they received.

    window_days  count only ratings entered in the last window_days before the
                 training cutoff (None = all training ratings)
    min_rating   count only ratings >= min_rating (None = every rating counts)

    Every user gets the same list, apart from removing what they have seen.
    """

    name = "MostPopular"

    def __init__(self, window_days: int | None = None, min_rating: int | None = None) -> None:
        super().__init__(window_days=window_days, min_rating=min_rating)
        self.window_days = window_days
        self.min_rating = min_rating
        self.counts: np.ndarray | None = None

    def _fit(self, train: TrainData) -> None:
        rows = train.ratings
        if self.window_days is not None:
            rows = rows[rows["timestamp"] >= train.cutoff - self.window_days * SECONDS_PER_DAY]
        if self.min_rating is not None:
            rows = rows[rows["rating"] >= self.min_rating]
        counts = np.zeros(train.n_items, dtype=np.float64)
        np.add.at(counts, train.ids.item_index(rows["item_id"]), 1.0)
        self.counts = counts

    def describe(self) -> str:
        window = "all-time" if self.window_days is None else f"last {self.window_days} days"
        ratings = "all ratings" if self.min_rating is None else f"ratings >= {self.min_rating}"
        return f"{window}, {ratings}"

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.counts, (len(users), self.train.n_items)).copy()


class RandomRecommender(Recommender):
    """Uniformly random scores. A sanity floor: anything below it is broken."""

    name = "Random"
    stochastic = True

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.seed = seed

    def _fit(self, train: TrainData) -> None:
        pass

    def score(self, users: np.ndarray) -> np.ndarray:
        # One generator per user keeps a user's scores independent of batching.
        out = np.empty((len(users), self.train.n_items))
        for row, user in enumerate(users):
            out[row] = np.random.default_rng([self.seed, int(user)]).random(self.train.n_items)
        return out
