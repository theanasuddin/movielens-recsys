"""Non-personalised baselines (Lecture 1.1, slide 9)."""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import TrainData
from recsys.models.base import Recommender

SECONDS_PER_DAY = 86_400


class MostPopular(Recommender):
    """Rank items by how many training ratings they received.

    window_days     count only ratings entered in the last window_days before the
                    training cutoff (None = all training ratings)
    window_ratings  count only the window_ratings most recent training ratings,
                    a window that holds the same amount of data at any cutoff
    min_rating      count only ratings >= min_rating (None = every rating counts)

    Every user gets the same list, apart from removing what they have seen.
    """

    name = "MostPopular"

    def __init__(self, window_days: int | None = None, min_rating: int | None = None,
                 window_ratings: int | None = None) -> None:
        if window_days is not None and window_ratings is not None:
            raise ValueError("Give window_days or window_ratings, not both")
        params = {"window_days": window_days, "min_rating": min_rating}
        if window_ratings is not None:
            params["window_ratings"] = window_ratings
        super().__init__(**params)
        self.window_days = window_days
        self.window_ratings = window_ratings
        self.min_rating = min_rating
        self.counts: np.ndarray | None = None

    def _fit(self, train: TrainData) -> None:
        rows = train.ratings
        if self.window_days is not None:
            rows = rows[rows["timestamp"] >= train.cutoff - self.window_days * SECONDS_PER_DAY]
        if self.window_ratings is not None:
            # A stable sort keeps the file order among equal timestamps.
            rows = rows.sort_values("timestamp", kind="stable").tail(self.window_ratings)
        # How much data the window holds, before the rating filter.
        self.window_size = len(rows)
        self.window_start = int(rows["timestamp"].min()) if len(rows) else train.cutoff
        if self.min_rating is not None:
            rows = rows[rows["rating"] >= self.min_rating]
        counts = np.zeros(train.n_items, dtype=np.float64)
        np.add.at(counts, train.ids.item_index(rows["item_id"]), 1.0)
        self.counts = counts

    def describe(self) -> str:
        if self.window_days is not None:
            window = f"last {self.window_days} days"
        elif self.window_ratings is not None:
            window = f"last {self.window_ratings:,} ratings"
        else:
            window = "all-time"
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
