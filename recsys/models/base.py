"""Common interface for all recommenders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from recsys.data.dataset import TrainData


class Recommender(ABC):
    """A model is fitted once on TrainData and then scores users.

    score() returns one row of item scores per requested user, with one column
    per item index. Higher means more recommended. The evaluator removes seen
    items and items outside the catalogue, so models do not need to.
    """

    name: str = "recommender"
    # True for models with a random component. build_model() then passes the
    # project seed to them unless the config sets one explicitly.
    stochastic: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._train: TrainData | None = None

    def fit(self, train: TrainData) -> Recommender:
        self._train = train
        self._fit(train)
        return self

    @abstractmethod
    def _fit(self, train: TrainData) -> None: ...

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray: ...

    @property
    def train(self) -> TrainData:
        if self._train is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        return self._train

    def describe(self) -> str:
        """Short settings string for the results table."""
        if not self.params:
            return "default"
        return ", ".join(f"{key}={value}" for key, value in self.params.items())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"
