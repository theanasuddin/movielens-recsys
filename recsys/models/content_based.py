"""Content-based profile matching over genres and release decade (Lecture 2.2, slides 4-7).

Each movie is a vector of its 18 genre indicators, optionally followed by a
one-hot release decade scaled by year_weight. A user's profile adds up the
vectors of the movies they rated, with one of two weightings:

    positive   1 for every rating >= min_rating, 0 otherwise
    centred    the rating minus the user's mean rating (slide 6), so movies
               the user disliked push the profile away from their features

A movie's score is the match between the profile and the movie's vector:
the dot product (as in Lecture 6.1, slide 7) or the cosine. Only public
catalogue metadata is used, no user attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.dataset import TrainData
from recsys.data.movielens import genre_matrix
from recsys.models.base import Recommender

WEIGHTINGS = ("positive", "centred")
MATCHES = ("cosine", "dot")
FIRST_DECADE, LAST_DECADE = 1910, 2000


def item_features(movies: pd.DataFrame, items: pd.Index, year_weight: float) -> np.ndarray:
    """Genre indicators, plus a scaled one-hot release decade when year_weight > 0."""
    genres = genre_matrix(movies).reindex(items, fill_value=0).to_numpy(dtype=np.float64)
    if year_weight == 0:
        return genres
    years = movies.set_index("item_id")["year"].reindex(items).astype("float64").fillna(1990.0)
    decade = (np.clip(years.to_numpy(), FIRST_DECADE, LAST_DECADE) - FIRST_DECADE) // 10
    one_hot = np.eye((LAST_DECADE - FIRST_DECADE) // 10 + 1)[decade.astype(int)]
    return np.hstack([genres, year_weight * one_hot])


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


class ContentProfile(Recommender):
    """Content-based recommender over movie metadata.

    movies       the movies table (genres and year)
    weighting    "positive" or "centred", see above
    min_rating   threshold for the positive weighting
    year_weight  weight of the release-decade features (0 = genres only)
    match        "cosine" or "dot"
    """

    name = "ContentProfile"

    def __init__(
        self,
        movies: pd.DataFrame,
        weighting: str = "positive",
        min_rating: int = 4,
        year_weight: float = 0.0,
        match: str = "cosine",
    ) -> None:
        super().__init__(weighting=weighting, min_rating=min_rating, year_weight=year_weight, match=match)
        if weighting not in WEIGHTINGS or match not in MATCHES or year_weight < 0:
            raise ValueError("Invalid ContentProfile settings")
        self.movies = movies
        self.weighting = weighting
        self.min_rating = min_rating
        self.year_weight = year_weight
        self.match = match

    def _fit(self, train: TrainData) -> None:
        features = item_features(self.movies, train.ids.items, self.year_weight)
        if self.weighting == "positive":
            weights = train.matrix("binary", self.min_rating)
        else:
            weights = train.matrix("rating").tocsr()
            counts = np.diff(weights.indptr)
            means = np.divide(np.asarray(weights.sum(axis=1)).ravel(), counts,
                              out=np.zeros(weights.shape[0]), where=counts > 0)
            weights.data -= np.repeat(means, counts).astype(weights.data.dtype)
        self.profiles = np.asarray(weights @ features)
        self.features = features
        if self.match == "cosine":
            self.profiles = _unit_rows(self.profiles)
            self.features = _unit_rows(features)

    def describe(self) -> str:
        weighting = f"positive (ratings >= {self.min_rating})" if self.weighting == "positive" else "centred"
        return f"{weighting} profile, year_weight={self.year_weight:g}, {self.match} match"

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.profiles[users] @ self.features.T
