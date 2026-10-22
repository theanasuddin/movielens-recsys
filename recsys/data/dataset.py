"""Index mapping and the training data object that every model receives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data.movielens import MovieLens


@dataclass(frozen=True)
class IdMap:
    """Maps raw MovieLens ids to row/column indices.

    The maps cover every user in users.dat and every movie in movies.dat, so
    indices stay the same across splits, models and parts of the project.
    """

    users: pd.Index
    items: pd.Index

    @classmethod
    def from_movielens(cls, ml: MovieLens) -> IdMap:
        return cls(pd.Index(np.sort(ml.users["user_id"].unique())), pd.Index(np.sort(ml.movies["item_id"].unique())))

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def user_index(self, user_ids) -> np.ndarray:
        idx = self.users.get_indexer(user_ids)
        if (idx < 0).any():
            raise KeyError("Unknown user ids")
        return idx

    def item_index(self, item_ids) -> np.ndarray:
        idx = self.items.get_indexer(item_ids)
        if (idx < 0).any():
            raise KeyError("Unknown item ids")
        return idx


def to_matrix(ratings: pd.DataFrame, ids: IdMap, values: str = "binary") -> sp.csr_matrix:
    """User x item CSR matrix from a ratings frame.

    values="binary" puts a 1 for every rating, values="rating" keeps the stars.
    """
    rows = ids.user_index(ratings["user_id"])
    cols = ids.item_index(ratings["item_id"])
    if values == "binary":
        data = np.ones(len(ratings), dtype=np.float32)
    elif values == "rating":
        data = ratings["rating"].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"values must be 'binary' or 'rating', got {values!r}")
    return sp.csr_matrix((data, (rows, cols)), shape=(ids.n_users, ids.n_items))


@dataclass(frozen=True)
class TrainData:
    """Everything a model may look at when it is fitted.

    ratings    training rows (user_id, item_id, rating, timestamp)
    cutoff     end of the training period in Unix seconds (exclusive); recency
               based models measure their window back from here
    catalogue  boolean mask over item indices: the candidate set
    seen       binary user x item matrix of training ratings; these items are
               never recommended back to the user
    """

    ratings: pd.DataFrame
    ids: IdMap
    cutoff: int
    catalogue: np.ndarray
    seen: sp.csr_matrix

    @property
    def n_users(self) -> int:
        return self.ids.n_users

    @property
    def n_items(self) -> int:
        return self.ids.n_items

    def matrix(self, values: str = "binary", min_rating: int | None = None) -> sp.csr_matrix:
        """Training matrix, optionally keeping only ratings >= min_rating."""
        rows = self.ratings if min_rating is None else self.ratings[self.ratings["rating"] >= min_rating]
        return to_matrix(rows, self.ids, values)

    def history_sizes(self) -> np.ndarray:
        """Number of training ratings per user index."""
        return np.asarray(self.seen.sum(axis=1)).ravel().astype(np.int64)


def make_train_data(ratings: pd.DataFrame, ids: IdMap, cutoff: int, min_item_ratings: int = 1) -> TrainData:
    """Build TrainData, dropping items with fewer than min_item_ratings training ratings.

    The candidate set is every movie with at least min_item_ratings ratings in
    the training data. Movies nobody rated before the cutoff cannot be scored
    by a collaborative model, so they are not candidates.
    """
    counts = ratings["item_id"].value_counts()
    kept = counts.index[counts >= min_item_ratings]
    ratings = ratings[ratings["item_id"].isin(kept)].reset_index(drop=True)

    catalogue = np.zeros(ids.n_items, dtype=bool)
    catalogue[ids.item_index(kept)] = True
    return TrainData(ratings=ratings, ids=ids, cutoff=cutoff, catalogue=catalogue, seen=to_matrix(ratings, ids))
