"""A tiny MovieLens-shaped dataset for fast tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data import IdMap, MovieLens

DAY = 86_400
T0 = int(pd.Timestamp("2000-06-01", tz="UTC").timestamp())


@pytest.fixture
def tiny_ml() -> MovieLens:
    # 4 users, 6 movies. Users 1-3 rate early; user 4 only appears late.
    ratings = pd.DataFrame(
        [
            (1, 10, 5, T0), (1, 20, 4, T0), (1, 30, 2, T0 + 1 * DAY),
            (2, 10, 4, T0), (2, 30, 5, T0 + 2 * DAY), (2, 40, 3, T0 + 3 * DAY),
            (3, 20, 5, T0 + 1 * DAY), (3, 50, 4, T0 + 2 * DAY),
            # later ratings
            (1, 40, 5, T0 + 10 * DAY), (2, 20, 4, T0 + 10 * DAY), (3, 10, 1, T0 + 11 * DAY),
            (1, 50, 4, T0 + 20 * DAY), (2, 50, 5, T0 + 21 * DAY), (4, 10, 5, T0 + 21 * DAY),
        ],
        columns=["user_id", "item_id", "rating", "timestamp"],
    ).astype({"user_id": "int32", "item_id": "int32", "rating": "int8", "timestamp": "int64"})
    movies = pd.DataFrame({
        "item_id": pd.Series([10, 20, 30, 40, 50, 60], dtype="int32"),
        "title": ["A (1995)", "B (1996)", "C (1990)", "D (1999)", "E (2000)", "F (1980)"],
        "year": pd.Series([1995, 1996, 1990, 1999, 2000, 1980], dtype="Int16"),
        "genres": ["Action", "Comedy", "Drama|Action", "Comedy", "Drama", "Horror"],
    })
    users = pd.DataFrame({
        "user_id": pd.Series([1, 2, 3, 4], dtype="int32"),
        "gender": ["F", "M", "M", "F"],
        "age": pd.Series([25, 35, 18, 25], dtype="int8"),
        "occupation": pd.Series([1, 2, 3, 4], dtype="int8"),
        "zip": ["00100", "33100", "90210", "12345"],
    })
    return MovieLens(ratings, movies, users)


@pytest.fixture
def tiny_ids(tiny_ml: MovieLens) -> IdMap:
    return IdMap.from_movielens(tiny_ml)


@pytest.fixture
def synthetic_ml() -> MovieLens:
    """60 users and 40 movies with ratings spread over the train, validation and test periods."""
    rng = np.random.default_rng(3)
    start, end = pd.Timestamp("2000-06-01", tz="UTC").timestamp(), pd.Timestamp("2002-06-01", tz="UTC").timestamp()
    taste = rng.normal(size=(60, 3)) @ rng.normal(size=(3, 40))
    rows = []
    for user in range(1, 61):
        items = rng.choice(40, size=20, replace=False)
        times = np.sort(rng.uniform(start, end, size=20)).astype(np.int64)
        for item, t in zip(items, times):
            stars = int(np.clip(np.round(3.5 + taste[user - 1, item]), 1, 5))
            rows.append((user, item + 1, stars, t))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"]).astype(
        {"user_id": "int32", "item_id": "int32", "rating": "int8", "timestamp": "int64"})
    genres = ["Action", "Comedy", "Drama|Romance", "Action|Thriller", "Children's|Animation"]
    movies = pd.DataFrame({
        "item_id": pd.Series(range(1, 41), dtype="int32"),
        "title": [f"Movie {i} ({1960 + i})" for i in range(1, 41)],
        "year": pd.Series([1960 + i for i in range(1, 41)], dtype="Int16"),
        "genres": [genres[i % len(genres)] for i in range(40)],
    })
    users = pd.DataFrame({"user_id": pd.Series(range(1, 61), dtype="int32"), "gender": "F",
                          "age": pd.Series([25] * 60, dtype="int8"), "occupation": pd.Series([1] * 60, dtype="int8"),
                          "zip": "00000"})
    return MovieLens(ratings, movies, users)
