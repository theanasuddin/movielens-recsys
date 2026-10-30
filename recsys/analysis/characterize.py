"""Descriptive statistics of MovieLens 1M for report section 1.1.

Everything here is computed from the raw ratings before any split or filter,
except held_out_profile() and popularity_drift(), which describe the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.movielens import MovieLens
from recsys.evaluation.metrics import gini

HOUR = 3_600
DAY = 86_400


def _date(seconds: float) -> str:
    return pd.Timestamp(int(seconds), unit="s", tz="UTC").strftime("%Y-%m-%d")


def dataset_summary(ml: MovieLens) -> pd.DataFrame:
    r = ml.ratings
    per_user = r.groupby("user_id").size()
    per_item = r.groupby("item_id").size()
    n_users, n_items = r["user_id"].nunique(), r["item_id"].nunique()
    rows = [
        ("ratings", len(r)),
        ("users", n_users),
        ("movies in movies.dat", len(ml.movies)),
        ("movies with at least one rating", n_items),
        ("density (ratings / (users x rated movies))", len(r) / (n_users * n_items)),
        ("ratings per user: min", per_user.min()),
        ("ratings per user: median", per_user.median()),
        ("ratings per user: mean", per_user.mean()),
        ("ratings per user: max", per_user.max()),
        ("ratings per movie: min", per_item.min()),
        ("ratings per movie: median", per_item.median()),
        ("ratings per movie: mean", per_item.mean()),
        ("ratings per movie: max", per_item.max()),
        ("movies with exactly 1 rating", int((per_item == 1).sum())),
        ("movies with fewer than 20 ratings", int((per_item < 20).sum())),
        ("Gini of ratings per movie", gini(per_item.to_numpy())),
        ("Gini of ratings per user", gini(per_user.to_numpy())),
        ("mean rating", r["rating"].mean()),
        ("share of ratings >= 4", (r["rating"] >= 4).mean()),
        ("first rating (UTC)", _date(r["timestamp"].min())),
        ("last rating (UTC)", _date(r["timestamp"].max())),
        ("date by which 90% of ratings were entered", _date(r["timestamp"].quantile(0.9))),
    ]
    for star in range(1, 6):
        rows.append((f"share of {star}-star ratings", (r["rating"] == star).mean()))
    return pd.DataFrame(rows, columns=["statistic", "value"])


def long_tail(ratings: pd.DataFrame, shares=(0.01, 0.05, 0.10, 0.20, 0.50)) -> pd.DataFrame:
    """Share of all ratings that go to the most-rated x% of movies."""
    counts = np.sort(ratings.groupby("item_id").size().to_numpy())[::-1]
    cumulative = np.cumsum(counts) / counts.sum()
    rows = []
    for share in shares:
        n = max(1, int(round(share * counts.size)))
        rows.append({"top_share_of_movies": share, "movies": n, "share_of_ratings": cumulative[n - 1]})
    bottom = counts.size // 2
    rows.append({
        "top_share_of_movies": "bottom 50%", "movies": counts.size - bottom,
        "share_of_ratings": counts[bottom:].sum() / counts.sum(),
    })
    return pd.DataFrame(rows)


def monthly_activity(ratings: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(ratings["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m")
    out = ratings.groupby(month).agg(ratings=("rating", "size"), users=("user_id", "nunique"))
    out["share_of_ratings"] = out["ratings"] / len(ratings)
    return out.reset_index(names="month")


def burstiness(ratings: pd.DataFrame) -> pd.DataFrame:
    """How concentrated each user's rating activity is in time."""
    t = ratings["timestamp"]
    first = ratings.groupby("user_id")["timestamp"].transform("min")
    span = ratings.groupby("user_id")["timestamp"].agg(lambda s: s.max() - s.min())
    same_second = ratings.duplicated(["user_id", "timestamp"], keep=False)
    rows = [
        ("users whose ratings all fall within 1 hour", (span < HOUR).mean()),
        ("users whose ratings all fall within 1 day", (span < DAY).mean()),
        ("median span between a user's first and last rating (days)", span.median() / DAY),
        ("ratings entered within 1 day of the user's first rating", ((t - first) < DAY).mean()),
        ("ratings sharing their exact second with another rating of the same user", same_second.mean()),
        ("largest number of ratings one user entered in the same second",
         int(ratings.groupby(["user_id", "timestamp"]).size().max())),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def held_out_profile(history: pd.DataFrame, held_out: pd.DataFrame, relevance_threshold: int = 4) -> dict:
    """Who is in a held-out part, and how far their ratings are from their history."""
    hist_size = history.groupby("user_id").size()
    last = history.groupby("user_id")["timestamp"].max()
    users = held_out["user_id"].unique()
    with_relevant = held_out.loc[held_out["rating"] >= relevance_threshold, "user_id"].unique()
    warm = np.isin(users, hist_size.index)
    warm_rows = held_out[held_out["user_id"].isin(hist_size.index)]
    gap = warm_rows["timestamp"] - warm_rows["user_id"].map(last)
    evaluable = np.intersect1d(with_relevant, hist_size.index)
    return {
        "ratings": len(held_out),
        "users": len(users),
        "users without any earlier rating": int((~warm).sum()),
        "share of users without any earlier rating": float((~warm).mean()) if len(users) else np.nan,
        "evaluable users (>= 1 relevant item and >= 1 earlier rating)": len(evaluable),
        "median earlier ratings of evaluable users": float(hist_size.reindex(evaluable).median()),
        "rows within 1 day of the user's last earlier rating": float((gap < DAY).mean()),
        "rows more than 30 days after the user's last earlier rating": float((gap > 30 * DAY).mean()),
        "rows on movies with no earlier rating": float((~held_out["item_id"].isin(history["item_id"])).mean()),
    }


def popularity_drift(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most-rated movies in the training period next to those of the test period."""
    titles = movies.set_index("item_id")["title"]
    top_train = train["item_id"].value_counts().head(n)
    top_test = test["item_id"].value_counts().head(n)
    return pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "train_title": titles.reindex(top_train.index).to_numpy(),
        "train_ratings": top_train.to_numpy(),
        "test_title": titles.reindex(top_test.index).to_numpy(),
        "test_ratings": top_test.to_numpy(),
    })


def top_overlap(train: pd.DataFrame, test: pd.DataFrame, n: int) -> int:
    top_train = set(train["item_id"].value_counts().head(n).index)
    top_test = set(test["item_id"].value_counts().head(n).index)
    return len(top_train & top_test)
