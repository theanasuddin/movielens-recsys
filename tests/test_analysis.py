from __future__ import annotations

import pandas as pd
import pytest

from recsys.analysis import burstiness, held_out_profile, long_tail, popularity_drift
from tests.conftest import DAY, T0


def test_long_tail_shares_add_up(tiny_ml):
    table = long_tail(tiny_ml.ratings, shares=(0.2, 0.5))
    # Counts per movie in tiny_ml: 10:4, 20:3, 30:2, 40:2, 50:3 (14 ratings).
    top = table.set_index("top_share_of_movies")
    assert top.loc[0.2, "movies"] == 1 and top.loc[0.2, "share_of_ratings"] == pytest.approx(4 / 14)
    assert top.loc[0.5, "share_of_ratings"] + top.loc["bottom 50%", "share_of_ratings"] == pytest.approx(1.0)


def test_held_out_profile_counts_cold_users(tiny_ml):
    r = tiny_ml.ratings
    history, held_out = r[r["timestamp"] < T0 + 10 * DAY], r[r["timestamp"] >= T0 + 10 * DAY]
    profile = held_out_profile(history, held_out)
    assert profile["users"] == 4
    assert profile["users without any earlier rating"] == 1          # user 4
    assert profile["evaluable users (>= 1 relevant item and >= 1 earlier rating)"] == 2   # users 1 and 2


def test_burstiness_same_second_share(tiny_ml):
    stats = burstiness(tiny_ml.ratings).set_index("statistic")["value"]
    # User 1 rated 10 and 20 at T0, user 2 rated 10 at T0 alone: 2 of 14 rows share a second.
    assert stats["ratings sharing their exact second with another rating of the same user"] == pytest.approx(2 / 14)


def test_popularity_drift_lists_top_titles(tiny_ml):
    r = tiny_ml.ratings
    drift = popularity_drift(r[r["timestamp"] < T0 + 10 * DAY], r[r["timestamp"] >= T0 + 10 * DAY], tiny_ml.movies, n=2)
    assert isinstance(drift, pd.DataFrame) and len(drift) == 2
    assert drift.loc[0, "test_title"] in {"A (1995)", "E (2000)"}
