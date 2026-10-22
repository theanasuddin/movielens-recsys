from __future__ import annotations

import pandas as pd
import pytest

from recsys.config import SplitConfig
from recsys.data import split_boundaries, temporal_split
from tests.conftest import DAY, T0


def _date(seconds: int) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


def test_calendar_split_is_strictly_time_ordered(tiny_ml):
    cfg = SplitConfig(mode="calendar", val_start=_date(T0 + 10 * DAY), test_start=_date(T0 + 20 * DAY))
    split = temporal_split(tiny_ml.ratings, cfg)

    assert len(split.train) + len(split.val) + len(split.test) == len(tiny_ml.ratings)
    assert split.train["timestamp"].max() < split.val_start <= split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test_start <= split.test["timestamp"].min()
    # Ratings exactly at a cut point belong to the later part.
    assert (split.val["timestamp"] == T0 + 10 * DAY).sum() == 2
    assert (split.test["timestamp"] == T0 + 20 * DAY).sum() == 1


def test_train_val_concatenates_in_time_order(tiny_ml):
    cfg = SplitConfig(mode="calendar", val_start=_date(T0 + 10 * DAY), test_start=_date(T0 + 20 * DAY))
    split = temporal_split(tiny_ml.ratings, cfg)
    assert len(split.train_val) == len(split.train) + len(split.val)
    assert split.train_val["timestamp"].max() < split.test_start


def test_quantile_split_uses_rating_quantiles(tiny_ml):
    cfg = SplitConfig(mode="quantile", quantiles=(0.5, 0.75))
    val_start, test_start = split_boundaries(tiny_ml.ratings, cfg)
    split = temporal_split(tiny_ml.ratings, cfg)
    assert val_start < test_start
    assert len(split.train) <= 0.5 * len(tiny_ml.ratings) < len(split.train) + len(split.val)


def test_rejects_cut_points_in_wrong_order(tiny_ml):
    cfg = SplitConfig(mode="calendar", val_start="2001-07-01", test_start="2001-01-01")
    with pytest.raises(ValueError):
        temporal_split(tiny_ml.ratings, cfg)
