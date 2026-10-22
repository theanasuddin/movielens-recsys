"""Global temporal split (Lecture 3.1, slide 4).

One pair of cut points applies to every user: ratings entered before
val_start train, ratings in [val_start, test_start) validate, and ratings from
test_start onwards test. Comparisons are strict, so ratings that share a
timestamp always end up in the same part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.config import SplitConfig


def _to_seconds(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def split_boundaries(ratings: pd.DataFrame, cfg: SplitConfig) -> tuple[int, int]:
    """Return (val_start, test_start) as Unix seconds."""
    if cfg.mode == "calendar":
        val_start, test_start = _to_seconds(cfg.val_start), _to_seconds(cfg.test_start)
    elif cfg.mode == "quantile":
        q_val, q_test = cfg.quantiles
        ts = ratings["timestamp"].to_numpy()
        val_start = int(np.quantile(ts, q_val, method="higher"))
        test_start = int(np.quantile(ts, q_test, method="higher"))
    else:
        raise ValueError(f"Unknown split mode {cfg.mode!r}")
    if not val_start < test_start:
        raise ValueError("The validation cut must come before the test cut")
    return val_start, test_start


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: int
    test_start: int

    @property
    def train_val(self) -> pd.DataFrame:
        """Training data for the final test run: train and validation together."""
        return pd.concat([self.train, self.val], ignore_index=True)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("validation", self.val), ("test", self.test)):
            ts = pd.to_datetime(part["timestamp"], unit="s", utc=True)
            rows.append({
                "part": name,
                "ratings": len(part),
                "share_of_ratings": len(part) / self.n_ratings,
                "users": part["user_id"].nunique(),
                "items": part["item_id"].nunique(),
                "first_rating": ts.min(),
                "last_rating": ts.max(),
            })
        return pd.DataFrame(rows)

    @property
    def n_ratings(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def temporal_split(ratings: pd.DataFrame, cfg: SplitConfig) -> TemporalSplit:
    val_start, test_start = split_boundaries(ratings, cfg)
    t = ratings["timestamp"]
    return TemporalSplit(
        train=ratings[t < val_start].reset_index(drop=True),
        val=ratings[(t >= val_start) & (t < test_start)].reset_index(drop=True),
        test=ratings[t >= test_start].reset_index(drop=True),
        val_start=val_start,
        test_start=test_start,
    )
