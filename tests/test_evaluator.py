from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.config import EvaluationConfig
from recsys.data import TrainData, make_train_data
from recsys.evaluation import evaluate, top_k
from recsys.models.base import Recommender
from tests.conftest import DAY, T0


class FixedScores(Recommender):
    """Returns the same score vector for every user."""

    name = "Fixed"

    def __init__(self, scores: np.ndarray) -> None:
        super().__init__()
        self.scores = np.asarray(scores, dtype=float)

    def _fit(self, train: TrainData) -> None:
        pass

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(users), 1))


def _setup(tiny_ml, tiny_ids, min_item_ratings=1):
    r = tiny_ml.ratings
    train_rows = r[r["timestamp"] < T0 + 10 * DAY]
    held_out = r[r["timestamp"] >= T0 + 10 * DAY]
    train = make_train_data(train_rows, tiny_ids, cutoff=T0 + 10 * DAY, min_item_ratings=min_item_ratings)
    return train, held_out


def test_seen_items_and_non_catalogue_items_are_never_recommended(tiny_ml, tiny_ids):
    train, held_out = _setup(tiny_ml, tiny_ids)
    # Movie 60 has no training ratings, so it is outside the catalogue even
    # though the model gives it the highest score.
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 99.0])
    result = evaluate(FixedScores(scores).fit(train), train, held_out, EvaluationConfig(k=2))

    item_60 = tiny_ids.item_index([60])[0]
    assert item_60 not in result.recommendations
    for row, user in enumerate(result.per_user["user_id"]):
        seen = set(train.ratings.loc[train.ratings["user_id"] == user, "item_id"])
        recommended = set(tiny_ids.items[result.recommendations[row]])
        assert not recommended & seen


def test_metrics_for_a_known_case(tiny_ml, tiny_ids):
    train, held_out = _setup(tiny_ml, tiny_ids)
    # Order 40 > 50 > 20 > 30 > 10. User 1 has seen 10, 20, 30 and gets [40, 50];
    # both are relevant for user 1 (ratings 5 and 4).
    scores = np.zeros(6)
    scores[tiny_ids.item_index([40, 50, 20, 30, 10])] = [5, 4, 3, 2, 1]
    result = evaluate(FixedScores(scores).fit(train), train, held_out, EvaluationConfig(k=2))

    user1 = result.per_user.set_index("user_id").loc[1]
    assert user1["precision"] == pytest.approx(1.0)
    assert user1["recall"] == pytest.approx(1.0)
    assert user1["ndcg"] == pytest.approx(1.0)
    # User 3's later rating is a 1, so user 3 has nothing relevant and is not evaluated.
    assert 3 not in set(result.per_user["user_id"])


def test_users_without_history_are_skipped(tiny_ml, tiny_ids):
    train, held_out = _setup(tiny_ml, tiny_ids)
    result = evaluate(FixedScores(np.arange(6.0)).fit(train), train, held_out, EvaluationConfig(k=2), min_history=1)
    assert 4 not in set(result.per_user["user_id"])  # user 4 only rates after the cutoff


def test_item_filter_shrinks_the_catalogue(tiny_ml, tiny_ids):
    train, _ = _setup(tiny_ml, tiny_ids, min_item_ratings=2)
    kept = set(tiny_ids.items[train.catalogue])
    counts = train.ratings["item_id"].value_counts()
    assert kept == set(counts.index) and (counts >= 2).all()


def test_top_k_breaks_ties_with_the_given_order():
    scores = np.array([[1.0, 1.0, 1.0, 0.0]])
    assert top_k(scores, 2, np.array([2, 0, 1, 3])).tolist() == [[2, 0]]
    assert top_k(scores, 2, np.array([1, 2, 0, 3])).tolist() == [[1, 2]]


def test_top_k_rejects_nan():
    with pytest.raises(ValueError):
        top_k(np.array([[np.nan, 1.0]]), 1, np.array([0, 1]))


def test_batching_does_not_change_results(tiny_ml, tiny_ids):
    train, held_out = _setup(tiny_ml, tiny_ids)
    model = FixedScores(np.array([1.0, 1.0, 2.0, 2.0, 3.0, 0.0])).fit(train)
    a = evaluate(model, train, held_out, EvaluationConfig(k=2, batch_size=1), seed=3)
    b = evaluate(model, train, held_out, EvaluationConfig(k=2, batch_size=100), seed=3)
    pd.testing.assert_frame_equal(a.per_user, b.per_user)
    np.testing.assert_array_equal(a.recommendations, b.recommendations)
