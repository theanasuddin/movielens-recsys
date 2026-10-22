from __future__ import annotations

import numpy as np
import pytest

from recsys.data import make_train_data
from recsys.models import MostPopular, RandomRecommender, build_model
from tests.conftest import DAY, T0


def _train(tiny_ml, tiny_ids):
    rows = tiny_ml.ratings[tiny_ml.ratings["timestamp"] < T0 + 10 * DAY]
    return make_train_data(rows, tiny_ids, cutoff=T0 + 10 * DAY)


def _counts(model, tiny_ids, items):
    return model.counts[tiny_ids.item_index(items)].tolist()


def test_most_popular_counts_all_ratings(tiny_ml, tiny_ids):
    model = MostPopular().fit(_train(tiny_ml, tiny_ids))
    assert _counts(model, tiny_ids, [10, 20, 30, 40, 50, 60]) == [2, 2, 2, 1, 1, 0]


def test_most_popular_positive_only(tiny_ml, tiny_ids):
    model = MostPopular(min_rating=4).fit(_train(tiny_ml, tiny_ids))
    # Movie 30 got a 2 and a 5, movie 40 only a 3.
    assert _counts(model, tiny_ids, [10, 20, 30, 40, 50]) == [2, 2, 1, 0, 1]


def test_most_popular_recent_window(tiny_ml, tiny_ids):
    # Cutoff is day 10; a 8.5-day window keeps ratings from day 1.5 onwards.
    model = MostPopular(window_days=8.5).fit(_train(tiny_ml, tiny_ids))
    assert _counts(model, tiny_ids, [10, 20, 30, 40, 50]) == [0, 0, 1, 1, 1]


def test_most_popular_scores_are_the_same_for_every_user(tiny_ml, tiny_ids):
    model = MostPopular().fit(_train(tiny_ml, tiny_ids))
    scores = model.score(np.array([0, 1, 2]))
    assert scores.shape == (3, tiny_ids.n_items)
    assert (scores == scores[0]).all()


def test_random_is_reproducible_and_batch_independent(tiny_ml, tiny_ids):
    train = _train(tiny_ml, tiny_ids)
    a = RandomRecommender(seed=7).fit(train).score(np.array([0, 1, 2]))
    b = RandomRecommender(seed=7).fit(train).score(np.array([2]))
    c = RandomRecommender(seed=8).fit(train).score(np.array([0]))
    np.testing.assert_array_equal(a[2], b[0])
    assert not np.array_equal(a[0], c[0])


def test_build_model_passes_the_seed_to_stochastic_models():
    assert build_model("random", default_seed=11).seed == 11
    assert build_model("random", default_seed=11, seed=5).seed == 5
    assert isinstance(build_model("most_popular", default_seed=11), MostPopular)
    with pytest.raises(ValueError):
        build_model("random")
    with pytest.raises(KeyError):
        build_model("no_such_model")
