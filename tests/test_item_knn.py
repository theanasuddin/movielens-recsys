from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data import IdMap, make_train_data
from recsys.models import ItemKNN, build_model
from recsys.models.item_knn import item_similarity, keep_top_k


@pytest.fixture
def small():
    # Lecture 1.2 running example (slide 4), minus Bob's missing i1.
    # Users: Ann, Bob, Cleo, Dan, Eva = 1..5. Items i1..i5 = 1..5.
    table = {
        1: {1: 5, 2: 3, 3: 4, 4: 4},
        2: {2: 1, 3: 2, 4: 3, 5: 3},
        3: {1: 4, 2: 3, 3: 4, 4: 3, 5: 5},
        4: {1: 3, 2: 3, 3: 1, 4: 5, 5: 4},
        5: {1: 1, 2: 5, 3: 5, 4: 2, 5: 1},
    }
    rows = [(u, i, r, 0) for u, items in table.items() for i, r in items.items()]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    ids = IdMap(pd.Index([1, 2, 3, 4, 5]), pd.Index([1, 2, 3, 4, 5]))
    return make_train_data(ratings, ids, cutoff=1), table


def _column(table, item, centred=False):
    col = np.zeros(5)
    for u, items in table.items():
        if item in items:
            mean = np.mean(list(items.values())) if centred else 0.0
            col[u - 1] = items[item] - mean if centred else 1.0
    return col


def _cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def test_cosine_and_jaccard_match_hand_computation(small):
    train, table = small
    sim_cos = item_similarity(train, "cosine")
    sim_jac = item_similarity(train, "jaccard")
    a, b = _column(table, 1), _column(table, 5)   # i1: users 1,3,4,5   i5: users 2,3,4,5
    assert sim_cos[0, 4] == pytest.approx(_cos(a, b), abs=1e-6)          # 3 / (2 * 2)
    assert sim_jac[0, 4] == pytest.approx(3 / 5, abs=1e-6)
    assert np.allclose(np.diag(sim_cos), 0) and np.allclose(sim_cos, sim_cos.T)


def test_adjusted_cosine_centres_by_user_mean(small):
    train, table = small
    sim = item_similarity(train, "adjusted_cosine")
    for i, j in [(1, 2), (1, 4), (3, 5)]:
        expected = _cos(_column(table, i, centred=True), _column(table, j, centred=True))
        assert sim[i - 1, j - 1] == pytest.approx(expected, abs=1e-5)


def test_keep_top_k_keeps_k_positive_neighbours_per_column():
    sim = np.array([
        [0.0, 0.9, 0.1, -0.5],
        [0.9, 0.0, 0.4, 0.2],
        [0.1, 0.4, 0.0, 0.3],
        [-0.5, 0.2, 0.3, 0.0],
    ], dtype=np.float32)
    pruned = keep_top_k(sim, 2).toarray()
    assert (np.count_nonzero(pruned, axis=0) <= 2).all()
    # Item 0's two neighbours are items 1 and 2; the negative item 3 never enters.
    assert pruned[:, 0].tolist() == pytest.approx([0.0, 0.9, 0.1, 0.0])
    assert pruned[:, 3].tolist() == pytest.approx([0.0, 0.2, 0.3, 0.0])
    assert keep_top_k(sim, 10).toarray()[3, 0] == 0.0


def test_score_sums_neighbour_similarities_over_the_history(small):
    train, _ = small
    model = ItemKNN(k=2, similarity="cosine").fit(train)
    sim = keep_top_k(item_similarity(train, "cosine"), 2).toarray()
    history = train.matrix("binary").toarray()
    np.testing.assert_allclose(model.score(np.arange(5)), history @ sim, rtol=1e-6)


def test_positive_only_signal_drops_low_ratings_from_history(small):
    train, table = small
    model = ItemKNN(k=4, similarity="cosine", min_rating=4).fit(train)
    bob = model.history[1].toarray().ravel()
    assert bob.sum() == 0   # Bob never gave a 4 or 5


def test_fit_from_similarity_equals_fit(small):
    train, _ = small
    a = ItemKNN(k=3, similarity="jaccard").fit(train)
    b = ItemKNN(k=3, similarity="jaccard").fit_from_similarity(train, item_similarity(train, "jaccard"))
    np.testing.assert_allclose(a.score(np.arange(5)), b.score(np.arange(5)))


def test_registry_and_bad_similarity():
    assert isinstance(build_model("item_knn", k=5), ItemKNN)
    with pytest.raises(ValueError):
        ItemKNN(similarity="euclid")
