"""Correctness checks for the Part 2 models.

The two self-implemented gradient models get a finite-difference check of
the exact gradient functions they use, plus a test that they learn planted
structure. EASE is compared with the constrained least-squares problem it
claims to solve, user-kNN with brute-force voting, and the content profile
with the worked example of Lecture 2.2, slide 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data import IdMap, make_train_data
from recsys.models import BPRMF, EASE, BiasedMF, ContentProfile, UserKNN, build_model
from recsys.models.bpr import triple_gradients
from recsys.models.ease import ease_weights
from recsys.models.matrix_factorization import rating_gradients
from recsys.models.sgd import group_sum


def _train_from(table: dict[int, dict[int, int]], n_users: int, n_items: int):
    rows = [(u, i, r, 0) for u, items in table.items() for i, r in items.items()]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    ids = IdMap(pd.Index(range(n_users)), pd.Index(range(n_items)))
    return make_train_data(ratings, ids, cutoff=1)


def _numeric_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        up = f()
        x[idx] = old - eps
        down = f()
        x[idx] = old
        grad[idx] = (up - down) / (2 * eps)
    return grad


def test_group_sum_matches_add_at():
    rng = np.random.default_rng(0)
    index = rng.integers(0, 5, 40)
    values = rng.normal(size=(40, 3))
    unique, summed, counts = group_sum(index, values)
    expected = np.zeros((5, 3))
    np.add.at(expected, index, values)
    np.testing.assert_allclose(summed, expected[unique])
    np.testing.assert_array_equal(counts, np.bincount(index)[unique])


# ---------------------------------------------------------------- BiasedMF

def test_mf_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    pu, qi = rng.normal(size=(1, 4)), rng.normal(size=(1, 4))
    bu, bi = rng.normal(size=1), rng.normal(size=1)
    mu, r, reg, reg_bias = 3.5, np.array([4.0]), 0.3, 0.7

    def loss():
        pred = mu + bu + bi + np.sum(pu * qi, axis=1)
        return float((0.5 * (pred - r) ** 2).sum() + 0.5 * reg * (pu ** 2).sum() + 0.5 * reg * (qi ** 2).sum()
                     + 0.5 * reg_bias * (bu ** 2 + bi ** 2).sum())

    err = mu + bu + bi + np.sum(pu * qi, axis=1) - r
    analytic = rating_gradients(err, pu, qi, bu, bi, reg, reg_bias)
    for got, param in zip(analytic, (pu, qi, bu, bi)):
        np.testing.assert_allclose(got, _numeric_grad(loss, param), atol=1e-6)


def test_mf_recovers_a_planted_low_rank_rating_matrix():
    rng = np.random.default_rng(2)
    n_users, n_items = 60, 40
    true = 3.5 + rng.normal(0, 0.4, (n_users, 1)) + rng.normal(0, 0.4, (1, n_items)) \
        + rng.normal(0, 0.6, (n_users, 2)) @ rng.normal(0, 0.6, (2, n_items))
    observed = rng.random((n_users, n_items)) < 0.7
    table = {u: {i: float(true[u, i]) for i in range(n_items) if observed[u, i]} for u in range(n_users)}
    train = _train_from(table, n_users, n_items)
    model = BiasedMF(factors=2, epochs=300, lr=0.05, reg=0.001, batch_size=256, seed=3).fit(train)

    u, i = np.nonzero(~observed)
    held_out_rmse = np.sqrt(np.mean((model.predict(u, i) - true[u, i]) ** 2))
    mean_rmse = np.sqrt(np.mean((true[observed].mean() - true[u, i]) ** 2))
    assert model.train_rmse[-1] < model.train_rmse[0]
    assert held_out_rmse < 0.3 * mean_rmse


def test_mf_is_reproducible_and_scores_match_predictions(tiny_ml, tiny_ids):
    train = make_train_data(tiny_ml.ratings, tiny_ids, cutoff=10**10)
    a = BiasedMF(factors=3, epochs=4, batch_size=4, seed=123).fit(train)
    b = BiasedMF(factors=3, epochs=4, batch_size=4, seed=123).fit(train)
    np.testing.assert_array_equal(a.score(np.arange(4)), b.score(np.arange(4)))
    users, items = np.meshgrid(np.arange(4), np.arange(6), indexing="ij")
    np.testing.assert_allclose(a.score(np.arange(4)), a.predict(users.ravel(), items.ravel()).reshape(4, 6), rtol=1e-5)


# ---------------------------------------------------------------- BPR-MF

def test_bpr_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    pu, qi, qj = (rng.normal(size=(1, 5)) for _ in range(3))
    bi, bj = rng.normal(size=1), rng.normal(size=1)
    reg = 0.2

    def objective():
        x = np.sum(pu * (qi - qj), axis=1) + bi - bj
        return float(-np.log1p(np.exp(-x)).sum()
                     - 0.5 * reg * ((pu ** 2).sum() + (qi ** 2).sum() + (qj ** 2).sum() + (bi ** 2).sum() + (bj ** 2).sum()))

    *analytic, _ = triple_gradients(pu, qi, qj, bi, bj, reg)
    for got, param in zip(analytic, (pu, qi, qj, bi, bj)):
        np.testing.assert_allclose(got, _numeric_grad(objective, param), atol=1e-6)


def _two_block_data(seed: int = 5):
    """Users 0-9 rate movies 0-9, users 10-19 rate movies 10-19, six each."""
    rng = np.random.default_rng(seed)
    table = {}
    for u in range(20):
        block = np.arange(10) if u < 10 else np.arange(10, 20)
        table[u] = {int(i): 5 for i in rng.choice(block, size=6, replace=False)}
    return _train_from(table, 20, 20)


def test_bpr_learns_planted_preferences():
    train = _two_block_data()
    model = BPRMF(factors=4, epochs=150, lr=0.1, reg=0.01, batch_size=32, seed=6).fit(train)
    scores = model.score(np.arange(20))
    scores[train.seen.toarray() > 0] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :3]
    in_block = np.where(np.arange(20)[:, None] < 10, top < 10, top >= 10)
    assert in_block.mean() >= 0.9
    assert model.log_likelihood[-1] > model.log_likelihood[0]


def test_bpr_negatives_are_never_in_the_training_signal():
    train = _two_block_data()
    model = BPRMF(factors=2, epochs=0, seed=7).fit(train)
    users = np.repeat(np.arange(20), 50)
    j, keep = model._draw_negatives(users)
    signal = train.matrix("binary").toarray() > 0
    assert not signal[users[keep], j[keep]].any()


# ---------------------------------------------------------------- EASE

def test_ease_matches_per_column_ridge_regression():
    rng = np.random.default_rng(8)
    x = (rng.random((30, 8)) < 0.4).astype(float)
    lam = 2.0
    import scipy.sparse as sp
    b = ease_weights(sp.csr_matrix(x), lam)

    assert np.allclose(np.diag(b), 0.0)
    for j in range(8):
        others = [c for c in range(8) if c != j]
        a = x[:, others]
        w = np.linalg.solve(a.T @ a + lam * np.eye(len(others)), a.T @ x[:, j])
        np.testing.assert_allclose(b[others, j], w, atol=1e-10)


def test_ease_scores_are_history_times_weights():
    train = _two_block_data()
    model = EASE(lam=1.0, min_rating=None).fit(train)
    expected = train.matrix("binary").toarray() @ model.weights
    np.testing.assert_allclose(model.score(np.arange(20)), expected)


# ---------------------------------------------------------------- User-kNN

def test_user_knn_matches_brute_force_voting():
    rng = np.random.default_rng(9)
    x = (rng.random((40, 25)) < 0.3).astype(int)
    table = {u: {int(i): 5 for i in np.flatnonzero(x[u])} for u in range(40)}
    train = _train_from(table, 40, 25)
    k = 4
    model = UserKNN(k=k, similarity="cosine", min_rating=None).fit(train)
    got = model.score(np.arange(40))

    norms = np.linalg.norm(x, axis=1)
    sim = np.divide(x @ x.T, np.outer(norms, norms), out=np.zeros((40, 40)), where=np.outer(norms, norms) > 0)
    np.fill_diagonal(sim, 0)
    checked = 0
    for u in range(40):
        ordered = np.sort(sim[u])[::-1]
        if np.isclose(ordered[k - 1], ordered[k]):
            continue  # a tie at the k-th neighbour may be broken either way
        neighbours = np.argsort(-sim[u])[:k]
        neighbours = neighbours[sim[u, neighbours] > 0]
        np.testing.assert_allclose(got[u], sim[u, neighbours] @ x[neighbours], rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked >= 20
    assert (np.count_nonzero(model.neighbours.toarray(), axis=1) <= k).all()


# ---------------------------------------------------------------- Content profile

def _lecture_movies():
    # Lecture 2.2, slide 5: i1..i5 described by Action, Drama, Comedy.
    return pd.DataFrame({
        "item_id": [1, 2, 3, 4, 5],
        "title": [f"i{n} (1995)" for n in range(1, 6)],
        "year": pd.Series([1995] * 5, dtype="Int16"),
        "genres": ["Action|Drama", "Comedy", "Drama|Comedy", "Action|Drama", "Action"],
    })


def test_content_profile_reproduces_lecture_example():
    # Bob rated i2..i5 with 1, 2, 3, 3 (Lecture 1.2, slide 4); his mean is 2.25.
    ratings = pd.DataFrame({"user_id": [7] * 4, "item_id": [2, 3, 4, 5], "rating": [1, 2, 3, 3], "timestamp": 0})
    ids = IdMap(pd.Index([7]), pd.Index([1, 2, 3, 4, 5]))
    train = make_train_data(ratings, ids, cutoff=1)
    model = ContentProfile(_lecture_movies(), weighting="centred", match="dot").fit(train)

    genres = ["Action", "Drama", "Comedy"]
    from recsys.data.movielens import GENRES
    profile = model.profiles[0, [GENRES.index(g) for g in genres]]
    np.testing.assert_allclose(profile, [1.5, 0.5, -1.5])          # slide 6
    assert model.score(np.array([0]))[0, 0] == pytest.approx(2.0)  # i1 = Action + Drama


def test_content_profile_options():
    ratings = pd.DataFrame({"user_id": [7, 7], "item_id": [1, 2], "rating": [5, 2], "timestamp": 0})
    ids = IdMap(pd.Index([7]), pd.Index([1, 2, 3, 4, 5]))
    train = make_train_data(ratings, ids, cutoff=1)
    positive = ContentProfile(_lecture_movies(), weighting="positive", min_rating=4).fit(train)
    with_years = ContentProfile(_lecture_movies(), year_weight=0.5).fit(train)
    # Only i1 (Action, Drama) is positive, so i4 with the same genres matches perfectly.
    assert positive.score(np.array([0]))[0, 3] == pytest.approx(1.0)
    assert with_years.features.shape[1] > positive.features.shape[1]
    with pytest.raises(ValueError):
        ContentProfile(_lecture_movies(), weighting="liked")


def test_registry_builds_the_part2_models():
    assert isinstance(build_model("bpr_mf", default_seed=1), BPRMF)
    assert isinstance(build_model("ease", lam=10.0), EASE)
    assert isinstance(build_model("user_knn", k=5), UserKNN)
    assert build_model("biased_mf", default_seed=9).seed == 9
