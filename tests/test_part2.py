"""Tests for the Part 2 tuning helpers, statistics and the runner as a whole."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.config import EvaluationConfig, load_config
from recsys.data import IdMap, MovieLens, make_train_data
from recsys.evaluation import evaluate, holm, metrics_against, relevance_matrix
from recsys.experiments import RunContext, part2
from recsys.models import BPRMF, MostPopular
from recsys.tuning import expand_grid, run_trial, sample_space
from recsys.utils import REPO_ROOT, OutputPaths
from tests.conftest import DAY, T0


def test_expand_grid_crosses_values_and_joins_blocks():
    grid = expand_grid([{"a": [1, 2], "b": ["x", "y"]}, {"a": [3], "b": ["z"]}])
    assert grid == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


def test_sample_space_is_seeded_and_respects_bounds():
    space = {"lr": {"loguniform": [0.01, 0.1]}, "factors": {"choice": [16, 32]}}
    a, b = sample_space(space, 20, seed=1), sample_space(space, 20, seed=1)
    assert a == b
    assert all(0.01 <= s["lr"] <= 0.1 and s["factors"] in (16, 32) for s in a)


def test_holm_adjustment():
    adjusted = holm(np.array([0.01, 0.04, 0.03, 0.2]))
    # Sorted: 0.01*4 = 0.04, 0.03*3 = 0.09, 0.04*2 = 0.08 -> 0.09 (kept non-decreasing), 0.2*1 = 0.2.
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.2])


def test_metrics_against_scores_the_same_lists_on_a_subset(tiny_ml, tiny_ids):
    r = tiny_ml.ratings
    train = make_train_data(r[r["timestamp"] < T0 + 10 * DAY], tiny_ids, cutoff=T0 + 10 * DAY)
    held_out = r[r["timestamp"] >= T0 + 10 * DAY]
    result = evaluate(MostPopular().fit(train), train, held_out, EvaluationConfig(k=2))
    everything = relevance_matrix(held_out, tiny_ids, 4)
    same, keep = metrics_against(result, everything, tiny_ids)
    np.testing.assert_allclose(same["ndcg"], result.per_user["ndcg"])
    assert keep.all()
    only_late = relevance_matrix(held_out[held_out["timestamp"] >= T0 + 20 * DAY], tiny_ids, 4)
    late, keep = metrics_against(result, only_late, tiny_ids)
    assert set(late["user_id"]) == {1, 2} and keep.sum() == 2


def test_run_trial_picks_the_best_checked_epoch():
    rng = np.random.default_rng(0)
    table = {u: {int(i): 5 for i in rng.choice(20, size=6, replace=False)} for u in range(30)}
    rows = [(u, i, r, 0) for u, items in table.items() for i, r in items.items()]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    train = make_train_data(ratings, IdMap(pd.Index(range(30)), pd.Index(range(20))), cutoff=1)
    calls = []

    def score(model):
        calls.append(len(model.log_likelihood))
        result = evaluate(model, train, ratings.iloc[:0], EvaluationConfig(k=2))
        result.per_user = pd.DataFrame({"ndcg": [1.0 / len(model.log_likelihood)]})  # best at the first check
        return result

    epochs, result, curve = run_trial(BPRMF(factors=2, seed=1), train, score, max_epochs=50, eval_every=5, patience=2)
    assert epochs == 5 and calls == [5, 10, 15]
    assert [e for e, _ in curve] == [5, 10, 15]


# ---------------------------------------------------------------- runner

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


def test_part2_runner_end_to_end(synthetic_ml, tmp_path):
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    ctx = RunContext(config=config, data=synthetic_ml, out=OutputPaths(tmp_path).create(), quick=True)
    part2.run(ctx)

    sheets = pd.Series([sheet for sheet, _ in ctx.results._rows]).value_counts()
    assert sheets["Part 2 - Models"] == len(part2.MODEL_KEYS)
    assert sheets["Part 2 - Sparsity"] == 3 * len(part2.MODEL_KEYS)
    assert sheets["Part 2 - Disaggregated"] == 7 * len(part2.MODEL_KEYS)
    models = ctx.results.rows("Part 2 - Models")
    assert {row["Self-implemented?"] for row in models if row["Model"] in ("BPR-MF", "BiasedMF")} == {"Yes"}
    for name in ("part2_validation_trials.csv", "part2_paired_tests.csv", "part2_breakdowns.csv",
                 "part2_sparsity_summary.csv", "part2_biasedmf_rating_error.csv", "part2_item_groups.csv",
                 "part2_recommendation_profile.csv", "part2_popularity_windows.csv"):
        assert (tmp_path / "tables" / name).is_file()
    windows = pd.read_csv(tmp_path / "tables" / "part2_popularity_windows.csv")
    trials = pd.read_csv(tmp_path / "tables" / "part2_validation_trials.csv")
    popularity = trials[(trials["model"] == "most_popular") & (trials["stage"] == "full data")]
    # The check recomputes the same validation scores the tuning saw.
    np.testing.assert_allclose(windows["validation_ndcg@10"], popularity["ndcg@10"])
    assert windows["chosen_on_validation"].sum() == 1
    for name in ("fig3_part2_models.png", "fig4_part2_breakdowns.png", "fig5_part2_sparsity.png"):
        assert (tmp_path / "figures" / name).is_file()
