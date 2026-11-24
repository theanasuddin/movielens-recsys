"""Tests for Part 3: MMR, the measures beyond accuracy and the runner.

The worked examples come from Lecture 5.1: five items described by the genres
Action, Drama and Comedy (slides 8 and 9) and the 70/30 user of slide 11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from recsys.config import load_config
from recsys.evaluation import bootstrap_ratio, bootstrap_system_diff, metrics
from recsys.evaluation.diversity import (
    genre_calibration_error,
    genre_shares,
    history_distribution,
    intra_list_distance,
    intra_list_diversity,
    item_cosine,
    list_distribution,
    self_information,
    top_genre_gap,
)
from recsys.experiments import RunContext, part3
from recsys.reranking.mmr import mmr_rerank, mmr_rerank_batch
from recsys.utils import REPO_ROOT, OutputPaths

# Slide 8: i1 (A, D) 0.95, i4 (A, D) 0.90, i3 (D, C) 0.70, i5 (A) 0.60, i2 (C) 0.40.
LECTURE_ITEMS = np.array([1, 4, 3, 5, 2])
LECTURE_RELEVANCE = np.array([0.95, 0.90, 0.70, 0.60, 0.40])
LECTURE_GENRES = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]])


def _genres_of(items):
    return LECTURE_GENRES[[list(LECTURE_ITEMS).index(i) for i in items]]


# ---------------------------------------------------------------- MMR

def test_slide_8_top_three_by_relevance_has_ild_one_third():
    ild = intra_list_diversity(np.array([[0, 1, 2]]), _genres_of([1, 4, 3]))
    assert ild[0] == pytest.approx(1 / 3)


def test_slide_9_mmr_picks_i1_i2_i3():
    picked = mmr_rerank(LECTURE_ITEMS, LECTURE_RELEVANCE, LECTURE_GENRES, k=3, lambda_=0.5, normalise=False)
    np.testing.assert_array_equal(picked, [1, 2, 3])
    # The exchange rate of slide 9: ILD 0.33 -> 0.60, relevance 2.55 -> 2.05.
    assert intra_list_diversity(np.array([[0, 1, 2]]), _genres_of(picked))[0] == pytest.approx(0.598, abs=1e-3)
    relevance = dict(zip(LECTURE_ITEMS, LECTURE_RELEVANCE))
    assert sum(relevance[i] for i in picked) == pytest.approx(2.05)


def test_min_max_normalisation_changes_the_exchange_rate():
    # Stretching relevance to [0, 1] makes the near-duplicate i4 worth its
    # similarity penalty, so the same lambda picks a different list.
    picked = mmr_rerank(LECTURE_ITEMS, LECTURE_RELEVANCE, LECTURE_GENRES, k=3, lambda_=0.5, normalise=True)
    np.testing.assert_array_equal(picked, [1, 3, 4])


def test_lambda_one_keeps_the_input_order_ties_included():
    picked = mmr_rerank(np.array([10, 20, 30, 40]), np.array([0.9, 0.9, 0.5, 0.5]),
                        np.array([[1, 0], [1, 0], [1, 0], [0, 1]]), k=4, lambda_=1.0)
    np.testing.assert_array_equal(picked, [10, 20, 30, 40])


def test_mmr_can_choose_a_different_genre():
    picked = mmr_rerank(np.array([10, 20, 30]), np.array([0.9, 0.8, 0.7]),
                        np.array([[1, 0], [1, 0], [0, 1]]), k=2, lambda_=0.5)
    np.testing.assert_array_equal(picked, [10, 30])


def test_batch_matches_one_list_at_a_time():
    rng = np.random.default_rng(0)
    scores = -np.sort(-rng.random((6, 15)), axis=1)
    vectors = (rng.random((6, 15, 5)) < 0.4).astype(float)
    vectors[..., 0] = np.maximum(vectors[..., 0], vectors.sum(axis=2) == 0)
    for lambda_ in (0.0, 0.3, 0.7, 1.0):
        batch = mmr_rerank_batch(scores, vectors, k=5, lambda_=lambda_)
        for row in range(6):
            single = mmr_rerank(np.arange(15), scores[row], vectors[row], k=5, lambda_=lambda_)
            np.testing.assert_array_equal(batch[row], single)


def test_empty_slots_are_never_picked():
    scores = np.array([[0.9, 0.5, 0.4, -np.inf, -np.inf]])
    vectors = np.ones((1, 5, 2))
    picked = mmr_rerank_batch(scores, vectors, k=3, lambda_=0.2)
    assert set(picked[0]) == {0, 1, 2}
    with pytest.raises(ValueError, match="at least k"):
        mmr_rerank_batch(scores, vectors, k=4, lambda_=0.2)


def test_invalid_lambda_is_rejected():
    with pytest.raises(ValueError, match="lambda_"):
        mmr_rerank([1], [0.9], [[1, 0]], lambda_=1.2)


# ---------------------------------------------------------------- measures

def test_ild_manual_example_and_identical_genres():
    genres = np.array([[1, 0], [1, 0], [0, 1]])
    assert intra_list_diversity(np.array([[0, 1, 2]]), genres)[0] == pytest.approx(2 / 3)
    assert intra_list_diversity(np.array([[0, 1]]), genres)[0] == pytest.approx(0.0)


def test_slide_11_calibration_of_the_70_30_user():
    # Movies 0-9 are drama only, 10-19 action only; the user liked 7 dramas and 3 action films.
    genres = np.array([[1, 0]] * 10 + [[0, 1]] * 10)
    history = np.zeros((1, 20))
    history[0, list(range(7)) + [10, 11, 12]] = 1
    sorted_list = np.array([list(range(10))])                      # 100% drama
    calibrated = np.array([list(range(7)) + [13, 14, 15]])          # 70% drama, 30% action
    users = np.array([0])
    # JSD between (0.7, 0.3) and (1, 0), in bits.
    m = np.array([0.85, 0.15])
    expected = 0.5 * (0.7 * np.log2(0.7 / m[0]) + 0.3 * np.log2(0.3 / m[1]) + np.log2(1 / m[0]))
    assert genre_calibration_error(sorted_list, genres, csr_matrix(history), users)[0] == pytest.approx(expected)
    assert genre_calibration_error(calibrated, genres, csr_matrix(history), users)[0] == pytest.approx(0.0)
    shares = genre_shares(genres)
    p = history_distribution(csr_matrix(history), users, shares)
    assert top_genre_gap(p, list_distribution(sorted_list, shares))[0] == pytest.approx(0.3)
    assert top_genre_gap(p, list_distribution(calibrated, shares))[0] == pytest.approx(0.0)


def test_calibration_edge_cases():
    genres = np.array([[1, 0], [0, 1]])
    users = np.array([0])
    assert genre_calibration_error(np.array([[1]]), genres, csr_matrix([[1, 0]]), users)[0] == pytest.approx(1.0)
    assert np.isnan(genre_calibration_error(np.array([[0, 1]]), genres, csr_matrix([[0, 0]]), users)[0])


def test_multi_genre_movies_split_their_weight():
    shares = genre_shares(np.array([[1, 1, 0], [0, 0, 1]]))
    np.testing.assert_allclose(list_distribution(np.array([[0, 1]]), shares), [[0.25, 0.25, 0.5]])


def test_collaborative_distance():
    # Users x items: movies 0 and 1 share both raters, movie 2 has its own.
    ratings = csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    similarity, lookup = item_cosine(ratings, np.array([0, 1, 2]))
    assert similarity[0, 1] == pytest.approx(1.0) and similarity[0, 2] == pytest.approx(0.0)
    distance = intra_list_distance(np.array([[0, 1], [0, 2]]), similarity, lookup)
    np.testing.assert_allclose(distance, [0.0, 1.0], atol=1e-12)


def test_self_information():
    assert self_information(np.array([[0, 1]]), np.array([0.5, 0.25]))[0] == pytest.approx(1.5)


def test_ratio_bootstrap_interval():
    rate = bootstrap_ratio(np.array([1, 0, 2, 1]), np.array([2, 2, 2, 2]), 500, 0.95, seed=3)
    assert rate.mean == pytest.approx(0.5) and 0.0 <= rate.low <= 0.5 <= rate.high <= 1.0


def test_system_bootstrap_interval():
    lists = np.array([[0, 1], [2, 3], [0, 1]])
    other = np.array([[0, 1], [0, 1], [0, 1]])
    coverage = lambda l: metrics.catalogue_coverage(l, 10)   # noqa: E731
    same = bootstrap_system_diff(lists, lists, coverage, 200, 0.95, seed=1)
    assert (same.mean, same.low, same.high) == (0.0, 0.0, 0.0)
    diff = bootstrap_system_diff(lists, other, coverage, 200, 0.95, seed=1)
    assert diff.mean == pytest.approx(0.2) and diff.low <= diff.mean <= diff.high


# ---------------------------------------------------------------- runner helpers

def test_lambda_is_the_most_diverse_within_the_loss_tolerance():
    summary = pd.DataFrame({"lambda": [1.0, 0.75, 0.5, 0.0], "ndcg": [0.200, 0.197, 0.190, 0.150],
                            "ild": [0.70, 0.76, 0.86, 0.90]})
    assert part3.choose_lambda(summary, max_drop=0.02) == 0.75
    assert part3.choose_lambda(summary, max_drop=0.0) == 1.0


def test_part3_runner_end_to_end(synthetic_ml, tmp_path):
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    ctx = RunContext(config=config, data=synthetic_ml, out=OutputPaths(tmp_path).create(), quick=True)
    part3.run(ctx)    # also asserts that lambda = 1 gives back EASE's own lists

    sweep = pd.read_csv(tmp_path / "tables" / "part3_sweep.csv")
    assert set(sweep["stage"]) == {"validation", "test"} and sweep["chosen_on_validation"].sum() == 2
    chosen = sweep.loc[sweep["chosen_on_validation"], "lambda"].iloc[0]
    quick = config.part3["quick"]
    rows = ctx.results.rows(part3.SHEET)
    # One row per lambda on the main pool, plus the ablation pools at the chosen lambda, 0.5 and 0.
    assert len(rows) == len(quick["lambdas"]) + len({chosen, 0.5, 0.0}) * (len(quick["pool_sizes"]) - 1)
    assert "none (plain EASE)" in {r["Re-ranking method"] for r in rows if r["Trade-off parameter"] == 1.0}
    for name in ("part3_differences.csv", "part3_pool_sizes.csv", "part3_who_pays.csv", "part3_genre_exposure.csv",
                 "part3_exposure.csv", "part3_test_per_user.csv", "part3_recommendations.csv"):
        assert (tmp_path / "tables" / name).is_file()
    assert (tmp_path / "figures" / "fig6_part3_frontier.png").is_file()


def test_part3_refuses_a_base_model_that_part2_did_not_choose(synthetic_ml, tmp_path):
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    out = OutputPaths(tmp_path).create()
    pd.DataFrame({"model": ["ease", "item_knn"], "params": ['{"lam": 300.0, "min_rating": 4}', "{}"],
                  "reference": [True, False]}).to_csv(out.tables / "part2_selected.csv", index=False)
    ctx = RunContext(config=config, data=synthetic_ml, out=out, quick=True)
    with pytest.raises(ValueError, match="best Part 2 model"):
        part3.run(ctx)
