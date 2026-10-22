"""Our metric code against hand-worked numbers and against scikit-learn.

The course allows our own metric code if we show it agrees with a reference
on a sample. The first test reproduces Lecture 3.1, slide 8; the second
compares NDCG@10 with sklearn.metrics.ndcg_score on random rankings.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from recsys.evaluation import metrics


def test_lecture_3_1_slide_8_example():
    # Bob's relevant items are {i1, i7}. Three top-5 lists:
    lists = {
        "B": ["i1", "i7", "i9", "i4", "i3"],
        "A": ["i4", "i1", "i9", "i7", "i3"],
        "C": ["i9", "i3", "i4", "i1", "i7"],
    }
    relevant = {"i1", "i7"}
    hits = np.array([[item in relevant for item in lst] for lst in lists.values()])
    n_rel = np.full(3, len(relevant))

    np.testing.assert_allclose(metrics.precision_at_k(hits), [0.4, 0.4, 0.4])
    np.testing.assert_allclose(metrics.recall_at_k(hits, n_rel), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(metrics.ndcg_at_k(hits, n_rel), [1.00, 0.65, 0.50], atol=0.005)


def test_ndcg_matches_sklearn_on_random_rankings():
    rng = np.random.default_rng(0)
    n_users, n_items, k = 500, 200, 10
    relevance = (rng.random((n_users, n_items)) < 0.05).astype(float)
    relevance[relevance.sum(axis=1) == 0, 0] = 1.0  # every user needs one relevant item
    scores = rng.random((n_users, n_items))

    top = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(relevance, top, axis=1).astype(bool)
    ours = metrics.ndcg_at_k(hits, relevance.sum(axis=1).astype(int))

    reference = np.array([ndcg_score(relevance[u:u + 1], scores[u:u + 1], k=k) for u in range(n_users)])
    np.testing.assert_allclose(ours, reference, atol=1e-12)


def test_ndcg_ideal_uses_at_most_k_relevant_items():
    hits = np.ones((1, 10), dtype=bool)
    assert metrics.ndcg_at_k(hits, np.array([25]))[0] == pytest.approx(1.0)
    assert metrics.recall_at_k(hits, np.array([25]))[0] == pytest.approx(10 / 25)


def test_coverage_counts_distinct_items():
    recs = np.array([[0, 1, 2], [1, 2, 3]])
    assert metrics.catalogue_coverage(recs, catalogue_size=8) == pytest.approx(4 / 8)


def test_gini_extremes():
    assert metrics.gini(np.ones(10)) == pytest.approx(0.0)
    assert metrics.gini(np.r_[np.zeros(9), 5.0]) == pytest.approx(0.9)
    assert metrics.gini(np.zeros(4)) == 0.0
