"""Focused Part 2 regression tests on small deterministic data."""
import numpy as np
import pandas as pd
from recsys.data.dataset import make_train_data
from recsys.models import UserKNN, BiasedMF, ContentProfile
from recsys.evaluation import evaluate
from recsys.config import EvaluationConfig


def test_part2_models_fit_and_rank(tiny_ml, tiny_ids):
    train=make_train_data(tiny_ml.ratings.iloc[:8],tiny_ids,cutoff=1000000000)
    models=[UserKNN(k=2),BiasedMF(factors=3,epochs=3,lr=0.1,batch_size=4,seed=42),
            ContentProfile(tiny_ml.movies,year_weight=.5)]
    for model in models:
        model.fit(train)
        scores=model.score(np.array([0,1]))
        assert scores.shape==(2,6)
        assert np.isfinite(scores).all()
        result=evaluate(model,train,tiny_ml.ratings.iloc[8:11],EvaluationConfig(k=3,batch_size=2),seed=42)
        assert np.isfinite(result.mean('ndcg'))


def test_mf_is_reproducible(tiny_ml,tiny_ids):
    train=make_train_data(tiny_ml.ratings.iloc[:8],tiny_ids,cutoff=1000000000)
    a=BiasedMF(factors=3,epochs=4,seed=123,batch_size=4).fit(train)
    b=BiasedMF(factors=3,epochs=4,seed=123,batch_size=4).fit(train)
    assert np.array_equal(a.score(np.array([0])),b.score(np.array([0])))


def test_part2_equal_validation_budgets():
    from recsys.experiments.part2 import grids
    for quick, expected in [(False, 8), (True, 2)]:
        options = grids(quick)
        assert set(options) == {'most_popular', 'item_knn', 'user_knn', 'biased_mf', 'content_profile'}
        assert all(len(rows) == expected for rows in options.values())
        assert all(len({str(r) for r in rows}) == expected for rows in options.values())
