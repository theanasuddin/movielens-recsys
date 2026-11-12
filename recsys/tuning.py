"""Hyperparameter search on the validation period (Part 2).

A search space is either a grid, written as blocks of {parameter: [values]}
whose values are crossed (the blocks are then put together), or a random
space of {parameter: {"choice": [...]}} and {parameter: {"loguniform": [low,
high]}} entries, sampled with a fixed seed.

Models that train in epochs (BiasedMF, BPR-MF) also choose their number of
epochs on validation: run_trial checks them every `eval_every` epochs and
stops after `patience` checks without improvement, or at `max_epochs`.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np

from recsys.data.dataset import TrainData
from recsys.evaluation import EvaluationResult
from recsys.models.base import Recommender


def expand_grid(blocks: list[dict[str, list]]) -> list[dict[str, Any]]:
    settings = []
    for block in blocks:
        names = list(block)
        for values in itertools.product(*(block[name] for name in names)):
            settings.append(dict(zip(names, values)))
    return settings


def sample_space(space: dict[str, dict], n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    settings = []
    for _ in range(n):
        setting = {}
        for name, spec in space.items():
            if "choice" in spec:
                options = spec["choice"]
                setting[name] = options[int(rng.integers(len(options)))]
            elif "loguniform" in spec:
                low, high = spec["loguniform"]
                setting[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                raise ValueError(f"Unknown search spec for {name!r}: {spec}")
        settings.append(setting)
    return settings


def candidate_settings(space: dict, random_trials: int, seed: int) -> list[dict[str, Any]]:
    if "grid" in space:
        return expand_grid(space["grid"])
    if "random" in space:
        return sample_space(space["random"], random_trials, seed)
    raise ValueError("A search space needs a 'grid' or a 'random' entry")


def run_trial(
    model: Recommender,
    train: TrainData,
    score: Callable[[Recommender], EvaluationResult],
    max_epochs: int,
    eval_every: int,
    patience: int,
) -> tuple[int | None, EvaluationResult, list[tuple[int, float]]]:
    """Fit one setting and score it on validation.

    Returns (chosen epochs or None, validation result, [(epoch, NDCG@10), ...]).
    Models without run_epoch() are fitted once.
    """
    if not hasattr(model, "run_epoch"):
        model.fit(train)
        return None, score(model), []

    model.initialise(train)
    best_epoch, best, curve, stale = 0, None, [], 0
    for epoch in range(1, max_epochs + 1):
        model.run_epoch()
        if epoch % eval_every and epoch != max_epochs:
            continue
        result = score(model)
        curve.append((epoch, result.mean("ndcg")))
        if best is None or result.mean("ndcg") > best.mean("ndcg"):
            best_epoch, best, stale = epoch, result, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch, best, curve
