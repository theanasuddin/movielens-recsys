"""Model registry. Add new models here so the experiment runners can build them by name."""

from __future__ import annotations

from typing import Any

from recsys.models.base import Recommender
from recsys.models.popularity import MostPopular, RandomRecommender

MODELS: dict[str, type[Recommender]] = {
    "random": RandomRecommender,
    "most_popular": MostPopular,
}


def build_model(name: str, default_seed: int | None = None, **params: Any) -> Recommender:
    """Build a model by registry name. Stochastic models get default_seed
    unless params already contains a seed."""
    try:
        cls = MODELS[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; known models: {sorted(MODELS)}") from None
    if cls.stochastic and "seed" not in params:
        if default_seed is None:
            raise ValueError(f"{cls.name} is stochastic and needs a seed")
        params["seed"] = default_seed
    return cls(**params)


__all__ = ["MODELS", "MostPopular", "RandomRecommender", "Recommender", "build_model"]
