"""Model registry. Add new models here so the experiment runners can build them by name."""

from __future__ import annotations

from typing import Any

from recsys.models.base import Recommender
from recsys.models.bpr import BPRMF
from recsys.models.content_based import ContentProfile
from recsys.models.ease import EASE
from recsys.models.item_knn import ItemKNN
from recsys.models.matrix_factorization import BiasedMF
from recsys.models.popularity import MostPopular, RandomRecommender
from recsys.models.user_knn import UserKNN

MODELS: dict[str, type[Recommender]] = {
    "random": RandomRecommender,
    "most_popular": MostPopular,
    "item_knn": ItemKNN,
    "user_knn": UserKNN,
    "biased_mf": BiasedMF,
    "bpr_mf": BPRMF,
    "ease": EASE,
    "content_profile": ContentProfile,
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


__all__ = [
    "BPRMF", "EASE", "MODELS", "BiasedMF", "ContentProfile", "ItemKNN", "MostPopular",
    "RandomRecommender", "Recommender", "UserKNN", "build_model",
]
