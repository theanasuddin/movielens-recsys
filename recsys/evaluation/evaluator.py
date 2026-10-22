"""Top-k evaluation against the full candidate catalogue.

For each evaluated user the model scores every item. Items the user rated in
the training data and items outside the catalogue are removed, the top k of the
rest form the list, and the list is compared with the user's relevant held-out
items (ratings >= relevance_threshold). No negative sampling is used
(Lecture 3.1, slide 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.config import EvaluationConfig
from recsys.data.dataset import TrainData, to_matrix
from recsys.evaluation import metrics
from recsys.models.base import Recommender

METRIC_NAMES = ("precision", "recall", "ndcg")


@dataclass
class EvaluationResult:
    per_user: pd.DataFrame          # user_id, n_history, n_relevant, precision, recall, ndcg
    recommendations: np.ndarray     # item indices, shape (n_users, k), row order = per_user
    catalogue_size: int
    k: int

    @property
    def coverage(self) -> float:
        return metrics.catalogue_coverage(self.recommendations, self.catalogue_size)

    def mean(self, metric: str) -> float:
        return float(self.per_user[metric].mean())

    def summary(self) -> dict[str, float]:
        out = {f"{name}@{self.k}": self.mean(name) for name in METRIC_NAMES}
        out["coverage"] = self.coverage
        out["n_users"] = len(self.per_user)
        return out


def top_k(scores: np.ndarray, k: int, tie_order: np.ndarray) -> np.ndarray:
    """Column indices of the k highest scores in each row.

    Equal scores are ordered by tie_order, a fixed permutation of the item
    indices, so ties do not silently favour low movie ids.
    """
    if np.isnan(scores).any():
        raise ValueError("Model returned NaN scores")
    permuted = scores[:, tie_order]
    order = np.argsort(-permuted, axis=1, kind="stable")[:, :k]
    return tie_order[order]


def evaluation_users(train: TrainData, held_out: pd.DataFrame, relevance_threshold: int, min_history: int) -> np.ndarray:
    """User indices with at least one relevant held-out item and enough history."""
    relevant = held_out[held_out["rating"] >= relevance_threshold]
    users = np.unique(train.ids.user_index(relevant["user_id"]))
    return users[train.history_sizes()[users] >= min_history]


def evaluate(
    model: Recommender,
    train: TrainData,
    held_out: pd.DataFrame,
    cfg: EvaluationConfig,
    min_history: int = 1,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate a fitted model on held-out ratings (validation or test)."""
    ids = train.ids
    relevant = to_matrix(held_out[held_out["rating"] >= cfg.relevance_threshold], ids)
    users = evaluation_users(train, held_out, cfg.relevance_threshold, min_history)
    tie_order = np.random.default_rng(seed).permutation(ids.n_items)
    history = train.history_sizes()

    recs, hit_rows, n_rel = [], [], []
    for start in range(0, len(users), cfg.batch_size):
        batch = users[start:start + cfg.batch_size]
        scores = np.array(model.score(batch), dtype=np.float64)
        scores[:, ~train.catalogue] = -np.inf
        seen_rows, seen_cols = train.seen[batch].nonzero()
        scores[seen_rows, seen_cols] = -np.inf

        top = top_k(scores, cfg.k, tie_order)
        rel = relevant[batch].toarray() > 0
        recs.append(top)
        hit_rows.append(np.take_along_axis(rel, top, axis=1))
        n_rel.append(rel.sum(axis=1))

    k = cfg.k
    recommendations = np.vstack(recs) if recs else np.empty((0, k), dtype=np.int64)
    hits = np.vstack(hit_rows) if hit_rows else np.empty((0, k), dtype=bool)
    n_relevant = np.concatenate(n_rel) if n_rel else np.empty(0, dtype=np.int64)

    per_user = pd.DataFrame({
        "user_id": ids.users[users].to_numpy(),
        "n_history": history[users],
        "n_relevant": n_relevant,
        "precision": metrics.precision_at_k(hits),
        "recall": metrics.recall_at_k(hits, n_relevant),
        "ndcg": metrics.ndcg_at_k(hits, n_relevant),
    })
    return EvaluationResult(per_user, recommendations, int(train.catalogue.sum()), k)
