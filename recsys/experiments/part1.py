"""Part 1: data, protocol and baselines.

Current state: the protocol (split, candidate set, metrics) runs end to end
with the reference models listed under part1.reference_models in the config.
Data characterisation, item-based kNN tuning and the popularity variants are
added on top of this runner.
"""

from __future__ import annotations

import logging

import pandas as pd

from recsys.data import make_train_data
from recsys.evaluation import EvaluationResult, bootstrap_ci, evaluate
from recsys.experiments.context import RunContext
from recsys.models import Recommender, build_model
from recsys.utils import save_table

log = logging.getLogger(__name__)

SHEET = "Part 1 - Baselines"


def _fmt_date(seconds: int) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%d")


def baseline_row(ctx: RunContext, model: Recommender, result: EvaluationResult, extra_note: str = "") -> dict:
    cfg = ctx.config
    ci = bootstrap_ci(result.per_user["ndcg"], cfg.evaluation.bootstrap_samples, cfg.evaluation.confidence, cfg.seed)
    note = (
        f"test users={len(result.per_user)}; candidates={result.catalogue_size} items; "
        f"NDCG@10 {int(cfg.evaluation.confidence * 100)}% bootstrap CI [{ci.low:.4f}, {ci.high:.4f}]"
    )
    if extra_note:
        note += f"; {extra_note}"
    summary = result.summary()
    return {
        "Model": model.name,
        "Variant / settings": model.describe(),
        "Similarity": getattr(model, "similarity", None),
        "k": getattr(model, "k", None),
        "Precision@10": summary["precision@10"],
        "Recall@10": summary["recall@10"],
        "NDCG@10": summary["ndcg@10"],
        "Catalogue coverage": summary["coverage"],
        "Seed": cfg.seed,
        "Notes": note,
    }


def run(ctx: RunContext) -> None:
    cfg = ctx.config
    split = ctx.split
    log.info(
        "Temporal split: validation from %s, test from %s",
        _fmt_date(split.val_start), _fmt_date(split.test_start),
    )
    save_table(split.summary(), ctx.out.tables / "split_summary.csv")

    # Final protocol: fit on train + validation, evaluate once on test.
    train_full = make_train_data(
        split.train_val, ctx.ids, cutoff=split.test_start,
        min_item_ratings=cfg.filtering.min_item_train_ratings,
    )
    for spec in cfg.part1.get("reference_models", []):
        model = build_model(spec["name"], default_seed=cfg.seed, **(spec.get("params") or {}))
        model.fit(train_full)
        result = evaluate(model, train_full, split.test, cfg.evaluation, cfg.filtering.min_user_history, seed=cfg.seed)
        log.info("%s (%s): %s", model.name, model.describe(), result.summary())
        ctx.results.add(SHEET, baseline_row(ctx, model, result))
