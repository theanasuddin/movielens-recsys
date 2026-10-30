"""Part 1: data, protocol and baselines.

Steps, in the order run() calls them:
1. characterise the data (tables and Figure 1);
2. tune both baselines on the validation period only (Figure 2);
3. refit the chosen settings on train + validation and evaluate once on test,
   with bootstrap intervals and paired tests;
4. sensitivity checks: item/user filtering thresholds, and an 80/10/10
   count-based split in place of the calendar split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from recsys import analysis
from recsys.config import SplitConfig
from recsys.data import TemporalSplit, TrainData, make_train_data, temporal_split
from recsys.evaluation import EvaluationResult, bootstrap_ci, evaluate, paired_bootstrap_diff, wilcoxon_signed_rank
from recsys.experiments.context import RunContext
from recsys.models import ItemKNN, MostPopular, Recommender, build_model
from recsys.models.item_knn import item_similarity
from recsys.reporting import plots
from recsys.utils import save_table

log = logging.getLogger(__name__)

SHEET = "Part 1 - Baselines"
SELECTION_METRIC = "ndcg@10"


def _fmt_date(seconds: int) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%d")


def _none_if_nan(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else int(value)


def _grids(ctx: RunContext) -> dict[str, Any]:
    p1 = ctx.config.part1
    grids = {
        "knn_k": p1["item_knn"]["k"],
        "knn_variants": p1["item_knn"]["variants"],
        "pop_window_days": p1["most_popular"]["window_days"],
        "pop_min_rating": p1["most_popular"]["min_rating"],
        "item_thresholds": p1["filter_sensitivity"]["min_item_train_ratings"],
        "user_thresholds": p1["filter_sensitivity"]["min_user_history"],
    }
    if ctx.quick:
        grids.update(p1.get("quick", {}))
    return grids


# --------------------------------------------------------------------------
# 1. Data characterisation
# --------------------------------------------------------------------------

def characterise(ctx: RunContext, alternative_splits: dict[str, TemporalSplit]) -> None:
    ml, split, tables = ctx.data, ctx.split, ctx.out.tables
    save_table(analysis.dataset_summary(ml), tables / "data_summary.csv")
    save_table(analysis.long_tail(ml.ratings), tables / "long_tail.csv")
    save_table(analysis.monthly_activity(ml.ratings), tables / "monthly_activity.csv")
    save_table(analysis.burstiness(ml.ratings), tables / "burstiness.csv")
    save_table(split.summary(), tables / "split_summary.csv")

    profiles = []
    for name, s in {"calendar (main)": split, **alternative_splits}.items():
        for part, history, held_out in (("validation", s.train, s.val), ("test", s.train_val, s.test)):
            profile = analysis.held_out_profile(history, held_out, ctx.config.evaluation.relevance_threshold)
            profiles.append({"split": name, "part": part, "starts": _fmt_date(s.val_start if part == "validation" else s.test_start), **profile})
    save_table(pd.DataFrame(profiles), tables / "held_out_profiles.csv")

    drift = analysis.popularity_drift(split.train_val, split.test, ml.movies)
    save_table(drift, tables / "popularity_drift_top10.csv")
    overlap = pd.DataFrame([
        {"top_n": n, "movies_in_both_top_lists": analysis.top_overlap(split.train_val, split.test, n)}
        for n in (10, 50, 100)
    ])
    save_table(overlap, tables / "popularity_drift_overlap.csv")
    plots.data_overview(ml, split, ctx.out.figures / "fig1_data_overview.png")
    plots.rating_distribution(ml, ctx.out.figures / "rating_distribution.png")


# --------------------------------------------------------------------------
# 2. Tuning on the validation period
# --------------------------------------------------------------------------

@dataclass
class Selection:
    popularity: dict[str, Any]
    knn: dict[str, Any]
    popularity_grid: pd.DataFrame
    knn_grid: pd.DataFrame


def _train_data(ctx: RunContext, ratings: pd.DataFrame, cutoff: int, min_item_ratings: int | None = None) -> TrainData:
    threshold = ctx.config.filtering.min_item_train_ratings if min_item_ratings is None else min_item_ratings
    return make_train_data(ratings, ctx.ids, cutoff=cutoff, min_item_ratings=threshold)


def _evaluate(ctx: RunContext, model: Recommender, train: TrainData, held_out: pd.DataFrame, **kwargs) -> EvaluationResult:
    cfg = ctx.config
    kwargs.setdefault("min_history", cfg.filtering.min_user_history)
    return evaluate(model, train, held_out, cfg.evaluation, seed=cfg.seed, **kwargs)


def _best(grid: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    # idxmax returns the first maximum, so ties go to the earlier grid entry.
    row = grid.loc[grid[SELECTION_METRIC].idxmax()]
    params = {col: row[col] for col in columns}
    for col in ("window_days", "min_rating"):
        if col in params:
            params[col] = _none_if_nan(params[col])
    if "k" in params:
        params["k"] = int(params["k"])
    params["value"] = float(row[SELECTION_METRIC])
    return params


def tune(ctx: RunContext, split: TemporalSplit, grids: dict[str, Any]) -> Selection:
    train = _train_data(ctx, split.train, cutoff=split.val_start)

    pop_rows = []
    for window in grids["pop_window_days"]:
        for min_rating in grids["pop_min_rating"]:
            model = MostPopular(window_days=window, min_rating=min_rating).fit(train)
            result = _evaluate(ctx, model, train, split.val)
            pop_rows.append({"window_days": window, "min_rating": min_rating, **result.summary()})
    pop_grid = pd.DataFrame(pop_rows)

    knn_rows = []
    for variant in grids["knn_variants"]:
        sim = item_similarity(train, variant["similarity"], variant.get("min_rating"))
        for k in grids["knn_k"]:
            model = ItemKNN(k=k, similarity=variant["similarity"], min_rating=variant.get("min_rating"))
            result = _evaluate(ctx, model.fit_from_similarity(train, sim), train, split.val)
            knn_rows.append({"similarity": variant["similarity"], "min_rating": variant.get("min_rating"), "k": k, **result.summary()})
        log.info("Tuned item-kNN %s", variant)
    knn_grid = pd.DataFrame(knn_rows)

    return Selection(
        popularity=_best(pop_grid, ["window_days", "min_rating"]),
        knn=_best(knn_grid, ["similarity", "min_rating", "k"]),
        popularity_grid=pop_grid,
        knn_grid=knn_grid,
    )


def popularity_window_sizes(split: TemporalSplit, windows: list[int | None]) -> pd.DataFrame:
    """How much data each recency window holds before the validation and the test cut.

    A window measured in days covers very different amounts of activity at the
    two cuts, because activity collapsed after 2000 (Figure 1d).
    """
    rows = []
    for used_for, ratings, cutoff in (("validation", split.train, split.val_start), ("test", split.train_val, split.test_start)):
        for days in windows:
            if days is None:
                continue
            part = ratings[ratings["timestamp"] >= cutoff - days * 86_400]
            rows.append({
                "used_for": used_for, "window_days": days,
                "window": f"{_fmt_date(cutoff - days * 86_400)} to {_fmt_date(cutoff - 1)}",
                "ratings": len(part), "ratings_ge_4": int((part["rating"] >= 4).sum()),
                "users": part["user_id"].nunique(), "movies": part["item_id"].nunique(),
            })
    return pd.DataFrame(rows)


def _popularity(params: dict[str, Any]) -> MostPopular:
    return MostPopular(window_days=params["window_days"], min_rating=params["min_rating"])


def _knn(params: dict[str, Any]) -> ItemKNN:
    return ItemKNN(k=params["k"], similarity=params["similarity"], min_rating=params["min_rating"])


# --------------------------------------------------------------------------
# 3. Test evaluation
# --------------------------------------------------------------------------

def baseline_row(ctx: RunContext, model: Recommender, result: EvaluationResult, note: str = "") -> dict[str, Any]:
    cfg = ctx.config
    ci = bootstrap_ci(result.per_user["ndcg"], cfg.evaluation.bootstrap_samples, cfg.evaluation.confidence, cfg.seed)
    parts = [
        note,
        f"NDCG@10 {int(cfg.evaluation.confidence * 100)}% bootstrap CI [{ci.low:.4f}, {ci.high:.4f}]",
        f"{len(result.per_user)} test users, {result.catalogue_size} candidate movies",
    ]
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
        "Notes": "; ".join(p for p in parts if p),
    }


def _validation_note(value: float) -> str:
    return f"validation NDCG@10 {value:.4f}"


def evaluate_on_test(ctx: RunContext, split: TemporalSplit, selection: Selection, grids: dict[str, Any]) -> dict[str, EvaluationResult]:
    cfg = ctx.config
    train = _train_data(ctx, split.train_val, cutoff=split.test_start)
    results: dict[str, EvaluationResult] = {}

    random = build_model("random", default_seed=cfg.seed).fit(train)
    results["random"] = _evaluate(ctx, random, train, split.test)
    ctx.results.add(SHEET, baseline_row(ctx, random, results["random"], "sanity floor (Lecture 1.1, slide 9)"))

    for row in selection.popularity_grid.to_dict("records"):
        params = {"window_days": _none_if_nan(row["window_days"]), "min_rating": _none_if_nan(row["min_rating"])}
        model = _popularity(params).fit(train)
        result = _evaluate(ctx, model, train, split.test)
        chosen = params == {key: selection.popularity[key] for key in params}
        note = ("selected on validation; " if chosen else "") + _validation_note(row[SELECTION_METRIC])
        ctx.results.add(SHEET, baseline_row(ctx, model, result, note))
        if chosen:
            results["most_popular"] = result
        if params == {"window_days": None, "min_rating": None}:
            results["most_popular_all_time"] = result
        if params == {"window_days": None, "min_rating": 4}:
            results["most_popular_all_time_positive"] = result

    # The selected item-kNN, then the best k of every other similarity / signal
    # combination, so the report can show how much the similarity matters.
    grid = selection.knn_grid.assign(min_rating=selection.knn_grid["min_rating"].map(_none_if_nan))
    for (similarity, min_rating), rows in grid.groupby(["similarity", "min_rating"], sort=False, dropna=False):
        best = rows.loc[rows[SELECTION_METRIC].idxmax()]
        params = {"similarity": similarity, "min_rating": _none_if_nan(min_rating), "k": int(best["k"])}
        chosen = params == {key: selection.knn[key] for key in params}
        model = _knn(params).fit(train)
        result = _evaluate(ctx, model, train, split.test)
        note = ("selected on validation; " if chosen else "best k for this similarity on validation, not selected; ")
        ctx.results.add(SHEET, baseline_row(ctx, model, result, note + _validation_note(best[SELECTION_METRIC])))
        if chosen:
            results["item_knn"] = result

    paired_tests(ctx, results)
    per_user = pd.concat(
        [res.per_user.assign(model=name) for name, res in results.items()], ignore_index=True
    )
    save_table(per_user, ctx.out.tables / "part1_test_per_user.csv")
    return results


def paired_tests(ctx: RunContext, results: dict[str, EvaluationResult]) -> None:
    cfg = ctx.config.evaluation
    pairs = [
        ("item_knn", "most_popular"),
        ("item_knn", "most_popular_all_time"),
        ("most_popular", "most_popular_all_time"),
        ("most_popular", "most_popular_all_time_positive"),
    ]
    rows = []
    for a, b in pairs:
        if a not in results or b not in results or a == b:
            continue
        ua, ub = results[a].per_user, results[b].per_user
        if not np.array_equal(ua["user_id"].to_numpy(), ub["user_id"].to_numpy()):
            raise RuntimeError("Paired comparison needs the same users for both models")
        for metric in ("ndcg", "precision", "recall"):
            diff = paired_bootstrap_diff(ua[metric], ub[metric], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
            _, p_value = wilcoxon_signed_rank(ua[metric], ub[metric])
            rows.append({
                "model_a": a, "model_b": b, "metric": f"{metric}@10", "mean_a": ua[metric].mean(), "mean_b": ub[metric].mean(),
                "mean_difference": diff.mean, "ci_low": diff.low, "ci_high": diff.high,
                "users_a_better": int((ua[metric] > ub[metric]).sum()), "users_b_better": int((ua[metric] < ub[metric]).sum()),
                "wilcoxon_p": p_value,
            })
    save_table(pd.DataFrame(rows), ctx.out.tables / "part1_paired_tests.csv")


# --------------------------------------------------------------------------
# 4. Sensitivity checks
# --------------------------------------------------------------------------

def filter_sensitivity(ctx: RunContext, split: TemporalSplit, selection: Selection, grids: dict[str, Any]) -> pd.DataFrame:
    """Refit the selected baselines under different filtering thresholds.

    Hyperparameters stay at the values chosen under the main protocol, so any
    change comes from the filter alone.
    """
    full = split.train_val
    rows = []

    def record(label: str, threshold: int, ground_truth: str, train: TrainData, min_history: int) -> None:
        drop = ground_truth == "dropped"
        knn = _evaluate(ctx, _knn(selection.knn).fit(train), train, split.test, min_history=min_history, drop_unreachable=drop)
        pop = _evaluate(ctx, _popularity(selection.popularity).fit(train), train, split.test, min_history=min_history, drop_unreachable=drop)
        rows.append({
            "filter": label, "threshold": threshold, "ground_truth_for_removed_movies": ground_truth,
            "train_rows_removed": len(full) - len(train.ratings),
            "train_rows_removed_share": 1 - len(train.ratings) / len(full),
            "candidate_movies": int(train.catalogue.sum()), "test_users": len(knn.per_user),
            "item_knn_ndcg@10": knn.mean("ndcg"), "most_popular_ndcg@10": pop.mean("ndcg"),
            "gap_ndcg@10": knn.mean("ndcg") - pop.mean("ndcg"),
            "item_knn_recall@10": knn.mean("recall"), "most_popular_recall@10": pop.mean("recall"),
        })

    for threshold in grids["item_thresholds"]:
        train = _train_data(ctx, full, cutoff=split.test_start, min_item_ratings=threshold)
        for ground_truth in ("kept", "dropped"):
            if threshold == 1 and ground_truth == "dropped":
                continue  # nothing is removed at threshold 1
            record("min training ratings per movie", threshold, ground_truth, train, ctx.config.filtering.min_user_history)

    train = _train_data(ctx, full, cutoff=split.test_start)
    for threshold in grids["user_thresholds"]:
        if threshold == 1:
            continue  # same as the main protocol, already recorded above
        record("min training ratings per evaluated user", threshold, "kept", train, threshold)

    table = pd.DataFrame(rows)
    main = table.iloc[0]
    table["change_item_knn_vs_main"] = table["item_knn_ndcg@10"] - main["item_knn_ndcg@10"]
    table["change_most_popular_vs_main"] = table["most_popular_ndcg@10"] - main["most_popular_ndcg@10"]
    save_table(table, ctx.out.tables / "part1_filter_sensitivity.csv")
    return table


def split_sensitivity(ctx: RunContext, alternative_splits: dict[str, TemporalSplit], main: Selection, grids: dict[str, Any]) -> None:
    """Re-run tuning and testing with another split, and check what its tuned
    settings would have given on the main test set."""
    rows = []
    main_train = _train_data(ctx, ctx.split.train_val, cutoff=ctx.split.test_start)
    for name, split in alternative_splits.items():
        selection = tune(ctx, split, grids)
        save_table(selection.knn_grid, ctx.out.tables / f"part1_validation_itemknn_{_slug(name)}.csv")
        save_table(selection.popularity_grid, ctx.out.tables / f"part1_validation_popularity_{_slug(name)}.csv")
        own_train = _train_data(ctx, split.train_val, cutoff=split.test_start)
        for label, params, factory in (("ItemKNN", selection.knn, _knn), ("MostPopular", selection.popularity, _popularity)):
            own = _evaluate(ctx, factory(params).fit(own_train), own_train, split.test)
            transfer = _evaluate(ctx, factory(params).fit(main_train), main_train, ctx.split.test)
            settings = {key: value for key, value in params.items() if key != "value"}
            rows.append({
                "split": name, "model": label, "selected_settings": settings,
                "validation_ndcg@10": params["value"],
                "own_test_users": len(own.per_user), "own_test_ndcg@10": own.mean("ndcg"),
                "own_test_coverage": own.coverage,
                "main_test_ndcg@10_with_these_settings": transfer.mean("ndcg"),
            })
    for label, params in (("ItemKNN", main.knn), ("MostPopular", main.popularity)):
        rows.append({
            "split": "calendar (main)", "model": label,
            "selected_settings": {key: value for key, value in params.items() if key != "value"},
            "validation_ndcg@10": params["value"],
        })
    save_table(pd.DataFrame(rows), ctx.out.tables / "part1_split_sensitivity.csv")


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.split(" (")[0]).strip("_")


# --------------------------------------------------------------------------

def alternative_splits(ctx: RunContext) -> dict[str, TemporalSplit]:
    out = {}
    for spec in ctx.config.part1.get("split_sensitivity", []):
        cfg = replace(SplitConfig(), **{k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()})
        if cfg.mode == "quantile":
            q_val, q_test = cfg.quantiles
            shares = [round(100 * q_val), round(100 * (q_test - q_val)), round(100 * (1 - q_test))]
            label = "count-based " + "/".join(str(s) for s in shares)
        else:
            label = f"calendar {cfg.val_start} / {cfg.test_start}"
        out[label] = temporal_split(ctx.data.ratings, cfg)
    return out


def run(ctx: RunContext) -> None:
    split = ctx.split
    grids = _grids(ctx)
    log.info("Temporal split: validation from %s, test from %s", _fmt_date(split.val_start), _fmt_date(split.test_start))
    others = alternative_splits(ctx)

    characterise(ctx, others)

    selection = tune(ctx, split, grids)
    save_table(selection.knn_grid, ctx.out.tables / "part1_validation_itemknn.csv")
    save_table(selection.popularity_grid, ctx.out.tables / "part1_validation_popularity.csv")
    save_table(popularity_window_sizes(split, grids["pop_window_days"]), ctx.out.tables / "popularity_window_sizes.csv")
    log.info("Selected on validation: item-kNN %s, MostPopular %s", selection.knn, selection.popularity)
    plots.knn_validation(selection.knn_grid, selection.popularity["value"], selection.knn,
                         ctx.out.figures / "fig2_itemknn_validation.png", k_metric=SELECTION_METRIC)

    results = evaluate_on_test(ctx, split, selection, grids)
    for name, result in results.items():
        log.info("Test %s: %s", name, result.summary())

    filter_sensitivity(ctx, split, selection, grids)
    split_sensitivity(ctx, others, selection, grids)
