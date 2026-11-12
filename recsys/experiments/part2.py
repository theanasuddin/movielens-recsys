"""Part 2: models and honest comparison.

Steps, in the order run() calls them:
1. tune every model on the validation period over the search space declared
   in configs/default.yaml (the same procedure for baselines and new models);
2. refit each chosen setting on train + validation and score it once on
   test, with five seeds for the two stochastic models;
3. compare the best model on validation with every other model on paired
   per-user values (bootstrap interval, Wilcoxon test, Holm correction);
4. break the test results down by user activity, item popularity and test
   period;
5. repeat the comparison with 50% and 20% of the training interactions.

The protocol is Part 1's: same split, relevance threshold, candidate set and
evaluator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.data import TrainData, make_train_data
from recsys.evaluation import (
    EvaluationResult,
    bootstrap_ci,
    evaluate,
    holm,
    metrics_against,
    paired_bootstrap_diff,
    relevance_matrix,
    wilcoxon_signed_rank,
)
from recsys.experiments.context import RunContext
from recsys.models import Recommender, build_model
from recsys.models.popularity import SECONDS_PER_DAY
from recsys.reporting import plots
from recsys.tuning import candidate_settings, run_trial, sample_space
from recsys.utils import save_table

log = logging.getLogger(__name__)

MODELS_SHEET = "Part 2 - Models"
BREAKDOWN_SHEET = "Part 2 - Disaggregated"
SPARSITY_SHEET = "Part 2 - Sparsity"

MODEL_KEYS = ("most_popular", "item_knn", "user_knn", "biased_mf", "bpr_mf", "ease", "content_profile")
PARADIGMS = {
    "most_popular": "Non-personalised (Part 1 baseline)",
    "item_knn": "Neighbourhood CF (Part 1 baseline)",
    "user_knn": "Neighbourhood CF",
    "biased_mf": "Latent-factor CF",
    "bpr_mf": "Ranking-loss CF",
    "ease": "Linear autoencoder (item-item)",
    "content_profile": "Content-based",
}
# The two models we designate and document as self-implemented. The other
# models are our own numpy code too, but are not counted for the requirement.
SELF_IMPLEMENTED = ("biased_mf", "bpr_mf")
METRICS = ("precision", "recall", "ndcg")
RARE_ITEM_RATINGS = 20


# --------------------------------------------------------------------------
# Settings and helpers
# --------------------------------------------------------------------------

@dataclass
class Settings:
    spaces: dict[str, dict]
    random_trials: int
    max_epochs: int
    eval_every: int
    patience: int
    test_seeds: list[int]
    activity_quantiles: list[float]
    head_share: float
    fractions: list[float]
    sampling_seeds: list[int]
    retune_trials: int
    max_grid: int | None = None


def settings(ctx: RunContext) -> Settings:
    p2 = ctx.config.part2
    values = {
        "spaces": p2["models"],
        "random_trials": p2["random_trials"],
        "max_epochs": p2["max_epochs"],
        "eval_every": p2["eval_every"],
        "patience": p2["patience"],
        "test_seeds": list(p2["test_seeds"]),
        "activity_quantiles": list(p2["breakdowns"]["activity_quantiles"]),
        "head_share": p2["breakdowns"]["head_share"],
        "fractions": list(p2["sparsity"]["fractions"]),
        "sampling_seeds": list(p2["sparsity"]["sampling_seeds"]),
        "retune_trials": p2["sparsity"]["retune_trials"],
    }
    if ctx.quick:
        values.update(p2.get("quick", {}))
    return Settings(**values)


def _train(ctx: RunContext, ratings: pd.DataFrame, cutoff: int, history: pd.DataFrame | None = None) -> TrainData:
    return make_train_data(ratings, ctx.ids, cutoff, ctx.config.filtering.min_item_train_ratings, history=history)


def _evaluate(ctx: RunContext, model: Recommender, train: TrainData, held_out: pd.DataFrame) -> EvaluationResult:
    cfg = ctx.config
    return evaluate(model, train, held_out, cfg.evaluation, cfg.filtering.min_user_history, seed=cfg.seed)


def make(ctx: RunContext, key: str, params: dict[str, Any], seed: int, epochs: int | None = None) -> Recommender:
    params = dict(params)
    if key == "content_profile":
        params["movies"] = ctx.data.movies
    if epochs is not None:
        params["epochs"] = epochs
    return build_model(key, default_seed=seed, **params)


def _as_text(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def subsample(ratings: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep `fraction` of each user's ratings (at least one), chosen at random."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(ratings), dtype=bool)
    for rows in ratings.groupby("user_id", sort=True).indices.values():
        n = max(1, int(round(fraction * len(rows))))
        keep[rng.choice(rows, size=n, replace=False)] = True
    return ratings[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. Tuning on validation
# --------------------------------------------------------------------------

@dataclass
class Choice:
    params: dict[str, Any]
    epochs: int | None
    validation_ndcg: float
    trials: int


def tune(ctx: RunContext, s: Settings, key: str, options: list[dict], train: TrainData,
         held_out: pd.DataFrame, stage: str, trial_rows: list, curve_rows: list) -> Choice:
    best: Choice | None = None
    for number, params in enumerate(options, 1):
        start = time.perf_counter()
        epochs, result, curve = run_trial(
            make(ctx, key, params, ctx.config.seed), train, lambda m: _evaluate(ctx, m, train, held_out),
            s.max_epochs, s.eval_every, s.patience,
        )
        summary = result.summary()
        trial_rows.append({
            "stage": stage, "model": key, "trial": number, "params": _as_text(params), "epochs": epochs,
            "ndcg@10": summary["ndcg@10"], "precision@10": summary["precision@10"],
            "recall@10": summary["recall@10"], "coverage": summary["coverage"],
            "seconds": round(time.perf_counter() - start, 1),
        })
        curve_rows.extend({"stage": stage, "model": key, "trial": number, "epoch": e, "ndcg@10": v} for e, v in curve)
        if best is None or summary["ndcg@10"] > best.validation_ndcg:
            best = Choice(dict(params), epochs, summary["ndcg@10"], len(options))
    log.info("%s %s: best of %d trials %s, validation NDCG@10 %.4f", stage, key, len(options), best.params,
             best.validation_ndcg)
    return best


def options_for(ctx: RunContext, s: Settings, key: str) -> list[dict]:
    space = s.spaces[key]
    options = candidate_settings(space, s.random_trials, ctx.config.seed)
    if s.max_grid is not None and "grid" in space:
        options = options[: s.max_grid]
    return options


def retune_options(ctx: RunContext, s: Settings, key: str, chosen: Choice) -> list[dict]:
    """The full-data choice first, then random draws from the same space."""
    space = s.spaces[key]
    extra = s.retune_trials - 1
    if "grid" in space:
        grid = [p for p in options_for(ctx, s, key) if p != chosen.params]
        rng = np.random.default_rng(ctx.config.seed)
        picked = rng.choice(len(grid), size=min(extra, len(grid)), replace=False) if grid else []
        return [chosen.params] + [grid[i] for i in sorted(picked)]
    return [chosen.params] + sample_space(space["random"], extra, ctx.config.seed + 1)


# --------------------------------------------------------------------------
# 2. Test runs
# --------------------------------------------------------------------------

@dataclass
class TestRun:
    key: str
    choice: Choice
    seeds: list[int]
    results: list[EvaluationResult] = field(default_factory=list)
    models: list[Recommender] = field(default_factory=list)

    def per_user(self) -> pd.DataFrame:
        """Per-user metrics, averaged over the seeds for the stochastic models."""
        frames = [r.per_user.set_index("user_id") for r in self.results]
        mean = sum(f[list(METRICS)] for f in frames) / len(frames)
        return frames[0][["n_history", "n_relevant"]].join(mean).reset_index()

    def mean(self, metric: str) -> float:
        return float(self.per_user()[metric].mean())

    @property
    def coverage(self) -> float:
        return float(np.mean([r.coverage for r in self.results]))

    def seed_values(self, metric: str = "ndcg") -> np.ndarray:
        return np.array([r.mean(metric) for r in self.results])


def run_tests(ctx: RunContext, s: Settings, choices: dict[str, Choice], train: TrainData) -> dict[str, TestRun]:
    runs = {}
    for key in MODEL_KEYS:
        choice = choices[key]
        stochastic = make(ctx, key, choice.params, ctx.config.seed).stochastic
        run = TestRun(key, choice, s.test_seeds if stochastic else [ctx.config.seed])
        for seed in run.seeds:
            model = make(ctx, key, choice.params, seed, epochs=choice.epochs).fit(train)
            run.results.append(_evaluate(ctx, model, train, ctx.split.test))
            run.models.append(model)
        runs[key] = run
        log.info("Test %s: NDCG@10 %.4f over %d seed(s)", key, run.mean("ndcg"), len(run.seeds))
    return runs


def popularity_windows(ctx: RunContext, s: Settings, choice: Choice, val_train: TrainData,
                       full_train: TrainData) -> pd.DataFrame:
    """Every most-popular setting at both cuts: the data and days its window
    holds, its validation score and, as a check only, its test score.

    Part 1 found that a window in days holds very different amounts of data
    at the validation and the test cut; this shows whether a window of the
    most recent ratings carries over better. Selection still uses validation.
    """
    rows = []
    for params in options_for(ctx, s, "most_popular"):
        row = {"params": _as_text(params), "chosen_on_validation": params == choice.params}
        for stage, train, held_out in (("validation", val_train, ctx.split.val), ("test", full_train, ctx.split.test)):
            model = make(ctx, "most_popular", params, ctx.config.seed).fit(train)
            row["setting"] = model.describe()
            row[f"{stage}_ndcg@10"] = _evaluate(ctx, model, train, held_out).mean("ndcg")
            row[f"{stage}_window_ratings"] = model.window_size
            row[f"{stage}_window_days"] = (train.cutoff - model.window_start) / SECONDS_PER_DAY
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Paired comparisons
# --------------------------------------------------------------------------

def paired_tests(ctx: RunContext, runs: dict[str, TestRun], reference: str) -> pd.DataFrame:
    cfg = ctx.config.evaluation
    ref = runs[reference].per_user()
    rows = []
    for key in MODEL_KEYS:
        if key == reference:
            continue
        other = runs[key].per_user()
        if not np.array_equal(ref["user_id"].to_numpy(), other["user_id"].to_numpy()):
            raise RuntimeError("Paired comparison needs the same users for both models")
        row = {"reference": reference, "model": key}
        for metric in METRICS:
            diff = paired_bootstrap_diff(ref[metric], other[metric], cfg.bootstrap_samples, cfg.confidence,
                                         ctx.config.seed)
            row.update({f"{metric}_difference": diff.mean, f"{metric}_ci_low": diff.low,
                        f"{metric}_ci_high": diff.high})
        _, row["wilcoxon_p"] = wilcoxon_signed_rank(ref["ndcg"], other["ndcg"])
        row["users_reference_better"] = int((ref["ndcg"] > other["ndcg"]).sum())
        row["users_model_better"] = int((ref["ndcg"] < other["ndcg"]).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["wilcoxon_p"].to_numpy())
    return table


# --------------------------------------------------------------------------
# 4. Breakdowns
# --------------------------------------------------------------------------

def _group_row(ctx: RunContext, key: str, dimension: str, group: str, definition: str,
               per_user: pd.DataFrame, coverage: float) -> dict[str, Any]:
    cfg = ctx.config.evaluation
    row = {"model": key, "dimension": dimension, "group": group, "definition": definition,
           "users": len(per_user), "coverage": coverage}
    if per_user.empty:
        return {**row, "precision": np.nan, "recall": np.nan, "ndcg": np.nan, "ndcg_ci_low": np.nan,
                "ndcg_ci_high": np.nan}
    ci = bootstrap_ci(per_user["ndcg"], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
    return {**row, "precision": per_user["precision"].mean(), "recall": per_user["recall"].mean(),
            "ndcg": ci.mean, "ndcg_ci_low": ci.low, "ndcg_ci_high": ci.high}


def _seed_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    indexed = [f.set_index("user_id")[list(METRICS)] for f in frames]
    return (sum(indexed) / len(indexed)).reset_index()


def breakdowns(ctx: RunContext, s: Settings, runs: dict[str, TestRun], train: TrainData) -> tuple[pd.DataFrame, dict]:
    """Per-group metrics for every model, plus the per-user values for paired checks."""
    ids, test = ctx.ids, ctx.split.test
    threshold = ctx.config.evaluation.relevance_threshold
    catalogue_size = int(train.catalogue.sum())

    # User activity: thirds of the evaluated users' training history.
    first = runs[MODEL_KEYS[0]].per_user()
    low, high = np.quantile(first["n_history"], s.activity_quantiles)
    user_groups = {
        "light": (first["n_history"] <= low, f"<= {low:.0f} training ratings"),
        "medium": ((first["n_history"] > low) & (first["n_history"] <= high), f"{low:.0f}-{high:.0f} training ratings"),
        "heavy": (first["n_history"] > high, f"> {high:.0f} training ratings"),
    }

    # Item popularity: the most-rated head_share of the candidate movies is the head.
    counts = np.bincount(ids.item_index(train.ratings["item_id"]), minlength=ids.n_items)
    candidates = np.flatnonzero(train.catalogue)
    n_head = int(np.ceil(s.head_share * len(candidates)))
    head = np.zeros(ids.n_items, dtype=bool)
    head[candidates[np.argsort(-counts[candidates], kind="stable")[:n_head]]] = True
    tail = train.catalogue & ~head
    head_min = int(counts[head].min())
    relevant = relevance_matrix(test, ids, threshold)
    item_groups = {
        "head": (relevant @ sp.diags(head.astype(float)), head, f"top {s.head_share:.0%} most-rated movies ({head.sum()}, >= {head_min} ratings)"),
        "tail": (relevant @ sp.diags(tail.astype(float)), tail, f"other {tail.sum()} candidate movies (< {head_min} ratings)"),
    }

    # Test period: before and after the median timestamp of the test ratings.
    middle = int(np.median(test["timestamp"]))
    day = pd.Timestamp(middle, unit="s", tz="UTC").strftime("%Y-%m-%d")
    time_groups = {
        "early": (relevance_matrix(test[test["timestamp"] < middle], ids, threshold), f"test ratings before {day}"),
        "late": (relevance_matrix(test[test["timestamp"] >= middle], ids, threshold), f"test ratings from {day}"),
    }

    rows, per_user_groups = [], {}
    for key in MODEL_KEYS:
        run = runs[key]
        per_user = run.per_user()
        for group, (mask, definition) in user_groups.items():
            mask = mask.to_numpy()
            coverage = np.mean([np.unique(r.recommendations[mask]).size / catalogue_size for r in run.results])
            rows.append(_group_row(ctx, key, "User activity", group, definition, per_user[mask], coverage))
            per_user_groups[(key, "User activity", group)] = per_user[mask]
        for group, (rel, members, definition) in item_groups.items():
            scored = [metrics_against(r, rel.tocsr(), ids)[0] for r in run.results]
            in_group = np.mean([np.isin(np.flatnonzero(members), np.unique(r.recommendations)).mean() for r in run.results])
            group_users = _seed_mean(scored)
            rows.append(_group_row(ctx, key, "Item popularity", group, definition, group_users, in_group))
            per_user_groups[(key, "Item popularity", group)] = group_users
        for group, (rel, definition) in time_groups.items():
            outcomes = [metrics_against(r, rel, ids) for r in run.results]
            coverage = np.mean([np.unique(r.recommendations[keep]).size / catalogue_size
                                for r, (_, keep) in zip(run.results, outcomes)])
            group_users = _seed_mean([o[0] for o in outcomes])
            rows.append(_group_row(ctx, key, "Test period", group, definition, group_users, coverage))
            per_user_groups[(key, "Test period", group)] = group_users
    return pd.DataFrame(rows), {"per_user": per_user_groups, "head": head, "tail": tail}


def breakdown_differences(ctx: RunContext, per_user_groups: dict, reference: str) -> pd.DataFrame:
    """Reference model minus every other model inside each group (paired, bootstrap)."""
    cfg = ctx.config.evaluation
    rows = []
    for (key, dimension, group), frame in per_user_groups.items():
        if key == reference:
            continue
        ref = per_user_groups[(reference, dimension, group)]
        merged = ref.merge(frame, on="user_id", suffixes=("_ref", "_other"))
        if merged.empty:
            continue
        diff = paired_bootstrap_diff(merged["ndcg_ref"], merged["ndcg_other"], cfg.bootstrap_samples,
                                     cfg.confidence, ctx.config.seed)
        rows.append({"reference": reference, "model": key, "dimension": dimension, "group": group,
                     "users": len(merged), "ndcg_difference": diff.mean, "ci_low": diff.low, "ci_high": diff.high})
    return pd.DataFrame(rows)


def _newest_releases(ctx: RunContext) -> np.ndarray:
    """Mask of the movies from the latest release year in the data (2000 for MovieLens 1M)."""
    years = ctx.data.movies.set_index("item_id")["year"].reindex(ctx.ids.items)
    return (years == years.max()).fillna(False).to_numpy(dtype=bool)


def recommendation_profile(ctx: RunContext, runs: dict[str, TestRun], train: TrainData, tail: np.ndarray) -> pd.DataFrame:
    """What each model puts in its lists: popularity of the movies, and the share of
    tail movies, of rarely rated movies, and of new releases among the tail slots."""
    counts = np.bincount(train.ids.item_index(train.ratings["item_id"]), minlength=train.n_items)
    newest = _newest_releases(ctx)
    rows = []
    for key, run in runs.items():
        recs = np.vstack([r.recommendations for r in run.results])
        tail_slots = recs[tail[recs]]
        rows.append({
            "model": key,
            "median_training_ratings_of_recommended_movies": float(np.median(counts[recs])),
            "share_of_slots_on_tail_movies": float(tail[recs].mean()),
            f"share_of_slots_on_movies_with_<{RARE_ITEM_RATINGS}_ratings": float((counts[recs] < RARE_ITEM_RATINGS).mean()),
            "share_of_tail_slots_on_newest_releases": float(newest[tail_slots].mean()) if tail_slots.size else 0.0,
        })
    return pd.DataFrame(rows)


def item_group_summary(ctx: RunContext, head: np.ndarray, tail: np.ndarray) -> pd.DataFrame:
    """How the relevant test ratings split between head and tail movies."""
    test = ctx.split.test
    relevant = test[test["rating"] >= ctx.config.evaluation.relevance_threshold]
    index = ctx.ids.item_index(relevant["item_id"])
    newest = _newest_releases(ctx)
    rows = []
    for group, members in (("head", head), ("tail", tail)):
        in_group = members[index]
        rows.append({"group": group, "movies": int(members.sum()), "relevant_test_ratings": int(in_group.sum()),
                     "share_of_relevant_test_ratings": float(in_group.mean()),
                     "share_of_them_on_newest_releases": float(newest[index[in_group]].mean()) if in_group.any() else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Sparsity ablation
# --------------------------------------------------------------------------

def sparsity(ctx: RunContext, s: Settings, choices: dict[str, Choice], runs: dict[str, TestRun],
             trial_rows: list, curve_rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on a fraction of the interactions; the evaluation task stays the same.

    The full history stays in the seen mask and the candidate set, so a model
    is never asked to recommend a movie the user already rated (Part 1 fixes
    that task). Each model is re-tuned at each level with an equal, smaller
    budget, then tested on every sampling seed.
    """
    split, seed = ctx.split, ctx.config.seed
    rows, frozen_rows = [], []
    for key, run in runs.items():
        per_user = run.per_user()
        rows.append({"model": key, "fraction": 1.0, "sampling_seed": None, "params": _as_text(run.choice.params),
                     "epochs": run.choice.epochs, "users": len(per_user), "coverage": run.coverage,
                     **{m: per_user[m].mean() for m in METRICS}, "per_user": per_user})
    for fraction in s.fractions:
        val_train = _train(ctx, subsample(split.train, fraction, s.sampling_seeds[0]), split.val_start,
                           history=split.train)
        test_trains = {sampling_seed: _train(ctx, subsample(split.train_val, fraction, sampling_seed),
                                             split.test_start, history=split.train_val)
                       for sampling_seed in s.sampling_seeds}
        for key in MODEL_KEYS:
            stage = f"sparsity {fraction:.0%}"
            choice = tune(ctx, s, key, retune_options(ctx, s, key, choices[key]), val_train, split.val, stage,
                          trial_rows, curve_rows)
            for sampling_seed, train in test_trains.items():
                model = make(ctx, key, choice.params, seed, epochs=choice.epochs).fit(train)
                result = _evaluate(ctx, model, train, split.test)
                rows.append({"model": key, "fraction": fraction, "sampling_seed": sampling_seed,
                             "params": _as_text(choice.params), "epochs": choice.epochs,
                             "users": len(result.per_user), "coverage": result.coverage,
                             **{m: result.mean(m) for m in METRICS}, "per_user": result.per_user})
            # The full-data setting without re-tuning, on the first sample only.
            first = test_trains[s.sampling_seeds[0]]
            fixed = make(ctx, key, choices[key].params, seed, epochs=choices[key].epochs).fit(first)
            frozen_rows.append({"model": key, "fraction": fraction, "ndcg": _evaluate(ctx, fixed, first, split.test).mean("ndcg"),
                                "retuned_ndcg": rows[-len(test_trains)]["ndcg"]})
            log.info("Sparsity %.0f%% %s: NDCG@10 %.4f", 100 * fraction, key,
                     np.mean([r["ndcg"] for r in rows[-len(test_trains):]]))
    return pd.DataFrame(rows), pd.DataFrame(frozen_rows)


def sparsity_summary(ctx: RunContext, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and spread per model and level, the ranking, and a paired test of the top two."""
    cfg = ctx.config.evaluation
    summary_rows, top_two_rows = [], []
    for fraction, level in detail.groupby("fraction", sort=False):
        per_model = {}
        for key, rows in level.groupby("model", sort=False):
            per_model[key] = _seed_mean(list(rows["per_user"]))
            summary_rows.append({
                "model": key, "fraction": fraction, "samples": len(rows),
                **{m: rows[m].mean() for m in METRICS}, "ndcg_sd_over_samples": rows["ndcg"].std(ddof=1) if len(rows) > 1 else 0.0,
                "coverage": rows["coverage"].mean(),
            })
        table = pd.DataFrame([r for r in summary_rows if r["fraction"] == fraction]).sort_values("ndcg", ascending=False)
        for rank, row in enumerate(table.itertuples(), 1):
            next(r for r in summary_rows if r["fraction"] == fraction and r["model"] == row.model)["rank"] = rank
        first, second = table["model"].iloc[0], table["model"].iloc[1]
        merged = per_model[first].merge(per_model[second], on="user_id", suffixes=("_a", "_b"))
        diff = paired_bootstrap_diff(merged["ndcg_a"], merged["ndcg_b"], cfg.bootstrap_samples, cfg.confidence,
                                     ctx.config.seed)
        _, p_value = wilcoxon_signed_rank(merged["ndcg_a"], merged["ndcg_b"])
        top_two_rows.append({"fraction": fraction, "first": first, "second": second, "ndcg_difference": diff.mean,
                             "ci_low": diff.low, "ci_high": diff.high, "wilcoxon_p": p_value})
    return pd.DataFrame(summary_rows), pd.DataFrame(top_two_rows)


# --------------------------------------------------------------------------
# Results rows and run()
# --------------------------------------------------------------------------

def _seeds_text(seeds: list[int]) -> str:
    return ", ".join(str(s) for s in seeds)


def model_rows(ctx: RunContext, runs: dict[str, TestRun], paired: pd.DataFrame, reference: str) -> None:
    cfg = ctx.config.evaluation
    for key, run in runs.items():
        per_user = run.per_user()
        ci = bootstrap_ci(per_user["ndcg"], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
        variability = f"NDCG@10 {int(cfg.confidence * 100)}% CI over users [{ci.low:.4f}, {ci.high:.4f}]"
        if len(run.seeds) > 1:
            values = run.seed_values()
            variability += f"; sd over {len(values)} seeds {values.std(ddof=1):.4f} (range {values.min():.4f}-{values.max():.4f})"
        if key == reference:
            variability += "; reference model for the paired tests"
        else:
            row = paired[paired["model"] == key].iloc[0]
            variability += (f"; {runs[reference].models[0].name} minus this model: {row.ndcg_difference:+.4f} "
                            f"[{row.ndcg_ci_low:+.4f}, {row.ndcg_ci_high:+.4f}], Holm-adjusted Wilcoxon p = {row.holm_p:.2g}")
        notes = (f"validation NDCG@10 {run.choice.validation_ndcg:.4f}, best of {run.choice.trials} trials; "
                 f"{len(per_user)} test users")
        if key in SELF_IMPLEMENTED:
            notes = "self-implemented (numpy, from Lecture 2.1 and the original paper); " + notes
        ctx.results.add(MODELS_SHEET, {
            "Model": run.models[0].name,
            "Paradigm": PARADIGMS[key],
            "Self-implemented?": "Yes" if key in SELF_IMPLEMENTED else "No",
            "Key hyperparameters": run.models[0].describe(),
            "Precision@10": run.mean("precision"),
            "Recall@10": run.mean("recall"),
            "NDCG@10": run.mean("ndcg"),
            "Catalogue coverage": run.coverage,
            "Variability / CI / test": variability,
            "Seed(s)": _seeds_text(run.seeds),
            "Notes": notes,
        })


def breakdown_rows(ctx: RunContext, runs: dict[str, TestRun], table: pd.DataFrame) -> None:
    for row in table.itertuples():
        run = runs[row.model]
        coverage_note = ("coverage = share of the group's movies recommended to anyone"
                         if row.dimension == "Item popularity" else "coverage = lists of this group's users")
        ctx.results.add(BREAKDOWN_SHEET, {
            "Model": run.models[0].name, "Breakdown dimension": row.dimension, "Group": row.group,
            "Definition / threshold": row.definition, "Precision@10": row.precision, "Recall@10": row.recall,
            "NDCG@10": row.ndcg, "Catalogue coverage": row.coverage, "Seed(s)": _seeds_text(run.seeds),
            "Notes": (f"{row.users} users; NDCG@10 95% CI [{row.ndcg_ci_low:.4f}, {row.ndcg_ci_high:.4f}]; "
                      f"{coverage_note}"),
        })


def sparsity_rows(ctx: RunContext, s: Settings, runs: dict[str, TestRun], summary: pd.DataFrame) -> None:
    for row in summary.itertuples():
        run = runs[row.model]
        if row.fraction == 1.0:
            seeds, note = _seeds_text(run.seeds), "full training data (same as the Part 2 - Models row)"
        else:
            seeds = f"sampling {_seeds_text(s.sampling_seeds)}; model {ctx.config.seed}"
            note = (f"mean of {row.samples} user-stratified samples (NDCG sd {row.ndcg_sd_over_samples:.4f}); "
                    f"re-tuned with {s.retune_trials} trials; full history kept out of the lists")
        ctx.results.add(SPARSITY_SHEET, {
            "Model": run.models[0].name, "Training interactions (%)": round(100 * row.fraction),
            "Precision@10": row.precision, "Recall@10": row.recall, "NDCG@10": row.ndcg,
            "Catalogue coverage": row.coverage, "Seed(s)": seeds,
            "Notes": f"rank {row.rank} of {len(MODEL_KEYS)} at this level; {note}",
        })


def mf_rating_error(ctx: RunContext, runs: dict[str, TestRun], train: TrainData) -> pd.DataFrame:
    """Rating error of the BiasedMF test models against two trivial predictors."""
    test = ctx.split.test
    users, items = ctx.ids.user_index(test["user_id"]), ctx.ids.item_index(test["item_id"])
    truth = test["rating"].to_numpy(dtype=float)
    mu = train.ratings["rating"].mean()
    item_mean = test["item_id"].map(train.ratings.groupby("item_id")["rating"].mean()).fillna(mu).to_numpy()
    rows = [{"predictor": "global mean", "test_rmse": float(np.sqrt(np.mean((mu - truth) ** 2)))},
            {"predictor": "item mean", "test_rmse": float(np.sqrt(np.mean((item_mean - truth) ** 2)))}]
    for seed, model in zip(runs["biased_mf"].seeds, runs["biased_mf"].models):
        pred = np.clip(model.predict(users, items), 1, 5)
        rows.append({"predictor": f"BiasedMF seed {seed}", "test_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                     "final_train_rmse": model.train_rmse[-1] if model.train_rmse else None})
    return pd.DataFrame(rows)


def run(ctx: RunContext) -> None:
    s = settings(ctx)
    split, tables = ctx.split, ctx.out.tables
    trial_rows: list[dict] = []
    curve_rows: list[dict] = []

    val_train = _train(ctx, split.train, split.val_start)
    choices = {key: tune(ctx, s, key, options_for(ctx, s, key), val_train, split.val, "full data",
                         trial_rows, curve_rows) for key in MODEL_KEYS}
    reference = max(MODEL_KEYS, key=lambda k: choices[k].validation_ndcg)
    log.info("Best model on validation: %s", reference)

    full_train = _train(ctx, split.train_val, split.test_start)
    runs = run_tests(ctx, s, choices, full_train)
    windows = popularity_windows(ctx, s, choices["most_popular"], val_train, full_train)

    paired = paired_tests(ctx, runs, reference)
    model_rows(ctx, runs, paired, reference)

    groups, extra = breakdowns(ctx, s, runs, full_train)
    breakdown_rows(ctx, runs, groups)
    group_differences = breakdown_differences(ctx, extra["per_user"], reference)
    profile = recommendation_profile(ctx, runs, full_train, extra["tail"])
    item_groups = item_group_summary(ctx, extra["head"], extra["tail"])

    detail, frozen = sparsity(ctx, s, choices, runs, trial_rows, curve_rows)
    summary, top_two = sparsity_summary(ctx, detail)
    sparsity_rows(ctx, s, runs, summary)

    save_table(pd.DataFrame(trial_rows), tables / "part2_validation_trials.csv")
    save_table(pd.DataFrame(curve_rows), tables / "part2_training_curves.csv")
    save_table(pd.DataFrame([{"model": k, "params": _as_text(c.params), "epochs": c.epochs,
                              "validation_ndcg@10": c.validation_ndcg, "trials": c.trials,
                              "reference": k == reference} for k, c in choices.items()]),
               tables / "part2_selected.csv")
    save_table(pd.DataFrame([{"model": k, "seed": seed, **r.summary()} for k, run in runs.items()
                             for seed, r in zip(run.seeds, run.results)]), tables / "part2_test_by_seed.csv")
    save_table(pd.concat([run.per_user().assign(model=k) for k, run in runs.items()]), tables / "part2_test_per_user.csv")
    save_table(windows, tables / "part2_popularity_windows.csv")
    save_table(paired, tables / "part2_paired_tests.csv")
    save_table(groups, tables / "part2_breakdowns.csv")
    save_table(group_differences, tables / "part2_breakdown_differences.csv")
    save_table(profile, tables / "part2_recommendation_profile.csv")
    save_table(item_groups, tables / "part2_item_groups.csv")
    save_table(mf_rating_error(ctx, runs, full_train), tables / "part2_biasedmf_rating_error.csv")
    save_table(detail.drop(columns="per_user"), tables / "part2_sparsity_detail.csv")
    save_table(summary, tables / "part2_sparsity_summary.csv")
    save_table(top_two, tables / "part2_sparsity_top_two.csv")
    save_table(frozen, tables / "part2_sparsity_frozen_settings.csv")

    figures, cfg = ctx.out.figures, ctx.config.evaluation
    names = {k: run.models[0].name for k, run in runs.items()}
    overall = []
    for key, run in runs.items():
        ci = bootstrap_ci(run.per_user()["ndcg"], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
        overall.append({"label": names[key], "ndcg": ci.mean, "ndcg_low": ci.low, "ndcg_high": ci.high,
                        "coverage": run.coverage})
    order = sorted(MODEL_KEYS, key=lambda k: runs[k].mean("ndcg"))   # best model drawn on top
    # In colour: the reference model, the leader at the sparsest level and the
    # Part 1 item-kNN baseline (three colours at most); the rest are grey.
    sparsest = summary[summary["fraction"] == summary["fraction"].min()]
    leader = sparsest.sort_values("ndcg", ascending=False)["model"].iloc[0]
    highlight = list(dict.fromkeys([reference, leader, "item_knn", "most_popular"]))[:3]
    plots.part2_overall(pd.DataFrame(overall), figures / "fig3_part2_models.png")
    plots.part2_breakdowns(groups, names, order, figures / "fig4_part2_breakdowns.png")
    plots.part2_sparsity(summary, names, highlight, figures / "fig5_part2_sparsity.png")
