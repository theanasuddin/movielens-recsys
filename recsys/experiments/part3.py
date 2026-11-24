"""Part 3, Track A: diversity and popularity bias (Lecture 5.1).

The best Part 2 model, EASE, retrieves each user's top N movies, and maximal
marginal relevance (recsys/reranking/mmr.py) re-ranks them into the top 10,
trading relevance against genre similarity with lambda. Steps, in the order
run() calls them:
1. retrieve the candidate pools on validation (EASE fitted on train) and on
   test (EASE refitted on train + validation), with the evaluator's rules;
2. sweep lambda and score every list on accuracy and on the four measures of
   slide 7 (intra-list diversity, coverage, Gini, calibration error), plus
   novelty, tail share and a collaborative item distance;
3. choose lambda on validation with a rule fixed before looking at test;
4. compare every test setting with plain EASE (lambda = 1) on paired
   intervals, and record how deep into the pool MMR reaches;
5. vary the pool size, the retrieval limit of slide 10;
6. break the cost down by user activity, taste breadth, genre and popularity.

The protocol is Part 1's: same split, relevance threshold, candidate set,
seen-movie rule and tie order.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from recsys.data import TrainData, genre_matrix, make_train_data
from recsys.data.movielens import GENRES
from recsys.evaluation import (
    bootstrap_ratio,
    bootstrap_system_diff,
    evaluate,
    evaluation_users,
    paired_bootstrap_diff,
    relevance_matrix,
    top_k,
)
from recsys.evaluation import metrics
from recsys.evaluation.diversity import (
    genre_shares,
    history_distribution,
    intra_list_distance,
    intra_list_diversity,
    item_cosine,
    jensen_shannon,
    jensen_shannon_parts,
    list_distribution,
    self_information,
    top_genre_gap,
)
from recsys.experiments.context import RunContext
from recsys.models import EASE
from recsys.reporting import plots
from recsys.reranking.mmr import mmr_rerank_batch
from recsys.utils import save_table

log = logging.getLogger(__name__)

SHEET = "Part 3A - Diversity"
DISTANCE = "1 - cosine of the 18 binary genre vectors"
BATCH = 256                       # lists re-ranked at once
PER_USER = ("precision", "recall", "ndcg", "ild", "ild_cf", "calibration", "novelty", "tail_share")


# --------------------------------------------------------------------------
# Settings and helpers
# --------------------------------------------------------------------------

@dataclass
class Settings:
    base: dict[str, Any]
    pool: int
    lambdas: list[float]
    pool_sizes: list[int]
    max_ndcg_drop: float
    activity_quantiles: list[float]
    head_share: float


def settings(ctx: RunContext) -> Settings:
    p3 = ctx.config.part3
    steps = int(round(1 / p3["lambda_step"]))
    values = {
        "base": dict(p3["base_model"]),
        "pool": p3["candidate_pool"],
        "lambdas": [round(i / steps, 4) for i in range(steps + 1)],
        "pool_sizes": list(p3["pool_sizes"]),
        "max_ndcg_drop": p3["max_relative_ndcg_drop"],
        # The user groups and the head/tail split are Part 2's.
        "activity_quantiles": list(ctx.config.part2["breakdowns"]["activity_quantiles"]),
        "head_share": ctx.config.part2["breakdowns"]["head_share"],
    }
    if ctx.quick:
        values.update(p3.get("quick", {}))
    s = Settings(**values)
    if 1.0 not in s.lambdas or s.pool not in s.pool_sizes:
        raise ValueError("part3 needs lambda = 1 (plain EASE) and the main pool size among the pool sizes")
    return s


def check_base_model(ctx: RunContext, s: Settings) -> None:
    """Part 3 starts from the best Part 2 model; compare with Part 2's choice when it is on disk."""
    path = ctx.out.tables / "part2_selected.csv"
    if not path.is_file():
        log.warning("No %s; taking the EASE setting from the config", path.name)
        return
    selected = pd.read_csv(path).set_index("model")
    best = selected.index[selected["reference"].astype(bool)].item()
    params = json.loads(selected.loc["ease", "params"])
    if best != "ease" or float(params["lam"]) != float(s.base["lam"]) or params["min_rating"] != s.base["min_rating"]:
        raise ValueError(f"part3.base_model {s.base} is not the best Part 2 model ({best}: {params})")


def _train(ctx: RunContext, ratings: pd.DataFrame, cutoff: int) -> TrainData:
    return make_train_data(ratings, ctx.ids, cutoff, ctx.config.filtering.min_item_train_ratings)


def head_tail(train: TrainData, head_share: float) -> np.ndarray:
    """Tail mask by Part 2's rule: the most-rated head_share of the candidates is the head."""
    counts = np.bincount(train.ids.item_index(train.ratings["item_id"]), minlength=train.n_items)
    candidates = np.flatnonzero(train.catalogue)
    n_head = int(np.ceil(head_share * len(candidates)))
    head = np.zeros(train.n_items, dtype=bool)
    head[candidates[np.argsort(-counts[candidates], kind="stable")[:n_head]]] = True
    return train.catalogue & ~head


def exposure_gini(lists: np.ndarray, catalogue: np.ndarray) -> float:
    """Gini of how often each candidate movie is recommended, unrecommended ones included."""
    return metrics.gini(np.bincount(lists.ravel(), minlength=len(catalogue))[catalogue])


# --------------------------------------------------------------------------
# 1. Candidate pools
# --------------------------------------------------------------------------

@dataclass
class Stage:
    """One evaluation period: the users, their EASE candidate pools and what
    the beyond-accuracy measures need from this period's training data."""
    name: str
    train: TrainData
    users: np.ndarray           # evaluated user indices, sorted
    items: np.ndarray           # (n_users, max pool) candidates, best first
    scores: np.ndarray          # EASE scores of the candidates
    relevant: np.ndarray        # (n_users, n_items) relevant held-out movies
    plain: np.ndarray           # EASE's own top-k from the evaluator
    history: np.ndarray         # (n_users, 18) genre distribution of the positive history
    tail: np.ndarray            # item mask
    popularity: np.ndarray      # share of training users who rated each movie

    @property
    def n_history(self) -> np.ndarray:
        return self.train.history_sizes()[self.users]


def build_stage(ctx: RunContext, s: Settings, name: str, ratings: pd.DataFrame, cutoff: int,
                held_out: pd.DataFrame, shares: np.ndarray) -> Stage:
    cfg = ctx.config
    train = _train(ctx, ratings, cutoff)
    model = EASE(lam=s.base["lam"], min_rating=s.base["min_rating"]).fit(train)
    users = evaluation_users(train, held_out, cfg.evaluation.relevance_threshold, cfg.filtering.min_user_history)
    size = min(max(s.pool_sizes), int(train.catalogue.sum()))
    tie_order = np.random.default_rng(cfg.seed).permutation(train.n_items)   # the evaluator's tie order
    items, scores = [], []
    for start in range(0, len(users), cfg.evaluation.batch_size):
        batch = users[start:start + cfg.evaluation.batch_size]
        batch_scores = np.array(model.score(batch), dtype=np.float64)
        batch_scores[:, ~train.catalogue] = -np.inf
        rows, cols = train.seen[batch].nonzero()
        batch_scores[rows, cols] = -np.inf
        top = top_k(batch_scores, size, tie_order)
        items.append(top)
        scores.append(np.take_along_axis(batch_scores, top, axis=1))
    plain = evaluate(model, train, held_out, cfg.evaluation, cfg.filtering.min_user_history, seed=cfg.seed)
    raters = np.asarray(train.matrix("binary").sum(axis=0)).ravel()
    return Stage(
        name=name, train=train, users=users, items=np.vstack(items), scores=np.vstack(scores),
        relevant=relevance_matrix(held_out, train.ids, cfg.evaluation.relevance_threshold)[users].toarray() > 0,
        plain=plain.recommendations,
        history=history_distribution(train.matrix("binary", cfg.evaluation.relevance_threshold), users, shares),
        tail=head_tail(train, s.head_share),
        popularity=raters / max(1, int((train.history_sizes() > 0).sum())),
    )


# --------------------------------------------------------------------------
# 2. Re-ranking and scoring
# --------------------------------------------------------------------------

def rerank(stage: Stage, genres: np.ndarray, pool: int, lambda_: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    """MMR over each user's top `pool`; returns the lists and the pool positions they came from."""
    positions = np.vstack([
        mmr_rerank_batch(stage.scores[start:start + BATCH, :pool], genres[stage.items[start:start + BATCH, :pool]],
                         k, lambda_)
        for start in range(0, len(stage.users), BATCH)
    ])
    return np.take_along_axis(stage.items[:, :pool], positions, axis=1), positions


def score_lists(stage: Stage, lists: np.ndarray, positions: np.ndarray, genres: np.ndarray, shares: np.ndarray,
                cf: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-user and whole-system measures of one set of lists."""
    hits = np.take_along_axis(stage.relevant, lists, axis=1)
    n_relevant = stage.relevant.sum(axis=1)
    distribution = list_distribution(lists, shares)
    valid = ~np.isnan(stage.history).any(axis=1)
    calibration, under, over = (np.full(len(lists), np.nan) for _ in range(3))
    calibration[valid] = jensen_shannon(stage.history[valid], distribution[valid])
    under[valid], over[valid] = jensen_shannon_parts(stage.history[valid], distribution[valid])
    per_user = pd.DataFrame({
        "user_id": stage.train.ids.users[stage.users].to_numpy(),
        "n_history": stage.n_history,
        "n_relevant": n_relevant,
        "precision": metrics.precision_at_k(hits),
        "recall": metrics.recall_at_k(hits, n_relevant),
        "ndcg": metrics.ndcg_at_k(hits, n_relevant),
        "ild": intra_list_diversity(lists, genres),
        "ild_cf": intra_list_distance(lists, *cf) if cf is not None else np.nan,
        "calibration": calibration,
        "calibration_under": under,
        "calibration_over": over,
        "top_genre_gap": top_genre_gap(stage.history, distribution),
        "novelty": self_information(lists, stage.popularity),
        "tail_share": stage.tail[lists].mean(axis=1),
        "depth": positions.mean(axis=1) + 1,
    })
    catalogue = stage.train.catalogue
    system = {
        "coverage": metrics.catalogue_coverage(lists, int(catalogue.sum())),
        "gini": exposure_gini(lists, catalogue),
        "distinct_movies": int(np.unique(lists).size),
        "depth_p90": float(np.quantile(positions + 1, 0.9)),
    }
    return per_user, system


@dataclass
class Setting:
    lambda_: float
    pool: int
    lists: np.ndarray
    per_user: pd.DataFrame
    system: dict[str, float]
    seconds: float

    def summary(self, stage: str) -> dict[str, Any]:
        means = self.per_user[[c for c in PER_USER if c in self.per_user]].mean()
        extra = self.per_user[["calibration_under", "calibration_over", "top_genre_gap"]].mean()
        return {"stage": stage, "pool": self.pool, "lambda": self.lambda_, "users": len(self.per_user),
                **means.to_dict(), **extra.to_dict(),
                "depth_mean": self.per_user["depth"].mean(), **self.system,
                "ms_per_list": 1000 * self.seconds / len(self.per_user)}


def sweep(ctx: RunContext, s: Settings, stage: Stage, genres: np.ndarray, shares: np.ndarray, pool: int,
          cf: tuple[np.ndarray, np.ndarray] | None = None) -> dict[float, Setting]:
    k = ctx.config.evaluation.k
    out = {}
    for lambda_ in s.lambdas:
        start = time.perf_counter()
        lists, positions = rerank(stage, genres, pool, lambda_, k)
        seconds = time.perf_counter() - start
        per_user, system = score_lists(stage, lists, positions, genres, shares, cf)
        out[lambda_] = Setting(lambda_, pool, lists, per_user, system, seconds)
    return out


def collaborative_lookup(stage: Stage, pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Item cosine over who rated what, for every movie in this stage's pools."""
    return item_cosine(stage.train.matrix("binary"), np.unique(stage.items[:, :pool]))


# --------------------------------------------------------------------------
# 3-4. Choosing lambda and comparing with plain EASE
# --------------------------------------------------------------------------

def choose_lambda(summary: pd.DataFrame, max_drop: float) -> float:
    """The highest validation ILD whose NDCG@10 is within max_drop (relative) of
    plain EASE; ties go to the larger lambda, the smaller change."""
    base = summary.loc[summary["lambda"] == 1.0, "ndcg"].item()
    eligible = summary[summary["ndcg"] >= (1 - max_drop) * base]
    return float(eligible.sort_values(["ild", "lambda"], ascending=[False, False])["lambda"].iloc[0])


def differences(ctx: RunContext, runs: dict[float, Setting], catalogue: np.ndarray, tail: np.ndarray) -> pd.DataFrame:
    """Every setting minus plain EASE, with 95% intervals: paired bootstrap over
    users for the per-user measures, resampled users for the system measures."""
    cfg = ctx.config.evaluation
    base = runs[1.0]
    system_measures = {
        "coverage": lambda lists: metrics.catalogue_coverage(lists, int(catalogue.sum())),
        "gini": lambda lists: exposure_gini(lists, catalogue),
        "tail_slots": lambda lists: float(tail[lists].mean()),
    }
    rows = []
    for lambda_, run in sorted(runs.items(), reverse=True):
        row: dict[str, Any] = {"lambda": lambda_}
        for metric in PER_USER:
            a, b = run.per_user[metric].to_numpy(), base.per_user[metric].to_numpy()
            keep = ~(np.isnan(a) | np.isnan(b))
            if not keep.any():
                continue
            d = paired_bootstrap_diff(a[keep], b[keep], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
            row.update({f"{metric}_difference": d.mean, f"{metric}_ci_low": d.low, f"{metric}_ci_high": d.high})
        for name, statistic in system_measures.items():
            d = bootstrap_system_diff(run.lists, base.lists, statistic, cfg.bootstrap_samples, cfg.confidence,
                                      ctx.config.seed)
            row.update({f"{name}_difference": d.mean, f"{name}_ci_low": d.low, f"{name}_ci_high": d.high})
        delta = run.per_user["ndcg"].to_numpy() - base.per_user["ndcg"].to_numpy()
        row.update({"users_better": int((delta > 1e-12).sum()), "users_worse": int((delta < -1e-12).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Who pays
# --------------------------------------------------------------------------

def user_groups(stage: Stage, s: Settings) -> dict[tuple[str, str], tuple[np.ndarray, str]]:
    """Part 2's activity thirds, and thirds of taste breadth (entropy of the
    genre mix of the positive history, in bits)."""
    n_history = stage.n_history
    low, high = np.quantile(n_history, s.activity_quantiles)
    groups = {
        ("User activity", "light"): (n_history <= low, f"<= {low:.0f} training ratings"),
        ("User activity", "medium"): ((n_history > low) & (n_history <= high), f"{low:.0f}-{high:.0f} training ratings"),
        ("User activity", "heavy"): (n_history > high, f"> {high:.0f} training ratings"),
    }
    p = stage.history
    valid = ~np.isnan(p).any(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    entropy[~valid] = np.nan
    a, b = np.nanquantile(entropy, s.activity_quantiles)
    groups.update({
        ("Taste breadth", "narrow"): (valid & (entropy <= a), f"genre entropy <= {a:.2f} bits"),
        ("Taste breadth", "medium"): (valid & (entropy > a) & (entropy <= b), f"{a:.2f}-{b:.2f} bits"),
        ("Taste breadth", "broad"): (valid & (entropy > b), f"> {b:.2f} bits"),
    })
    return groups


def swaps(ease: np.ndarray, lists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks over the slots: EASE movies MMR dropped, and movies MMR brought in."""
    removed = ~(ease[:, :, None] == lists[:, None, :]).any(axis=2)
    added = ~(lists[:, :, None] == ease[:, None, :]).any(axis=2)
    return removed, added


def who_pays(ctx: RunContext, s: Settings, stage: Stage, runs: dict[float, Setting], lambdas: list[float]) -> pd.DataFrame:
    """Per user group: the change in accuracy, diversity and calibration, and
    how often the movies MMR dropped, and the ones it brought in, were relevant."""
    cfg = ctx.config.evaluation
    base = runs[1.0].per_user
    ease_hits = np.take_along_axis(stage.relevant, runs[1.0].lists, axis=1)
    rows = []
    for lambda_ in lambdas:
        run = runs[lambda_].per_user
        removed, added = swaps(runs[1.0].lists, runs[lambda_].lists)
        hits = np.take_along_axis(stage.relevant, runs[lambda_].lists, axis=1)
        for (dimension, group), (mask, definition) in user_groups(stage, s).items():
            row: dict[str, Any] = {"lambda": lambda_, "dimension": dimension, "group": group,
                                   "definition": definition, "users": int(mask.sum()),
                                   "ease_ndcg": base["ndcg"][mask].mean(), "mmr_ndcg": run["ndcg"][mask].mean()}
            for metric in ("ndcg", "ild", "calibration"):
                a, b = run[metric].to_numpy()[mask], base[metric].to_numpy()[mask]
                keep = ~(np.isnan(a) | np.isnan(b))
                if not keep.any():
                    row.update({f"{metric}_difference": np.nan, f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan})
                    continue
                d = paired_bootstrap_diff(a[keep], b[keep], cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
                row.update({f"{metric}_difference": d.mean, f"{metric}_ci_low": d.low, f"{metric}_ci_high": d.high})
            delta = run["ndcg"].to_numpy()[mask] - base["ndcg"].to_numpy()[mask]
            row.update({"users_better": int((delta > 1e-12).sum()), "users_worse": int((delta < -1e-12).sum()),
                        "swapped_per_user": float(removed[mask].sum(axis=1).mean())})
            # How often the dropped and the brought-in movies were relevant, users resampled.
            for name, slots, relevant in (("removed", removed, ease_hits), ("added", added, hits)):
                n_slots, n_hits = slots[mask].sum(axis=1), (slots & relevant)[mask].sum(axis=1)
                if n_slots.sum() == 0:
                    row.update({f"{name}_hit_rate": np.nan, f"{name}_ci_low": np.nan, f"{name}_ci_high": np.nan})
                    continue
                rate = bootstrap_ratio(n_hits, n_slots, cfg.bootstrap_samples, cfg.confidence, ctx.config.seed)
                row.update({f"{name}_hit_rate": rate.mean, f"{name}_ci_low": rate.low, f"{name}_ci_high": rate.high})
            rows.append(row)
    return pd.DataFrame(rows)


def added_movies(ctx: RunContext, stage: Stage, runs: dict[float, Setting], lambda_: float, top: int = 15) -> pd.DataFrame:
    """The movies MMR brings into the most lists at one lambda."""
    ids = stage.train.ids
    _, added = swaps(runs[1.0].lists, runs[lambda_].lists)
    counts = np.bincount(runs[lambda_].lists[added], minlength=ids.n_items)
    order = np.argsort(-counts, kind="stable")[:top]
    order = order[counts[order] > 0]
    movies = ctx.data.movies.set_index("item_id").reindex(ids.items[order])
    ratings = np.bincount(ids.item_index(stage.train.ratings["item_id"]), minlength=ids.n_items)
    return pd.DataFrame({"lambda": lambda_, "movie_id": ids.items[order], "title": movies["title"].to_numpy(),
                         "genres": movies["genres"].to_numpy(), "lists_added_to": counts[order],
                         "training_ratings": ratings[order], "tail": stage.tail[order]})


def genre_exposure(stage: Stage, runs: dict[float, Setting], lambdas: list[float], shares: np.ndarray,
                   held_out: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Share of the recommendation slots each genre gets, next to its share of
    the catalogue and of the relevant test ratings."""
    ids = stage.train.ids
    relevant = held_out[held_out["rating"] >= threshold]
    table = pd.DataFrame({
        "genre": GENRES,
        "catalogue_share": shares[stage.train.catalogue].mean(axis=0),
        "relevant_test_share": shares[ids.item_index(relevant["item_id"])].mean(axis=0),
    })
    for lambda_ in lambdas:
        label = "ease" if lambda_ == 1.0 else f"mmr_{lambda_:g}"
        table[f"{label}_slot_share"] = shares[runs[lambda_].lists].reshape(-1, shares.shape[1]).mean(axis=0)
    return table


def exposure_summary(stage: Stage, runs: dict[float, Setting], lambdas: list[float]) -> pd.DataFrame:
    """How exposure spreads over movies, head and tail."""
    catalogue = stage.train.catalogue
    rows = []
    for lambda_ in lambdas:
        lists = runs[lambda_].lists
        counts = np.bincount(lists.ravel(), minlength=len(catalogue))
        tail_counts = counts[stage.tail]
        _, added = swaps(runs[1.0].lists, lists)
        added_counts = np.bincount(lists[added], minlength=len(catalogue))
        rows.append({
            "lambda": lambda_,
            "distinct_movies": int((counts > 0).sum()),
            "coverage": float((counts[catalogue] > 0).mean()),
            "gini": exposure_gini(lists, catalogue),
            "tail_slot_share": float(tail_counts.sum() / counts.sum()),
            "distinct_tail_movies": int((tail_counts > 0).sum()),
            "top10_movies_share_of_slots": float(np.sort(counts)[-10:].sum() / counts.sum()),
            "top10_tail_movies_share_of_tail_slots": float(np.sort(tail_counts)[-10:].sum() / max(1, tail_counts.sum())),
            # Slots MMR changed, and how many different movies filled them.
            "changed_slot_share": float(added.mean()),
            "distinct_added_movies": int((added_counts > 0).sum()),
            "top10_added_movies_share_of_added_slots": float(np.sort(added_counts)[-10:].sum() / max(1, added.sum())),
        })
    return pd.DataFrame(rows)


def recommendation_lists(ctx: RunContext, stage: Stage, runs: dict[float, Setting], lambdas: list[float]) -> pd.DataFrame:
    """The lists themselves, with titles, so changes can be read movie by movie."""
    ids = stage.train.ids
    titles = ctx.data.movies.set_index("item_id")["title"].reindex(ids.items).to_numpy()
    frames = []
    for lambda_ in lambdas:
        lists = runs[lambda_].lists
        n_users, k = lists.shape
        flat = lists.ravel()
        frames.append(pd.DataFrame({
            "lambda": lambda_,
            "user_id": np.repeat(runs[lambda_].per_user["user_id"].to_numpy(), k),
            "rank": np.tile(np.arange(1, k + 1), n_users),
            "movie_id": ids.items[flat],
            "title": titles[flat],
            "tail": stage.tail[flat],
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Results workbook
# --------------------------------------------------------------------------

def _ci(diff: pd.Series, metric: str, digits: int = 4) -> str:
    return (f"{diff[f'{metric}_difference']:+.{digits}f} "
            f"[{diff[f'{metric}_ci_low']:+.{digits}f}, {diff[f'{metric}_ci_high']:+.{digits}f}]")


def sheet_rows(ctx: RunContext, s: Settings, main: pd.DataFrame, diffs: pd.DataFrame, pools: pd.DataFrame,
               chosen: float) -> None:
    base = f"EASE (lambda={s.base['lam']:g}, ratings >= {s.base['min_rating']})"
    d = diffs.set_index("lambda")

    def add(row: pd.Series, method: str, note: str) -> None:
        ctx.results.add(SHEET, {
            "Base model": base, "Re-ranking method": method, "Trade-off parameter": row["lambda"],
            "Item distance": DISTANCE, "Precision@10": row["precision"], "Recall@10": row["recall"],
            "NDCG@10": row["ndcg"], "Intra-list diversity": row["ild"], "Catalogue coverage": row["coverage"],
            "Gini": row["gini"], "Calibration error": row["calibration"], "Notes": note,
        })

    for _, row in main.sort_values("lambda", ascending=False).iterrows():
        lambda_ = row["lambda"]
        note = (f"{row['users']} test users; top {s.pool} candidates; novelty {row['novelty']:.2f} bits; "
                f"tail slots {row['tail_share']:.1%}")
        if lambda_ != 1.0:
            diff = d.loc[lambda_]
            note += (f"; vs plain EASE: NDCG@10 {_ci(diff, 'ndcg')}, ILD {_ci(diff, 'ild')}, "
                     f"coverage {_ci(diff, 'coverage')}, Gini {_ci(diff, 'gini')}, "
                     f"calibration {_ci(diff, 'calibration')}")
        if lambda_ == chosen:
            note += "; chosen on validation"
        add(row, "none (plain EASE)" if lambda_ == 1.0 else f"MMR over the top {s.pool}", note)
    shown = {chosen, 0.5, 0.0} & set(s.lambdas)
    ablation = pools[(pools["pool"] != s.pool) & pools["lambda"].isin(shown)]
    for _, row in ablation.sort_values(["pool", "lambda"], ascending=[True, False]).iterrows():
        add(row, f"MMR over the top {row['pool']} (pool-size ablation)",
            f"{row['users']} test users; test only; tail slots {row['tail_share']:.1%}")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(ctx: RunContext) -> None:
    s = settings(ctx)
    check_base_model(ctx, s)
    split, cfg, tables = ctx.split, ctx.config, ctx.out.tables
    k = cfg.evaluation.k
    genres = genre_matrix(ctx.data.movies).reindex(ctx.ids.items).to_numpy(dtype=np.float64)
    shares = genre_shares(genres)

    val = build_stage(ctx, s, "validation", split.train, split.val_start, split.val, shares)
    test = build_stage(ctx, s, "test", split.train_val, split.test_start, split.test, shares)

    # 2-3. The lambda sweep on the main pool, and the choice on validation.
    runs = {}
    for stage in (val, test):
        runs[stage.name] = sweep(ctx, s, stage, genres, shares, s.pool, collaborative_lookup(stage, s.pool))
        # Lambda = 1 must give back EASE's own top-k, tie order included.
        np.testing.assert_array_equal(runs[stage.name][1.0].lists, stage.plain)
    log.info("Check passed: lambda = 1 reproduces EASE's top-%d on validation and test", k)
    main = pd.DataFrame([run.summary(stage) for stage, by_lambda in runs.items() for run in by_lambda.values()])
    chosen = choose_lambda(main[main["stage"] == "validation"], s.max_ndcg_drop)
    main["chosen_on_validation"] = main["lambda"] == chosen
    log.info("Chose lambda %.2f on validation (largest ILD within %.0f%% of plain EASE's NDCG@10)",
             chosen, 100 * s.max_ndcg_drop)

    # 4. Every test setting against plain EASE.
    diffs = differences(ctx, runs["test"], test.train.catalogue, test.tail)

    # 5. The pool-size ablation (test only: it explains, it chooses nothing).
    pool_runs = {pool: sweep(ctx, s, test, genres, shares, pool) for pool in s.pool_sizes if pool != s.pool}
    pool_runs[s.pool] = runs["test"]
    smallest = min(s.pool_sizes)
    if smallest <= k:
        # A pool of k movies leaves MMR nothing to choose, only an order.
        for run in pool_runs[smallest].values():
            np.testing.assert_allclose(run.per_user["ild"], runs["test"][1.0].per_user["ild"])
    pools = pd.DataFrame([run.summary("test") for pool in sorted(pool_runs) for run in pool_runs[pool].values()])

    # 6. Who pays, at the chosen lambda and at 0.5, where the frontier turns.
    contrast = sorted({chosen, 0.5} & set(s.lambdas) - {1.0}, reverse=True)
    groups = who_pays(ctx, s, test, runs["test"], contrast)
    shown = [1.0] + contrast
    genre_table = genre_exposure(test, runs["test"], shown + ([0.0] if 0.0 not in shown else []), shares,
                                 split.test, cfg.evaluation.relevance_threshold)
    exposure = exposure_summary(test, runs["test"], shown + ([0.0] if 0.0 not in shown else []))

    # Per-user values for the settings the report discusses; part3_differences holds every lambda.
    kept = sorted({1.0, chosen, 0.5, 0.0} & set(s.lambdas), reverse=True)
    per_user = pd.concat([runs["test"][lambda_].per_user.assign(**{"lambda": lambda_}) for lambda_ in kept],
                         ignore_index=True)
    save_table(main, tables / "part3_sweep.csv")
    save_table(diffs, tables / "part3_differences.csv")
    save_table(pools, tables / "part3_pool_sizes.csv")
    save_table(groups, tables / "part3_who_pays.csv")
    save_table(genre_table, tables / "part3_genre_exposure.csv")
    save_table(exposure, tables / "part3_exposure.csv")
    save_table(per_user.drop(columns=["n_relevant"]), tables / "part3_test_per_user.csv", float_format="%.6g")
    save_table(recommendation_lists(ctx, test, runs["test"], sorted({1.0, chosen}, reverse=True)),
               tables / "part3_recommendations.csv")
    save_table(pd.concat([added_movies(ctx, test, runs["test"], lambda_) for lambda_ in contrast], ignore_index=True),
               tables / "part3_added_movies.csv")

    test_main = main[main["stage"] == "test"]
    plots.part3_frontier(test_main, diffs, pools, chosen, ctx.out.figures / "fig6_part3_frontier.png")
    sheet_rows(ctx, s, test_main, diffs, pools, chosen)
    log.info("Part 3: test NDCG@10 %.4f -> %.4f, ILD %.3f -> %.3f at lambda %.2f", runs["test"][1.0].per_user["ndcg"].mean(),
             runs["test"][chosen].per_user["ndcg"].mean(), runs["test"][1.0].per_user["ild"].mean(),
             runs["test"][chosen].per_user["ild"].mean(), chosen)
