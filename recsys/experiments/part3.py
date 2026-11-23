
"""Part 3 Track A: EASE Top-100 candidate reranking with genre-based MMR.

This runner intentionally reuses the existing temporal split, EASE model,
internal ID map, top-k tie order, and course results workbook schema.

Validation: EASE trained only on train; sweep MMR lambda on validation.
Test: EASE trained on train+validation; report lambda sweep on test.
The recommended operating point is chosen *on validation only*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from recsys.experiments.part3_analysis import (
    create_part3_analysis,
    analyze_popularity_exposure,
)


import numpy as np
import pandas as pd

from recsys.data import genre_matrix, make_train_data
from recsys.evaluation import (
    EvaluationResult,
    evaluation_users,
    paired_bootstrap_diff,
    relevance_matrix,
    top_k,
)
from recsys.evaluation import metrics
from recsys.evaluation.diversity import genre_calibration_error, intra_list_diversity
from recsys.experiments.context import RunContext
from recsys.models import EASE
from recsys.reporting import plots
import matplotlib.pyplot as plt  # plots selects the Agg backend before pyplot loads
from recsys.reranking.mmr import mmr_rerank
from recsys.utils import save_table

log = logging.getLogger(__name__)
SHEET = "Part 3A - Diversity"


@dataclass
class RerankRun:
    result: EvaluationResult
    ild: np.ndarray                  # ILD per evaluated user
    calibration: np.ndarray          # JSD per user; nan if no positive history
    gini: float
    tail_slot_share: float
    candidate_pool: int

    def summary(self, lambda_: float) -> dict:
        result = self.result
        finite = np.isfinite(self.calibration)
        return {
            "lambda": float(lambda_),
            "n_users": len(result.per_user),
            "precision": result.mean("precision"),
            "recall": result.mean("recall"),
            "ndcg": result.mean("ndcg"),
            "ild": float(self.ild.mean()),
            "coverage": result.coverage,
            "gini": self.gini,
            "calibration": float(self.calibration[finite].mean()) if finite.any() else np.nan,
            "calibration_users": int(finite.sum()),
            "tail_slot_share": self.tail_slot_share,
            "candidate_pool": self.candidate_pool,
        }


def rerank_all(
    model: EASE,
    train,
    held_out: pd.DataFrame,
    evaluation_config,
    genres: np.ndarray,
    lambdas: list[float],
    pool_size: int,
    seed: int,
    min_history: int,
    head_share: float = 0.10,
) -> dict[float, RerankRun]:
    """Evaluate multiple MMR weights on identical users and full-catalogue ground truth."""
    ids = train.ids
    k = evaluation_config.k
    if pool_size < k:
        raise ValueError(f"candidate_pool ({pool_size}) must be >= evaluation k ({k})")
    if genres.shape[0] != ids.n_items:
        raise ValueError("genre vectors must align with train.ids.items")
    if len(lambdas) != len(set(lambdas)) or not lambdas or any(not (0 <= lam <= 1) for lam in lambdas):
        raise ValueError("lambdas must be distinct values between 0 and 1")

    users = evaluation_users(train, held_out, evaluation_config.relevance_threshold, min_history)
    if not len(users):
        raise ValueError("no evaluable users for this time split")
    # Same tie order used in the Part 2 evaluator.
    tie_order = np.random.default_rng(seed).permutation(ids.n_items)
    relevant = relevance_matrix(held_out, ids, evaluation_config.relevance_threshold)
    n_relevant = np.asarray(relevant[users].sum(axis=1)).ravel().astype(int)
    history_size = train.history_sizes()[users]

    catalogue_size = int(train.catalogue.sum())
    candidate_pool = min(pool_size, catalogue_size)
    selected: dict[float, list[np.ndarray]] = {float(lam): [] for lam in lambdas}

    for start in range(0, len(users), evaluation_config.batch_size):
        batch = users[start:start + evaluation_config.batch_size]
        scores = np.asarray(model.score(batch), dtype=np.float64)
        # EXACTLY the existing evaluator's eligibility rules.
        scores[:, ~train.catalogue] = -np.inf
        seen_rows, seen_cols = train.seen[batch].nonzero()
        scores[seen_rows, seen_cols] = -np.inf
        candidates = top_k(scores, candidate_pool, tie_order)

        for local_row, row in enumerate(candidates):
            # top_k can include -inf placeholders if a user has very few
            # eligible movies; do not accidentally recommend those movies.
            row = row[np.isfinite(scores[local_row, row])]
            if len(row) < k:
                raise ValueError(
                    f"User {int(ids.users[batch[local_row]])} has fewer than {k} eligible movies"
                )
            row_scores = scores[local_row, row]
            row_genres = genres[row]
            for lam in lambdas:
                selected[float(lam)].append(
                    mmr_rerank(row, row_scores, row_genres, k=k, lambda_=float(lam))
                )

    positive_history = train.matrix("binary", evaluation_config.relevance_threshold)
    # Head/tail grouping based only on ratings available before evaluation.
    train_counts = np.bincount(ids.item_index(train.ratings["item_id"]), minlength=ids.n_items)
    ranked_items = np.argsort(-train_counts[train.catalogue], kind="stable")
    catalogue_items = np.flatnonzero(train.catalogue)
    n_head = max(1, int(round(head_share * catalogue_size)))
    head_items = catalogue_items[ranked_items[:n_head]]
    tail_mask = train.catalogue.copy()
    tail_mask[head_items] = False

    output: dict[float, RerankRun] = {}
    for lam, lists in selected.items():
        recs = np.stack(lists)
        # Ground truth is the full held-out set, not just the Top-100 pool.
        rel_for_users = relevant[users].toarray() > 0
        hits = np.take_along_axis(rel_for_users, recs, axis=1)
        per_user = pd.DataFrame({
            "user_id": ids.users[users].to_numpy(),
            "n_history": history_size,
            "n_relevant": n_relevant,
            "precision": metrics.precision_at_k(hits),
            "recall": metrics.recall_at_k(hits, n_relevant),
            "ndcg": metrics.ndcg_at_k(hits, n_relevant),
        })
        result = EvaluationResult(per_user, recs, catalogue_size, k)
        per_user_ild = intra_list_diversity(recs, genres)
        per_user_calibration = genre_calibration_error(recs, genres, positive_history, users)
        exposure = np.bincount(recs.ravel(), minlength=ids.n_items)[train.catalogue]
        output[lam] = RerankRun(
            result=result,
            ild=per_user_ild,
            calibration=per_user_calibration,
            gini=metrics.gini(exposure),
            tail_slot_share=float(tail_mask[recs].mean()),
            candidate_pool=candidate_pool,
        )
    return output


def _plot_frontier(summary: pd.DataFrame, path) -> None:
    """Display the lambda sweep, without claiming every point is Pareto-optimal."""
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(plots.FULL_WIDTH * 0.63, 3.0))
    summary = summary.sort_values("ild")
    ax.plot(summary["ild"], summary["ndcg"], marker="o", color=plots.SERIES[0])
    for _, row in summary.iterrows():
        ax.annotate(f"{row['lambda']:.2g}", (row['ild'], row['ndcg']),
                    xytext=(3, 5), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Intra-list diversity (1 - genre cosine)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("EASE + MMR: accuracy–diversity trade-off (test)")
    plots.save(fig, path)


def _choose_lambda(validation: pd.DataFrame, max_relative_ndcg_drop: float) -> float:
    """Choose maximum validation ILD within a preset NDCG-loss tolerance."""
    baseline = float(validation.loc[validation["lambda"] == 1.0, "ndcg"].iloc[0])
    eligible = validation[validation["ndcg"] >= (1 - max_relative_ndcg_drop) * baseline]
    # On ties choose the larger lambda (conservative toward relevance).
    picked = eligible.sort_values(["ild", "lambda"], ascending=[False, False]).iloc[0]
    return float(picked["lambda"])


def _summary_rows(runs: dict[float, RerankRun]) -> pd.DataFrame:
    return pd.DataFrame([run.summary(lam) for lam, run in sorted(runs.items(), reverse=True)])


def run(ctx: RunContext) -> None:
    """Entry point invoked by run_all.py --parts 3."""
    options = ctx.config.part3
    pool_size = int(options.get("candidate_pool", 100))
    ease_lam = float(options.get("ease_lam", 5000))
    ease_min_rating = int(options.get("ease_min_rating", 4))
    lambdas = [float(x) for x in options.get("lambdas", [1.0, .9, .75, .5, .25, .1, 0.0])]
    if 1.0 not in lambdas:
        raise ValueError("part3.lambdas must contain 1.0 for the EASE control")
    tolerance = float(options.get("max_relative_ndcg_drop", 0.02))
    if not 0 <= tolerance <= 1:
        raise ValueError("max_relative_ndcg_drop must lie in [0, 1]")

    cfg = ctx.config
    genres = genre_matrix(ctx.data.movies).reindex(ctx.ids.items).to_numpy(dtype=np.float64)

    def train_data(ratings, cutoff):
        return make_train_data(ratings, ctx.ids, cutoff, cfg.filtering.min_item_train_ratings)

    # Validation: no use of validation/test interactions in EASE training.
    train_val_cut = train_data(ctx.split.train, ctx.split.val_start)
    val_ease = EASE(lam=ease_lam, min_rating=ease_min_rating).fit(train_val_cut)
    val_runs = rerank_all(val_ease, train_val_cut, ctx.split.val, cfg.evaluation, genres,
                          lambdas, pool_size, cfg.seed, cfg.filtering.min_user_history)
    val_summary = _summary_rows(val_runs)
    selected_lambda = _choose_lambda(val_summary, tolerance)
    log.info("Selected lambda %.2f on validation (max relative NDCG loss %.1f%%)",
             selected_lambda, 100 * tolerance)

    # Test: fit EASE again on train+validation, then evaluate once on test.
    train_test_cut = train_data(ctx.split.train_val, ctx.split.test_start)
    test_ease = EASE(lam=ease_lam, min_rating=ease_min_rating).fit(train_test_cut)
    test_runs = rerank_all(test_ease, train_test_cut, ctx.split.test, cfg.evaluation, genres,
                           lambdas, pool_size, cfg.seed, cfg.filtering.min_user_history)
    test_summary = _summary_rows(test_runs)
    test_summary["chosen_on_validation"] = test_summary["lambda"].eq(selected_lambda)

    # Lambda=1 MUST reproduce EASE's unmodified top-10 rankings.
    from recsys.evaluation import evaluate
    control = evaluate(test_ease, train_test_cut, ctx.split.test, cfg.evaluation,
                       cfg.filtering.min_user_history, seed=cfg.seed)
    np.testing.assert_array_equal(control.recommendations, test_runs[1.0].result.recommendations)
    log.info("Check passed: MMR lambda=1.0 exactly matches original EASE Top-%d", cfg.evaluation.k)

    # Confidence intervals for accuracy/diversity changes relative to EASE.
    comparison_rows = []
    base = test_runs[1.0]
    for lam, run in sorted(test_runs.items(), reverse=True):
        ndcg_diff = paired_bootstrap_diff(
            run.result.per_user["ndcg"].to_numpy(), base.result.per_user["ndcg"].to_numpy(),
            cfg.evaluation.bootstrap_samples, cfg.evaluation.confidence, cfg.seed)
        ild_diff = paired_bootstrap_diff(
            run.ild, base.ild, cfg.evaluation.bootstrap_samples, cfg.evaluation.confidence, cfg.seed)
        comparison_rows.append({"lambda": lam, "delta_ndcg": ndcg_diff.mean,
                                "ndcg_ci_low": ndcg_diff.low, "ndcg_ci_high": ndcg_diff.high,
                                "delta_ild": ild_diff.mean,
                                "ild_ci_low": ild_diff.low, "ild_ci_high": ild_diff.high})
    changes = pd.DataFrame(comparison_rows)
    test_summary = test_summary.merge(changes, on="lambda")

    save_table(val_summary, ctx.out.tables / "part3_validation.csv")
    save_table(test_summary, ctx.out.tables / "part3_test.csv")
    save_table(pd.concat([
        run.result.per_user.assign(lambda_=lam, ild=run.ild, calibration=run.calibration)
        for lam, run in sorted(test_runs.items(), reverse=True)
    ], ignore_index=True), ctx.out.tables / "part3_test_per_user.csv")
    _plot_frontier(test_summary, ctx.out.figures / "fig6_part3_accuracy_diversity.png")
    create_part3_analysis(ctx.out.tables, ctx.out.figures)
    analyze_popularity_exposure(
        test_runs,
        train_test_cut,
        selected_lambda,
        ctx.out.tables,
        ctx.out.figures,
        ctx.data.movies,
    )

    for row in test_summary.itertuples(index=False):
        lam = float(row[0])  # lambda is first column
        ctx.results.add(SHEET, {
            "Base model": f"EASE (lambda={ease_lam:g}, ratings >= {ease_min_rating})",
            "Re-ranking method": "MMR" if lam < 1.0 else "MMR control (EASE order)",
            "Trade-off parameter": lam,
            "Item distance": "1 - cosine similarity (18 binary genres)",
            "Precision@10": row.precision,
            "Recall@10": row.recall,
            "NDCG@10": row.ndcg,
            "Intra-list diversity": row.ild,
            "Catalogue coverage": row.coverage,
            "Gini": row.gini,
            "Calibration error": row.calibration,
            "Notes": (f"Top-{pool_size} EASE candidates, test users={row.n_users}; "
                      f"Jensen-Shannon genre calibration on {row.calibration_users} users; "
                      f"tail slot share={row.tail_slot_share:.3f}; "
                      f"paired ΔNDCG@10 95% CI [{row.ndcg_ci_low:+.4f}, {row.ndcg_ci_high:+.4f}]; "
                      + ("chosen on validation" if bool(row.chosen_on_validation) else "")),
        })
    log.info("Part 3 wrote %d test rows, selected lambda %.2f", len(test_summary), selected_lambda)
