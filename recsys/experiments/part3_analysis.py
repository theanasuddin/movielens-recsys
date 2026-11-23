
"""Extra Part 3 analysis: supporting graphs and user groups."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def paired_interval(differences, seed=42, samples=1000):
    """Mean paired difference and 95% bootstrap interval."""
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)

    bootstrap_means = np.array([
        d[rng.integers(0, len(d), size=len(d))].mean()
        for _ in range(samples)
    ])

    low, high = np.quantile(bootstrap_means, [0.025, 0.975])

    return float(d.mean()), float(low), float(high)


def create_part3_analysis(tables_dir, figures_dir):
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(tables / "part3_test.csv")
    per_user = pd.read_csv(tables / "part3_test_per_user.csv")

    # Read the lambda selected on validation.
    chosen_rows = summary.loc[
        summary["chosen_on_validation"]
        .astype(str).str.lower() == "true",
        "lambda"
    ]

    if len(chosen_rows) != 1:
        raise ValueError("Expected exactly one selected lambda")

    selected_lambda = float(chosen_rows.iloc[0])

    # --------------------------------------------------
    # FIGURE 7: Beyond-accuracy metrics versus lambda
    # --------------------------------------------------

    ordered = summary.sort_values(
        "lambda", ascending=False
    )

    plot_settings = [
        ("coverage", 100, "Catalogue coverage (%)"),
        ("gini", 1, "Gini (lower = less concentrated)"),
        ("calibration", 1, "Calibration error (JSD)"),
        ("tail_slot_share", 100, "Tail recommendation slots (%)"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), constrained_layout=True
    )

    for ax, (column, scale, ylabel) in zip(
        axes.flat, plot_settings
    ):
        ax.plot(
            ordered["lambda"],
            ordered[column] * scale,
            marker="o"
        )

        ax.axvline(
            selected_lambda,
            linestyle="--",
            alpha=0.65,
            label=f"Selected lambda = {selected_lambda:g}"
        )

        ax.invert_xaxis()
        ax.set_xlabel("MMR lambda")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)

    fig.suptitle(
        "EASE + MMR: Beyond-Accuracy Trade-offs (Test)"
    )

    fig.savefig(
        figures / "fig7_part3_beyond_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------
    # USER-GROUP ANALYSIS
    # --------------------------------------------------

    # Get original EASE per-user results.
    baseline = per_user.loc[
        np.isclose(per_user["lambda_"], 1.0),
        ["user_id", "n_history", "ndcg", "ild"]
    ].copy()

    # Get selected MMR per-user results.
    selected = per_user.loc[
        np.isclose(
            per_user["lambda_"], selected_lambda
        ),
        ["user_id", "n_history", "ndcg", "ild"]
    ].copy()

    # Ensure each user appears only once per setting.
    if (baseline["user_id"].duplicated().any()
            or selected["user_id"].duplicated().any()):
        raise ValueError("Duplicate user IDs found")

    # Compare the same users before and after MMR.
    comparison = baseline.merge(
        selected,
        on="user_id",
        suffixes=("_base", "_mmr"),
        validate="one_to_one"
    )

    if len(comparison) != len(baseline) or len(comparison) != len(selected):
        raise ValueError("Evaluation users do not match")

    if not (
        comparison["n_history_base"]
        == comparison["n_history_mmr"]
    ).all():
        raise ValueError("History counts do not match")

    # Reuse the activity groups from the Part 2 report.
    labels = [
        "Light (<=129)",
        "Medium (130-319)",
        "Heavy (>=320)"
    ]

    comparison["activity_group"] = pd.cut(
        comparison["n_history_base"],
        bins=[-np.inf, 129, 319, np.inf],
        labels=labels
    )

    comparison["delta_ndcg"] = (
        comparison["ndcg_mmr"]
        - comparison["ndcg_base"]
    )

    comparison["delta_ild"] = (
        comparison["ild_mmr"]
        - comparison["ild_base"]
    )

    rows = []

    for group, sub in comparison.groupby(
        "activity_group", observed=True
    ):
        ndcg_mean, ndcg_low, ndcg_high = paired_interval(
            sub["delta_ndcg"]
        )

        ild_mean, ild_low, ild_high = paired_interval(
            sub["delta_ild"]
        )

        rows.append({
            "activity_group": str(group),
            "n_users": len(sub),
            "lambda": selected_lambda,

            "ease_ndcg": sub["ndcg_base"].mean(),
            "mmr_ndcg": sub["ndcg_mmr"].mean(),
            "delta_ndcg": ndcg_mean,
            "ndcg_ci_low": ndcg_low,
            "ndcg_ci_high": ndcg_high,

            "ease_ild": sub["ild_base"].mean(),
            "mmr_ild": sub["ild_mmr"].mean(),
            "delta_ild": ild_mean,
            "ild_ci_low": ild_low,
            "ild_ci_high": ild_high,

            "users_accuracy_improved": int(
                (sub["delta_ndcg"] > 1e-12).sum()
            ),
            "users_accuracy_worsened": int(
                (sub["delta_ndcg"] < -1e-12).sum()
            ),
            "users_accuracy_unchanged": int(
                (sub["delta_ndcg"].abs() <= 1e-12).sum()
            )
        })

    groups = pd.DataFrame(rows)

    if len(groups) != 3:
        raise ValueError("Expected three activity groups")

    groups.to_csv(
        tables / "part3_user_groups.csv",
        index=False
    )

    # --------------------------------------------------
    # FIGURE 8: Impact by user activity
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2),
        constrained_layout=True
    )

    y = np.arange(len(groups))

    configurations = [
        (
            axes[0],
            "delta_ndcg",
            "ndcg_ci_low",
            "ndcg_ci_high",
            "Change in NDCG@10"
        ),
        (
            axes[1],
            "delta_ild",
            "ild_ci_low",
            "ild_ci_high",
            "Change in ILD"
        )
    ]

    for ax, metric, low, high, title in configurations:
        mean = groups[metric].to_numpy()

        errors = np.vstack([
            mean - groups[low].to_numpy(),
            groups[high].to_numpy() - mean
        ])

        ax.errorbar(
            mean, y,
            xerr=errors,
            fmt="o",
            capsize=4
        )

        ax.axvline(
            0, color="gray",
            linestyle="--"
        )

        ax.set_yticks(
            y, groups["activity_group"]
        )

        ax.invert_yaxis()
        ax.set_xlabel(
            "MMR - EASE (95% bootstrap CI)"
        )
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Effect of MMR (lambda={selected_lambda:g}) "
        "by User Activity"
    )

    fig.savefig(
        figures / "fig8_part3_user_groups.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"Part 3 additional analysis: "
        f"lambda={selected_lambda:g}"
    )

    print(groups[
        ["activity_group", "n_users",
         "delta_ndcg", "delta_ild"]
    ].to_string(index=False))

def analyze_popularity_exposure(
    test_runs,
    train,
    selected_lambda,
    tables_dir,
    figures_dir,
    movies,
):
    """Compare movie exposure under EASE and selected MMR.

    Head = top 10% of eligible movies by training ratings.
    Tail = remaining eligible movies.
    """

    from pathlib import Path
    from recsys.evaluation import metrics

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    item_ids = np.asarray(train.ids.items)
    n_items = len(item_ids)

    catalogue = np.asarray(
        train.catalogue, dtype=bool
    )
    eligible = np.flatnonzero(catalogue)

    # Same head/tail definition used by part3.py.
    train_counts = np.bincount(
        train.ids.item_index(train.ratings["item_id"]),
        minlength=n_items,
    )

    ordering = np.argsort(
        -train_counts[catalogue], kind="stable"
    )

    n_head = max(1, int(round(0.10 * len(eligible))))
    head = eligible[ordering[:n_head]]

    tail = catalogue.copy()
    tail[head] = False

    # Match internal movie indices to MovieLens titles.
    title_lookup = movies.set_index("item_id")["title"]

    titles = (
        title_lookup.reindex(item_ids)
        .fillna("(title unavailable)")
        .to_numpy()
    )

    counts_by_model = {}
    recommendations_rows = []
    summary_rows = []

    # Compare original EASE with validation-selected MMR.
    for label, lam in [
        ("EASE", 1.0),
        ("MMR", selected_lambda),
    ]:
        run = test_runs[float(lam)]

        recs = np.asarray(
            run.result.recommendations,
            dtype=int,
        )

        users = (
            run.result.per_user["user_id"]
            .to_numpy()
        )

        if recs.ndim != 2 or len(users) != len(recs):
            raise ValueError(
                "Recommendations and users do not align"
            )

        if not catalogue[recs].all():
            raise ValueError(
                "Recommended movie outside catalogue"
            )

        # Count how often every movie is recommended.
        counts = np.bincount(
            recs.ravel(),
            minlength=n_items,
        )

        counts_by_model[label] = counts

        tail_counts = counts[tail]

        total_tail = int(tail_counts.sum())
        unique_tail = int(np.count_nonzero(tail_counts))
        total = int(counts.sum())

        # Summary measurements.
        summary_rows.append({
            "model": label,
            "lambda": float(lam),
            "recommendation_slots": total,

            "unique_movies": int(
                np.count_nonzero(counts[catalogue])
            ),

            "catalogue_coverage": float(
                np.count_nonzero(counts[catalogue])
                / len(eligible)
            ),

            "gini": float(
                metrics.gini(counts[catalogue])
            ),

            "tail_slots": total_tail,

            "tail_slot_share": total_tail / total,

            "unique_tail_movies": unique_tail,

            "repeated_tail_slots": (
                total_tail - unique_tail
            ),

            "top_10_tail_movies_share_of_tail_slots": (
                float(
                    np.sort(tail_counts)[-10:].sum()
                    / total_tail
                )
                if total_tail else np.nan
            ),

            "top_10_movies_share_of_all_slots": float(
                np.sort(counts[catalogue])[-10:].sum()
                / total
            ),
        })

        # Save the actual recommended movie IDs.
        n_users, k = recs.shape
        flat_recs = recs.ravel()

        recommendations_rows.append(
            pd.DataFrame({
                "model": label,
                "lambda": float(lam),

                "user_id": np.repeat(users, k),

                "rank": np.tile(
                    np.arange(1, k + 1),
                    n_users,
                ),

                "movie_id": item_ids[flat_recs],

                "title": titles[flat_recs],

                "head_or_tail": np.where(
                    tail[flat_recs],
                    "tail",
                    "head",
                ),
            })
        )

    # --------------------------------------------------
    # Save exposure summary
    # --------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        tables_dir / "part3_exposure_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save per-movie recommendation frequencies
    # --------------------------------------------------

    movie_exposure = pd.DataFrame({
        "movie_id": item_ids[eligible],

        "title": titles[eligible],

        "training_ratings": train_counts[eligible],

        "head_or_tail": np.where(
            tail[eligible],
            "tail",
            "head",
        ),

        "ease_exposures": (
            counts_by_model["EASE"][eligible]
        ),

        "mmr_exposures": (
            counts_by_model["MMR"][eligible]
        ),
    })

    movie_exposure["mmr_minus_ease"] = (
        movie_exposure["mmr_exposures"]
        - movie_exposure["ease_exposures"]
    )

    movie_exposure.sort_values(
        "mmr_minus_ease",
        ascending=False,
        inplace=True,
    )

    movie_exposure.to_csv(
        tables_dir / "part3_exposure_per_movie.csv",
        index=False,
    )

    # Save recommendation lists.
    pd.concat(
        recommendations_rows,
        ignore_index=True,
    ).to_csv(
        tables_dir / "part3_recommendations.csv",
        index=False,
    )

    # --------------------------------------------------
    # Figure 9: Movie exposure concentration
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        constrained_layout=True,
    )

    for ax, group_mask, title in [
        (
            axes[0],
            catalogue,
            "All candidate movies",
        ),
        (
            axes[1],
            tail,
            "Tail movies only",
        ),
    ]:
        for label in ["EASE", "MMR"]:
            freq = np.sort(
                counts_by_model[label][group_mask]
            )[::-1]

            # Exclude zero-exposure movies from log plot.
            freq = freq[freq > 0]

            ax.plot(
                np.arange(1, len(freq) + 1),
                freq,
                label=label,
            )

        ax.set_yscale("log")

        ax.set_xlabel(
            "Rank by recommendation frequency"
        )

        ax.set_ylabel(
            "Number of recommendations (log scale)"
        )

        ax.set_title(title)
        ax.grid(alpha=0.2)

    axes[0].legend()

    fig.suptitle(
        "Movie exposure: EASE vs MMR "
        f"(lambda={selected_lambda:g})"
    )

    fig.savefig(
        figures_dir / "fig9_part3_exposure.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nPart 3 popularity-exposure analysis:")
    print(summary.to_string(index=False))

    return summary
