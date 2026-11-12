"""Figures for the report.

All figures share one style: thin lines, hairline grid, text in ink colours,
series colours only on marks. The three series colours are the first three
slots of a palette we checked for colour-vision deficiency separation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; also keeps output identical on every OS

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.data.movielens import MovieLens
from recsys.data.split import TemporalSplit

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REFERENCE = "#898781"

FULL_WIDTH = 6.9   # inches; the text width of the report template (US Letter, 0.8 in margins)

# Plain tick labels on log axes: 1, 10, 100, 1,000 instead of powers of ten.
_plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelsize": 7,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _note(ax: plt.Axes, x, y, text: str, **kwargs) -> None:
    ax.annotate(text, (x, y), fontsize=6.3, color=INK_2, **kwargs)


def data_overview(ml: MovieLens, split: TemporalSplit, path: Path) -> None:
    """Figure 1: long tail, user activity and activity over time with the split."""
    apply_style()
    r = ml.ratings
    fig, (ax_items, ax_users, ax_time) = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.95))

    # (a) Long tail: ratings per movie, movies sorted from most to least rated.
    per_item = np.sort(r.groupby("item_id").size().to_numpy())[::-1]
    rank = np.arange(1, per_item.size + 1)
    head = int(round(0.10 * per_item.size))
    half = per_item.size // 2
    head_share = per_item[:head].sum() / per_item.sum()
    tail_share = per_item[half:].sum() / per_item.sum()
    ax_items.fill_between(rank, per_item, 0.8, color=SERIES[0], alpha=0.10, linewidth=0)
    ax_items.plot(rank, per_item, color=SERIES[0])
    ax_items.set_yscale("log")
    ax_items.set_ylim(0.8, per_item.max() * 1.8)
    ax_items.set_xlim(0, per_item.size)
    for x in (head, half):
        ax_items.axvline(x, color=AXIS, linewidth=0.6)
    _note(ax_items, head + 50, 1.5, f"top 10%:\n{head_share:.1%}\nof ratings", va="bottom")
    _note(ax_items, half + 50, 1.5, f"bottom 50%:\n{tail_share:.1%}\nof ratings", va="bottom")
    ax_items.yaxis.set_major_formatter(_plain)
    ax_items.set_title("(a) Ratings per movie")
    ax_items.set_xlabel("Movie rank (most rated first)")
    ax_items.set_ylabel("Ratings (log scale)")

    # (b) Ratings per user as a complementary CDF on log-log axes.
    per_user = np.sort(r.groupby("user_id").size().to_numpy())
    share_at_least = 1.0 - np.arange(per_user.size) / per_user.size
    ax_users.step(per_user, share_at_least, where="post", color=SERIES[0])
    ax_users.set_xscale("log")
    ax_users.set_yscale("log")
    median = np.median(per_user)
    ax_users.axvline(median, color=AXIS, linewidth=0.6)
    _note(ax_users, median * 1.12, 0.02, f"median {median:.0f}")
    _note(ax_users, 21, 0.0015, "minimum is\n20 ratings", va="bottom")
    ax_users.set_xticks([20, 100, 500, 2000])
    ax_users.xaxis.set_major_formatter(_plain)
    ax_users.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_users.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"))
    ax_users.set_title("(b) Ratings per user")
    ax_users.set_xlabel("Ratings by the user (log scale)")
    ax_users.set_ylabel("Users with at least x")

    # (c) Weekly activity with the split boundaries.
    when = pd.to_datetime(r["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    weekly = when.dt.to_period("W").dt.start_time.value_counts().sort_index()
    ax_time.plot(weekly.index, weekly.to_numpy(), color=SERIES[0])
    ax_time.set_yscale("log")
    ax_time.set_ylim(top=weekly.max() * 5)
    val_day, test_day = pd.Timestamp(split.val_start, unit="s"), pd.Timestamp(split.test_start, unit="s")
    for day in (val_day, test_day):
        ax_time.axvline(day, color=INK_2, linewidth=0.7)
    # Part names centred in their own period, above the curve.
    periods = ((weekly.index.min(), val_day, "train"), (val_day, test_day, "val."), (test_day, weekly.index.max(), "test"))
    for start, end, label in periods:
        _note(ax_time, start + (end - start) / 2, weekly.max() * 3, label, ha="center", va="center")
    ax_time.xaxis.set_major_locator(mdates.YearLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_time.yaxis.set_major_formatter(_plain)
    ax_time.set_title("(c) Ratings entered per week")
    ax_time.set_ylabel("Ratings (log scale)")

    fig.tight_layout(w_pad=1.2)
    save(fig, path)


def rating_distribution(ml: MovieLens, path: Path) -> None:
    """Share of each star rating (the numbers are quoted in the report text)."""
    apply_style()
    shares = ml.ratings["rating"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    bars = ax.bar(shares.index, shares.to_numpy(), width=0.5, color=SERIES[0])
    for bar, value in zip(bars, shares.to_numpy()):
        ax.annotate(f"{value:.1%}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.3, color=INK_2)
    ax.set_ylim(0, shares.max() * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_title("Rating distribution")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Share of ratings")
    fig.tight_layout()
    save(fig, path)


def knn_validation(grid: pd.DataFrame, popularity_best: float, selected: dict, path: Path, k_metric: str = "ndcg") -> None:
    """Figure 2: validation NDCG@10 of item-kNN against k, per similarity and training signal."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 1.85), sharey=True)
    colours = {"cosine": SERIES[0], "jaccard": SERIES[1], "adjusted_cosine": SERIES[2]}
    labels = {"cosine": "cosine", "jaccard": "Jaccard", "adjusted_cosine": "adjusted cosine"}
    for ax, signal in zip(axes, (None, 4)):
        part = grid[grid["min_rating"].isna()] if signal is None else grid[grid["min_rating"] == signal]
        for similarity, rows in part.groupby("similarity", sort=False):
            rows = rows.sort_values("k")
            ax.plot(rows["k"], rows[k_metric], color=colours[similarity], marker="o", markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=labels[similarity])
        ax.axhline(popularity_best, color=REFERENCE, linewidth=0.8)
        _note(ax, part["k"].min(), popularity_best, "best MostPopular", xytext=(0, 3), textcoords="offset points")
        if selected["min_rating"] == signal:
            ax.plot([selected["k"]], [selected["value"]], marker="o", markersize=7, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=0.9, linestyle="none")
            _note(ax, selected["k"], selected["value"], "selected", xytext=(-14, 6), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xticks(sorted(part["k"].unique()))
        ax.get_xaxis().set_major_formatter(_plain)
        ax.minorticks_off()
        ax.set_xlabel("Neighbourhood size k (log scale)")
        title = "all ratings" if signal is None else f"only ratings >= {signal}"
        ax.set_title(f"Training signal: {title}")
    axes[0].set_ylabel("Validation NDCG@10")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


# --------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------

GREY = "#b4b2aa"   # lines that are context, not the point


def _dot_rows(ax: plt.Axes, y: np.ndarray, mean, low, high, colour: str, size: float = 4.0) -> None:
    ax.hlines(y, low, high, color=colour, linewidth=1.2, alpha=0.8)
    ax.plot(mean, y, "o", color=colour, markersize=size, markeredgecolor="white", markeredgewidth=0.7)


def part2_overall(rows: pd.DataFrame, path: Path) -> None:
    """Figure 3: test NDCG@10 with 95% intervals over users, and catalogue coverage.

    rows: model label, ndcg, ndcg_low, ndcg_high, coverage; drawn best on top.
    """
    apply_style()
    rows = rows.sort_values("ndcg").reset_index(drop=True)
    y = np.arange(len(rows))
    fig, (ax_ndcg, ax_cov) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.25), sharey=True,
                                          gridspec_kw={"width_ratios": [2.2, 1]})
    _dot_rows(ax_ndcg, y, rows["ndcg"], rows["ndcg_low"], rows["ndcg_high"], SERIES[0], size=4.5)
    for yi, mean, high in zip(y, rows["ndcg"], rows["ndcg_high"]):
        _note(ax_ndcg, high, yi, f"  {mean:.3f}", va="center")
    ax_ndcg.set_xlim(0, rows["ndcg_high"].max() * 1.18)
    ax_ndcg.set_yticks(y, rows["label"])
    ax_ndcg.grid(axis="y", visible=False)
    ax_ndcg.set_title("(a) Test NDCG@10, 95% interval over users")

    ax_cov.hlines(y, 0, rows["coverage"], color=AXIS, linewidth=0.8)
    ax_cov.plot(rows["coverage"], y, "o", color=SERIES[0], markersize=4.5, markeredgecolor="white", markeredgewidth=0.7)
    ax_cov.set_xlim(0, max(rows["coverage"].max() * 1.2, 0.05))
    ax_cov.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_cov.grid(axis="y", visible=False)
    ax_cov.set_title("(b) Catalogue coverage")
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


def part2_breakdowns(groups: pd.DataFrame, labels: dict[str, str], order: list[str], path: Path) -> None:
    """Figure 4: test NDCG@10 per group, one panel per breakdown, with 95% intervals."""
    apply_style()
    panels = (
        ("User activity", ["light", "medium", "heavy"], "(a) User activity"),
        ("Item popularity", ["head"], "(b) Head movies"),
        ("Item popularity", ["tail"], "(c) Tail movies"),
        ("Test period", ["early", "late"], "(d) Test period"),
    )
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_WIDTH, 2.6), sharey=True)
    for ax, (dimension, group_names, title) in zip(axes, panels):
        part = groups[groups["dimension"] == dimension]
        offsets = np.linspace(-0.2, 0.2, len(group_names)) if len(group_names) > 1 else [0.0]
        for slot, (group, offset) in enumerate(zip(group_names, offsets)):
            rows = part[part["group"] == group].set_index("model").reindex(order)
            _dot_rows(ax, y + offset, rows["ndcg"], rows["ndcg_ci_low"], rows["ndcg_ci_high"], SERIES[slot], size=3.6)
            ax.plot([], [], "o", color=SERIES[slot], markersize=3.6, label=group)
        if len(group_names) > 1:
            ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("Test NDCG@10")
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(y, [labels[k] for k in order])
    fig.tight_layout(w_pad=0.8)
    save(fig, path)


def part2_sparsity(summary: pd.DataFrame, labels: dict[str, str], highlight: list[str], path: Path) -> None:
    """Figure 5: test NDCG@10 as the training interactions shrink, with the
    highlighted models in colour and the rest in grey, labelled at the right."""
    apply_style()
    levels = [1.0, 0.5, 0.2]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ends = []
    for key in summary["model"].unique():
        values = summary[summary["model"] == key].set_index("fraction").reindex(levels)["ndcg"].to_numpy()
        colour = SERIES[highlight.index(key)] if key in highlight else GREY
        ax.plot(x, values, color=colour, marker="o", markersize=3.5, markeredgecolor="white",
                markeredgewidth=0.6, linewidth=1.3 if key in highlight else 1.0, zorder=3 if key in highlight else 2)
        ends.append((values[-1], labels[key]))
    # Labels at the right end, nudged apart where lines end close together,
    # with a hairline back to the line when a label had to move.
    ends.sort()
    span = summary["ndcg"].max() - summary["ndcg"].min()
    gap = 0.075 * span
    placed = []
    for value, label in ends:
        position = max(value, placed[-1] + gap) if placed else value
        placed.append(position)
        if abs(position - value) > 1e-9:
            ax.plot([x[-1] + 0.03, x[-1] + 0.12], [value, position], color=AXIS, linewidth=0.5)
        _note(ax, x[-1] + 0.14, position, label, va="center")
    ax.set_xticks(x, ["100%", "50%", "20%"])
    ax.set_xlim(-0.15, x[-1] + 0.9)
    ax.spines["bottom"].set_bounds(0, x[-1])
    ax.set_xlabel("Share of training interactions kept")
    ax.set_ylabel("Test NDCG@10")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, path)
