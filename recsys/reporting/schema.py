"""Sheet names and column headers of the course results template.

Copied verbatim from Recommender_Systems_Results_Template.xlsm (header row 6,
data from row 7). The workbook writer checks the template against this list,
so a renamed or reordered column fails loudly instead of being misfiled.
"""

from __future__ import annotations

HEADER_ROW = 6
FIRST_DATA_ROW = 7
INSTRUCTIONS_SHEET = "Instructions"

# Cells on the Instructions sheet that hold the team details.
INSTRUCTION_CELLS = {
    "team": "B3",
    "members": "B4",
    "repository": "B5",
    "part3_track": "B6",
    "last_updated": "B7",
}

SHEETS: dict[str, tuple[str, ...]] = {
    "Part 1 - Baselines": (
        "Model", "Variant / settings", "Similarity", "k", "Precision@10", "Recall@10",
        "NDCG@10", "Catalogue coverage", "Seed", "Notes",
    ),
    "Part 2 - Models": (
        "Model", "Paradigm", "Self-implemented?", "Key hyperparameters", "Precision@10",
        "Recall@10", "NDCG@10", "Catalogue coverage", "Variability / CI / test", "Seed(s)", "Notes",
    ),
    "Part 2 - Disaggregated": (
        "Model", "Breakdown dimension", "Group", "Definition / threshold", "Precision@10",
        "Recall@10", "NDCG@10", "Catalogue coverage", "Seed(s)", "Notes",
    ),
    "Part 2 - Sparsity": (
        "Model", "Training interactions (%)", "Precision@10", "Recall@10", "NDCG@10",
        "Catalogue coverage", "Seed(s)", "Notes",
    ),
    "Part 3A - Diversity": (
        "Base model", "Re-ranking method", "Trade-off parameter", "Item distance", "Precision@10",
        "Recall@10", "NDCG@10", "Intra-list diversity", "Catalogue coverage", "Gini",
        "Calibration error", "Notes",
    ),
    "Part 3B - Exposure": (
        "Base model", "Mitigation", "Mitigation strength", "Provider-proxy group",
        "Group size (items)", "Catalogue share", "Position-weighted exposure", "Notes",
    ),
    "Part 3B - Consumer": (
        "Base model", "Mitigation", "Mitigation strength", "User attribute", "User group",
        "Group size (users)", "Accuracy metric", "Accuracy value", "Seed(s)", "Notes",
    ),
    "Part 3B - Frontier": (
        "Base model", "Mitigation", "Mitigation strength", "Performance metric",
        "Performance value", "Fairness metric", "Fairness value", "Notes",
    ),
    "Part 3C - Groups": (
        "Group type", "Group size", "Aggregation strategy", "Group satisfaction",
        "Disagreement", "Minimum member satisfaction", "Seed(s)", "Notes",
    ),
    "Part 3C - Sequential": (
        "Group type", "Group size", "Aggregation strategy", "Round", "Group satisfaction",
        "Disagreement", "Minimum member satisfaction", "Carried-forward satisfaction / state",
        "Seed", "Notes",
    ),
}

# Columns that hold metric values; written with four decimals in the workbook.
METRIC_COLUMNS = frozenset({
    "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage", "Intra-list diversity",
    "Gini", "Calibration error", "Catalogue share", "Position-weighted exposure",
    "Accuracy value", "Performance value", "Fairness value", "Group satisfaction",
    "Disagreement", "Minimum member satisfaction",
})
