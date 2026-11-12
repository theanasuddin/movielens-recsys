"""A helper shared by the two models trained with stochastic gradients
(BiasedMF and BPRMF)."""

from __future__ import annotations

import numpy as np


def group_sum(index: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add up the rows of `values` that belong to the same index.

    Returns (unique indices, summed rows, how many rows each index had). A
    mini-batch can contain the same user or movie several times; this turns
    its per-row gradients into one gradient per user or movie. It does the
    same as np.add.at, but sorting first is much faster.
    """
    order = np.argsort(index, kind="stable")
    index, values = index[order], values[order]
    unique, starts, counts = np.unique(index, return_index=True, return_counts=True)
    return unique, np.add.reduceat(values, starts, axis=0), counts
