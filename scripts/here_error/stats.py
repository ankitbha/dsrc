"""Medians, cluster bootstrap, Spearman correlation and thirds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from here_error import params


@dataclass(frozen=True)
class MedianCI:
    median: float
    lo: float
    hi: float
    n: int
    n_clusters: int


def _percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_median(values, clusters, n_boot: int = params.BOOTSTRAP_N, seed: int = params.SEED) -> MedianCI:
    """Median with a 95% percentile interval from resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(clusters)
    if values.size == 0:
        raise ValueError("no values to summarise")
    uniq = list(dict.fromkeys(labels.tolist()))
    groups = [values[labels == u] for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[b] = np.median(np.concatenate([groups[i] for i in pick]))
    lo, hi = _percentile_ci(draws)
    return MedianCI(float(np.median(values)), lo, hi, int(values.size), len(groups))


def row_bootstrap_median(values, n_boot: int = params.BOOTSTRAP_N, seed: int = params.SEED) -> MedianCI:
    """Median with a 95% percentile interval from resampling single rows."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    lo, hi = _percentile_ci(np.median(values[idx], axis=1))
    return MedianCI(float(np.median(values)), lo, hi, int(values.size), int(values.size))


def average_ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties; NaN when either series is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx, ry = average_ranks(x), average_ranks(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def thirds(values) -> np.ndarray:
    """Group index 0, 1, 2 by the 1/3 and 2/3 quantiles of the values (ties stay together)."""
    v = np.asarray(values, dtype=float)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    return np.where(v <= q1, 0, np.where(v <= q2, 1, 2))


def median_half_width(sd: float, n_clusters: int) -> float:
    """Half-width of the 95% interval on a median of independent clusters.

    1.96 * sd / sqrt(n) is the interval on a mean; the median of roughly normal data has
    a standard error larger by sqrt(pi / 2), about 1.2533.
    """
    return 1.96 * 1.2533141 * sd / np.sqrt(n_clusters)
