"""Local gains of the controlled network and the minimum-penetration certificate.

The model is the sampled comparison system of SNS Assumption 3 and Proposition 1 at the
traces' 10 s sampling:

    y(t + dt) - y(t) ~= dt (-A y(t) + B y(t)),   A = diag(a_e),  B = [b_ej] >= 0,  b_ee = 0,

where `y_e(t) = |rho_e^pulse(t) - rho_e^base(t)|` is segment `e`'s absolute density
deviation between an episode with an extra-vehicle pulse and the unperturbed episode of
the same seed and penetration. Per seed, the twelve sites' deviations are stacked and,
for each segment, the increments are regressed on the twelve current deviations with no
intercept: `a_e = -coef_ee / dt`, `b_ej = |coef_ej| / dt` (plan D17).

Across seeds, one-sided 95% t bounds give `a_LCB` (lower) and `b_UCB` (upper, floored at
zero). The spectral radius of a nonnegative matrix does not decrease when any entry
increases, and every entry of `M = A_LCB^-1 B_UCB` bounds the corresponding entry of
`A^-1 B` from above, so `rho(M)` is an upper confidence bound on `rho(A^-1 B)` (a union
over the 144 one-sided bounds; a Bonferroni level is reported beside it). If any
`a_LCB <= 0` there is no certificate at that penetration (plan D18).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The bounded entries: 12 diagonal gains and 132 off-diagonal couplings.
BOUNDED_ENTRIES: int = 144


# ------------------------------------------------------------------ Student t


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0.0 else tail


def t_quantile(p: float, df: float) -> float:
    """The `p` quantile of Student's t with `df` degrees of freedom, by bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ---------------------------------------------------------------- estimation


def deviations(pulsed: Sequence[dict], base: Sequence[dict], t_from: float,
               t_to: float) -> np.ndarray:
    """`(samples, segments)` of `|rho_pulse - rho_base|` on `[t_from, t_to]`.

    The two traces must carry the same sample instants over the window.
    """
    def window(trace):
        return {s["t"]: s["density"] for s in trace if t_from <= s["t"] <= t_to}

    pulse_w, base_w = window(pulsed), window(base)
    if not pulse_w or sorted(pulse_w) != sorted(base_w):
        raise ValueError("the pulsed and unperturbed traces do not share their samples "
                         f"on [{t_from}, {t_to}]")
    times = sorted(pulse_w)
    return np.abs(np.array([pulse_w[t] for t in times], float)
                  - np.array([base_w[t] for t in times], float))


def fit_gains(runs: Sequence[np.ndarray], dt: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """`(a, B)` from one seed's stacked site runs, each `(samples, segments)`.

    For each segment `e`, `y_e(t + dt) - y_e(t)` is regressed on all `y_j(t)` with no
    intercept; `a_e = -coef_ee / dt`, `b_ej = |coef_ej| / dt` for `j != e`, `b_ee = 0`.
    """
    X = np.concatenate([run[:-1] for run in runs], axis=0)
    Y = np.concatenate([run[1:] - run[:-1] for run in runs], axis=0)
    coefficients, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (j, e): Y_e ~ sum_j X_j c_je
    coef = coefficients.T  # coef[e, j]
    a = -np.diag(coef) / dt
    B = np.abs(coef) / dt
    np.fill_diagonal(B, 0.0)
    return a, B


def confidence_bounds(a_seeds: np.ndarray, b_seeds: np.ndarray, level: float = 0.95,
                      ) -> dict[str, Any]:
    """One-sided `level` t bounds over seeds: lower on each `a_e`, upper on each `b_ej`.

    `a_LCB = mean - t se`, `b_UCB = max(0, mean + t se)`, `t = t_{level, n-1}`.
    """
    a_seeds, b_seeds = np.asarray(a_seeds, float), np.asarray(b_seeds, float)
    n = a_seeds.shape[0]
    if n < 2:
        raise ValueError("bounds over seeds need at least two seeds")
    t = t_quantile(level, n - 1)
    a_hat, b_hat = a_seeds.mean(axis=0), b_seeds.mean(axis=0)
    a_se = a_seeds.std(axis=0, ddof=1) / math.sqrt(n)
    b_se = b_seeds.std(axis=0, ddof=1) / math.sqrt(n)
    b_ucb = np.maximum(0.0, b_hat + t * b_se)
    np.fill_diagonal(b_ucb, 0.0)
    return {"n": n, "level": level, "t": t, "a_hat": a_hat, "b_hat": b_hat,
            "a_lcb": a_hat - t * a_se, "b_ucb": b_ucb}


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, float)))))


def rho_upper_bound(a_lcb: np.ndarray, b_ucb: np.ndarray) -> float | None:
    """`rho(A_LCB^-1 B_UCB)`, or None (no certificate) when any `a_LCB <= 0`."""
    a_lcb = np.asarray(a_lcb, float)
    if np.any(a_lcb <= 0.0):
        return None
    matrix = np.asarray(b_ucb, float) / a_lcb[:, None]
    np.fill_diagonal(matrix, 0.0)
    return spectral_radius(matrix)


def rho_point(a_hat: np.ndarray, b_hat: np.ndarray) -> float | None:
    """`rho(A_hat^-1 B_hat)`, the point estimate beside the bound."""
    return rho_upper_bound(a_hat, b_hat)


def estimate_level(seed_runs: Mapping[int, Sequence[np.ndarray]], dt: float = 10.0,
                   level: float = 0.95) -> dict[str, Any]:
    """The certificate at one penetration from each seed's twelve site deviations."""
    seeds = sorted(seed_runs)
    fits = [fit_gains(seed_runs[s], dt) for s in seeds]
    a_seeds = np.array([a for a, _ in fits])
    b_seeds = np.array([B for _, B in fits])
    bounds = confidence_bounds(a_seeds, b_seeds, level)
    bonferroni = confidence_bounds(a_seeds, b_seeds, 1.0 - (1.0 - level) / BOUNDED_ENTRIES)
    upper = rho_upper_bound(bounds["a_lcb"], bounds["b_ucb"])
    return {
        "seeds": seeds,
        "a_hat": bounds["a_hat"].tolist(), "B_hat": bounds["b_hat"].tolist(),
        "a_LCB": bounds["a_lcb"].tolist(), "B_UCB": bounds["b_ucb"].tolist(),
        "t": bounds["t"], "level": level,
        "rho_hat": rho_point(bounds["a_hat"], bounds["b_hat"]),
        "rho_UCB": upper,
        "certificate": upper is not None and upper < 1.0,
        "no_certificate_reason": (None if upper is not None else
                                  "a lower bound on some a_e is at or below zero"),
        "rho_UCB_bonferroni": rho_upper_bound(bonferroni["a_lcb"], bonferroni["b_ucb"]),
        "bonferroni_level": bonferroni["level"],
        "t_bonferroni": bonferroni["t"],
        "per_seed": {str(s): {"a": a.tolist(), "B": B.tolist()} for s, (a, B) in zip(seeds, fits)},
    }


def min_penetration(rho_ucb_by_level: Mapping[float, float | None]) -> dict[str, Any]:
    """`p_min`: the smallest level with `rho_UCB < 1` (plan section 3.6).

    The next level below `p_min` is the first where the bound is not met. Because a
    level can certify below one that does not, the result also carries
    `p_min_contiguous`, the smallest level such that every level at or above it
    certifies, and `non_monotone` when the two differ.
    """
    levels = sorted(rho_ucb_by_level, reverse=True)
    certified = {p: (rho_ucb_by_level[p] is not None and rho_ucb_by_level[p] < 1.0)
                 for p in levels}
    passing = [p for p in levels if certified[p]]
    p_min = min(passing) if passing else None
    contiguous = None
    for p in levels:
        if not certified[p]:
            break
        contiguous = p
    below = [p for p in levels if p_min is not None and p < p_min]
    first_not_met = (max(below) if below else None) if p_min is not None else (
        levels[0] if levels else None)
    return {"p_min": p_min, "first_not_met": first_not_met,
            "p_min_contiguous": contiguous, "non_monotone": p_min != contiguous,
            "certified": {str(p): certified[p] for p in levels}}


def linearity_ratio(y_large: np.ndarray, y_small: np.ndarray, site: int) -> float:
    """Peak self-deviation at the pulsed site, larger pulse over smaller pulse."""
    small = float(np.max(y_small[:, site]))
    return float(np.max(y_large[:, site])) / small if small > 0.0 else math.nan


# --------------------------------------------------------------------- sites


def load_routes(path: Path) -> dict[str, list[str]]:
    root = ET.parse(path).getroot()
    return {route.get("id"): route.get("edges").split() for route in root.findall("route")}


def pulse_sites(segments: Sequence[Sequence[str]],
                routes: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Per super-segment, where a pulse enters: the segment's first edge on the longest
    route through it, with the route's suffix from that edge to the exit.

    "Longest" is the most edges; a tie goes to the route id first in sorted order.
    """
    sites = []
    for index, segment in enumerate(segments):
        members = set(segment)
        through = [(len(routes[r]), r) for r in sorted(routes)
                   if members & set(routes[r])]
        if not through:
            raise ValueError(f"no route passes segment {index}")
        longest = max(length for length, _ in through)
        route = next(r for length, r in through if length == longest)
        edges = list(routes[route])
        first = next(i for i, edge in enumerate(edges) if edge in members)
        sites.append({"site": index, "route": route, "edge": edges[first],
                      "suffix": edges[first:]})
    return sites
