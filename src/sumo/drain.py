"""Drain time: how long after demand stops the network takes to leave the jammed state.

Inputs are the 10 s traces of `feed_episode` (per super-segment vehicle-weighted
space-mean speed and count, plus the gate's held count and SUMO's pending count). A
segment is jammed at a sample when it holds a vehicle and its space-mean speed is below
`v_jam`. The network has drained at `t_d` when no segment was jammed at any sample in
`[t_d - H, t_d]` and nothing is waiting outside it: held at the gate and pending in SUMO
are both zero (plan decision D9), and, after `t_d` up to the cap, no jam spell and no
waiting spell lasting at least `H` begins (plan Amendment 1, A3). Without that last
condition a run that is clear when the clock starts and jams a few minutes later counts
as drained at `t_0 + H`.

The jam threshold is a SPEED, fitted to the speed distribution of the uncontrolled
runs: `v_jam = (s1 + s2) / 4` from a three-piece linear fit to the empirical CDF of the
pooled segment speeds (plan D10). It is deliberately not the reward's critical density;
this module never reads that constant or the reward's density.

Censored runs (still jammed or still holding demand at the cap) are kept with
`drained=False` and enter the Kaplan-Meier estimate as censored at the cap.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

#: The policy's lowest speed action, the fixed sensitivity threshold.
FIXED_THRESHOLD_KMH: float = 30.0
#: Grid spacing of the CDF fit.
GRID_KMH: float = 0.5
#: A per-segment fit with `s2 < SEPARATION * s1` has no separated jam mode and falls
#: back to the pooled threshold.
SEPARATION: float = 1.5
#: A fit with `s2 >= REFUSAL_RATIO * s1` is refused (plan Amendment 1, A4). The 2023
#: paper takes `(s1 + s2) / 4` as the threshold because, with `s2 <= 3 s1`, it lies
#: below `s1`; a pool that is mostly jam (the surge scenario after its demand stops)
#: puts the knots at 3.5 and 25 km/h and the threshold above `s1`.
REFUSAL_RATIO: float = 3.0
#: Amendment 1, A4: the primary threshold is fitted on the sustained scenario's
#: published and no-control episodes over this window, a normal-operation pool.
NETWORK_WINDOW_S: tuple[float, float] = (300.0, 2500.0)


class CdfFit(NamedTuple):
    s1: float
    s2: float
    residual: float


def jam_threshold(fit: CdfFit) -> float:
    return (fit.s1 + fit.s2) / 4.0


def fit_three_piece_cdf(speeds: Sequence[float]) -> CdfFit:
    """Breakpoints of the least-squares continuous three-piece line through the CDF.

    The empirical CDF is evaluated on a 0.5 km/h grid from 0 to the maximum speed. For
    every pair of interior grid points `s1 < s2` the continuous piecewise-linear
    function with knots at `s1` and `s2` is fitted by least squares (basis 1, x,
    (x - s1)+, (x - s2)+); the pair with the smallest residual sum of squares is
    returned with that residual.
    """
    values = np.asarray([s for s in speeds if s is not None and not math.isnan(s)], float)
    if not len(values):
        raise ValueError("no speeds to fit")
    top = math.ceil(values.max() / GRID_KMH) * GRID_KMH
    grid = np.arange(0.0, top + GRID_KMH / 2, GRID_KMH)
    if len(grid) < 4:
        raise ValueError("the speeds span fewer than four grid points")
    cdf = np.searchsorted(np.sort(values), grid, side="right") / len(values)
    hinge = np.maximum(grid[None, :] - grid[:, None], 0.0)  # hinge[k] = (x - grid[k])+
    best = (math.inf, 0, 0)
    for i in range(1, len(grid) - 2):
        # All s2 > s1 at once: stack the designs and solve each least-squares problem.
        js = np.arange(i + 1, len(grid) - 1)
        design = np.stack([np.broadcast_to(np.ones_like(grid), (len(js), len(grid))),
                           np.broadcast_to(grid, (len(js), len(grid))),
                           np.broadcast_to(hinge[i], (len(js), len(grid))),
                           hinge[js]], axis=2)
        gram = np.einsum("jnk,jnl->jkl", design, design)
        rhs = np.einsum("jnk,n->jk", design, cdf)
        coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        residuals = ((np.einsum("jnk,jk->jn", design, coefficients) - cdf) ** 2).sum(axis=1)
        j = int(np.argmin(residuals))
        if residuals[j] < best[0]:
            best = (float(residuals[j]), i, int(js[j]))
    return CdfFit(float(grid[best[1]]), float(grid[best[2]]), best[0])


def _speeds_counts(trace: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    speed = np.array([[math.nan if v is None else v for v in s["speed_kmh"]] for s in trace],
                     dtype=float)
    count = np.array([s["count"] for s in trace], dtype=float)
    return speed, count


def jam_indicator(trace: Sequence[dict], v_jam: float | np.ndarray) -> np.ndarray:
    """`(samples, segments)`: occupied and slower than `v_jam` (scalar or per segment)."""
    speed, count = _speeds_counts(trace)
    with np.errstate(invalid="ignore"):
        return (count > 0) & np.nan_to_num(speed < np.asarray(v_jam, float), nan=False)


def _times(trace: Sequence[dict]) -> np.ndarray:
    return np.array([s["t"] for s in trace], dtype=float)


def spells(flags: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    """`(first, last)` sample time of every run of consecutive flagged samples."""
    out, start = [], None
    for flag, t in zip(flags, times):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            out.append((start, previous))
            start = None
        previous = t
    if start is not None:
        out.append((start, previous))
    return out


def long_spell_starts(trace: Sequence[dict], H: float, cap: float,
                      v_jam: float | np.ndarray) -> list[float]:
    """Start times of the jam spells (any segment jammed) and waiting spells (held +
    pending > 0) that last at least `H`, counted up to the cap.

    A spell of `n` consecutive samples at spacing `dt` lasts `n dt`.
    """
    times = _times(trace)
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    inside = times <= cap
    jammed = jam_indicator(trace, v_jam).any(axis=1) & inside
    waiting = (np.array([s["held"] + s["pending"] for s in trace]) > 0) & inside
    return sorted(start for flags in (jammed, waiting)
                  for start, last in spells(flags, times)
                  if last - start + spacing >= H)


def drain_time(trace: Sequence[dict], t_0: float, H: float, cap: float,
               v_jam: float | np.ndarray) -> tuple[float, bool]:
    """`(t_d - t_0, True)` for the first sample `t_d >= t_0 + H` with no jammed segment
    on `[t_d - H, t_d]`, zero held and pending at `t_d`, and no jam or waiting spell of
    at least `H` starting after `t_d` up to the cap; `(cap - t_0, False)` if no such
    sample exists by the cap.
    """
    times = _times(trace)
    if times[-1] < cap:
        raise ValueError(f"the trace ends at {times[-1]} s, before the cap at {cap} s")
    jammed_any = jam_indicator(trace, v_jam).any(axis=1)
    waiting = np.array([s["held"] + s["pending"] for s in trace]) > 0
    jam_times = times[jammed_any]
    later = long_spell_starts(trace, H, cap, v_jam)
    for index, t_d in enumerate(times):
        if t_d < t_0 + H or t_d > cap:
            continue
        if waiting[index]:
            continue
        if np.any((jam_times >= t_d - H) & (jam_times <= t_d)):
            continue
        if any(start > t_d for start in later):
            continue
        return float(t_d - t_0), True
    return float(cap - t_0), False


def onset(trace: Sequence[dict], t_start: float, v_jam: float | np.ndarray) -> float | None:
    """The first jammed sample at or after `t_start`, or None."""
    times = _times(trace)
    jammed = jam_indicator(trace, v_jam).any(axis=1) & (times >= t_start)
    return float(times[jammed][0]) if jammed.any() else None


def jam_segment_minutes(trace: Sequence[dict], v_jam: float | np.ndarray, t_start: float,
                        cap: float) -> float:
    """Sample spacing x jammed (segment, sample) pairs on `[t_start, cap]`, in minutes.

    A lower bound for a censored run, which was still jammed when the trace stopped.
    """
    times = _times(trace)
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    inside = (times >= t_start) & (times <= cap)
    return spacing * float(jam_indicator(trace, v_jam)[inside].sum()) / 60.0


@dataclass
class KaplanMeier:
    times: list[float]
    survival: list[float]
    median: float | None
    n: int

    def at(self, t: float) -> float:
        value = 1.0
        for time, survival in zip(self.times, self.survival):
            if time <= t:
                value = survival
        return value

    def as_dict(self) -> dict[str, Any]:
        return {"times": self.times, "survival": self.survival, "median": self.median,
                "n": self.n}


def kaplan_meier(times: Sequence[float], drained: Sequence[bool]) -> KaplanMeier:
    """The product-limit survival estimate of "not yet drained", censoring at the cap.

    At a time with both events and censorings the events are counted first, the usual
    convention. The median is the first time at which the estimate is at or below 0.5,
    None when it never gets there (reported as beyond the cap).
    """
    if len(times) != len(drained):
        raise ValueError("one drained flag per time")
    pairs = sorted(zip((float(t) for t in times), (bool(d) for d in drained)))
    at_risk = len(pairs)
    survival = 1.0
    out_times, out_survival = [], []
    index = 0
    while index < len(pairs):
        time = pairs[index][0]
        events = censored = 0
        while index < len(pairs) and pairs[index][0] == time:
            events += pairs[index][1]
            censored += not pairs[index][1]
            index += 1
        if events:
            survival *= 1.0 - events / at_risk
        at_risk -= events + censored
        out_times.append(time)
        out_survival.append(survival)
    median = next((t for t, s in zip(out_times, out_survival) if s <= 0.5), None)
    return KaplanMeier(out_times, out_survival, median, len(pairs))


def _pooled_speeds(traces: Sequence[Sequence[dict]], t_from: float, cap: float,
                   segment: int | None = None) -> list[float]:
    pooled = []
    for trace in traces:
        for sample in trace:
            if not t_from <= sample["t"] <= cap:
                continue
            for e, (speed, count) in enumerate(zip(sample["speed_kmh"], sample["count"])):
                if segment is not None and e != segment:
                    continue
                if count > 0 and speed is not None and not math.isnan(speed):
                    pooled.append(float(speed))
    return pooled


def refusal_reason(fit: CdfFit) -> str | None:
    """Why a fit cannot give a threshold, or None when it can."""
    if fit.s2 >= REFUSAL_RATIO * fit.s1:
        return (f"s2 = {fit.s2:g} >= {REFUSAL_RATIO:g} s1 = {REFUSAL_RATIO * fit.s1:g}: "
                "(s1 + s2)/4 would not lie below s1")
    return None


def threshold_entry(speeds: Sequence[float]) -> dict[str, Any]:
    """A fit and its threshold, or the fit and its recorded refusal."""
    try:
        fit = fit_three_piece_cdf(speeds)
    except ValueError as error:
        return {"s1": None, "s2": None, "residual": None, "v_jam": None,
                "refused": True, "reason": str(error), "samples": len(speeds)}
    reason = refusal_reason(fit)
    return {"s1": fit.s1, "s2": fit.s2, "residual": fit.residual,
            "v_jam": None if reason else jam_threshold(fit), "refused": reason is not None,
            "reason": reason, "samples": len(speeds)}


def thresholds_from_traces(traces: Sequence[Sequence[dict]], t_from: float, t_to: float,
                           fallback_v_jam: float | None = None) -> dict[str, Any]:
    """A pooled threshold and per-segment thresholds from `traces` on `[t_from, t_to]`.

    Segments holding at least one vehicle. The pooled fit is refused when
    `s2 >= 3 s1`. A segment's own fit is used unless it is refused, has `s2 < 1.5 s1`
    (no separated jam mode) or cannot be fitted; then the segment falls back to
    `fallback_v_jam`, or to the pooled threshold when no fallback is given.
    """
    pooled_speeds = _pooled_speeds(traces, t_from, t_to)
    pooled = threshold_entry(pooled_speeds)
    fallback = fallback_v_jam if fallback_v_jam is not None else pooled["v_jam"]
    segments = len(traces[0][0]["speed_kmh"])
    per_segment = []
    for e in range(segments):
        entry = {"segment": e, **threshold_entry(_pooled_speeds(traces, t_from, t_to, e))}
        weak = (entry["s1"] is not None and not entry["refused"]
                and entry["s2"] < SEPARATION * entry["s1"])
        if entry["refused"] or weak:
            entry.update(fallback=True, own_v_jam=entry["v_jam"], v_jam=fallback)
        else:
            entry["fallback"] = False
        per_segment.append(entry)
    return {"t_from": t_from, "t_to": t_to, "samples": len(pooled_speeds),
            "pooled": pooled, "per_segment": per_segment, "fixed_kmh": FIXED_THRESHOLD_KMH}


def drain_summary(per_seed: dict[str, dict[int, tuple[float, bool]]],
                  cap_minus_t0: float, baseline: str = "no_control") -> dict[str, Any]:
    """Fraction drained, the KM curve and median per arm, and paired differences.

    Every seed stays in its arm's denominator whether or not it drained. The paired
    difference (arm minus baseline) is taken only over seeds where both drained, and the
    seeds it used are listed.
    """
    arms = {}
    for arm, rows in per_seed.items():
        seeds = sorted(rows)
        times = [rows[s][0] for s in seeds]
        flags = [rows[s][1] for s in seeds]
        km = kaplan_meier(times, flags)
        arms[arm] = {"n": len(seeds), "drained": sum(flags),
                     "fraction_drained": sum(flags) / len(seeds) if seeds else None,
                     "km": km.as_dict(), "cap_minus_t0": cap_minus_t0,
                     "per_seed": [{"seed": s, "time_s": rows[s][0], "drained": rows[s][1]}
                                  for s in seeds]}
    paired = {}
    if baseline in per_seed:
        base = per_seed[baseline]
        for arm, rows in per_seed.items():
            if arm == baseline:
                continue
            both = sorted(s for s in rows if s in base and rows[s][1] and base[s][1])
            differences = [rows[s][0] - base[s][0] for s in both]
            paired[arm] = {
                "seeds": both, "differences": differences,
                "mean": statistics.fmean(differences) if differences else None,
                "two_se": (2.0 * statistics.stdev(differences) / math.sqrt(len(differences))
                           if len(differences) >= 2 else None),
            }
    return {"arms": arms, "paired": paired}
