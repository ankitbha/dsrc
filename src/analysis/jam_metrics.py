"""Jam metrics for the jam-avoidance program (plan 2026-10-16 sections 4.3, 5.7, 6, J20).

Inputs are the 10 s traces of `jam_episode` (per segment vehicle-weighted space-mean speed,
occupancy and count; held and pending; and the per-segment means of the camera estimate
`k_hat_mean` and the true local occupancy `k_true_mean`).

A segment is jammed at a sample when it holds a vehicle and its speed is below `v_jam`,
the network threshold of the feed program (`results/feed/drain_thresholds.json`, pooled
fit, 19.0 km/h), read and never refitted. A jam spell is a run of consecutive jammed
samples on one segment; `n` samples at spacing `dt` last `n dt`. Onset is the start of the
first spell lasting at least `H = 120 s` that starts at or after the scenario's clock
start; a spell still short of `H` at the cap is not an onset. A run with no onset is
censored at the cap and kept. Jam time is in jammed segment-minutes. This module reads
speeds, never the reward's density.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from src.sumo import drain as _drain

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = REPO_ROOT / "results" / "feed" / "drain_thresholds.json"
HOLD_S: float = 120.0
#: The fixed-threshold sensitivity, the policy's lowest action speed.
FIXED_KMH: float = _drain.FIXED_THRESHOLD_KMH
#: Section 4.3: the precursors are measured against this window and must hold this long.
BASELINE_S: tuple[float, float] = (600.0, 900.0)
PRECURSOR_HOLD_S: float = 30.0
SPEED_PRECURSOR_SHARE: float = 0.8
DENSITY_PRECURSOR_SD: float = 3.0
#: Criterion C's scenarios and window (section 5.7).
SELECTION_SCENARIOS: tuple[str, ...] = ("B5", "B7.5", "B9", "I_exit", "I_mid")
SELECTION_WINDOW_S: tuple[float, float] = (900.0, 3600.0)


def load_v_jam(path: Path = THRESHOLDS) -> tuple[float, str]:
    """`(v_jam, sha256 of the file)` from the feed program's network-pooled fit."""
    raw = Path(path).read_bytes()
    v = json.loads(raw)["network"]["pooled"]["v_jam"]
    if v is None:
        raise ValueError(f"{path} has no pooled network threshold")
    return float(v), hashlib.sha256(raw).hexdigest()


def _times(trace: Sequence[dict]) -> np.ndarray:
    return np.array([s["t"] for s in trace], dtype=float)


def _spacing(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.0


class Spell(NamedTuple):
    segment: int
    start: float
    last: float
    length_s: float


class Onset(NamedTuple):
    t: float
    segment: int


def jam_spells(trace: Sequence[dict], v_jam: float) -> list[Spell]:
    """Every per-segment run of consecutive jammed samples."""
    times = _times(trace)
    dt = _spacing(times)
    flags = _drain.jam_indicator(trace, v_jam)
    out = []
    for segment in range(flags.shape[1]):
        for start, last in _drain.spells(flags[:, segment], times):
            out.append(Spell(segment, float(start), float(last), float(last - start + dt)))
    return out


def onset(trace: Sequence[dict], v_jam: float, t_start: float, cap: float,
          H: float = HOLD_S) -> Onset | None:
    """The earliest start, at or after `t_start`, of a spell of at least `H` by the cap."""
    inside = [s for s in trace if s["t"] <= cap]
    candidates = [Onset(s.start, s.segment) for s in jam_spells(inside, v_jam)
                  if s.start >= t_start - 1e-9 and s.length_s >= H - 1e-9]
    return min(candidates) if candidates else None


def prevented(trace: Sequence[dict], v_jam: float, t_start: float, cap: float,
              H: float = HOLD_S) -> bool:
    return onset(trace, v_jam, t_start, cap, H) is None


def first_sample_onset(trace: Sequence[dict], v_jam: float, t_start: float) -> float | None:
    """Block D's onset, the first jammed sample at or after `t_start` (open item 6)."""
    return _drain.onset(trace, t_start, v_jam)


def jam_time(trace: Sequence[dict], v_jam: float, t_start: float, cap: float) -> float:
    """Jammed segment-minutes on `[t_start, cap]`."""
    return _drain.jam_segment_minutes(trace, v_jam, t_start, cap)


def held_minutes(trace: Sequence[dict], t_start: float, cap: float) -> tuple[float, float]:
    """`(held, pending)` vehicle-minutes on `[t_start, cap]`."""
    times = _times(trace)
    dt = _spacing(times)
    inside = [s for s in trace if t_start <= s["t"] <= cap]
    return (dt * sum(s["held"] for s in inside) / 60.0,
            dt * sum(s["pending"] for s in inside) / 60.0)


def drain(trace: Sequence[dict], v_jam: float, t_0: float, cap: float,
          H: float = HOLD_S) -> tuple[float, bool]:
    """`drain.drain_time` with the clock at the perturbation's end (J20)."""
    return _drain.drain_time(trace, t_0, H, cap, v_jam)


def km_onset(onsets: Sequence[float | None], t_start: float, cap: float) -> _drain.KaplanMeier:
    """Kaplan-Meier of "no onset yet" against time after `t_start`, censored at the cap."""
    times = [(o - t_start) if o is not None else (cap - t_start) for o in onsets]
    return _drain.kaplan_meier(times, [o is not None for o in onsets])


def _first_held(flags: np.ndarray, times: np.ndarray, hold: float) -> float | None:
    dt = _spacing(times)
    for start, last in _drain.spells(flags, times):
        if last - start + dt >= hold - 1e-9:
            return float(start)
    return None


def _series(trace: Sequence[dict], key: str, segment: int) -> np.ndarray:
    return np.array([math.nan if (s.get(key) is None or s[key][segment] is None)
                     else float(s[key][segment]) for s in trace], dtype=float)


def leads(trace: Sequence[dict], v_jam: float, t_start: float, cap: float,
          H: float = HOLD_S, baseline: tuple[float, float] = BASELINE_S) -> dict[str, Any]:
    """Section 4.3 on the first segment to start a 120 s jam spell.

    `W_feed` = onset minus the first time at or after `t_start` the segment's speed is
    below 80% of its baseline mean for 30 s; `W_cam` and `W_true` the same with the mean
    camera estimate (and the mean true occupancy) above its baseline mean plus 3 SD for
    30 s. None where the precursor never appears before onset.
    """
    first = onset(trace, v_jam, t_start, cap, H)
    out: dict[str, Any] = {"onset": None, "segment": None, "W_feed": None, "W_cam": None,
                           "W_true": None}
    if first is None:
        return out
    out.update(onset=first.t, segment=first.segment)
    times = _times(trace)
    base = (times >= baseline[0]) & (times < baseline[1])
    window = (times >= t_start) & (times <= first.t)
    speed = _series(trace, "speed_kmh", first.segment)
    if np.isfinite(speed[base]).any():
        level = SPEED_PRECURSOR_SHARE * float(np.nanmean(speed[base]))
        with np.errstate(invalid="ignore"):
            t_v = _first_held(window & (speed < level), times, PRECURSOR_HOLD_S)
        out["W_feed"] = None if t_v is None else first.t - t_v
    for key, name in (("k_hat_mean", "W_cam"), ("k_true_mean", "W_true")):
        series = _series(trace, key, first.segment)
        if not np.isfinite(series[base]).any():
            continue
        level = float(np.nanmean(series[base]) + DENSITY_PRECURSOR_SD * np.nanstd(series[base]))
        with np.errstate(invalid="ignore"):
            t_k = _first_held(window & (series > level), times, PRECURSOR_HOLD_S)
        out[name] = None if t_k is None else first.t - t_k
    return out


def speed_density_break(traces: Sequence[Sequence[dict]], free_flow_kmh: float = 60.0,
                        bin_width: float = 0.01, min_samples: int = 20,
                        share: float = 0.95) -> dict[str, Any]:
    """`k_b`: the lower edge of the first occupancy bin whose median segment speed is below
    `share` of free flow (section 6.2), from the pooled occupied segment-samples."""
    pairs = []
    for trace in traces:
        for s in trace:
            for k, v, n in zip(s["density"], s["speed_kmh"], s.get("count") or [1] * len(s["density"])):
                if n and v is not None and not math.isnan(v):
                    pairs.append((float(k), float(v)))
    if not pairs:
        return {"k_b": None, "bins": []}
    k = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    edges = np.arange(0.0, k.max() + bin_width, bin_width)
    index = np.floor(k / bin_width + 1e-9).astype(int)
    bins, k_b = [], None
    for b in range(len(edges)):
        inside = v[index == b]
        if len(inside) < min_samples:
            continue
        median = float(np.median(inside))
        bins.append({"lower": round(b * bin_width, 10), "n": int(len(inside)),
                     "median_speed_kmh": median})
        if k_b is None and median < share * free_flow_kmh:
            k_b = round(b * bin_width, 10)
    return {"k_b": k_b, "bins": bins, "free_flow_kmh": free_flow_kmh, "share": share,
            "bin_width": bin_width, "samples": len(pairs)}


def max_ramp(trace: Sequence[dict], segment: int, onset: float,
             window_s: float = PRECURSOR_HOLD_S) -> dict[str, Any]:
    """The largest rise of the segment's occupancy over `window_s` up to onset (`r`)."""
    times = _times(trace)
    density = np.array([s["density"][segment] for s in trace], dtype=float)
    dt = _spacing(times)
    lag = int(round(window_s / dt)) if dt else 0
    best = None
    for i in range(lag, len(times)):
        if times[i] > onset:
            break
        rise = density[i] - density[i - lag]
        if best is None or rise > best[0]:
            best = (float(rise), float(times[i]))
    if best is None:
        return {"rise": None, "rate_per_s": None, "at": None}
    return {"rise": best[0], "rate_per_s": best[0] / window_s, "at": best[1]}


def criterion_c(J: Mapping[str, Mapping[str, Mapping[int, float]]], arm: str,
                scenarios: Sequence[str] = SELECTION_SCENARIOS,
                plan: str = "plan", no_control: str = "no_control") -> dict[str, Any]:
    """Section 5.7: the mean over scenarios of `mean_seeds(J_a - J_plan) / mean_seeds(J_nc)`.

    `J[scenario][arm][seed]` is jammed segment-minutes on [900, 3,600] s. The arm, the
    plan and no control must share their seeds in every scenario; a scenario where no
    control has no jam time has no term and is listed.
    """
    terms, excluded = {}, []
    for scenario in scenarios:
        rows = J[scenario]
        seeds = set(rows[arm])
        if seeds != set(rows[plan]) or seeds != set(rows[no_control]):
            raise ValueError(f"{scenario}: {arm}, {plan} and {no_control} are not paired on "
                             f"the same seeds")
        ordered = sorted(seeds)
        baseline = statistics.fmean(rows[no_control][s] for s in ordered)
        if baseline <= 0.0:
            excluded.append(scenario)
            continue
        terms[scenario] = statistics.fmean(rows[arm][s] - rows[plan][s] for s in ordered) / baseline
    return {"arm": arm, "C": statistics.fmean(terms.values()) if terms else None,
            "terms": terms, "excluded_scenarios": excluded}


def episode_metrics(trace: Sequence[dict], *, v_jam: float, start_s: float, end_s: float,
                    cap: float, H: float = HOLD_S) -> dict[str, Any]:
    """Every per-episode jam quantity of section 8.3, from one trace."""
    first = onset(trace, v_jam, start_s, cap, H)
    held, pending = held_minutes(trace, start_s, cap)
    times = _times(trace)
    if len(times) and times[-1] >= cap and cap > end_s:
        drain_s, drained = drain(trace, v_jam, end_s, cap, H)
        open_spell = _drain.open_spell_at_cap(trace, cap, v_jam)
    else:
        drain_s, drained, open_spell = None, None, None
    return {
        "v_jam": v_jam, "hold_s": H, "start_s": start_s, "end_s": end_s, "cap_s": cap,
        "onset_s": first.t if first else None,
        "onset_segment": first.segment if first else None,
        "onset_after_start_s": (first.t - start_s) if first else None,
        "first_sample_onset_s": first_sample_onset(trace, v_jam, start_s),
        "prevented": first is None,
        "jam_time_min": jam_time(trace, v_jam, start_s, cap),
        "jam_time_min_900_3600": jam_time(trace, v_jam, SELECTION_WINDOW_S[0],
                                          min(SELECTION_WINDOW_S[1], cap)),
        "jam_time_min_30kmh": jam_time(trace, FIXED_KMH, start_s, cap),
        "held_min": held, "pending_min": pending,
        "drain_s": drain_s, "drained": drained, "open_spell_at_cap": open_spell,
    }


@dataclass(frozen=True)
class Paired:
    """A paired difference across seeds: mean, two standard errors and the seeds."""

    seeds: list[int]
    differences: list[float]
    mean: float | None
    two_se: float | None

    @classmethod
    def of(cls, a: Mapping[int, float], b: Mapping[int, float]) -> "Paired":
        seeds = sorted(set(a) & set(b))
        diffs = [a[s] - b[s] for s in seeds]
        mean = statistics.fmean(diffs) if diffs else None
        two_se = (2.0 * statistics.stdev(diffs) / math.sqrt(len(diffs))
                  if len(diffs) >= 2 else None)
        return cls(seeds, diffs, mean, two_se)

    def as_dict(self) -> dict[str, Any]:
        return {"seeds": self.seeds, "differences": self.differences, "mean": self.mean,
                "two_se": self.two_se}
