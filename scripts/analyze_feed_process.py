#!/usr/bin/env python3
"""Measure the feed's update process and write `results/feed/feed_process.json`.

    .venv/bin/python scripts/analyze_feed_process.py \\
        [--polling outputs/feed/polling/afternoon outputs/feed/polling/night]

Plan section 2.5. For every polling session and area, and for every drive of
2026-09-08 that logged HERE responses:

* versions (distinct `sourceUpdated`, in stamp order), gaps `T_{k+1} - T_k`, the stamp's
  second of the minute, and counts of each;
* per version the availability-lag bracket: upper `L_up = first receipt - T_k`, lower
  `L_lo = send time of the last successful query before that receipt - T_k`, clipped at
  zero (none for a session's first version); `L_bar = max L_up`, `L_low = min L_lo`,
  `J = L_bar - L_low`;
* the availability phase support `(T_k + L_up_k) mod 60` and `psi*`, the second whose
  window `[psi, psi + r_bar]` is farthest from it;
* request round-trip times (`r_bar` is their maximum) and `Age` header statistics;
* the `(gap, L_up, phase)` sequence the rule arms replay.

The drive logs carry no send times. Their receipt instants are on the Jetson's
monotonic clock, anchored to wall time by the median of `t_wall - t_mono` over every
`metadata.jsonl` line, and the lower bound uses the previous response's receipt
instead of its send, which is later by one request's round trip. They are the
placeholder process until polling sessions exist: the artefact's `placeholder` flag is
true when it holds no polling session, and the experiment runner refuses a placeholder
unless told `--allow-placeholder`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVES = REPO_ROOT.parent / "dsrc_drive_data_20260908"
DEFAULT_OUT = REPO_ROOT / "results" / "feed" / "feed_process.json"
DRIVE_SOURCE = "drive_2026-09-08"
#: `r_bar` when no round-trip time was measured (the drives have no send times).
ASSUMED_R_BAR_S = 1.0


def parse_stamp(stamp: str) -> float:
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc).timestamp()


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "p50": statistics.median(values),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "max": max(values)}


def _key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def psi_star(support: Iterable[float], r_bar: float) -> tuple[int, float]:
    """The whole second whose window `[psi, psi + r_bar]` lies farthest, around the
    minute, from every point of the availability phase support; and that distance."""
    points = [float(p) % 60.0 for p in support]
    if not points:
        return 0, 60.0
    best_psi, best_margin = 0, -1.0
    for psi in range(60):
        margin = math.inf
        for point in points:
            offset = (point - psi) % 60.0
            if offset <= r_bar:
                distance = 0.0
            else:
                distance = min(offset - r_bar, 60.0 - offset)
            margin = min(margin, distance)
        if margin > best_margin + 1e-12:
            best_psi, best_margin = psi, margin
    return best_psi, best_margin


def _session(observations: list[tuple[float, float, str | None, bool]],
             rtts: Sequence[float] = (), ages: Sequence[float] = (),
             r_bar: float | None = None) -> dict[str, Any]:
    """One area's process from `(t_send, t_recv, sourceUpdated, ok)` in receipt order."""
    observations = sorted(observations, key=lambda o: o[1])
    first_seen: dict[str, dict[str, Any]] = {}
    newest = -math.inf
    regressions = 0
    previous_send = None
    for t_send, t_recv, stamp, ok in observations:
        if not ok or stamp is None:
            continue
        T = parse_stamp(stamp)
        if stamp not in first_seen:
            if T < newest:
                regressions += 1  # an older version first seen after a newer one
            first_seen[stamp] = {
                "sourceUpdated": stamp, "T": T, "first_receipt": t_recv,
                "L_up": t_recv - T,
                # A version is not available before its own stamp, so a lower end
                # below zero (a long gap between queries) is clipped to zero.
                "L_lo": max(0.0, previous_send - T) if previous_send is not None else None,
                "n_responses": 0, "phase": int(T % 60)}
        elif T < newest:
            regressions += 1
        first_seen[stamp]["n_responses"] += 1
        newest = max(newest, T)
        previous_send = t_send
    versions = sorted(first_seen.values(), key=lambda v: v["T"])
    gaps = [b["T"] - a["T"] for a, b in zip(versions, versions[1:])]
    lowers = [v["L_lo"] for v in versions if v["L_lo"] is not None]
    L_bar = max((v["L_up"] for v in versions), default=None)
    L_low = min(lowers) if lowers else None
    support = [(v["T"] + v["L_up"]) % 60.0 for v in versions]
    measured_r_bar = max(rtts) if rtts else None
    window = r_bar if r_bar is not None else (measured_r_bar if measured_r_bar is not None
                                              else ASSUMED_R_BAR_S)
    psi, margin = psi_star(support, window)
    sequence = [[gap, v["L_up"], v["phase"]] for gap, v in zip(gaps, versions)]
    availability_gaps = [gap + b["L_up"] - a["L_up"]
                         for gap, a, b in zip(gaps, versions, versions[1:])]
    return {
        "responses": sum(1 for o in observations if o[3] and o[2] is not None),
        "requests": len(observations),
        "versions": len(versions),
        "per_version": versions,
        "gaps": gaps,
        "gap_counts": {_key(g): n for g, n in sorted(Counter(gaps).items())},
        "phase_counts": {str(p): n for p, n in sorted(Counter(v["phase"] for v in versions).items())},
        "L_up": _stats([v["L_up"] for v in versions]),
        "L_bar": L_bar, "L_low": L_low,
        "J": (L_bar - L_low) if (L_bar is not None and L_low is not None) else None,
        "g_min": min(gaps) if gaps else None,
        "replayed_lag_spread": ((max(v["L_up"] for v in versions)
                                 - min(v["L_up"] for v in versions)) if versions else None),
        "min_availability_gap": min(availability_gaps) if availability_gaps else None,
        "availability_phase_support": support,
        "psi_star": psi, "psi_margin_s": margin,
        "r_bar": window,
        "r_bar_source": ("given" if r_bar is not None else
                         "measured" if measured_r_bar is not None else "assumed"),
        "rtt_s": _stats(rtts),
        "age_header_s": _stats(ages),
        "regressions": regressions,
        "sequence": sequence,
    }


def analyze_records(records: Iterable[dict], r_bar: float | None = None) -> dict[str, Any]:
    """Per area, the process measured by one polling session's request records."""
    by_area: dict[str, list[dict]] = {}
    for record in records:
        if record.get("type", "request") != "request":
            continue
        if str(record.get("area", "")).startswith("coverage"):
            continue
        by_area.setdefault(record["area"], []).append(record)
    out = {}
    for area, rows in by_area.items():
        observations = [(r["t_send_wall"], r["t_recv_wall"], r.get("sourceUpdated"),
                         r.get("status") == 200) for r in rows]
        rtts = [r["rtt_s"] for r in rows if r.get("status") == 200 and r.get("rtt_s") is not None]
        ages = []
        for r in rows:
            try:
                ages.append(float(r.get("Age")))
            except (TypeError, ValueError):
                pass
        out[area] = {"area": area, "source": "polling",
                     **_session(observations, rtts, ages, r_bar)}
    return out


def _anchor(metadata: Path) -> dict[str, float]:
    offsets = []
    with metadata.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("t_wall") and row.get("t_mono"):
                offsets.append(row["t_wall"] - row["t_mono"])
    median = statistics.median(offsets)
    within = sum(1 for o in offsets if abs(o - median) < 0.05)
    return {"offset_s": median, "anchors": len(offsets), "within_50ms": within,
            "spread_s": max(offsets) - min(offsets)}


def analyze_drive_logs(root: Path) -> dict[str, Any]:
    """The same analysis on the drives' HERE responses, one session per run."""
    root = Path(root)
    sessions, pooled = {}, []
    for run in sorted(root.glob("run_*")):
        index = run / "here_index.jsonl"
        if not index.exists() or index.stat().st_size == 0:
            continue
        anchor = _anchor(run / "metadata.jsonl")
        observations = []
        for line in index.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ok = row.get("status") == 200
            stamp = None
            if ok:
                stamp = json.loads((run / row["file"]).read_text()).get("sourceUpdated")
            t_recv = row["received_t_mono"] + anchor["offset_s"]
            # No send time was logged: the receipt stands in for it in the lower bound.
            observations.append((t_recv, t_recv, stamp, ok))
        session = {"area": "nj", "source": DRIVE_SOURCE, "run": run.name,
                   "anchor": anchor, **_session(observations)}
        session["lower_bound_note"] = ("previous response's receipt, not its send: later "
                                       "than the send by one round trip")
        sessions[f"drive_{run.name}"] = session
        pooled.extend(session["per_version"])
    gaps = [g for s in sessions.values() for g in s["gaps"]]
    overall = {
        "responses": sum(s["responses"] for s in sessions.values()),
        "versions": len(pooled),
        "L_up": _stats([v["L_up"] for v in pooled]),
        "L_bar": max(v["L_up"] for v in pooled),
        "L_low": min(s["L_low"] for s in sessions.values() if s["L_low"] is not None),
        "gap_counts": {_key(g): n for g, n in sorted(Counter(gaps).items())},
        "within_run_gaps": len(gaps),
        "phase_counts": {str(p): n for p, n in sorted(Counter(v["phase"] for v in pooled).items())},
        "g_min": min(gaps),
    }
    overall["J"] = overall["L_bar"] - overall["L_low"]
    return {"source": DRIVE_SOURCE, "root": str(root), "sessions": sessions,
            "overall": overall,
            "clock_note": ("receipt times are on the Jetson's clock; the phone's differed by "
                           "1.15 to 1.18 s (summary.json phone/wall_clock_offset_s)")}


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_artefact(path: Path, *, drive: dict | None, polling: dict | None) -> dict[str, Any]:
    """Write the artefact whole-then-rename and return it.

    `polling` maps a session id to `analyze_records`' per-area output. Sessions are named
    `{session}_{area}` for polling and `drive_{run}` for the drives. The primary source
    is polling when any polling session exists, else the drives, and then the artefact
    is a placeholder.
    """
    sessions: dict[str, Any] = {}
    for session_id, areas in (polling or {}).items():
        for area, data in areas.items():
            sessions[f"{session_id}_{area}"] = {"session": session_id, **data}
    polling_names = list(sessions)
    if drive is not None:
        sessions.update(drive["sessions"])
    if not sessions:
        raise ValueError("nothing to write: no polling session and no drive analysis")
    primary_names = polling_names or list(drive["sessions"])
    default = max(primary_names, key=lambda n: (len(sessions[n]["sequence"]), n))
    primary_versions = [v for n in primary_names for v in sessions[n]["per_version"]]
    artefact = {
        "schema": "feed_process/1",
        "generated_by": "scripts/analyze_feed_process.py",
        "git_sha": _git_sha(),
        "generated_wall": time.time(),
        "placeholder": not polling_names,
        "primary_source": "polling" if polling_names else DRIVE_SOURCE,
        "primary": {
            "sessions": primary_names,
            "default_session": default,
            "L_bar": max(v["L_up"] for v in primary_versions),
            "L_up": _stats([v["L_up"] for v in primary_versions]),
        },
        "drive": None if drive is None else {
            "source": drive["source"], "overall": drive["overall"],
            "clock_note": drive["clock_note"]},
        "sessions": sessions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artefact, indent=1))
    temporary.rename(path)
    return artefact


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drives", default=str(DEFAULT_DRIVES))
    parser.add_argument("--polling", nargs="*", default=[],
                        help="polling session directories, each with requests.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    drive = analyze_drive_logs(Path(args.drives)) if Path(args.drives).exists() else None
    polling = {}
    for directory in args.polling:
        directory = Path(directory)
        records = [json.loads(line) for line in
                   (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
        polling[directory.name] = analyze_records(records)
    artefact = write_artefact(Path(args.out), drive=drive, polling=polling or None)
    if drive is not None:
        o = drive["overall"]
        print(f"drives: {o['responses']} responses, {o['versions']} versions; first-receipt "
              f"lag min {o['L_up']['min']:.1f} median {o['L_up']['p50']:.1f} "
              f"p90 {o['L_up']['p90']:.1f} max {o['L_up']['max']:.1f} s; gaps "
              f"{o['gap_counts']}; phases {o['phase_counts']}")
    for name in artefact["primary"]["sessions"]:
        s = artefact["sessions"][name]
        print(f"  {name}: {s['versions']} versions, L_bar {s['L_bar']:.1f} s, "
              f"L_low {s['L_low'] if s['L_low'] is None else round(s['L_low'], 1)}, "
              f"g_min {s['g_min']}, psi* {s['psi_star']} (margin {s['psi_margin_s']:.1f} s, "
              f"r_bar {s['r_bar']} {s['r_bar_source']})")
    print(f"wrote {args.out} (placeholder: {artefact['placeholder']}, default session "
          f"{artefact['primary']['default_session']}, sha256 {sha256(Path(args.out))[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
