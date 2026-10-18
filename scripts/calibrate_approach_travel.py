#!/usr/bin/env python3
"""Measure each approach's free-flow travel time, and check that the boundary keeps time.

    .venv/bin/python scripts/calibrate_approach_travel.py --seed 1
    .venv/bin/python scripts/build_mainz_scenario.py --approach \\
        --approach-travel data/mainz/mainz_approach_travel.json
    .venv/bin/python scripts/calibrate_approach_travel.py --check --seeds 1 2

Approach plan 2026-10-17 sections 2.2 and 6. The first command runs BG (`mainz_bg_A`) under
no control for 7,200 s plus `W` and writes `data/mainz/mainz_approach_travel.json`: per
entry, the number of vehicles and the mean, median, standard deviation and 5th and 95th
percentiles of the time from a vehicle's scheduled far-end departure to the first 0.5 s
step at which it is on its entry link, with every vehicle's raw crossing, the seed and the
sha256 of every file the run read. The second command (the second pass) rebuilds the
approach demand with the per-entry means as `T_A`. The third runs the boundary checks of
section 6 on the rebuilt files:

* arrivals, BG under no control: per entry the mean of (arrival - `entry_time`) within
  +/-3 s; in every 300 s window on `[W, cap - 300 s]` the arrivals at each entry within
  max(6, 0.15 x scheduled) of the scheduled count; every vehicle whose `entry_time` is at
  or before `cap - 300 s` reaches its entry link;
* the burst, B9 under no control: at every entry whose approach shows no queue in
  `[900 + W, 1,200 + W)`, arrivals in that window within 10% of the scheduled count; all
  eight entries recorded with their mean lags.

Training seeds only (1 and 2). The checks write `outputs/jam_approach/boundary_checks.json`
and exit 1 when a check fails. Nothing here chooses a threshold or an arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DATA = REPO_ROOT / "data" / "mainz"
APPROACH = DATA / "mainz_approach.json"
TRAVEL = DATA / "mainz_approach_travel.json"
CHECKS = REPO_ROOT / "outputs" / "jam_approach" / "boundary_checks.json"
STEP_S = 0.5
DEMAND_S = 7200.0
#: Section 6's tolerances.
MEAN_LAG_TOLERANCE_S = 3.0
WINDOW_S = 300.0
WINDOW_FLOOR = 6
WINDOW_SHARE = 0.15
BURST = (900.0, 1200.0)
BURST_SHARE = 0.10
#: The burst check's run ends this long after the burst window, so its vehicles arrive.
BURST_TAIL_S = 900.0
TRACE_EVERY_S = 10.0
SEEDS = (1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_crossings(steps: Iterable[tuple[float, Mapping[str, str]]],
                    entry_of: Mapping[str, str]) -> dict[str, float]:
    """The first step time at which each vehicle is on its own entry link.

    `steps` is the per-step road id of each vehicle, `(t, {vehicle: road id})` in time
    order; `entry_of` is each vehicle's entry link. A vehicle never seen on its entry link
    is absent from the result.
    """
    first: dict[str, float] = {}
    for t, roads in steps:
        for vehicle, road in roads.items():
            if vehicle not in first and road == entry_of.get(vehicle):
                first[vehicle] = float(t)
    return first


def travel_stats(values: Iterable[float]) -> dict[str, Any]:
    """Count, mean, median, sample standard deviation and 5th and 95th percentiles (linear
    interpolation between order statistics)."""
    v = [float(x) for x in values]
    if not v:
        return {"n": 0, "mean_s": None, "median_s": None, "sd_s": None, "p5_s": None,
                "p95_s": None}
    return {"n": len(v), "mean_s": statistics.fmean(v), "median_s": statistics.median(v),
            "sd_s": statistics.stdev(v) if len(v) > 1 else 0.0,
            "p5_s": float(np.percentile(v, 5)), "p95_s": float(np.percentile(v, 95))}


def arrival_check(rows: list[dict], crossings: Mapping[str, float], W: float,
                  cap: float) -> dict[str, Any]:
    """Section 6's first check on one seed. `rows` is the `_A` schedule (with
    `entry_time`), `crossings` the vehicles' first steps on their entry links."""
    entries = sorted({r["entry"] for r in rows}, key=int)
    due_until = cap - WINDOW_S
    lags = {e: [] for e in entries}
    missing = []
    for r in rows:
        arrival = crossings.get(r["id"])
        if arrival is not None:
            lags[r["entry"]].append(arrival - r["entry_time"])
        elif r["entry_time"] <= due_until + 1e-9:
            missing.append(r["id"])
    windows = []
    start = W
    while start + WINDOW_S <= due_until + 1e-9:
        windows.append((start, start + WINDOW_S))
        start += WINDOW_S
    per_entry, worst_window, ok = {}, 0, True
    for e in entries:
        scheduled = np.array([r["entry_time"] for r in rows if r["entry"] == e])
        arrived = np.array([crossings[r["id"]] for r in rows
                            if r["entry"] == e and r["id"] in crossings])
        failures = []
        for low, high in windows:
            s = int(((scheduled >= low) & (scheduled < high)).sum())
            a = int(((arrived >= low) & (arrived < high)).sum())
            worst_window = max(worst_window, abs(a - s))
            if abs(a - s) > max(WINDOW_FLOOR, WINDOW_SHARE * s):
                failures.append({"window": [low, high], "scheduled": s, "arrived": a})
        mean_lag = statistics.fmean(lags[e]) if lags[e] else None
        lag_ok = mean_lag is not None and abs(mean_lag) <= MEAN_LAG_TOLERANCE_S
        ok = ok and lag_ok and not failures
        per_entry[e] = {"n": len(lags[e]), "mean_lag_s": mean_lag,
                        "sd_lag_s": statistics.stdev(lags[e]) if len(lags[e]) > 1 else None,
                        "min_lag_s": min(lags[e], default=None),
                        "max_lag_s": max(lags[e], default=None),
                        "mean_lag_ok": lag_ok, "window_failures": failures}
    passes = ok and not missing
    return {"passes": passes, "entries": per_entry, "windows": len(windows),
            "largest_window_difference": worst_window, "missing": missing,
            "due": sum(1 for r in rows if r["entry_time"] <= due_until + 1e-9)}


def burst_check(rows: list[dict], crossings: Mapping[str, float], queues: Mapping[str, list],
                W: float) -> dict[str, Any]:
    """Section 6's second check on one seed. `queues[entry]` is the approach's queue length
    at every trace sample in the burst window."""
    low, high = BURST[0] + W, BURST[1] + W
    per_entry, ok = {}, True
    for e in sorted({r["entry"] for r in rows}, key=int):
        mine = [r for r in rows if r["entry"] == e]
        scheduled = sum(1 for r in mine if low <= r["entry_time"] < high)
        arrived = sum(1 for r in mine if r["id"] in crossings
                      and low <= crossings[r["id"]] < high)
        lags = [crossings[r["id"]] - r["entry_time"] for r in mine
                if low <= r["entry_time"] < high and r["id"] in crossings]
        queue = max(queues.get(e) or [0.0])
        tested = queue <= 0.0
        within = scheduled > 0 and abs(arrived - scheduled) <= BURST_SHARE * scheduled
        if tested and not within:
            ok = False
        per_entry[e] = {"scheduled": scheduled, "arrived": arrived,
                        "difference_share": (arrived - scheduled) / scheduled if scheduled
                        else None,
                        "mean_lag_s": statistics.fmean(lags) if lags else None,
                        "max_queue_m": queue, "tested": tested,
                        "within_10_percent": within}
    return {"passes": ok, "window": [low, high], "entries": per_entry}


def run(scenario: str, seed: int, duration_s: float, *, queues: bool = False) -> dict[str, Any]:
    """One no-control episode on the approach boundary from t = 0, recording every
    vehicle's first step on its entry link and, with `queues`, each approach's queue every
    10 s."""
    from src.sumo.binding import _sumo
    from src.sumo.mainz import MainzEnv
    from train_mainz_src import SCENARIOS

    paths = SCENARIOS[scenario]
    rows = json.loads((REPO_ROOT / paths["schedule"]).read_text())
    entry_of = {r["id"]: r["entry"] for r in rows}
    env = MainzEnv(seed=seed, duration_s=duration_s, warmup_s=0.0, step_length_s=STEP_S,
                   gate_entries=False, net_file=REPO_ROOT / paths["net"],
                   route_file=REPO_ROOT / paths["routes"],
                   segment_file=REPO_ROOT / paths["segments"],
                   schedule_file=REPO_ROOT / paths["schedule"],
                   approach_file=REPO_ROOT / paths["approach"])
    entries = env.approach["entries"]
    inserted: dict[str, float] = {}
    trace: list[dict[str, Any]] = []
    pending = [0]

    def steps():
        """Step SUMO and yield `(t, {vehicle: entry link})` for the vehicles on one."""
        for _ in range(int(round(duration_s / STEP_S))):
            env._step()
            t = env.time_s()
            # SUMO's own departure time: the step at which it inserted the vehicle. The clock
            # read after `simulationStep` is one step later.
            for vehicle in _sumo.simulation.getDepartedIDList():
                inserted[vehicle] = float(_sumo.vehicle.getDeparture(vehicle))
            pending[0] = max(pending[0], env.pending())
            if queues and abs(t / TRACE_EVERY_S - round(t / TRACE_EVERY_S)) < 1e-9:
                report = env.approach_report()
                trace.append({"t": t, "queue_m": report["queue_m"],
                              "on_approach": report["on_approach"]})
            yield t, {v: e for e in entries for v in _sumo.edge.getLastStepVehicleIDs(e)}

    started = time.perf_counter()
    try:
        env.reset()
        crossings = entry_crossings(steps(), entry_of)
        colliding, teleports = env.colliding_vehicles, env.collision_teleports
    finally:
        env.close()
    pending_max = pending[0]
    late_insertions = sum(1 for r in rows if r["id"] in inserted
                          and inserted[r["id"]] - r["depart"] > STEP_S + 1e-9)
    return {"scenario": scenario, "seed": seed, "duration_s": duration_s, "rows": rows,
            "crossings": crossings, "inserted": len(inserted),
            "late_insertions": late_insertions, "pending_max": pending_max,
            "colliding_vehicles": colliding, "collision_teleports": teleports,
            "entries": entries, "trace": trace,
            "wall_s": time.perf_counter() - started,
            "files": {name: {"path": paths[name], "sha256": _sha256(REPO_ROOT / paths[name])}
                      for name in ("net", "routes", "schedule", "approach")}}


def calibrate(seed: int, out: Path) -> dict[str, Any]:
    meta = json.loads(APPROACH.read_text())
    W = float(meta["W"])
    result = run("mainz_bg_A", seed, DEMAND_S + W)
    by_entry: dict[str, list[float]] = {e: [] for e in result["entries"]}
    vehicles = []
    for r in result["rows"]:
        crossing = result["crossings"].get(r["id"])
        vehicles.append([r["id"], r["entry"], r["depart"], crossing])
        if crossing is not None:
            by_entry[r["entry"]].append(crossing - r["depart"])
    record = {
        "schema": "mainz_approach_travel/1", "scenario": "mainz_bg_A", "arm": "no_control",
        "seed": seed, "step_length_s": STEP_S, "duration_s": result["duration_s"],
        "statistic": "travel time from the scheduled far-end departure to the first 0.5 s "
                     "step on the entry link",
        "W_run": W, "T_A_run": meta["T_A"], "calibrated_run": meta.get("calibrated"),
        "net_sha256": result["files"]["net"]["sha256"], "files": result["files"],
        "entries": {e: travel_stats(by_entry[e]) for e in result["entries"]},
        "never_reached": sum(1 for v in vehicles if v[3] is None),
        "late_insertions": result["late_insertions"], "pending_max": result["pending_max"],
        "colliding_vehicles": result["colliding_vehicles"],
        "collision_teleports": result["collision_teleports"],
        "wall_s": result["wall_s"],
        "vehicles_columns": ["id", "entry", "far_end_depart_s", "first_entry_step_s"],
        "vehicles": vehicles,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return record


def check(seeds: tuple[int, ...], out: Path) -> dict[str, Any]:
    meta = json.loads(APPROACH.read_text())
    if not meta.get("calibrated"):
        raise SystemExit(f"{APPROACH} is the provisional first pass; rebuild with "
                         "--approach-travel before checking")
    W = float(meta["W"])
    report: dict[str, Any] = {"W": W, "T_A": meta["T_A"], "approach_sha256": _sha256(APPROACH),
                              "arrivals": {}, "burst": {}}
    for seed in seeds:
        bg = run("mainz_bg_A", seed, DEMAND_S + W)
        report["arrivals"][str(seed)] = {
            **arrival_check(bg["rows"], bg["crossings"], W, DEMAND_S + W),
            "pending_max": bg["pending_max"], "late_insertions": bg["late_insertions"],
            "colliding_vehicles": bg["colliding_vehicles"],
            "collision_teleports": bg["collision_teleports"], "wall_s": bg["wall_s"]}
        b9 = run("mainz_b9000_A", seed, BURST[1] + W + BURST_TAIL_S, queues=True)
        low, high = BURST[0] + W, BURST[1] + W
        queues = {e: [s["queue_m"][j] for s in b9["trace"] if low <= s["t"] < high]
                  for j, e in enumerate(b9["entries"])}
        report["burst"][str(seed)] = {
            **burst_check(b9["rows"], b9["crossings"], queues, W),
            "pending_max": b9["pending_max"], "colliding_vehicles": b9["colliding_vehicles"],
            "collision_teleports": b9["collision_teleports"], "wall_s": b9["wall_s"]}
    report["passes"] = (all(r["passes"] for r in report["arrivals"].values())
                        and all(r["passes"] for r in report["burst"].values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1, help="the calibration's seed")
    parser.add_argument("--check", action="store_true",
                        help="run section 6's arrival and burst checks instead")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    seeds = [args.seed] if not args.check else args.seeds
    outside = sorted(set(seeds) - set(SEEDS))
    if outside:
        raise SystemExit(f"the calibration and its checks run on training seeds 1 and 2 only; "
                         f"refusing {outside}")
    if args.check:
        out = Path(args.out) if args.out else CHECKS
        report = check(tuple(args.seeds), out)
        for seed, entry in report["arrivals"].items():
            lags = [v["mean_lag_s"] for v in entry["entries"].values()]
            print(f"seed {seed} arrivals: {'pass' if entry['passes'] else 'FAIL'}; mean lags "
                  f"{min(lags):+.2f} to {max(lags):+.2f} s; largest window difference "
                  f"{entry['largest_window_difference']}; missing {len(entry['missing'])} "
                  f"of {entry['due']}")
        for seed, entry in report["burst"].items():
            cells = ", ".join(f"{e} {v['difference_share']:+.0%} lag {v['mean_lag_s']:.1f} s"
                              + ("" if v["tested"] else f" (queue {v['max_queue_m']:.0f} m)")
                              for e, v in entry["entries"].items())
            print(f"seed {seed} burst: {'pass' if entry['passes'] else 'FAIL'}; {cells}")
        print(f"wrote {out}")
        return 0 if report["passes"] else 1
    out = Path(args.out) if args.out else TRAVEL
    record = calibrate(args.seed, out)
    for entry, stats in record["entries"].items():
        print(f"  {entry}: n {stats['n']}, mean {stats['mean_s']:.2f} s, median "
              f"{stats['median_s']:.2f}, sd {stats['sd_s']:.2f}, p5 {stats['p5_s']:.2f}, "
              f"p95 {stats['p95_s']:.2f}")
    print(f"wrote {out} ({record['wall_s']:.1f} s; {record['never_reached']} never reached "
          f"an entry link; pending at most {record['pending_max']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
