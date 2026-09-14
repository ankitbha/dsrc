#!/usr/bin/env python3
"""Why the throughput window opens at 900 s, and what the vehicles after it do.

Both answers come out of the trip records `measure_mainz_trip_speeds.py` already
wrote, so this runs no simulation. A `tripinfo` record carries the insertion time
(`depart`), the time in the network (`duration`) and the distance driven
(`routeLength`), which is enough for two reconstructions:

  Served flow, per window. The count of trips whose `arrival` falls in each
  window. This is the quantity the paper's throughput figure measures, and it is
  zero until the first trips complete.

  Occupancy, per instant. A vehicle occupies the network over
  `[depart, depart + duration)`, so counting the intervals covering t gives
  vehicles-in-network at t. Cross-checked against insertions minus arrivals
  accumulated over the same windows, which is an independent path to the same
  number.

The two disagree about what 900 s means, and the disagreement is the point.
Served flow reaches its sustained level by then. Occupancy does not settle at
all: insertions exceed arrivals in every window of the episode, so the network
accumulates from the first second to the last. The window makes the throughput
figure a measurement of the controller rather than of the fill. It does not make
the episode a steady state, and nothing in the run does.

    python3 scripts/measure_mainz_window_start.py --tripinfo <dir> --out <dir>

`--tripinfo` is the directory `measure_mainz_trip_speeds.py --out` wrote its
`tripinfo/` subdirectory into.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

#: Copied from the published evaluation so these figures are comparable to it.
WINDOW_START_S = 900.0
DURATION_S = 2500.0
ARMS = ("no_control", "here", "src")


def read_trips(path: Path) -> list[dict]:
    trips = []
    for el in ET.parse(path).getroot():
        if el.tag != "tripinfo":
            continue
        arrival = el.get("arrival")
        trips.append({
            # SUMO's own insertion time. Not the route file's requested depart,
            # which is what `departDelay` measures the difference against.
            "depart_s": float(el.get("depart")),
            "duration_s": float(el.get("duration", 0.0)),
            "route_length_m": float(el.get("routeLength", 0.0)),
            "time_loss_s": float(el.get("timeLoss", 0.0)),
            "arrival_s": float(arrival) if arrival not in (None, "", "-1") else None,
        })
    return trips


def speed_summary(trips: list[dict]) -> dict:
    """Two speeds, because they answer different questions.

    The aggregate is total distance over total time, which is the network's mean
    speed over everyone. The per-vehicle mean weights a short trip the same as a
    long one.
    """
    usable = [t for t in trips if t["duration_s"] > 0]
    total_m = sum(t["route_length_m"] for t in usable)
    total_s = sum(t["duration_s"] for t in usable)
    per_vehicle = [t["route_length_m"] / t["duration_s"] * 3.6 for t in usable]
    return {
        "vehicles": len(trips),
        "arrived": sum(1 for t in trips if t["arrival_s"] is not None),
        "aggregate_speed_kmh": (total_m / total_s * 3.6) if total_s else None,
        "per_vehicle_mean_speed_kmh": statistics.fmean(per_vehicle) if per_vehicle else None,
        "per_vehicle_median_speed_kmh": statistics.median(per_vehicle) if per_vehicle else None,
        "mean_time_loss_s": (
            statistics.fmean([t["time_loss_s"] for t in trips]) if trips else None
        ),
    }


def paired_difference(arm_rows: list[dict], base_rows: list[dict], key: str) -> dict:
    """Mean and two standard errors of the per-seed difference.

    Paired on seed, because each arm runs the same traffic realisation under the
    same seed and an unpaired interval would carry the between-seed spread that
    the pairing removes.
    """
    diffs = [a[key] - b[key] for a, b in zip(arm_rows, base_rows)]
    mean = statistics.fmean(diffs)
    two_se = 2.0 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {"mean": mean, "two_standard_errors": two_se,
            "resolved": abs(mean) > two_se}


def served_flow(per_seed: list[list[dict]], width_s: float) -> list[dict]:
    edges = [w * width_s for w in range(int(DURATION_S // width_s) + 1)]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        arrived = statistics.fmean([
            sum(1 for t in trips if t["arrival_s"] is not None and lo <= t["arrival_s"] < hi)
            for trips in per_seed
        ])
        inserted = statistics.fmean([
            sum(1 for t in trips if lo <= t["depart_s"] < hi) for trips in per_seed
        ])
        out.append({"from_s": lo, "to_s": hi, "inserted": inserted,
                    "arrived": arrived, "net": inserted - arrived})
    return out


def occupancy(per_seed: list[list[dict]], step_s: float) -> list[dict]:
    out = []
    for t in range(0, int(DURATION_S) + 1, int(step_s)):
        counts = [
            sum(1 for tr in trips
                if tr["depart_s"] <= t < tr["depart_s"] + tr["duration_s"])
            for trips in per_seed
        ]
        out.append({"t_s": float(t), "vehicles": statistics.fmean(counts)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tripinfo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", nargs=2, type=int, default=[16, 30])
    args = ap.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    trips_by_arm = {
        arm: [read_trips(args.tripinfo / f"{arm}_seed{s}.xml") for s in seeds]
        for arm in ARMS
    }

    speeds: dict[str, dict] = {}
    per_seed_rows: dict[str, dict[str, list[dict]]] = {}
    for arm, per_seed in trips_by_arm.items():
        every = [speed_summary(t) for t in per_seed]
        after = [speed_summary([x for x in t if x["depart_s"] >= WINDOW_START_S])
                 for t in per_seed]
        per_seed_rows[arm] = {"every_vehicle": every, "departed_after_window": after}
        speeds[arm] = {
            population: {
                key: statistics.fmean([r[key] for r in rows if r[key] is not None])
                for key in rows[0]
            }
            for population, rows in per_seed_rows[arm].items()
        }

    paired = {}
    for arm in ("here", "src"):
        paired[arm] = {
            population: {
                key: paired_difference(per_seed_rows[arm][population],
                                       per_seed_rows["no_control"][population], key)
                for key in ("aggregate_speed_kmh", "per_vehicle_mean_speed_kmh",
                            "mean_time_loss_s")
            }
            for population in ("every_vehicle", "departed_after_window")
        }

    flow = served_flow(trips_by_arm["no_control"], 200.0)
    occ = occupancy(trips_by_arm["no_control"], 50.0)
    at_window = next(p["vehicles"] for p in occ if p["t_s"] == WINDOW_START_S)
    peak = max(p["vehicles"] for p in occ)
    sustained = [w["arrived"] for w in flow if w["from_s"] >= 800.0]

    payload = {
        "what": "Why the throughput window opens at 900 s, and the speed of the "
                "vehicles that entered after it. Reconstructed from the trip "
                "records of the published evaluation; no simulation was run.",
        "generator": "scripts/measure_mainz_window_start.py",
        "config": {"seeds": seeds, "duration_s": DURATION_S,
                   "window_start_s": WINDOW_START_S,
                   "tripinfo_source": str(args.tripinfo)},
        "why_the_window_opens_at_900s": {
            "statement":
                "Served flow is zero until the first trips complete and reaches "
                "its sustained level by the window start. Over 200 s windows the "
                "no-control arm completes 0, 43, 79 and 139 trips in the four "
                "windows before 800 s, and between "
                f"{min(sustained):.0f} and {max(sustained):.0f} in every window "
                "after. Measuring throughput from t=0 would average the fill "
                "into the result.",
            "served_flow_per_200s": flow,
        },
        "the_network_does_not_reach_a_steady_state": {
            "statement":
                "Insertions exceed arrivals in every window of the episode, so "
                "occupancy rises throughout: "
                f"{at_window:.0f} vehicles at the window start and "
                f"{peak:.0f} at the peak, which is the last sample. The window "
                "makes the throughput figure a measurement of the controller "
                "rather than of the fill. It does not make the episode a steady "
                "state, and nothing in the run does.",
            "occupancy_at_window_start": at_window,
            "peak_occupancy": peak,
            "occupancy_per_50s": occ,
        },
        "speeds": speeds,
        "paired_vs_no_control": paired,
        "per_seed": per_seed_rows,
    }
    (out / "mainz_window_start.json").write_text(json.dumps(payload, indent=1) + "\n")

    print(f"{'arm':<11} {'n/seed':>7} {'aggregate':>10} {'per-veh':>9} {'median':>8}")
    for population in ("every_vehicle", "departed_after_window"):
        print(f"\n-- {population} " + "-" * 40)
        for arm in ARMS:
            d = speeds[arm][population]
            print(f"{arm:<11} {d['vehicles']:7.1f} {d['aggregate_speed_kmh']:9.2f} "
                  f"{d['per_vehicle_mean_speed_kmh']:8.2f} "
                  f"{d['per_vehicle_median_speed_kmh']:7.2f}")
    print(f"\nwrote {out / 'mainz_window_start.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
