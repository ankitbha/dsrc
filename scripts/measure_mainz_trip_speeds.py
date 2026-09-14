#!/usr/bin/env python3
"""Mean speed over every vehicle, completed trips included, for the three Mainz arms.

The evaluation artefact reports `space_mean_speed_kmh`, which is the vehicle-weighted
mean over the vehicles *inside* the network at each sample. A vehicle that has reached
its destination has left, so it stops contributing. That statistic therefore describes
the resident population and not the demand.

This measures the quantity that does not have that gap: SUMO's own per-vehicle trip
record, over every vehicle that departed, with unfinished trips written as well. Two
figures come out of it. The aggregate is the total distance driven over the total time
driven, which is the network's mean speed including everyone. The per-vehicle mean
averages each vehicle's own distance over its own time, which weights a short trip the
same as a long one. They answer different questions and both are reported.

`departDelay` is reported beside them, because it is the part the in-network statistics
cannot see at all: time a vehicle spent waiting to be let in, whether held by the
policy's entry gate or refused insertion by the simulator for want of a safe gap.

    python3 scripts/measure_mainz_trip_speeds.py --out <dir> [--seeds 16 30] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (str(REPO), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_mainz_src import SCENARIOS, evaluate_no_control, run_episode  # noqa: E402
from evaluate_mainz_checkpoints import CHECKPOINT_FEATURES, load_arm_model  # noqa: E402
from src.sumo import binding  # noqa: E402
from src.sumo.mainz import HERE_FEATURES  # noqa: E402

#: The published evaluation's configuration, copied so these runs are comparable to it.
DURATION_S = 2500.0
STEP_LENGTH = 0.5
WINDOW_START_S = 900.0

_tripinfo_path: dict[str, Path | None] = {"path": None}


def _patch_sumo_start() -> None:
    """Append the trip-record options to whatever command the environment builds.

    Wrapping the call rather than copying `MainzEnv.reset` keeps every other option
    exactly as the environment sets it, so nothing here can drift from the runs the
    published figures came from.
    """
    original = binding._sumo.start

    def start(cmd, *args, **kwargs):
        target = _tripinfo_path["path"]
        if target is not None:
            cmd = list(cmd) + [
                "--tripinfo-output", str(target),
                # Without this a vehicle still on the network at the end is absent
                # from the record, which would drop exactly the slow ones.
                "--tripinfo-output.write-unfinished", "true",
            ]
        return original(cmd, *args, **kwargs)

    binding._sumo.start = start


def read_trips(path: Path) -> list[dict]:
    trips = []
    for el in ET.parse(path).getroot():
        if el.tag != "tripinfo":
            continue
        duration = float(el.get("duration", 0.0))
        length = float(el.get("routeLength", 0.0))
        trips.append({
            "id": el.get("id"),
            "duration_s": duration,
            "route_length_m": length,
            "depart_delay_s": float(el.get("departDelay", 0.0)),
            "time_loss_s": float(el.get("timeLoss", 0.0)),
            "arrived": el.get("arrival") not in (None, "", "-1"),
            "speed_kmh": (length / duration * 3.6) if duration > 0 else None,
        })
    return trips


def summarise(trips: list[dict]) -> dict:
    speeds = [t["speed_kmh"] for t in trips if t["speed_kmh"] is not None]
    total_m = sum(t["route_length_m"] for t in trips)
    total_s = sum(t["duration_s"] for t in trips)
    arrived = [t for t in trips if t["arrived"]]
    return {
        "vehicles": len(trips),
        "arrived": len(arrived),
        "still_running_at_end": len(trips) - len(arrived),
        # Total distance over total time: the network's mean speed over everyone.
        "aggregate_speed_kmh": (total_m / total_s * 3.6) if total_s else None,
        # Each vehicle weighted equally, whatever the length of its trip.
        "per_vehicle_mean_speed_kmh": statistics.fmean(speeds) if speeds else None,
        "per_vehicle_median_speed_kmh": statistics.median(speeds) if speeds else None,
        "total_distance_km": total_m / 1000.0,
        "mean_depart_delay_s": statistics.fmean([t["depart_delay_s"] for t in trips]) if trips else None,
        "mean_time_loss_s": statistics.fmean([t["time_loss_s"] for t in trips]) if trips else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", nargs=2, type=int, default=[16, 30])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-dir", type=Path, default=REPO / "results" / "checkpoints")
    args = ap.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    if args.limit:
        seeds = seeds[: args.limit]
    out = args.out.expanduser()
    (out / "tripinfo").mkdir(parents=True, exist_ok=True)

    _patch_sumo_start()
    paths = SCENARIOS["mainz"]
    loaded = {name: load_arm_model(name, args.checkpoint_dir, 12)
              for name in ("here", "src")}

    per_arm: dict[str, list[dict]] = {}
    for arm in ("no_control", "here", "src"):
        per_seed = []
        for seed in seeds:
            target = out / "tripinfo" / f"{arm}_seed{seed}.xml"
            _tripinfo_path["path"] = target
            if arm == "no_control":
                evaluate_no_control((seed,), HERE_FEATURES, DURATION_S,
                                    STEP_LENGTH, WINDOW_START_S, paths)
            else:
                a = loaded[arm]
                run_episode(a.model, seed, a.features, DURATION_S, 0.0, None,
                            STEP_LENGTH, WINDOW_START_S, True, paths)
            s = summarise(read_trips(target))
            s["seed"] = seed
            per_seed.append(s)
            print(f"  {arm:11} seed {seed}: {s['vehicles']:4d} vehicles, "
                  f"aggregate {s['aggregate_speed_kmh']:.2f} km/h, "
                  f"per-vehicle {s['per_vehicle_mean_speed_kmh']:.2f} km/h, "
                  f"depart delay {s['mean_depart_delay_s']:.1f} s", flush=True)
        per_arm[arm] = per_seed

    summary = {}
    for arm, rows in per_arm.items():
        summary[arm] = {
            key: statistics.fmean([r[key] for r in rows])
            for key in ("vehicles", "arrived", "still_running_at_end",
                        "aggregate_speed_kmh", "per_vehicle_mean_speed_kmh",
                        "per_vehicle_median_speed_kmh", "total_distance_km",
                        "mean_depart_delay_s", "mean_time_loss_s")
        }
    payload = {
        "what": "Mean speed over every vehicle that departed, completed trips included, "
                "for the three Mainz arms over the evaluation seeds.",
        "config": {"seeds": seeds, "duration_s": DURATION_S,
                   "step_length_s": STEP_LENGTH, "window_start_s": WINDOW_START_S},
        "per_seed": per_arm,
        "mean_over_seeds": summary,
    }
    (out / "trip_speeds.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'trip_speeds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
