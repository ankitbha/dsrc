#!/usr/bin/env python3
"""Does Mainz's SERVED FLOW rise to a peak and fall? The gate before any training run.

    .venv/bin/python scripts/measure_mainz_fundamental_diagram.py

SRC's mechanism is the capacity drop. Flow rises with density to a maximum and falls
beyond it, so a controller that holds density below critical recovers the flow the
collapse would have lost. If served flow does not fall, there is nothing to recover and
no controller can gain throughput however well trained. This is the measurement that
decides whether an arm comparison can say anything, and it must be made in the
configuration the arms will be trained in.

IT WAS PREVIOUSLY MADE IN A DIFFERENT ONE, which is the mistake this script exists to
stop repeating. `measure_fd_exit.py` traces the curve on the exit super-segment under a
rush that ramps from a quarter of its peak and back, so the exit meter alternately binds
and releases and the measurement passes through every traffic state. Training then ran
under the published demand, a flat 18,000 veh/h for 1,200 s, where the meter is
saturated for the whole episode: the exit road held 48 to 50 vehicles in every arm, a
saturated signal discharges at its own rate whatever is upstream, and served flow was
clamped near 2,150 veh/h in all three arms. A curve measured under one demand does not
license an experiment under another.

So demand here is FLAT AND SUSTAINED for the whole episode, which is what training uses,
and flow is counted over a window that opens after the network has filled. The exit
configuration is swept alongside, because where the exit constraint sits relative to the
network's own capacity decides whether the binding bottleneck is inside the network or
at its boundary.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The .inpx demand, which every swept rate is expressed against.
BASE_VEH_PER_HOUR = 18000.0


def build(rate: float, green: float | None, duration_s: float,
          block_after_s: float | None) -> None:
    command = [str(REPO_ROOT / ".venv" / "bin" / "python"),
               str(REPO_ROOT / "scripts" / "build_mainz_scenario.py"),
               "--av-fraction", "1.0",
               "--duration-s", f"{duration_s:.0f}",
               "--demand-scale", f"{rate / BASE_VEH_PER_HOUR:.6f}",
               "--demand-duration-s", f"{duration_s:.0f}"]
    command += (["--no-exit-meter"] if green is None
                else ["--exit-green-fraction", f"{green:.4f}"])
    if block_after_s is not None:
        command += ["--junction-block-after-s", f"{block_after_s:g}"]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit("building the scenario failed")


def run(rate: float, green: float | None, seed: int, args) -> dict:
    import libsumo as traffic

    import src.sumo.mainz as mainz

    build(rate, green, args.duration_s, args.block_after_s)
    # No control, and no entry gate: the gate stands in for a policy, and this is the
    # network's own behaviour that any policy would have to improve on.
    env = mainz.MainzEnv(seed=seed, duration_s=args.duration_s, features=("speed",),
                         step_length_s=args.step, gate_entries=False)
    try:
        env.reset()
        env._advance(args.window_start)
        env.mark_window()
        lane_km = sum(env._static[edge]["length_m"] * env._static[edge]["lanes"]
                      for segment in env.segments for edge in segment) / 1000.0
        while float(traffic.simulation.getTime()) < args.duration_s:
            env.release()
            env._advance(mainz.DECISION_INTERVAL_S)
        metrics = env.metrics()
        queue = sum(int(traffic.edge.getLastStepVehicleNumber(edge))
                    for edge in ("exit_road", "exit_tail")
                    if edge in traffic.edge.getIDList())
    finally:
        env.close()
    return {
        "served": metrics["flow_veh_per_h"],
        "vehicles": metrics["mean_vehicles"],
        "density": metrics["mean_vehicles"] / lane_km,
        "speed": metrics["space_mean_speed_kmh"],
        "held": metrics["waiting_to_enter"],
        "exit_queue": queue,
    }


def verdict(rows: list[tuple[float, dict]]) -> None:
    peak_rate, peak = max(rows, key=lambda row: row[1]["served"])
    beyond = [row for row in rows if row[1]["density"] > peak["density"]]
    print(f"\n  peak served flow {peak['served']:.0f} veh/h at "
          f"{peak['density']:.1f} veh/km/lane ({peak_rate:.0f} veh/h offered)")
    if not beyond:
        print("  FAILS: the peak is at the highest density reached, so nothing was")
        print("  sampled past capacity and no capacity drop is in range.")
        return
    worst = min(beyond, key=lambda row: row[1]["served"])[1]
    fall = 100.0 * (peak["served"] - worst["served"]) / peak["served"]
    print(f"  lowest served flow past the peak {worst['served']:.0f} veh/h at "
          f"{worst['density']:.1f} veh/km/lane, a fall of {fall:.1f}%")
    # Only the saturated points say anything about capacity. Below saturation served
    # flow equals demand, so including those points reports the sweep's own range as
    # though it were variation in capacity.
    saturated = [row for rate, row in rows if row["served"] < 0.95 * rate]
    if fall < 2.0:
        if saturated:
            low = min(r["served"] for r in saturated)
            high = max(r["served"] for r in saturated)
            print(f"  FAILS: over the {len(saturated)} points where demand exceeds what "
                  f"the network serves, served flow stays between {low:.0f} and "
                  f"{high:.0f} veh/h, a range of {100.0 * (high - low) / high:.1f}%, "
                  f"while density runs "
                  f"{min(r['density'] for r in saturated):.1f} to "
                  f"{max(r['density'] for r in saturated):.1f} veh/km/lane and speed "
                  f"falls from {max(r['speed'] for r in saturated):.1f} to "
                  f"{min(r['speed'] for r in saturated):.1f} km/h.")
        print("  Served flow is a constant set by a saturated bottleneck rather than a")
        print("  function of density. No controller can recover flow never lost.")
    else:
        print("  PASSES: served flow rises to a peak and falls, so there is a capacity")
        print("  drop for a controller to prevent.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--duration-s", type=float, default=2500.0)
    parser.add_argument("--window-start", type=float, default=900.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[16])
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[1800, 2400, 3000, 3600, 4800, 6000, 9000, 18000])
    parser.add_argument("--block-after-s", type=float, default=None,
                        help="pass through to --junction-block-after-s, so the sweep "
                             "can be run on a network whose junctions block")
    parser.add_argument("--greens", nargs="+", default=["none", "0.98", "0.85"],
                        help="exit meter green fractions; 'none' leaves the exit "
                             "unmetered, so the network's own capacity is the limit")
    args = parser.parse_args()

    for label in args.greens:
        green = None if label == "none" else float(label)
        print(f"\n=== exit {'unmetered' if green is None else f'metered at {green} green'}"
              f", flat demand, no control, dt {args.step} ===")
        print(f"  {'offered':>8} {'served':>8} {'ratio':>7} {'density':>8} {'speed':>7} "
              f"{'vehicles':>9} {'held':>7} {'exit q':>7}")
        rows = []
        for rate in args.rates:
            runs = [run(rate, green, seed, args) for seed in args.seeds]
            row = {k: statistics.fmean(r[k] for r in runs) for k in runs[0]}
            rows.append((rate, row))
            print(f"  {rate:>8.0f} {row['served']:>8.0f} {row['served']/rate:>7.2f} "
                  f"{row['density']:>8.1f} {row['speed']:>7.2f} {row['vehicles']:>9.0f} "
                  f"{row['held']:>7.0f} {row['exit_queue']:>7.0f}", flush=True)
        verdict(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
