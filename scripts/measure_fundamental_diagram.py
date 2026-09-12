"""The flow-density relationship, which decides whether control has anything to do.

Every mixed-autonomy control result exploits a CAPACITY DROP: served flow falling
once density passes a critical point, so that a controller which holds density
below it recovers throughput. If served flow rises monotonically with demand there
is no such opportunity, and a perfect-information oracle cannot beat inaction --
which is what was measured on this network with SUMO's default Krauss model.

The predecessor paper (arXiv:2506.11973, Appendix A) tuned Wiedemann-99 for exactly
this: CC2 and CC6 "help introduce stop-and-go dynamics", CC4 and CC5 "influence
capacity drop effects in the fundamental diagram". This sweeps demand and reports
density, speed and served flow so the curve can be seen rather than assumed.

    .venv/bin/python scripts/measure_fundamental_diagram.py --human-model w99_calibrated
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

SEEDS = (7, 17, 27)


def run(*, rate, seed, topology, human_model, dt, duration_steps, warmup_steps, work_dir):
    demand = dict(load_named_config("demand", "sumo_saturating"))
    demand["total_vehicles_per_hour"] = float(rate)
    demand["av_penetration"] = 0.0
    config = {
        "topology": load_named_config("topology", topology),
        "demand": demand, "duration_steps": duration_steps, "dt": dt,
        "warmup_steps": warmup_steps, "work_dir": work_dir,
    }
    if human_model:
        config["human_model"] = load_named_config("human_model", human_model)
    env = SumoTopologyEnv(topology, config)
    env.reset(seed=seed)
    try:
        densities, speeds = [], []
        for _ in range(duration_steps):
            _, _, _, _, info = env.step({})
            metrics = info["metrics"]
            segments = env.get_segment_metrics()
            occupied = [m for m in segments.values() if m["vehicle_count"]]
            if occupied:
                densities.append(statistics.fmean(m["density"] for m in occupied))
            speeds.append(metrics["mean_speed"])
        return {
            "arrived": env.arrived_total,
            "density": statistics.fmean(densities) if densities else 0.0,
            "speed": statistics.fmean(speeds),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="inverted_tree")
    parser.add_argument("--human-model", default=None,
                        help="omit for SUMO's default Krauss model")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration-steps", type=int, default=6000)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--rates", type=int, nargs="+",
                        default=[600, 900, 1200, 1500, 1800, 2100, 2400, 3000])
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    seconds = args.duration_steps * args.dt
    print(f"{args.topology}, car following: {args.human_model or 'Krauss (SUMO default)'}, "
          f"dt {args.dt}, {seconds:.0f} s episodes, seeds {tuple(args.seeds)}.\n")
    # TWO STANDARD ERRORS ACROSS SEEDS, on every row. The curve recorded in
    # configs/demand/sumo_saturating.yaml rises non-monotonically -- 948, 1130, 1072,
    # 1298, 1052 -- which is either a real shape or seed noise, and three seeds with no
    # spread reported cannot tell the two apart. A peak that is not separated from its
    # neighbours by more than this bar is not a peak.
    print(f"  {'offered':>8} {'density':>9} {'speed':>8} {'served/h':>9} "
          f"{'+/-2se':>8} {'served/offered':>15}")
    measured = []
    for rate in args.rates:
        rows = [run(rate=rate, seed=s, topology=args.topology,
                    human_model=args.human_model, dt=args.dt,
                    duration_steps=args.duration_steps, warmup_steps=args.warmup_steps,
                    work_dir=args.work_dir) for s in args.seeds]
        flows = [r["arrived"] * (3600.0 / seconds) for r in rows]
        served = statistics.fmean(flows)
        error = (2.0 * statistics.stdev(flows) / len(flows) ** 0.5
                 if len(flows) > 1 else float("nan"))
        measured.append({"rate": rate, "served": served, "error": error,
                         "density": statistics.fmean(r["density"] for r in rows),
                         "speed": statistics.fmean(r["speed"] for r in rows)})
        print(f"  {rate:>8} {measured[-1]['density']:>9.1f} "
              f"{measured[-1]['speed']:>8.2f} {served:>9.0f} {error:>8.0f} "
              f"{served / rate:>14.2f}", flush=True)

    # CAPACITY IS THE BEST FLOW WHILE DEMAND WAS STILL BEING MET, not the largest
    # number in the column. Once the network saturates, a later point can read higher
    # than the pre-breakdown one through noise or a transient, and taking the global
    # maximum then places the peak past the breakdown and reports that nothing was
    # sampled beyond it.
    free = [row for row in measured if row["served"] >= 0.95 * row["rate"]]
    peak = max(free or measured, key=lambda row: row["served"])
    beyond = [row for row in measured if row["rate"] > peak["rate"]]
    print(f"\n  peak served flow {peak['served']:.0f} veh/h at "
          f"{peak['density']:.1f} veh/km/lane ({peak['rate']} offered)")
    if not beyond:
        print("  FAILS: the peak is at the highest density reached, so nothing was")
        print("  sampled past capacity.")
        return 0
    worst = min(beyond, key=lambda row: row["served"])
    fall = peak["served"] - worst["served"]
    bar = (peak["error"] ** 2 + worst["error"] ** 2) ** 0.5
    print(f"  lowest served flow past the peak {worst['served']:.0f} veh/h at "
          f"{worst['density']:.1f} veh/km/lane: a fall of {fall:.0f} veh/h "
          f"({100.0 * fall / peak['served']:.1f}%) against a combined bar of "
          f"{bar:.0f} veh/h")
    if fall > bar:
        print("  PASSES: served flow rises to a peak and falls by more than the seed")
        print("  spread, so there is a capacity drop for a controller to prevent.")
    else:
        print("  FAILS: the fall is inside the seed spread, so it is not resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
