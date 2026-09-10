"""Under a demand above capacity, does control raise throughput and lower latent demand?

The accounting is complete: every scheduled vehicle is served, on the road, or
latent (waiting to enter). A controller has to move the first up and the third
down; moving vehicles from the road into the queue is not an improvement.

It also tests a structural hypothesis about this topology. The six entry leaves are
SINGLE LANE, so an AV that slows there cannot be overtaken and is a rolling
roadblock rather than a meter -- ramp metering works because the meter sits beside
the road, not in it. `--leaf-lanes 2` rebuilds the leaves with two lanes, which is
the smallest change that lets an AV hold back its own lane without blocking the
others.

    .venv/bin/python scripts/measure_saturated_control.py --leaf-lanes 1 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

SEEDS = (7, 17, 27, 37, 47)


def run(*, arm, speed, seed, leaf_lanes, dt, duration_steps, warmup_steps, demand_name,
        work_dir):
    topology = dict(load_named_config("topology", "inverted_tree"))
    road = dict(topology["road"])
    road["lane_counts"] = {**road["lane_counts"], "leaves": int(leaf_lanes)}
    topology["road"] = road
    demand = dict(load_named_config("demand", demand_name))
    demand["av_penetration"] = 0.0 if arm == "no_av" else 0.2
    env = SumoTopologyEnv("inverted_tree", {
        "topology": topology, "demand": demand,
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": duration_steps, "dt": dt, "warmup_steps": warmup_steps,
        "work_dir": work_dir})
    env.reset(seed=seed)
    try:
        downstream = env.view.downstream_segments()
        latent = []
        for _ in range(duration_steps):
            if arm != "no_av":
                segments = env.get_segment_metrics()
                busy = {name for name, m in segments.items()
                        if m["jam_fraction"] > 0.25 or m["queue_length"] > 0}
                for snapshot in env.vehicle_snapshots():
                    if snapshot.role != "av" or snapshot.vehicle_id not in env.agent_ids:
                        continue
                    if arm == "metering" and not (
                            set(downstream.get(snapshot.segment_id, ())) & busy):
                        continue
                    env.command_speed(snapshot.vehicle_id, speed)
            _, _, _, _, info = env.step({})
            latent.append(info["metrics"]["latent_demand"])
        return {"served": env.arrived_total, "latent": latent[-1],
                "latent_mean": statistics.fmean(latent)}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", default="sumo_oversaturated")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration-steps", type=int, default=6000)
    # Long enough that the road is already saturated and latent demand is growing
    # when the episode starts: at 3000 veh/h that takes about 1200 s.
    parser.add_argument("--warmup-steps", type=int, default=12000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--leaf-lanes", type=int, nargs="+", default=[1])
    parser.add_argument("--speeds", type=float, nargs="+", default=[10.0, 16.0])
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    kw = dict(dt=args.dt, duration_steps=args.duration_steps,
              warmup_steps=args.warmup_steps, demand_name=args.demand,
              work_dir=args.work_dir)
    print(f"{args.demand}, dt {args.dt}, {args.duration_steps * args.dt:.0f} s episodes "
          f"after {args.warmup_steps * args.dt:.0f} s of warm-up, seeds {tuple(args.seeds)}.")
    print("Served should rise and latent should fall. Both, or neither counts.\n")
    for leaf_lanes in args.leaf_lanes:
        print(f"  leaves with {leaf_lanes} lane(s):")
        print(f"    {'arm':>22} {'served':>8} {'sd':>6} {'vs no_av':>9} "
              f"{'latent':>8} {'vs no_av':>9}")
        base = [run(arm="no_av", speed=0.0, seed=s, leaf_lanes=leaf_lanes, **kw)
                for s in args.seeds]
        bs = [r["served"] for r in base]
        bl = [r["latent"] for r in base]
        print(f"    {'no_av':>22} {statistics.fmean(bs):>8.1f} "
              f"{statistics.stdev(bs):>6.1f} {'-':>9} {statistics.fmean(bl):>8.0f} {'-':>9}")
        for arm in ("metering", "always"):
            for speed in args.speeds:
                rows = [run(arm=arm, speed=speed, seed=s, leaf_lanes=leaf_lanes, **kw)
                        for s in args.seeds]
                served = [r["served"] for r in rows]
                lat = [r["latent"] for r in rows]
                ds = statistics.fmean(v - b for v, b in zip(served, bs))
                dl = statistics.fmean(v - b for v, b in zip(lat, bl))
                print(f"    {f'{arm} at {speed:g} m/s':>22} {statistics.fmean(served):>8.1f} "
                      f"{statistics.stdev(served):>6.1f} {ds:>+9.1f} "
                      f"{statistics.fmean(lat):>8.0f} {dl:>+9.0f}", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
