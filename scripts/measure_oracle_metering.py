"""Is there an exploitable inefficiency on this network, for ANY controller?

The uniform-speed oracle showed no gain at a converged step size (task 92), but a
uniform command is the weakest possible controller: it cannot respond to where the
congestion is. This asks the prior question that a training result cannot answer --
whether a controller with PERFECT information and the project's own declared
mechanism can beat doing nothing.

The mechanism is backpressure-inspired speed metering, which the project names as
legitimate: an AV slows only while the segment DOWNSTREAM of it is congested, and
otherwise does nothing. It reads the true simulator state, not the sensing model,
and it needs no learning, so a null here is about the network and a gain here is
about learning or sensing.

    .venv/bin/python scripts/measure_oracle_metering.py
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


def run(*, arm, metered_speed, seed, dt, duration_steps, warmup_steps, demand, work_dir):
    env = SumoTopologyEnv("inverted_tree", {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", demand),
        "duration_steps": duration_steps, "dt": dt,
        "warmup_steps": warmup_steps, "work_dir": work_dir})
    env.reset(seed=seed)
    try:
        downstream = env.view.downstream_segments()
        for _ in range(duration_steps):
            if arm != "no_av":
                segments = env.get_segment_metrics()
                congested = {
                    segment for segment, metrics in segments.items()
                    if metrics["jam_fraction"] > 0.25 or metrics["queue_length"] > 0
                }
                for snapshot in env.vehicle_snapshots():
                    if snapshot.role != "av" or snapshot.vehicle_id not in env.agent_ids:
                        continue
                    ahead = set(downstream.get(snapshot.segment_id, ()))
                    if arm == "metering" and not (ahead & congested):
                        continue  # nothing congested ahead: do not interfere
                    env.command_speed(snapshot.vehicle_id, metered_speed)
            env.step({})
        return env.arrived_total
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", default="sumo_burst")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration-steps", type=int, default=9000)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--speeds", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    print(f"{args.demand}, dt {args.dt}, {args.duration_steps * args.dt:.0f} s episodes, "
          f"seeds {tuple(args.seeds)}. Arrivals.\n")
    kw = dict(dt=args.dt, duration_steps=args.duration_steps,
              warmup_steps=args.warmup_steps, demand=args.demand, work_dir=args.work_dir)
    baseline = [run(arm="no_av", metered_speed=0.0, seed=s, **kw) for s in args.seeds]
    print(f"  {'arm':>28} {'arrivals':>9} {'sd':>6} {'vs no_av':>10} {'paired sd':>10}")
    print(f"  {'no_av':>28} {statistics.fmean(baseline):>9.1f} "
          f"{statistics.stdev(baseline):>6.1f} {'-':>10} {'-':>10}")
    for arm in ("metering", "always"):
        for speed in args.speeds:
            values = [run(arm=arm, metered_speed=speed, seed=s, **kw) for s in args.seeds]
            paired = [v - b for v, b in zip(values, baseline)]
            label = (f"metering at {speed:g} m/s" if arm == "metering"
                     else f"always {speed:g} m/s")
            print(f"  {label:>28} {statistics.fmean(values):>9.1f} "
                  f"{statistics.stdev(values):>6.1f} {statistics.fmean(paired):>+10.1f} "
                  f"{statistics.stdev(paired):>10.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
