"""Is the throughput effect physics, or an artefact of the simulation step?

At dt 1.0 an uncommanded fleet completes 132.0 trips per 600 s and AVs held at
10 m/s complete 155.2, a 17.6% gain. At dt 0.1 the uncommanded fleet completes
168.4 and the same commanded arm 161.2 -- the gain is gone and the baseline is 28%
higher. A vehicle at 24 m/s advances 24 m per step at dt 1.0, so junction gap
acceptance is resolved at 24 m granularity; commanding a lower speed shortens the
per-step displacement and recovers resolution the discretisation had destroyed.

This sweeps the step size to test that reading. If the effect is an artefact it
shrinks monotonically as dt falls and the uncommanded baseline rises to a limit; if
it is physics it survives at every step size.

    .venv/bin/python scripts/measure_step_size_convergence.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

DEFAULT_SEEDS = (3, 7, 11, 19, 23)
EPISODE_S = 600.0
WARMUP_S = 300.0


def run_one(*, commanded, seed, dt, topology, demand, work_dir):
    duration_steps = int(round(EPISODE_S / dt))
    env = SumoTopologyEnv(topology, {
        "topology": load_named_config("topology", topology),
        "demand": load_named_config("demand", demand),
        "duration_steps": duration_steps, "dt": dt,
        "warmup_steps": int(round(WARMUP_S / dt)), "work_dir": work_dir})
    env.reset(seed=seed)
    try:
        for _ in range(duration_steps):
            if commanded is not None:
                for agent_id in list(env.agent_ids):
                    env.command_speed(agent_id, commanded)
            env.step({})
        return env.arrived_total
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="inverted_tree")
    parser.add_argument("--demand", default="sumo_saturating")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=float, nargs="+",
                        default=[1.0, 0.5, 0.2, 0.1, 0.05])
    parser.add_argument("--commanded", type=float, default=10.0)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    print(f"{args.topology}, {args.demand}, {EPISODE_S:.0f} s episode after "
          f"{WARMUP_S:.0f} s of warm-up, seeds {tuple(args.seeds)}.\n")
    print(f"{'dt':>6} {'steps':>7} {'uncommanded':>12} {'sd':>6} "
          f"{str(args.commanded) + ' m/s':>12} {'sd':>6} {'gain':>8}")
    for dt in args.steps:
        idle = [run_one(commanded=None, seed=s, dt=dt, topology=args.topology,
                        demand=args.demand, work_dir=args.work_dir) for s in args.seeds]
        held = [run_one(commanded=args.commanded, seed=s, dt=dt, topology=args.topology,
                        demand=args.demand, work_dir=args.work_dir) for s in args.seeds]
        idle_mean, held_mean = statistics.fmean(idle), statistics.fmean(held)
        print(f"{dt:>6} {int(round(EPISODE_S / dt)):>7} {idle_mean:>12.1f} "
              f"{statistics.stdev(idle):>6.1f} {held_mean:>12.1f} "
              f"{statistics.stdev(held):>6.1f} "
              f"{100 * (held_mean - idle_mean) / idle_mean:>7.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
