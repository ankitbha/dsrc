"""Arrivals and team reward with every AV commanded to hold one speed.

This is the measurement behind the arm tables in
`plans/plan_task_84_sumo_simulator.md` and in `configs/training/mappo_sumo.yaml`.
It exists as a script because an independent audit could not reproduce those
tables from the repository: the seed set was not recorded, and recovering it took
an exhaustive search over 5-subsets. Quote no arm figure that this script does not
produce.

The commanded arms are an oracle, not a controller. Every AV is given the same
speed through `command_speed` for the whole episode, which no policy can express;
the point is to locate the effect a policy would have to find.

    .venv/bin/python scripts/measure_commanded_arms.py --dt 0.1 \
        --duration-steps 6000 --warmup-steps 3000
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.rl.rewards import DEFAULT_REWARD_WEIGHTS, build_team_reward  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

#: The seeds every recorded arm table was measured on. Named here because an audit
#: had to recover them by search when they were not written down.
DEFAULT_SEEDS = (3, 7, 11, 19, 23)


def run_one(*, commanded, seed, topology, demand, weights, dt, duration_steps,
            warmup_steps, work_dir):
    env = SumoTopologyEnv(topology, {
        "topology": load_named_config("topology", topology),
        "demand": load_named_config("demand", demand),
        "duration_steps": duration_steps, "dt": dt,
        "warmup_steps": warmup_steps, "work_dir": work_dir})
    env.reset(seed=seed)
    try:
        rewards, roadblock = [], []
        for _ in range(duration_steps):
            if commanded is not None:
                for agent_id in list(env.agent_ids):
                    env.command_speed(agent_id, commanded)
            _, _, _, _, info = env.step({})
            rewards.append(build_team_reward(info["metrics"], weights))
            roadblock.append(float(info["metrics"].get("rolling_roadblock_score", 0.0)))
        return {
            "arrivals": env.arrived_total,
            "reward_per_step": statistics.fmean(rewards),
            "roadblock": statistics.fmean(roadblock),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="inverted_tree")
    parser.add_argument("--demand", default="sumo_saturating")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration-steps", type=int, default=6000)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--commands", type=float, nargs="+",
                        default=[8.0, 10.0, 12.0, 15.0, 20.0, 24.0],
                        help="commanded speeds in m/s; an uncommanded arm is always included")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--throughput-weight", type=float, default=0.10)
    parser.add_argument("--jam-weight", type=float, default=-2.0)
    args = parser.parse_args()

    weights = {**DEFAULT_REWARD_WEIGHTS,
               "throughput_recent": args.throughput_weight,
               "jam_fraction": args.jam_weight}
    seconds = args.duration_steps * args.dt
    print(f"{args.topology}, {args.demand}, dt {args.dt}, "
          f"{args.duration_steps} steps = {seconds:.0f} s after "
          f"{args.warmup_steps * args.dt:.0f} s of warm-up, seeds {tuple(args.seeds)}.")
    print("Reward is the unscaled team reward per step; the per-second column is "
          "comparable across dt.\n")
    print(f"{'arm':>13} {'arrivals':>9} {'sd':>6} {'reward/step':>12} "
          f"{'reward/s':>9} {'roadblock':>10}")
    for commanded in [None, *args.commands]:
        results = [run_one(commanded=commanded, seed=seed, topology=args.topology,
                           demand=args.demand, weights=weights, dt=args.dt,
                           duration_steps=args.duration_steps,
                           warmup_steps=args.warmup_steps, work_dir=args.work_dir)
                   for seed in args.seeds]
        arrivals = [r["arrivals"] for r in results]
        reward = statistics.fmean(r["reward_per_step"] for r in results)
        name = "uncommanded" if commanded is None else f"{commanded:g} m/s"
        spread = statistics.stdev(arrivals) if len(arrivals) > 1 else 0.0
        print(f"{name:>13} {statistics.fmean(arrivals):>9.1f} {spread:>6.1f} "
              f"{reward:>12.4f} {reward / args.dt:>9.2f} "
              f"{statistics.fmean(r['roadblock'] for r in results):>10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
