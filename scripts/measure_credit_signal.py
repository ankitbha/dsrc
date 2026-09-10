#!/usr/bin/env python3
"""Does an agent's own action move its reward more than another agent's action does?

That question is the premise of task 103. A team reward divided among tens of agents
is moved by any of them, so the part attributable to the agent being credited is
small; a reward measured over the agent's own neighbourhood should be moved mostly
by that agent. Until now this was an argument rather than a measurement.

The measurement is a counterfactual on identical traffic. For one seed:

  * run the warm-up, take the first few controllable vehicles in sorted order;
  * hold ONE of them at 5 m/s for a 20 s window, then repeat from the same seed
    holding it at 25 m/s, and record EVERY tracked agent's reward in both;
  * the change in agent i's reward when agent i was the commanded one is the credit
    its own action earns; the change when agent j was commanded is the noise another
    agent's action puts into the same signal.

A ratio near 1 means an agent cannot tell its own contribution from its neighbours'.
The higher the ratio the more the advantage estimate is about the action.

    .venv/bin/python scripts/measure_credit_signal.py --seeds 7 17 27

Each arm runs in its own subprocess: libsumo keeps the simulation in module-level
state, so one process holds one simulation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.rl.rewards import build_local_reward, build_team_reward  # noqa: E402


def run_one(*, seed, speed, commanded_rank, ranks, window_steps, training, work_dir):
    """One arm. Returns each tracked agent's summed reward over the window.

    The tracked agents are chosen by a rule that cannot depend on the arm: the first
    `ranks` controllable vehicle ids in sorted order at the first observed step.
    Choosing by position or speed would select different vehicles once the arms
    diverge, and the comparison would be between two agents rather than between two
    actions.
    """
    from src.sumo.env import SumoTopologyEnv

    config = {
        "topology": load_named_config("topology", str(training["topology"])),
        "demand": load_named_config("demand", str(training["demand"])),
        "human_model": load_named_config("human_model", str(training["human_model"])),
        "duration_steps": window_steps,
        "dt": float(training["dt"]),
        "warmup_steps": int(training["warmup_steps"]),
        "sensing": dict(training.get("sensing") or {}),
        "work_dir": work_dir,
    }
    env = SumoTopologyEnv(str(training["topology"]), config)
    env.reset(seed=seed)
    try:
        # THE FASTEST agents at the first observed step, not the lowest-numbered.
        # Sorting by id selects the oldest vehicles, which after a 300 s warm-up on
        # a congested road are the ones deepest in the queue: the first version of
        # this script picked three vehicles all travelling at 0.0 m/s, and every
        # commanded speed from 0.5 to 30 m/s produced a bit-identical trajectory.
        # A speed command can only bind on a vehicle that is moving, so that is the
        # population the credit signal is about.
        #
        # The choice is made BEFORE any command is issued, so it is identical in
        # every arm and the comparison stays between two actions rather than two
        # vehicles.
        speeds = {s.vehicle_id: s.speed_mps for s in env.vehicle_snapshots()}
        tracked = sorted(env.agent_ids, key=lambda a: -speeds.get(a, 0.0))[:ranks]
        commanded = tracked[commanded_rank] if 0 <= commanded_rank < len(tracked) else None
        team = 0.0
        local = {agent: 0.0 for agent in tracked}
        for _ in range(window_steps):
            if commanded is not None and speed is not None and commanded in env.agent_ids:
                env.command_speed(commanded, speed)
            _, _, _, _, info = env.step({})
            team += build_team_reward(info["metrics"], training.get("reward_weights"))
            neighbourhood = info.get("neighbourhood") or {}
            for agent in tracked:
                values = neighbourhood.get(agent)
                if values:
                    local[agent] += build_local_reward(
                        values, training.get("local_reward_weights"))
        return {"tracked": tracked, "commanded": commanded, "team": team, "local": local}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="mappo_sumo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--ranks", type=int, default=3,
                        help="how many controllable vehicles are tracked and commanded")
    parser.add_argument("--window-s", type=float, default=20.0,
                        help="the advantage horizon, 1/(1-gae_lambda) decisions")
    parser.add_argument("--slow", type=float, default=5.0)
    parser.add_argument("--fast", type=float, default=25.0)
    parser.add_argument("--work-dir", default="/tmp/dsrc_credit")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=None)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--speed", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    training = load_named_config("training", args.training)
    window_steps = max(1, int(round(args.window_s / float(training["dt"]))))

    if args.one:
        print(json.dumps(run_one(
            seed=args.seed, speed=(None if args.speed < 0 else args.speed),
            commanded_rank=args.rank, ranks=args.ranks, window_steps=window_steps,
            training=training,
            work_dir=f"{args.work_dir}/s{args.seed}_r{args.rank}_v{args.speed:g}")))
        return 0

    def launch(seed, rank, speed):
        return subprocess.Popen(
            [sys.executable, __file__, "--one", "--training", args.training,
             "--seed", str(seed), "--rank", str(rank), "--speed", str(speed),
             "--ranks", str(args.ranks), "--window-s", str(args.window_s),
             "--work-dir", args.work_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def collect(jobs):
        out = {}
        for key, process in jobs:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"  {key} failed: {stderr[-400:]}")
                continue
            out[key] = json.loads(stdout.strip().splitlines()[-1])
        return out

    # THE CONTROL FIRST: two uncommanded runs at the same seed must agree exactly. If
    # they do not, the simulation is not reproducible under a fixed seed and every
    # difference below is unattributable rather than small.
    control = collect([(("a",), launch(args.seeds[0], -1, -1.0)),
                       (("b",), launch(args.seeds[0], -1, -1.0))])
    if len(control) < 2:
        print("the control runs did not complete")
        return 1
    drift = abs(control[("a",)]["team"] - control[("b",)]["team"])
    print(f"control: two uncommanded runs at seed {args.seeds[0]} differ by {drift:.3e}")
    if drift > 1e-9:
        print("  NOT REPRODUCIBLE -- nothing below is attributable")
        return 1

    combinations = list(itertools.product(args.seeds, range(args.ranks), (args.slow, args.fast)))
    results = {}
    for start in range(0, len(combinations), args.workers):
        batch = combinations[start:start + args.workers]
        results.update(collect([(key, launch(*key)) for key in batch]))
        print(f"  {len(results)}/{len(combinations)} arms done", flush=True)

    own, other, team_own, team_other = [], [], [], []
    for seed in args.seeds:
        for commanded in range(args.ranks):
            slow = results.get((seed, commanded, args.slow))
            fast = results.get((seed, commanded, args.fast))
            if not slow or not fast or slow["tracked"] != fast["tracked"]:
                continue
            team_delta = abs(fast["team"] - slow["team"])
            for rank, agent in enumerate(slow["tracked"]):
                delta = abs(fast["local"][agent] - slow["local"][agent])
                (own if rank == commanded else other).append(delta)
            (team_own if commanded == 0 else team_other).append(team_delta)

    if not own or not other:
        print("not enough pairs completed")
        return 1

    print(f"\nOver a {args.window_s:g} s window, {args.slow:g} m/s against "
          f"{args.fast:g} m/s, {len(own)} own and {len(other)} cross pairs:")
    print(f"  local reward moved by the agent's OWN action     "
          f"{statistics.fmean(own):8.4f}")
    print(f"  local reward moved by ANOTHER agent's action     "
          f"{statistics.fmean(other):8.4f}")
    print(f"  ratio own / other                                "
          f"{statistics.fmean(own) / max(statistics.fmean(other), 1e-12):8.2f}")
    print(f"  team reward moved by ONE agent's action          "
          f"{statistics.fmean(team_own + team_other):8.4f}")
    print(f"  team reward over the same window (uncommanded)   "
          f"{control[('a',)]['team']:8.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"own": own, "other": other, "team": team_own + team_other,
             "uncommanded_team_window": control[("a",)]["team"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
