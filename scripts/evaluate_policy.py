#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline import build_config
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger
from src.rl.controller import LearnedPolicyController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DSRC actor checkpoint.")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--topology", default="ring")
    parser.add_argument("--demand", default="medium")
    parser.add_argument("--human-model", default="normal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-steps", type=int, default=120)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--initial-human-vehicles", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs/metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = LearnedPolicyController.from_checkpoint(args.actor, device=args.device)
    env_args = argparse.Namespace(
        controller="random_av",
        topology=args.topology,
        demand=args.demand,
        human_model=args.human_model,
        av_penetration=None,
        seed=args.seed,
        duration_steps=args.duration_steps,
        dt=1.0,
        controlled_vehicles=args.controlled_vehicles,
        initial_human_vehicles=args.initial_human_vehicles,
        output_root=args.output_root,
    )
    config = build_config(env_args)
    config["controller"] = {"name": "learned_policy", "family": "rl", "safety_mode": "integrated_rl"}
    env = HighwayTopologyEnv(args.topology, config)
    observations, reset_info = env.reset(seed=args.seed)
    experiment_id = f"learned_policy_{args.topology}_{args.demand}_seed{args.seed}"
    logger = MetricsLogger(experiment_id=experiment_id, output_root=args.output_root)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        actions = controller.act(observations, global_state=None)
        observations, _, terminated, truncated, info = env.step(actions)
        logger.record_step(info.get("metrics", {}))
        logger.record_segments(time_s=float(info.get("time", 0.0)), segment_metrics=env.get_segment_metrics())
    paths = logger.write_episode(
        {
            **env.get_episode_summary(),
            "controller": "learned_policy",
            "actor": str(args.actor),
            "seed": args.seed,
            "reset_info": reset_info,
        }
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
