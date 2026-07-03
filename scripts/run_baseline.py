#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import BASELINE_NAMES, make_baseline
from src.config.loaders import deep_merge, load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one infrastructure-free DSRC baseline episode.")
    parser.add_argument("--controller", required=True, choices=BASELINE_NAMES)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--demand", default="medium")
    parser.add_argument("--human-model", default="normal")
    parser.add_argument("--av-penetration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--initial-human-vehicles", type=int, default=12)
    parser.add_argument("--output-root", default="outputs/metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    controller = make_baseline(args.controller)
    controller.reset(
        env_metadata={
            "topology_id": args.topology,
            "demand": args.demand,
            "human_model": args.human_model,
        },
        seed=args.seed,
    )
    env = HighwayTopologyEnv(args.topology, config)
    observations, reset_info = env.reset(seed=args.seed)
    experiment_id = f"{args.controller}_{args.topology}_{args.demand}_seed{args.seed}"
    logger = MetricsLogger(experiment_id=experiment_id, output_root=args.output_root)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        actions = controller.act(observations, global_state=None)
        observations, _, terminated, truncated, info = env.step(actions)
        logger.record_step(info.get("metrics", {}))
        logger.record_segments(time_s=float(info.get("time", 0.0)), segment_metrics=env.get_segment_metrics())

    summary = {
        **env.get_episode_summary(),
        "controller": args.controller,
        "seed": args.seed,
        "reset_info": reset_info,
    }
    paths = logger.write_episode(summary)
    for key, value in paths.items():
        print(f"{key}: {value}")
    return 0


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    topology_cfg = load_named_config("topology", args.topology)
    demand_cfg = load_named_config("demand", args.demand)
    human_model_cfg = load_named_config("human_model", args.human_model)

    demand_overrides: dict[str, Any] = {}
    if args.av_penetration is not None:
        demand_overrides["av_penetration"] = args.av_penetration
    if args.controller == "no_av":
        demand_overrides["av_penetration"] = 0.0

    demand_cfg = deep_merge(demand_cfg, demand_overrides) if demand_overrides else demand_cfg
    controlled_vehicles = 0 if args.controller == "no_av" else args.controlled_vehicles
    if args.topology == "ring":
        # keep the total ring population identical across controllers: no_av
        # replaces the would-be AVs with humans instead of running fewer vehicles
        initial_humans = args.initial_human_vehicles + (args.controlled_vehicles - controlled_vehicles)
    else:
        initial_humans = 0
    controller_cfg = {
        "name": args.controller,
        "family": "baseline",
        "safety_mode": make_baseline(args.controller).metadata.safety_mode,
    }
    return {
        "topology": topology_cfg,
        "demand": demand_cfg,
        "human_model": human_model_cfg,
        "controller": controller_cfg,
        "duration_steps": args.duration_steps,
        "dt": args.dt,
        "controlled_vehicles": controlled_vehicles,
        "initial_human_vehicles": initial_humans,
    }


if __name__ == "__main__":
    raise SystemExit(main())
