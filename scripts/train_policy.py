#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loaders import load_named_config
from src.rl.ppo import PPOConfig
from src.rl.trainers import TrainingConfig, make_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DSRC PPO/IPPO/MAPPO policy.")
    parser.add_argument("--training", default="shared_ppo", help="Training config name or YAML path.")
    parser.add_argument("--topology", default="ring")
    parser.add_argument("--demand", default="medium")
    parser.add_argument("--human-model", default="normal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--duration-steps", type=int, default=120)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--initial-human-vehicles", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs/checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_bundle(args)
    training_config = TrainingConfig.from_mapping(config)
    ppo_config = PPOConfig.from_mapping(config.get("training", config))
    trainer = make_trainer(training_config, ppo_config, device=args.device)
    result = trainer.train()
    print(f"checkpoint_dir: {result['output_dir']}")
    print(f"updates: {result['updates']}")
    print(f"best_score: {result['best_score']}")
    return 0


def load_training_bundle(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.training)
    if path.exists():
        loaded = yaml.safe_load(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"training config must be a mapping: {path}")
        training = loaded
    else:
        training = load_named_config("training", args.training)
    if args.total_updates is not None:
        training["total_updates"] = args.total_updates
    if args.rollout_steps is not None:
        training["rollout_steps"] = args.rollout_steps
    return {
        "training": training,
        "env": {
            "topology": args.topology,
            "demand": args.demand,
            "human_model": args.human_model,
            "duration_steps": args.duration_steps,
            "controlled_vehicles": args.controlled_vehicles,
            "initial_human_vehicles": args.initial_human_vehicles,
        },
        "seed": args.seed,
        "output_root": args.output_root,
    }


if __name__ == "__main__":
    raise SystemExit(main())
