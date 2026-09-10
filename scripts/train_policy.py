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
    # These default to None so the training config's own declarations win. They
    # previously carried literal defaults -- topology "ring", demand "medium",
    # duration_steps 120 -- which SILENTLY OVERRODE the config, because
    # `TrainingConfig.from_mapping` prefers the `env` block this script builds. A
    # config declaring 9000-step episodes at dt 0.1 was resolved to 120 steps at
    # dt 1.0, which is the step size the throughput result was retracted at.
    parser.add_argument("--topology", default=None)
    parser.add_argument("--demand", default=None)
    parser.add_argument("--human-model", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--duration-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--controlled-vehicles", type=int, default=None)
    parser.add_argument("--initial-human-vehicles", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs/checkpoints")
    parser.add_argument("--resume-from", default=None, help="Resume training from an existing checkpoint directory.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume model weights from latest_actor.pt/latest_critic.pt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_bundle(args)
    training_config = TrainingConfig.from_mapping(config)
    ppo_config = PPOConfig.from_mapping(config.get("training", config))
    trainer = make_trainer(training_config, ppo_config, device=args.device)
    result = trainer.train(resume_from=args.resume_from, resume_latest=args.resume_latest)
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
    # Only what was actually given on the command line. `TrainingConfig.from_mapping`
    # prefers `env` over everything else, so putting a default here overrides the
    # config silently.
    overrides = {
        "topology": args.topology,
        "demand": args.demand,
        "human_model": args.human_model,
        "duration_steps": args.duration_steps,
        "dt": args.dt,
        "controlled_vehicles": args.controlled_vehicles,
        "initial_human_vehicles": args.initial_human_vehicles,
    }
    env = {key: value for key, value in overrides.items() if value is not None}
    # The config's own declarations, lifted to where `from_mapping` looks for them
    # when `env` does not carry an override.
    declared = {
        key: training[key]
        for key in ("topology", "demand", "human_model", "duration_steps", "dt",
                    "controlled_vehicles", "initial_human_vehicles")
        if key in training
    }
    return {
        "training": training,
        "env": env,
        **declared,
        "seed": args.seed,
        "output_root": args.output_root,
    }


if __name__ == "__main__":
    raise SystemExit(main())
