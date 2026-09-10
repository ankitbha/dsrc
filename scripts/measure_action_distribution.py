#!/usr/bin/env python3
"""What a trained actor actually chooses, per head, over real observations.

Task 103's gate has three parts and the training metrics carry only two of them:
the score and the summed entropy. The third -- whether the action distribution has
left uniform -- needs the actor's own output, so it is measured here rather than
inferred from the entropy. Entropy is a single number over both heads and can fall
because one head became deterministic while the other did not, which is a different
outcome from a policy that has learned a joint preference.

    .venv/bin/python scripts/measure_action_distribution.py \
        --actor <dir>/latest_actor.pt --training mappo_sumo --decisions 200

The observations come from running the policy in the environment it trained on, not
from a fixture: a distribution measured on observations the policy never sees says
nothing about what it does.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.rl.actions import ACTION_VALUES, ActionSpec  # noqa: E402
from src.rl.controller import LearnedPolicyController  # noqa: E402
from src.rl.encoders import encode_local_batch  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--training", default="mappo_sumo")
    parser.add_argument("--decisions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    training = load_named_config("training", args.training)
    dt = float(training["dt"])
    interval = training.get("decision_interval_s")
    every = 1 if interval is None else max(1, int(round(float(interval) / dt)))
    spec = ActionSpec(str(training["action_profile"]))  # type: ignore[arg-type]

    config = {
        "topology": load_named_config("topology", str(training["topology"])),
        "demand": load_named_config("demand", str(training["demand"])),
        "human_model": load_named_config("human_model", str(training["human_model"])),
        "duration_steps": args.decisions * every,
        "dt": dt,
        "warmup_steps": int(training["warmup_steps"]),
        "sensing": dict(training.get("sensing") or {}),
    }
    if args.work_dir:
        config["work_dir"] = args.work_dir

    controller = LearnedPolicyController.from_checkpoint(Path(args.actor), device="cpu")
    actor = controller.actor
    env = SumoTopologyEnv(str(training["topology"]), config)
    observations, _ = env.reset(seed=args.seed)
    totals = {head: torch.zeros(spec.head_size(head)) for head in spec.active_heads}
    observed = 0
    try:
        for step in range(args.decisions * every):
            actions = {}
            if step % every == 0:
                agent_ids, obs_tensor = encode_local_batch(observations)
                if agent_ids:
                    with torch.no_grad():
                        distributions = actor.distributions(obs_tensor)
                    for head in spec.active_heads:
                        totals[head] += distributions[head].probs.sum(dim=0)
                    observed += len(agent_ids)
                    actions = controller.act(observations, global_state=None)
            observations, _, _, _, _ = env.step(actions)
    finally:
        env.close()

    if not observed:
        print("no agent was ever observed; nothing to report")
        return 1

    result = {"observations": observed, "heads": {}}
    joint_modal = 1.0
    summed_entropy = 0.0
    for head in spec.active_heads:
        probabilities = (totals[head] / observed).tolist()
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        joint_modal *= max(probabilities)
        summed_entropy += entropy
        result["heads"][head] = {
            "values": list(ACTION_VALUES[head]),
            "mean_probability": [round(p, 4) for p in probabilities],
            "entropy": round(entropy, 4),
            "max_entropy": round(math.log(len(probabilities)), 4),
        }
    combinations = math.prod(spec.head_size(head) for head in spec.active_heads)
    result["summed_entropy"] = round(summed_entropy, 4)
    result["max_summed_entropy"] = round(
        sum(math.log(spec.head_size(head)) for head in spec.active_heads), 4)
    result["joint_modal_share"] = round(joint_modal, 4)
    result["uniform_joint_share"] = round(1.0 / combinations, 4)

    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
