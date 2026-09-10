#!/usr/bin/env python3
"""Is the advantage aligned with the action that earned it? With a valid null.

    .venv/bin/python scripts/measure_action_alignment.py --training mappo_src

**This replaces the permutation floor used in tasks 105 to 118, which is not a valid
null when advantages are temporally correlated.** That floor permuted advantages
across the batch, destroying two things at once: the association between action and
advantage, which is what the test isolates, and the temporal profile of each agent's
trajectory. A segment agent holds about nineteen consecutive decisions whose
advantages are smooth; shuffling makes the sequence rough, a rough sequence cancels
less when summed against `grad log pi`, and the floor comes out high. The segment arms
read z = -1.18 and -1.75 -- a measured value systematically BELOW its own null, which
is the signature of a null that is not exchangeable with the data.

**The null used here.** For each transition, resample an action from the policy at
the SAME observation and keep the advantage:

    a'_i ~ pi(.|s_i),   g_null = mean_i A_i * grad log pi(a'_i | s_i)

States, advantages and their temporal structure are all preserved; only the pairing
between an advantage and the action that earned it is broken. Under the score-function
identity the expectation of that gradient is zero, so it is the right reference, and
it is estimated by repeating the resample.

The ceiling is unchanged: a synthetic advantage built to correlate with the action,
which shows the statistic can move on this batch.
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.rl.ppo import PPOConfig  # noqa: E402
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything  # noqa: E402


def gradient_norm(trainer, observations, actions, old_log_probs, advantages, clip):
    log_probs, _ = trainer.actor.evaluate_actions(observations, actions)
    ratio = torch.exp(log_probs - old_log_probs)
    loss = -torch.min(ratio * advantages,
                      torch.clamp(ratio, 1 - clip, 1 + clip) * advantages).mean()
    trainer.actor.zero_grad(set_to_none=True)
    loss.backward()
    return float(torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                for q in trainer.actor.parameters()
                                if q.grad is not None)))


def alignment_z(trainer, observations, actions, old_log_probs, advantages, *,
                clip: float, resamples: int = 60, seed: int = 0):
    """How far the measured gradient sits above a null that keeps the advantages.

    Returns `(z, measured, null_mean, null_sd)`. The null draws a fresh action from
    the policy at each observation and keeps that transition's advantage, so states,
    advantages and their temporal structure survive and only the pairing between an
    advantage and the action that earned it is broken.
    """
    measured = gradient_norm(trainer, observations, actions, old_log_probs,
                             advantages, clip)
    torch.manual_seed(seed)
    null = []
    for _ in range(resamples):
        with torch.no_grad():
            _, resampled, resampled_log_probs, _ = trainer.actor.sample(observations)
        null.append(gradient_norm(trainer, observations, resampled,
                                  resampled_log_probs, advantages, clip))
    mean, sd = statistics.fmean(null), statistics.pstdev(null)
    return (measured - mean) / max(sd, 1e-12), measured, mean, sd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="mappo_src")
    parser.add_argument("--decisions", type=int, default=None,
                        help="rollout length in DECISIONS; defaults to the config's")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--resamples", type=int, default=60)
    args = parser.parse_args()

    cfg = load_named_config("training", args.training)
    env_keys = ("topology", "demand", "human_model", "duration_steps", "dt")
    bundle = {"training": cfg, "env": {k: cfg[k] for k in env_keys if k in cfg}, "seed": 7}
    base_t = TrainingConfig.from_mapping(bundle)
    base_p = PPOConfig.from_mapping(cfg)
    decisions = args.decisions or base_t.rollout_steps

    print(f"{args.training}: {decisions} decisions per rollout, "
          f"{args.resamples} resamples per seed")
    print(f"{'seed':>5} {'n':>7} {'measured':>10} {'null mean':>10} {'null sd':>9} "
          f"{'z':>7} {'ceiling':>8}")
    zs = []
    for seed in args.seeds:
        seed_everything(0)
        t = dataclasses.replace(base_t, rollout_steps=decisions,
                                work_dir=f"/tmp/dsrc_align_{args.training}_{seed}")
        trainer = make_trainer(t, base_p, device="cpu")
        buffer, _ = trainer.collect_rollout(seed=seed)
        batch = buffer.compute_returns_and_advantages(
            gamma=base_p.gamma, gae_lambda=base_p.gae_lambda,
            group_by_agent=trainer.advantage_group_by_agent)
        observations, advantages = batch.observations, batch.advantages
        z, measured, mean, sd = alignment_z(
            trainer, observations, batch.actions, batch.old_log_probs, advantages,
            clip=base_p.clip_coef, resamples=args.resamples, seed=seed)

        ceiling_adv = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
        ceiling_adv = (ceiling_adv - ceiling_adv.mean()) / (ceiling_adv.std() + 1e-8)
        ceiling = gradient_norm(trainer, observations, batch.actions,
                                batch.old_log_probs, ceiling_adv, base_p.clip_coef)

        zs.append(z)
        print(f"{seed:>5} {advantages.shape[0]:>7} {measured:>10.5f} {mean:>10.5f} "
              f"{sd:>9.5f} {z:>+7.2f} {ceiling/max(mean, 1e-12):>8.1f}", flush=True)

    if len(zs) > 1:
        se = statistics.stdev(zs) / len(zs) ** 0.5
        print(f"\nmean z {statistics.fmean(zs):+.2f} +/- {se:.2f} over {len(zs)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
