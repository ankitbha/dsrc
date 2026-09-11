#!/usr/bin/env python3
"""What correlation does a gradient-norm z correspond to? Measured, not assumed.

Every null in the simulation leg is a z from `measure_action_alignment.py`, and every
statement about what a null EXCLUDES converts that z into an action-advantage
correlation. That conversion was done wrong twice:

  - by 2/sqrt(n), the sampling error of a correlation coefficient, which does not
    describe a statistic that is not a correlation;
  - by extrapolating linearly from the instrument's ceiling, where the synthetic
    advantage has correlation exactly 1.0 with the indicator of `slow`. The
    relationship is NOT linear: a gradient norm is the norm of a noise vector plus an
    aligned component, and a small aligned component adds almost nothing in quadrature.

This measures the curve. One rollout is collected, the resampled-action null is
computed once on it, and advantages are then synthesised at a range of known
correlations with the action. The output is the z the instrument reads at each, and
the correlation at which z reaches 2 is the smallest effect a single-seed reading of
that arm could show.

    .venv/bin/python scripts/measure_alignment_calibration.py --training mappo_src

The noise component is the REAL advantage orthogonalised against the action indicator,
not i.i.d. noise. With i.i.d. noise the c = 0 row reads z = -0.59 rather than 0,
because the null's advantages are temporally autocorrelated and i.i.d. ones are not,
so the curve does not pass through its own null.
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_action_alignment import gradient_norm  # noqa: E402
from src.config.loaders import load_named_config  # noqa: E402
from src.rl.ppo import PPOConfig  # noqa: E402
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything  # noqa: E402

LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def _standardise(vector: torch.Tensor) -> torch.Tensor:
    return (vector - vector.mean()) / (vector.std() + 1e-12)


def _indicator(actions: torch.Tensor, head: int = 0, value: int = 0) -> torch.Tensor:
    chose = (actions[:, head] == value).float()
    if float(chose.std()) < 1e-8:
        raise ValueError(
            "every agent chose the same action on this head, so no agent never chose "
            "the other values and the correlation is undefined")
    return _standardise(chose)


def correlation_with_action(actions: torch.Tensor, advantage: torch.Tensor,
                            head: int = 0, value: int = 0) -> float:
    """Point-biserial correlation between choosing `value` on `head` and `advantage`."""
    chose = (actions[:, head] == value).float()
    if float(chose.std()) < 1e-8 or float(advantage.std()) < 1e-8:
        return 0.0
    centred_a = chose - chose.mean()
    centred_b = advantage - advantage.mean()
    denominator = float(centred_a.norm() * centred_b.norm())
    return float(centred_a @ centred_b) / denominator if denominator else 0.0


def build_advantage(actions: torch.Tensor, noise: torch.Tensor,
                    target: float, head: int = 0, value: int = 0) -> torch.Tensor:
    """An advantage correlating with the action at exactly `target`.

    `noise` is orthogonalised against the indicator first, so the mix hits the target
    rather than approaching it: any correlation the caller's vector already carries
    with the action would otherwise add to the one being requested.
    """
    unit = _indicator(actions, head, value)
    residual = noise.flatten().float()
    residual = residual - (residual @ unit) / (unit @ unit) * unit
    residual = _standardise(residual)
    mixed = target * unit + (1.0 - target * target) ** 0.5 * residual
    return _standardise(mixed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="mappo_src")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decisions", type=int, default=None)
    parser.add_argument("--resamples", type=int, default=60)
    args = parser.parse_args()

    cfg = load_named_config("training", args.training)
    env_keys = ("topology", "demand", "human_model", "duration_steps", "dt")
    bundle = {"training": cfg, "env": {k: cfg[k] for k in env_keys if k in cfg},
              "seed": args.seed}
    base_t = TrainingConfig.from_mapping(bundle)
    base_p = PPOConfig.from_mapping(cfg)

    seed_everything(args.seed)
    t = dataclasses.replace(
        base_t, rollout_steps=args.decisions or base_t.rollout_steps,
        work_dir=f"/tmp/dsrc_calibration_{args.training}_{args.seed}")
    trainer = make_trainer(t, base_p, device="cpu")
    buffer, _ = trainer.collect_rollout(seed=args.seed)
    batch = buffer.compute_returns_and_advantages(
        gamma=base_p.gamma, gae_lambda=base_p.gae_lambda,
        group_by_agent=trainer.advantage_group_by_agent)

    torch.manual_seed(args.seed)
    null = []
    for _ in range(args.resamples):
        with torch.no_grad():
            _, resampled, resampled_lp, _ = trainer.actor.sample(batch.observations)
        null.append(gradient_norm(trainer, batch.observations, resampled, resampled_lp,
                                  batch.advantages, base_p.clip_coef))
    null_mean, null_sd = statistics.fmean(null), statistics.pstdev(null)

    print(f"{args.training} seed {args.seed}: n {batch.actions.shape[0]}, "
          f"null {null_mean:.5f} +/- {null_sd:.5f}, {args.resamples} resamples")
    print(f"{'target c':>9} {'actual c':>9} {'grad norm':>10} {'z':>8} {'z/c':>8}")
    rows = []
    for level in LEVELS:
        advantage = build_advantage(batch.actions, batch.advantages, level)
        norm = gradient_norm(trainer, batch.observations, batch.actions,
                             batch.old_log_probs, advantage, base_p.clip_coef)
        z = (norm - null_mean) / max(null_sd, 1e-12)
        actual = correlation_with_action(batch.actions, advantage)
        rows.append((level, z))
        print(f"{level:>9.2f} {actual:>+9.4f} {norm:>10.5f} {z:>+8.2f} "
              f"{(z / level if level else float('nan')):>8.1f}")

    for index in range(1, len(rows)):
        (low, z_low), (high, z_high) = rows[index - 1], rows[index]
        if z_low < 2.0 <= z_high:
            crossing = low + (2.0 - z_low) * (high - low) / (z_high - z_low)
            print(f"\nz crosses 2 at c = {crossing:.3f}. Linear extrapolation from the "
                  f"ceiling would say {2.0 / rows[-1][1]:.3f}.")
            return 0
    print("\nz does not cross 2 over the range measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
