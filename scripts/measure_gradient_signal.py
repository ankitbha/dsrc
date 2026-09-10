#!/usr/bin/env python3
"""Is the policy-gradient norm above its own noise floor at all?

TWO SENSES OF THE WORD "ORACLE" WERE CONFLATED and one is gone from this file. The
metering oracle elsewhere in this project is a CONTROLLER: perfect state, hand
written, it drives real vehicles and is measured on vehicles served. The quantity
below was also called an oracle and is not a controller at all -- it drives nothing.
It is a synthetic advantage vector, +1 where the policy chose one particular value
and -1 otherwise, injected into the gradient formula to check that the statistic can
move. It is the instrument's CEILING and is named that here.

THE FLOOR NEEDS ITS OWN ERROR BAR, which this script does not give it: it takes ONE
permutation, which is a single draw from the floor's distribution rather than the
floor. Measured with 60 permutations by
`scripts/measure_gradient_floor_distribution.py`, that distribution has a standard
deviation of 20 to 30% of its mean, so a ratio of 1.4 from this script is inside it.
Read the two together, and prefer the z-score the other one reports.

    .venv/bin/python scripts/measure_gradient_signal.py

THIS IS THE INSTRUMENT THAT SHOULD HAVE BEEN BUILT FIRST. Before it existed, four
comparisons were made and reported -- the team reward against a per-agent reward,
a critic with and without privileged neighbourhood features, discount horizons from
10 to 1000 decisions, and action hold lengths from 1 s to 20 s -- and every one of
them was a comparison between two noise floors. The differences were all within a
factor of 1.1 and none of them meant anything, because nothing had established that
the statistic could read a signal at all.

Every arm measured so far -- team reward against local reward, critic with and
without the neighbourhood, discount horizons from 10 to 1000 decisions -- gives a
policy-gradient norm near 0.05. That is either a signal that nothing changes or a
floor that nothing can go below, and the two are indistinguishable without measuring
the floor.

Three arms on the SAME batch:

  measured  the advantages as computed;
  shuffled  the same advantages permuted across the batch, which destroys any
            correlation with the action while keeping the distribution exactly;
  ceiling   a SYNTHETIC advantage constructed to correlate perfectly with the action
            (+1 where the policy chose `slow`, -1 otherwise), which is what a strong
            signal looks like on this batch.

If measured is at shuffled, no reward tried so far carries action-attributable
signal at this policy and the comparisons between them were between two floors.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
sys.argv = ["x", "--training", "mappo_sumo"]
import torch
from scripts.train_policy import parse_args, load_training_bundle
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.ppo import PPOConfig

cfg = load_training_bundle(parse_args())
base_t = TrainingConfig.from_mapping(cfg)
base_p = PPOConfig.from_mapping(cfg.get("training", cfg))


def gradient_norm(trainer, batch, advantages, clip):
    log_probs, _ = trainer.actor.evaluate_actions(batch.observations, batch.actions)
    ratio = torch.exp(log_probs - batch.old_log_probs)
    loss = -torch.min(ratio * advantages,
                      torch.clamp(ratio, 1 - clip, 1 + clip) * advantages).mean()
    trainer.actor.zero_grad(set_to_none=True)
    loss.backward()
    return float(torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                for q in trainer.actor.parameters()
                                if q.grad is not None)))


results = {"measured": [], "shuffled": [], "ceiling": []}
for seed in (7, 17, 27):
    seed_everything(0)
    t = dataclasses.replace(base_t, rollout_steps=150,
                            work_dir=f"/tmp/dsrc_fprobe_{seed}")
    trainer = make_trainer(t, base_p, device="cpu")
    buf, _ = trainer.collect_rollout(seed=seed)
    batch = buf.compute_returns_and_advantages(
        gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
    n = batch.advantages.shape[0]
    results["measured"].append(
        gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef))
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    results["shuffled"].append(
        gradient_norm(trainer, batch, batch.advantages[permutation], base_p.clip_coef))
    # Column 0 of the action indices is `desired_speed_bin`; index 0 is `slow`.
    chose_slow = (batch.actions[:, 0] == 0).float()
    ceiling = (chose_slow * 2.0 - 1.0)
    ceiling = (oracle - oracle.mean()) / (oracle.std() + 1e-8)
    results["ceiling"].append(gradient_norm(trainer, batch, ceiling, base_p.clip_coef))
    print(f"  seed {seed}: n {n}, slow share {float(chose_slow.mean()):.3f}", flush=True)

print()
for label in ("measured", "shuffled", "ceiling"):
    values = results[label]
    print(f"  {label:>9}: {statistics.fmean(values):.6f}  {[round(v, 5) for v in values]}")
print(f"\n  measured / shuffled = "
      f"{statistics.fmean(results['measured'])/statistics.fmean(results['shuffled']):.3f}")
print(f"  ceiling  / shuffled = "
      f"{statistics.fmean(results['ceiling'])/statistics.fmean(results['shuffled']):.3f}")
