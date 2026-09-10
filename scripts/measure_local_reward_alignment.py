"""The one vehicle-level arm whose reward is NOT shared, under the valid null.

Every other arm measured pays all agents the same reward at each step, so their
reward sequences are identical and the cross-agent variation in the advantage comes
from the critic. At the vehicle level that IS the configuration under test -- a
shared team reward is the credit-assignment problem -- but the per-agent local reward
of task 103 is the exception, and it was only ever measured against the permutation
floor that has since been shown invalid.

`local_reward_weight 0.5` blends each agent's own neighbourhood term with the team
reward, so the reward sequences genuinely differ across agents.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/scripts")
import torch
from measure_action_alignment import alignment_z, action_advantage_correlation
from src.config.loaders import load_named_config
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.ppo import PPOConfig

cfg = load_named_config("training", "mappo_sumo")
keys = ("topology", "demand", "human_model", "duration_steps", "dt")
bundle = {"training": cfg, "env": {k: cfg[k] for k in keys if k in cfg}, "seed": 7}
base_t = TrainingConfig.from_mapping(bundle)
base_p = PPOConfig.from_mapping(cfg)

print(f"{'arm':>14} {'seed':>5} {'n':>7} {'z':>7} {'corr(slow,A)':>13}")
for label, weight in (("team only", 0.0), ("local 0.5", 0.5)):
    zs, corrs = [], []
    for seed in (7, 17, 27):
        # Vary the network draw with the seed, not just the traffic: with
        # seed_everything(0) the actor and critic were bit-identical across seeds
        # and the cross-seed standard error understated the uncertainty.
        seed_everything(seed)
        t = dataclasses.replace(base_t, rollout_steps=150,
                                work_dir=f"/tmp/dsrc_localalign_{__import__('os').getpid()}_{weight}_{seed}")
        p = dataclasses.replace(base_p, local_reward_weight=weight)
        trainer = make_trainer(t, p, device="cpu")
        buf, _ = trainer.collect_rollout(seed=seed)
        batch = buf.compute_returns_and_advantages(
            gamma=p.gamma, gae_lambda=p.gae_lambda, group_by_agent=True)
        z, *_ = alignment_z(trainer, batch.observations, batch.actions,
                            batch.old_log_probs, batch.advantages,
                            clip=p.clip_coef, resamples=60, seed=seed)
        corr = action_advantage_correlation(batch.actions, batch.advantages)
        zs.append(z); corrs.append(corr)
        print(f"{label:>14} {seed:>5} {batch.advantages.shape[0]:>7} {z:>+7.2f} "
              f"{corr:>+13.4f}", flush=True)
    se = statistics.stdev(zs) / len(zs) ** 0.5
    print(f"{label:>14} {'MEAN':>5} {'':>7} {statistics.fmean(zs):>+7.2f} +/- {se:.2f}"
          f"   corr {statistics.fmean(corrs):+.4f}\n", flush=True)
