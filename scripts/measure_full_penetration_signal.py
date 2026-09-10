"""The one arm with a positive reading, on ten seeds instead of three.

`sumo_saturating` at 100% penetration gave a mean z of +0.82 in the floor-distribution
measurement -- under one standard deviation on three seeds, and the largest of any arm
tried. Three seeds cannot separate that from noise. This runs ten, and reports the
mean z with its standard error, so the arm is either dismissed on evidence or kept.

A mean z of +0.82 over ten seeds would have a standard error near 0.32 if the per-seed
spread holds, which would make it about 2.6 standard errors and worth pursuing. A mean
near zero settles it the other way.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
sys.argv = ["x", "--training", "mappo_sumo"]
import torch
from scripts.train_policy import parse_args, load_training_bundle
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.ppo import PPOConfig
import src.rl.trainers as trainers_module

cfg = load_training_bundle(parse_args())
base_t = TrainingConfig.from_mapping(cfg)
base_p = PPOConfig.from_mapping(cfg.get("training", cfg))
PERMUTATIONS = 60
SEEDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)
original_env_config = trainers_module.BasePPOTrainer.env_config


def env_config(self):
    config = original_env_config(self)
    demand = dict(config["demand"])
    demand["av_penetration"] = 1.0
    config["demand"] = demand
    return config


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


trainers_module.BasePPOTrainer.env_config = env_config
zs = []
try:
    for seed in SEEDS:
        seed_everything(0)
        t = dataclasses.replace(base_t, demand="sumo_saturating", rollout_steps=150,
                                work_dir=f"/tmp/dsrc_p100_{seed}")
        trainer = make_trainer(t, base_p, device="cpu")
        buf, _ = trainer.collect_rollout(seed=seed)
        batch = buf.compute_returns_and_advantages(
            gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
        n = batch.advantages.shape[0]
        measured = gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef)
        generator = torch.Generator().manual_seed(seed)
        floor = [gradient_norm(trainer, batch,
                               batch.advantages[torch.randperm(n, generator=generator)],
                               base_p.clip_coef)
                 for _ in range(PERMUTATIONS)]
        mean, sd = statistics.fmean(floor), statistics.pstdev(floor)
        z = (measured - mean) / max(sd, 1e-12)
        zs.append(z)
        print(f"  seed {seed:>3}: n {n:5d} measured {measured:.5f} floor {mean:.5f} "
              f"+/- {sd:.5f}  z {z:+.2f}", flush=True)
finally:
    trainers_module.BasePPOTrainer.env_config = original_env_config

mean_z = statistics.fmean(zs)
se = statistics.stdev(zs) / len(zs) ** 0.5
print(f"\nsumo_saturating at 100% penetration, {len(zs)} seeds:")
print(f"  mean z {mean_z:+.3f} +/- {se:.3f} (standard error)")
print(f"  {'ABOVE the floor' if abs(mean_z) > 2 * se else 'AT the floor'} "
      f"on the two-standard-error bar this project uses")
