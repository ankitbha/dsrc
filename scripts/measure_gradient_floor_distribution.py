"""Give the floor an error bar, which it never had.

Every ratio reported in tasks 105 to 107 divided the measured gradient norm by the
norm from ONE random permutation of the advantages. One permutation is a single draw
from the floor's distribution, not the floor, so a ratio of 1.4 was never shown to be
outside it. This estimates the floor from many permutations per rollout and reports
where the measured value sits in that distribution, in standard deviations.

If a measured value is inside the permutation distribution it is a floor reading. If
it is outside, that arm has signal and the conclusion drawn from it was wrong.
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
original_env_config = trainers_module.BasePPOTrainer.env_config


def with_penetration(penetration):
    def env_config(self):
        config = original_env_config(self)
        demand = dict(config["demand"])
        demand["av_penetration"] = penetration
        config["demand"] = demand
        return config
    return env_config


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


ARMS = [
    ("capacity_drop pen 0.25", "sumo_capacity_drop", 0.25, base_p.gamma, base_p.gae_lambda),
    ("saturating pen 0.25", "sumo_saturating", 0.25, base_p.gamma, base_p.gae_lambda),
    ("saturating pen 1.00", "sumo_saturating", 1.00, base_p.gamma, base_p.gae_lambda),
    ("capacity_drop gamma .999", "sumo_capacity_drop", 0.25, 0.999, 0.995),
]

print(f"{'arm':>26} {'seed':>5} {'measured':>9} {'floor mean':>11} {'floor sd':>9} "
      f"{'z':>7}")
for label, demand, penetration, gamma, lam in ARMS:
    trainers_module.BasePPOTrainer.env_config = with_penetration(penetration)
    zs = []
    try:
        for seed in (7, 17, 27):
            seed_everything(0)
            t = dataclasses.replace(base_t, demand=demand, rollout_steps=150,
                                    work_dir=f"/tmp/dsrc_fd_{demand}_{penetration}_{seed}")
            trainer = make_trainer(t, base_p, device="cpu")
            buf, _ = trainer.collect_rollout(seed=seed)
            batch = buf.compute_returns_and_advantages(
                gamma=gamma, gae_lambda=lam, group_by_agent=True)
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
            print(f"{label:>26} {seed:>5} {measured:>9.5f} {mean:>11.5f} {sd:>9.5f} "
                  f"{z:>+7.2f}", flush=True)
    finally:
        trainers_module.BasePPOTrainer.env_config = original_env_config
    print(f"{label:>26} {'mean':>5} {'':>9} {'':>11} {'':>9} {statistics.fmean(zs):>+7.2f}",
          flush=True)
