"""Does raising AV penetration give the action attributable credit?

The last lever that does not require a design decision. Everything else measured so
far leaves the policy-gradient norm at its noise floor: four bin rescalings that
raised the binding share from 6% to 29%, three operating points from 900 to 2400
veh/h, hold lengths from 1 to 20 s, horizons from 10 to 1000 decisions, and four
reward decompositions.

Two demands, so the reading is not a property of one operating point. Same controls:
the shuffled advantage is the floor and an action-correlated advantage is the ceiling.
`ratio - 1` divided by `oracle - 1` estimates the correlation between the advantage
and the action, which is the quantity that matters and is comparable across arms.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
sys.argv = ["x", "--training", "mappo_sumo"]
import torch
from src.config.loaders import load_named_config
from scripts.train_policy import parse_args, load_training_bundle
from src.rl.trainers import TrainingConfig, make_trainer, seed_everything
from src.rl.ppo import PPOConfig
import src.rl.trainers as trainers_module

cfg = load_training_bundle(parse_args())
base_t = TrainingConfig.from_mapping(cfg)
base_p = PPOConfig.from_mapping(cfg.get("training", cfg))
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


print(f"{'demand':>20} {'pen':>5} {'agents':>7} {'meas/floor':>11} "
      f"{'oracle':>7} {'implied corr':>13}")
for demand in ("sumo_saturating", "sumo_capacity_drop"):
    for penetration in (0.25, 0.5, 1.0):
        trainers_module.BasePPOTrainer.env_config = with_penetration(penetration)
        ratios, ceilings, agents = [], [], []
        try:
            for seed in (7, 17, 27):
                seed_everything(0)
                t = dataclasses.replace(base_t, demand=demand, rollout_steps=150,
                                        work_dir=f"/tmp/dsrc_pen_{demand}_{penetration}_{seed}")
                trainer = make_trainer(t, base_p, device="cpu")
                buf, metrics = trainer.collect_rollout(seed=seed)
                batch = buf.compute_returns_and_advantages(
                    gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
                n = batch.advantages.shape[0]
                measured = gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef)
                permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
                shuffled = gradient_norm(trainer, batch, batch.advantages[permutation],
                                         base_p.clip_coef)
                oracle_advantage = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
                oracle_advantage = ((oracle_advantage - oracle_advantage.mean())
                                    / (oracle_advantage.std() + 1e-8))
                ceilings.append(gradient_norm(trainer, batch, oracle_advantage,
                                              base_p.clip_coef) / max(shuffled, 1e-12))
                ratios.append(measured / max(shuffled, 1e-12))
                agents.append(float(metrics.get("active_av_count", 0) or 0))
        finally:
            trainers_module.BasePPOTrainer.env_config = original_env_config
        r, c = statistics.fmean(ratios), statistics.fmean(ceilings)
        print(f"{demand:>20} {penetration:>5} {statistics.fmean(agents):>7.1f} "
              f"{r:>11.3f} {c:>7.1f} {(r - 1) / max(c - 1, 1e-9):>13.4f}", flush=True)
