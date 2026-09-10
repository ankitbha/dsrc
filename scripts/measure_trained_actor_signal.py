"""Does a partly trained policy have more action signal than a fresh one?

Every reading in tasks 105 to 107 was taken at a randomly initialised actor. If the
correlation between the advantage and the action rises as the policy trains, those
readings describe a starting condition rather than the problem, and the conclusion
drawn from them is wrong. This is the control on that.

The arms are the same actor architecture with different weights: the initialisation
the earlier readings used, and the checkpoint the seed-7 run has reached. Everything
else -- environment, seeds, reward, controls -- is identical.
"""
import sys, dataclasses, statistics
sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
from pathlib import Path

# Captured BEFORE argv is replaced for the config loader, which is what silently
# turned the checkpoint path into None on the first attempt.
CHECKPOINT = Path(sys.argv[1])
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


print(f"{'actor':>14} {'meas/floor':>11} {'per seed':>28} {'ceiling':>7} {'implied corr':>13}")
for label in ("fresh", "trained"):
    ratios, ceilings = [], []
    for seed in (7, 17, 27):
        seed_everything(0)
        t = dataclasses.replace(base_t, rollout_steps=150,
                                work_dir=f"/tmp/dsrc_tr_{label}_{seed}")
        trainer = make_trainer(t, base_p, device="cpu")
        if label == "trained":
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            trainer.actor.load_state_dict(payload["state_dict"])
        buf, _ = trainer.collect_rollout(seed=seed)
        batch = buf.compute_returns_and_advantages(
            gamma=base_p.gamma, gae_lambda=base_p.gae_lambda, group_by_agent=True)
        n = batch.advantages.shape[0]
        measured = gradient_norm(trainer, batch, batch.advantages, base_p.clip_coef)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        shuffled = gradient_norm(trainer, batch, batch.advantages[permutation],
                                 base_p.clip_coef)
        ceiling = (batch.actions[:, 0] == 0).float() * 2.0 - 1.0
        ceiling = (oracle - oracle.mean()) / (oracle.std() + 1e-8)
        ceilings.append(gradient_norm(trainer, batch, ceiling, base_p.clip_coef)
                        / max(shuffled, 1e-12))
        ratios.append(measured / max(shuffled, 1e-12))
    r, c = statistics.fmean(ratios), statistics.fmean(ceilings)
    print(f"{label:>14} {r:>11.3f} {str([round(x, 3) for x in ratios]):>28} "
          f"{c:>7.1f} {(r - 1) / max(c - 1, 1e-9):>13.4f}", flush=True)
