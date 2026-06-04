from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.rl.models import MultiCategoricalActor
from src.rl.rollout_buffer import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_clip: float = 10.0
    reward_scale: float = 0.05

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> PPOConfig:
        cfg = dict(config or {})
        opt = cfg.get("optimization", {})
        if not isinstance(opt, dict):
            opt = {}
        return cls(
            learning_rate=float(opt.get("learning_rate", cfg.get("learning_rate", cls.learning_rate))),
            clip_coef=float(opt.get("clip_coef", cfg.get("clip_coef", cls.clip_coef))),
            value_coef=float(opt.get("value_coef", cfg.get("value_coef", cls.value_coef))),
            entropy_coef=float(opt.get("entropy_coef", cfg.get("entropy_coef", cls.entropy_coef))),
            max_grad_norm=float(opt.get("max_grad_norm", cfg.get("max_grad_norm", cls.max_grad_norm))),
            update_epochs=int(opt.get("update_epochs", cfg.get("update_epochs", cls.update_epochs))),
            minibatch_size=int(opt.get("minibatch_size", cfg.get("minibatch_size", cls.minibatch_size))),
            gamma=float(opt.get("gamma", cfg.get("gamma", cls.gamma))),
            gae_lambda=float(opt.get("gae_lambda", cfg.get("gae_lambda", cls.gae_lambda))),
            reward_clip=float(opt.get("reward_clip", cfg.get("reward_clip", cls.reward_clip))),
            reward_scale=float(opt.get("reward_scale", cfg.get("reward_scale", cls.reward_scale))),
        )


def ppo_update(
    *,
    actor: MultiCategoricalActor,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    observations = batch.observations.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.old_log_probs.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    value_observations = batch.value_observations.to(device)
    action_masks = batch.action_masks.to(device)
    batch_size = observations.shape[0]
    minibatch_size = max(1, min(config.minibatch_size, batch_size))
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "loss": []}
    for _ in range(config.update_epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            log_probs, entropy = actor.evaluate_actions(observations[indices], actions[indices], action_masks[indices])
            values = critic(value_observations[indices])
            ratio = torch.exp(log_probs - old_log_probs[indices])
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            policy_loss = -torch.min(ratio * advantages[indices], clipped_ratio * advantages[indices]).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[indices])
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], config.max_grad_norm)
            optimizer.step()
            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_mean.detach().cpu()))
            stats["loss"].append(float(loss.detach().cpu()))
    return {key: sum(values) / max(len(values), 1) for key, values in stats.items()}
