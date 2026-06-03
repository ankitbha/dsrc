from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    value_observations: torch.Tensor


class RolloutBuffer:
    """Flat transition storage for dynamic active-agent rollouts."""

    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.value_observations: list[torch.Tensor] = []
        self.agent_ids: list[str] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
        value_observation: torch.Tensor | None = None,
        agent_id: str = "",
    ) -> None:
        self.observations.append(observation.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.values.append(float(value.detach().cpu().item()))
        self.dones.append(bool(done))
        self.value_observations.append((value_observation if value_observation is not None else observation).detach().cpu())
        self.agent_ids.append(agent_id)

    def clear(self) -> None:
        self.__init__()

    def compute_returns_and_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalize_advantages: bool = True,
        group_by_agent: bool = True,
    ) -> RolloutBatch:
        if not self.rewards:
            raise ValueError("cannot build a rollout batch from an empty buffer")
        advantages = [0.0 for _ in self.rewards]
        by_agent: dict[str, list[int]] = {}
        for index, agent_id in enumerate(self.agent_ids):
            key = agent_id or "__shared__" if group_by_agent else "__shared__"
            by_agent.setdefault(key, []).append(index)
        for indices in by_agent.values():
            last_gae = 0.0
            for position in reversed(range(len(indices))):
                index = indices[position]
                has_next = position < len(indices) - 1
                next_index = indices[position + 1] if has_next else -1
                next_value = self.values[next_index] if has_next else 0.0
                next_non_terminal = 0.0 if self.dones[index] else 1.0
                delta = self.rewards[index] + gamma * next_value * next_non_terminal - self.values[index]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[index] = last_gae
        returns = torch.tensor([advantage + value for advantage, value in zip(advantages, self.values)], dtype=torch.float32)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        if normalize_advantages and advantage_tensor.numel() > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
        return RolloutBatch(
            observations=torch.stack(self.observations),
            actions=torch.stack(self.actions).long(),
            old_log_probs=torch.stack(self.log_probs).float(),
            returns=returns,
            advantages=advantage_tensor,
            value_observations=torch.stack(self.value_observations),
        )
