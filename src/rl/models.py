from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.rl.actions import ACTION_HEADS, ACTION_VALUES, ActionSpec, action_indices_to_flat, flat_to_action_indices, indices_to_action


def mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MultiCategoricalActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        action_spec: ActionSpec | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_spec = action_spec or ActionSpec("full")
        self.backbone = mlp(input_dim, hidden_sizes, hidden_sizes[-1] if hidden_sizes else input_dim)
        body_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.heads = nn.ModuleDict(
            {head: nn.Linear(body_dim, len(ACTION_VALUES[head])) for head in self.action_spec.active_heads}
        )

    def distributions(self, obs: torch.Tensor) -> dict[str, Categorical]:
        body = self.backbone(obs)
        return {head: Categorical(logits=layer(body)) for head, layer in self.heads.items()}

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        distributions = self.distributions(obs)
        defaults = self.action_spec.default_indices()
        index_rows: list[tuple[int, int, int, int]] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for row_index in range(obs.shape[0]):
            indices = dict(defaults)
            row_log_probs = []
            row_entropies = []
            for head, distribution in distributions.items():
                row_distribution = Categorical(logits=distribution.logits[row_index])
                if deterministic:
                    sampled = torch.argmax(row_distribution.logits, dim=-1)
                else:
                    sampled = row_distribution.sample()
                indices[head] = int(sampled.item())
                row_log_probs.append(row_distribution.log_prob(sampled))
                row_entropies.append(row_distribution.entropy())
            index_rows.append(action_indices_to_flat(indices))
            log_probs.append(torch.stack(row_log_probs).sum() if row_log_probs else torch.tensor(0.0, device=obs.device))
            entropies.append(torch.stack(row_entropies).sum() if row_entropies else torch.tensor(0.0, device=obs.device))
        actions = [indices_to_action(flat_to_action_indices(indices), self.action_spec) for indices in index_rows]
        return (
            actions,
            torch.tensor(index_rows, dtype=torch.long, device=obs.device),
            torch.stack(log_probs),
            torch.stack(entropies),
        )

    def evaluate_actions(self, obs: torch.Tensor, action_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distributions = self.distributions(obs)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        head_to_column = {head: index for index, head in enumerate(ACTION_HEADS)}
        for head, distribution in distributions.items():
            column = head_to_column[head]
            head_actions = action_indices[:, column]
            log_probs.append(distribution.log_prob(head_actions))
            entropies.append(distribution.entropy())
        if not log_probs:
            batch = obs.shape[0]
            return torch.zeros(batch, device=obs.device), torch.zeros(batch, device=obs.device)
        return torch.stack(log_probs, dim=0).sum(dim=0), torch.stack(entropies, dim=0).sum(dim=0)

    def act_mapping(
        self,
        agent_ids: list[str],
        obs_tensor: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, dict[str, str]]:
        with torch.no_grad():
            actions, _, _, _ = self.sample(obs_tensor, deterministic=deterministic)
        return {agent_id: action for agent_id, action in zip(agent_ids, actions, strict=True)}

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"input_dim": self.input_dim, "action_profile": self.action_spec.profile}


class LocalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.value = mlp(input_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)


class GlobalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.value = mlp(input_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(state).squeeze(-1)


def load_actor_from_checkpoint(checkpoint: Mapping[str, Any], map_location: str | torch.device = "cpu") -> MultiCategoricalActor:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    input_dim = int(metadata.get("input_dim", checkpoint.get("input_dim", 0)))
    action_profile = str(metadata.get("action_profile", checkpoint.get("action_profile", "full")))
    hidden_sizes = tuple(int(value) for value in checkpoint.get("hidden_sizes", (128, 128)))
    actor = MultiCategoricalActor(input_dim, hidden_sizes=hidden_sizes, action_spec=ActionSpec(action_profile))  # type: ignore[arg-type]
    actor.load_state_dict(checkpoint["state_dict"])
    actor.to(map_location)
    actor.eval()
    return actor
