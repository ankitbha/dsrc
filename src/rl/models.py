from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.rl.actions import ACTION_HEADS, ACTION_VALUES, ActionSpec, flat_to_action_indices, indices_to_action


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

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        distributions = self.distributions(obs)
        defaults = self.action_spec.default_indices()
        batch_size = obs.shape[0]
        action_indices = torch.empty((batch_size, len(ACTION_HEADS)), dtype=torch.long, device=obs.device)
        for column, head in enumerate(ACTION_HEADS):
            action_indices[:, column] = defaults[head]
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        head_to_column = {head: index for index, head in enumerate(ACTION_HEADS)}
        for head, distribution in distributions.items():
            column = head_to_column[head]
            head_distribution = distribution
            if action_masks is not None:
                head_distribution = Categorical(
                    logits=_masked_logits(
                        distribution.logits,
                        action_masks[:, column, : distribution.logits.shape[-1]],
                        self.action_spec.default_indices()[head],
                    )
                )
            if deterministic:
                sampled = torch.argmax(head_distribution.logits, dim=-1)
            else:
                sampled = head_distribution.sample()
            action_indices[:, column] = sampled
            log_prob_terms.append(head_distribution.log_prob(sampled))
            entropy_terms.append(head_distribution.entropy())
        log_probs = (
            torch.stack(log_prob_terms, dim=0).sum(dim=0)
            if log_prob_terms
            else torch.zeros(batch_size, device=obs.device)
        )
        entropies = (
            torch.stack(entropy_terms, dim=0).sum(dim=0)
            if entropy_terms
            else torch.zeros(batch_size, device=obs.device)
        )
        actions = [
            indices_to_action(flat_to_action_indices(tuple(int(value) for value in row.tolist())), self.action_spec)
            for row in action_indices
        ]
        return (
            actions,
            action_indices,
            log_probs,
            entropies,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_indices: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distributions = self.distributions(obs)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        head_to_column = {head: index for index, head in enumerate(ACTION_HEADS)}
        for head, distribution in distributions.items():
            column = head_to_column[head]
            head_actions = action_indices[:, column]
            head_distribution = distribution
            if action_masks is not None:
                head_distribution = Categorical(
                    logits=_masked_logits(
                        distribution.logits,
                        action_masks[:, column, : distribution.logits.shape[-1]],
                        self.action_spec.default_indices()[head],
                    )
                )
            log_probs.append(head_distribution.log_prob(head_actions))
            entropies.append(head_distribution.entropy())
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
        action_masks: torch.Tensor | None = None,
    ) -> dict[str, dict[str, str]]:
        with torch.no_grad():
            actions, _, _, _ = self.sample(obs_tensor, deterministic=deterministic, action_masks=action_masks)
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


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor, default_index: int) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool).clone()
    all_invalid = ~mask.any(dim=-1)
    if bool(all_invalid.any()):
        mask[all_invalid] = False
        mask[all_invalid, int(default_index)] = True
    return logits.masked_fill(~mask, -1.0e9)
