from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.controllers import BaseController, ControllerMetadata
from src.envs.base_ctde_env import AVActionMap, AVObservationMap, GlobalState
from src.rl.encoders import encode_action_mask_batch, encode_local_batch
from src.rl.models import MultiCategoricalActor, load_actor_from_checkpoint


class LearnedPolicyController(BaseController):
    """Evaluation-time controller that uses only local observations."""

    def __init__(
        self,
        actor: MultiCategoricalActor,
        *,
        name: str = "learned_policy",
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            ControllerMetadata(
                name=name,
                family="rl",
                requires_global_state=False,
                cooperation_mode="local_aggregate",
                safety_mode="integrated_rl",
            )
        )
        self.actor = actor.to(device)
        self.actor.eval()
        self.deterministic = deterministic
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> LearnedPolicyController:
        checkpoint = torch.load(path, map_location=device)
        actor = load_actor_from_checkpoint(checkpoint, map_location=device)
        return cls(actor, deterministic=deterministic, device=device)

    def act(
        self,
        local_obs: AVObservationMap,
        global_state: GlobalState | None = None,
    ) -> AVActionMap:
        if global_state is not None:
            raise ValueError("learned policy execution must not consume global_state")
        agent_ids, obs_tensor = encode_local_batch(local_obs)
        if not agent_ids:
            return {}
        obs_tensor = obs_tensor.to(self.device)
        action_masks = encode_action_mask_batch(local_obs, agent_ids, self.actor.action_spec).to(self.device)
        return self.actor.act_mapping(agent_ids, obs_tensor, deterministic=self.deterministic, action_masks=action_masks)
