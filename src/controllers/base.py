from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from src.envs.base_ctde_env import AVActionMap, AVObservationMap, GlobalState


@dataclass(frozen=True)
class ControllerMetadata:
    name: str
    family: str
    version: str = "v1"
    requires_global_state: bool = False


class BaseController(ABC):
    """Stable controller contract shared by baselines and learned policies."""

    metadata: ControllerMetadata

    def __init__(self, metadata: ControllerMetadata) -> None:
        self.metadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    def reset(
        self,
        env_metadata: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Reset controller state at episode start."""

    @abstractmethod
    def act(
        self,
        local_obs: AVObservationMap,
        global_state: GlobalState | None = None,
    ) -> AVActionMap:
        """Produce one public action per AV identifier."""

