from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, Sequence, TypeAlias, TypedDict


LanePreference: TypeAlias = Literal["keep", "left", "right", "leftmost", "rightmost"]


class AVAction(TypedDict):
    desired_speed: float
    desired_lane: LanePreference


LocalObservation: TypeAlias = Mapping[str, Any]
AVObservationMap: TypeAlias = Mapping[str, LocalObservation]
RewardMap: TypeAlias = Mapping[str, float]
GlobalState: TypeAlias = Mapping[str, Any]
SegmentMetrics: TypeAlias = Mapping[str, Mapping[str, Any]]
EpisodeSummary: TypeAlias = Mapping[str, Any]
InfoDict: TypeAlias = Mapping[str, Any]
AVActionMap: TypeAlias = Mapping[str, AVAction]


class BaseCTDEEnv(ABC):
    """Stable public DSRC environment contract.

    Concrete topology wrappers should adapt simulator-specific behavior into this API.
    """

    topology_id: str
    agent_ids: Sequence[str]
    config: Mapping[str, Any]

    @abstractmethod
    def reset(
        self,
        config: Mapping[str, Any] | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[AVObservationMap, InfoDict]:
        """Reset the environment and return AV-indexed local observations."""

    @abstractmethod
    def step(
        self,
        av_actions: AVActionMap,
    ) -> tuple[AVObservationMap, RewardMap, bool, bool, InfoDict]:
        """Step the environment with public AV actions."""

    @abstractmethod
    def get_local_observations(self) -> AVObservationMap:
        """Return the latest AV-indexed local observations."""

    @abstractmethod
    def get_global_state(self) -> GlobalState:
        """Return centralized state used by CTDE critics and analysis code."""

    @abstractmethod
    def get_segment_metrics(self) -> SegmentMetrics:
        """Return canonical per-segment metrics keyed by segment identifier."""

    @abstractmethod
    def get_episode_summary(self) -> EpisodeSummary:
        """Return the canonical episode summary record."""

