from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from src.envs.wrappers import HEADWAY_BINS, LANE_PREFERENCES, MERGE_MODES, SPEED_BINS


ActionProfile = Literal["speed_only", "speed_headway", "full"]
ACTION_HEADS = ("desired_speed_bin", "desired_headway_bin", "lane_preference", "merge_mode")
ACTION_VALUES: dict[str, tuple[str, ...]] = {
    "desired_speed_bin": tuple(SPEED_BINS),
    "desired_headway_bin": tuple(HEADWAY_BINS),
    "lane_preference": tuple(LANE_PREFERENCES),
    "merge_mode": tuple(MERGE_MODES),
}
FORCED_ACTIONS: dict[str, str] = {
    "desired_headway_bin": "normal",
    "lane_preference": "keep",
    "merge_mode": "normal",
}


@dataclass(frozen=True)
class ActionSpec:
    profile: ActionProfile = "full"

    @property
    def active_heads(self) -> tuple[str, ...]:
        if self.profile == "speed_only":
            return ("desired_speed_bin",)
        if self.profile == "speed_headway":
            return ("desired_speed_bin", "desired_headway_bin")
        if self.profile == "full":
            return ACTION_HEADS
        raise ValueError(f"unsupported action profile '{self.profile}'")

    def head_size(self, head: str) -> int:
        return len(ACTION_VALUES[head])

    def default_indices(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for head in ACTION_HEADS:
            value = FORCED_ACTIONS.get(head, ACTION_VALUES[head][0])
            indices[head] = ACTION_VALUES[head].index(value)
        return indices


def indices_to_action(indices: Mapping[str, int], spec: ActionSpec) -> dict[str, str]:
    defaults = spec.default_indices()
    action: dict[str, str] = {}
    for head in ACTION_HEADS:
        index = int(indices.get(head, defaults[head]))
        action[head] = ACTION_VALUES[head][index]
    return action


def action_to_indices(action: Mapping[str, Any], spec: ActionSpec | None = None) -> dict[str, int]:
    indices: dict[str, int] = {}
    for head in ACTION_HEADS:
        values = ACTION_VALUES[head]
        value = str(action[head])
        if value not in values:
            raise ValueError(f"unsupported {head} '{value}'")
        indices[head] = values.index(value)
    if spec is None:
        return indices
    defaults = spec.default_indices()
    for head in ACTION_HEADS:
        if head not in spec.active_heads:
            indices[head] = defaults[head]
    return indices


def action_indices_to_flat(indices: Mapping[str, int]) -> tuple[int, int, int, int]:
    return tuple(int(indices[head]) for head in ACTION_HEADS)  # type: ignore[return-value]


def flat_to_action_indices(values: tuple[int, int, int, int]) -> dict[str, int]:
    return {head: int(values[index]) for index, head in enumerate(ACTION_HEADS)}
