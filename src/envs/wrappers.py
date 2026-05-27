from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.envs.base_ctde_env import AVAction, AVActionMap, LanePreference


LANE_PREFERENCES: tuple[LanePreference, ...] = (
    "keep",
    "left",
    "right",
    "leftmost",
    "rightmost",
)


def default_agent_ids(count: int) -> list[str]:
    """Create canonical AV identifiers for wrappers and tests."""
    return [f"av_{index}" for index in range(count)]


def tuple_observation_to_mapping(
    agent_ids: Iterable[str],
    observations: Iterable[Any],
) -> dict[str, Any]:
    """Convert tuple-based simulator observations into public AV-indexed mappings."""
    return {agent_id: observation for agent_id, observation in zip(agent_ids, observations, strict=True)}


def validate_action(action: Mapping[str, Any]) -> AVAction:
    """Validate one public AV action object."""
    if "desired_speed" not in action:
        raise ValueError("action is missing required field 'desired_speed'")
    if "desired_lane" not in action:
        raise ValueError("action is missing required field 'desired_lane'")

    desired_speed = float(action["desired_speed"])
    desired_lane = action["desired_lane"]
    if desired_lane not in LANE_PREFERENCES:
        raise ValueError(f"unsupported desired_lane '{desired_lane}'")

    return {
        "desired_speed": desired_speed,
        "desired_lane": desired_lane,
    }


def validate_action_mapping(
    av_actions: Mapping[str, Mapping[str, Any]],
    expected_agent_ids: Iterable[str] | None = None,
) -> dict[str, AVAction]:
    """Validate and normalize a public AV action mapping."""
    normalized = {agent_id: validate_action(action) for agent_id, action in av_actions.items()}

    if expected_agent_ids is not None:
        expected = set(expected_agent_ids)
        actual = set(normalized)
        if actual != expected:
            raise ValueError(f"action mapping keys {sorted(actual)} do not match expected AV ids {sorted(expected)}")

    return normalized

