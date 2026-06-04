from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from src.rl.actions import ACTION_HEADS, ACTION_VALUES, ActionSpec


LOCAL_OBS_FIELDS: tuple[str, ...] = (
    "is_active",
    "ego_speed",
    "ego_acceleration",
    "ego_lane",
    "ego_headway_s",
    "target_headway_s",
    "time_since_last_lane_change",
    "lane_changes_last_km",
    "distance_to_next_merge",
    "distance_to_downstream_bottleneck",
    "leader_gap",
    "leader_relative_speed",
    "follower_gap",
    "follower_relative_speed",
    "left_lane_front_gap",
    "left_lane_rear_gap",
    "right_lane_front_gap",
    "right_lane_rear_gap",
    "target_lane_front_gap",
    "target_lane_rear_gap",
    "target_lane_rear_required_decel",
    "downstream_congestion_estimate",
    "merge_pressure",
    "segment_target_speed",
    "uncongested_low_speed_flag",
    "local_density_bin",
    "local_mean_speed_bin",
    "local_queue_estimate",
    "active_vehicle_count_local",
    "active_av_count_local",
    "nearby_av_count",
    "nearby_av_density",
    "nearby_av_mean_speed",
)
COOPERATION_FIELDS: tuple[str, ...] = (
    "segment_target_speed",
    "merge_pressure",
    "downstream_congestion_estimate",
)
LANE_DISTRIBUTION_LANES = ("0", "1", "2")

GLOBAL_FIELDS: tuple[str, ...] = (
    "time",
    "active_vehicle_count",
    "active_av_count",
    "completed_vehicle_count",
)
PHYSICAL_GLOBAL_FIELDS: tuple[str, ...] = (
    "time",
    "active_vehicle_count",
    "active_av_count",
)
SEGMENT_FIELDS: tuple[str, ...] = (
    "vehicle_count",
    "av_count",
    "mean_speed",
    "speed_std",
    "density",
    "queue_length",
    "jam_fraction",
    "inflow",
    "outflow",
    "rolling_roadblock_score",
    "all_lane_av_low_speed_occupancy",
)
DEMAND_FIELDS: tuple[str, ...] = (
    "current_vehicles_per_hour",
    "av_penetration",
    "spawned_vehicle_count",
    "completed_vehicle_count",
    "skipped_spawn_count",
)
PHYSICAL_DEMAND_FIELDS: tuple[str, ...] = (
    "current_vehicles_per_hour",
    "av_penetration",
)


def local_obs_dim() -> int:
    return len(LOCAL_OBS_FIELDS) + len(COOPERATION_FIELDS) + len(LANE_DISTRIBUTION_LANES)


def encode_local_observation(obs: Mapping[str, Any]) -> torch.Tensor:
    values = [_number(obs.get(field)) for field in LOCAL_OBS_FIELDS]
    cooperation = obs.get("cooperation", {})
    if not isinstance(cooperation, Mapping):
        cooperation = {}
    values.extend(_number(cooperation.get(field)) for field in COOPERATION_FIELDS)
    lane_distribution = obs.get("nearby_av_lane_distribution", {})
    if not isinstance(lane_distribution, Mapping):
        lane_distribution = {}
    values.extend(_number(lane_distribution.get(lane_id)) for lane_id in LANE_DISTRIBUTION_LANES)
    return torch.tensor(values, dtype=torch.float32)


def global_state_dim(max_segments: int = 10, max_branches: int = 6) -> int:
    return len(GLOBAL_FIELDS) + max_segments * len(SEGMENT_FIELDS) + len(DEMAND_FIELDS) + max_branches * 3


def physical_global_state_dim(max_segments: int = 10) -> int:
    return len(PHYSICAL_GLOBAL_FIELDS) + max_segments * len(SEGMENT_FIELDS) + len(PHYSICAL_DEMAND_FIELDS)


def encode_global_state(
    state: Mapping[str, Any],
    *,
    max_segments: int = 10,
    max_branches: int = 6,
) -> torch.Tensor:
    values = [_number(state.get(field)) for field in GLOBAL_FIELDS]
    segment_state = state.get("segment_state", {})
    if not isinstance(segment_state, Mapping):
        segment_state = {}
    for segment_id in sorted(segment_state)[:max_segments]:
        segment = segment_state[segment_id]
        if not isinstance(segment, Mapping):
            segment = {}
        values.extend(_number(segment.get(field)) for field in SEGMENT_FIELDS)
    values.extend([0.0] * ((max_segments - min(len(segment_state), max_segments)) * len(SEGMENT_FIELDS)))

    demand_state = state.get("demand_state", {})
    if not isinstance(demand_state, Mapping):
        demand_state = {}
    values.extend(_number(demand_state.get(field)) for field in DEMAND_FIELDS)

    branch_state = state.get("branch_state", {})
    if not isinstance(branch_state, Mapping):
        branch_state = {}
    completed = branch_state.get("per_branch_completed", {})
    spawned = branch_state.get("per_branch_spawned", {})
    travel = branch_state.get("branch_travel_time_mean", {})
    if not isinstance(completed, Mapping):
        completed = {}
    if not isinstance(spawned, Mapping):
        spawned = {}
    if not isinstance(travel, Mapping):
        travel = {}
    branch_ids = sorted(set(completed) | set(spawned) | set(travel))[:max_branches]
    for branch_id in branch_ids:
        values.extend((_number(spawned.get(branch_id)), _number(completed.get(branch_id)), _number(travel.get(branch_id))))
    values.extend([0.0] * ((max_branches - len(branch_ids)) * 3))
    return torch.tensor(values, dtype=torch.float32)


def encode_physical_global_state(
    state: Mapping[str, Any],
    *,
    max_segments: int = 10,
) -> torch.Tensor:
    values = [_number(state.get(field)) for field in PHYSICAL_GLOBAL_FIELDS]
    segment_state = state.get("segment_state", {})
    if not isinstance(segment_state, Mapping):
        segment_state = {}
    for segment_id in sorted(segment_state)[:max_segments]:
        segment = segment_state[segment_id]
        if not isinstance(segment, Mapping):
            segment = {}
        values.extend(_number(segment.get(field)) for field in SEGMENT_FIELDS)
    values.extend([0.0] * ((max_segments - min(len(segment_state), max_segments)) * len(SEGMENT_FIELDS)))

    demand_state = state.get("demand_state", {})
    if not isinstance(demand_state, Mapping):
        demand_state = {}
    values.extend(_number(demand_state.get(field)) for field in PHYSICAL_DEMAND_FIELDS)
    return torch.tensor(values, dtype=torch.float32)


def encode_local_batch(observations: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], torch.Tensor]:
    agent_ids = sorted(observations)
    if not agent_ids:
        return [], torch.empty((0, local_obs_dim()), dtype=torch.float32)
    return agent_ids, torch.stack([encode_local_observation(observations[agent_id]) for agent_id in agent_ids])


def action_mask_shape() -> tuple[int, int]:
    return len(ACTION_HEADS), max(len(values) for values in ACTION_VALUES.values())


def encode_action_mask(obs: Mapping[str, Any], spec: ActionSpec | None = None) -> torch.Tensor:
    spec = spec or ActionSpec("full")
    heads, values_per_head = action_mask_shape()
    mask_tensor = torch.ones((heads, values_per_head), dtype=torch.bool)
    raw_mask = obs.get("action_mask", {})
    if not isinstance(raw_mask, Mapping):
        raw_mask = {}
    active_heads = set(spec.active_heads)
    for head_index, head in enumerate(ACTION_HEADS):
        values = ACTION_VALUES[head]
        if head not in active_heads:
            mask_tensor[head_index, :] = False
            default_index = spec.default_indices()[head]
            mask_tensor[head_index, default_index] = True
            continue
        head_mask = raw_mask.get(head, {})
        if not isinstance(head_mask, Mapping):
            continue
        for value_index, value in enumerate(values):
            mask_tensor[head_index, value_index] = bool(head_mask.get(value, True))
        if not bool(mask_tensor[head_index, : len(values)].any()):
            mask_tensor[head_index, : len(values)] = True
    return mask_tensor


def encode_action_mask_batch(
    observations: Mapping[str, Mapping[str, Any]],
    agent_ids: list[str],
    spec: ActionSpec | None = None,
) -> torch.Tensor:
    heads, values_per_head = action_mask_shape()
    if not agent_ids:
        return torch.empty((0, heads, values_per_head), dtype=torch.bool)
    return torch.stack([encode_action_mask(observations[agent_id], spec) for agent_id in agent_ids])


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 200.0 if result > 0 else -200.0
    return result
