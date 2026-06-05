from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.envs.base_ctde_env import AVAction
from src.envs.wrappers import decode_headway_bin, decode_speed_bin, lane_preference_to_action
from src.safety.constraints import SafetyConstraints
from src.safety.etiquette import is_all_lane_low_speed_occupancy, is_low_speed_uncongested, is_passing_lane_slow_hold


@dataclass
class SafetyState:
    last_lane_change_time_s: float | None = None
    lane_changes_last_km: int = 0
    distance_since_window_start_m: float = 0.0
    last_lane_index: Any | None = None


@dataclass(frozen=True)
class SafetyContext:
    time_s: float
    ego_speed_mps: float = 0.0
    free_flow_speed_mps: float = 30.0
    min_contextual_speed_mps: float = 12.0
    local_density_veh_per_km: float = 0.0
    downstream_congested: bool = False
    leader_gap_m: float = float("inf")
    leader_relative_speed_mps: float = 0.0
    follower_gap_m: float = float("inf")
    follower_relative_speed_mps: float = 0.0
    target_lane_exists: bool = True
    target_lane_front_gap_m: float = float("inf")
    target_lane_front_relative_speed_mps: float = 0.0
    target_lane_rear_gap_m: float = float("inf")
    target_lane_rear_relative_speed_mps: float = 0.0
    target_lane_rear_required_decel_mps2: float = 0.0
    all_lanes_av_occupied: bool = False
    av_mean_speed_mps: float = 30.0
    in_passing_lane: bool = False
    local_mean_speed_mps: float = 30.0
    near_merge: bool = False


@dataclass(frozen=True)
class SafetyDecision:
    target_speed_mps: float
    target_headway_s: float
    lane_action: str | None
    acceleration_mps2: float = 0.0
    emergency_override: bool = False
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    penalty_terms: dict[str, float] = field(default_factory=dict)


def empty_diagnostics() -> dict[str, list[dict[str, Any]]]:
    return {
        "safety_masked_action": [],
        "etiquette_blocked_action": [],
        "follower_disruption_blocked": [],
        "external_safety_override": [],
        "simulator_blocked_action": [],
    }


def apply_safety_layer(
    action: AVAction,
    state: SafetyState,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
    agent_id: str | None = None,
) -> SafetyDecision:
    constraints = constraints or SafetyConstraints()
    diagnostics = empty_diagnostics()
    target_speed = decode_speed_bin(
        action["desired_speed_bin"],
        free_flow_speed_mps=context.free_flow_speed_mps,
        min_contextual_speed_mps=context.min_contextual_speed_mps,
    )
    target_headway = decode_headway_bin(action["desired_headway_bin"])
    if action["merge_mode"] == "create_gap":
        target_headway += constraints.merge_gap_headway_bonus_s
    lane_action = None if action["merge_mode"] == "hold_lane" else lane_preference_to_action(action["lane_preference"])

    if is_low_speed_uncongested(target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints):
        target_speed = max(target_speed, context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps)
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "low_speed_uncongested"})

    if is_all_lane_low_speed_occupancy(
        context.all_lanes_av_occupied,
        context.av_mean_speed_mps,
        context.free_flow_speed_mps,
        context.downstream_congested,
        constraints,
    ):
        lane_action = None
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "all_lane_low_speed_occupancy"})

    if is_passing_lane_slow_hold(context.in_passing_lane, target_speed, context.local_mean_speed_mps, constraints):
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "passing_lane_slow_hold"})

    if lane_action is not None:
        dwell = float("inf") if state.last_lane_change_time_s is None else context.time_s - state.last_lane_change_time_s
        if dwell < constraints.lane_change_dwell_s:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "lane_change_dwell"})
        elif _lane_change_count_exceeded(state, constraints):
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "lane_changes_per_km"})
        elif not context.target_lane_exists:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_missing"})
        elif context.target_lane_front_gap_m < constraints.min_front_gap_m:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_front_gap"})
        elif context.target_lane_rear_gap_m < constraints.min_rear_gap_m:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_rear_gap"})
        elif _ttc_from_relative_speed(context.target_lane_front_gap_m, -context.target_lane_front_relative_speed_mps) < constraints.min_lane_change_ttc_s:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_front_ttc"})
        elif _ttc_from_relative_speed(context.target_lane_rear_gap_m, context.target_lane_rear_relative_speed_mps) < constraints.min_lane_change_ttc_s:
            lane_action = None
            diagnostics["safety_masked_action"].append({"agent_id": agent_id, "reason": "target_lane_rear_ttc"})
        elif context.target_lane_rear_required_decel_mps2 > constraints.max_follower_braking_mps2:
            lane_action = None
            diagnostics["follower_disruption_blocked"].append({"agent_id": agent_id, "reason": "target_lane_rear_braking"})

    acceleration, emergency_override = physical_control_command(target_speed, target_headway, context, constraints)
    if emergency_override:
        diagnostics["external_safety_override"].append({"agent_id": agent_id, "reason": "forward_ttc"})
    penalty_terms = safety_penalty_terms(diagnostics, emergency_override=emergency_override)
    return SafetyDecision(
        target_speed_mps=target_speed,
        target_headway_s=target_headway,
        lane_action=lane_action,
        acceleration_mps2=acceleration,
        emergency_override=emergency_override,
        diagnostics=diagnostics,
        penalty_terms=penalty_terms,
    )


def physical_control_command(
    target_speed_mps: float,
    target_headway_s: float,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
) -> tuple[float, bool]:
    constraints = constraints or SafetyConstraints()
    forward_ttc = _forward_ttc(context)
    critical_gap = context.leader_gap_m < constraints.min_front_gap_m
    if forward_ttc < constraints.min_forward_ttc_s or critical_gap:
        return -constraints.emergency_decel_mps2, True

    desired_speed = min(target_speed_mps, context.free_flow_speed_mps)
    desired_gap = max(constraints.min_front_gap_m, target_headway_s * max(context.ego_speed_mps, 0.0))
    if context.leader_gap_m < desired_gap:
        leader_speed = max(0.0, context.ego_speed_mps + context.leader_relative_speed_mps)
        gap_ratio = max(0.0, context.leader_gap_m / max(desired_gap, 1e-6))
        desired_speed = min(desired_speed, leader_speed * gap_ratio)

    acceleration = constraints.speed_control_kp * (desired_speed - context.ego_speed_mps)
    return (
        max(-constraints.max_decel_mps2, min(constraints.max_accel_mps2, acceleration)),
        False,
    )


def safety_action_mask(
    state: SafetyState,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
) -> dict[str, dict[str, bool]]:
    constraints = constraints or SafetyConstraints()
    lateral_safe = _lane_preference_safe(state, context, constraints)
    slow_safe = not is_low_speed_uncongested(
        decode_speed_bin("slow", context.free_flow_speed_mps, context.min_contextual_speed_mps),
        context.free_flow_speed_mps,
        context.local_density_veh_per_km,
        constraints,
    )
    return {
        "desired_speed_bin": {"slow": slow_safe, "nominal": True, "fast": True},
        "desired_headway_bin": {"normal": True, "larger": True, "largest": True},
        "lane_preference": {
            "keep": True,
            "prefer_left_if_safe": lateral_safe,
            "prefer_right_if_safe": lateral_safe,
        },
        "merge_mode": {"normal": True, "create_gap": True, "hold_lane": True},
    }


def safety_penalty_terms(
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    emergency_override: bool = False,
) -> dict[str, float]:
    return {
        "unsafe_lane_preference": float(bool(diagnostics.get("safety_masked_action"))),
        "follower_disruption": float(bool(diagnostics.get("follower_disruption_blocked"))),
        "low_speed_uncongested": float(
            any(event.get("reason") == "low_speed_uncongested" for event in diagnostics.get("etiquette_blocked_action", []))
        ),
        "emergency_override": float(emergency_override),
        "excessive_lane_change": float(
            any(event.get("reason") == "lane_changes_per_km" for event in diagnostics.get("safety_masked_action", []))
        ),
    }


def _lane_preference_safe(
    state: SafetyState,
    context: SafetyContext,
    constraints: SafetyConstraints,
) -> bool:
    dwell = float("inf") if state.last_lane_change_time_s is None else context.time_s - state.last_lane_change_time_s
    return (
        dwell >= constraints.lane_change_dwell_s
        and not _lane_change_count_exceeded(state, constraints)
        and context.target_lane_exists
        and context.target_lane_front_gap_m >= constraints.min_front_gap_m
        and context.target_lane_rear_gap_m >= constraints.min_rear_gap_m
        and _ttc_from_relative_speed(context.target_lane_front_gap_m, -context.target_lane_front_relative_speed_mps)
        >= constraints.min_lane_change_ttc_s
        and _ttc_from_relative_speed(context.target_lane_rear_gap_m, context.target_lane_rear_relative_speed_mps)
        >= constraints.min_lane_change_ttc_s
        and context.target_lane_rear_required_decel_mps2 <= constraints.max_follower_braking_mps2
    )


def _lane_change_count_exceeded(state: SafetyState, constraints: SafetyConstraints) -> bool:
    return state.lane_changes_last_km >= constraints.max_lane_changes_per_km


def _forward_ttc(context: SafetyContext) -> float:
    return _ttc_from_relative_speed(context.leader_gap_m, -context.leader_relative_speed_mps)


def _ttc_from_relative_speed(gap_m: float, closing_speed_mps: float) -> float:
    if closing_speed_mps <= 0:
        return float("inf")
    return max(0.0, gap_m / max(closing_speed_mps, 1e-6))
