from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.safety.constraints import SafetyConstraints
from src.safety.etiquette import is_low_speed_uncongested


@dataclass(frozen=True)
class SafetyContext:
    """Everything the bound on a recommended speed reads.

    Every field here is read by `apply_safety_layer` or by
    `physical_control_command` below. The lane, rear-gap, cooperation and
    passing-lane fields were removed with the rules that read them; a field
    carried into a decision that never looks at it is indistinguishable, in
    the record, from one that was measured and did not matter.
    """

    ego_speed_mps: float = 0.0
    free_flow_speed_mps: float = 30.0
    local_density_veh_per_km: float = 0.0
    leader_gap_m: float = float("inf")
    leader_relative_speed_mps: float = 0.0
    #: Distance to the joining node of the nearest vehicle converging on it from a
    #: different arc, projected onto that node so it reads as a following
    #: distance. Infinite when nobody is converging.
    #:
    #: There is deliberately no `distance_to_next_merge_m` here. It was added,
    #: plumbed through two dataclasses and read by nothing, while looking
    #: load-bearing because five tests passed it as an input. The live system has
    #: no map matching and so cannot supply it at all.
    merge_conflict_gap_m: float = float("inf")
    merge_conflict_relative_speed_mps: float = 0.0


@dataclass(frozen=True)
class SafetyDecision:
    target_speed_mps: float
    target_headway_s: float
    acceleration_mps2: float = 0.0
    emergency_override: bool = False
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    penalty_terms: dict[str, float] = field(default_factory=dict)


def empty_diagnostics() -> dict[str, list[dict[str, Any]]]:
    """The two buckets the surviving rules write to.

    `safety_masked_action`, `follower_disruption_blocked` and
    `simulator_blocked_action` were removed with the lane and follower rules.
    An always-empty bucket reads as "this never happened" where it means "this
    can no longer happen", which are different statements about a drive.
    """
    return {
        "etiquette_blocked_action": [],
        "external_safety_override": [],
    }


def apply_safety_layer(
    target_speed_mps: float,
    target_headway_s: float,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
    agent_id: str | None = None,
) -> SafetyDecision:
    """Bound a proposed speed and following distance.

    Takes the speed directly rather than an action to decode. The controller
    this bounds emits one speed per super-segment, so there is no bin to decode
    and no lane or merge head to read.
    """
    constraints = constraints or SafetyConstraints()
    diagnostics = empty_diagnostics()
    target_speed = target_speed_mps
    target_headway = target_headway_s

    if is_low_speed_uncongested(target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints):
        target_speed = max(target_speed, context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps)
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "low_speed_uncongested"})

    acceleration, emergency_override = physical_control_command(target_speed, target_headway, context, constraints)
    if emergency_override:
        diagnostics["external_safety_override"].append({"agent_id": agent_id, "reason": "forward_ttc"})
    penalty_terms = safety_penalty_terms(diagnostics, emergency_override=emergency_override)
    return SafetyDecision(
        target_speed_mps=target_speed,
        target_headway_s=target_headway,
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
    hazard_gap, hazard_relative_speed = _forward_hazard(context)
    critical_gap = hazard_gap < constraints.min_front_gap_m
    if forward_ttc < constraints.min_forward_ttc_s or critical_gap:
        return -constraints.emergency_decel_mps2, True

    desired_speed = min(target_speed_mps, context.free_flow_speed_mps)
    desired_gap = max(constraints.min_front_gap_m, target_headway_s * max(context.ego_speed_mps, 0.0))
    if hazard_gap < desired_gap:
        leader_speed = max(0.0, context.ego_speed_mps + hazard_relative_speed)
        gap_ratio = max(0.0, hazard_gap / max(desired_gap, 1e-6))
        desired_speed = min(desired_speed, leader_speed * gap_ratio)

    acceleration = constraints.speed_control_kp * (desired_speed - context.ego_speed_mps)
    return (
        max(-constraints.max_decel_mps2, min(constraints.max_accel_mps2, acceleration)),
        False,
    )


def safety_penalty_terms(
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    emergency_override: bool = False,
) -> dict[str, float]:
    """One term per rule that can fire. `unsafe_lane_preference`,
    `follower_disruption` and `excessive_lane_change` were removed with their
    rules: each was computed from a diagnostics bucket nothing appends to any
    more, so each was a constant 0.0 reported as a measurement.
    """
    return {
        "low_speed_uncongested": float(
            any(event.get("reason") == "low_speed_uncongested" for event in diagnostics.get("etiquette_blocked_action", []))
        ),
        "emergency_override": float(emergency_override),
    }


def _forward_hazard(context: SafetyContext) -> tuple[float, float]:
    """The nearest thing ahead the ego must not hit, and its closing speed.

    Usually the leader. At a joining node it can instead be the node itself: a
    vehicle converging from a sibling arc is on nobody's route, so no headway rule
    sees it, and on `inverted_tree` 27 of 51 terminating collisions were exactly
    that pair.

    `merge_conflict_gap_m` is already a following distance: the sensing model
    projects both vehicles onto the shared node and reports the gap to the nearest
    one that will arrive first, or infinity when the ego arrives first. So this is
    a plain `min` against the leader, and the converging vehicle carries its own
    relative speed rather than being treated as a stationary point.
    """
    gap = context.leader_gap_m
    relative_speed = context.leader_relative_speed_mps
    if context.merge_conflict_gap_m < gap:
        gap = context.merge_conflict_gap_m
        relative_speed = context.merge_conflict_relative_speed_mps
    return gap, relative_speed


def _forward_ttc(context: SafetyContext) -> float:
    gap, relative_speed = _forward_hazard(context)
    return _ttc_from_relative_speed(gap, -relative_speed)


def _ttc_from_relative_speed(gap_m: float, closing_speed_mps: float) -> float:
    if closing_speed_mps <= 0:
        return float("inf")
    return max(0.0, gap_m / max(closing_speed_mps, 1e-6))
