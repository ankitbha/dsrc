"""Vendored safety and etiquette layer, run on the device advisory path.

`SafetyConstraints`, `SafetyContext`, `SafetyState`, `SafetyDecision`,
`apply_safety_layer`, `physical_control_command` and `safety_penalty_terms`
mirror `src/safety/{constraints,etiquette,safety_layer}.py` field for field
and line for line. Guarded against drift by `specs/safety_contract_golden.json`
rather than by importing the original: `deployment/jetson/tests/
test_safety_contract.py` checks this module against that file unconditionally,
`tests/test_safety_contract_matches_golden.py` checks `src/safety/` against
the same file, and neither imports the other side. Task 143's finding is why:
a check written as "compare the vendored copy against the original" goes
vacuous the day the original is deleted; a golden file cannot.

Two things do NOT come from `src/safety/`, by design (plan_task144 decision 1):

- This module calls `sim_contract.decode_speed_bin` / `decode_headway_bin`
  rather than carrying its own copies, so the device has exactly one decoder
  for each -- the one `AdvisoryDecoder` also uses. Two decoders on one
  device, decided independently, is the failure mode `src/envs/wrappers.py`
  and `sim_contract.py` already are two copies of.
- `lane_preference_to_action` (`src/envs/wrappers.py`) is inlined as
  `_lane_preference_to_action` below: a three-entry literal mapping with no
  state and no drift risk, used nowhere else on the device, so a second
  vendored copy of it does not create the two-decoder problem the point
  above exists to avoid.

`SafetyInputs` and the rule-evaluability machinery below (`RULE_NAMES`,
`evaluate_rules`, `run_safety_gate`) have no equivalent in `src/safety/` at
all -- decision 3's per-rule evaluability census, keyed on
`perception.provenance`. A filter fed `fallback_neutral` inputs either never
fires, which is a fact about the fallback and not about traffic, or fires on
a constant, which is worse; printing a firing rate without saying which one
happened is the defect this task exists to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from perception import provenance
from perception.observation_builder import ObservationResult
from policy import sim_contract
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET

# --- Vendored from src/safety/constraints.py ---------------------------------


@dataclass(frozen=True)
class SafetyConstraints:
    """Conservative defaults for hard safety and etiquette checks."""

    lane_change_dwell_s: float = 15.0
    max_lane_changes_per_km: float = 2.0
    max_follower_braking_mps2: float = 2.5
    comfortable_decel_mps2: float = 3.0
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    emergency_decel_mps2: float = 6.0
    min_front_gap_m: float = 5.0
    min_rear_gap_m: float = 5.0
    min_forward_ttc_s: float = 2.0
    min_lane_change_ttc_s: float = 2.5
    speed_control_kp: float = 0.6
    min_uncongested_speed_mps: float = 12.0
    uncongested_density_threshold_veh_per_km: float = 12.0
    low_speed_free_flow_delta_mps: float = 8.0
    merge_gap_headway_bonus_s: float = 0.8


# --- Vendored from src/safety/safety_layer.py --------------------------------


@dataclass
class SafetyState:
    last_lane_change_time_s: float | None = None
    lane_changes_last_km: int = 0
    distance_since_window_start_m: float = 0.0
    absolute_distance_m: float = 0.0
    lane_change_distances_m: list[float] = field(default_factory=list)
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
    merge_conflict_gap_m: float = float("inf")
    merge_conflict_relative_speed_mps: float = 0.0


@dataclass(frozen=True)
class SafetyDecision:
    target_speed_mps: float
    target_headway_s: float
    lane_action: str | None
    acceleration_mps2: float = 0.0
    emergency_override: bool = False
    diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    penalty_terms: dict[str, float] = field(default_factory=dict)


def _lane_preference_to_action(lane_preference: str) -> str | None:
    """Vendored from src/envs/wrappers.py:lane_preference_to_action -- a
    three-entry literal with no decoder-drift risk (see module docstring)."""
    return {
        "keep": None,
        "prefer_left_if_safe": "LANE_LEFT",
        "prefer_right_if_safe": "LANE_RIGHT",
    }[lane_preference]


def _is_low_speed_uncongested(
    target_speed_mps: float,
    free_flow_speed_mps: float,
    density_veh_per_km: float,
    constraints: SafetyConstraints,
) -> bool:
    """Vendored from src/safety/etiquette.py:is_low_speed_uncongested."""
    return (
        density_veh_per_km < constraints.uncongested_density_threshold_veh_per_km
        and target_speed_mps < free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps
    )


def _is_all_lane_low_speed_occupancy(
    all_lanes_av_occupied: bool,
    av_mean_speed_mps: float,
    free_flow_speed_mps: float,
    downstream_congested: bool,
    constraints: SafetyConstraints,
) -> bool:
    """Vendored from src/safety/etiquette.py:is_all_lane_low_speed_occupancy."""
    return (
        all_lanes_av_occupied
        and not downstream_congested
        and av_mean_speed_mps < free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps
    )


def _is_passing_lane_slow_hold(
    in_passing_lane: bool,
    ego_speed_mps: float,
    local_mean_speed_mps: float,
    constraints: SafetyConstraints,
) -> bool:
    """Vendored from src/safety/etiquette.py:is_passing_lane_slow_hold.

    The second argument is named `ego_speed_mps` in the original, but every
    caller (there and here) passes the DECODED TARGET speed, not
    `context.ego_speed_mps` -- kept exactly as the original calls it.
    """
    return in_passing_lane and ego_speed_mps < local_mean_speed_mps - constraints.low_speed_free_flow_delta_mps


def empty_diagnostics() -> dict[str, list[dict[str, Any]]]:
    return {
        "safety_masked_action": [],
        "etiquette_blocked_action": [],
        "follower_disruption_blocked": [],
        "external_safety_override": [],
        "simulator_blocked_action": [],
    }


def apply_safety_layer(
    action: Mapping[str, str],
    state: SafetyState,
    context: SafetyContext,
    constraints: SafetyConstraints | None = None,
    agent_id: str | None = None,
) -> SafetyDecision:
    """Verbatim port of src/safety/safety_layer.py:apply_safety_layer, with
    `decode_speed_bin`/`decode_headway_bin` and `lane_preference_to_action`
    coming from this module's own imports/helpers rather than
    src.envs.wrappers (see module docstring)."""
    constraints = constraints or SafetyConstraints()
    diagnostics = empty_diagnostics()
    target_speed = sim_contract.decode_speed_bin(
        action["desired_speed_bin"],
        free_flow_speed_mps=context.free_flow_speed_mps,
        min_contextual_speed_mps=context.min_contextual_speed_mps,
    )
    target_headway = sim_contract.decode_headway_bin(action["desired_headway_bin"])
    if action["merge_mode"] == "create_gap":
        target_headway += constraints.merge_gap_headway_bonus_s
    lane_action = None if action["merge_mode"] == "hold_lane" else _lane_preference_to_action(action["lane_preference"])

    if _is_low_speed_uncongested(target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints):
        target_speed = max(target_speed, context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps)
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "low_speed_uncongested"})

    if _is_all_lane_low_speed_occupancy(
        context.all_lanes_av_occupied,
        context.av_mean_speed_mps,
        context.free_flow_speed_mps,
        context.downstream_congested,
        constraints,
    ):
        lane_action = None
        diagnostics["etiquette_blocked_action"].append({"agent_id": agent_id, "reason": "all_lane_low_speed_occupancy"})

    if _is_passing_lane_slow_hold(context.in_passing_lane, target_speed, context.local_mean_speed_mps, constraints):
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
    """Verbatim port of src/safety/safety_layer.py:physical_control_command."""
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
    """Verbatim port of src/safety/safety_layer.py:safety_penalty_terms."""
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


def _lane_change_count_exceeded(state: SafetyState, constraints: SafetyConstraints) -> bool:
    if state.lane_change_distances_m:
        window_start = max(0.0, state.absolute_distance_m - 1000.0)
        recent_changes = sum(1 for distance in state.lane_change_distances_m if distance >= window_start)
    else:
        recent_changes = state.lane_changes_last_km
    return recent_changes >= constraints.max_lane_changes_per_km


def _forward_hazard(context: SafetyContext) -> tuple[float, float]:
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


def _dwell_s(state: SafetyState, context: SafetyContext) -> float:
    return float("inf") if state.last_lane_change_time_s is None else context.time_s - state.last_lane_change_time_s


# --- decision 3: the per-field input-class partition -------------------------

#: (A) Configured road property. Accepted as given; never makes a rule
#: not_evaluable -- these are operator statements about the road, not
#: measurements this tick owes.
INPUT_CLASS_CONFIGURED = "configured"
#: (B) Evidence-required. A rule reading one of these is RULE_NOT_EVALUABLE
#: when the field's provenance is in perception.provenance.SUBSTITUTED (with
#: one further carve-out for local_density_veh_per_km -- see
#: `safety_inputs_from_observation`).
INPUT_CLASS_EVIDENCE_REQUIRED = "evidence_required"
#: (C) Structurally absent. No sensor exists on this rig; not_evaluable
#: unconditionally, regardless of what a tag would otherwise say.
INPUT_CLASS_STRUCTURAL = "structural"

#: decision 3's 3 + 6 + 14 = 23 SafetyContext fields, plus the 3 SafetyState
#: names the rules that read lane-change history need (last_lane_change_time_s,
#: lane_changes_last_km, lane_change_distances_m -- the device cannot advance
#: any of the three, so they are class (C) unconditionally).
CONFIGURED_FIELDS: tuple[str, ...] = ("time_s", "free_flow_speed_mps", "min_contextual_speed_mps")
EVIDENCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "ego_speed_mps",
    "leader_gap_m",
    "leader_relative_speed_mps",
    "local_density_veh_per_km",
    "local_mean_speed_mps",
    "target_lane_front_gap_m",
)
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "follower_gap_m",
    "follower_relative_speed_mps",
    "target_lane_rear_gap_m",
    "target_lane_rear_relative_speed_mps",
    "target_lane_rear_required_decel_mps2",
    "target_lane_exists",
    "in_passing_lane",
    "target_lane_front_relative_speed_mps",
    "all_lanes_av_occupied",
    "av_mean_speed_mps",
    "downstream_congested",
    "near_merge",
    "merge_conflict_gap_m",
    "merge_conflict_relative_speed_mps",
    "last_lane_change_time_s",
    "lane_changes_last_km",
    "lane_change_distances_m",
)


@dataclass(frozen=True)
class SafetyInputField:
    """One SafetyContext (or SafetyState history) field's value, its
    decision-3 input class, and whether it counts as evidence this tick."""

    value: Any
    input_class: str
    #: The `perception.provenance` class behind the value, or None for a
    #: field with no observation-builder counterpart at all (most of class C).
    source: str | None
    evidence: bool


@dataclass(frozen=True)
class SafetyInputs:
    """Every SafetyContext field (plus the three lane-history counters),
    each carrying decision 3's per-field verdict. Built once per tick by
    `safety_inputs_from_observation`, from `ObservationResult` plus the
    running `SafetyState`.
    """

    fields: dict[str, SafetyInputField]

    def context(self) -> SafetyContext:
        values = {name: f.value for name, f in self.fields.items() if name not in (
            "last_lane_change_time_s", "lane_changes_last_km", "lane_change_distances_m",
        )}
        return SafetyContext(**values)

    def state(self) -> SafetyState:
        return SafetyState(
            last_lane_change_time_s=self.fields["last_lane_change_time_s"].value,
            lane_changes_last_km=self.fields["lane_changes_last_km"].value,
            lane_change_distances_m=list(self.fields["lane_change_distances_m"].value),
        )

    def is_evidence(self, name: str) -> bool:
        return self.fields[name].evidence


def _density_is_evidence(source: str | None, last_detection_age_s: float | None, max_age_s: float) -> bool:
    """decision 3's carve-out: SOURCE_DERIVED_EMPTY (a camera that has
    produced zero in-range tracks) counts as evidence for the density field
    only when `last_detection_age_s` bounds how long ago that camera last saw
    anything at all -- otherwise a blind camera and an empty road are the
    same reading, and this task's job is telling them apart, not merging
    them back together.
    """
    if source is None or provenance.is_substituted(source):
        return False
    if source != provenance.SOURCE_DERIVED_EMPTY:
        return True
    return last_detection_age_s is not None and math.isfinite(last_detection_age_s) and last_detection_age_s <= max_age_s


def safety_inputs_from_observation(
    obs_result: ObservationResult,
    state: SafetyState,
    *,
    time_s: float,
    min_contextual_speed_mps: float,
    density_max_age_s: float,
) -> SafetyInputs:
    """Build `SafetyInputs` from one tick's `ObservationResult` plus the
    running lane-change `SafetyState`, per decision 3's three-class
    partition. `density_max_age_s` is the bound `local_density_veh_per_km`'s
    `derived_empty` carve-out checks `obs_diagnostics.last_detection_age_s`
    against -- twice `BuilderConfig.gps_stale_after_s`, per decision 3.
    """
    obs = obs_result.obs
    src = obs_result.field_sources
    diag = obs_result.diagnostics
    cooperation = obs.get("cooperation", {}) if isinstance(obs.get("cooperation"), Mapping) else {}

    def configured(name: str, value: Any) -> SafetyInputField:
        return SafetyInputField(value=value, input_class=INPUT_CLASS_CONFIGURED, source=None, evidence=True)

    def evidence_required(name: str, value: Any, source_key: str) -> SafetyInputField:
        source = src.get(source_key)
        return SafetyInputField(
            value=value, input_class=INPUT_CLASS_EVIDENCE_REQUIRED, source=source,
            evidence=source is not None and not provenance.is_substituted(source),
        )

    def structural(value: Any, source: str | None = None) -> SafetyInputField:
        return SafetyInputField(value=value, input_class=INPUT_CLASS_STRUCTURAL, source=source, evidence=False)

    density_source = src.get("local_density_bin")
    density_evidence = _density_is_evidence(density_source, diag.get("last_detection_age_s"), density_max_age_s)

    fields: dict[str, SafetyInputField] = {
        "time_s": configured("time_s", time_s),
        "free_flow_speed_mps": configured(
            "free_flow_speed_mps", float(cooperation.get("segment_target_speed", 30.0))
        ),
        "min_contextual_speed_mps": configured("min_contextual_speed_mps", min_contextual_speed_mps),
        "ego_speed_mps": evidence_required("ego_speed_mps", float(obs.get("ego_speed", 0.0)), "ego_speed"),
        "leader_gap_m": evidence_required("leader_gap_m", float(obs.get("leader_gap", float("inf"))), "leader_gap"),
        "leader_relative_speed_mps": evidence_required(
            "leader_relative_speed_mps", float(obs.get("leader_relative_speed", 0.0)), "leader_relative_speed"
        ),
        "local_density_veh_per_km": SafetyInputField(
            value=float(diag.get("density_veh_per_km", 0.0)),
            input_class=INPUT_CLASS_EVIDENCE_REQUIRED,
            source=density_source,
            evidence=density_evidence,
        ),
        "local_mean_speed_mps": evidence_required(
            "local_mean_speed_mps", float(diag.get("mean_speed_mps", 0.0)), "local_mean_speed_bin"
        ),
        "target_lane_front_gap_m": evidence_required(
            "target_lane_front_gap_m", float(obs.get("target_lane_front_gap", float("inf"))), "target_lane_front_gap"
        ),
        "downstream_congested": structural(
            bool(obs.get("downstream_congestion_estimate", 0.0) > 0.0), src.get("downstream_congestion_estimate")
        ),
        "follower_gap_m": structural(float(obs.get("follower_gap", float("inf"))), src.get("follower_gap")),
        "follower_relative_speed_mps": structural(
            float(obs.get("follower_relative_speed", 0.0)), src.get("follower_relative_speed")
        ),
        "target_lane_exists": structural(True),
        "target_lane_front_relative_speed_mps": structural(0.0),
        "target_lane_rear_gap_m": structural(float(obs.get("target_lane_rear_gap", float("inf"))), src.get("target_lane_rear_gap")),
        "target_lane_rear_relative_speed_mps": structural(0.0),
        "target_lane_rear_required_decel_mps2": structural(
            float(obs.get("target_lane_rear_required_decel", 0.0)), src.get("target_lane_rear_required_decel")
        ),
        "all_lanes_av_occupied": structural(False),
        "av_mean_speed_mps": structural(
            float(obs.get("nearby_av_mean_speed", 30.0)), src.get("nearby_av_mean_speed")
        ),
        "in_passing_lane": structural(False),
        "near_merge": structural(False),
        "merge_conflict_gap_m": structural(float("inf")),
        "merge_conflict_relative_speed_mps": structural(0.0),
        "last_lane_change_time_s": structural(state.last_lane_change_time_s),
        "lane_changes_last_km": structural(state.lane_changes_last_km),
        "lane_change_distances_m": structural(list(state.lane_change_distances_m)),
    }
    return SafetyInputs(fields=fields)


# --- decision 3 / step 9: the twelve rules as total independent predicates ---

#: The order `apply_safety_layer`'s elif chain checks the eight lane/merge
#: guards in -- used both to build `LANE_GUARD_RULES` below and by
#: tests/test_safety_gate_pipeline.py's chain-order invariant.
_LANE_GUARD_CHAIN_ORDER: tuple[str, ...] = (
    "lane_change_dwell",
    "lane_changes_per_km",
    "target_lane_missing",
    "target_lane_front_gap",
    "target_lane_rear_gap",
    "target_lane_front_ttc",
    "target_lane_rear_ttc",
    "target_lane_rear_braking",
)

#: All twelve rules, in the order E4 and the plan's steps name them.
RULE_NAMES: tuple[str, ...] = (
    "low_speed_uncongested",
    "all_lane_low_speed_occupancy",
    "passing_lane_slow_hold",
    *_LANE_GUARD_CHAIN_ORDER,
    "forward_ttc",
)

#: The lane/merge-action guards: decision 3 says the lane action is withheld
#: when any of these is not_evaluable. Identical set to the chain order above,
#: named separately because "which rules guard the lane action" and "what
#: order does the chain check them in" are two different facts a reader
#: might want independently.
LANE_GUARD_RULES: tuple[str, ...] = _LANE_GUARD_CHAIN_ORDER

#: Which SafetyInputs field names each rule's not_evaluable verdict depends
#: on. A rule is not_evaluable when ANY named field is not evidence.
RULE_READS: dict[str, tuple[str, ...]] = {
    "low_speed_uncongested": ("local_density_veh_per_km",),
    "all_lane_low_speed_occupancy": ("all_lanes_av_occupied", "av_mean_speed_mps", "downstream_congested"),
    "passing_lane_slow_hold": ("in_passing_lane", "local_mean_speed_mps"),
    "lane_change_dwell": ("last_lane_change_time_s",),
    "lane_changes_per_km": ("lane_changes_last_km", "lane_change_distances_m"),
    "target_lane_missing": ("target_lane_exists",),
    "target_lane_front_gap": ("target_lane_front_gap_m",),
    "target_lane_rear_gap": ("target_lane_rear_gap_m",),
    "target_lane_front_ttc": ("target_lane_front_gap_m", "target_lane_front_relative_speed_mps"),
    "target_lane_rear_ttc": ("target_lane_rear_gap_m", "target_lane_rear_relative_speed_mps"),
    "target_lane_rear_braking": ("target_lane_rear_required_decel_mps2",),
    "forward_ttc": (
        "leader_gap_m", "leader_relative_speed_mps",
        "merge_conflict_gap_m", "merge_conflict_relative_speed_mps",
    ),
}


@dataclass(frozen=True)
class RuleRecord:
    """One rule's status this tick, evaluated as a total predicate rather
    than through the elif chain's short-circuit -- mirrors
    `policy.sensing_controller.RuleCheck` field for field, so a reader who
    already knows that vocabulary reads this one for free.
    """

    status: str
    missing: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"status": self.status}
        if self.missing:
            record["missing"] = list(self.missing)
        for key, value in self.evidence.items():
            record[key] = round(value, 4) if isinstance(value, float) else value
        return record


def _target_speed(action: Mapping[str, str], context: SafetyContext) -> float:
    return sim_contract.decode_speed_bin(
        action["desired_speed_bin"],
        free_flow_speed_mps=context.free_flow_speed_mps,
        min_contextual_speed_mps=context.min_contextual_speed_mps,
    )


def _evaluate_one_rule(
    name: str,
    *,
    action: Mapping[str, str],
    inputs: SafetyInputs,
    context: SafetyContext,
    state: SafetyState,
    constraints: SafetyConstraints,
) -> RuleRecord:
    missing = tuple(f for f in RULE_READS[name] if not inputs.is_evidence(f))
    if missing:
        evidence = {f"{f}_source": inputs.fields[f].source for f in missing}
        return RuleRecord(status=RULE_NOT_EVALUABLE, missing=missing, evidence=evidence)

    target_speed = _target_speed(action, context)
    if name == "low_speed_uncongested":
        fired = _is_low_speed_uncongested(
            target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints,
        )
        evidence = {
            "target_speed_mps": target_speed,
            "free_flow_speed_mps": context.free_flow_speed_mps,
            "threshold_mps": context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps,
            "local_density_veh_per_km": context.local_density_veh_per_km,
            "density_threshold_veh_per_km": constraints.uncongested_density_threshold_veh_per_km,
        }
    elif name == "all_lane_low_speed_occupancy":
        fired = _is_all_lane_low_speed_occupancy(
            context.all_lanes_av_occupied, context.av_mean_speed_mps, context.free_flow_speed_mps,
            context.downstream_congested, constraints,
        )
        evidence = {
            "all_lanes_av_occupied": context.all_lanes_av_occupied,
            "av_mean_speed_mps": context.av_mean_speed_mps,
            "threshold_mps": context.free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps,
            "downstream_congested": context.downstream_congested,
        }
    elif name == "passing_lane_slow_hold":
        fired = _is_passing_lane_slow_hold(context.in_passing_lane, target_speed, context.local_mean_speed_mps, constraints)
        evidence = {
            "in_passing_lane": context.in_passing_lane,
            "target_speed_mps": target_speed,
            "threshold_mps": context.local_mean_speed_mps - constraints.low_speed_free_flow_delta_mps,
        }
    elif name == "lane_change_dwell":
        dwell = _dwell_s(state, context)
        fired = dwell < constraints.lane_change_dwell_s
        evidence = {"dwell_s": dwell, "threshold_s": constraints.lane_change_dwell_s}
    elif name == "lane_changes_per_km":
        fired = _lane_change_count_exceeded(state, constraints)
        evidence = {
            "lane_changes_last_km": state.lane_changes_last_km,
            "threshold": constraints.max_lane_changes_per_km,
        }
    elif name == "target_lane_missing":
        fired = not context.target_lane_exists
        evidence = {"target_lane_exists": context.target_lane_exists}
    elif name == "target_lane_front_gap":
        fired = context.target_lane_front_gap_m < constraints.min_front_gap_m
        evidence = {"gap_m": context.target_lane_front_gap_m, "threshold_m": constraints.min_front_gap_m}
    elif name == "target_lane_rear_gap":
        fired = context.target_lane_rear_gap_m < constraints.min_rear_gap_m
        evidence = {"gap_m": context.target_lane_rear_gap_m, "threshold_m": constraints.min_rear_gap_m}
    elif name == "target_lane_front_ttc":
        ttc = _ttc_from_relative_speed(context.target_lane_front_gap_m, -context.target_lane_front_relative_speed_mps)
        fired = ttc < constraints.min_lane_change_ttc_s
        evidence = {"ttc_s": ttc, "threshold_s": constraints.min_lane_change_ttc_s}
    elif name == "target_lane_rear_ttc":
        ttc = _ttc_from_relative_speed(context.target_lane_rear_gap_m, context.target_lane_rear_relative_speed_mps)
        fired = ttc < constraints.min_lane_change_ttc_s
        evidence = {"ttc_s": ttc, "threshold_s": constraints.min_lane_change_ttc_s}
    elif name == "target_lane_rear_braking":
        fired = context.target_lane_rear_required_decel_mps2 > constraints.max_follower_braking_mps2
        evidence = {
            "required_decel_mps2": context.target_lane_rear_required_decel_mps2,
            "threshold_mps2": constraints.max_follower_braking_mps2,
        }
    elif name == "forward_ttc":
        gap, relative_speed = _forward_hazard(context)
        ttc = _ttc_from_relative_speed(gap, -relative_speed)
        critical_gap = gap < constraints.min_front_gap_m
        fired = ttc < constraints.min_forward_ttc_s or critical_gap
        evidence = {"gap_m": gap, "ttc_s": ttc, "threshold_s": constraints.min_forward_ttc_s, "critical_gap": critical_gap}
    else:  # pragma: no cover -- RULE_NAMES is closed and this branches on all of it
        raise ValueError(f"unknown rule '{name}'")

    return RuleRecord(status=RULE_FIRED if fired else RULE_QUIET, evidence=evidence)


def evaluate_rules(
    action: Mapping[str, str],
    inputs: SafetyInputs,
    state: SafetyState,
    constraints: SafetyConstraints,
) -> dict[str, RuleRecord]:
    """All twelve rules, each a total predicate over its own evaluability --
    never short-circuited by chain position, unlike `apply_safety_layer`'s
    lane-guard elif chain (decision 3's "never confuse a short-circuited
    guard with an unevaluable one")."""
    context = inputs.context()
    return {
        name: _evaluate_one_rule(name, action=action, inputs=inputs, context=context, state=state, constraints=constraints)
        for name in RULE_NAMES
    }


@dataclass(frozen=True)
class SafetyGateResult:
    """One tick's gate outcome: the raw (proposed) decision, the bounded one
    actually shown to the driver, and the per-rule evaluability census.

    `raw_diagnostics` is `apply_safety_layer`'s own (unchanged) elif-chain
    diagnostics dict -- kept on the result for a tool that wants to attribute
    an event the way the original decision does (plan_task144 E5's own
    methodology: "running apply_safety_layer... [with] events" is read off
    exactly this), but deliberately left out of `to_record()`. The persisted
    per-tick record uses `rules` (decision 3's independent, evaluability-aware
    census) as its one account of what each rule did; carrying both would
    let a reader play them against each other instead of trusting the one
    decision 3 says is the record.
    """

    proposed_speed_mps: float
    proposed_headway_s: float
    proposed_lane_action: str | None
    bounded_speed_mps: float
    bounded_headway_s: float
    bounded_lane_action: str | None
    delta_speed_mps: float
    emergency_override: bool
    lane_withheld: str | None
    evaluable_count: int
    not_evaluable_count: int
    rules: dict[str, RuleRecord]
    raw_diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "proposed": {
                "speed_mps": round(self.proposed_speed_mps, 2),
                "headway_s": round(self.proposed_headway_s, 2),
                "lane_action": self.proposed_lane_action,
            },
            "bounded": {
                "speed_mps": round(self.bounded_speed_mps, 2),
                "headway_s": round(self.bounded_headway_s, 2),
                "lane_action": self.bounded_lane_action,
            },
            "delta_speed_mps": round(self.delta_speed_mps, 2),
            "emergency_override": self.emergency_override,
            "lane_withheld": self.lane_withheld,
            "evaluable": self.evaluable_count,
            "not_evaluable": self.not_evaluable_count,
            "rules": {name: rule.to_record() for name, rule in self.rules.items()},
        }


def run_safety_gate(
    action: Mapping[str, str],
    inputs: SafetyInputs,
    state: SafetyState,
    constraints: SafetyConstraints | None = None,
    *,
    withhold_lane_when_not_evaluable: bool = True,
) -> SafetyGateResult:
    """Run the vendored `apply_safety_layer` (the bound the driver is shown)
    and the independent per-rule census (decision 3) from the same inputs,
    then apply decision 3's own addition: withhold the lane action outright
    when any of its guards could not be evaluated (open item 1), rather than
    letting an unevaluated guard's "safe" default speak for it.
    """
    constraints = constraints or SafetyConstraints()
    context = inputs.context()
    decision = apply_safety_layer(action, state, context, constraints)
    rules = evaluate_rules(action, inputs, state, constraints)

    lane_guard_not_evaluable = any(rules[name].status == RULE_NOT_EVALUABLE for name in LANE_GUARD_RULES)
    bounded_lane_action = decision.lane_action
    lane_withheld: str | None = None
    if withhold_lane_when_not_evaluable and lane_guard_not_evaluable and bounded_lane_action is not None:
        bounded_lane_action = None
        lane_withheld = RULE_NOT_EVALUABLE

    proposed_speed = _target_speed(action, context)
    evaluable = sum(1 for r in rules.values() if r.status != RULE_NOT_EVALUABLE)
    return SafetyGateResult(
        proposed_speed_mps=proposed_speed,
        proposed_headway_s=sim_contract.decode_headway_bin(action["desired_headway_bin"]),
        proposed_lane_action=_lane_preference_to_action(action["lane_preference"]) if action["merge_mode"] != "hold_lane" else None,
        bounded_speed_mps=decision.target_speed_mps,
        bounded_headway_s=decision.target_headway_s,
        bounded_lane_action=bounded_lane_action,
        delta_speed_mps=decision.target_speed_mps - proposed_speed,
        emergency_override=decision.emergency_override,
        lane_withheld=lane_withheld,
        evaluable_count=evaluable,
        not_evaluable_count=len(rules) - evaluable,
        rules=rules,
        raw_diagnostics=decision.diagnostics,
    )
