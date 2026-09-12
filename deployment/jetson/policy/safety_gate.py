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
from dataclasses import dataclass, field, replace as dataclasses_replace
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

#: decision 3's 3 + 6 + 14 = 23 SafetyContext fields, plus the 4 SafetyState
#: names the rules that read lane-change history need (last_lane_change_time_s,
#: lane_changes_last_km, lane_change_distances_m, and -- added validator round
#: 1, F1 follow-up -- absolute_distance_m, read by lane_changes_per_km's own
#: window branch. The device cannot advance any of the four, so they are
#: class (C) unconditionally.
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
    "absolute_distance_m",
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
    #: The `density_max_age_s` bound this tick's `local_density_veh_per_km`
    #: evidence verdict was computed against (see `_density_is_evidence`).
    #: Carried here, not just consumed and discarded, so `run_safety_gate`
    #: can put it in the persisted record (validator round 1, F4) --
    #: `safety_inputs_from_observation` always passes it explicitly (no
    #: default there); the default here is for direct construction in tests
    #: that do not exercise the density carve-out or the recorded config.
    density_max_age_s: float = 4.0

    def context(self) -> SafetyContext:
        values = {name: f.value for name, f in self.fields.items() if name not in (
            "last_lane_change_time_s", "lane_changes_last_km", "lane_change_distances_m", "absolute_distance_m",
        )}
        return SafetyContext(**values)

    def inert_context(self) -> SafetyContext:
        """The context `apply_safety_layer` reads (validator round 1, F1):
        identical to `context()` except that every field with
        `is_evidence(name)` False this tick is replaced by
        `INERT_CONTEXT_VALUES[name]` rather than the observation's own
        substituted value. `evaluate_rules`'s census keeps reading the
        UNMODIFIED `context()` -- this method exists only for the copy that
        reaches the driver-facing decision, so a not_evaluable rule is inert
        in the number the driver sees, not only in the record (decision 3:
        "nothing. It is recorded and the advisory is unchanged").

        A field absent from `INERT_CONTEXT_VALUES` (`ego_speed_mps`,
        `follower_gap_m`, `follower_relative_speed_mps`, `near_merge`) gates
        none of the twelve rules and is passed through unchanged regardless
        of evidence -- there is no comparison to make inert.
        """
        values = {name: f.value for name, f in self.fields.items() if name not in (
            "last_lane_change_time_s", "lane_changes_last_km", "lane_change_distances_m", "absolute_distance_m",
        )}
        for name, inert_value in INERT_CONTEXT_VALUES.items():
            if not self.fields[name].evidence:
                values[name] = inert_value
        return SafetyContext(**values)

    def inert_state(self, state: SafetyState) -> SafetyState:
        """The state `apply_safety_layer` reads (validator round 1, F1
        follow-up, found by the coordinator): mirrors `inert_context()`
        for the four `SafetyState` fields that method cannot reach (it
        only builds a `SafetyContext`) and that decision 3's evidence
        machinery covers: `last_lane_change_time_s`, `lane_changes_last_km`,
        `lane_change_distances_m` and `absolute_distance_m`. All four are
        always class (C) -- never evidence -- and were being passed to
        `apply_safety_layer` RAW, unneutralised.

        The other two `SafetyState` fields (`distance_since_window_start_m`,
        `last_lane_index`) are read by no function `apply_safety_layer`
        reaches at all -- not merely ungated, genuinely dead as far as the
        twelve rules are concerned -- and are copied through unchanged,
        from `state` itself, not from `self.fields`, which never carries
        them.

        This method has no sibling named `state()`: an earlier one existed,
        had zero callers (`run_safety_gate` always uses its own `state`
        argument, never a reconstruction from `SafetyInputs`), and was
        removed rather than left beside this one for a later reader to
        wire to by mistake.
        """
        return dataclasses_replace(
            state,
            last_lane_change_time_s=(
                self.fields["last_lane_change_time_s"].value
                if self.is_evidence("last_lane_change_time_s")
                else INERT_STATE_VALUES["last_lane_change_time_s"]
            ),
            lane_changes_last_km=(
                self.fields["lane_changes_last_km"].value
                if self.is_evidence("lane_changes_last_km")
                else INERT_STATE_VALUES["lane_changes_last_km"]
            ),
            lane_change_distances_m=(
                list(self.fields["lane_change_distances_m"].value)
                if self.is_evidence("lane_change_distances_m")
                else list(INERT_STATE_VALUES["lane_change_distances_m"])
            ),
            absolute_distance_m=(
                self.fields["absolute_distance_m"].value
                if self.is_evidence("absolute_distance_m")
                else INERT_STATE_VALUES["absolute_distance_m"]
            ),
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
        # validator round 1, F1 follow-up: read by _lane_change_count_exceeded
        # inside the window branch, alongside lane_change_distances_m, and by
        # no other function -- structurally absent for the same reason as the
        # three above (no odometry sensor exists on this rig at all).
        "absolute_distance_m": structural(state.absolute_distance_m),
    }
    return SafetyInputs(fields=fields, density_max_age_s=density_max_age_s)


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
    # absolute_distance_m added (validator round 1, F1 follow-up): read by
    # _lane_change_count_exceeded's window branch alongside
    # lane_change_distances_m, and previously in no rule's RULE_READS at
    # all. Evaluability is NOT the generic "every field here must be
    # evidence" rule for this entry -- see _lane_changes_per_km_missing,
    # which computes it by which branch the tick's own
    # lane_change_distances_m value actually selects.
    "lane_changes_per_km": ("lane_changes_last_km", "lane_change_distances_m", "absolute_distance_m"),
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

#: validator round 2 (coordinator sweep, 2026-09-12): built the most
#: favourable observation `safety_inputs_from_observation` can produce --
#: every source `measured`, a fresh `last_detection_age_s` -- and checked
#: which rules still have no field able to carry evidence. Seven of twelve
#: are unreachable on this rig BY CONSTRUCTION, not by chance of what a
#: drive happened to record, because the fields they read are class (C)
#: unconditionally, regardless of any observation:
#: `all_lane_low_speed_occupancy` (no cooperating peers -- reads
#: `all_lanes_av_occupied`/`av_mean_speed_mps`/`downstream_congested`, all
#: V2V-only), `lane_change_dwell` and `lane_changes_per_km` (no
#: lane-change detector), `target_lane_missing` (no lane detection),
#: `target_lane_rear_gap`/`target_lane_rear_ttc`/`target_lane_rear_braking`
#: (no rear sensor). The remaining five are reachable, wholly
#: (`low_speed_uncongested`, `target_lane_front_gap`) or partly --
#: `passing_lane_slow_hold` (`local_mean_speed_mps` yes, `in_passing_lane`
#: no), `target_lane_front_ttc` (gap yes, relative speed no), `forward_ttc`
#: (leader pair yes, merge-conflict pair no). This is why a per-rule
#: evaluability refinement (`_forward_ttc_missing`,
#: `_lane_changes_per_km_missing`) can only ever be pinned by a hand-built
#: `SafetyInputs` for `lane_changes_per_km` (wholly unreachable) and, on
#: its unreachable merge-conflict axis only, for `forward_ttc` -- see the
#: docstrings on both functions and on their tests in
#: `tests/test_safety_gate.py`.

#: validator round 1, F1: the value substituted for a SafetyContext field's
#: slot in the copy `apply_safety_layer` reads (`SafetyInputs.inert_context`),
#: whenever `is_evidence(name)` is False for that field this tick. Each entry
#: is derived from the one comparison RULE_READS says the field gates --
#: never guessed -- and states which direction is inert and why. Checked: no
#: field here is read by two rules with opposite senses (a field read by two
#: rules -- the front and rear gap/relative-speed pairs -- is read by both
#: with the SAME sense, "larger gap / non-closing is safer", so one value
#: serves both; see the notes below).
#:
#: A field absent from this table gates none of the twelve rules and is left
#: out on purpose: `ego_speed_mps` (evidence-required) only feeds
#: `physical_control_command`'s unrecorded `acceleration_mps2`, never
#: `target_speed_mps`/`lane_action`/`emergency_override`; `follower_gap_m`,
#: `follower_relative_speed_mps` and `near_merge` (all structural) are read
#: nowhere at all in this module. `inert_context()` passes all four through
#: unchanged regardless of evidence, since there is no comparison to make
#: inert.
INERT_CONTEXT_VALUES: dict[str, Any] = {
    # low_speed_uncongested fires on density < uncongested_density_threshold
    # (12.0 veh/km): any value at or above the threshold is inert, and inf
    # is the unambiguous choice.
    "local_density_veh_per_km": float("inf"),
    # all_lane_low_speed_occupancy is `all_lanes_av_occupied and not
    # downstream_congested and av_mean_speed_mps < free_flow - 8`. All three
    # reads are class (C) -- always substituted together on this rig -- so
    # each is given its own independently-sufficient direction rather than
    # relying on the other two:
    "all_lanes_av_occupied": False,           # AND on False is False
    "downstream_congested": True,             # `not True` is False
    "av_mean_speed_mps": float("inf"),        # never below free_flow - 8
    # passing_lane_slow_hold is `in_passing_lane and target_speed <
    # local_mean_speed - 8`. in_passing_lane is class (C) (always
    # substituted) and alone always neutralises the AND; local_mean_speed_mps
    # is class (B) (not always substituted alongside it) and is given its
    # own independent direction too.
    "in_passing_lane": False,                 # AND on False is False
    "local_mean_speed_mps": float("-inf"),    # target_speed can never clear threshold - 8
    # target_lane_missing fires on `not target_lane_exists`.
    "target_lane_exists": True,               # `not True` is False
    # target_lane_front_gap (gap < 5.0 m) and target_lane_front_ttc (ttc =
    # gap / closing_speed) read this field with the SAME sense -- larger gap
    # is safer for both -- so one value serves both rules.
    "target_lane_front_gap_m": float("inf"),
    # target_lane_front_ttc's closing_speed is
    # -target_lane_front_relative_speed_mps; 0.0 makes closing_speed 0,
    # which `_ttc_from_relative_speed` already treats as infinite ttc on its
    # own, independent of whatever the gap above is doing.
    "target_lane_front_relative_speed_mps": 0.0,
    # target_lane_rear_gap and target_lane_rear_ttc: the rear pair, same
    # reasoning and same sense as the front pair for both rules.
    "target_lane_rear_gap_m": float("inf"),
    # target_lane_rear_ttc's closing_speed is
    # +target_lane_rear_relative_speed_mps (no negation, unlike the front
    # pair -- a follower closes on POSITIVE relative speed): 0.0 is inert
    # here for the same reason as the front pair.
    "target_lane_rear_relative_speed_mps": 0.0,
    # target_lane_rear_braking fires on required_decel > 2.5 m/s^2; 0.0 (no
    # braking required) is at the safe end and is also the field's own
    # dataclass default.
    "target_lane_rear_required_decel_mps2": 0.0,
    # forward_ttc's hazard is min(leader_gap_m, merge_conflict_gap_m). An
    # infinite leader gap can never be the smaller of the two and can never
    # itself read as critical (gap < min_front_gap_m), independent of
    # whatever the merge-conflict pair (always class (C) below) holds.
    "leader_gap_m": float("inf"),
    # forward_ttc's closing_speed from the leader pair is
    # -leader_relative_speed_mps; 0.0 makes it non-positive, so
    # `_ttc_from_relative_speed` returns infinite ttc regardless of the gap.
    "leader_relative_speed_mps": 0.0,
    # Always class (C) on this rig (no merge-conflict sensor exists at all);
    # same sense and same reasoning as the leader pair.
    "merge_conflict_gap_m": float("inf"),
    "merge_conflict_relative_speed_mps": 0.0,
}


#: validator round 1, F5: which `RuleRecord.evidence`/`to_record()` key each
#: rule's fired/quiet determination compares against its threshold -- used
#: by `score_safety.py` and `eval_run._safety_lines` to count how many of a
#: rule's EVALUABLE ticks carry a non-finite compared value. A firing rate
#: computed entirely over ticks where the compared quantity is inf on every
#: one (`target_lane_front_gap`'s own reproduction: 1,229 of 1,229 "evaluable"
#: ticks, all tagged `derived`, gap_m == inf on all of them) is not a rate at
#: all -- the threshold could never physically be crossed, so "0.0%" reads
#: as a safety measurement when it is a description of the substituted
#: geometry. Omitted for a rule whose comparand cannot be non-finite:
#: `target_lane_missing`'s `target_lane_exists` is a bool, and
#: `lane_changes_per_km`'s `lane_changes_last_km` is a bounded count.
RULE_COMPARED_VALUE_KEY: dict[str, str] = {
    "low_speed_uncongested": "local_density_veh_per_km",
    "all_lane_low_speed_occupancy": "av_mean_speed_mps",
    "passing_lane_slow_hold": "target_speed_mps",
    "lane_change_dwell": "dwell_s",
    "target_lane_front_gap": "gap_m",
    "target_lane_rear_gap": "gap_m",
    "target_lane_front_ttc": "ttc_s",
    "target_lane_rear_ttc": "ttc_s",
    "target_lane_rear_braking": "required_decel_mps2",
    "forward_ttc": "gap_m",
}


#: validator round 1, F5 (provenance-consistency finding, coordinator
#: measurement 2026-09-12): rule name -> (this field's own `field_sources`
#: key, the `field_sources` key it is DEFINED to alias verbatim).
#: `perception/observation_builder.py` sets `target_lane_front_gap`'s VALUE
#: (`float(leader_gap)`) and its SOURCE (`src["leader_gap"]`) from the same
#: `leader_gap` variable, unconditionally, in the same two dict literals --
#: no other writer of either key exists under `perception/`. Under today's
#: contract the two `field_sources` entries are therefore always equal; a
#: persisted tick where they disagree (measured: every one of
#: `outputs/task42_usb/baseline_run_20260902_183446`'s 1,229 ticks --
#: `target_lane_front_gap` tagged `derived`, `leader_gap` tagged
#: `fallback_neutral`, both non-finite) was written under a DIFFERENT,
#: superseded builder, and cannot be trusted as evidence regardless of what
#: its own recorded source claims. `score_safety.py` and `eval_run.py` use
#: this to flag such ticks rather than silently either trusting or
#: discarding them. Named concretely rather than built as a general
#: alias-checking mechanism -- this is the one rule this task found it
#: changing an evaluability verdict for.
RULE_ALIASED_PROVENANCE_FIELDS: dict[str, tuple[str, str]] = {
    "target_lane_front_gap": ("target_lane_front_gap", "leader_gap"),
}

#: F10 (validator round 1, Fix 10): the `field_sources`/`obs` key
#: target_lane_front_gap_m's value is actually aliased from -- named once
#: here and recorded in `target_lane_front_gap`/`target_lane_front_ttc`'s
#: own evidence (below) so a reader of the per-tick record cannot mistake
#: an evaluable gap for a genuine target-lane measurement.
TARGET_LANE_FRONT_GAP_SOURCE_SLOT = "leader_gap"

#: validator round 1, F1 follow-up (found by the coordinator, 2026-09-12):
#: `SafetyContext` and `SafetyState` are the complete set of containers
#: `apply_safety_layer` reads sensed data from -- derived, not assumed, by
#: the transitive closure of every function it calls: the only
#: module-level data touched anywhere in that closure is
#: `sim_contract.decode_speed_bin`/`decode_headway_bin`, contract constants
#: rather than sensed values. `inert_context()` neutralises the first
#: container; this table and `SafetyInputs.inert_state()` neutralise the
#: second, which `run_safety_gate` was passing to `apply_safety_layer` RAW.
#: Latent only because nothing on this rig ever advances these past their
#: class defaults (`None`, `0`, `()`, and -- for `absolute_distance_m`, see
#: below -- `0.0`), which happen to already be inert -- the same shape F1
#: itself had before a policy emitted `slow`. `SafetyConstraints` needs no
#: inert treatment: it is operator configuration, not sensing, so it was
#: never a candidate for substitution in the first place, not an oversight.
#:
#: One uniform neutralisation policy cannot serve every rule this table
#: feeds, and `lane_changes_per_km` is why: `_lane_change_count_exceeded`
#: consults `lane_change_distances_m` (and `absolute_distance_m` alongside
#: it) OR `lane_changes_last_km`, never both -- an alternative, not a
#: conjunction -- so evaluability must be computed by WHICH BRANCH the
#: inert-or-real `lane_change_distances_m` value actually selects, not by
#: requiring every field in this table as evidence unconditionally (see
#: `_lane_changes_per_km_missing`). The values below stay a single fixed
#: substitution each; only which of them EVALUABILITY depends on, per
#: tick, is rule-specific.
#:
#: Each entry derived from the one comparison the reading rule makes, the
#: same way as `INERT_CONTEXT_VALUES`:
INERT_STATE_VALUES: dict[str, Any] = {
    # lane_change_dwell masks when `time_s - last < lane_change_dwell_s`;
    # `_dwell_s` already special-cases `None` to mean "no lane change yet"
    # (dwell = inf), so `None` is the value the code already treats as
    # inert, not a new convention introduced here.
    "last_lane_change_time_s": None,
    # lane_changes_per_km's `_lane_change_count_exceeded` takes the
    # window-based branch whenever `lane_change_distances_m` is non-empty
    # and does not read `lane_changes_last_km` at all in that case; else it
    # falls back to `lane_changes_last_km` directly. `()` is falsy, so it
    # sends the function to the `lane_changes_last_km` branch; `0` there is
    # below `max_lane_changes_per_km` for any positive threshold.
    "lane_change_distances_m": (),
    "lane_changes_last_km": 0,
    # Read only inside the window branch, alongside `lane_change_distances_m`
    # (`window_start = max(0.0, absolute_distance_m - 1000.0)`), and in NO
    # rule's `RULE_READS` before this fix. `+inf` pushes `window_start` to
    # `+inf`, so no real (finite) recorded distance can ever read as
    # "recent" -- the same gap-inertness derivation as `leader_gap_m`,
    # applied to a distance-since-start counter instead of a distance-ahead
    # one. This field can never itself be evidence (no odometry sensor
    # exists on this rig at all), so whenever the window branch is the one
    # a tick's `lane_change_distances_m` selects, `lane_changes_per_km` is
    # not_evaluable regardless of `lane_change_distances_m`'s own evidence
    # -- a second sensor this rule needs that this rig will never have,
    # the same shape as `forward_ttc`'s permanently-absent merge-conflict
    # pair.
    "absolute_distance_m": float("inf"),
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


def _forward_ttc_missing(inputs: SafetyInputs, constraints: SafetyConstraints) -> tuple[str, ...]:
    """`forward_ttc`'s evaluability, computed over the inputs the tick's
    OWN firing condition actually consulted, not the static list of
    inputs it might consult (validator round 1, corrected ruling --
    the first version of this function, which required only a pair's
    gap and relative speed together as a unit, was itself wrong: it
    suppressed a real emergency brake on a genuinely measured 3 m gap
    whenever that gap's relative speed was substituted, which is the
    NORMAL state while a track is newly acquired --
    `perception/observation_builder.py` sets a leader's gap to MEASURED
    as soon as a leader exists, but its relative speed only once
    `rel_speed_valid`, which lags by design).

    `physical_control_command` fires forward_ttc's emergency override on
    `ttc < min_forward_ttc_s OR hazard_gap < min_front_gap_m` -- two
    disjuncts over the SAME operative pair (whichever of leader /
    merge-conflict supplies the smaller gap, decided the same way as
    before: a pair without gap evidence sits at its inert +inf, which can
    never be the minimum). The two disjuncts do not need the same
    evidence:

    - the gap disjunct (`hazard_gap < min_front_gap_m`) reads only the
      operative gap;
    - the ttc disjunct reads the operative gap AND its relative speed.

    So: if the operative gap itself lacks evidence, neither disjunct can
    be evaluated at all. If it has evidence and is already close enough
    to fire the gap disjunct, the OR is true regardless of the relative
    speed's evidence -- the ttc disjunct's own answer cannot change the
    result. Only when the gap disjunct reads false (on real evidence)
    does the relative speed become necessary, because only then does the
    ttc disjunct decide the outcome.
    """
    leader_gap_effective = (
        inputs.fields["leader_gap_m"].value if inputs.is_evidence("leader_gap_m")
        else INERT_CONTEXT_VALUES["leader_gap_m"]
    )
    merge_gap_effective = (
        inputs.fields["merge_conflict_gap_m"].value if inputs.is_evidence("merge_conflict_gap_m")
        else INERT_CONTEXT_VALUES["merge_conflict_gap_m"]
    )
    if merge_gap_effective < leader_gap_effective:
        gap_field, relspeed_field = "merge_conflict_gap_m", "merge_conflict_relative_speed_mps"
    else:
        gap_field, relspeed_field = "leader_gap_m", "leader_relative_speed_mps"

    if not inputs.is_evidence(gap_field):
        # Neither disjunct can be evaluated without the operative gap.
        return (
            (gap_field,) if inputs.is_evidence(relspeed_field)
            else (gap_field, relspeed_field)
        )
    if inputs.fields[gap_field].value < constraints.min_front_gap_m:
        # The gap disjunct alone already makes the OR true; the ttc
        # disjunct's relative-speed requirement never has to be asked.
        return ()
    # The gap disjunct read false on real evidence; only the ttc disjunct
    # can still decide "fired", and it needs the relative speed too.
    return () if inputs.is_evidence(relspeed_field) else (relspeed_field,)


def _lane_changes_per_km_missing(inputs: SafetyInputs) -> tuple[str, ...]:
    """`lane_changes_per_km`'s evaluability, computed over the inputs the
    tick's own branch actually consulted (the same principle as
    `_forward_ttc_missing`, validator round 1's correction: the first
    version of this fix required both `lane_change_distances_m` and
    `lane_changes_last_km` as evidence together, which was also wrong --
    `_lane_change_count_exceeded` consults them as ALTERNATIVES, never
    both, so requiring both would report `not_evaluable` on a tick whose
    `lane_changes_last_km` is real evidence and decisive, only because an
    unconsulted `lane_change_distances_m` happened to also lack it).

    `_lane_change_count_exceeded` takes the window-based branch whenever
    `lane_change_distances_m` -- after inert substitution, mirroring
    `_forward_ttc_missing`'s gap comparison -- is non-empty, and reads
    `lane_changes_last_km` only in the else branch. Whichever branch is
    selected is the one whose fields must have evidence.

    The window branch also reads `absolute_distance_m`, in no rule's
    `RULE_READS` before this fix and never itself evidence (no odometry
    sensor exists on this rig at all) -- so whenever the window branch is
    the one selected, this rule is not_evaluable regardless of
    `lane_change_distances_m`'s own evidence, the same shape as
    `forward_ttc`'s permanently-absent merge-conflict pair: a second
    sensor this rule needs that this rig will never have.
    """
    distances_effective = (
        inputs.fields["lane_change_distances_m"].value if inputs.is_evidence("lane_change_distances_m")
        else INERT_STATE_VALUES["lane_change_distances_m"]
    )
    if distances_effective:
        window_fields = ("lane_change_distances_m", "absolute_distance_m")
        return tuple(f for f in window_fields if not inputs.is_evidence(f))
    return () if inputs.is_evidence("lane_changes_last_km") else ("lane_changes_last_km",)


def _evaluate_one_rule(
    name: str,
    *,
    action: Mapping[str, str],
    inputs: SafetyInputs,
    context: SafetyContext,
    state: SafetyState,
    constraints: SafetyConstraints,
) -> RuleRecord:
    if name == "forward_ttc":
        missing = _forward_ttc_missing(inputs, constraints)
    elif name == "lane_changes_per_km":
        missing = _lane_changes_per_km_missing(inputs)
    else:
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
        evidence = {
            "gap_m": context.target_lane_front_gap_m, "threshold_m": constraints.min_front_gap_m,
            # F10 (plan_task144's own out-of-scope finding, still true with
            # the gate live): perception/observation_builder.py sets this
            # slot to the CURRENT lane's leader_gap
            # (src["target_lane_front_gap"] = src["leader_gap"]), not a
            # measurement of the lane being changed INTO. Named here so a
            # reader of this rule's evidence cannot mistake "evaluable, gap
            # 3.0 m" for a genuine target-lane reading.
            "gap_source_slot": TARGET_LANE_FRONT_GAP_SOURCE_SLOT,
        }
    elif name == "target_lane_rear_gap":
        fired = context.target_lane_rear_gap_m < constraints.min_rear_gap_m
        evidence = {"gap_m": context.target_lane_rear_gap_m, "threshold_m": constraints.min_rear_gap_m}
    elif name == "target_lane_front_ttc":
        ttc = _ttc_from_relative_speed(context.target_lane_front_gap_m, -context.target_lane_front_relative_speed_mps)
        fired = ttc < constraints.min_lane_change_ttc_s
        evidence = {
            "ttc_s": ttc, "threshold_s": constraints.min_lane_change_ttc_s,
            # Same F10 caveat as target_lane_front_gap above: the gap half
            # of this ttc is the current lane's leader_gap.
            "gap_source_slot": TARGET_LANE_FRONT_GAP_SOURCE_SLOT,
        }
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
class SafetyGateConfig:
    """validator round 1, F3/F4: the gate configuration this tick's decision
    and census were actually run under, persisted alongside them so a replay
    tool (`score_safety.py`) can reproduce the SAME run rather than silently
    assuming its own current defaults. Both F3 (`score_safety.py` always
    replayed with `withhold_lane_when_not_evaluable=True`, so it could not
    read a run recorded with the flag off) and F4 (it reconstructed `time_s`
    as the tick's epoch `t_wall`, when the live pipeline passes
    `time.monotonic()`) trace back to this information never having been
    recorded at all.
    """

    enabled: bool
    withhold_lane_when_not_evaluable: bool
    time_s: float
    min_contextual_speed_mps: float
    density_max_age_s: float

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "withhold_lane_when_not_evaluable": self.withhold_lane_when_not_evaluable,
            "time_s": self.time_s,
            "min_contextual_speed_mps": self.min_contextual_speed_mps,
            "density_max_age_s": self.density_max_age_s,
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
    config: SafetyGateConfig
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
            "config": self.config.to_record(),
        }


def run_safety_gate(
    action: Mapping[str, str],
    inputs: SafetyInputs,
    state: SafetyState,
    constraints: SafetyConstraints | None = None,
    *,
    withhold_lane_when_not_evaluable: bool = True,
    enabled: bool = True,
) -> SafetyGateResult:
    """Run the vendored `apply_safety_layer` (the bound the driver is shown)
    and the independent per-rule census (decision 3) from the same inputs,
    then apply decision 3's own addition: withhold the lane action outright
    when any of its guards could not be evaluated (open item 1), rather than
    letting an unevaluated guard's "safe" default speak for it.

    `apply_safety_layer` is run against `inputs.inert_context()` and
    `inputs.inert_state(state)`, not the raw `context`/`state` (validator
    round 1, F1, and its follow-up: `SafetyContext` and `SafetyState` are
    the complete set of containers the decision reads sensed data from --
    derived from `apply_safety_layer`'s own transitive call closure, not
    assumed). Every field either container lacks evidence for is replaced
    by its `INERT_CONTEXT_VALUES`/`INERT_STATE_VALUES` entry before it
    reaches the decision, so a not_evaluable rule cannot move the number
    the driver is shown even when the observation's own substituted value
    for that field is not itself inert (`local_density_veh_per_km`'s 0.0
    default was the reproduced case: 0.0 is BELOW the uncongested threshold,
    the opposite of inert). `evaluate_rules`'s census, below, keeps reading
    the unmodified `context`/`state` -- the record must still say what was
    actually observed, not what the decision was neutralised against.

    This closes the invariant along one axis only (which containers the
    decision reads are neutralised); `forward_ttc`'s own evaluability
    computation (`_forward_ttc_missing`) closes a second, different axis --
    an evidence requirement wider than the comparison that actually
    determines the rule's result.

    `enabled=False` (validator round 1, F2/F3 -- `safety.enabled`) is the
    actual rollback mechanism `config.yaml`/`ARCHITECTURE.md` had wrongly
    attributed to `withhold_lane_when_not_evaluable`: the census and the
    full record are still produced, but `apply_safety_layer` is not run at
    all, so `bounded_* == proposed_*` exactly (including headway -- even the
    unconditional `create_gap` bonus does not apply, since that bonus is
    `apply_safety_layer`'s own first step) and the lane action is never
    withheld. `withhold_lane_when_not_evaluable` continues to cover only the
    lane/merge action, as it always has.
    """
    constraints = constraints or SafetyConstraints()
    context = inputs.context()
    rules = evaluate_rules(action, inputs, state, constraints)

    proposed_speed = _target_speed(action, context)
    proposed_headway = sim_contract.decode_headway_bin(action["desired_headway_bin"])
    proposed_lane_action = (
        _lane_preference_to_action(action["lane_preference"]) if action["merge_mode"] != "hold_lane" else None
    )

    if enabled:
        inert_context = inputs.inert_context()
        inert_state = inputs.inert_state(state)
        decision = apply_safety_layer(action, inert_state, inert_context, constraints)
        bounded_speed = decision.target_speed_mps
        bounded_headway = decision.target_headway_s
        bounded_lane_action = decision.lane_action
        emergency_override = decision.emergency_override
        raw_diagnostics = decision.diagnostics

        lane_guard_not_evaluable = any(rules[name].status == RULE_NOT_EVALUABLE for name in LANE_GUARD_RULES)
        lane_withheld: str | None = None
        if withhold_lane_when_not_evaluable and lane_guard_not_evaluable and bounded_lane_action is not None:
            bounded_lane_action = None
            lane_withheld = RULE_NOT_EVALUABLE
    else:
        bounded_speed = proposed_speed
        bounded_headway = proposed_headway
        bounded_lane_action = proposed_lane_action
        emergency_override = False
        raw_diagnostics = empty_diagnostics()
        lane_withheld = None

    evaluable = sum(1 for r in rules.values() if r.status != RULE_NOT_EVALUABLE)
    config = SafetyGateConfig(
        enabled=enabled,
        withhold_lane_when_not_evaluable=withhold_lane_when_not_evaluable,
        time_s=context.time_s,
        min_contextual_speed_mps=context.min_contextual_speed_mps,
        density_max_age_s=inputs.density_max_age_s,
    )
    return SafetyGateResult(
        proposed_speed_mps=proposed_speed,
        proposed_headway_s=proposed_headway,
        proposed_lane_action=proposed_lane_action,
        bounded_speed_mps=bounded_speed,
        bounded_headway_s=bounded_headway,
        bounded_lane_action=bounded_lane_action,
        delta_speed_mps=bounded_speed - proposed_speed,
        emergency_override=emergency_override,
        lane_withheld=lane_withheld,
        evaluable_count=evaluable,
        not_evaluable_count=len(rules) - evaluable,
        rules=rules,
        config=config,
        raw_diagnostics=raw_diagnostics,
    )
