"""Vendored safety and etiquette layer, run on the device advisory path.

`SafetyConstraints`, `SafetyContext`, `SafetyDecision`,
`apply_safety_layer`, `physical_control_command` and `safety_penalty_terms`
mirror `src/safety/{constraints,etiquette,safety_layer}.py` field for field
and line for line. Guarded against drift by `specs/safety_contract_golden.json`
rather than by importing the original: `deployment/jetson/tests/
test_safety_contract.py` checks this module against that file unconditionally,
`tests/test_safety_contract_matches_golden.py` checks `src/safety/` against
the same file, and neither imports the other side. Task 143's finding is why:
a check written as "compare the vendored copy against the original" goes
vacuous the day the original is deleted; a golden file cannot.

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
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET

# --- Vendored from src/safety/constraints.py ---------------------------------


@dataclass(frozen=True)
class SafetyConstraints:
    """Conservative defaults for the bound on a recommended speed.

    Every entry is read by `apply_safety_layer` or `physical_control_command`.
    The lane-change, rear-gap and merge-bonus constraints were removed with the
    rules that read them: the advisory is a speed, so a constraint earns its
    place by bounding that speed.
    """

    #: Below this density the road is uncongested, and a recommendation more
    #: than `low_speed_free_flow_delta_mps` under free flow is raised.
    uncongested_density_threshold_veh_per_km: float = 12.0
    low_speed_free_flow_delta_mps: float = 8.0
    #: The forward hazard bounds: closer than the gap, or sooner than the
    #: time-to-collision, and the decision becomes an emergency deceleration.
    min_front_gap_m: float = 5.0
    min_forward_ttc_s: float = 2.0
    emergency_decel_mps2: float = 6.0
    #: The proportional speed controller and the ordinary acceleration bounds
    #: it is clipped to.
    speed_control_kp: float = 0.6
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0


# --- Vendored from src/safety/safety_layer.py --------------------------------


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
    """Port of src/safety/safety_layer.py:apply_safety_layer.

    Takes the speed directly rather than an action to decode: the controller
    this bounds emits one speed per super-segment, so there is no bin to decode
    and no lane or merge head to read.
    """
    constraints = constraints or SafetyConstraints()
    diagnostics = empty_diagnostics()
    target_speed = target_speed_mps
    target_headway = target_headway_s

    if _is_low_speed_uncongested(target_speed, context.free_flow_speed_mps, context.local_density_veh_per_km, constraints):
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
    """Verbatim port of src/safety/safety_layer.py:safety_penalty_terms.

    One term per rule that can fire. `unsafe_lane_preference`,
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

#: decision 3's partition, over the seven SafetyContext fields that remain.
#: The four lane-history names came with the rules that read lane-change
#: history and went with them; the device could never advance any of the four,
#: so nothing that could be measured was lost.
CONFIGURED_FIELDS: tuple[str, ...] = ("free_flow_speed_mps",)
EVIDENCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "ego_speed_mps",
    "leader_gap_m",
    "leader_relative_speed_mps",
    "local_density_veh_per_km",
)
#: The merge-conflict pair is the whole of class (C) now. It needs a map
#: match to a joining node, which this rig has no way to make, so it is
#: structurally absent rather than merely unmeasured this tick.
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "merge_conflict_gap_m",
    "merge_conflict_relative_speed_mps",
)


@dataclass(frozen=True)
class SafetyInputField:
    """One SafetyContext field's value, its decision-3 input class, and
    whether it counts as evidence this tick."""

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
    running `ObservationResult`.
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
        return SafetyContext(**{name: f.value for name, f in self.fields.items()})

    def inert_context(self) -> SafetyContext:
        """The context `apply_safety_layer` reads (validator round 1, F1):
        identical to `context()` except that every field with
        `is_evidence(name)` False this tick is replaced by
        `INERT_CONTEXT_VALUES[name]` rather than the observation's own
        substituted value, so a not_evaluable rule is inert in the number
        the driver sees, not only in the record (decision 3: "nothing. It
        is recorded and the advisory is unchanged"). `evaluate_rules`'s
        census also reads this method now, not the raw `context()`
        (validator round 3: the two disagreeing about which
        disjunct/branch a partial-evidence tick's `missing` reasoned about
        let the census assert `fired` on values the decision never
        evidenced -- see `evaluate_rules`'s own docstring for the
        reproduction).

        `ego_speed_mps` is absent from `INERT_CONTEXT_VALUES` and is passed
        through unchanged regardless of evidence: it gates neither rule, and
        only feeds `physical_control_command`'s unrecorded
        `acceleration_mps2`, so there is no comparison to make inert.
        """
        values = {name: f.value for name, f in self.fields.items()}
        for name, inert_value in INERT_CONTEXT_VALUES.items():
            if not self.fields[name].evidence:
                values[name] = inert_value
        return SafetyContext(**values)

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
    *,
    density_max_age_s: float,
) -> SafetyInputs:
    """Build `SafetyInputs` from one tick's `ObservationResult`, per decision
    3's three-class partition. `density_max_age_s` is the bound
    `local_density_veh_per_km`'s `derived_empty` carve-out checks
    `obs_diagnostics.last_detection_age_s` against -- twice
    `BuilderConfig.gps_stale_after_s`, per decision 3.

    The running `SafetyState` was a parameter here, read for the four
    lane-history counters. It went with the lane rules: nothing on this rig
    could ever advance any of the four, and the state object had no other
    field.
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
        "free_flow_speed_mps": configured(
            "free_flow_speed_mps", float(cooperation.get("segment_target_speed", 30.0))
        ),
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
        # A joining node needs a map match this rig cannot make, so these two
        # are structurally absent rather than unmeasured this tick.
        "merge_conflict_gap_m": structural(float("inf")),
        "merge_conflict_relative_speed_mps": structural(0.0),
    }
    return SafetyInputs(fields=fields, density_max_age_s=density_max_age_s)


# --- decision 3 / step 9: the twelve rules as total independent predicates ---

#: The order `apply_safety_layer`'s elif chain checks the eight lane/merge
#: guards in -- used by
#: tests/test_safety_gate_pipeline.py's chain-order invariant.
#: The two rules that survive the speed-only advisory. A rule earns its place
#: by bounding the recommended speed: `low_speed_uncongested` raises it, and
#: `forward_ttc` drives the emergency override. The nine that only nulled the
#: lane action went with the lane action itself.
RULE_NAMES: tuple[str, ...] = (
    "low_speed_uncongested",
    "forward_ttc",
)

#: Which SafetyInputs field names each rule's not_evaluable verdict depends
#: on. A rule is not_evaluable when ANY named field is not evidence.
RULE_READS: dict[str, tuple[str, ...]] = {
    "low_speed_uncongested": ("local_density_veh_per_km",),
    "forward_ttc": (
        "leader_gap_m", "leader_relative_speed_mps",
        "merge_conflict_gap_m", "merge_conflict_relative_speed_mps",
    ),
}

#: validator round 2/3 sweep (2026-09-12), re-read against the two rules that
#: remain: both can be evaluable through `safety_inputs_from_observation`.
#: `low_speed_uncongested` is gated by a single class-(B) field, and
#: `forward_ttc` is reachable only because of `_forward_ttc_missing`'s own
#: per-disjunct refinement below -- under the generic "every `RULE_READS`
#: field must be evidence" rule its permanently-class-(C) merge-conflict pair
#: would make it permanently not evaluable. The nine rules the sweep found
#: unreachable are gone: each of them only nulled the lane action, and the
#: lane action went with them.

#: validator round 1, F1: the value substituted for a SafetyContext field's
#: slot in the copy `apply_safety_layer` reads (`SafetyInputs.inert_context`),
#: whenever `is_evidence(name)` is False for that field this tick. Each entry
#: is derived from the one comparison RULE_READS says the field gates --
#: never guessed -- and states which direction is inert and why.
#:
#: Two fields are absent from this table on purpose. `ego_speed_mps` gates
#: neither rule: it only feeds `physical_control_command`'s unrecorded
#: `acceleration_mps2`, never `target_speed_mps` or `emergency_override`.
#: `free_flow_speed_mps` is class (A), accepted as given. `inert_context()`
#: passes both through unchanged regardless of evidence, since there is no
#: comparison to make inert.
INERT_CONTEXT_VALUES: dict[str, Any] = {
    # low_speed_uncongested fires on density < uncongested_density_threshold
    # (12.0 veh/km): any value at or above the threshold is inert, and inf
    # is the unambiguous choice.
    "local_density_veh_per_km": float("inf"),
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

}

#: F10 (validator round 1, Fix 10): the `field_sources`/`obs` key
#: target_lane_front_gap_m's value is actually aliased from -- named once
#: here and recorded in `target_lane_front_gap`/`target_lane_front_ttc`'s
#: own evidence (below) so a reader of the per-tick record cannot mistake
#: an evaluable gap for a genuine target-lane measurement.
TARGET_LANE_FRONT_GAP_SOURCE_SLOT = "leader_gap"


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



def _evaluate_one_rule(
    name: str,
    *,
    target_speed: float,
    inputs: SafetyInputs,
    context: SafetyContext,
    constraints: SafetyConstraints,
) -> RuleRecord:
    if name == "forward_ttc":
        missing = _forward_ttc_missing(inputs, constraints)
    else:
        missing = tuple(f for f in RULE_READS[name] if not inputs.is_evidence(f))
    if missing:
        evidence = {f"{f}_source": inputs.fields[f].source for f in missing}
        return RuleRecord(status=RULE_NOT_EVALUABLE, missing=missing, evidence=evidence)

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
    target_speed: float,
    inputs: SafetyInputs,
    constraints: SafetyConstraints,
) -> dict[str, RuleRecord]:
    """Both rules, each a total predicate over its own evaluability.

    Evaluates each predicate over `inputs.inert_context()`, not the raw
    `context()` (validator round 3 correction). Decision 3 originally said the
    census reads the unmodified context; that was wrong for `forward_ttc`,
    whose `missing` is computed from EFFECTIVE (inert-or-real) values to
    decide which disjunct applies (`_forward_ttc_missing`). Running the
    predicate itself on the RAW context let that choice disagree
    with what `missing` reasoned about: reproduced by the coordinator --
    leader pair not evidence (raw gap 6.0 m), merge pair evidence (gap
    6.0 m). `missing` picks the merge pair (its real 6.0 m beats the
    leader's inert +inf) and reports `()`; the raw predicate then compares
    `merge_conflict_gap_m(6.0) < leader_gap_m(6.0)`, which is False, so it
    silently kept the LEADER pair -- and reported `fired`, with `ttc_s`
    computed from the leader's un-evidenced gap and relative speed. A
    positive claim on substituted values is worse than the `not_evaluable`
    problem F1 fixed: that at least said "unknown".

    The census's job is to describe truthfully what the decision did, and
    a census computed over values the decision never saw describes a
    hypothetical, not this tick. Nothing is lost by neutralising here:
    `Tick.to_record()`'s `obs`/`field_sources` still carry the actual
    observed values and their provenance regardless of what this function
    reports; only the `evidence` dict below changes, to name the values
    the rule actually compared.
    """
    context = inputs.inert_context()
    return {
        name: _evaluate_one_rule(name, target_speed=target_speed, inputs=inputs, context=context, constraints=constraints)
        for name in RULE_NAMES
    }


@dataclass(frozen=True)
class SafetyGateConfig:
    """validator round 1, F3/F4: the gate configuration this tick's decision
    and census were actually run under, persisted alongside them so a replay
    tool (`score_safety.py`) can reproduce the SAME run rather than silently
    assuming its own current defaults. Both F3 (`score_safety.py` always
    replayed with the lane guards in place, so it could not
    read a run recorded with the flag off) and F4 (it reconstructed `time_s`
    as the tick's epoch `t_wall`, when the live pipeline passes
    `time.monotonic()`) trace back to this information never having been
    recorded at all.

    `min_contextual_speed_mps` was recorded here too. No surviving rule reads
    it: `low_speed_uncongested` raises a recommendation to
    `free_flow_speed_mps - low_speed_free_flow_delta_mps`, not to a
    contextual minimum. It reached this record from `config.yaml` through the
    pipeline and the context without ever being compared against anything,
    which is a configured number that cannot change any outcome.
    """

    enabled: bool
    #: The clock the gate ran on -- `time.monotonic()` in the live pipeline.
    #: Not read by either rule; recorded so a replay can order ticks on the
    #: same clock the decision was made on.
    time_s: float
    density_max_age_s: float

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "time_s": self.time_s,
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
    bounded_speed_mps: float
    bounded_headway_s: float
    delta_speed_mps: float
    emergency_override: bool
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
            },
            "bounded": {
                "speed_mps": round(self.bounded_speed_mps, 2),
                "headway_s": round(self.bounded_headway_s, 2),
            },
            "delta_speed_mps": round(self.delta_speed_mps, 2),
            "emergency_override": self.emergency_override,
            "evaluable": self.evaluable_count,
            "not_evaluable": self.not_evaluable_count,
            "rules": {name: rule.to_record() for name, rule in self.rules.items()},
            "config": self.config.to_record(),
        }


def run_safety_gate(
    proposed_speed_mps: float,
    proposed_headway_s: float,
    inputs: SafetyInputs,
    constraints: SafetyConstraints | None = None,
    *,
    time_s: float,
    enabled: bool = True,
) -> SafetyGateResult:
    """Run the vendored `apply_safety_layer` (the bound the driver is shown)
    and the independent per-rule census (decision 3) from the same inputs,
    from the same inputs. A rule that cannot be evaluated is inert: it is
    recorded as such and cannot move the bounded speed in either direction.

    `apply_safety_layer` is run against `inputs.inert_context()`, not the raw
    `context` (validator round 1, F1, and its follow-up: `SafetyContext` is
    the complete set of containers the decision reads sensed data from --
    derived from `apply_safety_layer`'s own transitive call closure, not
    assumed). Every field it lacks evidence for is replaced
    by its `INERT_CONTEXT_VALUES` entry before it
    reaches the decision, so a not_evaluable rule cannot move the number
    the driver is shown even when the observation's own substituted value
    for that field is not itself inert (`local_density_veh_per_km`'s 0.0
    default was the reproduced case: 0.0 is BELOW the uncongested threshold,
    the opposite of inert). `evaluate_rules`'s census, below, ALSO now reads
    `inert_context()`, not the raw `context()`
    (validator round 3 correction -- see `evaluate_rules`'s own docstring:
    a census computed on raw values could disagree with `missing` about
    which disjunct/branch applied and assert `fired` on substituted data).
    The record still says what was actually observed: `Tick.to_record()`'s
    `obs`/`field_sources` carry that regardless of what this function
    reports; only the per-rule `evidence` dict changes, to the values the
    rule actually compared.

    This closes the invariant along two independent axes: which container the
    decision AND the census read (neutralised in both, identically), and,
    separately, `forward_ttc`'s own evaluability computation
    (`_forward_ttc_missing`) closing an evidence requirement wider than the
    comparison that actually determines its result.

    `enabled=False` (validator round 1, F2/F3 -- `safety.enabled`) is the
    rollback mechanism: the census and the full record are still produced,
    but `apply_safety_layer` is not run at all, so `bounded_* == proposed_*`
    exactly.
    """
    constraints = constraints or SafetyConstraints()
    rules = evaluate_rules(proposed_speed_mps, inputs, constraints)

    proposed_speed = proposed_speed_mps
    proposed_headway = proposed_headway_s

    if enabled:
        inert_context = inputs.inert_context()
        decision = apply_safety_layer(proposed_speed, proposed_headway, inert_context, constraints)
        bounded_speed = decision.target_speed_mps
        bounded_headway = decision.target_headway_s
        emergency_override = decision.emergency_override
        raw_diagnostics = decision.diagnostics
    else:
        bounded_speed = proposed_speed
        bounded_headway = proposed_headway
        emergency_override = False
        raw_diagnostics = empty_diagnostics()

    evaluable = sum(1 for r in rules.values() if r.status != RULE_NOT_EVALUABLE)
    config = SafetyGateConfig(
        enabled=enabled,
        time_s=time_s,
        density_max_age_s=inputs.density_max_age_s,
    )
    return SafetyGateResult(
        proposed_speed_mps=proposed_speed,
        proposed_headway_s=proposed_headway,
        bounded_speed_mps=bounded_speed,
        bounded_headway_s=bounded_headway,
        delta_speed_mps=bounded_speed - proposed_speed,
        emergency_override=emergency_override,
        evaluable_count=evaluable,
        not_evaluable_count=len(rules) - evaluable,
        rules=rules,
        config=config,
        raw_diagnostics=raw_diagnostics,
    )
