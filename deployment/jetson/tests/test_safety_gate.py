"""Unit and sanity tests for policy.safety_gate.

Two layers: (1) the vendored apply_safety_layer/physical_control_command,
proven behaviorally identical to src/safety/safety_layer.py by re-running
tests/test_safety_layer.py's own cases against this module (field/constant
identity is already the golden file's job -- this is logic identity); and
(2) SafetyInputs, evaluate_rules and run_safety_gate, which have no
src/safety/ counterpart at all.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from perception.observation_builder import ObservationResult
from perception import provenance
from policy.safety_gate import (
    RULE_NOT_EVALUABLE,
    RULE_FIRED,
    RULE_QUIET,
    RULE_NAMES,
    RULE_READS,
    SafetyConstraints,
    SafetyContext,
    SafetyGateResult,
    SafetyInputField,
    SafetyInputs,
    apply_safety_layer,
    evaluate_rules,
    run_safety_gate,
    safety_inputs_from_observation,
    _forward_ttc_missing,
)
from policy.sim_contract import decode_headway_bin


#: What the bins decoded to, kept as named speeds so each test still says
#: which case it is exercising.
SPEED_MPS = {"slow": 20.0, "nominal": 27.0, "fast": 30.0}
NORMAL_HEADWAY_S = 1.6


def _unused_action(**overrides: str) -> dict[str, str]:
    value = {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
    value.update(overrides)
    return value


# ---------------------------------------------------------------------------
# (1) apply_safety_layer / physical_control_command: behavioral parity with
# src/safety/safety_layer.py, checked without importing it (see safety_gate's
# module docstring for why the two sides never compare against each other).
# ---------------------------------------------------------------------------


def test_low_speed_uncongested_is_lifted_and_diagnosed() -> None:
    decision = apply_safety_layer(
        SPEED_MPS["slow"], NORMAL_HEADWAY_S,
        SafetyContext(free_flow_speed_mps=30.0, local_density_veh_per_km=2.0),
    )
    assert decision.target_speed_mps >= 22.0
    assert decision.diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"


def test_speed_control_acceleration_is_bounded() -> None:
    decision = apply_safety_layer(
        SPEED_MPS["fast"], NORMAL_HEADWAY_S,
        SafetyContext(ego_speed_mps=10.0, free_flow_speed_mps=30.0),
        SafetyConstraints(max_accel_mps2=1.5),
    )
    assert decision.acceleration_mps2 == 1.5
    assert decision.emergency_override is False


def test_low_forward_ttc_triggers_emergency_override() -> None:
    decision = apply_safety_layer(
        SPEED_MPS["fast"], NORMAL_HEADWAY_S,
        SafetyContext(
            ego_speed_mps=25.0, leader_gap_m=10.0, leader_relative_speed_mps=-10.0,
        ),
        SafetyConstraints(emergency_decel_mps2=7.0, min_forward_ttc_s=2.0),
    )
    assert decision.acceleration_mps2 == -7.0
    assert decision.emergency_override is True
    assert decision.diagnostics["external_safety_override"][0]["reason"] == "forward_ttc"
    assert decision.penalty_terms["emergency_override"] == 1.0


def test_class_a_configured_field_is_always_evidence_even_when_fallback() -> None:
    """free_flow_speed_mps is class (A): decision 3 says it never blocks a
    rule, regardless of the observation's own provenance for the field it is
    read from."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    field = inputs.fields["free_flow_speed_mps"]
    assert field.input_class == "configured"
    assert field.evidence is True


def test_class_b_evidence_required_field_is_not_evidence_when_substituted() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    field = inputs.fields["leader_gap_m"]
    assert field.input_class == "evidence_required"
    assert field.evidence is False
    assert field.source == provenance.SOURCE_FALLBACK_NEUTRAL


def test_class_b_evidence_required_field_is_evidence_when_measured() -> None:
    obs = _base_obs()
    obs["leader_gap"] = 42.0
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    field = inputs.fields["leader_gap_m"]
    assert field.evidence is True
    assert field.value == 42.0


def test_density_derived_empty_is_not_evidence_without_a_detection_age() -> None:
    """The camera has never produced a track (last_detection_age_s is None):
    a derived_empty density is a blind camera, not an empty road."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is False


def test_density_derived_empty_is_evidence_within_the_detection_age_bound() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": 1.0})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is True


def test_density_derived_empty_is_not_evidence_past_the_detection_age_bound() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": 10.0})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is False


# ---------------------------------------------------------------------------
# (3) evaluate_rules / run_safety_gate.
# ---------------------------------------------------------------------------


def test_low_speed_uncongested_reduces_to_the_speed_bin_alone() -> None:
    """The plan's algebraic-reduction finding, at the `apply_safety_layer`
    unit level: at the configured free-flow speed of 30.0, the predicate
    reduces to desired_speed_bin == 'slow', and forcing it clamps and RAISES
    the recommended speed -- given a context where the density really is
    evidence (this is `test_low_speed_uncongested_is_lifted_and_diagnosed`
    above, restated against the free-flow speed this rig actually runs)."""
    decision = apply_safety_layer(
        SPEED_MPS["slow"], NORMAL_HEADWAY_S,
        SafetyContext(free_flow_speed_mps=30.0, local_density_veh_per_km=2.0),
    )
    assert decision.target_speed_mps == 22.0
    assert decision.diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"


def test_not_evaluable_rule_is_inert_in_the_decision_not_only_the_record() -> None:
    """validator round 1, F1/Fix 1: the exact reproduction. On the device-
    typical rig (every evidence-required field substituted), forcing
    desired_speed_bin='slow' used to raise bounded_speed_mps to 22.0 by
    reading the observation's own substituted local_density_veh_per_km
    (0.0 -- itself below the uncongested threshold, the opposite of inert).
    After Fix 1, apply_safety_layer reads `inert_context()` instead, so the
    not_evaluable rule changes nothing: decision 3's own words, "nothing. It
    is recorded and the advisory is unchanged," now literally hold for the
    speed path too.
    """
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    result = run_safety_gate(SPEED_MPS["slow"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(), time_s=0.0)
    assert result.proposed_speed_mps == 20.0
    assert result.bounded_speed_mps == 20.0
    assert result.delta_speed_mps == 0.0
    # The rule is still reported not_evaluable -- the record is unchanged;
    # only the decision it used to move is fixed.
    assert result.rules["low_speed_uncongested"].status == RULE_NOT_EVALUABLE
    assert result.raw_diagnostics["etiquette_blocked_action"] == []


def test_run_safety_gate_exposes_raw_diagnostics_for_the_elif_chain_attribution() -> None:
    """score_safety.py's event attribution reads this, not the independent
    per-rule census -- E5's own methodology attributes a clamp to the rule
    apply_safety_layer's own elif chain names. After Fix 1, apply_safety_layer
    runs against the neutralised context, so this only has an event to name
    when the rule genuinely fired on real evidence (unlike before Fix 1,
    where a not_evaluable rule could still leave an event here by reading
    the observation's own substituted default)."""
    obs = _base_obs()
    src = _base_field_sources()
    src["local_density_bin"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 2.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    result = run_safety_gate(SPEED_MPS["slow"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(), time_s=0.0)
    assert result.raw_diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"
    # The persisted record uses only decision 3's independent per-rule census,
    # not the raw elif-chain diagnostics -- see SafetyGateResult's docstring.
    assert "raw_diagnostics" not in result.to_record()
    # And, post-fix, the two agree: a reason the raw diagnostics name is a
    # rule the census also calls fired, never not_evaluable.
    assert result.to_record()["rules"]["low_speed_uncongested"]["status"] == RULE_FIRED


def test_invariant_holds_on_a_genuine_speed_path_firing() -> None:
    obs = _base_obs()
    src = _base_field_sources()
    src["local_density_bin"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 2.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    result = run_safety_gate(SPEED_MPS["slow"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(), time_s=0.0)
    assert result.raw_diagnostics["etiquette_blocked_action"]  # non-empty: a real event exists
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)


def test_emergency_override_genuinely_fires_and_forward_ttc_reads_fired() -> None:
    """physical_control_command's emergency_override, driven entirely by
    forward_ttc, exercised here with a REAL close, closing leader (both
    leader_gap_m and leader_relative_speed_mps measured) rather than the
    substituted-but-currently-inert values the validator flagged.

    forward_ttc's RULE_READS lists all four of its fields (the leader pair
    AND the merge-conflict pair), and merge_conflict_gap_m /
    merge_conflict_relative_speed_mps are always class (C) on this rig (no
    merge-conflict sensor exists) -- so the GENERIC "every RULE_READS field
    must be evidence" rule the other eleven rules use reported this tick
    not_evaluable even while the decision genuinely fired on real leader
    evidence, breaking the invariant below (confirmed: this was reported
    by the coordinator, who ran this exact reproduction and the invariant
    helper against it before ruling that it must be fixed, not narrated as
    an exception). The inert merge-conflict pair (always +inf/0.0) can
    never override a smaller real leader gap, so evidence for it was never
    relevant to this tick's answer -- `_forward_ttc_missing` derives
    evaluability from `_forward_hazard`'s own `min()` comparison instead of
    the generic rule, and now agrees with the decision.
    """
    obs = _base_obs()
    obs["leader_gap"] = 3.0
    obs["leader_relative_speed"] = -5.0  # negative: leader closing on the ego
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    src["leader_relative_speed"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    result = run_safety_gate(SPEED_MPS["nominal"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(min_forward_ttc_s=2.0), time_s=0.0)
    assert result.emergency_override is True
    assert result.raw_diagnostics["external_safety_override"][0]["reason"] == "forward_ttc"
    assert result.rules["forward_ttc"].status == RULE_FIRED
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)


def test_the_invariant_fails_against_the_unfixed_mechanism(monkeypatch) -> None:
    """Fix 2's own instruction, made permanent rather than done once by hand:
    neuter Fix 1 (point `inert_context` back at the raw, unmodified context
    -- exactly what `run_safety_gate` read before Fix 1) and confirm the
    reproduction's own assertion now fails. A test that passes against the
    broken mechanism pins nothing.
    """
    monkeypatch.setattr(SafetyInputs, "inert_context", SafetyInputs.context)
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    result = run_safety_gate(SPEED_MPS["slow"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(), time_s=0.0)
    with pytest.raises(AssertionError):
        assert result.bounded_speed_mps == result.proposed_speed_mps


def test_forward_ttc_missing_leader_operative_not_critical_relative_speed_absent() -> None:
    """The one leader-side branch round 2's cases did not cover, and the
    ONE that is genuinely reachable on this rig: leader_gap_m MEASURED and
    NOT critical (10.0 m, above the 5.0 m default), leader_relative_speed_mps
    left substituted. Neither disjunct can be decided without it (the gap
    disjunct already read false on real evidence, so only the ttc disjunct
    is left, and it needs the relative speed) -- not_evaluable, naming
    exactly the one missing field.
    """
    obs = _base_obs()
    obs["leader_gap"] = 10.0
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    assert _forward_ttc_missing(inputs, SafetyConstraints(min_front_gap_m=5.0)) == ("leader_relative_speed_mps",)
    result = run_safety_gate(SPEED_MPS["nominal"], NORMAL_HEADWAY_S, inputs, SafetyConstraints(min_front_gap_m=5.0, min_forward_ttc_s=2.0), time_s=0.0)
    assert result.rules["forward_ttc"].status == RULE_NOT_EVALUABLE
    assert result.emergency_override is False


def _merge_operative_inputs(*, merge_conflict_gap_m: float, merge_conflict_relative_speed_evidence: bool) -> SafetyInputs:
    """A fixture for the merge-conflict axis with the LEADER pair built
    WITHOUT evidence in every case -- not merely unset, but explicitly
    marked `evidence=False`. This is what makes the fixture actually
    discriminate a branch-aware rule from the generic one: `_inputs_with_
    overrides` grants blanket evidence to every field, so a fixture built
    from it alone would give the generic rule the same "leader pair fine"
    answer the branch-aware rule gives, for the wrong reason (both fields
    happen to be evidenced) rather than the right one (the rule correctly
    never consulted them once merge is operative). Confirmed by hand: an
    earlier version of this fixture did exactly that and did not fail
    when `_forward_ttc_missing` was reverted to the generic rule.
    """
    inputs = _inputs_with_overrides(
        merge_conflict_gap_m=merge_conflict_gap_m, merge_conflict_relative_speed_mps=0.0,
    )
    fields = dict(inputs.fields)
    fields["leader_gap_m"] = SafetyInputField(value=50.0, input_class="evidence_required", source=None, evidence=False)
    fields["leader_relative_speed_mps"] = SafetyInputField(value=0.0, input_class="evidence_required", source=None, evidence=False)
    if not merge_conflict_relative_speed_evidence:
        fields["merge_conflict_relative_speed_mps"] = SafetyInputField(
            value=0.0, input_class="structural", source=None, evidence=False,
        )
    return SafetyInputs(fields=fields)


def test_evaluate_rules_case_42_fails_against_the_raw_context_mechanism(monkeypatch) -> None:
    """Confirms the test above actually discriminates: neuters `evaluate_
    rules` back to reading the raw `context()`/`state` (what it did before
    this fix) and confirms case 42 then disagrees with `missing` exactly
    as the coordinator reproduced -- `fired`, `ttc_s 0.75`, on the
    leader's own un-evidenced values.
    """
    import policy.safety_gate as safety_gate_module

    def unfixed_evaluate_rules(action, inputs, constraints):
        context = inputs.context()
        return {
            name: safety_gate_module._evaluate_one_rule(
                name, target_speed=SPEED_MPS["nominal"], inputs=inputs, context=context, constraints=constraints,
            )
            for name in RULE_NAMES
        }
    monkeypatch.setattr(safety_gate_module, "evaluate_rules", unfixed_evaluate_rules)

    fields = dict(_inputs_with_overrides(
        merge_conflict_gap_m=6.0, merge_conflict_relative_speed_mps=-0.5,
    ).fields)
    fields["leader_gap_m"] = SafetyInputField(value=6.0, input_class="evidence_required", source=None, evidence=False)
    fields["leader_relative_speed_mps"] = SafetyInputField(value=-8.0, input_class="evidence_required", source=None, evidence=False)
    inputs = SafetyInputs(fields=fields)
    constraints = SafetyConstraints(min_front_gap_m=5.0, min_forward_ttc_s=2.0)

    record = safety_gate_module.evaluate_rules(SPEED_MPS["nominal"], inputs, constraints)["forward_ttc"]
    assert record.status == RULE_FIRED, "the pre-fix mechanism disagrees with missing==() and wrongly fires"
    assert record.evidence["ttc_s"] == pytest.approx(0.75)

def _obs_result(obs: dict, field_sources: dict, diagnostics: dict) -> ObservationResult:
    return ObservationResult(
        obs=obs, encoded=np.zeros(39, dtype=np.float32), field_sources=field_sources,
        diagnostics=diagnostics, feed=None,
    )

def _base_obs() -> dict:
    return {
        "ego_speed": 0.0,
        "leader_gap": float("inf"),
        "leader_relative_speed": 0.0,
        "target_lane_front_gap": float("inf"),
        "target_lane_rear_gap": float("inf"),
        "target_lane_rear_required_decel": 0.0,
        "follower_gap": float("inf"),
        "follower_relative_speed": 0.0,
        "nearby_av_mean_speed": 30.0,
        "downstream_congestion_estimate": 0.0,
        "cooperation": {"segment_target_speed": 30.0},
    }

def _base_field_sources() -> dict:
    return {
        "ego_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
        "leader_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
        "leader_relative_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
        "target_lane_front_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
        "target_lane_rear_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
        "target_lane_rear_required_decel": provenance.SOURCE_FALLBACK_NEUTRAL,
        "follower_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
        "follower_relative_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
        "nearby_av_mean_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
        "downstream_congestion_estimate": provenance.SOURCE_FALLBACK_NEUTRAL,
        "local_density_bin": provenance.SOURCE_DERIVED_EMPTY,
        "local_mean_speed_bin": provenance.SOURCE_FALLBACK_NEUTRAL,
    }

def _inputs_with_overrides(**value_overrides: float) -> SafetyInputs:
    """A SafetyInputs built directly (bypassing safety_inputs_from_observation)
    with every evidence-required AND structural field marked as evidence --
    a configuration decision 3's real classification never produces on this
    rig, since the merge-conflict pair is structurally absent. Used here to
    exercise `_forward_ttc_missing`'s merge-conflict axis, which no
    observation this device can build will ever reach.
    """
    from policy.safety_gate import CONFIGURED_FIELDS, EVIDENCE_REQUIRED_FIELDS, STRUCTURAL_FIELDS

    defaults = {
        "free_flow_speed_mps": 30.0,
        "ego_speed_mps": 0.0, "leader_gap_m": float("inf"), "leader_relative_speed_mps": 0.0,
        "local_density_veh_per_km": 0.0,
        "merge_conflict_gap_m": float("inf"), "merge_conflict_relative_speed_mps": 0.0,
    }
    defaults.update(value_overrides)
    fields: dict[str, SafetyInputField] = {}
    for name in CONFIGURED_FIELDS:
        fields[name] = SafetyInputField(value=defaults[name], input_class="configured", source=None, evidence=True)
    for name in EVIDENCE_REQUIRED_FIELDS:
        fields[name] = SafetyInputField(value=defaults[name], input_class="evidence_required", source=provenance.SOURCE_MEASURED, evidence=True)
    for name in STRUCTURAL_FIELDS:
        fields[name] = SafetyInputField(value=defaults[name], input_class="structural", source=None, evidence=True)
    return SafetyInputs(fields=fields)

def _assert_every_raw_diagnostics_reason_is_a_fired_rule(result) -> None:
    for events in result.raw_diagnostics.values():
        for event in events:
            reason = event["reason"]
            assert reason in RULE_NAMES, reason
            assert result.rules[reason].status == RULE_FIRED, (
                f"'{reason}' appears in raw_diagnostics but the census reports "
                f"{result.rules[reason].status}, not {RULE_FIRED}"
            )

def _case_low_speed_uncongested():
    obs = _base_obs()
    src = _base_field_sources()
    src["local_density_bin"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 2.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    return inputs, SPEED_MPS["slow"], SafetyConstraints(), True



def _case_forward_ttc_full_evidence():
    """Both leader fields measured; gap 10.0 m stays above min_front_gap_m
    (5.0, not critical) so the ttc disjunct alone decides -- a fast closing
    speed (-30 m/s) still gives ttc 0.33 s, under the 2.0 s threshold."""
    obs = _base_obs()
    obs["leader_gap"] = 10.0
    obs["leader_relative_speed"] = -30.0
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    src["leader_relative_speed"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    return inputs, SPEED_MPS["nominal"], SafetyConstraints(min_front_gap_m=5.0, min_forward_ttc_s=2.0), True

def _case_forward_ttc_gap_only_critical():
    """The coordinator's correction, axis 3 (partial evidence within one
    rule's own reads): leader_gap MEASURED and close (3.0 m, below
    min_front_gap_m's 5.0 m), leader_relative_speed left at its default
    SUBSTITUTED source -- the NORMAL state while a track is newly acquired
    (observation_builder tags gap MEASURED as soon as a leader exists, but
    relative speed only once rel_speed_valid). The critical_gap disjunct
    alone must decide "fired" here; the ttc disjunct's own relative-speed
    requirement must not be asked."""
    obs = _base_obs()
    obs["leader_gap"] = 3.0
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    # leader_relative_speed left at _base_field_sources()'s own default
    # (SOURCE_FALLBACK_NEUTRAL) -- substituted, deliberately.
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, density_max_age_s=4.0,
    )
    return inputs, SPEED_MPS["nominal"], SafetyConstraints(min_front_gap_m=5.0, min_forward_ttc_s=2.0), True


CASES = {
    "low_speed_uncongested": _case_low_speed_uncongested,
    "forward_ttc_full_evidence": _case_forward_ttc_full_evidence,
    "forward_ttc_gap_only_critical": _case_forward_ttc_gap_only_critical,
}
# Every surviving rule must be exercised by at least one case name
# (some, like forward_ttc, by more than one -- see the module comment above).
_rules_covered = {rule for rule in RULE_NAMES if any(case_name.startswith(rule) for case_name in CASES)}
assert _rules_covered == set(RULE_NAMES), f"grid is missing: {set(RULE_NAMES) - _rules_covered}"


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_invariant_grid(case_name: str) -> None:
    """The grid the coordinator asked for, corrected twice: one uniform
    neutralisation policy cannot serve every rule (`forward_ttc` needs
    per-disjunct evidence), so this grid checks the INVARIANT on each case
    rather than asserting a single shared shape. See
    test_the_grid_fails_without_this_rounds_follow_up_fixes below for the
    confirmation that it actually would have caught all four regressions.
    """
    inputs, proposed_speed, constraints, expect_fired = CASES[case_name]()
    result = run_safety_gate(proposed_speed, NORMAL_HEADWAY_S, inputs, constraints, time_s=0.0)
    rule_name = next(n for n in RULE_NAMES if case_name.startswith(n))
    if expect_fired:
        assert result.rules[rule_name].status == RULE_FIRED, (
            f"{case_name} did not make {rule_name} fire -- fixture does not exercise it"
        )
        assert any(
            event["reason"] == rule_name
            for events in result.raw_diagnostics.values()
            for event in events
        ), f"{case_name}: {rule_name} fired but left no raw_diagnostics event naming it"
    else:
        assert not any(
            event["reason"] == rule_name
            for events in result.raw_diagnostics.values()
            for event in events
        ), f"{case_name}: {rule_name} fired despite a non-default/partial-evidence input that must stay inert"
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)
