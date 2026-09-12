"""Unit and sanity tests for policy.safety_gate.

Two layers: (1) the vendored apply_safety_layer/physical_control_command,
proven behaviorally identical to src/safety/safety_layer.py by re-running
tests/test_safety_layer.py's own cases against this module (field/constant
identity is already the golden file's job -- this is logic identity); and
(2) SafetyInputs, evaluate_rules and run_safety_gate, which have no
src/safety/ counterpart at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.observation_builder import ObservationResult
from perception import provenance
from policy.safety_gate import (
    RULE_NOT_EVALUABLE,
    RULE_FIRED,
    RULE_QUIET,
    LANE_GUARD_RULES,
    RULE_NAMES,
    SafetyConstraints,
    SafetyContext,
    SafetyInputField,
    SafetyInputs,
    SafetyState,
    apply_safety_layer,
    evaluate_rules,
    run_safety_gate,
    safety_inputs_from_observation,
)
from policy.sim_contract import decode_headway_bin


def action(**overrides: str) -> dict[str, str]:
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


def test_lane_change_dwell_blocks_lateral_action() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_left_if_safe"),
        SafetyState(last_lane_change_time_s=5.0),
        SafetyContext(time_s=10.0),
    )
    assert decision.lane_action is None
    assert decision.diagnostics["safety_masked_action"][0]["reason"] == "lane_change_dwell"


def test_max_lane_changes_per_km_blocks_lateral_action() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_right_if_safe"),
        SafetyState(lane_changes_last_km=2, distance_since_window_start_m=1000.0),
        SafetyContext(time_s=30.0),
    )
    assert decision.lane_action is None
    assert decision.diagnostics["safety_masked_action"][0]["reason"] == "lane_changes_per_km"


def test_unsafe_rear_gap_blocks_follower_disruption() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_left_if_safe"),
        SafetyState(),
        SafetyContext(time_s=30.0, target_lane_rear_required_decel_mps2=3.0),
    )
    assert decision.lane_action is None
    assert decision.diagnostics["follower_disruption_blocked"][0]["reason"] == "target_lane_rear_braking"


def test_low_speed_uncongested_is_lifted_and_diagnosed() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="slow"),
        SafetyState(),
        SafetyContext(time_s=0.0, free_flow_speed_mps=30.0, local_density_veh_per_km=2.0),
    )
    assert decision.target_speed_mps >= 22.0
    assert decision.diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"


def test_merge_create_gap_increases_headway_without_lane_blocking() -> None:
    decision = apply_safety_layer(
        action(merge_mode="create_gap"),
        SafetyState(),
        SafetyContext(time_s=0.0, near_merge=True),
        SafetyConstraints(),
    )
    assert decision.target_headway_s > decode_headway_bin("normal")
    assert decision.lane_action is None


def test_speed_control_acceleration_is_bounded() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="fast"),
        SafetyState(),
        SafetyContext(time_s=0.0, ego_speed_mps=10.0, free_flow_speed_mps=30.0),
        SafetyConstraints(max_accel_mps2=1.5),
    )
    assert decision.acceleration_mps2 == 1.5
    assert decision.emergency_override is False


def test_low_forward_ttc_triggers_emergency_override() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="fast"),
        SafetyState(),
        SafetyContext(
            time_s=0.0, ego_speed_mps=25.0, leader_gap_m=10.0, leader_relative_speed_mps=-10.0,
        ),
        SafetyConstraints(emergency_decel_mps2=7.0, min_forward_ttc_s=2.0),
    )
    assert decision.acceleration_mps2 == -7.0
    assert decision.emergency_override is True
    assert decision.diagnostics["external_safety_override"][0]["reason"] == "forward_ttc"
    assert decision.penalty_terms["emergency_override"] == 1.0


def test_unsafe_target_lane_front_gap_blocks_lane_preference() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_left_if_safe"),
        SafetyState(),
        SafetyContext(time_s=30.0, target_lane_front_gap_m=3.0),
        SafetyConstraints(min_front_gap_m=5.0),
    )
    assert decision.lane_action is None
    assert decision.diagnostics["safety_masked_action"][0]["reason"] == "target_lane_front_gap"


def test_an_empty_merge_conflict_changes_nothing() -> None:
    with_merge = apply_safety_layer(
        action(), SafetyState(), SafetyContext(time_s=10.0, ego_speed_mps=27.0, merge_conflict_gap_m=float("inf")),
    )
    without = apply_safety_layer(action(), SafetyState(), SafetyContext(time_s=10.0, ego_speed_mps=27.0))
    assert with_merge.acceleration_mps2 == pytest.approx(without.acceleration_mps2)


# ---------------------------------------------------------------------------
# (2) SafetyInputs: decision 3's per-field input-class partition.
# ---------------------------------------------------------------------------


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


def test_class_a_configured_field_is_always_evidence_even_when_fallback() -> None:
    """free_flow_speed_mps is class (A): decision 3 says it never blocks a
    rule, regardless of the observation's own provenance for the field it is
    read from."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    field = inputs.fields["free_flow_speed_mps"]
    assert field.input_class == "configured"
    assert field.evidence is True


def test_class_b_evidence_required_field_is_not_evidence_when_substituted() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
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
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    field = inputs.fields["leader_gap_m"]
    assert field.evidence is True
    assert field.value == 42.0


def test_class_c_structural_field_is_never_evidence() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    field = inputs.fields["target_lane_rear_gap_m"]
    assert field.input_class == "structural"
    assert field.evidence is False


def test_density_derived_empty_is_not_evidence_without_a_detection_age() -> None:
    """The camera has never produced a track (last_detection_age_s is None):
    a derived_empty density is a blind camera, not an empty road."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is False


def test_density_derived_empty_is_evidence_within_the_detection_age_bound() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": 1.0})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is True


def test_density_derived_empty_is_not_evidence_past_the_detection_age_bound() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": 10.0})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    assert inputs.fields["local_density_veh_per_km"].evidence is False


# ---------------------------------------------------------------------------
# (3) evaluate_rules / run_safety_gate.
# ---------------------------------------------------------------------------


def test_all_twelve_rules_not_evaluable_on_the_recorded_corpus_shape() -> None:
    """The plan's headline finding, reproduced from a synthetic tick shaped
    like the actual outputs/task42_usb corpus: every rule not_evaluable
    except target_lane_front_gap, which needs its own field measured."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    rules = evaluate_rules(action(), inputs, SafetyState(), SafetyConstraints())
    assert set(rules) == set(RULE_NAMES)
    for name, rule in rules.items():
        assert rule.status == RULE_NOT_EVALUABLE, name


def test_target_lane_front_gap_is_evaluable_once_its_own_field_is_derived() -> None:
    obs = _base_obs()
    src = _base_field_sources()
    src["target_lane_front_gap"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    rules = evaluate_rules(action(lane_preference="prefer_left_if_safe"), inputs, SafetyState(), SafetyConstraints())
    assert rules["target_lane_front_gap"].status == RULE_QUIET  # inf gap, no violation
    # its neighbouring rule still needs target_lane_front_relative_speed_mps, always structural
    assert rules["target_lane_front_ttc"].status == RULE_NOT_EVALUABLE


def test_evaluate_rules_fires_on_concrete_evidence() -> None:
    obs = _base_obs()
    obs["target_lane_front_gap"] = 3.0
    src = _base_field_sources()
    src["target_lane_front_gap"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    rules = evaluate_rules(action(lane_preference="prefer_left_if_safe"), inputs, SafetyState(), SafetyConstraints(min_front_gap_m=5.0))
    assert rules["target_lane_front_gap"].status == RULE_FIRED
    assert rules["target_lane_front_gap"].evidence["gap_m"] == 3.0


def test_run_safety_gate_withholds_lane_action_when_a_guard_is_not_evaluable() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(lane_preference="prefer_left_if_safe"), inputs, SafetyState(), SafetyConstraints(),
        withhold_lane_when_not_evaluable=True,
    )
    assert result.proposed_lane_action == "LANE_LEFT"
    assert result.bounded_lane_action is None
    assert result.lane_withheld == RULE_NOT_EVALUABLE


def test_run_safety_gate_flag_off_restores_the_ungated_lane_action() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(lane_preference="prefer_left_if_safe"), inputs, SafetyState(), SafetyConstraints(),
        withhold_lane_when_not_evaluable=False,
    )
    assert result.bounded_lane_action == "LANE_LEFT"
    assert result.lane_withheld is None


def test_run_safety_gate_records_no_withholding_when_lane_preference_is_keep() -> None:
    """Nothing to withhold: the lane action was already None before the gate,
    so lane_withheld must stay None rather than falsely claiming a mask."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(lane_preference="keep"), inputs, SafetyState(), SafetyConstraints())
    assert result.bounded_lane_action is None
    assert result.lane_withheld is None


def test_low_speed_uncongested_reduces_to_the_speed_bin_alone() -> None:
    """The plan's algebraic-reduction finding, at the `apply_safety_layer`
    unit level: at the configured free-flow speed of 30.0, the predicate
    reduces to desired_speed_bin == 'slow', and forcing it clamps and RAISES
    the recommended speed -- given a context where the density really is
    evidence (this is `test_low_speed_uncongested_is_lifted_and_diagnosed`
    above, restated against the free-flow speed this rig actually runs)."""
    decision = apply_safety_layer(
        action(desired_speed_bin="slow"),
        SafetyState(),
        SafetyContext(time_s=0.0, free_flow_speed_mps=30.0, local_density_veh_per_km=2.0),
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
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(desired_speed_bin="slow"), inputs, SafetyState(), SafetyConstraints())
    assert result.proposed_speed_mps == 20.0
    assert result.bounded_speed_mps == 20.0
    assert result.delta_speed_mps == 0.0
    # The rule is still reported not_evaluable -- the record is unchanged;
    # only the decision it used to move is fixed.
    assert result.rules["low_speed_uncongested"].status == RULE_NOT_EVALUABLE
    assert result.raw_diagnostics["etiquette_blocked_action"] == []


def test_a_genuine_low_density_still_clamps_after_the_fix() -> None:
    """The other half of Fix 1: neutralising NOT-evidence fields must not
    also neutralise a real one. With local_density_veh_per_km measured
    (source `derived`, not `derived_empty` or substituted) at 2.0 veh/km,
    `inert_context()` leaves it untouched and the clamp still fires exactly
    as `test_low_speed_uncongested_reduces_to_the_speed_bin_alone` shows at
    the apply_safety_layer level."""
    obs = _base_obs()
    src = _base_field_sources()
    src["local_density_bin"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 2.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    assert inputs.is_evidence("local_density_veh_per_km") is True
    result = run_safety_gate(action(desired_speed_bin="slow"), inputs, SafetyState(), SafetyConstraints())
    assert result.rules["low_speed_uncongested"].status == RULE_FIRED
    assert result.bounded_speed_mps == 22.0
    assert result.delta_speed_mps == pytest.approx(2.0)


def _inputs_with_overrides(**value_overrides: float) -> SafetyInputs:
    """A SafetyInputs built directly (bypassing safety_inputs_from_observation)
    with every evidence-required and structural field marked as evidence --
    a configuration decision 3's real classification never produces on this
    rig (only target_lane_front_gap_m can ever be evidence among the lane
    guards), used here purely to exercise the chain-order invariant itself,
    independent of what this device can actually measure.
    """
    from policy.safety_gate import CONFIGURED_FIELDS, EVIDENCE_REQUIRED_FIELDS, STRUCTURAL_FIELDS

    defaults = {
        "time_s": 30.0, "free_flow_speed_mps": 30.0, "min_contextual_speed_mps": 12.0,
        "ego_speed_mps": 0.0, "leader_gap_m": float("inf"), "leader_relative_speed_mps": 0.0,
        "local_density_veh_per_km": 0.0, "local_mean_speed_mps": 30.0, "target_lane_front_gap_m": float("inf"),
        "follower_gap_m": float("inf"), "follower_relative_speed_mps": 0.0,
        "target_lane_rear_gap_m": float("inf"), "target_lane_rear_relative_speed_mps": 0.0,
        "target_lane_rear_required_decel_mps2": 0.0, "target_lane_exists": True,
        "in_passing_lane": False, "target_lane_front_relative_speed_mps": 0.0,
        "all_lanes_av_occupied": False, "av_mean_speed_mps": 30.0, "downstream_congested": False,
        "near_merge": False, "merge_conflict_gap_m": float("inf"), "merge_conflict_relative_speed_mps": 0.0,
        "last_lane_change_time_s": None, "lane_changes_last_km": 0, "lane_change_distances_m": [],
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
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(desired_speed_bin="slow"), inputs, SafetyState(), SafetyConstraints())
    assert result.raw_diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"
    # The persisted record uses only decision 3's independent per-rule census,
    # not the raw elif-chain diagnostics -- see SafetyGateResult's docstring.
    assert "raw_diagnostics" not in result.to_record()
    # And, post-fix, the two agree: a reason the raw diagnostics name is a
    # rule the census also calls fired, never not_evaluable.
    assert result.to_record()["rules"]["low_speed_uncongested"]["status"] == RULE_FIRED


def test_chain_order_invariant_first_fired_lane_guard_matches_the_decision() -> None:
    """decision 3's own check (step 9): when guards ARE evaluable, the first
    one that fires in chain order is the one apply_safety_layer's elif chain
    actually names -- the independent per-rule evaluation must not disagree
    with the short-circuited one about WHICH reason applies. Exercised with
    every guard marked evidence (see _inputs_with_overrides), since decision
    3's real classification allows only one lane guard to ever be evaluable.
    """
    context = SafetyContext(
        time_s=30.0,
        target_lane_front_gap_m=3.0,   # would fire target_lane_front_gap...
        target_lane_rear_gap_m=1.0,    # ...and target_lane_rear_gap, later in chain order
    )
    state = SafetyState()
    constraints = SafetyConstraints(min_front_gap_m=5.0, min_rear_gap_m=5.0)
    decision = apply_safety_layer(action(lane_preference="prefer_left_if_safe"), state, context, constraints)
    assert decision.diagnostics["safety_masked_action"][0]["reason"] == "target_lane_front_gap"

    inputs = _inputs_with_overrides(target_lane_front_gap_m=3.0, target_lane_rear_gap_m=1.0)
    rules = evaluate_rules(action(lane_preference="prefer_left_if_safe"), inputs, state, constraints)
    fired_in_chain_order = [name for name in LANE_GUARD_RULES if rules[name].status == RULE_FIRED]
    assert fired_in_chain_order[0] == "target_lane_front_gap" == decision.diagnostics["safety_masked_action"][0]["reason"]


# ---------------------------------------------------------------------------
# validator round 1, Fix 2: the invariant that proves Fix 1 -- every reason
# apply_safety_layer's own (neutralised-context) diagnostics name is a rule
# the independent census calls RULE_FIRED, never RULE_NOT_EVALUABLE or
# RULE_QUIET. Covers the speed path, the lane path and emergency_override at
# once, on genuine firings so the check is not vacuous.
# ---------------------------------------------------------------------------


def _assert_every_raw_diagnostics_reason_is_a_fired_rule(result) -> None:
    for events in result.raw_diagnostics.values():
        for event in events:
            reason = event["reason"]
            assert reason in RULE_NAMES, reason
            assert result.rules[reason].status == RULE_FIRED, (
                f"'{reason}' appears in raw_diagnostics but the census reports "
                f"{result.rules[reason].status}, not {RULE_FIRED}"
            )


def test_invariant_holds_vacuously_when_nothing_is_evidence() -> None:
    """No reason is ever recorded for a rule the census could not evaluate --
    checked directly on the reproduction fixture, where every one of the
    twelve rules is not_evaluable and raw_diagnostics must therefore be
    empty everywhere, not merely consistent with an empty census by luck."""
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(desired_speed_bin="slow", lane_preference="prefer_left_if_safe"),
        inputs, SafetyState(), SafetyConstraints(),
    )
    assert all(events == [] for events in result.raw_diagnostics.values())
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)  # vacuous, but must not raise


def test_invariant_holds_on_a_genuine_speed_path_firing() -> None:
    obs = _base_obs()
    src = _base_field_sources()
    src["local_density_bin"] = provenance.SOURCE_DERIVED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 2.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(desired_speed_bin="slow"), inputs, SafetyState(), SafetyConstraints())
    assert result.raw_diagnostics["etiquette_blocked_action"]  # non-empty: a real event exists
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)


def test_invariant_holds_on_a_genuine_lane_path_firing() -> None:
    """target_lane_front_gap masks the lane action through apply_safety_layer's
    own elif chain, not through withholding -- every lane guard is marked
    evidence (_inputs_with_overrides), so lane_withheld stays None and the
    masking is apply_safety_layer's own, exactly what the invariant must
    cover for the lane path."""
    constraints = SafetyConstraints(min_front_gap_m=5.0)
    inputs = _inputs_with_overrides(target_lane_front_gap_m=3.0)
    result = run_safety_gate(
        action(lane_preference="prefer_left_if_safe"), inputs, SafetyState(), constraints,
    )
    assert result.lane_withheld is None
    assert result.bounded_lane_action is None
    assert result.raw_diagnostics["safety_masked_action"][0]["reason"] == "target_lane_front_gap"
    _assert_every_raw_diagnostics_reason_is_a_fired_rule(result)


def test_emergency_override_can_genuinely_fire_while_forward_ttc_reads_not_evaluable() -> None:
    """physical_control_command's emergency_override, driven entirely by
    forward_ttc, exercised here with a REAL close, closing leader (both
    leader_gap_m and leader_relative_speed_mps measured) rather than the
    substituted-but-currently-inert values the validator flagged.

    This is a DOCUMENTED, NARROWER exception to the general invariant below
    -- found while writing this test, not assumed: forward_ttc's RULE_READS
    requires all four of its fields as evidence (the leader pair AND the
    merge-conflict pair) to be evaluable at all, but merge_conflict_gap_m /
    merge_conflict_relative_speed_mps are always class (C) on this rig (no
    merge-conflict sensor exists), so forward_ttc's census reads
    not_evaluable on every tick regardless of the leader pair. The decision
    is still CORRECT here -- it fired on real leader evidence, and the
    inert merge-conflict pair (always +inf/0.0) can never override a
    smaller real leader gap -- so this is the census criterion being
    stricter than the decision needs, not the decision using unevidenced
    data. Left as found and reported rather than silently loosened:
    changing which fields forward_ttc's evaluability depends on is a
    decision-3-level call, not a validator-fix-pass one.
    """
    obs = _base_obs()
    obs["leader_gap"] = 3.0
    obs["leader_relative_speed"] = -5.0  # negative: leader closing on the ego
    src = _base_field_sources()
    src["leader_gap"] = provenance.SOURCE_MEASURED
    src["leader_relative_speed"] = provenance.SOURCE_MEASURED
    obs_result = _obs_result(obs, src, {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(), inputs, SafetyState(), SafetyConstraints(min_forward_ttc_s=2.0))
    assert result.emergency_override is True
    assert result.raw_diagnostics["external_safety_override"][0]["reason"] == "forward_ttc"
    assert result.rules["forward_ttc"].status == RULE_NOT_EVALUABLE


def test_contrapositive_all_not_evaluable_speed_lane_and_override_are_untouched() -> None:
    """validator round 1, Fix 2's legible contrapositive: with every one of
    the twelve rules not_evaluable, the gate changes nothing except by the
    one explicit, principled path -- withholding the lane action -- never by
    a masked elif branch silently agreeing with the withdrawal for its own
    (wrong) reason.
    """
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(desired_speed_bin="slow", lane_preference="prefer_left_if_safe"),
        inputs, SafetyState(), SafetyConstraints(),
    )
    assert all(r.status == RULE_NOT_EVALUABLE for r in result.rules.values())
    assert result.bounded_speed_mps == result.proposed_speed_mps
    assert result.delta_speed_mps == 0.0
    # None BY WITHHOLDING: lane_withheld names the reason...
    assert result.bounded_lane_action is None
    assert result.lane_withheld == RULE_NOT_EVALUABLE
    # ...not by masking: apply_safety_layer's own elif chain (run against the
    # neutralised context) must have found no reason of its own to null the
    # lane action, or an event would appear here.
    assert result.raw_diagnostics["safety_masked_action"] == []
    assert result.raw_diagnostics["follower_disruption_blocked"] == []
    assert result.raw_diagnostics["etiquette_blocked_action"] == []
    assert result.emergency_override is False


def test_headway_invariant_the_merge_bonus_is_the_one_unconditional_transformation() -> None:
    """Fix 2's named carve-out: `apply_safety_layer` applies the create_gap
    merge headway bonus before any of the twelve rules run, so it is not
    gated by RULE_FIRED the way target_speed_mps, lane_action and
    emergency_override are. Read end to end (module docstring / code review),
    it is the ONLY unconditional transformation apply_safety_layer makes --
    every other value it returns is either a plain decode of the action bin
    (no rule involved, nothing to gate) or moves only inside one of the
    twelve rules' own `if`/`elif` bodies.
    """
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    constraints = SafetyConstraints()

    normal = run_safety_gate(action(merge_mode="normal"), inputs, SafetyState(), constraints)
    assert not any(r.status == RULE_FIRED for r in normal.rules.values())
    assert normal.bounded_headway_s == pytest.approx(normal.proposed_headway_s)

    create_gap = run_safety_gate(action(merge_mode="create_gap"), inputs, SafetyState(), constraints)
    assert not any(r.status == RULE_FIRED for r in create_gap.rules.values())
    assert create_gap.bounded_headway_s == pytest.approx(
        create_gap.proposed_headway_s + constraints.merge_gap_headway_bonus_s
    )


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
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(action(desired_speed_bin="slow"), inputs, SafetyState(), SafetyConstraints())
    with pytest.raises(AssertionError):
        assert result.bounded_speed_mps == result.proposed_speed_mps


def test_safety_enabled_false_bounded_equals_proposed_and_lane_is_not_withheld() -> None:
    """validator round 1, Fix 3: `enabled=False` is the actual rollback --
    the census and full record are still produced, but bounded_* equals
    proposed_* exactly (including headway: even the unconditional
    create_gap bonus does not apply, since that bonus is
    apply_safety_layer's own first step and apply_safety_layer does not run
    at all) and the lane action is never withheld.
    """
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=1.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(desired_speed_bin="slow", lane_preference="prefer_left_if_safe", merge_mode="create_gap"),
        inputs, SafetyState(), SafetyConstraints(), enabled=False,
    )
    assert result.bounded_speed_mps == result.proposed_speed_mps
    assert result.bounded_headway_s == result.proposed_headway_s
    assert result.bounded_lane_action == result.proposed_lane_action
    assert result.lane_withheld is None
    assert result.emergency_override is False
    # The census still ran and is still in the record.
    assert len(result.rules) == 12
    assert result.to_record()["config"]["enabled"] is False


def test_safety_gate_config_is_recorded() -> None:
    obs_result = _obs_result(_base_obs(), _base_field_sources(), {"density_veh_per_km": 0.0, "last_detection_age_s": None})
    inputs = safety_inputs_from_observation(
        obs_result, SafetyState(), time_s=42.0, min_contextual_speed_mps=12.0, density_max_age_s=4.0,
    )
    result = run_safety_gate(
        action(), inputs, SafetyState(), SafetyConstraints(), withhold_lane_when_not_evaluable=False,
    )
    config = result.to_record()["config"]
    assert config == {
        "enabled": True,
        "withhold_lane_when_not_evaluable": False,
        "time_s": 42.0,
        "min_contextual_speed_mps": 12.0,
        "density_max_age_s": 4.0,
    }
