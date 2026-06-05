from __future__ import annotations

from src.envs.wrappers import decode_headway_bin
from src.safety import SafetyConstraints, SafetyContext, SafetyState, apply_safety_layer, safety_action_mask


def action(**overrides: str) -> dict[str, str]:
    value = {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
    value.update(overrides)
    return value


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


def test_lane_change_window_uses_count_not_distance_rate_spike() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_right_if_safe"),
        SafetyState(lane_changes_last_km=1, distance_since_window_start_m=1.0),
        SafetyContext(time_s=30.0),
        SafetyConstraints(max_lane_changes_per_km=2.0),
    )
    mask = safety_action_mask(
        SafetyState(lane_changes_last_km=1, distance_since_window_start_m=1.0),
        SafetyContext(time_s=30.0),
        SafetyConstraints(max_lane_changes_per_km=2.0),
    )

    assert decision.lane_action == "LANE_RIGHT"
    assert mask["lane_preference"]["prefer_right_if_safe"] is True


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


def test_short_headway_applies_bounded_deceleration() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="fast", desired_headway_bin="largest"),
        SafetyState(),
        SafetyContext(
            time_s=0.0,
            ego_speed_mps=25.0,
            free_flow_speed_mps=30.0,
            leader_gap_m=20.0,
            leader_relative_speed_mps=-5.0,
        ),
        SafetyConstraints(max_decel_mps2=2.0),
    )

    assert decision.acceleration_mps2 == -2.0
    assert decision.emergency_override is False


def test_low_forward_ttc_triggers_emergency_override() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="fast"),
        SafetyState(),
        SafetyContext(
            time_s=0.0,
            ego_speed_mps=25.0,
            leader_gap_m=10.0,
            leader_relative_speed_mps=-10.0,
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


def test_unsafe_target_lane_rear_ttc_blocks_lane_preference() -> None:
    decision = apply_safety_layer(
        action(lane_preference="prefer_right_if_safe"),
        SafetyState(),
        SafetyContext(
            time_s=30.0,
            target_lane_rear_gap_m=12.0,
            target_lane_rear_relative_speed_mps=8.0,
        ),
        SafetyConstraints(min_lane_change_ttc_s=2.0),
    )

    assert decision.lane_action is None
    assert decision.diagnostics["safety_masked_action"][0]["reason"] == "target_lane_rear_ttc"


def test_safety_action_mask_blocks_unsafe_lateral_and_slow_uncongested() -> None:
    mask = safety_action_mask(
        SafetyState(last_lane_change_time_s=9.0),
        SafetyContext(
            time_s=10.0,
            free_flow_speed_mps=30.0,
            local_density_veh_per_km=2.0,
            target_lane_front_gap_m=3.0,
        ),
        SafetyConstraints(),
    )

    assert mask["desired_speed_bin"]["slow"] is False
    assert mask["lane_preference"]["keep"] is True
    assert mask["lane_preference"]["prefer_left_if_safe"] is False
    assert mask["lane_preference"]["prefer_right_if_safe"] is False
