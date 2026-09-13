from __future__ import annotations

import pytest

from src.envs.wrappers import decode_headway_bin
from src.safety import SafetyConstraints, SafetyContext, SafetyState, apply_safety_layer


def action(**overrides: str) -> dict[str, str]:
    value = {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
    value.update(overrides)
    return value








def test_low_speed_uncongested_is_lifted_and_diagnosed() -> None:
    decision = apply_safety_layer(
        action(desired_speed_bin="slow"),
        SafetyState(),
        SafetyContext(time_s=0.0, free_flow_speed_mps=30.0, local_density_veh_per_km=2.0),
    )
    assert decision.target_speed_mps >= 22.0
    assert decision.diagnostics["etiquette_blocked_action"][0]["reason"] == "low_speed_uncongested"



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



