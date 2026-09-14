from __future__ import annotations

from src.safety.constraints import SafetyConstraints


def is_low_speed_uncongested(
    target_speed_mps: float,
    free_flow_speed_mps: float,
    density_veh_per_km: float,
    constraints: SafetyConstraints,
) -> bool:
    return (
        density_veh_per_km < constraints.uncongested_density_threshold_veh_per_km
        and target_speed_mps < free_flow_speed_mps - constraints.low_speed_free_flow_delta_mps
    )
