from __future__ import annotations

from dataclasses import dataclass


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
