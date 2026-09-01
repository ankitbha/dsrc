"""Vendored simulation contract: observation encoding + action schema.

This file mirrors the parts of the simulation repo that define the
actor's input/output interface, so the edge runtime never has to import
the simulation stack (``src.rl.actions`` pulls in ``highway_env`` etc.,
which we do not want on the Jetson).

Mirrored from dsrc commit d477dba:
  - src/rl/encoders.py      (LOCAL_OBS_FIELDS, COOPERATION_FIELDS,
                             LANE_DISTRIBUTION_LANES, FIELD_SCALES,
                             encode_local_observation semantics)
  - src/rl/actions.py       (ACTION_HEADS, ACTION_VALUES, FORCED_ACTIONS)
  - src/envs/wrappers.py    (bin value orders, decode_speed_bin,
                             decode_headway_bin)
  - specs/observation_schema.md (neutral fallback values)

tests/test_sim_contract.py asserts equality against the originals
whenever the sim package is importable. If you change the sim contract,
re-run that test on a machine with the sim repo and update this file.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

SIM_COMMIT = "d477dba"

# --- Observation contract (src/rl/encoders.py) -------------------------------

LOCAL_OBS_FIELDS: tuple[str, ...] = (
    "is_active",
    "ego_speed",
    "ego_acceleration",
    "ego_lane",
    "ego_headway_s",
    "target_headway_s",
    "time_since_last_lane_change",
    "lane_changes_last_km",
    "distance_to_next_merge",
    "distance_to_downstream_bottleneck",
    "leader_gap",
    "leader_relative_speed",
    "follower_gap",
    "follower_relative_speed",
    "left_lane_front_gap",
    "left_lane_rear_gap",
    "right_lane_front_gap",
    "right_lane_rear_gap",
    "target_lane_front_gap",
    "target_lane_rear_gap",
    "target_lane_rear_required_decel",
    "downstream_congestion_estimate",
    "merge_pressure",
    "segment_target_speed",
    "uncongested_low_speed_flag",
    "local_density_bin",
    "local_mean_speed_bin",
    "local_queue_estimate",
    "active_vehicle_count_local",
    "active_av_count_local",
    "nearby_av_count",
    "nearby_av_density",
    "nearby_av_mean_speed",
)
COOPERATION_FIELDS: tuple[str, ...] = (
    "segment_target_speed",
    "merge_pressure",
    "downstream_congestion_estimate",
)
LANE_DISTRIBUTION_LANES: tuple[str, ...] = ("0", "1", "2")

FIELD_SCALES: dict[str, float] = {
    "ego_speed": 40.0,
    "ego_acceleration": 8.0,
    "ego_lane": 3.0,
    "ego_headway_s": 10.0,
    "target_headway_s": 10.0,
    "time_since_last_lane_change": 120.0,
    "lane_changes_last_km": 5.0,
    "distance_to_next_merge": 500.0,
    "distance_to_downstream_bottleneck": 500.0,
    "leader_gap": 150.0,
    "leader_relative_speed": 40.0,
    "follower_gap": 150.0,
    "follower_relative_speed": 40.0,
    "left_lane_front_gap": 150.0,
    "left_lane_rear_gap": 150.0,
    "right_lane_front_gap": 150.0,
    "right_lane_rear_gap": 150.0,
    "target_lane_front_gap": 150.0,
    "target_lane_rear_gap": 150.0,
    "target_lane_rear_required_decel": 8.0,
    "downstream_congestion_estimate": 1.0,
    "merge_pressure": 1.0,
    "segment_target_speed": 40.0,
    "uncongested_low_speed_flag": 1.0,
    "local_density_bin": 2.0,
    "local_mean_speed_bin": 2.0,
    "local_queue_estimate": 20.0,
    "active_vehicle_count_local": 50.0,
    "active_av_count_local": 50.0,
    "nearby_av_count": 50.0,
    "nearby_av_density": 100.0,
    "nearby_av_mean_speed": 40.0,
}


def local_obs_dim() -> int:
    return len(LOCAL_OBS_FIELDS) + len(COOPERATION_FIELDS) + len(LANE_DISTRIBUTION_LANES)


def encoded_slot_names() -> tuple[str, ...]:
    """The 39 slot names `encode_local_observation` fills, in encoder order.

    Dotted names for the two nested blocks (`cooperation.*`,
    `nearby_av_lane_distribution.<lane>`), so a flat field and the same value
    read through the nested block it is also part of have distinct names in
    anything keyed on this tuple -- `field_sources` included. Additive: it
    introduces no new constant, so `contract_fingerprint` (below), which
    hashes only `LOCAL_OBS_FIELDS` and `FIELD_SCALES`, is unaffected.
    """
    return (
        *LOCAL_OBS_FIELDS,
        *(f"cooperation.{field}" for field in COOPERATION_FIELDS),
        *(f"nearby_av_lane_distribution.{lane}" for lane in LANE_DISTRIBUTION_LANES),
    )


def encode_local_observation(obs: Mapping[str, Any]) -> np.ndarray:
    """numpy twin of src.rl.encoders.encode_local_observation (torch-free)."""
    values = [_field_number(field, obs.get(field)) for field in LOCAL_OBS_FIELDS]
    cooperation = obs.get("cooperation", {})
    if not isinstance(cooperation, Mapping):
        cooperation = {}
    values.extend(_field_number(field, cooperation.get(field)) for field in COOPERATION_FIELDS)
    lane_distribution = obs.get("nearby_av_lane_distribution", {})
    if not isinstance(lane_distribution, Mapping):
        lane_distribution = {}
    values.extend(_number(lane_distribution.get(lane_id)) for lane_id in LANE_DISTRIBUTION_LANES)
    return np.asarray(values, dtype=np.float32)


def _field_number(field: str, value: Any) -> float:
    return _number(value, FIELD_SCALES.get(field, 1.0))


def _number(value: Any, scale: float = 1.0) -> float:
    # Verbatim port of src.rl.encoders._number; every branch matters
    # (bools bypass scaling, inf is clamped to 200 before scaling).
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        result = 200.0 if result > 0 else -200.0
        limit = 5.0 * max(float(scale), 1e-9)
        result = max(-limit, min(limit, result))
    return result / max(float(scale), 1e-9)


# --- Action contract (src/rl/actions.py, src/envs/wrappers.py) ---------------

ACTION_HEADS: tuple[str, ...] = (
    "desired_speed_bin",
    "desired_headway_bin",
    "lane_preference",
    "merge_mode",
)
ACTION_VALUES: dict[str, tuple[str, ...]] = {
    "desired_speed_bin": ("slow", "nominal", "fast"),
    "desired_headway_bin": ("normal", "larger", "largest"),
    "lane_preference": ("keep", "prefer_left_if_safe", "prefer_right_if_safe"),
    "merge_mode": ("normal", "create_gap", "hold_lane"),
}
FORCED_ACTIONS: dict[str, str] = {
    "desired_headway_bin": "normal",
    "lane_preference": "keep",
    "merge_mode": "normal",
}

ACTION_PROFILES: dict[str, tuple[str, ...]] = {
    "speed_only": ("desired_speed_bin",),
    "speed_headway": ("desired_speed_bin", "desired_headway_bin"),
    "full": ACTION_HEADS,
}


def active_heads(profile: str) -> tuple[str, ...]:
    try:
        return ACTION_PROFILES[profile]
    except KeyError:
        raise ValueError(f"unsupported action profile '{profile}'") from None


def default_indices() -> dict[str, int]:
    indices: dict[str, int] = {}
    for head in ACTION_HEADS:
        value = FORCED_ACTIONS.get(head, ACTION_VALUES[head][0])
        indices[head] = ACTION_VALUES[head].index(value)
    return indices


def indices_to_action(indices: Mapping[str, int]) -> dict[str, str]:
    defaults = default_indices()
    return {
        head: ACTION_VALUES[head][int(indices.get(head, defaults[head]))]
        for head in ACTION_HEADS
    }


# --- Action decoding (src/envs/wrappers.py) ----------------------------------

SPEED_BIN_OFFSETS_MPS: dict[str, float] = {"slow": -10.0, "nominal": -3.0, "fast": 0.0}
HEADWAY_BIN_S: dict[str, float] = {"normal": 1.6, "larger": 2.2, "largest": 3.0}


def decode_speed_bin(
    speed_bin: str,
    free_flow_speed_mps: float = 30.0,
    min_contextual_speed_mps: float = 12.0,
) -> float:
    return max(min_contextual_speed_mps, free_flow_speed_mps + SPEED_BIN_OFFSETS_MPS[speed_bin])


def decode_headway_bin(headway_bin: str) -> float:
    return HEADWAY_BIN_S[headway_bin]


# --- Neutral fallbacks (specs/observation_schema.md) --------------------------


def neutral_cooperation(free_flow_speed_mps: float) -> dict[str, float]:
    """Cooperation-block values mandated when no nearby AVs are sensed."""
    return {
        "segment_target_speed": float(free_flow_speed_mps),
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
    }


def bin_index(value: float, edges: tuple[float, ...] | list[float]) -> int:
    """Mirror of src.sensing.local._bin: count of edges <= value."""
    return int(sum(float(value) >= edge for edge in edges))


def contract_fingerprint() -> str:
    """A short hash over everything a bundle's inputs depend on.

    The dimension alone was the only thing guarding a loaded bundle, and it does not
    move when the contract does: reordering two `LOCAL_OBS_FIELDS` entries or changing
    a `FIELD_SCALES` value leaves it at 39, so an old module loads silently against a
    new contract and every field lands in the wrong slot -- which is exactly the class
    of change the dimension check's own error message names.

    Field NAMES in order and their scales, because those two decide what the encoder
    puts where and how far it stretches it. `SIM_COMMIT` is deliberately not in here:
    the sim moves for reasons that do not touch this contract, and a guard that fires
    on every unrelated commit would be turned off within a week.
    """
    import hashlib
    import json as _json

    payload = _json.dumps(
        {
            "fields": list(LOCAL_OBS_FIELDS),
            "scales": {k: FIELD_SCALES[k] for k in sorted(FIELD_SCALES)},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
