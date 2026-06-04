from __future__ import annotations

import math

import numpy as np
import pytest

from src.road.topology_factory import build_topology
from src.safety import SafetyConstraints, SafetyState
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot


def snapshot(
    vehicle_id: str,
    *,
    role: str = "av",
    lane_id: int = 1,
    longitudinal_m: float = 100.0,
    speed_mps: float = 20.0,
    segment_id: str = "straight_upstream",
) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id=vehicle_id,
        role=role,
        segment_id=segment_id,
        lane_index=("s0", "s1", lane_id),
        lane_id=lane_id,
        position=(longitudinal_m, lane_id * 4.0),
        longitudinal_m=longitudinal_m,
        speed_mps=speed_mps,
        acceleration_mps2=0.0,
        free_flow_speed_mps=30.0,
    )


def build_obs(
    snapshots,
    *,
    config: SensingConfig | None = None,
    current_av_ids=("av_0",),
    time_s: float = 0.0,
    rng_seed: int = 7,
):
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(config)
    return builder.build_all(
        time_s=time_s,
        topology=topology,
        snapshots=snapshots,
        current_av_ids=list(current_av_ids),
        safety_states={agent_id: SafetyState() for agent_id in current_av_ids},
        target_headways={agent_id: 1.6 for agent_id in current_av_ids},
        target_lanes={agent_id: ("s0", "s1", 1) for agent_id in current_av_ids},
        segment_metrics={"straight_upstream": {"density": 4.0, "jam_fraction": 0.0}},
        constraints=SafetyConstraints(),
        rng=np.random.RandomState(rng_seed),
    )


def test_local_observation_contains_public_schema_sensor_and_cooperation() -> None:
    obs = build_obs([snapshot("av_0")])["av_0"]

    for key in (
        "is_active",
        "ego_speed",
        "ego_acceleration",
        "ego_lane",
        "ego_headway_s",
        "target_headway_s",
        "current_segment",
        "leader_gap",
        "follower_gap",
        "local_density_bin",
        "local_mean_speed_bin",
        "active_vehicle_count_local",
        "active_av_count_local",
        "nearby_av_count",
        "nearby_av_density",
        "nearby_av_mean_speed",
        "nearby_av_lane_distribution",
    ):
        assert key in obs
    assert obs["sensor"] == {
        "range_m": 150.0,
        "latency_s": 0.0,
        "position_noise_std": 0.0,
        "speed_noise_std": 0.0,
    }
    assert obs["cooperation"] == {
        "segment_target_speed": 30.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
    }


def test_local_counts_exclude_ego_and_respect_range() -> None:
    obs = build_obs(
        [
            snapshot("av_0", longitudinal_m=100.0),
            snapshot("av_1", longitudinal_m=130.0),
            snapshot("human_0", role="human", longitudinal_m=190.0),
            snapshot("av_2", longitudinal_m=260.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["active_vehicle_count_local"] == 2
    assert obs["active_av_count_local"] == 1
    assert obs["nearby_av_count"] == 1
    assert obs["nearby_av_density"] == pytest.approx(5.0)
    assert obs["nearby_av_mean_speed"] == pytest.approx(20.0)


def test_neutral_cooperation_fallback_when_no_nearby_avs() -> None:
    obs = build_obs(
        [
            snapshot("av_0", speed_mps=18.0),
            snapshot("human_0", role="human", longitudinal_m=130.0, speed_mps=12.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["nearby_av_count"] == 0
    assert obs["nearby_av_density"] == 0.0
    assert obs["nearby_av_mean_speed"] == pytest.approx(30.0)
    assert obs["nearby_av_lane_distribution"] == {}
    assert obs["cooperation"]["segment_target_speed"] == pytest.approx(30.0)
    assert obs["cooperation"]["merge_pressure"] == 0.0
    assert obs["cooperation"]["downstream_congestion_estimate"] == 0.0


def test_nearby_av_aggregates_hide_neighbor_identities() -> None:
    obs = build_obs(
        [
            snapshot("av_0", lane_id=1, longitudinal_m=100.0),
            snapshot("av_1", lane_id=0, longitudinal_m=130.0, speed_mps=10.0),
            snapshot("av_2", lane_id=2, longitudinal_m=140.0, speed_mps=20.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["nearby_av_count"] == 2
    assert obs["nearby_av_mean_speed"] == pytest.approx(15.0)
    assert obs["nearby_av_lane_distribution"] == {"0": 0.5, "2": 0.5}
    assert "av_1" not in str(obs["nearby_av_lane_distribution"])
    assert "av_2" not in str(obs["cooperation"])


def test_lane_relative_gaps_are_range_limited() -> None:
    obs = build_obs(
        [
            snapshot("av_0", lane_id=1, longitudinal_m=100.0, speed_mps=20.0),
            snapshot("human_front", role="human", lane_id=1, longitudinal_m=130.0, speed_mps=15.0),
            snapshot("human_rear", role="human", lane_id=1, longitudinal_m=80.0, speed_mps=25.0),
            snapshot("human_left", role="human", lane_id=0, longitudinal_m=120.0),
            snapshot("human_right", role="human", lane_id=2, longitudinal_m=90.0),
            snapshot("human_far", role="human", lane_id=1, longitudinal_m=400.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["leader_gap"] == pytest.approx(30.0)
    assert obs["leader_relative_speed"] == pytest.approx(-5.0)
    assert obs["follower_gap"] == pytest.approx(20.0)
    assert obs["follower_relative_speed"] == pytest.approx(5.0)
    assert obs["left_lane_front_gap"] == pytest.approx(20.0)
    assert math.isinf(obs["left_lane_rear_gap"])
    assert math.isinf(obs["right_lane_front_gap"])
    assert obs["right_lane_rear_gap"] == pytest.approx(10.0)
    assert obs["leader_gap"] < 300.0


def test_density_speed_bins_and_queue_estimate() -> None:
    obs = build_obs(
        [
            snapshot("av_0", longitudinal_m=100.0, speed_mps=20.0),
            snapshot("human_0", role="human", longitudinal_m=110.0, speed_mps=4.0),
            snapshot("human_1", role="human", longitudinal_m=120.0, speed_mps=6.0),
            snapshot("human_2", role="human", longitudinal_m=130.0, speed_mps=8.0),
        ],
        config=SensingConfig(range_m=50.0, density_bin_edges_veh_per_km=(12.0, 30.0), mean_speed_bin_edges_mps=(5.0, 10.0)),
    )["av_0"]

    assert obs["local_density_bin"] == 2
    assert obs["local_mean_speed_bin"] == 1
    assert obs["local_queue_estimate"] == 1


def test_noise_is_seed_reproducible_and_changes_continuous_measurements() -> None:
    snapshots = [
        snapshot("av_0", longitudinal_m=100.0, speed_mps=20.0),
        snapshot("human_front", role="human", longitudinal_m=130.0, speed_mps=15.0),
    ]
    noisy = SensingConfig(position_noise_std=2.0, speed_noise_std=1.0)

    obs_a = build_obs(snapshots, config=noisy, rng_seed=17)["av_0"]
    obs_b = build_obs(snapshots, config=noisy, rng_seed=17)["av_0"]
    obs_clean = build_obs(snapshots, config=SensingConfig(), rng_seed=17)["av_0"]

    assert obs_a["leader_gap"] == pytest.approx(obs_b["leader_gap"])
    assert obs_a["leader_relative_speed"] == pytest.approx(obs_b["leader_relative_speed"])
    assert obs_a["leader_gap"] != pytest.approx(obs_clean["leader_gap"])


def test_latency_uses_oldest_frame_until_buffer_is_warm_then_delayed_frame() -> None:
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(SensingConfig(latency_s=1.0))
    kwargs = {
        "topology": topology,
        "current_av_ids": ["av_0"],
        "safety_states": {"av_0": SafetyState()},
        "target_headways": {"av_0": 1.6},
        "target_lanes": {"av_0": ("s0", "s1", 1)},
        "segment_metrics": {"straight_upstream": {"density": 1.0, "jam_fraction": 0.0}},
        "constraints": SafetyConstraints(),
    }

    first = builder.build_all(
        time_s=0.0,
        snapshots=[snapshot("av_0", speed_mps=10.0)],
        rng=np.random.RandomState(3),
        **kwargs,
    )["av_0"]
    delayed = builder.build_all(
        time_s=1.0,
        snapshots=[snapshot("av_0", speed_mps=20.0)],
        rng=np.random.RandomState(3),
        **kwargs,
    )["av_0"]

    assert first["ego_speed"] == pytest.approx(10.0)
    assert delayed["ego_speed"] == pytest.approx(10.0)


def test_lane_gap_context_reuses_frame_recorded_by_build_all() -> None:
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(SensingConfig())
    snapshots = [
        snapshot("av_0", longitudinal_m=100.0),
        snapshot("human_rear", role="human", longitudinal_m=80.0),
    ]
    kwargs = {
        "topology": topology,
        "current_av_ids": ["av_0"],
        "safety_states": {"av_0": SafetyState()},
        "target_headways": {"av_0": 1.6},
        "target_lanes": {"av_0": ("s0", "s1", 1)},
        "segment_metrics": {"straight_upstream": {"density": 1.0, "jam_fraction": 0.0}},
        "constraints": SafetyConstraints(),
    }

    builder.build_all(time_s=2.0, snapshots=snapshots, rng=np.random.RandomState(3), **kwargs)
    frame_count = len(builder.buffer._frames)
    builder.lane_gap_context(
        ego_id="av_0",
        time_s=2.0,
        topology=topology,
        snapshots=snapshots,
        target_lane=("s0", "s1", 1),
        rng=np.random.RandomState(3),
    )

    assert len(builder.buffer._frames) == frame_count
