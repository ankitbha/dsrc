from __future__ import annotations

import csv
import json

import pytest

from src.envs.topology_env import HighwayTopologyEnv, VehicleRuntime
from src.metrics import MetricThresholds, MetricsLogger, compute_segment_metrics, compute_step_metrics, jain_fairness
from src.rl.rewards import build_team_reward
from src.sensing.local import VehicleSnapshot


def action() -> dict[str, str]:
    return {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }


def demand_config(**overrides):
    cfg = {
        "id": "test",
        "total_vehicles_per_hour": 36000,
        "av_penetration": 1.0,
        "branch_split": {"main": 1.0},
        "burst": {"enabled": False},
        "speed_distribution": {"mean_mps": 24.0, "std_mps": 0.0, "min_mps": 12.0, "max_mps": 32.0},
        "spawn_min_gap_m": 0.0,
    }
    cfg.update(overrides)
    return cfg


def test_fairness_and_step_metrics_are_stable_for_empty_or_balanced_counts() -> None:
    thresholds = MetricThresholds()
    assert jain_fairness([]) == 1.0
    assert jain_fairness([10, 10]) == pytest.approx(1.0)
    assert jain_fairness([10, 0]) == pytest.approx(0.5)

    metrics = compute_step_metrics(
        time_s=12.0,
        active_vehicle_records=[
            {
                "role": "av",
                "speed": 10.0,
                "acceleration": -4.0,
                "distance_traveled_m": 100.0,
                "lane_changed_this_step": True,
                "free_flow_speed_mps": 30.0,
                "segment_density": 2.0,
            },
            {
                "role": "human",
                "speed": 20.0,
                "acceleration": 0.0,
                "distance_traveled_m": 100.0,
                "lane_changed_this_step": False,
                "free_flow_speed_mps": 30.0,
                "segment_density": 2.0,
            },
        ],
        segment_metrics={"s0": {"queue_length": 1, "jam_fraction": 0.5, "rolling_roadblock_score": 0.0}},
        diagnostics={"safety_masked_action": [{}], "follower_disruption_blocked": [{}, {}]},
        completed_vehicle_count=3,
        recent_completion_times=[1.0, 11.0],
        completed_travel_times=[20.0, 30.0],
        branch_completed={"main": 3},
        branch_travel_times={"main": [20.0, 30.0]},
        lane_change_dwell_times=[16.0],
        hard_braking_count=1,
        hard_brakes_caused_by_av=1,
        follower_delay_imposed_by_av=2.0,
        rear_ttc_after_av_lane_change_min=float("inf"),
        thresholds=thresholds,
    )

    assert metrics["active_vehicle_count"] == 2
    assert metrics["mean_speed"] == pytest.approx(15.0)
    assert metrics["hard_braking_count"] == 1
    assert metrics["safety_masked_action_count"] == 1
    assert metrics["follower_disruption_blocked_count"] == 2
    assert metrics["travel_time_mean"] == pytest.approx(25.0)
    assert metrics["av_low_speed_uncongested_fraction"] == pytest.approx(1.0)


def test_segment_metrics_include_lane_use_and_roadblock_fields() -> None:
    metrics = compute_segment_metrics(
        segment_ids=("seg",),
        segment_lengths_m={"seg": 1000.0},
        lane_counts={"seg": 2},
        active_vehicle_records=[
            {"role": "av", "segment_id": "seg", "lane_id": 0, "speed": 10.0, "branch_id": "main", "free_flow_speed_mps": 30.0},
            {"role": "av", "segment_id": "seg", "lane_id": 1, "speed": 11.0, "branch_id": "main", "free_flow_speed_mps": 30.0},
            {"role": "human", "segment_id": "seg", "lane_id": 1, "speed": 4.0, "branch_id": "main", "free_flow_speed_mps": 30.0},
        ],
        step_inflow={"seg": 2},
        step_outflow={"seg": 1},
        thresholds=MetricThresholds(),
    )

    segment = metrics["seg"]
    assert segment["vehicle_count"] == 3
    assert segment["av_count"] == 2
    assert segment["lane_counts"] == {"0": 1, "1": 2}
    assert segment["lane_av_counts"] == {"0": 1, "1": 1}
    assert segment["queue_length"] == 1
    assert segment["all_lane_av_low_speed_occupancy"] == 1.0


def test_logger_writes_json_and_csv_artifacts(tmp_path) -> None:
    logger = MetricsLogger(experiment_id="exp_test", output_root=tmp_path)
    logger.record_step({"time": 1.0, "mean_speed": 12.0, "branch_throughput": {"main": 2}})
    logger.record_segments(time_s=1.0, segment_metrics={"seg": {"vehicle_count": 2, "lane_counts": {"0": 2}}})

    paths = logger.write_episode({"topology_id": "straight_single_lane", "completed_vehicle_count": 2})

    assert json.loads((tmp_path / "exp_test" / "episode_summary.json").read_text())["completed_vehicle_count"] == 2
    with open(paths["step_metrics"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["mean_speed"] == "12.0"
    with open(paths["segment_metrics"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["segment_id"] == "seg"


@pytest.mark.parametrize(
    "topology_id",
    ["ring", "straight_single_lane", "straight_multilane", "merge", "inverted_tree", "inverted_tree_bottleneck"],
)
def test_env_exposes_canonical_metrics_for_all_topologies(topology_id: str) -> None:
    env = HighwayTopologyEnv(topology_id, {"controlled_vehicles": 2, "duration_steps": 2})
    obs, _ = env.reset(seed=13)
    for local_obs in obs.values():
        assert "action_mask" in local_obs
    obs, _, _, _, info = env.step({agent_id: action() for agent_id in obs})

    for key in (
        "mean_speed",
        "speed_std",
        "queue_length_total",
        "hard_braking_count",
        "lane_change_count",
        "throughput_recent",
        "rolling_roadblock_score",
        "fairness_jain",
    ):
        assert key in info["metrics"]
    for segment in env.get_segment_metrics().values():
        assert "lane_counts" in segment
        assert "lane_fractions" in segment
        assert "rolling_roadblock_score" in segment
        assert "all_lane_av_low_speed_occupancy" in segment


def test_segment_metrics_cache_reuses_state_and_invalidates_on_step_and_reset() -> None:
    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 2, "duration_steps": 2})
    obs, _ = env.reset(seed=13)

    first = env.get_segment_metrics()
    assert env.get_segment_metrics() is first

    obs, _, _, _, _ = env.step({agent_id: action() for agent_id in obs})
    after_step = env.get_segment_metrics()
    assert after_step is env.get_segment_metrics()
    assert after_step is not first

    env.reset(seed=14)
    after_reset = env.get_segment_metrics()
    assert after_reset is env.get_segment_metrics()
    assert after_reset is not after_step


def test_rear_ttc_after_av_lane_change_uses_closing_rear_vehicle() -> None:
    env = HighwayTopologyEnv("straight_multilane")
    lane_index = ("s0", "s1", 1)
    env._vehicle_runtime = {
        "av_0": VehicleRuntime(
            vehicle_id="av_0",
            role="av",
            branch_id="main",
            spawn_time_s=0.0,
            previous_speed_mps=10.0,
            previous_lane_index=("s0", "s1", 0),
            previous_segment_id="straight_upstream",
            lane_changed_this_step=True,
        ),
        "human_0": VehicleRuntime(
            vehicle_id="human_0",
            role="human",
            branch_id="main",
            spawn_time_s=0.0,
            previous_speed_mps=20.0,
            previous_lane_index=lane_index,
            previous_segment_id="straight_upstream",
        ),
    }
    env._vehicle_snapshots = lambda: [  # type: ignore[method-assign]
        VehicleSnapshot(
            vehicle_id="av_0",
            role="av",
            segment_id="straight_upstream",
            lane_index=lane_index,
            lane_id=1,
            position=(30.0, 0.0),
            longitudinal_m=30.0,
            speed_mps=10.0,
            acceleration_mps2=0.0,
            free_flow_speed_mps=30.0,
        ),
        VehicleSnapshot(
            vehicle_id="human_0",
            role="human",
            segment_id="straight_upstream",
            lane_index=lane_index,
            lane_id=1,
            position=(10.0, 0.0),
            longitudinal_m=10.0,
            speed_mps=20.0,
            acceleration_mps2=0.0,
            free_flow_speed_mps=30.0,
        ),
    ]

    assert env._rear_ttc_after_av_lane_change_min() == pytest.approx(2.0)


def test_rear_ttc_after_av_lane_change_wraps_on_ring() -> None:
    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 0, "initial_human_vehicles": 0})
    env.reset(seed=3)
    lane_index = ("r0", "r1", 0)
    lane_length = env.road.network.get_lane(lane_index).length
    env._vehicle_runtime = {
        "av_0": VehicleRuntime(
            vehicle_id="av_0",
            role="av",
            branch_id="main",
            spawn_time_s=0.0,
            previous_speed_mps=10.0,
            previous_lane_index=lane_index,
            previous_segment_id="ring",
            lane_changed_this_step=True,
        ),
        "human_0": VehicleRuntime(
            vehicle_id="human_0",
            role="human",
            branch_id="main",
            spawn_time_s=0.0,
            previous_speed_mps=20.0,
            previous_lane_index=lane_index,
            previous_segment_id="ring",
        ),
    }
    env._vehicle_snapshots = lambda: [  # type: ignore[method-assign]
        VehicleSnapshot(
            vehicle_id="av_0",
            role="av",
            segment_id="ring",
            lane_index=lane_index,
            lane_id=0,
            position=(0.0, 0.0),
            longitudinal_m=10.0,
            speed_mps=10.0,
            acceleration_mps2=0.0,
            free_flow_speed_mps=30.0,
        ),
        VehicleSnapshot(
            vehicle_id="human_0",
            role="human",
            segment_id="ring",
            lane_index=lane_index,
            lane_id=0,
            position=(0.0, 0.0),
            longitudinal_m=lane_length - 10.0,
            speed_mps=20.0,
            acceleration_mps2=0.0,
            free_flow_speed_mps=30.0,
        ),
    ]

    assert env._rear_ttc_after_av_lane_change_min() == pytest.approx(20.0 / 10.0)


def test_human_crash_reward_penalty_is_impulsive_but_metric_persists() -> None:
    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 0, "initial_human_vehicles": 1, "duration_steps": 3})
    env.reset(seed=5)
    for vehicle in env._human_vehicles.values():
        vehicle.crashed = True

    _, _, _, _, first_info = env.step({})
    _, _, _, _, second_info = env.step({})

    assert first_info["metrics"]["collision_count"] == 1
    assert first_info["metrics"]["new_collision_count"] == 1
    assert second_info["metrics"]["collision_count"] == 1
    assert second_info["metrics"]["new_collision_count"] == 0
    assert build_team_reward(first_info["metrics"]) < build_team_reward(second_info["metrics"])


def test_initial_human_spawn_avoids_existing_vehicle_overlap() -> None:
    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 2, "initial_human_vehicles": 3})
    env.reset(seed=7)
    by_lane: dict[tuple[str, str, int], list[float]] = {}
    for vehicle in env._active_vehicles():
        assert vehicle.lane_index is not None
        lane = env.road.network.get_lane(vehicle.lane_index)
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        by_lane.setdefault(vehicle.lane_index, []).append(float(longitudinal))

    for lane_index, positions in by_lane.items():
        lane_length = env.road.network.get_lane(lane_index).length
        for index, position in enumerate(positions):
            for other in positions[index + 1 :]:
                gap = abs(position - other)
                assert min(gap, lane_length - gap) >= 8.9


def test_active_vehicle_records_use_precomputed_segment_density_and_reverse_lookup() -> None:
    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 2, "initial_human_vehicles": 2})
    env.reset(seed=11)
    records = env._active_vehicle_records()
    segment_metrics = env.get_segment_metrics()

    for record in records:
        segment_id = record["segment_id"]
        assert record["segment_density"] == pytest.approx(segment_metrics[segment_id]["density"])
    for agent_id, vehicle in env._av_vehicles.items():
        assert env._agent_id_for_vehicle(vehicle) == agent_id


def test_exited_vehicles_count_for_throughput_and_travel_time_not_active_segments() -> None:
    env = HighwayTopologyEnv(
        "straight_single_lane",
        {
            "duration_steps": 30,
            "dt": 10.0,
            "demand": demand_config(total_vehicles_per_hour=72000, av_penetration=1.0),
            "max_speed_mps": 40.0,
        },
    )
    obs, _ = env.reset(seed=5)
    completed = 0
    last_info = {}
    for _ in range(30):
        obs, _, _, _, last_info = env.step({agent_id: action() for agent_id in obs})
        completed = env.get_global_state()["completed_vehicle_count"]
        if completed:
            break
    if not completed:
        pytest.skip("no vehicle exited during deterministic smoke horizon")

    active_segment_count = sum(segment["vehicle_count"] for segment in env.get_segment_metrics().values())
    summary = env.get_episode_summary()
    assert active_segment_count == env.get_global_state()["active_vehicle_count"]
    assert summary["completed_vehicle_count"] == completed
    assert summary["travel_time_count"] == completed
    assert last_info["metrics"]["travel_time_count"] == completed
