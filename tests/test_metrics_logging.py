from __future__ import annotations

import csv
import json

import pytest

from src.envs.topology_env import HighwayTopologyEnv, VehicleRuntime
from src.metrics import MetricThresholds, MetricsLogger, compute_segment_metrics, compute_step_metrics, jain_fairness
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
