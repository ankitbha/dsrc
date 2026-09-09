from __future__ import annotations

import argparse
import json

import pytest

from scripts import validate_topology_baselines as validation


def result(
    *,
    topology: str,
    baseline: str,
    demand: str,
    seed: int = 7,
    rows: list[dict] | None = None,
    summary: dict | None = None,
) -> validation.RunResult:
    spec = validation.RunSpec(
        topology=topology,
        baseline=baseline,
        demand=demand,
        seed=seed,
        duration_steps=10,
        dt=1.0,
        av_penetration=None,
    )
    return validation.RunResult(
        spec=spec,
        paths={"episode_summary": "episode_summary.json", "step_metrics": "step_metrics.csv", "segment_metrics": "segment_metrics.csv"},
        summary=summary or {},
        step_rows=rows or [],
        segment_rows=[],
        hard_failures=[],
        notes=[],
    )


def test_build_matrix_uses_burst_duration() -> None:
    args = argparse.Namespace(duration_steps=10, burst_duration_steps=30, dt=1.0, av_penetration=None)
    matrix = validation.build_matrix(("ring",), ("random_av",), ("high", "burst"), (1, 2), args)
    durations = {(spec.demand, spec.seed): spec.duration_steps for spec in matrix}
    assert durations[("high", 1)] == 10
    assert durations[("burst", 2)] == 30


def test_action_counts_capture_baseline_intent() -> None:
    counts = validation._action_counts(
        {
            "av_0": {
                "desired_speed_bin": "slow",
                "desired_headway_bin": "largest",
                "lane_preference": "keep",
                "merge_mode": "create_gap",
            },
            "av_1": {
                "desired_speed_bin": "fast",
                "desired_headway_bin": "normal",
                "lane_preference": "prefer_left_if_safe",
                "merge_mode": "hold_lane",
            },
        }
    )
    assert counts["action_count"] == 2
    assert counts["action_create_gap_count"] == 1
    assert counts["action_hold_lane_count"] == 1
    assert counts["action_lane_preference_count"] == 1
    assert counts["action_slow_count"] == 1
    assert counts["action_fast_count"] == 1


def test_directional_checks_report_pass_warn_and_insufficient_signal() -> None:
    rows = [{"mean_speed": 20.0, "speed_std": 2.0, "jam_fraction": 0.0, "queue_length_total": 0.0}]
    results = [
        result(topology="ring", baseline="selfish_av", demand="low", rows=[{"mean_speed": 25.0}]),
        result(topology="ring", baseline="density_lookup", demand="low", rows=[{"mean_speed": 15.0}]),
        result(topology="ring", baseline="dynamic_speed_limit", demand="low", rows=[{"mean_speed": 16.0}]),
        result(topology="ring", baseline="random_av", demand="low", rows=[{"safety_masked_action_count": 2.0}]),
        result(topology="ring", baseline="cooperative_smoothing", demand="low", rows=[{"safety_masked_action_count": 0.0}]),
        result(topology="ring", baseline="random_av", demand="high", rows=rows),
        result(topology="ring", baseline="density_lookup", demand="high", rows=[{"speed_std": 1.0, "jam_fraction": 0.0}]),
    ]
    checks = validation.directional_checks(results)
    statuses = {check["status"] for check in checks}
    assert "pass" in statuses
    assert "insufficient_signal" in statuses


def test_write_reports_outputs_summary_files(tmp_path) -> None:
    run = result(
        topology="straight_single_lane",
        baseline="random_av",
        demand="high",
        summary={"completed_vehicle_count": 1, "active_vehicle_count": 2, "travel_time_mean": 3.0, "fairness_jain": 1.0},
    )
    directional = [
        {
            "topology": "straight_single_lane",
            "demand": "high",
            "check": "example",
            "status": "pass",
            "observed": 1.0,
            "reference": 0.0,
            "detail": ">",
        }
    ]
    validation.write_reports(tmp_path, [run], directional)
    assert (tmp_path / "run_summary.csv").exists()
    assert (tmp_path / "directional_checks.csv").exists()
    data = json.loads((tmp_path / "validation_summary.json").read_text())
    assert data["run_count"] == 1
    assert (tmp_path / "validation_summary.md").exists()


class TestArcsJoinWhereVehiclesCross:
    """A lane must end where its successor begins.

    Before this, `next_lane` sent a vehicle crossing node `c` from `('b2','c',1)`
    to a lane starting **ten metres** away laterally, and several other transitions
    moved it 2 to 4 m. A vehicle is 2 m wide, so it was being placed into occupied
    space, and no car-following bound can prevent a collision caused that way --
    task 72's residual was exactly six side-by-side events, and tightening MOBIL's
    braking parameter from 2.0 to 0.05 changed the count not at all.

    Two causes. The `SineLane`s used `phase=pi/2`, which puts the sine at plus or
    minus its full amplitude AT the arc ends rather than zero, and the nominal
    endpoints did not match the successors' starts.
    """

    TOLERANCE_M = 0.5

    def _jumps(self, topology_id):
        import numpy as np

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology(
            topology_id, load_named_config("topology", topology_id)
        ).road_network
        out = {}
        for index in sorted(network.lanes_dict()):
            lane = network.get_lane(index)
            end = np.array(lane.position(lane.length, 0))
            try:
                nxt = network.next_lane(index, route=None, position=end)
            except Exception:
                continue
            if nxt is None or nxt == index:
                continue
            start = np.array(network.get_lane(nxt).position(0.0, 0))
            out[(index, nxt)] = float(np.linalg.norm(end - start))
        return out

    @pytest.mark.parametrize("topology_id", ["inverted_tree", "inverted_tree_bottleneck"])
    def test_every_transition_is_continuous(self, topology_id):
        jumps = self._jumps(topology_id)
        assert jumps, "no transitions found; the fixture assumption is broken"
        worst = max(jumps.items(), key=lambda kv: kv[1])
        assert worst[1] <= self.TOLERANCE_M, (
            f"{worst[0][0]} -> {worst[0][1]} jumps {worst[1]:.2f} m laterally"
        )
