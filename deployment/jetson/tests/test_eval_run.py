"""eval_run.analyze over synthetic run logs: metrics and gate logic."""

import json

import pytest

from eval_run import analyze

T0 = 1_750_000_000.0
RATE_HZ = 30.0


def make_tick(i: int, *, e2e_ms=20.0, ego_speed=20.0, leader_gap=35.0,
              gps_fresh=True, leader_rel_measured=True) -> dict:
    has_leader = leader_gap is not None
    gap = leader_gap if has_leader else float("inf")
    return {
        "type": "tick",
        "tick_id": i,
        "frame_id": i,
        "t_wall": T0 + i / RATE_HZ,
        "e2e_ms": e2e_ms,
        "stage_ms": {"detect": 17.0, "track_distance": 0.4, "observe": 0.5,
                     "policy_advisory": 0.7, "capture_to_start": 1.0},
        "fps": RATE_HZ,
        "n_detections": 2 if has_leader else 0,
        "vehicles": (
            [{"id": 1, "cls": 2, "conf": 0.8, "dist_m": gap, "lat_m": 0.1,
              "rel_mps": -1.5 if leader_rel_measured else None,
              "method": "ground_plane", "bbox": [0, 0, 10, 10]}]
            if has_leader else []
        ),
        "obs": {"leader_gap": gap, "ego_speed": ego_speed},
        "field_sources": {
            "leader_relative_speed": "measured" if (has_leader and leader_rel_measured)
            else "fallback_neutral",
        },
        "obs_diagnostics": {
            "missingness": 0.3,
            "fallback_fields": ["follower_gap", "merge_pressure"],
            "gps_fresh": gps_fresh,
            "leader_track_id": 1 if has_leader else None,
        },
        "action": {"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                   "lane_preference": "keep", "merge_mode": "normal"},
        "advisory": {"recommended_speed_mps": 24.0, "recommended_speed_display": 53.7,
                     "units": "mph", "headway_target_s": 1.6, "lane_text": "keep lane",
                     "merge_text": "normal", "confidence_label": "low"},
        "gps": {"valid": gps_fresh, "lat": 39.0, "lon": -77.0,
                "speed_mps": ego_speed, "heading_deg": 105.0, "num_sats": 10, "hdop": 0.8},
        "n_peers": 0,
    }


def write_run(tmp_path, ticks, scenario=None, summary=None):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        if scenario is not None:
            f.write(json.dumps(scenario) + "\n")
        for t in ticks:
            f.write(json.dumps(t) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary or {
        "ticks": len(ticks), "camera_dropped_frames": 0, "policy_trained": False,
    }))
    return run_dir


def scenario_record(speed=20.0, dropouts=()):
    return {
        "type": "scenario",
        "scenario_path": "test.json",
        "description": "synthetic",
        "video_source": "file:test.webm",
        "gps_profile": {
            "start": {"lat": 39.0, "lon": -77.0, "heading_deg": 105.0},
            "rate_hz": 5,
            "speed_profile_mps": speed,
            "dropouts_s": [list(d) for d in dropouts],
            "noise": {"speed_std_mps": 0.0, "pos_std_m": 0.0},
            "seed": 0,
            "loop": False,
        },
        "gps_start_wall": T0,
        "gps_start_mono": 100.0,
    }


def test_healthy_run_passes_all_gates(tmp_path):
    ticks = [make_tick(i) for i in range(90)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
    result = analyze(run_dir)
    assert result["n_ticks"] == 90
    assert result["tick_rate_hz_median"] == pytest.approx(30.0, rel=0.05)
    assert result["gps"]["speed_rmse_mps"] == pytest.approx(0.0, abs=1e-6)
    assert result["perception"]["leader_present_fraction"] == 1.0
    assert result["perception"]["leader_rel_speed_measured_fraction"] == 1.0
    assert all(g["pass"] in (True, None) for g in result["gates"].values())
    assert result["overall_pass"]


def test_latency_gate_fails_on_slow_run(tmp_path):
    ticks = [make_tick(i, e2e_ms=250.0) for i in range(60)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
    result = analyze(run_dir)
    assert result["gates"]["latency_jetson_p95"]["pass"] is False
    assert not result["overall_pass"]


def test_speed_rmse_gate_catches_unit_bug(tmp_path):
    # ego speed logged in knots-ish scale vs scripted 20 m/s truth
    ticks = [make_tick(i, ego_speed=10.3) for i in range(60)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record(speed=20.0))
    result = analyze(run_dir)
    assert result["gates"]["gps_speed_rmse"]["pass"] is False


def test_scripted_dropout_does_not_fail_freshness_gate(tmp_path):
    # dropout covers t in [0.5, 1.5); ticks there report stale GPS
    dropout = (0.5, 1.5)
    ticks = []
    for i in range(90):
        elapsed = i / RATE_HZ
        in_drop = dropout[0] <= elapsed < dropout[1] + 0.3
        ticks.append(make_tick(i, gps_fresh=not in_drop))
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record(dropouts=[dropout]))
    result = analyze(run_dir)
    assert result["gps"]["fresh_fraction_overall"] < 0.95
    assert result["gates"]["gps_fresh"]["pass"] is True  # judged outside dropouts
    assert result["gps"]["speed_max_drift_during_dropout_mps"] is not None


def test_no_gps_run_marks_gps_gates_not_applicable(tmp_path):
    ticks = [make_tick(i, gps_fresh=False) for i in range(60)]
    for t in ticks:
        t["gps"]["valid"] = False
    run_dir = write_run(tmp_path, ticks)  # no scenario record either
    result = analyze(run_dir)
    assert result["gates"]["gps_fresh"]["pass"] is None
    assert result["gates"]["gps_speed_rmse"]["pass"] is None
    assert result["overall_pass"]  # remaining applicable gates still pass


def test_empty_traffic_fails_perception_gate(tmp_path):
    ticks = [make_tick(i, leader_gap=None) for i in range(60)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
    result = analyze(run_dir)
    assert result["gates"]["perception_coverage"]["pass"] is False
    assert result["perception"]["leader_present_fraction"] == 0.0
