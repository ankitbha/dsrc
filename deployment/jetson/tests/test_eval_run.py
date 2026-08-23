"""eval_run.analyze over synthetic run logs: metrics and gate logic."""

import json

import pytest

from eval_run import analyze

T0 = 1_750_000_000.0
RATE_HZ = 30.0


def make_tick(i: int, *, e2e_ms=20.0, ego_speed=20.0, leader_gap=35.0,
              gps_fresh=True, leader_rel_measured=True, jetson_ms=None,
              link_ms=None) -> dict:
    has_leader = leader_gap is not None
    gap = leader_gap if has_leader else float("inf")
    return {
        "type": "tick",
        "tick_id": i,
        "frame_id": i,
        "t_wall": T0 + i / RATE_HZ,
        "e2e_ms": e2e_ms,
        # None means "absent from this run", which is what a pre-split recording
        # looks like -- the fallback path. A value exercises the gate proper.
        **({} if jetson_ms is None else {"jetson_ms": jetson_ms}),
        **({} if link_ms is None else {"link_ms": link_ms}),
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


def test_the_gate_is_on_the_jetson_segment_not_end_to_end(tmp_path):
    """The one behaviour change to a published number, and it had no test.

    Every existing eval_run test omits `jetson_ms`, so they all ran through the
    pre-split fallback and asserted a gate NAME rather than a gate: reverting the
    threshold to `e2e_ms` left the whole suite green.

    A run whose link is slow and whose Jetson is fast must PASS. The threshold is
    a claim about this hardware, and charging it for a link the Jetson does not
    control would fail a run for the network's behaviour.
    """
    ticks = [make_tick(i, e2e_ms=900.0, jetson_ms=30.0, link_ms=870.0) for i in range(120)]
    result = analyze(write_run(tmp_path, ticks, scenario=scenario_record()))
    assert result["gates"]["latency_jetson_p95"]["pass"] is True, (
        "a fast Jetson behind a slow link failed a gate about the Jetson"
    )
    assert result["latency_ms"]["jetson_ms"]["p95"] == pytest.approx(30.0, abs=0.1)
    assert result["latency_ms"]["e2e_ms"]["p95"] == pytest.approx(900.0, abs=0.1)
    assert result["latency_ms"]["jetson_ms_source"] == "measured"


def test_a_slow_jetson_fails_the_gate_even_behind_a_fast_link(tmp_path):
    ticks = [make_tick(i, e2e_ms=260.0, jetson_ms=250.0, link_ms=10.0) for i in range(120)]
    result = analyze(write_run(tmp_path, ticks, scenario=scenario_record()))
    assert result["gates"]["latency_jetson_p95"]["pass"] is False


def test_a_pre_split_run_says_the_gated_number_was_substituted(tmp_path):
    """The fallback is correct -- on a local camera the two coincided -- but a
    reader has to be told it happened rather than shown a number that looks
    measured."""
    ticks = [make_tick(i, e2e_ms=42.0) for i in range(120)]
    result = analyze(write_run(tmp_path, ticks, scenario=scenario_record()))
    assert result["latency_ms"]["jetson_ms_source"] == "absent from this run; e2e used"
    assert result["latency_ms"]["jetson_ms"]["p95"] == pytest.approx(42.0, abs=0.1)
    assert result["latency_ms"]["link_ms"] is None, (
        "a run with no link segment reported one"
    )


def test_the_link_segment_reports_its_count_and_its_negatives(tmp_path):
    """A converted capture stamp may land after the arrival it preceded, so the
    segment can be negative -- and with only mean/p50/p95/max, a negative merely
    lowered the p95 and was otherwise invisible."""
    ticks = [make_tick(i, jetson_ms=20.0, link_ms=(-2.0 if i < 10 else 8.0))
             for i in range(120)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
    link = analyze(run_dir)["latency_ms"]["link_ms"]
    assert link["n"] == 120
    assert link["negative"] == 10
    assert link["min"] == pytest.approx(-2.0, abs=0.01)
