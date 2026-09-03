"""eval_run.analyze over synthetic run logs: metrics and gate logic."""

import json

import pytest

from eval_run import (
    GATE_TICK_COVERAGE_MISSING_FRACTION,
    _failure_lines,
    _join_failure_episodes,
    _log_health_lines,
    analyze,
    join_phone_log,
    load_phone_log,
    render_markdown,
    stage_timings,
)

T0 = 1_750_000_000.0
RATE_HZ = 30.0


def make_tick(i: int, *, e2e_ms=20.0, ego_speed=20.0, leader_gap=35.0,
              gps_fresh=True, leader_rel_measured=True, jetson_ms=None,
              link_ms=None, field_sources=None, missingness=0.3) -> dict:
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
        "field_sources": field_sources if field_sources is not None else {
            "leader_relative_speed": "measured" if (has_leader and leader_rel_measured)
            else "fallback_neutral",
        },
        "obs_diagnostics": {
            "missingness": missingness,
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


def test_no_leader_and_no_tracks_are_absent_not_a_zeroed_out_gap(tmp_path):
    """B1: `pctl([])` returns all zeros, and a `leader_gap_m` of `{p50: 0.0,
    mean: 0.0}` states the vehicle ahead is touching the bumper -- not the
    same fact as "no leader was ever observed". `stage_timings` already
    avoids exactly this shape for its own stats; the perception block must
    too.
    """
    ticks = [make_tick(i, leader_gap=None) for i in range(60)]
    run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
    result = analyze(run_dir)
    assert result["perception"]["leader_gap_m"] is None
    assert result["perception"]["track_lifetime_s"] is None

    md = render_markdown(result, [])
    assert "no leader observed on any tick" in md
    assert "no tracks" in md
    assert "gap p50 0.0 m" not in md
    assert "0 tracks, lifetime p50 0.0 s" not in md


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


class TestATruncatedLogIsNotACompleteRun:
    """A run that lost records must not certify as one that did not."""

    def _truncate_last_record(self, run_dir):
        path = run_dir / "metadata.jsonl"
        lines = path.read_text().splitlines()
        # A half-written final record, which is what an unflushed 1 MiB buffer leaves
        # behind -- at ~1.5 KB a record that is some seven hundred ticks, not one.
        lines[-1] = lines[-1][: len(lines[-1]) // 2]
        path.write_text("\n".join(lines) + "\n")

    def test_a_truncated_final_record_is_counted_and_fails_the_run(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(40)])
        self._truncate_last_record(run_dir)
        result = analyze(run_dir)

        integrity = result["log_integrity"]
        assert integrity["unparseable_lines"] >= 1
        assert integrity["log_complete"] is False
        assert result["overall_pass"] is False, (
            "a run whose log was truncated certified as a complete one"
        )

    def test_a_log_short_of_the_count_the_run_reported_fails(self, tmp_path):
        # The evidence was sitting in summary.json the whole time, unread.
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(30)],
                            summary={"ticks": 90, "camera_dropped_frames": 0,
                                     "policy_trained": False})
        result = analyze(run_dir)

        assert result["log_integrity"]["tick_ids_absent_from_log"] == 60
        assert result["log_integrity"]["log_complete"] is False
        assert result["overall_pass"] is False

    def test_an_intact_log_still_passes_its_integrity_check(self, tmp_path):
        # The fix must not fail every run: a complete log reports complete.
        integrity = analyze(write_run(tmp_path, [make_tick(i) for i in range(40)]))["log_integrity"]
        assert integrity["log_complete"] is True
        assert integrity["tick_ids_absent_from_log"] == 0
        assert integrity["unparseable_lines"] == 0

    def test_a_summary_with_no_tick_count_is_unmeasurable_not_complete(self, tmp_path):
        """A5: `summary["ticks"]` absent leaves `shortfall` at `None`, which
        the old code folded straight into `log_complete: True` -- an
        unmeasured drive certifying as a measured, complete one. It must
        instead read as a third state that does not pass the verdict.
        """
        # `summary={}` would fall back to `write_run`'s own default (an empty
        # dict is falsy) -- a truthy dict with no "ticks" key is what actually
        # exercises the unmeasurable branch.
        run_dir = write_run(
            tmp_path, [make_tick(i) for i in range(40)],
            summary={"camera_dropped_frames": 0, "policy_trained": False},
        )
        result = analyze(run_dir)
        assert result["log_integrity"]["ticks_the_run_reported"] is None
        assert result["log_integrity"]["log_complete"] is None
        assert result["overall_pass"] is False


class TestTickCoverageGate:
    """A5: `log_integrity` compares the log against what the run itself
    reported producing, so a real outage that reduces both sides together is
    invisible to it -- `missing_ticks` reads zero by construction. This gate
    reads the gap distribution instead, so a real interruption fails it even
    when the run's own reported tick count and every ratio-based gate agree
    nothing is missing.
    """

    def test_a_real_outage_fails_the_gate_and_the_overall_verdict(self, tmp_path):
        ticks = [make_tick(i) for i in range(90)]
        # A 10 s gap after tick 40 (300 ticks' worth at 30 Hz) -- every tick
        # keeps its own relative spacing, just shifted forward in time.
        for t in ticks[40:]:
            t["t_wall"] += 10.0
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)

        gate = result["gates"]["tick_coverage"]
        assert gate["pass"] is False
        assert gate["value"] > GATE_TICK_COVERAGE_MISSING_FRACTION
        # The ratio-based gates this outage does not touch still read healthy --
        # the point of the finding: nothing else in the report says a span of
        # the drive is missing.
        assert result["gates"]["latency_jetson_p95"]["pass"] is True
        assert result["overall_pass"] is False

    def test_an_uninterrupted_run_passes_the_gate(self, tmp_path):
        ticks = [make_tick(i) for i in range(90)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        gate = result["gates"]["tick_coverage"]
        assert gate["pass"] is True
        assert gate["value"] == 0.0


class TestTickCoverageDistinguishesRateChangeFromLoss:
    """A9/A10/A11: a single population median assumes one characteristic
    period. A controller legitimately running at more than one rate has no
    such period (A9/A10); a loss spread evenly across a drive never widens
    any one gap past a multiple of the median at all (A11). Both are size
    questions a gap-shape heuristic answers wrong in opposite directions.
    """

    def test_a_legitimate_rate_change_reports_zero_missing(self, tmp_path):
        ticks = [make_tick(i) for i in range(90)]
        # The last 20 ticks held a real, slower cadence (6 Hz instead of the
        # base 30 Hz) for long enough to be its own regime -- every tick
        # that was produced is still here, tick_id 0..89 with no hole, just
        # spaced out differently in wall time.
        base_t_wall = ticks[69]["t_wall"]
        for offset, t in enumerate(ticks[70:], start=1):
            t["t_wall"] = base_t_wall + offset * (1.0 / 6.0)
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        gate = result["gates"]["tick_coverage"]
        assert gate["value"] == 0.0
        assert gate["pass"] is True

    def test_an_evenly_distributed_loss_is_caught_through_tick_id_holes(self, tmp_path):
        all_ticks = [make_tick(i) for i in range(90)]
        # Every third tick deleted -- no single gap stands out (each
        # surviving gap is a uniform multiple of the base period), so a
        # heuristic reading only gap SIZE cannot see this at all. tick_id
        # itself holes exactly where a tick used to be.
        kept = [t for i, t in enumerate(all_ticks) if i % 3 != 2]
        run_dir = write_run(
            tmp_path, kept, scenario=scenario_record(),
            summary={"ticks": len(kept), "camera_dropped_frames": 0, "policy_trained": False},
        )
        result = analyze(run_dir)
        gate = result["gates"]["tick_coverage"]
        assert gate["pass"] is False
        assert gate["value"] == pytest.approx(30 / 90, abs=0.01)
        assert result["overall_pass"] is False


class TestTickCoverageGateIsNoneWhenUnmeasurable:
    """B13: `coverage_fraction is not None else None` mutated to `else True`
    left every existing test passing -- a drive whose coverage cannot be
    computed at all (fewer than three ticks) would then render the gate as
    a pass rather than not applicable, the same silent-pass shape
    `log_complete`'s three-state fix (A5) exists to rule out.
    """

    def test_a_two_tick_drive_leaves_the_gate_not_applicable(self, tmp_path):
        ticks = [make_tick(0), make_tick(1)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        gate = result["gates"]["tick_coverage"]
        assert gate["pass"] is None
        assert gate["value"] is None


# -- joining the phone's own log against the Jetson's ticks -----------------


def write_phone_log(tmp_path, lines: list[dict]):
    path = tmp_path / "session.jsonl"
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def an_inbound_advisory(*, capture_ns: int, wire_ns: int | None, recv_ns: int,
                         recv_wall_ns: int) -> dict:
    header = {"ch": "advisory", "seq": 1, "t_mono_ns": recv_ns - 1_000_000,
              "t_wall_ns": recv_wall_ns, "n": 0, "t_capture_mono_ns": capture_ns}
    if wire_ns is not None:
        header["t_wire_mono_ns"] = wire_ns
    return {"dir": "in", "recv_mono_ns": recv_ns, "recv_wall_ns": recv_wall_ns, "header": header}


def a_shown_line(*, capture_ns: int, shown_ns: int) -> dict:
    return {"dir": "shown", "t_capture_mono_ns": capture_ns, "shown_mono_ns": shown_ns}


def a_timebase_estimate(*, source: str, t_wall: float, offset_ns: int = 0,
                         rtt_min_ns: int = 10_000_000, t_reference_ns: int = 0) -> dict:
    return {
        "type": "timebase_estimate", "source": source, "t_wall": t_wall,
        "estimate_id": 1, "offset_ns": offset_ns, "t_reference_ns": t_reference_ns,
        "rtt_min_ns": rtt_min_ns, "offset_samples": 10,
        "skew_ppm": None, "skew_stderr_ppm": None, "skew_samples": 0,
    }


class TestLoadPhoneLog:

    def test_an_outbound_header_is_not_mistaken_for_either_shape(self, tmp_path):
        # A bare frame header, exactly what SessionLog writes for anything it
        # sent -- it carries no `dir` key at all.
        path = write_phone_log(tmp_path, [
            {"ch": "camera", "seq": 1, "t_mono_ns": 1, "t_wall_ns": 1, "n": 0},
        ])
        inbound, shown, failures = load_phone_log(path)
        assert inbound == [] and shown == [] and failures == []

    def test_an_inbound_non_advisory_frame_is_ignored(self, tmp_path):
        path = write_phone_log(tmp_path, [
            {"dir": "in", "recv_mono_ns": 1, "recv_wall_ns": 1,
             "header": {"ch": "rate_cmd"}},
        ])
        inbound, shown, failures = load_phone_log(path)
        assert inbound == []

    def test_inbound_advisories_and_shown_lines_are_both_picked_out(self, tmp_path):
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=200, recv_ns=300, recv_wall_ns=400)
        shown = a_shown_line(capture_ns=100, shown_ns=350)
        path = write_phone_log(tmp_path, [advisory, shown])
        loaded_advisories, loaded_shown, loaded_failures = load_phone_log(path)
        assert loaded_advisories == [advisory]
        assert loaded_shown == [shown]
        assert loaded_failures == []

    def test_a_fail_line_is_picked_out_and_ignored_by_the_other_two_shapes(self, tmp_path):
        fail_line = {
            "dir": "fail", "at_mono_ns": 1, "at_wall_ns": 2,
            "kind": "link.dial_failed", "n": 3, "detail": "ConnectException",
        }
        path = write_phone_log(tmp_path, [fail_line])
        inbound, shown, failures = load_phone_log(path)
        assert inbound == [] and shown == []
        assert failures == [fail_line]


class TestJoinPhoneLog:

    def test_matches_on_the_exact_capture_stamp(self):
        tick = {"tick_id": 5, "t_capture_mono_ns": 100, "stages": {"detect": {"ms": 1.0}}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=None, recv_ns=300, recv_wall_ns=400)
        result = join_phone_log([tick], [], [advisory], [])

        assert result["matched"] == 1
        assert result["unmatched"] == 0
        row = result["rows"][0]
        assert row["tick_id"] == 5
        assert row["stages"]["detect"] == {"ms": 1.0}
        assert "return" in row["stages"] and "render" in row["stages"]

    def test_an_advisory_with_no_matching_tick_is_counted(self):
        advisory = an_inbound_advisory(capture_ns=999, wire_ns=None, recv_ns=300, recv_wall_ns=400)
        result = join_phone_log([], [], [advisory], [])
        assert result["matched"] == 0
        assert result["unmatched"] == 1
        assert result["advisories_seen_by_the_phone"] == 1

    def test_a_jetson_tick_whose_advisory_never_returned_still_gets_a_row(self):
        """The join checked in the other direction: a tick with no matching
        inbound advisory used to vanish from `rows` entirely, along with
        every Jetson-side stage it carried -- and `rows` is the whole
        population `stage_timings` is computed over. The dropped tick's own
        stages must survive; only `return`/`render` -- the two stages only
        the phone can supply -- are absent.
        """
        matched_tick = {"tick_id": 1, "t_capture_mono_ns": 100, "stages": {"detect": {"ms": 1.0}}}
        orphaned_tick = {"tick_id": 2, "t_capture_mono_ns": 200, "stages": {"detect": {"ms": 2.0}}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=None, recv_ns=300, recv_wall_ns=400)

        result = join_phone_log([matched_tick, orphaned_tick], [], [advisory], [])

        assert result["matched"] == 1
        assert result["unmatched"] == 0
        assert result["jetson_ticks_with_no_advisory"] == 1
        assert len(result["rows"]) == 2

        orphaned_row = next(r for r in result["rows"] if r["tick_id"] == 2)
        assert orphaned_row["stages"]["detect"] == {"ms": 2.0}  # the Jetson-side stage survives
        assert orphaned_row["stages"]["return"]["basis"] == "absent"
        assert orphaned_row["stages"]["return"]["ms"] is None
        assert orphaned_row["stages"]["render"]["basis"] == "absent"
        assert orphaned_row["stages"]["render"]["ms"] is None

        # The matched tick's own reason differs -- it went through the real
        # `_return_stage` path (absent here only because this advisory
        # carries no wire stamp), not the generic no-advisory-at-all reason.
        matched_row = next(r for r in result["rows"] if r["tick_id"] == 1)
        assert matched_row["stages"]["return"]["reason"] == "advisory carried no wire stamp"

    def test_a_run_with_no_orphaned_ticks_reports_zero_not_omitted(self):
        tick = {"tick_id": 1, "t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=None, recv_ns=300, recv_wall_ns=400)
        result = join_phone_log([tick], [], [advisory], [])
        assert result["jetson_ticks_with_no_advisory"] == 0

    def test_return_is_absent_without_a_wire_stamp(self):
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=None, recv_ns=300, recv_wall_ns=400)
        row = join_phone_log([tick], [], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"
        assert "wire stamp" in row["stages"]["return"]["reason"]

    def test_return_is_absent_with_no_estimate_near_enough(self):
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=200, recv_ns=300, recv_wall_ns=400)
        # An estimate that exists, but nowhere near this receipt's wall time.
        estimate = a_timebase_estimate(source="round_trip", t_wall=1_000_000.0)
        row = join_phone_log([tick], [estimate], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"

    def test_return_recovers_a_planted_one_way_delay_within_its_bound(self):
        """A planted world -- a known Jetson wire departure, a known clock
        offset between the two devices, and a known one-way delay for this
        specific advisory -- so the recovered value can be checked against a
        truth rather than only checked for being present. A prior version of
        this test asserted only `basis == "converted"` and that the fields
        were non-None, which a sign flip and a wrong-direction conversion
        both satisfy just as well as the correct arithmetic does.
        """
        from transport.timebase import NS_PER_S, TimebaseEstimator, TimeSyncSample

        base_ns = 4_000_000_000_000
        estimator = TimebaseEstimator(mono_clock=lambda: base_ns + 20 * NS_PER_S)
        for i in range(10):
            t1 = base_ns + i * NS_PER_S
            estimator.add(TimeSyncSample(
                exchange_id=i, t1_local_send_ns=t1,
                t2_remote_recv_ns=t1 + 5_000_000 + 2_000_000_000,
                t3_remote_send_ns=t1 + 5_000_000 + 2_000_000_000,
                t4_local_recv_ns=t1 + 10_000_000,
            ))
        estimate = estimator.estimate
        wall_now = 1_755_000_000.0
        timebase_estimate = {
            "type": "timebase_estimate", "source": "round_trip", "t_wall": wall_now,
            **estimate.to_record(),
        }

        # The wire departure sits exactly at the estimate's own reference instant, so
        # extrapolation drift is zero and the bound is exactly half the round trip --
        # a clean number to check the recovered value against.
        wire_ns = estimate.t_reference_ns
        truth_delay_ns = 7_000_000  # the advisory's actual one-way travel time
        # The phone's own clock reading of that same wire departure, plus how long
        # the packet actually took to arrive.
        recv_ns = wire_ns + estimate.offset_ns + truth_delay_ns
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=wire_ns, recv_ns=recv_ns, recv_wall_ns=int(wall_now * 1e9),
        )
        tick = {"t_capture_mono_ns": 100, "stages": {}}

        row = join_phone_log([tick], [timebase_estimate], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "converted"
        assert row["stages"]["return"]["source"] == "round_trip"
        truth_ms = truth_delay_ns / 1e6
        ms = row["stages"]["return"]["ms"]
        bound_ms = row["stages"]["return"]["bound_ms"]
        assert ms == pytest.approx(truth_ms, abs=1e-3)
        assert abs(ms - truth_ms) <= bound_ms

    def test_the_round_trip_estimate_wins_even_when_a_one_way_line_is_nearer(self):
        """Both a round-trip and a one-way line exist for this receipt's wall
        time, at different distances and with different offsets. Removing the
        source filter used to let whichever line was nearest answer a
        round-trip request while still being stamped `source: "round_trip"` --
        a one-way number, with a one-way bound, presented as if it were
        bounded by half a round trip. The one-way line here is deliberately
        the nearer one, so a filter-less join would pick it.
        """
        wall_now = 1_755_000_000.0
        nearer_one_way = a_timebase_estimate(
            source="one_way", t_wall=wall_now, offset_ns=999_000_000, rtt_min_ns=500_000_000,
        )
        nearer_one_way["estimate_id"] = 77
        farther_round_trip = a_timebase_estimate(
            source="round_trip", t_wall=wall_now - 20.0, offset_ns=2_000_000, rtt_min_ns=10_000_000,
        )
        farther_round_trip["estimate_id"] = 42
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            # recv_ns == the round-trip offset, so a round-trip conversion lands
            # exactly on t_reference_ns (0) and wire_ns (0): return_ms == 0.0.
            capture_ns=100, wire_ns=0, recv_ns=2_000_000, recv_wall_ns=int(wall_now * 1e9),
        )

        row = join_phone_log(
            [tick], [farther_round_trip, nearer_one_way], [advisory], [],
        )["rows"][0]
        assert row["stages"]["return"]["basis"] == "converted"
        assert row["stages"]["return"]["source"] == "round_trip"
        assert row["stages"]["return"]["estimate_id"] == 42
        assert row["stages"]["return"]["ms"] == pytest.approx(0.0, abs=1e-6)

    def test_return_skips_an_estimate_the_live_adapter_would_have_refused(self):
        # `usable=False` is what the live gate (staleness, the RTT ceiling, too
        # few samples) looked like at the moment this line was written -- an
        # offline join must refuse it exactly as the live conversion did.
        wall_now = 1_755_000_000.0
        unusable = a_timebase_estimate(source="round_trip", t_wall=wall_now, offset_ns=0)
        unusable["usable"] = False
        unusable["why_not_usable"] = "only 1 samples in the offset window"
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [unusable], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"
        assert row["stages"]["return"]["reason"] == (
            "no usable timebase estimate near this receipt's wall time"
        )

    def test_an_old_log_without_the_usable_field_falls_back_to_the_sample_floor(self):
        # A run logged before `usable` existed carries neither it nor
        # `why_not_usable` at all -- the only signal left is the same
        # sample-count floor the live gate checks first.
        wall_now = 1_755_000_000.0
        old_style = a_timebase_estimate(source="round_trip", t_wall=wall_now, offset_ns=0)
        old_style["offset_samples"] = 2  # below MIN_OFFSET_SAMPLES, no `usable` key
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [old_style], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"

    def test_an_old_log_round_trip_line_above_the_rtt_ceiling_is_refused(self):
        # Unlike staleness, the RTT ceiling is a property of the estimate itself --
        # `rtt_min_ns` is written into every persisted line, old format or new -- so
        # an old-format line above the ceiling must be refused exactly as the live
        # gate would have refused it, not accepted just because `usable` is absent.
        from transport.timebase import MAX_ACCEPTABLE_RTT_NS

        wall_now = 1_755_000_000.0
        old_style = a_timebase_estimate(
            source="round_trip", t_wall=wall_now, offset_ns=0,
            rtt_min_ns=MAX_ACCEPTABLE_RTT_NS + 1,  # no `usable` key: an old-format line
        )
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [old_style], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"

    def test_an_old_log_one_way_line_above_the_round_trip_ceiling_still_converts(self):
        # `rtt_min_ns` means something different on a one-way line -- a spread of
        # observed delays, not half a round trip -- and `OneWayEstimator` has no
        # ceiling clause on it. Applying the round-trip bound here would refuse an
        # estimate the live one-way path accepts.
        from transport.timebase import MAX_ACCEPTABLE_RTT_NS

        wall_now = 1_755_000_000.0
        old_style = a_timebase_estimate(
            source="one_way", t_wall=wall_now, offset_ns=0,
            rtt_min_ns=MAX_ACCEPTABLE_RTT_NS + 1,  # no `usable` key: an old-format line
        )
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [old_style], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "converted"
        assert row["stages"]["return"]["source"] == "one_way"

    def test_return_does_not_convert_against_the_previous_sessions_estimate(self):
        """Both estimators are rebuilt whole on every redial, so their
        `estimate_id` counters restart at 1 on the new session -- an estimate
        left over from the phone that just hung up can sit within the wall-time
        match window of an advisory the NEW phone sent. `session_id` is what
        tells the two apart when wall time alone cannot.
        """
        wall_now = 1_755_000_000.0
        stale = a_timebase_estimate(source="round_trip", t_wall=wall_now, offset_ns=5_000_000_000)
        stale["session_id"] = 1
        tick = {"t_capture_mono_ns": 100, "stages": {}, "session_id": 2}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [stale], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "absent"

    def test_return_converts_against_the_current_sessions_estimate(self):
        wall_now = 1_755_000_000.0
        current = a_timebase_estimate(source="round_trip", t_wall=wall_now, offset_ns=0)
        current["session_id"] = 2
        tick = {"t_capture_mono_ns": 100, "stages": {}, "session_id": 2}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=0, recv_ns=0, recv_wall_ns=int(wall_now * 1e9),
        )
        row = join_phone_log([tick], [current], [advisory], [])["rows"][0]
        assert row["stages"]["return"]["basis"] == "converted"

    def test_render_is_measured_between_two_phone_clock_stamps(self):
        # Nanosecond deltas large enough to survive the record's millisecond
        # rounding -- a real render segment is tens of milliseconds, not
        # hundreds of nanoseconds.
        recv_ns, shown_ns = 300_000_000, 550_000_000
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(
            capture_ns=100, wire_ns=None, recv_ns=recv_ns, recv_wall_ns=400
        )
        shown = a_shown_line(capture_ns=100, shown_ns=shown_ns)
        row = join_phone_log([tick], [], [advisory], [shown])["rows"][0]
        assert row["stages"]["render"]["basis"] == "measured"
        assert row["stages"]["render"]["ms"] == pytest.approx((shown_ns - recv_ns) / 1e6)
        assert row["stages"]["render"]["clock"] == "phone"

    def test_render_is_absent_when_the_advisory_was_never_shown(self):
        # Named for what the join actually knows -- no `advisory_shown` line
        # exists for this capture stamp -- not for a specific mechanism.
        # `AdvisoryHolder.accept` replaces `latest` unconditionally with no
        # supersession count, so a newer advisory arriving first is the
        # ordinary cause here, not an expiry; a dropped `SessionLog` line or a
        # null `liveLog` produce the same absence and this join cannot tell
        # any of the three apart.
        tick = {"t_capture_mono_ns": 100, "stages": {}}
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=None, recv_ns=300, recv_wall_ns=400)
        row = join_phone_log([tick], [], [advisory], [])["rows"][0]
        assert row["stages"]["render"]["basis"] == "absent"
        assert row["stages"]["render"]["reason"] == "no advisory_shown line for this capture stamp"


def test_analyze_with_a_phone_log_produces_the_phone_join(tmp_path):
    ticks = [make_tick(i) for i in range(10)]
    ticks[3]["t_capture_mono_ns"] = 555
    ticks[3]["stages"] = {"detect": {"ms": 1.0}}
    run_dir = write_run(tmp_path, ticks)

    phone_log = write_phone_log(tmp_path, [
        an_inbound_advisory(capture_ns=555, wire_ns=None, recv_ns=600, recv_wall_ns=700),
        a_shown_line(capture_ns=555, shown_ns=650),
    ])

    result = analyze(run_dir, phone_log)
    join = result["phone_join"]
    assert join["matched"] == 1
    assert join["rows"][0]["stages"]["render"]["basis"] == "measured"


def test_analyze_without_a_phone_log_leaves_the_join_absent(tmp_path):
    run_dir = write_run(tmp_path, [make_tick(i) for i in range(10)])
    result = analyze(run_dir)
    assert result["phone_join"] is None


def _append_lines(run_dir, records: list[dict]) -> None:
    with open(run_dir / "metadata.jsonl", "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestLoadRecordsDataclass:
    """D18: `load_records` returns a frozen dataclass, one field per record
    type. Written because four same-typed adjacent members -- `thermal_samples`,
    `thermal_events`, `failure_scans`, `failure_events` -- is exactly where a
    swapped assignment survives most tests and passes anyway."""

    def test_each_record_type_lands_in_its_own_field(self, tmp_path):
        from eval_run import load_records

        run_dir = tmp_path / "loaded"
        run_dir.mkdir()
        path = run_dir / "metadata.jsonl"
        with open(path, "w") as f:
            for record in [
                {"type": "thermal_sample", "t_mono": 1.0, "marker": "TS"},
                {"type": "thermal_event", "t_mono": 1.0, "marker": "TE"},
                {"type": "failure_scan", "seq": 1, "t_mono": 1.0, "marker": "FS"},
                {"type": "failure_event", "phase": "open", "episode_id": 1, "marker": "FE"},
            ]:
                f.write(json.dumps(record) + "\n")
        loaded = load_records(path)
        assert [r["marker"] for r in loaded.thermal_samples] == ["TS"]
        assert [r["marker"] for r in loaded.thermal_events] == ["TE"]
        assert [r["marker"] for r in loaded.failure_scans] == ["FS"]
        assert [r["marker"] for r in loaded.failure_events] == ["FE"]


class TestPreTask38RunIsNotAFailedDrive:

    def test_a_run_with_no_failure_records_reports_failures_none(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(10)])
        result = analyze(run_dir)
        assert result["failures"] is None
        assert result["overall_pass"] == (
            analyze(run_dir)["overall_pass"]  # unchanged by this task's own logic
        )
        markdown = render_markdown(result, [])
        assert "## Failures" not in markdown


class TestFailuresSectionRendersOnRealRecordShapes:
    """Task 33 and task 36 both shipped a measurement with no surface. This
    builds a real `metadata.jsonl` with ticks, scans and two episodes and
    asserts the `## Failures` heading, an episode line with a duration, and
    the words `NOT EVALUABLE` on a fixture where a source was unreadable."""

    def test_the_section_renders_with_a_duration_and_not_evaluable(self, tmp_path):
        ticks = [make_tick(i) for i in range(5)]
        for t in ticks:
            t["failures"] = {
                "open": [], "open_n": 0, "episodes": 2, "scan_age_s": 0.4,
                "basis": "measured", "unreadable_n": 1, "reason": None,
            }
        run_dir = write_run(tmp_path, ticks, summary={
            "ticks": len(ticks), "camera_dropped_frames": 0, "policy_trained": False,
            "failures": {
                "scan": {"passes": 5, "seq_last": 5,
                         "interval_s": {"p50": 1.0, "p95": 1.0, "max": 1.0},
                         "sources_n": 30},
                "sources": {
                    "gps.not_fresh": {
                        "status": "fired", "passes_attempted": 5, "passes_readable": 5,
                        "episodes": 1, "total": 12, "by_reason": {"stale": 12},
                        "first_t_mono": 1.0, "last_t_mono": 4.0,
                        "events_written": 2, "episodes_not_kept": 0,
                    },
                    "phone.dropped": {
                        "status": "not_evaluable", "passes_attempted": 5, "passes_readable": 2,
                        "episodes": 0, "total": 0, "by_reason": {},
                        "missing": ["telemetry"], "first_t_mono": None, "last_t_mono": None,
                        "events_written": 0, "episodes_not_kept": 0,
                    },
                },
                "outcomes": {"recovered": 1},
                "counter_went_backwards": {}, "blind_ticks": 0, "pipeline_exception": None,
            },
        })
        _append_lines(run_dir, [
            {"type": "failure_scan", "seq": 1, "t_wall": 1.0, "t_mono": 1.0,
             "session_id": "a1b2", "ticks_seen": 5, "sources_n": 30,
             "sources_readable": 29, "unreadable": ["phone.dropped"], "open": []},
            {"type": "failure_event", "phase": "open", "episode_id": 1,
             "source": "gps.not_fresh", "reason": "stale", "device": "jetson",
             "t_wall": 1.0, "t_mono": 1.0, "basis": "measured", "bound_s": 1.0,
             "session_id": None, "tick_id": 1, "channel": None, "value": 4.2,
             "first_pass_n": 1, "detail": "gps_age_s 4.213"},
            {"type": "failure_event", "phase": "close", "episode_id": 1,
             "source": "gps.not_fresh", "device": "jetson",
             "t_wall": 4.0, "t_mono": 4.0, "outcome": "recovered",
             "duration_s": 3.0, "n": 12, "last_t_mono": 3.9,
             "close_after_s": 3.0, "basis": "measured", "bound_s": 1.0,
             "session_id": None},
        ])
        result = analyze(run_dir)
        assert result["failures"] is not None
        assert len(result["failures"]["episodes"]) == 1
        assert result["failures"]["episodes"][0]["duration_s"] == 3.0

        markdown = render_markdown(result, [])
        assert "## Failures" in markdown
        assert "3.0" in markdown or "duration" in markdown.lower()
        assert "NOT EVALUABLE" in markdown
        assert "gps.not_fresh" in markdown


class TestCounterWentBackwardsRendersEachStep:
    """C1: `_backwards` changed from `dict[str, dict]` to `dict[str, list[dict]]`
    (accumulating one entry per occurrence instead of the last one
    overwriting the rest), and `_failure_lines` was not updated to match --
    `step.get("from")` on a list element that is itself a list raised
    `AttributeError`, so `render_markdown` never returned and no
    `report.md` was written for any drive whose counter went backwards. The
    only fixture this block had was an empty dict, whose loop body never
    runs either way."""

    def test_two_backwards_steps_on_one_source_both_render(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(2)], summary={
            "ticks": 2, "camera_dropped_frames": 0, "policy_trained": False,
            "failures": {
                "scan": {"passes": 2, "seq_last": 2,
                         "interval_s": {"p50": 1.0, "p95": 1.0, "max": 1.0},
                         "sources_n": 30},
                "sources": {},
                "outcomes": {},
                "counter_went_backwards": {
                    "camera.dropped_unconsumed": [
                        {"from": 40, "to": 3, "t_mono": 1.0},
                        {"from": 9, "to": 1, "t_mono": 2.0},
                    ],
                },
                "blind_ticks": 0, "pipeline_exception": None,
            },
        })
        result = analyze(run_dir)
        markdown = render_markdown(result, [])
        assert "## Failures" in markdown
        assert "camera.dropped_unconsumed: counter went backwards, 40 -> 3" in markdown
        assert "camera.dropped_unconsumed: counter went backwards, 9 -> 1" in markdown


class TestFiredRowQuantityMatchesItsDeclaredCumulativeFlag:
    """m1: `pipeline.exception` was declared `cumulative=False` in the
    registry while `note_pipeline_exception` credits one real occurrence per
    call, the same way `camera.blind_ticks` (`cumulative=True`) does. A
    source's declared flag is what `_failure_lines` reads to choose between
    "occurrences" and "passes with the condition active" -- disagreeing with
    what `total` actually counts made the rendered report describe three
    exceptions as three seconds the condition happened to still be true."""

    def test_a_cumulative_source_reports_occurrences_not_passes(self):
        failures = {
            "summary": {
                "scan": {"passes": 3, "seq_last": 3, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {
                    "pipeline.exception": {
                        "status": "fired", "passes_attempted": 3, "passes_readable": 3,
                        "episodes": 1, "total": 3, "by_reason": {"RuntimeError": 3},
                        "cumulative": True, "first_t_mono": 1.0, "last_t_mono": 3.0,
                        "events_written": 2, "episodes_not_kept": 0,
                    },
                },
                "outcomes": {}, "counter_went_backwards": {},
                "blind_ticks": 0, "pipeline_exception": "RuntimeError",
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "pipeline.exception: FIRED -- 1 episode(s), 3 occurrences" in text
        assert "passes with the condition active" not in text


class TestFiredLineNamesAnUnobservableOutcome:
    """B12: a source that went unwatchable while its condition was still
    active closes its episode with outcome `unobservable`, not `recovered`
    -- but the FIRED line rendered only the duration ("longest 3.0 s"),
    which a reader reads as a resolution. The aggregate `outcomes` line
    does carry `1 unobservable`; the per-source line did not.
    """

    def test_an_unobservable_outcome_and_the_unreadable_pass_count_are_named(self):
        failures = {
            "episodes": [
                {"source": "phone.here_errors", "duration_s": 3.0, "outcome": "unobservable"},
            ],
            "summary": {
                "scan": {"passes": 301, "seq_last": 301, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {
                    "phone.here_errors": {
                        "status": "fired", "passes_attempted": 301, "passes_readable": 297,
                        "episodes": 1, "total": 4, "by_reason": {}, "cumulative": False,
                    },
                },
                "outcomes": {"unobservable": 1}, "counter_went_backwards": {},
                "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "longest 3.0 s (unobservable)" in text
        assert "unreadable on 4 of 301 passes" in text

    def test_a_recovered_outcome_names_neither(self):
        failures = {
            "episodes": [
                {"source": "phone.here_errors", "duration_s": 3.0, "outcome": "recovered"},
            ],
            "summary": {
                "scan": {"passes": 301, "seq_last": 301, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {
                    "phone.here_errors": {
                        "status": "fired", "passes_attempted": 301, "passes_readable": 301,
                        "episodes": 1, "total": 4, "by_reason": {}, "cumulative": False,
                    },
                },
                "outcomes": {"recovered": 1}, "counter_went_backwards": {},
                "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "longest 3.0 s" in text
        assert "(recovered)" not in text
        assert "unreadable on" not in text


class TestNotEvaluableLineNamesTheUnreadableCount:
    """D3: the NOT EVALUABLE line rendered `passes_readable` in the slot
    that names how many passes the source was NOT readable -- the correct
    quantity is `passes_attempted - passes_readable`. On the drive,
    `phone.here_errors` was unreadable on 2 of 301 passes and the line
    asserted 299; `wire.decode_errors` and `wire.send_rejected` were
    unreadable on 80 and the line asserted 221."""

    def test_the_numerator_is_the_unreadable_pass_count_not_the_readable_one(self):
        failures = {
            "summary": {
                "scan": {"passes": 301, "seq_last": 301, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {
                    "phone.here_errors": {
                        "status": "not_evaluable", "passes_attempted": 301, "passes_readable": 299,
                        "episodes": 0, "total": 0, "by_reason": {}, "missing": ["session"],
                    },
                    "wire.decode_errors": {
                        "status": "not_evaluable", "passes_attempted": 301, "passes_readable": 221,
                        "episodes": 0, "total": 0, "by_reason": {}, "missing": ["session"],
                    },
                },
                "outcomes": {}, "counter_went_backwards": {},
                "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "phone.here_errors: NOT EVALUABLE on 2 of 301 passes" in text
        assert "wire.decode_errors: NOT EVALUABLE on 80 of 301 passes" in text
        assert "NOT EVALUABLE on 299 of 301 passes" not in text
        assert "NOT EVALUABLE on 221 of 301 passes" not in text


class TestNotEvaluableWithNoReasonNamesItsOwnAbsence:
    """B2: a `not_evaluable` row with an empty (or absent) `missing` list
    rendered `-- missing ;`, on a record that names no reason at all --
    reachable on a log from a build that predates `missing` being written.
    Every other row on every other drive carries a real reason; this one
    must at least say it carries none, not print an empty list as though it
    were one.
    """

    def test_an_empty_missing_list_renders_a_named_absence_not_a_bare_semicolon(self):
        failures = {
            "summary": {
                "scan": {"passes": 301, "seq_last": 301, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {
                    "phone.here_errors": {
                        "status": "not_evaluable", "passes_attempted": 301, "passes_readable": 2,
                        "episodes": 0, "total": 0, "by_reason": {}, "missing": [],
                    },
                },
                "outcomes": {}, "counter_went_backwards": {},
                "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "missing ;" not in text
        assert "missing a reason this record does not carry" in text


class TestEpisodesLineNamesTwoQuantitiesWhenTheyDisagree:
    """D6: `episodes` (the open/close record pairs a reader can join from
    the log -- only a source with `event_records=True` writes one) and the
    summary's own `outcomes` total (every episode that closed, on every
    source) are two different populations that happen to agree whenever no
    `event_records=False` source closed an episode. `31 episodes: 34
    recovered` presented one number standing in for both."""

    def test_matching_counts_render_the_single_number_line(self):
        failures = {
            "episodes": [{"source": "gps.not_fresh", "duration_s": 1.0}],
            "summary": {
                "scan": {"passes": 1, "seq_last": 1, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {},
                "outcomes": {"recovered": 1},
                "counter_went_backwards": {}, "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "- 1 episodes: 1 recovered" in text

    def test_disagreeing_counts_name_both_populations(self):
        failures = {
            "episodes": [{"source": "gps.not_fresh", "duration_s": 1.0} for _ in range(31)],
            "summary": {
                "scan": {"passes": 1, "seq_last": 1, "interval_s": {"p50": 1.0, "max": 1.0}, "sources_n": 30},
                "sources": {},
                "outcomes": {"recovered": 30, "open_at_end": 1, "unobservable": 3},
                "counter_went_backwards": {}, "blind_ticks": 0, "pipeline_exception": None,
            },
        }
        text = "\n".join(_failure_lines(failures))
        assert "31 episodes recorded" in text
        assert "34 episodes closed in total" in text
        assert "3 closed on a source with no event record" in text
        # The old single-number line must not appear -- it would silently
        # claim the two populations are one.
        assert "31 episodes: " not in text


class TestLogHealthReachesTheReport:

    def test_a_dead_writer_is_named_in_the_failures_section(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        (run_dir / "log_health.json").write_text(json.dumps({
            "t_wall": 1.0, "t_mono": 1.0, "dropped_records": 7,
            "writer_failure": "OSError: No space left on device",
            "queue_depth": 50000, "thread_alive_at_close": False,
            "path": "metadata.jsonl", "bytes_on_disk": 1234,
        }))
        result = analyze(run_dir)
        assert result["failures"]["log_health"]["dropped_records"] == 7
        markdown = render_markdown(result, [])
        assert "## Failures" in markdown
        assert "7 records dropped" in markdown
        assert "writer failed" in markdown


class TestLogHealthAbsentWriterFailureIsUnknownNotHealthy:
    """B7: `log_health.get("writer_failure") is None` reads both "the field
    is present and null" (genuinely healthy) and "the field is entirely
    absent" (a log from before it existed) as the same "writer healthy"
    line. Absent is not the same fact as healthy.
    """

    def test_an_absent_field_reads_as_unknown(self):
        text = "\n".join(_log_health_lines({"log_health": {}}))
        assert "writer healthy" not in text
        assert "unknown" in text
        assert "0 records dropped" in text

    def test_a_present_null_field_still_reads_as_healthy(self):
        text = "\n".join(_log_health_lines({"log_health": {"writer_failure": None}}))
        assert "writer healthy" in text

    def test_a_present_failure_still_reads_as_failed(self):
        text = "\n".join(
            _log_health_lines({"log_health": {"writer_failure": "OSError: boom"}})
        )
        assert "writer failed: OSError: boom" in text


class TestPhoneFailuresJoinOffline:

    def test_no_phone_log_says_not_read(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        _append_lines(run_dir, [
            {"type": "failure_scan", "seq": 1, "t_wall": 1.0, "t_mono": 1.0,
             "session_id": None, "ticks_seen": 3, "sources_n": 30,
             "sources_readable": 30, "unreadable": [], "open": []},
        ])
        result = analyze(run_dir)
        markdown = render_markdown(result, [])
        assert "phone-side failures: not read" in markdown

    def test_a_supplied_phone_log_reports_the_kind_breakdown(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        _append_lines(run_dir, [
            {"type": "failure_scan", "seq": 1, "t_wall": 1.0, "t_mono": 1.0,
             "session_id": None, "ticks_seen": 3, "sources_n": 30,
             "sources_readable": 30, "unreadable": [], "open": []},
        ])
        phone_log = write_phone_log(tmp_path, [
            {"dir": "fail", "at_mono_ns": 1, "at_wall_ns": 2,
             "kind": "link.dial_failed", "n": 3, "detail": "ConnectException"},
        ])
        result = analyze(run_dir, phone_log)
        markdown = render_markdown(result, [])
        assert "phone (offline)" in markdown
        assert "link.dial_failed 3" in markdown

    def test_a_suppressed_count_is_added_to_n_not_dropped(self, tmp_path):
        # M9(a): D12's compensation -- an occurrence the per-second rate cap
        # held back rides forward on the next accepted line's `suppressed`
        # field. Summing `n` alone throws that count away.
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        _append_lines(run_dir, [
            {"type": "failure_scan", "seq": 1, "t_wall": 1.0, "t_mono": 1.0,
             "session_id": None, "ticks_seen": 3, "sources_n": 30,
             "sources_readable": 30, "unreadable": [], "open": []},
        ])
        phone_log = write_phone_log(tmp_path, [
            {"dir": "fail", "at_mono_ns": 1, "at_wall_ns": 2,
             "kind": "link.dial_failed", "n": 1, "detail": "x", "suppressed": 0},
            {"dir": "fail", "at_mono_ns": 2_000_000_000, "at_wall_ns": 2,
             "kind": "link.dial_failed", "n": 1, "detail": "x", "suppressed": 99},
        ])
        result = analyze(run_dir, phone_log)
        markdown = render_markdown(result, [])
        assert "link.dial_failed 101" in markdown

    def test_a_kind_at_its_lifetime_cap_is_flagged_as_possibly_incomplete(self, tmp_path):
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        _append_lines(run_dir, [
            {"type": "failure_scan", "seq": 1, "t_wall": 1.0, "t_mono": 1.0,
             "session_id": None, "ticks_seen": 3, "sources_n": 30,
             "sources_readable": 30, "unreadable": [], "open": []},
        ])
        phone_log = write_phone_log(tmp_path, [
            {"dir": "fail", "at_mono_ns": i * 1_000_000_000, "at_wall_ns": 2,
             "kind": "link.dial_failed", "n": 1, "detail": "x", "suppressed": 0}
            for i in range(64)
        ])
        result = analyze(run_dir, phone_log)
        markdown = render_markdown(result, [])
        assert "link.dial_failed 64" in markdown
        assert "lifetime cap" in markdown

    def test_a_jetson_log_predating_this_feature_still_shows_phone_failures(self, tmp_path):
        # MINOR: `failures_result` returns `None` for a Jetson log with no
        # tick block, no scan/event records and no summary -- `analyze` used
        # to skip assigning `phone` in exactly that case, so a real
        # `--phone-log` was silently dropped rather than shown on its own.
        run_dir = write_run(tmp_path, [make_tick(i) for i in range(3)])
        phone_log = write_phone_log(tmp_path, [
            {"dir": "fail", "at_mono_ns": 1, "at_wall_ns": 2,
             "kind": "link.dial_failed", "n": 3, "detail": "ConnectException"},
        ])
        result = analyze(run_dir, phone_log)
        assert result["failures"] is not None
        markdown = render_markdown(result, [])
        assert "## Failures" in markdown
        assert "predates the failure event log" in markdown
        assert "link.dial_failed 3" in markdown
        # Must not read as a truncated run -- that message names a different
        # cause and would send a reader looking for a wiring bug that is not
        # there.
        assert "did not reach its normal teardown" not in markdown


class TestJoinFailureEpisodesKeepsNAndFirstPassNSeparate:
    """M9(b): `n` used to mean two different quantities depending on whether
    the episode had closed -- occurrences over the whole episode when it
    had, movement on the opening pass alone when it had not -- under one
    key. A reader could not tell which it was looking at without also
    checking `closed`."""

    def test_a_closed_episode_reports_n_from_the_close_record(self):
        events = [
            {"phase": "open", "episode_id": 1, "source": "gps.not_fresh", "reason": "stale",
             "device": "jetson", "t_mono": 1.0, "first_pass_n": 1},
            {"phase": "close", "episode_id": 1, "source": "gps.not_fresh", "outcome": "recovered",
             "duration_s": 3.0, "n": 12, "t_mono": 4.0},
        ]
        episode = _join_failure_episodes(events)[0]
        assert episode["closed"] is True
        assert episode["n"] == 12
        assert episode["first_pass_n"] == 1

    def test_an_open_episode_reports_n_as_none_not_first_pass_n(self):
        # The log was truncated before this episode's close record -- `n`
        # (occurrences over the whole episode) is genuinely unknown, and
        # standing in `first_pass_n` (the opening pass's own movement) used
        # to claim a number for it anyway.
        events = [
            {"phase": "open", "episode_id": 2, "source": "gps.not_fresh", "reason": "stale",
             "device": "jetson", "t_mono": 1.0, "first_pass_n": 5},
        ]
        episode = _join_failure_episodes(events)[0]
        assert episode["closed"] is False
        assert episode["n"] is None
        assert episode["first_pass_n"] == 5


class TestStageTimings:
    """The table's whole purpose is that a stage which was not measured cannot be
    read as one that was, so each basis is checked for what it contributes."""

    def test_an_instant_contributes_no_duration(self):
        # `capture` is a point in time and carries `ms: 0.0` as a placeholder. Counting
        # it as a duration reports a stage that took no time, which on a real run put
        # `capture | n 900 | min 0.0 | mean 0.0` in the table.
        ticks = [{"stages": {"capture": {"ms": 0.0, "basis": "instant", "clock": "phone"}}}
                 for _ in range(5)]
        out = stage_timings(ticks)["capture"]
        assert out["basis"] == {"instant": 5}
        assert out["stats"] is None, "an instant must not report duration statistics"

    def test_an_absent_stage_carries_its_reason_and_no_value(self):
        ticks = [
            {"stages": {"transport": {"ms": None, "basis": "absent", "clock": "cross",
                                      "reason": "only 3 samples in the offset window"}}},
            {"stages": {"transport": {"ms": 20.0, "basis": "converted", "clock": "cross",
                                      "bound_ms": 4.0}}},
        ]
        out = stage_timings(ticks)["transport"]
        assert out["basis"] == {"absent": 1, "converted": 1}
        assert out["absent_reasons"] == {"only 3 samples in the offset window": 1}
        # One value, not two: the absent tick contributes nothing rather than a zero
        # that would halve the mean.
        assert out["stats"]["n"] == 1
        assert out["stats"]["mean"] == pytest.approx(20.0)
        assert out["bound_ms"]["n"] == 1

    def test_a_measured_stage_reports_the_values_it_was_given(self):
        ticks = [{"stages": {"detect": {"ms": v, "basis": "measured", "clock": "jetson"}}}
                 for v in (10.0, 20.0, 30.0)]
        out = stage_timings(ticks)["detect"]
        assert out["stats"]["n"] == 3
        assert out["stats"]["mean"] == pytest.approx(20.0)
        assert out["absent_reasons"] == {}

    def test_a_stage_absent_on_every_tick_reports_no_statistics(self):
        # The case that must not render as zeros: nothing was measured at all.
        ticks = [{"stages": {"render": {"ms": None, "basis": "absent", "clock": "phone",
                                        "reason": "no advisory_shown line"}}}
                 for _ in range(4)]
        out = stage_timings(ticks)["render"]
        assert out["stats"] is None
        assert out["absent_reasons"] == {"no advisory_shown line": 4}


def _grounded_field_sources() -> dict:
    """A real 39-key `field_sources` map, produced by the actual builder --
    not hand-typed, so this fixture cannot drift from what
    `ObservationBuilder` actually emits.
    """
    from perception.observation_builder import BuilderConfig, ObservationBuilder
    from sensors.gps_reader import GpsFix

    builder = ObservationBuilder(BuilderConfig())
    result = None
    for i in range(5):
        t = 1000.0 + i * 0.1
        fix = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0, heading_deg=90.0,
                    fix_quality=1, num_sats=8, hdop=1.0, t_mono=t, t_wall=0.0)
        result = builder.build([], fix, t)
    return result.field_sources


class TestObservationProvenance:
    """`observation`'s four new keys, computed from each tick's own
    `field_sources` -- which every log back to the beginning carries, so an
    old, 1-key fixture reports `covers_encoder: False` rather than crashing
    (the `jetson_ms_source` precedent), and a full 39-key one reports the
    real rollup.
    """

    def test_a_pre_task_36_style_fixture_reports_covers_encoder_false(self, tmp_path):
        # `make_tick`'s own default `field_sources` is a single key -- never
        # the full 33- or 39-field map -- which is exactly the shape a log
        # missing this task's coverage takes.
        ticks = [make_tick(i) for i in range(10)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["covers_encoder"] is False
        assert obs["provenance_fields"] == 1

    def test_a_full_map_reports_covers_encoder_true_and_by_source_sums_to_one(self, tmp_path):
        sources = _grounded_field_sources()
        ticks = [make_tick(i, field_sources=sources) for i in range(10)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["provenance_fields"] == 39
        assert obs["covers_encoder"] is True
        # Each class's fraction is independently rounded to three places, so
        # the sum is close to but not always exactly 1.0.
        assert sum(obs["by_source"].values()) == pytest.approx(1.0, abs=0.01)
        assert "derived_empty" in obs["by_source"]

    def test_fields_by_source_names_the_derived_empty_fields(self, tmp_path):
        sources = _grounded_field_sources()
        ticks = [make_tick(i, field_sources=sources) for i in range(5)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        derived_empty = result["observation"]["fields_by_source"]["derived_empty"]
        assert "local_density_bin" in derived_empty
        assert derived_empty["local_density_bin"] == pytest.approx(1.0)

    def test_report_md_names_the_by_source_line_and_derived_empty_fields(self, tmp_path):
        sources = _grounded_field_sources()
        ticks = [make_tick(i, field_sources=sources) for i in range(5)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        report = render_markdown(result, [])
        assert "provenance covers 39 of 39 encoder slots" in report
        assert "by source:" in report
        assert "local_density_bin" in report

    def test_a_pre_task_36_style_fixture_still_renders_a_report(self, tmp_path):
        # Does not crash on the shape every pre-task-36 log has, and states
        # the incomplete coverage rather than omitting the line.
        ticks = [make_tick(i) for i in range(5)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        report = render_markdown(result, [])
        assert "encoder-field missingness" in report
        assert "provenance covers 1 of 39 encoder slots" in report

    def test_a_mixed_run_reports_the_mixture_instead_of_the_first_ticks_size(self, tmp_path):
        # `by_source` pools every tick's `field_sources` regardless of its
        # size, so reading `provenance_fields`/`covers_encoder` off only the
        # first tick describes a run that is not actually uniform.
        sources = _grounded_field_sources()
        ticks = (
            [make_tick(i) for i in range(5)]
            + [make_tick(i, field_sources=sources) for i in range(5, 10)]
        )
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["provenance_fields_mixed"] is True
        assert obs["covers_encoder"] is False
        # Pins the "first tick" half of the name: the 1-key ticks are first,
        # so this is 1, not 39 (the last tick's size) and not 5 (the number
        # of distinct sizes) -- `render_markdown` below prints this number
        # as "first tick has {pf}", and nothing else makes that sentence true.
        assert obs["provenance_fields"] == 1

    def test_the_mixed_guard_still_refuses_coverage_with_the_full_map_first(self, tmp_path):
        # Same mixture as above, reordered: the 39-key ticks come first this
        # time. `covers_encoder` must still be False -- a run whose maps are
        # not all the same size is not a run with settled coverage, whichever
        # tick happens to be first. (With the 1-key ticks first, the union of
        # every name ever seen already equals the full 39-name set, because
        # that one key is itself one of the 39 -- so this order is the one
        # that actually exercises the mixed short-circuit rather than passing
        # by coincidence.)
        sources = _grounded_field_sources()
        ticks = (
            [make_tick(i, field_sources=sources) for i in range(5)]
            + [make_tick(i) for i in range(5, 10)]
        )
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["provenance_fields_mixed"] is True
        assert obs["covers_encoder"] is False
        assert obs["provenance_fields"] == 39

    def test_covers_encoder_is_false_on_a_same_size_name_swap(self, tmp_path):
        # The defect `_covers_encoder` was written to catch, reproduced at
        # the surface an operator reads: a map the same SIZE as the full
        # contract but not the same set of NAMES is not coverage. Deleting
        # "ego_speed" and adding "not_a_real_slot" keeps the count at 39.
        sources = dict(_grounded_field_sources())
        del sources["ego_speed"]
        sources["not_a_real_slot"] = "measured"
        ticks = [make_tick(i, field_sources=sources) for i in range(10)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["provenance_fields"] == 39
        assert obs["provenance_fields_mixed"] is False
        assert obs["covers_encoder"] is False
        assert "ego_speed" not in obs["fields_by_source"].get("measured", {})

    def test_an_empty_field_sources_tick_still_counts_toward_the_mixture(self, tmp_path):
        # A tick with no provenance at all (`field_sources == {}`) used to be
        # skipped when sizing the run, so a run half carrying no provenance
        # and half carrying the full map read as uniform and complete.
        sources = _grounded_field_sources()
        ticks = (
            [make_tick(i, field_sources={}) for i in range(5)]
            + [make_tick(i, field_sources=sources) for i in range(5, 10)]
        )
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["provenance_fields"] == 0
        assert obs["provenance_fields_mixed"] is True
        assert obs["covers_encoder"] is False

    def test_report_md_names_the_mixture(self, tmp_path):
        sources = _grounded_field_sources()
        ticks = (
            [make_tick(i) for i in range(5)]
            + [make_tick(i, field_sources=sources) for i in range(5, 10)]
        )
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        report = render_markdown(result, [])
        assert "varies across ticks" in report


class TestMissingnessSpread:
    """A mean alone hides a bimodal (or, here, trimodal) missingness run --
    two ticks can average to a percentage neither one produced. The report
    line has to carry the range and, when there are only a few, how many
    distinct values actually occurred.
    """

    def test_report_md_names_the_spread_and_distinct_value_count(self, tmp_path):
        # Three ticks at each of 0.6, 0.7, 0.8: mean 70.0%, min 60.0%, p50
        # 70.0%, p95 80.0%, max 80.0%, 3 distinct values -- chosen so p50 and
        # p95 land on two different values and neither is the minimum, the
        # same shape a mean-only line cannot distinguish from a run where
        # every tick sat at 70.0%.
        ticks = (
            [make_tick(i, missingness=0.6) for i in range(3)]
            + [make_tick(i, missingness=0.7) for i in range(3, 6)]
            + [make_tick(i, missingness=0.8) for i in range(6, 9)]
        )
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        obs = result["observation"]
        assert obs["missingness"]["mean"] == pytest.approx(0.7)
        report = render_markdown(result, [])
        line = next(l for l in report.splitlines() if "encoder-field missingness" in l)
        # The old wording stays exactly as it read before -- comparability
        # against the two runs already recorded depends on it -- with the
        # spread added alongside it, not in place of it.
        assert "mean 70.0%" in line
        assert "min 60.0%" in line
        assert "p50 70.0%" in line
        assert "p95 80.0%" in line
        assert "max 80.0%" in line
        assert "3 distinct values" in line

    def test_a_uniform_run_names_one_distinct_value_singular(self, tmp_path):
        ticks = [make_tick(i, missingness=0.3) for i in range(5)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        report = render_markdown(result, [])
        line = next(l for l in report.splitlines() if "encoder-field missingness" in l)
        assert "1 distinct value" in line
        assert "1 distinct values" not in line

    def test_many_distinct_values_are_not_named_individually(self, tmp_path):
        # 20 ticks, each its own missingness value: naming every one of them
        # would not tell a reader the metric is discrete, so past the small-
        # count cutoff the count itself is omitted rather than printed as a
        # number that swamps the line.
        ticks = [make_tick(i, missingness=round(0.3 + 0.01 * i, 3)) for i in range(20)]
        run_dir = write_run(tmp_path, ticks, scenario=scenario_record())
        result = analyze(run_dir)
        report = render_markdown(result, [])
        line = next(l for l in report.splitlines() if "encoder-field missingness" in l)
        assert "distinct value" not in line
        assert "min " in line and "max " in line
