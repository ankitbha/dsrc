"""eval_run.analyze over synthetic run logs: metrics and gate logic."""

import json

import pytest

from eval_run import analyze, join_phone_log, load_phone_log

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

        assert result["log_integrity"]["missing_ticks"] == 60
        assert result["log_integrity"]["log_complete"] is False
        assert result["overall_pass"] is False

    def test_an_intact_log_still_passes_its_integrity_check(self, tmp_path):
        # The fix must not fail every run: a complete log reports complete.
        integrity = analyze(write_run(tmp_path, [make_tick(i) for i in range(40)]))["log_integrity"]
        assert integrity["log_complete"] is True
        assert integrity["missing_ticks"] == 0
        assert integrity["unparseable_lines"] == 0


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
        inbound, shown = load_phone_log(path)
        assert inbound == [] and shown == []

    def test_an_inbound_non_advisory_frame_is_ignored(self, tmp_path):
        path = write_phone_log(tmp_path, [
            {"dir": "in", "recv_mono_ns": 1, "recv_wall_ns": 1,
             "header": {"ch": "rate_cmd"}},
        ])
        inbound, shown = load_phone_log(path)
        assert inbound == []

    def test_inbound_advisories_and_shown_lines_are_both_picked_out(self, tmp_path):
        advisory = an_inbound_advisory(capture_ns=100, wire_ns=200, recv_ns=300, recv_wall_ns=400)
        shown = a_shown_line(capture_ns=100, shown_ns=350)
        path = write_phone_log(tmp_path, [advisory, shown])
        loaded_advisories, loaded_shown = load_phone_log(path)
        assert loaded_advisories == [advisory]
        assert loaded_shown == [shown]


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
