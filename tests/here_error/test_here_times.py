import pytest

from here_fixtures import fix, make_body, road, segment
from here_error import here_times as ht
from here_error import load
from here_error.models import Pass, PassTimes

EAST = [(0, 0), (500, 0), (1000, 0)]
SUBS = [
    {"length": 300.0, "speed": 10.0, "freeFlow": 15.0, "jamFactor": 3.0},
    {"length": 400.0, "speed": 20.0, "freeFlow": 25.0, "jamFactor": 1.0},
    {"length": 300.0, "speed": 5.0, "freeFlow": 10.0, "jamFactor": 7.0},
]


def sub_segment(**kw):
    return load.parse_segment(road("Main St", EAST, subs=SUBS, speed=12.0, free=18.0, **kw))


def make_pass(seg, c0=200.0, c1=800.0, t0=1000.0, t1=1060.0, speeds=(10.0, 10.0), key=None, run="run_t", pid="t-001"):
    fixes = (fix(t0, c0, 0, speed=speeds[0]), fix(t1, c1, 0, speed=speeds[1]))
    return Pass(pid, run, seg.key, seg.description, 2, fixes, c0, c1, False, c1 - c0, c1 - c0, None)


def test_sub_segment_integration_matches_hand_calculation():
    seg = sub_segment()
    shape = 1000.0
    # 200..800 crosses: 100 m at 10, 400 m at 20, 100 m at 5 -> 10 + 20 + 20 = 50 s.
    assert ht.travel_time_s(seg, shape, 200, 800, free_flow=False) == pytest.approx(50.0, rel=1e-3)
    # free flow: 100/15 + 400/25 + 100/10
    assert ht.travel_time_s(seg, shape, 200, 800, free_flow=True) == pytest.approx(100 / 15 + 16 + 10, rel=1e-3)
    # jam factor weighted by metres: (100*3 + 400*1 + 100*7) / 600
    assert ht.matched_jam_factor(seg, shape, 200, 800) == pytest.approx(1400 / 600, rel=1e-3)


def test_sub_segment_boundaries_scale_to_the_shape_length():
    seg = sub_segment()
    # The sub-segment lengths sum to 1000 while the shape is 2000 m: boundaries double.
    assert ht.travel_time_s(seg, 2000.0, 0, 2000, free_flow=False) == pytest.approx(600 / 10 + 800 / 20 + 600 / 5)


def test_fallback_to_segment_speed_without_sub_segments():
    seg = segment("Plain", EAST, speed=10.0, free=20.0)
    assert ht.travel_time_s(seg, 1000.0, 200, 700, free_flow=False) == pytest.approx(50.0)
    assert ht.travel_time_s(seg, 1000.0, 200, 700, free_flow=True) == pytest.approx(25.0)


def test_no_speed_gives_none():
    r = road("NoSpeed", EAST)
    del r["currentFlow"]["speed"]
    seg = load.parse_segment(r)
    assert ht.travel_time_s(seg, 1000.0, 0, 500, free_flow=False) is None


def test_signed_error_direction():
    assert ht.signed_error(120.0, 100.0) == pytest.approx(0.2)   # HERE longer than the car
    assert ht.signed_error(80.0, 100.0) == pytest.approx(-0.2)


def test_stopped_seconds():
    seg = segment("S", EAST)
    fixes = tuple(fix(i, 100 + i, 0, speed=0.0 if 2 <= i < 5 else 8.0) for i in range(10))
    p = Pass("x", "r", seg.key, "S", 2, fixes, 100, 109, False, 9, 9, None)
    assert ht.stopped_seconds(p) == 3.0


def body_with(seq, data_t, speed, resp=None):
    seg = segment("Main St", EAST, speed=speed, free=20.0)
    return make_body("run_t", seq, [seg], data_t, resp), seg


def test_reading_selection_rules():
    bodies = [body_with(0, 900.0, 10.0)[0], body_with(1, 960.0, 20.0)[0], body_with(2, 1030.0, 30.0)[0]]
    seg = body_with(0, 0, 1)[1]
    readings = ht.index_readings(bodies)[seg.key]
    first, last = 1000.0, 1060.0
    # Primary: latest data time at or before the first fix.
    r, why = ht.select_primary(readings, first)
    assert r.seq == 1 and why is None
    # Exactly at the first fix counts as at-or-before.
    assert ht.select_primary(readings, 960.0)[0].seq == 1
    # Too old: the latest is more than 300 s before the pass.
    r, why = ht.select_primary(readings, 1400.0 + 160.0)
    assert r is None and why == "reading_too_old"
    # None: every reading is after the pass.
    r, why = ht.select_primary(readings, 800.0)
    assert r is None and why == "no_reading"
    # Sensitivity: nearest data time to the midpoint (1030) is body 2 even though it is after the start.
    assert ht.select_sensitivity(readings, (first + last) / 2).seq == 2


def test_arrived_reading_must_have_reached_the_caller():
    seg = body_with(0, 0, 1)[1]
    bodies = [body_with(0, 900.0, 10.0, resp=950.0)[0],      # arrived 50 s before the pass
              body_with(1, 990.0, 20.0, resp=1090.0)[0]]     # data time before the pass, arrived 90 s after its start
    readings = ht.index_readings(bodies)[seg.key]
    assert ht.select_primary(readings, 1000.0)[0].seq == 1
    assert ht.select_arrived(readings, 1000.0).seq == 0
    assert ht.select_arrived(readings, 920.0) is None


def test_shuffled_pairing_needs_20_minutes_and_takes_the_nearest():
    bodies = [body_with(i, t, 10.0)[0] for i, t in enumerate([0.0, 100.0, 1500.0, 3000.0, 4500.0])]
    seg = body_with(0, 0, 1)[1]
    readings = ht.index_readings(bodies)[seg.key]
    # Pass at 3000..3060: seq 0 is 3000 s before it, seq 1 2900 s, seq 2 1500 s, seq 4 1440 s after it,
    # and seq 3, at its first fix, is 0 s away.
    r = ht.select_shuffled(readings, 3000.0, 3060.0)
    assert r.seq == 4          # 1440 s away, nearer than seq 2 at 1500 s
    assert ht.select_shuffled(readings, 100.0, 160.0).seq == 2      # seq 2 is 1340 s after the pass, seq 3 2840 s
    assert ht.select_shuffled(readings[:2], 50.0, 60.0) is None     # nothing at least 1200 s away


def test_compute_pass_times_end_to_end_and_v4():
    seg = sub_segment()
    p = make_pass(seg)
    bodies = [make_body("run_t", 0, [seg], 900.0, 940.0), make_body("run_t", 1, [seg], 5000.0, 5040.0)]
    idx = ht.index_readings(bodies)
    t = ht.compute_pass_times(p, 1000.0, idx, idx)
    assert t.exclusion is None
    assert t.reading_seq == 0 and t.reading_age_s == 100.0
    assert t.reading_arrived_before_start is True
    assert t.here_s == pytest.approx(50.0, rel=1e-3)
    assert t.observed_s == 60.0
    assert t.signed_error == pytest.approx(-10 / 60, rel=1e-3)       # HERE shorter than the car
    assert t.free_flow_error == pytest.approx((100 / 15 + 26 - 60) / 60, rel=1e-3)
    assert t.arrived_seq == 0 and t.arrived_signed_error == pytest.approx(t.signed_error)
    assert t.shuffled_seq == 1                                       # 3940 s later
    v4 = ht.v4_summary([t])
    assert v4.n == 1 and v4.median_abs_real == pytest.approx(abs(t.signed_error))


def test_pass_with_no_reading_is_excluded_with_reason():
    seg = sub_segment()
    p = make_pass(seg, t0=100.0, t1=160.0)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    t = ht.compute_pass_times(p, 1000.0, idx, idx)
    assert t.exclusion == "no_reading" and t.signed_error is None


def test_reading_without_speed_is_excluded():
    r = road("Main St", EAST)
    del r["currentFlow"]["speed"]
    seg = load.parse_segment(r)
    p = make_pass(seg)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    assert ht.compute_pass_times(p, 1000.0, idx, idx).exclusion == "reading_without_speed"


def test_matching_exclusion_carries_through():
    seg = sub_segment()
    p = make_pass(seg)
    p = Pass(**{**p.__dict__, "exclusion": "v3_failed"})
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    assert ht.compute_pass_times(p, 1000.0, idx, idx).exclusion == "v3_failed"


def test_sub_segments_keep_their_order_along_the_shape():
    # 300 / 400 / 300 m at 10 / 20 / 5 m/s. A matched portion of 100 to 850 m crosses:
    # 200 m at 10 (20 s), 400 m at 20 (20 s), 150 m at 5 (30 s) = 70 s. With the sub-segments laid in
    # reverse order the same portion would take 200/5 + 400/20 + 150/10 = 75 s.
    seg = sub_segment()
    assert ht.travel_time_s(seg, 1000.0, 100, 850, free_flow=False) == pytest.approx(70.0, rel=1e-3)
    # free flow 15 / 25 / 10 m/s: 200/15 + 400/25 + 150/10
    assert ht.travel_time_s(seg, 1000.0, 100, 850, free_flow=True) == pytest.approx(200 / 15 + 16 + 15, rel=1e-3)


def test_stop_threshold_is_1_m_per_s_and_uses_the_earlier_fix():
    seg = segment("S", EAST)
    speeds = [0.9, 1.1, 0.9, 1.1]
    fixes = tuple(fix(i, 100 + i, 0, speed=v) for i, v in enumerate(speeds))
    p = Pass("x", "r", seg.key, "S", 2, fixes, 100, 103, False, 3, 3, None)
    # Intervals 0->1 and 2->3 start at a fix below 1 m/s; 1 m/s itself is not stopped.
    assert ht.stopped_seconds(p) == 2.0


def test_equal_data_times_prefer_the_later_response_and_data_time_beats_response_time():
    seg = body_with(0, 0, 1)[1]
    tie = ht.index_readings([body_with(0, 900.0, 10.0, resp=950.0)[0], body_with(1, 900.0, 20.0, resp=960.0)[0]])[seg.key]
    assert ht.select_primary(tie, 1000.0)[0].seq == 1
    mixed = ht.index_readings([body_with(0, 900.0, 10.0, resp=970.0)[0], body_with(1, 950.0, 20.0, resp=960.0)[0]])[seg.key]
    assert ht.select_primary(mixed, 1000.0)[0].seq == 1          # newest data, although it arrived first
    assert ht.select_sensitivity(mixed, 905.0).seq == 0          # sensitivity goes by data time, not arrival


def test_arrival_flag_compares_with_the_first_fix_not_the_last():
    seg = sub_segment()
    p = make_pass(seg, t0=1000.0, t1=1060.0)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0, 1030.0)])
    assert ht.compute_pass_times(p, 1000.0, idx, idx).reading_arrived_before_start is False


def test_reading_age_is_measured_from_the_first_fix():
    seg = sub_segment()
    p = make_pass(seg, t0=1000.0, t1=1060.0)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 940.0)])
    assert ht.compute_pass_times(p, 1000.0, idx, idx).reading_age_s == 60.0


def test_jam_factor_is_weighted_by_matched_metres():
    seg = sub_segment()
    # The portion 100..600 m covers 200 m of the first sub-segment (jam 3) and 300 m of the second (jam 1).
    assert ht.matched_jam_factor(seg, 1000.0, 100, 600) == pytest.approx((200 * 3 + 300 * 1) / 500)


def test_v4_summary_compares_real_and_shuffled_on_the_same_passes():
    def pt(pid, signed, shuffled):
        return PassTimes(pid, 0, 0.0, 0.0, True, 100.0, 100.0, 70.0, signed, 0.0, 1.0, 0.9, 0.0, 100.0, None, None, None, None, 1, shuffled, None)

    times = [pt("a", 0.10, 0.50), pt("b", -0.20, -0.60), pt("c", 0.05, None), pt("d", 0.30, 0.90)]
    v4 = ht.v4_summary(times)
    assert v4.n == 3
    assert v4.median_abs_real == pytest.approx(0.20)         # passes a, b, d only: |0.10|, |0.20|, |0.30|
    assert v4.median_abs_shuffled == pytest.approx(0.60)    # |0.50|, |0.60|, |0.90|
    assert ht.v4_summary([pt("c", 0.05, None)]).n == 0


UNCAPPED_SUBS = [
    {"length": 300.0, "speed": 10.0, "speedUncapped": 12.0, "freeFlow": 15.0, "jamFactor": 3.0},
    {"length": 400.0, "speed": 20.0, "speedUncapped": 25.0, "freeFlow": 25.0, "jamFactor": 1.0},
    {"length": 300.0, "speed": 5.0, "speedUncapped": 5.0, "freeFlow": 10.0, "jamFactor": 7.0},
]


def test_uncapped_time_differs_from_the_capped_time_where_the_speeds_differ():
    r = road("Main St", EAST, subs=UNCAPPED_SUBS, speed=12.0, free=18.0)
    r["currentFlow"]["speedUncapped"] = 14.0
    seg = load.parse_segment(r)
    # 200..800 m: 100 m, 400 m, 100 m.
    assert ht.travel_time_s(seg, 1000.0, 200, 800, free_flow=False) == pytest.approx(10 + 20 + 20, rel=1e-3)
    assert ht.travel_time_s(seg, 1000.0, 200, 800, free_flow=False, uncapped=True) == pytest.approx(100 / 12 + 400 / 25 + 100 / 5, rel=1e-3)
    p = make_pass(seg)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    t = ht.compute_pass_times(p, 1000.0, idx, idx)
    assert t.here_s == pytest.approx(50.0, rel=1e-3)
    assert t.here_uncapped_s == pytest.approx(100 / 12 + 16 + 20, rel=1e-3)
    assert t.uncapped_signed_error == pytest.approx((t.here_uncapped_s - 60) / 60)
    assert t.signed_error != pytest.approx(t.uncapped_signed_error)


def test_uncapped_time_equals_capped_time_when_the_body_has_no_uncapped_field():
    seg = sub_segment()
    p = make_pass(seg)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    t = ht.compute_pass_times(p, 1000.0, idx, idx)
    assert t.here_uncapped_s == pytest.approx(t.here_s)


def test_unusable_uncapped_speed_falls_back_to_the_capped_time_not_free_flow():
    from dataclasses import replace

    seg = replace(segment("Plain", EAST, speed=10.0, free=20.0), speed_uncapped_mps=0.0)
    p = make_pass(seg, c0=200.0, c1=700.0)
    idx = ht.index_readings([make_body("run_t", 0, [seg], 900.0)])
    t = ht.compute_pass_times(p, 1000.0, idx, idx)
    assert t.here_s == pytest.approx(50.0) and t.free_flow_s == pytest.approx(25.0)
    assert t.here_uncapped_s == pytest.approx(50.0)
