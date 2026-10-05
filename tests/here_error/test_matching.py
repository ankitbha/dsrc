import numpy as np
import pytest

from here_fixtures import drive_east, fix, make_body, segment
from here_error import matching

EAST = [(0, 0), (500, 0), (1000, 0)]
WEST = list(reversed(EAST))
NORTH = [(500, -500), (500, 0), (500, 500)]


def run_match(fixes, *segs):
    body = make_body("run_t", 0, list(segs), 0.0)
    return matching.extract_passes("run_t", tuple(fixes), (body,))


def test_exact_pass_chainage_and_observed_time():
    east = segment("Main St", EAST)
    passes, _ = run_match(drive_east(1000, 100, 700), east)
    (p,) = passes
    assert p.chain_start_m == pytest.approx(100.0, abs=0.5)
    assert p.chain_end_m == pytest.approx(700.0, abs=0.5)
    assert p.matched_m == pytest.approx(600.0, abs=1.0)
    assert p.observed_s == 60.0
    assert len(p.fixes) == 61
    assert p.exclusion is None
    assert p.pass_id.endswith("-001")
    assert p.description == "Main St"


def test_heading_picks_the_direction():
    east = segment("Main St eastbound", EAST)
    west = segment("Main St westbound", WEST)
    (p,) = run_match(drive_east(0, 100, 700), east, west)[0]
    assert p.description == "Main St eastbound"
    westward = [fix(i, 700 - 10 * i, 0, heading=270.0) for i in range(61)]
    (q,) = run_match(westward, east, west)[0]
    assert q.description == "Main St westbound"
    assert q.matched_m == pytest.approx(600.0, abs=1.0)


def test_stop_at_a_signal_stays_in_the_pass():
    east = segment("Main St", EAST)
    cross = segment("Cross St", NORTH)
    fixes = drive_east(0, 100, 500)  # t = 0..40, reaches the crossing
    stop = [fix(41 + i, 505, 3, speed=0.0, heading=None) for i in range(20)]
    after = drive_east(61, 515, 700)
    passes, _ = run_match(fixes + stop + after, east, cross)
    (p,) = passes
    assert p.description == "Main St"
    assert p.observed_s == 79.0  # t=0 .. t=79: the 20 s stop is inside
    assert p.matched_m == pytest.approx(595.0, abs=1.0)


def test_crossing_road_gets_its_own_pass():
    east = segment("Main St", EAST)
    cross = segment("Cross St", NORTH)
    north = [fix(i, 502, -400 + 10 * i, heading=0.0) for i in range(81)]
    (p,) = run_match(north, east, cross)[0]
    assert p.description == "Cross St"
    assert p.matched_m == pytest.approx(800.0, abs=1.0)


def test_reversed_chainage_is_rejected_with_a_reason():
    east = segment("Main St", EAST)
    out = drive_east(0, 100, 600, speed=1.5)  # speed below the heading test
    back = [fix(51 + i, 590 - 10 * i, 0, speed=1.5, heading=270.0) for i in range(20)]
    (p,) = run_match(out + back, east)[0]
    assert p.exclusion == "chainage_reversed"


def test_small_chainage_fallback_is_tolerated():
    east = segment("Main St", EAST)
    fixes = drive_east(0, 100, 600) + [fix(51, 590, 0, speed=0.5, heading=None)] + drive_east(52, 600, 700)
    (p,) = run_match(fixes, east)[0]
    assert p.exclusion is None


def test_pass_under_200_m_is_excluded():
    east = segment("Main St", EAST)
    (p,) = run_match(drive_east(0, 100, 250), east)[0]
    assert p.exclusion == "shorter_than_min"


def test_v3_failure_is_flagged():
    east = segment("Main St", EAST)
    good = run_match(drive_east(0, 100, 700, speed=10.0), east)[0][0]
    assert good.exclusion is None
    assert good.v3_rel_diff == pytest.approx(0.0, abs=1e-3)
    bad = run_match(drive_east(0, 100, 700, speed=15.0), east)[0][0]
    assert bad.exclusion == "v3_failed"
    assert bad.v3_rel_diff == pytest.approx(0.5, abs=0.01)


def test_fixes_beyond_either_end_do_not_join_the_pass():
    east = segment("Main St", EAST)
    # x = -5 and 1005 are within 15 m of the end points but beyond them.
    fixes = [fix(i, -5 + 10 * i, 0) for i in range(102)]  # x = -5 .. 1005
    (p,) = run_match(fixes, east)[0]
    assert p.fixes[0].utc_s == 1.0 and p.fixes[-1].utc_s == 100.0
    assert p.observed_s == pytest.approx(99.0)
    assert p.chain_start_m == pytest.approx(5.0, abs=0.5)
    assert p.chain_end_m == pytest.approx(995.0, abs=0.5)


def test_gap_longer_than_three_seconds_splits_the_pass():
    east = segment("Main St", EAST)
    a = drive_east(0, 100, 400)
    b = drive_east(35, 450, 800)  # fix at t=30 -> 35: 5 s gap
    passes, _ = run_match(a + b, east)
    assert len(passes) == 2


def test_segment_length_flag():
    ok = segment("A", EAST)
    odd = segment("B", [(0, 100), (1000, 100)], stated_length=1100)
    _, cat = run_match(drive_east(0, 100, 700), ok, odd)
    assert not cat[ok.key].length_flag
    assert cat[odd.key].length_flag


def test_off_road_fixes_match_nothing():
    east = segment("Main St", EAST)
    passes, _ = run_match(drive_east(0, 100, 700, y=40.0), east)
    assert passes == []


def test_plot_writes_a_png(tmp_path):
    east = segment("Main St", EAST)
    fixes = drive_east(0, 100, 700)
    passes, cat = run_match(fixes, east)
    out = tmp_path / "v2" / "p.png"
    matching.plot_pass(passes[0], cat, tuple(fixes), out)
    assert out.stat().st_size > 1000


# ---- boundaries and rules that a single straight-road fixture does not exercise ---------------


def test_stopped_car_stays_on_its_segment_when_a_nearer_ramp_appears():
    main = segment("Main St", EAST)
    ramp = segment("Ramp", [(495, 4), (1000, 4)])             # runs 1 m from the fixes, main runs 3 m
    moving = [fix(i, 10 * i, 3, heading=90.0) for i in range(50)]                    # x = 0 .. 490
    stopped = [fix(50 + i, 500, 3, speed=0.0, heading=None) for i in range(20)]
    after = [fix(70 + i, 500 + 10 * i, 3, heading=90.0) for i in range(40)]           # x = 500 .. 890
    passes, _ = run_match(moving + stopped + after, main, ramp)
    assert [p.description for p in passes] == ["Main St"]
    assert passes[0].matched_m == pytest.approx(890.0, abs=1.0)


def test_first_candidate_is_the_nearest_not_the_farthest():
    near = segment("Near", EAST)
    far = segment("Far", [(0, 9), (1000, 9)])
    fixes = [fix(i, 100 + 10 * i, 2) for i in range(60)]
    (p,) = run_match(fixes, near, far)[0]
    assert p.description == "Near"


def test_heading_tolerance_is_45_degrees():
    east = segment("Main St", EAST)
    assert run_match([fix(i, 100 + 10 * i, 0, heading=130.0) for i in range(60)], east)[0]       # 40 degrees off
    assert not run_match([fix(i, 100 + 10 * i, 0, heading=150.0) for i in range(60)], east)[0]   # 60 degrees off
    assert not run_match([fix(i, 100 + 10 * i, 0, heading=270.0) for i in range(60)], east)[0]   # opposite


def test_heading_is_ignored_below_the_minimum_speed():
    east = segment("Main St", EAST)
    slow = [fix(i, 100 + 1.5 * i, 0, speed=1.5, heading=270.0) for i in range(200)]
    assert run_match(slow, east)[0]


def test_match_radius_is_15_m():
    east = segment("Main St", EAST)
    assert run_match([fix(i, 100 + 10 * i, 14.0) for i in range(60)], east)[0]
    assert not run_match([fix(i, 100 + 10 * i, 16.0) for i in range(60)], east)[0]
    assert not run_match([fix(i, 100 + 10 * i, 20.0) for i in range(60)], east)[0]


def test_chainage_is_taken_at_the_first_and_last_fix_not_the_extremes():
    east = segment("Main St", EAST)
    start = [fix(0, 105, 0, speed=1.5, heading=None), fix(1, 100, 0, speed=1.5, heading=None)]   # jitter backwards
    body = [fix(2 + i, 110 + 10 * i, 0) for i in range(60)]                                        # to x = 700
    end = [fix(62, 695, 0, speed=1.5, heading=None)]
    (p,) = run_match(start + body + end, east)[0]
    assert p.chain_start_m == pytest.approx(105.0, abs=0.5)
    assert p.chain_end_m == pytest.approx(695.0, abs=0.5)


def test_v3_integrates_speed_with_the_trapezoid_rule():
    east = segment("Main St", EAST)
    # Constant acceleration of 1 m/s^2 from rest: speed t, position t^2 / 2.
    fixes = [fix(t, t * t / 2.0, 0, speed=float(t)) for t in range(31)]
    (p,) = run_match(fixes, east)[0]
    assert p.path_speed_m == pytest.approx(450.0)
    assert p.exclusion is None


def test_gap_of_exactly_three_seconds_stays_in_the_pass():
    east = segment("Main St", EAST)
    fixes = drive_east(0, 100, 400) + drive_east(33, 430, 700)    # 30 -> 33: 3 s
    assert len(run_match(fixes, east)[0]) == 1


def test_match_radius_applies_inside_the_bounding_box_of_a_bent_segment():
    bent = segment("Bent", [(0, 0), (500, 0), (500, 500)])
    # (100..400, 20) lies inside the segment's bounding box but 20 m from it.
    assert not run_match([fix(i, 100 + 10 * i, 20.0) for i in range(30)], bent)[0]
    assert run_match([fix(i, 100 + 10 * i, 10.0) for i in range(30)], bent)[0]


def test_plot_title_and_labels_never_present_a_description_as_a_road():
    from dataclasses import replace

    east = segment("Cross St", EAST)
    passes, _ = run_match(drive_east(0, 100, 700), east)
    p = replace(passes[0], road_id="R07")
    title = matching.pass_title(p)
    assert title.splitlines()[0] == f"{p.pass_id}, road R07, segment ending at Cross St"
    assert matching.neighbour_label(east) == "ends at Cross St"
    assert "road (not assigned)" in matching.pass_title(passes[0])
