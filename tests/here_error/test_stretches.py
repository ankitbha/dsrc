import pytest

from here_fixtures import fix
from here_error.here_times import AnalysedPass
from here_error.models import Pass, PassTimes
from here_error.stretches import build_stretches


def ap(pid, t0, length, run="run_t", dur=100.0):
    fixes = (fix(t0, 0, 0), fix(t0 + dur, length, 0))
    p = Pass(pid, run, "k", "R", 2, fixes, 0.0, length, False, length, length, None)
    t = PassTimes(pid, 0, 0.0, 10.0, True, dur, dur * 1.1, dur * 0.9, 0.1, -0.1, 1.0, 0.99, 5.0, dur - 5,
                  None, None, None, None, None, None, None)
    return AnalysedPass(p, t)


def test_closes_when_minimum_reached_and_sums_times():
    rows = [ap("a", 0, 900), ap("b", 110, 900), ap("c", 220, 900), ap("d", 330, 900)]
    r = build_stretches(rows)
    (s1,) = r.stretches
    assert s1.pass_ids == ("a", "b", "c") and s1.matched_m == 2700
    assert s1.observed_s == 300.0 and s1.here_s == pytest.approx(330.0)
    assert s1.signed_error == pytest.approx(0.1)
    assert r.dropped_passes == 1 and r.dropped_m == 900   # "d" alone is under 2 km


def test_remainder_below_minimum_is_dropped_and_counted():
    rows = [ap("a", 0, 1200), ap("b", 110, 1200), ap("c", 220, 500)]
    r = build_stretches(rows)
    assert [s.pass_ids for s in r.stretches] == [("a", "b")]
    assert r.dropped_passes == 1 and r.dropped_m == 500


def test_gap_over_60_s_breaks_a_stretch():
    rows = [ap("a", 0, 1500), ap("b", 100 + 61, 1500)]   # a ends at 100, b starts at 161
    r = build_stretches(rows)
    assert r.stretches == ()
    assert r.dropped_passes == 2
    rows = [ap("a", 0, 1500), ap("b", 100 + 60, 1500)]   # exactly 60 s is allowed
    assert len(build_stretches(rows).stretches) == 1


def test_maximum_closes_before_a_pass_that_would_exceed_it():
    rows = [ap("a", 0, 1500), ap("b", 110, 3600)]        # 1500 < 2000, adding 3600 -> 5100 > 5000
    r = build_stretches(rows)
    assert [s.pass_ids for s in r.stretches] == [("b",)]   # b alone is 3600 m, a stretch
    assert r.dropped_passes == 1 and r.dropped_m == 1500


def test_runs_are_never_joined():
    rows = [ap("a", 0, 1200, run="run_1"), ap("b", 110, 1200, run="run_2")]
    assert build_stretches(rows).stretches == ()
