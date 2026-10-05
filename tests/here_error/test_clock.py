import pytest

from here_error import clock


def test_offset_and_spread_on_constant_offset():
    utc = [100.0 + i for i in range(50)]
    wall = [u + 1.2 for u in utc]
    off = clock.estimate_offset(wall, utc)
    assert off.median_s == pytest.approx(1.2)
    assert off.spread_s == pytest.approx(0.0, abs=1e-9)
    assert off.passes_gate
    assert clock.wall_to_utc(500.0, off) == pytest.approx(498.8)


def test_gate_fails_on_two_second_spread():
    utc = [100.0 + i for i in range(100)]
    # Half the series at +1 s, half at +3 s: 5th to 95th percentile spread is 2 s.
    wall = [u + (1.0 if i < 50 else 3.0) for i, u in enumerate(utc)]
    off = clock.estimate_offset(wall, utc)
    assert off.spread_s == pytest.approx(2.0)
    assert not off.passes_gate


def test_gate_ignores_a_few_outliers():
    utc = [float(i) for i in range(200)]
    wall = [u + 1.0 for u in utc]
    wall[10] += 2.0
    wall[11] += 2.0
    assert clock.estimate_offset(wall, utc).passes_gate


def test_empty_series_refused():
    with pytest.raises(ValueError):
        clock.estimate_offset([], [])


def test_dusk_split_is_local_time():
    # 2026-09-08 19:14:59 EDT = 23:14:59Z ; 19:15:00 EDT = 23:15:00Z
    import datetime as dt

    base = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.timezone.utc).timestamp()
    assert not clock.is_after_dusk(base - 1)
    assert clock.is_after_dusk(base)
