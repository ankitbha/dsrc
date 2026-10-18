"""Jam metrics split into network and approach parts, and the storage measures (approach
plan sections 5.2 to 5.4, step S6). No SUMO."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.analysis.jam_metrics as jm  # noqa: E402

NETWORK = list(range(12))
APPROACH = list(range(12, 36))
W = 240.0


def _trace(jammed=lambda t, s: False, cap=7440.0, extra=None):
    """A 36-column trace every 10 s from 0 to `cap`: 50 km/h everywhere, 10 km/h where
    `jammed(t, s)`; one vehicle per super-segment."""
    out = []
    t = 0.0
    while t <= cap + 1e-9:
        sample = {"t": t, "speed_kmh": [10.0 if jammed(t, s) else 50.0 for s in range(36)],
                  "density": [0.05] * 36, "count": [1] * 36, "held": 0, "pending": 0,
                  "running": 900, "k_hat_mean": [None] * 36, "k_true_mean": [None] * 36,
                  "plan_mean": [1.0] * 36, "shown_mean": [1.0] * 36,
                  "active_share": [0.0] * 36}
        if extra:
            sample.update(extra(t))
        out.append(sample)
        t += 10.0
    return out


class TestColumns:
    def test_select_columns_keeps_only_the_named_segments(self):
        trace = _trace(lambda t, s: s == 20)
        network = jm.select_columns(trace, NETWORK)
        approach = jm.select_columns(trace, APPROACH)
        assert len(network[0]["speed_kmh"]) == 12 and len(approach[0]["count"]) == 24
        assert approach[0]["speed_kmh"][20 - 12] == 10.0
        assert network[0]["held"] == 0 and network[0]["t"] == 0.0
        assert len(trace[0]["speed_kmh"]) == 36  # the input is not changed

    def test_jam_time_is_additive_over_the_parts(self):
        trace = _trace(lambda t, s: (s in (3, 20) and 1500.0 <= t < 2100.0)
                       or (s == 30 and 3000.0 <= t < 3300.0))
        parts = {name: jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0,
                                          cap=7200.0, columns=columns, offset_s=W)
                 for name, columns in (("all", None), ("network", NETWORK),
                                       ("approach", APPROACH))}
        for key in ("jam_time_min", "jam_time_min_900_3600", "jam_time_min_30kmh"):
            assert parts["all"][key] == pytest.approx(parts["network"][key]
                                                      + parts["approach"][key])
        assert parts["network"]["jam_time_min"] == pytest.approx(10.0)
        assert parts["approach"]["jam_time_min"] == pytest.approx(15.0)

    def test_a_jam_only_on_an_approach_is_an_onset_for_all_but_not_for_the_network(self):
        trace = _trace(lambda t, s: s == 20 and 1500.0 <= t < 1800.0)
        every = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=7200.0,
                                   offset_s=W)
        network = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0,
                                     cap=7200.0, columns=NETWORK, offset_s=W)
        approach = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0,
                                      cap=7200.0, columns=APPROACH, offset_s=W)
        assert every["onset_segment"] == 20 and every["onset_s"] == 1500.0
        assert every["prevented"] is False and network["prevented"] is True
        assert approach["onset_segment"] == 20 - 12
        assert network["jam_time_min"] == 0.0

    def test_the_clock_moves_by_the_offset(self):
        # A spell at [1000, 1200) s is before the shifted start at 1,140 s for its onset.
        trace = _trace(lambda t, s: s == 4 and 1000.0 <= t < 1400.0)
        gated = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=7200.0)
        moved = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=7200.0,
                                   offset_s=W)
        assert gated["onset_s"] == 1000.0 and moved["onset_s"] is None
        assert (moved["start_s"], moved["end_s"], moved["cap_s"]) == (1140.0, 1440.0, 7440.0)
        assert moved["offset_s"] == W
        assert moved["jam_time_min"] == pytest.approx(26 / 6.0)
        assert moved["drained"] is True and moved["drain_s"] == pytest.approx(120.0)

    def test_without_columns_or_offset_the_result_is_the_gated_one(self):
        trace = _trace(lambda t, s: s == 4 and 1000.0 <= t < 1400.0)
        twelve = jm.select_columns(trace, NETWORK)
        out = jm.episode_metrics(twelve, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=7200.0)
        assert "offset_s" not in out and "columns" not in out
        assert out["onset_s"] == 1000.0


SCHEDULE = [{"id": "a", "entry": "140", "depart": 100.0},
            {"id": "b", "entry": "255", "depart": 150.0},
            {"id": "c", "entry": "140", "depart": 400.0}]
TRAVEL = {"140": 190.0, "255": 195.0}


class TestStorage:
    def test_free_flow_occupancy_on_three_vehicles(self):
        times = [0.0, 100.0, 150.0, 289.9, 290.0, 344.9, 345.0, 400.0, 589.0, 590.0]
        assert jm.free_flow_occupancy(SCHEDULE, TRAVEL, times).tolist() == [
            0, 1, 2, 2, 1, 1, 0, 1, 1, 0]

    def test_approach_excess_is_on_approach_minus_free_flow_occupancy(self):
        # 2 vehicles more than free flow on [1,140, 1,740] s, sampled every 10 s.
        rows = [{"id": f"v{k}", "entry": "140", "depart": 1000.0 + 10.0 * k}
                for k in range(200)]
        n_ff = lambda t: jm.free_flow_occupancy(rows, TRAVEL, [t])[0]
        trace = _trace(cap=7440.0, extra=lambda t: {
            "on_approach": int(n_ff(t)) + (2 if 1140.0 <= t <= 1740.0 else 0)})
        out = jm.approach_excess_minutes(trace, rows, TRAVEL, start_s=900.0, cap=7200.0,
                                         offset_s=W)
        assert out == pytest.approx(10.0 * 2 * 61 / 60.0)

    def test_system_excess_charges_running_held_and_pending(self):
        trace = _trace(cap=3600.0, extra=lambda t: {"running": 1000, "held": 3, "pending": 2})
        out = jm.system_excess_minutes(trace, start_s=900.0, cap=3600.0)
        assert out == pytest.approx(10.0 * 1005 * 271 / 60.0)
        rows = [{"id": "a", "entry": "140", "depart": 0.0}]
        always = {"140": 1.0e6}
        shifted = jm.system_excess_minutes(trace, start_s=900.0, cap=3360.0, offset_s=W,
                                           rows=rows, travel_s=always)
        assert shifted == pytest.approx(10.0 * 1004 * 247 / 60.0)


class TestFlags:
    def _queues(self, length, pending=0):
        return [{"t": 0.0, "queue_m": [0.0] * 8, "pending": 0,
                 "slow_near_insertion": [0] * 8},
                {"t": 10.0, "queue_m": [0.0] * 7 + [length], "pending": pending,
                 "slow_near_insertion": [0] * 7 + [2]}]

    def test_the_flag_fires_past_2000_m_and_on_any_pending_vehicle(self):
        assert jm.queue_flags(self._queues(2001.0))["approach_short"] is True
        assert jm.queue_flags(self._queues(1999.0))["approach_short"] is False
        assert jm.queue_flags(self._queues(0.0, pending=1))["approach_short"] is True

    def test_the_flag_reports_its_maxima(self):
        out = jm.queue_flags(self._queues(1500.0, pending=0))
        assert out["queue_max_m"] == [0.0] * 7 + [1500.0]
        assert out["pending_max"] == 0 and out["slow_near_insertion_max"] == [0] * 7 + [2]
        assert out["limit_m"] == 2000.0
        assert jm.queue_flags(self._queues(2500.0), limit_m=4000.0)["approach_short"] is False
