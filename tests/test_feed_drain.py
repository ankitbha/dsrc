"""The drain-time metric on hand-built traces. No SUMO."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.sumo import drain
from src.sumo.drain import (
    drain_summary, drain_time, fit_three_piece_cdf, jam_indicator, jam_segment_minutes,
    jam_threshold, kaplan_meier, onset, thresholds_from_traces,
)


def _trace(end: int, *, jam_until: int = 0, held_until: int = 0, pending_until: int = 0,
           start: int = 10, segments: int = 3, jam_from: int = 0):
    """Samples every 10 s; segment 1 is jammed (8 km/h) on [jam_from, jam_until)."""
    trace = []
    for t in range(start, end + 1, 10):
        jammed = jam_from <= t < jam_until
        speeds = [45.0] * segments
        if jammed:
            speeds[1] = 8.0
        trace.append({"t": float(t), "speed_kmh": speeds, "count": [5] * segments,
                      "held": 3 if t < held_until else 0,
                      "pending": 1 if t < pending_until else 0})
    return trace


class TestTheCdfFit:
    def test_bimodal_speeds_put_the_breakpoints_between_the_modes(self):
        rng = np.random.default_rng(0)
        speeds = np.concatenate([rng.normal(15.0, 3.0, 4000), rng.normal(50.0, 3.0, 6000)])
        fit = fit_three_piece_cdf(speeds)
        # The plan expected s1 in [12, 22]. The least-squares three-piece line through
        # this CDF puts s1 at the plateau's start, 23.5 km/h, because the first piece
        # runs from 0 km/h and must also cover the empty range below the jam mode.
        assert 15.0 < fit.s1 < 32.5 < fit.s2 < 50.0
        assert (fit.s1, fit.s2) == (23.5, 43.5)
        assert jam_threshold(fit) < fit.s1
        assert fit.residual >= 0.0

    def test_the_returned_pair_beats_every_pair_with_s1_in_12_to_22(self):
        rng = np.random.default_rng(0)
        speeds = np.concatenate([rng.normal(15.0, 3.0, 4000), rng.normal(50.0, 3.0, 6000)])
        grid = np.arange(0.0, math.ceil(speeds.max() / 0.5) * 0.5 + 0.25, 0.5)
        cdf = np.searchsorted(np.sort(speeds), grid, side="right") / len(speeds)

        def sse(a, b):
            design = np.stack([np.ones_like(grid), grid, np.maximum(grid - a, 0.0),
                               np.maximum(grid - b, 0.0)], axis=1)
            coefficients, *_ = np.linalg.lstsq(design, cdf, rcond=None)
            return float(((design @ coefficients - cdf) ** 2).sum())

        fit = fit_three_piece_cdf(speeds)
        assert sse(fit.s1, fit.s2) == pytest.approx(fit.residual)
        best_in_plan_range = min(sse(a, b) for a in np.arange(12.0, 22.5, 0.5)
                                 for b in np.arange(40.0, 55.5, 0.5))
        assert fit.residual < best_in_plan_range

    def test_the_fit_is_the_least_squares_one_on_the_grid(self):
        # A CDF that is exactly three-piece linear with knots at 10 and 40 is recovered.
        rng = np.random.default_rng(1)
        speeds = np.concatenate([rng.uniform(0.0, 10.0, 5000), rng.uniform(10.0, 40.0, 1000),
                                 rng.uniform(40.0, 60.0, 4000)])
        fit = fit_three_piece_cdf(speeds)
        assert abs(fit.s1 - 10.0) <= 1.0 and abs(fit.s2 - 40.0) <= 1.0

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError):
            fit_three_piece_cdf([])


class TestJamIndicator:
    def test_jammed_needs_vehicles_and_a_speed_under_the_threshold(self):
        trace = [{"t": 10.0, "speed_kmh": [10.0, None, 40.0], "count": [3, 0, 2],
                  "held": 0, "pending": 0},
                 {"t": 20.0, "speed_kmh": [10.0, 5.0, 40.0], "count": [0, 1, 2],
                  "held": 0, "pending": 0}]
        assert jam_indicator(trace, 20.0).tolist() == [[True, False, False],
                                                       [False, True, False]]

    def test_a_per_segment_threshold(self):
        trace = [{"t": 10.0, "speed_kmh": [10.0, 10.0], "count": [1, 1], "held": 0,
                  "pending": 0}]
        assert jam_indicator(trace, np.array([5.0, 15.0])).tolist() == [[False, True]]


class TestDrainTime:
    def test_jams_ending_at_2000_and_held_clearing_at_2300_drain_at_2300(self):
        trace = _trace(9000, jam_until=2000, held_until=2300)
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (1100.0, True)

    def test_pending_until_2600_moves_the_drain_to_2600(self):
        trace = _trace(9000, jam_until=2000, held_until=2300, pending_until=2600)
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (1400.0, True)

    def test_the_hold_period_is_counted_from_the_last_jammed_sample(self):
        # Last jammed sample is 2490; no jam on [t_d - 120, t_d] first holds at 2620.
        trace = _trace(9000, jam_until=2500)
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (1420.0, True)

    def test_jammed_at_the_cap_is_censored(self):
        trace = _trace(9000, jam_until=9001)
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (7800.0, False)

    def test_pending_forced_to_one_moves_the_drain_to_the_cap(self):
        trace = _trace(9000, jam_until=2000)
        for sample in trace:
            sample["pending"] = 1
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (7800.0, False)

    def test_the_threshold_is_live(self):
        trace = _trace(9000, jam_until=2000)
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0)[0] == 920.0
        # At 5 km/h the 8 km/h samples are not jams, so it drains at t_0 + H.
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 5.0)[0] == 120.0

    def test_a_trace_that_stops_before_the_cap_is_refused(self):
        with pytest.raises(ValueError):
            drain_time(_trace(5000), 1200.0, 120.0, 9000.0, 20.0)


class TestOnsetAndMinutes:
    def test_onset_is_the_first_jammed_sample_from_the_start(self):
        trace = _trace(3000, jam_from=950, jam_until=1500)
        assert onset(trace, 900.0, 20.0) == 950.0
        assert onset(_trace(3000), 900.0, 20.0) is None

    def test_jam_segment_minutes(self):
        trace = _trace(3000, jam_from=1000, jam_until=1600)  # 60 jammed samples
        assert jam_segment_minutes(trace, 20.0, 900.0, 3000.0) == pytest.approx(10.0)


class TestKaplanMeier:
    def test_the_hand_values_with_one_censored(self):
        km = kaplan_meier([10.0, 20.0, 30.0, 40.0], [True, True, False, True])
        assert km.times == [10.0, 20.0, 30.0, 40.0]
        assert km.survival == pytest.approx([0.75, 0.5, 0.5, 0.0])
        assert km.median == 20.0
        assert km.n == 4

    def test_a_median_never_reached_is_reported_as_beyond_the_cap(self):
        km = kaplan_meier([10.0, 7800.0, 7800.0], [True, False, False])
        assert km.median is None
        assert km.survival[-1] == pytest.approx(2 / 3)

    def test_ties_between_events_and_censoring_count_the_event_first(self):
        km = kaplan_meier([10.0, 10.0, 20.0], [True, False, True])
        assert km.survival == pytest.approx([2 / 3, 0.0])


class TestThresholds:
    def test_pooled_per_segment_and_fixed_thresholds_are_all_reported(self):
        rng = np.random.default_rng(2)
        traces = []
        for _ in range(3):
            trace = []
            for t in range(10, 9001, 10):
                jam = rng.random() < 0.4
                trace.append({"t": float(t), "count": [4, 4, 0],
                              "speed_kmh": [float(rng.normal(12 if jam else 50, 3)),
                                            float(rng.normal(12 if jam else 50, 3)), None],
                              "held": 0, "pending": 0})
            traces.append(trace)
        out = thresholds_from_traces(traces, 600.0, 9000.0)
        assert out["fixed_kmh"] == 30.0
        pooled = out["pooled"]
        assert pooled["v_jam"] == pytest.approx((pooled["s1"] + pooled["s2"]) / 4.0)
        assert 10.0 < pooled["v_jam"] < 25.0
        assert len(out["per_segment"]) == 3
        assert pooled["refused"] is False
        # Segment 2 never has a vehicle, so it falls back to the pooled threshold.
        assert out["per_segment"][2]["fallback"] is True
        assert out["per_segment"][2]["refused"] is True
        assert out["per_segment"][2]["v_jam"] == pooled["v_jam"]
        assert out["samples"] == 3 * 2 * len(range(600, 9001, 10))


class TestSummary:
    def test_censored_runs_stay_in_the_denominator(self):
        per_seed = {"no_control": {16: (7800.0, False), 17: (2000.0, True),
                                   18: (7800.0, False)},
                    "here": {16: (1000.0, True), 17: (900.0, True), 18: (1500.0, True)}}
        out = drain_summary(per_seed, cap_minus_t0=7800.0)
        assert out["arms"]["no_control"]["fraction_drained"] == pytest.approx(1 / 3)
        assert out["arms"]["no_control"]["n"] == 3
        assert out["arms"]["no_control"]["km"]["median"] is None
        assert out["arms"]["here"]["fraction_drained"] == 1.0
        # Paired only where both drained: seed 17.
        assert out["paired"]["here"]["seeds"] == [17]
        assert out["paired"]["here"]["differences"] == [-1100.0]


def test_the_module_never_reads_the_reward_density():
    source = Path(drain.__file__).read_text()
    assert "DENSITY_CRITICAL" not in source
    assert "densities(" not in source


class TestLaterSpells:
    """Amendment 1, A3: a later spell of at least H undoes an earlier drain."""

    def test_clear_at_t0_then_jammed_with_pending_to_the_cap_is_censored(self):
        trace = _trace(9000, jam_from=1400, jam_until=9001, pending_until=9001)
        for sample in trace:
            if sample["t"] < 1400:
                sample["pending"] = 0
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (7800.0, False)

    def test_a_later_spell_shorter_than_h_does_not_censor(self):
        trace = _trace(9000, jam_from=5000, jam_until=5100)  # 10 samples, 100 s
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (120.0, True)

    def test_a_later_spell_of_h_moves_the_drain_past_it(self):
        trace = _trace(9000, jam_from=5000, jam_until=5120)  # 12 samples, 120 s
        # Last jammed sample 5,110 s; the first clear [t_d - 120, t_d] ends at 5,240 s.
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (5240.0 - 1200.0, True)

    def test_a_later_waiting_spell_counts_like_a_jam(self):
        trace = _trace(9000)
        for sample in trace:
            if 6000 <= sample["t"] < 6200:
                sample["held"] = 2
        t, drained = drain_time(trace, 1200.0, 120.0, 9000.0, 20.0)
        assert drained and t == 6200.0 - 1200.0

    def test_spells_are_runs_of_consecutive_samples(self):
        from src.sumo.drain import spells

        flags = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
        times = np.arange(10.0, 101.0, 10.0)
        assert spells(flags, times) == [(20.0, 30.0), (50.0, 50.0), (80.0, 100.0)]



class TestRefusal:
    """Amendment 1, A4: a fit with s2 >= 3 s1 gives no threshold, and says why."""

    def test_the_surge_like_pool_is_refused(self):
        from src.sumo.drain import threshold_entry

        rng = np.random.default_rng(4)
        # Mostly stopped and crawling traffic: the pool of the surge scenario after its
        # demand stops, whose fit has knots at 3.5 and 25.0 km/h.
        speeds = np.concatenate([rng.uniform(0.0, 3.5, 3500), rng.uniform(3.5, 25.0, 4500),
                                 rng.uniform(25.0, 40.0, 2000)])
        entry = threshold_entry(speeds)
        assert abs(entry["s1"] - 3.5) <= 0.5 and abs(entry["s2"] - 25.0) <= 1.0
        assert entry["refused"] is True and entry["v_jam"] is None
        assert "3 s1" in entry["reason"]

    def test_the_boundary_is_refused_and_below_it_accepted(self):
        from src.sumo.drain import CdfFit, refusal_reason

        assert refusal_reason(CdfFit(10.0, 30.0, 0.0)) is not None
        assert refusal_reason(CdfFit(10.0, 29.5, 0.0)) is None
        assert refusal_reason(CdfFit(30.0, 51.0, 0.0)) is None

    def test_a_segment_with_s2_under_1_5_s1_falls_back(self, monkeypatch):
        from src.sumo.drain import CdfFit

        # Every pool fits to knots 10 and 12 km/h: s2 / s1 = 1.2, no separated jam mode.
        monkeypatch.setattr(drain, "fit_three_piece_cdf", lambda speeds: CdfFit(10.0, 12.0, 0.0))
        trace = [{"t": 10.0, "speed_kmh": [40.0, 40.0], "count": [1, 1], "held": 0,
                  "pending": 0}]
        out = thresholds_from_traces([trace], 0.0, 100.0, fallback_v_jam=17.0)
        assert out["pooled"]["v_jam"] == pytest.approx(5.5)
        assert all(e["fallback"] is True and e["v_jam"] == 17.0 for e in out["per_segment"])
        assert all(e["own_v_jam"] == pytest.approx(5.5) for e in out["per_segment"])

    def test_a_refused_segment_falls_back_and_a_good_one_keeps_its_own(self, monkeypatch):
        from src.sumo.drain import CdfFit

        fits = iter([CdfFit(30.0, 51.0, 0.0), CdfFit(3.5, 25.0, 0.0), CdfFit(28.0, 50.0, 0.0)])
        monkeypatch.setattr(drain, "fit_three_piece_cdf", lambda speeds: next(fits))
        trace = [{"t": 10.0, "speed_kmh": [40.0, 40.0], "count": [1, 1], "held": 0,
                  "pending": 0}]
        out = thresholds_from_traces([trace], 0.0, 100.0)
        assert out["pooled"]["v_jam"] == pytest.approx(20.25)
        refused, good = out["per_segment"]
        assert refused["refused"] is True and refused["v_jam"] == pytest.approx(20.25)
        assert good["fallback"] is False and good["v_jam"] == pytest.approx(19.5)


class TestOpenSpells:
    """Amendment 2, B5: a drained run may end inside a short spell open at the cap."""

    def test_a_short_jam_at_the_cap_is_reported_and_does_not_undo_the_drain(self):
        from src.sumo.drain import open_spell_at_cap

        trace = _trace(9000, jam_from=2000, jam_until=2100)
        for sample in trace:
            if sample["t"] >= 8950:
                sample["speed_kmh"][1] = 8.0  # 6 samples, 60 s, still jammed at 9,000 s
        # Both later spells are shorter than H, so the drain stands at t_0 + H.
        assert drain_time(trace, 1200.0, 120.0, 9000.0, 20.0) == (120.0, True)
        spell = open_spell_at_cap(trace, 9000.0, 20.0)
        assert spell == {"kinds": ["jam"], "at": 9000.0,
                         "jam": {"start": 8950.0, "length_s": 60.0}}

    def test_a_waiting_spell_at_the_cap_and_a_clear_end(self):
        from src.sumo.drain import open_spell_at_cap

        trace = _trace(9000, held_until=0)
        assert open_spell_at_cap(trace, 9000.0, 20.0) is None
        for sample in trace:
            if sample["t"] >= 8990:
                sample["pending"] = 2
        assert open_spell_at_cap(trace, 9000.0, 20.0)["waiting"] == {"start": 8990.0,
                                                                     "length_s": 20.0}
