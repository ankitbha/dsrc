"""Jam metrics (plan sections 4.3, 5.7 and J20, step S10). No SUMO."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.analysis import jam_metrics as jm  # noqa: E402

V = 19.0


def trace_from(jammed_runs: dict[int, list[tuple[int, int]]], n: int = 60, t0: float = 900.0,
               segments: int = 12, extra=None) -> list[dict]:
    """10 s samples from `t0`; `jammed_runs[segment]` lists `(first, count)` sample runs
    at 10 km/h; every other occupied segment runs at 50 km/h."""
    trace = []
    for i in range(n):
        speed = [50.0] * segments
        for segment, runs in jammed_runs.items():
            for first, count in runs:
                if first <= i < first + count:
                    speed[segment] = 10.0
        sample = {"t": t0 + 10.0 * i, "speed_kmh": speed, "count": [5] * segments,
                  "density": [0.05] * segments, "held": 0, "pending": 0}
        if extra:
            sample.update(extra(i))
        trace.append(sample)
    return trace


class TestSpellsAndOnset:
    def test_spells_of_6_12_and_13_samples_give_onset_at_the_12_sample_spell(self):
        trace = trace_from({3: [(2, 6)], 5: [(20, 12)], 7: [(10, 13)]}, n=60)
        spells = jm.jam_spells(trace, V)
        assert sorted((s.segment, s.start, s.length_s) for s in spells) == [
            (3, 920.0, 60.0), (5, 1100.0, 120.0), (7, 1000.0, 130.0)]
        onset = jm.onset(trace, V, t_start=900.0, cap=1490.0)
        assert onset == jm.Onset(1000.0, 7)
        # With the 13-sample spell gone, onset is the 12-sample spell's start.
        trace = trace_from({3: [(2, 6)], 5: [(20, 12)]}, n=60)
        assert jm.onset(trace, V, 900.0, 1490.0) == jm.Onset(1100.0, 5)

    def test_no_spell_long_enough_gives_no_onset(self):
        trace = trace_from({3: [(2, 11)]}, n=60)
        assert jm.onset(trace, V, 900.0, 1490.0) is None
        assert jm.prevented(trace, V, 900.0, 1490.0)

    def test_a_spell_cut_by_the_cap_is_not_an_onset(self):
        trace = trace_from({4: [(50, 10)]}, n=60)
        assert jm.onset(trace, V, 900.0, 1490.0) is None

    def test_a_spell_before_the_clock_starts_is_not_counted(self):
        trace = trace_from({4: [(0, 20)]}, n=60, t0=800.0)
        assert jm.onset(trace, V, 900.0, 1390.0) is None

    def test_an_empty_segment_is_never_jammed(self):
        trace = trace_from({4: [(0, 30)]}, n=60)
        for s in trace:
            s["count"][4] = 0
        assert jm.onset(trace, V, 900.0, 1490.0) is None

    def test_first_sample_onset_is_block_d_s(self):
        trace = trace_from({3: [(2, 1)], 5: [(20, 12)]}, n=60)
        assert jm.first_sample_onset(trace, V, 900.0) == 920.0


class TestTimes:
    def test_jam_time_is_segment_minutes(self):
        trace = trace_from({3: [(0, 6)], 5: [(10, 12)]}, n=60)
        assert jm.jam_time(trace, V, 900.0, 1490.0) == pytest.approx((6 + 12) * 10.0 / 60.0)
        assert jm.jam_time(trace, V, 1000.0, 1490.0) == pytest.approx(12 * 10.0 / 60.0)

    def test_held_and_pending_vehicle_minutes(self):
        trace = trace_from({}, n=12, extra=lambda i: {"held": 3, "pending": 1 if i < 6 else 0})
        held, pending = jm.held_minutes(trace, 900.0, 1010.0)
        assert held == pytest.approx(12 * 3 * 10.0 / 60.0)
        assert pending == pytest.approx(6 * 1 * 10.0 / 60.0)

    def test_drain_uses_the_drain_module(self):
        trace = trace_from({3: [(0, 30)]}, n=120)
        # Jammed to 1,190 s; clear from 1,200 s; t_0 = 1,200 s and H = 120 s.
        assert jm.drain(trace, V, t_0=1200.0, cap=2090.0) == (120.0, True)

    def test_km_on_100_200_and_a_censored_run(self):
        km = jm.km_onset([1000.0, 1100.0, None], t_start=900.0, cap=3600.0)
        assert km.at(99.0) == 1.0
        assert km.at(100.0) == pytest.approx(2 / 3)
        assert km.at(200.0) == pytest.approx(1 / 3)
        assert km.at(2700.0) == pytest.approx(1 / 3)
        assert km.n == 3 and km.median == 200.0


class TestLeads:
    def _lead_trace(self):
        """Segment 10: baseline 50 km/h and k 0.05 on [600, 900); k rises from 1,000 s,
        speed falls below 40 km/h from 1,200 s, a 120 s jam spell from 1,400 s."""
        trace = []
        for i in range(120):
            t = 600.0 + 10.0 * i
            speed = [50.0] * 12
            k_hat = [0.05] * 12
            k_true = [0.05] * 12
            k_hat[10] = 0.05 + (0.001 if i % 2 else -0.001)
            if t >= 1000.0:
                k_hat[10] = 0.2
            if t >= 1200.0:
                speed[10] = 35.0
            if 1400.0 <= t < 1600.0:
                speed[10] = 10.0
            trace.append({"t": t, "speed_kmh": speed, "count": [5] * 12,
                          "density": [0.05] * 12, "held": 0, "pending": 0,
                          "k_hat_mean": k_hat, "k_true_mean": k_true})
        return trace

    def test_feed_and_camera_leads_on_the_first_segment_to_jam(self):
        leads = jm.leads(self._lead_trace(), V, t_start=900.0, cap=1790.0)
        assert leads["segment"] == 10 and leads["onset"] == 1400.0
        assert leads["W_feed"] == 200.0
        assert leads["W_cam"] == 400.0
        assert leads["W_true"] is None  # k_true never departs

    def _ramp_trace(self, speed_at, k_hat_at, k_hat_base=lambda i: 0.05):
        """Segment 10 only; 10 s samples on [600, 1790]; a 200 s jam spell from 1,400 s."""
        trace = []
        for i in range(120):
            t = 600.0 + 10.0 * i
            speed = [50.0] * 12
            k_hat = [0.05] * 12
            speed[10] = 50.0 if t < 900.0 else speed_at(t)
            k_hat[10] = k_hat_base(i) if t < 900.0 else k_hat_at(t)
            if 1400.0 <= t < 1600.0:
                speed[10] = 10.0
            trace.append({"t": t, "speed_kmh": speed, "count": [5] * 12, "density": [0.05] * 12,
                          "held": 0, "pending": 0, "k_hat_mean": k_hat, "k_true_mean": [0.05] * 12})
        return trace

    def test_the_speed_precursor_is_80_percent_of_the_baseline(self):
        # Speed falls 1 km/h per 10 s from 49.5 at 900 s to a floor of 30: below 40 (80% of
        # 50) first at 1,000 s (39.5); below 45 (90%) at 950 s (kills a 90% precursor, M51).
        trace = self._ramp_trace(lambda t: max(30.0, 49.5 - (t - 900.0) / 10.0),
                                 lambda t: 0.05)
        assert jm.leads(trace, V, 900.0, 1790.0)["W_feed"] == 1400.0 - 1000.0

    def test_the_density_precursor_is_three_standard_deviations(self):
        # Baseline 0.04 / 0.06 alternating (mean 0.05, SD 0.01); from 900 s k_hat rises
        # 0.001 per 10 s from 0.0505: above 0.08 (3 SD) first at 1,200 s (0.0805), above
        # 0.07 (2 SD) at 1,100 s (kills a 2 SD precursor, M52).
        trace = self._ramp_trace(lambda t: 50.0, lambda t: 0.0505 + 0.0001 * (t - 900.0),
                                 k_hat_base=lambda i: 0.04 if i % 2 else 0.06)
        assert jm.leads(trace, V, 900.0, 1790.0)["W_cam"] == pytest.approx(1400.0 - 1200.0)

    def test_a_precursor_must_hold_30_s(self):
        # Below 40 km/h for two samples (20 s) at 1,000 s, then from 1,200 s for good:
        # the 30 s hold skips the short dip (kills a 20 s hold, M53).
        trace = self._ramp_trace(lambda t: 35.0 if (1000.0 <= t <= 1010.0 or t >= 1200.0)
                                 else 50.0, lambda t: 0.05)
        assert jm.leads(trace, V, 900.0, 1790.0)["W_feed"] == 1400.0 - 1200.0

    def test_a_precursor_only_at_or_after_onset_is_none(self):
        # Speed stays at baseline until the jam itself, and k_hat rises only after it: no
        # lead, rather than one of zero or less (kills a search not cut at onset, M60).
        trace = self._ramp_trace(lambda t: 50.0, lambda t: 0.3 if t >= 1450.0 else 0.05)
        leads = jm.leads(trace, V, 900.0, 1790.0)
        assert leads["onset"] == 1400.0
        assert leads["W_feed"] is None and leads["W_cam"] is None

    def test_the_leads_say_whether_the_camera_s_estimate_was_traced(self):
        with_camera = self._lead_trace()
        assert jm.leads(with_camera, V, 900.0, 1790.0)["k_hat_available"] is True
        without = [{**sample, "k_hat_mean": [None] * 12} for sample in with_camera]
        out = jm.leads(without, V, 900.0, 1790.0)
        assert out["k_hat_available"] is False and out["W_cam"] is None
        assert jm.leads(trace_from({}, n=60, t0=600.0), V, 900.0, 1190.0)["k_hat_available"] is False

    def test_no_onset_gives_no_leads(self):
        trace = trace_from({}, n=60, t0=600.0)
        assert jm.leads(trace, V, 900.0, 1190.0)["onset"] is None


class TestTheory:
    def test_speed_density_break_is_the_first_bin_below_95_percent_of_free_flow(self):
        trace = []
        for i in range(300):
            k = 0.002 * i
            speed = 60.0 if k < 0.1 else 50.0
            trace.append({"t": float(i), "speed_kmh": [speed], "count": [5], "density": [k]})
        result = jm.speed_density_break([trace], free_flow_kmh=60.0, bin_width=0.01,
                                        min_samples=3)
        assert result["k_b"] == pytest.approx(0.10)

    def test_the_break_is_at_95_percent_of_free_flow(self):
        # Median speed 60 below 0.10, 56 on [0.10, 0.11) (below 57 = 95%, above 54 = 90%),
        # 50 beyond: k_b is 0.10 at 95% and would be 0.11 at 90% (kills M56).
        trace = []
        for i in range(300):
            k = 0.002 * i
            speed = 60.0 if k < 0.1 else (56.0 if k < 0.11 else 50.0)
            trace.append({"t": float(i), "speed_kmh": [speed], "count": [5], "density": [k]})
        result = jm.speed_density_break([trace], free_flow_kmh=60.0, bin_width=0.01,
                                        min_samples=3)
        assert result["k_b"] == pytest.approx(0.10)

    def test_max_ramp_is_the_largest_30_s_rise_before_onset(self):
        trace = [{"t": 900.0 + 10.0 * i, "density": [0.05 + (0.003 * i if i < 20 else 0.0)]}
                 for i in range(40)]
        trace[10]["density"][0] += 0.02  # a 0.02 bump inside the rise
        out = jm.max_ramp(trace, segment=0, onset=1090.0)
        assert out["rise"] == pytest.approx(0.009 + 0.02)
        assert out["rate_per_s"] == pytest.approx((0.009 + 0.02) / 30.0)


class TestPaired:
    def test_two_standard_errors_by_hand(self):
        # Differences 1, 2, 3: mean 2, sample SD 1, so two standard errors are 2 / sqrt(3)
        # (kills one standard error, M34).
        paired = jm.Paired.of({1: 3.0, 2: 5.0, 3: 7.0}, {1: 2.0, 2: 3.0, 3: 4.0})
        assert paired.seeds == [1, 2, 3] and paired.differences == [1.0, 2.0, 3.0]
        assert paired.mean == pytest.approx(2.0)
        assert paired.two_se == pytest.approx(2.0 / 3 ** 0.5)

    def test_one_seed_has_no_standard_error(self):
        assert jm.Paired.of({1: 3.0}, {1: 2.0}).two_se is None


class TestCriterionC:
    def test_hand_table(self):
        J = {
            "B5": {"plan": {11: 10.0, 12: 20.0}, "no_control": {11: 40.0, 12: 60.0},
                   "rule": {11: 5.0, 12: 15.0}},
            "B9": {"plan": {11: 30.0, 12: 30.0}, "no_control": {11: 100.0, 12: 100.0},
                   "rule": {11: 40.0, 12: 30.0}},
        }
        out = jm.criterion_c(J, "rule", scenarios=("B5", "B9"))
        # B5: mean(-5, -5) / mean(40, 60) = -0.1; B9: mean(10, 0) / 100 = 0.05.
        assert out["terms"] == {"B5": pytest.approx(-0.1), "B9": pytest.approx(0.05)}
        assert out["C"] == pytest.approx(-0.025)
        assert jm.criterion_c(J, "plan", scenarios=("B5", "B9"))["C"] == 0.0

    def test_unpaired_seeds_are_refused(self):
        J = {"B5": {"plan": {11: 10.0}, "no_control": {11: 40.0}, "rule": {12: 5.0}}}
        with pytest.raises(ValueError):
            jm.criterion_c(J, "rule", scenarios=("B5",))


class TestThreshold:
    def test_v_jam_is_read_from_the_drain_thresholds_with_its_sha256(self):
        v, sha = jm.load_v_jam()
        data = json.loads((REPO_ROOT / "results/feed/drain_thresholds.json").read_text())
        assert v == data["network"]["pooled"]["v_jam"] == 19.0
        assert len(sha) == 64

    def test_the_module_never_reads_the_reward_s_density(self):
        source = Path(jm.__file__).read_text()
        assert "DENSITY_CRITICAL" not in source
        assert "densities(" not in source

    def test_the_30_km_h_sensitivity_changes_jam_time(self):
        trace = trace_from({}, n=30)
        for s in trace[:10]:
            s["speed_kmh"][2] = 25.0
        assert jm.jam_time(trace, 19.0, 900.0, 1190.0) == 0.0
        assert jm.jam_time(trace, 30.0, 900.0, 1190.0) > 0.0


class TestEpisodeMetrics:
    def test_one_record_carries_every_metric(self):
        trace = trace_from({5: [(30, 20)]}, n=300, t0=0.0,
                           extra=lambda i: {"held": 2 if 100 <= i < 110 else 0,
                                            "pending": 0})
        out = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=2990.0)
        assert out["onset_s"] is None  # the spell is before 900 s
        # Jam time is counted from the clock start, not from 0 s (kills M30).
        assert out["jam_time_min"] == 0.0 and out["jam_time_min_900_3600"] == 0.0
        trace = trace_from({5: [(100, 20)]}, n=300, t0=0.0)
        out = jm.episode_metrics(trace, v_jam=19.0, start_s=900.0, end_s=1200.0, cap=2990.0)
        assert out["onset_s"] == 1000.0 and out["onset_segment"] == 5
        assert out["onset_after_start_s"] == 100.0
        assert out["first_sample_onset_s"] == 1000.0
        assert out["prevented"] is False
        assert out["jam_time_min"] == pytest.approx(20 * 10.0 / 60.0)
        assert out["drained"] is True and out["drain_s"] == 120.0
        assert out["jam_time_min_30kmh"] >= out["jam_time_min"]
        for key in ("held_min", "pending_min", "jam_time_min_900_3600"):
            assert key in out
