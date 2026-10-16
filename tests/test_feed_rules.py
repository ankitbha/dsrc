"""The feed process, the agreement rules and Lemma 1, without SUMO.

The fleet fixtures use 1,000 vehicles with uniform phases from a fixed seed, so every
fraction below is deterministic; the tolerances are about two binomial standard
deviations at n = 1,000 (0.016 at a fraction of one half).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.sumo.feed import (
    FeedProcess, FixedPhase, NoRule, Synchronous, VehicleFeedState, VersionKeyed,
    command_disagreement, disagreement, draw_phases, lemma_bound, lemma_cycle_average,
    lemma_skip_bound, version_from_schedule,
)

N = 1000


def _fleet(seed: int = 7) -> list[VehicleFeedState]:
    return [VehicleFeedState(f"v{i}", float(u)) for i, u in enumerate(draw_phases(seed, N))]


def _versions_at(rule, fleet, feed, instants) -> np.ndarray:
    """Applied version of every vehicle (rows) at every instant (columns)."""
    end = float(max(instants))
    out = np.empty((len(fleet), len(instants)), dtype=int)
    for row, state in enumerate(fleet):
        switch, versions = rule.switch_schedule(state, feed, end)
        position = np.searchsorted(switch, instants, side="right") - 1
        out[row] = np.where(position >= 0, versions[np.maximum(position, 0)], 0)
    return out


def _delta_series(matrix: np.ndarray) -> np.ndarray:
    return np.array([disagreement(matrix[:, j]) for j in range(matrix.shape[1])])


def _synthetic_40_50(count: int = 40, seed: int = 3) -> FeedProcess:
    lags = np.random.default_rng(seed).uniform(40.0, 50.0, count)
    lags[0], lags[1] = 40.0, 50.0  # the bracket's two ends are attained
    return FeedProcess.synthetic(60.0 * np.arange(1, count + 1), lags)


class TestTheProcess:
    def test_a_query_returns_the_newest_available_version(self):
        feed = FeedProcess.synthetic([100.0, 160.0, 220.0], [10.0, 5.0, 20.0])
        assert feed.version_at(109.9) == 0
        assert feed.version_at(110.0) == 1
        assert feed.version_at(165.0) == 2
        assert feed.version_at(239.9) == 2
        assert feed.version_at(240.0) == 3
        assert list(feed.version_at(np.array([0.0, 110.0, 300.0]))) == [0, 1, 3]

    def test_a_version_overtaken_by_its_successor_is_never_returned(self):
        # A_1 = 160 comes after A_2 = 140: version 1 is skipped by every query.
        feed = FeedProcess.synthetic([100.0, 126.0], [60.0, 14.0])
        assert feed.version_at(150.0) == 2
        assert feed.version_at(170.0) == 2

    def test_aligned_has_zero_lag_at_the_decision_instants(self):
        feed = FeedProcess.aligned([300.0, 360.0, 420.0])
        assert list(feed.available_s) == [300.0, 360.0, 420.0]
        assert feed.version_at(359.9) == 1 and feed.version_at(360.0) == 2

    def test_unordered_publication_is_refused(self):
        with pytest.raises(ValueError):
            FeedProcess.synthetic([60.0, 60.0], [1.0, 1.0])

    def test_from_artefact_replays_the_gaps_in_order_from_the_seeded_start(self, tmp_path):
        sequence = [[60.0, 40.0], [26.0, 45.5], [34.0, 50.0], [60.0, 42.0], [120.0, 33.0]]
        path = tmp_path / "feed_process.json"
        path.write_text(json.dumps({"placeholder": True, "source": "test",
                                    "sessions": {"s": {"sequence": sequence}}}))
        starts = set()
        for seed in range(12):
            feed = FeedProcess.from_artefact(path, "s", seed, 5.0, 1000.0)
            start = feed.meta["start_index"]
            assert start == int(np.random.default_rng(seed).integers(5))
            starts.add(start)
            assert feed.publish_s[0] == 5.0
            expected_gaps = [sequence[(start + j) % 5][0] for j in range(len(feed.gaps()))]
            expected_lags = [sequence[(start + j) % 5][1] for j in range(feed.count)]
            assert list(feed.gaps()) == expected_gaps
            assert list(feed.lag_s) == expected_lags
            assert feed.publish_s[-1] <= 1000.0 < feed.publish_s[-1] + sequence[
                (start + feed.count - 1) % 5][0]
            assert feed.meta["placeholder"] is True
            assert len(feed.meta["artefact_sha256"]) == 64
        assert len(starts) > 1

    def test_an_unknown_session_names_the_ones_present(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"sessions": {"a": {"sequence": [[60, 1]]}}}))
        with pytest.raises(KeyError, match="'a'"):
            FeedProcess.from_artefact(path, "b", 0, 0.0, 100.0)


class TestVersionKeyed:
    def test_d_at_l_bar_plus_p_q_gives_zero_disagreement_at_every_second(self):
        feed = _synthetic_40_50()
        rule = VersionKeyed(P_q=10.0, D=60.0)
        instants = np.arange(0.0, 60.0 * 40, 1.0)
        matrix = _versions_at(rule, _fleet(), feed, instants)
        assert np.nanmax(_delta_series(matrix)) == 0.0
        # And it applies every version, each on exactly [T_k + D, T_{k+1} + D).
        for k in range(1, 38):
            start = int(feed.publish_of(k) + 60.0)
            assert (matrix[:, start] == k).all() and (matrix[:, start - 1] == k - 1).all()

    @pytest.mark.parametrize("tau", [0.0, 5.0])
    def test_d_at_l_bar_gives_the_lemma_late_fraction(self, tau):
        feed = _synthetic_40_50()
        D, L_bar, P_q = 50.0, 50.0, 10.0
        rule = VersionKeyed(P_q=P_q, D=D)
        versions = range(2, 38)
        instants = np.array([feed.publish_of(k) + D + tau for k in versions])
        matrix = _versions_at(rule, _fleet(), feed, instants)
        late = np.array([(matrix[:, j] != k).mean() for j, k in enumerate(versions)])
        # Exact expectation per version uses its own lag; the lemma's bound uses L_bar.
        expected = np.array([max(0.0, 1.0 - (D - feed.lag_s[k - 1] + tau) / P_q)
                             for k in versions])
        bound = lemma_bound(D, L_bar, P_q, 60.0, tau)
        assert abs(late.mean() - expected.mean()) < 0.03
        assert np.all(late <= bound + 0.05)
        # The version whose lag is L_bar (version 2) attains the bound.
        assert abs(late[0] - bound) < 0.03

    def test_a_late_fetch_applies_on_receipt(self):
        feed = FeedProcess.synthetic([0.0, 60.0], [30.0, 30.0])
        rule = VersionKeyed(P_q=10.0, D=20.0)
        state = VehicleFeedState("v", 0.5)  # queries at ..., 25, 35, 45, ...
        instants, versions = rule.switch_schedule(state, feed, 200.0)
        assert list(versions) == [1, 2]
        assert list(instants) == [35.0, 95.0]

    def test_an_early_fetch_waits_for_its_switch_instant(self):
        feed = FeedProcess.synthetic([0.0, 60.0], [30.0, 30.0])
        state = VehicleFeedState("v", 0.5)
        instants, versions = VersionKeyed(P_q=10.0, D=50.0).switch_schedule(state, feed, 200.0)
        assert list(instants) == [50.0, 110.0] and list(versions) == [1, 2]

    def test_applied_version_reads_the_switch_schedule(self):
        feed = _synthetic_40_50()
        rule = VersionKeyed(P_q=10.0, D=55.0)
        state = VehicleFeedState("v", 0.37)
        instants, versions = rule.switch_schedule(state, feed, 2000.0)
        for t in (0.0, 100.0, 333.3, 1999.0):
            assert rule.applied_version(state, feed, t) == version_from_schedule(
                instants, versions, t)


class TestNoRule:
    def test_query_every_60_s_averages_a_quarter_and_peaks_at_a_half(self):
        feed = FeedProcess.synthetic(60.0 * np.arange(1, 41), np.full(40, 45.0))
        instants = np.arange(600.0, 1800.0, 1.0)  # twenty whole cycles
        delta = _delta_series(_versions_at(NoRule(60.0), _fleet(), feed, instants))
        assert abs(delta.mean() - 0.25) < 0.02
        assert abs(delta.max() - 0.5) < 0.03

    def test_query_every_10_s_averages_p_q_over_4g(self):
        feed = FeedProcess.synthetic(60.0 * np.arange(1, 41), np.full(40, 45.0))
        instants = np.arange(600.0, 1800.0, 1.0)
        delta = _delta_series(_versions_at(NoRule(10.0), _fleet(), feed, instants))
        assert abs(delta.mean() - 10.0 / 240.0) < 0.01


class TestFixedPhase:
    feed = FeedProcess.synthetic(60.0 * np.arange(1, 41) + 36.0, np.full(40, 45.0))
    # Availability falls at second 21 of every minute.

    def test_psi_outside_the_availability_window_gives_zero(self):
        instants = np.arange(600.0, 1800.0, 1.0)
        matrix = _versions_at(FixedPhase(psi=50.0, r_bar=1.0), _fleet(), self.feed, instants)
        assert np.nanmax(_delta_series(matrix)) == 0.0

    def test_psi_inside_the_window_splits_the_fleet_for_the_minute(self):
        instants = np.arange(600.0, 1800.0, 1.0)
        matrix = _versions_at(FixedPhase(psi=20.5, r_bar=1.0), _fleet(), self.feed, instants)
        delta = _delta_series(matrix)
        # Half the requests land before second 21 and keep the previous version for the
        # whole minute; the other half fetch the new one.
        assert abs(np.nanmean(delta) - 0.5) < 0.05


class TestSynchronous:
    def test_every_vehicle_switches_at_the_availability_instant(self):
        feed = FeedProcess.synthetic([0.0, 60.0, 120.0], [30.0, 31.5, 29.0])
        instants, versions = Synchronous().switch_schedule(VehicleFeedState("v", 0.9), feed, 500)
        assert list(instants) == [30.0, 91.5, 149.0] and list(versions) == [1, 2, 3]


class TestLemma:
    def test_the_bound_at_its_corners(self):
        assert lemma_bound(50.0, 50.0, 10.0, 60.0, 0.0) == 1.0
        assert lemma_bound(50.0, 50.0, 10.0, 60.0, 5.0) == pytest.approx(0.5)
        assert lemma_bound(55.0, 50.0, 10.0, 60.0, 2.0) == pytest.approx(0.3)
        assert lemma_bound(60.0, 50.0, 10.0, 60.0, 0.0) == 0.0
        assert lemma_bound(30.0, 50.0, 10.0, 60.0, 10.0) == 1.0

    def test_the_cycle_average_matches_its_closed_form(self):
        # (P_q - D + L_bar)^2 / (2 P_q g) on [L_bar, L_bar + P_q).
        assert lemma_cycle_average(50.0, 50.0, 10.0, 60.0) == pytest.approx(100 / 1200)
        assert lemma_cycle_average(55.0, 50.0, 10.0, 60.0) == pytest.approx(25 / 1200)
        assert lemma_cycle_average(60.0, 50.0, 10.0, 60.0) == 0.0
        # Below L_bar: 20 s at one, then the 10 s ramp, over 60 s.
        assert lemma_cycle_average(30.0, 50.0, 10.0, 60.0) == pytest.approx(25.0 / 60.0)

    @pytest.mark.parametrize("x", [-25.0, -5.0, 0.0, 3.0, 9.9, 12.0])
    def test_the_cycle_average_is_the_integral_of_the_bound(self, x):
        g = 60.0
        taus = np.linspace(0.0, g, 600001)[:-1]
        numeric = np.mean([lemma_bound(50.0 + x, 50.0, 10.0, g, t) for t in taus[::100]])
        assert lemma_cycle_average(50.0 + x, 50.0, 10.0, g) == pytest.approx(numeric, abs=2e-3)

    def test_the_skip_term(self):
        assert lemma_skip_bound(10.0, 26.0, 10.0) == 0.0
        assert lemma_skip_bound(10.0, 26.0, 20.0) == pytest.approx(0.4)
        assert lemma_skip_bound(10.0, 26.0, 40.0) == 1.0


class TestDisagreement:
    def test_the_fraction_off_the_plurality(self):
        assert disagreement([3, 3, 3, 2]) == pytest.approx(0.25)
        assert disagreement([1, 2, 2, 3, 3, 3]) == pytest.approx(0.5)
        assert disagreement([5]) == 0.0
        assert math.isnan(disagreement([]))

    def test_different_versions_with_equal_commands_disagree_on_nothing(self):
        vectors = {1: [0.5, 1.0], 2: [0.5, 0.75]}
        assert command_disagreement([1, 2], [0, 0], vectors) == 0.0
        assert disagreement([1, 2]) == 0.5

    def test_commands_are_compared_within_a_segment(self):
        vectors = {1: [0.5, 1.0], 2: [0.5, 0.75]}
        # Two on segment 1 disagree (1.0 vs 0.75); the one on segment 0 does not.
        assert command_disagreement([1, 2, 1], [1, 1, 0], vectors) == pytest.approx(1 / 3)

    def test_no_version_is_a_command_of_its_own(self):
        assert command_disagreement([0, 1], [0, 0], {1: [0.5]}) == 0.5
