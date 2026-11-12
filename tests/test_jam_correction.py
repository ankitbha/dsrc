"""The corrections and the lookahead rule (plan section 5, step S5). No SUMO."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.rl.src_q import SrcQNetwork, greedy_actions  # noqa: E402
from src.sumo.correction import (  # noqa: E402
    FRACTIONS, Hysteresis, LearnedCorrection, LookaheadRule, RuleCorrection, TakeTheSlower,
    batch_greedy, rule_margin, shown_fraction, step_down,
)


class TestStepAndMin:
    def test_step_down(self):
        assert step_down(1.0) == 0.75
        assert step_down(0.75) == 0.5
        assert step_down(0.5) == 0.5

    def test_step_down_vectorised(self):
        assert step_down(np.array([1.0, 0.75, 0.5])).tolist() == [0.75, 0.5, 0.5]

    def test_shown_never_exceeds_the_plan_over_all_nine_pairs(self):
        for plan, correction in itertools.product(FRACTIONS, FRACTIONS):
            assert shown_fraction(plan, correction) == min(plan, correction)
            assert shown_fraction(plan, correction) <= plan

    def test_shown_vectorised(self):
        plan = np.array([1.0, 0.75, 0.5, 1.0])
        corr = np.array([0.75, 1.0, 0.5, 0.5])
        assert shown_fraction(plan, corr).tolist() == [0.75, 0.75, 0.5, 0.5]


class TestHysteresis:
    def test_holds_30_s_and_needs_two_release_ticks(self):
        h = Hysteresis()
        on = lambda t, trig, rel: bool(h.update(["v"], t, np.array([trig]), np.array([rel]))[0])
        assert on(0.0, True, False)
        # The release condition holds from 5 s on, but 30 s have not passed.
        for t in (5.0, 10.0, 15.0, 20.0, 25.0):
            assert on(t, False, True)
        # At 30 s the condition has held on two consecutive ticks: released.
        assert not on(30.0, False, True)

    def test_one_release_tick_is_not_enough(self):
        h = Hysteresis()
        h.update(["v"], 0.0, np.array([True]), np.array([False]))
        for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
            h.update(["v"], t, np.array([False]), np.array([False]))
        assert h.update(["v"], 35.0, np.array([False]), np.array([True]))[0]
        assert not h.update(["v"], 40.0, np.array([False]), np.array([True]))[0]

    def test_an_interrupted_release_starts_counting_again(self):
        h = Hysteresis()
        h.update(["v"], 0.0, np.array([True]), np.array([False]))
        for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
            h.update(["v"], t, np.array([False]), np.array([False]))
        assert h.update(["v"], 35.0, np.array([False]), np.array([True]))[0]
        assert h.update(["v"], 40.0, np.array([False]), np.array([False]))[0]
        assert h.update(["v"], 45.0, np.array([False]), np.array([True]))[0]
        assert not h.update(["v"], 50.0, np.array([False]), np.array([True]))[0]

    def test_inactive_vehicles_stay_off_without_a_trigger(self):
        h = Hysteresis()
        assert h.update(["a", "b"], 0.0, np.array([False, True]),
                        np.array([True, False])).tolist() == [False, True]


class TestRule:
    def test_margin(self):
        assert rule_margin(np.array([0.178 * 0.9]), np.array([0.0]))[0] == pytest.approx(0.1)
        # A rising trend uses the 10 s horizon; a falling one is ignored.
        assert rule_margin(np.array([0.1]), np.array([0.001]))[0] == pytest.approx(
            (0.178 - 0.1 - 0.01) / 0.178)
        assert rule_margin(np.array([0.1]), np.array([-0.01]))[0] == pytest.approx(
            (0.178 - 0.1) / 0.178)

    def test_fires_at_the_threshold_and_releases_at_threshold_plus_0_05(self):
        rule = RuleCorrection(m_on=0.1)
        k_at = lambda m: np.array([0.178 * (1.0 - m)])
        zero = np.array([0.0])
        assert rule.update(["v"], 0.0, k_at(0.1), zero)[0]
        for t in (5.0, 10.0, 15.0, 20.0, 25.0):
            assert rule.update(["v"], t, k_at(0.149), zero)[0]
        assert rule.update(["v"], 30.0, k_at(0.15), zero)[0]
        assert not rule.update(["v"], 35.0, k_at(0.15), zero)[0]

    def test_does_not_fire_above_the_threshold(self):
        rule = RuleCorrection(m_on=0.1)
        assert not rule.update(["v"], 0.0, np.array([0.178 * 0.89]), np.array([0.0]))[0]

    def test_records_its_parameters(self):
        assert RuleCorrection(m_on=0.2, tau_l=30.0).describe() == {
            "kind": "rule", "m_on": 0.2, "tau_l": 30.0, "k_c": 0.178,
            "release_offset": 0.05, "min_on_s": 30.0, "release_ticks": 2}


class TestLearned:
    def test_fires_at_theta_and_releases_below_theta_minus_0_1(self):
        learned = LearnedCorrection(predictor=None, theta=0.45)
        assert learned.update_from_probability(["v"], 0.0, np.array([0.45]))[0]
        for t in (5.0, 10.0, 15.0, 20.0, 25.0):
            assert learned.update_from_probability(["v"], t, np.array([0.36]))[0]
        assert learned.update_from_probability(["v"], 30.0, np.array([0.34]))[0]
        assert not learned.update_from_probability(["v"], 35.0, np.array([0.34]))[0]


class TestLookahead:
    def test_jam_factor_5_on_segment_9_with_threshold_4(self):
        jam = np.zeros(12)
        jam[9] = 5.0
        rule = LookaheadRule(J_th=4.0)
        fractions = rule.fractions(jam, np.array([7, 8, 9, 10, 0]))
        assert fractions.tolist() == [0.75, 0.5, 1.0, 1.0, 1.0]

    def test_the_nearest_downstream_segment_wins(self):
        jam = np.zeros(12)
        jam[[4, 5]] = 6.0
        assert LookaheadRule(J_th=4.0).fractions(jam, np.array([3, 2])).tolist() == [0.5, 0.75]

    def test_below_threshold_is_ignored(self):
        jam = np.zeros(12)
        jam[9] = 3.9
        assert LookaheadRule(J_th=4.0).fractions(jam, np.array([8])).tolist() == [1.0]


def _stub_network(prefer_slow_below_kmh: float) -> SrcQNetwork:
    """A 2-segment, 5-feature network that picks action 0 on a segment whose speed feature
    is below a threshold and action 2 otherwise, built by setting its weights."""
    net = SrcQNetwork(2, 5, 3)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        first, second = net.stack[0], net.stack[2]
        # Hidden unit s carries segment s's speed (feature 0).
        first.weight[0, 0] = 1.0
        first.weight[1, 5] = 1.0
        for s in range(2):
            # Q(slow) = threshold, Q(fast) = speed; Q(mid) = -inf-ish.
            second.bias[3 * s + 0] = prefer_slow_below_kmh
            second.weight[3 * s + 2, s] = 1.0
            second.bias[3 * s + 1] = -1e6
    return net


class TestTakeTheSlower:
    def test_batch_greedy_equals_greedy_per_state_on_the_real_checkpoint(self):
        net = SrcQNetwork(12, 5, 3)
        net.load_state_dict(torch.load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                                       weights_only=True))
        rng = np.random.default_rng(0)
        states = rng.uniform(0, 60, size=(40, 12, 5)).astype(np.float32)
        batch = batch_greedy(net, states)
        for k in range(len(states)):
            assert batch[k].tolist() == greedy_actions(net, torch.tensor(states[k])).tolist()

    def test_returns_the_slower_of_the_plan_and_the_rerun(self):
        net = _stub_network(prefer_slow_below_kmh=30.0)
        x = np.zeros((2, 5), dtype=np.float32)
        x[:, 0] = 50.0  # both segments fast in the feed
        x[:, 1] = 60.0  # free flow
        plan = batch_greedy(net, x[None])[0]
        assert plan.tolist() == [2, 2]
        slower = TakeTheSlower(net)
        corrected, rerun = slower.correct(["a", "b", "c"], 0.0, np.stack([x, x, x]),
                                          np.array([0, 1, 0]),
                                          np.array([20.0, 45.0, np.nan]) / 3.6)
        # a: the camera's 20 km/h makes the re-run pick 0.5 on its segment; b's 45 km/h
        # does not; c saw nothing, so its state is unchanged.
        assert rerun.tolist() == [0.5, 1.0, 1.0]
        assert corrected.tolist() == [0.5, 1.0, 1.0]
        assert slower.active.tolist() == [True, False, False]

    def test_the_rerun_writes_speed_and_jam_factor(self):
        x = np.zeros((2, 5), dtype=np.float32)
        x[:, 1] = 60.0
        changed = TakeTheSlower.replace_row(x, 1, 30.0)
        assert changed[1, 0] == 30.0 and changed[1, 2] == pytest.approx(5.0)
        assert changed[0].tolist() == x[0].tolist()
        assert TakeTheSlower.replace_row(x, 0, 90.0)[0, 2] == 0.0

    def test_a_held_correction_keeps_its_fraction_until_release(self):
        net = _stub_network(prefer_slow_below_kmh=30.0)
        x = np.zeros((1, 2, 5), dtype=np.float32)
        x[:, :, 0] = 50.0
        x[:, :, 1] = 60.0
        slower = TakeTheSlower(net)
        corr, _ = slower.correct(["a"], 0.0, x, np.array([0]), np.array([20.0 / 3.6]))
        assert corr.tolist() == [0.5]
        # The camera now sees free road, but the 30 s hold keeps the step down.
        corr, rerun = slower.correct(["a"], 5.0, x, np.array([0]), np.array([55.0 / 3.6]))
        assert rerun.tolist() == [1.0] and corr.tolist() == [0.5]
        slower.correct(["a"], 25.0, x, np.array([0]), np.array([55.0 / 3.6]))
        corr, _ = slower.correct(["a"], 30.0, x, np.array([0]), np.array([55.0 / 3.6]))
        assert corr.tolist() == [1.0]
