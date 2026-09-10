"""The arithmetic that decides whether the pre-registered run found anything.

`paired_verdict` is what turns five paired numbers into "real" or "no effect", so
if it is wrong the conclusion is wrong whatever the simulation did. It is tested
here against cases with known answers, including the one that matters most: a
difference that looks large but is not distinguishable from noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_burst_scenario import paired_verdict  # noqa: E402


def _rows(values, key="arrivals"):
    return [{"seed": seed, key: value} for seed, value in enumerate(values)]


class TestThePreRegisteredBar:

    def test_a_consistent_difference_counts(self):
        # Every seed improves by about 20, with little spread.
        reference = _rows([100, 102, 98, 101, 99])
        arm = _rows([120, 122, 118, 121, 119])
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(20.0)
        assert se == pytest.approx(0.0, abs=1e-9)
        assert verdict == "real"

    def test_a_large_but_inconsistent_difference_does_not(self):
        # The same mean difference, but the sign flips across seeds. This is the
        # case the bar exists for: a headline of "+20 arrivals" that is noise.
        reference = _rows([100, 100, 100, 100, 100])
        arm = _rows([180, 40, 190, 30, 160])
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(20.0)
        assert se > 10.0
        assert verdict == "no effect"

    def test_a_loss_is_also_reported_as_real(self):
        # The bar is on the magnitude, so a consistent regression must be caught
        # rather than reported as no effect.
        reference = _rows([100, 102, 98, 101, 99])
        arm = _rows([80, 82, 78, 81, 79])
        mean, _, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(-20.0)
        assert verdict == "real"

    def test_pairing_is_by_seed_and_not_by_order(self):
        # The rows may arrive in any order. Pairing cannot change the MEAN -- the
        # two sums are the same however they are matched -- so the mean proves
        # nothing here and the spread is what separates the two. Matched by seed
        # the differences are +10 each; matched by order they are +110 and -90,
        # which turns a unanimous result into noise.
        reference = [{"seed": 1, "arrivals": 100}, {"seed": 2, "arrivals": 200}]
        arm = [{"seed": 2, "arrivals": 210}, {"seed": 1, "arrivals": 110}]
        mean, standard_error, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(10.0)
        assert standard_error == pytest.approx(0.0, abs=1e-9), (
            f"the paired differences have a spread of {standard_error:.1f}, which "
            "means the rows were matched by position rather than by seed"
        )
        assert verdict == "real"

    def test_an_unpaired_seed_is_dropped_rather_than_compared(self):
        reference = [{"seed": 1, "arrivals": 100}, {"seed": 2, "arrivals": 100}]
        arm = [{"seed": 1, "arrivals": 110}, {"seed": 99, "arrivals": 500}]
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        # One pair only, which is not enough to estimate a spread.
        assert verdict == "too few pairs"
        assert mean is None and se is None

    def test_a_missing_measurement_does_not_count_as_zero(self):
        # `recovery_s` is None when the queue never cleared, and treating that as
        # zero would report the fastest possible recovery for a run that never
        # recovered at all.
        reference = [{"seed": s, "recovery_s": 100.0} for s in range(5)]
        arm = [{"seed": s, "recovery_s": None} for s in range(5)]
        mean, se, verdict = paired_verdict(arm, reference, "recovery_s")
        assert verdict == "too few pairs"
