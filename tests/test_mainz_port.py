"""The Mainz port's pure functions and its constants.

Replaces `test_src_port.py`, which despite its name tested `SumoTopologyEnv` on
`inverted_tree` and not this module at all. Every case here pins a defect that was
actually found while the port was built, so each one can fail.
"""
from __future__ import annotations

import math

import pytest

from src.sumo.mainz import (
    DENSITY_CRITICAL,
    DENSITY_CRITICAL_SRC,
    HERE_FEATURES,
    SPEED_ACTION_FRACTIONS,
    SRC_FEATURES,
    VEHICLE_LENGTH_M,
    _nanmean,
    jam_factor,
)


class TestJamFactor:
    """HERE does not publish its formula, so ours is stated and bounded."""

    def test_it_is_zero_at_free_flow_and_ten_at_a_standstill(self):
        assert jam_factor(60.0, 60.0) == pytest.approx(0.0)
        assert jam_factor(0.0, 60.0) == pytest.approx(10.0)

    def test_it_clips_rather_than_going_negative_above_free_flow(self):
        assert jam_factor(80.0, 60.0) == pytest.approx(0.0)

    def test_a_zero_free_flow_does_not_divide(self):
        assert jam_factor(0.0, 0.0) == pytest.approx(0.0)


class TestEmptyEdgesAreNotZeros:
    """An edge with no traffic has no density, which is not a density of zero.

    Averaging empty edges in as zeros held the mean density at 0.23 against a threshold
    of 0.3, so the congestion term never fired and the reward was 0.2 * speed with its
    other half dead.
    """

    def test_nan_is_excluded_rather_than_counted(self):
        assert _nanmean([1.0, math.nan, 3.0]) == pytest.approx(2.0)

    def test_all_nan_is_nan_and_not_zero(self):
        assert math.isnan(_nanmean([math.nan, math.nan]))

    def test_an_empty_list_is_nan_and_not_zero(self):
        assert math.isnan(_nanmean([]))


class TestTheThreshold:
    """`DENSITY_CRITICAL` is measured on this fleet, not adopted from the paper."""

    def test_it_is_the_measured_critical_density_not_srcs_published_ratio(self):
        assert DENSITY_CRITICAL == pytest.approx(0.178)
        assert DENSITY_CRITICAL_SRC == 0.3
        assert DENSITY_CRITICAL != DENSITY_CRITICAL_SRC

    def test_it_corresponds_to_the_measured_veh_per_km_and_jam(self):
        # EIDM peaks at 39.6 veh/km/lane; density here is occupancy, count * length
        # over lane-metres, so 39.6 veh/km/lane is 39.6 * 4.5 / 1000 of the roadway.
        assert 39.6 * VEHICLE_LENGTH_M / 1000.0 == pytest.approx(DENSITY_CRITICAL, abs=1e-3)
        # And against the measured jam density of 128.5 veh/km/lane that is 0.308,
        # within 3% of the 0.3 SRC publishes.
        assert 39.6 / 128.5 == pytest.approx(0.308, abs=0.005)


class TestTheActionSet:
    """SRC's 30/45/60 km/h are a half, three quarters and all of Mainz's 60 km/h limit.

    Held as fractions so the set carries to a network with a different limit; taken
    literally on a 108 km/h road they would be an order to crawl.
    """

    def test_the_fractions_reproduce_srcs_speeds_on_a_sixty_limit(self):
        assert [round(f * 60.0) for f in SPEED_ACTION_FRACTIONS] == [30, 45, 60]

    def test_the_top_action_is_the_limit_and_never_above_it(self):
        assert max(SPEED_ACTION_FRACTIONS) == 1.0


class TestTheTwoObservations:
    """The whole point of the port is that these two differ and score the same."""

    def test_here_carries_only_what_a_traffic_api_returns(self):
        assert set(HERE_FEATURES) == {"speed", "free_flow", "jam_factor", "lanes", "length_km"}

    def test_src_carries_the_four_fields_a_vehicle_cannot_obtain(self):
        assert {"density", "gap", "input_rate", "exit_rate"} <= set(SRC_FEATURES)
