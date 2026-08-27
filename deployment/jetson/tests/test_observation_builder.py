from __future__ import annotations

import math
import time

import numpy as np
import pytest

from perception.distance import TrackedVehicle
from perception.observation_builder import BuilderConfig, ObservationBuilder, PeerState
from policy import sim_contract
from sensors.gps_reader import GpsFix


def make_vehicle(track_id: int, dist: float, lateral: float, rel: float = 0.0, rel_valid: bool = True) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
        cls=2,
        conf=0.9,
        distance_m=dist,
        lateral_m=lateral,
        rel_speed_mps=rel,
        rel_speed_valid=rel_valid,
        method="ground_plane",
    )


def fresh_fix(speed: float = 20.0) -> GpsFix:
    now = time.monotonic()
    return GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=speed, heading_deg=90.0,
        fix_quality=1, num_sats=8, hdop=1.0, altitude_m=5.0,
        utc_epoch_s=time.time(), t_mono=now, t_wall=time.time(),
    )


@pytest.fixture
def builder() -> ObservationBuilder:
    return ObservationBuilder(BuilderConfig())


def test_empty_scene_uses_spec_neutral_fallbacks(builder: ObservationBuilder) -> None:
    result = builder.build([], GpsFix(), time.monotonic())
    obs = result.obs
    # spec: neutral fallback values when nothing is sensed
    assert obs["leader_gap"] == math.inf
    assert obs["follower_gap"] == math.inf
    assert obs["nearby_av_count"] == 0
    assert obs["nearby_av_density"] == 0.0
    assert obs["nearby_av_mean_speed"] == builder.config.free_flow_speed_mps
    assert obs["nearby_av_lane_distribution"] == {}
    assert obs["cooperation"]["segment_target_speed"] == builder.config.free_flow_speed_mps
    assert obs["cooperation"]["merge_pressure"] == 0.0
    assert obs["cooperation"]["downstream_congestion_estimate"] == 0.0
    assert obs["active_vehicle_count_local"] == 0
    assert result.encoded.shape == (sim_contract.local_obs_dim(),)
    assert result.field_sources["leader_gap"] == "fallback_neutral"
    assert result.diagnostics["gps_fresh"] is False


def test_leader_selection_and_lane_split(builder: ObservationBuilder) -> None:
    vehicles = [
        make_vehicle(1, 60.0, 0.2),    # ego lane, far
        make_vehicle(2, 35.0, -0.4, rel=-2.0),  # ego lane, near -> leader
        make_vehicle(3, 25.0, -3.6),   # left lane
        make_vehicle(4, 50.0, 3.9),    # right lane
    ]
    result = builder.build(vehicles, fresh_fix(20.0), time.monotonic())
    obs = result.obs
    assert obs["leader_gap"] == 35.0
    assert obs["leader_relative_speed"] == -2.0
    assert obs["left_lane_front_gap"] == 25.0
    assert obs["right_lane_front_gap"] == 50.0
    assert obs["target_lane_front_gap"] == obs["leader_gap"]
    assert obs["ego_headway_s"] == pytest.approx(35.0 / 20.0)
    assert result.field_sources["leader_gap"] == "measured"


def test_density_and_bins_use_sim_formula(builder: ObservationBuilder) -> None:
    # 3 forward vehicles in 80 m, symmetrized to 6 over +-80 m
    vehicles = [make_vehicle(i, 20.0 + i * 10, 0.0 if i == 0 else (-3.7 if i == 1 else 3.7)) for i in range(3)]
    result = builder.build(vehicles, fresh_fix(20.0), time.monotonic())
    obs = result.obs
    assert obs["active_vehicle_count_local"] == 6
    expected_density = 6 / (2 * 80.0 / 1000.0)  # 37.5 veh/km
    assert result.diagnostics["density_veh_per_km"] == pytest.approx(expected_density, abs=0.01)
    # edges (12, 30) -> 37.5 lands in bin 2
    assert obs["local_density_bin"] == 2
    assert obs["local_density_bin"] == sim_contract.bin_index(
        expected_density, builder.config.density_bin_edges_veh_per_km
    )


def test_queue_estimate_counts_slow_vehicles(builder: ObservationBuilder) -> None:
    # ego 6 m/s; leader rel -3 -> abs 3 m/s < 5 -> queued
    vehicles = [make_vehicle(1, 30.0, 0.0, rel=-3.0), make_vehicle(2, 50.0, 0.0, rel=10.0)]
    result = builder.build(vehicles, fresh_fix(6.0), time.monotonic())
    assert result.obs["local_queue_estimate"] == 2  # 1 slow, symmetrized x2


def test_gps_speed_hold_on_staleness(builder: ObservationBuilder) -> None:
    t = time.monotonic()
    builder.build([], fresh_fix(22.0), t)
    stale = GpsFix(valid=True, speed_mps=99.0, t_mono=t - 10.0, t_wall=time.time())
    result = builder.build([], stale, t + 0.1)
    # stale fix is not trusted; last fresh speed is held
    assert result.obs["ego_speed"] == 22.0
    assert result.field_sources["ego_speed"] == "fallback_neutral"


def test_peers_populate_cooperation_fields(builder: ObservationBuilder) -> None:
    peers = [
        PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1),
        PeerState(peer_id="b", distance_m=120.0, speed_mps=26.0, lane_id=2),
    ]
    result = builder.build([], fresh_fix(20.0), time.monotonic(), peers)
    obs = result.obs
    assert obs["nearby_av_count"] == 2
    assert obs["nearby_av_mean_speed"] == pytest.approx(25.0)
    assert obs["cooperation"]["segment_target_speed"] == pytest.approx(25.0)
    assert obs["nearby_av_lane_distribution"] == {"1": 0.5, "2": 0.5}


def test_uncongested_low_speed_flag_mirrors_etiquette(builder: ObservationBuilder) -> None:
    # empty road (density 0 < 12), speed 10 < 30 - 8 -> flag on
    result = builder.build([], fresh_fix(10.0), time.monotonic())
    assert result.obs["uncongested_low_speed_flag"] is True
    result = builder.build([], fresh_fix(28.0), time.monotonic())
    assert result.obs["uncongested_low_speed_flag"] is False


def test_target_headway_feedback(builder: ObservationBuilder) -> None:
    builder.set_target_headway(2.2)
    result = builder.build([], fresh_fix(20.0), time.monotonic())
    assert result.obs["target_headway_s"] == 2.2


class TestTheFreshnessPredicateIsWhollyPinned:
    """Half of it was not, and the deleted half is what the comment above it defends."""

    def _fix_with_bound(self, *, age_s: float, bound_s: float):
        from sensors.gps_reader import GpsFix
        from sensors.phone_source import TimebaseStamp

        now = 1000.0
        return now, GpsFix(
            valid=True, lat=51.49, lon=-0.20, speed_mps=20.0, heading_deg=90.0,
            fix_quality=1, num_sats=9, hdop=0.9, t_mono=now - age_s, t_wall=0.0,
            timebase=TimebaseStamp(t_capture_mono=now - age_s, t_arrival_mono=now,
                                   bound_s=bound_s, proxy=False, estimate_id=1))

    def test_a_bound_too_wide_to_resolve_refuses_the_fix(self):
        # `timebase_unresolved` -> False survived the whole suite. The comment above
        # it names the case: "at a 10 s bound a stamp nine seconds into this clock's
        # future read as measured".
        builder = ObservationBuilder(BuilderConfig())
        now, gps = self._fix_with_bound(age_s=0.1, bound_s=10.0)
        result = builder.build([], gps, now)
        assert result.diagnostics["gps_fresh"] is False
        assert result.field_sources["ego_speed"] == "fallback_neutral"

    def test_the_bound_is_charged_before_the_staleness_comparison(self):
        # `gps_age + uncertainty_s <= stale_after` -> `gps_age <= stale_after` also
        # survived. A fix inside the window on its own is outside it once the cost of
        # not knowing when it arrived is charged -- which is the whole point of
        # carrying a bound rather than a bare age.
        builder = ObservationBuilder(BuilderConfig())
        stale_after = BuilderConfig().gps_stale_after_s

        now, inside = self._fix_with_bound(age_s=stale_after * 0.6, bound_s=0.0)
        assert builder.build([], inside, now).diagnostics["gps_fresh"] is True

        builder = ObservationBuilder(BuilderConfig())
        now, borderline = self._fix_with_bound(age_s=stale_after * 0.6,
                                               bound_s=stale_after * 0.5)
        assert builder.build([], borderline, now).diagnostics["gps_fresh"] is False, (
            "the fix is inside the window only if the bound costs nothing"
        )


class TestTheAccelerationProvenanceMatchesTheBranchTaken:
    """Two ways to fall back, reported as one."""

    def _drive(self, builder, *, samples: int, dt: float, start: float = 1000.0):
        from sensors.gps_reader import GpsFix

        result = None
        for i in range(samples):
            now = start + i * dt
            gps = GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=20.0 - 3.0 * i * dt,
                         heading_deg=90.0, fix_quality=1, num_sats=9, hdop=0.9,
                         t_mono=now, t_wall=0.0)
            result = builder.build([], gps, now)
        return result

    def test_a_window_too_short_to_fit_a_slope_is_not_called_derived(self):
        # The provenance was read off the sample COUNT, which cannot see the
        # window-span guard below it -- so a neutral 0.0 was tagged `derived`. At the
        # shipped 30 fps the ten-sample slice spans exactly the 0.3 s guard, so which
        # branch ran was decided by frame-timing noise: measured at 533 of 888 ticks
        # under constant braking.
        result = self._drive(ObservationBuilder(BuilderConfig()), samples=5, dt=0.01)
        assert result.obs["ego_acceleration"] == pytest.approx(0.0)
        assert result.field_sources["ego_acceleration"] == "fallback_neutral", (
            "a window too short to fit a slope was reported as derived"
        )

    def test_a_window_long_enough_is_derived_and_carries_the_slope(self):
        # The other side, so the fix is not just "always say fallback".
        result = self._drive(ObservationBuilder(BuilderConfig()), samples=8, dt=0.2)
        assert result.field_sources["ego_acceleration"] == "derived"
        assert result.obs["ego_acceleration"] == pytest.approx(-3.0, abs=0.2)

    def test_too_few_samples_is_also_a_fallback(self):
        result = self._drive(ObservationBuilder(BuilderConfig()), samples=2, dt=1.0)
        assert result.field_sources["ego_acceleration"] == "fallback_neutral"
