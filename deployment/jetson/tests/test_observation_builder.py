from __future__ import annotations

import math
import time

import numpy as np
import pytest

from perception import provenance
from perception.distance import TrackedVehicle
from perception.observation_builder import (
    OBS_FIELDS,
    BuilderConfig,
    ObservationBuilder,
    PeerState,
    bin_index,
)
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


def test_the_observation_is_exactly_the_declared_field_set(builder: ObservationBuilder) -> None:
    """`OBS_FIELDS` is the one place the set is written down, and both `obs`
    and `field_sources` are checked against it by NAME. A field produced but
    untagged, or tagged but not produced, is what the coverage flag exists to
    catch, and a count comparison cannot tell those apart."""
    result = builder.build([], GpsFix(), time.monotonic())
    assert set(result.obs) == set(OBS_FIELDS)
    assert set(result.field_sources) == set(OBS_FIELDS)
    assert result.diagnostics["provenance"]["covers_obs"] is True


def test_empty_scene_uses_spec_neutral_fallbacks(builder: ObservationBuilder) -> None:
    result = builder.build([], GpsFix(), time.monotonic())
    obs = result.obs
    # spec: neutral fallback values when nothing is sensed
    assert obs["leader_gap"] == math.inf
    assert obs["leader_relative_speed"] == 0.0
    assert obs["segment_target_speed"] == builder.config.free_flow_speed_mps
    assert obs["active_vehicle_count_local"] == 0
    assert result.field_sources["leader_gap"] == "fallback_neutral"
    assert result.diagnostics["gps_fresh"] is False


def test_leader_selection_ignores_the_adjacent_lanes(builder: ObservationBuilder) -> None:
    """The lane split still runs -- it is what decides which vehicle is the
    leader -- but only the ego lane's nearest reaches the observation. A
    vehicle nearer in the left lane must not become the leader, which is the
    case a builder that had dropped the lane split entirely would get wrong."""
    vehicles = [
        make_vehicle(1, 60.0, 0.2),    # ego lane, far
        make_vehicle(2, 35.0, -0.4, rel=-2.0),  # ego lane, near -> leader
        make_vehicle(3, 25.0, -3.6),   # left lane, NEARER than the leader
        make_vehicle(4, 50.0, 3.9),    # right lane
    ]
    result = builder.build(vehicles, fresh_fix(20.0), time.monotonic())
    obs = result.obs
    assert obs["leader_gap"] == 35.0
    assert obs["leader_relative_speed"] == -2.0
    assert result.field_sources["leader_gap"] == "measured"
    # All four are in range, so all four are counted, symmetrized.
    assert obs["active_vehicle_count_local"] == 8


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
    assert obs["local_density_bin"] == bin_index(
        expected_density, builder.config.density_bin_edges_veh_per_km
    )


def test_gps_speed_hold_on_staleness(builder: ObservationBuilder) -> None:
    t = time.monotonic()
    builder.build([], fresh_fix(22.0), t)
    stale = GpsFix(valid=True, speed_mps=99.0, t_mono=t - 10.0, t_wall=time.time())
    result = builder.build([], stale, t + 0.1)
    # stale fix is not trusted; last fresh speed is held
    assert result.obs["ego_speed"] == 22.0
    assert result.field_sources["ego_speed"] == "fallback_neutral"


def test_peers_set_the_segment_target_speed(builder: ObservationBuilder) -> None:
    """The one field a cooperating peer still reaches. It is the mean of the
    beacons' own speeds, computed over the receptions rather than read from
    one, so `derived` and not `measured`."""
    peers = [
        PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1),
        PeerState(peer_id="b", distance_m=120.0, speed_mps=26.0, lane_id=2),
    ]
    result = builder.build([], fresh_fix(20.0), time.monotonic(), peers)
    assert result.obs["segment_target_speed"] == pytest.approx(25.0)
    assert result.field_sources["segment_target_speed"] == "derived"

    # The control: no peers, and the field reads the configured free-flow
    # speed as a fallback rather than a measurement.
    no_peers = builder.build([], fresh_fix(20.0), time.monotonic())
    assert no_peers.obs["segment_target_speed"] == builder.config.free_flow_speed_mps
    assert no_peers.field_sources["segment_target_speed"] == "fallback_neutral"


def test_no_field_ever_carries_a_feed_class(builder: ObservationBuilder) -> None:
    # `SOURCE_FEED` is reserved for the traffic feed, which owns no
    # observation field. A V2V peer beacon is not the traffic feed, so
    # nothing in this builder should ever write "feed_derived".
    peers = [PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1)]
    for peer_list in ([], peers):
        result = builder.build([], fresh_fix(20.0), time.monotonic(), peer_list)
        assert provenance.SOURCE_FEED not in result.field_sources.values()



class TestFuseTiming:
    """`last_timings["fuse_ms"]` -- the peer/cooperation merge plus the traffic
    feed's ownership decision, timed as one sub-segment of `observe`. Same
    precedent as `TrtYoloDetector.last_timings`."""

    def test_absent_before_the_first_build_not_a_placeholder_zero(
        self, builder: ObservationBuilder
    ) -> None:
        # Unlike `last_feed_ownership`, which has a real neutral value to start
        # from, there is no build yet to report a duration for -- a caller
        # reading the key here is reading a missing value, not a zero-length
        # fuse, so the key itself must not exist until `build()` has run once.
        assert "fuse_ms" not in builder.last_timings

    def test_a_build_always_reports_a_non_negative_duration(
        self, builder: ObservationBuilder
    ) -> None:
        builder.build([], fresh_fix(20.0), time.monotonic())
        assert builder.last_timings["fuse_ms"] >= 0.0

    def test_the_reading_is_from_the_most_recent_build_not_accumulated(
        self, builder: ObservationBuilder
    ) -> None:
        builder.build([], fresh_fix(20.0), time.monotonic())
        first = builder.last_timings["fuse_ms"]
        builder.build([], fresh_fix(20.0), time.monotonic())
        second = builder.last_timings["fuse_ms"]
        # Both are real durations of the same code path, not a running total --
        # an accumulator would make the second call strictly larger than the
        # first every time, which a single sample cannot disprove on its own,
        # so the shape checked here is that the field holds one call's cost.
        assert second < first + 0.050, (first, second)

    def test_peers_present_still_produces_a_timed_fuse(
        self, builder: ObservationBuilder
    ) -> None:
        peers = [PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1)]
        builder.build([], fresh_fix(20.0), time.monotonic(), peers)
        assert builder.last_timings["fuse_ms"] >= 0.0


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

    def test_cold_start_and_a_stale_window_are_the_same_class(self):
        # "Distinct from the stale case by nothing but the class" -- and the
        # class is the same, `fallback_neutral`, in both. `_speed_slope` has
        # no third string to tell a too-short window from a stale one, and
        # that is correct: both are substitutions, and the controller (which
        # only reads the class, not the internal reason) must not have to
        # tell them apart.
        cold_start = self._drive(ObservationBuilder(BuilderConfig()), samples=2, dt=1.0)
        builder = ObservationBuilder(BuilderConfig())
        for i in range(8):
            builder.build([], GpsFix(valid=True, speed_mps=20.0 - 3.0 * i * 0.2, t_mono=1000.0 + i * 0.2, t_wall=0.0),
                          1000.0 + i * 0.2)
        stale = builder.build([], GpsFix(valid=False), 1000.0 + 8 * 0.2 + 10.0)
        assert cold_start.field_sources["ego_acceleration"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert stale.field_sources["ego_acceleration"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert cold_start.field_sources["ego_acceleration"] == stale.field_sources["ego_acceleration"]

    def test_the_threshold_is_the_gps_window_not_a_new_literal(self):
        # A dead receiver does not report `valid=False` -- it keeps returning
        # the same fix, which only grows older against the tick clock, so
        # `gps_fresh` (not an abrupt "no signal") is what actually crosses
        # `gps_stale_after_s` here. Brake hard under fresh GPS, then keep
        # calling `build()` with the SAME fix as it ages in place. At 1.9 s
        # it is still inside the window and `ego_speed` is still trusted, so
        # the slope is still `derived`; at 2.1 s (past `gps_stale_after_s`)
        # both fields go stale on the same tick and the value is exactly
        # 0.0 -- the substituted neutral, not a stale slope.
        cfg = BuilderConfig()
        builder = ObservationBuilder(cfg)
        t = 1000.0
        speed = 20.0
        for i in range(8):
            t = 1000.0 + i * 0.2
            gps = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)
            result = builder.build([], gps, t)
            speed -= 3.0 * 0.2
        assert result.field_sources["ego_acceleration"] == provenance.SOURCE_DERIVED

        held = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)

        within = builder.build([], held, t + 1.9)
        assert within.field_sources["ego_speed"] == "measured"
        assert within.field_sources["ego_acceleration"] == provenance.SOURCE_DERIVED

        past = builder.build([], held, t + 2.1)
        assert past.field_sources["ego_speed"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert past.field_sources["ego_acceleration"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert past.obs["ego_acceleration"] == 0.0

    def test_a_dead_receiver_holding_a_valid_fix_does_not_double_the_dropout_window(self):
        """A dead receiver presents as the SAME `GpsFix(valid=True)` growing
        older, not as `valid=False` -- the only shape the other tests in
        this class use. Appending a sample under `gps_fresh` on every tick,
        stamped with the tick's own clock rather than the fix's, used to let
        the window look freshly appended for a further `gps_stale_after_s`
        after `gps_age` itself had already crossed the threshold, so
        `ego_acceleration` stayed `derived` for twice as long as `ego_speed`
        did. On every tick of this held-fix dropout the two must be
        substituted together, not one two seconds after the other.
        """
        cfg = BuilderConfig()
        builder = ObservationBuilder(cfg)
        t = 1000.0
        speed = 20.0
        dt = 0.1
        for i in range(20):
            t = 1000.0 + i * dt
            gps = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)
            builder.build([], gps, t)
            speed -= 3.0 * dt

        held = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)
        dropout_start = t
        for i in range(1, 40):
            now = dropout_start + i * dt
            result = builder.build([], held, now)
            speed_src = result.field_sources["ego_speed"]
            accel_src = result.field_sources["ego_acceleration"]
            assert provenance.is_substituted(accel_src) == provenance.is_substituted(speed_src), (
                now - dropout_start, speed_src, accel_src,
            )

    def test_the_guard_boundary_changes_the_value_not_just_the_label(self):
        """D6 replaces a frozen slope with the neutral 0.0 in the VALUE the
        sensing controller compares against its event threshold, not only in
        a record a reader might assume is record-only. So a tick on either
        side of the staleness guard reports a different `ego_acceleration`,
        not merely a different `field_sources` label.
        """
        cfg = BuilderConfig()
        builder = ObservationBuilder(cfg)
        t = 1000.0
        speed = 20.0
        for i in range(8):
            t = 1000.0 + i * 0.2
            gps = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)
            result = builder.build([], gps, t)
            speed -= 3.0 * 0.2
        assert result.field_sources["ego_acceleration"] == provenance.SOURCE_DERIVED
        assert result.obs["ego_acceleration"] != pytest.approx(0.0)

        held = GpsFix(valid=True, speed_mps=speed, t_mono=t, t_wall=0.0)
        past = builder.build([], held, t + 2.1)
        assert past.field_sources["ego_acceleration"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert past.obs["ego_acceleration"] == pytest.approx(0.0)
        assert past.obs["ego_acceleration"] != pytest.approx(result.obs["ego_acceleration"])

    def test_a_dropout_gap_is_not_fitted_as_a_slope_once_gps_returns(self):
        """A real dropout (`gps.valid` False, not a stale fix aging in
        place) leaves a hole in the MIDDLE of the window, not at its edge.
        The span guard in `_speed_slope` only asks whether the window's
        first and last timestamps are 0.3 s apart, so it cannot see the
        hole -- and before `speed_samples` was cleared on every non-fresh
        tick, the pre-dropout samples were still sitting in the window when
        GPS returned. A tick after an 8 s dropout, with speed dropping from
        a constant 25 m/s to a constant 10 m/s, used to fit a slope across
        nine pre-dropout samples and one post-dropout sample and report it
        as `derived` -- a real acceleration measurement of an interval that
        contains no measurement at all.
        """
        cfg = BuilderConfig()
        builder = ObservationBuilder(cfg)
        dt = 0.1
        t = 1000.0
        for i in range(10):
            t = 1000.0 + i * dt
            builder.build([], GpsFix(valid=True, speed_mps=25.0, t_mono=t, t_wall=0.0), t)

        dropout_ticks = round(8.0 / dt)
        for i in range(1, dropout_ticks + 1):
            now = t + i * dt
            builder.build([], GpsFix(valid=False, t_mono=now, t_wall=0.0), now)
        t += dropout_ticks * dt

        # The instant GPS returns, at a different (but also constant)
        # speed: the window must not be trusted yet, however large the
        # slope across the gap would have been.
        t += dt
        first_return = builder.build(
            [], GpsFix(valid=True, speed_mps=10.0, t_mono=t, t_wall=0.0), t
        )
        assert first_return.field_sources["ego_acceleration"] == (
            provenance.SOURCE_FALLBACK_NEUTRAL
        )
        assert first_return.obs["ego_acceleration"] == pytest.approx(0.0)

        # Once the window has refilled from fresh, post-dropout samples
        # only, the label returns to `derived` -- and now with the value the
        # window actually supports: zero, because the speed either side of
        # this later window is the same constant 10 m/s.
        result = first_return
        for i in range(1, 6):
            t += dt
            result = builder.build(
                [], GpsFix(valid=True, speed_mps=10.0, t_mono=t, t_wall=0.0), t
            )
        assert result.field_sources["ego_acceleration"] == provenance.SOURCE_DERIVED
        assert result.obs["ego_acceleration"] == pytest.approx(0.0, abs=1e-6)

    def test_speed_slope_refuses_a_stale_window_on_its_own_authority(self):
        """`_speed_slope` carries its own `if not gps_fresh: return 0.0,
        False` rather than trusting the window to already be empty by the
        time it is called with `gps_fresh=False`. Every other test in this
        class reaches `_speed_slope` through `build`, which clears the
        window on the same tick it would pass `gps_fresh=False` in, so a
        window with samples never actually reaches this guard through
        `build`. Populate the window directly and call `_speed_slope`
        itself, bypassing `build`, so the guard is exercised on a window it
        did not clear.
        """
        builder = ObservationBuilder(BuilderConfig())
        t = 1000.0
        for i in range(5):
            builder._ego.speed_samples.append((t + i * 0.2, 20.0 - 3.0 * i * 0.2))
        span = builder._ego.speed_samples[-1][0] - builder._ego.speed_samples[0][0]
        assert len(builder._ego.speed_samples) >= 3
        assert span > 0.3

        accel, derived = builder._speed_slope(False)
        assert derived is False
        assert accel == pytest.approx(0.0)


class TestCoverageAndMissingness:
    """`field_sources` covers every field `obs` carries, and `missingness` is
    a statement about that whole set."""

    @staticmethod
    def _gps(speed: float, t: float) -> GpsFix:
        return GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=speed, heading_deg=90.0,
                      fix_quality=1, num_sats=8, hdop=1.0, t_mono=t, t_wall=0.0)

    def _warmed_up(self) -> tuple[ObservationBuilder, float]:
        """A builder with enough fresh, constant-speed history that
        `ego_acceleration` reads `derived` (the ordinary case) rather than a
        fresh builder's own first-tick `fallback_neutral`."""
        builder = ObservationBuilder(BuilderConfig())
        t = 1000.0
        for i in range(5):
            t = 1000.0 + i * 0.1
            builder.build([], self._gps(20.0, t), t)
        return builder, t

    def test_field_sources_covers_every_field(self):
        builder, t = self._warmed_up()
        result = builder.build([], self._gps(20.0, t + 0.1), t + 0.1)
        assert set(result.field_sources) == set(OBS_FIELDS)
        assert len(result.field_sources) == len(OBS_FIELDS) == 7

    def test_the_no_peer_no_vehicle_fresh_gps_tick_pins_missingness(self):
        """A lone instrumented car on an empty road, with a fresh fix.

        Three of the seven are substituted: `leader_gap` and
        `leader_relative_speed` have no leader to read, and
        `segment_target_speed` has no peer, so it reports the configured
        free-flow speed. Two are `derived_empty` -- the count really is zero,
        nothing failed. `ego_speed` is measured and `ego_acceleration` is
        derived from a window of measurements. So 3/7 = 0.429.

        The number moved from 0.718 over 39 fields, and it is not the same
        statistic: the 39-field version was dominated by constants for
        sensors this rig does not have, which is a fact about the contract
        rather than about the road.
        """
        builder, t = self._warmed_up()
        result = builder.build([], self._gps(20.0, t + 0.1), t + 0.1)

        assert result.diagnostics["provenance"]["fields"] == 7
        assert result.diagnostics["missingness"] == 0.429
        assert result.diagnostics["provenance"]["by_source"]["derived_empty"] == 2
        assert result.diagnostics["provenance"]["covers_obs"] is True

        assert result.field_sources["local_density_bin"] == provenance.SOURCE_DERIVED_EMPTY
        assert result.field_sources["active_vehicle_count_local"] == provenance.SOURCE_DERIVED_EMPTY
        assert result.field_sources["ego_acceleration"] == provenance.SOURCE_DERIVED
        assert result.field_sources["leader_gap"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert result.field_sources["leader_relative_speed"] == provenance.SOURCE_FALLBACK_NEUTRAL
        assert result.field_sources["segment_target_speed"] == provenance.SOURCE_FALLBACK_NEUTRAL

    def test_by_source_sums_to_fields_across_a_sweep(self):
        builder, t = self._warmed_up()
        vehicles_options = [
            [],
            [make_vehicle(1, 30.0, 0.0)],
            [make_vehicle(i, 20.0 + i, 0.0) for i in range(6)],
        ]
        peers_options: list = [
            None,
            [PeerState(peer_id="a", distance_m=50.0, speed_mps=20.0, lane_id=1)],
            [PeerState(peer_id="b", distance_m=50.0, speed_mps=20.0, lane_id=None)],
        ]
        for vehicles in vehicles_options:
            for peers in peers_options:
                result = builder.build(vehicles, self._gps(20.0, t + 0.1), t + 0.1, peers)
                prov = result.diagnostics["provenance"]
                assert sum(prov["by_source"].values()) == prov["fields"] == 7

    def test_covers_obs_checks_names_not_just_a_count(self):
        # Same key count as the real seven (one field dropped, one name the
        # observation never carries put in its place) -- a count comparison
        # cannot tell this apart from real coverage.
        good = {name: "measured" for name in OBS_FIELDS}
        assert ObservationBuilder._covers_obs(good) is True
        bad = dict(good)
        del bad["ego_speed"]
        bad["not_a_real_field"] = "measured"
        assert len(bad) == len(good)
        assert ObservationBuilder._covers_obs(bad) is False


class TestTheDensityPath:

    @staticmethod
    def _gps(speed: float, t: float) -> GpsFix:
        return GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=speed, heading_deg=90.0,
                      fix_quality=1, num_sats=8, hdop=1.0, t_mono=t, t_wall=0.0)

    def test_zero_in_range_tracks_are_derived_empty(self):
        builder = ObservationBuilder(BuilderConfig())
        result = builder.build([], self._gps(20.0, 1000.0), 1000.0)
        assert result.obs["local_density_bin"] == 0
        assert result.field_sources["local_density_bin"] == provenance.SOURCE_DERIVED_EMPTY
        assert result.field_sources["active_vehicle_count_local"] == provenance.SOURCE_DERIVED_EMPTY

    def test_one_in_range_track_is_derived_at_bin_one(self):
        # Under shipped constants (edges 12.0, 30.0; symmetrize_counts=True;
        # effective_range_m=80.0) density = 12.5 * n_forward, so bin 0 <=> 0
        # in-range tracks and bin 1 at a single one -- what makes gating the
        # disagreement rule on `derived_empty` (D4) a real choice rather than
        # a no-op.
        builder = ObservationBuilder(BuilderConfig())
        vehicles = [make_vehicle(1, 30.0, 0.0, rel=0.0)]
        result = builder.build(vehicles, self._gps(20.0, 1000.0), 1000.0)
        assert result.obs["local_density_bin"] == 1
        assert result.field_sources["local_density_bin"] == provenance.SOURCE_DERIVED

    def test_density_counts_tracks_whose_speed_was_never_measurable(self):
        """Density counts tracks, not speeds. Six in range with no usable
        relative speed is still six vehicles, so the bin is `derived` from a
        real count rather than falling back -- the distinction the removed
        `local_queue_estimate` used to draw on the same population."""
        vehicles = [make_vehicle(i, 20.0 + i * 5, 0.0, rel_valid=False) for i in range(6)]
        builder = ObservationBuilder(BuilderConfig())
        result = builder.build(vehicles, self._gps(20.0, 1000.0), 1000.0)
        assert result.obs["active_vehicle_count_local"] == 12
        assert result.field_sources["local_density_bin"] == provenance.SOURCE_DERIVED


class TestLastDetectionAge:

    @staticmethod
    def _gps(speed: float, t: float) -> GpsFix:
        return GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=speed, heading_deg=90.0,
                      fix_quality=1, num_sats=8, hdop=1.0, t_mono=t, t_wall=0.0)

    def test_none_before_the_first_in_range_detection(self):
        builder = ObservationBuilder(BuilderConfig())
        result = builder.build([], self._gps(20.0, 1000.0), 1000.0)
        assert result.diagnostics["last_detection_age_s"] is None

    def test_zero_on_the_tick_that_has_one(self):
        builder = ObservationBuilder(BuilderConfig())
        vehicles = [make_vehicle(1, 30.0, 0.0)]
        result = builder.build(vehicles, self._gps(20.0, 1000.0), 1000.0)
        assert result.diagnostics["last_detection_age_s"] == 0.0

    def test_it_grows_monotonically_after_the_last_detection(self):
        builder = ObservationBuilder(BuilderConfig())
        vehicles = [make_vehicle(1, 30.0, 0.0)]
        builder.build(vehicles, self._gps(20.0, 1000.0), 1000.0)
        ages = []
        for dt in (1.0, 2.0, 5.0):
            result = builder.build([], self._gps(20.0, 1000.0 + dt), 1000.0 + dt)
            ages.append(result.diagnostics["last_detection_age_s"])
        assert ages == sorted(ages)
        assert ages == [1.0, 2.0, 5.0]


class TestBuilderConfigFromFullConfig:
    """B11 (validation round 2): the merge `run_demo.build_components` and
    `src.analysis.observation_parity._production_builder_config` both need
    -- config["observation"] plus the gps override -- lives once here, not
    as a copy at each call site.
    """

    def _config(self, **overrides):
        base = {
            "observation": {"effective_range_m": 80.0, "assumed_lane": 1},
            "gps": {"stale_after_s": 2.0},
            "v2v": {"range_m": 150.0},
        }
        base.update(overrides)
        return base

    def test_reads_the_observation_section(self):
        cfg = BuilderConfig.from_full_config(self._config())
        assert cfg.effective_range_m == 80.0

    def test_a_key_this_builder_no_longer_has_is_ignored_rather_than_raising(self):
        """`config.yaml`'s observation section still carries `assumed_lane`,
        which `run_demo` reads for the V2V beacon, and the bin edges the
        removed fields used. `from_dict` takes only the names it declares, so
        a shipped config keeps loading."""
        cfg = BuilderConfig.from_full_config(self._config())
        assert not hasattr(cfg, "assumed_lane")

    def test_overrides_gps_stale_after_s_from_the_gps_section(self):
        cfg = BuilderConfig.from_full_config(self._config(gps={"stale_after_s": 9.0}))
        assert cfg.gps_stale_after_s == 9.0

    def test_a_value_named_in_both_the_observation_section_and_an_override_takes_the_override(self):
        """The override sections win, matching build_components' own
        assignment order (observation section first, then overwritten)."""
        cfg = BuilderConfig.from_full_config(
            self._config(
                observation={"effective_range_m": 80.0, "gps_stale_after_s": 1.0},
                gps={"stale_after_s": 9.0},
            )
        )
        assert cfg.gps_stale_after_s == 9.0
