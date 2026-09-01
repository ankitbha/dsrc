"""The offset a responder can form, and what it is worth.

The Jetson answers time-sync pings and never initiates one, so it never learns
when its pong landed and cannot build a `TimeSyncSample`. These cover the
estimate it can build instead: where it sits relative to the truth, which way it
is wrong, and what its bound does and does not cover.
"""

from __future__ import annotations

import pytest

from transport.timebase import (
    ASSUMED_SKEW_PPM,
    MAX_SAMPLE_AGE_S,
    MIN_OFFSET_SAMPLES,
    OFFSET_WINDOW_S,
    ConvertedInstant,
    OneWayEstimator,
    OneWaySample,
    TimebaseNotReady,
)

NS = 1_000_000_000
# The real one, measured between this Mac and the Jetson: both count from their
# own boot and they are 67.57 hours apart.
PLANTED_OFFSET_NS = int(67.57 * 3600 * NS)


class Clock:
    """A local clock the test drives, so windows can be stepped deliberately."""

    def __init__(self, now_ns: int = 1_000 * NS) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * NS)


def arrival(clock: Clock, exchange_id: int, delay_s: float,
            offset_ns: int = PLANTED_OFFSET_NS) -> OneWaySample:
    """A ping that left the peer at `offset_ns` ahead and took `delay_s` to land."""
    t2 = clock.now_ns
    t1 = t2 + offset_ns - int(delay_s * NS)
    return OneWaySample(exchange_id, t1_remote_send_ns=t1, t2_local_recv_ns=t2)


def converged(clock: Clock, delay_s: float = 0.030) -> OneWayEstimator:
    """An estimator past its admission gate, which conversion now requires."""
    estimator = OneWayEstimator(mono_clock=clock)
    for i in range(MIN_OFFSET_SAMPLES):
        estimator.add(arrival(clock, i, delay_s=delay_s))
        clock.advance(0.25)
    return estimator


class TestWhereTheEstimateSits:

    def test_a_single_arrival_underestimates_the_offset_by_its_delay(self):
        # The property the whole design rests on. Not "approximately right" --
        # wrong in a known direction by a known amount.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        estimator.add(arrival(clock, 1, delay_s=0.030))

        estimate = estimator.estimate()
        assert estimate.offset_ns == PLANTED_OFFSET_NS - int(0.030 * NS)
        assert estimate.offset_ns < PLANTED_OFFSET_NS

    def test_the_fastest_arrival_wins_not_the_latest_or_the_mean(self):
        # The analogue of min-RTT selection. A mean would drag the estimate down
        # by the average delay instead of the smallest one, and taking the most
        # recent would let one slow arrival undo every good sample.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i, delay in enumerate([0.200, 0.008, 0.140, 0.400]):
            estimator.add(arrival(clock, i, delay_s=delay))
            clock.advance(1.0)

        assert estimator.estimate().offset_ns == PLANTED_OFFSET_NS - int(0.008 * NS)

    def test_the_estimate_is_never_above_the_truth(self):
        # One-sidedness is what makes this usable: it does not wander around the
        # offset, it sits under it. A consumer can reason about the direction of
        # its error, which it could not do with a two-sided estimate.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i, delay in enumerate([0.005, 0.001, 0.050, 0.0001, 0.3]):
            estimator.add(arrival(clock, i, delay_s=delay))
            clock.advance(0.5)
            assert estimator.estimate().offset_ns <= PLANTED_OFFSET_NS


class TestWhatTheBoundCovers:

    def test_the_bound_is_the_delay_spread_not_half_a_round_trip(self):
        # The number is reused from the round-trip estimator's field, so this
        # pins what it MEANS here. Spread of observed delays: 200 ms - 8 ms.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i, delay in enumerate([0.200, 0.008]):
            estimator.add(arrival(clock, i, delay_s=delay))
            clock.advance(1.0)

        assert estimator.estimate().rtt_min_ns == pytest.approx(int(0.192 * NS), abs=NS // 1000)

    def test_a_constant_delay_reports_no_spread_while_still_being_wrong(self):
        # The failure a spread cannot see. Every arrival delayed by exactly the
        # same 80 ms gives a spread of zero -- looking perfect -- while every
        # converted stamp is 80 ms off. This is why the bound is documented as
        # covering variation only, and why this is unfit for latency work.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i in range(5):
            estimator.add(arrival(clock, i, delay_s=0.080))
            clock.advance(1.0)

        estimate = estimator.estimate()
        assert estimate.rtt_min_ns == 0
        assert PLANTED_OFFSET_NS - estimate.offset_ns == int(0.080 * NS)

    def test_the_record_says_one_way_so_a_reader_cannot_mistake_it(self):
        clock = Clock()
        estimator = converged(clock, delay_s=0.01)

        record = estimator.to_record()
        assert record["one_way"] is True
        assert "delay_spread_ns" in record
        # Named for what it is. A key called `rtt_min_ns` in a one-way record
        # would invite exactly the comparison this estimator cannot support.
        assert "rtt_min_ns" not in record

    def test_usable_and_why_not_usable_agree_with_the_live_gate(self):
        # Same surface as `TimebaseEstimator`: a persisted line has to say
        # whether the estimate it carries is one an offline reader may convert
        # against, not just what the offset was.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)

        assert estimator.usable is False
        assert estimator.why_not_usable() == "no samples"
        record = estimator.to_record()
        assert record["usable"] is False
        assert record["why_not_usable"] == "no samples"

        estimator = converged(clock, delay_s=0.01)
        assert estimator.usable is True
        assert estimator.why_not_usable() is None
        record = estimator.to_record()
        assert record["usable"] is True
        assert record["why_not_usable"] is None


class TestConversion:

    def test_conversion_makes_a_peer_stamp_local_and_slightly_too_recent(self):
        # The end-to-end consequence, in the direction a consumer feels it: the
        # offset is under the truth, so a converted capture lands LATER than it
        # really was, and a reading looks fresher than it is by the delay floor.
        clock = Clock()
        estimator = converged(clock, delay_s=0.030)

        captured_local_truth = clock.now_ns - int(0.5 * NS)
        captured_remote = captured_local_truth + PLANTED_OFFSET_NS
        converted = estimator.to_local(captured_remote)

        assert isinstance(converted, ConvertedInstant)
        assert converted.t_remote_mono_ns - captured_local_truth == int(0.030 * NS)

    def test_a_67_hour_offset_is_what_conversion_removes(self):
        # Unconverted, `gps_age` comes out at -243,264 s against a 2.0 s
        # staleness threshold, so `gps_fresh` is False on every tick of every
        # drive and ego speed silently falls back to neutral while the loop
        # keeps producing advisories that look fine.
        clock = Clock()
        estimator = converged(clock, delay_s=0.01)

        fixed_remote = clock.now_ns + PLANTED_OFFSET_NS - int(0.4 * NS)
        age_s = (clock.now_ns - estimator.to_local(fixed_remote).t_remote_mono_ns) / NS
        assert 0.0 < age_s < 2.0

    def test_converting_before_any_sample_is_refused_not_guessed(self):
        estimator = OneWayEstimator(mono_clock=Clock())
        with pytest.raises(TimebaseNotReady):
            estimator.to_local(PLANTED_OFFSET_NS)

    def test_one_packet_is_not_enough_to_convert(self):
        # A single arrival gives a spread of zero, which reads downstream as a
        # perfectly known offset: the adapter puts it in `TimebaseStamp.bound_s`
        # and the observation builder charges it as `uncertainty_s`. Maximum trust
        # from one packet is the claim the round-trip path refuses to make, and
        # this one has no business making it either.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i in range(MIN_OFFSET_SAMPLES - 1):
            estimator.add(arrival(clock, i, delay_s=0.01))
            clock.advance(0.25)

        with pytest.raises(TimebaseNotReady):
            estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS)

        estimator.add(arrival(clock, 99, delay_s=0.01))
        assert estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS) is not None

    def test_an_offset_nobody_has_refreshed_stops_being_used(self):
        # If the phone's sync stops while camera and GPS keep streaming, the
        # offset freezes. Converting on it anyway would go on producing confident
        # stamps from a measurement nothing has confirmed for minutes.
        clock = Clock()
        estimator = converged(clock)
        assert estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS) is not None

        clock.advance(MAX_SAMPLE_AGE_S + 1.0)
        with pytest.raises(TimebaseNotReady):
            estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS)

    def test_the_bound_widens_as_the_estimate_ages(self):
        # An offset measured a while ago is worth less than one measured now,
        # whether or not anyone fitted a slope to it. Frozen, the bound claimed
        # the same certainty seconds later as at the instant of measurement.
        clock = Clock()
        estimator = converged(clock, delay_s=0.01)
        reference = clock.now_ns + PLANTED_OFFSET_NS

        near = estimator.to_local(reference).bound_ns
        far = estimator.to_local(reference + int(4.0 * NS)).bound_ns

        assert far > near
        assert far - near == pytest.approx(int(4.0 * NS * ASSUMED_SKEW_PPM / 1e6), rel=0.05)

    def test_an_instant_beyond_the_samples_is_refused(self):
        # The same guard the round-trip estimator has. An offset measured now
        # says nothing about an instant hours away, and converting anyway would
        # produce a confident number with no support under it.
        clock = Clock()
        estimator = converged(clock, delay_s=0.01)

        far = clock.now_ns + PLANTED_OFFSET_NS + int((OFFSET_WINDOW_S + 100_000) * NS)
        with pytest.raises(TimebaseNotReady):
            estimator.to_local(far)


class TestWindowing:

    def test_samples_older_than_the_window_stop_counting(self):
        # Without pruning, one lucky arrival sets the estimate for the whole
        # drive and a clock that has since drifted is never re-measured.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        estimator.add(arrival(clock, 1, delay_s=0.001))

        clock.advance(OFFSET_WINDOW_S + 1.0)
        estimator.add(arrival(clock, 2, delay_s=0.100))

        assert estimator.estimate().offset_samples == 1
        assert estimator.estimate().offset_ns == PLANTED_OFFSET_NS - int(0.100 * NS)

    def test_an_unstamped_arrival_is_refused_and_counted(self):
        estimator = OneWayEstimator(mono_clock=Clock())
        assert estimator.add(OneWaySample(1, t1_remote_send_ns=5, t2_local_recv_ns=0)) is False
        assert estimator.samples_refused == 1
        assert estimator.refused_by_reason == {"local_recv_not_stamped": 1}
        assert estimator.estimate() is None


class TestNoSkewFit:

    def test_a_one_way_estimate_never_claims_a_skew(self):
        # Fitting a slope through points each displaced by an unknown delay
        # produces a confident-looking number whose error nobody can state. The
        # round-trip estimator earns its skew term; this one does not have one,
        # and `to_local` must not silently apply a stale or zero one as though
        # it had been measured.
        clock = Clock()
        estimator = OneWayEstimator(mono_clock=clock)
        for i in range(10):
            estimator.add(arrival(clock, i, delay_s=0.01))
            clock.advance(1.0)

        estimate = estimator.estimate()
        assert estimate.skew_ppm is None
        assert estimate.skew_samples == 0


class TestRefusalReasonsAreBounded:
    """`PhoneClockAdapter` keys `proxy_reasons` on the refusal reason.

    A reason carrying a formatted age made every tenth of a second its own bucket,
    so a run summary meant to say "proxied N frames because the sync died" said N
    buckets of one instead -- unbounded, and never pruned, because `newest` stops
    moving the moment the pings stop.
    """

    def test_a_stalled_sync_produces_one_reason_not_one_per_conversion(self):
        clock = Clock()
        estimator = converged(clock, delay_s=0.01)
        clock.advance(MAX_SAMPLE_AGE_S + 1.0)

        seen = set()
        for _ in range(40):
            clock.advance(0.1)
            try:
                estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS)
            except TimebaseNotReady as exc:
                seen.add(exc.reason)

        assert len(seen) == 1, f"one stalled sync produced {len(seen)} distinct reasons: {seen}"


def test_the_published_bound_is_the_one_the_estimator_computed():
    # `test_the_bound_is_the_delay_spread_not_half_a_round_trip` asserts on
    # `estimate().rtt_min_ns`, the raw field, and never on what `to_local` actually
    # publishes -- so halving the bound on its way out survived the whole suite.
    #
    # That value is not decorative. It becomes `TimebaseStamp.bound_s`, which
    # `ObservationBuilder` charges against `gps_stale_after_s` before deciding
    # `timebase_unresolved`, so a halved bound turns a fix that should be refused
    # into one that reads as measured.
    clock = Clock()
    estimator = OneWayEstimator(mono_clock=clock)
    for i in range(MIN_OFFSET_SAMPLES):
        # A spread of observed delays, so the bound is not trivially zero.
        estimator.add(arrival(clock, i, delay_s=0.200 if i % 2 else 0.008))
        clock.advance(1.0)

    estimate = estimator.estimate()
    assert estimate is not None
    # A REMOTE instant -- the peer's clock runs `PLANTED_OFFSET_NS` ahead of ours.
    stamped = estimator.to_local(clock.now_ns + PLANTED_OFFSET_NS)
    assert stamped is not None
    assert stamped.bound_ns >= estimate.rtt_min_ns, (
        f"the published bound {stamped.bound_ns} is below the spread "
        f"{estimate.rtt_min_ns} the estimator computed"
    )
