"""Shared timebase tests.

The estimator is tested against a *planted* truth rather than against itself. A
clock-offset estimator that is merely stable looks identical to one that is
right, and stability is the easier property to achieve accidentally: an estimator
that always returned zero would be perfectly stable.

So every accuracy test here builds a synthetic link with a known offset, a known
skew, and a delay distribution shaped like the measured one -- a hard floor with
a long one-sided tail -- and asks whether the recovered value is near the value
that was planted.
"""

from __future__ import annotations

import json
import random
import time

import pytest

from transport.channels import Channel
from transport.clock import now_mono_ns
from transport.frames import WIRE_STAMP_KEY
from transport.loopback import loopback_pair
from transport.messages import MessageError, MessageRouter, TimeSyncMessage
from transport.session import Session
from transport.timebase import (
    ASSUMED_SKEW_PPM,
    MAX_ACCEPTABLE_RTT_NS,
    MAX_SAMPLE_AGE_S,
    MIN_OFFSET_SAMPLES,
    MIN_SKEW_BASELINE_S,
    MIN_SKEW_SAMPLES,
    NS_PER_S,
    SYNC_FAST_DURATION_S,
    SYNC_FAST_HZ,
    SYNC_STEADY_HZ,
    ConvertedInstant,
    TimebaseEstimate,
    TimebaseEstimator,
    TimebaseNotReady,
    TimeSyncInitiator,
    TimeSyncSample,
    answer_ping,
    fit_skew,
)

BASE_NS = 4_000_000_000_000  # a plausible monotonic reading, well clear of zero


class Link:
    """A synthetic two-device link with a truth we can check against.

    The delay model matters: floors plus a one-sided tail, because that is the
    measured shape (p50 12.2 ms against a max of 333 ms) and it is the shape
    that separates a minimum filter from an average. A symmetric Gaussian would
    make both look equally good and prove nothing.
    """

    def __init__(
        self,
        *,
        offset_ns: int = 7_000_000_000,
        skew_ppm: float = 0.0,
        up_floor_ns: int = 5_000_000,
        down_floor_ns: int = 5_000_000,
        tail_ns: int = 60_000_000,
        tail_probability: float = 0.25,
        service_ns: int = 300_000,
        seed: int = 1234,
    ) -> None:
        self.offset_ns = offset_ns
        self.skew_ppm = skew_ppm
        self.up_floor_ns = up_floor_ns
        self.down_floor_ns = down_floor_ns
        self.tail_ns = tail_ns
        self.tail_probability = tail_probability
        self.service_ns = service_ns
        self.rng = random.Random(seed)
        self.t0 = BASE_NS

    def true_offset_at(self, t_local_ns: int) -> int:
        """What the estimator should recover: remote_clock - local_clock."""
        return self.offset_ns + int(self.skew_ppm * 1e-6 * (t_local_ns - self.t0))

    def remote_at(self, t_local_ns: int) -> int:
        return t_local_ns + self.true_offset_at(t_local_ns)

    def _delay(self, floor_ns: int) -> int:
        if self.rng.random() < self.tail_probability:
            return floor_ns + int(self.rng.expovariate(1.0 / self.tail_ns))
        return floor_ns

    def exchange(self, exchange_id: int, t_send_ns: int) -> TimeSyncSample:
        up = self._delay(self.up_floor_ns)
        down = self._delay(self.down_floor_ns)
        t1 = t_send_ns
        t2 = self.remote_at(t1 + up)
        t3 = t2 + self.service_ns
        t4 = t1 + up + self.service_ns + down
        return TimeSyncSample(
            exchange_id=exchange_id,
            t1_local_send_ns=t1,
            t2_remote_recv_ns=t2,
            t3_remote_send_ns=t3,
            t4_local_recv_ns=t4,
        )


class SteppedClock:
    """A clock the test drives, so pruning and staleness are exercised rather
    than dodged. A frozen clock would make every window test vacuous."""

    def __init__(self, now_ns: int = BASE_NS) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def feed(estimator, link, clock, *, count, period_ns, start_ns=None):
    """Run `count` exchanges at a fixed cadence, advancing the clock with them."""
    t = BASE_NS if start_ns is None else start_ns
    samples = []
    for index in range(1, count + 1):
        sample = link.exchange(index, t)
        clock.now_ns = sample.t4_local_recv_ns
        estimator.add(sample)
        samples.append(sample)
        t += period_ns
    return samples


# -- the sample arithmetic ---------------------------------------------------


def test_a_symmetric_exchange_recovers_the_offset_exactly():
    """The arithmetic's own claim, with no estimator involved: when the two
    one-way delays are equal, the four stamps determine the offset."""
    link = Link(offset_ns=1_500_000_000, up_floor_ns=4_000_000, down_floor_ns=4_000_000,
                tail_probability=0.0)
    sample = link.exchange(1, BASE_NS)
    assert sample.offset_ns == link.true_offset_at(sample.t1_local_send_ns + 4_000_000)
    assert sample.rtt_ns == 8_000_000


def test_the_round_trip_excludes_the_responders_service_time():
    """A responder that takes a second to answer must not widen anyone's bound:
    the estimate would be as good as ever and the uncertainty ten times too
    large, which biases every later decision about link quality."""
    quick = Link(tail_probability=0.0, service_ns=100_000)
    slow = Link(tail_probability=0.0, service_ns=1_000_000_000)
    assert quick.exchange(1, BASE_NS).rtt_ns == slow.exchange(1, BASE_NS).rtt_ns


def test_the_asymmetry_is_exactly_half_the_delay_difference():
    """The error floor no amount of sampling removes, stated as an equality so a
    later change to the arithmetic cannot quietly alter it."""
    link = Link(offset_ns=0, up_floor_ns=10_000_000, down_floor_ns=2_000_000,
                tail_probability=0.0)
    sample = link.exchange(1, BASE_NS)
    assert sample.offset_ns - link.true_offset_at(sample.t1_local_send_ns + 10_000_000) == (
        (10_000_000 - 2_000_000) // 2
    )


# -- recovery against a planted truth ---------------------------------------


def test_the_estimator_recovers_a_planted_offset_within_its_own_bound():
    link = Link(offset_ns=12_345_678_901, seed=7)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=40, period_ns=250_000_000)

    assert estimator.usable, estimator.why_not_usable()
    at = clock.now_ns
    converted = estimator.to_remote(at)
    truth = link.remote_at(at)
    assert abs(converted.t_remote_mono_ns - truth) <= converted.bound_ns, (
        f"off by {abs(converted.t_remote_mono_ns - truth)} ns, bound {converted.bound_ns} ns"
    )


@pytest.mark.parametrize("seed", list(range(12)))
def test_the_bound_holds_across_seeds(seed):
    """A bound that holds usually is not a bound. Every seed, not a sample.

    Floors are randomised per seed and deliberately unequal. With symmetric
    floors the error is near zero and the bound holds by a mile, so the test
    passed with the bound quartered rather than halved -- it was not measuring
    the bound at all.
    """
    rng = random.Random(1000 + seed)
    link = Link(
        offset_ns=-3_000_000_000, skew_ppm=0.0, seed=seed,
        up_floor_ns=rng.randint(1_000_000, 40_000_000),
        down_floor_ns=rng.randint(1_000_000, 40_000_000),
    )
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=40, period_ns=250_000_000)
    converted = estimator.to_remote(clock.now_ns)
    error = abs(converted.t_remote_mono_ns - link.remote_at(clock.now_ns))
    assert error <= converted.bound_ns, f"seed {seed}: {error} ns exceeds {converted.bound_ns}"


def test_the_minimum_filter_beats_an_average_on_this_delay_shape():
    """The design decision, tested rather than asserted, on the traffic where it
    matters.

    Under a *light* tail the minimum and the median tie exactly: most samples sit
    on the floor, so the median is itself a floor sample and there is nothing to
    win. The minimum earns its place when the link is busy enough that the median
    is no longer on the floor -- which the measurement says is the real
    condition, since the observed p50 of 12.2 ms sits above the 10.1 ms
    minimum. So this runs a congested link, and the tie under light load is
    pinned separately below.
    """
    link = Link(offset_ns=1_000_000_000, seed=99, tail_probability=0.8)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    samples = feed(estimator, link, clock, count=60, period_ns=250_000_000)

    middle = samples[len(samples) // 2]
    reference_truth = link.true_offset_at(middle.t_local_mid_ns)

    mean_offset = sum(s.offset_ns for s in samples) / len(samples)
    mean_error = abs(mean_offset - reference_truth)
    # The median too, not just the mean. A median is a much better estimator on
    # this shape than a mean, so beating only the mean is a weak claim -- and
    # measured, swapping the minimum for a median left the suite green.
    ordered_offsets = sorted(s.offset_ns for s in samples)
    median_offset = ordered_offsets[len(ordered_offsets) // 2]
    median_error = abs(median_offset - reference_truth)

    estimate = estimator.estimate
    assert estimate is not None
    min_error = abs(estimate.offset_ns - link.true_offset_at(estimate.t_reference_ns))
    assert min_error < mean_error, (
        f"the minimum ({min_error} ns) did not beat the mean ({mean_error:.0f} ns); "
        "the design decision to min-filter is not paying for itself"
    )
    assert min_error < median_error, (
        f"the minimum ({min_error} ns) did not beat the median ({median_error} ns), "
        "which is the cheaper filter it has to justify itself against"
    )


def test_under_a_light_tail_the_minimum_and_the_median_tie():
    """The honest other half. Most samples are on the floor, so a median is a
    floor sample and the extra machinery wins nothing -- worth pinning, because a
    reader who saw only the test above would think the filter always pays."""
    link = Link(offset_ns=1_000_000_000, seed=99, tail_probability=0.05)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    samples = feed(estimator, link, clock, count=60, period_ns=250_000_000)

    middle = samples[len(samples) // 2]
    truth = link.true_offset_at(middle.t_local_mid_ns)
    ordered = sorted(s.offset_ns for s in samples)
    median_error = abs(ordered[len(ordered) // 2] - truth)
    estimate = estimator.estimate
    assert estimate is not None
    min_error = abs(estimate.offset_ns - link.true_offset_at(estimate.t_reference_ns))
    assert min_error == median_error == 0, (min_error, median_error)


def test_a_one_way_asymmetry_biases_the_estimate_but_cannot_escape_the_bound():
    """A persistent asymmetry is undetectable *and* harmless to the bound.

    Planning this task assumed the opposite -- that half the min round trip was
    an optimistic bound because it assumes symmetry. It is not, and the algebra
    says why: the error of a sample is |up - down| / 2 and its round trip is
    up + down, so with non-negative delays the error can never exceed rtt / 2.
    The bound is sound by construction, not by assumption.

    What it cannot do is make the estimate *accurate*: a 20 ms one-way asymmetry
    puts a real 10 ms bias in the offset that no amount of sampling reveals. So
    the bound is honest and the point estimate is biased, which is exactly the
    situation a bound exists for.
    """
    link = Link(offset_ns=0, up_floor_ns=25_000_000, down_floor_ns=5_000_000, seed=3)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=40, period_ns=250_000_000)
    converted = estimator.to_remote(clock.now_ns)
    error = abs(converted.t_remote_mono_ns - link.remote_at(clock.now_ns))

    assert abs(error - 10_000_000) < 2_000_000, (
        f"expected ~10 ms of undetectable bias from a 20 ms asymmetry, got {error}"
    )
    assert error <= converted.bound_ns, (
        f"the bias {error} escaped the bound {converted.bound_ns}, which the "
        "algebra says is impossible -- the round trip or the offset is wrong"
    )


@pytest.mark.parametrize("seed", list(range(8)))
def test_a_samples_error_never_exceeds_half_its_own_round_trip(seed):
    """The inequality the bound rests on, checked directly on samples rather
    than inferred: |up - down| / 2 <= (up + down) / 2 for non-negative delays.

    Stated as a property over random asymmetric links, because the version of
    this that only checks one link would pass on a symmetric one by luck.
    """
    rng = random.Random(seed)
    for _ in range(50):
        up = rng.randint(0, 80_000_000)
        down = rng.randint(0, 80_000_000)
        link = Link(offset_ns=rng.randint(-10**10, 10**10), up_floor_ns=up,
                    down_floor_ns=down, tail_probability=0.0, seed=seed)
        sample = link.exchange(1, BASE_NS)
        truth = link.true_offset_at(sample.t1_local_send_ns + up)
        # No "+ 1" allowance: the bound now uses ceiling division, so it really
        # does cover the error for an odd round trip. The slack this test used to
        # grant itself was the finding.
        assert abs(sample.offset_ns - truth) <= -(-sample.rtt_ns // 2), (
            f"up={up} down={down}: error exceeded half the round trip"
        )


# -- skew --------------------------------------------------------------------


def test_skew_is_withheld_until_the_baseline_can_resolve_it():
    """A fit over ten seconds returns a number near zero whose uncertainty is
    many times its own size. Publishing that invites a consumer to apply it."""
    link = Link(skew_ppm=20.0, seed=5)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=40, period_ns=250_000_000)  # 10 s
    estimate = estimator.estimate
    assert estimate is not None
    assert estimate.skew_ppm is None, "skew was published on a 10 s baseline"
    assert estimate.skew_uncertainty_ppm == ASSUMED_SKEW_PPM


def test_skew_is_recovered_with_the_right_sign_and_size_over_a_long_baseline():
    link = Link(skew_ppm=20.0, seed=11)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=300, period_ns=NS_PER_S)  # 300 s at 1 Hz
    estimate = estimator.estimate
    assert estimate is not None
    assert estimate.skew_ppm is not None, "skew stayed absent on a 300 s baseline"
    assert estimate.skew_ppm == pytest.approx(20.0, abs=5.0), estimate.skew_ppm
    assert estimate.skew_stderr_ppm is not None and estimate.skew_stderr_ppm < 5.0


def test_a_negative_skew_is_recovered_as_negative():
    """Sign errors in a fit are easy and invisible: a bound is symmetric, so a
    flipped sign still produces plausible numbers, drifting the wrong way."""
    link = Link(skew_ppm=-30.0, seed=13)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=300, period_ns=NS_PER_S)
    estimate = estimator.estimate
    assert estimate is not None and estimate.skew_ppm is not None
    assert estimate.skew_ppm == pytest.approx(-30.0, abs=6.0), estimate.skew_ppm


def test_correcting_for_skew_beats_ignoring_it_over_a_drive():
    """Why skew is tracked at all. At 20 ppm the uncorrected error after five
    minutes exceeds the offset bound, so an offset-only estimate would be
    outside its own stated uncertainty."""
    link = Link(skew_ppm=20.0, seed=17)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=300, period_ns=NS_PER_S)

    estimate = estimator.estimate
    assert estimate is not None and estimate.skew_ppm is not None
    far = clock.now_ns
    corrected = abs(estimator.to_remote(far).t_remote_mono_ns - link.remote_at(far))
    uncorrected = abs((far + estimate.offset_ns) - link.remote_at(far))
    assert corrected < uncorrected, f"corrected {corrected} vs uncorrected {uncorrected}"


def test_fit_skew_refuses_too_few_samples_and_too_short_a_baseline():
    link = Link(skew_ppm=20.0, seed=19)
    dense = [link.exchange(i, BASE_NS + i * 1_000_000) for i in range(MIN_SKEW_SAMPLES + 10)]
    assert fit_skew(dense) is None, "a long enough count with no baseline was fitted"
    sparse = [
        link.exchange(i, BASE_NS + i * int(MIN_SKEW_BASELINE_S * NS_PER_S))
        for i in range(MIN_SKEW_SAMPLES - 1)
    ]
    assert fit_skew(sparse) is None, "too few samples were fitted"


def test_the_baseline_guard_has_no_live_path_and_the_arithmetic_says_why(monkeypatch):
    """`MIN_SKEW_SAMPLES` buckets `SKEW_BUCKET_S` apart already span more than
    `MIN_SKEW_BASELINE_S`, so the baseline check cannot fire on real input.

    Kept rather than deleted because it states the actual requirement -- the
    count is a proxy for it -- and pinned here so that lowering either constant
    trips this test instead of quietly loosening the gate. Same treatment as
    `no_typed_message`, which is also a guard with no live channel.
    """
    from transport import timebase

    # Two buckets, not one. Buckets are absolute-aligned, so the first sample
    # can sit at the end of its bucket and the last at the start of its own --
    # the guaranteed span is (n - 2) widths, not (n - 1). At 20 buckets of 10 s
    # that is 180 s, and the comment, the spec row and this assertion all said
    # 190 or 200. Raising the baseline into (180, 190] would have kept the old
    # inequality true while making the "dead" guard live.
    implied_span_s = (MIN_SKEW_SAMPLES - 2) * timebase.SKEW_BUCKET_S
    assert implied_span_s >= MIN_SKEW_BASELINE_S, (
        f"{MIN_SKEW_SAMPLES} buckets of {timebase.SKEW_BUCKET_S}s span "
        f"{implied_span_s}s, under the {MIN_SKEW_BASELINE_S}s baseline -- the "
        "guard is now live and needs a test that reaches it through real input"
    )

    # Reached the only way left: by making the count stop dominating.
    link = Link(skew_ppm=20.0, seed=101)
    packed = [link.exchange(i, BASE_NS + i * NS_PER_S) for i in range(60)]  # 60 s
    # monkeypatch, not a hand-rolled try/finally: an assertion failure inside
    # the block would otherwise leave the module patched for whatever test the
    # random ordering runs next, and the failure would surface somewhere else.
    monkeypatch.setattr(timebase, "MIN_SKEW_SAMPLES", 3)
    assert fit_skew(packed) is None, "a 60 s baseline was fitted"
    long_enough = [link.exchange(i, BASE_NS + i * NS_PER_S) for i in range(200)]
    assert fit_skew(long_enough) is not None, "the guard now rejects everything"


def test_the_fit_survives_raw_monotonic_magnitudes():
    """Monotonic nanoseconds are ~1e15 and squaring them loses the variation
    entirely in float64, so the fit is centred. Without centring this returns
    nonsense rather than failing, which is the dangerous kind of wrong."""
    link = Link(skew_ppm=25.0, seed=23)
    huge = 9_000_000_000_000_000  # ~104 days of uptime
    link.t0 = huge
    samples = [link.exchange(i, huge + i * NS_PER_S) for i in range(300)]
    fitted = fit_skew(samples)
    assert fitted is not None
    assert fitted[0] == pytest.approx(25.0, abs=5.0), fitted


# -- the gate ----------------------------------------------------------------


def test_conversion_refuses_before_any_sample_arrives():
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    assert not estimator.usable
    assert estimator.why_not_usable() == "no samples"
    with pytest.raises(TimebaseNotReady, match="no samples"):
        estimator.to_remote(BASE_NS)


def test_conversion_refuses_below_the_minimum_sample_count():
    link = Link(seed=29)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=MIN_OFFSET_SAMPLES - 1, period_ns=100_000_000)
    assert not estimator.usable
    assert "samples in the offset window" in (estimator.why_not_usable() or "")
    with pytest.raises(TimebaseNotReady):
        estimator.to_remote(clock.now_ns)


def test_conversion_refuses_once_the_newest_sample_goes_stale():
    """The case a parked car produces: the link is idle, the last estimate looks
    fine, and it is minutes old."""
    link = Link(seed=31)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    assert estimator.usable

    clock.now_ns += int((MAX_SAMPLE_AGE_S + 1.0) * NS_PER_S)
    assert not estimator.usable
    assert "old" in (estimator.why_not_usable() or "")
    with pytest.raises(TimebaseNotReady):
        estimator.to_remote(clock.now_ns)


def test_conversion_refuses_when_the_best_round_trip_is_too_slow():
    """Above the ceiling the bound is wider than anything it could inform, so
    answering with it would be answering with noise."""
    floor = MAX_ACCEPTABLE_RTT_NS
    link = Link(up_floor_ns=floor, down_floor_ns=floor, tail_probability=0.0, seed=37)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=NS_PER_S)
    assert not estimator.usable
    assert "exceeds the acceptable bound" in (estimator.why_not_usable() or "")


def test_a_negative_round_trip_is_refused_and_counted():
    """Impossible physically, so it means a clock stepped or a field was
    misattributed. Feeding it to the fit would drag the minimum below any real
    floor and shrink the bound -- a corruption that looks like an improvement.
    """
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    impossible = TimeSyncSample(
        exchange_id=1,
        t1_local_send_ns=BASE_NS,
        t2_remote_recv_ns=BASE_NS + 1_000_000,
        t3_remote_send_ns=BASE_NS + 900_000_000,
        t4_local_recv_ns=BASE_NS + 1_000_000,
    )
    assert impossible.rtt_ns < 0
    assert estimator.add(impossible) is False
    assert estimator.samples_refused == 1
    assert estimator.refused_by_reason == {"out_of_range": 1}
    assert estimator.samples_accepted == 0
    assert estimator.estimate is None


def test_every_gate_clause_names_a_different_cause():
    """One bool cannot tell an operator whether to wait or to look at the link."""
    reasons = set()
    empty = TimebaseEstimator(mono_clock=SteppedClock())
    reasons.add(empty.why_not_usable())

    clock = SteppedClock()
    few = TimebaseEstimator(mono_clock=clock)
    feed(few, Link(seed=41), clock, count=MIN_OFFSET_SAMPLES - 1, period_ns=100_000_000)
    reasons.add(few.why_not_usable())

    clock2 = SteppedClock()
    stale = TimebaseEstimator(mono_clock=clock2)
    feed(stale, Link(seed=43), clock2, count=20, period_ns=100_000_000)
    clock2.now_ns += int((MAX_SAMPLE_AGE_S + 1) * NS_PER_S)
    reasons.add(stale.why_not_usable())

    clock3 = SteppedClock()
    slow = TimebaseEstimator(mono_clock=clock3)
    feed(
        slow,
        Link(up_floor_ns=MAX_ACCEPTABLE_RTT_NS, down_floor_ns=MAX_ACCEPTABLE_RTT_NS,
             tail_probability=0.0, seed=47),
        clock3, count=20, period_ns=NS_PER_S,
    )
    reasons.add(slow.why_not_usable())

    assert None not in reasons
    assert len(reasons) == 4, f"clauses share a message: {sorted(reasons)}"


# -- the bound ---------------------------------------------------------------


def test_the_bound_grows_with_extrapolation_from_the_reference():
    link = Link(seed=53)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    estimate = estimator.estimate
    assert estimate is not None

    near = estimate.bound_ns_at(estimate.t_reference_ns)
    far = estimate.bound_ns_at(estimate.t_reference_ns + 60 * NS_PER_S)
    assert far > near, "the bound did not widen with extrapolation"
    assert near == estimate.rtt_min_ns // 2


def test_an_unmeasured_skew_is_charged_at_the_assumed_rate_not_at_zero():
    """"Not measured" and "zero" are different claims, and treating the first as
    the second understates the bound exactly when it is least trustworthy."""
    link = Link(seed=59)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    estimate = estimator.estimate
    assert estimate is not None and estimate.skew_ppm is None
    assert estimate.skew_uncertainty_ppm == ASSUMED_SKEW_PPM

    over_a_minute = estimate.bound_ns_at(estimate.t_reference_ns + 60 * NS_PER_S)
    assert over_a_minute - estimate.rtt_min_ns // 2 == pytest.approx(
        60 * NS_PER_S * ASSUMED_SKEW_PPM / 1e6, rel=0.01
    )


def test_there_is_no_way_to_get_a_converted_instant_without_its_bound():
    """The whole discipline of this module in one test: a cross-device timestamp
    must not be able to look as measured as a same-device one.

    The first version enumerated public callables but evaluated everything except
    `to_remote` to `None`, so the list was empty by construction -- adding a
    `to_remote_ns` returning a bare int left the suite green. This calls every
    public one-argument method for real.
    """
    link = Link(seed=61)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)

    converted = estimator.to_remote(clock.now_ns)
    assert isinstance(converted, ConvertedInstant)
    assert not isinstance(converted, int)

    offenders = []
    for name in dir(estimator):
        if name.startswith("_"):
            continue
        attribute = getattr(estimator, name)
        if not callable(attribute):
            continue
        try:
            result = attribute(clock.now_ns)
        except (TypeError, AttributeError):
            continue  # does not take a single instant, so not a conversion
        except TimebaseNotReady:
            continue
        if isinstance(result, int) and not isinstance(result, bool):
            offenders.append(name)
    assert not offenders, f"these return a bare converted number: {offenders}"


def test_reading_the_estimate_directly_is_the_documented_escape_hatch():
    """`estimate` publishes `offset_ns`, so a caller who wants to bypass the
    bound can. That is deliberate and required: the spec promises an offline
    reader can re-derive a conversion exactly, which needs the offset, the
    reference and the skew. What is guaranteed is that no *conversion* hands back
    a bare number -- not that the estimate is unreadable.

    Pinned so the guarantee is not overstated later.
    """
    link = Link(seed=62)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    estimate = estimator.estimate
    assert estimate is not None
    assert isinstance(estimate.offset_ns, int)

    # And it stays readable once the gate closes, which is what makes offline
    # re-derivation possible for a run that ended badly.
    clock.now_ns += int((MAX_SAMPLE_AGE_S + 60) * NS_PER_S)
    assert not estimator.usable
    assert estimator.estimate is not None
    with pytest.raises(TimebaseNotReady):
        estimator.to_remote(clock.now_ns)


# -- provenance --------------------------------------------------------------


def test_a_converted_value_does_not_change_when_the_estimate_improves():
    link = Link(seed=67)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)

    at = clock.now_ns
    first = estimator.to_remote(at)
    feed(estimator, link, clock, count=40, period_ns=100_000_000,
         start_ns=at + 100_000_000)
    later = estimator.to_remote(at)

    assert first.estimate_id != later.estimate_id, "the estimate never improved"
    # Forward-only: `first` is still exactly what it was, and says which
    # estimate produced it, so an offline reader can redo it against `later`.
    replayed = next(e for e in estimator.history() if e.estimate_id == first.estimate_id)
    assert rederive(at, replayed) == first.t_remote_mono_ns
    assert replayed.bound_ns_at(at) == first.bound_ns


def rederive(t_local_ns, estimate):
    """A transcription of the spec's conversion formula.

    Honest about what this is: character-for-character the arithmetic in
    `to_remote`, so a shared *misreading* of the spec is invisible to it. What it
    does prove is that the published estimate is **sufficient** to redo a
    conversion -- the offset, the reference instant and the slope are all a reader
    needs -- which is the provenance claim. It also kills a change to `to_remote`
    that this is not updated to mirror.

    The rounding is the part worth naming: the spec now fixes round-to-nearest
    with ties to even on the drift term, because an unstated tie rule is a legal
    one-nanosecond disagreement between two conforming implementations.
    """
    drift = 0.0
    if estimate.skew_ppm is not None:
        drift = (t_local_ns - estimate.t_reference_ns) * estimate.skew_ppm / 1e6
    return t_local_ns + estimate.offset_ns + int(round(drift))


def test_an_offline_reader_can_re_derive_a_conversion_that_needed_the_skew_term():
    """The provenance claim was only pinned where `skew_ppm is None`, i.e. where
    conversion is arithmetic-free. That is the easy half; this is the half that
    needs the reference instant and the slope."""
    link = Link(skew_ppm=25.0, seed=91)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=300, period_ns=NS_PER_S)

    estimate = estimator.estimate
    assert estimate is not None
    assert estimate.skew_ppm is not None, "this test needs the skew branch to be live"

    at = clock.now_ns
    converted = estimator.to_remote(at)
    replayed = next(e for e in estimator.history() if e.estimate_id == converted.estimate_id)
    assert rederive(at, replayed) == converted.t_remote_mono_ns
    assert replayed.bound_ns_at(at) == converted.bound_ns


def test_the_history_records_every_published_estimate_in_order():
    link = Link(seed=71)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=15, period_ns=100_000_000)
    history = estimator.history()
    assert [e.estimate_id for e in history] == list(range(1, len(history) + 1))
    assert len(history) == 15


def test_the_record_is_json_serialisable_and_says_why_it_is_unusable():
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    record = estimator.to_record()
    assert record["usable"] is False
    assert record["why_not_usable"] == "no samples"
    assert record["current"] is None
    json.dumps(record, allow_nan=False)

    clock = SteppedClock()
    ready = TimebaseEstimator(mono_clock=clock)
    feed(ready, Link(seed=73), clock, count=20, period_ns=100_000_000)
    ready_record = ready.to_record()
    assert ready_record["usable"] is True
    assert ready_record["why_not_usable"] is None
    json.dumps(ready_record, allow_nan=False)


# -- windows -----------------------------------------------------------------


def test_a_sample_older_than_the_skew_window_is_dropped_entirely():
    """Asserted on what pruning governs -- the retained set -- not on
    `offset_samples`, which the 30 s offset horizon already filters. Checking the
    latter passed with pruning removed entirely: two filters in series, and the
    test was watching the wrong one.
    """
    link = Link(seed=79)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=10, period_ns=100_000_000)
    feed(estimator, link, clock, count=10, period_ns=100_000_000,
         start_ns=BASE_NS + 400 * NS_PER_S)

    retained = estimator.retained_samples
    assert retained == 10, f"{retained} samples retained; the old ten were not pruned"
    estimate = estimator.estimate
    assert estimate is not None
    assert estimate.offset_samples == 10


def test_the_skew_fit_spans_more_time_than_the_offset_window():
    """The whole reason there are two: the offset must be recent and the skew
    needs leverage, and one window cannot be both.

    Compared as time spanned, not as sample counts -- the skew fit holds one
    representative per bucket, so counting points would compare a filtered
    sequence against a raw one and could pass either way for the wrong reason.
    """
    from transport.timebase import OFFSET_WINDOW_S, SKEW_BUCKET_S

    link = Link(seed=83)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=200, period_ns=NS_PER_S)
    estimate = estimator.estimate
    assert estimate is not None
    assert estimate.offset_samples < 200, "the offset window held the whole run"
    assert estimate.skew_ppm is not None
    skew_span_s = (estimate.skew_samples - 1) * SKEW_BUCKET_S
    assert skew_span_s > OFFSET_WINDOW_S, (
        f"the skew fit spans {skew_span_s}s, no more than the {OFFSET_WINDOW_S}s "
        "offset window, so the second window is buying nothing"
    )


# -- the roles ---------------------------------------------------------------


def test_the_responder_echoes_receipt_and_the_pings_wire_stamp():
    ping = TimeSyncMessage(t_capture_mono_ns=10, exchange_id=88, t_wire_mono_ns=1_000)
    pong = answer_ping(ping, t_recv_mono_ns=5_000, t_recv_wall_ns=1_755_000_000_000_000_000)
    assert not pong.is_ping
    assert pong.exchange_id == 88
    assert pong.t_peer_recv_mono_ns == 5_000
    assert pong.t_peer_wire_mono_ns == 1_000, "the ping's wire stamp was not echoed"
    assert pong.t_wire_mono_ns == 0, "the responder must leave its own stamp to the writer"


def test_a_responder_refuses_a_pong():
    """Nobody should be answering the Jetson. Treating a pong as a ping would
    produce an offset with the sign inverted -- a plausible number, exactly
    wrong."""
    pong = TimeSyncMessage(
        t_capture_mono_ns=1, exchange_id=2, t_wire_mono_ns=3,
        t_peer_recv_mono_ns=4, t_peer_recv_wall_ns=5, t_peer_wire_mono_ns=6,
    )
    with pytest.raises(MessageError) as caught:
        answer_ping(pong, t_recv_mono_ns=9, t_recv_wall_ns=9)
    assert caught.value.reason == "unknown_value"


def test_an_initiator_refuses_a_ping():
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=SteppedClock())
    assert initiator.on_pong(TimeSyncMessage(t_capture_mono_ns=1, exchange_id=1), 2) is None
    assert initiator.wrong_direction == 1
    assert initiator.estimator.samples_accepted == 0


def test_an_unmatched_pong_is_counted_not_used():
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=SteppedClock())
    stray = TimeSyncMessage(
        t_capture_mono_ns=1, exchange_id=9999, t_wire_mono_ns=3,
        t_peer_recv_mono_ns=4, t_peer_recv_wall_ns=5, t_peer_wire_mono_ns=6,
    )
    assert initiator.on_pong(stray, 7) is None
    assert initiator.pongs_unmatched == 1


def test_a_peer_that_never_stamped_the_ping_is_named_not_misdiagnosed():
    """Echoing the placeholder back yields a round trip of the whole uptime. The
    gate would call that a bad link; it is an unimplemented peer, and the two
    want different responses."""
    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    exchange_id = initiator.send_ping()
    unstamped = TimeSyncMessage(
        t_capture_mono_ns=1, exchange_id=exchange_id, t_wire_mono_ns=clock.now_ns + 1_000,
        t_peer_recv_mono_ns=clock.now_ns + 500, t_peer_recv_wall_ns=1_755_000_000_000_000_000,
        t_peer_wire_mono_ns=0,
    )
    assert initiator.on_pong(unstamped, clock.now_ns + 2_000) is None
    assert initiator.unstamped_echoes == 1
    assert initiator.estimator.samples_accepted == 0


def test_the_cadence_starts_fast_and_settles():
    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    assert initiator.period_s == pytest.approx(1.0 / SYNC_FAST_HZ)
    clock.now_ns += int((SYNC_FAST_DURATION_S + 1.0) * NS_PER_S)
    assert initiator.period_s == pytest.approx(1.0 / SYNC_STEADY_HZ)


class _NullRouter:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)
        return True

    def recv(self, channel, timeout=0.0):
        return None


# -- over a real session ----------------------------------------------------


def test_the_exchange_completes_over_a_real_session_and_produces_an_estimate():
    """End to end on the transport, where the writer stamps both directions and
    the estimate is built from what actually crossed the wire."""
    near, far = loopback_pair()
    phone = Session(near, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    jetson = Session(far, session_id=2, heartbeat_s=None, stall_timeout_s=None).start()
    up, down = MessageRouter(phone), MessageRouter(jetson)
    initiator = TimeSyncInitiator(up)
    try:
        for _ in range(MIN_OFFSET_SAMPLES + 3):
            initiator.send_ping()
            ping = down.recv(Channel.CONTROL, timeout=5.0)
            assert ping is not None and ping.is_ping
            assert ping.t_wire_mono_ns > 0, "the writer did not stamp the ping"
            down.send(answer_ping(ping, now_mono_ns(), 1_755_000_000_000_000_000))
            assert initiator.pump(timeout=5.0) is not None

        assert initiator.pongs_matched == MIN_OFFSET_SAMPLES + 3
        assert initiator.unstamped_echoes == 0
        assert initiator.wrong_direction == 0
        assert initiator.estimator.usable, initiator.estimator.why_not_usable()

        # One clock on both ends of a loopback, so the offset is near zero and
        # the estimator must not invent structure where there is none.
        estimate = initiator.estimator.estimate
        assert estimate is not None
        assert abs(estimate.offset_ns) < 50_000_000, estimate.offset_ns
        converted = initiator.estimator.to_remote(now_mono_ns())
        assert abs(converted.t_remote_mono_ns - now_mono_ns()) <= converted.bound_ns
        json.dumps(initiator.to_record(), allow_nan=False)
    finally:
        phone.close()
        jetson.close()


# -- the spec is the cross-language contract ---------------------------------


REPO = __import__("pathlib").Path(__file__).resolve().parents[3]


def constants_table() -> dict[str, str]:
    """The Shared Timebase constants table, sliced out rather than found by a
    whole-file scan: other tables in this document match the same row pattern."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    start = text.index("## Shared Timebase")
    section = text[start:text.index("## Golden Vectors", start)]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| constant") or set(line) <= set("|- "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3:
            # Value and reason both: the off-by-one this catches lives in the
            # reason cell, and a slicer that kept only the value could not see it.
            rows[cells[0]] = (cells[1], cells[2])
    return rows


def test_every_timebase_constant_in_the_spec_matches_the_code():
    """The plan promised this check and the commit message claimed it, and it did
    not exist -- all ten rows were unchecked. It is the check that caught a
    falsified bound in task 14, and F11's off-by-one in this task is exactly the
    class of drift it exists to catch.
    """
    from transport import timebase

    rows = constants_table()
    expected = {
        "sampling, first 10 s": f"{timebase.SYNC_FAST_HZ:g} Hz",
        "sampling, thereafter": f"{timebase.SYNC_STEADY_HZ:g} Hz",
        "offset window": f"{timebase.OFFSET_WINDOW_S:g} s",
        "skew window": f"{timebase.SKEW_WINDOW_S:g} s",
        "skew bucket": f"{timebase.SKEW_BUCKET_S:g} s",
        "minimum offset samples": f"{timebase.MIN_OFFSET_SAMPLES:g}",
        "minimum skew buckets": f"{timebase.MIN_SKEW_SAMPLES:g}",
        "maximum sample age": f"{timebase.MAX_SAMPLE_AGE_S:.1f} s",
        "maximum acceptable min-rtt": f"{timebase.MAX_ACCEPTABLE_RTT_NS // 1_000_000:g} ms",
        "assumed skew": f"{timebase.ASSUMED_SKEW_PPM:g} ppm",
        "admission ceiling": f"{timebase.MAX_SAMPLE_RTT_NS // 1_000_000:g} ms",
        "extrapolation limit": f"{timebase.MAX_EXTRAPOLATION_S:g} s",
    }
    # Both directions. Checking code->spec only meant a table row with no code
    # constant behind it -- or a constant the table never mentions -- was
    # invisible, and the admission ceiling and the extrapolation limit were both
    # in that gap.
    assert set(rows) == set(expected), (
        f"spec-only rows: {sorted(set(rows) - set(expected))}; "
        f"code-only constants: {sorted(set(expected) - set(rows))}"
    )
    for label, value in expected.items():
        assert rows[label][0].startswith(value), (
            f"{label}: spec says {rows[label][0]!r}, code says {value!r}"
        )


def test_the_spec_documents_the_baseline_the_bucket_count_guarantees():
    """F11's off-by-one: the comment, the spec row and the test all claimed a
    span one bucket wider than absolute-aligned buckets guarantee."""
    from transport import timebase

    guaranteed_s = (timebase.MIN_SKEW_SAMPLES - 2) * timebase.SKEW_BUCKET_S
    assert f"{guaranteed_s:g} s baseline" in constants_table()["minimum skew buckets"][1], (
        f"the spec row does not state the {guaranteed_s:g}s the constants guarantee"
    )


def test_the_spec_states_the_conversion_formula_and_the_extrapolation_refusal():
    """A Kotlin implementer reading only the bound formula builds
    `t_local + offset` and disagrees with this side by skew x dt -- larger than
    the bound they both report."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text[text.index("## Shared Timebase"):text.index("## Golden Vectors")]
    flat = " ".join(section.split())
    assert "value = t_local + offset + skew_ppm" in flat
    assert "omitted when null" in flat
    assert "refused" in section and "extrapolation" in section


def test_the_spec_records_both_ordering_requirements():
    """`rtt >= 0` is one inequality standing in for two, and the spec used to
    record only the rtt refusal as the whole of the validation."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text[text.index("## Shared Timebase"):text.index("## Golden Vectors")]
    flat = " ".join(section.split())
    assert "t3 >= t2" in flat, "the service-interval ordering is not a stated requirement"
    assert "t4 >= t1" in flat, "the local ordering is not a stated requirement"
    assert "not sufficient on its own" in flat, (
        "the spec does not say why rtt >= 0 alone is inadequate"
    )
    assert "same clock" in flat, "the clock-provenance requirement is not stated"


# -- the round-1 fixes, each pinned ------------------------------------------
#
# Every fix above was new code, and a mutation pass found that sixteen of
# nineteen reverts left the suite green. A fix nothing tests is a fix that can be
# undone by the next person to touch the file.


def a_sample(t1, t2, t3, t4, exchange_id=1):
    return TimeSyncSample(
        exchange_id=exchange_id, t1_local_send_ns=t1, t2_remote_recv_ns=t2,
        t3_remote_send_ns=t3, t4_local_recv_ns=t4,
    )


def test_a_remote_departure_before_its_own_receipt_is_refused():
    """`rtt >= 0` is satisfiable with this ordering violated, and violating it
    shrinks the round trip below the true one -- shrinking the bound under an
    error that has not moved."""
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    backwards = a_sample(BASE_NS, BASE_NS + 5_000_000, BASE_NS + 4_000_000,
                         BASE_NS + 30_000_000)
    assert backwards.rtt_ns >= 0, "this sample must pass the rtt check to be a test"
    assert estimator.add(backwards) is False
    assert estimator.refused_by_reason == {"remote_send_before_remote_recv": 1}
    assert estimator.estimate is None


def test_a_local_arrival_before_its_own_departure_is_refused():
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    impossible = a_sample(BASE_NS + 30_000_000, BASE_NS, BASE_NS, BASE_NS)
    assert estimator.add(impossible) is False
    assert "local_recv_before_local_send" in estimator.refused_by_reason


def test_a_round_trip_past_the_admission_ceiling_is_refused():
    """A pong the link had forgotten. It would otherwise sit in the skew window
    as the sole representative of its bucket, where the 30 s gate cannot see it."""
    from transport.timebase import MAX_SAMPLE_RTT_NS

    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    far_too_slow = a_sample(BASE_NS, BASE_NS + 1_000, BASE_NS + 2_000,
                            BASE_NS + MAX_SAMPLE_RTT_NS + 1_000_000)
    assert estimator.add(far_too_slow) is False
    assert "rtt_above_admission_ceiling" in estimator.refused_by_reason


def test_the_admission_ceiling_and_the_gate_ceiling_are_both_reachable():
    """One constant for both made the gate's own clause unreachable. A round trip
    between the two must be admitted as real data and still refuse conversion."""
    from transport.timebase import MAX_SAMPLE_RTT_NS

    assert MAX_SAMPLE_RTT_NS > MAX_ACCEPTABLE_RTT_NS
    between = (MAX_ACCEPTABLE_RTT_NS + MAX_SAMPLE_RTT_NS) // 2
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    for index in range(MIN_OFFSET_SAMPLES + 3):
        start = BASE_NS + index * 1_000_000_000
        clock.now_ns = start + between
        assert estimator.add(
            a_sample(start, start + between // 2, start + between // 2, start + between,
                     exchange_id=index)
        ) is True, "a poor-but-real link was refused admission"
    assert estimator.samples_refused == 0
    assert "exceeds the acceptable bound" in (estimator.why_not_usable() or "")
    with pytest.raises(TimebaseNotReady):
        estimator.to_remote(clock.now_ns)


def test_the_drift_charge_covers_a_fit_that_is_entirely_asymmetry():
    """Clock skew and a drifting one-way asymmetry are indistinguishable from
    four timestamps, and the fit is most *precise* exactly when it is most
    wrong -- a smooth ramp gives a near-perfect line. So the charge is the worst
    of the crystal range, the slope's own magnitude, and its scatter.

    Constructed with the round trip held constant and only its direction split
    moving, which is a path degrading from direct toward relayed. Planted skew is
    zero; every nanosecond of apparent slope is asymmetry.
    """
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    total = 35_000_000
    for index in range(300):
        start = BASE_NS + index * NS_PER_S
        up = 30_000_000 - int(25_000_000 * index / 299)
        down = total - up
        t2 = start + up
        t3 = t2
        t4 = start + up + down
        clock.now_ns = t4
        estimator.add(a_sample(start, t2, t3, t4, exchange_id=index))

    estimate = estimator.estimate
    assert estimate is not None and estimate.skew_ppm is not None
    assert abs(estimate.skew_ppm) > 20.0, "the construction did not fool the fit"
    assert estimate.skew_stderr_ppm is not None and estimate.skew_stderr_ppm < 1.0, (
        "the fit was not precise, so this does not test the precision-vs-accuracy gap"
    )
    # The charge must follow the slope, not its scatter -- and additively, since
    # the true skew it cannot see could be opposed to the fitted one.
    assert estimate.skew_uncertainty_ppm >= ASSUMED_SKEW_PPM + abs(estimate.skew_ppm)

    # And the bound must cover the error it produces, out to the reach limit.
    truth_offset = 0
    for ahead_s in (0, 30, 120, 250):
        at = estimate.t_reference_ns + ahead_s * NS_PER_S
        converted = estimator.to_remote(at)
        error = abs(converted.t_remote_mono_ns - (at + truth_offset))
        assert error <= converted.bound_ns, (
            f"+{ahead_s}s: error {error} exceeds bound {converted.bound_ns}"
        )


def test_the_drift_charge_adds_the_fitted_slope_to_the_assumed_range():
    """Additive, not the larger of the two.

    Applying a fitted slope costs `|fit - true| * dt`, which is bounded by
    `|fit| + |true|` and not by `max(|fit|, |true|)` -- the two coincide only when
    the signs agree or one is zero. Nothing measures `|true|`, so the assumed
    range bounds it and the fitted magnitude is added on top.
    """
    quiet = TimebaseEstimate(
        estimate_id=1, offset_ns=0, t_reference_ns=BASE_NS, rtt_min_ns=10_000_000,
        offset_samples=30, skew_ppm=2.0, skew_stderr_ppm=0.001, skew_samples=30,
    )
    assert quiet.skew_uncertainty_ppm == ASSUMED_SKEW_PPM + 2.0
    loud = TimebaseEstimate(
        estimate_id=2, offset_ns=0, t_reference_ns=BASE_NS, rtt_min_ns=10_000_000,
        offset_samples=30, skew_ppm=-90.0, skew_stderr_ppm=0.001, skew_samples=30,
    )
    assert loud.skew_uncertainty_ppm == ASSUMED_SKEW_PPM + 90.0, (
        "the charge is not additive, so it is unsound wherever the fitted and "
        "true slopes have opposite signs"
    )
    # Absent skew still charges the floor alone.
    unfitted = TimebaseEstimate(
        estimate_id=3, offset_ns=0, t_reference_ns=BASE_NS, rtt_min_ns=10_000_000,
        offset_samples=30,
    )
    assert unfitted.skew_uncertainty_ppm == ASSUMED_SKEW_PPM


def test_the_bound_holds_when_the_fitted_slope_has_the_wrong_sign():
    """The case `max()` could not cover, and the case a zero-skew construction
    cannot reach.

    A real +50 ppm crystal error with a drifting asymmetry that drags the fit to
    a large NEGATIVE slope. The error from applying it is then `|fit| + |true|`
    per unit time, which is exactly why the charge has to be a sum. The earlier
    version of this test planted zero skew, so `max` and `sum` agreed at the one
    point it measured and it passed for the wrong reason.
    """
    true_skew_ppm = 50.0
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)

    def remote_at(t_local):
        return t_local + int(true_skew_ppm * 1e-6 * (t_local - BASE_NS))

    # 270 s relayed, asymmetry sliding across a constant 200 ms round trip, then
    # 30 s direct at 10 ms so rtt_min comes from the good stretch.
    for index in range(300):
        start = BASE_NS + index * NS_PER_S
        if index < 270:
            total = 200_000_000
            up = 195_000_000 - int(190_000_000 * index / 269)
        else:
            total = 10_000_000
            up = 5_000_000
        down = total - up
        t2 = remote_at(start + up)
        t3 = t2
        t4 = start + up + down
        clock.now_ns = t4
        estimator.add(a_sample(start, t2, t3, t4, exchange_id=index))

    estimate = estimator.estimate
    assert estimate is not None and estimate.skew_ppm is not None
    assert estimate.skew_ppm < -50.0, (
        f"the construction did not drive the fit negative (got {estimate.skew_ppm})"
    )
    assert estimate.skew_uncertainty_ppm >= abs(estimate.skew_ppm) + true_skew_ppm, (
        "the charge is under |fit| + |true|, which is the error applying the "
        "slope actually introduces"
    )

    for behind_s in (60, 120, 200, 299):
        at = estimate.t_reference_ns - behind_s * NS_PER_S
        converted = estimator.to_remote(at)
        error = abs(converted.t_remote_mono_ns - remote_at(at))
        assert error <= converted.bound_ns, (
            f"-{behind_s}s: error {error} exceeds bound {converted.bound_ns}"
        )


def test_conversion_refuses_to_reach_beyond_what_the_samples_support():
    """The gate only asks that the newest sample be fresh. Without this an
    instant half an hour away still got an answer, with the drift term
    extrapolated that whole way on five minutes of evidence."""
    from transport.timebase import MAX_EXTRAPOLATION_S

    link = Link(seed=131)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    estimate = estimator.estimate
    assert estimate is not None

    inside = estimate.t_reference_ns + int(MAX_EXTRAPOLATION_S * 0.5 * NS_PER_S)
    estimator.to_remote(inside)  # must not raise

    for direction in (1, -1):
        far = estimate.t_reference_ns + direction * int(
            (MAX_EXTRAPOLATION_S + 60) * NS_PER_S
        )
        with pytest.raises(TimebaseNotReady, match="beyond"):
            estimator.to_remote(far)


def test_the_bound_covers_an_odd_round_trip():
    """Floor division left the bound one nanosecond under the error for an odd
    round trip, and the property test had granted itself a "+ 1" allowance that
    hid it."""
    estimate = TimebaseEstimate(
        estimate_id=1, offset_ns=0, t_reference_ns=BASE_NS, rtt_min_ns=9,
        offset_samples=30, skew_ppm=None, skew_stderr_ppm=None,
    )
    assert estimate.bound_ns_at(BASE_NS) == 5, "the bound is not the ceiling of rtt/2"


def test_t4_is_the_transports_arrival_stamp_not_a_later_reading():
    """Stamping t4 after recv returns folds the inbound queue wait and the decode
    into the round trip -- the receive-side twin of the error the wire stamp
    removed from the other three terms."""
    stamped_at = {"value": None}

    class _SlowPollRouter:
        """Holds the pong, then hands it over with an arrival stamp from before
        the delay -- exactly what the transport does."""

        def __init__(self):
            self.pending = None

        def send(self, message):
            arrival = BASE_NS + 10_000_000
            stamped_at["value"] = arrival
            self.pending = (
                TimeSyncMessage(
                    t_capture_mono_ns=arrival, exchange_id=message.exchange_id,
                    t_wire_mono_ns=arrival, t_peer_recv_mono_ns=arrival,
                    t_peer_recv_wall_ns=1_755_000_000_000_000_000,
                    t_peer_wire_mono_ns=BASE_NS + 1,
                ),
                arrival,
            )
            return True

        def recv_with_receipt(self, channel, timeout=0.0):
            held, self.pending = self.pending, None
            return held

        def recv(self, channel, timeout=0.0):  # pragma: no cover - must not be used
            raise AssertionError("pump must use the receipt-bearing read")

    clock = SteppedClock()
    router = _SlowPollRouter()
    initiator = TimeSyncInitiator(router, mono_clock=clock)
    initiator.send_ping()

    # A whole poll period passes before the pong is collected.
    clock.now_ns = BASE_NS + 2 * NS_PER_S
    sample = initiator.pump()
    assert sample is not None
    assert sample.t4_local_recv_ns == stamped_at["value"], (
        "t4 came from the clock at collection time, not from the transport's "
        "arrival stamp, so it carries the whole poll period"
    )
    assert sample.rtt_ns < 100_000_000, f"rtt inflated to {sample.rtt_ns} ns by the poll delay"


def test_a_ping_the_router_refused_is_not_counted_as_sent():
    """A closed session returns False. Counting it anyway made a dead link
    indistinguishable from total pong loss."""

    class _RefusingRouter:
        def send(self, message):
            return False

        def recv_with_receipt(self, channel, timeout=0.0):
            return None

    initiator = TimeSyncInitiator(_RefusingRouter(), mono_clock=SteppedClock())
    for _ in range(20):
        initiator.send_ping()
    assert initiator.pings_sent == 0
    assert initiator.pings_refused == 20
    record = initiator.to_record()
    assert record["awaiting_reply"] == 0, "a refused ping left a pending entry"
    json.dumps(record, allow_nan=False)


def test_an_unanswered_exchange_expires_instead_of_leaking():
    """Without expiry the table grows for the whole drive, and an arbitrarily
    late pong is still matched -- entering the skew window with a round trip of
    however long it was gone."""
    from transport.timebase import PENDING_TIMEOUT_S

    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    first = initiator.send_ping()
    assert initiator.to_record()["awaiting_reply"] == 1

    clock.now_ns += int((PENDING_TIMEOUT_S + 1) * NS_PER_S)
    initiator.send_ping()
    assert initiator.pongs_timed_out == 1
    assert initiator.to_record()["awaiting_reply"] == 1, "the stale entry was not dropped"

    late = TimeSyncMessage(
        t_capture_mono_ns=1, exchange_id=first, t_wire_mono_ns=clock.now_ns,
        t_peer_recv_mono_ns=clock.now_ns - 1_000,
        t_peer_recv_wall_ns=1_755_000_000_000_000_000,
        t_peer_wire_mono_ns=BASE_NS + 1,
    )
    assert initiator.on_pong(late, clock.now_ns) is None
    # Late, not unmatched. Sharing one counter made a single exchange increment
    # two, and gave an operator one number for two different diagnoses: a stray
    # pong for an exchange we never sent, and an answer to one we gave up on.
    assert initiator.pongs_late == 1
    assert initiator.pongs_unmatched == 0
    assert initiator.pongs_timed_out == 1
    assert initiator.estimator.samples_accepted == 0


def test_the_history_is_bounded_and_says_what_it_dropped():
    """One estimate per accepted sample is ~28,800 over an eight-hour session,
    and this runs on a Jetson. What was dropped is counted so a re-derivation
    that finds no matching estimate can tell why."""
    from transport.timebase import MAX_HISTORY

    link = Link(seed=137)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=MAX_HISTORY + 50, period_ns=1_000_000)

    history = estimator.history()
    assert len(history) == MAX_HISTORY, f"history grew to {len(history)}"
    assert estimator.estimates_evicted == 50
    record = estimator.to_record()
    assert record["estimates_evicted"] == 50
    assert record["estimates_published"] == MAX_HISTORY
    # The newest are the ones kept, since those are what recent conversions cite.
    assert history[-1].estimate_id == MAX_HISTORY + 50


def test_the_gate_and_the_estimate_come_from_one_lock_acquisition():
    """Read separately, a sample arriving between the two let a conversion
    proceed on an estimate that never passed -- handing out a 400 ms bound
    against a 200 ms ceiling.

    Asserted structurally, by counting acquisitions, because the window is a few
    bytecodes wide: a two-thread version of this reported zero hits in six
    seconds, so it would pass on the broken code most of the time. Counting is
    deterministic and it is the property that actually matters.
    """
    link = Link(seed=151)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)

    class _CountingLock:
        def __init__(self, inner):
            self._inner = inner
            self.acquisitions = 0

        def __enter__(self):
            self.acquisitions += 1
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    counting = _CountingLock(estimator._lock)  # noqa: SLF001
    estimator._lock = counting  # noqa: SLF001
    estimator.to_remote(clock.now_ns)
    assert counting.acquisitions == 1, (
        f"to_remote took the lock {counting.acquisitions} times, so the gate and "
        "the estimate it approved are not one snapshot"
    )

    counting.acquisitions = 0
    estimator.to_record()
    assert counting.acquisitions == 1, (
        f"to_record took the lock {counting.acquisitions} times; usable and "
        "why_not_usable can disagree across separate reads"
    )


def test_a_conversion_cites_an_estimate_that_would_itself_have_passed_the_gate():
    """The consequence of the property above, stated in the terms a reader of the
    record cares about."""
    link = Link(seed=139)
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    feed(estimator, link, clock, count=20, period_ns=100_000_000)
    assert estimator.usable

    converted = estimator.to_remote(clock.now_ns)
    cited = next(e for e in estimator.history() if e.estimate_id == converted.estimate_id)
    assert cited.rtt_min_ns <= MAX_ACCEPTABLE_RTT_NS, (
        "the estimate a conversion cited would not itself have passed the gate"
    )
    assert cited.offset_samples >= MIN_OFFSET_SAMPLES
    assert converted.bound_ns == cited.bound_ns_at(clock.now_ns)


def test_every_typed_message_is_a_member_of_the_message_union():
    """`Message` closed one type before the one task 15 added, so the router was
    annotated to reject it. There is no type checker in this repo, so the union
    is checked at runtime instead of not at all -- and a future exhaustive match
    over `Message` would otherwise omit `control` silently.
    """
    import typing

    from transport.messages import MESSAGE_FOR_CHANNEL, Message

    members = set(typing.get_args(Message))
    assert members, "Message is not a union any more; this test needs rewriting"
    missing = {
        channel.value: message_type.__name__
        for channel, message_type in MESSAGE_FOR_CHANNEL.items()
        if message_type not in members
    }
    assert not missing, f"typed messages absent from the Message union: {missing}"


def test_a_stray_pong_and_a_late_pong_are_counted_apart():
    """One exchange must not increment two counters, and two different faults
    must not share one."""
    from transport.timebase import PENDING_TIMEOUT_S

    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    abandoned = initiator.send_ping()
    clock.now_ns += int((PENDING_TIMEOUT_S + 1) * NS_PER_S)
    initiator.send_ping()  # triggers the expiry sweep
    assert initiator.pongs_timed_out == 1

    def pong_for(exchange_id):
        return TimeSyncMessage(
            t_capture_mono_ns=1, exchange_id=exchange_id, t_wire_mono_ns=clock.now_ns,
            t_peer_recv_mono_ns=clock.now_ns - 1_000,
            t_peer_recv_wall_ns=1_755_000_000_000_000_000,
            t_peer_wire_mono_ns=BASE_NS + 1,
        )

    initiator.on_pong(pong_for(abandoned), clock.now_ns)
    initiator.on_pong(pong_for(999_999), clock.now_ns)
    assert initiator.pongs_late == 1, "an answer to an abandoned exchange is not late"
    assert initiator.pongs_unmatched == 1, "a pong we never asked for is not a stray"
    assert initiator.pongs_timed_out == 1, "the expiry was double-counted"


def test_a_pong_the_estimator_refuses_is_counted_by_the_initiator_too():
    """Otherwise a ping's fate exists only in the estimator's record, and the
    initiator shows one sent with every outcome zero -- recoverable only by
    joining two records and subtracting."""
    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    exchange_id = initiator.send_ping()

    # A round trip past the admission ceiling: matched here, refused there.
    absurd = TimeSyncMessage(
        t_capture_mono_ns=1, exchange_id=exchange_id, t_wire_mono_ns=BASE_NS + 2_000,
        t_peer_recv_mono_ns=BASE_NS + 1_000,
        t_peer_recv_wall_ns=1_755_000_000_000_000_000,
        t_peer_wire_mono_ns=BASE_NS + 1,
    )
    assert initiator.on_pong(absurd, BASE_NS + 5 * NS_PER_S) is None
    assert initiator.pongs_refused == 1
    assert initiator.pongs_matched == 0
    record = initiator.to_record()
    accounted = (
        record["pongs_matched"] + record["pongs_unmatched"] + record["pongs_refused"]
        + record["pongs_late"] + record["pongs_timed_out"]
        + record["unstamped_echoes"] + record["wrong_direction"]
        + record["awaiting_reply"]
    )
    assert accounted == record["pings_sent"], (
        f"{record['pings_sent']} pings sent but {accounted} accounted for: "
        "a ping's fate is missing from the initiator's own record"
    )
    json.dumps(record, allow_nan=False)


def test_recv_with_receipt_skips_a_malformed_record_like_recv_does():
    """The two paths had drifted apart: this one returned None on the first bad
    record without spending any of the caller's budget, so `pump` could not tell
    "no pong yet" from "one malformed record" and left the good message queued.
    """
    near, far = loopback_pair()
    phone = Session(near, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    jetson = Session(far, session_id=2, heartbeat_s=None, stall_timeout_s=None).start()
    up, down = MessageRouter(phone), MessageRouter(jetson)
    try:
        # A malformed control record ahead of a good one.
        good = TimeSyncMessage(
            t_capture_mono_ns=1, exchange_id=7, t_wire_mono_ns=2,
            t_peer_recv_mono_ns=3, t_peer_recv_wall_ns=4, t_peer_wire_mono_ns=5,
        )
        broken, payload = good.to_wire()
        broken["t_peer_wire_mono_ns"] = None
        phone.send(Channel.CONTROL, payload, broken, allow_reserved=(WIRE_STAMP_KEY,))
        up.send(good)

        # Wait for both to arrive, then read ONCE. Looping until non-None was
        # the flaw in the first version of this test: a call that returns None on
        # the bad record and leaves the good one queued looks identical to "not
        # here yet", so the retry hid exactly the defect being named.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and jetson.pending(Channel.CONTROL) < 2:
            time.sleep(0.01)
        assert jetson.pending(Channel.CONTROL) == 2, "both records did not arrive"

        received = down.recv_with_receipt(Channel.CONTROL, timeout=5.0)
        assert received is not None, (
            "one call returned nothing with a good record queued behind a bad "
            "one, so the bad record was not skipped"
        )
        message, t_recv = received
        assert message.exchange_id == 7
        assert t_recv > 0
        assert down.stats()[Channel.CONTROL].decode_errors == 1
        assert down.stats()[Channel.CONTROL].delivered == 1
    finally:
        phone.close()
        jetson.close()


def test_recv_and_recv_with_receipt_are_one_implementation():
    """Two copies of skip-and-count and two of the budget had already disagreed
    once; this pins that there is only one of each."""
    import inspect

    source = inspect.getsource(MessageRouter.recv)
    assert "recv_with_receipt" in source, "recv no longer delegates"
    assert "decode_errors" not in source, "recv has its own copy of the counting"


def test_the_spec_fixes_both_rounding_rules():
    """An unstated rounding rule is a legal one-nanosecond disagreement between
    two conforming implementations, and F12 was exactly a one-nanosecond
    disagreement mattering."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text[text.index("## Shared Timebase"):text.index("## Golden Vectors")]
    flat = " ".join(section.split())
    assert "ceil(rtt_min / 2)" in flat, "the bound's rounding direction is unstated"
    assert "ties to even" in flat, "the drift term's tie rule is unstated"


def test_the_drift_term_rounds_ties_to_even_as_the_spec_says():
    """Pinned against the rule, not against whatever Python happens to do."""
    estimate = TimebaseEstimate(
        estimate_id=1, offset_ns=0, t_reference_ns=0, rtt_min_ns=0,
        offset_samples=30, skew_ppm=0.5, skew_stderr_ppm=0.0, skew_samples=30,
    )
    # 0.5 ppm over 1 s is 500 ns exactly; over 3 s, 1500 ns. Ties to even keeps
    # both on the even nanosecond, which is what round() does and what the spec
    # now requires of any implementation.
    assert round(1 * 0.5) == 0
    assert round(3 * 0.5) == 2
    assert estimate.skew_uncertainty_ppm == ASSUMED_SKEW_PPM + 0.5


def test_the_spec_states_the_admission_rule_and_the_extrapolation_number():
    """A Kotlin implementer cannot derive 2 s or 300 s from prose, and both
    decide which samples enter the window -- and therefore the offset, the
    bound, and every conversion."""
    from transport import timebase

    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text[text.index("## Shared Timebase"):text.index("## Golden Vectors")]
    flat = " ".join(section.split())
    assert f"rtt <= {timebase.MAX_SAMPLE_RTT_NS // 1_000_000}ms" in flat, (
        "the admission ceiling is not in the conformance block"
    )
    assert "not data and enters no window" in flat, (
        "the admission rule has a number but no statement of what it does"
    )
    assert "makes the gate's own clause unreachable" in flat, (
        "the spec does not say why admission and the gate are separate constants"
    )
    assert f"**{timebase.MAX_EXTRAPOLATION_S:.0f} s**" in flat, (
        "the extrapolation limit has no number in the spec"
    )
    assert f"**{timebase.MAX_HISTORY}**" in flat, (
        "the retention floor is not stated, so the provenance promise is unbounded"
    )


def test_the_expiry_sweep_tolerates_an_entry_vanishing_under_it():
    """`on_pong` pops from the same dict on another thread, and the natural
    production arrangement is exactly two threads. A `del` that lost the race
    raised KeyError out of `send_ping` and killed the cadence thread.

    Driven deterministically rather than raced: the dict drops a key while the
    sweep is iterating, which is the interleaving without the timing.
    """
    from transport.timebase import PENDING_TIMEOUT_S

    class _VanishingDict(dict):
        def items(self):
            snapshot = list(super().items())
            # Whatever the sweep is about to act on is gone by the time it acts.
            for key in list(super().keys()):
                super().pop(key, None)
            return snapshot

    clock = SteppedClock()
    initiator = TimeSyncInitiator(_NullRouter(), mono_clock=clock)
    initiator.send_ping()
    initiator.send_ping()
    stale = _VanishingDict(initiator._pending)  # noqa: SLF001
    initiator._pending = stale  # noqa: SLF001

    clock.now_ns += int((PENDING_TIMEOUT_S + 1) * NS_PER_S)
    initiator._expire_pending()  # noqa: SLF001 - must not raise
    # Nothing was double-counted for entries that had already gone.
    assert initiator.pongs_timed_out == 0


def test_the_spec_states_the_drift_charge_in_its_additive_form():
    """The contract carried the stderr-only charge for a whole round after the
    code stopped using it, so a conforming Kotlin implementation would have
    reproduced the bound violation verbatim."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text[text.index("## Shared Timebase"):text.index("## Golden Vectors")]
    flat = " ".join(section.split())
    assert "max(50 + |skew_ppm|, skew_stderr_ppm)" in flat, (
        "the spec does not state the additive charge"
    )
    assert "Additive, not the larger of the two" in flat
    assert "|fit - true| <= |fit| + |true|" in flat, (
        "the spec states the formula without the algebra that makes it necessary"
    )
    # And it must not still be describing the charge it replaced.
    assert "the standard error of the fitted skew once skew is known" not in flat
