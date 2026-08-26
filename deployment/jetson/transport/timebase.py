"""Shared timebase: the one sanctioned way to compare the two devices' clocks.

`clock.py` forbids comparing a phone monotonic value with a Jetson one, and for
good reason -- they are unrelated counters. This module relates them, and the
whole discipline is that it never hands back a bare number: a converted instant
carries the uncertainty it was produced with, so a cross-device timestamp cannot
be mistaken for a same-device one.

The estimator takes the offset from the *minimum* round trip in a window rather
than an average. Measured over a real link the delay has a hard floor and a long
tail -- p50 12.2 ms against a max of 333 ms -- and the least-delayed sample is
the one least distorted by asymmetry. An average draws the tail in; a minimum
discards it. That is also why this does not follow the EMA in
`sensors/time_sync.py`: that tracker averages wall-minus-GPS-UTC, whose noise is
roughly symmetric, so the precedent does not transfer.

Offset and skew are fitted over different windows because they need different
baselines. Two independent crystals differ by 10-50 ppm, so at 20 ppm a ten
minute drive accumulates ~12 ms -- more than the offset bound itself -- while
over ten seconds the same skew moves 0.2 ms, far under the noise floor. A window
short enough to keep the offset fresh cannot see skew at all.

Stdlib only, like the rest of transport/, and no I/O: the estimator is fed
samples and knows nothing about sessions.
"""

from __future__ import annotations

import itertools
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from transport.channels import Channel
from transport.clock import MonoClock, WallClock, now_mono_ns, now_wall_ns
from transport.messages import MessageError, TimeSyncMessage

REASON_WRONG_DIRECTION = "unknown_value"

# Sampling: fast until the first estimate is trustworthy, then cheap. The first
# advisory of a drive should be alignable within seconds, and after that two
# ~200 B frames per second is about 0.1% of the camera stream.
SYNC_FAST_HZ = 4.0
SYNC_FAST_DURATION_S = 10.0
SYNC_STEADY_HZ = 1.0

OFFSET_WINDOW_S = 30.0
SKEW_WINDOW_S = 300.0

# Gate. Below MIN_OFFSET_SAMPLES the minimum is not yet a floor, so the bound
# derived from it would be a guess wearing a number's clothes.
MIN_OFFSET_SAMPLES = 5
MAX_SAMPLE_AGE_S = 5.0
# A link-health floor, not a bound-width limit. The distinction matters now that
# the drift term routinely dominates: a converged estimator can hand out a 90 ms
# bound on a 10 ms link, far wider than the 100 ms this clause refuses, and that
# is correct -- the bound is honest and only the consumer knows what width it can
# use. What this clause catches is a link so poor that the offset itself is barely
# constrained by any sample in the window.
MAX_ACCEPTABLE_RTT_NS = 200_000_000

# Admission, which is a different question from the gate. The gate above asks
# "is this link good enough to inform anything"; this asks "is this sample real
# data at all". Using one constant for both made the gate's own clause
# unreachable -- a sample it would have rejected never got in. A 400 ms round
# trip is a poor link and belongs in the window; a 600 s one is a pong the link
# had forgotten, and it would sit in the skew window as the sole representative
# of its bucket where the 30 s gate can never see it.
MAX_SAMPLE_RTT_NS = 2_000_000_000

# Skew is withheld until it can be resolved. A fit over too short a baseline
# returns a value near zero whose uncertainty is many times its own size, and
# publishing that invites a consumer to apply it as a measurement.
# The requirement is leverage in time, and the bucket width times the minimum
# count already guarantees it: buckets are absolute-aligned, so 20 of them
# guarantee (20 - 2) x 10 = 180 s, not 190 -- the first sample can sit at the end
# of its bucket and the last at the start of its own. 180 s still clears the
# baseline below, so the
# runtime baseline check below has no live path today. It is kept because it
# states the actual requirement -- a future change to either constant could drop
# the implied span under it -- and a test pins the arithmetic so that change gets
# noticed rather than silently loosening the gate.
MIN_SKEW_BASELINE_S = 120.0
MIN_SKEW_SAMPLES = 20
# The fit runs over one representative per bucket -- the least-delayed sample in
# it -- not over every sample. Fitting raw samples puts the whole delay tail in
# the residuals: at 1 Hz with a 60 ms tail that noise is tens of ms against a
# 6 ms signal from 20 ppm over 300 s, and the recovered slope came out with the
# wrong sign. Min-filtering first is what makes the signal visible.
SKEW_BUCKET_S = 10.0

# The drift charge, and it is a FLOOR rather than a fallback. Clock skew and a
# linearly-drifting one-way path asymmetry are observationally identical from
# four timestamps -- both make the min-filtered offset move at a constant rate --
# so a fitted slope can be entirely asymmetry and the fit cannot tell. Its
# standard error is no help: it measures scatter about the line, and a smoothly
# drifting asymmetry produces a near-perfect line, so the fit is at its most
# precise exactly when it is most wrong. Measured, that put a 60 ms error under a
# 17.5 ms bound.
#
# This constant is therefore load-bearing rather than a fallback: after the fix
# it is the ONLY thing bounding the true skew, and the whole guarantee rests on
# it. 50 ppm is the top of the ordinary range for initial accuracy at 25 C -- it
# is NOT the figure over temperature and aging, and a phone in a windscreen mount
# in the sun is exactly that excursion. A true 100 ppm would leave the bound
# short by 50 ppm x 300 s = 15 ms, more than rtt_min/2 on any healthy link. It is
# recorded here as a stated premise of the guarantee, not a measured fact.
ASSUMED_SKEW_PPM = 50.0

# How far outside the reference instant a conversion may reach. Beyond this the
# fit is being extended into a region no sample supports, and the drift term
# grows without any evidence behind it.
MAX_EXTRAPOLATION_S = SKEW_WINDOW_S

# How long an unanswered exchange stays matchable. Generous against the round
# trip, short against the drive: a pong later than this is not a slow answer, it
# is an answer to a question the link has forgotten.
PENDING_TIMEOUT_S = 10.0

# How long an abandoned exchange stays distinguishable from one we never sent.
# Past it a pong is a stray, which is the honest label: nothing that old can
# produce a usable sample, because the admission ceiling refuses the round trip.
LATE_WINDOW_S = 60.0

# The published history is bounded, which bounds the re-derivation promise with
# it: at the steady 1 Hz this is about 68 minutes, inside a long drive. A
# ConvertedInstant older than that carries an estimate_id that no longer resolves,
# so the spec says an implementation retains at least this many rather than
# promising every conversion is re-derivable forever. Evictions are counted so a
# reader that finds no matching estimate can tell why instead of guessing.
MAX_HISTORY = 4096

NS_PER_S = 1_000_000_000


class TimebaseNotReady(RuntimeError):
    """Conversion refused: no estimate good enough to answer with.

    Deliberately an error rather than a widened number. A wide bound is easy to
    ignore, and this project has already shipped a report where every solve
    failed and the output read as perfect. A caller that cannot handle a refusal
    logs the raw same-device stamp and says which one it logged.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TimeSyncSample:
    """One completed exchange. Four stamps, two clocks, no cross-clock maths
    except the two differences the protocol sanctions."""

    exchange_id: int
    t1_local_send_ns: int
    t2_remote_recv_ns: int
    t3_remote_send_ns: int
    t4_local_recv_ns: int
    # The responder's WALL clock at receipt, carried through because the same
    # midpoint arithmetic on the two wall clocks is an independent estimate of
    # the same offset -- and both ends run NTP, so that pair does not drift while
    # the monotonic pair does. The difference of the two slopes is the true
    # monotonic skew, which is the quantity ASSUMED_SKEW_PPM stands in for and
    # the only thing that can say whether that premise holds. Optional because a
    # peer may not supply it; a consumer must treat None as "no cross-check",
    # never as zero.
    t2_remote_recv_wall_ns: int | None = None
    # And this side's arrival wall stamp, from our own reader at the same instant
    # as t4. With both ends' wall stamps taken at arrival rather than at handling,
    # the wall pair refers to the same two instants the monotonic pair does.
    t4_local_recv_wall_ns: int | None = None

    @property
    def rtt_ns(self) -> int:
        """Round trip with the responder's own service time removed, so a slow
        responder inflates neither this nor the bound derived from it."""
        return (self.t4_local_recv_ns - self.t1_local_send_ns) - (
            self.t3_remote_send_ns - self.t2_remote_recv_ns
        )

    @property
    def offset_ns(self) -> int:
        """remote_clock - local_clock. Both terms are same-clock differences."""
        return (
            (self.t2_remote_recv_ns - self.t1_local_send_ns)
            + (self.t3_remote_send_ns - self.t4_local_recv_ns)
        ) // 2

    @property
    def t_local_mid_ns(self) -> int:
        """The local instant this sample's offset describes."""
        return (self.t1_local_send_ns + self.t4_local_recv_ns) // 2


@dataclass(frozen=True)
class TimebaseEstimate:
    estimate_id: int
    offset_ns: int
    t_reference_ns: int
    rtt_min_ns: int
    offset_samples: int
    skew_ppm: float | None = None
    skew_stderr_ppm: float | None = None
    skew_samples: int = 0

    @property
    def skew_uncertainty_ppm(self) -> float:
        """What to charge per second of extrapolation.

        Applying a fitted slope costs `|fit - true| * dt`, and

            |fit - true| <= |fit| + |true|

        so the charge is a **sum**, not the larger of the two. Nothing here
        measures `|true|` -- that is the whole difficulty, since a drifting path
        asymmetry and a real crystal error are indistinguishable from four
        timestamps -- so `ASSUMED_SKEW_PPM` bounds it, and the fitted magnitude
        is added on top rather than compared against.

        Taking the maximum instead was unsound wherever `fit` and `true` have
        opposite signs, which is precisely what a drifting asymmetry larger than
        the crystal error produces. Measured: a true +50 ppm fitted as -497.9 ppm
        put a 164 ms error under a 154 ms bound, while the sum lands on 547.9 ppm
        -- exactly the required `|fit - true|`, which is how you can tell the
        algebra is tight rather than merely bigger.

        Never zero, because "not measured" and "zero" are different claims and
        only one of them is true here.
        """
        if self.skew_ppm is None or self.skew_stderr_ppm is None:
            return ASSUMED_SKEW_PPM
        return max(ASSUMED_SKEW_PPM + abs(self.skew_ppm), self.skew_stderr_ppm)

    def bound_ns_at(self, t_local_ns: int) -> int:
        """Half the least asymmetry we cannot rule out, plus what drift accrues
        over the extrapolation from the reference instant."""
        drift = abs(t_local_ns - self.t_reference_ns) * self.skew_uncertainty_ppm / 1e6
        # Ceiling, not floor. For an odd round trip the error reaches
        # (rtt + 1) // 2 while floor division bounds it at (rtt - 1) // 2, so the
        # unconditional claim was off by exactly one nanosecond -- and the test
        # that proved the inequality had written itself a "+ 1" allowance the
        # code did not have.
        return -(-self.rtt_min_ns // 2) + int(math.ceil(drift))

    def to_record(self) -> dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "offset_ns": self.offset_ns,
            "t_reference_ns": self.t_reference_ns,
            "rtt_min_ns": self.rtt_min_ns,
            "offset_samples": self.offset_samples,
            "skew_ppm": self.skew_ppm,
            "skew_stderr_ppm": self.skew_stderr_ppm,
            "skew_samples": self.skew_samples,
            "skew_uncertainty_ppm": self.skew_uncertainty_ppm,
        }


@dataclass(frozen=True)
class ConvertedInstant:
    """A cross-device instant and what it is worth.

    There is deliberately no accessor returning the value alone. A converted
    timestamp must not be able to look as measured as a same-device one, and
    `estimate_id` is what lets an offline reader re-derive it later against a
    better estimate.
    """

    t_remote_mono_ns: int
    bound_ns: int
    estimate_id: int

    def to_record(self) -> dict[str, Any]:
        return {
            "t_remote_mono_ns": self.t_remote_mono_ns,
            "bound_ns": self.bound_ns,
            "estimate_id": self.estimate_id,
        }


def min_filtered(samples: Iterable[TimeSyncSample]) -> list[TimeSyncSample]:
    """One sample per time bucket: the least-delayed one in it.

    The same reasoning as the offset, applied along the time axis. Without this
    the fit's residuals are the whole delay tail, which at the measured shape is
    tens of milliseconds against a few milliseconds of real signal.
    """
    buckets: dict[int, TimeSyncSample] = {}
    width = int(SKEW_BUCKET_S * NS_PER_S)
    for sample in samples:
        key = sample.t_local_mid_ns // width
        held = buckets.get(key)
        if held is None or sample.rtt_ns < held.rtt_ns:
            buckets[key] = sample
    return [buckets[key] for key in sorted(buckets)]


def fit_skew(samples: Iterable[TimeSyncSample]) -> tuple[float, float, int] | None:
    """Least squares of offset against local time, in ppm, with its standard
    error and the number of points fitted. None when the baseline or the count
    cannot support a fit.

    Returns ppm rather than a raw slope because ppm is the unit the hardware
    spec is written in and the unit the bound charges in.
    """
    ordered = min_filtered(sorted(samples, key=lambda s: s.t_local_mid_ns))
    if len(ordered) < MIN_SKEW_SAMPLES:
        return None
    baseline_ns = ordered[-1].t_local_mid_ns - ordered[0].t_local_mid_ns
    if baseline_ns < MIN_SKEW_BASELINE_S * NS_PER_S:
        return None

    # Centred on the mean so the normal equations stay well conditioned: raw
    # monotonic nanoseconds are ~1e15 and squaring them loses the variation
    # entirely in float64.
    xs = [float(s.t_local_mid_ns) for s in ordered]
    ys = [float(s.offset_ns) for s in ordered]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx

    residuals = [y - (y_mean + slope * (x - x_mean)) for x, y in zip(xs, ys)]
    dof = len(ordered) - 2
    if dof <= 0:
        return None
    residual_variance = sum(r * r for r in residuals) / dof
    stderr = math.sqrt(residual_variance / sxx)
    return slope * 1e6, stderr * 1e6, len(ordered)


class TimebaseEstimator:
    """Sliding windows of samples, and the estimate they support.

    Thread-safe because the sampler feeding it and the consumer converting are
    different threads in every real arrangement.
    """

    def __init__(self, *, mono_clock: MonoClock = now_mono_ns) -> None:
        self._mono = mono_clock
        self._lock = threading.Lock()
        self._samples: list[TimeSyncSample] = []
        self._ids = itertools.count(1)
        self._published: deque[TimebaseEstimate] = deque(maxlen=MAX_HISTORY)
        self._current: TimebaseEstimate | None = None
        self.samples_accepted = 0
        self.samples_refused = 0
        self.estimates_evicted = 0
        self.estimates_published = 0
        self.refused_by_reason: dict[str, int] = {}

    # -- feeding -----------------------------------------------------------

    def add(self, sample: TimeSyncSample) -> bool:
        """Take a sample, or refuse and count it. True if it was used.

        A negative round trip is impossible: it means a clock went backwards or
        a field was misattributed between the two halves of the exchange. Either
        way it must not reach the fit, where it would drag the minimum below any
        real floor and make the bound arbitrarily small -- the one failure that
        would look like an improvement.
        """
        # The guarantee that a sample's error cannot exceed half its round trip
        # needs BOTH orderings. `rtt_ns >= 0` is satisfiable with either one
        # violated: a responder reporting a longer service interval than really
        # elapsed shrinks rtt below up+down while leaving the offset error
        # intact, which shrinks the bound under a real error -- the one failure
        # that looks like an improvement.
        if sample.t3_remote_send_ns < sample.t2_remote_recv_ns:
            self._refuse("remote_send_before_remote_recv")
            return False
        if sample.t4_local_recv_ns < sample.t1_local_send_ns:
            self._refuse("local_recv_before_local_send")
            return False
        if sample.rtt_ns < 0:
            self._refuse("out_of_range")
            return False
        if sample.rtt_ns > MAX_SAMPLE_RTT_NS:
            self._refuse("rtt_above_admission_ceiling")
            return False
        with self._lock:
            self._samples.append(sample)
            self.samples_accepted += 1
            self._prune_locked()
            self._recompute_locked()
        return True

    def _refuse(self, reason: str) -> None:
        with self._lock:
            self.samples_refused += 1
            self.refused_by_reason[reason] = self.refused_by_reason.get(reason, 0) + 1

    def _prune_locked(self) -> None:
        horizon = self._mono() - int(SKEW_WINDOW_S * NS_PER_S)
        self._samples = [s for s in self._samples if s.t_local_mid_ns >= horizon]

    def _recompute_locked(self) -> None:
        offset_horizon = self._mono() - int(OFFSET_WINDOW_S * NS_PER_S)
        recent = [s for s in self._samples if s.t_local_mid_ns >= offset_horizon]
        if not recent:
            self._current = None
            return
        best = min(recent, key=lambda s: s.rtt_ns)
        skew = fit_skew(self._samples)
        estimate = TimebaseEstimate(
            estimate_id=next(self._ids),
            offset_ns=best.offset_ns,
            t_reference_ns=best.t_local_mid_ns,
            rtt_min_ns=best.rtt_ns,
            offset_samples=len(recent),
            skew_ppm=None if skew is None else skew[0],
            skew_stderr_ppm=None if skew is None else skew[1],
            skew_samples=0 if skew is None else skew[2],
        )
        self._current = estimate
        # A bounded deque evicts in O(1); popping from the front of a list was
        # O(n) on every publish once the bound was reached.
        if len(self._published) == MAX_HISTORY:
            self.estimates_evicted += 1
        self._published.append(estimate)
        self.estimates_published += 1

    # -- reading -----------------------------------------------------------

    @property
    def retained_samples(self) -> int:
        """How many samples survive pruning. Exposed because `offset_samples`
        is filtered again by the 30 s horizon, so it cannot show whether the
        skew window is being pruned at all."""
        with self._lock:
            return len(self._samples)

    @property
    def estimate(self) -> TimebaseEstimate | None:
        with self._lock:
            return self._current

    def _gate_locked(self) -> tuple[str | None, TimebaseEstimate | None]:
        """The gate and the estimate it passed, from one snapshot.

        Returned together because they have to be the same estimate. Reading the
        verdict and then the estimate took two lock acquisitions, and a sample
        arriving in between published a new one -- so a conversion could proceed
        on an estimate that never passed the gate, with a bound twice the
        ceiling. Measured: a 400 ms bound handed out against a 200 ms limit.

        Likewise `usable` and `why_not_usable` are two views of one answer. When
        they each recomputed the gate, crossing MAX_SAMPLE_AGE_S between the two
        calls was enough for a record to publish `usable: true` beside a reason
        it was not -- and that record is what an offline reader attributes the
        run from.
        """
        current = self._current
        now = self._mono()
        if current is None:
            return "no samples", None
        if current.offset_samples < MIN_OFFSET_SAMPLES:
            return f"only {current.offset_samples} samples in the offset window", current
        newest = max((s.t_local_mid_ns for s in self._samples), default=None)
        if newest is None or now - newest > int(MAX_SAMPLE_AGE_S * NS_PER_S):
            age_s = "never" if newest is None else f"{(now - newest) / NS_PER_S:.1f}s"
            return f"newest sample is {age_s} old", current
        if current.rtt_min_ns > MAX_ACCEPTABLE_RTT_NS:
            return (
                f"min rtt {current.rtt_min_ns / 1e6:.1f}ms exceeds the acceptable bound",
                current,
            )
        return None, current

    def why_not_usable(self) -> str | None:
        """The single gate, and the reason it failed. None means usable.

        One predicate with a nameable cause, rather than a bool a caller has to
        guess about: "not ready" and "the link went bad" want different
        responses from an operator.
        """
        with self._lock:
            return self._gate_locked()[0]

    @property
    def usable(self) -> bool:
        return self.why_not_usable() is None

    def _gated(self) -> tuple[str | None, TimebaseEstimate | None]:
        """One acquisition, so both conversions read one snapshot."""
        with self._lock:
            return self._gate_locked()

    def to_remote(self, t_local_mono_ns: int) -> ConvertedInstant:
        """Convert a local instant to the peer's clock, with its bound.

        Raises rather than guessing. Forward-only: the value returned is stamped
        with the estimate that produced it and never changes afterwards, so a
        live consumer never sees a number move under it while an offline reader
        can still re-derive it against a better later estimate.
        """
        reason, estimate = self._gated()
        if reason is not None:
            raise TimebaseNotReady(f"cannot convert: {reason}", reason)
        assert estimate is not None  # the gate returning None proves it
        # Refuse to extend the fit into a region no sample supports. The gate
        # only asks that the newest sample be fresh; without this, an instant
        # half an hour from the reference still got an answer, with a drift term
        # extrapolated that whole way on evidence that spans five minutes.
        reach_ns = abs(t_local_mono_ns - estimate.t_reference_ns)
        if reach_ns > int(MAX_EXTRAPOLATION_S * NS_PER_S):
            raise TimebaseNotReady(
                f"instant is {reach_ns / NS_PER_S:.1f}s from the reference, beyond the "
                f"{MAX_EXTRAPOLATION_S:.0f}s the samples support",
                "beyond extrapolation limit",
            )
        drift_ns = 0.0
        if estimate.skew_ppm is not None:
            drift_ns = (t_local_mono_ns - estimate.t_reference_ns) * estimate.skew_ppm / 1e6
        return ConvertedInstant(
            t_remote_mono_ns=t_local_mono_ns + estimate.offset_ns + int(round(drift_ns)),
            bound_ns=estimate.bound_ns_at(t_local_mono_ns),
            estimate_id=estimate.estimate_id,
        )

    def to_local(self, t_remote_mono_ns: int) -> ConvertedInstant:
        """Convert a PEER instant to this device's clock, with its bound.

        The direction a receiver needs, and the one the estimator did not have.
        `to_remote` serves a device converting its own stamps for the peer's
        benefit; a device converting stamps that arrive needs the inverse, and
        whichever side runs the estimator is the side that can do it.

        Exact inverse rather than a first-order approximation. From
        `t_remote = t_local + offset + s(t_local - t_ref)` with `s` the skew as a
        fraction:

            t_local = (t_remote - offset + s * t_ref) / (1 + s)

        Subtracting the offset and then applying the skew correction forwards
        would leave a second-order error in `s`. It is small -- at 50 ppm over
        300 s it is well under a microsecond -- but the closed form costs nothing
        and does not need that argument made for it.
        """
        reason, estimate = self._gated()
        if reason is not None:
            raise TimebaseNotReady(f"cannot convert: {reason}", reason)
        assert estimate is not None
        skew = 0.0 if estimate.skew_ppm is None else estimate.skew_ppm / 1e6
        numerator = t_remote_mono_ns - estimate.offset_ns + skew * estimate.t_reference_ns
        t_local = int(round(numerator / (1.0 + skew)))

        reach_ns = abs(t_local - estimate.t_reference_ns)
        if reach_ns > int(MAX_EXTRAPOLATION_S * NS_PER_S):
            raise TimebaseNotReady(
                f"instant is {reach_ns / NS_PER_S:.1f}s from the reference, beyond the "
                f"{MAX_EXTRAPOLATION_S:.0f}s the samples support",
                "beyond extrapolation limit",
            )
        return ConvertedInstant(
            t_remote_mono_ns=t_local,
            bound_ns=estimate.bound_ns_at(t_local),
            estimate_id=estimate.estimate_id,
        )

    def history(self) -> list[TimebaseEstimate]:
        with self._lock:
            return list(self._published)

    def to_record(self) -> dict[str, Any]:
        """One lock, one snapshot. This is the artifact a run is attributed
        from, and it used to be able to say `usable: true` beside the reason it
        was not."""
        with self._lock:
            reason, current = self._gate_locked()
            return {
                "samples_accepted": self.samples_accepted,
                "samples_refused": self.samples_refused,
                "refused_by_reason": dict(sorted(self.refused_by_reason.items())),
                # Retained, not published: the deque saturates at MAX_HISTORY, so
                # the length stopped being the lifetime count when the bound went
                # in. An operator reading a five-hour run saw 4096 and concluded
                # 4096 were published when it was nearer eighteen thousand.
                "estimates_retained": len(self._published),
                "estimates_published": self.estimates_published,
                "estimates_evicted": self.estimates_evicted,
                "retained_samples": len(self._samples),
                "usable": reason is None,
                "why_not_usable": reason,
                # The bound as it stands, so an operator does not have to
                # recompute it: the drift term dominates it and the gate
                # deliberately does not limit it, so it is the number a consumer
                # checks against its own tolerance.
                "bound_now_ns": None if current is None else current.bound_ns_at(self._mono()),
                "current": None if current is None else current.to_record(),
            }


def answer_ping(
    received: tuple[TimeSyncMessage, int],
    *,
    mono_clock: MonoClock = now_mono_ns,
) -> TimeSyncMessage:
    """The responder's whole job: echo the exchange with receipt stamped.

    `t_wire_mono_ns` goes out as the placeholder; the writer replaces it with the
    departure time, which is what makes t3 a departure rather than an enqueue.

    Takes what `MessageRouter.recv_with_receipt` returns, rather than a message
    and a timestamp separately. The receipt stamp **must** be the one the
    session's reader took on arrival: the initiator computes the responder's
    service interval as `t3 - t2`, t3 is stamped by this session's writer clock,
    and a t2 from any other clock makes that difference arbitrary -- entering the
    round trip with a minus sign, so an interval reported longer than really
    elapsed shrinks the bound under an error that has not moved. Nothing on the
    wire can detect it. Prose was guarding the one hazard with no wire-side
    check, so the signature guards it instead: a bare `now_mono_ns()` will not fit.

    Both of the peer-receipt stamps come from that receipt for the same reason:
    taken at one instant they describe one instant, and the wall pair built from
    them is then a genuine independent estimate of the same offset rather than
    one displaced by this responder's handling delay.
    """
    ping, receipt = received
    if not ping.is_ping:
        raise MessageError(
            f"exchange {ping.exchange_id} is a pong; a responder must not receive one",
            REASON_WRONG_DIRECTION,
        )
    return TimeSyncMessage(
        t_capture_mono_ns=mono_clock(),
        exchange_id=ping.exchange_id,
        t_peer_recv_mono_ns=receipt.t_recv_mono_ns,
        # The reader's wall stamp, not a reading taken here. Both halves of the
        # peer's receipt must be one instant, or a cross-clock pair built from
        # them is displaced by however long this responder took to get to it --
        # and a handling delay that drifts across a run is indistinguishable from
        # clock skew, which is the one thing the pair exists to separate.
        t_peer_recv_wall_ns=receipt.t_recv_wall_ns,
        # Echoed, because the initiator cannot read its own: its writer stamped
        # the frame after the caller let go of it.
        t_peer_wire_mono_ns=ping.t_wire_mono_ns,
    )


@dataclass
class _Pending:
    exchange_id: int
    t_sent_mono_ns: int


class TimeSyncInitiator:
    """The phone's side: emit pings on a cadence, match pongs, feed the estimator.

    Takes anything with `send(message)` and `recv(channel, timeout)`, duck-typed
    the way the rest of this package treats its collaborators, so it can be
    driven by a MessageRouter or by a test double with no session at all.
    """

    def __init__(
        self,
        router: Any,
        *,
        mono_clock: MonoClock = now_mono_ns,
        wall_clock: WallClock = now_wall_ns,
        estimator: TimebaseEstimator | None = None,
    ) -> None:
        self._router = router
        self._mono = mono_clock
        self._wall = wall_clock
        self.estimator = estimator or TimebaseEstimator(mono_clock=mono_clock)
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._started_ns = self._mono()
        self.pings_sent = 0
        self.pongs_matched = 0
        self.pongs_unmatched = 0
        self.wrong_direction = 0
        self.unstamped_echoes = 0
        self.pings_refused = 0
        self.pongs_timed_out = 0
        self.pongs_refused = 0
        self.pongs_late = 0
        # Exchange ids we gave up on, with when, so a pong arriving afterwards is
        # counted as late rather than as a stray for an exchange we never sent.
        # Pruned on the same horizon as the pending table: `discard` only fires
        # when a timed-out pong eventually arrives, which for a genuinely lost
        # one never happens -- so without pruning the set is monotone for exactly
        # the entries that dominate it, and grows fastest when the link is worst.
        self._timed_out_ids: dict[int, int] = {}

    @property
    def period_s(self) -> float:
        """Fast while converging, cheap afterwards."""
        elapsed_s = (self._mono() - self._started_ns) / NS_PER_S
        hz = SYNC_FAST_HZ if elapsed_s < SYNC_FAST_DURATION_S else SYNC_STEADY_HZ
        return 1.0 / hz

    def next_exchange_id(self) -> int:
        """The id the next `send_ping` will use, without consuming it.

        A caller that stamps its own clock alongside an exchange needs to key
        that stamp by the same id, and pairing by arrival order instead is wrong:
        `pump` returns the oldest queued pong, which can belong to an earlier
        exchange. Measured, that mispairing manufactured a 51 ppm slope on a
        loopback link with no skew at all.
        """
        return self._next_id

    def send_ping(self) -> int:
        exchange_id = self._next_id
        self._next_id += 1
        # A pre-send stamp, kept only as a floor to sanity-check the echo
        # against. t1 proper is this side's writer stamp, which comes back on
        # the pong -- a pre-send stamp would carry the queueing delay the wire
        # stamp exists to remove.
        self._expire_pending()
        sent_at = self._mono()
        queued = self._router.send(
            TimeSyncMessage(t_capture_mono_ns=sent_at, exchange_id=exchange_id)
        )
        if not queued:
            # A closed session returns False. Counting it anyway made a dead link
            # indistinguishable from total pong loss in the record, and left a
            # pending entry that could never be matched.
            self.pings_refused += 1
            return exchange_id
        self._pending[exchange_id] = _Pending(exchange_id, sent_at)
        self.pings_sent += 1
        return exchange_id

    def _expire_pending(self) -> None:
        """Drop exchanges too old to be answered, and count them.

        Without this the table grows for the whole drive -- one entry per lost
        pong -- and an arbitrarily late pong is still matched, entering the skew
        window with a round trip of however long it was gone.
        """
        horizon = self._mono() - int(PENDING_TIMEOUT_S * NS_PER_S)
        # An id older than the late window cannot produce a usable sample anyway:
        # the admission ceiling would refuse a round trip that long.
        late_horizon = self._mono() - int(LATE_WINDOW_S * NS_PER_S)
        for key in [k for k, at in list(self._timed_out_ids.items()) if at < late_horizon]:
            self._timed_out_ids.pop(key, None)
        stale = [key for key, pending in list(self._pending.items())
                 if pending.t_sent_mono_ns < horizon]
        for key in stale:
            # pop, not del: on_pong pops from the same dict on another thread,
            # and a del that lost the race raised KeyError out of send_ping and
            # killed the cadence thread.
            if self._pending.pop(key, None) is None:
                # on_pong matched it between the scan and the pop -- the race the
                # pop tolerates. Recording it as timed out anyway would label a
                # later duplicate `pongs_late`, which is the wrong one of the two
                # diagnoses this counter was split to distinguish.
                continue
            self.pongs_timed_out += 1
            self._timed_out_ids[key] = self._mono()

    def on_pong(
        self,
        pong: TimeSyncMessage,
        t_recv_mono_ns: int,
        t_recv_wall_ns: int | None = None,
    ) -> TimeSyncSample | None:
        """Match a pong to its ping and feed the sample. None if unusable.

        A ping arriving here is a protocol error, not a pong to interpret: the
        phone initiates, so nobody should be pinging it, and treating one as the
        other produces an offset with the sign inverted -- a plausible number
        that is exactly wrong.
        """
        if pong.is_ping:
            self.wrong_direction += 1
            return None
        pending = self._pending.pop(pong.exchange_id, None)
        if pending is None:
            # An exchange we already timed out is a late answer, not a stray one.
            # Sharing `pongs_unmatched` between the two made one exchange
            # increment two counters and gave an operator one number for two
            # different diagnoses.
            if pong.exchange_id in self._timed_out_ids:
                self._timed_out_ids.pop(pong.exchange_id, None)
                self.pongs_late += 1
            else:
                self.pongs_unmatched += 1
            return None
        echoed = pong.t_peer_wire_mono_ns
        # A peer that never stamped the ping echoes the placeholder back. Left
        # alone that yields a round trip of the whole uptime, which the gate
        # would reject as a bad link rather than as an unimplemented peer -- so
        # it is named here instead of misdiagnosed there.
        if not echoed or echoed < pending.t_sent_mono_ns:
            self.unstamped_echoes += 1
            return None
        sample = TimeSyncSample(
            exchange_id=pong.exchange_id,
            t1_local_send_ns=echoed,
            t2_remote_recv_ns=pong.t_peer_recv_mono_ns or 0,
            t3_remote_send_ns=pong.t_wire_mono_ns,
            t4_local_recv_ns=t_recv_mono_ns,
            t2_remote_recv_wall_ns=pong.t_peer_recv_wall_ns,
            t4_local_recv_wall_ns=t_recv_wall_ns,
        )
        if not self.estimator.add(sample):
            # Counted here as well as in the estimator's own reasons: without
            # this a ping's fate existed only in the other record, so the
            # initiator showed one sent and every outcome zero, recoverable only
            # by joining two records and subtracting.
            self.pongs_refused += 1
            return None
        self.pongs_matched += 1
        return sample

    def pump(self, timeout: float = 0.0) -> TimeSyncSample | None:
        """Drain one pong from the router, if there is one.

        t4 is the transport's arrival stamp, not a reading taken here. Stamping
        it after recv returns folds the inbound queue wait and the decode into
        the round trip -- at the 1 Hz steady cadence, up to a whole period -- and
        that is the one term of the four measured locally, so it was the one
        still carrying the error the wire stamp removed from the other three.
        """
        received = self._router.recv_with_receipt(Channel.CONTROL, timeout=timeout)
        if received is None:
            return None
        pong, receipt = received
        return self.on_pong(pong, receipt.t_recv_mono_ns, receipt.t_recv_wall_ns)

    def to_record(self) -> dict[str, Any]:
        return {
            "pings_sent": self.pings_sent,
            "pongs_matched": self.pongs_matched,
            "pongs_unmatched": self.pongs_unmatched,
            "wrong_direction": self.wrong_direction,
            "unstamped_echoes": self.unstamped_echoes,
            "pings_refused": self.pings_refused,
            "pongs_timed_out": self.pongs_timed_out,
            "pongs_refused": self.pongs_refused,
            "pongs_late": self.pongs_late,
            "awaiting_reply": len(self._pending),
            "estimator": self.estimator.to_record(),
        }


@dataclass(frozen=True)
class OneWaySample:
    """One arrival, seen from the side that only ever answers.

    `TimeSyncSample` needs four stamps and the **initiator** is the only side
    that has them: it knows when it sent, when the peer received, when the peer
    replied and when the reply landed. The responder learns three of those and
    never the fourth, because nobody tells it when its pong arrived.

    The spec settles which side we are. "The phone initiates and the Jetson only
    ever answers" -- and the phone enforces it, refusing a ping. So the Jetson,
    which is the side that must convert incoming stamps to its own clock, is
    structurally the side that cannot form a round-trip sample. This is what it
    can form instead.
    """

    exchange_id: int
    #: The peer's monotonic clock when it put the ping on the wire.
    t1_remote_send_ns: int
    #: Our monotonic clock when the reader took it off the wire.
    t2_local_recv_ns: int

    @property
    def gap_ns(self) -> int:
        """The offset, underestimated by exactly the one-way delay.

        With `remote = local + offset`, a ping stamped `t1` leaves at local time
        `t1 - offset` and lands at `t2 = t1 - offset + d` for a one-way delay
        `d > 0`. So `offset = t1 - t2 + d`, and this quantity is always **below**
        the true offset, by `d`.

        The error is therefore one-sided, which is the property that makes a
        one-way estimate usable at all: it does not wander either way around the
        truth, it sits under it, and the largest gap seen is the one that came
        by the fastest path.
        """
        return self.t1_remote_send_ns - self.t2_local_recv_ns


class OneWayEstimator:
    """An offset from arrivals alone, for the side that cannot round-trip.

    Same consumer surface as `TimebaseEstimator` -- `to_local`, `estimate`,
    `to_record` -- so `PhoneClockAdapter` takes either without knowing which.
    That is deliberate and it is also the risk: a number from here must never be
    mistaken for one from a completed exchange, so `to_record` says `one_way`
    and every estimate it publishes carries a bound that means something
    different from the round-trip one. Read `bound_ns` below before using it.

    **What it cannot do.** No skew fit. Skew needs a lever arm across samples
    whose individual errors are bounded, and these are not: fitting a slope
    through points each displaced by an unknown delay would produce a
    confident-looking number whose error nobody can state. The round-trip
    estimator earns its skew term; this one does not, so it does not have one.
    """

    def __init__(self, *, mono_clock: MonoClock = now_mono_ns) -> None:
        self._mono = mono_clock
        self._lock = threading.Lock()
        self._samples: list[OneWaySample] = []
        self._ids = itertools.count(1)
        self._current: TimebaseEstimate | None = None
        self.samples_accepted = 0
        self.samples_refused = 0
        self.estimates_published = 0
        self.refused_by_reason: dict[str, int] = {}
        #: Read by anything recording provenance. See the class docstring.
        self.one_way = True

    def add(self, sample: OneWaySample) -> bool:
        """Take an arrival, or refuse and count it."""
        if sample.t2_local_recv_ns <= 0:
            self._refuse("local_recv_not_stamped")
            return False
        with self._lock:
            self._samples.append(sample)
            self.samples_accepted += 1
            self._prune_locked()
            self._recompute_locked()
        return True

    def _refuse(self, reason: str) -> None:
        with self._lock:
            self.samples_refused += 1
            self.refused_by_reason[reason] = self.refused_by_reason.get(reason, 0) + 1

    def _prune_locked(self) -> None:
        horizon = self._mono() - int(OFFSET_WINDOW_S * NS_PER_S)
        self._samples = [s for s in self._samples if s.t2_local_recv_ns >= horizon]

    def _recompute_locked(self) -> None:
        if not self._samples:
            self._current = None
            return
        # The LARGEST gap, for the same reason the round-trip estimator takes the
        # smallest round trip: `gap = offset - d`, so the biggest gap is the one
        # whose packet crossed fastest, and it is the closest to the truth that
        # was actually observed. A mean would average in every slow arrival and
        # sit further below the offset than the best single sample.
        best = max(self._samples, key=lambda s: s.gap_ns)
        worst = min(self._samples, key=lambda s: s.gap_ns)
        self._current = TimebaseEstimate(
            estimate_id=next(self._ids),
            offset_ns=best.gap_ns,
            t_reference_ns=best.t2_local_recv_ns,
            # NOT half a round trip, and not comparable to the round-trip
            # estimator's bound. This is the spread of one-way delays actually
            # seen, so it covers how much delivery VARIED and says nothing about
            # the delay floor -- an unobservable, one-sided component that makes
            # every converted stamp look newer than it is by the fastest delay
            # in the window. On this pair that floor is a few tens of
            # milliseconds, which is why this is fit for a 2 s freshness
            # threshold and unfit for attributing latency.
            rtt_min_ns=best.gap_ns - worst.gap_ns,
            offset_samples=len(self._samples),
        )
        self.estimates_published += 1

    def estimate(self) -> TimebaseEstimate | None:
        with self._lock:
            return self._current

    def to_local(self, t_remote_mono_ns: int) -> ConvertedInstant:
        """Convert a peer instant to this device's clock.

        No skew term, so this is the plain inverse rather than the closed form
        the round-trip estimator needs.
        """
        with self._lock:
            estimate = self._current
        if estimate is None:
            raise TimebaseNotReady("cannot convert: no one-way samples yet", "no estimate")
        t_local = t_remote_mono_ns - estimate.offset_ns
        reach_ns = abs(t_local - estimate.t_reference_ns)
        if reach_ns > int(MAX_EXTRAPOLATION_S * NS_PER_S):
            raise TimebaseNotReady(
                f"instant is {reach_ns / NS_PER_S:.1f}s from the reference, beyond the "
                f"{MAX_EXTRAPOLATION_S:.0f}s the samples support",
                "beyond extrapolation limit",
            )
        return ConvertedInstant(
            t_remote_mono_ns=t_local,
            bound_ns=estimate.rtt_min_ns,
            estimate_id=estimate.estimate_id,
        )

    def to_record(self) -> dict[str, Any]:
        estimate = self.estimate()
        return {
            # First, and named this, because it is the thing a reader must not
            # miss: a bound from here is a delay spread, not half a round trip.
            "one_way": True,
            "samples_accepted": self.samples_accepted,
            "samples_refused": self.samples_refused,
            "refused_by_reason": dict(self.refused_by_reason),
            "estimates_published": self.estimates_published,
            "offset_ns": None if estimate is None else estimate.offset_ns,
            "delay_spread_ns": None if estimate is None else estimate.rtt_min_ns,
            "offset_samples": 0 if estimate is None else estimate.offset_samples,
        }
