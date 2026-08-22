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
MAX_ACCEPTABLE_RTT_NS = 200_000_000

# Skew is withheld until it can be resolved. A fit over too short a baseline
# returns a value near zero whose uncertainty is many times its own size, and
# publishing that invites a consumer to apply it as a measurement.
# The requirement is leverage in time, and the bucket width times the minimum
# count already guarantees it: 20 buckets 10 s apart span at least 190 s. So the
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

# What we charge for skew we have not measured. An unmeasured skew is not a zero
# skew: assuming zero would understate the bound exactly when it is least
# trustworthy, so the bound grows at the top of the ordinary crystal range.
ASSUMED_SKEW_PPM = 50.0

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

        The fitted standard error once skew is known, and the assumed crystal
        range while it is not -- never zero, because "not measured" and "zero"
        are different claims and only one of them is true here.
        """
        if self.skew_ppm is None or self.skew_stderr_ppm is None:
            return ASSUMED_SKEW_PPM
        return max(self.skew_stderr_ppm, 0.0)

    def bound_ns_at(self, t_local_ns: int) -> int:
        """Half the least asymmetry we cannot rule out, plus what drift accrues
        over the extrapolation from the reference instant."""
        drift = abs(t_local_ns - self.t_reference_ns) * self.skew_uncertainty_ppm / 1e6
        return self.rtt_min_ns // 2 + int(math.ceil(drift))

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
        self._published: list[TimebaseEstimate] = []
        self._current: TimebaseEstimate | None = None
        self.samples_accepted = 0
        self.samples_refused = 0
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
        if sample.rtt_ns < 0:
            self._refuse("out_of_range")
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
        self._published.append(estimate)

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

    def why_not_usable(self) -> str | None:
        """The single gate, and the reason it failed. None means usable.

        One predicate with a nameable cause, rather than a bool a caller has to
        guess about: "not ready" and "the link went bad" want different
        responses from an operator.
        """
        with self._lock:
            current = self._current
            newest = max((s.t_local_mid_ns for s in self._samples), default=None)
        if current is None:
            return "no samples"
        if current.offset_samples < MIN_OFFSET_SAMPLES:
            return f"only {current.offset_samples} samples in the offset window"
        if newest is None or self._mono() - newest > int(MAX_SAMPLE_AGE_S * NS_PER_S):
            age_s = "never" if newest is None else f"{(self._mono() - newest) / NS_PER_S:.1f}s"
            return f"newest sample is {age_s} old"
        if current.rtt_min_ns > MAX_ACCEPTABLE_RTT_NS:
            return f"min rtt {current.rtt_min_ns / 1e6:.1f}ms exceeds the acceptable bound"
        return None

    @property
    def usable(self) -> bool:
        return self.why_not_usable() is None

    def to_remote(self, t_local_mono_ns: int) -> ConvertedInstant:
        """Convert a local instant to the peer's clock, with its bound.

        Raises rather than guessing. Forward-only: the value returned is stamped
        with the estimate that produced it and never changes afterwards, so a
        live consumer never sees a number move under it while an offline reader
        can still re-derive it against a better later estimate.
        """
        reason = self.why_not_usable()
        if reason is not None:
            raise TimebaseNotReady(f"cannot convert: {reason}", reason)
        estimate = self.estimate
        assert estimate is not None  # why_not_usable() just proved it
        drift_ns = 0.0
        if estimate.skew_ppm is not None:
            drift_ns = (t_local_mono_ns - estimate.t_reference_ns) * estimate.skew_ppm / 1e6
        return ConvertedInstant(
            t_remote_mono_ns=t_local_mono_ns + estimate.offset_ns + int(round(drift_ns)),
            bound_ns=estimate.bound_ns_at(t_local_mono_ns),
            estimate_id=estimate.estimate_id,
        )

    def history(self) -> list[TimebaseEstimate]:
        with self._lock:
            return list(self._published)

    def to_record(self) -> dict[str, Any]:
        current = self.estimate
        with self._lock:
            refused = dict(sorted(self.refused_by_reason.items()))
            accepted, refused_total = self.samples_accepted, self.samples_refused
            published = len(self._published)
        return {
            "samples_accepted": accepted,
            "samples_refused": refused_total,
            "refused_by_reason": refused,
            "estimates_published": published,
            "retained_samples": self.retained_samples,
            "usable": self.usable,
            "why_not_usable": self.why_not_usable(),
            "current": None if current is None else current.to_record(),
        }


def answer_ping(
    ping: TimeSyncMessage,
    t_recv_mono_ns: int,
    t_recv_wall_ns: int,
    *,
    mono_clock: MonoClock = now_mono_ns,
) -> TimeSyncMessage:
    """The responder's whole job: echo the exchange with receipt stamped.

    `t_wire_mono_ns` goes out as the placeholder; the writer replaces it with the
    departure time, which is what makes t3 a departure rather than an enqueue.
    """
    if not ping.is_ping:
        raise MessageError(
            f"exchange {ping.exchange_id} is a pong; a responder must not receive one",
            REASON_WRONG_DIRECTION,
        )
    return TimeSyncMessage(
        t_capture_mono_ns=mono_clock(),
        exchange_id=ping.exchange_id,
        t_peer_recv_mono_ns=t_recv_mono_ns,
        t_peer_recv_wall_ns=t_recv_wall_ns,
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
        self._ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}
        self._started_ns = self._mono()
        self.pings_sent = 0
        self.pongs_matched = 0
        self.pongs_unmatched = 0
        self.wrong_direction = 0
        self.unstamped_echoes = 0

    @property
    def period_s(self) -> float:
        """Fast while converging, cheap afterwards."""
        elapsed_s = (self._mono() - self._started_ns) / NS_PER_S
        hz = SYNC_FAST_HZ if elapsed_s < SYNC_FAST_DURATION_S else SYNC_STEADY_HZ
        return 1.0 / hz

    def send_ping(self) -> int:
        exchange_id = next(self._ids)
        # A pre-send stamp, kept only as a floor to sanity-check the echo
        # against. t1 proper is this side's writer stamp, which comes back on
        # the pong -- a pre-send stamp would carry the queueing delay the wire
        # stamp exists to remove.
        self._pending[exchange_id] = _Pending(exchange_id, self._mono())
        self._router.send(TimeSyncMessage(t_capture_mono_ns=self._mono(), exchange_id=exchange_id))
        self.pings_sent += 1
        return exchange_id

    def on_pong(self, pong: TimeSyncMessage, t_recv_mono_ns: int) -> TimeSyncSample | None:
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
        )
        if not self.estimator.add(sample):
            return None
        self.pongs_matched += 1
        return sample

    def pump(self, timeout: float = 0.0) -> TimeSyncSample | None:
        """Drain one pong from the router, if there is one."""
        received = self._router.recv(Channel.CONTROL, timeout=timeout)
        if received is None:
            return None
        return self.on_pong(received, self._mono())

    def to_record(self) -> dict[str, Any]:
        return {
            "pings_sent": self.pings_sent,
            "pongs_matched": self.pongs_matched,
            "pongs_unmatched": self.pongs_unmatched,
            "wrong_direction": self.wrong_direction,
            "unstamped_echoes": self.unstamped_echoes,
            "awaiting_reply": len(self._pending),
            "estimator": self.estimator.to_record(),
        }
