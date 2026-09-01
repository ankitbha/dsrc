"""Time bases for the deployment.

Two clocks are used everywhere:
  - t_mono: time.monotonic(), for all latency math and sensor freshness.
    Never compare monotonic values across processes or reboots.
  - t_wall: time.time() (UTC epoch), for log records and cross-device
    correlation (e.g. matching GPS UTC timestamps offline).

GPS sentences carry UTC; ``GpsUtcOffsetTracker`` keeps a running estimate
of (wall clock - GPS UTC) so logs can be corrected offline if the Jetson
RTC drifts while off-network in the car.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def now_mono() -> float:
    return time.monotonic()


def now_wall() -> float:
    return time.time()


def capture_stamp_ns(t_capture_mono: float) -> int:
    """The one conversion of a monotonic capture instant into the integer
    nanoseconds a record or a wire message carries.

    Both `pipeline.Tick.to_record()` and `AdvisoryMessage.t_capture_mono_ns`
    (built in `policy.sensing_loop`) need to name the same tick with the same
    key, and `eval_run.py` joins the two logs on exact equality of that key.
    Rounding rather than truncating matters only at the boundary where the
    float's fractional-nanosecond part sits past .5 -- rare, but truncating in
    one place and rounding in the other disagreed on it.
    """
    return int(round(t_capture_mono * 1e9))


@dataclass(frozen=True)
class TimebaseStamp:
    """What a cross-device capture stamp is worth, carried with the reading.

    Attached to the reading rather than looked up beside it. The alternative --
    a map from id to stamp, or a "last stamp" attribute -- is a pairing problem,
    and pairing a stamp with the wrong record is a mistake this project has
    already made once: in the task 15 probe a wall stamp taken before one
    exchange was paired with whatever reply arrived next, and it manufactured a
    51 ppm clock skew out of a link that had none.

    `t_capture_mono` is on THIS device's clock, either converted from the peer's
    or proxied by arrival. `t_arrival_mono` is always exact, so the segment from
    it onwards never depends on the timebase.
    """

    t_capture_mono: float
    t_arrival_mono: float
    bound_s: float | None
    estimate_id: int | None
    proxy: bool
    #: Which estimator produced this, or "proxy" when neither could. A converted
    #: number must never look as measured as a same-device one, and a
    #: round-trip-converted number must not be pooled with a one-way-converted
    #: one -- the two have different error semantics, so this is what lets a
    #: consumer keep their samples in separate series.
    source: str = "proxy"
    #: Why the estimator refused, when `proxy` is True. None when it did not
    #: proxy. Kept apart from the adapter's aggregate `proxy_reasons` counter so
    #: a single stamp can say why for itself, not only the run as a whole.
    proxy_reason: str | None = None

    @property
    def link_s(self) -> float | None:
        """Capture to arrival, or None when it was not measured.

        None under the proxy, where capture IS arrival by construction, so the
        difference is zero for a reason that has nothing to do with the link. It
        used to return that zero, and the proxy is used precisely when the link
        is worst -- so those zeros pulled the reported link segment down exactly
        when it mattered most.

        Bounded, and can come out negative: the bound permits a converted capture
        stamp slightly later than the arrival it preceded. Reported as measured
        rather than clamped, because clamping would hide exactly the case where
        the bound is being tested.
        """
        if self.proxy:
            return None
        return self.t_arrival_mono - self.t_capture_mono

    def to_record(self) -> dict:
        return {
            "converted": not self.proxy,
            "proxy": self.proxy,
            "source": self.source,
            "proxy_reason": self.proxy_reason,
            "bound_ms": None if self.bound_s is None else round(self.bound_s * 1000.0, 3),
            "estimate_id": self.estimate_id,
            "link_ms": None if self.link_s is None else round(self.link_s * 1000.0, 3),
        }


# The three bases a stage's duration can rest on. A fourth, "instant", covers
# the one stage (capture) that names a reference point rather than a span.
STAGE_BASIS_MEASURED = "measured"
STAGE_BASIS_CONVERTED = "converted"
STAGE_BASIS_ABSENT = "absent"
STAGE_BASIS_INSTANT = "instant"


@dataclass(frozen=True)
class StageTiming:
    """One named segment of a tick's time, and how much the number is worth.

    The recurring defect this project keeps naming is a record that cannot
    distinguish failure from success: a stage that was proxied, never
    instrumented, or measured across an unsynchronised clock has to be
    distinguishable from one read straight off a single device's own clock, in
    the record itself rather than in a reader's memory of how the pipeline
    works. So every stage carries `basis` and `clock` beside its `ms`, and a
    stage that could not be measured carries `ms: None` and a `reason`, never a
    zero -- the same discipline `RollingStats.summary` and `TimebaseStamp.link_s`
    already apply to their own numbers.
    """

    ms: float | None
    basis: str
    clock: str
    #: Present only when `basis == "converted"`.
    bound_ms: float | None = None
    estimate_id: int | None = None
    source: str | None = None
    #: Present only when `basis == "absent"`.
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ms": None if self.ms is None else round(self.ms, 3),
            "basis": self.basis,
            "clock": self.clock,
        }
        if self.bound_ms is not None:
            record["bound_ms"] = round(self.bound_ms, 3)
        if self.estimate_id is not None:
            record["estimate_id"] = self.estimate_id
        if self.source is not None:
            record["source"] = self.source
        if self.reason is not None:
            record["reason"] = self.reason
        return record

    @classmethod
    def measured(cls, ms: float, *, clock: str) -> "StageTiming":
        return cls(ms=ms, basis=STAGE_BASIS_MEASURED, clock=clock)

    @classmethod
    def converted(
        cls, ms: float, *, bound_ms: float, estimate_id: int, source: str
    ) -> "StageTiming":
        return cls(
            ms=ms, basis=STAGE_BASIS_CONVERTED, clock="cross",
            bound_ms=bound_ms, estimate_id=estimate_id, source=source,
        )

    @classmethod
    def absent(cls, *, clock: str, reason: str) -> "StageTiming":
        return cls(ms=None, basis=STAGE_BASIS_ABSENT, clock=clock, reason=reason)

    @classmethod
    def instant(cls, *, clock: str) -> "StageTiming":
        """The reference point a tick's other stages are measured from.

        Zero rather than absent: `capture` is not unmeasured, it is by
        definition the origin every other phone-clock stage is a distance from.
        The shutter-to-callback bias that instant does not capture is a named,
        open limitation (see the task plan), not something this record hides.
        """
        return cls(ms=0.0, basis=STAGE_BASIS_INSTANT, clock=clock)


@dataclass
class Stamp:
    t_mono: float
    t_wall: float

    @classmethod
    def now(cls) -> "Stamp":
        return cls(t_mono=now_mono(), t_wall=now_wall())


class GpsUtcOffsetTracker:
    """EMA of (system wall clock - GPS UTC) in seconds."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._offset_s: float | None = None

    def update(self, gps_utc_epoch_s: float, wall_s: float) -> None:
        sample = wall_s - gps_utc_epoch_s
        if self._offset_s is None:
            self._offset_s = sample
        else:
            self._offset_s += self._alpha * (sample - self._offset_s)

    @property
    def offset_s(self) -> float | None:
        return self._offset_s
