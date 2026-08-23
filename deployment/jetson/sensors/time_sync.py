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


def now_mono() -> float:
    return time.monotonic()


def now_wall() -> float:
    return time.time()


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
            "bound_ms": None if self.bound_s is None else round(self.bound_s * 1000.0, 3),
            "estimate_id": self.estimate_id,
            "link_ms": None if self.link_s is None else round(self.link_s * 1000.0, 3),
        }


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
