"""Stage 0, gate V1: relate the phone's wall clock to GPS UTC."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np

from here_error import params


@dataclass(frozen=True)
class ClockOffset:
    """Phone wall clock minus GPS UTC, in seconds, over every valid fix of one phone log."""

    median_s: float
    p5_s: float
    p95_s: float
    n: int

    @property
    def spread_s(self) -> float:
        return self.p95_s - self.p5_s

    @property
    def passes_gate(self) -> bool:
        return self.spread_s <= params.V1_MAX_SPREAD_S


def estimate_offset(wall_s, utc_s) -> ClockOffset:
    wall = np.asarray(wall_s, dtype=float)
    utc = np.asarray(utc_s, dtype=float)
    if wall.size == 0 or wall.shape != utc.shape:
        raise ValueError("clock offset needs equal, non-empty wall and UTC series")
    d = wall - utc
    p5, med, p95 = np.percentile(d, [5, 50, 95])
    return ClockOffset(median_s=float(med), p5_s=float(p5), p95_s=float(p95), n=int(d.size))


def wall_to_utc(wall_s: float, offset: ClockOffset) -> float:
    """Convert a phone wall-clock time to the GPS UTC axis."""
    return wall_s - offset.median_s


def is_after_dusk(utc_s: float) -> bool:
    """Whether a time is at or after the local daylight/dark split used everywhere."""
    t = dt.datetime.fromtimestamp(utc_s, ZoneInfo(params.LOCAL_TZ))
    return (t.hour, t.minute) >= params.DUSK_SPLIT_LOCAL
