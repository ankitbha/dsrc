"""Nanosecond clocks for the wire.

Same two-clock discipline as sensors/time_sync.py, in nanoseconds because the
header fields are integers and a float second loses resolution at the scale a
wall-clock epoch reaches.

  monotonic  latency math, and meaningful only on the device that produced it.
             Never compare a phone monotonic value with a Jetson one.
  wall       UTC epoch, for correlating logs across the two devices. Can step.

Note that a wall-clock nanosecond value is around 1.75e18, which is past 2**53.
Any implementation that routes it through a double loses the low digits; the
golden vectors include a case that catches exactly that.
"""

from __future__ import annotations

import time
from typing import Callable

MonoClock = Callable[[], int]
WallClock = Callable[[], int]


def now_mono_ns() -> int:
    return time.monotonic_ns()


def now_wall_ns() -> int:
    return time.time_ns()
