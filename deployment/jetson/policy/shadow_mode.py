"""Whether a rate decision is gated for real, or only recorded.

The phone's half already exists and is right: `ConfigApplier` treats a shadow
command as changing "nothing at all -- not the rates, not the query, and not
`current`, which is what the phone is actually running", and counts what it
shadowed. This is the Jetson choosing which it is sending.

**The mode never reaches the decision.** `SensingController.decide` does not take a
mode and must not: task 43 checks that logged shadow decisions match what live
gating produces on the same input, and if the decision could see the mode that check
would compare a function against itself and pass whatever the code did. So the mode
is applied strictly on the way out, selecting one boolean on the command.

**What a shadow log is and is not.** In shadow the phone keeps running the reference
rates, because nothing is applied, so a shadow drive samples at full rate throughout
-- which is what makes one drive scorable against several candidate policies. But
the controller is then deciding from full-rate inputs. In live, dropping the camera
to 1 Hz changes what the next tick sees: fewer frames, a different density bin,
perhaps a different policy margin. So a shadow log predicts the **decision function**
exactly, tick for tick against the same inputs, and does **not** predict the
**trajectory** -- a live drive feeds its own reduced observations back in and a
shadow drive never does. The two diverge after the first change, and no amount of
shadow logging closes that.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

#: Recorded, not applied. The default, because a process that comes up gating for
#: real because nobody said otherwise is the wrong failure to have.
SHADOW = "shadow"

#: Applied. Chosen deliberately, never fallen into.
LIVE = "live"

MODES = (SHADOW, LIVE)


@dataclass(frozen=True)
class Flip:
    """One mode change, and when it happened.

    A drive that changed mode and cannot say at which tick is unattributable
    afterwards, and every decision before and after is scored together.
    """

    at_mono: float
    was: str
    now: str


class ModeHolder:
    """The current mode, flippable while the loop runs.

    Read on the tick loop and written by whatever flips it, so the read is one
    atomic operation: a flip cannot land between reading the mode for the rates and
    reading it for the flag, because there is only one read.
    """

    def __init__(self, mode: str = SHADOW, *, clock: Any = None) -> None:
        import time as _time

        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        self._now = clock or _time.monotonic
        self._lock = threading.Lock()
        self._mode = mode
        self._flips: list[Flip] = []

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def is_live(self) -> bool:
        return self.mode == LIVE

    def flip_to(self, mode: str) -> bool:
        """Change mode. False when it was already there, so a no-op is not a flip."""
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        with self._lock:
            if mode == self._mode:
                return False
            self._flips.append(Flip(at_mono=self._now(), was=self._mode, now=mode))
            self._mode = mode
            return True

    def to_record(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode,
                # Every flip, not just where it ended. A log that reports only the
                # final mode cannot say which decisions were gated.
                "flips": [{"at_mono": f.at_mono, "was": f.was, "now": f.now} for f in self._flips],
                "flip_count": len(self._flips),
                "shadow_predicts": "the decision function, not the trajectory",
            }


def command_for(decision: Any, mode: str, *, t_capture_mono_ns: int) -> Any:
    """Turn a decision into the command that carries it.

    The one place the mode is read. Everything else on the command comes from the
    decision unchanged -- same rates, same trigger, same query in either mode -- so
    a shadow log and a live drive differ by exactly one boolean, which is the
    property task 43's comparison rests on.
    """
    from transport.messages import RateCommand

    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    return RateCommand(
        t_capture_mono_ns=t_capture_mono_ns,
        rates=dict(decision.rates),
        trigger=decision.trigger,
        shadow=(mode != LIVE),
        here=decision.here_query,
    )
