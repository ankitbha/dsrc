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

**What a shadow log is and is not.** Three limits, and the first is the one that
bites hardest because it is present from the first tick rather than accumulating.

*The traffic feed is structurally absent from a pure shadow drive.* The phone makes
no HERE call until the Jetson tells it what to ask, and `ConfigApplier` returns on
the shadow branch **before** it reaches `setHereQuery`. So a drive that has only ever
been in shadow has no query, makes no call, and sends no `here` frame; `HereFeed`
stays empty, `feed_congestion` is None on every tick, and `Trigger.DISAGREEMENT` --
one of the controller's three raise rules -- cannot fire at all. A shadow log
therefore cannot credit or debit any candidate policy for that rule. This is not a
degraded input; it is a missing one.

*Reference rates are only reference until the first live segment.* Once a live
command has been applied the phone is running whatever it was told, and it keeps
running it across a live -> shadow flip: the last live rates and the last live query
both survive, because `setQuery(null)` is a deliberate no-op there. So the
discriminator is whether this drive has **ever** been live, not whether it has left
live -- a drive that is still live has not held the reference rates either. Both
`reference_rates_hold` and `structurally_absent` therefore key on that one fact, and
on a mixed drive the flip log gives the boundary: the feed cannot exist before the
first flip to live, and cannot be assumed to exist immediately after it, since the
phone still has to place the call and the response still has to arrive.

*And the trajectory diverges.* In shadow nothing is applied, so the controller
decides from whatever the phone is currently running. In live, dropping the camera to
1 Hz changes what the next tick sees: fewer frames, a different density bin, perhaps
a different margin. So a shadow log predicts the **decision function** exactly, tick
for tick against the same inputs, and does **not** predict the **trajectory** -- a
live drive feeds its own reduced observations back in and a shadow drive never does.

Making the feed available in shadow would mean letting a shadow command carry a query
into effect, which changes what `shadow` means on the wire and is a protocol
decision, not one this module may take on its own.
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

#: Inputs a drive that has only ever been in shadow cannot produce, and the rules
#: they make unreachable. The phone makes no HERE call until the Jetson sends a
#: query, and a shadow command never reaches `setHereQuery`, so nothing about the
#: traffic feed exists in such a log -- a candidate policy cannot be scored on a
#: rule that never had the chance to fire.
#:
#: This lists what is absent *when the condition holds*. `to_record` emits it only
#: for a drive that has never been live, because on a live drive every one of these
#: is present and naming them absent is the pure-shadow reading of a live log.
ABSENT_IN_PURE_SHADOW = (
    "feed_congestion",
    "source_disagreement",
)


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
        # Entering live is what contaminates a log, and a flip records the mode it
        # came FROM -- so no predicate over `_flips` alone sees a drive that started
        # live, and one over `f.was == LIVE` sees only drives that have LEFT live.
        # Held as a fact instead: set here, set on the way in, never cleared.
        self._ever_live = mode == LIVE

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
            if mode == LIVE:
                self._ever_live = True
            return True

    def to_record(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode,
                # Every flip, not just where it ended. A log that reports only the
                # final mode cannot say which decisions were gated.
                "flips": [{"at_mono": f.at_mono, "was": f.was, "now": f.now} for f in self._flips],
                "flip_count": len(self._flips),
                # Named rather than left to a reader to discover from an empty
                # column. A shadow log's gaps are structural, not incidental.
                "shadow_predicts": "the decision function, not the trajectory",
                # What is missing from THIS log, not what would be missing from a
                # pure shadow one. A live drive has the feed on every tick, so
                # emitting the list there asserted the pure-shadow reading of a live
                # log -- and did it alongside `reference_rates_hold: True`, so both
                # fields agreed on the wrong answer.
                "structurally_absent":
                    [] if self._ever_live else list(ABSENT_IN_PURE_SHADOW),
                "reference_rates_hold": not self._ever_live,
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
