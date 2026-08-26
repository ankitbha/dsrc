"""Four rates, decided here and commanded to the phone.

Three tasks of the phone reporting; this is where the Jetson decides. The phone's
own rule is that it reports and never lowers a rate on its own -- a handset that
quietly halved its camera rate when warm would leave this side comparing a model
against inputs it never asked for and cannot see it did not get. So every rate
change in the system originates here.

**Rates cannot express "off".** The wire constrains them to `(0, 1000]` Hz, so zero
is refused as `out_of_range`. A modality can be idled low and never disabled, and
anything that wants "off" needs a protocol change rather than a rate of zero. Named
because the obvious implementation of "stop the camera" is silently invalid.

**Changes are expensive, so they need evidence and patience.** Every change costs the
phone a sensor re-registration or a camera rebind. A controller that re-decided every
tick would thrash the hardware and produce a drive whose rates are noise rather than
decisions, so a rate moves only when its evidence crosses a band AND has held for a
dwell. `trigger` says which rule fired, from a closed set, because task 34 has to
attribute rate changes afterwards and free text makes that a text-mining problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The wire's own bounds. Zero is not a rate; see the module docstring.
MIN_RATE_HZ = 0.05
MAX_RATE_HZ = 1000.0

#: What each modality costs and therefore how it idles. IMU and GPS are the free
#: always-on tier -- two ~200-byte frames a second against a camera stream, about a
#: tenth of a percent -- so they never idle down: they are the detector that says
#: when to spend the expensive ones.
IDLE_RATES = {"camera_hz": 1.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.05}
ACTIVE_RATES = {"camera_hz": 5.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.2}

#: Thermal backoff is the only input that argues for less, so it is applied last and
#: wins. Multipliers rather than absolutes, so a backoff composes with whatever the
#: other rules decided rather than overwriting their reasoning.
THERMAL_SCALE = {
    "nominal": 1.0,
    "light": 1.0,
    "moderate": 0.6,
    "severe": 0.3,
    "critical": 0.15,
    "emergency": 0.15,
    "shutdown": 0.15,
    # An unknown status is not a nominal one. A drive that never heard from the
    # phone must not run hot rates on the assumption that silence means cool.
    "unknown": 0.6,
}

#: Skin temperature that triggers backoff before the status moves. Measured on the
#: handset: it warmed 5.4 C under camera load while `thermal_status` stayed
#: `nominal` for all 81 telemetry frames, so the status alone does not move until
#: the phone is already in trouble.
SKIN_WARM_C = 40.0
SKIN_HOT_C = 45.0

#: Policy margin below which the advice is one input change from flipping.
NARROW_MARGIN = 0.15

#: Longitudinal acceleration that counts as "something is happening", from the free
#: tier. Gentle enough to catch a lift off the throttle, firm enough not to fire on
#: road noise.
EVENT_ACCEL_MPS2 = 1.5

#: How long a rule's evidence must hold before a rate moves, and how long a raised
#: rate stays up after the evidence goes away. Asymmetric on purpose: going up late
#: misses the event that justified it, coming down early re-pays the cost of going
#: up again.
RAISE_DWELL_S = 0.5
HOLD_S = 5.0


class Trigger:
    """Why the rates are what they are. Closed, because task 34 attributes on it."""

    IDLE = "idle"
    EVENT = "event_from_free_tier"
    NARROW_MARGIN = "advisory_margin_narrow"
    DISAGREEMENT = "source_disagreement"
    THERMAL = "thermal_backoff"
    HOLD = "holding_after_event"

    ALL = frozenset({IDLE, EVENT, NARROW_MARGIN, DISAGREEMENT, THERMAL, HOLD})


@dataclass(frozen=True)
class Inputs:
    """Everything the controller is allowed to look at, in one place."""

    #: The free always-on tier.
    ego_acceleration: float | None = None
    ego_speed: float | None = None
    #: `top1 - top2` of the policy's own softmax, per active head. None before the
    #: first inference.
    policy_margin: float | None = None
    #: The feed's derived congestion, and what the camera sees locally.
    feed_congestion: float | None = None
    camera_density_bin: int | None = None
    #: The phone's last self-report. None means never heard from.
    thermal_status: str | None = None
    skin_temp_c: float | None = None


@dataclass(frozen=True)
class Decision:
    """Four rates, why, and the settings that ride with them."""

    rates: dict[str, float]
    trigger: str
    reasons: list[str] = field(default_factory=list)
    thermal_scale: float = 1.0

    def to_record(self) -> dict[str, Any]:
        return {
            "rates": dict(self.rates),
            "trigger": self.trigger,
            "reasons": list(self.reasons),
            "thermal_scale": self.thermal_scale,
        }


def _clamp(hz: float) -> float:
    """Into the wire's range. A rate that would be refused is a bug here, not there."""
    return max(MIN_RATE_HZ, min(MAX_RATE_HZ, hz))


def disagreement(feed_congestion: float | None, camera_density_bin: int | None) -> bool:
    """Whether the two views contradict each other about where we are.

    Not a disagreement inside the observation: task 28 established the feed reaches
    no field of it. This is the feed's congestion against what the camera sees, and
    a contradiction means one of them is wrong about our position -- worth spending
    samples to resolve rather than picking a winner here.
    """
    if feed_congestion is None or camera_density_bin is None:
        return False
    jammed_ahead = feed_congestion >= 0.5
    empty_here = camera_density_bin <= 0
    return jammed_ahead and empty_here


class SensingController:
    """Decides the four rates. Does not send them, and does not gate them.

    Gating is `RateCommand.shadow` and belongs to task 30; the tick loop that sends
    belongs to task 31. Keeping the decision separate from both is what lets shadow
    mode log what would have happened without doing it.
    """

    def __init__(self, *, clock: Any = None) -> None:
        import time as _time

        self._now = clock or _time.monotonic
        self._raised_since: float | None = None
        self._holding_until: float = 0.0
        self._last: Decision | None = None

    def decide(self, inputs: Inputs) -> Decision:
        now = self._now()
        reasons: list[str] = []

        wants_more = False
        trigger = Trigger.IDLE

        if inputs.ego_acceleration is not None and abs(inputs.ego_acceleration) >= EVENT_ACCEL_MPS2:
            wants_more = True
            trigger = Trigger.EVENT
            reasons.append(f"|accel| {abs(inputs.ego_acceleration):.1f} >= {EVENT_ACCEL_MPS2}")
        if inputs.policy_margin is not None and inputs.policy_margin <= NARROW_MARGIN:
            wants_more = True
            trigger = Trigger.NARROW_MARGIN
            reasons.append(f"policy margin {inputs.policy_margin:.3f} <= {NARROW_MARGIN}")
        if disagreement(inputs.feed_congestion, inputs.camera_density_bin):
            wants_more = True
            trigger = Trigger.DISAGREEMENT
            reasons.append("feed says jammed, camera sees empty road")

        # Dwell before raising: a single tick of evidence is noise, and paying a
        # camera rebind for it costs more than the tick was worth.
        if wants_more:
            if self._raised_since is None:
                self._raised_since = now
            active = (now - self._raised_since) >= RAISE_DWELL_S
            if active:
                self._holding_until = now + HOLD_S
        else:
            self._raised_since = None
            active = now < self._holding_until
            if active:
                trigger = Trigger.HOLD
                reasons.append(f"holding {self._holding_until - now:.1f}s after the last event")

        rates = dict(ACTIVE_RATES if active else IDLE_RATES)

        # Thermal last, so it composes with whatever the rules above decided rather
        # than overwriting their reasoning -- and so it always wins.
        scale = self._thermal_scale(inputs, reasons)
        if scale < 1.0:
            trigger = Trigger.THERMAL
            # The free tier is not scaled. It is what notices the next event, and
            # backing it off to save a tenth of a percent of the stream would blind
            # the controller exactly when it has decided to look less.
            for key in ("camera_hz", "here_hz"):
                rates[key] = rates[key] * scale

        decision = Decision(
            rates={k: _clamp(v) for k, v in rates.items()},
            trigger=trigger,
            reasons=reasons,
            thermal_scale=scale,
        )
        self._last = decision
        return decision

    def _thermal_scale(self, inputs: Inputs, reasons: list[str]) -> float:
        """How much to back off. Silence is not nominal."""
        if inputs.thermal_status is None and inputs.skin_temp_c is None:
            reasons.append("thermal unknown: no telemetry received")
            return THERMAL_SCALE["unknown"]

        scale = THERMAL_SCALE.get(inputs.thermal_status or "unknown", THERMAL_SCALE["unknown"])
        if inputs.thermal_status is not None and scale < 1.0:
            reasons.append(f"thermal status {inputs.thermal_status}")

        # Skin temperature moves before the status does. Measured on the handset:
        # 5.4 C of warming under camera load with the status `nominal` throughout.
        if inputs.skin_temp_c is not None:
            if inputs.skin_temp_c >= SKIN_HOT_C:
                reasons.append(f"skin {inputs.skin_temp_c:.1f}C >= {SKIN_HOT_C}")
                scale = min(scale, THERMAL_SCALE["severe"])
            elif inputs.skin_temp_c >= SKIN_WARM_C:
                reasons.append(f"skin {inputs.skin_temp_c:.1f}C >= {SKIN_WARM_C}")
                scale = min(scale, THERMAL_SCALE["moderate"])
        return scale
