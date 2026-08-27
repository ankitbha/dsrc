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

import math
from dataclasses import dataclass, field
from typing import Any

#: The wire refuses anything outside `(0, 1000]`, so zero is not a rate. The FLOOR
#: below is this module's own and is deliberately far under the idle rates: an
#: earlier 0.05 sat exactly at `here_hz`'s idle value, so every thermal backoff was
#: clamped straight back up and HERE -- the one modality whose sample is a cellular
#: HTTP call, and therefore the one backoff most wants to cut -- never backed off at
#: all. 0.0075 Hz is legal end to end; the wire was never the constraint.
MIN_RATE_HZ = 0.001
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

#: How old a thermal report may be and still describe the phone. Telemetry runs on
#: its own 1 Hz thread on the handset, guarded so a failing tick is logged and
#: skipped, so the stream can die while the session stays healthy -- it has done, on
#: Android 10. Without this, one `nominal` report pins full rates for the rest of a
#: drive, which is the "no news is cool news" reading the controller refuses.
MAX_TELEMETRY_AGE_S = 10.0

#: Bounds on the query radius. The lower keeps a stationary car asking about a
#: useful stretch; the upper keeps a slow rate from asking about a county.
MIN_QUERY_RADIUS_M = 500.0
MAX_QUERY_RADIUS_M = 10_000.0

#: How old a fix may be and still centre a query. Matches `PhoneGpsReader`'s own
#: staleness window, because the two are answering the same question about the same
#: reading, and symmetric for the same reason the other three predicates here are.
MAX_POSITION_AGE_S = 2.0

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

#: How long a gap between two `decide` calls has to be before the evidence either
#: side of it stops counting as the same evidence.
#:
#: The dwell measures whether evidence PERSISTED, and persistence cannot be
#: established across a stretch where nothing was observed. Without this, one tick of
#: hard braking, a 120 s redial, and one more tick satisfied the dwell outright and
#: raised the camera to 5 Hz -- a 120 s absence of evidence read as 120 s of held
#: evidence. `run_demo`'s worker `continue`s without calling the controller for as
#: long as a rebind takes, and nothing resets the controller on a redial.
#:
#: The bound cannot be `RAISE_DWELL_S`. At the idle camera rate a tick is already
#: 1 s apart, so resetting on any gap wider than the dwell resets on every normal
#: tick and the rates can never rise at all -- 22 tests fail. It is derived instead
#: from the slowest tick this controller can itself cause: the idle camera rate under
#: the deepest thermal backoff. A gap wider than that did not come from a rate this
#: controller chose, so it is the stream stopping rather than the controller slowing
#: down. Derived rather than typed, because a typed constant silently stops covering
#: the idle rate the moment either input changes; the factor is headroom for a tick
#: the Jetson itself was late for.
MAX_EVIDENCE_GAP_S = 2.0 / (IDLE_RATES["camera_hz"] * min(THERMAL_SCALE.values()))
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
    #: How long ago that report arrived. None means never. A report that has aged
    #: out is treated as no report at all -- a phone that said `nominal` once and
    #: then went quiet is not a cool phone.
    telemetry_age_s: float | None = None
    #: Where we are, for the HERE query that goes down with `here_hz`.
    lat: float | None = None
    lon: float | None = None
    #: `GpsFix.valid`. The wire deliberately lets an invalid fix carry whatever the
    #: receiver had -- `GpsRecord.from_wire` range-checks lat/lon only when `valid`
    #: is true, "an invalid fix is allowed to carry whatever the receiver had,
    #: including nothing" -- so coordinates alone do not mean there is a position.
    position_valid: bool = True
    #: How old that fix is. A tunnel leaves the last one in place, and without this
    #: every query for its length is centred on the entrance: at 20 m/s a 30 s
    #: dropout is 600 m, further than the smallest radius this controller asks for.
    position_age_s: float | None = None


@dataclass(frozen=True)
class Decision:
    """Four rates, why, and the settings that ride with them."""

    rates: dict[str, float]
    trigger: str
    reasons: list[str] = field(default_factory=list)
    thermal_scale: float = 1.0
    #: EVERY rule that fired, from the closed vocabulary. `trigger` is one word
    #: because the wire carries one; this is what task 34 attributes on, so a
    #: decision where three rules fired does not lose two of them to free text.
    rules_fired: list[str] = field(default_factory=list)
    #: Rates the floor pulled back up. Without this, `thermal_scale` is a false
    #: statement about the emitted rates: `IDLE_RATES x scale` would not reproduce
    #: them and nothing would say why.
    clamped: list[str] = field(default_factory=list)
    #: The query that goes down with `here_hz`. None when there is no position to
    #: ask about -- a query centred on a fix we do not have describes nowhere.
    here_query: Any = None

    def to_record(self) -> dict[str, Any]:
        return {
            "rates": dict(self.rates),
            "trigger": self.trigger,
            "rules_fired": list(self.rules_fired),
            "reasons": list(self.reasons),
            "thermal_scale": self.thermal_scale,
            "clamped": list(self.clamped),
            "here_radius_m": None if self.here_query is None else self.here_query.radius_m,
        }


def _usable_position(lat: float | None, lon: float | None, *,
                     valid: bool = True, age_s: float | None = None) -> bool:
    """Whether there is a fix worth building a query around.

    `None` is not how this codebase says "no position" -- `GpsFix.lat` and `.lon`
    default to NaN and `PhoneGpsReader` writes `_or_nan(...)` for every field the
    phone omits, so a dropout, a cold start or a tunnel arrives as NaN. A `is None`
    guard let that straight through and produced `circle:nan,nan;r=1333`.

    The cost is not the wasted call. `to_wire_number(nan)` encodes as null,
    `HereQuery.from_wire` refuses a null lat, and `MessageRouter.send` validates by
    round-tripping and RAISES rather than dropping -- deliberately, so a consumer
    cannot swallow its own bug. So one NaN fix takes all four rates down with it and
    the controller stops commanding anything at all for the length of the dropout.
    """
    if lat is None or lon is None:
        return False
    if not valid:
        # `HereFeed.at` already refuses exactly this, as `Outcome.UNUSABLE_FIX`.
        # This guard implemented two of that predicate's three conditions, so a fix
        # the protocol calls meaningless still bought a cellular call about a real
        # place -- the same shape as the NaN defect, one layer up.
        return False
    if age_s is not None and (not math.isfinite(age_s) or abs(age_s) > MAX_POSITION_AGE_S):
        return False
    # NaN and the infinities need no separate check: every comparison against NaN
    # is False, so a chained `-90 <= lat <= 90` rejects them along with anything off
    # the globe. Written this way round on purpose -- the negated form
    # `not (lat < -90 or lat > 90)` is the same for real numbers and TRUE for NaN,
    # which would put `circle:nan,nan` back on the wire. An explicit isfinite() was
    # here and mutation testing showed it could not fire; a comment that survives a
    # rewrite is worth more than a guard that does nothing.
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


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
        self._last_active = False
        self._last_at: float | None = None

    def decide(self, inputs: Inputs) -> Decision:
        now = self._now()
        reasons: list[str] = []
        fired: list[str] = []

        if inputs.ego_acceleration is not None and abs(inputs.ego_acceleration) >= EVENT_ACCEL_MPS2:
            fired.append(Trigger.EVENT)
            reasons.append(f"|accel| {abs(inputs.ego_acceleration):.1f} >= {EVENT_ACCEL_MPS2}")
        if inputs.policy_margin is not None and inputs.policy_margin <= NARROW_MARGIN:
            fired.append(Trigger.NARROW_MARGIN)
            reasons.append(f"policy margin {inputs.policy_margin:.3f} <= {NARROW_MARGIN}")
        if disagreement(inputs.feed_congestion, inputs.camera_density_bin):
            fired.append(Trigger.DISAGREEMENT)
            reasons.append("feed says jammed, camera sees empty road")

        wants_more = bool(fired)

        # Dwell before raising, and a hold after -- but a hold already running is
        # NOT cancelled by fresh evidence. Re-arming the dwell on every new event
        # dropped the rates back to idle mid-hold, so strictly more evidence gave
        # strictly lower rates: a hard-braking event 1.4 s into a valid 5 s hold
        # took the camera from 5 Hz to 1 Hz. With a signal straddling the
        # threshold it produced a camera rebind per tick, which is the exact
        # thrash the dwell and the hold exist to prevent.
        holding = now < self._holding_until
        # A gap in the tick stream is not dwell time. Checked before the dwell is
        # read, so evidence resumes its dwell from now rather than being credited
        # with the silence.
        gapped = (self._last_at is not None
                  and (now - self._last_at) > MAX_EVIDENCE_GAP_S)
        if wants_more:
            if self._raised_since is None or gapped:
                self._raised_since = now
            dwelled = (now - self._raised_since) >= RAISE_DWELL_S
            if dwelled:
                self._holding_until = now + HOLD_S
        else:
            self._raised_since = None
            dwelled = False

        # A dwell already in progress bridges the end of a hold. Without it,
        # evidence that reappears in the last RAISE_DWELL_S of a hold keeps the rate
        # up via `holding`, then drops to idle the moment the hold lapses -- two
        # camera rebinds with the evidence continuously present, measured at 69 such
        # ticks in a 3,000-drive sweep, and routine in stop-and-go traffic.
        #
        # `last_active` and not `_holding_until = now + HOLD_S`: refreshing the hold
        # on evidence that has not dwelled latches the rate high forever. A single
        # blip every four seconds -- evidence that never dwells -- held the camera at
        # 5 Hz indefinitely under that version. This bridges an in-progress dwell and
        # decays to idle the moment the evidence stops.
        # The stored fact, not a value the thermal multiplier has already mangled.
        # Re-deriving it as `camera_hz > IDLE_RATES["camera_hz"]` was exact only
        # while the scale stayed above 0.2: at `critical` an ACTIVE camera is
        # 5.0 x 0.15 = 0.75, BELOW the unscaled idle of 1.0, so the proxy read False
        # and the hold-boundary defect came back -- on a phone at critical, which is
        # the one moment a spurious camera rebind is least affordable.
        #
        # Bounded in time as well. `_last_active` is the previous DECISION, however
        # old, so after a stalled or event-driven caller one tick of evidence raised
        # the camera with no dwell at all. A regular tick loop never sees it because
        # `holding` covers everything inside HOLD_S, but the bridge should not
        # depend on the caller's cadence to be correct.
        recent = self._last_at is not None and (now - self._last_at) <= RAISE_DWELL_S
        bridged = wants_more and self._last_active and recent
        active = dwelled or holding or bridged
        rates = dict(ACTIVE_RATES if active else IDLE_RATES)
        # Kept before the backoff, because the query's radius is sized from it. Cut
        # the rate AND grow the radius by the same factor and the backoff cancels
        # itself: at `critical` the call count fell 6.7x while the area covered rose
        # 45x, so the phone's cellular bytes, decode work and heat -- the axis the
        # backoff exists to relieve -- were left unchanged or worse. Backing off
        # means seeing less of the road, which is the honest trade; it does not mean
        # asking one enormous question instead of several small ones.
        intended_here_hz = rates["here_hz"]

        scale = self._thermal_scale(inputs, reasons)
        if scale < 1.0:
            fired.append(Trigger.THERMAL)
            # The free tier is not scaled. It is what notices the next event, and
            # backing it off to save a tenth of a percent of the stream would blind
            # the controller exactly when it has decided to look less.
            for key in ("camera_hz", "here_hz"):
                rates[key] = rates[key] * scale

        clamped = [k for k, v in rates.items() if _clamp(v) != v]
        rates = {k: _clamp(v) for k, v in rates.items()}

        return self._record(
            rates=rates, fired=fired, reasons=reasons, scale=scale, clamped=clamped,
            active=active, dwelled=dwelled, holding=holding, inputs=inputs, now=now,
            intended_here_hz=intended_here_hz, bridged=bridged,
        )

    def _record(self, *, rates, fired, reasons, scale, clamped, active, dwelled,
                holding, inputs, now, intended_here_hz, bridged) -> Decision:
        """Name the decision in one word, without letting that word mislead.

        `trigger` used to be whichever rule was checked last, which produced two
        wrong statements. During the dwell it named a raise rule on a decision whose
        rates were the idle set. And when a raise and a backoff both fired it said
        `thermal_backoff` while the camera had tripled -- the rule that actually
        moved the rates surviving only in free text, which is what the closed
        vocabulary was introduced to avoid.

        So the word describes the RATE LEVEL, and thermal claims it only when the
        backoff is the whole story. Everything that fired is in `rules_fired`.
        """
        raises = [rule for rule in fired if rule != Trigger.THERMAL]
        # `bridged` included, or a bridged tick matches none of these branches and
        # falls through to `idle` -- an idle-level word on rates that are raised,
        # which is the round-2 defect inverted, in the field task 34 attributes on.
        if active and (dwelled or bridged) and raises:
            trigger = raises[0]
        elif active and holding:
            trigger = Trigger.HOLD
            reasons.append(f"holding {self._holding_until - now:.1f}s after the last event")
        elif scale < 1.0:
            trigger = Trigger.THERMAL
        else:
            trigger = Trigger.IDLE

        decision = Decision(
            rates=rates, trigger=trigger, reasons=reasons, thermal_scale=scale,
            rules_fired=fired, clamped=clamped,
            here_query=self._here_query(inputs, intended_here_hz),
        )
        self._last = decision
        self._last_active = active
        self._last_at = now
        return decision

    def _here_query(self, inputs: Inputs, here_hz: float) -> Any:
        """The query that goes down with `here_hz`.

        Radius follows the rate: asking less often means each answer has to cover
        more of the road ahead, because the vehicle travels further between
        responses. None without a fix -- a query centred on a position we do not
        have describes nowhere, and the phone would spend a cellular call on it.
        """
        if not _usable_position(inputs.lat, inputs.lon,
                                valid=inputs.position_valid,
                                age_s=inputs.position_age_s):
            return None
        from transport.messages import HereQuery

        speed = inputs.ego_speed if inputs.ego_speed and inputs.ego_speed > 0 else 20.0
        # What the vehicle covers before the next answer, with headroom, floored so
        # a stationary car still asks about a useful stretch.
        radius_m = max(MIN_QUERY_RADIUS_M, min(MAX_QUERY_RADIUS_M, speed / max(here_hz, 1e-6) * 2.0))
        return HereQuery(
            in_=f"circle:{inputs.lat:.5f},{inputs.lon:.5f};r={int(radius_m)}",
            location_ref="shape",
            lat=float(inputs.lat), lon=float(inputs.lon), radius_m=float(radius_m),
        )

    def _thermal_scale(self, inputs: Inputs, reasons: list[str]) -> float:
        """How much to back off. Silence is not nominal."""
        if inputs.thermal_status is None and inputs.skin_temp_c is None:
            reasons.append("thermal unknown: no telemetry received")
            return THERMAL_SCALE["unknown"]

        # abs(), because `PhoneGpsReader.is_stale` states the rule outright and
        # there are now four such predicates in this codebase: a stamp from this
        # clock's future is not fresh. Not reachable through PhoneLink today --
        # the age comes from the Jetson's own monotonic clock -- but a predicate
        # that disagrees with its siblings is one caller away from mattering.
        age = inputs.telemetry_age_s
        if age is not None and (not math.isfinite(age) or abs(age) > MAX_TELEMETRY_AGE_S):
            # A report that has aged out is no report. The phone's telemetry thread
            # can die while the session stays healthy, and one `nominal` from
            # minutes ago would otherwise pin full rates for the rest of the drive.
            reasons.append(f"thermal stale: {age}s old")
            return THERMAL_SCALE["unknown"]

        scale = THERMAL_SCALE.get(inputs.thermal_status or "unknown", THERMAL_SCALE["unknown"])
        if scale < 1.0 and inputs.thermal_status is not None:
            reasons.append(f"thermal status {inputs.thermal_status}")
        elif scale < 1.0:
            # Reachable when a status is absent but a skin reading is not. A 40%
            # rate cut with nothing saying why is not a decision anyone can audit.
            reasons.append("thermal status unknown")

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
