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
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

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

#: The two rates thermal backoff actually touches. IMU and GPS are the free tier and
#: are exempt from it -- backing them off to save a fraction of a percent of the
#: stream would blind the controller exactly when it has decided to look less.
THERMAL_SCALED_KEYS = ("camera_hz", "here_hz")

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


#: The three states one rule's check can be in on a given tick. A rule missing from
#: `rules_fired` used to be all three of these at once -- fired-but-lost, quiet, or
#: never evaluated -- and a reader could not tell which without re-reading the code.
#: `fired` says the comparison ran and crossed its threshold; `quiet` says it ran and
#: did not; `not_evaluable` says the input it needs was absent, named in `missing`.
RULE_FIRED = "fired"
RULE_QUIET = "quiet"
RULE_NOT_EVALUABLE = "not_evaluable"

#: The four rules attribution reports on, in the order `decide` checks them. `IDLE`
#: and `HOLD` are rate-level words the trigger can say and never appear here: they
#: describe the rates, not a comparison that ran against an input.
RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT, Trigger.THERMAL)

#: Why the thermal rule backed the rates off, when it did. Closed set: `status` covers
#: both a named status below `nominal` and a missing status defaulting to `unknown`;
#: `skin_warm`/`skin_hot` claim the word only when the skin reading strictly lowered
#: the scale the status alone had already reached; `no_telemetry`/`stale_telemetry`/
#: `unstamped_telemetry` are the three ways silence -- total, aged-out, or unstamped --
#: is read as the `unknown` tier rather than as `nominal`.
THERMAL_CAUSE_STATUS = "status"
THERMAL_CAUSE_SKIN_WARM = "skin_warm"
THERMAL_CAUSE_SKIN_HOT = "skin_hot"
THERMAL_CAUSE_NO_TELEMETRY = "no_telemetry"
THERMAL_CAUSE_STALE_TELEMETRY = "stale_telemetry"
#: A status or skin reading with no age attached -- neither known-fresh nor
#: known-stale, because there is nothing to check `MAX_TELEMETRY_AGE_S`
#: against. Reached through a torn read of a phone link's telemetry and its
#: arrival time as two separate values (see `inputs_from`'s own fix for the
#: race), not through an ordinary missing report, which is `no_telemetry`.
THERMAL_CAUSE_UNSTAMPED_TELEMETRY = "unstamped_telemetry"
THERMAL_CAUSES = frozenset({
    THERMAL_CAUSE_STATUS, THERMAL_CAUSE_SKIN_WARM, THERMAL_CAUSE_SKIN_HOT,
    THERMAL_CAUSE_NO_TELEMETRY, THERMAL_CAUSE_STALE_TELEMETRY, THERMAL_CAUSE_UNSTAMPED_TELEMETRY,
})

#: The `"telemetry"` word `_thermal_scale`'s evidence carries: `"absent"` (no
#: status or skin reading at all), `"unstamped"` (a reading with no age to
#: check), `"stale"` (aged past `MAX_TELEMETRY_AGE_S`), or `"fresh"` (checked
#: and within it). Closed, so a reader censusing this field never needs a
#: fifth bucket for an unrecognised word.
THERMAL_TELEMETRY_STATES = frozenset({"absent", "unstamped", "stale", "fresh"})


@dataclass(frozen=True)
class RuleCheck:
    """One rule's status this tick, and enough evidence to say why.

    `missing` is populated only on `not_evaluable`, naming `Inputs` fields. `evidence`
    carries the numbers a `fired` or `quiet` verdict was reached from -- a threshold
    compared only against a value that produced it can drift by 10x with every test
    still green, so the record states both rather than the verdict alone.
    """

    status: str
    missing: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"status": self.status}
        if self.missing:
            record["missing"] = list(self.missing)
        for key, value in self.evidence.items():
            record[key] = round(value, 4) if isinstance(value, float) else value
        return record


@dataclass(frozen=True)
class Attribution:
    """Which rule, for which sensor, and why -- built where the checks run.

    `rules` always has exactly the four `RULES` entries, closing the three-state gap
    `rules_fired` alone leaves open. `gates` states the dwell/hold/bridge machinery a
    rule word cannot show by itself: a raise rule can fire and still be held off by
    the dwell, and `trigger` alone cannot distinguish that from the rule staying
    quiet. `per_sensor` gives each of the four rates its own composition chain --
    base, level sensitivity, thermal scale, clamp, previous value -- so "for which
    sensor" is answered by the record instead of by a reader who has memorised
    `IDLE_RATES`/`ACTIVE_RATES`.
    """

    rules: dict[str, RuleCheck]
    gates: dict[str, Any]
    per_sensor: dict[str, dict[str, Any]]
    first_decision: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "first_decision": self.first_decision,
            "rules": {name: check.to_record() for name, check in self.rules.items()},
            "gates": self.gates,
            "per_sensor": self.per_sensor,
        }


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
    #: `FeedOwnership.declined` for this tick, when the feed was asked and named a
    #: reason it owns nothing. None both when the feed owns `feed_congestion` and
    #: when nothing carried a reason forward -- those two absences are not told
    #: apart at this layer, which is task 36's subject.
    feed_declined: str | None = None
    #: The phone's last self-report. None means never heard from.
    thermal_status: str | None = None
    skin_temp_c: float | None = None
    #: How long ago that report arrived. None means never. A report that has aged
    #: out is treated as no report at all -- a phone that said `nominal` once and
    #: then went quiet is not a cool phone.
    telemetry_age_s: float | None = None
    #: The `field_sources` class of `ego_acceleration`, as `inputs_from` read
    #: it. `ego_acceleration` is None exactly when this class is a
    #: substitution (`provenance.SUBSTITUTED`): the controller is told the
    #: value was not evidence rather than handed the neutral that stood in
    #: for it.
    ego_acceleration_source: str | None = None
    #: The class behind `ego_speed`. No rule keys on it -- the HERE query
    #: radius is still sized from a held speed with no gate -- but the record
    #: of what `here_radius_m` rested on has to exist somewhere.
    ego_speed_source: str | None = None
    #: The class behind `camera_density_bin` (the obs field
    #: `local_density_bin`). `derived_empty` means the bin rests on an
    #: absence of detections, which is what a blind camera and an empty road
    #: both produce.
    camera_density_bin_source: str | None = None
    #: How long since the perception chain last produced an in-range track.
    #: The only bound available on whether "the camera saw nothing" is a
    #: statement about the road. None means it never has on this drive.
    camera_last_detection_age_s: float | None = None
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

    def to_record(self) -> dict[str, Any]:
        """Every field, at full precision. This is the replay substrate task 35
        scores candidates against, so a value here must be exactly what `decide`
        saw -- `RuleCheck.to_record` rounds its evidence floats to four places
        because evidence is read by a person, not replayed, and a replay built
        from a rounded copy can flip a threshold comparison this decision never
        made.
        """
        return {
            "ego_acceleration": self.ego_acceleration,
            "ego_speed": self.ego_speed,
            "policy_margin": self.policy_margin,
            "feed_congestion": self.feed_congestion,
            "camera_density_bin": self.camera_density_bin,
            "feed_declined": self.feed_declined,
            "thermal_status": self.thermal_status,
            "skin_temp_c": self.skin_temp_c,
            "telemetry_age_s": self.telemetry_age_s,
            "ego_acceleration_source": self.ego_acceleration_source,
            "ego_speed_source": self.ego_speed_source,
            "camera_density_bin_source": self.camera_density_bin_source,
            "camera_last_detection_age_s": self.camera_last_detection_age_s,
            "lat": self.lat,
            "lon": self.lon,
            "position_valid": self.position_valid,
            "position_age_s": self.position_age_s,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Inputs":
        """The strict inverse of `to_record`. A missing field and an unknown field
        are both refused by name rather than defaulted -- a schema drift between
        the writer and this reader must be loud, because a silently defaulted
        input is a silently different replay.
        """
        expected = {f.name for f in fields(cls)}
        present = set(record)
        missing = expected - present
        unknown = present - expected
        if missing or unknown:
            raise ValueError(
                f"decision_inputs does not match Inputs: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**record)


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
    #: Which rule, for which sensor, and why. Required and keyword-only: `_record` is
    #: the only place a `Decision` is built, so there is no caller for an optional
    #: default to protect, and an optional default would itself be a silent-absence
    #: path -- the defect class this field exists to close.
    attribution: "Attribution" = field(kw_only=True)
    #: The instant `decide` read its own clock (:409), carried onto the Decision it
    #: produced. Every gate -- dwell, hold, bridge, gap -- compares differences of
    #: this same instant, so a replay fed anything else, including a clock read a
    #: few microseconds apart by the tick loop, is a different drive at exactly the
    #: ticks that straddle a dwell boundary. Required and keyword-only for the same
    #: reason `attribution` is: there is one construction site and no caller for a
    #: default to protect.
    decided_at_mono: float = field(kw_only=True)

    def to_record(self) -> dict[str, Any]:
        return {
            "rates": dict(self.rates),
            "trigger": self.trigger,
            "rules_fired": list(self.rules_fired),
            "reasons": list(self.reasons),
            "thermal_scale": self.thermal_scale,
            "clamped": list(self.clamped),
            "here_radius_m": None if self.here_query is None else self.here_query.radius_m,
            "attribution": self.attribution.to_record(),
            "decided_at_mono": self.decided_at_mono,
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


#: The two literals `disagreement` compares against, named so attribution can echo
#: them beside the value they judged rather than leaving the threshold implicit in
#: the boolean it produced.
JAMMED_CONGESTION = 0.5
EMPTY_DENSITY_BIN = 0


def disagreement(feed_congestion: float | None, camera_density_bin: int | None) -> bool:
    """Whether the two views contradict each other about where we are.

    Not a disagreement inside the observation: task 28 established the feed reaches
    no field of it. This is the feed's congestion against what the camera sees, and
    a contradiction means one of them is wrong about our position -- worth spending
    samples to resolve rather than picking a winner here.
    """
    if feed_congestion is None or camera_density_bin is None:
        return False
    jammed_ahead = feed_congestion >= JAMMED_CONGESTION
    empty_here = camera_density_bin <= EMPTY_DENSITY_BIN
    return jammed_ahead and empty_here


def _disagreement_check(inputs: Inputs) -> RuleCheck:
    """`disagreement`'s verdict as a three-state check, with its evidence attached.

    Missing rather than quiet when a view is absent -- one source silent is not two
    sources agreeing, and the pin at scripts/remutate.py rests on `disagreement`
    itself returning False for it; this only reports that verdict, never changes it.
    """
    missing = tuple(
        name for name, value in (
            ("feed_congestion", inputs.feed_congestion),
            ("camera_density_bin", inputs.camera_density_bin),
        )
        if value is None
    )
    if missing:
        evidence: dict[str, Any] = {
            "camera_density_bin_source": inputs.camera_density_bin_source,
            "camera_last_detection_age_s": inputs.camera_last_detection_age_s,
        }
        if inputs.feed_declined is not None:
            evidence["feed_declined"] = inputs.feed_declined
        return RuleCheck(status=RULE_NOT_EVALUABLE, missing=missing, evidence=evidence)
    fired = disagreement(inputs.feed_congestion, inputs.camera_density_bin)
    return RuleCheck(
        status=RULE_FIRED if fired else RULE_QUIET,
        evidence={
            "feed_congestion": inputs.feed_congestion,
            "jammed_congestion": JAMMED_CONGESTION,
            "camera_density_bin": inputs.camera_density_bin,
            "empty_density_bin": EMPTY_DENSITY_BIN,
            "camera_density_bin_source": inputs.camera_density_bin_source,
            "camera_last_detection_age_s": inputs.camera_last_detection_age_s,
        },
    )


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
        checks: dict[str, RuleCheck] = {}

        if inputs.ego_acceleration is None:
            checks[Trigger.EVENT] = RuleCheck(
                status=RULE_NOT_EVALUABLE, missing=("ego_acceleration",),
                evidence={"ego_acceleration_source": inputs.ego_acceleration_source},
            )
        else:
            value = inputs.ego_acceleration
            event_fired = abs(value) >= EVENT_ACCEL_MPS2
            checks[Trigger.EVENT] = RuleCheck(
                status=RULE_FIRED if event_fired else RULE_QUIET,
                evidence={"value": value, "threshold": EVENT_ACCEL_MPS2,
                         "ego_acceleration_source": inputs.ego_acceleration_source},
            )
            if event_fired:
                reasons.append(f"|accel| {abs(value):.1f} >= {EVENT_ACCEL_MPS2}")

        if inputs.policy_margin is None:
            checks[Trigger.NARROW_MARGIN] = RuleCheck(status=RULE_NOT_EVALUABLE,
                                                       missing=("policy_margin",))
        else:
            margin = inputs.policy_margin
            margin_fired = margin <= NARROW_MARGIN
            checks[Trigger.NARROW_MARGIN] = RuleCheck(
                status=RULE_FIRED if margin_fired else RULE_QUIET,
                evidence={"value": margin, "threshold": NARROW_MARGIN},
            )
            if margin_fired:
                reasons.append(f"policy margin {margin:.3f} <= {NARROW_MARGIN}")

        checks[Trigger.DISAGREEMENT] = _disagreement_check(inputs)
        if checks[Trigger.DISAGREEMENT].status == RULE_FIRED:
            reasons.append("feed says jammed, camera sees empty road")

        # `wants_more` reads the three raise checks' verdicts rather than a list built
        # alongside them, so a rule that is not_evaluable or quiet cannot be counted
        # here by drifting out of step with what `checks` itself says.
        wants_more = any(checks[rule].status == RULE_FIRED
                         for rule in (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT))

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

        scale, thermal_cause, thermal_evidence = self._thermal_scale(inputs, reasons)
        thermal_fired = scale < 1.0
        checks[Trigger.THERMAL] = RuleCheck(
            status=RULE_FIRED if thermal_fired else RULE_QUIET,
            evidence={**thermal_evidence, "scale": scale, "cause": thermal_cause},
        )
        if thermal_fired:
            # The free tier is not scaled. It is what notices the next event, and
            # backing it off to save a tenth of a percent of the stream would blind
            # the controller exactly when it has decided to look less.
            for key in THERMAL_SCALED_KEYS:
                rates[key] = rates[key] * scale

        clamped = [k for k, v in rates.items() if _clamp(v) != v]
        rates = {k: _clamp(v) for k, v in rates.items()}

        # In `RULES` order, and derived from the same `checks` this tick built rather
        # than accumulated independently -- so `rules_fired` and the attribution
        # record's rule statuses cannot silently drift apart.
        fired = [rule for rule in RULES if checks[rule].status == RULE_FIRED]

        return self._record(
            rates=rates, fired=fired, checks=checks, reasons=reasons, scale=scale,
            clamped=clamped, active=active, dwelled=dwelled, holding=holding,
            inputs=inputs, now=now, intended_here_hz=intended_here_hz, bridged=bridged,
            gapped=gapped, wants_more=wants_more,
        )

    def _record(self, *, rates, fired, checks, reasons, scale, clamped, active, dwelled,
                holding, inputs, now, intended_here_hz, bridged, gapped,
                wants_more) -> Decision:
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

        first_decision = self._last is None
        # `_raised_since` is None exactly when no dwell is running -- reported as
        # null, not 0.0, so a tick with no dwell in progress cannot be read as one
        # that just armed. While a dwell IS running, 0.0 on the arming tick is a
        # measured zero: elapsed time really is zero, not an absent measurement.
        dwell_elapsed_s = None if self._raised_since is None else now - self._raised_since
        gates = {
            "wants_more": wants_more,
            "gapped": gapped,
            "dwell": {
                "elapsed_s": dwell_elapsed_s,
                "required_s": RAISE_DWELL_S,
                "satisfied": dwelled,
            },
            "hold": {
                "active": holding,
                "remaining_s": (self._holding_until - now) if holding else None,
            },
            "bridged": bridged,
            "level": "active" if active else "idle",
        }

        profile = ACTIVE_RATES if active else IDLE_RATES
        per_sensor: dict[str, dict[str, Any]] = {}
        for key, hz in rates.items():
            previous_hz = None if self._last is None else self._last.rates[key]
            per_sensor[key] = {
                "hz": hz,
                "base_hz": profile[key],
                "level_sensitive": IDLE_RATES[key] != ACTIVE_RATES[key],
                "thermal_exempt": key not in THERMAL_SCALED_KEYS,
                "scale": scale if key in THERMAL_SCALED_KEYS else 1.0,
                "clamped": key in clamped,
                "previous_hz": previous_hz,
                "changed": False if self._last is None else previous_hz != hz,
            }

        attribution = Attribution(
            rules={name: checks[name] for name in RULES},
            gates=gates,
            per_sensor=per_sensor,
            first_decision=first_decision,
        )

        decision = Decision(
            rates=rates, trigger=trigger, reasons=reasons, thermal_scale=scale,
            rules_fired=fired, clamped=clamped,
            here_query=self._here_query(inputs, intended_here_hz),
            attribution=attribution,
            # The exact instant the gates above compared, not a fresh read: a
            # second call here would return a different value in production and
            # replaying against it would not reproduce this decision.
            decided_at_mono=now,
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

    def _thermal_scale(
        self, inputs: Inputs, reasons: list[str]
    ) -> tuple[float, str | None, dict[str, Any]]:
        """How much to back off, the `THERMAL_CAUSES` member responsible, and its
        evidence. Silence is not nominal.

        Internal signature -- `decide` is the only caller -- so returning more than
        a float here changes nothing external. `cause` is None exactly when the
        scale is 1.0; every other return path names one member of the closed set.
        """
        if inputs.thermal_status is None and inputs.skin_temp_c is None:
            reasons.append("thermal unknown: no telemetry received")
            evidence = {"thermal_status": None, "skin_temp_c": None,
                       "telemetry": "absent", "telemetry_age_s": None}
            return THERMAL_SCALE["unknown"], THERMAL_CAUSE_NO_TELEMETRY, evidence

        age = inputs.telemetry_age_s
        if age is None:
            # A status or a skin reading with no age attached is not a fresh
            # report -- there is nothing to check `MAX_TELEMETRY_AGE_S`
            # against, so it cannot be confirmed fresh, and it is not the
            # total silence `no_telemetry` names either. Silence is not
            # nominal, and neither is a reading of unknown age.
            reasons.append("thermal unstamped: status or skin reading carries no age")
            evidence = {"thermal_status": inputs.thermal_status,
                       "skin_temp_c": inputs.skin_temp_c,
                       "telemetry": "unstamped", "telemetry_age_s": None}
            return THERMAL_SCALE["unknown"], THERMAL_CAUSE_UNSTAMPED_TELEMETRY, evidence

        # abs(), because `PhoneGpsReader.is_stale` states the rule outright and
        # there are now four such predicates in this codebase: a stamp from this
        # clock's future is not fresh. Not reachable through PhoneLink today --
        # the age comes from the Jetson's own monotonic clock -- but a predicate
        # that disagrees with its siblings is one caller away from mattering.
        if not math.isfinite(age) or abs(age) > MAX_TELEMETRY_AGE_S:
            # A report that has aged out is no report. The phone's telemetry thread
            # can die while the session stays healthy, and one `nominal` from
            # minutes ago would otherwise pin full rates for the rest of the drive.
            reasons.append(f"thermal stale: {age}s old")
            evidence = {"thermal_status": inputs.thermal_status,
                       "skin_temp_c": inputs.skin_temp_c,
                       "telemetry": "stale", "telemetry_age_s": age}
            return THERMAL_SCALE["unknown"], THERMAL_CAUSE_STALE_TELEMETRY, evidence

        scale = THERMAL_SCALE.get(inputs.thermal_status or "unknown", THERMAL_SCALE["unknown"])
        cause: str | None = None
        if scale < 1.0 and inputs.thermal_status is not None:
            reasons.append(f"thermal status {inputs.thermal_status}")
            cause = THERMAL_CAUSE_STATUS
        elif scale < 1.0:
            # Not reachable through `inputs_from`, the only production caller:
            # `thermal_status` and `skin_temp_c` there both come off one
            # `PhoneTelemetry`, and `PhoneTelemetry.thermal_status` is a required
            # string that `require_str` refuses to leave null. So a decoded
            # telemetry frame always carries a status, and the frame's total
            # absence is caught by the `no_telemetry` branch above, before this
            # line runs. This branch only runs when a test builds `Inputs` by
            # hand with a null status and a non-null skin reading.
            reasons.append("thermal status unknown")
            cause = THERMAL_CAUSE_STATUS

        # Skin temperature moves before the status does. Measured on the handset:
        # 5.4 C of warming under camera load with the status `nominal` throughout.
        # A `min` tie leaves the cause alone: skin claims it only when it strictly
        # lowers the scale the status had already reached, not when it merely ties.
        if inputs.skin_temp_c is not None:
            if inputs.skin_temp_c >= SKIN_HOT_C:
                reasons.append(f"skin {inputs.skin_temp_c:.1f}C >= {SKIN_HOT_C}")
                if THERMAL_SCALE["severe"] < scale:
                    cause = THERMAL_CAUSE_SKIN_HOT
                scale = min(scale, THERMAL_SCALE["severe"])
            elif inputs.skin_temp_c >= SKIN_WARM_C:
                reasons.append(f"skin {inputs.skin_temp_c:.1f}C >= {SKIN_WARM_C}")
                if THERMAL_SCALE["moderate"] < scale:
                    cause = THERMAL_CAUSE_SKIN_WARM
                scale = min(scale, THERMAL_SCALE["moderate"])

        evidence = {"thermal_status": inputs.thermal_status, "skin_temp_c": inputs.skin_temp_c,
                   "telemetry": "fresh", "telemetry_age_s": age}
        return scale, cause, evidence
