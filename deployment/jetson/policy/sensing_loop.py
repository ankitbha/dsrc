"""The tick loop's half of sensing control: decide, mark the mode, send.

Sections E and F built a controller, a mode holder, a traffic feed and a telemetry
reader. None of them had ever been constructed by a live path -- `run_demo.py
--phone` took camera and GPS off the handset and sent nothing back down the link.
This is the one place per tick where that closes.

**Two cadences, because the two channels fail differently.** `advisory` is
`latest_wins` at depth one, so the newest is the only one worth having and it goes
every tick. `rate_cmd` is `reliable` at depth 16, and the tick loop runs at camera
rate: a command per tick would put tens per second into a queue whose oldest is
dropped on overflow, so the channel designed never to lose a record would be losing
records continuously while the phone applied the same command over and over.

So a rate command goes down when something about it changed, and otherwise on a
heartbeat. Three reasons, counted apart:

*The decision changed.* Compared on the decided content -- rates, trigger, and the
shadow flag -- and never on `t_capture_mono_ns`, which differs every tick and would
make the test always true. The flag is in the comparison because a mode flip is the
whole difference between a recorded decision and a gated one, even when the rates
are identical.

*The query went stale in space.* The HERE query is centred on the vehicle and its
`in_` string is formatted to five decimal places, about a metre, so it differs on
almost every tick and cannot be part of the changed-content test. But a query is
about the road ahead, so it goes stale by distance travelled rather than by time:
resent once the vehicle has moved a quarter of the radius it last asked about.

*Nothing has been sent for a while.* Which is what puts a phone that reconnected
mid-drive back onto the current command, since a rebind leaves it running whatever
it had.

This is a transport concern layered on the controller's dwell and hold, not a second
policy. The controller decides; this decides whether the wire has already been told.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from geo import haversine_m
from policy.sensing_controller import Decision, Inputs, SensingController
from policy.shadow_mode import SHADOW, ModeHolder, command_for
from sensors.time_sync import capture_stamp_ns
from transport.messages import DROP_KEYS, RATE_KEYS

#: How long a rate command may go unsent while nothing changes.
RATE_CMD_HEARTBEAT_S = 5.0

#: How far the vehicle may travel from the point it last asked about, as a fraction
#: of that query's radius, before the query is resent. A fraction rather than a fixed
#: distance because the radius already scales with speed and query rate: at 2 km it
#: is 500 m, and at the 200 m floor it is 50 m.
QUERY_REFRESH_FRACTION = 0.25


@dataclass(frozen=True)
class TickOutcome:
    """What the loop did with one tick."""

    decision: Decision
    #: The command built for it, whatever its fate. Present even when nothing was
    #: sent, because the decision log task 35 scores is about what was DECIDED.
    command: Any
    advisory_sent: bool
    command_sent: bool
    #: Why the command went, or None when it did not. One of the three reasons
    #: above, so a drive's send pattern can be attributed rather than counted.
    send_reason: str | None
    #: The exact `Inputs` `decide` was called with. Required, not optional: an
    #: optional default here is a silent-absence path, the same rule task 34 D5
    #: applied to `attribution`. This is what makes the logged decision replayable
    #: as a pure function of the record instead of a reconstruction from rounded
    #: evidence.
    inputs: Inputs
    #: What the phone reports it is actually running, built by `reference_from`
    #: against this tick's own `now`. Required for the same reason `inputs` is.
    reference: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """The shape `run_demo` writes to `record["sensing"]`: the decision
        exactly as `Decision.to_record()` states it, the exact inputs it was
        decided from, what the phone says it was running while that decision was
        made, and whether the command reached the wire.
        """
        return {
            **self.decision.to_record(),
            "decision_inputs": self.inputs.to_record(),
            "reference": dict(self.reference),
            "shadow": self.command.shadow,
            "advisory_sent": self.advisory_sent,
            "command_sent": self.command_sent,
            "send_reason": self.send_reason,
        }


def _margin(head_probs: dict[str, list[float]]) -> float | None:
    """`top1 - top2`, over the head closest to a boundary.

    The minimum over active heads, not the mean: the policy is at a boundary if ANY
    of its heads is, and averaging would let three confident heads hide the one that
    is about to change its mind.
    """
    margins = []
    for probs in head_probs.values():
        if probs is None or len(probs) < 2:
            continue
        top1, top2 = sorted(probs, reverse=True)[:2]
        margins.append(float(top1) - float(top2))
    return min(margins) if margins else None


def inputs_from(tick: Any, phone: Any, *, now: float) -> Inputs:
    """Everything the controller may look at, gathered from one tick and the link.

    The ages are computed against `now` rather than read off the objects, because
    freshness is a question about this instant and the stamps are from earlier ones.
    """
    obs = tick.obs_result.obs
    feed = tick.obs_result.feed
    telemetry = getattr(phone, "telemetry", None)
    telemetry_at = getattr(phone, "telemetry_at_mono", None)
    return Inputs(
        ego_acceleration=obs.get("ego_acceleration"),
        ego_speed=obs.get("ego_speed"),
        policy_margin=_margin(tick.policy.head_probs),
        # The feed's own number, from beside the observation vector rather than in
        # it -- task 28 concluded the feed owns no observation field, and the
        # controller is the consumer that reading was published for.
        feed_congestion=None if feed is None else feed.downstream_congestion,
        # The feed's own named reason it owns nothing, when it has one -- so
        # disagreement's not_evaluable entry can say why the view is missing rather
        # than just that it is.
        feed_declined=None if feed is None else feed.declined,
        camera_density_bin=None if obs.get("local_density_bin") is None
        else int(obs["local_density_bin"]),
        thermal_status=getattr(telemetry, "thermal_status", None),
        skin_temp_c=getattr(telemetry, "skin_temp_c", None),
        telemetry_age_s=None if telemetry_at is None else now - telemetry_at,
        lat=tick.gps.lat,
        lon=tick.gps.lon,
        position_valid=bool(tick.gps.valid),
        position_age_s=tick.obs_result.diagnostics.get("gps_age_s"),
    )


def reference_from(phone: Any, *, now: float) -> dict[str, Any]:
    """What the phone says it is actually running, witnessed rather than assumed.

    `achieved` is the phone's windowed average and `dropped` is cumulative
    (`TelemetryReporter` on the phone side); both arrive on every telemetry frame
    and neither had a reader on this side before this task. Null together with
    `age_s`, and `absent` named, when the phone has never reported -- never
    zeros, because a phone reporting zero achieved and a phone never heard from
    are different drives.
    """
    telemetry = getattr(phone, "telemetry", None)
    if telemetry is None:
        return {"achieved": None, "dropped": None, "age_s": None, "absent": "no_telemetry"}
    telemetry_at = getattr(phone, "telemetry_at_mono", None)
    return {
        "achieved": {key: telemetry.achieved[key] for key in RATE_KEYS},
        "dropped": {key: telemetry.dropped[key] for key in DROP_KEYS},
        "age_s": None if telemetry_at is None else now - telemetry_at,
        "absent": None,
    }


class SensingLoop:
    """One decision per tick, and the two messages that may follow it."""

    def __init__(
        self,
        *,
        controller: SensingController | None = None,
        modes: ModeHolder | None = None,
        clock: Any = None,
        heartbeat_s: float = RATE_CMD_HEARTBEAT_S,
    ) -> None:
        self._now = clock or time.monotonic
        self.controller = controller or SensingController(clock=self._now)
        # Shadow unless told otherwise, all the way down: a loop that comes up
        # gating for real because nobody passed a flag is the wrong failure to have.
        self.modes = modes or ModeHolder(SHADOW, clock=self._now)
        self.heartbeat_s = heartbeat_s
        self._last_sent: tuple | None = None
        self._last_sent_at: float | None = None
        self._last_query: Any = None
        self.ticks = 0
        self.sends_by_reason: dict[str, int] = {}
        #: One decision, counted by its single word. Sums to `ticks`.
        self.decisions_by_trigger: dict[str, int] = {}
        #: Every rule's status, every tick -- `{rule_name: {status: count}}`. Answers
        #: "how often was this rule fired / quiet / not_evaluable" over a whole drive
        #: without re-reading every tick's attribution record.
        self.rules_by_status: dict[str, dict[str, int]] = {}

    def on_tick(self, tick: Any, phone: Any = None) -> TickOutcome:
        now = self._now()
        self.ticks += 1
        inputs = inputs_from(tick, phone, now=now)
        decision = self.controller.decide(inputs)
        reference = reference_from(phone, now=now)
        self.decisions_by_trigger[decision.trigger] = (
            self.decisions_by_trigger.get(decision.trigger, 0) + 1
        )
        for rule_name, check in decision.attribution.rules.items():
            counts = self.rules_by_status.setdefault(rule_name, {})
            counts[check.status] = counts.get(check.status, 0) + 1
        # The frame's capture stamp, on our clock -- the advisory and the command are
        # both about the tick that produced them, not about the moment of sending.
        # `capture_stamp_ns` is the same conversion `Tick.to_record()` uses, so this
        # value and the tick's own logged key are the identical integer.
        capture_ns = capture_stamp_ns(tick.t_capture_mono)
        command = command_for(decision, self.modes.mode, t_capture_mono_ns=capture_ns)

        advisory_sent = False
        command_sent = False
        reason = self._send_reason(command, now)
        if phone is not None:
            advisory_sent = phone.send_advisory(
                tick.advisory, t_capture_mono_ns=capture_ns
            )
            if reason is not None:
                command_sent = phone.send_rate_command(command)
        if reason is not None and command_sent:
            self._last_sent = _content(command)
            self._last_sent_at = now
            self._last_query = command.here
            self.sends_by_reason[reason] = self.sends_by_reason.get(reason, 0) + 1
        return TickOutcome(
            decision=decision,
            command=command,
            advisory_sent=advisory_sent,
            command_sent=command_sent,
            send_reason=reason if command_sent else None,
            inputs=inputs,
            reference=reference,
        )

    def _send_reason(self, command: Any, now: float) -> str | None:
        """Why this command should go down, or None when the phone already has it.

        Nothing has been sent yet is its own reason and not a heartbeat: a first
        command that had to wait out the heartbeat period would leave the phone on
        its startup rates for the opening seconds of every drive.
        """
        if self._last_sent is None:
            return "first"
        if _content(command) != self._last_sent:
            return "changed"
        if self._query_moved(command.here):
            return "query_moved"
        if self._last_sent_at is not None and now - self._last_sent_at >= self.heartbeat_s:
            return "heartbeat"
        return None

    def _query_moved(self, query: Any) -> bool:
        """Whether the vehicle has left the stretch it last asked about."""
        if query is None or self._last_query is None:
            # Gaining or losing a fix changes the command's content, so it is already
            # covered by "changed" -- and calling it a move would report a distance
            # from a position that does not exist.
            return False
        moved = haversine_m(self._last_query.lat, self._last_query.lon,
                            query.lat, query.lon)
        return moved >= QUERY_REFRESH_FRACTION * self._last_query.radius_m

    def to_record(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "sends_by_reason": dict(self.sends_by_reason),
            "rate_commands_sent": sum(self.sends_by_reason.values()),
            "heartbeat_s": self.heartbeat_s,
            "mode": self.modes.to_record(),
            "decisions_by_trigger": dict(self.decisions_by_trigger),
            "rules_by_status": {name: dict(counts) for name, counts in self.rules_by_status.items()},
        }


def _content(command: Any) -> tuple:
    """What makes two commands the same command, for the phone.

    Deliberately not `t_capture_mono_ns`, which differs every tick and would make
    every command look new -- and deliberately not the query's coordinates, which
    differ by metres every tick for the same reason. The radius IS included: it is
    the controller's statement about how far ahead it is asking, and it changes only
    when the rate or the speed does.
    """
    return (
        tuple(sorted(command.rates.items())),
        command.trigger,
        command.shadow,
        None if command.here is None else command.here.radius_m,
    )
