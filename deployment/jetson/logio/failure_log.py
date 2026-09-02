"""A time axis, an episode, and a reader for the failures this repository already
detects.

Task 38's own inventory found 186 distinct failure detections across the phone and
the Jetson -- GPS dropout, HERE quota exhaustion, dropped frames, transport stalls,
all of it -- and almost none of it carries an instant, an episode, or a reader.
`HereFeed.refused_by_reason` moving from 3 to 4 says nothing about which second it
moved in; a counter has no second endpoint to say a condition ended; and
`{"type": "system_error"}`, the repository's one existing failure record, is written
by one line in `metadata_logger.py` and read by nothing. This module builds none of
the detection -- every source it reads already exists -- and builds only the three
missing things: a 1 Hz scan that makes "nothing failed" different from "nothing was
watching", an episode with a beginning, an end and a recovery outcome, and the
`eval_run.py` section that reads them back.

**The registry is a projection, not a second detector.** Every `Source` below names
an existing counter, dict or field and reads it; none of them changes what the
object they read does or how it counts. Where the sampler's own view disagrees with
the counter it reads -- a session-scoped field reset by a redial faster than the
sampler noticed, say -- the counter is right and the sampler says so
(`counter_went_backwards`) rather than clamping the disagreement away.

**Which of the four vocabularies this is.** Three closed sets already exist and this
module imports rather than restates them: `RULE_FIRED` / `RULE_QUIET` /
`RULE_NOT_EVALUABLE` answer "was detection running" (task 34's question), and
`STAGE_BASIS_MEASURED` / `_CONVERTED` / `_ABSENT` answer "how well is the instant
known" (task 33's question). A failure's own identity is a `(source, reason)` pair,
where `reason` is drawn from that source's own existing closed set -- `Outcome`,
`REASONS`, `SessionEndReason`, `DROP_KEYS` -- never from a set invented here. The one
genuinely new vocabulary is an episode's `outcome`, three members long, and the third
member, `unobservable`, is the reason this module exists: a source that stopped
being readable while an episode was open must never report a recovery nobody
observed, and it must not report the episode as still open either, since the
interval after the source went dark was not observed.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET
from sensors.here_feed import Outcome
from sensors.thermal import ABSENT_NO_SAMPLE_YET, ABSENT_SAMPLER_STOPPED
from sensors.time_sync import (
    STAGE_BASIS_ABSENT,
    STAGE_BASIS_CONVERTED,
    STAGE_BASIS_MEASURED,
)
from transport.channels import Channel
from transport.messages import DROP_KEYS, REASONS
from transport.session import SessionEndReason

# Re-exported, not retyped -- see `thermal.py`'s own re-export for the precedent.
FAILURE_BASIS_MEASURED = STAGE_BASIS_MEASURED
FAILURE_BASIS_CONVERTED = STAGE_BASIS_CONVERTED
FAILURE_BASIS_ABSENT = STAGE_BASIS_ABSENT
#: A scan whose age is more than twice its own interval. Task 37's D4 precedent:
#: derived from `interval_s` rather than typed, so a reader can undo it from
#: `bound_s` alone.
FAILURE_BASIS_STALE = "stale"

#: Re-exported from `thermal.py` for the same reason `thermal.py` re-exports
#: `time_sync`'s: these two absence reasons mean here precisely what they mean
#: there, and a second copy of the strings is the drift task 36 gave the
#: vocabulary one home to prevent.
FAILURE_ABSENT_NO_SAMPLE_YET = ABSENT_NO_SAMPLE_YET
FAILURE_ABSENT_SAMPLER_STOPPED = ABSENT_SAMPLER_STOPPED

#: The one closed set this task adds. Three members; see the module docstring for
#: why the third exists.
OUTCOME_RECOVERED = "recovered"
OUTCOME_OPEN_AT_END = "open_at_end"
OUTCOME_UNOBSERVABLE = "unobservable"
OUTCOMES = frozenset({OUTCOME_RECOVERED, OUTCOME_OPEN_AT_END, OUTCOME_UNOBSERVABLE})

#: Why a source could not be read this pass. Closed.
MISSING_NO_PHONE = "phone"            # no PhoneLink was constructed
MISSING_NO_SESSION = "session"        # a link exists, no session is bound
MISSING_NO_TELEMETRY = "telemetry"    # spelled as sensing_loop.reference_from's is
MISSING_SESSION_MOVED = "session_changed"  # the pass straddled a rebind (D15)
MISSING_NO_SOURCE = "source"          # the object itself is absent this run
MISSING = frozenset({
    MISSING_NO_PHONE, MISSING_NO_SESSION, MISSING_NO_TELEMETRY,
    MISSING_SESSION_MOVED, MISSING_NO_SOURCE,
})

#: The instrument disagreeing with the counter it reads. Never clamped -- see the
#: module docstring.
BACKWARDS = "counter_went_backwards"

#: The ten failure kinds that reach an analysis only through the phone's own
#: `SessionLog` (D11) -- not a registry source, because nothing on the Jetson
#: reads them at all; they are what §"Phone" fact 5 names as reaching, before
#: this task, exactly one place: a logcat line at teardown. Asserted equal to
#: Kotlin's `FailureKinds.ALL` by `InteropTest`, the mechanism
#: `scripts/refusal_reasons.py` already uses, so "this matches Kotlin" is
#: executed rather than asserted by hand.
PHONE_OFFLINE_KINDS = frozenset({
    "link.dial_failed", "link.session_ended", "imu.no_hardware",
    "imu.timebase_mismatched", "here.unconfigured", "service.come_up_failed",
    "service.permission_revoked", "service.teardown_failed",
    "service.resources_held", "log.self",
})

#: How many consecutive still passes close an open episode (D7). A pass count, not
#: a time: `close_after_s` is this times the sampler's own `interval_s`, so a
#: faster sampler closes sooner and the bound travels with the record instead of
#: being typed as a constant seconds value.
QUIET_PASSES_TO_CLOSE = 3

#: The episode cap per source (D10), `phone_link.refusals`' precedent verbatim.
MAX_EPISODES_PER_SOURCE = 100

#: A detail string longer than this is truncated and the record says so. Named
#: because `gps_reader`'s own `last_error` is `f"{type(exc).__name__}: {exc}"` and
#: a `serial.SerialException` message can run past 200 characters.
DETAIL_MAX_LEN = 200


def _capped(text: str | None) -> tuple[str | None, bool]:
    """`(text, truncated)`. `None` stays `None` -- an absent detail is not the
    same as an empty or truncated one."""
    if text is None:
        return None, False
    if len(text) <= DETAIL_MAX_LEN:
        return text, False
    return text[:DETAIL_MAX_LEN], True


# ---------------------------------------------------------------------------
# What one source's accessor reports on one pass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSnapshot:
    """The typed accessor's answer for one source, one pass.

    `readable=False` is `not_evaluable` for this pass; every other field is
    meaningless when it is False and `missing` names why.

    `by_reason`'s meaning depends on `Source.cumulative`. For a cumulative
    source it is the object's own running total, sliced by reason -- the
    sampler diffs it against the previous pass to find what moved. For a
    predicate source (one sampled once per pass with no counter of its own,
    such as a boolean or a sticky "latest error" field) it is this pass's own
    state: empty when nothing is active, one entry mapping the active reason
    to `1` when something is.

    `session_id` is this source's own session identity, when it has one --
    `None` for a run-scoped source. `value` and `channel` are optional
    context carried onto an occurrence record; `detail` is a free-text detail,
    capped at write time.
    """

    readable: bool
    by_reason: Mapping[str, int] = field(default_factory=dict)
    session_id: Any = None
    missing: str | None = None
    value: float | None = None
    channel: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class Source:
    """One row of the registry: the site an existing counter lives at, and how
    to turn its current state into a `SourceSnapshot`.

    `vocabulary` is a frozenset, a predicate `(str) -> bool`, or `None`.
    `None` together with `reason` set is a single-reason source, where every
    occurrence carries the same fixed word. `None` with `reason` also `None`
    is a freeform source -- "the string itself, capped" -- where the reason
    is whatever detail the accessor names for that occurrence and no
    membership check applies.

    `cumulative` distinguishes a real running counter (diffed pass to pass;
    a decrease is `counter_went_backwards`) from a predicate sampled once per
    pass (active or not, no counter to go backwards).
    """

    name: str
    read: Callable[["_Context"], SourceSnapshot]
    vocabulary: frozenset[str] | Callable[[str], bool] | None
    reason: str | None
    scope: str  # "run", "session", or "mixed" (D14's own word for phone.dropped)
    cumulative: bool
    event_records: bool
    device: str  # "jetson" or "phone"


def reason_is_valid(source: Source, reason: str) -> bool:
    """Whether `reason` is a member of `source`'s own declared vocabulary.

    A freeform source (`vocabulary is None and reason is None`) accepts any
    string -- there is nothing to check, by design, because the source's own
    field was never a closed set to begin with (a sticky error string, an
    errno name, a proxy-reason string). Every other source is checked, and
    this is the one function a corpus-driving test calls once per emitted
    reason so a fifth vocabulary cannot be built by accident.
    """
    if source.reason is not None:
        return reason == source.reason
    if source.vocabulary is None:
        return True
    if callable(source.vocabulary):
        return bool(source.vocabulary(reason))
    return reason in source.vocabulary


# ---------------------------------------------------------------------------
# The episode.
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    episode_id: int
    source: str
    reason: str
    opened_t_mono: float
    opened_t_wall: float
    last_t_mono: float
    n: int
    first_pass_n: int
    session_id: Any
    tick_id: int | None
    basis: str
    bound_s: float | None
    value: float | None = None
    channel: str | None = None
    detail: str | None = None
    closed_t_mono: float | None = None
    closed_t_wall: float | None = None
    outcome: str | None = None

    def open_record(self, device: str) -> dict[str, Any]:
        detail, _ = _capped(self.detail)
        return {
            "type": "failure_event", "phase": "open", "episode_id": self.episode_id,
            "source": self.source, "reason": self.reason, "device": device,
            "t_wall": self.opened_t_wall, "t_mono": self.opened_t_mono,
            "basis": self.basis, "bound_s": self.bound_s,
            "session_id": self.session_id, "tick_id": self.tick_id,
            "channel": self.channel, "value": self.value,
            "first_pass_n": self.first_pass_n, "detail": detail,
        }

    def close_record(self, device: str, *, close_after_s: float) -> dict[str, Any]:
        assert self.outcome is not None and self.closed_t_mono is not None
        duration_s = self.closed_t_mono - self.opened_t_mono
        return {
            "type": "failure_event", "phase": "close", "episode_id": self.episode_id,
            "source": self.source, "device": device,
            "t_wall": self.closed_t_wall, "t_mono": self.closed_t_mono,
            "outcome": self.outcome, "duration_s": round(duration_s, 3),
            "n": self.n, "last_t_mono": self.last_t_mono,
            "close_after_s": close_after_s, "basis": self.basis,
            "bound_s": self.bound_s, "session_id": self.session_id,
        }


@dataclass
class _SourceState:
    passes_attempted: int = 0
    passes_readable: int = 0
    baseline_total: int = 0
    baseline_by_reason: dict[str, int] = field(default_factory=dict)
    baseline_session_id: Any = None
    open_episode: Episode | None = None
    quiet_streak: int = 0
    run_total: int = 0
    by_reason_total: dict[str, int] = field(default_factory=dict)
    episodes_closed: int = 0
    episodes_not_kept: int = 0
    events_written: int = 0
    first_t_mono: float | None = None
    last_t_mono: float | None = None
    last_missing: str | None = None
    last_readable: bool = False
    #: The instant of the last pass this source was readable, updated on
    #: every readable pass regardless of whether anything moved. Distinct
    #: from `Episode.last_t_mono` (the last pass an OCCURRENCE was observed):
    #: an `unobservable` close measures to this instant, since it is the last
    #: moment this drive actually watched the source, not the last moment it
    #: saw movement or the moment the gap was noticed.
    last_readable_t_mono: float | None = None
    #: `last_readable` starts False, which is correct: the first pass has not
    #: run yet, and `passes_attempted == 0` is what a reader checks before
    #: trusting `status` at all.


# ---------------------------------------------------------------------------
# The context one pass reads from -- one snapshot of the live objects, taken
# once, so every source in this pass sees the same session (D15).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    phone: Any
    camera: Any
    gps: Any
    blind_ticks_total: int
    pipeline_exception: str | None
    pass_session_id: Any


def _phone_session(phone: Any) -> Any:
    return getattr(phone, "session", None) if phone is not None else None


def _fixed(reason_word: str, total: int) -> SourceSnapshot:
    return SourceSnapshot(readable=True, by_reason={reason_word: total} if total else {})


# -- camera sources -----------------------------------------------------------


def _read_camera_dropped_unconsumed(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    total = getattr(ctx.camera, "dropped_frames", None)
    if total is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    return _fixed("unconsumed", int(total))


def _read_camera_decode_failures(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None or not hasattr(ctx.camera, "decode_failures"):
        # Only `PhoneCameraStream` has this counter at all -- a local
        # `CameraStream` never decodes a JPEG, so the question does not apply.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    snap = _fixed("jpeg_decode", int(ctx.camera.decode_failures))
    return SourceSnapshot(readable=True, by_reason=snap.by_reason, session_id=ctx.pass_session_id)


def _read_camera_file_recoveries(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    total = getattr(ctx.camera, "file_recoveries", None)
    if total is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    return _fixed("decoder_poisoned", int(total))


def _read_camera_end_of_stream(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    active = bool(getattr(ctx.camera, "end_of_stream", False))
    return SourceSnapshot(readable=True, by_reason={"end_of_stream": 1} if active else {})


def _read_camera_reader_failure(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None or not hasattr(ctx.camera, "failure"):
        # Only a phone-fed source (`_PhoneSource`) has a `.failure` field --
        # a local `CameraStream` has no reader thread that can name one.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    failure = ctx.camera.failure
    detail, _ = _capped(failure)
    return SourceSnapshot(
        readable=True, session_id=ctx.pass_session_id, detail=detail,
        by_reason={detail: 1} if detail else {},
    )


def _read_camera_blind_ticks(ctx: _Context) -> SourceSnapshot:
    # Managed directly by `FailureSampler.note_no_frame`, not by polling here
    # -- see the module docstring on the two direct-notification sources.
    # Always readable and never itself reports movement: `sample_once` checks
    # this pseudo-source's own quiet timer separately.
    return SourceSnapshot(readable=True, by_reason={})


# -- gps sources ----------------------------------------------------------


def _read_gps_not_fresh(ctx: _Context) -> SourceSnapshot:
    if ctx.gps is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    if not ctx.gps.is_stale():
        return SourceSnapshot(readable=True, by_reason={})
    fix = ctx.gps.latest()
    if getattr(fix, "t_mono", 0.0) <= 0.0:
        reason = "absent"
    elif not getattr(fix, "valid", False):
        reason = "invalid"
    else:
        reason = "stale"
    age = None
    try:
        import sensors.time_sync as _ts

        age = fix.age_s(_ts.now_mono())
    except Exception:  # pragma: no cover - value is informational only
        age = None
    detail = None if age is None else f"gps_age_s {age:.3f}"
    return SourceSnapshot(readable=True, by_reason={reason: 1}, value=age, detail=detail)


def _read_gps_parse_errors(ctx: _Context) -> SourceSnapshot:
    if ctx.gps is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    return _fixed("nmea_parse", int(ctx.gps.diagnostics.parse_errors))


def _read_gps_ingest_errors(ctx: _Context) -> SourceSnapshot:
    if ctx.gps is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    return _fixed("ingest_raised", int(ctx.gps.diagnostics.ingest_errors))


def _read_gps_last_error(ctx: _Context) -> SourceSnapshot:
    if ctx.gps is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    last_error = ctx.gps.diagnostics.last_error
    detail, _ = _capped(last_error or None)
    return SourceSnapshot(readable=True, detail=detail, by_reason={detail: 1} if detail else {})


def _read_gps_rate_unconfigured(ctx: _Context) -> SourceSnapshot:
    if ctx.gps is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    active = not bool(ctx.gps.diagnostics.rate_configured)
    return SourceSnapshot(
        readable=True, by_reason={"ubx_rate_config_failed": 1} if active else {},
    )


# -- here sources -----------------------------------------------------------


def _outcome_members() -> frozenset[str]:
    return frozenset(
        v for k, v in vars(Outcome).items() if not k.startswith("_") and isinstance(v, str)
    )


_OUTCOME_MEMBERS = _outcome_members()


def _here_refused_reason_valid(reason: str) -> bool:
    base = reason.split(":", 1)[0]
    return base in _OUTCOME_MEMBERS


def _read_here_refused(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None or getattr(phone, "here", None) is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    by_reason = dict(phone.here.refused_by_reason)
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


def _read_here_reader_failures(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    total = int(phone.here_failures)
    detail, _ = _capped(phone.here_failure)
    key = detail if detail is not None else "here_reader_failed"
    return SourceSnapshot(
        readable=True, by_reason={key: total} if total else {}, detail=detail,
    )


def _read_here_proxied_stamps(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None or getattr(phone, "here", None) is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    total = int(phone.here.proxied_stamps)
    return SourceSnapshot(
        readable=True, by_reason={"proxy": total} if total else {}, session_id=ctx.pass_session_id,
    )


# -- phone-telemetry sources --------------------------------------------------


def _read_phone_dropped(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    telemetry = getattr(phone, "telemetry", None)
    if telemetry is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)
    dropped = getattr(telemetry, "dropped", None) or {}
    by_reason = {k: int(v) for k, v in dropped.items() if k in DROP_KEYS}
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


def _read_phone_here_errors(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    telemetry = getattr(phone, "telemetry", None)
    if telemetry is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)
    total = int(getattr(telemetry, "here_errors", 0))
    return SourceSnapshot(
        readable=True, by_reason={"here_error": total} if total else {}, session_id=ctx.pass_session_id,
    )


# -- link sources -------------------------------------------------------------


def _read_link_down(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    session = _phone_session(phone)
    down = session is None or bool(getattr(session, "is_closed", False))
    return SourceSnapshot(readable=True, by_reason={"no_session": 1} if down else {})


_SESSION_END_REASONS = frozenset(e.value for e in SessionEndReason)


def _read_link_session_end(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    by_reason: dict[str, int] = {}
    for entry in phone.rebinds:
        reason = entry.get("previous_end_reason")
        if reason is not None:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    session = _phone_session(phone)
    if session is not None and bool(getattr(session, "is_closed", False)) and phone.supervisor_ended is not None:
        # The current, terminal session: it ended and nothing replaced it.
        # An `end_reason` here is a fact this drive is done with, so it is
        # counted once, not once per pass it stays terminal.
        reason = getattr(session, "end_reason", None)
        reason = getattr(reason, "value", reason)
        if reason is not None:
            key = f"{reason}:final"
            by_reason[key] = 1
    return SourceSnapshot(readable=True, by_reason=by_reason)


def _link_session_end_valid(reason: str) -> bool:
    base = reason.split(":", 1)[0]
    return base in _SESSION_END_REASONS


def _read_link_refusals(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    total = len(phone.refusals) + phone.refusals_not_kept
    detail, _ = _capped(phone.refusals[-1] if phone.refusals else None)
    key = detail if detail is not None else "refused"
    return SourceSnapshot(readable=True, by_reason={key: total} if total else {}, detail=detail)


def _read_link_displaced(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    return _fixed("displaced", int(phone._listener.displaced))


def _read_link_workers_leaked(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    return _fixed("handshake_worker_leaked", int(phone._listener.handshake_workers_leaked))


def _read_link_sends_lost(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    by_reason = {
        "no_session": int(phone.sends_without_a_session),
        "refused": int(phone.sends_refused),
    }
    return SourceSnapshot(readable=True, by_reason={k: v for k, v in by_reason.items() if v})


_SUPERVISOR_ENDED_FIXED = frozenset({"readers_would_not_stop", "rebind_failed", "stopped"})
_GAVE_UP_PATTERN = re.compile(r"^gave_up_after_.+s$")


def _supervisor_ended_valid(reason: str) -> bool:
    return reason in _SUPERVISOR_ENDED_FIXED or bool(_GAVE_UP_PATTERN.match(reason))


def _read_link_supervisor_ended(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    ended = phone.supervisor_ended
    return SourceSnapshot(readable=True, by_reason={ended: 1} if ended else {})


# -- wire sources -------------------------------------------------------------


def _session_stats_or_none(phone: Any) -> Any:
    session = _phone_session(phone)
    if session is None or bool(getattr(session, "is_closed", False)):
        return None
    return session.stats()


def _read_wire_dropped(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    stats = _session_stats_or_none(phone)
    if stats is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    outbound = sum(c.dropped_outbound for c in stats.channels.values())
    inbound = sum(c.dropped_inbound for c in stats.channels.values())
    by_reason = {k: v for k, v in {"outbound": outbound, "inbound": inbound}.items() if v}
    channel = None
    if by_reason:
        channel = max(
            stats.channels.items(),
            key=lambda kv: kv[1].dropped_outbound + kv[1].dropped_inbound,
        )[0].value
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id, channel=channel)


def _read_wire_seq_gaps(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    stats = _session_stats_or_none(phone)
    if stats is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    total = sum(c.seq_gaps for c in stats.channels.values())
    missing_seqs = sum(c.missing_seqs for c in stats.channels.values())
    channel = None
    if total:
        channel = max(stats.channels.items(), key=lambda kv: kv[1].seq_gaps)[0].value
    return SourceSnapshot(
        readable=True, by_reason={"seq_gap": total} if total else {},
        session_id=ctx.pass_session_id, channel=channel, value=float(missing_seqs),
    )


def _read_wire_decode_errors(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    router = getattr(phone, "router", None)
    if router is None or _session_stats_or_none(phone) is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    per_channel = router.stats()
    by_reason: dict[str, int] = {}
    winning_channel, winning_total = None, 0
    for channel, stats in per_channel.items():
        channel_total = 0
        for reason, count in stats.errors_by_reason.items():
            by_reason[reason] = by_reason.get(reason, 0) + count
            channel_total += count
        if channel_total > winning_total:
            winning_channel, winning_total = channel.value, channel_total
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id, channel=winning_channel)


def _read_wire_send_rejected(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    router = getattr(phone, "router", None)
    if router is None or _session_stats_or_none(phone) is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    per_channel = router.stats()
    by_reason: dict[str, int] = {}
    for stats in per_channel.values():
        for reason, count in stats.rejected_by_reason.items():
            by_reason[reason] = by_reason.get(reason, 0) + count
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


# -- acceptor and clock sources ------------------------------------------------


def _read_acceptor_accept_errors(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    acceptor = getattr(phone, "_acceptor", None)
    if acceptor is None or not hasattr(acceptor, "stats"):
        # `LoopbackAcceptor` has no accept-retry concept and so no `stats()`
        # -- readable in production, where a real `TcpAcceptor` always sits
        # under a `PhoneLink`, and not evaluable under the loopback backend
        # the test suite drives instead.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    stats = acceptor.stats()
    by_reason = dict(stats.get("accept_errors_by_errno") or {})
    return SourceSnapshot(readable=True, by_reason=by_reason)


def _read_clock_proxied(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    adapter = getattr(phone, "adapter", None)
    if adapter is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    by_reason = dict(adapter.proxy_reasons)
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


def _read_pipeline_exception(ctx: _Context) -> SourceSnapshot:
    # Managed directly by `FailureSampler.note_pipeline_exception` -- see
    # `_read_camera_blind_ticks`.
    return SourceSnapshot(readable=True, by_reason={})


# ---------------------------------------------------------------------------
# The registry.
#
# The table in plans/plan_task38_failure_log.md enumerates thirty rows under a
# heading that says "twenty-eight". All thirty are built here: the table is
# the concrete spec and the prose count is the part that disagrees with it.
# See the implementation report for the discrepancy in full.
# ---------------------------------------------------------------------------

REGISTRY: tuple[Source, ...] = (
    Source("camera.blind_ticks", _read_camera_blind_ticks, None, "no_frame",
           "run", True, True, "jetson"),
    Source("camera.dropped_unconsumed", _read_camera_dropped_unconsumed, None, "unconsumed",
           "run", True, False, "jetson"),
    Source("camera.decode_failures", _read_camera_decode_failures, None, "jpeg_decode",
           "session", True, True, "jetson"),
    Source("camera.file_recoveries", _read_camera_file_recoveries, None, "decoder_poisoned",
           "run", True, False, "jetson"),
    Source("camera.end_of_stream", _read_camera_end_of_stream, None, "end_of_stream",
           "run", False, True, "jetson"),
    Source("camera.reader_failure", _read_camera_reader_failure, None, None,
           "session", False, True, "jetson"),
    Source("gps.not_fresh", _read_gps_not_fresh,
           frozenset({"stale", "invalid", "absent"}), None,
           "run", False, True, "jetson"),
    Source("gps.parse_errors", _read_gps_parse_errors, None, "nmea_parse",
           "run", True, False, "jetson"),
    Source("gps.ingest_errors", _read_gps_ingest_errors, None, "ingest_raised",
           "run", True, False, "jetson"),
    Source("gps.last_error", _read_gps_last_error, None, None,
           "run", False, False, "jetson"),
    Source("gps.rate_unconfigured", _read_gps_rate_unconfigured, None, "ubx_rate_config_failed",
           "run", False, False, "jetson"),
    Source("here.refused", _read_here_refused, _here_refused_reason_valid, None,
           "session", True, True, "jetson"),
    Source("here.reader_failures", _read_here_reader_failures, None, None,
           "run", True, True, "jetson"),
    Source("here.proxied_stamps", _read_here_proxied_stamps, None, "proxy",
           "session", True, False, "jetson"),
    Source("phone.dropped", _read_phone_dropped, frozenset(DROP_KEYS), None,
           "mixed", True, True, "phone"),
    Source("phone.here_errors", _read_phone_here_errors, None, "here_error",
           "session", True, True, "phone"),
    Source("link.down", _read_link_down, None, "no_session",
           "run", False, True, "jetson"),
    Source("link.session_end", _read_link_session_end, _link_session_end_valid, None,
           "run", True, True, "jetson"),
    Source("link.refusals", _read_link_refusals, None, None,
           "run", True, False, "jetson"),
    Source("link.displaced", _read_link_displaced, None, "displaced",
           "run", True, True, "jetson"),
    Source("link.workers_leaked", _read_link_workers_leaked, None, "handshake_worker_leaked",
           "run", True, False, "jetson"),
    Source("link.sends_lost", _read_link_sends_lost, frozenset({"no_session", "refused"}), None,
           "run", True, False, "jetson"),
    Source("link.supervisor_ended", _read_link_supervisor_ended, _supervisor_ended_valid, None,
           "run", False, True, "jetson"),
    Source("wire.dropped", _read_wire_dropped, frozenset({"outbound", "inbound"}), None,
           "session", True, True, "jetson"),
    Source("wire.seq_gaps", _read_wire_seq_gaps, None, "seq_gap",
           "session", True, True, "jetson"),
    Source("wire.decode_errors", _read_wire_decode_errors, frozenset(REASONS), None,
           "session", True, True, "jetson"),
    Source("wire.send_rejected", _read_wire_send_rejected, frozenset(REASONS), None,
           "session", True, False, "jetson"),
    Source("acceptor.accept_errors", _read_acceptor_accept_errors, None, None,
           "run", True, False, "jetson"),
    Source("clock.proxied", _read_clock_proxied, None, None,
           "session", True, False, "jetson"),
    Source("pipeline.exception", _read_pipeline_exception, None, None,
           "run", False, True, "jetson"),
)

_BY_NAME: dict[str, Source] = {s.name: s for s in REGISTRY}


def _dominant_reason(current: Mapping[str, int], previous: Mapping[str, int]) -> str | None:
    """Which key's delta is largest between two per-reason snapshots.

    Used only to name the reason an episode OPENS under; once open, the
    episode keeps that name for its own life (D9 already discards
    per-occurrence granularity inside one episode, and re-picking a reason
    every pass would make an episode's name flap under it).
    """
    best_key, best_delta = None, 0
    for key, value in current.items():
        delta = value - previous.get(key, 0)
        if delta > best_delta:
            best_key, best_delta = key, delta
    return best_key


def _pctl(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction: float) -> float:
        rank = fraction * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    return {"p50": at(0.50), "p95": at(0.95), "max": ordered[-1]}


class FailureSampler:
    """The 1 Hz thread that turns the registry into episodes, plus two direct
    entry points (`note_no_frame`, `note_pipeline_exception`) the tick loop
    calls for the two sources this repository detects nowhere else -- see the
    module docstring's account of the two failures the log's own honesty
    depends on.

    Shaped like `ThermalSampler`: `start()` / `stop()`, `latest(now)` for the
    per-tick block, `to_record()` for the summary, one lock guarding every
    field the tick loop and the sampler's own thread both touch.
    """

    def __init__(
        self,
        sink: Any,
        *,
        phone: Any = None,
        camera: Any = None,
        gps: Any = None,
        pipeline: Any = None,
        interval_s: float = 1.0,
        quiet_passes_to_close: int = QUIET_PASSES_TO_CLOSE,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._sink = sink
        self._phone = phone
        self._camera = camera
        self._gps = gps
        self._pipeline = pipeline
        self._interval_s = interval_s
        self._quiet_passes_to_close = quiet_passes_to_close
        self._close_after_s = quiet_passes_to_close * interval_s
        self._now = clock
        self._wall = wall_clock

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        self._state: dict[str, _SourceState] = {s.name: _SourceState() for s in REGISTRY}
        self._next_episode_id = 1
        self._passes = 0
        self._seq = 0
        self._last_pass_mono: float | None = None
        self._interval_samples: list[float] = []
        self._last_tick_counter: int | None = None
        self._backwards: dict[str, dict[str, Any]] = {}
        self._outcome_counts: dict[str, int] = {}

        # -- the two direct-notification pseudo-sources --------------------
        self._blind_ticks_total = 0
        #: True for exactly one no-frame tick that has not yet been joined by
        #: a second: the first blind tick of a possible outage, not yet worth
        #: an episode. Cleared either by a second tick arriving (which opens
        #: the episode, crediting both) or by the quiet timer in
        #: `_check_pseudo_source_quiet` deciding none is coming.
        self._blind_pending = False
        self._blind_last_mono: float | None = None
        self._pipeline_exception: str | None = None
        self._pipeline_exception_at_mono: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FailureSampler":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="failure-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            self._close_all_open(outcome=OUTCOME_OPEN_AT_END)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self.sample_once()
            except Exception:
                # As `ThermalSampler._loop`: one lost pass, not a dead thread.
                pass
            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.0, self._interval_s - elapsed))
        self._running = False

    # -- one pass ------------------------------------------------------------

    def sample_once(self) -> None:
        now = self._now()
        t_wall = self._wall()
        with self._lock:
            if self._last_pass_mono is not None:
                self._interval_samples.append(now - self._last_pass_mono)
            self._last_pass_mono = now
            self._seq += 1
            seq = self._seq

            phone = self._phone
            pass_session_id = None
            session = _phone_session(phone)
            if session is not None:
                pass_session_id = session.session_id

            ticks_seen = None
            if self._pipeline is not None:
                counter = getattr(self._pipeline, "_tick_counter", None)
                if counter is not None:
                    if self._last_tick_counter is None:
                        ticks_seen = 0
                    else:
                        ticks_seen = counter - self._last_tick_counter
                    self._last_tick_counter = counter

            ctx = _Context(
                phone=phone, camera=self._camera, gps=self._gps,
                blind_ticks_total=self._blind_ticks_total,
                pipeline_exception=self._pipeline_exception,
                pass_session_id=pass_session_id,
            )
            for source in REGISTRY:
                self._process_source(source, ctx, now, t_wall)
            self._check_pseudo_source_quiet(now, t_wall)

            self._passes += 1
            unreadable = [s.name for s in REGISTRY if not self._state[s.name].last_readable]
            open_sources = [s.name for s in REGISTRY if self._state[s.name].open_episode is not None]
            sources_n = len(REGISTRY)
            record = {
                "type": "failure_scan", "seq": seq, "t_wall": t_wall, "t_mono": now,
                "session_id": pass_session_id, "ticks_seen": ticks_seen,
                "sources_n": sources_n, "sources_readable": sources_n - len(unreadable),
                "unreadable": unreadable, "open": open_sources,
            }
        if self._sink is not None:
            self._sink.write(record)

    def _process_source(self, source: Source, ctx: _Context, now: float, t_wall: float) -> None:
        st = self._state[source.name]
        st.passes_attempted += 1
        snap = source.read(ctx)

        if snap.readable and snap.session_id is not None and ctx.pass_session_id is not None \
                and snap.session_id != ctx.pass_session_id:
            # D15: the pass straddled a rebind between this source's own read
            # and the pass's session snapshot. Not evaluable for this pass
            # rather than a delta attributed to the wrong session.
            snap = SourceSnapshot(readable=False, missing=MISSING_SESSION_MOVED)

        if not snap.readable:
            st.last_readable = False
            st.last_missing = snap.missing or MISSING_NO_SOURCE
            if st.open_episode is not None:
                self._close_episode(st, source, now, t_wall, OUTCOME_UNOBSERVABLE)
            return

        st.last_readable = True
        st.last_missing = None
        st.passes_readable += 1
        st.last_readable_t_mono = now

        if source.scope in ("session", "mixed") and snap.session_id is not None:
            if st.baseline_session_id is not None and snap.session_id != st.baseline_session_id:
                # D14: a new session for a session-scoped source. Its open
                # episode, if any, cannot say whether the old condition
                # recovered -- the source that would have told us was
                # replaced whole. Its baseline resets to the new source's
                # own zero.
                if st.open_episode is not None:
                    self._close_episode(st, source, now, t_wall, OUTCOME_UNOBSERVABLE)
                st.baseline_total = 0
                st.baseline_by_reason = {}
            st.baseline_session_id = snap.session_id

        active = False
        delta = 0
        reason: str | None = None
        if source.cumulative:
            total = sum(snap.by_reason.values())
            delta = total - st.baseline_total
            if delta < 0:
                self._backwards[source.name] = {
                    "from": st.baseline_total, "to": total, "t_mono": now,
                }
                st.baseline_total = total
                st.baseline_by_reason = dict(snap.by_reason)
                active = False
            else:
                if delta > 0:
                    active = True
                    reason = _dominant_reason(snap.by_reason, st.baseline_by_reason)
                st.baseline_total = total
                st.baseline_by_reason = dict(snap.by_reason)
        else:
            if snap.by_reason:
                active = True
                delta = 1
                reason = next(iter(snap.by_reason))

        if active and reason is not None:
            st.run_total += delta
            st.by_reason_total[reason] = st.by_reason_total.get(reason, 0) + delta
            if st.first_t_mono is None:
                st.first_t_mono = now
            st.last_t_mono = now
            st.quiet_streak = 0
            if st.open_episode is None:
                self._open_episode(st, source, now, t_wall, reason, delta, ctx.pass_session_id, snap)
            else:
                ep = st.open_episode
                ep.n += delta
                ep.last_t_mono = now
        elif st.open_episode is not None:
            st.quiet_streak += 1
            if st.quiet_streak >= self._quiet_passes_to_close:
                self._close_episode(st, source, now, t_wall, OUTCOME_RECOVERED)

    def _check_pseudo_source_quiet(self, now: float, t_wall: float) -> None:
        """The two direct-notification sources open the moment their own
        condition occurs (see `note_no_frame` / `note_pipeline_exception`),
        so this pass only ever needs to check whether one has gone quiet
        long enough to close -- the same `close_after_s` bound every other
        source uses, applied to elapsed time instead of consecutive passes,
        because nothing repolls a pseudo-source between direct calls.
        """
        st = self._state["camera.blind_ticks"]
        if self._blind_last_mono is not None and now - self._blind_last_mono >= self._close_after_s:
            if st.open_episode is not None:
                self._close_episode(st, _BY_NAME["camera.blind_ticks"], now, t_wall, OUTCOME_RECOVERED)
            # A single blind tick with nothing following it within the quiet
            # window was never worth an episode -- forgotten rather than
            # left to pair, much later, with an unrelated one.
            self._blind_pending = False

    def _open_episode(
        self, st: _SourceState, source: Source, now: float, t_wall: float,
        reason: str, first_pass_n: int, session_id: Any, snap: SourceSnapshot,
    ) -> None:
        tick_id = None
        if self._pipeline is not None:
            tick_id = getattr(self._pipeline, "_tick_counter", None)
        if self._episode_count(st) >= MAX_EPISODES_PER_SOURCE:
            st.episodes_not_kept += 1
            return
        episode = Episode(
            episode_id=self._next_episode_id, source=source.name, reason=reason,
            opened_t_mono=now, opened_t_wall=t_wall, last_t_mono=now,
            n=first_pass_n, first_pass_n=first_pass_n, session_id=session_id,
            tick_id=tick_id, basis=FAILURE_BASIS_MEASURED, bound_s=self._interval_s,
            value=snap.value, channel=snap.channel, detail=snap.detail,
        )
        self._next_episode_id += 1
        st.open_episode = episode
        if source.name == "camera.blind_ticks":
            self._blind_last_mono = now
        if source.event_records and self._sink is not None:
            self._sink.write(episode.open_record(source.device))
            st.events_written += 1

    def _episode_count(self, st: _SourceState) -> int:
        return st.episodes_closed + st.episodes_not_kept

    def _close_episode(
        self, st: _SourceState, source: Source, now: float, t_wall: float, outcome: str,
    ) -> None:
        episode = st.open_episode
        assert episode is not None
        if outcome == OUTCOME_UNOBSERVABLE:
            # The last instant this drive actually watched the source, not
            # the last occurrence and not the moment the gap was noticed --
            # see `_SourceState.last_readable_t_mono`'s own docstring.
            episode.closed_t_mono = st.last_readable_t_mono
        else:
            episode.closed_t_mono = now
        episode.closed_t_wall = t_wall
        episode.outcome = outcome
        st.open_episode = None
        st.quiet_streak = 0
        st.episodes_closed += 1
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
        if source.event_records and self._sink is not None:
            self._sink.write(episode.close_record(source.device, close_after_s=self._close_after_s))
            st.events_written += 1

    def _close_all_open(self, *, outcome: str) -> None:
        now = self._now()
        t_wall = self._wall()
        for source in REGISTRY:
            st = self._state[source.name]
            if st.open_episode is not None:
                self._close_episode(st, source, now, t_wall, outcome)

    # -- direct-notification entry points -------------------------------------

    def note_no_frame(self, *, end_of_stream: bool = False) -> None:
        """The tick loop's `frame is None` branch, which it already takes.
        Does not wait for the sampler's own pass -- see the module docstring.

        The first blind tick of a run of them opens nothing: a single missed
        poll at 5 Hz is ordinary and not worth an episode. The second is what
        confirms it is a run rather than a blip, and the episode it opens is
        credited with both -- `first_pass_n=2` -- so `n` never undercounts
        `blind_ticks` by the one tick that was, correctly, held back while
        waiting to see if a second would follow.
        """
        with self._lock:
            now = self._now()
            t_wall = self._wall()
            self._blind_ticks_total += 1
            self._blind_last_mono = now
            st = self._state["camera.blind_ticks"]
            st.passes_attempted += 1
            st.passes_readable += 1
            st.last_readable = True
            source = _BY_NAME["camera.blind_ticks"]
            if st.open_episode is not None:
                st.run_total += 1
                st.by_reason_total["no_frame"] = st.by_reason_total.get("no_frame", 0) + 1
                st.last_t_mono = now
                st.open_episode.n += 1
                st.open_episode.last_t_mono = now
            elif self._blind_pending:
                self._blind_pending = False
                st.run_total += 2
                st.by_reason_total["no_frame"] = st.by_reason_total.get("no_frame", 0) + 2
                if st.first_t_mono is None:
                    st.first_t_mono = now
                st.last_t_mono = now
                self._open_episode(
                    st, source, now, t_wall, "no_frame", 2, None,
                    SourceSnapshot(readable=True),
                )
            else:
                self._blind_pending = True

    def note_pipeline_exception(self, exc: BaseException) -> None:
        """`worker()`'s `except BaseException`. Writes one `failure_event`
        and never swallows the exception -- the caller still re-raises."""
        with self._lock:
            now = self._now()
            t_wall = self._wall()
            self._pipeline_exception = type(exc).__name__
            self._pipeline_exception_at_mono = now
            st = self._state["pipeline.exception"]
            st.passes_attempted += 1
            st.passes_readable += 1
            st.last_readable = True
            st.run_total += 1
            reason = type(exc).__name__
            st.by_reason_total[reason] = st.by_reason_total.get(reason, 0) + 1
            if st.first_t_mono is None:
                st.first_t_mono = now
            st.last_t_mono = now
            source = _BY_NAME["pipeline.exception"]
            detail, _ = _capped(str(exc))
            if st.open_episode is None:
                self._open_episode(
                    st, source, now, t_wall, reason, 1, None,
                    SourceSnapshot(readable=True, detail=detail),
                )

    # -- per tick --------------------------------------------------------------

    def latest(self, now: float | None = None) -> dict[str, Any]:
        """`record["failures"]`, written beside `record["thermal"]`."""
        now = self._now() if now is None else now
        with self._lock:
            if self._last_pass_mono is None:
                reason = FAILURE_ABSENT_SAMPLER_STOPPED if not self._running else FAILURE_ABSENT_NO_SAMPLE_YET
                return {
                    "open": None, "open_n": None, "episodes": None, "scan_age_s": None,
                    "basis": FAILURE_BASIS_ABSENT, "unreadable_n": None, "reason": reason,
                }
            age = now - self._last_pass_mono
            basis = FAILURE_BASIS_MEASURED if age <= 2 * self._interval_s else FAILURE_BASIS_STALE
            open_sources = [s.name for s in REGISTRY if self._state[s.name].open_episode is not None]
            unreadable_n = sum(1 for s in REGISTRY if not self._state[s.name].last_readable)
            episodes_total = sum(st.episodes_closed + (1 if st.open_episode else 0) for st in self._state.values())
            return {
                "open": open_sources, "open_n": len(open_sources), "episodes": episodes_total,
                "scan_age_s": round(age, 3), "basis": basis, "unreadable_n": unreadable_n,
                "reason": None,
            }

    # -- per run -----------------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """`summary["failures"]`, written once at the end of a run."""
        with self._lock:
            sources: dict[str, Any] = {}
            for source in REGISTRY:
                st = self._state[source.name]
                if st.run_total > 0:
                    status = RULE_FIRED
                elif st.passes_attempted == 0 or st.passes_attempted != st.passes_readable:
                    status = RULE_NOT_EVALUABLE
                else:
                    status = RULE_QUIET
                row: dict[str, Any] = {
                    "status": status,
                    "passes_attempted": st.passes_attempted,
                    "passes_readable": st.passes_readable,
                    "episodes": st.episodes_closed + (1 if st.open_episode else 0),
                    "total": st.run_total,
                    "by_reason": dict(st.by_reason_total),
                    "first_t_mono": st.first_t_mono,
                    "last_t_mono": st.last_t_mono,
                    "events_written": st.events_written,
                    "episodes_not_kept": st.episodes_not_kept,
                }
                if status == RULE_NOT_EVALUABLE and st.last_missing:
                    row["missing"] = [st.last_missing]
                sources[source.name] = row
            outcomes = dict(self._outcome_counts)

            interval_stats = _pctl(self._interval_samples) or {"p50": None, "p95": None, "max": None}
            basis_counts = {FAILURE_BASIS_MEASURED: self._passes, FAILURE_BASIS_STALE: 0, FAILURE_BASIS_ABSENT: 0}
            return {
                "scan": {
                    "passes": self._passes, "seq_last": self._seq,
                    "interval_s": interval_stats, "basis_counts": basis_counts,
                    "absent_reasons": {}, "sources_n": len(REGISTRY),
                },
                "sources": sources,
                "outcomes": outcomes,
                "counter_went_backwards": dict(self._backwards),
                "blind_ticks": self._blind_ticks_total,
                "pipeline_exception": self._pipeline_exception,
            }
