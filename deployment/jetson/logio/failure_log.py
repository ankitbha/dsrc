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
#: The accessor itself raised. No live trigger is known -- every accessor
#: guards the object it reads being absent or closed before touching a field
#: on it -- but an accessor added later that does not is a silent kill
#: switch without this: one raise would stop `_process_source` before it
#: reaches every source after it in the registry, freezing their `quiet`
#: reading forever rather than reporting the one that actually failed.
MISSING_ACCESSOR_RAISED = "accessor_raised"
MISSING = frozenset({
    MISSING_NO_PHONE, MISSING_NO_SESSION, MISSING_NO_TELEMETRY,
    MISSING_SESSION_MOVED, MISSING_NO_SOURCE, MISSING_ACCESSOR_RAISED,
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

#: The two sources with no counter of their own to scan: `camera.blind_ticks`
#: and `pipeline.exception` are reported entirely through direct calls
#: (`note_no_frame`, `note_frame`, `note_pipeline_exception`). Their
#: accessors always return `readable=True` with an empty `by_reason` -- there
#: is nothing for a scan pass to read -- so `_process_source` must not run
#: its movement or quiet-streak logic against them; only a direct call or
#: (for `camera.blind_ticks`) `_check_pseudo_source_quiet`'s own timer may
#: open or close their episodes.
PSEUDO_SOURCES = frozenset({"camera.blind_ticks", "pipeline.exception"})

#: How long an unbroken streak of `note_no_frame` calls must span before it
#: becomes an episode, and also the quiet bound used to close an open blind
#: episode once notifications stop arriving. Every other source closes on
#: `close_after_s`; `camera.blind_ticks` cannot, because a single failed
#: `wait_for_fresh` call can legitimately block for several seconds, longer
#: than `close_after_s` itself -- closing on that bound would end an episode
#: between two notifications that both belong to the same outage. The value
#: has margin over both the periods this sampler must be correct at (up to
#: 5 s) and the longest period observed on a real drive (about 4 s).
BLIND_EPISODE_MIN_S = 10.0

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
    #: Whether `detail` was already longer than `DETAIL_MAX_LEN` and was cut
    #: down by the accessor's own call to `_capped`. Carried alongside
    #: `detail` itself so `_open_episode` -- the one place a `SourceSnapshot`
    #: is turned into an `Episode` -- can count it on the source's row
    #: without re-running `_capped` a second time on text already cut.
    detail_truncated: bool = False


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
    #: False for an episode opened past `MAX_EPISODES_PER_SOURCE` (D10): it
    #: still runs the close logic and accumulates `n` like any other episode,
    #: but writes no `failure_event` record and does not count toward the
    #: source's own `episodes` total -- only toward `episodes_not_kept` and
    #: `suppressed`, which is what makes it "not kept" rather than "missing".
    kept: bool = True

    def open_record(self, device: str) -> dict[str, Any]:
        # `self.detail` is already capped -- the accessor that produced it, or
        # `note_pipeline_exception`, ran it through `_capped` once, and
        # `_open_episode` counted the truncation then. Re-capping already-cut
        # text here would be a second, redundant pass that could only ever be
        # a no-op.
        return {
            "type": "failure_event", "phase": "open", "episode_id": self.episode_id,
            "source": self.source, "reason": self.reason, "device": device,
            "t_wall": self.opened_t_wall, "t_mono": self.opened_t_mono,
            "basis": self.basis, "bound_s": self.bound_s,
            "session_id": self.session_id, "tick_id": self.tick_id,
            "channel": self.channel, "value": self.value,
            "first_pass_n": self.first_pass_n, "detail": self.detail,
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
    #: Occurrences credited to `run_total` that a closed, KEPT episode's own
    #: `n` accounts for. Kept separately from `episodes_closed` (a count of
    #: episodes) because the reconciliation this field exists for --
    #: `kept_total + suppressed + below_episode_threshold == run_total`, in
    #: `to_record`'s row -- needs the occurrence count, not the episode count.
    kept_n_total: int = 0
    #: Occurrences absorbed by an episode that closed with `kept=False`: past
    #: the cap (D10), the condition still happened, and this is where those
    #: occurrences are still counted after `_open_episode` stopped writing a
    #: record for them.
    suppressed_total: int = 0
    #: `camera.blind_ticks` only: no-frame ticks that `note_no_frame` credited
    #: to `run_total` immediately but whose streak resolved (a frame arrived,
    #: or the run ended) before it ran long enough to become an episode, so
    #: no episode ever enclosed them.
    below_episode_threshold: int = 0
    #: How many episodes on this source carried a `detail` longer than
    #: `DETAIL_MAX_LEN` and had it cut by `_capped`.
    truncated_details: int = 0
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
    pass_session_id: Any


def _phone_session(phone: Any) -> Any:
    # `PhoneLink.session` is set in `__init__` and never deleted -- `None`
    # until a session binds, never absent as an attribute -- so this reads it
    # directly rather than through `getattr` with a default, which would
    # accept an object with no `session` attribute at all as silently as one
    # that has not bound yet, and would hide the attribute from `ty`.
    return phone.session if phone is not None else None


def _fixed(reason_word: str, total: int) -> SourceSnapshot:
    return SourceSnapshot(readable=True, by_reason={reason_word: total} if total else {})


# -- camera sources -----------------------------------------------------------


def _read_camera_dropped_unconsumed(ctx: _Context) -> SourceSnapshot:
    # `dropped_frames` is a property on both camera backends (`CameraStream`
    # and `PhoneCameraStream` alike) -- read directly rather than through
    # `getattr` with a default, which would turn a renamed property into a
    # silent `None` instead of an `AttributeError` at the one place that
    # would notice.
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    snap = _fixed("unconsumed", int(ctx.camera.dropped_frames))
    return SourceSnapshot(readable=True, by_reason=snap.by_reason, session_id=ctx.pass_session_id)


def _read_camera_decode_failures(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None or not hasattr(ctx.camera, "decode_failures"):
        # Only `PhoneCameraStream` has this counter at all -- a local
        # `CameraStream` never decodes a JPEG, so the question does not apply.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    snap = _fixed("jpeg_decode", int(ctx.camera.decode_failures))
    return SourceSnapshot(readable=True, by_reason=snap.by_reason, session_id=ctx.pass_session_id)


def _read_camera_file_recoveries(ctx: _Context) -> SourceSnapshot:
    # `file_recoveries` is likewise present on both backends -- see
    # `_read_camera_dropped_unconsumed`.
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    return _fixed("decoder_poisoned", int(ctx.camera.file_recoveries))


def _read_camera_end_of_stream(ctx: _Context) -> SourceSnapshot:
    # Also present on both backends -- `run_demo`'s own tick loop reads
    # `camera.end_of_stream` unconditionally, regardless of which one is live.
    if ctx.camera is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    active = bool(ctx.camera.end_of_stream)
    return SourceSnapshot(readable=True, by_reason={"end_of_stream": 1} if active else {})


def _read_camera_reader_failure(ctx: _Context) -> SourceSnapshot:
    if ctx.camera is None or not hasattr(ctx.camera, "failure"):
        # Only a phone-fed source (`_PhoneSource`) has a `.failure` field --
        # a local `CameraStream` has no reader thread that can name one. This
        # `hasattr` stays: the two backends genuinely differ here, which is
        # what `dropped_frames` and `file_recoveries` above do not.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SOURCE)
    failure = ctx.camera.failure
    detail, truncated = _capped(failure)
    return SourceSnapshot(
        readable=True, session_id=ctx.pass_session_id, detail=detail,
        detail_truncated=truncated, by_reason={detail: 1} if detail else {},
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
    # `latest()` returns a `GpsFix` on every backend (`GpsReader`,
    # `PhoneGpsReader`, `SimulatedGps`) -- `t_mono` and `valid` are dataclass
    # fields with defaults, never absent, so read directly.
    fix = ctx.gps.latest()
    if fix.t_mono <= 0.0:
        reason = "absent"
    elif not fix.valid:
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
    detail, truncated = _capped(last_error or None)
    return SourceSnapshot(
        readable=True, detail=detail, detail_truncated=truncated,
        by_reason={detail: 1} if detail else {},
    )


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
    # `PhoneLink.here` is constructed in `__init__` and rebuilt whole on a
    # redial (D14) -- never absent as an attribute, so read directly.
    phone = ctx.phone
    if phone is None or phone.here is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    by_reason = dict(phone.here.refused_by_reason)
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


def _read_here_reader_failures(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    total = int(phone.here_failures)
    detail, truncated = _capped(phone.here_failure)
    key = detail if detail is not None else "here_reader_failed"
    return SourceSnapshot(
        readable=True, by_reason={key: total} if total else {}, detail=detail,
        detail_truncated=truncated,
    )


def _read_here_proxied_stamps(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None or phone.here is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    total = int(phone.here.proxied_stamps)
    return SourceSnapshot(
        readable=True, by_reason={"proxy": total} if total else {}, session_id=ctx.pass_session_id,
    )


# -- phone-telemetry sources --------------------------------------------------


def _read_phone_dropped(ctx: _Context) -> SourceSnapshot:
    # `PhoneLink.telemetry` is a property backed by a field set in `__init__`
    # -- `None` until the first report arrives, never an absent attribute --
    # so this reads it directly rather than through `getattr` with a default.
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    if _session_stats_or_none(phone) is None:
        # `PhoneLink.telemetry` is cleared on a rebind (`_rebind`), not on
        # session loss -- so while the link is down, it still holds the last
        # report received before the outage. Reading it without this gate
        # reports the phone's drop counters as current, unchanged, for the
        # whole outage, which is exactly the recovery `link.down` exists to
        # let every phone-side source refuse to claim.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    telemetry = phone.telemetry
    if telemetry is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)
    dropped = telemetry.dropped or {}
    by_reason = {k: int(v) for k, v in dropped.items() if k in DROP_KEYS}
    return SourceSnapshot(readable=True, by_reason=by_reason, session_id=ctx.pass_session_id)


def _read_phone_here_errors(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    if _session_stats_or_none(phone) is None:
        # See `_read_phone_dropped`: the same stale-snapshot hazard applies
        # to `here_errors`, read off the same `telemetry` object.
        return SourceSnapshot(readable=False, missing=MISSING_NO_SESSION)
    telemetry = phone.telemetry
    if telemetry is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_TELEMETRY)
    total = int(telemetry.here_errors)
    return SourceSnapshot(
        readable=True, by_reason={"here_error": total} if total else {}, session_id=ctx.pass_session_id,
    )


# -- link sources -------------------------------------------------------------


def _read_link_down(ctx: _Context) -> SourceSnapshot:
    phone = ctx.phone
    if phone is None:
        return SourceSnapshot(readable=False, missing=MISSING_NO_PHONE)
    # `Session.is_closed` is a property every `Session` has -- read directly.
    session = _phone_session(phone)
    down = session is None or bool(session.is_closed)
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
    if session is not None and bool(session.is_closed) and phone.supervisor_ended is not None:
        # The current, terminal session: it ended and nothing replaced it.
        # An `end_reason` here is a fact this drive is done with, so it is
        # counted once, not once per pass it stays terminal.
        reason = session.end_reason
        # `end_reason` is a `SessionEndReason | None` -- `.value` unwraps the
        # enum member to its string; a plain string (a test double that
        # skips the enum) or `None` passes through this `getattr` unchanged,
        # which is the one place here duck-typing a Union is legitimate.
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
    detail, truncated = _capped(phone.refusals[-1] if phone.refusals else None)
    key = detail if detail is not None else "refused"
    return SourceSnapshot(
        readable=True, by_reason={key: total} if total else {}, detail=detail,
        detail_truncated=truncated,
    )


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
    if session is None or bool(session.is_closed):
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
    # `PhoneLink.router` is set in `__init__` and rebuilt on a redial -- never
    # absent, so read directly rather than through `getattr`.
    router = phone.router
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
    # `PhoneLink.router` is set in `__init__` and rebuilt on a redial -- never
    # absent, so read directly rather than through `getattr`.
    router = phone.router
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
    # `PhoneLink._acceptor` is a constructor argument, always set -- read
    # directly. `hasattr(acceptor, "stats")` stays: `TcpAcceptor` and
    # `LoopbackAcceptor` genuinely differ here.
    acceptor = phone._acceptor
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
    # `PhoneLink.adapter` is constructed in `__init__` and rebuilt on a
    # redial -- never absent, so read directly.
    adapter = phone.adapter
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
# ---------------------------------------------------------------------------


def build_registry(*, camera: Any = None) -> tuple[Source, ...]:
    """The registry, built for the camera backend actually in use.

    Every row is fixed except one. `camera.dropped_unconsumed` reads
    `camera.dropped_frames`, and what that counter means depends on which
    backend produced it: a local `CameraStream` never rebinds, so its own
    `_drop_counter` only ever grows for the life of the run -- `scope="run"`.
    `PhoneCameraStream._on_rebound` resets the same field to zero on every
    redial by design, the same reset `camera.decode_failures` already
    accounts for with `scope="session"` -- so a `camera` built from a phone
    needs the same scope here, or the sampler reads a design reset as a real
    anomaly and reports `counter_went_backwards` on every redial.
    `hasattr(camera, "decode_failures")` is the same duck-typing
    `_read_camera_decode_failures` and `_read_camera_reader_failure` already
    use to tell the two backends apart.
    """
    camera_dropped_scope = "session" if hasattr(camera, "decode_failures") else "run"
    return (
        Source("camera.blind_ticks", _read_camera_blind_ticks, None, "no_frame",
               "run", True, True, "jetson"),
        Source("camera.dropped_unconsumed", _read_camera_dropped_unconsumed, None, "unconsumed",
               camera_dropped_scope, True, False, "jetson"),
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
               "run", True, True, "jetson"),
    )


#: The default registry, for callers with no camera to build it around --
#: `TestRegistryAccessorsResolve` and every other test that only needs the
#: fixed 30 rows. `FailureSampler` builds its own from the camera it was
#: actually given (see `build_registry`'s own docstring) rather than reading
#: this one.
REGISTRY: tuple[Source, ...] = build_registry()


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
    """The 1 Hz thread that turns the registry into episodes, plus the direct
    entry points (`note_no_frame`, `note_frame`, `note_pipeline_exception`)
    the tick loop calls for the two sources this repository detects nowhere
    else -- see the module docstring's account of the two failures the log's
    own honesty depends on.

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

        # Built from the camera this sampler was actually given (C2): a
        # module-level `REGISTRY` cannot know whether `camera.dropped_frames`
        # is going to reset on a redial, because that depends on which
        # backend this run constructed.
        self._registry: tuple[Source, ...] = build_registry(camera=camera)
        self._by_name: dict[str, Source] = {s.name: s for s in self._registry}
        self._state: dict[str, _SourceState] = {s.name: _SourceState() for s in self._registry}
        self._next_episode_id = 1
        self._passes = 0
        self._seq = 0
        self._last_pass_mono: float | None = None
        self._interval_samples: list[float] = []
        self._last_tick_counter: int | None = None
        #: Every occurrence of a `run`-scoped counter going backwards, per
        #: source, in the order seen -- a list rather than the last value
        #: overwriting the ones before it, so N redials of a source that
        #: should have been session-scoped leave N entries, not one.
        self._backwards: dict[str, list[dict[str, Any]]] = {}
        self._outcome_counts: dict[str, int] = {}

        # -- the two direct-notification pseudo-sources --------------------
        self._blind_ticks_total = 0
        #: The `t_mono` of the first `note_no_frame` call in the current
        #: blind streak, or `None` when the camera is not currently blind.
        #: Marks where an episode would be back-dated to if this streak goes
        #: on to become one. Cleared by `note_frame` (the streak ended in a
        #: frame), by promotion to an episode once the streak reaches
        #: `BLIND_EPISODE_MIN_S`, or by `end_of_stream` closing it out.
        self._blind_since: float | None = None
        #: How many `note_no_frame` calls belong to the streak named by
        #: `_blind_since`. Becomes the opened episode's `n` on promotion, or
        #: is credited to `below_episode_threshold` if the streak resolves
        #: (via `note_frame` or `end_of_stream`) before reaching that point.
        self._blind_count = 0
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
            st = self._state["camera.blind_ticks"]
            if self._blind_since is not None and st.open_episode is None:
                # A blind streak was under way (credited to `run_total` as it
                # happened) but the run ended before either `note_frame` or a
                # quiet pass got to decide its fate. If it had already run
                # long enough to be an episode, open it now, back-dated to
                # its first tick, so `_close_all_open` below closes it as
                # `open_at_end` like any other episode still running at
                # teardown. Otherwise it was never worth one, and its ticks
                # go to `below_episode_threshold` so `kept_total + suppressed
                # + below_episode_threshold == run_total` still balances.
                now = self._now()
                if now - self._blind_since >= BLIND_EPISODE_MIN_S:
                    self._open_episode(
                        st, self._by_name["camera.blind_ticks"], self._blind_since,
                        self._wall(), "no_frame", self._blind_count, None,
                        SourceSnapshot(readable=True),
                    )
                else:
                    st.below_episode_threshold += self._blind_count
                self._blind_since = None
                self._blind_count = 0
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
        """One pass over the registry.

        The pass-level counters (`_seq`, `_last_pass_mono`, `_passes`, the
        interval samples) commit together, after the registry loop -- not
        split, with some set before the loop and `_passes` only after it, the
        way a pass that raised partway through would leave them disagreeing
        about whether a pass had happened at all, and every source after the
        raise would keep whatever reading its last successful pass left it
        with, forever. Every accessor is guarded individually in
        `_process_source`, so this should not be reachable; the counters
        commit together anyway, as the second line of defence for a failure
        mode neither guard was proven to need.
        """
        now = self._now()
        t_wall = self._wall()
        with self._lock:
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
                pass_session_id=pass_session_id,
            )
            for source in self._registry:
                self._process_source(source, ctx, now, t_wall)
            self._check_pseudo_source_quiet(now, t_wall)

            if self._last_pass_mono is not None:
                self._interval_samples.append(now - self._last_pass_mono)
            self._last_pass_mono = now
            self._seq += 1
            seq = self._seq
            self._passes += 1
            unreadable = [s.name for s in self._registry if not self._state[s.name].last_readable]
            open_sources = [s.name for s in self._registry if self._state[s.name].open_episode is not None]
            sources_n = len(self._registry)
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
        try:
            snap = source.read(ctx)
        except Exception:
            # Every source after this one in the registry still gets its own
            # pass -- an accessor's own bug costs this one source a reading,
            # not the rest of the scan.
            snap = SourceSnapshot(readable=False, missing=MISSING_ACCESSOR_RAISED)

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
        # `last_missing` is not reset here: it names the reason the most
        # recent UNREADABLE pass gave, not the current pass. A source read
        # as `not_evaluable` at some point mid-run and readable again by
        # teardown must still report what was missing -- clearing it on
        # every readable pass left every such row with an empty `missing`.
        st.passes_readable += 1
        st.last_readable_t_mono = now

        if source.name in PSEUDO_SOURCES:
            # `camera.blind_ticks` and `pipeline.exception` report through
            # `note_no_frame` / `note_pipeline_exception`, not through a
            # counter this pass can see -- `snap.by_reason` is always empty
            # by construction. Falling through to the movement logic below
            # would read that emptiness as "quiet" every pass, and once
            # either source's episode is open (opened by a direct call),
            # this pass's own quiet streak would close it as `recovered`
            # after `quiet_passes_to_close` passes -- regardless of whether
            # direct calls kept arriving between scans. Only the dedicated
            # timer (`_check_pseudo_source_quiet`) or a direct call may
            # close these two sources' episodes.
            return

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
        #: Per-key deltas for a cumulative, multi-reason source (D5): every
        #: key that moved this pass, credited its own movement. `None` for a
        #: predicate source, where `by_reason_total` is credited by `reason`
        #: alone below, as before.
        reason_deltas: dict[str, int] | None = None
        if source.cumulative:
            total = sum(snap.by_reason.values())
            delta = total - st.baseline_total
            if delta < 0:
                # Appended, not assigned: a `run`-scoped source that should
                # have been `session`-scoped goes backwards on every redial,
                # and overwriting this entry left N redials looking like one.
                self._backwards.setdefault(source.name, []).append(
                    {"from": st.baseline_total, "to": total, "t_mono": now}
                )
                st.baseline_total = total
                st.baseline_by_reason = dict(snap.by_reason)
                active = False
            else:
                if delta > 0:
                    active = True
                    # `reason` still names the episode, and keeps naming it
                    # for the episode's whole life once open (D9) -- but the
                    # source's own `by_reason_total` is a different question,
                    # "how many times did each reason occur", and crediting
                    # only the largest-moving key silently drops any other
                    # key that moved in the same pass. `wire.decode_errors`,
                    # `phone.dropped`, `link.session_end` and
                    # `acceptor.accept_errors` are cumulative and
                    # multi-reason exactly like `clock.proxied`, so this is
                    # not a `clock.proxied`-only fix.
                    reason = _dominant_reason(snap.by_reason, st.baseline_by_reason)
                    reason_deltas = {
                        key: value - st.baseline_by_reason.get(key, 0)
                        for key, value in snap.by_reason.items()
                    }
                    reason_deltas = {k: v for k, v in reason_deltas.items() if v}
                st.baseline_total = total
                st.baseline_by_reason = dict(snap.by_reason)
        else:
            if snap.by_reason:
                active = True
                delta = 1
                reason = next(iter(snap.by_reason))

        if active and reason is not None:
            st.run_total += delta
            if reason_deltas:
                for key, value in reason_deltas.items():
                    st.by_reason_total[key] = st.by_reason_total.get(key, 0) + value
            else:
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
        """`camera.blind_ticks` opens and closes entirely off direct calls
        (`note_no_frame`, `note_frame`) rather than this pass's own scan, so
        there are two things left for a scan pass to notice between calls:

        A blind streak that has now run long enough to become an episode
        even though nothing has called `note_no_frame` again to notice --
        promoted here, back-dated to its first tick, exactly as
        `note_no_frame` would promote it on its own next call.

        An *open* episode that has gone quiet for `BLIND_EPISODE_MIN_S` with
        no further notification and no `note_frame` call. That silence does
        not mean the camera recovered -- it means this drive has stopped
        saying anything about the window, which could be a recovered camera
        or one still blind but failing slowly enough to notify less often
        than the bound. Recovery is only ever reported when `note_frame`
        observes a frame; a close driven by silence alone is `unobservable`.
        """
        st = self._state["camera.blind_ticks"]
        if st.open_episode is None and self._blind_since is not None:
            if now - self._blind_since >= BLIND_EPISODE_MIN_S:
                last_mono = self._blind_last_mono
                self._open_episode(
                    st, self._by_name["camera.blind_ticks"], self._blind_since, t_wall,
                    "no_frame", self._blind_count, None, SourceSnapshot(readable=True),
                )
                st.open_episode.last_t_mono = last_mono
                self._blind_since = None
                self._blind_count = 0
            return
        if (
            st.open_episode is not None
            and self._blind_last_mono is not None
            and now - self._blind_last_mono >= BLIND_EPISODE_MIN_S
        ):
            self._close_episode(
                st, self._by_name["camera.blind_ticks"], now, t_wall, OUTCOME_UNOBSERVABLE,
                observed_until=self._blind_last_mono,
            )

    def _open_episode(
        self, st: _SourceState, source: Source, now: float, t_wall: float,
        reason: str, first_pass_n: int, session_id: Any, snap: SourceSnapshot,
    ) -> None:
        """Start an episode -- always, even past the cap (D10).

        `_episode_count(st) >= MAX_EPISODES_PER_SOURCE` used to `return`
        before constructing anything, leaving `st.open_episode` at `None`.
        The next movement pass then found no open episode and called this
        method again, so one outage that kept moving past the cap counted as
        one `episodes_not_kept` per pass rather than once. A suppressed
        episode is still built and still assigned to `st.open_episode`, so
        the continuation branch in `_process_source` (`ep.n += delta`) runs
        for it exactly as it does for a kept one, and the cap is decided once,
        here, rather than re-decided on every pass that follows.
        """
        tick_id = None
        if self._pipeline is not None:
            tick_id = getattr(self._pipeline, "_tick_counter", None)
        if snap.detail_truncated:
            st.truncated_details += 1
        kept = self._episode_count(st) < MAX_EPISODES_PER_SOURCE
        episode = Episode(
            episode_id=self._next_episode_id, source=source.name, reason=reason,
            opened_t_mono=now, opened_t_wall=t_wall, last_t_mono=now,
            n=first_pass_n, first_pass_n=first_pass_n, session_id=session_id,
            tick_id=tick_id, basis=FAILURE_BASIS_MEASURED, bound_s=self._interval_s,
            value=snap.value, channel=snap.channel, detail=snap.detail, kept=kept,
        )
        self._next_episode_id += 1
        st.open_episode = episode
        if not kept:
            # Counted once, at the moment this episode is decided to be over
            # the cap -- not once per pass it goes on moving, which is the
            # defect this docstring names.
            st.episodes_not_kept += 1
            return
        if source.event_records and self._sink is not None:
            self._sink.write(episode.open_record(source.device))
            st.events_written += 1

    def _episode_count(self, st: _SourceState) -> int:
        return st.episodes_closed + st.episodes_not_kept

    def _close_episode(
        self, st: _SourceState, source: Source, now: float, t_wall: float, outcome: str,
        *, observed_until: float | None = None,
    ) -> None:
        episode = st.open_episode
        assert episode is not None
        if outcome == OUTCOME_UNOBSERVABLE:
            # The last instant this drive actually watched the source, not
            # the last occurrence and not the moment the gap was noticed.
            # For a regularly scanned source that instant is
            # `last_readable_t_mono` -- see its own docstring. It is wrong
            # for `camera.blind_ticks`: that pseudo-source's accessor
            # reports `readable=True` unconditionally (PSEUDO_SOURCES), so
            # `last_readable_t_mono` there tracks the most recent ordinary
            # SCAN PASS, which carries no information about the camera at
            # all, rather than the most recent evidence of blindness. A
            # caller that has that evidence (`_check_pseudo_source_quiet`,
            # via `_blind_last_mono`) passes it as `observed_until`, which
            # takes priority; a caller with none falls through to the
            # scanned-source behaviour unchanged.
            episode.closed_t_mono = observed_until if observed_until is not None else st.last_readable_t_mono
        else:
            episode.closed_t_mono = now
        episode.closed_t_wall = t_wall
        episode.outcome = outcome
        st.open_episode = None
        st.quiet_streak = 0
        if not episode.kept:
            # Its occurrences still count -- toward `suppressed`, not toward
            # a kept episode's own total -- but it never had a record and
            # never counted as one of the source's `episodes`.
            st.suppressed_total += episode.n
            return
        st.episodes_closed += 1
        st.kept_n_total += episode.n
        self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
        if source.event_records and self._sink is not None:
            # `camera.blind_ticks` closes on its own bound (`BLIND_EPISODE_MIN_S`,
            # see its docstring), not the sampler-wide `close_after_s` every
            # scanned source uses -- the record should name the bound that
            # actually governed this close, not the one that did not apply.
            close_after_s = (
                BLIND_EPISODE_MIN_S if source.name == "camera.blind_ticks" else self._close_after_s
            )
            self._sink.write(episode.close_record(source.device, close_after_s=close_after_s))
            st.events_written += 1

    def _close_all_open(self, *, outcome: str) -> None:
        now = self._now()
        t_wall = self._wall()
        for source in self._registry:
            st = self._state[source.name]
            if st.open_episode is not None:
                self._close_episode(st, source, now, t_wall, outcome)

    # -- direct-notification entry points -------------------------------------

    def note_no_frame(self, *, end_of_stream: bool = False) -> None:
        """The tick loop's `frame is None` branch, which it already takes.
        Does not wait for the sampler's own pass -- see the module docstring.

        Every call is credited to `run_total` immediately, so
        `camera.blind_ticks`' own total always equals the number of times
        this was called.

        An episode opens once an unbroken blind streak has run for
        `BLIND_EPISODE_MIN_S`, back-dated to the streak's first tick
        (`_blind_since`) -- not on a fixed tick count, because the failed-read
        period this camera produces is not fixed either. A streak that never
        reaches the threshold is not worth an episode; its ticks are counted
        in `below_episode_threshold` once the streak is known to be over
        (`note_frame` observing a frame, or `end_of_stream` here), which is
        what keeps `sum(episode.n) + suppressed + below_episode_threshold ==
        run_total` checkable.

        This call never reports a recovery. Closing an open episode as
        `recovered` happens only in `note_frame`, when a frame is actually
        observed -- the previous design closed on this method's own quiet
        timer, which reported `recovered` whenever notifications merely
        stopped arriving often enough, including while the camera was still
        blind. `end_of_stream` is the one exception: nothing will call this
        again, so an episode left open here is closed as `open_at_end`
        rather than left for a quiet timer that assumes more ticks might
        still come.

        Does not touch `passes_attempted` / `passes_readable` -- those count
        the sampler's own 1 Hz scan passes for every source alike, including
        this one (`_process_source` still runs its readable bookkeeping for
        `camera.blind_ticks`, only skipping its movement logic; see
        `PSEUDO_SOURCES`). Incrementing them here too used to inflate the
        pair with every direct call, so `passes_attempted` for this one
        source counted scans plus notifications combined and stopped
        meaning "how many scan passes watched it" the way it does for every
        other source.
        """
        with self._lock:
            now = self._now()
            t_wall = self._wall()
            self._blind_ticks_total += 1
            self._blind_last_mono = now
            st = self._state["camera.blind_ticks"]
            source = self._by_name["camera.blind_ticks"]

            st.run_total += 1
            st.by_reason_total["no_frame"] = st.by_reason_total.get("no_frame", 0) + 1
            if st.first_t_mono is None:
                st.first_t_mono = now
            st.last_t_mono = now

            if st.open_episode is not None:
                st.open_episode.n += 1
                st.open_episode.last_t_mono = now
            else:
                if self._blind_since is None:
                    self._blind_since = now
                    self._blind_count = 0
                self._blind_count += 1
                if now - self._blind_since >= BLIND_EPISODE_MIN_S:
                    self._open_episode(
                        st, source, self._blind_since, t_wall, "no_frame",
                        self._blind_count, None, SourceSnapshot(readable=True),
                    )
                    st.open_episode.last_t_mono = now
                    self._blind_since = None
                    self._blind_count = 0
                elif end_of_stream:
                    st.below_episode_threshold += self._blind_count
                    self._blind_since = None
                    self._blind_count = 0

            if end_of_stream and st.open_episode is not None:
                self._close_episode(st, source, now, t_wall, OUTCOME_OPEN_AT_END)

    def note_frame(self) -> None:
        """The tick loop's success path, called once a frame actually
        arrives -- the observation the old design had no way to make, and
        the only evidence this sampler ever has that the camera recovered.

        Closes an open blind episode as `recovered`, back-dated to
        `_blind_last_mono` (the last notified tick), because that is the
        last instant the outage is known to have still been going; the time
        between that tick and this frame arriving is dead time neither
        blind nor demonstrably working. A streak that had not yet reached
        `BLIND_EPISODE_MIN_S` is resolved into `below_episode_threshold`
        instead, the same way `end_of_stream` resolves one in `note_no_frame`.
        Either way the streak is over: `_blind_since` and `_blind_count`
        are cleared unconditionally.
        """
        with self._lock:
            st = self._state["camera.blind_ticks"]
            source = self._by_name["camera.blind_ticks"]
            if st.open_episode is not None:
                close_mono = self._blind_last_mono if self._blind_last_mono is not None else self._now()
                self._close_episode(st, source, close_mono, self._wall(), OUTCOME_RECOVERED)
            elif self._blind_since is not None:
                st.below_episode_threshold += self._blind_count
            self._blind_since = None
            self._blind_count = 0

    def note_pipeline_exception(self, exc: BaseException) -> None:
        """`worker()`'s `except BaseException`. Writes one `failure_event`
        and never swallows the exception -- the caller still re-raises.

        Leaves `passes_attempted` / `passes_readable` to the scan pass, for
        the same reason `note_no_frame` does (see its own docstring):
        `pipeline.exception` is the other direct-notification pseudo-source,
        and double-counting here would cost it the same meaning loss D7
        named for `camera.blind_ticks`.
        """
        with self._lock:
            now = self._now()
            t_wall = self._wall()
            self._pipeline_exception = type(exc).__name__
            self._pipeline_exception_at_mono = now
            st = self._state["pipeline.exception"]
            st.run_total += 1
            reason = type(exc).__name__
            st.by_reason_total[reason] = st.by_reason_total.get(reason, 0) + 1
            if st.first_t_mono is None:
                st.first_t_mono = now
            st.last_t_mono = now
            source = self._by_name["pipeline.exception"]
            detail, truncated = _capped(str(exc))
            if st.open_episode is None:
                self._open_episode(
                    st, source, now, t_wall, reason, 1, None,
                    SourceSnapshot(readable=True, detail=detail, detail_truncated=truncated),
                )
            else:
                st.open_episode.n += 1
                st.open_episode.last_t_mono = now

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
            open_sources = [s.name for s in self._registry if self._state[s.name].open_episode is not None]
            unreadable_n = sum(1 for s in self._registry if not self._state[s.name].last_readable)
            episodes_total = sum(
                st.episodes_closed + (1 if st.open_episode is not None and st.open_episode.kept else 0)
                for st in self._state.values()
            )
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
            for source in self._registry:
                st = self._state[source.name]
                if st.run_total > 0:
                    status = RULE_FIRED
                elif st.passes_attempted == 0 or st.passes_attempted != st.passes_readable:
                    status = RULE_NOT_EVALUABLE
                else:
                    status = RULE_QUIET
                open_ep = st.open_episode
                open_kept_n = open_ep.n if (open_ep is not None and open_ep.kept) else 0
                open_suppressed_n = open_ep.n if (open_ep is not None and not open_ep.kept) else 0
                row: dict[str, Any] = {
                    "status": status,
                    "passes_attempted": st.passes_attempted,
                    "passes_readable": st.passes_readable,
                    "episodes": st.episodes_closed + (1 if open_ep is not None and open_ep.kept else 0),
                    "total": st.run_total,
                    # A cumulative source's `total` is occurrences: a real
                    # counter moved that many times. A predicate source's
                    # `total` is passes: `delta = 1` once per pass the
                    # condition holds, so a sticky error active for the whole
                    # drive reports one "occurrence" per second it stayed
                    # true. Carried so a reader (`report.md`) can name the
                    # quantity for what it is instead of calling both
                    # "occurrences".
                    "cumulative": source.cumulative,
                    "by_reason": dict(st.by_reason_total),
                    "first_t_mono": st.first_t_mono,
                    "last_t_mono": st.last_t_mono,
                    "events_written": st.events_written,
                    "episodes_not_kept": st.episodes_not_kept,
                    # The reading rule this row makes checkable: `kept_total +
                    # suppressed + below_episode_threshold == total`. Every
                    # occurrence `total` counts lands in exactly one of the
                    # three -- inside a kept episode's own `n`, inside a
                    # suppressed (over-the-cap) episode's `n`, or, for
                    # `camera.blind_ticks` only, never enclosed in an episode
                    # at all.
                    "kept_total": st.kept_n_total + open_kept_n,
                    "suppressed": st.suppressed_total + open_suppressed_n,
                    "below_episode_threshold": st.below_episode_threshold,
                    "truncated_details": st.truncated_details,
                }
                if status == RULE_NOT_EVALUABLE and st.last_missing:
                    row["missing"] = [st.last_missing]
                sources[source.name] = row
            outcomes = dict(self._outcome_counts)

            interval_stats = _pctl(self._interval_samples) or {"p50": None, "p95": None, "max": None}
            return {
                "scan": {
                    "passes": self._passes, "seq_last": self._seq,
                    "interval_s": interval_stats, "sources_n": len(self._registry),
                },
                "sources": sources,
                "outcomes": outcomes,
                "counter_went_backwards": dict(self._backwards),
                "blind_ticks": self._blind_ticks_total,
                "pipeline_exception": self._pipeline_exception,
            }
