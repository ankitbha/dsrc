"""The Jetson's own temperature, and both devices' throttle events.

The Jetson has no thermal reading anywhere in this repository before this module:
nothing reads `/sys/class/thermal`, runs `tegrastats`, or otherwise measures the
Orin. `JetsonThermal` reads the zones directly, the same kind of file the phone's
`ThermalZones.kt` already reads, so the two devices' temperatures are the same
kind of number. `ThermalSampler` runs that reader on its own 1 Hz thread, beside
the tick loop rather than on it: the tick loop `continue`s without a record
whenever the camera yields no frame, and a thermal log tied to that path would go
silent exactly when a hot device stops delivering frames.

Two records, because thermal is two different problems. A temperature is task
33's stage-timing question -- "read now", "read a moment ago, here is how long
ago", or "not readable, and here is why" -- so `ThermalReading` reuses two of
`time_sync.StageTiming`'s three basis words outright. A throttle event is task
34's rule-attribution question -- "it fired", "it was watched and did not fire",
or "it could not be watched at all" -- so the event and summary blocks import
`RULE_FIRED` / `RULE_QUIET` / `RULE_NOT_EVALUABLE` from the controller rather
than restating them. `count == 0` on both `quiet` and `not_evaluable`, which is
exactly why the status word carries the distinction and not the count.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET
from sensors.time_sync import STAGE_BASIS_ABSENT, STAGE_BASIS_MEASURED

# Re-exported, not retyped: a temperature and a stage duration are the same
# "measured / displaced in time / absent, and never a zero" discipline, and a
# second copy of these two strings is exactly the drift task 36 gave the
# vocabulary one home to prevent.
THERMAL_BASIS_MEASURED = STAGE_BASIS_MEASURED
THERMAL_BASIS_ABSENT = STAGE_BASIS_ABSENT
#: Displaced in time rather than across a clock, so the bound it carries is
#: `age_s` rather than `time_sync`'s `bound_ms`.
THERMAL_BASIS_STALE = "stale"
THERMAL_BASES = frozenset({THERMAL_BASIS_MEASURED, THERMAL_BASIS_STALE, THERMAL_BASIS_ABSENT})

#: Why a Jetson temperature is absent. Closed: every arm of `JetsonThermal` and
#: `ThermalSampler` that produces no reading returns one of these six.
ABSENT_NO_THERMAL_ROOT = "no_thermal_root"     # the directory would not list
ABSENT_NO_ZONE_READABLE = "no_zone_readable"   # listed, no plausible temp in any zone
ABSENT_NO_SAMPLE_YET = "no_sample_yet"         # running, first pass not finished
ABSENT_SAMPLER_STOPPED = "sampler_stopped"     # disabled in config, never started, or told to stop
#: The zone this sampler is holding stopped appearing in an otherwise-readable
#: census -- distinct from `ABSENT_NO_ZONE_READABLE`, which means nothing at all
#: was plausible this pass. Nine other zones reading fine while the held one
#: drops out is not "no plausible temp in any zone".
ABSENT_ZONE_DISAPPEARED = "zone_disappeared"
#: One pass's own read raised an unexpected error -- distinct from every
#: other reason above, all of which are ordinary "nothing to read" outcomes
#: rather than a raise. This pass is recorded absent for this reason and the
#: sampler keeps running; it names a single bad pass, never a sampler that
#: has stopped.
ABSENT_READ_ERROR = "read_error"
ABSENT_REASONS = frozenset({
    ABSENT_NO_THERMAL_ROOT, ABSENT_NO_ZONE_READABLE,
    ABSENT_NO_SAMPLE_YET, ABSENT_SAMPLER_STOPPED, ABSENT_ZONE_DISAPPEARED,
    ABSENT_READ_ERROR,
})

#: Why the phone's thermal fields are absent, beyond the phone's own per-field
#: reasons (`headroom_absent` / `skin_temp_absent`, carried on the wire). Spelled
#: like `sensing_loop.reference_from`'s own `"no_telemetry"` because it answers
#: the identical question: nothing has arrived from the phone at all.
ABSENT_NO_TELEMETRY = "no_telemetry"

#: The phone reported no headroom and gave no reason for it -- not one of the
#: wire's three closed `thermal_headroom_absent` values. Either an older build
#: that predates the reason field but still predates headroom itself in some
#: other way, or a bug in a build that should have named a reason. Counting
#: nothing here would make "always answered" and "never answered and never
#: said why" the same empty dict.
HEADROOM_ABSENT_UNSPECIFIED = "unspecified"

#: What `events.<device>` can be missing, when its status is `not_evaluable`.
#: Closed: three different absences, and the record never merges them.
MISSING_COOLING_STATE = "cooling_device_cur_state"
MISSING_TELEMETRY = "telemetry"
MISSING_STATUS_CHANGES = "thermal_status_changes"

#: The same band `ThermalZones.kt` applies on the phone, and for the same
#: reason: a zone reports things that are not temperatures at all.
MIN_PLAUSIBLE_C = -40.0
MAX_PLAUSIBLE_C = 125.0

#: Zone `type` names to prefer, best first -- a guess at the Orin's own names
#: (open item 1), with `JetsonThermal`'s fallback to the hottest readable zone
#: existing precisely so a wrong guess degrades to something rather than to
#: nothing.
PREFERRED_ZONE_TYPES = (
    "tj-thermal", "cpu-thermal", "CPU-therm",
    "gpu-thermal", "GPU-therm", "soc0-thermal",
)


def _read_trimmed(path: Path) -> str | None:
    """A sysfs file's stripped contents, or None for anything that stops it:
    missing, unreadable, would-block, or empty. Never raises -- a permission
    denial or a would-block on one zone must not take the whole sample down.

    `TypeError` is caught alongside `OSError` because a read that would block
    surfaces through the buffered text layer as `None` rather than as an
    ordinary `OSError`, and stripping that `None` raises `TypeError` instead
    -- measured on three of this Orin's nine zones, on every pass. Reading
    through `os.open`/`os.read` directly would turn the same would-block into
    an ordinary `OSError` like any other zone's failure, but that path has
    not been exercised on the hardware, so this catches the failure actually
    seen.
    """
    try:
        text = path.read_text().strip()
    except (OSError, TypeError):
        return None
    return text or None


def _safe_call(fn: Any, default: Any) -> Any:
    """Runs `fn()`, returning `default` for any exception it raises instead of
    propagating it. Isolates one device's own read from the rest of the pass:
    a Jetson-side sysfs quirk raising here must not stop the phone's own
    processing later in the same pass, and must not reach `_loop` and take
    the sampler thread down with it.
    """
    try:
        return fn()
    except Exception:
        return default


@dataclass(frozen=True)
class ThermalReading:
    """One device's temperature at one instant, and how much the number is worth.

    `celsius` is None if and only if `basis == "absent"` -- a temperature this
    reader could not measure is never reported as a zero. `age_s` is how long
    ago the underlying sample was actually taken, present whenever there is a
    number to be old (`measured` and `stale` alike), null exactly when there is
    no number at all.
    """

    celsius: float | None
    zone: str | None
    basis: str
    age_s: float | None
    reason: str | None
    zones_n: int

    def to_record(self) -> dict[str, Any]:
        return {
            "temp_c": self.celsius,
            "zone": self.zone,
            "basis": self.basis,
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "reason": self.reason,
            "zones_n": self.zones_n,
        }


class JetsonThermal:
    """Reads `/sys/class/thermal` directly: ordinary files, no service, no
    subprocess, and injectable so every branch below is reachable against a
    fixture directory with no Orin attached.

    Holds no state of its own -- `ThermalSampler` is what remembers a chosen
    zone across passes -- so a fresh instance always re-examines the root.
    """

    def __init__(self, root: Path | str = "/sys/class/thermal") -> None:
        self.root = Path(root)

    def read_zones(self) -> tuple[dict[str, float], str | None]:
        """Every zone under root with a `type` and a currently plausible
        `temp`, keyed by type name. `(census, reason)`: the census is empty if
        and only if `reason` is not None, so a partial read is never mistaken
        for a complete one with nothing to report.
        """
        try:
            if not self.root.is_dir():
                return {}, ABSENT_NO_THERMAL_ROOT
            entries = sorted(self.root.glob("thermal_zone*"))
        except OSError:
            return {}, ABSENT_NO_THERMAL_ROOT

        census: dict[str, float] = {}
        for entry in entries:
            name = _read_trimmed(entry / "type")
            if name is None or name in census:
                continue
            raw = _read_trimmed(entry / "temp")
            celsius = None if raw is None else self.celsius_of(raw)
            if celsius is None:
                continue
            census[name] = celsius
        if not census:
            return {}, ABSENT_NO_ZONE_READABLE
        return census, None

    def read_cooling(self) -> tuple[dict[str, int], tuple[str, ...]]:
        """Every `cooling_device*`'s current state, keyed by its `type`, and
        the names of devices whose `type` listed but whose `cur_state` did
        not. `(states, missing)`: a device that listed but would not give a
        state is not the same as a directory with nothing in it at all -- one
        readable device among two does not mean the census is complete, and a
        caller that only asked "did anything read" could not tell the two
        apart.

        A device whose own `type` will not read is entered into `missing`
        keyed by its directory name, the same as one whose `cur_state` will
        not: both listed and neither gave a usable reading, so both count as
        attempted. Without this, such a device fell into neither `states` nor
        `missing`, which a caller cannot tell apart from a root with no
        `cooling_device*` entries at all.

        Deduplication against a repeated `type` name only ever skips a device
        that already has a state in `states` -- not one recorded in `missing`
        -- so a device sharing a name with one that failed to read still gets
        its own attempt.
        """
        try:
            if not self.root.is_dir():
                return {}, ()
            entries = sorted(self.root.glob("cooling_device*"))
        except OSError:
            return {}, ()

        states: dict[str, int] = {}
        missing: list[str] = []
        for entry in entries:
            name = _read_trimmed(entry / "type")
            if name is None:
                missing.append(entry.name)
                continue
            if name in states:
                continue
            raw = _read_trimmed(entry / "cur_state")
            if raw is None:
                missing.append(name)
                continue
            try:
                states[name] = int(raw)
            except ValueError:
                missing.append(name)
                continue
        return states, tuple(missing)

    @staticmethod
    def celsius_of(raw: str) -> float | None:
        """Interpret one `temp` file's contents: millidegrees or whole degrees
        separated by magnitude, then refused if it falls outside a band no
        phone or Jetson can actually occupy. Ported from `ThermalZones.kt`'s
        `celsiusOf` so both devices read a `temp` file the same way.
        """
        try:
            number = int(raw)
        except ValueError:
            return None
        celsius = number / 1000.0 if abs(number) >= 1000 else float(number)
        if celsius < MIN_PLAUSIBLE_C or celsius > MAX_PLAUSIBLE_C:
            return None
        return celsius


def _tick_event_record(
    status: str, count: int, last: dict[str, Any] | None, missing: tuple[str, ...] = (),
    **extra: Any,
) -> dict[str, Any]:
    """One `events.<device>` entry for the per-tick block. `missing` is present
    only on a `not_evaluable` entry, the way `RuleCheck.to_record` emits it.
    `extra` carries fields specific to one device's `not_evaluable` reading,
    such as the cooling census's own pass counts -- present only when given,
    so a device with nothing further to say does not grow new keys.
    """
    record: dict[str, Any] = {"status": status, "count": count, "last": last}
    if missing:
        record["missing"] = list(missing)
    record.update(extra)
    return record


def _summary_event_record(
    status: str, count: int, missing: tuple[str, ...], by_unit: dict[str, int] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """As `_tick_event_record`, but for the drive-level rollup, where `missing`
    is always present (possibly empty) because the summary is read on its own,
    with no sibling ticks to compare it against.
    """
    record: dict[str, Any] = {"status": status, "count": count, "missing": list(missing)}
    if by_unit is not None:
        record["by_unit"] = dict(by_unit)
    record.update(extra)
    return record


def _pctl(values: list[float]) -> dict[str, float]:
    """min/mean/p50/p95/max, linearly interpolating between the two nearest
    ranks -- the convention `numpy.percentile`'s default computes, and the one
    `eval_run.pctl` uses. The two must agree: this module's own p50/p95 and
    `eval_run`'s sit in the same `## Thermal` section, and a nearest-rank
    number next to a linearly interpolated one would read as one convention
    when it is two.
    """
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction: float) -> float:
        # `n == 1` needs no special case: `rank` is then always 0, `lo` and
        # `hi` both 0, and the interpolation term is zero -- the general
        # formula already returns `ordered[0]`.
        rank = fraction * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    return {
        "min": ordered[0], "mean": sum(ordered) / n,
        "p50": at(0.50), "p95": at(0.95), "max": ordered[-1],
    }


class ThermalSampler:
    """The 1 Hz thread: a Jetson temperature series, a Jetson throttle-event
    stream, and a relay of the phone's own thermal telemetry and status-change
    count -- independent of the tick loop, so a hot, silent camera does not
    also silence this log.

    `phone` is read fresh on every pass rather than captured once, because
    `PhoneLink.telemetry` can be reassigned (a redial) or go stale underneath
    a long-lived sampler.
    """

    def __init__(
        self,
        sink: Any,
        phone: Any = None,
        interval_s: float = 1.0,
        clock: Any = time.monotonic,
        jetson: JetsonThermal | None = None,
    ) -> None:
        self._sink = sink
        self._phone = phone
        self._interval_s = interval_s
        self._now = clock
        self._jetson = jetson if jetson is not None else JetsonThermal()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        #: Guards every field below. `sample_once` runs on this sampler's own
        #: thread; `latest()` and `to_record()` are called from the tick loop
        #: and the shutdown path respectively -- genuine cross-thread reads of
        #: mutable state, the same shape `GpsReader` already guards.
        self._lock = threading.Lock()

        # -- the held zone (D11) and the last successful reading -------------
        self._selected_zone: str | None = None
        self._selected_by: str | None = None
        self._last_census: dict[str, float] = {}
        self._last_sample_mono: float | None = None
        self._last_selected_celsius: float | None = None
        self._last_zones_reason: str | None = None

        # -- jetson cooling / throttle events ---------------------------------
        self._prev_cooling: dict[str, int] | None = None
        #: Passes where a `cooling_device*` entry existed to attempt (a pass
        #: with no `cooling_device*` directory at all attempts nothing, and
        #: does not count here) versus passes where every entry attempted also
        #: read. `quiet` is only reported when the two are equal -- otherwise
        #: some device was listed but never gave a `cur_state` on at least one
        #: pass, which is `not_evaluable`, not "readable throughout".
        self._cooling_passes_attempted = 0
        self._cooling_passes_readable = 0
        self._jetson_seq = 0
        self._jetson_event_count = 0
        self._jetson_last_event: dict[str, Any] | None = None
        self._jetson_events_by_unit: dict[str, int] = {}

        # -- phone status-change events ---------------------------------------
        self._prev_status_changes: int | None = None
        self._phone_seq = 0
        self._phone_event_count = 0
        self._phone_last_event: dict[str, Any] | None = None
        self._phone_status_changes_ever_seen = False
        #: Rises counted with neither `thermal_change_from` nor `_to` carried
        #: -- reachable from a torn read on the phone (D9's own transition
        #: fields sit behind a separate lock acquisition from the count that
        #: moved with them) as well as an older build. Counted apart from
        #: `_phone_event_count` rather than subtracted from it, so the total
        #: stays the number of real transitions either way.
        self._phone_count_without_descriptors = 0
        #: Passes where the rise was more than one, so only the most recent
        #: transition could be named and the ones between it and the last
        #: report have no `thermal_event` line of their own (D9).
        self._phone_gap_events = 0

        # -- summary accumulators ----------------------------------------------
        self._samples = 0
        self._selected_temp_samples: list[float] = []
        self._zones_seen: set[str] = set()
        self._per_zone_max: dict[str, float] = {}
        self._cooling_devices: set[str] = set()
        #: `stale` is a `ThermalReading.basis` value `_latest_jetson` assigns
        #: only when its own caller asks for a reading, never while a sample
        #: is being taken -- there is no pass here that could ever set it.
        self._basis_counts: dict[str, int] = {THERMAL_BASIS_MEASURED: 0, THERMAL_BASIS_ABSENT: 0}
        self._absent_reasons: dict[str, int] = {}

        self._phone_samples = 0
        self._phone_status_counts: dict[str, int] = {}
        self._phone_skin_temps: list[float] = []
        self._phone_skin_zone: str | None = None
        self._phone_headroom_absent_counts: dict[str, int] = {}
        self._phone_skin_absent_counts: dict[str, int] = {}
        self._phone_absent_counts: dict[str, int] = {}

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "ThermalSampler":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="thermal-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            pass_started = time.monotonic()
            try:
                self.sample_once()
            except Exception:
                # Either device's own read is already isolated inside
                # `sample_once`, so reaching here means something else failed
                # -- the sink, say. Either way, this one pass is skipped and
                # the loop keeps running at the same pace, exactly as if it
                # had not been attempted: `sampler_stopped` is reserved for a
                # sampler that was actually asked to stop or never started,
                # not for one that lost a pass.
                pass
            elapsed = time.monotonic() - pass_started
            self._stop_event.wait(max(0.0, self._interval_s - elapsed))
        self._running = False

    # -- one pass, on the thread or driven directly by a test -------------------

    def sample_once(self) -> None:
        now = self._now()
        # Each device's own read is isolated from the other: a Jetson-side
        # sysfs quirk raising here must not stop the phone-side processing
        # below in the same pass, and the failed read is recorded as this
        # pass's own absence rather than propagated.
        census, zones_reason = _safe_call(self._jetson.read_zones, ({}, ABSENT_READ_ERROR))
        cooling, cooling_missing = _safe_call(self._jetson.read_cooling, ({}, ()))
        telemetry = getattr(self._phone, "telemetry", None) if self._phone is not None else None
        telemetry_at = getattr(self._phone, "telemetry_at_mono", None) if self._phone is not None else None

        with self._lock:
            jetson_basis, jetson_reason = self._advance_zone_reading(census, zones_reason, now)
            self._process_cooling(cooling, cooling_missing, census, now)
            self._process_phone_events(telemetry, now)
            self._accumulate_phone_summary(telemetry, telemetry_at, now)
            self._samples += 1

        self._write_sample(now, census, cooling, jetson_basis, jetson_reason, telemetry, telemetry_at)

    def _advance_zone_reading(
        self, census: dict[str, float], zones_reason: str | None, now: float,
    ) -> tuple[str, str | None]:
        self._zones_seen.update(census)
        for name, value in census.items():
            self._per_zone_max[name] = max(self._per_zone_max.get(name, value), value)

        if zones_reason is not None:
            self._last_zones_reason = zones_reason
            self._basis_counts[THERMAL_BASIS_ABSENT] += 1
            self._absent_reasons[zones_reason] = self._absent_reasons.get(zones_reason, 0) + 1
            return THERMAL_BASIS_ABSENT, zones_reason

        if self._selected_zone is None:
            self._select_zone(census)
        value = census.get(self._selected_zone) if self._selected_zone else None
        if value is None:
            # Zones were readable this pass -- census is non-empty, or `reason`
            # above would already have returned -- but the held zone specifically
            # was not among them. Other zones reading fine rules out "no
            # plausible temp in any zone"; this is the held zone dropping out of
            # an otherwise-readable census, which is a different fact.
            self._last_zones_reason = ABSENT_ZONE_DISAPPEARED
            self._basis_counts[THERMAL_BASIS_ABSENT] += 1
            self._absent_reasons[ABSENT_ZONE_DISAPPEARED] = (
                self._absent_reasons.get(ABSENT_ZONE_DISAPPEARED, 0) + 1
            )
            return THERMAL_BASIS_ABSENT, ABSENT_ZONE_DISAPPEARED

        self._last_census = census
        self._last_sample_mono = now
        self._last_selected_celsius = value
        self._selected_temp_samples.append(value)
        self._basis_counts[THERMAL_BASIS_MEASURED] += 1
        return THERMAL_BASIS_MEASURED, None

    def _select_zone(self, census: dict[str, float]) -> None:
        for name in PREFERRED_ZONE_TYPES:
            if name in census:
                self._selected_zone = name
                self._selected_by = "preferred_name"
                return
        if census:
            self._selected_zone = max(census, key=census.get)
            self._selected_by = "hottest_at_first_sample"

    def _process_cooling(
        self, states: dict[str, int], missing: tuple[str, ...], census: dict[str, float], now: float,
    ) -> None:
        """`states` is whatever `cooling_device*` entries actually gave a
        `cur_state` this pass; `missing` is every entry that listed a `type`
        but did not. A pass with no `cooling_device*` directory at all
        attempts nothing and is not counted either way -- there is no device
        to have failed. A pass that attempted at least one device counts
        toward `_cooling_passes_attempted`, and toward `_cooling_passes_readable`
        only when nothing on it is missing: one device reading fine does not
        make the census complete while another next to it never answers.
        """
        attempted = set(states) | set(missing)
        if not attempted:
            return
        self._cooling_passes_attempted += 1
        self._cooling_devices.update(states)
        if not missing:
            self._cooling_passes_readable += 1
        if self._prev_cooling is not None:
            for name, value in states.items():
                previous = self._prev_cooling.get(name)
                if previous is not None and previous != value:
                    self._jetson_event_count += 1
                    self._jetson_events_by_unit[name] = self._jetson_events_by_unit.get(name, 0) + 1
                    self._jetson_last_event = {"at_mono": now, "unit": name, "from": previous, "to": value}
                    self._jetson_seq += 1
                    self._write_event(
                        device="jetson", seq=self._jetson_seq, now=now,
                        clock="jetson", at_ns=int(round(now * 1e9)),
                        source="cooling_device", unit=name, from_=previous, to=value,
                        temp_c=self._last_selected_celsius, zones=dict(census),
                    )
        # Merged rather than replaced: a device missing on this one pass keeps
        # the value it last gave, so it is still there to diff against once it
        # starts reading again, instead of silently forgetting its history.
        self._prev_cooling = {**(self._prev_cooling or {}), **states}

    def _cooling_fully_readable(self) -> bool:
        """Whether every `cooling_device*` attempted has read on every pass
        attempted -- `quiet` is reported only here. Reading on just one pass,
        or on most of them, is `not_evaluable`: the plan's own three-word
        vocabulary has no fourth word for "readable most of the time", and
        rounding that up to `quiet` is exactly the defect this method exists
        to close.
        """
        return (
            self._cooling_passes_attempted > 0
            and self._cooling_passes_attempted == self._cooling_passes_readable
        )

    def _process_phone_events(self, telemetry: Any, now: float) -> None:
        """Counts the phone's own `thermal_status_changes` as a delta against
        the last report, never as a copy of it. `PhoneLink._rebind` sets
        `_telemetry` to `None` on every redial, and the handset that answers
        next starts its own counter from 0 -- comparing that fresh count
        directly against the departed handset's last value would read as a
        drop back to zero, erasing every transition the previous handset
        actually reported. Clearing the baseline on an absent report means
        the next one, from whichever phone sends it, starts a new comparison
        instead of diffing against a counter that belonged to a phone that is
        gone.
        """
        if telemetry is None:
            self._prev_status_changes = None
            return
        changes = getattr(telemetry, "thermal_status_changes", None)
        if changes is None:
            self._prev_status_changes = None
            return
        self._phone_status_changes_ever_seen = True
        if self._prev_status_changes is not None:
            delta = changes - self._prev_status_changes
            if delta > 0:
                frm = getattr(telemetry, "thermal_change_from", None)
                to = getattr(telemetry, "thermal_change_to", None)
                # A rise is real evidence a transition happened even when
                # neither endpoint survived the report: on the phone,
                # `changesCount` and `lastTransition` are two separate reads
                # of the watcher's lock with the status poll between them, so
                # the very first transition of a service run can land with
                # the count already at 1 and the transition still null.
                # Discarding the rise here instead of naming it absent would
                # lose it for good -- the baseline below still advances past
                # it, so no later report ever sees the gap again.
                at_ns = getattr(telemetry, "thermal_change_at_mono_ns", None)
                self._phone_event_count += delta
                self._phone_last_event = {"at_mono": now, "from": frm, "to": to}
                self._phone_seq += 1
                if frm is None and to is None:
                    self._phone_count_without_descriptors += delta
                if delta > 1:
                    self._phone_gap_events += 1
                self._write_event(
                    device="phone", seq=self._phone_seq, now=now,
                    clock="phone", at_ns=at_ns,
                    source="thermal_status", unit=None, from_=frm, to=to,
                    temp_c=getattr(telemetry, "skin_temp_c", None), zones=None,
                )
        self._prev_status_changes = changes

    def _accumulate_phone_summary(self, telemetry: Any, telemetry_at: float | None, now: float) -> None:
        if telemetry is None:
            self._phone_absent_counts[ABSENT_NO_TELEMETRY] = (
                self._phone_absent_counts.get(ABSENT_NO_TELEMETRY, 0) + 1
            )
            return
        self._phone_samples += 1
        status = getattr(telemetry, "thermal_status", None)
        if status is not None:
            self._phone_status_counts[status] = self._phone_status_counts.get(status, 0) + 1
        skin = getattr(telemetry, "skin_temp_c", None)
        if skin is not None:
            self._phone_skin_temps.append(skin)
            self._phone_skin_zone = getattr(telemetry, "skin_temp_zone", None) or self._phone_skin_zone
        headroom = getattr(telemetry, "thermal_headroom", None)
        if headroom is None:
            # A null headroom with no stated reason is still a null headroom --
            # counting it nowhere would make "always answered" and "never
            # answered and never said why" the same empty dict.
            reason = getattr(telemetry, "thermal_headroom_absent", None) or HEADROOM_ABSENT_UNSPECIFIED
            self._phone_headroom_absent_counts[reason] = self._phone_headroom_absent_counts.get(reason, 0) + 1
        skin_absent = getattr(telemetry, "skin_temp_absent", None)
        if skin_absent is not None:
            self._phone_skin_absent_counts[skin_absent] = self._phone_skin_absent_counts.get(skin_absent, 0) + 1

    def _write_event(
        self, *, device: str, seq: int, now: float, clock: str, at_ns: int | None,
        source: str, unit: str | None, from_: Any, to: Any, temp_c: float | None,
        zones: dict[str, float] | None,
    ) -> None:
        if self._sink is None:
            return
        self._sink.write({
            "type": "thermal_event", "device": device, "seq": seq,
            "t_wall": time.time(), "t_mono": now,
            "clock": clock, "at_ns": at_ns,
            "source": source, "unit": unit,
            "from": from_, "to": to, "temp_c": temp_c,
            "zones": zones,
        })

    def _write_sample(
        self, now: float, census: dict[str, float], cooling: dict[str, int],
        jetson_basis: str, jetson_reason: str | None, telemetry: Any, telemetry_at: float | None,
    ) -> None:
        if self._sink is None:
            return
        if telemetry is None:
            phone_block: dict[str, Any] = {
                "status": None, "headroom": None, "headroom_absent": None,
                "skin_temp_c": None, "skin_zone": None, "skin_temp_absent": None,
                "at_mono": None, "age_s": None, "absent": ABSENT_NO_TELEMETRY,
            }
        else:
            phone_block = {
                "status": getattr(telemetry, "thermal_status", None),
                "headroom": getattr(telemetry, "thermal_headroom", None),
                "headroom_absent": getattr(telemetry, "thermal_headroom_absent", None),
                "skin_temp_c": getattr(telemetry, "skin_temp_c", None),
                "skin_zone": getattr(telemetry, "skin_temp_zone", None),
                "skin_temp_absent": getattr(telemetry, "skin_temp_absent", None),
                "at_mono": telemetry_at,
                "age_s": None if telemetry_at is None else round(now - telemetry_at, 3),
                "absent": None,
            }
        self._sink.write({
            "type": "thermal_sample", "t_wall": time.time(), "t_mono": now,
            "jetson": {
                "zones": dict(census), "cooling": dict(cooling),
                "basis": jetson_basis, "reason": jetson_reason,
            },
            "phone": phone_block,
        })

    # -- per tick ----------------------------------------------------------------

    def latest(self, now: float | None = None) -> dict[str, Any]:
        """The `record["thermal"]` block `run_demo` writes on every tick.
        Reads only this sampler's own state and `self._phone` -- never
        `sensing`, which can be `None` on a phoneless run.
        """
        now = self._now() if now is None else now
        with self._lock:
            return {
                "jetson": self._latest_jetson(now).to_record(),
                "phone": self._latest_phone(),
                "events": {
                    "jetson": self._latest_jetson_events(),
                    "phone": self._latest_phone_events(),
                },
            }

    def _latest_jetson(self, now: float) -> ThermalReading:
        if self._last_sample_mono is not None:
            age = now - self._last_sample_mono
            basis = THERMAL_BASIS_MEASURED if age <= 2 * self._interval_s else THERMAL_BASIS_STALE
            return ThermalReading(
                celsius=self._last_selected_celsius, zone=self._selected_zone,
                basis=basis, age_s=age, reason=None, zones_n=len(self._last_census),
            )
        # A reason from an actual attempt outranks the running/stopped guess:
        # once a pass has run and failed, that is why there is no reading,
        # whether or not the thread is still alive to try again.
        if self._last_zones_reason is not None:
            reason = self._last_zones_reason
        elif not self._running:
            reason = ABSENT_SAMPLER_STOPPED
        else:
            reason = ABSENT_NO_SAMPLE_YET
        return ThermalReading(celsius=None, zone=None, basis=THERMAL_BASIS_ABSENT,
                               age_s=None, reason=reason, zones_n=0)

    def _latest_phone(self) -> dict[str, Any]:
        telemetry = getattr(self._phone, "telemetry", None) if self._phone is not None else None
        telemetry_at = getattr(self._phone, "telemetry_at_mono", None) if self._phone is not None else None
        if telemetry is None:
            return {"headroom": None, "headroom_absent": None, "skin_zone": None,
                    "skin_temp_absent": None, "at_mono": None, "absent": ABSENT_NO_TELEMETRY}
        return {
            "headroom": getattr(telemetry, "thermal_headroom", None),
            "headroom_absent": getattr(telemetry, "thermal_headroom_absent", None),
            "skin_zone": getattr(telemetry, "skin_temp_zone", None),
            "skin_temp_absent": getattr(telemetry, "skin_temp_absent", None),
            "at_mono": telemetry_at,
            "absent": None,
        }

    def _latest_jetson_events(self) -> dict[str, Any]:
        # A transition this sampler actually observed and logged is `fired`
        # regardless of whether some other cooling device on the same pass,
        # or a later pass, ever gave a reading -- that incompleteness rides
        # along in `missing` and the pass counters instead of erasing the
        # count. `not_evaluable` is reserved for zero transitions with
        # incomplete observation, where there is nothing else to report.
        if self._jetson_event_count > 0:
            if self._cooling_fully_readable():
                return _tick_event_record(RULE_FIRED, self._jetson_event_count, self._jetson_last_event)
            return _tick_event_record(
                RULE_FIRED, self._jetson_event_count, self._jetson_last_event,
                (MISSING_COOLING_STATE,),
                passes_attempted=self._cooling_passes_attempted,
                passes_readable=self._cooling_passes_readable,
            )
        if not self._cooling_fully_readable():
            return _tick_event_record(
                RULE_NOT_EVALUABLE, 0, None, (MISSING_COOLING_STATE,),
                passes_attempted=self._cooling_passes_attempted,
                passes_readable=self._cooling_passes_readable,
            )
        # `quiet` is a claim of full observation, exactly like `not_evaluable`
        # is a claim of its absence -- both need the same pass counters as
        # evidence, or the claim can only be taken on faith.
        return _tick_event_record(
            RULE_QUIET, 0, None,
            passes_attempted=self._cooling_passes_attempted,
            passes_readable=self._cooling_passes_readable,
        )

    def _latest_phone_events(self) -> dict[str, Any]:
        telemetry = getattr(self._phone, "telemetry", None) if self._phone is not None else None
        if telemetry is None:
            return _tick_event_record(RULE_NOT_EVALUABLE, 0, None, (MISSING_TELEMETRY,))
        if getattr(telemetry, "thermal_status_changes", None) is None:
            return _tick_event_record(RULE_NOT_EVALUABLE, 0, None, (MISSING_STATUS_CHANGES,))
        status = RULE_FIRED if self._phone_event_count > 0 else RULE_QUIET
        return _tick_event_record(status, self._phone_event_count, self._phone_last_event)

    # -- per run -------------------------------------------------------------

    def to_record(self, *, jtop_available: bool | None = None) -> dict[str, Any]:
        """The `summary["thermal"]` block, written once at the end of a run."""
        with self._lock:
            return self._to_record_locked(jtop_available=jtop_available)

    def _to_record_locked(self, *, jtop_available: bool | None) -> dict[str, Any]:
        temp_stats = _pctl(self._selected_temp_samples) if self._selected_temp_samples else None
        skin_stats = _pctl(self._phone_skin_temps) if self._phone_skin_temps else None
        return {
            "jetson": {
                "samples": self._samples,
                "selected_zone": self._selected_zone,
                "selected_by": self._selected_by,
                "zones_seen": sorted(self._zones_seen),
                "temp_c": temp_stats,
                "per_zone_max_c": dict(self._per_zone_max),
                "cooling_devices": sorted(self._cooling_devices),
                "basis_counts": dict(self._basis_counts),
                "absent_reasons": dict(self._absent_reasons),
                "jtop_available": jtop_available,
            },
            "phone": {
                "samples": self._phone_samples,
                "status_counts": dict(self._phone_status_counts),
                "skin_temp_c": skin_stats,
                "skin_zone": self._phone_skin_zone,
                "headroom_absent_counts": dict(self._phone_headroom_absent_counts),
                "skin_temp_absent_counts": dict(self._phone_skin_absent_counts),
                "absent_counts": dict(self._phone_absent_counts),
            },
            "events": {
                "jetson": self._jetson_events_summary(),
                "phone": self._phone_events_summary(),
            },
        }

    def _jetson_events_summary(self) -> dict[str, Any]:
        # As `_latest_jetson_events`: a real count outranks incomplete
        # observation, which is carried alongside it rather than in place of
        # it.
        if self._jetson_event_count > 0:
            if self._cooling_fully_readable():
                return _summary_event_record(
                    RULE_FIRED, self._jetson_event_count, (), by_unit=self._jetson_events_by_unit,
                )
            return _summary_event_record(
                RULE_FIRED, self._jetson_event_count, (MISSING_COOLING_STATE,),
                by_unit=self._jetson_events_by_unit,
                passes_attempted=self._cooling_passes_attempted,
                passes_readable=self._cooling_passes_readable,
            )
        if not self._cooling_fully_readable():
            return _summary_event_record(
                RULE_NOT_EVALUABLE, 0, (MISSING_COOLING_STATE,), by_unit={},
                passes_attempted=self._cooling_passes_attempted,
                passes_readable=self._cooling_passes_readable,
            )
        return _summary_event_record(
            RULE_QUIET, 0, (), by_unit=self._jetson_events_by_unit,
            passes_attempted=self._cooling_passes_attempted,
            passes_readable=self._cooling_passes_readable,
        )

    def _phone_events_summary(self) -> dict[str, Any]:
        if self._phone_samples == 0 and not self._phone_status_changes_ever_seen:
            return _summary_event_record(RULE_NOT_EVALUABLE, 0, (MISSING_TELEMETRY,))
        if not self._phone_status_changes_ever_seen:
            return _summary_event_record(RULE_NOT_EVALUABLE, 0, (MISSING_STATUS_CHANGES,))
        status = RULE_FIRED if self._phone_event_count > 0 else RULE_QUIET
        return _summary_event_record(
            status, self._phone_event_count, (),
            count_without_descriptors=self._phone_count_without_descriptors,
            gap_events=self._phone_gap_events,
        )
