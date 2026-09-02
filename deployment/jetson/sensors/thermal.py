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
#: `ThermalSampler` that produces no reading returns one of these four.
ABSENT_NO_THERMAL_ROOT = "no_thermal_root"     # the directory would not list
ABSENT_NO_ZONE_READABLE = "no_zone_readable"   # listed, no plausible temp in any zone
ABSENT_NO_SAMPLE_YET = "no_sample_yet"         # running, first pass not finished
ABSENT_SAMPLER_STOPPED = "sampler_stopped"     # disabled in config, or the thread died
ABSENT_REASONS = frozenset({
    ABSENT_NO_THERMAL_ROOT, ABSENT_NO_ZONE_READABLE,
    ABSENT_NO_SAMPLE_YET, ABSENT_SAMPLER_STOPPED,
})

#: Why the phone's thermal fields are absent, beyond the phone's own per-field
#: reasons (`headroom_absent` / `skin_temp_absent`, carried on the wire). Spelled
#: like `sensing_loop.reference_from`'s own `"no_telemetry"` because it answers
#: the identical question: nothing has arrived from the phone at all.
ABSENT_NO_TELEMETRY = "no_telemetry"

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
    missing, unreadable, or empty. Never raises -- a permission denial on one
    zone must not take the whole sample down.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return text or None


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

    def read_cooling(self) -> tuple[dict[str, int], bool]:
        """Every `cooling_device*`'s current state, keyed by its `type`.
        `(states, readable)`: `readable` is False when nothing here could be
        read at all -- the caller's cue to report `not_evaluable` rather than
        a quiet zero.
        """
        try:
            if not self.root.is_dir():
                return {}, False
            entries = sorted(self.root.glob("cooling_device*"))
        except OSError:
            return {}, False

        states: dict[str, int] = {}
        for entry in entries:
            name = _read_trimmed(entry / "type")
            if name is None or name in states:
                continue
            raw = _read_trimmed(entry / "cur_state")
            if raw is None:
                continue
            try:
                states[name] = int(raw)
            except ValueError:
                continue
        return states, bool(states)

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
) -> dict[str, Any]:
    """One `events.<device>` entry for the per-tick block. `missing` is present
    only on a `not_evaluable` entry, the way `RuleCheck.to_record` emits it.
    """
    record: dict[str, Any] = {"status": status, "count": count, "last": last}
    if missing:
        record["missing"] = list(missing)
    return record


def _summary_event_record(
    status: str, count: int, missing: tuple[str, ...], by_unit: dict[str, int] | None = None,
) -> dict[str, Any]:
    """As `_tick_event_record`, but for the drive-level rollup, where `missing`
    is always present (possibly empty) because the summary is read on its own,
    with no sibling ticks to compare it against.
    """
    record: dict[str, Any] = {"status": status, "count": count, "missing": list(missing)}
    if by_unit is not None:
        record["by_unit"] = dict(by_unit)
    return record


def _pctl(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction: float) -> float:
        return ordered[min(n - 1, int(fraction * n))]

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
        self._cooling_ever_readable = False
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

        # -- summary accumulators ----------------------------------------------
        self._samples = 0
        self._selected_temp_samples: list[float] = []
        self._zones_seen: set[str] = set()
        self._per_zone_max: dict[str, float] = {}
        self._cooling_devices: set[str] = set()
        self._basis_counts: dict[str, int] = {b: 0 for b in THERMAL_BASES}
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
            try:
                self.sample_once()
            except Exception:
                # A sysfs read misbehaving must not take the pipeline down with
                # it. `_running` goes false, so a tick after this reports
                # `sampler_stopped` rather than quietly repeating the last
                # value forever.
                self._running = False
                return
            self._stop_event.wait(self._interval_s)

    # -- one pass, on the thread or driven directly by a test -------------------

    def sample_once(self) -> None:
        now = self._now()
        census, zones_reason = self._jetson.read_zones()
        cooling, cooling_readable = self._jetson.read_cooling()
        telemetry = getattr(self._phone, "telemetry", None) if self._phone is not None else None
        telemetry_at = getattr(self._phone, "telemetry_at_mono", None) if self._phone is not None else None

        with self._lock:
            jetson_basis, jetson_reason = self._advance_zone_reading(census, zones_reason, now)
            self._process_cooling(cooling, cooling_readable, census, now)
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
            # Zones were readable this pass, but the held zone specifically was
            # not among them -- still "no plausible temp in any zone" from the
            # point of view of the series this sampler is holding.
            self._last_zones_reason = ABSENT_NO_ZONE_READABLE
            self._basis_counts[THERMAL_BASIS_ABSENT] += 1
            self._absent_reasons[ABSENT_NO_ZONE_READABLE] = (
                self._absent_reasons.get(ABSENT_NO_ZONE_READABLE, 0) + 1
            )
            return THERMAL_BASIS_ABSENT, ABSENT_NO_ZONE_READABLE

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
        self, states: dict[str, int], readable: bool, census: dict[str, float], now: float,
    ) -> None:
        if not readable:
            return
        self._cooling_ever_readable = True
        self._cooling_devices.update(states)
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
        self._prev_cooling = dict(states)

    def _process_phone_events(self, telemetry: Any, now: float) -> None:
        if telemetry is None:
            return
        changes = getattr(telemetry, "thermal_status_changes", None)
        if changes is None:
            return
        self._phone_status_changes_ever_seen = True
        if self._prev_status_changes is not None and changes != self._prev_status_changes:
            frm = getattr(telemetry, "thermal_change_from", None)
            to = getattr(telemetry, "thermal_change_to", None)
            at_ns = getattr(telemetry, "thermal_change_at_mono_ns", None)
            self._phone_event_count = changes
            self._phone_last_event = {"at_mono": now, "from": frm, "to": to}
            self._phone_seq += 1
            self._write_event(
                device="phone", seq=self._phone_seq, now=now,
                clock="phone", at_ns=at_ns,
                source="thermal_status", unit=None, from_=frm, to=to,
                temp_c=getattr(telemetry, "skin_temp_c", None), zones=None,
            )
        self._phone_event_count = changes
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
        headroom_absent = getattr(telemetry, "thermal_headroom_absent", None)
        if headroom_absent is not None:
            self._phone_headroom_absent_counts[headroom_absent] = (
                self._phone_headroom_absent_counts.get(headroom_absent, 0) + 1
            )
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
        if not self._cooling_ever_readable:
            return _tick_event_record(RULE_NOT_EVALUABLE, 0, None, (MISSING_COOLING_STATE,))
        status = RULE_FIRED if self._jetson_event_count > 0 else RULE_QUIET
        return _tick_event_record(status, self._jetson_event_count, self._jetson_last_event)

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
        if not self._cooling_ever_readable:
            return _summary_event_record(RULE_NOT_EVALUABLE, 0, (MISSING_COOLING_STATE,), by_unit={})
        status = RULE_FIRED if self._jetson_event_count > 0 else RULE_QUIET
        return _summary_event_record(status, self._jetson_event_count, (), by_unit=self._jetson_events_by_unit)

    def _phone_events_summary(self) -> dict[str, Any]:
        if self._phone_samples == 0 and not self._phone_status_changes_ever_seen:
            return _summary_event_record(RULE_NOT_EVALUABLE, 0, (MISSING_TELEMETRY,))
        if not self._phone_status_changes_ever_seen:
            return _summary_event_record(RULE_NOT_EVALUABLE, 0, (MISSING_STATUS_CHANGES,))
        status = RULE_FIRED if self._phone_event_count > 0 else RULE_QUIET
        return _summary_event_record(status, self._phone_event_count, ())
