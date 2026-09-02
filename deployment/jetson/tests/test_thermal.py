"""sensors.thermal: the Jetson's own temperature and both devices' throttle
events. Also pins the boundary of this task's one behaviour change on the
phone -- `Inputs` unchanged and the controller byte-identical -- since those
are exactly what would move if the change escaped its fence.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import fields
from pathlib import Path

import pytest

from eval_run import pctl as eval_run_pctl
from policy import sensing_controller
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE, RULE_QUIET, Inputs, SensingController
from sensors import thermal
from sensors.thermal import (
    ABSENT_NO_SAMPLE_YET,
    ABSENT_NO_TELEMETRY,
    ABSENT_NO_THERMAL_ROOT,
    ABSENT_NO_ZONE_READABLE,
    ABSENT_REASONS,
    ABSENT_SAMPLER_STOPPED,
    ABSENT_ZONE_DISAPPEARED,
    HEADROOM_ABSENT_UNSPECIFIED,
    MISSING_COOLING_STATE,
    MISSING_STATUS_CHANGES,
    MISSING_TELEMETRY,
    JetsonThermal,
    ThermalSampler,
    _pctl,
)
from transport.messages import DROP_KEYS, RATE_KEYS, PhoneTelemetry


def _zone(root: Path, index: int, type_: str, temp: str | None) -> None:
    directory = root / f"thermal_zone{index}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "type").write_text(type_)
    if temp is not None:
        (directory / "temp").write_text(temp)


def _cooling(root: Path, index: int, type_: str, state: str) -> None:
    directory = root / f"cooling_device{index}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "type").write_text(type_)
    (directory / "cur_state").write_text(state)


class _Clock:
    """A monotonic clock a test can move by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Sink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def _telemetry(**overrides) -> PhoneTelemetry:
    fields_ = dict(
        t_capture_mono_ns=0, thermal_status="nominal",
        achieved={k: 1.0 for k in RATE_KEYS}, dropped={k: 0 for k in DROP_KEYS},
        here_calls=0, here_errors=0,
    )
    fields_.update(overrides)
    return PhoneTelemetry(**fields_)


class _FakePhone:
    """Just enough of `PhoneLink` for `ThermalSampler`: `telemetry` and
    `telemetry_at_mono`, reassignable between passes the way a redial or a
    fresh 1 Hz report would reassign them in production.
    """

    def __init__(self, telemetry: PhoneTelemetry | None = None, telemetry_at_mono: float | None = None) -> None:
        self.telemetry = telemetry
        self.telemetry_at_mono = telemetry_at_mono


# -- round 3: measured on the device -----------------------------------------


def test_read_trimmed_catches_a_would_block_type_error():
    """Measured on the device: three of this Orin's nine zones answer EAGAIN
    on every pass, and a read that would block surfaces through the buffered
    text layer as `None` rather than as an `OSError` -- `str.strip()` on that
    `None` then raises `TypeError`, which the old `except OSError` did not
    catch. `_read_trimmed`'s own docstring promises it never raises; this
    pins that promise directly, without a device.
    """

    class _WouldBlock:
        def read_text(self) -> str:
            raise TypeError("can't concat NoneType to bytes")

    assert thermal._read_trimmed(_WouldBlock()) is None


# -- 1. a zone that will not read is absent with a reason, not a zero --------


def test_a_zone_that_will_not_read_is_absent_with_a_reason(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", temp=None)  # type readable, temp file absent

    census, reason = JetsonThermal(root=root).read_zones()
    assert census == {}
    assert reason == ABSENT_NO_ZONE_READABLE

    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()
    reading = sampler.latest(now=100.0)["jetson"]
    assert reading["basis"] == "absent"
    assert reading["reason"] == ABSENT_NO_ZONE_READABLE
    assert reading["temp_c"] is None


# -- 2. the five absence reasons are each reachable and distinct ------------


def test_the_five_absence_reasons_are_each_reachable_and_distinct(tmp_path):
    seen: set[str] = set()

    # arm 1: the root would not list at all.
    missing_root = JetsonThermal(root=tmp_path / "does-not-exist")
    _, reason = missing_root.read_zones()
    assert reason == ABSENT_NO_THERMAL_ROOT
    seen.add(reason)

    # arm 2: the root lists, but nothing in it is a plausible temperature.
    no_zone_root = tmp_path / "no-zone"
    _zone(no_zone_root, 0, "soc", "100000000")  # excluded: 100000 C, implausible
    _, reason = JetsonThermal(root=no_zone_root).read_zones()
    assert reason == ABSENT_NO_ZONE_READABLE
    seen.add(reason)

    # arm 3: the sampler is running but has not completed a first pass.
    started = ThermalSampler(sink=None, jetson=JetsonThermal(root=no_zone_root), clock=_Clock(100.0))
    started._running = True
    reading = started.latest(now=100.0)["jetson"]
    assert reading["reason"] == ABSENT_NO_SAMPLE_YET
    seen.add(reading["reason"])

    # arm 4: the sampler was never started (equally true after `stop()`).
    stopped = ThermalSampler(sink=None, jetson=JetsonThermal(root=no_zone_root), clock=_Clock(100.0))
    reading = stopped.latest(now=100.0)["jetson"]
    assert reading["reason"] == ABSENT_SAMPLER_STOPPED
    seen.add(reading["reason"])

    # arm 5: the held zone specifically drops out of an otherwise-readable
    # census -- nine other zones read fine, so this is not "no plausible temp
    # in any zone" (arm 2's reason). Once a zone has ever been measured,
    # `latest()` reports it `measured`/`stale` forever (D4) rather than
    # falling back to absent, so this reason only surfaces in the drive-level
    # `absent_reasons` rollup, not in a later `latest()` call.
    disappearing_root = tmp_path / "disappearing"
    _zone(disappearing_root, 0, "cpu-thermal", "40000")
    for i in range(1, 10):
        _zone(disappearing_root, i, f"zone{i}", "41000")
    jetson = JetsonThermal(root=disappearing_root)
    sampler = ThermalSampler(sink=None, jetson=jetson, clock=_Clock(100.0))
    sampler.sample_once()
    assert sampler.latest(now=100.0)["jetson"]["zone"] == "cpu-thermal"
    (disappearing_root / "thermal_zone0" / "type").unlink()  # the held zone vanishes
    sampler.sample_once()
    absent_reasons = sampler.to_record()["jetson"]["absent_reasons"]
    assert ABSENT_ZONE_DISAPPEARED in absent_reasons
    assert sampler.to_record()["jetson"]["zones_seen"] == sorted(
        ["cpu-thermal"] + [f"zone{i}" for i in range(1, 10)]
    )
    seen.add(ABSENT_ZONE_DISAPPEARED)

    assert seen == ABSENT_REASONS
    assert len(seen) == 5


# -- 3. a value that is not a temperature is refused -------------------------


def test_a_value_that_is_not_a_temperature_is_refused(tmp_path):
    """The `soc`/`ibat` case `ThermalZones.kt` documents on the phone, ported:
    a reading whose magnitude-converted value falls outside the plausible band
    is excluded from the census and cannot be selected.

    NOTE ON THE PLAN: plan's own worked example for this test names raw
    readings `100000` and `-351000`. `100000` millidegrees converts to 100.0 C,
    which is *inside* [-40, 125] and would not be excluded -- that example
    value does not demonstrate the refusal it is meant to. This test instead
    uses the two raw strings `ThermalZonesTest.kt`'s existing, already-shipped
    "not a temperature at all" test uses (`100000000`, `-2000000`), which do
    convert to values outside the band.
    """
    root = tmp_path / "root"
    _zone(root, 0, "skin", "100000000")   # -> 100000.0 C, refused
    _zone(root, 1, "quiet_therm", "-2000000")  # -> -2000.0 C, refused
    _zone(root, 2, "xo_therm", "28926")   # -> 28.926 C, the only plausible one

    census, reason = JetsonThermal(root=root).read_zones()
    assert reason is None
    assert set(census) == {"xo_therm"}
    assert census["xo_therm"] == pytest.approx(28.926)


# -- 4. millidegrees and degrees are separated by magnitude ------------------

#: The same literal table `ThermalZonesTest.kt` states in
#: `celsiusOf matches the shared table also stated in Python`. One copy read
#: by both suites is not possible across languages, so a divergence between
#: the two implementations shows as two different expected lists in review.
MAGNITUDE_TABLE = [
    ("47500", 47.5),
    ("47", 47.0),
    ("999", None),          # below the magnitude threshold, so 999.0 C: implausible
    ("1000", 1.0),          # exactly at the magnitude threshold: millidegrees, not degrees
    ("-1000", -1.0),        # the same threshold, negative
    ("-40000", -40.0),      # exactly at the plausible floor
    ("125000", 125.0),      # exactly at the plausible ceiling
    ("-40001", None),
    ("125001", None),
    ("100000000", None),
    ("-2000000", None),
]


@pytest.mark.parametrize("raw,expected", MAGNITUDE_TABLE)
def test_millidegrees_and_degrees_are_separated_by_magnitude(raw, expected):
    result = JetsonThermal.celsius_of(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# -- 5, 6, 7: staleness -------------------------------------------------------


def test_a_stale_sample_is_stale_not_measured_and_carries_its_age(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), interval_s=1.0, clock=clock)
    sampler.sample_once()

    clock.advance(2.5)  # past 2 x interval_s (2.0)
    reading = sampler.latest(now=clock.now)["jetson"]
    assert reading["basis"] == "stale"
    assert reading["age_s"] == pytest.approx(2.5)
    assert reading["temp_c"] == pytest.approx(40.0)


def test_a_stale_sample_is_never_collapsed_to_absent(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), interval_s=1.0, clock=clock)
    sampler.sample_once()

    clock.advance(600.0)
    reading = sampler.latest(now=clock.now)["jetson"]
    assert reading["basis"] == "stale"
    assert reading["age_s"] == pytest.approx(600.0)


def test_the_freshness_bound_follows_the_configured_interval(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), interval_s=0.2, clock=clock)
    sampler.sample_once()

    clock.advance(0.5)  # > 2 x 0.2
    reading = sampler.latest(now=clock.now)["jetson"]
    assert reading["basis"] == "stale"


# -- 8. quiet and not_evaluable are different records (headline) ------------


def test_quiet_and_not_evaluable_are_different_records(tmp_path):
    # Half A: the cooling directory reads throughout and never changes.
    readable_root = tmp_path / "readable"
    _zone(readable_root, 0, "cpu-thermal", "40000")
    _cooling(readable_root, 0, "pwm-fan", "0")
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=readable_root), clock=_Clock(100.0))
    sampler.sample_once()
    sampler.sample_once()
    quiet = sampler.latest(now=100.0)["events"]["jetson"]
    assert quiet == {"status": RULE_QUIET, "count": 0, "last": None}

    # Half B: no cooling device exists to read at all.
    unreadable_root = tmp_path / "unreadable"
    _zone(unreadable_root, 0, "cpu-thermal", "40000")
    sampler2 = ThermalSampler(sink=None, jetson=JetsonThermal(root=unreadable_root), clock=_Clock(100.0))
    sampler2.sample_once()
    not_evaluable = sampler2.latest(now=100.0)["events"]["jetson"]
    assert not_evaluable == {
        "status": RULE_NOT_EVALUABLE, "count": 0, "last": None,
        "missing": ["cooling_device_cur_state"],
        "passes_attempted": 0, "passes_readable": 0,
    }


# -- 9. a cooling transition emits exactly one event, with the device name --


def test_a_cooling_transition_emits_exactly_one_event_with_the_device_name(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    _cooling(root, 0, "pwm-fan", "0")
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()

    _cooling(root, 0, "pwm-fan", "1")
    clock.advance(1.0)
    sampler.sample_once()

    events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(events) == 1
    event = events[0]
    assert event["device"] == "jetson"
    assert event["unit"] == "pwm-fan"
    assert event["from"] == 0
    assert event["to"] == 1
    assert event["zones"] == {"cpu-thermal": 40.0}

    # Counted, not derived by subtraction from an unrelated total -- task 34's
    # `superseded = received - shown - expired` defect, in this task's shape.
    assert sampler.latest(now=clock.now)["events"]["jetson"]["count"] == 1


# -- 10. an event is emitted with no tick loop running -----------------------


def test_an_event_is_emitted_with_no_tick_loop_running(tmp_path):
    """The sampler is driven directly -- no `pipeline`, no `Tick`, no
    `run_demo` loop of any kind -- and still produces both record types.
    Pins D3's reason for running the sampler beside the tick path rather than
    on it.
    """
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    _cooling(root, 0, "pwm-fan", "0")
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), clock=clock)

    sampler.sample_once()
    _cooling(root, 0, "pwm-fan", "1")
    clock.advance(1.0)
    sampler.sample_once()

    types = [r["type"] for r in sink.records]
    assert types.count("thermal_sample") == 2
    assert types.count("thermal_event") == 1


# -- 11. the status words are the controller's own objects -------------------


def test_the_status_words_are_the_controllers_own_objects():
    assert thermal.RULE_QUIET is sensing_controller.RULE_QUIET
    assert thermal.RULE_FIRED is sensing_controller.RULE_FIRED
    assert thermal.RULE_NOT_EVALUABLE is sensing_controller.RULE_NOT_EVALUABLE


# -- 12. Inputs is unchanged --------------------------------------------------

EXPECTED_INPUTS_FIELDS = (
    "ego_acceleration", "ego_speed", "policy_margin", "feed_congestion",
    "camera_density_bin", "feed_declined", "thermal_status", "skin_temp_c",
    "telemetry_age_s", "ego_acceleration_source", "ego_speed_source",
    "camera_density_bin_source", "camera_last_detection_age_s", "lat", "lon",
    "position_valid", "position_age_s",
)


def test_inputs_is_unchanged_at_seventeen_fields():
    names = tuple(f.name for f in fields(Inputs))
    assert names == EXPECTED_INPUTS_FIELDS
    assert len(names) == 17


# -- 13. score_shadow still scores a task-36 log after this task ------------


def test_score_shadow_still_scores_a_log_after_this_task(tmp_path):
    import score_shadow
    # Reused rather than re-typed: `write_run` there builds a real
    # `SensingLoop` drive, exactly the fixture this regression needs and the
    # one score_shadow's own suite already trusts.
    from tests.test_score_shadow import write_run as write_score_shadow_run

    run_dir = write_score_shadow_run(tmp_path, n=12)
    result = score_shadow.score(run_dir)
    assert "refused" not in result
    assert result["replay_identity"]["status"] == "ok"
    assert result["replay_identity"]["mismatched"] == 0


# -- 14. the tick block does not depend on sensing ---------------------------


def test_the_tick_block_does_not_depend_on_sensing():
    """Built with `phone=None` and never referencing a `SensingLoop` or a
    `sensing` record at all -- `run_demo.py` can build `record["thermal"]`
    on a phoneless run, where `sensing` is itself `None` (run_demo.py:444).
    """
    sampler = ThermalSampler(sink=None, phone=None, clock=_Clock(100.0))
    block = sampler.latest(now=100.0)
    assert set(block) == {"jetson", "phone", "events"}
    assert block["phone"]["absent"] == ABSENT_NO_TELEMETRY


# -- 15. at_mono agrees with the reference witness ---------------------------


def test_at_mono_agrees_with_the_reference_witness():
    from policy.sensing_loop import reference_from

    telemetry = _telemetry(thermal_headroom_absent="not_a_number")
    phone = _FakePhone(telemetry=telemetry, telemetry_at_mono=1234.5678)

    now = 1235.0
    reference = reference_from(phone, now=now)
    sampler = ThermalSampler(sink=None, phone=phone, clock=lambda: now)
    tick_block = sampler.latest(now=now)

    assert reference["at_mono"] == 1234.5678
    assert tick_block["phone"]["at_mono"] == reference["at_mono"]


# -- 16. eval_run prints the thermal section, on real record shapes --------


def test_eval_run_prints_the_thermal_section(tmp_path):
    from eval_run import analyze, render_markdown
    from tests.test_eval_run import make_tick, write_run

    root = tmp_path / "sysroot"
    _zone(root, 0, "cpu-thermal", "47500")
    _cooling(root, 0, "pwm-fan", "0")

    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), interval_s=1.0, clock=clock)
    sampler.sample_once()
    _cooling(root, 0, "pwm-fan", "1")  # a real transition, so `events` is not vacuously []
    clock.advance(1.0)
    sampler.sample_once()

    ticks = [make_tick(i) for i in range(5)]
    for t in ticks:
        t["thermal"] = sampler.latest(now=clock.now)

    run_dir = write_run(tmp_path, ticks, summary={
        "ticks": len(ticks), "camera_dropped_frames": 0, "policy_trained": False,
        "thermal": sampler.to_record(jtop_available=True),
    })
    with open(run_dir / "metadata.jsonl", "a") as f:
        for record in sink.records:
            f.write(json.dumps(record) + "\n")

    result = analyze(run_dir)
    md = render_markdown(result, [])

    assert "## Thermal" in md
    assert "47.5" in md
    assert result["thermal"] is not None
    assert result["thermal"]["ticks_by_basis"] == {"measured": 5}
    expected_events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(expected_events) == 1, "the fixture must produce a real event, not an empty list"
    assert result["thermal"]["events"] == expected_events


def test_eval_run_prints_not_evaluable_when_cooling_is_missing(tmp_path):
    from eval_run import analyze, render_markdown
    from tests.test_eval_run import make_tick, write_run

    root = tmp_path / "sysroot_no_cooling"
    _zone(root, 0, "cpu-thermal", "47500")  # no cooling_device* at all

    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()

    ticks = [make_tick(i) for i in range(3)]
    for t in ticks:
        t["thermal"] = sampler.latest(now=clock.now)
    run_dir = write_run(tmp_path, ticks, summary={
        "ticks": len(ticks), "camera_dropped_frames": 0, "policy_trained": False,
        "thermal": sampler.to_record(jtop_available=None),
    })

    result = analyze(run_dir)
    md = render_markdown(result, [])
    assert "NOT EVALUABLE" in md


# -- 17. a pre-task-37 run does not crash and does not fail ------------------


def test_a_pre_task_37_run_does_not_crash_and_is_not_failed(tmp_path):
    from eval_run import analyze
    from tests.test_eval_run import make_tick, write_run

    ticks = [make_tick(i) for i in range(30)]  # no "thermal" key anywhere
    run_dir = write_run(tmp_path, ticks)

    result = analyze(run_dir)
    assert result["thermal"] is None
    assert result["overall_pass"] is True


# -- confirmed validation findings, task 37 round 2 --------------------------


# C1: a redial must not erase the departed handset's events or invent a
# from-None-to-None one.


def test_a_phone_redial_accumulates_instead_of_resetting_the_count(tmp_path):
    """`PhoneLink._rebind` sets `_telemetry` to `None` on every redial, and the
    replacement handset's own `thermal_status_changes` restarts at 0. Copying
    that value straight into the sampler's count erased the departed
    handset's real transitions and printed an event whose `from`/`to` were
    both None; this pins that the count survives the redial and that no such
    event is ever written.
    """
    phone = _FakePhone()
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    # Old handset: baseline, then one report carrying two transitions (D9).
    phone.telemetry = _telemetry(thermal_status_changes=0)
    sampler.sample_once()
    phone.telemetry = _telemetry(
        thermal_status_changes=2, thermal_change_from="light", thermal_change_to="severe",
    )
    clock.advance(1.0)
    sampler.sample_once()
    before = sampler.latest(now=clock.now)["events"]["phone"]
    assert before["count"] == 2
    assert before["last"] == {"at_mono": clock.now, "from": "light", "to": "severe"}

    # Redial: telemetry drops to None before the new handset's first report.
    phone.telemetry = None
    clock.advance(1.0)
    sampler.sample_once()

    # New handset's own counter restarts at 0 -- must not compare against the
    # departed handset's last count of 2.
    phone.telemetry = _telemetry(thermal_status_changes=0)
    clock.advance(1.0)
    sampler.sample_once()
    after = sampler.latest(now=clock.now)["events"]["phone"]
    assert after["status"] == RULE_FIRED
    assert after["count"] == 2, "the old handset's real events must survive the redial"
    assert after["last"] == before["last"]

    phantom_events = [
        r for r in sink.records
        if r["type"] == "thermal_event" and r["device"] == "phone"
        and r["from"] is None and r["to"] is None
    ]
    assert phantom_events == [], "no event may ever carry both from and to null"

    # A genuine transition on the new handset accumulates on top of the old
    # handset's total rather than restarting from it.
    phone.telemetry = _telemetry(
        thermal_status_changes=1, thermal_change_from="nominal", thermal_change_to="light",
    )
    clock.advance(1.0)
    sampler.sample_once()
    final = sampler.latest(now=clock.now)["events"]["phone"]
    assert final["count"] == 3
    assert final["last"] == {"at_mono": clock.now, "from": "nominal", "to": "light"}


def test_the_baseline_resets_on_a_redial_even_when_the_new_counters_briefly_coincide(tmp_path):
    """If the baseline were not cleared when telemetry goes absent, a new
    handset's first report -- whose own counter starts from its own boot and
    can already be higher than the departed handset's last value -- would be
    diffed against that stale value and produce a spurious event straddling
    two different phones.
    """
    phone = _FakePhone()
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    phone.telemetry = _telemetry(thermal_status_changes=0)
    sampler.sample_once()
    phone.telemetry = _telemetry(thermal_status_changes=1, thermal_change_from="nominal", thermal_change_to="light")
    clock.advance(1.0)
    sampler.sample_once()
    before = sampler.latest(now=clock.now)["events"]["phone"]
    assert before["count"] == 1

    phone.telemetry = None  # redial
    clock.advance(1.0)
    sampler.sample_once()

    # The new handset's own counter is already at 5 on its first report --
    # unrelated to the departed handset's last value of 1.
    phone.telemetry = _telemetry(
        thermal_status_changes=5, thermal_change_from="moderate", thermal_change_to="severe",
    )
    clock.advance(1.0)
    sampler.sample_once()

    after = sampler.latest(now=clock.now)["events"]["phone"]
    assert after["count"] == 1, "the new handset's first report must only set a baseline, not 4 new events"
    assert after["last"] == before["last"]


def test_a_count_rise_with_no_transition_fields_is_still_counted_and_logged(tmp_path):
    """A delta greater than zero can appear with no redial at all if a report
    carries a raised `thermal_status_changes` but no `thermal_change_from`/
    `to` -- reachable from a torn read on the phone (`changesCount` and
    `lastTransition` are two separate lock acquisitions with the status poll
    between them, so the very first transition of a service run can land with
    the count already at 1 and the transition still null) as well as an older
    build. Round 2: this used to be discarded outright, which is a different
    defect from the redial case above -- there, `_prev_status_changes` is
    `None` and no delta is computed at all; here a real baseline exists and a
    real rise happened, so it must be counted and logged, with the missing
    descriptors named absent rather than invented.
    """
    phone = _FakePhone()
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    phone.telemetry = _telemetry(thermal_status_changes=0)
    sampler.sample_once()
    phone.telemetry = _telemetry(thermal_status_changes=1)  # no from/to carried
    clock.advance(1.0)
    sampler.sample_once()

    events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(events) == 1
    assert events[0]["from"] is None
    assert events[0]["to"] is None

    tick_events = sampler.latest(now=clock.now)["events"]["phone"]
    assert tick_events["status"] == RULE_FIRED
    assert tick_events["count"] == 1

    summary = sampler.to_record()["events"]["phone"]
    assert summary["count"] == 1
    assert summary["count_without_descriptors"] == 1


def test_a_half_described_transition_is_not_counted_as_without_descriptors(tmp_path):
    """Only a rise with *neither* endpoint named counts toward
    `count_without_descriptors` -- one whose `from` (or `to`) survived the
    read still describes something, and folding it into the same bucket as a
    fully-absent one would hide the difference between "nothing came
    through" and "half of it did".
    """
    phone = _FakePhone()
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    phone.telemetry = _telemetry(thermal_status_changes=0)
    sampler.sample_once()
    phone.telemetry = _telemetry(thermal_status_changes=1, thermal_change_from="nominal", thermal_change_to=None)
    clock.advance(1.0)
    sampler.sample_once()

    events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(events) == 1
    assert events[0]["from"] == "nominal"
    assert events[0]["to"] is None

    summary = sampler.to_record()["events"]["phone"]
    assert summary["count"] == 1
    assert summary["count_without_descriptors"] == 0


def test_a_multi_transition_gap_is_recorded_in_the_summary(tmp_path):
    """D9: a report carrying a rise of more than one collapses to a single
    `thermal_event` line -- only the most recent transition is nameable -- so
    `sum(count)` legitimately does not equal the number of lines here, the
    plan's own stated exception. The gap itself must still be visible rather
    than only inferable by cross-referencing the count against the lines.
    """
    phone = _FakePhone()
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    phone.telemetry = _telemetry(thermal_status_changes=0)
    sampler.sample_once()
    phone.telemetry = _telemetry(
        thermal_status_changes=2, thermal_change_from="light", thermal_change_to="severe",
    )
    clock.advance(1.0)
    sampler.sample_once()

    events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(events) == 1, "only the most recent transition is nameable"

    summary = sampler.to_record()["events"]["phone"]
    assert summary["count"] == 2
    assert summary["gap_events"] == 1


# C2: cooling readable on some passes and not others is not_evaluable, never
# quiet -- unless a real transition fired, which must survive the same
# incomplete observation instead of being reported as zero.


def test_partial_cooling_readability_is_not_evaluable_not_quiet(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    _cooling(root, 0, "pwm-fan", "0")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: readable

    (root / "cooling_device0" / "cur_state").unlink()  # pass 2: denied
    clock.advance(1.0)
    sampler.sample_once()

    summary_events = sampler.to_record()["events"]["jetson"]
    assert summary_events["status"] == RULE_NOT_EVALUABLE
    assert summary_events["missing"] == [MISSING_COOLING_STATE]
    assert summary_events["passes_attempted"] == 2
    assert summary_events["passes_readable"] == 1

    tick_events = sampler.latest(now=clock.now)["events"]["jetson"]
    assert tick_events["status"] == RULE_NOT_EVALUABLE
    assert tick_events["passes_attempted"] == 2
    assert tick_events["passes_readable"] == 1


def test_a_fired_jetson_event_survives_a_later_unreadable_pass(tmp_path):
    """Round 2's own regression: `pwm-fan` reads 0 then 1 (one real
    `thermal_event` line), then a later pass cannot read `cur_state` at all.
    Reporting `not_evaluable`/`count: 0` here would discard a transition this
    sampler actually observed and logged -- the incompleteness belongs beside
    the real count, in `missing` and the pass counters, not in place of it.
    """
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "0")
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: readable, state 0

    _cooling(root, 0, "pwm-fan", "1")
    clock.advance(1.0)
    sampler.sample_once()  # pass 2: readable, state 1 -- one event

    (root / "cooling_device0" / "cur_state").unlink()
    clock.advance(1.0)
    sampler.sample_once()  # pass 3: cur_state will not read

    events = [r for r in sink.records if r["type"] == "thermal_event"]
    assert len(events) == 1

    tick = sampler.latest(now=clock.now)["events"]["jetson"]
    assert tick["status"] == RULE_FIRED
    assert tick["count"] == 1
    assert tick["missing"] == [MISSING_COOLING_STATE]
    assert tick["passes_attempted"] == 3
    assert tick["passes_readable"] == 2

    summary = sampler.to_record()["events"]["jetson"]
    assert summary["status"] == RULE_FIRED
    assert summary["count"] == 1
    assert summary["by_unit"] == {"pwm-fan": 1}
    assert summary["missing"] == [MISSING_COOLING_STATE]
    assert summary["passes_attempted"] == 3
    assert summary["passes_readable"] == 2


# Round 2 M1: a cooling device whose own `type` stops reading is attempted
# and unreadable, not filed as nothing having been attempted at all.


def test_a_cooling_device_whose_type_stops_reading_counts_as_attempted(tmp_path):
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "0")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: readable

    (root / "cooling_device0" / "type").unlink()
    clock.advance(1.0)
    sampler.sample_once()  # pass 2: `type` itself will not read

    (root / "cooling_device0" / "type").write_text("pwm-fan")
    clock.advance(1.0)
    sampler.sample_once()  # pass 3: readable again

    summary = sampler.to_record()["events"]["jetson"]
    assert summary["status"] == RULE_NOT_EVALUABLE
    assert summary["passes_attempted"] == 3
    assert summary["passes_readable"] == 2


def test_read_cooling_reports_a_device_whose_type_will_not_read_as_missing(tmp_path):
    root = tmp_path / "root"
    directory = root / "cooling_device0"
    directory.mkdir(parents=True)
    (directory / "cur_state").write_text("0")  # no `type` file at all

    states, missing = JetsonThermal(root=root).read_cooling()
    assert states == {}
    assert missing == ("cooling_device0",)


# Round 2 m9: a device sharing a `type` name with one that failed to read
# still gets its own attempt, rather than being skipped because the name is
# in `missing`.


def test_a_second_device_sharing_a_failed_names_type_still_gets_read(tmp_path):
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "0")
    (root / "cooling_device0" / "cur_state").unlink()  # listed, unreadable
    _cooling(root, 1, "pwm-fan", "1")  # same type name, readable

    states, missing = JetsonThermal(root=root).read_cooling()
    assert states == {"pwm-fan": 1}
    assert missing == ("pwm-fan",)


def test_prev_cooling_merges_rather_than_replaces_across_a_missed_pass(tmp_path):
    """A device that fails to read on one pass keeps the value it last gave,
    so a real transition spanning the gap is still detected once it reads
    again -- replacing the whole map on that pass instead of merging into it
    would forget the device entirely and miss the transition.
    """
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "0")
    _cooling(root, 1, "tegra-heavy", "0")
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: both read, state 0

    (root / "cooling_device1" / "cur_state").unlink()  # pass 2: tegra-heavy unreadable
    clock.advance(1.0)
    sampler.sample_once()

    _cooling(root, 1, "tegra-heavy", "1")  # pass 3: reads again, now 1
    clock.advance(1.0)
    sampler.sample_once()

    events = [r for r in sink.records if r["type"] == "thermal_event" and r["unit"] == "tegra-heavy"]
    assert len(events) == 1
    assert events[0]["from"] == 0
    assert events[0]["to"] == 1


# C3: a cooling device that lists but will not read is named as missing, not
# folded into "nothing here read".


def test_read_cooling_reports_a_listed_device_that_will_not_give_a_state(tmp_path):
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "0")
    _cooling(root, 1, "tegra-heavy", "0")
    (root / "cooling_device1" / "cur_state").unlink()

    states, missing = JetsonThermal(root=root).read_cooling()
    assert states == {"pwm-fan": 0}
    assert missing == ("tegra-heavy",)


def test_read_cooling_reports_an_unparseable_state_as_missing_too(tmp_path):
    root = tmp_path / "root"
    _cooling(root, 0, "pwm-fan", "not-a-number")

    states, missing = JetsonThermal(root=root).read_cooling()
    assert states == {}
    assert missing == ("pwm-fan",)


# M1: the phone half of ThermalSampler is exercised through `sample_once`, not
# only through `.latest()` on state a test set by hand.


def test_no_phone_at_all_reports_not_evaluable_not_quiet(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    sampler = ThermalSampler(sink=None, phone=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()

    tick_events = sampler.latest(now=100.0)["events"]["phone"]
    assert tick_events == {
        "status": RULE_NOT_EVALUABLE, "count": 0, "last": None, "missing": [MISSING_TELEMETRY],
    }
    summary_events = sampler.to_record()["events"]["phone"]
    assert summary_events["status"] == RULE_NOT_EVALUABLE
    assert summary_events["missing"] == [MISSING_TELEMETRY]


def test_an_older_phone_build_reports_not_evaluable_not_quiet(tmp_path):
    """A build that predates this task never sends `thermal_status_changes` at
    all, decoding to None. Driven through `sample_once`, not `.latest()` set
    by hand, since `_process_phone_events` is what decides this.
    """
    phone = _FakePhone(telemetry=_telemetry(thermal_status_changes=None))
    sampler = ThermalSampler(sink=None, phone=phone, clock=_Clock(100.0))
    sampler.sample_once()

    tick_events = sampler.latest(now=100.0)["events"]["phone"]
    assert tick_events == {
        "status": RULE_NOT_EVALUABLE, "count": 0, "last": None, "missing": [MISSING_STATUS_CHANGES],
    }


def test_the_populated_phone_branches_are_reached_through_sample_once(tmp_path):
    """`_accumulate_phone_summary`'s and `_write_sample`'s populated-phone
    branches, and `_latest_phone_events`'s fired/quiet branch, are only
    reached by driving `sample_once` against a real telemetry object.
    """
    phone = _FakePhone(telemetry_at_mono=50.0)
    sink = _Sink()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=sink, phone=phone, clock=clock)

    phone.telemetry = _telemetry(
        thermal_status="severe", skin_temp_c=41.5, skin_temp_zone="xo_therm",
        thermal_headroom_absent="not_a_number", thermal_status_changes=0,
    )
    sampler.sample_once()

    record = sampler.to_record()["phone"]
    assert record["samples"] == 1
    assert record["status_counts"] == {"severe": 1}
    assert record["skin_temp_c"]["p50"] == pytest.approx(41.5)
    assert record["skin_zone"] == "xo_therm"
    assert record["headroom_absent_counts"] == {"not_a_number": 1}

    sample_line = next(r for r in sink.records if r["type"] == "thermal_sample")
    assert sample_line["phone"]["status"] == "severe"
    assert sample_line["phone"]["skin_temp_c"] == pytest.approx(41.5)
    assert sample_line["phone"]["absent"] is None

    tick_events = sampler.latest(now=clock.now)["events"]["phone"]
    assert tick_events["status"] == RULE_QUIET  # reported and evaluable, zero transitions so far


# M3: the 1 Hz thread itself, and its own exception handler.


def test_starting_the_real_thread_produces_samples(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    sink = _Sink()
    sampler = ThermalSampler(sink=sink, jetson=JetsonThermal(root=root), interval_s=0.05)
    sampler.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(sink.records) < 3:
            time.sleep(0.02)
    finally:
        sampler.stop()
    assert len(sink.records) >= 3
    assert all(r["type"] == "thermal_sample" for r in sink.records)


def test_the_loop_stops_itself_when_a_sample_raises(tmp_path):
    """`_loop`'s exception handler is the only producer of the 'thread died'
    half of `ABSENT_SAMPLER_STOPPED` -- proven on the real thread, not by
    calling `sample_once` directly.
    """

    class _ExplodingJetson(JetsonThermal):
        def read_zones(self):
            raise RuntimeError("sysfs blew up")

    sampler = ThermalSampler(sink=None, jetson=_ExplodingJetson(root=tmp_path), interval_s=0.02)
    sampler.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and sampler._running:
        time.sleep(0.02)
    # Read before stop(), which sets `_running` False itself regardless of
    # whether the loop's own exception handler already did -- checking after
    # stop() would not tell the two apart.
    still_running_after_the_raise = sampler._running
    sampler.stop()

    assert still_running_after_the_raise is False
    reading = sampler.latest(now=time.monotonic())["jetson"]
    assert reading["reason"] == ABSENT_SAMPLER_STOPPED


# M4: the hottest-zone fallback is the entire mitigation for the guessed Orin
# zone names.


def test_select_zone_falls_back_to_the_hottest_when_no_preferred_name_matches(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "unexpected-a", "30000")
    _zone(root, 1, "unexpected-b", "55000")
    _zone(root, 2, "unexpected-c", "40000")
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()

    reading = sampler.latest(now=100.0)["jetson"]
    assert reading["zone"] == "unexpected-b"  # the hottest of the three
    assert reading["temp_c"] == pytest.approx(55.0)
    assert sampler.to_record()["jetson"]["selected_by"] == "hottest_at_first_sample"


# M5: summary["thermal"]'s own accumulators, asserted against the records
# that produced them.


def test_summary_thermal_accumulators_are_assertable_end_to_end(tmp_path):
    root = tmp_path / "root"
    jetson = JetsonThermal(root=root)
    phone = _FakePhone()
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, phone=phone, jetson=jetson, clock=clock)

    sampler.sample_once()  # pass 1: absent -- the root does not exist yet

    _zone(root, 0, "cpu-thermal", "40000")
    _zone(root, 1, "gpu-thermal", "45000")
    phone.telemetry = _telemetry(thermal_status="nominal", thermal_headroom_absent="not_a_number")
    clock.advance(1.0)
    sampler.sample_once()  # pass 2: measured on two zones, phone present

    _zone(root, 0, "cpu-thermal", "42000")
    _zone(root, 1, "gpu-thermal", "48000")
    phone.telemetry = None
    clock.advance(1.0)
    sampler.sample_once()  # pass 3: hotter, phone silent again

    record = sampler.to_record()

    jetson_record = record["jetson"]
    assert jetson_record["samples"] == 3, "every attempted pass, not only the measured ones"
    assert jetson_record["basis_counts"]["absent"] == 1
    assert jetson_record["basis_counts"]["measured"] == 2
    assert jetson_record["absent_reasons"] == {ABSENT_NO_THERMAL_ROOT: 1}
    assert jetson_record["zones_seen"] == ["cpu-thermal", "gpu-thermal"]
    assert jetson_record["per_zone_max_c"] == {
        "cpu-thermal": pytest.approx(42.0), "gpu-thermal": pytest.approx(48.0),
    }
    assert jetson_record["temp_c"]["p50"] == pytest.approx(41.0)  # held zone: 40.0, 42.0
    assert jetson_record["temp_c"]["p95"] == pytest.approx(41.9)

    phone_record = record["phone"]
    assert phone_record["samples"] == 1, "only the pass with a real telemetry object"
    assert phone_record["absent_counts"] == {ABSENT_NO_TELEMETRY: 2}, "passes 1 and 3"
    assert phone_record["headroom_absent_counts"] == {"not_a_number": 1}


def test_the_summary_event_record_always_carries_a_missing_list_even_when_empty(tmp_path):
    """Unlike the tick-level record, the summary-level one always carries
    `missing`, even empty, because the summary is read on its own with no
    sibling ticks to compare it against.
    """
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    _cooling(root, 0, "pwm-fan", "0")
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()

    events = sampler.to_record()["events"]["jetson"]
    assert events["status"] == RULE_QUIET
    assert events["missing"] == []


def test_a_null_headroom_with_no_stated_reason_is_still_counted(tmp_path):
    """A null `thermal_headroom` with no `thermal_headroom_absent` used to be
    counted nowhere, so `headroom_absent_counts == {}` could mean either
    'always answered' or 'never answered and never said why'.
    """
    phone = _FakePhone(telemetry=_telemetry(thermal_headroom=None, thermal_headroom_absent=None))
    sampler = ThermalSampler(sink=None, phone=phone, clock=_Clock(100.0))
    sampler.sample_once()

    assert sampler.to_record()["phone"]["headroom_absent_counts"] == {HEADROOM_ABSENT_UNSPECIFIED: 1}


# M8: the disappearing held zone gets its own reason, not the "nothing at all
# was plausible" one.


def test_a_disappearing_held_zone_is_not_no_zone_readable(tmp_path):
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    for i in range(1, 10):
        _zone(root, i, f"zone{i}", "41000")
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()
    assert sampler.latest(now=100.0)["jetson"]["zone"] == "cpu-thermal"

    (root / "thermal_zone0" / "type").unlink()  # the held zone vanishes; nine others still read
    sampler.sample_once()

    record = sampler.to_record()["jetson"]
    assert record["absent_reasons"] == {ABSENT_ZONE_DISAPPEARED: 1}
    assert ABSENT_NO_ZONE_READABLE not in record["absent_reasons"]
    assert len(record["zones_seen"]) == 10  # the nine that still read, plus the one that vanished


# M9: the phone status line names every status seen, not only the modal one.


def test_report_line_names_every_phone_status_not_only_the_modal_one(tmp_path):
    from eval_run import analyze, render_markdown
    from tests.test_eval_run import make_tick, write_run

    root = tmp_path / "sysroot"
    _zone(root, 0, "cpu-thermal", "40000")
    clock = _Clock(100.0)
    phone = _FakePhone()
    sampler = ThermalSampler(sink=None, phone=phone, jetson=JetsonThermal(root=root), clock=clock)

    for status, n in (("nominal", 100), ("severe", 78)):
        for _ in range(n):
            phone.telemetry = _telemetry(thermal_status=status)
            sampler.sample_once()
            clock.advance(1.0)

    run_dir = write_run(tmp_path, [make_tick(0)], summary={
        "ticks": 1, "camera_dropped_frames": 0, "policy_trained": False,
        "thermal": sampler.to_record(jtop_available=None),
    })
    result = analyze(run_dir)
    md = render_markdown(result, [])
    assert "severe" in md
    assert "nominal" in md
    # The counts themselves, not only that both names appear: the line is
    # sorted by descending count, so the more frequent status is named first.
    assert "nominal 100" in md
    assert "severe 78" in md
    assert md.index("nominal 100") < md.index("severe 78")


def test_the_not_evaluable_line_names_the_pass_counts(tmp_path):
    from eval_run import analyze, render_markdown
    from tests.test_eval_run import make_tick, write_run

    root = tmp_path / "sysroot"
    _zone(root, 0, "cpu-thermal", "40000")
    _cooling(root, 0, "pwm-fan", "0")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: readable

    (root / "cooling_device0" / "cur_state").unlink()
    clock.advance(1.0)
    sampler.sample_once()  # pass 2: unreadable
    clock.advance(1.0)
    sampler.sample_once()  # pass 3: unreadable

    run_dir = write_run(tmp_path, [make_tick(0)], summary={
        "ticks": 1, "camera_dropped_frames": 0, "policy_trained": False,
        "thermal": sampler.to_record(jtop_available=None),
    })
    md = render_markdown(analyze(run_dir), [])
    assert "(1 of 3 passes fully readable)" in md


def test_the_fired_line_names_the_pass_counts_when_incomplete(tmp_path):
    """C2's tick/summary fix reaches `eval_run`'s rendering too: a real count
    observed under incomplete observation must say so on the "fired" line,
    not only on the "not evaluable" one.
    """
    from eval_run import analyze, render_markdown
    from tests.test_eval_run import make_tick, write_run

    root = tmp_path / "sysroot"
    _cooling(root, 0, "pwm-fan", "0")
    clock = _Clock(100.0)
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=clock)
    sampler.sample_once()  # pass 1: readable, state 0

    _cooling(root, 0, "pwm-fan", "1")
    clock.advance(1.0)
    sampler.sample_once()  # pass 2: readable, state 1 -- fires

    (root / "cooling_device0" / "cur_state").unlink()
    clock.advance(1.0)
    sampler.sample_once()  # pass 3: unreadable

    run_dir = write_run(tmp_path, [make_tick(0)], summary={
        "ticks": 1, "camera_dropped_frames": 0, "policy_trained": False,
        "thermal": sampler.to_record(jtop_available=None),
    })
    md = render_markdown(analyze(run_dir), [])
    assert "throttle events, jetson: fired -- 1 transitions (2 of 3 passes fully readable)" in md


# MINOR: the two percentile conventions in the same report section must agree.


def test_the_two_percentile_conventions_agree():
    # n = 8, not 7: with an odd sample count, `0.5 * (n - 1)` is always a
    # whole number, so p50 lands on a real value under both a nearest-rank
    # and a linearly-interpolated convention and the assertion below cannot
    # tell the two apart. An even count makes both p50 and p95 land between
    # two ranks, so both assertions actually exercise the interpolation.
    values = [40.0, 41.5, 42.0, 47.5, 50.0, 33.2, 48.9, 45.0]
    a = _pctl(values)
    b = eval_run_pctl(values)
    assert a["min"] == pytest.approx(b["min"])
    assert a["mean"] == pytest.approx(b["mean"])
    assert a["p50"] == pytest.approx(b["p50"])
    assert a["p95"] == pytest.approx(b["p95"])
    assert a["max"] == pytest.approx(b["max"])


# MINOR: a log whose sampler wrote records but whose summary was never
# written still gets a "## Thermal" section.


def test_thermal_lines_render_something_when_the_summary_is_missing():
    from eval_run import _thermal_lines

    lines = _thermal_lines({"summary": None, "ticks_by_basis": {"measured": 2, "stale": 168}, "events": []})
    text = "\n".join(lines)
    assert "## Thermal" in text
    assert "stale 168" in text
    assert "no summary was written" in text


# m7: `basis_counts` carries no "stale" entry that is 0 by construction.


def test_basis_counts_has_no_entry_that_can_never_be_incremented(tmp_path):
    """`stale` is a `ThermalReading.basis` value `_latest_jetson` assigns only
    when its own caller asks for a reading -- never while a sample is being
    taken, which is the only time this accumulator is touched. There is no
    code path that could ever move it, so it does not belong in the JSON
    where it would read as a measurement of something that happens.
    """
    root = tmp_path / "root"
    _zone(root, 0, "cpu-thermal", "40000")
    sampler = ThermalSampler(sink=None, jetson=JetsonThermal(root=root), clock=_Clock(100.0))
    sampler.sample_once()
    assert set(sampler.to_record()["jetson"]["basis_counts"]) == {
        thermal.THERMAL_BASIS_MEASURED, thermal.THERMAL_BASIS_ABSENT,
    }


# m8: `_thermal_lines` must not raise on a record that carries
# `passes_attempted` without its sibling `passes_readable`.


def test_thermal_lines_do_not_crash_on_a_partial_passes_record():
    from eval_run import _thermal_lines

    lines = _thermal_lines({
        "summary": {
            "jetson": {"samples": 0, "temp_c": None},
            "phone": {"samples": 0},
            "events": {
                "jetson": {
                    "status": RULE_NOT_EVALUABLE, "count": 0,
                    "missing": [MISSING_COOLING_STATE], "passes_attempted": 2,
                },
                "phone": {"status": RULE_QUIET, "count": 0},
            },
        },
        "ticks_by_basis": {},
    })
    assert any("NOT EVALUABLE" in line for line in lines)


# -- the no-rate-change contract: the controller is byte-identical ----------


def test_the_controller_is_byte_identical_after_this_task():
    """Task 34's round-2 method: replay a large number of randomized `Inputs`
    through `SensingController.decide` and hash every `Decision.to_record()`.
    This task does not touch `sensing_controller.py` at all, so the digest
    below must be exact -- it is the one test that would catch D1 (folding the
    Jetson's temperature into `_thermal_scale`) being taken by accident.
    """
    state = {"t": 1000.0}

    def _advancing_clock() -> float:
        state["t"] += 0.05
        return state["t"]

    controller = SensingController(clock=_advancing_clock)
    rng = random.Random(1337)
    records = []
    for _ in range(15_000):
        inputs = Inputs(
            ego_acceleration=rng.choice([None, rng.uniform(-4, 4)]),
            ego_speed=rng.uniform(0, 30),
            policy_margin=rng.choice([None, rng.uniform(0, 1)]),
            feed_congestion=rng.choice([None, rng.uniform(0, 1)]),
            camera_density_bin=rng.choice([None, rng.randint(0, 5)]),
            feed_declined=rng.choice([None, "stale"]),
            thermal_status=rng.choice(
                [None, "nominal", "light", "moderate", "severe", "critical", "unknown"]
            ),
            skin_temp_c=rng.choice([None, rng.uniform(20, 50)]),
            telemetry_age_s=rng.choice([None, rng.uniform(0, 20)]),
            ego_acceleration_source="measured",
            ego_speed_source="measured",
            camera_density_bin_source="measured",
            camera_last_detection_age_s=rng.choice([None, rng.uniform(0, 5)]),
            lat=rng.uniform(-90, 90),
            lon=rng.uniform(-180, 180),
            position_valid=rng.choice([True, False]),
            position_age_s=rng.choice([None, rng.uniform(0, 5)]),
        )
        records.append(controller.decide(inputs).to_record())
    digest = hashlib.md5(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
    assert digest == "c884167b75931f2c07db6a5d2983d982"
