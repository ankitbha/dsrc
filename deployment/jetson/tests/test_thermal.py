"""sensors.thermal: the Jetson's own temperature and both devices' throttle
events. Also pins the boundary of this task's one behaviour change on the
phone -- `Inputs` unchanged and the controller byte-identical -- since those
are exactly what would move if the change escaped its fence.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from pathlib import Path

import pytest

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
    JetsonThermal,
    ThermalSampler,
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


# -- 2. the four absence reasons are each reachable and distinct ------------


def test_the_four_absence_reasons_are_each_reachable_and_distinct(tmp_path):
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

    assert seen == ABSENT_REASONS
    assert len(seen) == 4


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
