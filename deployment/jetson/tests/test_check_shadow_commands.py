"""check_shadow_commands.py: the shadow/live command comparison (task 43,
steps 43.2-43.7) and the phone-side logcat parser (43.5-43.6).

Drives are built from a real `SensingLoop`, the same convention
`test_score_shadow.py` uses and for the same reason: the `sensing` block a
test scores here must be byte-for-byte what `run_demo.py` actually writes.
Unlike that file's fixture, every tick here also carries the top-level
`t_capture_mono_ns` key `Tick.to_record()` writes -- `command_for` needs it
and `TickOutcome.to_record()` alone does not carry it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

import check_shadow_commands as csc
from policy.advisory import Advisory
from policy.sensing_controller import Inputs, SensingController
from policy.sensing_loop import SensingLoop
from policy.shadow_mode import LIVE, SHADOW, ModeHolder, command_for
from sensors.time_sync import capture_stamp_ns
from transport.messages import ACTION_HEADS


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeGps:
    lat: float = 51.49
    lon: float = -0.20
    valid: bool = True


@dataclass
class FakeObs:
    obs: dict
    feed: Any = None
    diagnostics: dict = field(default_factory=lambda: {"gps_age_s": 0.4})
    field_sources: dict = field(default_factory=dict)


@dataclass
class FakePolicy:
    head_probs: dict


@dataclass
class FakeTick:
    obs_result: FakeObs
    policy: FakePolicy
    gps: FakeGps
    advisory: Advisory
    t_capture_mono: float = 1000.0
    tick_id: int = 0


def _advisory() -> Advisory:
    return Advisory(
        recommended_speed_mps=13.4, recommended_speed_display=30.0,
        current_speed_display=28.0, units="mph", headway_target_s=2.0,
        lane_text="keep lane", merge_text="no merge", traffic_text="moderate",
        confidence=0.8, confidence_label="high",
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"})


def _tick(tick_id: int, *, accel: float = 0.0, density: int = 2) -> FakeTick:
    obs = {"ego_acceleration": accel, "ego_speed": 20.0, "local_density_bin": float(density)}
    return FakeTick(
        obs_result=FakeObs(obs=obs),
        policy=FakePolicy(head_probs={head: [0.95, 0.05] for head in ACTION_HEADS}),
        gps=FakeGps(),
        advisory=_advisory(),
        t_capture_mono=1000.0 + tick_id,
        tick_id=tick_id,
    )


class Phone:
    def __init__(self) -> None:
        self.telemetry = None
        self.telemetry_at_mono = None

    def send_advisory(self, *a, **k):
        return True

    def send_rate_command(self, *a, **k):
        return True


def write_shadow_drive(
    tmp_path: Path, n: int = 12, *, mode: str = SHADOW, accel_ticks: frozenset = frozenset(),
    rate_cmd_sent: int | None = None, name: str = "drive",
    t_wall_start: float | None = None, log_health_t_wall: float | None = None,
    wall_clock_offset_s: float | None = None, sessions: list | None = None,
) -> Path:
    """A run directory with both `metadata.jsonl` and a `summary.json`
    naming the drive's mode -- `check()` needs the latter to know what
    `sensing.shadow` is supposed to say.

    `t_wall_start`/`log_health_t_wall`, given, also write a `t_wall` on
    every tick (one second apart, starting there) and a `log_health.json`
    -- what `_run_window` (A2, validation round 1) needs. `None` (the
    default) omits both, matching every test written before that check
    existed.

    `wall_clock_offset_s`, given, writes `summary["phone"]["wall_clock_
    offset_s"]` -- what `_run_window` shifts the window by (B10, validation
    round 2). `None` omits the key, matching a run from before that fix or
    one with no phone session.

    `sessions`, given, writes `summary["phone"]["sessions"]` -- the
    FINISHED sessions before the current one, each with its own
    `wall_clock_offset_s` (B15, validation round 3), simulating a run that
    rebound.
    """
    clock = Clock()
    modes = ModeHolder(mode, clock=clock)
    loop = SensingLoop(clock=clock, modes=modes)
    phone = Phone()
    lines = []
    for i in range(n):
        clock.advance(0.1)
        accel = 3.0 if i in accel_ticks else 0.0
        tick = _tick(i, accel=accel)
        outcome = loop.on_tick(tick, phone)
        record = {
            "type": "tick",
            "tick_id": i,
            "t_capture_mono_ns": capture_stamp_ns(tick.t_capture_mono),
            "sensing": outcome.to_record(),
        }
        if t_wall_start is not None:
            record["t_wall"] = t_wall_start + i
        lines.append(record)

    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
    sent = rate_cmd_sent if rate_cmd_sent is not None else loop.to_record()["rate_commands_sent"]
    phone_record: dict[str, Any] = {"wire": {"channels": {"rate_cmd": {"sent": sent}}}}
    if wall_clock_offset_s is not None:
        phone_record["wall_clock_offset_s"] = wall_clock_offset_s
    if sessions is not None:
        phone_record["sessions"] = sessions
    (run_dir / "summary.json").write_text(json.dumps({
        "sensing": loop.to_record(),
        "phone": phone_record,
    }))
    if log_health_t_wall is not None:
        (run_dir / "log_health.json").write_text(json.dumps({"t_wall": log_health_t_wall}))
    return run_dir


# -- 43.2/43.3: the command replay, on a real recorded drive ----------------


def test_check_agrees_on_a_small_recorded_shadow_drive(tmp_path):
    run_dir = write_shadow_drive(tmp_path, n=15, accel_ticks=frozenset({5, 6}))
    result = csc.check(run_dir)
    assert "refused" not in result
    assert result["drive_mode"] == SHADOW
    assert result["ticks"] == 15
    assert result["command_replay"]["mismatched"] == 0
    assert result["logged_shadow_flag"]["ok"] is True
    assert result["overall_ok"] is True


def test_check_agrees_on_a_live_drive_too(tmp_path):
    run_dir = write_shadow_drive(tmp_path, n=10, mode=LIVE)
    result = csc.check(run_dir)
    assert result["drive_mode"] == LIVE
    assert result["logged_shadow_flag"]["expected_shadow"] is False
    assert result["overall_ok"] is True


def test_check_refuses_a_run_missing_the_capture_stamp(tmp_path):
    """`test_score_shadow.py`'s own fixture omits this key -- a `sensing`-only
    log has nothing for `command_for` to be called with."""
    run_dir = tmp_path / "no_capture"
    run_dir.mkdir()
    clock = Clock()
    loop = SensingLoop(clock=clock, modes=ModeHolder(SHADOW, clock=clock))
    phone = Phone()
    clock.advance(0.1)
    outcome = loop.on_tick(_tick(0), phone)
    with open(run_dir / "metadata.jsonl", "w") as f:
        f.write(json.dumps({"type": "tick", "tick_id": 0, "sensing": outcome.to_record()}) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"sensing": loop.to_record()}))
    result = csc.check(run_dir)
    assert result["refused"] == csc.REFUSAL_CAPTURE_STAMP_ABSENT
    assert result["first_tick_id"] == 0


def test_check_refuses_when_the_drive_mode_cannot_be_read(tmp_path):
    run_dir = write_shadow_drive(tmp_path, n=3)
    (run_dir / "summary.json").write_text(json.dumps({}))
    result = csc.check(run_dir)
    assert result["refused"] == csc.REFUSAL_DRIVE_MODE_UNKNOWN


def test_check_refuses_a_run_with_no_metadata(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    result = csc.check(run_dir)
    assert result["refused"] == csc.REFUSAL_NO_METADATA


# -- 43.4: the logged shadow flag against the drive's own recorded mode -----


def test_logged_shadow_flag_catches_a_flipped_column():
    clock = Clock()
    controller = SensingController(clock=clock)
    clock.advance(0.1)
    decision = controller.decide(Inputs(ego_speed=20.0))
    sensing_ticks = [
        {"tick_id": 0, "sensing": {"shadow": True}},
        {"tick_id": 1, "sensing": {"shadow": False}},  # corrupted: this drive is SHADOW
    ]
    result = csc.check_logged_shadow_flag(sensing_ticks, SHADOW)
    assert result["ok"] is False
    assert result["mismatched_tick_ids"] == [1]
    del decision  # unused; the fixture only needs the two literal dicts above


# -- 43.2/43.3, compare_tick_commands directly -------------------------------


def _replayed_decision(inputs: Inputs) -> Any:
    clock = Clock()
    controller = SensingController(clock=clock)
    clock.advance(0.1)
    return controller.decide(inputs)


def test_compare_tick_commands_agrees_when_command_for_is_unmodified():
    decision = _replayed_decision(Inputs(ego_speed=20.0, lat=51.49, lon=-0.20))
    tick = {"tick_id": 0, "t_capture_mono_ns": 123}
    result = csc.compare_tick_commands(tick, decision)
    assert result == {"tick_id": 0, "ok": True, "mismatches": []}


# -- 43.8: pin the comparison as a test, and mutate command_for -------------


class TestMutationsAreKilled:
    """Each mutation named in the plan (43.8), applied to `command_for`
    itself and confirmed caught -- either by `compare_tick_commands`
    directly, or (the aliasing mutation) by a property pinned separately,
    because a snapshot value comparison cannot see a shared reference that
    nothing has mutated yet.
    """

    def test_dropping_the_query_on_the_shadow_branch_is_caught(self, monkeypatch):
        decision = _replayed_decision(Inputs(ego_speed=20.0, lat=51.49, lon=-0.20))
        assert decision.here_query is not None, "fixture must exercise a real query"

        real_command_for = command_for

        def mutant(decision, mode, *, t_capture_mono_ns):
            cmd = real_command_for(decision, mode, t_capture_mono_ns=t_capture_mono_ns)
            if mode == SHADOW:
                cmd = replace(cmd, here=None)
            return cmd

        monkeypatch.setattr(csc, "command_for", mutant)
        result = csc.compare_tick_commands({"tick_id": 0, "t_capture_mono_ns": 1}, decision)
        assert result["ok"] is False
        assert "here" in result["mismatches"]

    def test_returning_the_same_shadow_flag_for_both_modes_is_caught(self, monkeypatch):
        decision = _replayed_decision(Inputs(ego_speed=20.0))

        real_command_for = command_for

        def mutant(decision, mode, *, t_capture_mono_ns):
            # Ignores the mode entirely: both calls come back live.
            return real_command_for(decision, LIVE, t_capture_mono_ns=t_capture_mono_ns)

        monkeypatch.setattr(csc, "command_for", mutant)
        result = csc.compare_tick_commands({"tick_id": 0, "t_capture_mono_ns": 1}, decision)
        assert result["ok"] is False
        assert "shadow" in result["mismatches"]

    def test_command_for_does_not_alias_the_rates_dict_between_calls(self):
        """"Copy rates by reference": if `command_for` built `rates=decision.rates`
        instead of `rates=dict(decision.rates)`, both commands would share
        one dict, and mutating one after construction would silently
        corrupt the other -- a defect no snapshot-value comparison would
        ever see, because at the moment of comparison the values still
        agree. Pinned directly, the way `test_shadow_mode.py:73`'s identity
        check pins the equivalent property for `here`.
        """
        decision = _replayed_decision(Inputs(ego_speed=20.0))
        shadow_cmd = command_for(decision, SHADOW, t_capture_mono_ns=1)
        live_cmd = command_for(decision, LIVE, t_capture_mono_ns=1)
        assert shadow_cmd.rates is not live_cmd.rates
        shadow_cmd.rates["camera_hz"] = -999.0
        assert live_cmd.rates["camera_hz"] != -999.0


# -- Phone-side: parse_config_applier_stats / check_phone_applier ----------


#: Verbatim off ZY227VV4XC during task 42's smoke run.
REAL_LOGCAT_LINE = (
    "09-05 01:13:50.408 20794 20794 I SensingService: config applier stats "
    "Stats(applied=0, shadowed=7, lastTrigger=advisory_margin_narrow, "
    "currentRates={}, hereConfigured=false)"
)


def test_parse_config_applier_stats_reads_the_real_device_line():
    stats = csc.parse_config_applier_stats(REAL_LOGCAT_LINE)
    assert stats == {
        "applied": 0, "shadowed": 7, "last_trigger": "advisory_margin_narrow",
        "current_rates": {}, "here_configured": False,
    }


def test_parse_config_applier_stats_reads_populated_rates_and_null_trigger():
    line = (
        "config applier stats Stats(applied=3, shadowed=0, lastTrigger=null, "
        "currentRates={camera_hz=2.0, gps_hz=1.0, imu_hz=50.0, here_hz=0.5}, "
        "hereConfigured=true)"
    )
    stats = csc.parse_config_applier_stats(line)
    assert stats == {
        "applied": 3, "shadowed": 0, "last_trigger": None,
        "current_rates": {"camera_hz": 2.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.5},
        "here_configured": True,
    }


def test_parse_config_applier_stats_is_none_when_the_line_never_appeared():
    assert csc.parse_config_applier_stats("nothing relevant in this buffer\n") is None


def test_parse_config_applier_stats_takes_the_last_line_after_a_rebind():
    two_teardowns = (
        "config applier stats Stats(applied=1, shadowed=1, lastTrigger=first, "
        "currentRates={}, hereConfigured=false)\n"
        "config applier stats Stats(applied=5, shadowed=9, lastTrigger=second, "
        "currentRates={}, hereConfigured=false)\n"
    )
    stats = csc.parse_config_applier_stats(two_teardowns)
    assert stats["applied"] == 5 and stats["shadowed"] == 9


# -- A2 (validation round 1): windowed logcat parsing -----------------------


#: Verbatim off ZY227VV4XC, `-v epoch` format, the same line
#: REAL_LOGCAT_LINE carries in the older (unwindowed) `-v time` format.
REAL_EPOCH_LOGCAT_LINE = (
    "         1788585230.408 20794 20794 I SensingService: config applier stats "
    "Stats(applied=0, shadowed=7, lastTrigger=advisory_margin_narrow, "
    "currentRates={}, hereConfigured=false)"
)


def test_parse_config_applier_stats_reads_a_real_epoch_line_inside_its_window():
    stats = csc.parse_config_applier_stats(
        REAL_EPOCH_LOGCAT_LINE, window=(1788585200.0, 1788585260.0),
    )
    assert stats is not None and stats["shadowed"] == 7


def test_parse_config_applier_stats_with_window_ignores_a_line_outside_it():
    """The exact defect (A2): the buffer holds a real line, and without a
    window it would be the only match -- but it is from a run that is not
    this one."""
    stats = csc.parse_config_applier_stats(
        REAL_EPOCH_LOGCAT_LINE, window=(1788589000.0, 1788589060.0),
    )
    assert stats is None


def test_parse_config_applier_stats_with_window_picks_the_in_window_line_from_two():
    """Run 1's teardown line an hour before run 3's, both still in the
    buffer -- the shape A2 was reported against ("the 01:13 line was still
    the only match at 02:10"). Windowed, run 1's own check must not pick up
    run 3's line even though it is chronologically the last match."""
    run1_line = (
        "         1788585230.408 20794 20794 I SensingService: config applier "
        "stats Stats(applied=0, shadowed=7, lastTrigger=a, currentRates={}, "
        "hereConfigured=false)"
    )
    run3_line = (
        "         1788588830.408 20794 20794 I SensingService: config applier "
        "stats Stats(applied=0, shadowed=99, lastTrigger=b, currentRates={}, "
        "hereConfigured=false)"
    )
    buffer = run1_line + "\n" + run3_line + "\n"
    stats = csc.parse_config_applier_stats(buffer, window=(1788585190.0, 1788585260.0))
    assert stats is not None and stats["shadowed"] == 7


def test_parse_config_applier_stats_with_window_takes_the_last_match_inside_it():
    two_in_window = (
        "         1788585210.0 1 1 I SensingService: config applier stats "
        "Stats(applied=1, shadowed=1, lastTrigger=first, currentRates={}, "
        "hereConfigured=false)\n"
        "         1788585220.0 1 1 I SensingService: config applier stats "
        "Stats(applied=5, shadowed=9, lastTrigger=second, currentRates={}, "
        "hereConfigured=false)\n"
    )
    stats = csc.parse_config_applier_stats(two_in_window, window=(1788585200.0, 1788585260.0))
    assert stats["applied"] == 5 and stats["shadowed"] == 9


# -- _run_window -------------------------------------------------------------


def _write_minimal_run(
    tmp_path, *, first_tick_t_wall, log_health_t_wall, name="run",
    wall_clock_offset_s=None, sessions=None,
):
    """`sessions`, given, writes `summary["phone"]["sessions"]` -- one
    entry per FINISHED session, each carrying its own `wall_clock_offset_s`
    (B15, validation round 3), simulating a run that rebound at least once.
    `wall_clock_offset_s` is always the CURRENT (last, still-active) session's
    own reading; `sessions[0]` names the FIRST one, which `_run_window`
    compares it against.
    """
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        f.write(json.dumps({"type": "tick", "tick_id": 0, "t_wall": first_tick_t_wall}) + "\n")
    if log_health_t_wall is not None:
        (run_dir / "log_health.json").write_text(json.dumps({"t_wall": log_health_t_wall}))
    if wall_clock_offset_s is not None or sessions is not None:
        phone_record: dict[str, Any] = {}
        if wall_clock_offset_s is not None:
            phone_record["wall_clock_offset_s"] = wall_clock_offset_s
        if sessions is not None:
            phone_record["sessions"] = sessions
        (run_dir / "summary.json").write_text(json.dumps({"phone": phone_record}))
    return run_dir


def test_run_window_reads_first_tick_and_log_health_plus_margin(tmp_path):
    """The real numbers from task 42's smoke run: first tick t_wall
    1788585200.572, log_health.json t_wall 1788585229.66 -- the phone's own
    teardown line landed at 1788585230.408, 0.75s after log_health's own
    timestamp and comfortably inside `[start, end + margin]`."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1788585200.572, log_health_t_wall=1788585229.6617892,
    )
    window = csc._run_window(run_dir)
    assert window == (1788585200.572, 1788585229.6617892 + csc.RUN_WINDOW_MARGIN_S)
    start, end = window
    assert start <= 1788585230.408 <= end


def test_run_window_shifts_by_the_recorded_wall_clock_offset(tmp_path):
    """B10 (validation round 2): `start`/`end` are on the Jetson's clock as
    recorded, but the logcat lines this window is matched against are
    timestamped on the phone's -- shifted by the measured offset so both
    sides of the comparison are expressed in the same clock's terms."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=0.935,
    )
    window = csc._run_window(run_dir)
    assert window == (1000.935, 1010.935 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_shifts_negative_when_the_phone_is_behind(tmp_path):
    """The one-sided-margin defect this fixes: an unshifted window
    tolerates a phone running behind for the whole run's duration and one
    running ahead nowhere past margin_s. Shifting by a negative offset
    moves BOTH bounds down, which a fixed one-sided margin cannot do."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=-45.0,
    )
    window = csc._run_window(run_dir)
    assert window == (955.0, 965.0 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_falls_back_to_unshifted_when_no_offset_was_recorded(tmp_path):
    """A run from before B10 (no summary.json at all) or one with no phone
    session (offset never set) gets the prior, unshifted behaviour rather
    than refusing outright."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
    )
    window = csc._run_window(run_dir)
    assert window == (1000.0, 1010.0 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_shift_closes_the_beyond_margin_contamination_case(tmp_path):
    """A2's own failure returning "one layer down": with an offset large
    enough that the unshifted window would have missed this run's own
    line, the shift recovers it."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=45.0,  # bigger than RUN_WINDOW_MARGIN_S (30s)
    )
    window = csc._run_window(run_dir)
    start, end = window
    # The phone's own teardown line, ~0.75s after log_health's Jetson-clock
    # t_wall PLUS the phone's offset -- outside the unshifted window
    # ([1000, 1040]), inside the shifted one.
    phone_clock_teardown_line = 1010.0 + 45.0 + 0.75
    assert not (1000.0 <= phone_clock_teardown_line <= 1010.0 + csc.RUN_WINDOW_MARGIN_S)
    assert start <= phone_clock_teardown_line <= end


def test_run_window_refuses_when_sessions_disagree_beyond_the_margin(tmp_path):
    """B15 (validation round 3): a rebind's two sessions can each have
    measured a different phone. `start` fell inside the FIRST session
    (offset 0.9); the CURRENT session's offset (46.0) is what
    `wall_clock_offset_s` reports post-rebind -- 45.1s apart, more than
    `RUN_WINDOW_MARGIN_S` (30s), so one shift cannot be trusted for both
    halves of the window and this refuses rather than silently using the
    current session's offset for the whole thing."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=46.0, sessions=[{"wall_clock_offset_s": 0.9}],
    )
    assert csc._run_window(run_dir) is None


def test_run_window_uses_the_current_session_when_offsets_agree_within_margin(tmp_path):
    """The ordinary case: a rebind happened, but both sessions measured
    close to the same phone-Jetson offset (0.9 vs 0.95, 0.05s apart) --
    well within the margin, so the CURRENT (last, covering `end`) session's
    offset is used for the whole window, same as a run with no rebind at
    all."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=0.95, sessions=[{"wall_clock_offset_s": 0.9}],
    )
    window = csc._run_window(run_dir)
    assert window == (1000.95, 1010.95 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_with_an_empty_sessions_list_behaves_like_no_rebind(tmp_path):
    """A run with `sessions: []` (no rebind, `PhoneLink.to_record`'s own
    docstring: `sessions` holds every session BEFORE the current one) has
    nothing to compare the current offset against, and must not refuse."""
    run_dir = _write_minimal_run(
        tmp_path, first_tick_t_wall=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=0.935, sessions=[],
    )
    window = csc._run_window(run_dir)
    assert window == (1000.935, 1010.935 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_skips_a_leading_non_tick_record(tmp_path):
    """A run that opened with a `failure_event` has one ahead of tick 0;
    the window's start is the first TICK's t_wall, not the first line's."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        f.write(json.dumps({"type": "failure_event", "t_wall": 1.0}) + "\n")
        f.write(json.dumps({"type": "tick", "tick_id": 0, "t_wall": 100.0}) + "\n")
    (run_dir / "log_health.json").write_text(json.dumps({"t_wall": 200.0}))
    window = csc._run_window(run_dir)
    assert window == (100.0, 200.0 + csc.RUN_WINDOW_MARGIN_S)


def test_run_window_is_none_without_log_health(tmp_path):
    run_dir = _write_minimal_run(tmp_path, first_tick_t_wall=1.0, log_health_t_wall=None)
    assert csc._run_window(run_dir) is None


def test_run_window_is_none_without_any_tick(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.jsonl").write_text("")
    (run_dir / "log_health.json").write_text(json.dumps({"t_wall": 1.0}))
    assert csc._run_window(run_dir) is None


def test_run_window_is_none_without_metadata_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert csc._run_window(run_dir) is None


# -- _phone_session_offsets (B15, validation round 3) ------------------------


def test_phone_session_offsets_with_no_rebind_reports_the_same_value_twice():
    first, current = csc._phone_session_offsets({"wall_clock_offset_s": 0.935})
    assert first == 0.935
    assert current == 0.935


def test_phone_session_offsets_with_a_rebind_reports_the_first_sessions_own_value():
    first, current = csc._phone_session_offsets({
        "wall_clock_offset_s": 46.0,
        "sessions": [{"wall_clock_offset_s": 0.9}],
    })
    assert first == 0.9
    assert current == 46.0


def test_phone_session_offsets_with_multiple_rebinds_uses_the_earliest_session():
    first, current = csc._phone_session_offsets({
        "wall_clock_offset_s": 3.0,
        "sessions": [{"wall_clock_offset_s": 1.0}, {"wall_clock_offset_s": 2.0}],
    })
    assert first == 1.0
    assert current == 3.0


def test_phone_session_offsets_is_none_none_with_no_phone_summary_at_all():
    assert csc._phone_session_offsets(None) == (None, None)
    assert csc._phone_session_offsets({}) == (None, None)


# -- pull_config_applier_stats: uses -v epoch and the given window ----------


class RecordingRunAdb:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.calls: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, args, *, capture_output, text, timeout):
        self.calls.append(list(args))

        class Result:
            pass

        result = Result()
        result.stdout = self.stdout
        result.returncode = self.returncode
        return result


def test_pull_config_applier_stats_uses_v_epoch():
    run = RecordingRunAdb(stdout=REAL_EPOCH_LOGCAT_LINE + "\n")
    stats = csc.pull_config_applier_stats(
        "ZY227VV4XC", (1788585200.0, 1788585260.0), run=run,
    )
    assert stats is not None and stats["shadowed"] == 7
    assert run.calls == [
        ["adb", "-s", "ZY227VV4XC", "logcat", "-d", "-v", "epoch", "-s", "SensingService:I"]
    ]


def test_pull_config_applier_stats_respects_the_window():
    run = RecordingRunAdb(stdout=REAL_EPOCH_LOGCAT_LINE + "\n")
    stats = csc.pull_config_applier_stats(
        "ZY227VV4XC", (1788589000.0, 1788589060.0), run=run,
    )
    assert stats is None


def test_pull_config_applier_stats_is_none_when_adb_is_unreachable():
    def raising_run(args, *, capture_output, text, timeout):
        raise FileNotFoundError("no adb")

    assert csc.pull_config_applier_stats(
        "ZY227VV4XC", (0.0, 1.0), run=raising_run,
    ) is None


def test_check_phone_applier_passes_on_the_real_smoke_run_numbers():
    """applied=0, shadowed=7 against the Jetson's own rate_cmd.sent=7 --
    task 42's actual smoke-run numbers."""
    result = csc.check_phone_applier(
        {"applied": 0, "shadowed": 7}, drive_mode=SHADOW, commands_sent=7,
    )
    assert result["ok"] is True


def test_check_phone_applier_fails_if_anything_was_applied_on_a_shadow_drive():
    result = csc.check_phone_applier(
        {"applied": 1, "shadowed": 6}, drive_mode=SHADOW, commands_sent=7,
    )
    assert result["ok"] is False


def test_check_phone_applier_fails_if_shadowed_undercounts_what_was_sent():
    """A dropped or misrouted command would let shadowed fall behind sent
    without applied moving at all -- caught only by comparing against the
    Jetson's own count, not by applied alone."""
    result = csc.check_phone_applier(
        {"applied": 0, "shadowed": 5}, drive_mode=SHADOW, commands_sent=7,
    )
    assert result["ok"] is False


def test_check_phone_applier_on_a_live_drive_defers_to_task_44():
    """B3, validation round 1: the live-mode branch was unpinned in the
    failing direction (mutating `ok = shadowed == 0` to `ok = True` left
    the suite green) and never read `applied` at all. Rather than assert
    `applied == commands_sent` -- a live-mode correctness claim this task
    gathered no evidence for -- it returns `ok: None` regardless, deferring
    to task 44, which the module docstring for `check_phone_applier` now
    states."""
    passing_numbers = csc.check_phone_applier(
        {"applied": 7, "shadowed": 0}, drive_mode=LIVE, commands_sent=7,
    )
    assert passing_numbers["ok"] is None
    assert passing_numbers["applied"] == 7
    assert passing_numbers["shadowed"] == 0

    # The failing direction the old test never exercised at all: something
    # WAS shadowed, and applied fell short of commands_sent. Still `None`,
    # not silently `True` -- a mutation collapsing this branch to a
    # constant `True` is caught by the `is None` on both cases, not just
    # one.
    bad_numbers = csc.check_phone_applier(
        {"applied": 3, "shadowed": 4}, drive_mode=LIVE, commands_sent=7,
    )
    assert bad_numbers["ok"] is None
    assert bad_numbers["applied"] == 3
    assert bad_numbers["shadowed"] == 4


def test_rate_cmd_commands_sent_reads_the_transport_channel_counter():
    summary = {"phone": {"wire": {"channels": {"rate_cmd": {"sent": 7}}}}}
    assert csc.rate_cmd_commands_sent(summary) == 7


def test_rate_cmd_commands_sent_is_none_when_absent():
    assert csc.rate_cmd_commands_sent({}) is None


# -- 43.7: cross-check the same conclusion from the Jetson alone ------------


def test_a_pure_shadow_drive_is_witnessed_three_independent_ways(tmp_path):
    """The phone-side check (43.6) is the load-bearing half; this pins that
    three OTHER, independent readings of the same log agree with it, all
    from the Jetson's own record, none of them this module's own code:

      structurally_absent   `ModeHolder.to_record()` (policy/shadow_mode.py)
      attribution.rules     `SensingController`'s own per-tick attribution
      _comparable            `eval_run.py`'s achieved-vs-commanded gate

    A log where these three disagreed would mean the mode record, the
    controller's own attribution, and the reporting layer had drifted apart
    -- each is a different piece of code reaching the same fact.
    """
    from eval_run import _comparable

    run_dir = write_shadow_drive(tmp_path, n=20)
    loaded_summary = json.loads((run_dir / "summary.json").read_text())
    mode_record = loaded_summary["sensing"]["mode"]
    assert mode_record["structurally_absent"] == ["feed_congestion", "source_disagreement"]

    ticks = [json.loads(line) for line in (run_dir / "metadata.jsonl").read_text().splitlines()]
    statuses = {t["sensing"]["attribution"]["rules"]["source_disagreement"]["status"] for t in ticks}
    assert statuses == {"not_evaluable"}

    comparable, reason = _comparable(ever_live=False, distinct_reports=[], ticks=len(ticks))
    assert comparable is False
    assert "mode shadow" in reason


# -- A1 (validation round 1): overall_ok must fold in every outcome --------


def test_apply_phone_applier_check_not_requested_leaves_overall_ok_untouched(tmp_path):
    run_dir = write_shadow_drive(tmp_path, n=5, t_wall_start=1000.0, log_health_t_wall=1005.0)
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial=None, run_dir=run_dir)
    assert out["phone_applier"] == {"ok": None, "detail": "not requested"}
    assert out["overall_ok"] is True


def test_apply_phone_applier_check_reproduces_the_reported_defect(tmp_path, monkeypatch):
    """The validator's own repro: `--serial NOSUCHSERIAL` against a real
    run reported `phone_applier ok=False` and still exited 0, because the
    line folding the verdict into `overall_ok` lived inside the branch
    that only runs when `pull_config_applier_stats` returns something --
    not the branch where it returns `None`. Reproduced here by making
    `pull_config_applier_stats` return `None`, exactly what an
    unreachable/nonexistent serial does.
    """
    run_dir = write_shadow_drive(tmp_path, n=5, t_wall_start=1000.0, log_health_t_wall=1005.0)
    monkeypatch.setattr(csc, "pull_config_applier_stats", lambda serial, window: None)
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="NOSUCHSERIAL", run_dir=run_dir)
    assert out["phone_applier"]["ok"] is False
    assert out["overall_ok"] is False, (
        "the exact defect: phone_applier.ok=False must fail overall_ok, not "
        "leave it at whatever check() computed"
    )


def test_apply_phone_applier_check_window_unavailable_fails_overall_ok(tmp_path):
    run_dir = tmp_path / "no_window"
    run_dir.mkdir()  # no metadata.jsonl, no log_health.json at all
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["ok"] is False
    assert "window" in out["phone_applier"]["detail"]
    assert out["overall_ok"] is False


def test_apply_phone_applier_check_passes_and_names_the_window(tmp_path, monkeypatch):
    run_dir = write_shadow_drive(
        tmp_path, n=5, rate_cmd_sent=7, t_wall_start=1000.0, log_health_t_wall=1005.0,
    )
    seen_windows = []

    def fake_pull(serial, window):
        seen_windows.append(window)
        return {"applied": 0, "shadowed": 7}

    monkeypatch.setattr(csc, "pull_config_applier_stats", fake_pull)
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["ok"] is True
    assert out["overall_ok"] is True
    assert seen_windows == [(1000.0, 1005.0 + csc.RUN_WINDOW_MARGIN_S)]


def test_apply_phone_applier_check_no_match_names_the_window_in_the_detail(tmp_path, monkeypatch):
    run_dir = write_shadow_drive(tmp_path, n=5, t_wall_start=1000.0, log_health_t_wall=1005.0)
    monkeypatch.setattr(csc, "pull_config_applier_stats", lambda serial, window: None)
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    detail = out["phone_applier"]["detail"]
    assert "1000.000" in detail and f"{1005.0 + csc.RUN_WINDOW_MARGIN_S:.3f}" in detail


def test_apply_phone_applier_check_live_drive_does_not_change_overall_ok(tmp_path, monkeypatch):
    """`ok: None` (B3) must not corrupt a real bool: `True and None` is
    `None`, not `True`, so folding it in naively would turn a passing
    live-mode command_replay/shadow_flag result into a falsy-but-not-False
    `overall_ok`."""
    run_dir = write_shadow_drive(
        tmp_path, n=5, mode=LIVE, t_wall_start=1000.0, log_health_t_wall=1005.0,
    )
    monkeypatch.setattr(
        csc, "pull_config_applier_stats", lambda serial, window: {"applied": 5, "shadowed": 0},
    )
    result = {"overall_ok": True, "drive_mode": LIVE}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["ok"] is None
    assert out["overall_ok"] is True


# -- B14 (validation round 3): the offset used is named in the artifact -----


def test_apply_phone_applier_check_names_the_offset_it_used(tmp_path, monkeypatch):
    run_dir = write_shadow_drive(
        tmp_path, n=5, rate_cmd_sent=7, t_wall_start=1000.0, log_health_t_wall=1005.0,
        wall_clock_offset_s=0.935,
    )
    monkeypatch.setattr(
        csc, "pull_config_applier_stats", lambda serial, window: {"applied": 0, "shadowed": 7},
    )
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["wall_clock_offset_s"] == 0.935


def test_apply_phone_applier_check_names_none_when_no_offset_was_recorded(tmp_path, monkeypatch):
    """A genuine 0.0 offset and an absent one must not read alike: this run
    has no phone session at all (no `wall_clock_offset_s` key), so the
    window fell back to unshifted -- `None`, not `0.0`, records that
    nothing was measured (B14)."""
    run_dir = write_shadow_drive(
        tmp_path, n=5, rate_cmd_sent=7, t_wall_start=1000.0, log_health_t_wall=1005.0,
    )
    monkeypatch.setattr(
        csc, "pull_config_applier_stats", lambda serial, window: {"applied": 0, "shadowed": 7},
    )
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["wall_clock_offset_s"] is None


def test_apply_phone_applier_check_names_a_genuine_zero_offset(tmp_path, monkeypatch):
    """The other half of the same distinction: a MEASURED zero offset
    (the two devices' clocks agreed exactly) must read as `0.0`, not
    `None` -- `is None` in `_run_window`, not `or 0.0`, is what keeps the
    two apart."""
    run_dir = write_shadow_drive(
        tmp_path, n=5, rate_cmd_sent=7, t_wall_start=1000.0, log_health_t_wall=1005.0,
        wall_clock_offset_s=0.0,
    )
    monkeypatch.setattr(
        csc, "pull_config_applier_stats", lambda serial, window: {"applied": 0, "shadowed": 7},
    )
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["wall_clock_offset_s"] == 0.0


def test_apply_phone_applier_check_window_unavailable_names_no_offset(tmp_path):
    run_dir = tmp_path / "no_window"
    run_dir.mkdir()  # no metadata.jsonl, no log_health.json at all
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["wall_clock_offset_s"] is None


# -- B15 (validation round 3): a rebind whose sessions disagree -------------


def test_apply_phone_applier_check_refuses_and_names_both_sessions_offsets(tmp_path):
    run_dir = write_shadow_drive(
        tmp_path, n=5, t_wall_start=1000.0, log_health_t_wall=1010.0,
        wall_clock_offset_s=46.0, sessions=[{"wall_clock_offset_s": 0.9}],
    )
    result = {"overall_ok": True, "drive_mode": SHADOW}
    out = csc.apply_phone_applier_check(result, serial="ZY227VV4XC", run_dir=run_dir)
    assert out["phone_applier"]["ok"] is False
    assert out["overall_ok"] is False
    assert out["phone_applier"]["wall_clock_offset_s"] is None
    detail = out["phone_applier"]["detail"]
    assert "0.900000" in detail and "46.000000" in detail and "rebound" in detail
