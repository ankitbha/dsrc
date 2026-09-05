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
) -> Path:
    """A run directory with both `metadata.jsonl` and a `summary.json`
    naming the drive's mode -- `check()` needs the latter to know what
    `sensing.shadow` is supposed to say.
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
        lines.append({
            "type": "tick",
            "tick_id": i,
            "t_capture_mono_ns": capture_stamp_ns(tick.t_capture_mono),
            "sensing": outcome.to_record(),
        })

    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
    sent = rate_cmd_sent if rate_cmd_sent is not None else loop.to_record()["rate_commands_sent"]
    (run_dir / "summary.json").write_text(json.dumps({
        "sensing": loop.to_record(),
        "phone": {"wire": {"channels": {"rate_cmd": {"sent": sent}}}},
    }))
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


def test_check_phone_applier_on_a_live_drive_requires_nothing_shadowed():
    result = csc.check_phone_applier(
        {"applied": 7, "shadowed": 0}, drive_mode=LIVE, commands_sent=7,
    )
    assert result["ok"] is True


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
