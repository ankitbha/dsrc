"""score_shadow.py over synthetic logs: the replay-identity gate, refusals,
segments, and the three-state scoring discipline the whole task exists for.

Drives are built from a real `SensingLoop` rather than hand-built dicts, so
the `sensing` block each test scores is the same shape `run_demo.py` writes
-- the task 34 lesson that the emitted record is the one worth testing,
applied here to the log this tool reads rather than the log `sensing_loop.py`
writes.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import pytest

import score_shadow
from perception.feed_fusion import FeedOwnership
from policy.advisory import Advisory
from policy.sensing_controller import (
    RULE_FIRED,
    RULE_NOT_EVALUABLE,
    RULE_QUIET,
    RULES,
    RuleCheck,
    SensingController,
    Trigger,
)
from policy.sensing_loop import SensingLoop
from policy.shadow_mode import LIVE, SHADOW, ModeHolder
from transport.messages import ACTION_HEADS, PhoneTelemetry


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


def _tick(tick_id: int, *, accel=0.0, density=2, feed=None) -> FakeTick:
    return FakeTick(
        obs_result=FakeObs(obs={"ego_acceleration": accel, "ego_speed": 20.0,
                                "local_density_bin": float(density)}, feed=feed),
        policy=FakePolicy(head_probs={
            head: [0.95, 0.05] for head in ACTION_HEADS}),
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


def _telemetry(i: int, *, thermal_status: str = "nominal") -> PhoneTelemetry:
    return PhoneTelemetry(
        t_capture_mono_ns=0, thermal_status=thermal_status, thermal_headroom=None,
        achieved={"camera_hz": 4.97, "gps_hz": 1.0, "imu_hz": 49.8, "here_hz": 0.0},
        dropped={"camera": i, "gps": 0, "imu": 0, "here": 0},
        here_calls=0, here_errors=0,
    )


def _jammed_feed() -> FeedOwnership:
    return FeedOwnership(downstream_congestion=0.9, free_flow_mps=30.0, age_s=1.0)


def write_run(tmp_path: Path, n: int = 10, *, feed_ticks: frozenset = frozenset(),
              live_from: int | None = None, telemetry: bool = True,
              telemetry_at: frozenset[int] | None = None,
              accel_ticks: frozenset = frozenset(),
              thermal_status_at: Callable[[int], str] | None = None,
              name: str = "run") -> Path:
    """A run directory built from a real `SensingLoop`, so every `sensing`
    block scored here is byte-for-byte what `TickOutcome.to_record()` emits.

    `telemetry_at`, given, restricts the ticks on which `phone.telemetry`
    is (re)assigned to that set -- everywhere else, whatever was last
    assigned stays in place, exactly as `PhoneLink.telemetry` behaves when
    the reporting thread has died. `telemetry=True` with `telemetry_at=None`
    keeps the old behaviour: a fresh report on every tick.
    """
    clock = Clock()
    modes = ModeHolder(SHADOW, clock=clock)
    loop = SensingLoop(clock=clock, modes=modes)
    phone = Phone()
    lines = []
    for i in range(n):
        clock.advance(0.1)
        if live_from is not None and i == live_from:
            modes.flip_to(LIVE)
        status = thermal_status_at(i) if thermal_status_at else "nominal"
        reports_this_tick = (i in telemetry_at) if telemetry_at is not None else telemetry
        if reports_this_tick:
            phone.telemetry = _telemetry(i, thermal_status=status)
            phone.telemetry_at_mono = clock.now - 0.02
        feed = _jammed_feed() if i in feed_ticks else None
        density = 0 if i in feed_ticks else 2
        accel = 3.0 if i in accel_ticks else 0.0
        outcome = loop.on_tick(_tick(i, accel=accel, feed=feed, density=density), phone)
        lines.append({"type": "tick", "tick_id": i, "sensing": outcome.to_record()})

    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
    return run_dir


def _promoted_drive(tmp_path: Path, n: int = 20, live_from: int = 10,
                     tick_id_base: int = 0, name: str = "promoted") -> Path:
    """A drive that starts in shadow and is promoted to live partway through,
    written without a `summary.json` so `_limits` takes its derived-from-ticks
    path rather than echoing a mode block. `tick_id_base` lets a caller give
    it logged ids that are not their own position in the log, the same way
    `write_run` does.
    """
    clock = Clock()
    modes = ModeHolder(SHADOW, clock=clock)
    loop = SensingLoop(clock=clock, modes=modes)
    phone = Phone()
    lines = []
    for i in range(n):
        clock.advance(0.1)
        if i == live_from:
            modes.flip_to(LIVE)
        phone.telemetry = _telemetry(i)
        phone.telemetry_at_mono = clock.now - 0.02
        lines.append({"type": "tick", "tick_id": tick_id_base + i,
                      "sensing": loop.on_tick(_tick(i), phone).to_record()})

    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
    return run_dir


class TestReplayClock:

    def test_reading_before_the_first_tick_is_set_raises(self):
        clock = score_shadow.ReplayClock()
        with pytest.raises(RuntimeError):
            clock()

    def test_the_set_value_is_returned_on_every_call_until_set_again(self):
        clock = score_shadow.ReplayClock()
        clock.set(5.0)
        assert clock() == 5.0 == clock() == clock()
        clock.set(6.0)
        assert clock() == 6.0


class TestRefusals:
    """A log that cannot referee anyone, including its own incumbent, refuses
    by name rather than scoring an approximation."""

    def test_a_run_dir_with_no_metadata_jsonl_refuses_by_name(self, tmp_path):
        # The plan's own named example run, device-test-2026-08-25, is exactly
        # this shape: a phone-side session.jsonl and no Jetson metadata log.
        run_dir = tmp_path / "phone_only"
        run_dir.mkdir()
        (run_dir / "session.jsonl").write_text('{"dir": "in"}\n')
        result = score_shadow.score(run_dir)
        assert result["refused"] == score_shadow.REFUSAL_NO_METADATA

    def test_main_exits_2_on_a_missing_metadata_jsonl(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "phone_only"
        run_dir.mkdir()
        monkeypatch.setattr(sys, "argv", ["score_shadow.py", str(run_dir)])
        assert score_shadow.main() == 2
        assert not (run_dir / "shadow_score.json").exists()

    def test_no_tick_records_at_all(self, tmp_path):
        run_dir = tmp_path / "empty"
        run_dir.mkdir()
        (run_dir / "metadata.jsonl").write_text("")
        result = score_shadow.score(run_dir)
        assert result["refused"] == score_shadow.REFUSAL_NO_TICKS

    def test_a_phoneless_run_has_no_sensing_block_anywhere(self, tmp_path):
        run_dir = tmp_path / "phoneless"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({"type": "tick", "tick_id": 0}) + "\n")
        result = score_shadow.score(run_dir)
        assert result["refused"] == score_shadow.REFUSAL_PHONELESS

    def test_a_pre_task_35_log_has_sensing_but_no_decision_inputs(self, tmp_path):
        run_dir = tmp_path / "pre35"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({
                "type": "tick", "tick_id": 0,
                "sensing": {"rates": {}, "trigger": "idle", "shadow": True},
            }) + "\n")
        result = score_shadow.score(run_dir)
        assert result["refused"] == score_shadow.REFUSAL_PRE_TASK_35

    def test_main_exits_2_and_writes_nothing_on_a_refusal(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "phoneless"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({"type": "tick", "tick_id": 0}) + "\n")
        monkeypatch.setattr(sys, "argv", ["score_shadow.py", str(run_dir)])
        assert score_shadow.main() == 2
        assert not (run_dir / "shadow_score.json").exists()


class TestReplayIdentityGate:

    def test_an_intact_log_replays_with_zero_mismatches(self, tmp_path):
        run_dir = write_run(tmp_path, n=12)
        result = score_shadow.score(run_dir)
        assert result["replay_identity"] == {
            "status": "ok", "ticks": 12, "mismatched": 0, "first_mismatch": None,
        }

    def test_a_corrupted_tick_fails_the_gate_and_names_it(self, tmp_path):
        run_dir = write_run(tmp_path, n=12)
        lines = (run_dir / "metadata.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        records[5]["sensing"]["decision_inputs"]["ego_speed"] = 999.0
        with open(run_dir / "metadata.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = score_shadow.score(run_dir, {"copy": lambda clock: SensingController(clock=clock)})
        assert result["replay_identity"]["status"] == "failed"
        assert result["replay_identity"]["mismatched"] >= 1
        assert result["replay_identity"]["first_mismatch"]["tick_id"] == 5
        assert "keys" in result["replay_identity"]["first_mismatch"]
        assert "candidates" not in result

    def test_main_exits_2_on_a_failed_replay(self, tmp_path, monkeypatch):
        run_dir = write_run(tmp_path, n=6)
        lines = (run_dir / "metadata.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        records[2]["sensing"]["decision_inputs"]["ego_speed"] = 999.0
        with open(run_dir / "metadata.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        monkeypatch.setattr(sys, "argv", ["score_shadow.py", str(run_dir)])
        assert score_shadow.main() == 2


class TestLogCompleteness:
    """`summary["sensing"]["ticks"]` is the tick count `SensingLoop` itself
    recorded; `len(sensing_ticks)` is what this scorer read back off the
    log. A truncated log can carry fewer records than the run it came from
    reported, with everything else about it -- `unparseable_lines`,
    `replay_identity` -- reading clean.
    """

    def test_none_without_a_recorded_tick_count(self, tmp_path):
        run_dir = write_run(tmp_path, n=10)  # no summary.json written at all
        result = score_shadow.score(run_dir)
        assert result["log_completeness"] is None

    def test_a_complete_log_reports_zero_missing(self, tmp_path):
        run_dir = write_run(tmp_path, n=10)
        (run_dir / "summary.json").write_text(json.dumps({"sensing": {"ticks": 10}}))
        result = score_shadow.score(run_dir)
        assert result["log_completeness"] == {
            "ticks_recorded": 10, "ticks_scored": 10, "ticks_missing": 0,
        }

    def test_a_truncated_log_reports_the_shortfall_against_the_runs_own_count(self, tmp_path):
        # The run's own `SensingLoop` ran for 13 ticks; only 10 made it into
        # the log this scorer can read -- the shape of a log whose tail was
        # lost after `close()` had already written `summary.json`.
        run_dir = write_run(tmp_path, n=10)
        (run_dir / "summary.json").write_text(json.dumps({"sensing": {"ticks": 13}}))
        result = score_shadow.score(run_dir)
        assert result["log_completeness"] == {
            "ticks_recorded": 13, "ticks_scored": 10, "ticks_missing": 3,
        }
        table = score_shadow.render_table(result)
        assert "recorded=13" in table
        assert "scored=10" in table
        assert "missing=3" in table


class TestSegments:

    def test_a_pure_shadow_drive_is_reference_throughout(self, tmp_path):
        run_dir = write_run(tmp_path, n=10)
        result = score_shadow.score(run_dir)
        assert result["segments"] == {
            "reference_ticks": 10, "contaminated_ticks": 0, "first_live_tick_id": None,
        }
        assert result["limits"]["reference_rates_hold"] is True

    def test_a_drive_that_goes_live_at_tick_k_splits_there(self, tmp_path):
        run_dir = write_run(tmp_path, n=10, live_from=4)
        result = score_shadow.score(run_dir)
        assert result["segments"] == {
            "reference_ticks": 4, "contaminated_ticks": 6, "first_live_tick_id": 4,
        }
        assert result["limits"]["reference_rates_hold"] is False


class TestLimitsDerivedFromTicks:

    def test_a_drive_never_live_names_the_absent_inputs(self, tmp_path):
        run_dir = write_run(tmp_path, n=5)
        result = score_shadow.score(run_dir)
        assert result["limits"]["structurally_absent"] == ["feed_congestion", "source_disagreement"]
        assert result["limits"]["mode_derived_from_ticks"] is True

    def test_a_drive_born_live_names_nothing_absent(self, tmp_path):
        run_dir = write_run(tmp_path, n=5, live_from=0)
        result = score_shadow.score(run_dir)
        assert result["limits"]["structurally_absent"] == []

    def test_a_drive_promoted_partway_still_names_its_shadow_segments_absences(self, tmp_path):
        # `born_live` asks whether the FIRST tick was live, not whether any
        # tick ever was -- a drive promoted partway has a leading shadow
        # segment that genuinely had no feed in it, so the two feed-derived
        # inputs really are absent from that part of the log. Reading this as
        # "ever live" would report nothing absent, which is the exact error
        # `shadow_mode.ModeHolder.to_record`'s own comment names.
        run_dir = _promoted_drive(tmp_path, n=20, live_from=10)
        result = score_shadow.score(run_dir)
        assert "mode_derived_from_ticks" in result["limits"]
        assert result["limits"]["structurally_absent"] == ["feed_congestion", "source_disagreement"]
        assert result["limits"]["reference_rates_hold"] is False

    def test_summary_json_is_echoed_verbatim_when_present(self, tmp_path):
        run_dir = write_run(tmp_path, n=5)
        summary = {"sensing": {"mode": {
            "shadow_predicts": "the decision function, not the trajectory",
            "structurally_absent": [], "reference_rates_hold": False,
        }}}
        (run_dir / "summary.json").write_text(json.dumps(summary))
        result = score_shadow.score(run_dir)
        assert result["limits"] == {
            "shadow_predicts": "the decision function, not the trajectory",
            "structurally_absent": [], "reference_rates_hold": False,
        }


class TestReferenceWitness:

    def test_a_fully_reported_drive_counts_every_tick(self, tmp_path):
        # Every tick gets a fresh report here (`write_run`'s default), so this
        # is also the boundary case where the tick interval is no longer than
        # the telemetry interval: an age recomputed from `now` on every tick
        # settles to a near-constant value and never decreases, which is
        # exactly the drive an age-based arrival count misreads.
        run_dir = write_run(tmp_path, n=8, telemetry=True)
        result = score_shadow.score(run_dir)
        rw = result["reference_witness"]
        assert rw["ticks_with_achieved"] == 8
        assert rw["ticks_no_telemetry"] == 0
        assert rw["reports"] == 8
        assert rw["achieved_mean"] == {"camera_hz": 4.97, "gps_hz": 1.0,
                                       "imu_hz": 49.8, "here_hz": 0.0}
        assert rw["dropped_final"] == {"camera": 7, "gps": 0, "imu": 0, "here": 0}

    def test_a_drive_that_never_reported_counts_zero(self, tmp_path):
        run_dir = write_run(tmp_path, n=6, telemetry=False)
        result = score_shadow.score(run_dir)
        rw = result["reference_witness"]
        assert rw["ticks_with_achieved"] == 0
        assert rw["ticks_no_telemetry"] == 6
        assert rw["reports"] == 0
        assert rw["achieved_mean"] is None
        assert rw["dropped_final"] is None


class TestReferenceWitnessStalePredicateMatchesTheController:
    """`ticks_stale` has to answer exactly the question `_thermal_scale`
    answers about the same age (`sensing_controller.py`), or a report the
    incumbent discarded as too old could still be averaged into
    `achieved_mean` here, and the reverse. Built directly against
    `_reference_witness` rather than through a live drive, because
    `telemetry_age_s` comes off the Jetson's own monotonic clock -- neither
    a NaN nor a negative-beyond-bound age reaches this path through
    `reference_from` today.
    """

    @staticmethod
    def _reference(age_s, *, camera_hz=1.0, dropped=0, at_mono=0.0):
        return {
            "achieved": {"camera_hz": camera_hz, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.0},
            "dropped": {"camera": dropped, "gps": 0, "imu": 0, "here": 0},
            "age_s": age_s, "at_mono": at_mono, "absent": None,
        }

    def _ticks(self, ages):
        return [{"sensing": {"reference": self._reference(age, at_mono=float(i))}}
                for i, age in enumerate(ages)]

    def test_a_non_finite_age_does_not_poison_the_reported_ages(self):
        # `max()` and the mean propagate a NaN to the whole field, and `json.dumps`
        # writes it as a bare `NaN` that strict parsers refuse, so one unusable age
        # would cost a reader both numbers and the ability to load the file at all.
        witness = score_shadow._reference_witness(
            self._ticks([2.0, float("nan"), 4.0]))
        assert witness["age_s_max"] == 4.0
        assert witness["age_s_mean"] == pytest.approx(3.0)
        # Not silently forgotten: an age the controller could not use is stale.
        assert witness["ticks_stale"] == 1
        json.loads(json.dumps(witness))

    def test_every_known_age_lands_in_exactly_one_of_fresh_or_stale(self):
        # `_reference_witness` asserts this partition itself; reaching the
        # return statement at all is the assertion passing.
        rw = score_shadow._reference_witness(
            self._ticks([0.0, 5.0, 10.0, 10.1, -0.5, -60.0, float("inf"), float("nan")]))
        assert rw["ticks_with_achieved"] == 8

    def test_a_negative_age_far_beyond_the_bound_is_stale(self):
        rw = score_shadow._reference_witness(self._ticks([-60.0]))
        assert rw["ticks_stale"] == 1
        assert rw["achieved_mean"] is None

    def test_a_nan_age_is_stale_not_silently_dropped_from_both_partitions(self):
        rw = score_shadow._reference_witness(self._ticks([float("nan")]))
        assert rw["ticks_with_achieved"] == 1
        assert rw["ticks_stale"] == 1
        assert rw["achieved_mean"] is None


class TestReferenceWitnessExcludesStaleReports:
    """The defect class F1 is about: a witness field that is supposed to
    exclude stale reports has no way to prove it on a drive where every
    report is fresh, because the excluded and unexcluded computations then
    coincide. This drive's fresh and stale reports carry different numbers,
    so a field that fails to exclude the stale ones is observably wrong
    rather than accidentally right.
    """

    @staticmethod
    def _telemetry_at(camera_hz: float, dropped: int) -> PhoneTelemetry:
        return PhoneTelemetry(
            t_capture_mono_ns=0, thermal_status="nominal", thermal_headroom=None,
            achieved={"camera_hz": camera_hz, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.0},
            dropped={"camera": dropped, "gps": 0, "imu": 0, "here": 0},
            here_calls=0, here_errors=0,
        )

    def _write(self, tmp_path) -> Path:
        """Report A runs fresh for seven ticks, the last landing exactly on
        `MAX_TELEMETRY_AGE_S` (fresh, not stale -- the incumbent's own `>`).
        Report B arrives just before the tick loop stalls for 20 s, so every
        tick that ever observes B observes it already aged out: the last
        report and the last report the controller would have used are two
        different reports.
        """
        clock = Clock()
        loop = SensingLoop(clock=clock, modes=ModeHolder(SHADOW, clock=clock))
        phone = Phone()
        lines = []

        def emit(i):
            lines.append({"type": "tick", "tick_id": i,
                          "sensing": loop.on_tick(_tick(i), phone).to_record()})

        phone.telemetry = self._telemetry_at(9.0, 100)
        phone.telemetry_at_mono = clock.now
        for i in range(6):                                          # ages 0..5, fresh
            emit(i)
            clock.advance(1.0)
        clock.advance(score_shadow.MAX_TELEMETRY_AGE_S - 6.0)        # land exactly on the bound
        emit(6)                                                      # age == bound: fresh
        phone.telemetry = self._telemetry_at(1.0, 200)
        phone.telemetry_at_mono = clock.now
        clock.advance(20.0)                                          # the loop stalls
        for i in range(7, 12):                                       # ages 20..24, all stale
            emit(i)
            clock.advance(1.0)

        run_dir = tmp_path / "mixed_freshness"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        return run_dir

    def test_stale_reports_are_excluded_from_every_field_that_claims_to_exclude_them(self, tmp_path):
        rw = score_shadow.score(self._write(tmp_path))["reference_witness"]

        assert rw["ticks_with_achieved"] == 12
        assert rw["ticks_stale"] == 5
        # The stale reports ran at 1.0 Hz and the fresh ones at 9.0 Hz; a
        # mean that included the stale ticks would not be 9.0.
        assert rw["achieved_mean"]["camera_hz"] == 9.0
        # The last report overall is the 1.0 Hz one; the last report the
        # controller would have used is the 9.0 Hz one.
        assert rw["dropped_final"] == {"camera": 100, "gps": 0, "imu": 0, "here": 0}
        assert rw["age_s_max"] == pytest.approx(24.0)
        assert rw["age_s_mean"] < rw["age_s_max"]


class TestReferenceWitnessTellsALiveDriveFromADeadOne:
    """`PhoneLink.telemetry` holds the latest report and is cleared only on
    rebind, so a phone that reported once and then had its telemetry thread
    die is, by presence alone, indistinguishable from one reporting every
    tick -- both show `ticks_with_achieved` equal to the tick count. `age_s`
    is what actually tells the two drives apart.
    """

    def test_a_dead_phone_and_a_live_one_produce_different_witnesses(self, tmp_path):
        live = write_run(tmp_path, n=300, telemetry_at=frozenset(range(0, 300, 5)), name="live")
        dead = write_run(tmp_path, n=300, telemetry_at=frozenset({0}), name="dead")

        live_rw = score_shadow.score(live)["reference_witness"]
        dead_rw = score_shadow.score(dead)["reference_witness"]

        # The defect, reproduced: presence alone reports the same thing for
        # both drives.
        assert live_rw["ticks_with_achieved"] == 300
        assert dead_rw["ticks_with_achieved"] == 300

        assert live_rw != dead_rw
        assert live_rw["reports"] == 60
        assert dead_rw["reports"] == 1
        assert live_rw["ticks_stale"] == 0
        assert dead_rw["ticks_stale"] > 0
        assert live_rw["age_s_max"] < score_shadow.MAX_TELEMETRY_AGE_S
        assert dead_rw["age_s_max"] > score_shadow.MAX_TELEMETRY_AGE_S


class TestReferenceWitnessRespectsSegments:
    """The reference witness is a claim about the reference segment
    (`reference_rates_hold`'s own boundary, D9) -- a drive with zero
    reference ticks must not report a full-rate witness for a segment its
    own report calls zero ticks long.
    """

    def test_a_drive_born_live_has_a_zero_length_reference_witness(self, tmp_path):
        run_dir = write_run(tmp_path, n=5, live_from=0, telemetry=True)
        result = score_shadow.score(run_dir)
        assert result["segments"]["reference_ticks"] == 0
        assert result["reference_witness"] is None
        assert result["reference_witness_contaminated"]["ticks_with_achieved"] == 5

    def test_a_drive_that_goes_live_partway_splits_the_witness_too(self, tmp_path):
        run_dir = write_run(tmp_path, n=10, live_from=4, telemetry=True)
        result = score_shadow.score(run_dir)
        assert result["segments"]["reference_ticks"] == 4
        assert result["reference_witness"]["ticks_with_achieved"] == 4
        assert result["reference_witness_contaminated"]["ticks_with_achieved"] == 6

    def test_a_pure_shadow_drive_has_no_contaminated_witness(self, tmp_path):
        run_dir = write_run(tmp_path, n=5)
        result = score_shadow.score(run_dir)
        assert result["reference_witness_contaminated"] is None


class TestTheThreeStateScoringDiscipline:
    """The defect class this task exists for: a candidate that would differ on
    a rule it was never given the inputs to decide on must not score as
    agreeing on it.
    """

    def test_a_pure_shadow_drive_names_the_rule_never_exercised(self, tmp_path, monkeypatch):
        import policy.sensing_controller as sc

        run_dir = write_run(tmp_path, n=10)  # feed never present anywhere

        def strict_disagreement(clock):
            # A candidate that would decide differently on `source_disagreement`
            # if it were ever handed the inputs -- it never is, on this drive.
            monkeypatch.setattr(sc, "JAMMED_CONGESTION", 2.0)
            return sc.SensingController(clock=clock)

        result = score_shadow.score(run_dir, {"strict": strict_disagreement})
        c = result["candidates"]["strict"]
        per_rule = c["vs_incumbent"]["per_rule"][Trigger.DISAGREEMENT]
        assert per_rule == {"agree": 0, "differ": 0, "not_evaluable": 10}

        entry = next(e for e in c["rules_never_exercised"] if e["rule"] == Trigger.DISAGREEMENT)
        assert entry["ticks"] == 10
        # Only `feed_congestion` is actually missing on this drive:
        # `camera_density_bin` comes from the local observation, not the
        # traffic feed, and every tick here supplies one -- matching task
        # 34's device run, which named the same single field
        # (plan_task34.md:349-356).
        assert entry["missing"] == {"feed_congestion": 10}

    def test_a_partly_evaluable_drive_differs_only_where_it_could(self, tmp_path, monkeypatch):
        import policy.sensing_controller as sc

        run_dir = write_run(tmp_path, n=10, feed_ticks=frozenset({3, 4, 5}))

        def strict_disagreement(clock):
            monkeypatch.setattr(sc, "JAMMED_CONGESTION", 2.0)
            return sc.SensingController(clock=clock)

        result = score_shadow.score(run_dir, {"strict": strict_disagreement})
        c = result["candidates"]["strict"]
        per_rule = c["vs_incumbent"]["per_rule"][Trigger.DISAGREEMENT]
        assert per_rule["not_evaluable"] == 7
        assert per_rule["differ"] == 3
        assert per_rule["agree"] == 0
        assert per_rule["agree"] + per_rule["differ"] == 10 - per_rule["not_evaluable"]
        assert not any(e["rule"] == Trigger.DISAGREEMENT for e in c["rules_never_exercised"])

    def test_an_identical_candidate_agrees_on_every_evaluable_tick(self, tmp_path):
        run_dir = write_run(tmp_path, n=10, feed_ticks=frozenset({2, 3}))
        result = score_shadow.score(run_dir, {"copy": lambda clock: SensingController(clock=clock)})
        c = result["candidates"]["copy"]
        for rule in RULES:
            counts = c["vs_incumbent"]["per_rule"][rule]
            assert counts["differ"] == 0
        assert c["vs_incumbent"]["rates"]["differ"] == 0
        assert c["vs_incumbent"]["trigger"]["differ"] == 0


class TestRulesNeverExercisedIsLogLevel:
    """D7: a mandatory output, not a footnote inside a candidate's score --
    present even when zero candidates are supplied, because running the tool
    with none is exactly the log-validity check this states the result of.
    """

    def test_present_with_zero_candidates_on_a_pure_shadow_drive(self, tmp_path):
        run_dir = write_run(tmp_path, n=10)  # feed never present anywhere
        result = score_shadow.score(run_dir)
        assert result["candidates"] == {}
        entry = next(e for e in result["rules_never_exercised"] if e["rule"] == Trigger.DISAGREEMENT)
        assert entry["ticks"] == 10
        assert entry["missing"] == {"feed_congestion": 10}

    def test_rendered_unconditionally_with_zero_candidates(self, tmp_path):
        run_dir = write_run(tmp_path, n=6)
        result = score_shadow.score(run_dir)
        table = score_shadow.render_table(result)
        assert "RULE NEVER EXERCISED" in table
        assert Trigger.DISAGREEMENT in table


class TestCandidateWithoutAttribution:

    def test_a_candidate_that_cannot_report_attribution_is_refused_by_name(self, tmp_path):
        run_dir = write_run(tmp_path, n=5)

        class NoAttribution:
            def __init__(self, clock):
                pass

            def decide(self, inputs):
                return types.SimpleNamespace(to_record=lambda: {"rates": {}, "trigger": "idle"})

        result = score_shadow.score(run_dir, {"bad": NoAttribution})
        assert result["candidates"]["bad"] == {"refused": score_shadow.CANDIDATE_WITHOUT_ATTRIBUTION}

    def test_a_candidate_valid_on_tick_zero_but_malformed_later_is_refused(self, tmp_path):
        # The shape check used to inspect only tick 0 (score_shadow.py:279),
        # so a candidate malformed from the second tick on raised a raw
        # KeyError instead of refusing cleanly.
        run_dir = write_run(tmp_path, n=3)

        class ValidThenBroken:
            def __init__(self, clock):
                self._inner = SensingController(clock=clock)
                self.calls = 0

            def decide(self, inputs):
                self.calls += 1
                if self.calls == 1:
                    return self._inner.decide(inputs)
                return types.SimpleNamespace(to_record=lambda: {"rates": {}, "trigger": "idle"})

        result = score_shadow.score(run_dir, {"bad": ValidThenBroken})
        assert result["candidates"]["bad"] == {"refused": score_shadow.CANDIDATE_WITHOUT_ATTRIBUTION}

    def test_evaluating_past_a_missing_input_is_flagged_not_absorbed(self, tmp_path):
        # A candidate that reports `quiet` for a rule the incumbent could not
        # evaluate did not see more than the incumbent did on the same
        # `Inputs` -- it dropped the "missing input means not_evaluable"
        # discipline. That divergence must be visible beside the ordinary
        # `quiet` count, not silently folded into it.
        run_dir = write_run(tmp_path, n=6)  # feed never present anywhere

        class EvaluatesEverything:
            def __init__(self, clock):
                self._inner = SensingController(clock=clock)

            def decide(self, inputs):
                decision = self._inner.decide(inputs)
                rules = dict(decision.attribution.rules)
                for name, check in list(rules.items()):
                    if check.status == RULE_NOT_EVALUABLE:
                        rules[name] = RuleCheck(status=RULE_QUIET, evidence={})
                attribution = replace(decision.attribution, rules=rules)
                return replace(decision, attribution=attribution)

        result = score_shadow.score(run_dir, {"over_eager": EvaluatesEverything})
        c = result["candidates"]["over_eager"]
        assert c["rules"][Trigger.DISAGREEMENT][RULE_QUIET] == 6
        assert c["vs_incumbent"]["per_rule"][Trigger.DISAGREEMENT] == {
            "agree": 0, "differ": 0, "not_evaluable": 6,
        }
        assert c["candidate_evaluated_where_incumbent_could_not"][Trigger.DISAGREEMENT] == 6


class TestActivityBlock:
    """`activity.ticks_active` and `activity.raises` are emitted numbers in
    the plan's template that nothing named directly. Both are checked here
    against ground truth read straight off the drive's own logged decisions
    -- independent of the counting code under test -- on a drive with a
    known, unequal active/idle split.
    """

    def test_ticks_active_and_raises_match_the_logged_decisions(self, tmp_path):
        # Ticks 0-4: thermal backs off alone (no other raise rule fires).
        # Ticks 5-12: sustained hard acceleration, which both fires EVENT and
        # eventually dwells the rates active. Ticks 13-17: the feed
        # disagrees with the camera alone -- no acceleration, no thermal
        # backoff -- the only stretch on this drive where
        # `Trigger.DISAGREEMENT` is the sole raise rule firing, so a
        # candidate that stopped counting it as a raise would diverge here
        # even though it agrees with the incumbent on every rule's own
        # status. The three stretches are asymmetric by construction so an
        # active/idle inversion, or a dropped raise rule, cannot land on the
        # same count by coincidence.
        run_dir = write_run(
            tmp_path, n=18, accel_ticks=frozenset(range(5, 13)),
            feed_ticks=frozenset(range(13, 18)),
            thermal_status_at=lambda i: "severe" if i < 5 else "nominal",
        )
        lines = [json.loads(line) for line in (run_dir / "metadata.jsonl").read_text().splitlines()]

        expected_active = sum(1 for l in lines if l["sensing"]["attribution"]["gates"]["level"] == "active")
        expected_idle = len(lines) - expected_active
        assert 0 < expected_active < len(lines)
        assert expected_active != expected_idle

        raise_rules = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT)
        expected_raises = sum(
            1 for l in lines
            if any(l["sensing"]["attribution"]["rules"][r]["status"] == RULE_FIRED for r in raise_rules)
        )
        thermal_alone = [
            l for l in lines
            if l["sensing"]["attribution"]["rules"][Trigger.THERMAL]["status"] == RULE_FIRED
            and not any(l["sensing"]["attribution"]["rules"][r]["status"] == RULE_FIRED for r in raise_rules)
        ]
        # Otherwise a mutation that folds THERMAL into `raises` would land on
        # the same count as the correct one -- this is the whole point of
        # the thermal-only preamble above.
        assert thermal_alone
        disagreement_alone = [
            l for l in lines
            if l["sensing"]["attribution"]["rules"][Trigger.DISAGREEMENT]["status"] == RULE_FIRED
            and not any(l["sensing"]["attribution"]["rules"][r]["status"] == RULE_FIRED
                        for r in (Trigger.EVENT, Trigger.NARROW_MARGIN))
        ]
        # Otherwise a mutation that drops DISAGREEMENT from the raise rules
        # would land on the same count as the correct one: nothing else on
        # this drive fires alongside it.
        assert disagreement_alone

        result = score_shadow.score(run_dir, {"copy": lambda clock: SensingController(clock=clock)})
        assert result["replay_identity"]["status"] == "ok"
        c = result["candidates"]["copy"]
        assert c["activity"]["ticks_active"] == expected_active
        assert c["activity"]["raises"] == expected_raises


class TestVocabularyClosure:

    def test_every_rule_status_and_verdict_is_a_closed_member(self, tmp_path):
        run_dir = write_run(tmp_path, n=10, feed_ticks=frozenset({4}))
        result = score_shadow.score(run_dir, {"copy": lambda clock: SensingController(clock=clock)})
        c = result["candidates"]["copy"]
        for rule in RULES:
            assert set(c["rules"][rule]) == {RULE_FIRED, RULE_QUIET, RULE_NOT_EVALUABLE}
            assert set(c["vs_incumbent"]["per_rule"][rule]) == {"agree", "differ", "not_evaluable"}
        assert set(result["segments"]) == {"reference_ticks", "contaminated_ticks", "first_live_tick_id"}


class TestResolveCandidate:

    def test_splits_label_module_and_factory(self, monkeypatch):
        fake_module = types.SimpleNamespace(build=lambda clock: None)
        monkeypatch.setattr(
            score_shadow.importlib, "import_module",
            lambda name: fake_module if name == "my.module" else (_ for _ in ()).throw(AssertionError(name)),
        )
        label, factory = score_shadow._resolve_candidate("mine=my.module:build")
        assert label == "mine"
        assert factory is fake_module.build


class TestMainCLI:

    def test_no_json_skips_writing_the_file(self, tmp_path, monkeypatch):
        run_dir = write_run(tmp_path, n=5)
        monkeypatch.setattr(sys, "argv", ["score_shadow.py", str(run_dir), "--no-json"])
        assert score_shadow.main() == 0
        assert not (run_dir / "shadow_score.json").exists()

    def test_the_default_writes_shadow_score_json(self, tmp_path, monkeypatch):
        run_dir = write_run(tmp_path, n=5)
        monkeypatch.setattr(sys, "argv", ["score_shadow.py", str(run_dir)])
        assert score_shadow.main() == 0
        written = json.loads((run_dir / "shadow_score.json").read_text())
        assert written["replay_identity"]["status"] == "ok"

    def test_a_candidate_flag_reaches_the_result(self, tmp_path, monkeypatch, capsys):
        run_dir = write_run(tmp_path, n=5)
        monkeypatch.setattr(
            score_shadow, "_resolve_candidate",
            lambda spec: ("copy", lambda clock: SensingController(clock=clock)),
        )
        monkeypatch.setattr(sys, "argv",
                            ["score_shadow.py", str(run_dir), "--candidate", "copy=x:y", "--no-json"])
        assert score_shadow.main() == 0
        assert "candidate copy" in capsys.readouterr().out


class TestRenderTable:

    def test_a_refusal_renders_without_crashing(self):
        table = score_shadow.render_table({"run": "x", "refused": "phoneless_run"})
        assert "REFUSED" in table
        assert "phoneless_run" in table

    def test_a_rule_never_exercised_is_visible_even_when_agreement_is_total(self, tmp_path):
        run_dir = write_run(tmp_path, n=6)
        result = score_shadow.score(run_dir, {"copy": lambda clock: SensingController(clock=clock)})
        table = score_shadow.render_table(result)
        assert "RULE NEVER EXERCISED" in table
        assert Trigger.DISAGREEMENT in table

    def test_a_born_live_drive_renders_without_a_reference_segment_line(self, tmp_path):
        # `reference_witness` is None on this drive (zero reference ticks) --
        # the table must skip that line rather than subscript a null block,
        # the same way it already skips the contaminated line when that
        # segment is empty.
        run_dir = write_run(tmp_path, n=5, live_from=0)
        result = score_shadow.score(run_dir)
        table = score_shadow.render_table(result)
        assert "reference_witness (reference segment)" not in table
        assert "reference_witness (contaminated segment)" in table
