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
        run_dir = write_run(tmp_path, n=8, telemetry=True)
        result = score_shadow.score(run_dir)
        rw = result["reference_witness"]
        assert rw["ticks_with_achieved"] == 8
        assert rw["ticks_no_telemetry"] == 0
        assert rw["achieved_mean"] == {"camera_hz": 4.97, "gps_hz": 1.0,
                                       "imu_hz": 49.8, "here_hz": 0.0}
        assert rw["dropped_final"] == {"camera": 7, "gps": 0, "imu": 0, "here": 0}

    def test_a_drive_that_never_reported_counts_zero(self, tmp_path):
        run_dir = write_run(tmp_path, n=6, telemetry=False)
        result = score_shadow.score(run_dir)
        rw = result["reference_witness"]
        assert rw["ticks_with_achieved"] == 0
        assert rw["ticks_no_telemetry"] == 6
        assert rw["achieved_mean"] is None
        assert rw["dropped_final"] is None


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
        assert result["reference_witness"]["ticks_with_achieved"] == 0
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
        # eventually dwells the rates active. The split is asymmetric by
        # construction so an active/idle inversion cannot land on the same
        # count by coincidence.
        run_dir = write_run(
            tmp_path, n=13, accel_ticks=frozenset(range(5, 13)),
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
