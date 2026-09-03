"""Task 39's session summary: seven instrument axes, nine reconciliations,
the `## Session summary` / `## Sensing` sections, and the zero-tick branch.

Drives that need a real `sensing` block are built from a real `SensingLoop`
(the task 34 lesson `test_score_shadow.py` already applies to its own
fixtures), so the block each test reads is the same shape `run_demo.py`
writes rather than a hand-typed approximation of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from eval_run import (
    AXES,
    _tick_coverage,
    LoadedRecords,
    _axis_api_calls,
    _axis_failures,
    _axis_latency,
    _axis_provenance,
    _axis_rates,
    _axis_thermal,
    _axis_triggers,
    _census_and_violations,
    _here_calls_by_session,
    _read_log_health,
    _read_summary,
    _tally_rules_missing,
    _telemetry_window,
    _time_weighted_integral,
    _time_weighted_mean,
    load_records,
    main,
    reconciliations,
    render_markdown,
    render_session_summary_only,
    sensing_result,
    session_summary,
)
from policy.advisory import Advisory
from policy.sensing_controller import RULE_NOT_EVALUABLE
from policy.sensing_loop import SensingLoop
from policy.shadow_mode import LIVE, SHADOW, ModeHolder
from transport.messages import ACTION_HEADS, PhoneTelemetry

# --------------------------------------------------------------------------
# A real `sensing` block per tick, via a real SensingLoop -- see module
# docstring for why.
# --------------------------------------------------------------------------


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


def _tick(tick_id: int) -> FakeTick:
    return FakeTick(
        obs_result=FakeObs(obs={"ego_acceleration": 0.0, "ego_speed": 20.0,
                                 "local_density_bin": 2.0}),
        policy=FakePolicy(head_probs={head: [0.95, 0.05] for head in ACTION_HEADS}),
        gps=FakeGps(), advisory=_advisory(),
        t_capture_mono=1000.0 + tick_id, tick_id=tick_id,
    )


class Phone:
    def __init__(self) -> None:
        self.telemetry = None
        self.telemetry_at_mono = None

    def send_advisory(self, *a, **k):
        return True

    def send_rate_command(self, *a, **k):
        return True


def _telemetry(*, here_calls: int = 0, here_errors: int = 0) -> PhoneTelemetry:
    return PhoneTelemetry(
        t_capture_mono_ns=0, thermal_status="nominal", thermal_headroom=None,
        achieved={"camera_hz": 4.97, "gps_hz": 1.0, "imu_hz": 49.8, "here_hz": 0.0},
        dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0},
        here_calls=here_calls, here_errors=here_errors,
    )


def sensing_ticks(
    n: int, *, mode: str = SHADOW, telemetry_at: frozenset | None = None,
    here_calls_at: dict[int, int] | None = None, session_id_at: dict[int, Any] | None = None,
    telemetry_age_offset: float = 0.02, tick_id_base: int = 0,
) -> list[dict[str, Any]]:
    """`n` tick dicts, each carrying a real `sensing` block plus a top-level
    `session_id` (assigned independently of `SensingLoop`, matching
    `run_demo.py`, which reads it off `phone.session` rather than the
    decision).

    `telemetry_at`, given, restricts which ticks receive a fresh telemetry
    report (the rest keep whatever was last assigned, as `PhoneLink.telemetry`
    does between reports). `here_calls_at` overrides that tick's reported
    `here_calls`; the default is monotonically increasing by 1 per report.
    """
    clock = Clock()
    modes = ModeHolder(mode, clock=clock)
    loop = SensingLoop(clock=clock, modes=modes)
    phone = Phone()
    out = []
    for i in range(n):
        clock.advance(0.1)
        reports_this_tick = (i in telemetry_at) if telemetry_at is not None else True
        if reports_this_tick:
            calls = (here_calls_at or {}).get(i, i)
            phone.telemetry = _telemetry(here_calls=calls)
            phone.telemetry_at_mono = clock.now - telemetry_age_offset
        outcome = loop.on_tick(_tick(i), phone)
        record: dict[str, Any] = {
            "type": "tick", "tick_id": tick_id_base + i, "sensing": outcome.to_record(),
        }
        if session_id_at is not None:
            record["session_id"] = session_id_at.get(i, 1)
        else:
            record["session_id"] = 1
        out.append(record)
    return out


def loaded_from(ticks: list[dict[str, Any]], **kwargs) -> LoadedRecords:
    return LoadedRecords(
        ticks=ticks, scenario=None, timebase_estimates=[], unparseable=0,
        thermal_samples=kwargs.get("thermal_samples", []),
        thermal_events=kwargs.get("thermal_events", []),
        failure_scans=kwargs.get("failure_scans", []),
        failure_events=kwargs.get("failure_events", []),
    )


# --------------------------------------------------------------------------
# Axis mechanics -- section 6 rules 1-5, applied to a plain (non-sensing) axis
# --------------------------------------------------------------------------


class TestAxisIdentityRules:
    """Rules checkable within one axis record, exercised on `thermal` since
    its shape (nested `jetson.basis`/`jetson.reason`) is the richest of the
    four plain axes.
    """

    def test_measured_and_absent_ticks_partition_attempted(self):
        ticks = [
            {"thermal": {"jetson": {"basis": "measured"}}},
            {"thermal": {"jetson": {"basis": "absent", "reason": "sampler_stopped"}}},
        ]
        axis = _axis_thermal(ticks).to_record()
        assert axis["attempted"] == 2
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {"sampler_stopped": 1}
        # rule 2: three independently counted quantities still reconcile.
        assert axis["answered"] + sum(axis["unanswered_by_reason"].values()) == axis["attempted"]

    def test_two_different_absent_reasons_are_not_merged(self):
        """§7: changing the reason word on the absent tick moves the census,
        not just its total -- two distinct keys survive, neither merged into
        the other.
        """
        ticks = [
            {"thermal": {"jetson": {"basis": "absent", "reason": "sampler_stopped"}}},
            {"thermal": {"jetson": {"basis": "absent", "reason": "no_sample_yet"}}},
        ]
        axis = _axis_thermal(ticks).to_record()
        assert axis["unanswered_by_reason"] == {"sampler_stopped": 1, "no_sample_yet": 1}

    def test_a_word_outside_the_vocabulary_is_counted_verbatim_and_flagged(self):
        ticks = [{"thermal": {"jetson": {"basis": "absent", "reason": "gremlins"}}}]
        axis = _axis_thermal(ticks).to_record()
        assert axis["unanswered_by_reason"] == {"gremlins": 1}
        assert axis["vocabulary_violations"] == {"gremlins": 1}
        # rule 2 still holds even though the word is a violation.
        assert axis["answered"] + sum(axis["unanswered_by_reason"].values()) == axis["attempted"]

    def test_a_stale_tick_is_a_recognised_word_not_a_violation(self):
        ticks = [{"thermal": {"jetson": {"basis": "stale"}}}]
        axis = _axis_thermal(ticks).to_record()
        assert axis["unanswered_by_reason"] == {"stale": 1}
        assert axis["vocabulary_violations"] == {}

    def test_unanswered_by_reason_is_empty_iff_answered_equals_attempted(self):
        all_measured = _axis_thermal([{"thermal": {"jetson": {"basis": "measured"}}}]).to_record()
        assert all_measured["answered"] == all_measured["attempted"]
        assert all_measured["unanswered_by_reason"] == {}

        with_absent = _axis_thermal(
            [{"thermal": {"jetson": {"basis": "measured"}}},
             {"thermal": {"jetson": {"basis": "absent", "reason": "read_error"}}}]
        ).to_record()
        assert with_absent["answered"] != with_absent["attempted"]
        assert with_absent["unanswered_by_reason"] != {}

    def test_zero_of_zero_is_never_counted_as_answering(self):
        """No tick carries a thermal block at all: `attempted == 0`, and this
        is a real fact about the axis (thermal not enabled), not
        `unbuildable`. Rule 5: `attempted == 0` implies `answered == 0`.
        """
        axis = _axis_thermal([{"tick_id": 0}, {"tick_id": 1}]).to_record()
        assert axis["attempted"] == 0
        assert axis["answered"] == 0
        assert axis["unbuildable"] is None

    def test_census_and_violations_helper_never_drops_a_key(self):
        census, violations = _census_and_violations({"a": 1, "b": 2}, frozenset({"a"}))
        assert census == {"a": 1, "b": 2}
        assert violations == {"b": 2}


class TestFailuresAxis:
    def test_answered_counts_measured_basis(self):
        """§7: `failures.answered` moves on a tick whose `failures.basis` is
        `absent`/`sampler_stopped` instead of `measured`.
        """
        ticks = [
            {"failures": {"basis": "measured"}},
            {"failures": {"basis": "absent", "reason": "sampler_stopped"}},
        ]
        axis = _axis_failures(ticks).to_record()
        assert axis["attempted"] == 2
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {"sampler_stopped": 1}


class TestLatencyAxis:
    def test_answered_counts_jetson_ms_present(self):
        ticks = [{"jetson_ms": 12.0}, {}, {}]
        axis = _axis_latency(ticks).to_record()
        assert axis["attempted"] == 3
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {"absent from this run; e2e used": 2}


class TestProvenanceAxis:
    def _encoder_map(self) -> dict[str, str]:
        from policy import sim_contract
        return {name: "measured" for name in sim_contract.encoded_slot_names()}

    def test_a_full_correct_map_answers(self):
        ticks = [{"field_sources": self._encoder_map()}]
        axis = _axis_provenance(ticks).to_record()
        assert axis["attempted"] == 1
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {}

    def test_a_short_map_is_censused_by_its_size_and_flagged(self):
        short = dict(list(self._encoder_map().items())[:20])
        ticks = [{"field_sources": short}]
        axis = _axis_provenance(ticks).to_record()
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {"short: 20": 1}
        assert axis["vocabulary_violations"] == {"short: 20": 1}

    def test_a_full_size_map_of_only_substituted_values_does_not_answer(self):
        """M4: a map that covers every encoder slot by NAME but whose every
        value is a substituted class (never derived_empty here, but the same
        exclusion) measured nothing -- a shape check alone reads this as a
        full house.
        """
        from perception import provenance

        encoder_map = {name: provenance.SOURCE_FALLBACK_NEUTRAL for name in self._encoder_map()}
        ticks = [{"field_sources": encoder_map}]
        axis = _axis_provenance(ticks).to_record()
        assert axis["attempted"] == 1
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {provenance.SOURCE_FALLBACK_NEUTRAL: 1}
        # A recognised class word, not a violation.
        assert axis["vocabulary_violations"] == {}

    def test_a_full_size_map_of_only_derived_empty_does_not_answer(self):
        from perception import provenance

        encoder_map = {name: provenance.SOURCE_DERIVED_EMPTY for name in self._encoder_map()}
        ticks = [{"field_sources": encoder_map}]
        axis = _axis_provenance(ticks).to_record()
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {provenance.SOURCE_DERIVED_EMPTY: 1}

    def test_a_map_with_one_measured_field_among_substituted_ones_answers(self):
        from perception import provenance

        encoder_map = {name: provenance.SOURCE_FALLBACK_NEUTRAL for name in self._encoder_map()}
        first = next(iter(encoder_map))
        encoder_map[first] = "measured"
        ticks = [{"field_sources": encoder_map}]
        axis = _axis_provenance(ticks).to_record()
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {}


class TestSessionSummaryShape:
    """Rules 6 and 7: the axis list is complete and in order, and nothing in
    `session_summary` is a ratio, a percentage, or a single health word.
    """

    def test_axes_has_exactly_the_seven_members_in_order(self):
        loaded = loaded_from([])
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        assert [a["axis"] for a in session["axes"]] == list(AXES)
        assert len(session["axes"]) == len(AXES)

    def test_no_field_is_a_ratio_percentage_or_health_word(self):
        loaded = loaded_from([{"jetson_ms": 1.0}])
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        assert set(session.keys()) == {
            "axes", "tick_coverage", "reconciliations", "inputs", "sensing",
        }
        # `tick_coverage` (D1) is counts and durations only -- no ratio field
        # (the hardware-drive round's rule 7 concern, restated for it).
        if session["tick_coverage"] is not None:
            assert set(session["tick_coverage"].keys()) == {
                "actual_ticks", "span_s", "median_gap_s", "largest_gap_s",
                "next_largest_gap_s", "gap_multiple", "missing_ticks", "expected_ticks",
            }
        for axis in session["axes"]:
            assert set(axis.keys()) == {
                "axis", "attempted", "answered", "attempted_is", "answered_is",
                "unanswered_by_reason", "vocabulary", "vocabulary_violations",
                "unbuildable", "section", "not_evaluable_by_rule", "zero_attempted_context",
            }


class TestRatesAndApiCallsAndTriggersUnbuildable:
    """A drive with no phone at all has no `sensing` block on any tick, so
    three axes are `unbuildable` -- distinct from `attempted: 0`.
    """

    def test_no_phone_run_marks_three_axes_unbuildable(self):
        loaded = loaded_from([{"tick_id": 0}, {"tick_id": 1}])
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        by_axis = {a["axis"]: a for a in session["axes"]}
        for name in ("rates", "api_calls", "triggers"):
            assert by_axis[name]["attempted"] is None
            assert by_axis[name]["answered"] is None
            assert by_axis[name]["unbuildable"] == "no tick carries a sensing block (not a phone run)"
        assert session["sensing"] is None


class TestRatesAxis:
    def test_attempted_dedups_on_distinct_at_mono(self):
        """§7: two ticks echoing one report vs two ticks with distinct
        reports move `rates.attempted` between 1 and 2 -- the D7
        deduplication.
        """
        one_report = sensing_ticks(2, telemetry_at=frozenset({0}))
        axis_one = _axis_rates(one_report).to_record()
        assert axis_one["attempted"] == 1

        two_reports = sensing_ticks(2, telemetry_at=frozenset({0, 1}))
        axis_two = _axis_rates(two_reports).to_record()
        assert axis_two["attempted"] == 2

    def test_a_stale_report_is_not_answered(self):
        """§7: a report aged past `MAX_TELEMETRY_AGE_S` (10.0) moves
        `rates.answered` and is censused as `stale`.
        """
        ticks = sensing_ticks(2, telemetry_at=frozenset({0}), telemetry_age_offset=0.02)
        # Force the one report's age past the controller's own threshold.
        ticks[1]["sensing"]["reference"]["age_s"] = 11.0
        axis = _axis_rates(ticks).to_record()
        assert axis["attempted"] == 1
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {"stale": 1}

    def test_no_telemetry_ticks_are_censused_separately_from_stale(self):
        ticks = sensing_ticks(2, telemetry_at=frozenset())  # never reports
        axis = _axis_rates(ticks).to_record()
        assert axis["attempted"] == 2
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {"no_telemetry": 2}


class TestDistinctReports:
    """`sensing_result`'s own dedup (used for `achieved_time_mean` and the
    telemetry window), separate from `_axis_rates`'s inline one -- D7 applies
    to both, and this pins it on the code path `_axis_rates` does not cover.
    """

    def test_two_ticks_echoing_one_report_collapse_to_one(self):
        from eval_run import _distinct_reports

        ticks = sensing_ticks(3, telemetry_at=frozenset({0}))
        assert len(_distinct_reports(ticks)) == 1

    def test_distinct_reports_are_kept_apart(self):
        from eval_run import _distinct_reports

        ticks = sensing_ticks(3, telemetry_at=frozenset({0, 1, 2}))
        assert len(_distinct_reports(ticks)) == 3


class TestApiCallsAxis:
    def test_calls_total_accumulates_within_one_session(self):
        """§7: `here_calls` 0 -> 9 within one session moves `calls_total` to 9."""
        ticks = sensing_ticks(4, here_calls_at={0: 0, 1: 3, 2: 6, 3: 9})
        here = _here_calls_by_session(ticks)
        assert here["calls_total"] == 9
        assert here["by_session"] == [{"session_id": 1, "first": 0, "last": 9, "observations": 4}]

    def test_calls_total_is_zero_when_every_value_is_zero(self):
        ticks = sensing_ticks(3, here_calls_at={0: 0, 1: 0, 2: 0})
        here = _here_calls_by_session(ticks)
        assert here["calls_total"] == 0

    def test_by_session_splits_on_a_redial_and_never_goes_negative(self):
        """§7 (D11): the same counter series split across two `session_id`s,
        the second restarting at zero -- two rows, no negative total.
        """
        ticks = sensing_ticks(4, here_calls_at={0: 5, 1: 9, 2: 0, 3: 2})
        for i, t in enumerate(ticks):
            t["session_id"] = 1 if i < 2 else 2
        here = _here_calls_by_session(ticks)
        assert len(here["by_session"]) == 2
        assert here["calls_total"] == (9 - 5) + (2 - 0)
        assert here["calls_total"] >= 0
        assert here["uncounted_prefix"] == 5 + 0

    def test_a_no_telemetry_tick_is_counted_under_no_telemetry_and_does_not_move_calls_total(self):
        # Tick 0 has no telemetry assigned at all (the phone has never
        # reported); tick 1 is the first report.
        ticks = sensing_ticks(2, telemetry_at=frozenset({1}), here_calls_at={1: 4})
        axis = _axis_api_calls(ticks).to_record()
        assert axis["attempted"] == 2
        assert axis["answered"] == 1
        assert axis["unanswered_by_reason"] == {"no_telemetry": 1}
        here = _here_calls_by_session(ticks)
        assert here["calls_total"] == 0  # only one observation in the session; nothing to diff

    def test_absent_wins_over_a_present_here_calls_value(self):
        """M1's mirror-image case: a record that violates the shape rule --
        `absent` set (telemetry not present) but `here_calls` a real number
        anyway -- must still census under no_telemetry, not read as answered
        just because a number happens to be there.
        """
        ticks = sensing_ticks(1, telemetry_at=frozenset({0}), here_calls_at={0: 7})
        ticks[0]["sensing"]["reference"]["absent"] = "no_telemetry"
        axis = _axis_api_calls(ticks).to_record()
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {"no_telemetry": 1}

    def test_a_field_not_recorded_tick_is_censused_separately_from_no_telemetry(self):
        """M1: telemetry present (`absent` is `None`) but `here_calls` still
        null is a shape violation `reference_from` itself never produces --
        it gets its own word rather than being merged into no_telemetry.
        """
        ticks = sensing_ticks(1, telemetry_at=frozenset({0}), here_calls_at={0: 3})
        ticks[0]["sensing"]["reference"]["here_calls"] = None
        axis = _axis_api_calls(ticks).to_record()
        assert axis["answered"] == 0
        assert axis["unanswered_by_reason"] == {"field_not_recorded": 1}
        assert axis["unanswered_by_reason"].get("no_telemetry") is None


class TestHereCallsNotMeasuredAndBackwards:
    """C2 and M7: a drive whose log predates `here_calls` must not read as a
    measured zero, and a counter that decreases within one session must not
    be silently clamped.
    """

    def test_calls_total_is_none_when_no_tick_carries_here_calls(self):
        ticks = [
            {"sensing": {"reference": {"absent": None, "at_mono": float(i), "age_s": 0.02}},
             "session_id": 1}
            for i in range(3)
        ]  # pre-task-39 shape: reference present, no here_calls/here_errors keys at all
        here = _here_calls_by_session(ticks)
        assert here["calls_total"] is None
        assert here["errors_total"] is None
        assert here["not_measured"] == "no tick carries here_calls (log predates task 39)"

    def test_sensing_lines_report_not_measured_rather_than_a_measured_zero(self):
        """C2's real-drive reproduction: `## Sensing` used to print
        `0 total ... -- mode shadow; a shadow command never reaches
        setHereQuery` on a drive that never recorded the field at all.
        """
        from eval_run import _sensing_lines

        ticks = sensing_ticks(3, mode=SHADOW, telemetry_at=frozenset(range(3)))
        for t in ticks:
            t["sensing"]["reference"].pop("here_calls", None)
            t["sensing"]["reference"].pop("here_errors", None)
        result = sensing_result(ticks, {})
        assert result["here"]["calls_total"] is None
        text = "\n".join(_sensing_lines(result))
        assert "not measured" in text
        assert "0 total" not in text

    def test_zero_calls_because_requires_at_least_one_session_with_two_observations(self):
        ticks = sensing_ticks(1, mode=SHADOW, here_calls_at={0: 0})
        result = sensing_result(ticks, {})
        assert result["here"]["calls_total"] == 0
        assert result["here"]["zero_calls_because"] is None

    def test_zero_calls_because_is_set_with_two_or_more_observations_in_shadow(self):
        ticks = sensing_ticks(2, mode=SHADOW, here_calls_at={0: 0, 1: 0})
        result = sensing_result(ticks, {})
        assert result["here"]["calls_total"] == 0
        assert result["here"]["zero_calls_because"] is not None

    def test_a_counter_that_decreases_within_one_session_is_reported_not_silently_clamped(self):
        ticks = sensing_ticks(2, here_calls_at={0: 5, 1: 2})
        here = _here_calls_by_session(ticks)
        assert here["counter_went_backwards"] == [1]
        assert here["calls_total"] == 0  # still clamped for the total, not negative

    def test_no_backwards_session_is_an_empty_list_not_omitted(self):
        ticks = sensing_ticks(2, here_calls_at={0: 0, 1: 3})
        here = _here_calls_by_session(ticks)
        assert here["counter_went_backwards"] == []


class TestExpectedFromCommanded:
    def test_doubling_the_commanded_rate_doubles_the_integral(self):
        points = [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        base = _time_weighted_integral(points)
        doubled = _time_weighted_integral([(t, v * 2) for t, v in points])
        assert doubled == pytest.approx(base * 2)
        assert base > 0

    def test_the_integral_is_weighted_by_the_gap_between_points_not_by_sample_count(self):
        """Doubling every value alone cannot distinguish a correctly
        time-weighted integral from any naive aggregate that is merely
        linear in v (a plain sum, or a sample mean times the span) --
        doubling v doubles any of those too. This pins the exact
        gap-weighted value on an unevenly spaced series instead.
        """
        points = [(0.0, 2.0), (1.0, 2.0), (5.0, 2.0)]
        # weights = gaps + [last gap] = [1, 4, 4]; sum(v * w) = 2*1 + 2*4 + 2*4
        assert _time_weighted_integral(points) == pytest.approx(18.0)


class TestTelemetryWindow:
    def test_median_moves_with_the_gap_between_reports(self):
        ticks_1s = sensing_ticks(3, telemetry_at=frozenset({0, 1, 2}), telemetry_age_offset=0.0)
        # Reports 0.1 s apart by construction (clock.advance(0.1) each tick).
        window = _telemetry_window([t for t in ticks_1s if t.get("sensing")])
        assert window["observed_median_s"] == pytest.approx(0.1)

    def test_fewer_than_two_reports_is_unbuildable(self):
        one_report = sensing_ticks(1, telemetry_at=frozenset({0}))
        window = _telemetry_window(one_report)
        assert window["observed_median_s"] is None
        assert window["unbuildable"] is not None

    def test_median_reflects_the_actual_gap_not_a_hardcoded_constant(self):
        """The other median test's gap happens to be 0.1 s, which is also a
        plausible hardcoded stand-in -- this one uses a different gap (0.2 s,
        reports every other tick) so a constant-0.1 implementation is caught.
        """
        ticks = sensing_ticks(5, telemetry_at=frozenset({0, 2, 4}))
        window = _telemetry_window([t for t in ticks if t.get("sensing")])
        assert window["observed_median_s"] == pytest.approx(0.2)


class TestComparableAndPercentilesSuppressed:
    def test_a_shadow_drive_is_not_comparable(self):
        ticks = sensing_ticks(5, mode=SHADOW, telemetry_at=frozenset(range(5)))
        result = sensing_result(ticks, {})
        assert result["rates"]["camera_hz"]["comparable"] is False
        assert "shadow" in result["rates"]["camera_hz"]["not_comparable_because"]
        assert result["rates"]["camera_hz"]["achieved_time_mean"] is None

    def test_a_live_drive_with_enough_reports_is_comparable(self):
        ticks = sensing_ticks(5, mode=LIVE, telemetry_at=frozenset(range(5)))
        result = sensing_result(ticks, {})
        assert result["rates"]["camera_hz"]["comparable"] is True
        assert result["rates"]["camera_hz"]["not_comparable_because"] is None

    def test_a_rate_slower_than_the_window_suppresses_percentiles(self):
        """§7 (D9): commanded `here_hz` (0.05, idle) over a ~1 s window is
        one delivery per ~20 windows -- percentiles are suppressed. The
        faster `imu_hz` (50) on the same drive is not.
        """
        ticks = sensing_ticks(5, mode=LIVE, telemetry_at=frozenset(range(5)))
        result = sensing_result(ticks, {})
        here = result["rates"]["here_hz"]
        imu = result["rates"]["imu_hz"]
        assert here["percentiles_suppressed"] is not None
        assert here["percentiles"] == {"p50": None, "p95": None}
        assert imu["percentiles_suppressed"] is None
        assert imu["percentiles"]["p50"] is not None

    def test_lambda_per_window_is_persisted_on_the_record(self):
        """Open item (§9's `lambda_per_window` -- "computed then discarded"):
        it was used only to decide suppression and never written to the
        record; it now survives on `rates[key]`.
        """
        ticks = sensing_ticks(5, mode=LIVE, telemetry_at=frozenset(range(5)))
        result = sensing_result(ticks, {})
        assert result["rates"]["here_hz"]["lambda_per_window"] is not None
        assert result["rates"]["camera_hz"]["comparable"] is False or \
            result["rates"]["camera_hz"]["lambda_per_window"] is not None


def _minimal_sensing_tick(decided_at_mono: float, at_mono: float, rate_value: float) -> dict[str, Any]:
    """A hand-built tick carrying only what `sensing_result` reads -- used
    where a real `SensingLoop` drive cannot easily be steered to an exact
    `lambda_per_window` (D9's `== 1.0` edge) or to a malformed shape.
    """
    from transport.messages import DROP_KEYS

    return {
        "sensing": {
            "decided_at_mono": decided_at_mono,
            "shadow": False,
            "rates": {"camera_hz": 1.0, "gps_hz": rate_value, "imu_hz": 1.0, "here_hz": 1.0},
            "reference": {
                "absent": None, "at_mono": at_mono, "age_s": 0.02,
                "achieved": {"camera_hz": 1.0, "gps_hz": rate_value, "imu_hz": 1.0, "here_hz": 1.0},
                "dropped": {k: 0 for k in DROP_KEYS},
                "here_calls": 0, "here_errors": 0,
            },
            "attribution": {"rules": {}, "per_sensor": {}},
        },
        "session_id": 1,
    }


class TestLambdaExactlyOne:
    def test_lambda_exactly_one_reports_percentiles_and_names_the_quantisation(self):
        """D9: below 1.0 the percentiles are suppressed (already pinned
        above); AT exactly 1.0 -- one delivery per window -- they are
        reported, and this is the case that used to leave
        `percentiles_suppressed` null with no mention of the quantisation.
        """
        ticks = [_minimal_sensing_tick(float(i), float(i), 1.0) for i in range(3)]
        result = sensing_result(ticks, {})
        row = result["rates"]["gps_hz"]
        assert row["comparable"] is True
        assert row["lambda_per_window"] == pytest.approx(1.0)
        assert row["percentiles"]["p50"] == pytest.approx(1.0)
        assert row["percentiles"]["p95"] == pytest.approx(1.0)
        assert row["percentiles_suppressed"] is not None
        assert "quanti" in row["percentiles_suppressed"]


class TestSensingResultRobustness:
    """MINOR: `sensing_result` used to hard-index `reference`,
    `decided_at_mono`, `rates[key]` and `attribution`, and crashed on a
    malformed block -- taking the whole report down with it, including axes
    that never touch `sensing` at all.
    """

    def test_does_not_raise_on_a_null_phone_block(self):
        ticks = sensing_ticks(3, mode=LIVE, telemetry_at=frozenset(range(3)))
        result = sensing_result(ticks, {"phone": None})  # used to raise AttributeError
        assert result["here"]["responses_received"] is None

    def test_a_malformed_sensing_block_does_not_abort_the_computation(self):
        ticks = sensing_ticks(3, mode=LIVE, telemetry_at=frozenset(range(3)))
        ticks[1]["sensing"] = {}  # no reference/decided_at_mono/rates/attribution at all
        result = sensing_result(ticks, {})
        assert result is not None
        assert result["mode"] in (LIVE, SHADOW)

    def test_session_summary_does_not_raise_on_a_malformed_sensing_block(self):
        """The same malformed shape reaches the axis builders before
        `sensing_result` is even called (`session_summary` builds `axes`
        first) -- both must survive it for the report to survive at all.
        """
        ticks = sensing_ticks(3, mode=LIVE, telemetry_at=frozenset(range(3)))
        ticks[1]["sensing"] = {}
        loaded = loaded_from(ticks)
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        assert session["sensing"] is not None


class TestCommandedByValue:
    def test_census_and_time_mean_over_a_mixed_commanded_series(self):
        """§7: `camera_hz` at 1.0 on 3 points and 5.0 on 1, UNEVENLY spaced
        (the last point holds for 8 s, not 1) so the result is a genuine time
        mean -- an evenly-spaced fixture here cannot distinguish a time mean
        from a plain sample mean, which happen to agree whenever every gap is
        the same size, and this test's name claims that distinction.
        """
        points = [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (10.0, 5.0)]
        mean = _time_weighted_mean(points)
        sample_mean = sum(v for _, v in points) / len(points)
        assert mean != pytest.approx(sample_mean)
        # weights = [1, 1, 8, 8]; sum(v*w) = 1+1+8+40 = 50, total weight = 18
        assert mean == pytest.approx(50 / 18)

    def test_time_weighted_mean_is_not_a_plain_sample_mean(self):
        points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (100.0, 12.0)]
        sample_mean = sum(v for _, v in points) / len(points)
        # weights = gaps + [last gap] = [1, 1, 98, 98]; sum(v*w) = 12*98 = 1176,
        # total weight = 198 -- the last point's long hold dominates a plain
        # sample mean would never reflect.
        assert _time_weighted_mean(points) != pytest.approx(sample_mean)
        assert _time_weighted_mean(points) == pytest.approx(1176 / 198)


class TestTriggersRulesMissing:
    def test_the_missing_field_name_reaches_the_summary_not_just_the_count(self):
        ticks = [
            {"sensing": {"attribution": {"rules": {
                "advisory_margin_narrow": {"status": RULE_NOT_EVALUABLE, "missing": ["policy_margin"]},
            }}}},
            {"sensing": {"attribution": {"rules": {
                "advisory_margin_narrow": {"status": RULE_NOT_EVALUABLE, "missing": ["policy_margin"]},
            }}}},
            {"sensing": {"attribution": {"rules": {
                "source_disagreement": {"status": RULE_NOT_EVALUABLE, "missing": ["feed_congestion"]},
            }}}},
        ]
        missing = _tally_rules_missing(ticks)
        assert missing["advisory_margin_narrow"] == {"policy_margin": 2}
        assert missing["source_disagreement"] == {"feed_congestion": 1}


class TestTriggersAxis:
    def test_not_evaluable_is_not_unanswered(self):
        """A rule being `not_evaluable` is a legitimate state (D3): the
        `triggers` axis answers "did the attribution parse", and it parses
        here even though `source_disagreement` never evaluates.
        """
        ticks = sensing_ticks(5, mode=SHADOW, telemetry_at=frozenset(range(5)))
        axis = _axis_triggers(ticks).to_record()
        assert axis["attempted"] == axis["answered"] == 5
        assert axis["unanswered_by_reason"] == {}

    def test_not_evaluable_reaches_the_axis_record_itself(self):
        """M5: `{"attempted": 5, "answered": 5, "unanswered_by_reason": {}}`
        alone is indistinguishable from a drive where every rule fired --
        `not_evaluable_by_rule` is the axis record's own trace of the rules
        that never got the chance, without touching rule 2's identity.
        """
        ticks = sensing_ticks(5, mode=SHADOW, telemetry_at=frozenset(range(5)))
        axis = _axis_triggers(ticks).to_record()
        assert axis["answered"] == axis["attempted"]
        assert axis["not_evaluable_by_rule"].get("source_disagreement") == 5
        # rule 2 is unaffected by this field.
        assert axis["answered"] + sum(axis["unanswered_by_reason"].values()) == axis["attempted"]


# --------------------------------------------------------------------------
# Reconciliations -- section 6 rules 11-19
# --------------------------------------------------------------------------


def _failures_summary(**overrides) -> dict[str, Any]:
    base_row = {
        "status": "quiet", "passes_attempted": 10, "passes_readable": 10,
        "total": 0, "kept_total": 0, "suppressed": 0, "below_episode_threshold": 0,
        "events_written": 0,
    }
    sources = {f"source.{i}": dict(base_row) for i in range(30)}
    sources.update(overrides.pop("sources", {}))
    result = {
        "scan": {"sources_n": 30}, "sources": sources, "blind_ticks": 0,
    }
    result.update(overrides)
    return result


class TestReconciliations:
    def test_blind_ticks_holds_when_they_agree(self):
        summary = {"failures": _failures_summary(
            sources={"camera.blind_ticks": {
                "status": "fired", "passes_attempted": 10, "passes_readable": 10,
                "total": 40, "kept_total": 40, "suppressed": 0, "below_episode_threshold": 0,
                "events_written": 0,
            }},
            blind_ticks=40,
        )}
        loaded = loaded_from([])
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        assert recs["blind_ticks_matches_camera_source"]["status"] == "held"

    def test_blind_ticks_fails_and_names_both_numbers(self):
        """§7 / the plan's own worked example: `blind_ticks` 40 against a
        `camera.blind_ticks` row reporting `total: 0` fails, names both
        numbers, and does not resolve which is right.
        """
        summary = {"failures": _failures_summary(
            sources={"camera.blind_ticks": {
                "status": "quiet", "passes_attempted": 10, "passes_readable": 10,
                "total": 0, "kept_total": 0, "suppressed": 0, "below_episode_threshold": 0,
                "events_written": 0,
            }},
            blind_ticks=40,
        )}
        loaded = loaded_from([])
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        row = recs["blind_ticks_matches_camera_source"]
        assert row["status"] == "failed"
        assert "40" in row["detail"] and "0" in row["detail"]

    def test_a_missing_summary_json_is_unavailable_never_held(self):
        """Ten reconciliations: the nine that need `summary.json` (all
        `unavailable` here, since it was never written) plus
        `reference_absent_iff_fields_null` (M1), which needs only tick
        records and is `unavailable` here because there are none.
        """
        loaded = loaded_from([])
        recs = [r.to_record() for r in reconciliations(loaded, {})]
        assert len(recs) == 10
        assert all(r["status"] == "unavailable" for r in recs)

    def test_events_written_matches_open_and_close_records(self):
        """Verified against a real hardware drive: `events_written` is
        incremented on both the open write and the close write
        (`logio/failure_log.py`'s `_open_episode`/`_close_episode`), so the
        identity is against every `failure_event` record, not the `open`
        half alone -- the plan this task implements states the narrower
        (open-only) identity, which is wrong against the code as it stands.
        """
        summary = {"failures": _failures_summary(
            sources={"a": {
                "status": "fired", "passes_attempted": 1, "passes_readable": 1,
                "total": 1, "kept_total": 1, "suppressed": 0, "below_episode_threshold": 0,
                "events_written": 2,
            }},
        )}
        loaded = loaded_from(
            [],
            failure_events=[{"phase": "open", "episode_id": 1}, {"phase": "close", "episode_id": 1}],
        )
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        assert recs["events_written_matches_open_and_close_records"]["status"] == "held"

    def test_events_written_fails_when_a_record_is_missing(self):
        """§7: deleting one `failure_event` open record from the log fails
        the reconciliation rather than the summary silently agreeing.
        """
        summary = {"failures": _failures_summary(
            sources={"a": {
                "status": "fired", "passes_attempted": 1, "passes_readable": 1,
                "total": 1, "kept_total": 1, "suppressed": 0, "below_episode_threshold": 0,
                "events_written": 2,
            }},
        )}
        loaded = loaded_from([], failure_events=[{"phase": "close", "episode_id": 1}])
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        assert recs["events_written_matches_open_and_close_records"]["status"] == "failed"

    def test_triggers_reconciliation_fails_and_names_both_maps(self):
        """§7: altering `summary["sensing"]["rules_by_status"]` by one fails
        the reconciliation and prints both maps.
        """
        ticks = sensing_ticks(3, mode=SHADOW, telemetry_at=frozenset(range(3)))
        summary = {
            "failures": _failures_summary(),
            "sensing": {"decisions_by_trigger": {"idle": 999}, "rules_by_status": {}},
        }
        loaded = loaded_from(ticks)
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        row = recs["triggers_match_summary"]
        assert row["status"] == "failed"
        assert "computed" in row["detail"] and "summary" in row["detail"]

    def test_here_calls_ge_responses_received_holds_and_fails(self):
        ticks = sensing_ticks(3, here_calls_at={0: 0, 1: 5, 2: 9})
        loaded = loaded_from(ticks)
        held = reconciliations(
            loaded, {"failures": _failures_summary(), "phone": {"here": {"responses_received": 5}}},
        )
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "here_calls_ge_responses_received"
        ] == "held"
        failed = reconciliations(
            loaded, {"failures": _failures_summary(), "phone": {"here": {"responses_received": 50}}},
        )
        assert {r.to_record()["name"]: r.to_record()["status"] for r in failed}[
            "here_calls_ge_responses_received"
        ] == "failed"

    def test_source_based_reconciliations_are_unavailable_not_held_when_sources_is_empty(self):
        """C1: an empty `sources` map makes every source-row reconciliation's
        `bad` list empty by construction, which used to read as `held` -- a
        check that compared nothing. It must read `unavailable`.
        """
        summary = {"failures": {"sources": {}, "scan": {}, "blind_ticks": 0}}
        loaded = loaded_from([])
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        for name in (
            "blind_ticks_matches_camera_source",
            "source_status_fired_iff_total_positive",
            "source_totals_partition",
            "source_passes_readable_le_attempted",
            "events_written_matches_open_and_close_records",
        ):
            assert recs[name]["status"] == "unavailable", name
        # sources_count_is_30 is not one of the seven: 0 == None == 30 is
        # False, so it already failed loudly rather than holding vacuously.
        assert recs["sources_count_is_30"]["status"] == "failed"

    def test_here_reconciliation_is_unavailable_when_no_tick_carries_here_calls(self):
        """C1's real-drive case, reproduced: `calls_total` used to be
        computed as `0` from a field present on zero ticks, and held against
        a `responses_received` of `0` -- two zeros that were never actually
        compared.
        """
        ticks = [
            {"type": "tick", "sensing": {"reference": {"absent": None, "at_mono": float(i)}}}
            for i in range(5)
        ]  # no tick carries here_calls at all (pre-task-39 shape)
        loaded = loaded_from(ticks)
        summary = {"failures": _failures_summary(), "phone": {"here": {"responses_received": 0}}}
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        assert recs["here_calls_ge_responses_received"]["status"] == "unavailable"

    def test_triggers_reconciliation_is_unavailable_not_held_with_zero_sensing_ticks(self):
        """C1: with no sensing ticks at all, the computed maps and an empty
        summary rollup are both `{}` -- which used to compare equal and hold
        on zero decisions, on both sides.
        """
        loaded = loaded_from([{"tick_id": 0}])  # no sensing block on any tick
        summary = {"sensing": {"decisions_by_trigger": {}, "rules_by_status": {}}}
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, summary)}
        assert recs["triggers_match_summary"]["status"] == "unavailable"

    def test_source_status_fired_iff_total_positive_holds_and_fails(self):
        held = reconciliations(loaded_from([]), {"failures": _failures_summary(
            sources={"a": {**{"status": "fired", "total": 5}}},
        )})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "source_status_fired_iff_total_positive"
        ] == "held"
        failed = reconciliations(loaded_from([]), {"failures": _failures_summary(
            # status says FIRED but total is 0 -- disagreement.
            sources={"a": {"status": "fired", "total": 0}},
        )})
        row = {r.to_record()["name"]: r.to_record() for r in failed}[
            "source_status_fired_iff_total_positive"
        ]
        assert row["status"] == "failed"
        assert "a" in row["detail"]

    def test_source_totals_partition_holds_and_fails(self):
        held = reconciliations(loaded_from([]), {"failures": _failures_summary(
            sources={"a": {"kept_total": 2, "suppressed": 1, "below_episode_threshold": 0, "total": 3}},
        )})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "source_totals_partition"
        ] == "held"
        failed = reconciliations(loaded_from([]), {"failures": _failures_summary(
            sources={"a": {"kept_total": 2, "suppressed": 1, "below_episode_threshold": 0, "total": 99}},
        )})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in failed}[
            "source_totals_partition"
        ] == "failed"

    def test_source_passes_readable_le_attempted_holds_and_fails(self):
        held = reconciliations(loaded_from([]), {"failures": _failures_summary(
            sources={"a": {"passes_readable": 3, "passes_attempted": 5}},
        )})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "source_passes_readable_le_attempted"
        ] == "held"
        failed = reconciliations(loaded_from([]), {"failures": _failures_summary(
            # readable exceeds attempted -- impossible, and a real defect.
            sources={"a": {"passes_readable": 9, "passes_attempted": 5}},
        )})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in failed}[
            "source_passes_readable_le_attempted"
        ] == "failed"

    def test_sources_count_is_30_holds_and_fails(self):
        held = reconciliations(loaded_from([]), {"failures": _failures_summary()})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "sources_count_is_30"
        ] == "held"
        summary = _failures_summary()
        del summary["sources"]["source.0"]  # 29 sources, not 30
        failed = reconciliations(loaded_from([]), {"failures": summary})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in failed}[
            "sources_count_is_30"
        ] == "failed"

    def test_thermal_samples_count_holds_and_fails(self):
        loaded = loaded_from([], thermal_samples=[{"type": "thermal_sample"}] * 5)
        held = reconciliations(loaded, {"thermal": {"jetson": {"samples": 5}}})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in held}[
            "thermal_samples_count"
        ] == "held"
        failed = reconciliations(loaded, {"thermal": {"jetson": {"samples": 50}}})
        assert {r.to_record()["name"]: r.to_record()["status"] for r in failed}[
            "thermal_samples_count"
        ] == "failed"

    def test_reference_shape_holds_on_a_real_drive(self):
        ticks = sensing_ticks(4, telemetry_at=frozenset({0, 2}))
        loaded = loaded_from(ticks)
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, {})}
        assert recs["reference_absent_iff_fields_null"]["status"] == "held"

    def test_reference_shape_fails_on_the_shape_violation(self):
        """M1's shape violation, at the reconciliation level: `absent` set
        together with a real `here_calls` value on the same record.
        """
        ticks = sensing_ticks(2, telemetry_at=frozenset({0, 1}))
        ticks[0]["sensing"]["reference"]["absent"] = "no_telemetry"
        loaded = loaded_from(ticks)
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, {})}
        assert recs["reference_absent_iff_fields_null"]["status"] == "failed"

    def test_reference_shape_is_unavailable_on_a_log_that_predates_here_calls(self):
        """A real pre-task-39 drive: `achieved`/`dropped` are present (they
        pre-date this task) but `here_calls`/`here_errors` are not keys on
        the record at all, on every tick where telemetry genuinely arrived.
        That is not a shape violation -- it is a schema this rule postdates
        -- and reporting it as `failed` would fail on every log recorded
        before this task, the same backward-compatibility gap M1/C2 already
        closed one level up, in `api_calls` and the HERE block.
        """
        ticks = sensing_ticks(3, telemetry_at=frozenset(range(3)))
        for t in ticks:
            t["sensing"]["reference"].pop("here_calls", None)
            t["sensing"]["reference"].pop("here_errors", None)
        loaded = loaded_from(ticks)
        recs = {r.to_record()["name"]: r.to_record() for r in reconciliations(loaded, {})}
        assert recs["reference_absent_iff_fields_null"]["status"] == "unavailable"
        assert "predates task 39" in recs["reference_absent_iff_fields_null"]["detail"]

    def test_a_failed_reconciliation_is_rendered_keyed_on_its_axis(self):
        """MINOR / plan §5.4: a failed reconciliation used to be keyed only
        on its own check name, rendering with no visible connection to the
        `## Session summary` axis line it contradicts.
        """
        from eval_run import _reconciliation_line

        rec = {
            "name": "blind_ticks_matches_camera_source", "status": "failed",
            "detail": "40 vs 0",
        }
        line = _reconciliation_line(rec)
        assert line.startswith("- **failures**")
        assert "blind_ticks_matches_camera_source" in line


# --------------------------------------------------------------------------
# Rendering -- section 6 rules 20-22, and the zero-tick branch (D13)
# --------------------------------------------------------------------------


def write_run(tmp_path, ticks, summary=None, log_health=None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for t in ticks:
            f.write(json.dumps(t) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary if summary is not None else {}))
    if log_health is not None:
        (run_dir / "log_health.json").write_text(json.dumps(log_health))
    return run_dir


class TestRenderingRules:
    def test_report_md_always_contains_the_session_summary_header(self, tmp_path):
        loaded = loaded_from([{"jetson_ms": 1.0}])
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        md = render_session_summary_only(tmp_path, session, loaded_from([]))
        assert "## Session summary" in md

    def test_overall_line_carries_the_axis_clause_in_both_forms(self):
        from eval_run import _overall_clause

        loaded_all_answer = loaded_from([{"jetson_ms": 1.0}])
        session_all = session_summary(loaded_all_answer, {}, None, phone_log_supplied=False)
        clause_all = _overall_clause(session_all)
        assert "instrument axes" in clause_all

        loaded_partial = loaded_from([{}])  # jetson_ms absent -> latency axis does not fully answer
        session_partial = session_summary(loaded_partial, {}, None, phone_log_supplied=False)
        clause_partial = _overall_clause(session_partial)
        assert "did not answer" in clause_partial

    def test_the_answered_count_never_appears_without_the_full_enumeration(self):
        """A fixture with zero fully-answered axes cannot detect a renderer
        that lists only the axes that did NOT answer, because every axis in
        such a fixture is in that list anyway. This one has at least one
        fully-answered axis (`latency`, `thermal`) alongside ones that do not
        (`failures`, `provenance`) and ones that cannot be built (`rates`,
        `api_calls`, `triggers`, no phone), so the enumeration's coverage
        actually gets exercised.
        """
        from eval_run import _axis_fully_answered, _session_summary_lines

        loaded = loaded_from([{"jetson_ms": 1.0, "thermal": {"jetson": {"basis": "measured"}}}])
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        fully_answered = {a["axis"] for a in session["axes"] if _axis_fully_answered(a)}
        assert fully_answered, "fixture must contain at least one fully-answered axis"
        assert len(fully_answered) < len(AXES), "fixture must also contain a non-answering axis"

        lines = _session_summary_lines(session)
        text = "\n".join(lines)
        for axis in AXES:
            assert f"**{axis}**" in text

    def test_render_markdown_puts_the_axis_clause_on_the_overall_line(self, tmp_path):
        """End to end through `render_markdown`, not just `_overall_clause` in
        isolation -- the clause has to actually reach the `Overall` line
        `render_markdown` writes.
        """
        from eval_run import analyze
        from tests.test_eval_run import make_tick, write_run

        run_dir = write_run(tmp_path, [make_tick(i) for i in range(5)])
        loaded = load_records(run_dir / "metadata.jsonl")
        session = session_summary(loaded, _read_summary(run_dir), _read_log_health(run_dir),
                                   phone_log_supplied=False)
        result = analyze(run_dir, loaded=loaded)
        md = render_markdown(result, [], session)
        overall_lines = [line for line in md.splitlines() if line.startswith("**Overall:")]
        assert len(overall_lines) == 1
        assert "instrument axes" in overall_lines[0]

    def test_a_zero_of_zero_axis_is_excluded_from_fully_answered(self):
        """§6 rule 5: `attempted == 0` is never counted as answering, even
        though `answered == attempted` (both zero) -- distinct from an axis
        that genuinely answered every observation it attempted.
        """
        from eval_run import _axis_fully_answered

        never_attempted = _axis_thermal([{"tick_id": 0}]).to_record()  # no thermal block at all
        assert never_attempted["attempted"] == 0
        assert never_attempted["answered"] == 0
        assert _axis_fully_answered(never_attempted) is False

        genuinely_answered = _axis_thermal([{"thermal": {"jetson": {"basis": "measured"}}}]).to_record()
        assert _axis_fully_answered(genuinely_answered) is True

    def test_overall_clause_separates_did_not_answer_from_could_not_be_built(self):
        """M2: a phone-less drive's three unbuildable axes (rates, api_calls,
        triggers) used to be folded into "did not answer" on the `Overall`
        line, contradicting the `## Session summary` section it points to,
        which already separates the two.
        """
        from eval_run import _overall_clause, _session_summary_lines

        loaded = loaded_from([{"jetson_ms": 1.0}])  # no thermal/failures/provenance block, no sensing
        session = session_summary(loaded, {}, None, phone_log_supplied=False)
        by_axis = {a["axis"]: a for a in session["axes"]}
        for name in ("rates", "api_calls", "triggers"):
            assert by_axis[name]["unbuildable"] is not None

        clause = _overall_clause(session)
        # latency fully answers; thermal/failures/provenance do not (attempted
        # 0 or a field_sources mismatch); rates/api_calls/triggers cannot be
        # built at all -- did_not == 3, unbuildable == 3, not 6 and 3.
        assert "3 of 7 instrument axes did not answer" in clause
        assert "3 could not be built" in clause

        section_text = "\n".join(_session_summary_lines(session))
        header_line = next(line for line in section_text.splitlines() if line.startswith(f"{len(AXES)} axes"))
        # The Overall line's counts match the section's own.
        assert "3 did not, 3 could not be built" in header_line

    def test_rates_headline_uses_an_honest_noun_not_reports(self):
        """MINOR: `rates.attempted` mixes distinct telemetry reports and
        no-telemetry TICKS under one count -- rendering it as "reports"
        claims a unit the count is not purely made of.
        """
        from eval_run import _axis_headline_line

        ticks = sensing_ticks(2, telemetry_at=frozenset({1}))  # tick 0: no telemetry; tick 1: a report
        axis = _axis_rates(ticks).to_record()
        line = _axis_headline_line(axis, None)
        assert "telemetry observations" in line
        assert "reports answered" not in line

    def test_render_markdown_renders_the_sensing_section_end_to_end(self, tmp_path):
        """M6: the only end-to-end rendering test built ticks with no
        `sensing` block at all, so `## Sensing` was never actually rendered
        by any test -- the whole section could be deleted with the suite
        green.
        """
        from eval_run import analyze
        from tests.test_eval_run import make_tick

        sensing_blocks = sensing_ticks(5, mode=LIVE, telemetry_at=frozenset(range(5)))
        full_ticks = []
        for i in range(5):
            t = make_tick(i)
            t["sensing"] = sensing_blocks[i]["sensing"]
            t["session_id"] = sensing_blocks[i]["session_id"]
            full_ticks.append(t)
        run_dir = write_run(tmp_path, full_ticks)
        loaded = load_records(run_dir / "metadata.jsonl")
        session = session_summary(loaded, _read_summary(run_dir), _read_log_health(run_dir),
                                   phone_log_supplied=False)
        result = analyze(run_dir, loaded=loaded)
        md = render_markdown(result, [], session)
        assert "## Sensing" in md
        assert "HERE calls" in md
        assert "decisions by trigger" in md

    def test_clamped_and_thermal_scaled_ticks_are_rendered(self):
        """MINOR: `clamped_ticks`/`thermal_scaled_ticks` are computed and
        written to `report.json` but rendered nowhere.
        """
        from eval_run import _sensing_lines

        ticks = sensing_ticks(5, mode=LIVE, telemetry_at=frozenset(range(5)))
        for t in ticks:
            t["sensing"]["attribution"].setdefault("per_sensor", {})
            t["sensing"]["attribution"]["per_sensor"]["camera_hz"] = {"clamped": True, "scale": 0.6}
        result = sensing_result(ticks, {})
        assert result["rates"]["camera_hz"]["clamped_ticks"] == 5
        assert result["rates"]["camera_hz"]["thermal_scaled_ticks"] == 5
        text = "\n".join(_sensing_lines(result))
        assert "clamped on 5 ticks" in text
        assert "thermal-scaled on 5 ticks" in text


class TestZeroTickBranch:
    """D13: a drive with no tick records still produces a `report.md`
    containing `## Session summary`, and `main()` returns 2 rather than
    letting `analyze`'s `SystemExit` escape uncaught.
    """

    def test_main_writes_a_session_summary_only_report_and_returns_2(self, tmp_path, monkeypatch, capsys):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({"type": "failure_scan", "t_mono": 1.0, "ticks_seen": 0}) + "\n")
        (run_dir / "summary.json").write_text(json.dumps({}))

        monkeypatch.setattr("sys.argv", ["eval_run.py", str(run_dir), "--no-plots"])
        code = main()

        assert code == 2
        report_md = (run_dir / "report.md").read_text()
        assert "## Session summary" in report_md
        report_json = json.loads((run_dir / "report.json").read_text())
        assert report_json["n_ticks"] == 0
        assert report_json["analysis"] is None
        assert "session_summary" in report_json
        assert report_json["session_summary"]["axes"]

    def test_zero_tick_report_names_a_dropped_log(self, tmp_path, monkeypatch):
        """MINOR: `log_health`'s own detail (records dropped, writer status)
        used to be rendered only inside `## Failures`, which the zero-tick
        branch never produces -- so the fact that would explain the zero
        ticks (the log dropped records) never reached the report.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with open(run_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({"type": "failure_scan", "t_mono": 1.0, "ticks_seen": 0}) + "\n")
        (run_dir / "summary.json").write_text(json.dumps({}))
        (run_dir / "log_health.json").write_text(json.dumps({"dropped_records": 42, "writer_failure": None}))

        monkeypatch.setattr("sys.argv", ["eval_run.py", str(run_dir), "--no-plots"])
        main()

        report_md = (run_dir / "report.md").read_text()
        assert "42 records dropped" in report_md

    def test_metadata_jsonl_present_is_false_when_the_file_never_existed(self, tmp_path, monkeypatch):
        """MINOR: `inputs.metadata_jsonl` used to be a literal `True` that
        could never say otherwise.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps({}))
        # No metadata.jsonl written at all.

        monkeypatch.setattr("sys.argv", ["eval_run.py", str(run_dir), "--no-plots"])
        code = main()

        assert code == 2
        report_json = json.loads((run_dir / "report.json").read_text())
        assert report_json["session_summary"]["inputs"]["metadata_jsonl"] is False

    def test_analyze_still_raises_systemexit_directly(self, tmp_path):
        from eval_run import analyze

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metadata.jsonl").write_text(
            json.dumps({"type": "failure_scan", "t_mono": 1.0, "ticks_seen": 0}) + "\n"
        )
        with pytest.raises(SystemExit):
            analyze(run_dir)


# --------------------------------------------------------------------------
# Behaviour-change fences -- section 8
# --------------------------------------------------------------------------


class TestZeroBehaviourChange:
    def test_summary_json_key_set_is_unchanged_by_a_scripted_run(self):
        """`session_summary`/`sensing_result` are read-only over
        `summary.json` -- calling them must not add or remove a key.
        """
        summary = {"ticks": 3, "sensing": {"mode": {"mode": "shadow"}}}
        before = set(summary.keys())
        loaded = loaded_from(sensing_ticks(3))
        session_summary(loaded, summary, None, phone_log_supplied=False)
        assert set(summary.keys()) == before

    def test_render_markdown_without_a_session_is_unchanged(self, tmp_path):
        """A caller that does not pass `session` gets exactly the pre-task-39
        rendering -- no `## Session summary`, no `## Sensing`, no axis clause
        on the `Overall` line. Existing callers (other test files) rely on
        this.
        """
        from eval_run import analyze
        from tests.test_eval_run import make_tick, write_run

        run_dir = write_run(tmp_path, [make_tick(i) for i in range(5)])
        result = analyze(run_dir)

        md = render_markdown(result, [])
        assert "## Session summary" not in md
        assert "## Sensing" not in md
        assert md.count("**Overall: PASS**") == 1
        assert " -- the five gates" not in md


# --------------------------------------------------------------------------
# The hardware-drive round: a summary built from ratios cannot see an event
# that stops the records being produced (D1), and four axes that read a
# healthy line while the section they point at reports the opposite.
# --------------------------------------------------------------------------


def _ticks_at(gaps_s: list[float]) -> list[dict[str, Any]]:
    """Tick records whose wall clocks are separated by `gaps_s`."""
    t, out = 1000.0, []
    for i, gap in enumerate([0.0] + gaps_s):
        t += gap
        out.append({"type": "tick", "tick_id": i, "t_wall": t, "e2e_ms": 10.0})
    return out


class TestTickCoverageSeesWhatTheAxesCannot:
    """D1. Ticks come from phone frames, so an outage destroys `attempted`
    and `answered` together and every axis still reports fully answered. The
    coverage line reads the gap distribution, which is the only place in the
    records where the missing ticks leave a trace.
    """

    def test_an_uninterrupted_run_reports_no_missing_ticks(self):
        cov = _tick_coverage(_ticks_at([0.2] * 50))
        assert cov["missing_ticks"] == 0
        assert cov["actual_ticks"] == cov["expected_ticks"]

    def test_one_long_gap_is_counted_and_the_next_largest_is_named_beside_it(self):
        # 20 ticks at 0.2 s, one 54.58 s hole, 20 more -- the shape the real
        # degraded drive produced, where the next-largest gap was 0.31 s.
        cov = _tick_coverage(_ticks_at([0.2] * 20 + [54.58] + [0.2] * 20))
        assert cov["largest_gap_s"] == pytest.approx(54.58)
        assert cov["next_largest_gap_s"] == pytest.approx(0.2)
        assert cov["missing_ticks"] > 250
        assert cov["expected_ticks"] > cov["actual_ticks"]

    def test_the_estimate_is_not_span_times_rate_which_could_never_fail(self):
        # `expected = span x achieved rate` is algebraically forced: the rate
        # is derived from the same span and the same count, so it equals the
        # actual on every drive, outage or not. The interrupted run must not
        # reproduce that identity.
        cov = _tick_coverage(_ticks_at([0.2] * 20 + [54.58] + [0.2] * 20))
        span, actual = cov["span_s"], cov["actual_ticks"]
        forced = span * ((actual - 1) / span)
        assert cov["expected_ticks"] > forced + 1

    def test_fewer_than_three_ticks_is_absent_not_a_manufactured_zero(self):
        assert _tick_coverage(_ticks_at([0.2])) is None
