"""Unit and sanity tests for score_safety.py.

The sanity check that matters most is against the real corpus
(outputs/task42_usb/), run manually and reported in plans/
implementation_records.md -- gitignored harness data is not a fixture this
suite can depend on (feedback_harness_data_files_untracked_local). These
tests build small synthetic tick records instead, shaped like
Tick.to_record()'s obs/field_sources/obs_diagnostics/action, and check the
tool's own refusal and reporting logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parents[1]
if str(JETSON_DIR) not in sys.path:
    sys.path.insert(0, str(JETSON_DIR))

from perception import provenance  # noqa: E402
from score_safety import render_report, score_ticks  # noqa: E402


def _tick(*, leader_gap_source: str = provenance.SOURCE_FALLBACK_NEUTRAL, safety: dict | None = None) -> dict:
    return {
        "tick_id": 0,
        "t_wall": 1.0,
        "obs": {
            "ego_speed": 0.0, "leader_gap": float("inf"), "leader_relative_speed": 0.0,
            "target_lane_front_gap": float("inf"), "target_lane_rear_gap": float("inf"),
            "target_lane_rear_required_decel": 0.0, "follower_gap": float("inf"),
            "follower_relative_speed": 0.0, "nearby_av_mean_speed": 30.0,
            "downstream_congestion_estimate": 0.0, "cooperation": {"segment_target_speed": 30.0},
        },
        "field_sources": {
            "ego_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
            "leader_gap": leader_gap_source,
            "leader_relative_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
            "target_lane_front_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "target_lane_rear_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "target_lane_rear_required_decel": provenance.SOURCE_FALLBACK_NEUTRAL,
            "follower_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "follower_relative_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
            "nearby_av_mean_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
            "downstream_congestion_estimate": provenance.SOURCE_FALLBACK_NEUTRAL,
            "local_density_bin": provenance.SOURCE_DERIVED_EMPTY,
            "local_mean_speed_bin": provenance.SOURCE_FALLBACK_NEUTRAL,
        },
        "obs_diagnostics": {"density_veh_per_km": 0.0, "last_detection_age_s": None},
        "action": {
            "desired_speed_bin": "fast", "desired_headway_bin": "normal",
            "lane_preference": "prefer_left_if_safe", "merge_mode": "normal",
        },
        "safety": safety,
    }


def test_refuses_on_no_ticks() -> None:
    result = score_ticks([])
    assert result["refused"] == "no_ticks"


def test_reports_counterfactual_when_no_safety_block_present() -> None:
    result = score_ticks([_tick() for _ in range(5)])
    assert result["counterfactual"] is True
    assert result["n_ticks"] == 5


def test_a_rule_with_zero_evaluable_ticks_carries_no_rate() -> None:
    result = score_ticks([_tick() for _ in range(5)])
    entry = result["rule_census"]["forward_ttc"]
    assert entry["evaluable_ticks"] == 0
    assert entry["rate"] is None
    assert "fired_ticks" not in entry


def test_an_evaluable_rule_carries_a_rate() -> None:
    ticks = [_tick(leader_gap_source=provenance.SOURCE_MEASURED) for _ in range(4)]
    # These are still all-substituted for the OTHER field forward_ttc needs
    # (leader_relative_speed), so forward_ttc stays not_evaluable; check a
    # rule that only needs the one field this fixture measures instead.
    for t in ticks:
        t["obs"]["target_lane_front_gap"] = 42.0
        t["field_sources"]["target_lane_front_gap"] = provenance.SOURCE_DERIVED
    result = score_ticks(ticks)
    entry = result["rule_census"]["target_lane_front_gap"]
    assert entry["evaluable_ticks"] == 4
    assert entry["fired_ticks"] == 0
    assert entry["fired_fraction_of_evaluable"] == 0.0


def test_forced_slow_arm_no_longer_clamps_when_density_lacks_evidence() -> None:
    """validator round 1, Fix 1: apply_safety_layer now runs against the
    neutralised context, so forcing desired_speed_bin to 'slow' on a rig
    where density is not evidence (every _tick() fixture here, matching
    outputs/task42_usb's own corpus) no longer clamps. plan_task144's E5
    reproduced the OLD clamp -- 3,913 of 3,913 -- as its headline finding
    that a firing rate here would describe the fallback, not traffic; Fix 1
    means that clamp itself was the bug E5 was documenting, and it is gone.
    See test_forced_slow_arm_still_clamps_with_real_density_evidence for the
    arm that correctly still fires."""
    result = score_ticks([_tick() for _ in range(6)])
    forced_slow = result["arms"]["forced_slow"]
    assert forced_slow["clamped_ticks"] == 0
    assert forced_slow["events"] == {}
    recorded = result["arms"]["recorded"]
    assert recorded["clamped_ticks"] == 0


def test_forced_slow_arm_still_clamps_with_real_density_evidence() -> None:
    """The other half of the Fix 1 regression check: a rig where density
    genuinely IS evidence (source `derived`, value below the uncongested
    threshold) still clamps under forced_slow exactly as before -- Fix 1
    neutralises absent evidence, not real evidence."""
    ticks = [_tick() for _ in range(6)]
    for t in ticks:
        t["field_sources"]["local_density_bin"] = provenance.SOURCE_DERIVED
        t["obs_diagnostics"]["density_veh_per_km"] = 2.0
    result = score_ticks(ticks)
    forced_slow = result["arms"]["forced_slow"]
    assert forced_slow["clamped_ticks"] == 6
    assert forced_slow["mean_delta_speed_mps"] == 2.0
    assert forced_slow["events"] == {"etiquette_blocked_action:low_speed_uncongested": 6}
    recorded = result["arms"]["recorded"]
    assert recorded["clamped_ticks"] == 0


#: A complete, valid `safety.config` (validator round 1, Fix 4) -- used by
#: tests below that want a tick past the up-front config check so they can
#: exercise something else (the incumbent replay mismatch, here).
_VALID_SAFETY_CONFIG = {
    "enabled": True, "withhold_lane_when_not_evaluable": True, "time_s": 1.0,
    "min_contextual_speed_mps": 12.0, "density_max_age_s": 4.0,
}


def test_refuses_when_a_recorded_safety_block_does_not_replay() -> None:
    tick = _tick(safety={"config": _VALID_SAFETY_CONFIG, "bogus": True})
    result = score_ticks([tick])
    assert result["refused"] == "incumbent_replay_mismatch"
    assert result["n_mismatches"] == 1


def test_refuses_naming_config_when_a_safety_block_predates_fix_4() -> None:
    """A `safety` block recorded after task 144 shipped but before Fix 4
    added `config` -- must refuse and name `config` as missing, not
    silently assume this tool's own defaults for `withhold_lane_when_not_
    evaluable`/`time_s`/etc."""
    tick = _tick(safety={"bounded": {"speed_mps": 20.0}})
    result = score_ticks([tick])
    assert result["refused"] == "safety_config_missing_key"
    assert result["missing_key"] == "config"


def test_refuses_naming_the_specific_missing_config_key() -> None:
    incomplete_config = dict(_VALID_SAFETY_CONFIG)
    del incomplete_config["withhold_lane_when_not_evaluable"]
    tick = _tick(safety={"config": incomplete_config})
    result = score_ticks([tick])
    assert result["refused"] == "safety_config_missing_key"
    assert result["missing_key"] == "withhold_lane_when_not_evaluable"


def test_a_tick_with_no_safety_block_at_all_is_unaffected_by_the_config_check() -> None:
    """A run recorded before task 144 shipped the gate -- COUNTERFACTUAL
    reporting, using this tool's own module defaults, exactly as before
    Fix 4."""
    result = score_ticks([_tick() for _ in range(3)])
    assert "refused" not in result
    assert result["counterfactual"] is True


def test_render_report_names_rules_with_zero_evaluable_ticks_without_a_percentage() -> None:
    result = score_ticks([_tick() for _ in range(3)])
    report = render_report(result)
    assert "forward_ttc: not evaluable on any tick (0 of 3)" in report
    assert "forward_ttc: 0%" not in report


def test_evaluable_rule_with_non_finite_compared_value_says_so() -> None:
    """validator round 1, F5: target_lane_front_gap tagged `derived`
    (evidence) with the gap itself still inf on every tick -- the exact
    corpus shape (1,229 of 3,913 ticks, all `derived`, gap_m == inf on all
    of them). A 0.0% fired rate over this population is not a rate; the
    threshold could never physically be crossed."""
    ticks = []
    for _ in range(3):
        t = _tick()
        t["field_sources"]["target_lane_front_gap"] = provenance.SOURCE_DERIVED
        ticks.append(t)
    result = score_ticks(ticks)
    entry = result["rule_census"]["target_lane_front_gap"]
    assert entry["evaluable_ticks"] == 3
    assert entry["fired_ticks"] == 0
    assert entry["non_finite_compared_ticks"] == 3
    report = render_report(result)
    assert (
        "target_lane_front_gap: evaluable on 3 of 3 ticks; fired on 0 of 3 "
        "evaluable (0.0%); the compared value (gap_m) is inf on all 3"
    ) in report


def test_stale_provenance_ticks_are_flagged_not_silently_trusted_or_dropped() -> None:
    """validator round 1, F5's provenance-consistency finding (coordinator
    measurement 2026-09-12): reproduces baseline_run_20260902_183446's exact
    shape -- target_lane_front_gap tagged `derived` while leader_gap, the
    field it is defined to alias verbatim, is tagged `fallback_neutral`.
    Under today's observation_builder.py the two can never disagree, so a
    tick where they do was written under a superseded contract; the tool
    must still count it evaluable (that is what today's rule says of the
    recorded source) while flagging it, not silently excluding it."""
    ticks = []
    for _ in range(3):
        t = _tick(leader_gap_source=provenance.SOURCE_FALLBACK_NEUTRAL)
        t["field_sources"]["target_lane_front_gap"] = provenance.SOURCE_DERIVED
        ticks.append(t)
    result = score_ticks(ticks)
    entry = result["rule_census"]["target_lane_front_gap"]
    assert entry["evaluable_ticks"] == 3  # still counted evaluable -- not silently excluded
    assert entry["stale_provenance_ticks"] == 3
    report = render_report(result)
    assert (
        "all 3 disagree with leader_gap's own source (target_lane_front_gap "
        "is defined to alias it verbatim) -- recorded under a superseded "
        "contract, not evidence under today's rule"
    ) in report


def test_consistent_provenance_ticks_are_not_flagged() -> None:
    """The other half: when target_lane_front_gap's source genuinely agrees
    with leader_gap's (today's actual contract), nothing is flagged."""
    ticks = [_tick(leader_gap_source=provenance.SOURCE_DERIVED) for _ in range(3)]
    for t in ticks:
        t["field_sources"]["target_lane_front_gap"] = provenance.SOURCE_DERIVED
    result = score_ticks(ticks)
    entry = result["rule_census"]["target_lane_front_gap"]
    assert entry["stale_provenance_ticks"] == 0
    report = render_report(result)
    assert "disagree with leader_gap" not in report


def test_render_report_states_clamp_direction_and_magnitude_in_words() -> None:
    """validator round 1, F6: a raised count and a lowered count, each with
    its own signed magnitude -- never a bare 'clamped' (which reads as a
    reduction) or a mean pooled across both directions."""
    ticks = [_tick() for _ in range(6)]
    for t in ticks:
        t["field_sources"]["local_density_bin"] = provenance.SOURCE_DERIVED
        t["obs_diagnostics"]["density_veh_per_km"] = 2.0
    result = score_ticks(ticks)
    arm = result["arms"]["forced_slow"]
    assert arm["raised_ticks"] == 6
    assert arm["lowered_ticks"] == 0
    assert arm["mean_raise_mps"] == 2.0
    report = render_report(result)
    arms_section = report.split("-- counterfactual arms --")[1]
    assert (
        "forced_slow: recommended speed raised on 6 of 6 ticks (mean +2.00 m/s); "
        "lowered on 0 of 6 ticks" in arms_section
    )
    assert "clamped" not in arms_section
