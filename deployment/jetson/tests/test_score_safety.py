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


def test_forced_slow_arm_clamps_every_tick_and_raises_speed() -> None:
    result = score_ticks([_tick() for _ in range(6)])
    forced_slow = result["arms"]["forced_slow"]
    assert forced_slow["clamped_ticks"] == 6
    assert forced_slow["mean_delta_speed_mps"] == 2.0
    assert forced_slow["events"] == {"etiquette_blocked_action:low_speed_uncongested": 6}
    recorded = result["arms"]["recorded"]
    assert recorded["clamped_ticks"] == 0


def test_refuses_when_a_recorded_safety_block_does_not_replay() -> None:
    tick = _tick(safety={"bogus": True})
    result = score_ticks([tick])
    assert result["refused"] == "incumbent_replay_mismatch"
    assert result["n_mismatches"] == 1


def test_render_report_names_rules_with_zero_evaluable_ticks_without_a_percentage() -> None:
    result = score_ticks([_tick() for _ in range(3)])
    report = render_report(result)
    assert "forward_ttc: not evaluable on any tick (0 of 3)" in report
    assert "forward_ttc: 0%" not in report
