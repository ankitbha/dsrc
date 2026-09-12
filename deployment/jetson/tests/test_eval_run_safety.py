"""Unit tests for eval_run.py's task-144 safety rollup (safety_result/_safety_lines).

A dedicated file rather than additions to the large existing test_eval_run.py,
matching the project's own precedent (score_safety.py got its own
test_score_safety.py) and avoiding touching a file other work might also
need to touch.
"""

from __future__ import annotations

from eval_run import _safety_lines, safety_result


def _rule_block(status: str, **evidence) -> dict:
    block = {"status": status}
    if status == "not_evaluable":
        block["missing"] = list(evidence.keys()) or ["some_field"]
    block.update(evidence)
    return block


def _tick(safety: dict | None) -> dict:
    return {"safety": safety}


def _safety_block(*, delta: float, rules: dict) -> dict:
    return {
        "proposed": {"speed_mps": 30.0, "headway_s": 1.6, "lane_action": "LANE_LEFT"},
        "bounded": {"speed_mps": 30.0 + delta, "headway_s": 1.6, "lane_action": None},
        "delta_speed_mps": delta,
        "emergency_override": False,
        "lane_withheld": "not_evaluable",
        "evaluable": sum(1 for r in rules.values() if r["status"] != "not_evaluable"),
        "not_evaluable": sum(1 for r in rules.values() if r["status"] == "not_evaluable"),
        "rules": rules,
    }


def test_safety_result_is_none_with_no_ticks() -> None:
    assert safety_result([]) is None


def test_safety_result_is_none_when_no_tick_carries_a_safety_block() -> None:
    assert safety_result([{"safety": None}, {"safety": None}]) is None


def test_safety_result_census_and_clamp_rollup() -> None:
    ticks = [
        _tick(_safety_block(delta=0.0, rules={
            "forward_ttc": _rule_block("not_evaluable", leader_gap_m_source="fallback_neutral"),
            "target_lane_front_gap": _rule_block("quiet", gap_m=42.0, threshold_m=5.0),
        })),
        _tick(_safety_block(delta=2.0, rules={
            "forward_ttc": _rule_block("not_evaluable", leader_gap_m_source="fallback_neutral"),
            "target_lane_front_gap": _rule_block("fired", gap_m=3.0, threshold_m=5.0),
        })),
    ]
    result = safety_result(ticks)
    assert result["n_ticks"] == 2
    assert result["rules"]["forward_ttc"]["evaluable_ticks"] == 0
    assert result["rules"]["forward_ttc"]["fired_fraction_of_evaluable"] is None
    assert "fired_ticks" not in result["rules"]["forward_ttc"]
    assert result["rules"]["target_lane_front_gap"]["evaluable_ticks"] == 2
    assert result["rules"]["target_lane_front_gap"]["fired_ticks"] == 1
    assert result["rules"]["target_lane_front_gap"]["fired_fraction_of_evaluable"] == 0.5
    assert result["clamped_ticks"] == 1
    assert result["clamped_fraction"] == 0.5
    assert result["clamp_delta_mps"]["mean"] == 2.0
    assert result["lane_withheld_ticks"] == 2


def test_safety_result_ignores_ticks_predating_the_task() -> None:
    """A mixed log (recorded mid-upgrade) still rolls up the ticks that do
    carry a safety block, rather than refusing the whole run."""
    ticks = [
        {"safety": None},
        _tick(_safety_block(delta=0.0, rules={"forward_ttc": _rule_block("not_evaluable")})),
    ]
    result = safety_result(ticks)
    assert result["n_ticks"] == 1


def test_safety_lines_is_empty_for_a_run_that_predates_the_task() -> None:
    assert _safety_lines(None) == []


def test_safety_lines_names_zero_evaluable_rules_without_a_percentage() -> None:
    result = safety_result([_tick(_safety_block(delta=0.0, rules={
        "forward_ttc": _rule_block("not_evaluable", leader_gap_m_source="fallback_neutral"),
    }))])
    lines = _safety_lines(result)
    joined = "\n".join(lines)
    assert "## Safety" in joined
    assert "forward_ttc: not evaluable on any tick (0 of 1)" in joined
    assert "forward_ttc: 0%" not in joined
    # validator round 1, F6: named direction, not "clamped".
    assert "recommended speed raised on 0 of 1 ticks; lowered on 0 of 1 ticks" in joined
    assert "clamped" not in joined


def test_safety_lines_reports_a_rate_for_an_evaluable_rule() -> None:
    result = safety_result([
        _tick(_safety_block(delta=0.0, rules={"target_lane_front_gap": _rule_block("quiet")})),
        _tick(_safety_block(delta=0.0, rules={"target_lane_front_gap": _rule_block("fired")})),
    ])
    lines = _safety_lines(result)
    joined = "\n".join(lines)
    assert "target_lane_front_gap: evaluable on 2 of 2; fired on 1 (50.0%)" in joined


def test_safety_lines_names_a_non_finite_compared_value() -> None:
    """validator round 1, F5: the exact corpus shape -- evaluable (`derived`),
    quiet, and the compared gap is inf on every tick, so a 0.0% fired rate
    would otherwise read as a real (if low) firing rate."""
    result = safety_result([
        _tick(_safety_block(delta=0.0, rules={
            "target_lane_front_gap": _rule_block("quiet", gap_m=float("inf"), threshold_m=5.0),
        })),
        _tick(_safety_block(delta=0.0, rules={
            "target_lane_front_gap": _rule_block("quiet", gap_m=float("inf"), threshold_m=5.0),
        })),
    ])
    lines = _safety_lines(result)
    joined = "\n".join(lines)
    assert (
        "target_lane_front_gap: evaluable on 2 of 2; fired on 0 (0.0%); "
        "the compared value (gap_m) is inf on all 2"
    ) in joined


def test_safety_lines_states_raise_direction_with_median_and_mean() -> None:
    """validator round 1, F6."""
    result = safety_result([
        _tick(_safety_block(delta=2.0, rules={
            "low_speed_uncongested": _rule_block("fired", local_density_veh_per_km=2.0),
        })),
        _tick(_safety_block(delta=2.0, rules={
            "low_speed_uncongested": _rule_block("fired", local_density_veh_per_km=2.0),
        })),
    ])
    lines = _safety_lines(result)
    joined = "\n".join(lines)
    assert "recommended speed raised on 2 of 2 ticks, by a median of 2.00 m/s (mean 2.00)" in joined
    assert "lowered on 0 of 2 ticks" in joined
