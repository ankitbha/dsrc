"""Unit tests for ui/dashboard.py's driver-facing text construction.

render_dashboard itself draws into a raster image via cv2.putText, which a
unit test cannot assert text content against without OCR -- so the
recommended-speed line's construction is pulled out into
_recommended_speed_line, a pure function this file tests directly.
"""

from __future__ import annotations

import pytest

import ui.dashboard as dashboard
from policy.advisory import Advisory
from ui.dashboard import GREEN, RED, _recommended_speed_line


def _advisory(**over) -> Advisory:
    fields = dict(
        recommended_speed_mps=10.0,
        recommended_speed_display=22.0,
        current_speed_display=20.0,
        units="mph",
        headway_target_s=1.6,
        lane_text="Keep lane",
        merge_text="Normal driving",
        traffic_text="Light",
        confidence_label="high",
        confidence=0.9,
        action={
            "desired_speed_bin": "nominal", "desired_headway_bin": "normal",
            "lane_preference": "keep", "merge_mode": "normal",
        },
    )
    fields.update(over)
    return Advisory(**fields)


def test_recommended_speed_is_shown_normally_when_not_withheld() -> None:
    text, color = _recommended_speed_line(_advisory(speed_display_withheld=False))
    assert "22" in text
    assert "mph" in text
    assert color == GREEN


def test_recommended_speed_is_withheld_on_an_emergency_override() -> None:
    """validator round 1, F7: plan step 6 says to withhold the speed number
    on an emergency override. `Advisory.speed_display_withheld` was written
    into the per-tick record (task 144, open item 3) and read by no surface
    -- not `ui/dashboard.py`, not `advisory_message_from_advisory`, not
    `replay_demo.py`, and by no test -- before this fix."""
    text, color = _recommended_speed_line(
        _advisory(speed_display_withheld=True, recommended_speed_display=99.0)
    )
    assert "99" not in text
    assert "WITHHELD" in text
    assert color == RED


def test_the_withheld_test_fails_against_the_unfixed_renderer(monkeypatch) -> None:
    """Neuters the fix -- makes the rendered line ignore
    `speed_display_withheld` again, exactly as it read before this fix --
    and confirms the test above would then fail. Constructed directly on
    `Advisory` rather than by arranging a real tick to produce an override:
    `speed_display_withheld` is set from `gate_result.emergency_override`,
    and forward_ttc's own two other reads (the merge-conflict pair) are
    always class (C) on this rig, so a test that reached this branch only
    through a real tick would have a premise that may never be active --
    the same shape as a frozen clock that never advances.
    """
    def unfixed(adv: Advisory) -> tuple[str, tuple[int, int, int]]:
        return f"Recommended: {adv.recommended_speed_display:5.0f} {adv.units}", GREEN

    monkeypatch.setattr(dashboard, "_recommended_speed_line", unfixed)
    text, _ = dashboard._recommended_speed_line(
        _advisory(speed_display_withheld=True, recommended_speed_display=99.0)
    )
    with pytest.raises(AssertionError):
        assert "99" not in text
