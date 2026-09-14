"""Unit tests for ui/dashboard.py's driver-facing text construction.

render_dashboard itself draws into a raster image via cv2.putText, which a
unit test cannot assert text content against without OCR -- so the
recommended-speed line's construction is pulled out into
_recommended_speed_line, a pure function this file tests directly.
"""

from __future__ import annotations

import pytest

import ui.dashboard as dashboard
from policy.segment_advisory import SegmentAdvisoryRow
from ui.dashboard import GRAY, GREEN, RED, _recommended_speed_line


def _row(**over) -> SegmentAdvisoryRow:
    fields = dict(
        segment_id="S03",
        action_index=2,
        fraction=1.0,
        recommended_speed_mps=10.0,
        recommended_speed_display=22.0,
    )
    fields.update(over)
    return SegmentAdvisoryRow(**fields)


def test_recommended_speed_is_shown_normally_when_not_withheld() -> None:
    text, color = _recommended_speed_line(_row(), "mph", withheld=False)
    assert "22" in text
    assert "mph" in text
    assert color == GREEN


def test_recommended_speed_is_withheld_on_an_emergency_override() -> None:
    """validator round 1, F7: plan step 6 says to withhold the speed number
    on an emergency override. `speed_display_withheld` was written into the
    per-tick record (task 144, open item 3) and read by no surface -- not
    `ui/dashboard.py`, not `advisory_message_from_advisory`, not
    `replay_demo.py`, and by no test -- before this fix."""
    text, color = _recommended_speed_line(
        _row(recommended_speed_display=99.0), "mph", withheld=True
    )
    assert "99" not in text
    assert "WITHHELD" in text
    assert color == RED


def test_no_row_is_shown_as_no_advisory_rather_than_blank() -> None:
    """A rig driving a road its policy does not cover has no ego row at all.
    That is a third outcome, not a zero and not a withholding, so it gets its
    own line: a blank panel and a `0 mph` recommendation are both readable as
    an advisory that was made."""
    text, color = _recommended_speed_line(None, "mph", withheld=False)
    assert "none for this road" in text
    assert color == GRAY


def test_the_withheld_test_fails_against_the_unfixed_renderer(monkeypatch) -> None:
    """Neuters the fix -- makes the rendered line ignore the withheld flag
    again, exactly as it read before this fix -- and confirms the test above
    would then fail. Constructed directly on a row rather than by arranging a
    real tick to produce an override: the flag is set from
    `gate_result.emergency_override`, and forward_ttc's own two other reads
    (the merge-conflict pair) are always class (C) on this rig, so a test that
    reached this branch only through a real tick would have a premise that may
    never be active -- the same shape as a frozen clock that never advances.
    """
    def unfixed(
        row: SegmentAdvisoryRow | None, units: str, withheld: bool,
    ) -> tuple[str, tuple[int, int, int]]:
        return f"Recommended: {row.recommended_speed_display:5.0f} {units}", GREEN

    monkeypatch.setattr(dashboard, "_recommended_speed_line", unfixed)
    text, _ = dashboard._recommended_speed_line(
        _row(recommended_speed_display=99.0), "mph", withheld=True
    )
    with pytest.raises(AssertionError):
        assert "99" not in text
