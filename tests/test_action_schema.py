from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.envs.wrappers import (
    HEADWAY_BINS,
    LANE_PREFERENCES,
    MERGE_MODES,
    SPEED_BINS,
    validate_action,
)

REPO = Path(__file__).resolve().parents[1]

#: The four heads, and the module that owns each one's vocabulary. This check
#: used to live in `deployment/jetson/tests/test_transport_messages.py`,
#: against a copy of the vocabulary the advisory channel carried. The advisory
#: is a speed now and carries no action, so that copy was deleted and the
#: check moved here, beside the implementation the spec actually documents.
HEAD_VALUES: dict[str, tuple[str, ...]] = {
    "desired_speed_bin": SPEED_BINS,
    "desired_headway_bin": HEADWAY_BINS,
    "lane_preference": LANE_PREFERENCES,
    "merge_mode": MERGE_MODES,
}

#: `normal` appears in more than one spec block legitimately: it is a headway
#: bin and a merge mode. Any other overlap is drift.
LEGITIMATELY_SHARED_VALUES = {"normal"}


def spec_allowed_values(head: str) -> set[str]:
    """The values under one head's `allowed values:` block, and no further.

    Anchored to the block rather than to the next head: a window running to the
    next head swept up three other heads' values, so the check passed with a
    value listed under the wrong head -- which is precisely the drift it exists
    to catch.
    """
    text = (REPO / "specs" / "action_schema.md").read_text()
    after_head = text.split(f"`{head}`", 1)
    assert len(after_head) == 2, f"{head} is not documented"
    after_allowed = after_head[1].split("- allowed values:", 1)
    assert len(after_allowed) == 2, f"{head} has no allowed-values block"
    block = after_allowed[1].split("- meaning:", 1)[0]
    return set(re.findall(r"^\s*-\s+`([\w_]+)`\s*$", block, re.M))


def test_the_spec_blocks_match_the_implementation():
    for head, values in HEAD_VALUES.items():
        assert spec_allowed_values(head) == set(values), head


def test_the_spec_blocks_are_pairwise_disjoint_apart_from_the_shared_value():
    """A property of the spec text alone, independent of the code. A value
    appears under one head, except `normal`, which really is both a headway
    bin and a merge mode."""
    blocks = {head: spec_allowed_values(head) for head in HEAD_VALUES}
    heads = sorted(HEAD_VALUES)
    for left in heads:
        for right in heads:
            if left >= right:
                continue
            shared = blocks[left] & blocks[right]
            assert shared <= LEGITIMATELY_SHARED_VALUES, (
                f"{left} and {right} both list {sorted(shared - LEGITIMATELY_SHARED_VALUES)}"
            )
    assert blocks["desired_headway_bin"] & blocks["merge_mode"] == {"normal"}


def safe_action(**overrides: str) -> dict[str, str]:
    action = {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
    action.update(overrides)
    return action


def test_v2_action_schema_accepts_allowed_bins() -> None:
    assert validate_action(safe_action()) == safe_action()
    assert validate_action(safe_action(desired_speed_bin="slow"))["desired_speed_bin"] == "slow"
    assert validate_action(safe_action(desired_headway_bin="largest"))["desired_headway_bin"] == "largest"
    assert validate_action(safe_action(lane_preference="prefer_right_if_safe"))["lane_preference"] == "prefer_right_if_safe"
    assert validate_action(safe_action(merge_mode="create_gap"))["merge_mode"] == "create_gap"


@pytest.mark.parametrize(
    "bad_action",
    [
        {"desired_speed": 22.0, "desired_lane": "keep"},
        safe_action(lane_preference="left"),
        safe_action(lane_preference="right"),
        safe_action(merge_mode="block_lanes"),
        {"desired_speed_bin": "nominal", "lane_preference": "keep", "merge_mode": "normal"},
    ],
)
def test_v2_action_schema_rejects_legacy_or_unsafe_actions(bad_action: dict) -> None:
    with pytest.raises(ValueError):
        validate_action(bad_action)



class TestTheSpeedBinsAgainstTheOperatingPoint:
    """What the three speed values actually command, and when they can bind.

    Measured, not asserted: over 114,889 AV-steps at `sumo_capacity_drop` with a
    mean AV speed of 7.33 m/s, `slow` binds on 15.6% of them, `nominal` on 0.9% and
    `fast` on 0.07%. `setSpeed` is an upper bound that SUMO's car-following
    dominates, so a value above the speed the vehicle would take anyway changes
    nothing, and on 84% of AV-steps all three values do the same thing.

    These tests do not claim that is right or wrong -- rescaling the bins changes
    what the deployed actor's heads mean, which is a decision on record for the
    user. They make the numbers executable, so a change to the offsets or the floor
    shows up here rather than only in a plan file.
    """

    def test_the_three_values_at_a_thirty_metre_limit(self):
        from src.envs.wrappers import decode_speed_bin

        assert decode_speed_bin("slow", free_flow_speed_mps=30.0) == 20.0
        assert decode_speed_bin("nominal", free_flow_speed_mps=30.0) == 27.0
        assert decode_speed_bin("fast", free_flow_speed_mps=30.0) == 30.0

    def test_the_floor_holds_every_value_above_twelve(self):
        from src.envs.wrappers import decode_speed_bin

        # At a congested operating point the achievable speed is well under the
        # floor, so the floor and not the offset is what makes the values equivalent.
        for value in ("slow", "nominal", "fast"):
            assert decode_speed_bin(value, free_flow_speed_mps=8.0) == 12.0
        assert decode_speed_bin("slow", free_flow_speed_mps=15.0) == 12.0

    def test_the_values_are_ordered_and_distinct_at_the_limit(self):
        from src.envs.wrappers import decode_speed_bin

        # The control for the two tests above: the head is not degenerate by
        # construction. It is the operating point that makes it so.
        values = [decode_speed_bin(name, free_flow_speed_mps=30.0)
                  for name in ("slow", "nominal", "fast")]
        assert values == sorted(values)
        assert len(set(values)) == 3
