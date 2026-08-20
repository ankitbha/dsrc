"""Unit tests for the channel table, plus a check that it still agrees with
the spec. The table is a contract two languages implement, so code and
specs/transport_protocol.md drifting apart is a real failure, not a docs nit."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from transport.channels import (
    CHANNEL_POLICIES,
    Channel,
    Direction,
    OverflowPolicy,
    Priority,
    UnknownChannelError,
    channels_by_priority,
    parse_channel,
    policy_for,
)

SPEC = Path(__file__).resolve().parents[3] / "specs" / "transport_protocol.md"


def test_every_channel_has_a_policy():
    assert set(CHANNEL_POLICIES) == set(Channel)


def test_policy_keys_match_their_channel():
    for channel, policy in CHANNEL_POLICIES.items():
        assert policy.channel is channel


def test_there_is_no_default_policy():
    with pytest.raises(UnknownChannelError):
        policy_for("lidar")
    with pytest.raises(UnknownChannelError):
        parse_channel("lidar")


def test_policy_for_accepts_a_string_or_an_enum():
    assert policy_for("camera") is policy_for(Channel.CAMERA)


def test_depths_are_positive():
    for policy in CHANNEL_POLICIES.values():
        assert policy.depth >= 1, policy


def test_latest_wins_implies_depth_one():
    """A latest-wins channel with room for two is a contradiction: the second
    message would sit behind a message the policy says is already worthless."""
    for policy in CHANNEL_POLICIES.values():
        if policy.overflow is OverflowPolicy.LATEST_WINS:
            assert policy.depth == 1, policy


def test_priority_order_is_high_normal_bulk():
    assert Priority.HIGH < Priority.NORMAL < Priority.BULK


def test_tiers_are_in_drain_order_and_cover_every_channel():
    tiers = channels_by_priority()
    assert [priority for priority, _ in tiers] == sorted(p for p, _ in tiers)
    covered = [channel for _, channels in tiers for channel in channels]
    assert sorted(covered, key=lambda c: c.value) == sorted(Channel, key=lambda c: c.value)
    assert len(covered) == len(set(covered))


def test_the_advisory_and_camera_channels_are_the_droppers():
    droppers = {
        channel
        for channel, policy in CHANNEL_POLICIES.items()
        if policy.overflow is OverflowPolicy.LATEST_WINS
    }
    assert droppers == {Channel.ADVISORY, Channel.CAMERA}


def test_control_and_commands_outrank_sensor_data():
    for channel in (Channel.CONTROL, Channel.RATE_CMD, Channel.ADVISORY):
        assert policy_for(channel).priority is Priority.HIGH
    assert policy_for(Channel.CAMERA).priority is Priority.BULK
    for channel in (Channel.GPS, Channel.IMU, Channel.HERE, Channel.TELEMETRY):
        assert policy_for(channel).priority is Priority.NORMAL


# -- the spec is the contract ------------------------------------------------


def spec_channel_rows() -> dict[str, dict[str, str]]:
    text = SPEC.read_text()
    rows = {}
    for line in text.splitlines():
        match = re.match(
            r"^\|\s*`(\w+)`\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|", line
        )
        if match:
            name, direction, priority, overflow, depth = match.groups()
            rows[name] = {
                "direction": direction,
                "priority": priority,
                "overflow": overflow,
                "depth": depth,
            }
    return rows


def test_spec_table_is_parseable_and_complete():
    rows = spec_channel_rows()
    assert set(rows) == {c.value for c in Channel}, rows


@pytest.mark.parametrize("channel", list(Channel), ids=lambda c: c.value)
def test_code_matches_the_spec_table(channel):
    row = spec_channel_rows()[channel.value]
    policy = policy_for(channel)
    assert policy.direction is Direction(row["direction"])
    assert policy.priority is Priority[row["priority"].upper()]
    assert policy.overflow is OverflowPolicy(row["overflow"])
    assert policy.depth == int(row["depth"])
