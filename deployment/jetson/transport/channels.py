"""Channel table: what each channel is, and what happens when it backs up.

One connection carries every kind of traffic, so the transport needs a per-
channel answer to two questions: who goes first when several messages are
waiting, and who gets thrown away when the link cannot keep up. Those answers
are data, not code, and they live here.

The policies encode one asymmetry. A camera frame is a snapshot: if the link
is behind, the newest frame is the only one worth sending and the rest are
dead weight. A GPS fix, an IMU sample or a rate command is a record: no later
message repeats what it said. So the camera drops and the small channels
queue.

`direction` is documentation. The transport does not reject a frame for
arriving on a channel that nominally flows the other way; nothing needs that
rule and enforcing it would make the transport care what messages mean.

See specs/transport_protocol.md for the wire-level contract this mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping


class Channel(str, Enum):
    CONTROL = "control"
    RATE_CMD = "rate_cmd"
    ADVISORY = "advisory"
    GPS = "gps"
    IMU = "imu"
    HERE = "here"
    TELEMETRY = "telemetry"
    CAMERA = "camera"


class Priority(IntEnum):
    """Lower value drains first."""

    HIGH = 0
    NORMAL = 1
    BULK = 2


class Direction(str, Enum):
    UP = "up"        # phone -> jetson
    DOWN = "down"    # jetson -> phone
    BOTH = "both"


class OverflowPolicy(str, Enum):
    RELIABLE = "reliable"        # queue to `depth`, then drop the oldest
    LATEST_WINS = "latest_wins"  # hold one; a new message replaces it


@dataclass(frozen=True)
class ChannelPolicy:
    channel: Channel
    direction: Direction
    priority: Priority
    overflow: OverflowPolicy
    depth: int


CHANNEL_POLICIES: Mapping[Channel, ChannelPolicy] = {
    policy.channel: policy
    for policy in (
        ChannelPolicy(Channel.CONTROL, Direction.BOTH, Priority.HIGH, OverflowPolicy.RELIABLE, 8),
        ChannelPolicy(Channel.RATE_CMD, Direction.DOWN, Priority.HIGH, OverflowPolicy.RELIABLE, 16),
        ChannelPolicy(Channel.ADVISORY, Direction.DOWN, Priority.HIGH, OverflowPolicy.LATEST_WINS, 1),
        ChannelPolicy(Channel.GPS, Direction.UP, Priority.NORMAL, OverflowPolicy.RELIABLE, 64),
        ChannelPolicy(Channel.IMU, Direction.UP, Priority.NORMAL, OverflowPolicy.RELIABLE, 256),
        ChannelPolicy(Channel.HERE, Direction.UP, Priority.NORMAL, OverflowPolicy.RELIABLE, 16),
        ChannelPolicy(Channel.TELEMETRY, Direction.UP, Priority.NORMAL, OverflowPolicy.RELIABLE, 32),
        ChannelPolicy(Channel.CAMERA, Direction.UP, Priority.BULK, OverflowPolicy.LATEST_WINS, 1),
    )
}


class UnknownChannelError(ValueError):
    """A channel id with no policy. There is deliberately no default."""


def parse_channel(value: str) -> Channel:
    try:
        return Channel(value)
    except ValueError:
        raise UnknownChannelError(f"unknown channel {value!r}") from None


def policy_for(channel: Channel | str) -> ChannelPolicy:
    if not isinstance(channel, Channel):
        channel = parse_channel(str(channel))
    try:
        return CHANNEL_POLICIES[channel]
    except KeyError:
        raise UnknownChannelError(f"no policy for channel {channel!r}") from None


def channels_by_priority() -> tuple[tuple[Priority, tuple[Channel, ...]], ...]:
    """Channels grouped into priority tiers, tiers in drain order.

    The writer walks these tiers in order and round-robins inside a tier, so
    the group order here is the send order and the within-group order is only
    the starting point of the rotation.
    """
    tiers: dict[Priority, list[Channel]] = {}
    for policy in CHANNEL_POLICIES.values():
        tiers.setdefault(policy.priority, []).append(policy.channel)
    return tuple((priority, tuple(tiers[priority])) for priority in sorted(tiers))
