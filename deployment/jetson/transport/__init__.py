"""Framed bidirectional transport between the phone and the Jetson.

The phone opens the connection; the Jetson listens. See
specs/transport_protocol.md for the wire format, which is a cross-language
contract, and specs/transport_golden_frames.json for the frozen encodings both
implementations are held to.

The transport is opaque: it moves (channel, header, payload) and assigns no
meaning to any payload.
"""

from transport.channels import (
    CHANNEL_POLICIES,
    Channel,
    ChannelPolicy,
    Direction,
    OverflowPolicy,
    Priority,
    UnknownChannelError,
    channels_by_priority,
    parse_channel,
    policy_for,
)
from transport.clock import now_mono_ns, now_wall_ns
from transport.connection import ByteConnection, ConnectionClosed
from transport.endpoint import (
    Acceptor,
    SessionEnded,
    SessionEvent,
    SessionRefused,
    SessionStarted,
    TransportListener,
)
from transport.frames import (
    HEARTBEAT_KEY,
    HELLO_KEY,
    MAX_HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    PREFIX_BYTES,
    PROTOCOL_VERSION,
    RESERVED_EXTENSIONS,
    Frame,
    FramingError,
    decode,
    encode,
    encode_header,
    read_frame,
)
from transport.handshake import (
    ClockSample,
    HandshakeError,
    HandshakeResult,
    Hello,
    Role,
    VersionMismatch,
    hello_frame,
    parse_hello,
    perform_handshake,
)
from transport.loopback import LoopbackAcceptor, LoopbackConnection, loopback_pair
from transport.session import (
    DEFAULT_HEARTBEAT_S,
    DEFAULT_STALL_TIMEOUT_S,
    ChannelStats,
    ReceivedMessage,
    Session,
    SessionEndReason,
    SessionStats,
)

__all__ = [
    "Acceptor",
    "ByteConnection",
    "CHANNEL_POLICIES",
    "Channel",
    "ChannelPolicy",
    "ChannelStats",
    "ClockSample",
    "ConnectionClosed",
    "DEFAULT_HEARTBEAT_S",
    "DEFAULT_STALL_TIMEOUT_S",
    "Direction",
    "Frame",
    "FramingError",
    "HEARTBEAT_KEY",
    "HELLO_KEY",
    "HandshakeError",
    "HandshakeResult",
    "Hello",
    "LoopbackAcceptor",
    "LoopbackConnection",
    "MAX_HEADER_BYTES",
    "MAX_PAYLOAD_BYTES",
    "OverflowPolicy",
    "PREFIX_BYTES",
    "PROTOCOL_VERSION",
    "Priority",
    "RESERVED_EXTENSIONS",
    "ReceivedMessage",
    "Role",
    "Session",
    "SessionEndReason",
    "SessionEnded",
    "SessionEvent",
    "SessionRefused",
    "SessionStarted",
    "SessionStats",
    "TransportListener",
    "UnknownChannelError",
    "VersionMismatch",
    "channels_by_priority",
    "decode",
    "encode",
    "encode_header",
    "hello_frame",
    "loopback_pair",
    "now_mono_ns",
    "now_wall_ns",
    "parse_channel",
    "parse_hello",
    "policy_for",
    "read_frame",
]
