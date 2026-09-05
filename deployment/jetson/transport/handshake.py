"""The hello exchange that opens every connection.

Three things it establishes, in order of how much trouble their absence causes:

Version agreement. The phone app and the Jetson runtime are deployed
separately, so a mismatch is a question of when, not whether. Without a
handshake a stale peer silently misparses; with one it is refused and logged.

Identity. Which device and which role is on the other end, so a session in a
log can be attributed.

A first clock sample. Four timestamps -- local monotonic at send, the peer's
monotonic and wall, local monotonic at receipt -- which is what an offset
estimator needs to start. This module records them and computes nothing from
them; the estimator is a separate concern.

Both sides send before either reads. A hello is a couple of hundred bytes, far
inside any socket buffer, so neither send can block on the peer not having read
yet and the exchange cannot deadlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from transport.channels import Channel
from transport.clock import MonoClock, WallClock, now_mono_ns, now_wall_ns
from transport.connection import ByteConnection
from transport.frames import (
    HELLO_KEY,
    PROTOCOL_VERSION,
    Frame,
    FramingError,
    encode,
    read_frame,
)


class Role(str, Enum):
    PHONE = "phone"
    JETSON = "jetson"


class HandshakeError(Exception):
    """The connection did not open correctly. Terminal."""


class VersionMismatch(HandshakeError):
    def __init__(self, local: int, remote: int) -> None:
        super().__init__(f"protocol version mismatch: local {local}, remote {remote}")
        self.local = local
        self.remote = remote


@dataclass(frozen=True)
class Hello:
    device_id: str
    role: Role
    protocol_version: int = PROTOCOL_VERSION

    def to_header_value(self) -> dict[str, Any]:
        return {
            "protocol_version": int(self.protocol_version),
            "device_id": str(self.device_id),
            "role": self.role.value,
        }


@dataclass(frozen=True)
class ClockSample:
    """One round of timestamps. Local and remote monotonics are different
    clocks; the whole point of keeping all four is that the offset between
    them can be estimated later."""

    t_local_send_mono_ns: int
    t_local_recv_mono_ns: int
    t_remote_mono_ns: int
    t_remote_wall_ns: int
    #: This device's own wall clock at the same instant `t_local_send_mono_ns`
    #: was taken -- local only, never encoded onto the wire (`hello_frame`
    #: puts a wall stamp in the outgoing header too, but that is a separate
    #: call the OTHER side reads back as ITS `t_remote_wall_ns`). Captured so
    #: a later reader can compare THIS device's wall clock against
    #: `t_remote_wall_ns` without assuming the two were taken at the same
    #: moment some other way (B10, validation round 2): before this field
    #: existed, nothing outside this module read `t_remote_wall_ns` at all.
    t_local_wall_ns: int

    @property
    def round_trip_ns(self) -> int:
        return self.t_local_recv_mono_ns - self.t_local_send_mono_ns

    @property
    def remote_minus_local_wall_s(self) -> float:
        """Remote wall clock minus local wall clock, in seconds -- positive
        means the remote (the phone, on the Jetson's own listener side of
        every session) is ahead.

        NOT two simultaneous stamps (B13, validation round 3):
        `t_local_wall_ns` is taken at this device's own hello-send,
        `t_remote_wall_ns` is the peer's wall clock at ITS hello-send, and
        both sides send before either reads (this module's own docstring),
        so the two instants are up to one round trip apart, with the
        remote's always the later one. That makes this value systematically
        overstate how far ahead the remote is, by up to
        `wall_clock_offset_bound_s`. A caller that needs the bound alongside
        the estimate reads that property too, rather than treating this one
        as exact.
        """
        return (self.t_remote_wall_ns - self.t_local_wall_ns) / 1e9

    @property
    def wall_clock_offset_bound_s(self) -> float:
        """Upper bound on the error in `remote_minus_local_wall_s`, in
        seconds (B13, validation round 3).

        `t_remote_wall_ns` is the peer's wall clock at its OWN hello-send,
        not at the instant `t_local_wall_ns` was taken, and the two stamps
        are up to one round trip (`round_trip_ns`) apart. This codebase's
        rule for a cross-device number is to carry its bound alongside it
        (`TimebaseStamp.bound_s`, `StageTiming.converted(bound_ms=...)`);
        `remote_minus_local_wall_s` was the exception until this property
        existed.
        """
        return self.round_trip_ns / 1e9


@dataclass(frozen=True)
class HandshakeResult:
    local: Hello
    remote: Hello
    clock: ClockSample


def hello_frame(hello: Hello, *, t_mono_ns: int, t_wall_ns: int) -> Frame:
    return Frame(
        channel=Channel.CONTROL,
        seq=0,
        t_mono_ns=t_mono_ns,
        t_wall_ns=t_wall_ns,
        payload=b"",
        extensions={HELLO_KEY: hello.to_header_value()},
    )


def parse_hello(frame: Frame) -> Hello:
    if frame.channel is not Channel.CONTROL:
        raise HandshakeError(f"hello must arrive on control, got {frame.channel.value}")
    value = frame.extensions.get(HELLO_KEY)
    if value is None:
        raise HandshakeError("first frame carries no hello")
    if not isinstance(value, Mapping):
        raise HandshakeError(f"hello is {type(value).__name__}, expected object")

    missing = [k for k in ("protocol_version", "device_id", "role") if k not in value]
    if missing:
        raise HandshakeError(f"hello missing {', '.join(missing)}")

    version = value["protocol_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise HandshakeError(f"hello protocol_version is {type(version).__name__}, expected int")
    try:
        role = Role(value["role"])
    except ValueError:
        raise HandshakeError(f"unknown role {value['role']!r}") from None
    device_id = value["device_id"]
    if not isinstance(device_id, str) or not device_id:
        raise HandshakeError("hello device_id must be a non-empty string")

    return Hello(device_id=device_id, role=role, protocol_version=version)


def perform_handshake(
    connection: ByteConnection,
    local: Hello,
    *,
    mono_clock: MonoClock = now_mono_ns,
    wall_clock: WallClock = now_wall_ns,
) -> HandshakeResult:
    """Send our hello, read theirs, and refuse a version we do not speak.

    Raises HandshakeError (or VersionMismatch) without reading any data frame,
    so a mismatched peer never gets a message interpreted.
    """
    t_send = mono_clock()
    t_send_wall = wall_clock()
    connection.send_all(encode(hello_frame(local, t_mono_ns=t_send, t_wall_ns=t_send_wall)))

    try:
        frame = read_frame(connection.recv_exact)
    except FramingError as exc:
        raise HandshakeError(f"malformed hello: {exc}") from None
    t_recv = mono_clock()

    remote = parse_hello(frame)
    if remote.protocol_version != local.protocol_version:
        raise VersionMismatch(local.protocol_version, remote.protocol_version)

    return HandshakeResult(
        local=local,
        remote=remote,
        clock=ClockSample(
            t_local_send_mono_ns=t_send,
            t_local_recv_mono_ns=t_recv,
            t_remote_mono_ns=frame.t_mono_ns,
            t_remote_wall_ns=frame.t_wall_ns,
            t_local_wall_ns=t_send_wall,
        ),
    )
