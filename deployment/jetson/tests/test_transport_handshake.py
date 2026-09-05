"""Unit tests for the hello exchange: version agreement, identity, and the
first clock sample. A mismatched peer must be refused before any data frame
is interpreted."""

from __future__ import annotations

import threading

import pytest

from transport.channels import Channel
from transport.frames import HELLO_KEY, PROTOCOL_VERSION, Frame, encode
from transport.handshake import (
    HandshakeError,
    Hello,
    Role,
    VersionMismatch,
    hello_frame,
    parse_hello,
    perform_handshake,
)
from transport.loopback import loopback_pair

PHONE = Hello(device_id="moto-g-power", role=Role.PHONE)
JETSON = Hello(device_id="jetson-orin", role=Role.JETSON)


def raw_hello(value, channel=Channel.CONTROL) -> Frame:
    extensions = {} if value is None else {HELLO_KEY: value}
    return Frame(channel=channel, seq=0, t_mono_ns=1, t_wall_ns=2, extensions=extensions)


# -- encoding ----------------------------------------------------------------


def test_hello_frame_is_a_control_frame_at_seq_zero():
    frame = hello_frame(PHONE, t_mono_ns=5, t_wall_ns=6)
    assert frame.channel is Channel.CONTROL
    assert frame.seq == 0
    assert frame.payload == b""
    assert frame.extensions[HELLO_KEY]["role"] == "phone"


def test_hello_roundtrips_through_the_codec():
    frame = hello_frame(PHONE, t_mono_ns=5, t_wall_ns=6)
    from transport.frames import decode

    assert parse_hello(decode(encode(frame))) == PHONE


def test_default_version_is_the_protocol_version():
    assert PHONE.protocol_version == PROTOCOL_VERSION


# -- parse refusals ----------------------------------------------------------


def test_hello_must_arrive_on_control():
    with pytest.raises(HandshakeError, match="control"):
        parse_hello(raw_hello(PHONE.to_header_value(), channel=Channel.GPS))


def test_missing_hello_is_refused():
    with pytest.raises(HandshakeError, match="no hello"):
        parse_hello(raw_hello(None))


@pytest.mark.parametrize("value", ["text", 5, [1, 2], True])
def test_hello_must_be_an_object(value):
    with pytest.raises(HandshakeError, match="expected object"):
        parse_hello(raw_hello(value))


@pytest.mark.parametrize("missing", ["protocol_version", "device_id", "role"])
def test_incomplete_hello_is_refused(missing):
    value = PHONE.to_header_value()
    del value[missing]
    with pytest.raises(HandshakeError, match=missing):
        parse_hello(raw_hello(value))


def test_unknown_role_is_refused():
    value = PHONE.to_header_value()
    value["role"] = "laptop"
    with pytest.raises(HandshakeError, match="laptop"):
        parse_hello(raw_hello(value))


@pytest.mark.parametrize("device_id", ["", 7, None])
def test_bad_device_id_is_refused(device_id):
    value = PHONE.to_header_value()
    value["device_id"] = device_id
    with pytest.raises(HandshakeError, match="device_id"):
        parse_hello(raw_hello(value))


@pytest.mark.parametrize("version", ["1", 1.0, None, True])
def test_non_integer_version_is_refused(version):
    value = PHONE.to_header_value()
    value["protocol_version"] = version
    with pytest.raises(HandshakeError, match="protocol_version"):
        parse_hello(raw_hello(value))


# -- exchange ----------------------------------------------------------------


def exchange(phone_hello=PHONE, jetson_hello=JETSON):
    """Both sides handshake concurrently, as they do on a real connection."""
    phone_conn, jetson_conn = loopback_pair()
    results: dict[str, object] = {}

    def phone_side():
        try:
            results["phone"] = perform_handshake(phone_conn, phone_hello)
        except Exception as exc:  # recorded, asserted on by the caller
            results["phone"] = exc

    thread = threading.Thread(target=phone_side, daemon=True)
    thread.start()
    try:
        results["jetson"] = perform_handshake(jetson_conn, jetson_hello)
    except Exception as exc:
        results["jetson"] = exc
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    return results


def test_matching_versions_succeed_on_both_sides():
    results = exchange()
    assert results["jetson"].remote == PHONE
    assert results["jetson"].local == JETSON
    assert results["phone"].remote == JETSON


def test_mismatched_version_is_refused_with_both_versions_named():
    results = exchange(phone_hello=Hello("old-phone", Role.PHONE, protocol_version=99))
    error = results["jetson"]
    assert isinstance(error, VersionMismatch)
    assert error.local == PROTOCOL_VERSION and error.remote == 99
    assert "99" in str(error) and str(PROTOCOL_VERSION) in str(error)


def test_clock_sample_carries_all_four_timestamps():
    sample = exchange()["jetson"].clock
    assert sample.t_local_recv_mono_ns >= sample.t_local_send_mono_ns
    assert sample.round_trip_ns >= 0
    assert sample.t_remote_mono_ns > 0
    assert sample.t_remote_wall_ns > 1_600_000_000_000_000_000  # a real epoch in ns


def test_clock_sample_uses_injected_clocks():
    phone_conn, jetson_conn = loopback_pair()
    thread = threading.Thread(
        target=lambda: perform_handshake(
            phone_conn, PHONE, mono_clock=lambda: 500, wall_clock=lambda: 900
        ),
        daemon=True,
    )
    thread.start()
    ticks = iter([10, 20])
    result = perform_handshake(
        jetson_conn, JETSON, mono_clock=lambda: next(ticks), wall_clock=lambda: 77
    )
    thread.join(timeout=2.0)
    assert result.clock.t_local_send_mono_ns == 10
    assert result.clock.t_local_recv_mono_ns == 20
    assert result.clock.t_remote_mono_ns == 500
    assert result.clock.t_remote_wall_ns == 900
    # B10 (validation round 2): the Jetson's OWN wall clock at hello-send
    # time, captured so a caller outside this module can compare it
    # against t_remote_wall_ns without assuming the two were taken at the
    # same moment some other way.
    assert result.clock.t_local_wall_ns == 77


def test_remote_minus_local_wall_s_is_the_measured_clock_offset():
    """Positive means the remote (the phone, on every session this device
    accepts) is ahead -- matches the measured real-hardware direction
    (+0.935 to +0.952 s, phone ahead of the Jetson)."""
    phone_conn, jetson_conn = loopback_pair()
    thread = threading.Thread(
        target=lambda: perform_handshake(
            phone_conn, PHONE, mono_clock=lambda: 500, wall_clock=lambda: 100_935_000_000
        ),
        daemon=True,
    )
    thread.start()
    result = perform_handshake(
        jetson_conn, JETSON, mono_clock=lambda: 10, wall_clock=lambda: 100_000_000_000,
    )
    thread.join(timeout=2.0)
    assert result.clock.remote_minus_local_wall_s == pytest.approx(0.935, abs=1e-9)


def test_remote_minus_local_wall_s_is_negative_when_the_remote_is_behind():
    phone_conn, jetson_conn = loopback_pair()
    thread = threading.Thread(
        target=lambda: perform_handshake(
            phone_conn, PHONE, mono_clock=lambda: 500, wall_clock=lambda: 99_500_000_000
        ),
        daemon=True,
    )
    thread.start()
    result = perform_handshake(
        jetson_conn, JETSON, mono_clock=lambda: 10, wall_clock=lambda: 100_000_000_000,
    )
    thread.join(timeout=2.0)
    assert result.clock.remote_minus_local_wall_s == pytest.approx(-0.5, abs=1e-9)


def test_wall_clock_offset_bound_s_is_the_round_trip_in_seconds():
    """B13 (validation round 3): `remote_minus_local_wall_s` compares two
    hello-sends up to one round trip apart, not two simultaneous stamps --
    the bound on that error is exactly this side's own round trip,
    converted to seconds."""
    phone_conn, jetson_conn = loopback_pair()
    thread = threading.Thread(
        target=lambda: perform_handshake(
            phone_conn, PHONE, mono_clock=lambda: 500, wall_clock=lambda: 900
        ),
        daemon=True,
    )
    thread.start()
    ticks = iter([10_000_000, 45_000_000])
    result = perform_handshake(
        jetson_conn, JETSON, mono_clock=lambda: next(ticks), wall_clock=lambda: 77
    )
    thread.join(timeout=2.0)
    assert result.clock.round_trip_ns == 35_000_000
    assert result.clock.wall_clock_offset_bound_s == pytest.approx(0.035, abs=1e-9)


def test_wall_clock_offset_bound_s_is_zero_for_an_instant_round_trip():
    """Not a claim this ever happens on real hardware -- a same-tick
    round trip is the smallest possible bound, and the property should not
    invent a floor under it."""
    phone_conn, jetson_conn = loopback_pair()
    thread = threading.Thread(
        target=lambda: perform_handshake(
            phone_conn, PHONE, mono_clock=lambda: 500, wall_clock=lambda: 900
        ),
        daemon=True,
    )
    thread.start()
    result = perform_handshake(
        jetson_conn, JETSON, mono_clock=lambda: 10, wall_clock=lambda: 77,
    )
    thread.join(timeout=2.0)
    assert result.clock.wall_clock_offset_bound_s == 0.0


def test_a_non_hello_first_frame_is_refused():
    phone_conn, jetson_conn = loopback_pair()
    phone_conn.send_all(
        encode(Frame(channel=Channel.GPS, seq=0, t_mono_ns=1, t_wall_ns=2, payload=b"fix"))
    )
    with pytest.raises(HandshakeError):
        perform_handshake(jetson_conn, JETSON)


def test_a_malformed_first_frame_is_refused_as_a_handshake_error():
    phone_conn, jetson_conn = loopback_pair()
    phone_conn.send_all(b"\x00\x00\x00\x00\x00\x02{}")
    with pytest.raises(HandshakeError, match="malformed hello"):
        perform_handshake(jetson_conn, JETSON)


def test_mismatch_leaves_the_queued_data_frame_unread():
    """No message from a peer we do not speak to gets interpreted."""
    phone_conn, jetson_conn = loopback_pair()
    phone_conn.send_all(
        encode(hello_frame(Hello("old", Role.PHONE, protocol_version=99), t_mono_ns=1, t_wall_ns=2))
    )
    phone_conn.send_all(
        encode(Frame(channel=Channel.GPS, seq=0, t_mono_ns=3, t_wall_ns=4, payload=b"secret"))
    )
    with pytest.raises(VersionMismatch):
        perform_handshake(jetson_conn, JETSON)
    # The data frame is still sitting in the stream, untouched.
    assert phone_conn.unread_bytes > 0


# -- the clock sample is the first thing the offset estimator gets -----------


class ClockAdvancingConnection:
    """A connection whose reads advance the clock the handshake is told to use.

    This is what separates a receive stamp taken after the peer's hello was
    read from one taken before it. A clock fed from a fixed list cannot: both
    orderings consume the same two values.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.reads = 0

    @property
    def peer(self) -> str:
        return self._inner.peer

    def send_all(self, data: bytes) -> None:
        self._inner.send_all(data)

    def recv_exact(self, n: int) -> bytes:
        data = self._inner.recv_exact(n)
        self.reads += 1
        return data

    def close(self) -> None:
        self._inner.close()


def test_the_receive_stamp_is_taken_after_the_hello_has_been_read():
    """Otherwise round_trip_ns measures nothing at all, and it is the first
    sample the clock-offset estimator has to work from."""
    phone_conn, jetson_conn = loopback_pair()
    wrapped = ClockAdvancingConnection(jetson_conn)

    thread = threading.Thread(
        target=lambda: perform_handshake(phone_conn, PHONE), daemon=True
    )
    thread.start()
    result = perform_handshake(
        wrapped,
        JETSON,
        mono_clock=lambda: wrapped.reads * 1_000_000,
        wall_clock=lambda: 1_700_000_000_000_000_000,
    )
    thread.join(timeout=2.0)

    assert wrapped.reads >= 2  # prefix and header at least
    assert result.clock.t_local_send_mono_ns == 0
    assert result.clock.t_local_recv_mono_ns == wrapped.reads * 1_000_000
    assert result.clock.round_trip_ns > 0


def test_the_round_trip_spans_the_read_not_a_bare_clock_pair():
    phone_conn, jetson_conn = loopback_pair()
    wrapped = ClockAdvancingConnection(jetson_conn)
    thread = threading.Thread(
        target=lambda: perform_handshake(phone_conn, PHONE), daemon=True
    )
    thread.start()
    result = perform_handshake(
        wrapped, JETSON, mono_clock=lambda: wrapped.reads * 7, wall_clock=lambda: 1
    )
    thread.join(timeout=2.0)
    assert result.clock.round_trip_ns == wrapped.reads * 7
