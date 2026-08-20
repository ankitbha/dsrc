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
