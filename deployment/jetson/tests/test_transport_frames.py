"""Unit tests for the frame codec: exact bytes in, exact bytes out, and every
documented malformed input refused."""

from __future__ import annotations

import json
import struct

import pytest

from transport.channels import Channel
from transport.frames import (
    HEARTBEAT_KEY,
    MAX_HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    PREFIX_BYTES,
    Frame,
    FramingError,
    decode,
    encode,
    encode_header,
    read_frame,
)


def make_frame(**overrides) -> Frame:
    fields = {
        "channel": Channel.CAMERA,
        "seq": 3,
        "t_mono_ns": 111,
        "t_wall_ns": 222,
        "payload": b"abc",
    }
    fields.update(overrides)
    return Frame(**fields)


def raw_frame(header: object, payload: bytes, *, header_len=None, payload_len=None) -> bytes:
    if isinstance(header, (bytes, bytearray)):
        header_bytes = bytes(header)
    else:
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack(
        ">IH",
        len(payload) if payload_len is None else payload_len,
        len(header_bytes) if header_len is None else header_len,
    )
    return prefix + header_bytes + payload


def reader_for(data: bytes):
    cursor = {"at": 0}

    def recv_exact(n: int) -> bytes:
        chunk = data[cursor["at"] : cursor["at"] + n]
        cursor["at"] += len(chunk)
        return chunk

    return recv_exact


# -- roundtrip ---------------------------------------------------------------


@pytest.mark.parametrize("payload", [b"", b"x", b"abc", bytes(range(256)), b"\x00" * 5000])
def test_roundtrip_preserves_every_field(payload):
    frame = make_frame(payload=payload, extensions={"w": 1280, "note": "señal"})
    decoded = decode(encode(frame))
    assert decoded.channel is frame.channel
    assert decoded.seq == frame.seq
    assert decoded.t_mono_ns == frame.t_mono_ns
    assert decoded.t_wall_ns == frame.t_wall_ns
    assert decoded.payload == payload
    assert decoded.extensions == {"w": 1280, "note": "señal"}


def test_roundtrip_at_the_payload_limit():
    frame = make_frame(payload=b"\xa5" * MAX_PAYLOAD_BYTES)
    decoded = decode(encode(frame))
    assert len(decoded.payload) == MAX_PAYLOAD_BYTES


@pytest.mark.parametrize("channel", list(Channel))
def test_every_channel_roundtrips(channel):
    assert decode(encode(make_frame(channel=channel))).channel is channel


def test_large_integers_survive_exactly():
    frame = make_frame(
        seq=2**53 + 1,
        t_mono_ns=2**53,
        t_wall_ns=1_755_648_000_987_654_321,
        extensions={"big": 2**63 - 1},
    )
    decoded = decode(encode(frame))
    assert decoded.seq == 2**53 + 1
    assert decoded.t_mono_ns == 2**53
    assert decoded.t_wall_ns == 1_755_648_000_987_654_321
    assert decoded.extensions["big"] == 2**63 - 1


# -- exact wire bytes --------------------------------------------------------


def test_prefix_is_big_endian_payload_then_header():
    encoded = encode(make_frame(payload=b"\x00" * 258))
    payload_len, header_len = struct.unpack(">IH", encoded[:PREFIX_BYTES])
    assert payload_len == 258
    assert encoded[:4] == b"\x00\x00\x01\x02"  # 258 big-endian, not little
    assert header_len == len(encoded) - PREFIX_BYTES - 258


def test_header_json_is_canonical():
    header = encode_header(
        make_frame(extensions={"zz": 1, "aa": {"y": 2, "x": 1}}).header()
    ).decode()
    assert header.startswith('{"aa":{"x":1,"y":2},"ch":')  # sorted, recursively
    assert " " not in header  # no separator whitespace
    assert json.loads(header)["aa"] == {"x": 1, "y": 2}


def test_non_ascii_is_raw_utf8_not_escaped():
    encoded = encode(make_frame(extensions={"note": "señal"}))
    assert "señal".encode("utf-8") in encoded
    assert b"\\u" not in encoded


def test_declared_n_matches_the_prefix():
    encoded = encode(make_frame(payload=b"abcd"))
    payload_len, header_len = struct.unpack(">IH", encoded[:PREFIX_BYTES])
    header = json.loads(encoded[PREFIX_BYTES : PREFIX_BYTES + header_len])
    assert header["n"] == payload_len == 4


# -- refusals ----------------------------------------------------------------


def test_encode_refuses_oversize_payload():
    with pytest.raises(FramingError, match="exceeds"):
        encode(make_frame(payload=b"\x00" * (MAX_PAYLOAD_BYTES + 1)))


def test_encode_refuses_oversize_header():
    with pytest.raises(FramingError, match="header"):
        encode(make_frame(extensions={"pad": "x" * (MAX_HEADER_BYTES + 10)}))


@pytest.mark.parametrize("key", ["ch", "seq", "t_mono_ns", "t_wall_ns", "n"])
def test_extension_may_not_shadow_a_reserved_key(key):
    with pytest.raises(FramingError, match="collides"):
        encode(make_frame(extensions={key: "hijack"}))


def test_zero_length_header_refused():
    with pytest.raises(FramingError, match="header_len is 0"):
        decode(struct.pack(">IH", 0, 0))


def test_oversize_declared_header_refused():
    data = struct.pack(">IH", 0, MAX_HEADER_BYTES + 1)
    with pytest.raises(FramingError, match="header_len"):
        read_frame(reader_for(data))


def test_oversize_declared_payload_refused():
    data = struct.pack(">IH", MAX_PAYLOAD_BYTES + 1, 2) + b"{}"
    with pytest.raises(FramingError, match="payload_len"):
        read_frame(reader_for(data))


def test_never_asks_for_an_unvalidated_length():
    """A corrupt prefix must not turn into an allocation request."""
    requested: list[int] = []

    def recv_exact(n: int) -> bytes:
        requested.append(n)
        if len(requested) == 1:
            return struct.pack(">IH", 0xFFFFFFFF, 0xFFFF)
        return b"\x00" * n

    with pytest.raises(FramingError):
        read_frame(recv_exact)
    assert requested == [PREFIX_BYTES]
    assert max(requested) <= max(MAX_PAYLOAD_BYTES, MAX_HEADER_BYTES)


def test_header_not_utf8_refused():
    with pytest.raises(FramingError, match="UTF-8"):
        decode(raw_frame(b"\xff\xfe{}", b""))


def test_header_not_json_refused():
    with pytest.raises(FramingError, match="not JSON"):
        decode(raw_frame(b"{not json", b""))


@pytest.mark.parametrize("header", [[1, 2], "text", 42, None, True])
def test_header_must_be_an_object(header):
    with pytest.raises(FramingError, match="expected object"):
        decode(raw_frame(header, b""))


@pytest.mark.parametrize("missing", ["ch", "seq", "t_mono_ns", "t_wall_ns", "n"])
def test_missing_required_key_refused(missing):
    header = {"ch": "camera", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 0}
    del header[missing]
    with pytest.raises(FramingError, match=missing):
        decode(raw_frame(header, b""))


def test_declared_n_disagreeing_with_prefix_refused():
    header = {"ch": "camera", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 99}
    with pytest.raises(FramingError, match="disagrees"):
        decode(raw_frame(header, b"abc"))


def test_unknown_channel_refused():
    header = {"ch": "lidar", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 0}
    with pytest.raises(FramingError, match="lidar"):
        decode(raw_frame(header, b""))


@pytest.mark.parametrize("key", ["seq", "t_mono_ns", "t_wall_ns", "n"])
@pytest.mark.parametrize("value", ["1", 1.5, None, True, [1]])
def test_non_integer_header_numbers_refused(key, value):
    header = {"ch": "camera", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 0}
    header[key] = value
    with pytest.raises(FramingError, match="expected int"):
        decode(raw_frame(header, b""))


def test_short_prefix_refused():
    with pytest.raises(FramingError, match="short prefix"):
        read_frame(reader_for(b"\x00\x00"))


def test_short_header_refused():
    data = raw_frame({"ch": "camera", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 0}, b"")
    with pytest.raises(FramingError, match="short header"):
        read_frame(reader_for(data[:-4]))


def test_short_payload_refused():
    data = raw_frame({"ch": "camera", "seq": 1, "t_mono_ns": 2, "t_wall_ns": 3, "n": 8}, b"12345678")
    with pytest.raises(FramingError, match="short payload"):
        read_frame(reader_for(data[:-3]))


def test_trailing_bytes_refused_by_decode():
    with pytest.raises(FramingError, match="trailing"):
        decode(encode(make_frame()) + b"extra")


def test_two_frames_read_in_sequence_from_one_stream():
    first = make_frame(seq=1, payload=b"one")
    second = make_frame(seq=2, payload=b"two", channel=Channel.GPS)
    recv = reader_for(encode(first) + encode(second))
    assert read_frame(recv).payload == b"one"
    got = read_frame(recv)
    assert got.payload == b"two" and got.channel is Channel.GPS


def test_heartbeat_extension_survives_roundtrip():
    decoded = decode(encode(make_frame(channel=Channel.CONTROL, extensions={HEARTBEAT_KEY: True})))
    assert decoded.extensions[HEARTBEAT_KEY] is True


# -- non-finite numbers ------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused_as_framing_errors(value):
    """Python writes the bare tokens NaN and Infinity, which its own parser
    accepts and a strict parser elsewhere rejects: a bug that round-trips
    perfectly here and desyncs on the first interop attempt. A sensor reading
    that is unavailable is a plausible source of one, so the refusal has to be
    a FramingError like any other bad frame, not a bare ValueError escaping
    past the caller's except clause."""
    with pytest.raises(FramingError, match="not encodable"):
        encode(make_frame(extensions={"reading": value}))


def test_nested_non_finite_numbers_are_refused_too():
    with pytest.raises(FramingError, match="not encodable"):
        encode(make_frame(extensions={"imu": {"accel": [1.0, float("nan")]}}))


@pytest.mark.parametrize(
    "value",
    [b"raw bytes", {1, 2}, 1j, object()],
    ids=["bytes", "set", "complex", "object"],
)
def test_unserializable_values_are_refused_as_framing_errors(value):
    """These raise TypeError, not ValueError. A numpy scalar is in this class
    and is what sensors/ produces -- likelier by far than the NaN that
    motivated the guard -- so catching only ValueError let it escape the
    caller's except clause and kill the thread."""
    with pytest.raises(FramingError, match="not encodable"):
        encode(make_frame(extensions={"reading": value}))


def test_a_non_string_header_key_is_refused_as_a_framing_error():
    with pytest.raises(FramingError, match="not encodable"):
        encode(make_frame(extensions={7: "numeric key"}))


def test_finite_floats_still_encode():
    decoded = decode(encode(make_frame(extensions={"reading": 1.5, "zero": 0.0})))
    assert decoded.extensions == {"reading": 1.5, "zero": 0.0}


# -- header size boundary ----------------------------------------------------


def pad_to_header_size(target: int) -> Frame:
    """A frame whose encoded header is exactly `target` bytes."""
    probe = make_frame(payload=b"", extensions={"pad": ""})
    overhead = len(encode_header(probe.header()))
    return make_frame(payload=b"", extensions={"pad": "x" * (target - overhead)})


def test_header_at_exactly_the_limit_encodes():
    """The payload limit has both an at-limit and an over-limit test. Without
    the same pair here, an off-by-one lets us emit a header our own reader
    would refuse, so we would kill the session with our own frame."""
    frame = pad_to_header_size(MAX_HEADER_BYTES)
    encoded = encode(frame)
    assert len(encode_header(frame.header())) == MAX_HEADER_BYTES
    assert decode(encoded).extensions == frame.extensions


def test_header_one_byte_over_the_limit_is_refused():
    with pytest.raises(FramingError, match="header"):
        encode(pad_to_header_size(MAX_HEADER_BYTES + 1))


def test_a_header_at_the_limit_survives_a_round_trip_through_a_reader():
    frame = pad_to_header_size(MAX_HEADER_BYTES)
    assert read_frame(reader_for(encode(frame))).extensions == frame.extensions

@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_bare_non_json_literal_in_the_header_is_a_framing_error(literal):
    """`json` accepts all three by default and none of them is JSON.

    This was an unreconcilable divergence rather than a defect on either side, and
    the two costs were not comparable. Kotlin's parser has no branch for `N` or
    `I`, so it raises at the framing layer and the session ends. Python accepted
    the literal, put a float `nan` in the header, and left the decoder to refuse it
    as `non_finite` -- one dropped message, session intact. One side loses a record
    and the other loses the link, which is not something the refusal table can
    express.

    Refusing here makes both a framing error, which is the stricter reading and the
    one RFC 8259 supports. The encoder cannot produce any of the three
    (`allow_nan=False`), so a bare one means a peer that is not speaking the
    protocol.
    """
    header = (
        b'{"ch":"gps","seq":1,"n":0,"t_mono_ns":1,"t_wall_ns":2,"speed_mps":'
        + literal.encode()
        + b"}"
    )
    data = struct.pack(">IH", 0, len(header)) + header
    with pytest.raises(FramingError, match="not JSON"):
        read_frame(reader_for(data))


def test_a_quoted_nan_is_still_just_a_string():
    """The guard refuses the three *literals*, not the characters.

    A field whose value happens to be the text "NaN" is an ordinary string and must
    survive -- refusing it would be a parser that reads inside quotes.
    """
    header = b'{"ch":"gps","seq":1,"n":0,"t_mono_ns":1,"t_wall_ns":2,"note":"NaN"}'
    data = struct.pack(">IH", 0, len(header)) + header
    assert read_frame(reader_for(data)).extensions["note"] == "NaN"
