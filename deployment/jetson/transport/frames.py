"""Frame encoding: a binary length prefix, a JSON header, and opaque payload.

    [4B payload_len][2B header_len][JSON header][payload]

Both lengths are big-endian. The header is a JSON object; the payload is bytes
the transport never inspects.

Why a JSON header in front of raw bytes, rather than a struct or base64. The
metadata is the part that will change -- every later task wants to add a field
-- and JSON absorbs additions without either side needing a version bump. The
payload is the part that is large, and it is copied through untouched, so a
JPEG costs its own size and nothing more. Base64 would have cost a third more
on every frame.

Encoding is canonical: sorted keys, no insertion whitespace, UTF-8 without
escaping. Two implementations in two languages must produce identical bytes for
the same frame, and `specs/transport_golden_frames.json` holds them to it.

Every length is validated against a limit before a single byte is read into a
buffer. A corrupted prefix otherwise asks the reader to allocate whatever
number happened to land in those four bytes.

A framing error is terminal for the connection. The stream has no delimiter to
resynchronize on, so a reader that has lost its place cannot find it again;
see specs/transport_protocol.md.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from transport.channels import Channel, UnknownChannelError, parse_channel

PROTOCOL_VERSION = 1

MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 8192

_PREFIX = struct.Struct(">IH")
PREFIX_BYTES = _PREFIX.size  # 6

# Header keys the transport owns. Anything else is a message-level extension
# that the transport carries and ignores.
_REQUIRED_KEYS = ("ch", "seq", "t_mono_ns", "t_wall_ns", "n")

# Reserved header extensions the transport owns on every channel: the
# handshake and its own keepalive. Message-level extensions must not use them.
HELLO_KEY = "hello"
HEARTBEAT_KEY = "heartbeat"
# Stamped by the writer immediately before the bytes leave, on frames that ask
# for it. `t_mono_ns` is an enqueue stamp and so carries however long the frame
# then waited behind others; for a cross-device timebase that wait is the
# dominant error, larger than the network. Both keys ride such a frame and mean
# different things -- redefining `t_mono_ns` to departure time would have
# changed what every existing latency figure measures without changing a test.
WIRE_STAMP_KEY = "t_wire_mono_ns"
# The placeholder the transport substitutes at enqueue: the widest value a real
# stamp can ever be (2**63-1 ns is ~292 years of uptime). The enqueue-time
# encode therefore validates the widest header this frame can possibly produce,
# so the writer's re-encode can only ever be shorter and can never be the thing
# that pushes a header past MAX_HEADER_BYTES. Without this, a header that fit
# with a one-digit placeholder could fail on a sixteen-digit stamp -- in the
# writer thread, long after send() told the caller it was accepted.
WIRE_STAMP_RESERVE = 2**63 - 1
RESERVED_EXTENSIONS = (HELLO_KEY, HEARTBEAT_KEY, WIRE_STAMP_KEY)


class FramingError(ValueError):
    """Malformed frame. Not recoverable: the session must end."""


@dataclass(frozen=True)
class Frame:
    channel: Channel
    seq: int
    t_mono_ns: int
    t_wall_ns: int
    payload: bytes = b""
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def header(self) -> dict[str, Any]:
        header: dict[str, Any] = {
            "ch": self.channel.value,
            "seq": int(self.seq),
            "t_mono_ns": int(self.t_mono_ns),
            "t_wall_ns": int(self.t_wall_ns),
            "n": len(self.payload),
        }
        for key, value in self.extensions.items():
            if key in _REQUIRED_KEYS:
                raise FramingError(f"extension {key!r} collides with a reserved header key")
            header[key] = value
        return header

    @property
    def size_bytes(self) -> int:
        return len(encode(self))


def encode_header(header: Mapping[str, Any]) -> bytes:
    """Canonical JSON bytes. Any implementation must match this exactly.

    NaN and Infinity are refused rather than emitted. Python would write the
    bare tokens NaN and Infinity, which its own parser accepts and a strict
    parser in another language rejects -- a bug that round-trips perfectly on
    one side and desyncs on the first interop attempt.

    Everything json refuses arrives as FramingError, not just the non-finite
    floats: an unserializable value raises TypeError rather than ValueError,
    and a numpy scalar is a likelier thing for a caller to hand us than a NaN
    -- sensors/ produces them -- so both have to be caught or the exception
    escapes the caller's except clause and kills the thread.
    """
    try:
        return json.dumps(
            header,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise FramingError(f"header is not encodable: {exc}") from None


def encode(frame: Frame) -> bytes:
    payload = frame.payload
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise FramingError(f"payload {len(payload)} exceeds {MAX_PAYLOAD_BYTES}")
    header_bytes = encode_header(frame.header())
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise FramingError(f"header {len(header_bytes)} exceeds {MAX_HEADER_BYTES}")
    return _PREFIX.pack(len(payload), len(header_bytes)) + header_bytes + payload


def read_frame(recv_exact: Callable[[int], bytes]) -> Frame:
    """Read one frame using a blocking exact-read callable.

    `recv_exact(n)` must return exactly n bytes or raise. It is never asked for
    a length that has not already been bounds-checked.
    """
    prefix = recv_exact(PREFIX_BYTES)
    if len(prefix) != PREFIX_BYTES:
        raise FramingError(f"short prefix: {len(prefix)} of {PREFIX_BYTES} bytes")
    payload_len, header_len = _PREFIX.unpack(prefix)

    if header_len == 0:
        raise FramingError("header_len is 0")
    if header_len > MAX_HEADER_BYTES:
        raise FramingError(f"header_len {header_len} exceeds {MAX_HEADER_BYTES}")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise FramingError(f"payload_len {payload_len} exceeds {MAX_PAYLOAD_BYTES}")

    header_bytes = recv_exact(header_len)
    if len(header_bytes) != header_len:
        raise FramingError(f"short header: {len(header_bytes)} of {header_len} bytes")
    payload = recv_exact(payload_len) if payload_len else b""
    if len(payload) != payload_len:
        raise FramingError(f"short payload: {len(payload)} of {payload_len} bytes")

    return _frame_from_parts(header_bytes, payload, payload_len)


def decode(data: bytes) -> Frame:
    """Decode exactly one frame from a complete buffer.

    Rejects trailing bytes: callers of this function claim to hold one frame,
    and silently ignoring a second would hide a desync.
    """
    cursor = 0

    def take(n: int) -> bytes:
        nonlocal cursor
        chunk = data[cursor : cursor + n]
        cursor += len(chunk)
        return chunk

    frame = read_frame(take)
    if cursor != len(data):
        raise FramingError(f"{len(data) - cursor} trailing bytes after frame")
    return frame


MAX_INT64 = 2**63 - 1
MIN_INT64 = -(2**63)


def _reject_non_finite_float(text: str) -> float:
    """Refuse a float literal that overflows to infinity.

    `parse_constant` covers the three bare literals; this covers the same class
    arriving as ordinary digits. `1e999` is well-formed JSON and `float()` turns it
    into `inf` without complaint, so it walked past the constant guard and became a
    record the decoder refused one message later -- while Kotlin's parser refuses it
    and ends the session. Same asymmetry the constant guard was written for: one
    side loses a record, the other loses the link.
    """
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{text} is not finite")
    return value


def _reject_out_of_range_int(text: str) -> int:
    """Refuse an integer outside signed 64-bit.

    Python's ints are unbounded and Kotlin's `Long` is not, so `2**63` parsed
    cleanly here and was refused by the peer's parser -- and unlike the float case
    it produced *no refusal at all* on this side: the value stayed intact and rode
    through as a plausible sequence number.
    """
    value = int(text)
    if not MIN_INT64 <= value <= MAX_INT64:
        raise ValueError(f"{text} is outside signed 64-bit")
    return value


def _reject_duplicate_keys(pairs: list) -> dict:
    """Refuse a repeated key rather than letting the last one win.

    `{"a":1,"a":2}` is accepted by every JSON parser that builds a dict, and the
    value that survives is a property of the parser rather than of the message.
    Canonical JSON is the only form in which two headers are equal on this wire, and
    a duplicate key means the sender cannot have produced one -- so this is a
    malformed header, not a message to interpret. Kotlin's parser already refuses
    it.
    """
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _reject_json_constant(name: str) -> float:
    """Refuse `NaN`, `Infinity` and `-Infinity`, which json accepts by default.

    None of the three is JSON: RFC 8259 has no such literals, and the encoder here
    cannot produce them (`allow_nan=False`). So a bare one on the wire means a peer
    that is not speaking the protocol, and the two implementations disagreed about
    what that costs. Kotlin's parser has no branch for `N` or `I`, so it raises at
    the framing layer and the session ends. Python accepted it, put a float `nan`
    in the header, and left the decoder to refuse it as `non_finite` -- one dropped
    message, session intact.

    That is not a difference the refusal table can express: one side loses a record
    and the other loses the link. Refusing here makes both a framing error, which
    is the stricter of the two readings and the one that matches the spec.
    """
    raise ValueError(f"{name} is not JSON")


def _frame_from_parts(header_bytes: bytes, payload: bytes, payload_len: int) -> Frame:
    try:
        text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FramingError(f"header is not UTF-8: {exc}") from None
    try:
        header = json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_reject_non_finite_float,
            parse_int=_reject_out_of_range_int,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ValueError as exc:
        raise FramingError(f"header is not JSON: {exc}") from None
    if not isinstance(header, dict):
        raise FramingError(f"header is {type(header).__name__}, expected object")

    missing = [key for key in _REQUIRED_KEYS if key not in header]
    if missing:
        raise FramingError(f"header missing {', '.join(missing)}")

    try:
        channel = parse_channel(header["ch"])
    except UnknownChannelError as exc:
        raise FramingError(str(exc)) from None

    declared = header["n"]
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise FramingError(f"header 'n' is {type(declared).__name__}, expected int")
    if declared != payload_len:
        raise FramingError(f"header n={declared} disagrees with payload_len={payload_len}")

    ints = {}
    for key in ("seq", "t_mono_ns", "t_wall_ns"):
        value = header[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise FramingError(f"header {key!r} is {type(value).__name__}, expected int")
        ints[key] = value

    extensions = {k: v for k, v in header.items() if k not in _REQUIRED_KEYS}
    return Frame(
        channel=channel,
        seq=ints["seq"],
        t_mono_ns=ints["t_mono_ns"],
        t_wall_ns=ints["t_wall_ns"],
        payload=payload,
        extensions=extensions,
    )
