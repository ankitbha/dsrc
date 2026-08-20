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
RESERVED_EXTENSIONS = (HELLO_KEY, HEARTBEAT_KEY)


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
    """Canonical JSON bytes. Any implementation must match this exactly."""
    return json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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


def _frame_from_parts(header_bytes: bytes, payload: bytes, payload_len: int) -> Frame:
    try:
        text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FramingError(f"header is not UTF-8: {exc}") from None
    try:
        header = json.loads(text)
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
