#!/usr/bin/env python3
"""Generate specs/transport_golden_frames.json.

Run once. The file it writes is a frozen protocol artifact: the tests read it
and assert against it, and they never regenerate it. Changing a recorded byte
is a protocol change and needs a PROTOCOL_VERSION bump, because a peer already
in the field agrees with the old bytes.

Payloads are described by a generator rather than stored, so the 4 MiB case
costs nothing to commit. Whole-frame bytes are pinned by SHA-256 for the same
reason; the header bytes, which are where two implementations actually diverge,
are stored literally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from transport.channels import Channel  # noqa: E402
from transport.frames import (  # noqa: E402
    HEARTBEAT_KEY,
    HELLO_KEY,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    Frame,
    encode,
    encode_header,
)

PATTERN_DESCRIPTION = "payload[i] = (i * 37 + 11) % 256"


def pattern_payload(length: int) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(length))


CASES = [
    {
        "name": "empty_payload",
        "why": "the degenerate frame: payload_len 0, and no payload read at all",
        "channel": Channel.CONTROL,
        "seq": 0,
        "t_mono_ns": 1,
        "t_wall_ns": 2,
        "length": 0,
        "extensions": {},
    },
    {
        "name": "typical_jpeg",
        "why": "the common case: a camera frame at a realistic size",
        "channel": Channel.CAMERA,
        "seq": 1841,
        "t_mono_ns": 987_654_321_012,
        "t_wall_ns": 1_755_648_000_123_456_789,
        "length": 40_960,
        "extensions": {"w": 1280, "h": 720, "fmt": "jpeg"},
    },
    {
        "name": "max_payload",
        "why": "payload_len at the limit; one more byte must be refused",
        "channel": Channel.CAMERA,
        "seq": 2,
        "t_mono_ns": 3,
        "t_wall_ns": 4,
        "length": MAX_PAYLOAD_BYTES,
        "extensions": {},
    },
    {
        "name": "non_ascii_extension",
        "why": "UTF-8 in a header string, unescaped; catches ensure_ascii and "
               "any implementation that escapes or transcodes",
        "channel": Channel.TELEMETRY,
        "seq": 7,
        "t_mono_ns": 11,
        "t_wall_ns": 13,
        "length": 3,
        "extensions": {"note": "señal 温度 ±2°C \U0001f321", "unit": "°C"},
    },
    {
        "name": "large_ints",
        "why": "2**53 and one past it, plus a realistic wall clock. Any "
               "implementation that routes these through a double loses digits",
        "channel": Channel.GPS,
        "seq": 9_007_199_254_740_993,
        "t_mono_ns": 9_007_199_254_740_992,
        "t_wall_ns": 1_755_648_000_987_654_321,
        "length": 16,
        "extensions": {"big": 9_223_372_036_854_775_807, "neg": -9_007_199_254_740_993},
    },
    {
        "name": "nested_unsorted_extension",
        "why": "key ordering is canonical and recursive: nested objects sort too",
        "channel": Channel.HERE,
        "seq": 5,
        "t_mono_ns": 17,
        "t_wall_ns": 19,
        "length": 8,
        "extensions": {
            "zulu": 1,
            "alpha": {"z": [3, 2, 1], "a": {"n": None, "b": True, "a": 1.5}},
            "mike": "m",
        },
    },
    {
        "name": "hello",
        "why": "the frame that opens every connection",
        "channel": Channel.CONTROL,
        "seq": 0,
        "t_mono_ns": 123_456_789,
        "t_wall_ns": 1_755_648_000_000_000_000,
        "length": 0,
        "extensions": {
            HELLO_KEY: {
                "protocol_version": PROTOCOL_VERSION,
                "device_id": "moto-g-power",
                "role": "phone",
            }
        },
    },
    {
        "name": "heartbeat",
        "why": "the keepalive the transport consumes rather than delivering",
        "channel": Channel.CONTROL,
        "seq": 42,
        "t_mono_ns": 1_000_000_000,
        "t_wall_ns": 1_755_648_001_000_000_000,
        "length": 0,
        "extensions": {HEARTBEAT_KEY: True},
    },
]


def build() -> dict:
    cases = []
    for spec in CASES:
        payload = pattern_payload(spec["length"])
        frame = Frame(
            channel=spec["channel"],
            seq=spec["seq"],
            t_mono_ns=spec["t_mono_ns"],
            t_wall_ns=spec["t_wall_ns"],
            payload=payload,
            extensions=spec["extensions"],
        )
        encoded = encode(frame)
        header_bytes = encode_header(frame.header())
        cases.append(
            {
                "name": spec["name"],
                "why": spec["why"],
                "frame": {
                    "ch": spec["channel"].value,
                    "seq": spec["seq"],
                    "t_mono_ns": spec["t_mono_ns"],
                    "t_wall_ns": spec["t_wall_ns"],
                    "extensions": spec["extensions"],
                },
                "payload": {"kind": "pattern", "length": spec["length"]},
                "prefix_hex": encoded[:6].hex(),
                "header_hex": header_bytes.hex(),
                "header_len": len(header_bytes),
                "frame_len": len(encoded),
                "frame_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frozen": True,
        "note": (
            "Frozen cross-language encodings. Every implementation must encode each "
            "case to exactly these bytes and decode them back to these fields. "
            "Changing a byte here is a protocol change and needs a version bump."
        ),
        "payload_generator": PATTERN_DESCRIPTION,
        "byte_order": "big-endian",
        "header_encoding": (
            "UTF-8 JSON object, keys sorted recursively, separators ',' and ':', "
            "no non-ASCII escaping, NaN and Infinity rejected"
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "specs" / "transport_golden_frames.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing file; without this an existing file is left alone",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} exists and is frozen; pass --force only for a protocol change")
        return 1
    args.out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
