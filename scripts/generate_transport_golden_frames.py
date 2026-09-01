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
from transport.messages import (  # noqa: E402
    AdvisoryMessage,
    CameraFrame,
    HereQuery,
    GpsRecord,
    HereResponse,
    ImuSample,
    PhoneTelemetry,
    RateCommand,
    TimeSyncMessage,
)
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


# One case per typed message, so Kotlin is held to the same field names and
# types as well as the same framing. Every one uses fixed values -- no clocks,
# no randomness -- so the bytes are reproducible.
MESSAGES = [
    (
        "message_time_sync_ping",
        "a time-sync ping: the two peer-receipt fields null, and the wire stamp "
        "still the placeholder the sender's transport will overwrite",
        TimeSyncMessage(t_capture_mono_ns=1_000_000_008, exchange_id=17),
    ),
    (
        "message_time_sync_pong",
        "a time-sync pong: peer receipt in both clocks, and a wire stamp as the "
        "writer leaves it -- a wall value past 2**53, which is the case an "
        "implementation routing it through a double gets wrong",
        TimeSyncMessage(
            t_capture_mono_ns=1_000_000_009, exchange_id=17,
            t_wire_mono_ns=1_000_000_100,
            t_peer_recv_mono_ns=2_000_000_050,
            t_peer_recv_wall_ns=1_755_648_000_123_456_789,
            t_peer_wire_mono_ns=1_000_000_020,
        ),
    ),
    (
        "message_camera",
        "a camera frame: metadata in the header, JPEG untouched in the payload",
        CameraFrame(
            t_capture_mono_ns=1_000_000_000, frame_id=1841, width=1280, height=720,
            format="jpeg", quality=85, jpeg=pattern_payload(4096),
        ),
    ),
    (
        "message_gps_full",
        "a GPS fix with every field available",
        GpsRecord(
            t_capture_mono_ns=1_000_000_001, valid=True, fix_quality=1, num_sats=9,
            lat=51.5074, lon=-0.1278, speed_mps=13.4, heading_deg=91.2, hdop=0.9,
            altitude_m=35.0, utc_epoch_ns=1_755_648_000_000_000_000,
        ),
    ),
    (
        "message_gps_all_null",
        "the null convention: every optional field unavailable, none absent",
        GpsRecord(
            t_capture_mono_ns=1_000_000_002, valid=False, fix_quality=0, num_sats=0,
        ),
    ),
    (
        "message_imu",
        "an IMU sample, the highest-rate message",
        ImuSample(
            t_capture_mono_ns=1_000_000_003, ax=0.1, ay=-0.2, az=9.79,
            gx=0.01, gy=0.0, gz=-0.02, accuracy=3,
        ),
    ),
    (
        "message_here",
        "a HERE response: our metadata in the header, their body opaque",
        HereResponse(
            t_capture_mono_ns=1_000_000_004,
            request_url="https://data.traffic.hereapi.com/v7/flow?in=circle:51.5,-0.12;r=1500",
            status=200, content_type="application/json", query_lat=51.5,
            query_lon=-0.1278, query_radius_m=1500.0,
            t_request_mono_ns=999_000_000, t_response_mono_ns=1_000_000_004,
            body=pattern_payload(512),
        ),
    ),
    (
        "message_advisory",
        "the advisory as the driver sees it, plus the machine-readable action",
        AdvisoryMessage(
            t_capture_mono_ns=1_000_000_005, rec_speed_mps=11.176,
            rec_speed_display=25.0, current_speed_display=27.5, units="mph",
            headway_target_s=1.6, lane_text="Keep lane", merge_text="Normal driving",
            traffic_text="Moderate", confidence=0.87, confidence_label="high",
            action={
                "desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal",
            },
        ),
    ),
    (
        "message_rate_cmd",
        "a rate command, with the trigger that produced it and the shadow flag",
        RateCommand(
            t_capture_mono_ns=1_000_000_006,
            rates={"camera_hz": 5.0, "gps_hz": 5.0, "imu_hz": 50.0, "here_hz": 0.5},
            trigger="advisory_bin_boundary", shadow=True,
        ),
    ),
    (
        "message_rate_cmd_with_here",
        "a rate command that also sets the HERE query shape, which is optional",
        RateCommand(
            t_capture_mono_ns=1_000_000_006,
            rates={"camera_hz": 5.0, "gps_hz": 5.0, "imu_hz": 50.0, "here_hz": 0.5},
            trigger="advisory_bin_boundary", shadow=True,
            here=HereQuery(
                in_="corridor:40.7128,-74.0060;40.7580,-73.9855;r=200",
                location_ref="shape",
                lat=40.7128, lon=-74.0060, radius_m=9000.0,
            ),
        ),
    ),
    (
        "message_telemetry",
        "phone telemetry: thermal is a constraint, energy is deliberately not",
        PhoneTelemetry(
            t_capture_mono_ns=1_000_000_007, thermal_status="nominal",
            thermal_headroom=0.42,
            achieved={"camera_hz": 9.9, "gps_hz": 5.0, "imu_hz": 49.8, "here_hz": 0.5},
            dropped={"camera": 3, "gps": 0, "imu": 0, "here": 1},
            here_calls=30, here_errors=1,
        ),
    ),
    # A second telemetry case rather than an edit to the one above, because that one
    # is frozen and a frozen vector that moves is not a vector. `skin_temp_c` and
    # `skin_temp_zone` postdate it, so until now the two codecs had never been frozen
    # against each other on either field -- and they are the pair the sensing
    # controller backs off on before the status moves.
    (
        "message_telemetry_with_skin",
        "phone telemetry carrying an absolute temperature and the zone it came from",
        PhoneTelemetry(
            t_capture_mono_ns=1_000_000_009, thermal_status="moderate",
            thermal_headroom=0.11,
            achieved={"camera_hz": 5.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.05},
            dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0},
            here_calls=7, here_errors=0,
            # Null together or set together: the number cannot be interpreted, or
            # compared across devices, without the zone that produced it.
            skin_temp_c=41.5, skin_temp_zone="xo_therm",
        ),
    ),
    # Two more absent-tolerant additions, appended rather than edited into their
    # nearest existing case for the same reason as message_telemetry_with_skin:
    # a frozen vector that moves is not a vector.
    (
        "message_camera_with_encode_stamps",
        "a camera frame carrying the phone's own encode timing (task 33) -- "
        "absent-tolerant, so message_camera above stays byte-identical without it",
        CameraFrame(
            t_capture_mono_ns=1_000_000_010, frame_id=1842, width=1280, height=720,
            format="jpeg", quality=85, jpeg=pattern_payload(4096),
            t_encode_start_mono_ns=1_000_000_020, t_encode_done_mono_ns=1_000_000_035,
        ),
    ),
    (
        "message_time_sync_ping_with_prev",
        "a ping carrying the previous exchange (task 33): the trio a responder "
        "needs to reconstruct a round-trip sample with no pending state of its own",
        TimeSyncMessage(
            t_capture_mono_ns=1_000_000_011, exchange_id=18,
            prev_exchange_id=17, t_prev_pong_wire_mono_ns=1_000_000_100,
            t_prev_pong_recv_mono_ns=1_000_000_130,
        ),
    ),
]


def message_cases() -> list[dict]:
    """Frames built from typed messages, appended to the frame-level cases."""
    built = []
    for name, why, message in MESSAGES:
        extensions, payload = message.to_wire()
        built.append(
            {
                "name": name,
                "why": why,
                "channel": message.CHANNEL,
                "seq": 1,
                "t_mono_ns": 1_100_000_000,
                "t_wall_ns": 1_755_648_000_000_000_000,
                "length": len(payload),
                "extensions": extensions,
                "payload_bytes": payload,
            }
        )
    return built


def build() -> dict:
    cases = []
    for spec in CASES + message_cases():
        payload = spec.get("payload_bytes")
        if payload is None:
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
            "Frozen cross-language encodings, at two layers: frame cases pin the "
            "framing, message cases pin the typed field names and types. Every "
            "implementation must encode each case to exactly these bytes and decode "
            "them back to these fields. Changing a recorded byte is a protocol change "
            "and needs a version bump; ADDING a case is not, since it constrains "
            "nothing already agreed -- but because regeneration rewrites the whole "
            "file, a test checks every pre-existing case is byte-identical."
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
