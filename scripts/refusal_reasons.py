"""Print the reason Python gives for each refusal input, as one JSON object.

Exists because a Kotlin test asserting "this matches Python" cannot fail when it stops
matching. Round 3 shipped exactly that: the docstring said the expectations were recorded
by running this package, and one of them -- a null `action` -- had never matched, because
the fix landed in `_nested_object` and `AdvisoryMessage.from_wire` has its own inline check.

`InteropTest` already spawns Python, so the comparison can be executed instead of asserted
by hand. Keys here must match the Kotlin side's case names exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment" / "jetson"))

from transport.messages import (  # noqa: E402
    AdvisoryMessage,
    CameraFrame,
    GpsRecord,
    MessageError,
    RateCommand,
    TimeSyncMessage,
)

GPS = {
    "t_capture_mono_ns": 1, "valid": False, "lat": None, "lon": None, "speed_mps": None,
    "heading_deg": None, "fix_quality": 0, "num_sats": 0, "hdop": None,
    "altitude_m": None, "utc_epoch_ns": None,
}
CAMERA = {
    "t_capture_mono_ns": 1, "frame_id": 1, "width": 1280, "height": 720,
    "format": "jpeg", "quality": 85,
}
ACTION = {
    "desired_speed_bin": "nominal", "desired_headway_bin": "normal",
    "lane_preference": "keep", "merge_mode": "normal",
}
ADVISORY = {
    "t_capture_mono_ns": 1, "rec_speed_mps": 13.4, "rec_speed_display": 30.0,
    "current_speed_display": 28.0, "units": "mph", "headway_target_s": 2.0,
    "lane_text": "keep", "merge_text": "normal", "traffic_text": "clear",
    "confidence": 0.87, "confidence_label": "high", "action": ACTION,
}
RATES = {"camera_hz": 5.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.2}
RATE_CMD = {"t_capture_mono_ns": 1, "rates": RATES, "trigger": "thermal", "shadow": False}
PING = {
    "t_capture_mono_ns": 1, "exchange_id": 1, "t_wire_mono_ns": 0,
    "t_peer_recv_mono_ns": None, "t_peer_recv_wall_ns": None, "t_peer_wire_mono_ns": None,
}

CASES = {
    # non_finite was the one reason in the table's reach that no case exercised, and the
    # floor did not notice because "ACCEPTED" was being counted as a reason. It has to be
    # built here rather than parsed from a header: a bare NaN on the wire is now a framing
    # error on both sides, so a decoder only ever sees one from an in-process caller.
    "gps speed is not finite": (GpsRecord, {**GPS, "speed_mps": float("nan")}, b""),
    "gps altitude is infinite": (GpsRecord, {**GPS, "altitude_m": float("inf")}, b""),
    # The one entry of the shape vocabulary nothing exercised. Extensions are additive,
    # so an unknown key is preserved rather than refused -- which makes this an
    # acceptance on both sides, and the point is that the *shape* agrees.
    "gps unknown key holding a list is accepted by both": (
        GpsRecord, {**GPS, "future_field": [1, "two", None]}, b"",
    ),
    "gps count is null": (GpsRecord, {**GPS, "fix_quality": None}, b""),
    # require_int's own null path, which the case above does not reach: fix_quality goes
    # through check_count. A capture stamp is require_int on every message.
    "gps capture stamp is null": (GpsRecord, {**GPS, "t_capture_mono_ns": None}, b""),
    "camera frame id is null": (CameraFrame, {**CAMERA, "frame_id": None}, b""),
    # require_str's null path, which had no case and no fix.
    "camera format is null": (CameraFrame, {**CAMERA, "format": None}, b""),
    "advisory units is null": (AdvisoryMessage, {**ADVISORY, "units": None}, b""),
    "rate_cmd trigger is null": (RateCommand, {**RATE_CMD, "trigger": None}, b""),
    "gps required bool is null": (GpsRecord, {**GPS, "valid": None}, b""),
    "gps valid fix with null coordinates": (GpsRecord, {**GPS, "valid": True}, b""),
    "gps negative count": (GpsRecord, {**GPS, "num_sats": -1}, b""),
    "gps fractional count": (GpsRecord, {**GPS, "num_sats": 1.5}, b""),
    "camera quality zero is accepted by both": (CameraFrame, {**CAMERA, "quality": 0}, b""),
    "camera quality over one hundred is accepted by both": (CameraFrame, {**CAMERA, "quality": 101}, b""),
    "camera zero width is accepted by both": (CameraFrame, {**CAMERA, "width": 0}, b""),
    "camera negative frame id is accepted by both": (CameraFrame, {**CAMERA, "frame_id": -1}, b""),
    "camera empty format is accepted by both": (CameraFrame, {**CAMERA, "format": ""}, b""),
    "advisory action is null": (AdvisoryMessage, {**ADVISORY, "action": None}, b""),
    "advisory action is not an object": (AdvisoryMessage, {**ADVISORY, "action": 5}, b""),
    "advisory action head is an integer": (
        AdvisoryMessage, {**ADVISORY, "action": {**ACTION, "desired_speed_bin": 5}}, b"",
    ),
    "advisory action head outside the set": (
        AdvisoryMessage, {**ADVISORY, "action": {**ACTION, "merge_mode": "ram_it"}}, b"",
    ),
    "advisory action missing a head": (
        AdvisoryMessage,
        {**ADVISORY, "action": {k: v for k, v in ACTION.items() if k != "merge_mode"}},
        b"",
    ),
    "advisory units outside the three": (AdvisoryMessage, {**ADVISORY, "units": "furlongs"}, b""),
    "rate_cmd rates is null": (RateCommand, {**RATE_CMD, "rates": None}, b""),
    "rate_cmd zero rate": (RateCommand, {**RATE_CMD, "rates": {**RATES, "gps_hz": 0.0}}, b""),
    "rate_cmd rate above the ceiling": (
        RateCommand, {**RATE_CMD, "rates": {**RATES, "gps_hz": 1000.001}}, b"",
    ),
    "rate_cmd missing a rate key": (
        RateCommand,
        {**RATE_CMD, "rates": {k: v for k, v in RATES.items() if k != "imu_hz"}},
        b"",
    ),
    "control partial pong": (
        TimeSyncMessage, {**PING, "t_peer_recv_mono_ns": 2}, b"",
    ),
    "control absent peer field": (
        TimeSyncMessage, {k: v for k, v in PING.items() if k != "t_peer_wire_mono_ns"}, b"",
    ),
    "control negative exchange id": (TimeSyncMessage, {**PING, "exchange_id": -5}, b""),
    "control negative wire stamp": (TimeSyncMessage, {**PING, "t_wire_mono_ns": -7}, b""),
    "control payload on a channel that carries none": (TimeSyncMessage, PING, b"x"),

    # -- two faults at once ------------------------------------------------------
    #
    # Every row above injects exactly one fault, so the ORDER in which each decoder
    # applies its rules has never been compared. It differs: this side checks
    # utc_epoch_ns before the constructor and the counts ahead of lat/speed, while
    # Kotlin checks the capture stamp first and speed/heading ahead of the counts. One
    # systematic mapping bug in a phone build produces multi-fault records, and then
    # inboundRefusals and errors_by_reason name different causes for the same frames --
    # which is what per-reason counting exists to prevent.
    #
    # Recorded rather than papered over: DifferentialTest allows exactly these names to
    # disagree and fails on any other, so a NEW divergence is caught and fixing one of
    # these forces the list to be updated. See specs/transport_protocol.md for the
    # precedence this should settle on.
    "two faults: gps negative count and wrong-typed speed": (
        GpsRecord, {**GPS, "num_sats": -1, "speed_mps": "x"}, b"",
    ),
    "two faults: gps null capture stamp and wrong-typed utc": (
        GpsRecord, {**GPS, "t_capture_mono_ns": None, "utc_epoch_ns": "x"}, b"",
    ),
    "two faults: camera missing height and null frame id": (
        CameraFrame, {k: v for k, v in CAMERA.items() if k != "height"} | {"frame_id": None}, b"",
    ),
    "two faults: control negative exchange id and partial pong": (
        TimeSyncMessage, {**PING, "exchange_id": -5, "t_peer_recv_mono_ns": 5}, b"",
    ),
}


def shape(extensions, payload) -> str:
    """`key:type` for every key, sorted, plus the payload length.

    The Kotlin side computes the same string. Comparing case *names* proves only
    that both tables have a row called the same thing; this catches a case that has
    quietly stopped exercising the field it is named for.

    Structural rather than a digest of the values, deliberately. A value digest
    would have to agree with the other side's float formatting exactly, and it
    cannot be computed at all for the two non_finite cases -- canonical JSON
    refuses to encode a NaN on both sides, which is the property those cases exist
    to check.

    Recursive, and round 7 is why. Emitting a bare "obj" hid the contents of
    `action` and `rates`, so six of the thirty-two rows were still open to exactly
    the drift this exists to catch: moving `"ram_it"` from `merge_mode` to
    `lane_preference` on one side only is both reason- and shape-preserving at
    depth zero, and the suite stayed green.
    """
    import math

    # bool before int: True is an int of value 1, and the two must not be conflated
    # here any more than they are in the decoders. Mapping by exact type() rather
    # than isinstance is what keeps that true.
    def name_of(value):
        if isinstance(value, dict):
            return f"obj({object_shape(value)})"
        if isinstance(value, (list, tuple)):
            return "arr[" + ",".join(name_of(item) for item in value) + "]"
        if value is None:
            return "null"
        if type(value) is bool:
            return f"bool={'true' if value else 'false'}"
        if type(value) is int:
            return f"int={value}"
        if type(value) is float:
            # Rendered, not compared as a number: the two runtimes need not agree on
            # how a double prints. Both use shortest-round-trip for the magnitudes in
            # this table, and a disagreement fails every row at once rather than one
            # row quietly -- which is the right way for it to break.
            if not math.isfinite(value):
                return "real=nonfinite"
            return f"real={value!r}"
        if type(value) is str:
            return f"str={value}"
        return type(value).__name__

    def object_shape(mapping):
        return ",".join(sorted(f"{key}:{name_of(value)}" for key, value in mapping.items()))

    return f"{object_shape(extensions)}#{len(payload)}"


def reason(cls, extensions, payload) -> str:
    try:
        cls.from_wire(extensions, payload)
    except MessageError as exc:
        return getattr(exc, "reason", None) or "unknown"
    except Exception as exc:  # noqa: BLE001 - the point is to name it, not to handle it
        # Same spelling as the Kotlin side, so one assertion covers both. Letting this
        # propagate killed the script, which reads as "python failed to run" rather than
        # "this input crashed a decoder", and lost which of the thirty inputs did it.
        return f"CRASH:{type(exc).__name__}"
    return "ACCEPTED"


def main() -> int:
    # Ahead of the JSON, and on its own line: the Kotlin side parses the last line that
    # starts with "{", so a provenance line cannot break the parse. Which interpreter ran
    # is part of the result, not a detail of how it was produced.
    print(f"VERSION {sys.version.split()[0]}")
    print(
        json.dumps(
            {
                name: f"{reason(*case)}|{shape(case[1], case[2])}"
                for name, case in CASES.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
