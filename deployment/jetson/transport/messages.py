"""Typed messages on top of the opaque transport.

The layer below moves (channel, header, payload) and assigns no meaning. This is
the meaning, and it is a cross-language contract: Kotlin encodes the upstream
half, so every field name and type here is frozen in
specs/transport_golden_frames.json alongside the frame-level cases.

Two placement rules, per specs/transport_protocol.md. Small structured records
ride in the header as extension keys, because that is what an additive JSON
header is for and encoding a hundred-byte GPS fix twice would be waste. Blobs --
a camera JPEG, a HERE response body -- ride in the payload untouched; the HERE
body alone would exceed the header cap.

The channel is the discriminator, so no message carries a `kind`. A channel that
later needs two shapes would have to add one, which is an additive header field
and therefore cheap, but it would be a change.

Unavailable values are `null`, never absent and never a sentinel. Absent would
conflate "the sensor said nothing" with "the sender is an older build", and the
two are deployed separately. NaN cannot go on the wire at all -- framing refuses
it, because Python writes a bare NaN token that a strict parser elsewhere
rejects -- so encoders convert non-finite values to None and decoders map them
back for in-process types like GpsFix that use NaN. That round trip is lossless
in both directions and is the likeliest thing in this module to get wrong.

Note the import direction. This module imports from frames and channels; nothing
in the opaque layer imports this one, and a test asserts it. That is what keeps
"the transport is opaque" true rather than merely stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from transport.channels import Channel
from transport.clock import now_mono_ns
from transport.frames import RESERVED_EXTENSIONS, WIRE_STAMP_KEY

CAPTURE_KEY = "t_capture_mono_ns"

# Mirrored from policy/sim_contract.py, which is itself a vendored copy of the
# simulation's action schema -- the edge runtime must not import the sim stack,
# and the transport must not import policy. A test asserts this copy, that one
# and specs/action_schema.md all agree.
ACTION_HEADS: tuple[str, ...] = (
    "desired_speed_bin",
    "desired_headway_bin",
    "lane_preference",
    "merge_mode",
)
ACTION_VALUES: dict[str, tuple[str, ...]] = {
    "desired_speed_bin": ("slow", "nominal", "fast"),
    "desired_headway_bin": ("normal", "larger", "largest"),
    "lane_preference": ("keep", "prefer_left_if_safe", "prefer_right_if_safe"),
    "merge_mode": ("normal", "create_gap", "hold_lane"),
}

DISPLAY_UNITS: tuple[str, ...] = ("mph", "kmh", "mps")
RATE_KEYS: tuple[str, ...] = ("camera_hz", "gps_hz", "imu_hz", "here_hz")
DROP_KEYS: tuple[str, ...] = ("camera", "gps", "imu", "here")

# A commanded rate above this is a bug, not a request: the camera is the fastest
# sensor in the plan at 10 Hz and the IMU at 50 Hz.
MAX_RATE_HZ = 1000.0

# Counts. Non-negative because a negative one makes a summary under-count with
# no error anywhere, and bounded so a Kotlin Long can hold it -- the frame layer
# deliberately permits header integers past 2**53, but that permission is for
# timestamps, not for tallies.
MAX_COUNT = 2**63 - 1


# The reason vocabulary. A single drop count cannot answer "were these four
# thousand drops one bad field or four", which is the whole point of counting by
# reason, so every refusal carries one of these.
REASON_MISSING_FIELD = "missing_field"
REASON_WRONG_TYPE = "wrong_type"
REASON_NULL_NOT_ALLOWED = "null_not_allowed"
REASON_NON_FINITE = "non_finite"
REASON_OUT_OF_RANGE = "out_of_range"
REASON_UNKNOWN_VALUE = "unknown_value"
REASON_UNEXPECTED_PAYLOAD = "unexpected_payload"
REASON_RESERVED_KEY = "reserved_key"
REASON_NO_TYPED_MESSAGE = "no_typed_message"

REASONS: tuple[str, ...] = (
    REASON_MISSING_FIELD,
    REASON_WRONG_TYPE,
    REASON_NULL_NOT_ALLOWED,
    REASON_NON_FINITE,
    REASON_OUT_OF_RANGE,
    REASON_UNKNOWN_VALUE,
    REASON_UNEXPECTED_PAYLOAD,
    REASON_RESERVED_KEY,
    REASON_NO_TYPED_MESSAGE,
)


class InvalidMessage(ValueError):
    """A message this side built that its own decoder would refuse.

    Deliberately *not* a MessageError. That type means "the peer sent something
    invalid" and its whole handling idiom is drop-and-count; a consumer wrapping
    send in the same idiom would swallow its own bug. This one is the caller's
    fault and is meant to be loud, like FramingError from an oversize payload.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class MessageError(ValueError):
    """A frame that framed correctly but is not a valid message of its type.

    Not fatal to the session, unlike FramingError: framing succeeding proves the
    byte stream is still aligned, so one bad record costs one record.
    """

    def __init__(self, message: str, reason: str) -> None:
        # Required, not defaulted: a default absorbed two untagged raise sites
        # and filed an out-of-schema action value as a type error. Validated,
        # because the spec calls the vocabulary closed, and a typo would split
        # one reason into two buckets no reader could reconcile.
        if reason not in REASONS:
            raise AssertionError(f"{reason!r} is not one of the documented reasons")
        super().__init__(message)
        self.reason = reason


# -- the null convention -----------------------------------------------------


def to_wire_number(value: float | None) -> float | None:
    """None for anything that cannot be represented, including NaN and Inf."""
    if value is None:
        return None
    number = float(value)
    return None if not math.isfinite(number) else number


def from_wire_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    # bool before int: True is an int in Python, so {"speed_mps": true} would
    # otherwise decode to 1.0 m/s.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MessageError(
            f"{field} is {type(value).__name__}, expected a number or null",
            REASON_WRONG_TYPE,
        )
    number = float(value)
    if not math.isfinite(number):
        raise MessageError(
            f"{field} is {number}, which must have been sent as null", REASON_NON_FINITE
        )
    return number


def as_nan(value: float | None) -> float:
    """For in-process types that use NaN for absent, such as GpsFix."""
    return float("nan") if value is None else value


# -- validation helpers ------------------------------------------------------


def require(extensions: Mapping[str, Any], field: str) -> Any:
    if field not in extensions:
        raise MessageError(f"missing {field}", REASON_MISSING_FIELD)
    return extensions[field]


def require_int(extensions: Mapping[str, Any], field: str) -> int:
    value = require(extensions, field)
    # Null before type. The spec's refusal table gives "null where a value is required"
    # its own row, and folding it into wrong_type named the wrong cause on every message
    # with a required integer -- gps, imu, camera and control alike. The Kotlin side read
    # the table correctly; this is the general case of a fix that previously landed on one
    # instance only.
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageError(f"{field} is {type(value).__name__}, expected int", REASON_WRONG_TYPE)
    return value


def require_number(extensions: Mapping[str, Any], field: str) -> float:
    value = from_wire_number(require(extensions, field), field)
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    return value


def require_str(extensions: Mapping[str, Any], field: str) -> str:
    value = require(extensions, field)
    # Null before type, completing a fix that landed on require_int and require_bool and
    # missed this one. Nine string fields diverged from Kotlin because of it: camera
    # `format`; advisory `units`, `lane_text`, `merge_text`, `traffic_text`,
    # `confidence_label`; rate_cmd `trigger`; here `request_url`; telemetry
    # `thermal_status`.
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    if not isinstance(value, str):
        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)
    return value


def require_bool(extensions: Mapping[str, Any], field: str) -> bool:
    value = require(extensions, field)
    # Null before type, for the same reason as require_int: the table gives it a row.
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    if not isinstance(value, bool):
        raise MessageError(f"{field} is {type(value).__name__}, expected bool", REASON_WRONG_TYPE)
    return value


def optional_int(extensions: Mapping[str, Any], field: str) -> int | None:
    """An int or an explicit null. Bool is refused: it is an int in Python and
    would sail through as 0 or 1, which for a timestamp is a silent zero."""
    value = require(extensions, field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageError(
            f"{field} is {type(value).__name__}, expected int or null", REASON_WRONG_TYPE
        )
    return value


def optional_number(extensions: Mapping[str, Any], field: str) -> float | None:
    return from_wire_number(require(extensions, field), field)


def absentable_number(extensions: Mapping[str, Any], field: str) -> float | None:
    """A field that may be absent entirely, not merely null.

    `optional_number` requires the key to be present, which is right for a field
    that has always existed and is only sometimes unavailable. A field *added* to a
    shipped protocol is a different case: an older sender does not write it at all,
    and requiring it would turn every one of that sender's messages into a
    `missing_field` refusal. `here` on a rate command is the same shape, and the
    reason the spec calls extensions additive.
    """
    if field not in extensions:
        return None
    return from_wire_number(extensions[field], field)


def absentable_str(extensions: Mapping[str, Any], field: str) -> str | None:
    """As `absentable_number`, for a string."""
    if field not in extensions:
        return None
    value = extensions[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise MessageError(f"{field!r} is not a string", REASON_WRONG_TYPE)
    return value


def absentable_int(extensions: Mapping[str, Any], field: str) -> int | None:
    """As `absentable_number`, for an integer field added to a shipped protocol.

    Bool is refused the same way `optional_int` refuses it: True is an int in
    Python and would silently decode a timestamp as 0 or 1.
    """
    if field not in extensions:
        return None
    value = extensions[field]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageError(
            f"{field} is {type(value).__name__}, expected int or null", REASON_WRONG_TYPE
        )
    return value


def require_capture(extensions: Mapping[str, Any]) -> int:
    return require_int(extensions, CAPTURE_KEY)


def _nested_object(extensions: Mapping[str, Any], field: str, keys: tuple[str, ...]) -> Mapping:
    """A nested object with every known key present. Unknown keys are ignored.

    Additive on purpose. `rates`, `achieved` and `dropped` describe a sensor set
    that will grow -- the plan expects `telemetry` and `rate_cmd` to move -- and
    refusing an unknown key would break a rolling deploy in both directions at
    once: a new sender's every command dropped by an old receiver, and an old
    sender's every command dropped by a new one. Ignoring unknown keys is safe
    precisely because the known ones are required, so a typo still surfaces as a
    missing key rather than passing silently.

    `action` is different and stays strict: its heads are a closed set defined by
    specs/action_schema.md, not an extensible list.
    """
    value = require(extensions, field)
    # Null gets its own reason, because the spec's refusal table gives it one: "null where
    # a value is required -> null_not_allowed". Folding it into wrong_type was the one
    # refusal disagreement with the Kotlin side where Kotlin read the table correctly, and
    # a counter naming the wrong cause is what per-reason counting exists to avoid.
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    if not isinstance(value, Mapping):
        raise MessageError(
            f"{field} is {type(value).__name__}, expected object", REASON_WRONG_TYPE
        )
    missing = [key for key in keys if key not in value]
    if missing:
        raise MessageError(f"{field} missing {', '.join(missing)}", REASON_MISSING_FIELD)
    return value


def require_mapping_of_numbers(
    extensions: Mapping[str, Any], field: str, keys: tuple[str, ...]
) -> dict[str, float]:
    value = _nested_object(extensions, field, keys)
    return {key: require_number(value, key) for key in keys}


def check_count(value: Any, field: str) -> int:
    """A count: integral, non-negative, and inside a signed 64-bit range.

    Used on both sides. The encoder used to coerce with int(), which destroyed
    the evidence before the decoder could refuse it -- so a fractional count
    crossed the wire looking legitimate while the decoder was documented as
    refusing one.
    """
    if value is None:
        raise MessageError(f"{field} must not be null", REASON_NULL_NOT_ALLOWED)
    # bool before int: True is an int of value 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageError(
            f"{field} is {type(value).__name__}, expected int", REASON_WRONG_TYPE
        )
    if not 0 <= value <= MAX_COUNT:
        raise MessageError(f"{field} is {value}, outside [0, {MAX_COUNT}]", REASON_OUT_OF_RANGE)
    return value


def require_mapping_of_ints(
    extensions: Mapping[str, Any], field: str, keys: tuple[str, ...]
) -> dict[str, int]:
    value = _nested_object(extensions, field, keys)
    return {key: check_count(value[key], f"{field}.{key}") for key in keys}


def check_no_payload(payload: bytes, channel: Channel) -> None:
    if payload:
        raise MessageError(
            f"{channel.value} carries no payload, got {len(payload)} bytes",
            REASON_UNEXPECTED_PAYLOAD,
        )


def check_reserved(extensions: Mapping[str, Any], allow: tuple[str, ...] = ()) -> None:
    """Reserved keys belong to the transport. `allow` names the exceptions.

    Per-message and by exact name, so the one message that legitimately carries
    a transport-owned key does not open the other two for everything else.
    """
    clash = [key for key in extensions if key in RESERVED_EXTENSIONS and key not in allow]
    if clash:
        raise MessageError(
            f"{', '.join(sorted(clash))} are reserved for the transport", REASON_RESERVED_KEY
        )


# -- the messages ------------------------------------------------------------


@dataclass(frozen=True)
class CameraFrame:
    t_capture_mono_ns: int
    frame_id: int
    width: int
    height: int
    format: str
    jpeg: bytes
    quality: int | None = None
    #: When the phone's encoder started and finished turning the packed pixels
    #: into `jpeg`, on the phone's own clock -- the same one `t_capture_mono_ns`
    #: is on, so their differences need no cross-device conversion at all.
    #: Absent rather than merely null: these were added to a channel that
    #: already ships, and a phone built before they existed does not write
    #: them, so requiring the key would refuse every one of its frames.
    t_encode_start_mono_ns: int | None = None
    t_encode_done_mono_ns: int | None = None

    CHANNEL: ClassVar[Channel] = Channel.CAMERA

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        extensions: dict[str, Any] = {
            CAPTURE_KEY: int(self.t_capture_mono_ns),
            "frame_id": int(self.frame_id),
            "width": int(self.width),
            "height": int(self.height),
            "format": self.format,
            "quality": None if self.quality is None else int(self.quality),
        }
        if self.t_encode_start_mono_ns is not None:
            extensions["t_encode_start_mono_ns"] = int(self.t_encode_start_mono_ns)
        if self.t_encode_done_mono_ns is not None:
            extensions["t_encode_done_mono_ns"] = int(self.t_encode_done_mono_ns)
        return extensions, self.jpeg

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "CameraFrame":
        quality = require(extensions, "quality")
        if quality is not None and (isinstance(quality, bool) or not isinstance(quality, int)):
            raise MessageError(
                f"quality is {type(quality).__name__}, expected int or null", REASON_WRONG_TYPE
            )
        return cls(
            t_capture_mono_ns=require_capture(extensions),
            frame_id=require_int(extensions, "frame_id"),
            width=require_int(extensions, "width"),
            height=require_int(extensions, "height"),
            format=require_str(extensions, "format"),
            quality=quality,
            jpeg=payload,
            t_encode_start_mono_ns=absentable_int(extensions, "t_encode_start_mono_ns"),
            t_encode_done_mono_ns=absentable_int(extensions, "t_encode_done_mono_ns"),
        )


@dataclass(frozen=True)
class GpsRecord:
    t_capture_mono_ns: int
    valid: bool
    fix_quality: int
    num_sats: int
    lat: float | None = None
    lon: float | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    hdop: float | None = None
    altitude_m: float | None = None
    utc_epoch_ns: int | None = None

    CHANNEL: ClassVar[Channel] = Channel.GPS

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "valid": bool(self.valid),
                "fix_quality": int(self.fix_quality),
                "num_sats": int(self.num_sats),
                "lat": to_wire_number(self.lat),
                "lon": to_wire_number(self.lon),
                "speed_mps": to_wire_number(self.speed_mps),
                "heading_deg": to_wire_number(self.heading_deg),
                "hdop": to_wire_number(self.hdop),
                "altitude_m": to_wire_number(self.altitude_m),
                "utc_epoch_ns": None if self.utc_epoch_ns is None else int(self.utc_epoch_ns),
            },
            b"",
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "GpsRecord":
        check_no_payload(payload, Channel.GPS)
        valid = require_bool(extensions, "valid")
        lat = optional_number(extensions, "lat")
        lon = optional_number(extensions, "lon")
        # Checked only when the fix claims to be usable: an invalid fix is
        # allowed to carry whatever the receiver had, including nothing.
        if valid:
            if lat is None or not -90.0 <= lat <= 90.0:
                raise MessageError(f"lat {lat} out of range on a valid fix", REASON_OUT_OF_RANGE)
            if lon is None or not -180.0 <= lon <= 180.0:
                raise MessageError(f"lon {lon} out of range on a valid fix", REASON_OUT_OF_RANGE)
        utc = require(extensions, "utc_epoch_ns")
        if utc is not None and (isinstance(utc, bool) or not isinstance(utc, int)):
            raise MessageError(
                f"utc_epoch_ns is {type(utc).__name__}, expected int or null", REASON_WRONG_TYPE
            )
        return cls(
            t_capture_mono_ns=require_capture(extensions),
            valid=valid,
            fix_quality=check_count(require(extensions, "fix_quality"), "fix_quality"),
            num_sats=check_count(require(extensions, "num_sats"), "num_sats"),
            lat=lat,
            lon=lon,
            speed_mps=optional_number(extensions, "speed_mps"),
            heading_deg=optional_number(extensions, "heading_deg"),
            hdop=optional_number(extensions, "hdop"),
            altitude_m=optional_number(extensions, "altitude_m"),
            utc_epoch_ns=utc,
        )


@dataclass(frozen=True)
class ImuSample:
    t_capture_mono_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    accuracy: int | None = None

    CHANNEL: ClassVar[Channel] = Channel.IMU

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "ax": to_wire_number(self.ax),
                "ay": to_wire_number(self.ay),
                "az": to_wire_number(self.az),
                "gx": to_wire_number(self.gx),
                "gy": to_wire_number(self.gy),
                "gz": to_wire_number(self.gz),
                "accuracy": None if self.accuracy is None else int(self.accuracy),
            },
            b"",
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "ImuSample":
        check_no_payload(payload, Channel.IMU)
        accuracy = require(extensions, "accuracy")
        if accuracy is not None and (
            isinstance(accuracy, bool) or not isinstance(accuracy, int)
        ):
            raise MessageError(
                f"accuracy is {type(accuracy).__name__}, expected int or null",
                REASON_WRONG_TYPE,
            )
        axes = {name: require_number(extensions, name) for name in ("ax", "ay", "az", "gx", "gy", "gz")}
        return cls(t_capture_mono_ns=require_capture(extensions), accuracy=accuracy, **axes)


@dataclass(frozen=True)
class HereResponse:
    t_capture_mono_ns: int
    request_url: str
    status: int
    query_lat: float
    query_lon: float
    query_radius_m: float
    t_request_mono_ns: int
    t_response_mono_ns: int
    body: bytes
    content_type: str | None = None

    CHANNEL: ClassVar[Channel] = Channel.HERE

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "request_url": self.request_url,
                "status": int(self.status),
                "content_type": self.content_type,
                "query_lat": to_wire_number(self.query_lat),
                "query_lon": to_wire_number(self.query_lon),
                "query_radius_m": to_wire_number(self.query_radius_m),
                "t_request_mono_ns": int(self.t_request_mono_ns),
                "t_response_mono_ns": int(self.t_response_mono_ns),
            },
            self.body,
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "HereResponse":
        content_type = require(extensions, "content_type")
        if content_type is not None and not isinstance(content_type, str):
            raise MessageError(
                f"content_type is {type(content_type).__name__}, expected str or null",
                REASON_WRONG_TYPE,
            )
        return cls(
            t_capture_mono_ns=require_capture(extensions),
            request_url=require_str(extensions, "request_url"),
            status=require_int(extensions, "status"),
            content_type=content_type,
            query_lat=require_number(extensions, "query_lat"),
            query_lon=require_number(extensions, "query_lon"),
            query_radius_m=require_number(extensions, "query_radius_m"),
            t_request_mono_ns=require_int(extensions, "t_request_mono_ns"),
            t_response_mono_ns=require_int(extensions, "t_response_mono_ns"),
            body=payload,
        )


@dataclass(frozen=True)
class PhoneTelemetry:
    t_capture_mono_ns: int
    thermal_status: str
    achieved: dict[str, float]
    dropped: dict[str, int]
    here_calls: int
    here_errors: int
    thermal_headroom: float | None = None
    #: An absolute temperature for handsets that will not compute headroom, and the
    #: vendor-named kernel zone it came from. Null together: the number cannot be
    #: interpreted, or compared across devices, without the zone that produced it,
    #: because zone names do not mean what they look like.
    skin_temp_c: float | None = None
    skin_temp_zone: str | None = None

    CHANNEL: ClassVar[Channel] = Channel.TELEMETRY

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "thermal_status": self.thermal_status,
                "thermal_headroom": to_wire_number(self.thermal_headroom),
                "achieved": {key: to_wire_number(self.achieved[key]) for key in RATE_KEYS},
                # Passed through, not coerced and not validated here. Coercing
                # with int() destroyed the evidence before the decoder could
                # refuse it; validating only here made telemetry the one message
                # whose encoder checked anything. Validation is centralised in
                # MessageRouter.send, which holds every message to the same
                # thirteen refusal conditions its own decoder applies.
                "dropped": {key: self.dropped[key] for key in DROP_KEYS},
                "here_calls": self.here_calls,
                "here_errors": self.here_errors,
                **(
                    {"skin_temp_c": to_wire_number(self.skin_temp_c)}
                    if self.skin_temp_c is not None
                    else {}
                ),
                **(
                    {"skin_temp_zone": self.skin_temp_zone}
                    if self.skin_temp_zone is not None
                    else {}
                ),
            },
            b"",
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "PhoneTelemetry":
        check_no_payload(payload, Channel.TELEMETRY)
        return cls(
            t_capture_mono_ns=require_capture(extensions),
            thermal_status=require_str(extensions, "thermal_status"),
            thermal_headroom=optional_number(extensions, "thermal_headroom"),
            # Absent-tolerant, not merely nullable: a phone built before these existed
            # does not write them, and requiring them would refuse all of its telemetry.
            skin_temp_c=absentable_number(extensions, "skin_temp_c"),
            skin_temp_zone=absentable_str(extensions, "skin_temp_zone"),
            achieved=require_mapping_of_numbers(extensions, "achieved", RATE_KEYS),
            dropped=require_mapping_of_ints(extensions, "dropped", DROP_KEYS),
            here_calls=check_count(require(extensions, "here_calls"), "here_calls"),
            here_errors=check_count(require(extensions, "here_errors"), "here_errors"),
        )


@dataclass(frozen=True)
class AdvisoryMessage:
    t_capture_mono_ns: int
    rec_speed_mps: float
    rec_speed_display: float
    current_speed_display: float
    units: str
    headway_target_s: float
    lane_text: str
    merge_text: str
    traffic_text: str
    confidence: float
    confidence_label: str
    action: dict[str, str]

    CHANNEL: ClassVar[Channel] = Channel.ADVISORY

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "rec_speed_mps": to_wire_number(self.rec_speed_mps),
                "rec_speed_display": to_wire_number(self.rec_speed_display),
                "current_speed_display": to_wire_number(self.current_speed_display),
                "units": self.units,
                "headway_target_s": to_wire_number(self.headway_target_s),
                "lane_text": self.lane_text,
                "merge_text": self.merge_text,
                "traffic_text": self.traffic_text,
                "confidence": to_wire_number(self.confidence),
                "confidence_label": self.confidence_label,
                "action": {head: self.action[head] for head in ACTION_HEADS},
            },
            b"",
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "AdvisoryMessage":
        check_no_payload(payload, Channel.ADVISORY)
        units = require_str(extensions, "units")
        if units not in DISPLAY_UNITS:
            raise MessageError(f"units {units!r} not one of {DISPLAY_UNITS}", REASON_UNKNOWN_VALUE)
        action = require(extensions, "action")
        if action is None:
            # Same row of the table as every other required-but-null field. This check is
            # inline rather than going through _nested_object -- `action`'s heads are a
            # closed set, unlike the additive rate objects -- which is why the earlier fix
            # to _nested_object did not reach it.
            raise MessageError("action must not be null", REASON_NULL_NOT_ALLOWED)
        if not isinstance(action, Mapping):
            raise MessageError(
                f"action is {type(action).__name__}, expected object", REASON_WRONG_TYPE
            )
        missing = [head for head in ACTION_HEADS if head not in action]
        if missing:
            raise MessageError(f"action missing {', '.join(missing)}", REASON_MISSING_FIELD)
        unexpected = [head for head in action if head not in ACTION_HEADS]
        if unexpected:
            raise MessageError(
                f"action has unexpected {', '.join(sorted(unexpected))}",
                REASON_UNKNOWN_VALUE,
            )
        for head in ACTION_HEADS:
            value = action[head]
            if value not in ACTION_VALUES[head]:
                raise MessageError(
                    f"action.{head} is {value!r}, not one of {ACTION_VALUES[head]}",
                    # A value outside a closed set, exactly like `units`, which
                    # the spec gives an adjacent refusal row. Filed as a type
                    # error, it was wrong in the case the counter exists for.
                    REASON_UNKNOWN_VALUE,
                )
        return cls(
            t_capture_mono_ns=require_capture(extensions),
            rec_speed_mps=require_number(extensions, "rec_speed_mps"),
            rec_speed_display=require_number(extensions, "rec_speed_display"),
            current_speed_display=require_number(extensions, "current_speed_display"),
            units=units,
            headway_target_s=require_number(extensions, "headway_target_s"),
            lane_text=require_str(extensions, "lane_text"),
            merge_text=require_str(extensions, "merge_text"),
            traffic_text=require_str(extensions, "traffic_text"),
            confidence=require_number(extensions, "confidence"),
            confidence_label=require_str(extensions, "confidence_label"),
            action={head: str(action[head]) for head in ACTION_HEADS},
        )


@dataclass(frozen=True)
class HereQuery:
    """The shape of the HERE traffic query, chosen here and executed by the phone.

    Both fields are HERE's own parameters and are passed through verbatim. Neither
    is parsed on the phone: validating HERE's query grammar there would mean
    tracking their API from a device that cannot see it, and getting it wrong
    would refuse a query this side meant.
    """

    in_: str
    location_ref: str
    lat: float
    lon: float
    radius_m: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "in": self.in_,
            "location_ref": self.location_ref,
            "lat": to_wire_number(self.lat),
            "lon": to_wire_number(self.lon),
            "radius_m": to_wire_number(self.radius_m),
        }

    @classmethod
    def from_wire(cls, value: Any) -> "HereQuery | None":
        """Decode the optional ``here`` object.

        Absent is ``None`` -- "this command does not change the query" -- which is
        what lets the field be added without a flag day. The spec spells out the
        asymmetry: an unknown header field is ignored, so an old receiver tolerates
        a new sender, but a new receiver that *requires* a field refuses an old
        sender's command outright.

        Present but malformed is a refusal, not a shrug. A command that names a
        query and gets it wrong is not the same as one that says nothing, and
        ignoring it would leave the phone querying yesterday's corridor while this
        side believed it had moved.
        """
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise MessageError("'here' is not an object", REASON_WRONG_TYPE)
        lat = _here_number(value, "lat")
        lon = _here_number(value, "lon")
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise MessageError("'here' position is off the globe", REASON_OUT_OF_RANGE)
        return cls(
            in_=_here_text(value, "in"),
            location_ref=_here_text(value, "location_ref"),
            lat=lat,
            lon=lon,
            radius_m=_here_number(value, "radius_m"),
        )


def _here_number(value: Mapping[str, Any], key: str) -> float:
    if key not in value:
        raise MessageError(f"'here.{key}' is missing", REASON_MISSING_FIELD)
    number = value[key]
    if number is None:
        raise MessageError(f"'here.{key}' is null", REASON_NULL_NOT_ALLOWED)
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise MessageError(f"'here.{key}' is not a number", REASON_WRONG_TYPE)
    if not math.isfinite(number):
        raise MessageError(f"'here.{key}' is not finite", REASON_NON_FINITE)
    return float(number)


def _here_text(value: Mapping[str, Any], key: str) -> str:
    if key not in value:
        raise MessageError(f"'here.{key}' is missing", REASON_MISSING_FIELD)
    text = value[key]
    if text is None:
        raise MessageError(f"'here.{key}' is null", REASON_NULL_NOT_ALLOWED)
    if not isinstance(text, str):
        raise MessageError(f"'here.{key}' is not a string", REASON_WRONG_TYPE)
    if not text.strip():
        raise MessageError(f"'here.{key}' is blank", REASON_OUT_OF_RANGE)
    return text


@dataclass(frozen=True)
class RateCommand:
    t_capture_mono_ns: int
    rates: dict[str, float]
    trigger: str
    shadow: bool
    here: "HereQuery | None" = None

    CHANNEL: ClassVar[Channel] = Channel.RATE_CMD

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "rates": {key: to_wire_number(self.rates[key]) for key in RATE_KEYS},
                "trigger": self.trigger,
                "shadow": bool(self.shadow),
                **({"here": self.here.to_wire()} if self.here is not None else {}),
            },
            b"",
        )

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "RateCommand":
        check_no_payload(payload, Channel.RATE_CMD)
        rates = require_mapping_of_numbers(extensions, "rates", RATE_KEYS)
        for key, value in rates.items():
            # A non-positive rate would be applied as a period; an absurd one
            # would be a bug rather than a request.
            if not 0.0 < value <= MAX_RATE_HZ:
                raise MessageError(
                    f"rates.{key} is {value}, outside (0, {MAX_RATE_HZ}]", REASON_OUT_OF_RANGE
                )
        return cls(
            here=HereQuery.from_wire(extensions.get("here")),
            t_capture_mono_ns=require_capture(extensions),
            rates=rates,
            trigger=require_str(extensions, "trigger"),
            shadow=require_bool(extensions, "shadow"),
        )



@dataclass(frozen=True)
class TimeSyncMessage:
    """One type for both halves of the exchange, ping and pong.

    The channel is the discriminator for every other message, so a ping type and
    a pong type on `control` would need a `kind` field to tell them apart --
    which is what this protocol refuses. The null convention carries it instead:
    a message whose `t_peer_recv_mono_ns` is null is a ping, and one whose value
    is set is the pong for that `exchange_id`. Since the phone always initiates,
    the receiver's role settles which it is without a field claiming it.

    `t_wire_mono_ns` is transport-owned. It leaves here as the placeholder 0 and
    the writer replaces it immediately before the bytes go out, so what the peer
    reads is a departure time rather than an enqueue time.

    A pong also echoes back the ping's wire stamp in `t_peer_wire_mono_ns`. The
    initiator cannot read its own: the writer stamps the frame after the caller
    has let go of it, so without the echo the only t1 available would be a
    pre-send stamp carrying the queueing delay -- which is the error the wire
    stamp exists to remove. Removing it from the responder's departure and
    leaving it on the initiator's would have fixed one half of a symmetric
    calculation.

    A ping may also carry the previous exchange: `prev_exchange_id`,
    `t_prev_pong_wire_mono_ns` (that pong's wire departure, echoed back) and
    `t_prev_pong_recv_mono_ns` (the initiator's own receipt of that pong).
    Absent on the first ping of a session, and absent-tolerant the same way
    `skin_temp_c` is on telemetry: an older initiator does not write them, and
    requiring them would refuse every one of its pings. Together they let a
    responder -- which by the spec above never gets to initiate and so never
    forms a `TimeSyncSample` of its own -- reconstruct one from the pong it
    already sent and the ping that answers it, with no pending state of its
    own: `t1` is the pong's own departure (this side's clock, already known),
    `t2` is `t_prev_pong_recv_mono_ns` (the initiator's clock), `t3` is this
    ping's own `t_wire_mono_ns` (the initiator's clock), and `t4` is this
    side's receipt of it. All three are present together or absent together,
    for the same reason as the peer trio above.
    """

    t_capture_mono_ns: int
    exchange_id: int
    t_wire_mono_ns: int = 0
    t_peer_recv_mono_ns: int | None = None
    t_peer_recv_wall_ns: int | None = None
    t_peer_wire_mono_ns: int | None = None
    prev_exchange_id: int | None = None
    t_prev_pong_wire_mono_ns: int | None = None
    t_prev_pong_recv_mono_ns: int | None = None

    CHANNEL: ClassVar[Channel] = Channel.CONTROL
    RESERVED_ALLOWED: ClassVar[tuple[str, ...]] = (WIRE_STAMP_KEY,)

    @property
    def is_ping(self) -> bool:
        return self.t_peer_recv_mono_ns is None

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        extensions: dict[str, Any] = {
            CAPTURE_KEY: int(self.t_capture_mono_ns),
            "exchange_id": self.exchange_id,
            WIRE_STAMP_KEY: int(self.t_wire_mono_ns),
            "t_peer_recv_mono_ns": (
                None if self.t_peer_recv_mono_ns is None else int(self.t_peer_recv_mono_ns)
            ),
            "t_peer_recv_wall_ns": (
                None if self.t_peer_recv_wall_ns is None else int(self.t_peer_recv_wall_ns)
            ),
            "t_peer_wire_mono_ns": (
                None if self.t_peer_wire_mono_ns is None else int(self.t_peer_wire_mono_ns)
            ),
        }
        if self.prev_exchange_id is not None:
            assert self.t_prev_pong_wire_mono_ns is not None
            assert self.t_prev_pong_recv_mono_ns is not None
            extensions["prev_exchange_id"] = int(self.prev_exchange_id)
            extensions["t_prev_pong_wire_mono_ns"] = int(self.t_prev_pong_wire_mono_ns)
            extensions["t_prev_pong_recv_mono_ns"] = int(self.t_prev_pong_recv_mono_ns)
        return extensions, b""

    @classmethod
    def from_wire(cls, extensions: Mapping[str, Any], payload: bytes) -> "TimeSyncMessage":
        check_no_payload(payload, Channel.CONTROL)
        # Required fields first, cross-field consistency after. The other order
        # reports a subtle inconsistency while a basic field is simply missing,
        # which sends a reader looking in the wrong place.
        capture = require_capture(extensions)
        exchange_id = check_count(require_int(extensions, "exchange_id"), "exchange_id")
        wire = check_count(require_int(extensions, WIRE_STAMP_KEY), WIRE_STAMP_KEY)

        # All three peer fields, or none of them. A pong missing any one loses a
        # term of the offset arithmetic, and the estimator would then compute a
        # plausible number from an incomplete exchange -- which is worse than
        # refusing, because nothing downstream can tell.
        peer_fields = ("t_peer_recv_mono_ns", "t_peer_recv_wall_ns", "t_peer_wire_mono_ns")
        peer = {name: optional_int(extensions, name) for name in peer_fields}
        absent = sorted(name for name, value in peer.items() if value is None)
        if absent and len(absent) != len(peer_fields):
            present = sorted(set(peer_fields) - set(absent))
            raise MessageError(
                f"{', '.join(absent)} must not be null when {', '.join(present)} is set",
                REASON_NULL_NOT_ALLOWED,
            )

        # The previous-exchange trio: present together or absent together, same
        # rule as the peer trio, but on absence rather than on null -- an older
        # initiator omits the keys entirely rather than nulling them.
        prev_fields = (
            "prev_exchange_id", "t_prev_pong_wire_mono_ns", "t_prev_pong_recv_mono_ns",
        )
        prev = {name: absentable_int(extensions, name) for name in prev_fields}
        prev_absent = sorted(name for name, value in prev.items() if value is None)
        if prev_absent and len(prev_absent) != len(prev_fields):
            prev_present = sorted(set(prev_fields) - set(prev_absent))
            raise MessageError(
                f"{', '.join(prev_absent)} must be set when {', '.join(prev_present)} is",
                REASON_NULL_NOT_ALLOWED,
            )
        return cls(
            t_capture_mono_ns=capture,
            exchange_id=exchange_id,
            t_wire_mono_ns=wire,
            t_peer_recv_mono_ns=peer["t_peer_recv_mono_ns"],
            t_peer_recv_wall_ns=peer["t_peer_recv_wall_ns"],
            t_peer_wire_mono_ns=peer["t_peer_wire_mono_ns"],
            prev_exchange_id=prev["prev_exchange_id"],
            t_prev_pong_wire_mono_ns=prev["t_prev_pong_wire_mono_ns"],
            t_prev_pong_recv_mono_ns=prev["t_prev_pong_recv_mono_ns"],
        )


Message = (
    CameraFrame
    | GpsRecord
    | ImuSample
    | HereResponse
    | PhoneTelemetry
    | AdvisoryMessage
    | RateCommand
    | TimeSyncMessage
)


MESSAGE_FOR_CHANNEL: dict[Channel, type] = {
    Channel.CONTROL: TimeSyncMessage,
    Channel.CAMERA: CameraFrame,
    Channel.GPS: GpsRecord,
    Channel.IMU: ImuSample,
    Channel.HERE: HereResponse,
    Channel.TELEMETRY: PhoneTelemetry,
    Channel.ADVISORY: AdvisoryMessage,
    Channel.RATE_CMD: RateCommand,
}


def decode_message(channel: Channel, extensions: Mapping[str, Any], payload: bytes) -> Message:
    """The channel is the discriminator; there is no `kind` field to consult."""
    message_type = MESSAGE_FOR_CHANNEL.get(channel)
    if message_type is None:
        raise MessageError(f"{channel.value} carries no typed message", REASON_NO_TYPED_MESSAGE)
    return message_type.from_wire(extensions, payload)


# -- bridges to the in-process types -----------------------------------------
#
# Duck-typed on purpose. GpsFix lives in sensors/gps_reader.py behind pynmea2
# and Advisory in policy/advisory.py behind the actor runtime, and importing
# either would drag those dependencies into a module that is otherwise stdlib
# only -- which is what lets the golden vectors and this whole package run
# anywhere, including a Jetson with no model loaded. So these take anything
# carrying the right attribute names, and the conversion still lives in one
# place rather than being rewritten by each consumer.


def gps_record_from_fix(fix: Any, t_capture_mono_ns: int) -> GpsRecord:
    """A GpsRecord from anything shaped like sensors.gps_reader.GpsFix.

    That type uses NaN for every unavailable field, and NaN cannot go on the
    wire, so this is where the conversion happens.
    """
    utc = to_wire_number(getattr(fix, "utc_epoch_s", None))
    return GpsRecord(
        t_capture_mono_ns=int(t_capture_mono_ns),
        valid=bool(fix.valid),
        fix_quality=int(getattr(fix, "fix_quality", 0)),
        num_sats=int(getattr(fix, "num_sats", 0)),
        lat=to_wire_number(getattr(fix, "lat", None)),
        lon=to_wire_number(getattr(fix, "lon", None)),
        speed_mps=to_wire_number(getattr(fix, "speed_mps", None)),
        heading_deg=to_wire_number(getattr(fix, "heading_deg", None)),
        hdop=to_wire_number(getattr(fix, "hdop", None)),
        altitude_m=to_wire_number(getattr(fix, "altitude_m", None)),
        utc_epoch_ns=None if utc is None else int(utc * 1e9),
    )


def advisory_message_from_advisory(advisory: Any, t_capture_mono_ns: int) -> AdvisoryMessage:
    """An AdvisoryMessage from anything shaped like policy.advisory.Advisory."""
    return AdvisoryMessage(
        t_capture_mono_ns=int(t_capture_mono_ns),
        rec_speed_mps=float(advisory.recommended_speed_mps),
        rec_speed_display=float(advisory.recommended_speed_display),
        current_speed_display=float(advisory.current_speed_display),
        units=str(advisory.units),
        headway_target_s=float(advisory.headway_target_s),
        lane_text=str(advisory.lane_text),
        merge_text=str(advisory.merge_text),
        traffic_text=str(advisory.traffic_text),
        confidence=float(advisory.confidence),
        confidence_label=str(advisory.confidence_label),
        action={head: str(advisory.action[head]) for head in ACTION_HEADS},
    )


# -- routing -----------------------------------------------------------------


@dataclass
class ChannelMessageStats:
    channel: Channel
    delivered: int = 0
    decode_errors: int = 0
    last_error: str | None = None
    # Per reason, because one number cannot answer "were these four thousand
    # drops one bad field or four" -- which is the whole point of counting them.
    errors_by_reason: dict[str, int] = field(default_factory=dict)
    # Our own invalid sends, kept separate from the peer's: one is a bug here
    # and one is a bug there, and a summary that added them would hide both.
    send_rejected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "delivered": self.delivered,
            "decode_errors": self.decode_errors,
            "errors_by_reason": dict(sorted(self.errors_by_reason.items())),
            "send_rejected": self.send_rejected,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "last_error": self.last_error,
        }


class MessageRouter:
    """Typed send and receive over a Session, dropping what will not decode.

    The drop-and-count policy lives here rather than inside the decoders, so a
    caller that wants strictness can call decode_message directly and handle
    MessageError itself.
    """

    def __init__(self, session: Any, *, mono_clock: Any = now_mono_ns) -> None:
        self._session = session
        # The package's own monotonic source, the same one Session takes and
        # defaults to. Using time.monotonic() here put two clocks on one
        # deadline, so a test injecting a fake clock would have measured the
        # router against real time.
        self._mono = mono_clock
        self._stats = {channel: ChannelMessageStats(channel) for channel in Channel}

    @property
    def session(self) -> Any:
        return self._session

    def send(self, message: Message, *, wants_wire_stamp: bool = False) -> bool:
        """Encode and send, refusing anything our own decoder would reject.

        One rule, applied to every message rather than to the one type whose
        encoder happened to validate: a sender could otherwise construct and
        send a zero rate -- which the code comment beside the check calls
        "applied as a period" -- an out-of-schema action, or an out-of-range
        coordinate, and learn about it as a silent drop at the far end.

        Raises InvalidMessage, not MessageError, so a consumer cannot swallow
        its own bug with the drop-and-count idiom meant for the peer's.

        `wants_wire_stamp` asks the session to add the departure stamp at write
        time, without it being part of `message.to_wire()`'s own output --
        see `Session.send`. It never appears in `extensions` here, so it plays
        no part in the self-check above.
        """
        extensions, payload = message.to_wire()
        allowed: tuple[str, ...] = getattr(message, "RESERVED_ALLOWED", ())
        stats = self._stats[message.CHANNEL]
        try:
            # Inside the guard, not before it: a reserved key is our own bug on
            # exactly the same terms as an out-of-range rate, and letting one of
            # the two escape as MessageError left a caller two exception types
            # to handle for one mistake.
            check_reserved(extensions, allowed)
            decode_message(message.CHANNEL, extensions, payload)
        except MessageError as exc:
            stats.send_rejected += 1
            stats.rejected_by_reason[exc.reason] = (
                stats.rejected_by_reason.get(exc.reason, 0) + 1
            )
            raise InvalidMessage(str(exc), exc.reason) from None
        return self._session.send(
            message.CHANNEL, payload, extensions,
            allow_reserved=allowed, wants_wire_stamp=wants_wire_stamp,
        )

    def recv(self, channel: Channel, timeout: float | None = 0.0) -> Message | None:
        """The next decodable message, skipping and counting any that are not.

        A thin wrapper over `recv_with_receipt`, so there is one implementation
        of skip-and-count and one of the time budget. Two of them had already
        drifted apart: this one looped and the other returned on the first bad
        record without spending any of the caller's timeout.
        """
        received = self.recv_with_receipt(channel, timeout=timeout)
        return None if received is None else received[0]

    def recv_with_receipt(
        self, channel: Channel, timeout: float | None = 0.0
    ) -> tuple[Message, Any] | None:
        """The next decodable message and the instant the transport stamped it.

        Returns the transport's own `ReceivedMessage` alongside the decoded
        message, so its monotonic and wall stamps -- both taken by the reader at
        the same instant -- travel together.

        The arrival stamp matters for a timestamp exchange in a way it does not
        for a sensor reading: taking the time after this returns folds the
        inbound queue wait and the decode into it, which is the receive-side twin
        of the enqueue-versus-departure error the wire stamp exists to remove,
        and at a 1 Hz poll it is up to a whole period. It also fixes which clock
        the stamp comes from -- the session's own, the same one the writer uses
        for the departure stamp -- so the difference between them is meaningful.

        A bad record costs one record -- and one record's worth of the caller's
        time budget, not a fresh copy of it. Passing the original timeout on
        every skip made a stream of malformed messages block for an unbounded
        multiple of what was asked: measured 6.9x on a 50 Hz channel with one
        broken field, which is the shape a single bad phone build produces. The
        remaining budget is tracked instead -- clamped at zero, so an expired
        budget still polls the queue once rather than skipping it.
        """
        stats = self._stats[channel]
        deadline = None if timeout is None else self._mono() + int(timeout * 1e9)
        while True:
            if deadline is None:
                remaining: float | None = None
            else:
                # Never negative. Returning early on an expired budget skipped
                # the queue entirely, so the default timeout=0.0 -- the poll
                # idiom a control loop uses -- delivered nothing while messages
                # sat waiting. Session.recv checks its queue before its
                # deadline, and this has to agree.
                remaining = max(deadline - self._mono(), 0) / 1e9
            received = self._session.recv(channel, timeout=remaining)
            if received is None:
                return None
            try:
                message = decode_message(channel, received.extensions, received.payload)
            except MessageError as exc:
                stats.decode_errors += 1
                stats.errors_by_reason[exc.reason] = (
                    stats.errors_by_reason.get(exc.reason, 0) + 1
                )
                stats.last_error = str(exc)
                continue
            stats.delivered += 1
            # The whole receipt, not just the monotonic stamp: a caller building
            # a cross-clock pair needs both halves taken at one instant.
            return message, received

    def stats(self) -> dict[Channel, ChannelMessageStats]:
        snapshot = {}
        for channel, stats in self._stats.items():
            fields = dict(vars(stats))
            fields["errors_by_reason"] = dict(fields["errors_by_reason"])
            fields["rejected_by_reason"] = dict(fields["rejected_by_reason"])
            snapshot[channel] = ChannelMessageStats(**fields)
        return snapshot

    def to_record(self) -> dict[str, Any]:
        return {channel.value: stats.to_record() for channel, stats in self._stats.items()}
