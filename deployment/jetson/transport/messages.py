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
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from transport.channels import Channel
from transport.frames import RESERVED_EXTENSIONS

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


class MessageError(ValueError):
    """A frame that framed correctly but is not a valid message of its type.

    Not fatal to the session, unlike FramingError: framing succeeding proves the
    byte stream is still aligned, so one bad record costs one record.
    """

    def __init__(self, message: str, reason: str = REASON_WRONG_TYPE) -> None:
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
    if not isinstance(value, str):
        raise MessageError(f"{field} is {type(value).__name__}, expected str", REASON_WRONG_TYPE)
    return value


def require_bool(extensions: Mapping[str, Any], field: str) -> bool:
    value = require(extensions, field)
    if not isinstance(value, bool):
        raise MessageError(f"{field} is {type(value).__name__}, expected bool", REASON_WRONG_TYPE)
    return value


def optional_number(extensions: Mapping[str, Any], field: str) -> float | None:
    return from_wire_number(require(extensions, field), field)


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


def require_mapping_of_ints(
    extensions: Mapping[str, Any], field: str, keys: tuple[str, ...]
) -> dict[str, int]:
    """Counts, so integral. A fractional drop count is a bug in the sender, and
    silently truncating it would make the round trip not field-for-field."""
    value = _nested_object(extensions, field, keys)
    counts: dict[str, int] = {}
    for key in keys:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int):
            raise MessageError(
                f"{field}.{key} is {type(number).__name__}, expected int", REASON_WRONG_TYPE
            )
        counts[key] = number
    return counts


def check_no_payload(payload: bytes, channel: Channel) -> None:
    if payload:
        raise MessageError(
            f"{channel.value} carries no payload, got {len(payload)} bytes",
            REASON_UNEXPECTED_PAYLOAD,
        )


def check_reserved(extensions: Mapping[str, Any]) -> None:
    clash = [key for key in extensions if key in RESERVED_EXTENSIONS]
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

    CHANNEL: ClassVar[Channel] = Channel.CAMERA

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "frame_id": int(self.frame_id),
                "width": int(self.width),
                "height": int(self.height),
                "format": self.format,
                "quality": None if self.quality is None else int(self.quality),
            },
            self.jpeg,
        )

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
            fix_quality=require_int(extensions, "fix_quality"),
            num_sats=require_int(extensions, "num_sats"),
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
                f"content_type is {type(content_type).__name__}, expected str or null"
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

    CHANNEL: ClassVar[Channel] = Channel.TELEMETRY

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "thermal_status": self.thermal_status,
                "thermal_headroom": to_wire_number(self.thermal_headroom),
                "achieved": {key: to_wire_number(self.achieved[key]) for key in RATE_KEYS},
                "dropped": {key: int(self.dropped[key]) for key in DROP_KEYS},
                "here_calls": int(self.here_calls),
                "here_errors": int(self.here_errors),
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
            achieved=require_mapping_of_numbers(extensions, "achieved", RATE_KEYS),
            dropped=require_mapping_of_ints(extensions, "dropped", DROP_KEYS),
            here_calls=require_int(extensions, "here_calls"),
            here_errors=require_int(extensions, "here_errors"),
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
                    f"action.{head} is {value!r}, not one of {ACTION_VALUES[head]}"
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
class RateCommand:
    t_capture_mono_ns: int
    rates: dict[str, float]
    trigger: str
    shadow: bool

    CHANNEL: ClassVar[Channel] = Channel.RATE_CMD

    def to_wire(self) -> tuple[dict[str, Any], bytes]:
        return (
            {
                CAPTURE_KEY: int(self.t_capture_mono_ns),
                "rates": {key: to_wire_number(self.rates[key]) for key in RATE_KEYS},
                "trigger": self.trigger,
                "shadow": bool(self.shadow),
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
            t_capture_mono_ns=require_capture(extensions),
            rates=rates,
            trigger=require_str(extensions, "trigger"),
            shadow=require_bool(extensions, "shadow"),
        )


Message = (
    CameraFrame
    | GpsRecord
    | ImuSample
    | HereResponse
    | PhoneTelemetry
    | AdvisoryMessage
    | RateCommand
)

MESSAGE_FOR_CHANNEL: dict[Channel, type] = {
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

    def to_record(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "delivered": self.delivered,
            "decode_errors": self.decode_errors,
            "errors_by_reason": dict(sorted(self.errors_by_reason.items())),
            "last_error": self.last_error,
        }


class MessageRouter:
    """Typed send and receive over a Session, dropping what will not decode.

    The drop-and-count policy lives here rather than inside the decoders, so a
    caller that wants strictness can call decode_message directly and handle
    MessageError itself.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._stats = {channel: ChannelMessageStats(channel) for channel in Channel}

    @property
    def session(self) -> Any:
        return self._session

    def send(self, message: Message) -> bool:
        extensions, payload = message.to_wire()
        check_reserved(extensions)
        return self._session.send(message.CHANNEL, payload, extensions)

    def recv(self, channel: Channel, timeout: float | None = 0.0) -> Message | None:
        """The next decodable message, skipping and counting any that are not.

        A bad record costs one record -- and one record's worth of the caller's
        time budget, not a fresh copy of it. Passing the original timeout on
        every skip made a stream of malformed messages block for an unbounded
        multiple of what was asked: measured 6.9x on a 50 Hz channel with one
        broken field, which is the shape a single bad phone build produces. The
        remaining budget is tracked instead.
        """
        stats = self._stats[channel]
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is None:
                remaining: float | None = None
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
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
            return message

    def stats(self) -> dict[Channel, ChannelMessageStats]:
        snapshot = {}
        for channel, stats in self._stats.items():
            fields = dict(vars(stats))
            fields["errors_by_reason"] = dict(fields["errors_by_reason"])
            snapshot[channel] = ChannelMessageStats(**fields)
        return snapshot

    def to_record(self) -> dict[str, Any]:
        return {channel.value: stats.to_record() for channel, stats in self._stats.items()}
