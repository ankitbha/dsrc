"""The typed message layer: round trips, the null convention, and refusals.

The two things most worth pinning here are the NaN-to-null conversion, because
NaN cannot go on the wire at all and GpsFix uses it for every unavailable field,
and that a malformed message does not end a session -- unlike a framing error,
which must.
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

import pytest

from transport.channels import Channel
from transport.frames import WIRE_STAMP_KEY, FramingError, encode
from transport.loopback import loopback_pair
from transport.messages import (
    TimeSyncMessage,
    ACTION_HEADS,
    RATE_KEYS,
    ACTION_VALUES,
    CAPTURE_KEY,
    DISPLAY_UNITS,
    MAX_RATE_HZ,
    MESSAGE_FOR_CHANNEL,
    AdvisoryMessage,
    CameraFrame,
    GpsRecord,
    HereResponse,
    ImuSample,
    MessageError,
    MessageRouter,
    PhoneTelemetry,
    RateCommand,
    advisory_message_from_advisory,
    decode_message,
    gps_record_from_fix,
)
from transport.session import Session

REPO = Path(__file__).resolve().parents[3]


def a_camera_frame(**over):
    fields = dict(
        t_capture_mono_ns=111, frame_id=7, width=1280, height=720,
        format="jpeg", quality=85, jpeg=b"\xff\xd8fake",
    )
    fields.update(over)
    return CameraFrame(**fields)


def a_gps_record(**over):
    fields = dict(
        t_capture_mono_ns=222, valid=True, fix_quality=1, num_sats=9, lat=51.5074,
        lon=-0.1278, speed_mps=13.4, heading_deg=91.2, hdop=0.9, altitude_m=35.0,
        utc_epoch_ns=1_755_648_000_000_000_000,
    )
    fields.update(over)
    return GpsRecord(**fields)


def an_imu_sample(**over):
    fields = dict(t_capture_mono_ns=333, ax=0.1, ay=-0.2, az=9.79, gx=0.01, gy=0.0,
                  gz=-0.02, accuracy=3)
    fields.update(over)
    return ImuSample(**fields)


def a_here_response(**over):
    fields = dict(
        t_capture_mono_ns=444, request_url="https://data.traffic.hereapi.com/v7/flow",
        status=200, content_type="application/json", query_lat=51.5, query_lon=-0.12,
        query_radius_m=1500.0, t_request_mono_ns=440, t_response_mono_ns=444,
        body=b'{"results":[]}',
    )
    fields.update(over)
    return HereResponse(**fields)


def a_telemetry(**over):
    fields = dict(
        t_capture_mono_ns=555, thermal_status="nominal", thermal_headroom=0.42,
        achieved={"camera_hz": 9.9, "gps_hz": 5.0, "imu_hz": 49.8, "here_hz": 0.5},
        dropped={"camera": 3, "gps": 0, "imu": 0, "here": 1},
        here_calls=30, here_errors=1,
    )
    fields.update(over)
    return PhoneTelemetry(**fields)


def an_advisory(**over):
    fields = dict(
        t_capture_mono_ns=666, rec_speed_mps=11.18, rec_speed_display=25.0,
        current_speed_display=27.5, units="mph", headway_target_s=1.6,
        lane_text="Keep lane", merge_text="Normal driving", traffic_text="Moderate",
        confidence=0.87, confidence_label="high",
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"},
    )
    fields.update(over)
    return AdvisoryMessage(**fields)


def a_rate_command(**over):
    fields = dict(
        t_capture_mono_ns=777,
        rates={"camera_hz": 5.0, "gps_hz": 5.0, "imu_hz": 50.0, "here_hz": 0.5},
        trigger="advisory_bin_boundary", shadow=True,
    )
    fields.update(over)
    return RateCommand(**fields)


def a_time_sync_ping(**over):
    fields = dict(t_capture_mono_ns=888, exchange_id=4)
    fields.update(over)
    return TimeSyncMessage(**fields)


def a_time_sync_pong(**over):
    fields = dict(
        t_capture_mono_ns=999, exchange_id=4, t_wire_mono_ns=1_000,
        t_peer_recv_mono_ns=950, t_peer_recv_wall_ns=1_755_648_000_000_000_000,
    )
    fields.update(over)
    return TimeSyncMessage(**fields)


ALL_BUILDERS = [
    a_camera_frame, a_gps_record, an_imu_sample, a_here_response,
    a_telemetry, an_advisory, a_rate_command, a_time_sync_ping, a_time_sync_pong,
]


def roundtrip(message):
    extensions, payload = message.to_wire()
    return decode_message(message.CHANNEL, extensions, payload)


# -- round trips -------------------------------------------------------------


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_every_message_round_trips_field_for_field(build):
    message = build()
    assert roundtrip(message) == message


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_every_message_survives_the_frame_codec(build):
    """The header has to be JSON-encodable, which is the constraint the null
    convention exists to satisfy."""
    from transport.frames import Frame, decode

    message = build()
    extensions, payload = message.to_wire()
    frame = Frame(
        channel=message.CHANNEL, seq=1, t_mono_ns=999, t_wall_ns=1_000,
        payload=payload, extensions=extensions,
    )
    decoded = decode(encode(frame))
    assert decode_message(decoded.channel, decoded.extensions, decoded.payload) == message


def test_every_channel_with_a_message_is_covered_by_a_builder():
    covered = {build().CHANNEL for build in ALL_BUILDERS}
    assert covered == set(MESSAGE_FOR_CHANNEL)


def test_control_now_carries_the_time_sync_message():
    """It used to carry none -- it was the transport's own channel, hello and
    heartbeat -- and that is why `no_typed_message` exists. Task 15 gives it
    one, and the hello still spends control seq 0 alongside."""
    message = a_time_sync_ping()
    extensions, payload = message.to_wire()
    assert decode_message(Channel.CONTROL, extensions, payload) == message


def test_no_typed_message_is_now_a_guard_with_no_live_channel():
    """Every channel has a type, so nothing on the wire can produce this reason
    any more. The guard stays because `Channel` will grow -- adding a channel
    and forgetting its message type is exactly the mistake it catches -- but a
    reason no test can reach is a reason that rots, so this reaches it the only
    way left: by taking an entry back out.

    Deliberately not deleted from the vocabulary. `channels.py` already refuses
    to have a default for an unknown channel for the same reason.
    """
    import transport.messages as module

    assert set(module.MESSAGE_FOR_CHANNEL) == set(Channel), (
        "a channel has no typed message; this test's premise has changed"
    )
    saved = module.MESSAGE_FOR_CHANNEL.pop(Channel.CONTROL)
    try:
        with pytest.raises(MessageError, match="no typed message") as caught:
            decode_message(Channel.CONTROL, {CAPTURE_KEY: 1}, b"")
        assert caught.value.reason == "no_typed_message"
    finally:
        module.MESSAGE_FOR_CHANNEL[Channel.CONTROL] = saved
    assert set(module.MESSAGE_FOR_CHANNEL) == set(Channel), "the table was not restored"


# -- the null convention -----------------------------------------------------


def test_every_optional_field_round_trips_as_null():
    """One test per field rather than a representative sample: the bad case is
    a field nobody ever exercises with an unavailable value."""
    cases = [
        (a_camera_frame(quality=None), "quality"),
        (an_imu_sample(accuracy=None), "accuracy"),
        (a_here_response(content_type=None), "content_type"),
        (a_telemetry(thermal_headroom=None), "thermal_headroom"),
    ]
    for field in ("lat", "lon", "speed_mps", "heading_deg", "hdop", "altitude_m",
                  "utc_epoch_ns"):
        cases.append((a_gps_record(valid=False, **{field: None}), field))
    for message, field in cases:
        extensions, _payload = message.to_wire()
        assert field in extensions, f"{field} must be present, not absent"
        assert extensions[field] is None, f"{field} should be null"
        assert roundtrip(message) == message, field


def test_a_legitimate_zero_is_not_mistaken_for_absent():
    """The other half of the null convention, and the half that had no test.

    0, 0.0 and False are all falsy and all meaningful here: accuracy 0 is
    Android's SENSOR_STATUS_UNRELIABLE, speed 0.0 is a stationary vehicle, and
    shadow False is a command that actually gates. Any `if value:` where
    `if value is None:` belongs turns each of them into "the phone said
    nothing" -- exactly the conflation the null convention exists to prevent.
    """
    cases = [
        (a_camera_frame(quality=0), "quality", 0),
        (an_imu_sample(accuracy=0), "accuracy", 0),
        (a_gps_record(valid=False, utc_epoch_ns=0), "utc_epoch_ns", 0),
        (a_gps_record(valid=False, speed_mps=0.0), "speed_mps", 0.0),
        (a_gps_record(valid=False, altitude_m=0.0), "altitude_m", 0.0),
        (a_gps_record(valid=False, hdop=0.0), "hdop", 0.0),
        (a_gps_record(valid=True, lat=0.0, lon=0.0), "lat", 0.0),
        (a_telemetry(thermal_headroom=0.0), "thermal_headroom", 0.0),
        (a_rate_command(shadow=False), "shadow", False),
    ]
    for message, field, expected in cases:
        extensions, _payload = message.to_wire()
        assert extensions[field] == expected, f"{field} was mangled on the way out"
        assert extensions[field] is not None, f"{field} became null"
        assert roundtrip(message) == message, field


def test_a_zero_drop_count_survives():
    message = a_telemetry(dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0})
    extensions, _ = message.to_wire()
    assert extensions["dropped"] == {"camera": 0, "gps": 0, "imu": 0, "here": 0}
    assert roundtrip(message) == message


def test_as_nan_maps_only_none():
    """as_nan(0.0) becoming NaN would turn a stationary vehicle into a GPS
    outage for any consumer mapping a record back to a GpsFix."""
    from transport.messages import as_nan

    assert as_nan(0.0) == 0.0
    assert as_nan(-0.0) == 0.0
    assert math.isnan(as_nan(None))
    assert as_nan(12.5) == 12.5


def test_a_non_finite_value_becomes_null_rather_than_reaching_the_wire():
    """Framing refuses NaN deliberately -- Python writes a bare NaN token that a
    strict parser elsewhere rejects -- so the conversion has to happen here."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        record = a_gps_record(valid=False, speed_mps=bad, hdop=bad)
        extensions, _ = record.to_wire()
        assert extensions["speed_mps"] is None
        assert extensions["hdop"] is None
        # And the whole thing still frames, which is the point.
        from transport.frames import Frame

        encode(Frame(Channel.GPS, 1, 2, 3, b"", extensions))


def test_a_non_finite_value_that_escaped_the_conversion_is_refused_by_framing():
    """The failure mode if a field is ever missed: loud, in the caller's thread,
    rather than silent corruption."""
    from transport.frames import Frame

    with pytest.raises(FramingError, match="not encodable"):
        encode(Frame(Channel.GPS, 1, 2, 3, b"", {"speed_mps": float("nan")}))


def test_a_null_arriving_where_a_value_is_required_is_refused():
    extensions, payload = an_imu_sample().to_wire()
    extensions["ax"] = None
    with pytest.raises(MessageError, match="ax must not be null"):
        decode_message(Channel.IMU, extensions, payload)


def test_a_non_finite_number_arriving_on_the_wire_is_refused():
    """It cannot be produced by our encoder, but a peer could send it."""
    extensions, payload = an_imu_sample().to_wire()
    extensions["ax"] = float("inf")
    with pytest.raises(MessageError, match="must have been sent as null"):
        decode_message(Channel.IMU, extensions, payload)


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_a_missing_required_field_is_refused(build):
    message = build()
    extensions, payload = message.to_wire()
    for field in list(extensions):
        trimmed = {k: v for k, v in extensions.items() if k != field}
        with pytest.raises(MessageError, match=re.escape(field)):
            decode_message(message.CHANNEL, trimmed, payload)


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_a_missing_capture_stamp_is_refused(build):
    message = build()
    extensions, payload = message.to_wire()
    del extensions[CAPTURE_KEY]
    with pytest.raises(MessageError, match=CAPTURE_KEY):
        decode_message(message.CHANNEL, extensions, payload)


@pytest.mark.parametrize(
    "channel,build",
    [(Channel.GPS, a_gps_record), (Channel.IMU, an_imu_sample),
     (Channel.TELEMETRY, a_telemetry), (Channel.ADVISORY, an_advisory),
     (Channel.RATE_CMD, a_rate_command)],
    ids=["gps", "imu", "telemetry", "advisory", "rate_cmd"],
)
def test_a_payload_on_a_channel_that_carries_none_is_refused(channel, build):
    extensions, _payload = build().to_wire()
    with pytest.raises(MessageError, match="carries no payload"):
        decode_message(channel, extensions, b"unexpected")


def test_a_wrong_json_type_is_refused():
    cases = [
        (Channel.IMU, an_imu_sample, "ax", "not a number"),
        (Channel.GPS, a_gps_record, "valid", "yes"),
        (Channel.GPS, a_gps_record, "num_sats", 9.5),
        (Channel.CAMERA, a_camera_frame, "width", "1280"),
        (Channel.CAMERA, a_camera_frame, "format", 7),
        (Channel.HERE, a_here_response, "status", "200"),
        (Channel.ADVISORY, an_advisory, "lane_text", 3),
        (Channel.RATE_CMD, a_rate_command, "shadow", "true"),
        (Channel.TELEMETRY, a_telemetry, "here_calls", "30"),
    ]
    for channel, build, field, bad in cases:
        extensions, payload = build().to_wire()
        extensions[field] = bad
        with pytest.raises(MessageError, match=re.escape(field)):
            decode_message(channel, extensions, payload)


def test_a_boolean_is_not_accepted_where_an_int_is_required():
    """True is an int in Python and would sail through a naive check."""
    extensions, payload = a_camera_frame().to_wire()
    extensions["width"] = True
    with pytest.raises(MessageError, match="width"):
        decode_message(Channel.CAMERA, extensions, payload)


@pytest.mark.parametrize("head", ACTION_HEADS)
def test_an_action_value_outside_the_schema_is_refused(head):
    extensions, payload = an_advisory().to_wire()
    extensions["action"][head] = "sideways"
    with pytest.raises(MessageError, match=re.escape(f"action.{head}")):
        decode_message(Channel.ADVISORY, extensions, payload)


def test_an_action_missing_a_head_is_refused():
    extensions, payload = an_advisory().to_wire()
    del extensions["action"]["merge_mode"]
    with pytest.raises(MessageError, match="merge_mode"):
        decode_message(Channel.ADVISORY, extensions, payload)


def test_an_action_with_an_extra_head_is_refused():
    extensions, payload = an_advisory().to_wire()
    extensions["action"]["desired_altitude"] = "high"
    with pytest.raises(MessageError, match="unexpected"):
        decode_message(Channel.ADVISORY, extensions, payload)


def test_units_outside_the_three_are_refused():
    extensions, payload = an_advisory().to_wire()
    extensions["units"] = "furlongs_per_fortnight"
    with pytest.raises(MessageError, match="units"):
        decode_message(Channel.ADVISORY, extensions, payload)


@pytest.mark.parametrize("shadow", [True, False])
def test_both_shadow_states_round_trip(shadow):
    """The flag that distinguishes a command that gates from one that is merely
    recorded. Pinned in one state only, a mutation encoding False as null would
    drop every real command while every shadow one sailed through -- and task
    30's comparison would then see only the shadow arm and read as clean."""
    message = a_rate_command(shadow=shadow)
    extensions, _ = message.to_wire()
    assert extensions["shadow"] is shadow
    assert roundtrip(message).shadow is shadow


@pytest.mark.parametrize("units", DISPLAY_UNITS)
def test_each_allowed_unit_is_accepted(units):
    assert roundtrip(an_advisory(units=units)).units == units


@pytest.mark.parametrize("bad", [0.0, -1.0, MAX_RATE_HZ + 1.0])
def test_a_rate_outside_its_range_is_refused(bad):
    """A non-positive rate would be applied as a period."""
    extensions, payload = a_rate_command().to_wire()
    extensions["rates"]["camera_hz"] = bad
    with pytest.raises(MessageError, match="camera_hz"):
        decode_message(Channel.RATE_CMD, extensions, payload)


def test_a_rate_at_the_ceiling_is_accepted():
    assert roundtrip(a_rate_command(
        rates={"camera_hz": MAX_RATE_HZ, "gps_hz": 1.0, "imu_hz": 1.0, "here_hz": 1.0}
    )).rates["camera_hz"] == MAX_RATE_HZ


def test_a_rates_object_missing_a_key_is_refused():
    extensions, payload = a_rate_command().to_wire()
    del extensions["rates"]["imu_hz"]
    with pytest.raises(MessageError, match="imu_hz"):
        decode_message(Channel.RATE_CMD, extensions, payload)


@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_a_coordinate_out_of_range_on_a_valid_fix_is_refused(lat, lon):
    extensions, payload = a_gps_record(lat=lat, lon=lon).to_wire()
    with pytest.raises(MessageError, match="out of range"):
        decode_message(Channel.GPS, extensions, payload)


def test_an_invalid_fix_may_carry_anything_including_nothing():
    """An invalid fix is allowed to report whatever the receiver had."""
    record = a_gps_record(valid=False, lat=None, lon=None)
    assert roundtrip(record) == record


def test_a_null_coordinate_on_a_valid_fix_is_refused():
    extensions, payload = a_gps_record(valid=True, lat=None).to_wire()
    with pytest.raises(MessageError, match="out of range"):
        decode_message(Channel.GPS, extensions, payload)


# -- the bridges to the in-process types -------------------------------------


class FixLikeGpsFix:
    """The GpsFix shape, without importing it: that module needs pynmea2."""

    def __init__(self, **over):
        nan = float("nan")
        self.valid = over.get("valid", True)
        self.lat = over.get("lat", 51.5074)
        self.lon = over.get("lon", -0.1278)
        self.speed_mps = over.get("speed_mps", 13.4)
        self.heading_deg = over.get("heading_deg", nan)
        self.fix_quality = over.get("fix_quality", 1)
        self.num_sats = over.get("num_sats", 9)
        self.hdop = over.get("hdop", nan)
        self.altitude_m = over.get("altitude_m", 35.0)
        self.utc_epoch_s = over.get("utc_epoch_s", 1_755_648_000.0)


def test_a_fix_full_of_nan_becomes_a_record_full_of_null():
    nan = float("nan")
    fix = FixLikeGpsFix(valid=False, lat=nan, lon=nan, speed_mps=nan,
                        heading_deg=nan, hdop=nan, altitude_m=nan, utc_epoch_s=nan)
    record = gps_record_from_fix(fix, t_capture_mono_ns=42)
    extensions, _ = record.to_wire()
    for field in ("lat", "lon", "speed_mps", "heading_deg", "hdop", "altitude_m",
                  "utc_epoch_ns"):
        assert extensions[field] is None, field
    assert roundtrip(record) == record


def test_the_fix_conversion_is_lossless_in_both_directions():
    """A value that was NaN comes back as None and can be mapped back to NaN;
    a value that was present comes back unchanged."""
    from transport.messages import as_nan

    fix = FixLikeGpsFix()
    record = roundtrip(gps_record_from_fix(fix, t_capture_mono_ns=1))
    assert record.lat == pytest.approx(fix.lat)
    assert record.speed_mps == pytest.approx(fix.speed_mps)
    assert record.heading_deg is None and math.isnan(fix.heading_deg)
    assert math.isnan(as_nan(record.heading_deg))
    assert math.isnan(as_nan(record.hdop))
    assert record.utc_epoch_ns == 1_755_648_000_000_000_000


def test_the_bridge_works_against_the_real_gps_fix():
    """Where the dependency exists -- so this runs on the Jetson and skips here."""
    pytest.importorskip("pynmea2")
    from sensors.gps_reader import GpsFix

    fix = GpsFix(valid=True, lat=51.5, lon=-0.1, speed_mps=12.0, num_sats=8,
                 fix_quality=1, utc_epoch_s=1_755_648_000.0)
    record = gps_record_from_fix(fix, t_capture_mono_ns=5)
    assert record.valid is True
    assert record.lat == pytest.approx(51.5)
    # The fields GpsFix leaves as NaN must have become null.
    extensions, _ = record.to_wire()
    assert extensions["heading_deg"] is None
    assert extensions["hdop"] is None
    assert roundtrip(record) == record


def test_the_advisory_bridge_works_against_the_real_advisory():
    # policy.advisory arrives by way of the actor runtime, which pulls numpy and
    # torch. Guarded for the same reason the bridges are duck-typed: this suite
    # has to run where those are absent, which is the whole argument.
    pytest.importorskip("torch")
    from policy.advisory import Advisory

    advisory = Advisory(
        recommended_speed_mps=11.18, recommended_speed_display=25.0,
        current_speed_display=27.5, units="mph", headway_target_s=1.6,
        lane_text="Keep lane", merge_text="Normal driving", traffic_text="Moderate",
        confidence_label="high", confidence=0.87,
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"},
    )
    message = advisory_message_from_advisory(advisory, t_capture_mono_ns=9)
    assert roundtrip(message) == message
    assert message.action == advisory.action
    assert message.rec_speed_display == pytest.approx(advisory.recommended_speed_display)


# -- the action vocabulary is mirrored, so it must not drift ------------------


def test_the_action_vocabulary_matches_the_vendored_sim_contract():
    """messages.py keeps its own copy so the transport stays stdlib-only and
    does not import policy. Three copies now exist -- src/, the vendored
    sim_contract, and this -- so equality is asserted rather than assumed."""
    pytest.importorskip("numpy")
    from policy import sim_contract

    assert ACTION_VALUES == sim_contract.ACTION_VALUES
    assert ACTION_HEADS == sim_contract.ACTION_HEADS


def spec_allowed_values(head: str) -> set[str]:
    """The values under one head's `allowed values:` block, and no further.

    Anchored to the block rather than to the next head: a window running to the
    next head swept up three other heads' values, so the check passed with a
    value listed under the wrong head -- which is precisely the drift it exists
    to catch, since ACTION_VALUES is what the decoder validates against.
    """
    text = (REPO / "specs" / "action_schema.md").read_text()
    after_head = text.split(f"`{head}`", 1)
    assert len(after_head) == 2, f"{head} is not documented"
    after_allowed = after_head[1].split("- allowed values:", 1)
    assert len(after_allowed) == 2, f"{head} has no allowed-values block"
    block = after_allowed[1].split("- meaning:", 1)[0]
    return set(re.findall(r"^\s*-\s+`([\w_]+)`\s*$", block, re.M))


@pytest.mark.parametrize("head", ACTION_HEADS)
def test_the_action_vocabulary_matches_the_action_schema_spec(head):
    """Set equality, both directions: a value in the code and not the spec would
    accept an out-of-schema action, and one in the spec and not the code would
    silently drop a legitimate one."""
    assert spec_allowed_values(head) == set(ACTION_VALUES[head])


# `normal` is legitimately shared: it is a headway bin and a merge mode.
LEGITIMATELY_SHARED_VALUES = {"normal"}


def test_the_spec_blocks_are_pairwise_disjoint_apart_from_the_shared_value():
    """Independent of the code, which the previous version of this was not.

    It compared each block against ACTION_VALUES, and its sibling already
    asserts those are equal -- so it was empty by construction and passed even
    when the window was widened and the code widened to match. This asserts a
    property of the spec text alone: a value appears under one head, except
    `normal`, which really is both a headway bin and a merge mode.
    """
    blocks = {head: spec_allowed_values(head) for head in ACTION_HEADS}
    for left in ACTION_HEADS:
        for right in ACTION_HEADS:
            if left >= right:
                continue
            shared = blocks[left] & blocks[right]
            assert shared <= LEGITIMATELY_SHARED_VALUES, (
                f"{left} and {right} both list {sorted(shared - LEGITIMATELY_SHARED_VALUES)}"
            )
    assert blocks["desired_headway_bin"] & blocks["merge_mode"] == {"normal"}


SAMPLE_FOR_CHANNEL = {
    Channel.CAMERA: a_camera_frame, Channel.GPS: a_gps_record,
    Channel.IMU: an_imu_sample, Channel.HERE: a_here_response,
    Channel.TELEMETRY: a_telemetry, Channel.ADVISORY: an_advisory,
    Channel.RATE_CMD: a_rate_command, Channel.CONTROL: a_time_sync_ping,
}


def message_table_rows() -> dict[str, str]:
    """The message table only. The channel-policy table earlier in the same file
    matches the same row pattern, and a whole-file findall is last-wins, so the
    section is sliced out first rather than relied on to come second."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    section = text.split("### The message set", 1)
    assert len(section) == 2, "the message set section is missing"
    body = section[1].split("### ", 1)[0]
    return dict(re.findall(r"^\| `(\w+)` \| (.+?) \| .+ \|$", body, re.M))


@pytest.mark.parametrize("channel", sorted(MESSAGE_FOR_CHANNEL, key=lambda c: c.value),
                         ids=lambda c: c.value)
def test_the_message_table_matches_the_encoder_in_both_directions(channel):
    """Set equality. One-directional was blind to the spec listing a field the
    code never sends, which is the direction that breaks Kotlin: absent is a
    refusal condition here, so a documented-but-unsent field would have the
    phone drop every message of that type."""
    rows = message_table_rows()
    assert channel.value in rows, f"{channel.value} has no message row"
    documented = set(re.findall(r"`([\w_]+)`", rows[channel.value]))
    extensions, _payload = SAMPLE_FOR_CHANNEL[channel]().to_wire()
    emitted = {field for field in extensions if field != CAPTURE_KEY}
    assert documented == emitted, (
        f"{channel.value}: spec-only {sorted(documented - emitted)}, "
        f"code-only {sorted(emitted - documented)}"
    )


def test_the_message_table_covers_every_typed_channel():
    rows = message_table_rows()
    assert {c.value for c in MESSAGE_FOR_CHANNEL} <= set(rows)





# -- routing: a bad message is not a bad stream ------------------------------


def quiet_pair():
    left, right = loopback_pair()
    sender = Session(left, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    receiver = Session(right, session_id=2, heartbeat_s=None, stall_timeout_s=None).start()
    return sender, receiver


def wait_until(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.mark.parametrize("build", ALL_BUILDERS, ids=lambda b: b.__name__)
def test_a_message_crosses_a_real_session_intact(build):
    """Field for field, except the one field the transport owns.

    A time-sync message leaves with `t_wire_mono_ns` as the placeholder 0 and
    arrives with a real departure stamp, so plain equality would fail on the
    message whose whole point is that the transport rewrites it. Asserting the
    rest is unchanged *and* that this one changed is the stronger statement.
    """
    from dataclasses import replace as replace_fields

    sender, receiver = quiet_pair()
    out_router, in_router = MessageRouter(sender), MessageRouter(receiver)
    try:
        message = build()
        assert out_router.send(message) is True
        arrived = in_router.recv(message.CHANNEL, timeout=5.0)
        assert arrived is not None
        if WIRE_STAMP_KEY in getattr(message, "RESERVED_ALLOWED", ()):
            assert arrived.t_wire_mono_ns > 0, "the writer never stamped it"
            assert arrived.t_wire_mono_ns != message.t_wire_mono_ns
            assert replace_fields(arrived, t_wire_mono_ns=message.t_wire_mono_ns) == message
        else:
            assert arrived == message
        assert in_router.stats()[message.CHANNEL].delivered == 1
        assert in_router.stats()[message.CHANNEL].decode_errors == 0
    finally:
        sender.close()
        receiver.close()


def test_the_capture_stamp_precedes_the_enqueue_stamp():
    """The one latency this protocol lets you compute without the offset
    estimate: same device, same monotonic clock."""
    from transport.clock import now_mono_ns

    left, right = loopback_pair()
    sender = Session(left, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    receiver = Session(right, session_id=2, heartbeat_s=None, stall_timeout_s=None).start()
    try:
        captured = now_mono_ns()
        MessageRouter(sender).send(a_gps_record(t_capture_mono_ns=captured))
        received = receiver.recv(Channel.GPS, timeout=5.0)
        assert received is not None
        queueing_ns = received.frame.t_mono_ns - captured
        assert queueing_ns >= 0, "enqueue stamped before capture"
        assert queueing_ns < 2e9
    finally:
        sender.close()
        receiver.close()


def test_a_malformed_message_is_dropped_and_the_session_survives():
    """Unlike a framing error, which ends the session: framing succeeding proves
    the byte stream is still aligned, so one bad record costs one record."""
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        good = a_gps_record()
        extensions, payload = good.to_wire()
        del extensions["num_sats"]
        sender.send(Channel.GPS, payload, extensions)          # malformed
        MessageRouter(sender).send(a_gps_record(num_sats=11))   # then a good one

        received = router.recv(Channel.GPS, timeout=5.0)
        assert received is not None and received.num_sats == 11
        stats = router.stats()[Channel.GPS]
        assert stats.decode_errors == 1
        assert stats.delivered == 1
        assert "num_sats" in stats.last_error
        assert not receiver.is_closed, "a bad message must not end the session"
    finally:
        sender.close()
        receiver.close()


def test_a_drop_on_one_channel_does_not_touch_another():
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        extensions, payload = an_imu_sample().to_wire()
        del extensions["az"]
        sender.send(Channel.IMU, payload, extensions)
        # Waits for the drop to be counted, not for "nothing arrived" -- which
        # was also what the next assertion checks, so slow delivery failed the
        # test rather than waiting for it.
        assert wait_until(
            lambda: (router.recv(Channel.IMU, timeout=0.05) is None)
            and router.stats()[Channel.IMU].decode_errors == 1,
            timeout=5.0,
        )
        assert router.stats()[Channel.IMU].decode_errors == 1
        for channel in Channel:
            if channel is not Channel.IMU:
                assert router.stats()[channel].decode_errors == 0, channel
    finally:
        sender.close()
        receiver.close()


def test_the_message_counters_reconcile_with_what_the_transport_delivered():
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    out = MessageRouter(sender)
    try:
        for index in range(6):
            if index % 3 == 0:
                extensions, payload = an_imu_sample().to_wire()
                del extensions["gy"]
                sender.send(Channel.IMU, payload, extensions)
            else:
                out.send(an_imu_sample(t_capture_mono_ns=index))
        assert wait_until(
            lambda: receiver.stats().channels[Channel.IMU].received == 6, timeout=5.0
        )
        while router.recv(Channel.IMU, timeout=0.1) is not None:
            pass
        stats = router.stats()[Channel.IMU]
        transport_delivered = receiver.stats().channels[Channel.IMU].delivered
        assert stats.delivered + stats.decode_errors == transport_delivered
        assert stats.decode_errors == 2
    finally:
        sender.close()
        receiver.close()


def test_a_reserved_extension_on_a_message_is_refused_before_sending():
    """No encoder here produces `hello` or `heartbeat`, so this guards against a
    future field name colliding with the transport's own -- which would be
    consumed as transport traffic and never delivered."""
    from transport.messages import check_reserved

    extensions, _payload = a_gps_record().to_wire()
    check_reserved(extensions)  # the real thing is clean
    for reserved in ("hello", "heartbeat"):
        with pytest.raises(MessageError, match="reserved"):
            check_reserved({**extensions, reserved: True})


def test_the_router_record_is_json_shaped():
    left, _right = loopback_pair()
    session = Session(left, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    try:
        record = MessageRouter(session).to_record()
        assert set(record) == {channel.value for channel in Channel}
        json.dumps(record, allow_nan=False)
    finally:
        session.close()


# -- the opaque layer must not know about this one ---------------------------


def test_nothing_in_the_opaque_layer_imports_the_message_layer():
    """What keeps "the transport is opaque" true rather than merely stated.

    messages.py imports frames and channels; the dependency must not run the
    other way, or the byte mover starts caring what the bytes mean.
    """
    opaque = [
        "frames.py", "channels.py", "connection.py", "loopback.py",
        "handshake.py", "session.py", "endpoint.py", "tcp.py", "client.py",
    ]
    transport = REPO / "deployment" / "jetson" / "transport"
    offenders = []
    for name in opaque:
        text = (transport / name).read_text()
        if re.search(r"^\s*(from transport\.messages|import transport\.messages)", text, re.M):
            offenders.append(name)
    assert offenders == [], f"the opaque layer imports messages: {offenders}"


def test_the_message_layer_imports_nothing_outside_the_standard_library():
    """Which is what lets this package and the golden vectors run anywhere,
    including a Jetson with no model loaded and no pynmea2 in reach."""
    text = (REPO / "deployment" / "jetson" / "transport" / "messages.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, re.M)
    # Deliberately enumerated, and split so the two kinds of addition read
    # differently. This test caught `time` being added for the recv budget; the
    # right answer was not to allow it but to use transport.clock, which is the
    # package's own monotonic source and what Session already defaults to. A
    # widened stdlib allow-list would have been the weaker outcome.
    stdlib = {"__future__", "math", "dataclasses", "typing"}
    transport_internal = {"transport.channels", "transport.frames", "transport.clock"}
    allowed = stdlib | transport_internal
    assert set(imports) <= allowed, f"unexpected imports: {sorted(set(imports) - allowed)}"
    assert "time" not in imports, "use transport.clock, the package's own monotonic source"


# -- the branches mutation showed were unpinned -------------------------------


def test_a_boolean_is_not_accepted_where_a_number_is_required():
    """True is an int, so {"speed_mps": true} would decode to 1.0 m/s. The int
    path was guarded and tested; the number path was guarded and not."""
    extensions, payload = a_gps_record(valid=False).to_wire()
    extensions["speed_mps"] = True
    with pytest.raises(MessageError, match="speed_mps"):
        decode_message(Channel.GPS, extensions, payload)


def test_the_capture_stamp_must_be_an_integer():
    """The one field decision 8 makes mandatory on every message."""
    for build in ALL_BUILDERS:
        message = build()
        extensions, payload = message.to_wire()
        extensions[CAPTURE_KEY] = 1.5
        with pytest.raises(MessageError, match=CAPTURE_KEY):
            decode_message(message.CHANNEL, extensions, payload)


@pytest.mark.parametrize("lat,lon", [(90.0, 180.0), (-90.0, -180.0), (90.0, -180.0)])
def test_the_coordinate_bounds_are_inclusive(lat, lon):
    """The poles and the antimeridian are real places."""
    record = a_gps_record(lat=lat, lon=lon)
    assert roundtrip(record) == record


def test_a_nested_object_ignores_an_unknown_key_but_requires_the_known_ones():
    """Additive on purpose: refusing an unknown key would break a rolling deploy
    in both directions. Safe because the known keys are still required, so a
    misspelling surfaces as a missing key."""
    extensions, payload = a_rate_command().to_wire()
    extensions["rates"]["lidar_hz"] = 20.0
    decoded = decode_message(Channel.RATE_CMD, extensions, payload)
    assert set(decoded.rates) == set(RATE_KEYS)

    extensions, payload = a_rate_command().to_wire()
    extensions["rates"]["camra_hz"] = extensions["rates"].pop("camera_hz")
    with pytest.raises(MessageError, match="camera_hz"):
        decode_message(Channel.RATE_CMD, extensions, payload)


@pytest.mark.parametrize("field", ["dropped.camera", "here_calls"])
def test_a_boolean_is_not_accepted_where_a_count_is_required(field):
    """True is an int of value 1. This guard was added on the number path with a
    test and on the count path without -- the same shape, one function over."""
    extensions, payload = a_telemetry().to_wire()
    if field == "here_calls":
        extensions["here_calls"] = True
    else:
        extensions["dropped"]["camera"] = True
    with pytest.raises(MessageError, match="expected int"):
        decode_message(Channel.TELEMETRY, extensions, payload)


@pytest.mark.parametrize("bad", [-1, -5, 2**63, 2**70])
def test_a_count_outside_its_range_is_refused(bad):
    """A negative count makes a summary under-count with no error anywhere, and
    an unbounded one overflows a Kotlin Long."""
    extensions, payload = a_telemetry().to_wire()
    extensions["dropped"]["camera"] = bad
    with pytest.raises(MessageError, match="outside"):
        decode_message(Channel.TELEMETRY, extensions, payload)


def test_a_count_at_its_bounds_is_accepted():
    from transport.messages import MAX_COUNT

    message = a_telemetry(dropped={"camera": 0, "gps": MAX_COUNT, "imu": 1, "here": 0})
    assert roundtrip(message) == message


def test_a_null_count_reports_a_null_reason_not_a_type_error():
    """The same wire condition landed in two buckets depending on which nested
    object it was in, so a summary showed nulls under rates and not dropped."""
    extensions, payload = a_telemetry().to_wire()
    extensions["dropped"]["camera"] = None
    with pytest.raises(MessageError) as caught:
        decode_message(Channel.TELEMETRY, extensions, payload)
    assert caught.value.reason == "null_not_allowed"

    extensions, payload = a_rate_command().to_wire()
    extensions["rates"]["camera_hz"] = None
    with pytest.raises(MessageError) as other:
        decode_message(Channel.RATE_CMD, extensions, payload)
    assert other.value.reason == "null_not_allowed"


def test_to_wire_never_raises_so_a_caller_can_predict_it():
    """A pure projection. It used to coerce counts with int(), and then to
    validate them -- which made telemetry the one message whose encoder checked
    anything, so a sender could still emit five messages its own decoder
    refuses."""
    for message in [
        a_telemetry(dropped={"camera": 2.7, "gps": 0, "imu": 0, "here": 0}),
        a_telemetry(here_calls=-3),
        a_rate_command(rates={"camera_hz": 0.0, "gps_hz": 1.0, "imu_hz": 1.0, "here_hz": 1.0}),
        an_advisory(units="furlongs"),
        a_gps_record(lat=91.0),
    ]:
        extensions, payload = message.to_wire()
        assert isinstance(extensions, dict)
        # And no truncation on the way out: the value is carried as given, so
        # the receiver refuses it rather than seeing a rounded one.
    fractional = a_telemetry(dropped={"camera": 2.7, "gps": 0, "imu": 0, "here": 0})
    assert fractional.to_wire()[0]["dropped"]["camera"] == 2.7


SEND_REFUSALS = [
    ("fractional count",
     lambda: a_telemetry(dropped={"camera": 2.7, "gps": 0, "imu": 0, "here": 0}),
     "wrong_type"),
    ("negative count", lambda: a_telemetry(here_calls=-3), "out_of_range"),
    ("zero rate",
     lambda: a_rate_command(rates={"camera_hz": 0.0, "gps_hz": 1.0, "imu_hz": 1.0,
                                   "here_hz": 1.0}),
     "out_of_range"),
    ("negative rate",
     lambda: a_rate_command(rates={"camera_hz": -5.0, "gps_hz": 1.0, "imu_hz": 1.0,
                                   "here_hz": 1.0}),
     "out_of_range"),
    ("rate above the ceiling",
     lambda: a_rate_command(rates={"camera_hz": 5000.0, "gps_hz": 1.0, "imu_hz": 1.0,
                                   "here_hz": 1.0}),
     "out_of_range"),
    ("bad units", lambda: an_advisory(units="furlongs"), "unknown_value"),
    ("out-of-schema action",
     lambda: an_advisory(action={"desired_speed_bin": "nominal",
                                 "desired_headway_bin": "normal",
                                 "lane_preference": "keep",
                                 "merge_mode": "sideways"}),
     "unknown_value"),
    ("coordinate out of range on a valid fix", lambda: a_gps_record(lat=91.0),
     "out_of_range"),
]


@pytest.mark.parametrize(
    "label,build_bad,reason", SEND_REFUSALS, ids=[case[0] for case in SEND_REFUSALS]
)
def test_send_refuses_what_our_own_decoder_would_refuse(label, build_bad, reason):
    """One rule for every message, not for the one whose encoder happened to
    validate. A zero rate is the sharp case: the check beside it calls it
    "applied as a period", and a sender used to learn about it as a silent drop
    at the far end."""
    from transport.messages import InvalidMessage

    sender, receiver = quiet_pair()
    router = MessageRouter(sender)
    try:
        with pytest.raises(InvalidMessage) as caught:
            router.send(build_bad())
        assert caught.value.reason == reason, f"{label}: got {caught.value.reason!r}"
        stats = router.stats()[build_bad().CHANNEL]
        assert stats.send_rejected == 1
        assert stats.rejected_by_reason == {reason: 1}
        # Nothing reached the wire.
        assert receiver.stats().channels[build_bad().CHANNEL].received == 0
    finally:
        sender.close()
        receiver.close()


def test_an_invalid_send_is_not_a_message_error():
    """MessageError means the peer sent something bad and its idiom is
    drop-and-count. A consumer wrapping send in that idiom would swallow its own
    bug, so this is a separate type."""
    from transport.messages import InvalidMessage

    assert not issubclass(InvalidMessage, MessageError)
    sender, receiver = quiet_pair()
    try:
        with pytest.raises(InvalidMessage):
            MessageRouter(sender).send(a_telemetry(here_calls=-1))
        try:
            MessageRouter(sender).send(a_telemetry(here_calls=-1))
        except MessageError:  # pragma: no cover - would mean the types merged
            pytest.fail("InvalidMessage was caught as MessageError")
        except InvalidMessage:
            pass
    finally:
        sender.close()
        receiver.close()


def test_our_own_rejections_are_counted_apart_from_the_peers():
    """One is a bug here and one is a bug there; a summary adding them hides
    both."""
    from transport.messages import InvalidMessage

    sender, receiver = quiet_pair()
    out, back = MessageRouter(sender), MessageRouter(receiver)
    try:
        with pytest.raises(InvalidMessage):
            out.send(a_telemetry(here_calls=-1))
        bad, payload = an_imu_sample().to_wire()
        del bad["az"]
        sender.send(Channel.IMU, payload, bad)
        assert wait_until(
            lambda: back.recv(Channel.IMU, timeout=0.05) is None
            and back.stats()[Channel.IMU].decode_errors == 1,
            timeout=5.0,
        )
        assert out.stats()[Channel.TELEMETRY].send_rejected == 1
        assert out.stats()[Channel.TELEMETRY].decode_errors == 0
        assert back.stats()[Channel.IMU].decode_errors == 1
        assert back.stats()[Channel.IMU].send_rejected == 0
        json.dumps(out.to_record(), allow_nan=False)
    finally:
        sender.close()
        receiver.close()


def test_a_valid_message_still_sends():
    sender, receiver = quiet_pair()
    out, back = MessageRouter(sender), MessageRouter(receiver)
    try:
        assert out.send(a_telemetry()) is True
        assert back.recv(Channel.TELEMETRY, timeout=5.0) == a_telemetry()
        assert out.stats()[Channel.TELEMETRY].send_rejected == 0
    finally:
        sender.close()
        receiver.close()


def test_a_fractional_drop_count_is_refused():
    """A count, so integral; truncating it would make the round trip lie."""
    extensions, payload = a_telemetry().to_wire()
    extensions["dropped"]["camera"] = 2.7
    with pytest.raises(MessageError, match="dropped.camera"):
        decode_message(Channel.TELEMETRY, extensions, payload)


def test_the_router_refuses_a_reserved_key_when_sending():
    """The earlier test called check_reserved directly, so it passed whether or
    not send called it -- a test named "before sending" that did not test
    sending.

    InvalidMessage, not MessageError: a reserved key is our own bug on the same
    terms as an out-of-range rate, and two exception types for one mistake left
    a caller to catch both.
    """
    from transport.messages import InvalidMessage

    class RogueMessage:
        CHANNEL = Channel.GPS

        def to_wire(self):
            return {CAPTURE_KEY: 1, "heartbeat": True}, b""

    left, _right = loopback_pair()
    session = Session(left, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    try:
        router = MessageRouter(session)
        with pytest.raises(InvalidMessage, match="reserved") as caught:
            router.send(RogueMessage())
        assert caught.value.reason == "reserved_key"
        assert router.stats()[Channel.GPS].rejected_by_reason == {"reserved_key": 1}
    finally:
        session.close()


@pytest.mark.parametrize(
    "message",
    [a_camera_frame(jpeg=b""), a_here_response(body=b"")],
    ids=["camera", "here"],
)
def test_a_blob_channel_accepts_an_empty_payload(message):
    """Acceptance criterion 1 asks for both an empty and a full payload."""
    assert roundtrip(message) == message


def test_drops_are_counted_by_reason():
    """One number cannot answer whether four thousand drops were one bad field
    or four."""
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        missing, payload = an_imu_sample().to_wire()
        del missing["az"]
        sender.send(Channel.IMU, payload, missing)

        wrong_type, payload = an_imu_sample().to_wire()
        wrong_type["ax"] = "not a number"
        sender.send(Channel.IMU, payload, wrong_type)

        out_of_range, payload = a_gps_record(lat=91.0).to_wire()
        sender.send(Channel.GPS, payload, out_of_range)

        assert wait_until(
            lambda: router.recv(Channel.IMU, timeout=0.05) is None
            and router.recv(Channel.GPS, timeout=0.05) is None
            and router.stats()[Channel.IMU].decode_errors == 2
            and router.stats()[Channel.GPS].decode_errors == 1,
            timeout=5.0,
        )
        imu = router.stats()[Channel.IMU]
        assert imu.errors_by_reason == {"missing_field": 1, "wrong_type": 1}
        assert router.stats()[Channel.GPS].errors_by_reason == {"out_of_range": 1}
        assert sum(imu.errors_by_reason.values()) == imu.decode_errors
        json.dumps(router.to_record(), allow_nan=False)
    finally:
        sender.close()
        receiver.close()


def malformed_for(reason):
    """A (channel, extensions, payload) triple that must produce `reason`."""
    if reason == "missing_field":
        extensions, payload = an_imu_sample().to_wire()
        del extensions["az"]
        return Channel.IMU, extensions, payload
    if reason == "wrong_type":
        extensions, payload = an_imu_sample().to_wire()
        extensions["ax"] = "not a number"
        return Channel.IMU, extensions, payload
    if reason == "null_not_allowed":
        extensions, payload = an_imu_sample().to_wire()
        extensions["ax"] = None
        return Channel.IMU, extensions, payload
    if reason == "non_finite":
        extensions, payload = an_imu_sample().to_wire()
        extensions["ax"] = float("inf")
        return Channel.IMU, extensions, payload
    if reason == "out_of_range":
        extensions, payload = a_gps_record(lat=91.0).to_wire()
        return Channel.GPS, extensions, payload
    if reason == "unknown_value":
        extensions, payload = an_advisory().to_wire()
        extensions["units"] = "furlongs"
        return Channel.ADVISORY, extensions, payload
    if reason == "unexpected_payload":
        extensions, _payload = a_gps_record().to_wire()
        return Channel.GPS, extensions, b"unexpected"
    if reason == "no_typed_message":
        # Covered by its own test now: every channel has a typed message, so
        # this reason is unreachable without removing one from the table.
        raise AssertionError("no_typed_message has no live channel; see its own test")
    raise AssertionError(f"no fixture for {reason}")


@pytest.mark.parametrize(
    "reason",
    ["missing_field", "wrong_type", "null_not_allowed", "non_finite", "out_of_range",
     "unknown_value", "unexpected_payload"],
)
def test_each_reason_is_produced_by_the_condition_it_names(reason):
    """Three of nine were pinned, which is round-one coverage on brand-new code.
    `reserved_key` is covered separately on the send path, and
    `no_typed_message` in its own test -- it has no live channel since task 15
    gave `control` a type."""
    channel, extensions, payload = malformed_for(reason)
    with pytest.raises(MessageError) as caught:
        decode_message(channel, extensions, payload)
    assert caught.value.reason == reason, f"got {caught.value.reason!r}"


def test_an_out_of_schema_action_value_is_an_unknown_value_not_a_type_error():
    """Adjacent to `units` in the spec's refusal table, and it was filed under
    wrong_type -- wrong in the one case the counter exists for, since this is
    where the phone and the sim contract can disagree."""
    extensions, payload = an_advisory().to_wire()
    extensions["action"]["merge_mode"] = "sideways"
    with pytest.raises(MessageError) as caught:
        decode_message(Channel.ADVISORY, extensions, payload)
    assert caught.value.reason == "unknown_value"


def test_the_reason_vocabulary_is_closed():
    """The spec calls it closed, so a typo must not reach a summary and split one
    reason into two buckets no reader can reconcile."""
    from transport.messages import REASONS

    with pytest.raises(AssertionError):
        MessageError("x", "nul_not_alowed")
    for reason in REASONS:
        assert MessageError("x", reason).reason == reason


def test_every_reason_the_code_can_emit_is_in_the_vocabulary():
    """The reverse direction of the spec check, which only walked REASONS.

    Walks every construction site rather than only `raise MessageError(...)`:
    binding it first and raising on the next line is the same defect and was
    invisible, and `reason=` as a keyword is correct style that the positional
    count rejected.
    """
    import ast
    import transport.messages as module
    from transport.messages import REASONS

    source = (REPO / "deployment" / "jetson" / "transport" / "messages.py").read_text()
    emitted = set()
    sites = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) not in {"MessageError", "InvalidMessage"}:
            continue
        sites += 1
        if len(node.args) >= 2:
            given = node.args[1]
        else:
            keywords = {kw.arg: kw.value for kw in node.keywords}
            assert "reason" in keywords, (
                f"a MessageError at line {node.lineno} has no explicit reason"
            )
            given = keywords["reason"]
        if isinstance(given, ast.Name):
            emitted.add(given.id)
        elif isinstance(given, ast.Attribute):  # exc.reason, re-raising another
            continue
        else:
            raise AssertionError(f"line {node.lineno}: reason is not a named constant")
    assert sites >= 10, f"only {sites} construction sites found; the walk stopped seeing them"
    resolved = {getattr(module, name) for name in emitted}
    assert resolved <= set(REASONS), f"outside the vocabulary: {resolved - set(REASONS)}"


def test_the_ast_walk_would_catch_a_reason_bound_before_it_is_raised():
    """The test above is only worth its runtime if it sees the shape the old one
    missed, so this hands it that shape directly."""
    import ast

    for source in [
        "err = MessageError('x', 'typo_not_in_vocabulary')\nraise err",
        "raise MessageError('x', reason='typo_not_in_vocabulary')",
        "raise MessageError('x')",
    ]:
        found = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MessageError"
        ]
        assert len(found) == 1, source


def test_every_reason_in_the_vocabulary_has_a_row_in_the_refusal_table():
    """Mentioned somewhere in the prose was the old bar, and two reasons had no
    row at all under a sentence claiming every one did. A row is what a Kotlin
    implementer reads to decide which reason to emit, so a row is the bar."""
    from transport.messages import REASONS

    rows = [
        line
        for line in (REPO / "specs" / "transport_protocol.md").read_text().splitlines()
        if line.startswith("|") and "dropped, counted" in line
    ]
    for reason in REASONS:
        assert any(f"`{reason}`" in row for row in rows), f"{reason} has no refusal row"


def test_the_spec_carries_the_numeric_bounds_the_code_enforces():
    """Both could be halved in the spec with every test still green, which for a
    cross-language contract means the other implementation reads the wrong
    number and its messages are dropped here.

    Checks every interval in the document, not that the right one appears
    somewhere: the rate ceiling is written twice, so falsifying the refusal row
    alone left an "is in the spec" assertion satisfied by the other copy.
    """
    import re

    from transport.messages import MAX_COUNT, MAX_RATE_HZ

    text = (REPO / "specs" / "transport_protocol.md").read_text()
    rates = re.findall(r"\(0, ([0-9.]+)\]", text)
    counts = re.findall(r"\[0, ([0-9]+)\]", text)
    assert rates, "no rate interval in the spec at all"
    assert counts, "no count interval in the spec at all"
    assert set(rates) == {f"{MAX_RATE_HZ:g}"}, f"rate intervals disagree with the code: {rates}"
    assert set(counts) == {str(MAX_COUNT)}, f"count intervals disagree with the code: {counts}"


def test_the_refusal_table_covers_every_documented_channel_object():
    """The three nested objects and the strict action head are the parts a
    reader is most likely to implement additively by accident."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    for phrase in ["nested object", "action", "reserved for the transport"]:
        assert phrase in text, phrase


def test_recv_honours_its_budget_across_skipped_messages():
    """A bad record costs one record, and one record's worth of the caller's
    time budget -- not a fresh copy of it. Malformed messages arriving slower
    than the timeout used to restart it on every skip: 6.9x measured."""
    import time as clock

    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    stop = threading.Event()

    def dribble():
        while not stop.is_set():
            extensions, payload = an_imu_sample().to_wire()
            del extensions["az"]
            sender.send(Channel.IMU, payload, extensions)
            stop.wait(0.02)

    dribbler = threading.Thread(target=dribble, daemon=True)
    dribbler.start()
    try:
        started = clock.monotonic()
        assert router.recv(Channel.IMU, timeout=0.25) is None
        elapsed = clock.monotonic() - started
        assert elapsed < 0.75, f"a 0.25s budget took {elapsed:.2f}s"
        assert router.stats()[Channel.IMU].decode_errors > 0
    finally:
        stop.set()
        dribbler.join(timeout=3.0)
        sender.close()
        receiver.close()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"timeout": 0.0}, {"timeout": -1.0}, {"timeout": 1e-9}, {"timeout": 5.0},
     {"timeout": None}],
    ids=["default", "zero", "negative", "tiny", "generous", "none"],
)
def test_recv_polls_a_queued_message_across_the_whole_argument_domain(kwargs):
    """The domain enumerated rather than sampled, because the default was the
    broken one: an expired budget returned before the queue was looked at, so
    router.recv(channel) -- the poll idiom a control loop uses -- delivered
    nothing while messages sat waiting, with delivered=0 and decode_errors=0 as
    the only evidence."""
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        MessageRouter(sender).send(an_imu_sample(accuracy=2))
        assert wait_until(lambda: receiver.pending(Channel.IMU) == 1, timeout=5.0)
        message = router.recv(Channel.IMU, **kwargs)
        assert message is not None, f"a queued message was not delivered for {kwargs}"
        assert message.accuracy == 2
        assert router.stats()[Channel.IMU].delivered == 1
    finally:
        sender.close()
        receiver.close()


def test_recv_skips_a_bad_message_to_reach_a_good_one_on_an_exhausted_budget():
    """Both already queued, and a budget too small to survive a skip. The
    message must not be lost just because the clock ran out mid-skip."""
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        bad, payload = an_imu_sample().to_wire()
        del bad["az"]
        sender.send(Channel.IMU, payload, bad)
        MessageRouter(sender).send(an_imu_sample(accuracy=7))
        assert wait_until(lambda: receiver.pending(Channel.IMU) == 2, timeout=5.0)
        message = router.recv(Channel.IMU, timeout=1e-9)
        assert message is not None, "the good message was lost to an expired budget"
        assert message.accuracy == 7
        assert router.stats()[Channel.IMU].decode_errors == 1
    finally:
        sender.close()
        receiver.close()


def test_recv_returns_none_on_an_empty_queue_without_blocking():
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    import time as clock

    try:
        started = clock.monotonic()
        assert router.recv(Channel.IMU) is None
        # Tight on purpose. At 0.5 s this passed with a 400 ms floor in place,
        # which is not what "without blocking" means; measured, the poll costs
        # 0.5 us at the median and 32 us at the worst of 2000, so 10 ms is still
        # two orders of margin over the behaviour and kills a floor of 38 ms.
        assert clock.monotonic() - started < 0.01
    finally:
        sender.close()
        receiver.close()


def test_the_router_uses_the_injected_clock_for_its_budget():
    """One deadline, one clock. Using time.monotonic() while Session took an
    injectable one put the router's budget on real time and the session's on the
    fake, so a test injecting a clock would have measured nothing.

    A *frozen* clock cannot show that: the budget never expires under either
    one, so the assertion held with mono_clock ignored. This clock jumps a
    minute per call, which separates them by the whole timeout.
    """
    import time as clock

    sender, receiver = quiet_pair()
    calls = {"n": 0}

    def jumping_clock():
        calls["n"] += 1
        return calls["n"] * 60_000_000_000

    router = MessageRouter(receiver, mono_clock=jumping_clock)
    try:
        # A generous budget on a clock that has already blown through it. The
        # router must return at once; on time.monotonic() this call waits the
        # full twenty seconds, so the bound below is not a timing guess.
        started = clock.monotonic()
        assert router.recv(Channel.IMU, timeout=20.0) is None
        assert clock.monotonic() - started < 1.0, "the budget was measured on the wrong clock"
        assert calls["n"] >= 2, "the injected clock was never consulted"

        # And an expired budget still polls the queue once, so a message that is
        # already waiting comes back rather than being skipped.
        MessageRouter(sender).send(an_imu_sample(accuracy=4))
        assert wait_until(lambda: receiver.pending(Channel.IMU) == 1, timeout=5.0)
        message = router.recv(Channel.IMU, timeout=20.0)
        assert message is not None and message.accuracy == 4
    finally:
        sender.close()
        receiver.close()


def test_recv_with_no_timeout_still_returns_a_message_after_skips():
    sender, receiver = quiet_pair()
    router = MessageRouter(receiver)
    try:
        bad, payload = an_imu_sample().to_wire()
        del bad["az"]
        sender.send(Channel.IMU, payload, bad)
        MessageRouter(sender).send(an_imu_sample(accuracy=1))
        # Bounded on a worker: timeout=None blocks until a decodable message
        # arrives, so a regression here would hang the suite rather than fail
        # it, and there is no pytest-timeout installed to catch that.
        result: list[object] = []
        worker = threading.Thread(
            target=lambda: result.append(router.recv(Channel.IMU, timeout=None)),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "recv(timeout=None) never returned"
        assert result and result[0] is not None and result[0].accuracy == 1
        assert router.stats()[Channel.IMU].decode_errors == 1
    finally:
        sender.close()
        receiver.close()
