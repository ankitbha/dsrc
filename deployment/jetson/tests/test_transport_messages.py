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
from transport.frames import FramingError, encode
from transport.loopback import loopback_pair
from transport.messages import (
    ACTION_HEADS,
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


ALL_BUILDERS = [
    a_camera_frame, a_gps_record, an_imu_sample, a_here_response,
    a_telemetry, an_advisory, a_rate_command,
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


def test_control_carries_no_typed_message():
    """It is the transport's own channel: hello and heartbeat."""
    with pytest.raises(MessageError, match="no typed message"):
        decode_message(Channel.CONTROL, {CAPTURE_KEY: 1}, b"")


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
    from policy import sim_contract

    assert ACTION_VALUES == sim_contract.ACTION_VALUES
    assert ACTION_HEADS == sim_contract.ACTION_HEADS


def test_the_action_vocabulary_matches_the_action_schema_spec():
    text = (REPO / "specs" / "action_schema.md").read_text()
    for head, values in ACTION_VALUES.items():
        section = text.split(f"`{head}`", 1)
        assert len(section) == 2, f"{head} is not documented"
        body = section[1].split("`desired", 1)[0].split("## ", 1)[0]
        for value in values:
            assert f"`{value}`" in body, f"{head}: {value} missing from the spec"


def test_the_message_table_matches_the_protocol_spec():
    """The spec is what Kotlin implements, so code and prose drifting apart is a
    real failure -- the same check the channel table already gets."""
    text = (REPO / "specs" / "transport_protocol.md").read_text()
    rows = dict(re.findall(r"^\| `(\w+)` \| (.+?) \|", text, re.M))
    for channel, message_type in MESSAGE_FOR_CHANNEL.items():
        assert channel.value in rows, f"{channel.value} has no message row"
        documented = rows[channel.value]
        # Every field the encoder emits, except the capture stamp, must appear.
        sample = {
            Channel.CAMERA: a_camera_frame, Channel.GPS: a_gps_record,
            Channel.IMU: an_imu_sample, Channel.HERE: a_here_response,
            Channel.TELEMETRY: a_telemetry, Channel.ADVISORY: an_advisory,
            Channel.RATE_CMD: a_rate_command,
        }[channel]()
        extensions, _payload = sample.to_wire()
        for field in extensions:
            if field == CAPTURE_KEY:
                continue
            assert f"`{field}`" in documented, f"{channel.value}: {field} undocumented"


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
    sender, receiver = quiet_pair()
    out_router, in_router = MessageRouter(sender), MessageRouter(receiver)
    try:
        message = build()
        assert out_router.send(message) is True
        assert in_router.recv(message.CHANNEL, timeout=5.0) == message
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
        assert wait_until(lambda: router.recv(Channel.IMU, timeout=0.2) is None)
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
    allowed = {"__future__", "math", "dataclasses", "typing", "transport.channels",
               "transport.frames"}
    assert set(imports) <= allowed, f"unexpected imports: {sorted(set(imports) - allowed)}"
