"""Unit and sanity tests for Session: priority, overflow, counters, lifetime.

Several tests deliberately stall the writer by giving the loopback pair a tiny
buffer and not reading from it. That is the only way to observe queueing at
all: on an unbounded in-memory pipe the writer drains instantly and no policy
ever fires.
"""

from __future__ import annotations

import threading
import time

import pytest

from transport.channels import Channel, policy_for
from transport.frames import HEARTBEAT_KEY, MAX_PAYLOAD_BYTES, Frame, encode, read_frame
from transport.loopback import loopback_pair
from transport.session import Session, SessionEndReason

SMALL_BUFFER = 64


def wait_until(predicate, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def quiet_session(connection, session_id=1, **kwargs):
    """A session with the timer disabled, so ordering tests are not perturbed
    by keepalives arriving in the middle of them."""
    options = {"heartbeat_s": None, "stall_timeout_s": None}
    options.update(kwargs)
    return Session(connection, session_id=session_id, **options).start()


def stalled_writer_session(**kwargs):
    """A session whose writer is blocked inside send_all on a full buffer, with
    a raw connection on the other end for reading at a controlled pace."""
    near, far = loopback_pair(max_buffer_bytes=SMALL_BUFFER)
    session = quiet_session(near, **kwargs)
    session.send(Channel.CAMERA, b"\x00" * 2000)
    assert wait_until(lambda: near.unread_bytes >= SMALL_BUFFER), "writer never blocked"
    return session, far


def drain_channels(connection, count, timeout=3.0):
    """Read `count` frames straight off the wire, returning their channels."""
    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < count and time.monotonic() < deadline:
        frames.append(read_frame(connection.recv_exact))
    return frames


# -- basic traffic -----------------------------------------------------------


def test_a_message_crosses_and_arrives_intact():
    left, right = loopback_pair()
    sender = quiet_session(left, session_id=1)
    receiver = quiet_session(right, session_id=2)
    try:
        assert sender.send(Channel.GPS, b"fix", {"lat": 1.5}) is True
        message = receiver.recv(Channel.GPS, timeout=2.0)
        assert message is not None
        assert message.payload == b"fix"
        assert message.extensions == {"lat": 1.5}
        assert message.channel is Channel.GPS
        assert message.t_recv_mono_ns > 0
    finally:
        sender.close()
        receiver.close()


def test_traffic_flows_both_ways_on_one_connection():
    left, right = loopback_pair()
    phone = quiet_session(left, session_id=1)
    jetson = quiet_session(right, session_id=2)
    try:
        phone.send(Channel.CAMERA, b"jpeg")
        jetson.send(Channel.ADVISORY, b"slow")
        assert jetson.recv(Channel.CAMERA, timeout=2.0).payload == b"jpeg"
        assert phone.recv(Channel.ADVISORY, timeout=2.0).payload == b"slow"
    finally:
        phone.close()
        jetson.close()


def test_five_hundred_messages_arrive_intact_and_in_order():
    """Sustained traffic across four channels loses nothing and stays ordered.

    Sent and drained in batches rather than as one 500-deep burst, because the
    queues are finite on purpose: dumping 500 messages into a depth-16 channel
    nobody is reading is meant to drop, and a test that did it would be
    asserting the policy is broken.
    """
    channels = [Channel.GPS, Channel.IMU, Channel.HERE, Channel.TELEMETRY]
    left, right = loopback_pair()
    sender = quiet_session(left, session_id=1)
    receiver = quiet_session(right, session_id=2)
    try:
        expected = {channel: [] for channel in channels}
        got = {channel: [] for channel in channels}
        for index in range(125):
            for channel in channels:
                payload = f"{channel.value}-{index}".encode()
                expected[channel].append(payload)
                assert sender.send(channel, payload)
            for channel in channels:
                message = receiver.recv(channel, timeout=2.0)
                assert message is not None, (channel, index)
                got[channel].append(message.payload)

        assert got == expected
        assert sum(len(v) for v in got.values()) == 500
        for channel in channels:
            stats = receiver.stats().channels[channel]
            assert stats.dropped_inbound == 0
            assert stats.seq_gaps == 0
            assert sender.stats().channels[channel].dropped_outbound == 0
    finally:
        sender.close()
        receiver.close()


# -- sequence numbers --------------------------------------------------------


def test_control_sequence_continues_past_the_hello():
    """The handshake already spent control seq 0, so the session starts at 1."""
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        session.send(Channel.CONTROL, b"x")
        assert read_frame(far.recv_exact).seq == 1
    finally:
        session.close()


def test_data_channels_start_at_zero_and_increment():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for _ in range(3):
            session.send(Channel.GPS, b"f")
        assert [read_frame(far.recv_exact).seq for _ in range(3)] == [0, 1, 2]
    finally:
        session.close()


def test_sequence_numbers_are_independent_per_channel():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        session.send(Channel.GPS, b"a")
        session.send(Channel.IMU, b"b")
        session.send(Channel.GPS, b"c")
        frames = [read_frame(far.recv_exact) for _ in range(3)]
        by_channel = {}
        for frame in frames:
            by_channel.setdefault(frame.channel, []).append(frame.seq)
        assert by_channel[Channel.GPS] == [0, 1]
        assert by_channel[Channel.IMU] == [0]
    finally:
        session.close()


def test_a_gap_in_received_sequence_is_counted():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in (0, 1, 5):
            far.send_all(
                encode(Frame(channel=Channel.GPS, seq=seq, t_mono_ns=1, t_wall_ns=2, payload=b"f"))
            )
        assert wait_until(lambda: session.stats().channels[Channel.GPS].received == 3)
        stats = session.stats().channels[Channel.GPS]
        assert stats.seq_gaps == 1
        assert stats.missing_seqs == 3  # 2, 3, 4
    finally:
        session.close()


def test_the_first_frame_on_a_channel_is_never_a_gap():
    """Otherwise every fresh session would report a phantom drop."""
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        far.send_all(
            encode(Frame(channel=Channel.GPS, seq=100, t_mono_ns=1, t_wall_ns=2, payload=b"f"))
        )
        assert wait_until(lambda: session.stats().channels[Channel.GPS].received == 1)
        assert session.stats().channels[Channel.GPS].seq_gaps == 0
        assert session.stats().channels[Channel.GPS].missing_seqs == 0
    finally:
        session.close()


# -- priority ----------------------------------------------------------------


def test_a_command_overtakes_a_queued_camera_frame():
    """The documented guarantee: a high-priority message waits only for the
    frame already going out, not for what is queued behind it."""
    session, far = stalled_writer_session()
    try:
        session.send(Channel.CAMERA, b"\x01" * 100)  # fills camera's depth-1 queue
        session.send(Channel.RATE_CMD, b"rate")
        channels = [frame.channel for frame in drain_channels(far, 3)]
        assert channels == [Channel.CAMERA, Channel.RATE_CMD, Channel.CAMERA]
    finally:
        session.close()


def test_a_command_overtakes_a_hundred_deep_sensor_backlog():
    near, far = loopback_pair(max_buffer_bytes=SMALL_BUFFER)
    session = quiet_session(near)
    try:
        session.send(Channel.IMU, b"\x00" * 2000)
        assert wait_until(lambda: near.unread_bytes >= SMALL_BUFFER)
        for _ in range(100):
            session.send(Channel.IMU, b"s")
        session.send(Channel.RATE_CMD, b"rate")

        channels = [frame.channel for frame in drain_channels(far, 102)]
        assert channels[0] is Channel.IMU  # the frame already in flight
        assert channels[1] is Channel.RATE_CMD
        assert channels.count(Channel.RATE_CMD) == 1
    finally:
        session.close()


def test_equal_priority_channels_take_turns():
    """Round-robin inside a tier: no channel drains fully while a peer of the
    same priority waits."""
    session, far = stalled_writer_session()
    normal = [Channel.GPS, Channel.IMU, Channel.HERE, Channel.TELEMETRY]
    try:
        for _ in range(3):
            for channel in normal:
                session.send(channel, b"x")
        frames = drain_channels(far, 13)
        assert frames[0].channel is Channel.CAMERA  # the blocking frame
        rotation = [frame.channel for frame in frames[1:]]
        for start in (0, 4, 8):
            assert set(rotation[start : start + 4]) == set(normal), rotation
    finally:
        session.close()


# -- overflow ----------------------------------------------------------------


def test_latest_wins_keeps_exactly_the_newest_and_counts_the_rest():
    session, far = stalled_writer_session()
    try:
        for index in range(6):
            session.send(Channel.CAMERA, f"frame{index}".encode())
        stats = session.stats().channels[Channel.CAMERA]
        # One frame is in flight; of the six queued behind it, five were
        # displaced and one remains.
        assert session.outbound_pending(Channel.CAMERA) == 1
        assert stats.dropped_outbound == 5
        assert stats.queued == 7  # the blocking frame plus six

        frames = drain_channels(far, 2)
        assert frames[1].payload == b"frame5"
    finally:
        session.close()


def test_a_reliable_channel_drops_nothing_until_its_bound():
    depth = policy_for(Channel.HERE).depth
    session, far = stalled_writer_session()
    try:
        for _ in range(depth):
            session.send(Channel.HERE, b"resp")
        assert session.stats().channels[Channel.HERE].dropped_outbound == 0
        assert session.outbound_pending(Channel.HERE) == depth

        session.send(Channel.HERE, b"one too many")
        assert session.stats().channels[Channel.HERE].dropped_outbound == 1
        assert session.outbound_pending(Channel.HERE) == depth
    finally:
        session.close()


def test_a_reliable_channel_drops_the_oldest():
    depth = policy_for(Channel.HERE).depth
    session, far = stalled_writer_session()
    try:
        for index in range(depth + 1):
            session.send(Channel.HERE, f"{index}".encode())
        payloads = []
        for frame in drain_channels(far, depth + 1):
            if frame.channel is Channel.HERE:
                payloads.append(frame.payload)
        assert payloads[0] == b"1"  # 0 was displaced, not 1
        assert b"0" not in payloads
    finally:
        session.close()


def test_dropped_count_matches_the_messages_that_never_arrived():
    """Counted against the delivered set, not inferred by subtraction."""
    session, far = stalled_writer_session()
    sent = [f"frame{index}".encode() for index in range(6)]
    try:
        for payload in sent:
            session.send(Channel.CAMERA, payload)
        arrived = {frame.payload for frame in drain_channels(far, 2)}
        never_arrived = [payload for payload in sent if payload not in arrived]
        assert len(never_arrived) == session.stats().channels[Channel.CAMERA].dropped_outbound
    finally:
        session.close()


def test_inbound_overflow_is_counted_too():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in range(4):
            far.send_all(
                encode(
                    Frame(
                        channel=Channel.CAMERA,
                        seq=seq,
                        t_mono_ns=1,
                        t_wall_ns=2,
                        payload=f"{seq}".encode(),
                    )
                )
            )
        assert wait_until(lambda: session.stats().channels[Channel.CAMERA].received == 4)
        stats = session.stats().channels[Channel.CAMERA]
        assert stats.dropped_inbound == 3
        assert session.pending(Channel.CAMERA) == 1
        assert session.recv(Channel.CAMERA).payload == b"3"
    finally:
        session.close()


# -- counters ----------------------------------------------------------------


def test_counters_are_exact_over_a_clean_run():
    left, right = loopback_pair()
    sender = quiet_session(left, session_id=1)
    receiver = quiet_session(right, session_id=2)
    try:
        for _ in range(10):
            sender.send(Channel.GPS, b"fix")
        assert wait_until(lambda: sender.stats().channels[Channel.GPS].sent == 10)
        assert wait_until(lambda: receiver.stats().channels[Channel.GPS].received == 10)

        out = sender.stats().channels[Channel.GPS]
        assert (out.queued, out.sent, out.dropped_outbound) == (10, 10, 0)
        assert out.bytes_sent > 0

        inbound = receiver.stats().channels[Channel.GPS]
        assert (inbound.received, inbound.delivered, inbound.dropped_inbound) == (10, 10, 0)
        assert inbound.bytes_received == 30  # three payload bytes each
    finally:
        sender.close()
        receiver.close()


def test_stats_is_a_snapshot_not_a_live_handle():
    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        session.send(Channel.GPS, b"a")
        snapshot = session.stats()
        snapshot.channels[Channel.GPS].queued = 999
        assert session.stats().channels[Channel.GPS].queued == 1
    finally:
        session.close()


def test_stats_record_is_json_shaped():
    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        record = session.stats().to_record()
        assert record["session_id"] == 1
        assert record["end_reason"] is None
        assert set(record["channels"]) == {channel.value for channel in Channel}
    finally:
        session.close()


def test_high_water_marks_record_the_deepest_backlog():
    session, _ = stalled_writer_session()
    try:
        for _ in range(5):
            session.send(Channel.HERE, b"x")
        assert session.stats().channels[Channel.HERE].outbound_high_water == 5
    finally:
        session.close()


# -- caller-side errors ------------------------------------------------------


def test_an_oversize_payload_raises_in_the_callers_thread():
    from transport.frames import FramingError

    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        with pytest.raises(FramingError):
            session.send(Channel.CAMERA, b"\x00" * (MAX_PAYLOAD_BYTES + 1))
        assert not session.is_closed
    finally:
        session.close()


def test_an_extension_shadowing_a_reserved_key_raises_in_send():
    from transport.frames import FramingError

    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        with pytest.raises(FramingError, match="collides"):
            session.send(Channel.GPS, b"x", {"seq": 1})
    finally:
        session.close()


# -- lifetime ----------------------------------------------------------------


def test_close_records_the_reason_and_refuses_further_sends():
    near, _ = loopback_pair()
    session = quiet_session(near)
    session.close()
    assert session.is_closed
    assert session.end_reason is SessionEndReason.CLOSED_LOCAL
    assert session.send(Channel.GPS, b"x") is False


def test_the_first_end_reason_wins():
    near, _ = loopback_pair()
    session = quiet_session(near)
    session.close(SessionEndReason.DISPLACED)
    session.close(SessionEndReason.CLOSED_LOCAL)
    assert session.end_reason is SessionEndReason.DISPLACED


def test_no_thread_outlives_close():
    before = {thread.name for thread in threading.enumerate()}
    near, _ = loopback_pair()
    session = quiet_session(near, session_id=77)
    assert any("session77" in name for name in {t.name for t in threading.enumerate()})
    session.close()
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if "session77" in thread.name and thread.name not in before
        ]
    )


def test_a_peer_hangup_ends_the_session():
    near, far = loopback_pair()
    ended: list[SessionEndReason] = []
    session = quiet_session(near, on_end=lambda _s, reason: ended.append(reason))
    far.close()
    assert wait_until(lambda: session.is_closed)
    assert session.end_reason is SessionEndReason.PEER_CLOSED
    assert ended == [SessionEndReason.PEER_CLOSED]
    session.close()


def test_garbage_on_the_wire_ends_the_session_as_a_framing_error():
    near, far = loopback_pair()
    session = quiet_session(near)
    far.send_all(b"\x00\x00\x00\x00\x00\x02[]")  # valid prefix, header is an array
    assert wait_until(lambda: session.is_closed)
    assert session.end_reason is SessionEndReason.FRAMING_ERROR
    session.close()


def test_a_truncated_frame_is_never_delivered():
    near, far = loopback_pair()
    session = quiet_session(near)
    whole = encode(Frame(channel=Channel.GPS, seq=0, t_mono_ns=1, t_wall_ns=2, payload=b"abcdefgh"))
    far.send_all(whole[:-3])
    far.close()
    assert wait_until(lambda: session.is_closed)
    assert session.pending(Channel.GPS) == 0
    assert session.recv(Channel.GPS) is None
    session.close()


def test_recv_returns_none_on_timeout_and_after_the_session_ends():
    near, far = loopback_pair()
    session = quiet_session(near)
    started = time.monotonic()
    assert session.recv(Channel.GPS, timeout=0.15) is None
    assert time.monotonic() - started >= 0.1
    session.close()
    assert session.recv(Channel.GPS, timeout=None) is None


def test_queued_messages_stay_readable_until_drained():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        far.send_all(
            encode(Frame(channel=Channel.GPS, seq=0, t_mono_ns=1, t_wall_ns=2, payload=b"fix"))
        )
        assert wait_until(lambda: session.pending(Channel.GPS) == 1)
        assert session.recv(Channel.GPS).payload == b"fix"
        assert session.recv(Channel.GPS) is None
    finally:
        session.close()


# -- timer -------------------------------------------------------------------


def test_keepalives_are_exchanged_and_never_delivered_to_a_caller():
    left, right = loopback_pair()
    phone = Session(left, session_id=1, heartbeat_s=0.03, stall_timeout_s=None).start()
    jetson = Session(right, session_id=2, heartbeat_s=0.03, stall_timeout_s=None).start()
    try:
        assert wait_until(lambda: jetson.stats().heartbeats_received >= 2)
        assert phone.stats().heartbeats_sent >= 2
        assert jetson.pending(Channel.CONTROL) == 0
        assert jetson.recv(Channel.CONTROL, timeout=0.05) is None
    finally:
        phone.close()
        jetson.close()


def test_a_silent_peer_ends_the_session_as_stalled():
    near, _far = loopback_pair()
    session = Session(near, session_id=1, heartbeat_s=None, stall_timeout_s=0.2).start()
    try:
        assert wait_until(lambda: session.is_closed, timeout=3.0)
        assert session.end_reason is SessionEndReason.STALLED
    finally:
        session.close()


def test_a_talking_peer_is_not_declared_stalled():
    left, right = loopback_pair()
    phone = Session(left, session_id=1, heartbeat_s=0.05, stall_timeout_s=0.4).start()
    jetson = Session(right, session_id=2, heartbeat_s=0.05, stall_timeout_s=0.4).start()
    try:
        time.sleep(1.0)
        assert not phone.is_closed
        assert not jetson.is_closed
    finally:
        phone.close()
        jetson.close()


def test_heartbeat_frames_carry_the_reserved_extension():
    near, far = loopback_pair()
    session = Session(near, session_id=1, heartbeat_s=0.02, stall_timeout_s=None).start()
    try:
        frame = read_frame(far.recv_exact)
        assert frame.channel is Channel.CONTROL
        assert frame.extensions[HEARTBEAT_KEY] is True
        assert frame.payload == b""
    finally:
        session.close()
