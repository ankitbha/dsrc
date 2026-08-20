"""Unit and sanity tests for Session: priority, overflow, counters, lifetime.

Several tests deliberately stall the writer by giving the loopback pair a tiny
buffer and not reading from it. That is the only way to observe queueing at
all: on an unbounded in-memory pipe the writer drains instantly and no policy
ever fires.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import pytest

from transport.channels import Channel, policy_for
from transport.connection import ConnectionClosed
from transport.frames import (
    HEARTBEAT_KEY,
    MAX_PAYLOAD_BYTES,
    Frame,
    FramingError,
    encode,
    read_frame,
)
from transport.loopback import loopback_pair
from transport.session import (
    DEFAULT_HEARTBEAT_S,
    RX_CHUNK_BYTES,
    DEFAULT_STALL_TIMEOUT_S,
    Session,
    SessionEndReason,
)

SMALL_BUFFER = 64


def wait_until(predicate, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@contextlib.contextmanager
def captured_thread_exceptions():
    """Collect exceptions that escape session threads.

    Asserting on these is what pins the `raise` in each loop's catch-all. The
    session ends and the counters are right either way, so re-raising is only
    observable here -- and without it a MemoryError on a large write and a
    backend wrapper bug both present as a bare transport_error.
    """
    escaped: list[BaseException] = []
    previous = threading.excepthook
    threading.excepthook = lambda args: escaped.append(args.exc_value)
    try:
        yield escaped
    finally:
        threading.excepthook = previous


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


def drain_channels(connection, count, timeout=6.0):
    """Read `count` frames off the wire, failing rather than hanging.

    read_frame blocks, so a deadline checked between frames is decorative: once
    a read blocks, the loop never re-evaluates it. A regression that stopped
    the writer used to hang four tests forever, and a hung suite reports
    nothing at all. The reads run on a worker with a join deadline instead.
    """
    frames = []
    errors = []

    def reader():
        try:
            while len(frames) < count:
                frames.append(read_frame(connection.recv_exact))
        except Exception as exc:  # recorded so the assertion can name it
            errors.append(exc)

    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    worker.join(timeout)
    assert len(frames) == count, (
        f"read {len(frames)} of {count} frames in {timeout}s"
        + (f"; last error {errors[-1]!r}" if errors else "")
    )
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
        assert drain_channels(far, 1)[0].seq == 1
    finally:
        session.close()


def test_data_channels_start_at_zero_and_increment():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for _ in range(3):
            session.send(Channel.GPS, b"f")
        assert [frame.seq for frame in drain_channels(far, 3)] == [0, 1, 2]
    finally:
        session.close()


def test_sequence_numbers_are_independent_per_channel():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        session.send(Channel.GPS, b"a")
        session.send(Channel.IMU, b"b")
        session.send(Channel.GPS, b"c")
        frames = drain_channels(far, 3)
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
    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        with pytest.raises(FramingError):
            session.send(Channel.CAMERA, b"\x00" * (MAX_PAYLOAD_BYTES + 1))
        assert not session.is_closed
    finally:
        session.close()


def test_an_extension_shadowing_a_reserved_key_raises_in_send():
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
        frame = drain_channels(far, 1)[0]
        assert frame.channel is Channel.CONTROL
        assert frame.extensions[HEARTBEAT_KEY] is True
        assert frame.payload == b""
    finally:
        session.close()


# -- reserved extensions belong to the transport ----------------------------


@pytest.mark.parametrize("key", ["hello", "heartbeat"])
def test_send_refuses_a_reserved_extension(key):
    """A caller message carrying one of these would be read as transport
    traffic by the peer and consumed instead of delivered -- lost with no drop
    counted and no sequence gap, so invisible in the session summary too. Task
    14 defines message fields on top of this and could pick either name."""
    near, _ = loopback_pair()
    session = quiet_session(near)
    try:
        with pytest.raises(FramingError, match="reserved"):
            session.send(Channel.GPS, b"x", {key: True})
    finally:
        session.close()


def test_a_data_channel_message_carrying_heartbeat_is_still_delivered():
    """Belt and braces for the receive side: if a peer ever does send one, it
    is a caller's message on a data channel and must arrive."""
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        far.send_all(
            encode(
                Frame(
                    channel=Channel.GPS,
                    seq=0,
                    t_mono_ns=1,
                    t_wall_ns=2,
                    payload=b"realdata",
                    extensions={HEARTBEAT_KEY: True},
                )
            )
        )
        message = session.recv(Channel.GPS, timeout=2.0)
        assert message is not None
        assert message.payload == b"realdata"
        stats = session.stats().channels[Channel.GPS]
        assert (stats.received, stats.delivered, stats.dropped_inbound) == (1, 1, 0)
        assert session.stats().heartbeats_received == 0
    finally:
        session.close()


def test_a_control_channel_heartbeat_is_still_consumed():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        far.send_all(
            encode(
                Frame(
                    channel=Channel.CONTROL,
                    seq=1,
                    t_mono_ns=1,
                    t_wall_ns=2,
                    extensions={HEARTBEAT_KEY: True},
                )
            )
        )
        assert wait_until(lambda: session.stats().heartbeats_received == 1)
        assert session.pending(Channel.CONTROL) == 0
    finally:
        session.close()


# -- receive order and inbound overflow at depth > 1 ------------------------


def test_recv_returns_the_oldest_queued_message():
    """Every other test drains after one message per channel, where FIFO and
    LIFO are the same thing."""
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in range(5):
            far.send_all(
                encode(
                    Frame(
                        channel=Channel.GPS,
                        seq=seq,
                        t_mono_ns=1,
                        t_wall_ns=2,
                        payload=f"fix{seq}".encode(),
                    )
                )
            )
        assert wait_until(lambda: session.pending(Channel.GPS) == 5)
        drained = [session.recv(Channel.GPS).payload for _ in range(5)]
        assert drained == [b"fix0", b"fix1", b"fix2", b"fix3", b"fix4"]
    finally:
        session.close()


def test_a_reliable_inbound_queue_drops_the_oldest_at_its_bound():
    depth = policy_for(Channel.GPS).depth
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in range(depth + 3):
            far.send_all(
                encode(
                    Frame(
                        channel=Channel.GPS,
                        seq=seq,
                        t_mono_ns=1,
                        t_wall_ns=2,
                        payload=f"{seq}".encode(),
                    )
                )
            )
        assert wait_until(
            lambda: session.stats().channels[Channel.GPS].received == depth + 3, timeout=5.0
        )
        stats = session.stats().channels[Channel.GPS]
        assert stats.dropped_inbound == 3
        assert session.pending(Channel.GPS) == depth
        # The three oldest went, so the queue starts at 3 and ends at the newest.
        first = session.recv(Channel.GPS).payload
        assert first == b"3"
        remaining = [session.recv(Channel.GPS).payload for _ in range(depth - 1)]
        assert remaining[-1] == f"{depth + 2}".encode()
    finally:
        session.close()


def test_inbound_high_water_is_the_peak_not_the_current_depth():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in range(5):
            far.send_all(
                encode(Frame(channel=Channel.GPS, seq=seq, t_mono_ns=1, t_wall_ns=2, payload=b"f"))
            )
        assert wait_until(lambda: session.pending(Channel.GPS) == 5)
        for _ in range(5):
            session.recv(Channel.GPS)
        far.send_all(
            encode(Frame(channel=Channel.GPS, seq=5, t_mono_ns=1, t_wall_ns=2, payload=b"f"))
        )
        assert wait_until(lambda: session.stats().channels[Channel.GPS].received == 6)
        assert session.stats().channels[Channel.GPS].inbound_high_water == 5
    finally:
        session.close()


def test_outbound_high_water_is_the_peak_not_the_current_depth():
    session, far = stalled_writer_session()
    try:
        for _ in range(5):
            session.send(Channel.HERE, b"x")
        assert session.stats().channels[Channel.HERE].outbound_high_water == 5
        drain_channels(far, 6)  # the blocking camera frame plus the five
        assert wait_until(lambda: session.outbound_pending(Channel.HERE) == 0)
        session.send(Channel.HERE, b"x")
        assert session.stats().channels[Channel.HERE].outbound_high_water == 5
    finally:
        session.close()


def test_a_repeated_or_reordered_sequence_number_is_not_counted_as_a_gap():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for seq in (0, 1, 1, 0, 2):
            far.send_all(
                encode(Frame(channel=Channel.GPS, seq=seq, t_mono_ns=1, t_wall_ns=2, payload=b"f"))
            )
        assert wait_until(lambda: session.stats().channels[Channel.GPS].received == 5)
        stats = session.stats().channels[Channel.GPS]
        assert stats.seq_gaps == 0
        assert stats.missing_seqs == 0
    finally:
        session.close()


# -- counters -------------------------------------------------------------


def test_bytes_sent_counts_whole_frames_not_payloads():
    near, far = loopback_pair()
    session = quiet_session(near)
    try:
        for _ in range(4):
            session.send(Channel.GPS, b"fix")
        frames = drain_channels(far, 4)
        on_the_wire = sum(len(encode(frame)) for frame in frames)
        assert wait_until(lambda: session.stats().channels[Channel.GPS].sent == 4)
        stats = session.stats().channels[Channel.GPS]
        assert stats.bytes_sent == on_the_wire
        assert stats.bytes_sent > 4 * len(b"fix")
    finally:
        session.close()


def test_everything_queued_is_sent_dropped_or_abandoned():
    """The identity has to close from the counters alone. Recovering a loss as
    queued - sent - dropped is derivation by subtraction, which is how a
    counting bug hides."""
    session, _far = stalled_writer_session()
    for index in range(10):
        session.send(Channel.HERE, f"{index}".encode())
    session.close()

    for channel, stats in session.stats().channels.items():
        assert stats.queued == (
            stats.sent + stats.dropped_outbound + stats.abandoned_outbound
        ), f"{channel.value}: {stats.to_record()}"
        assert session.outbound_pending(channel) == 0

    here = session.stats().channels[Channel.HERE]
    assert here.abandoned_outbound > 0
    assert "abandoned_outbound" in here.to_record()
    # The frame the writer had already taken is a camera frame in this fixture.
    # Asserting only on HERE left that half of the accounting free.
    camera = session.stats().channels[Channel.CAMERA]
    assert camera.queued == 1
    assert camera.sent + camera.abandoned_outbound == 1


def test_a_clean_run_abandons_nothing():
    left, right = loopback_pair()
    sender = quiet_session(left, session_id=1)
    receiver = quiet_session(right, session_id=2)
    try:
        for _ in range(10):
            sender.send(Channel.GPS, b"fix")
        assert wait_until(lambda: receiver.stats().channels[Channel.GPS].received == 10)
    finally:
        sender.close()
        receiver.close()
    stats = sender.stats().channels[Channel.GPS]
    assert (stats.queued, stats.sent, stats.abandoned_outbound) == (10, 10, 0)


# -- what the stall timer actually watches ----------------------------------


def test_the_stall_timer_watches_arrivals_not_our_own_transmissions():
    """The case the timer exists for: half-open, we transmit fine, nothing
    comes back. Every other timer test either disables our heartbeat or has
    both sides beating, so both clocks move together and the two are
    indistinguishable."""
    near, far = loopback_pair()
    drain = threading.Event()

    def keep_reading():
        while not drain.is_set():
            try:
                far.recv_exact(1)
            except Exception:
                return

    reader = threading.Thread(target=keep_reading, daemon=True)
    reader.start()
    session = Session(near, session_id=1, heartbeat_s=0.05, stall_timeout_s=0.3).start()
    try:
        assert wait_until(lambda: session.is_closed, timeout=3.0)
        assert session.end_reason is SessionEndReason.STALLED
        assert session.stats().heartbeats_sent > 0  # we were talking the whole time
    finally:
        drain.set()
        session.close()


def test_a_slow_but_continuous_frame_is_not_declared_stalled():
    """A frame larger than the link can deliver inside the timeout must not
    kill the session -- it would reconnect, re-send, and never recover."""
    near, far = loopback_pair()
    session = Session(
        near, session_id=1, heartbeat_s=None, stall_timeout_s=0.4, rx_chunk_bytes=512
    ).start()
    payload = bytes(range(256)) * 80  # 20 KiB at 4 KB/s: five times the timeout
    blob = encode(Frame(channel=Channel.CAMERA, seq=0, t_mono_ns=1, t_wall_ns=2, payload=payload))

    def dribble():
        for offset in range(0, len(blob), 200):
            if session.is_closed:
                return
            try:
                far.send_all(blob[offset : offset + 200])
            except Exception:
                return
            time.sleep(0.05)

    threading.Thread(target=dribble, daemon=True).start()
    try:
        message = session.recv(Channel.CAMERA, timeout=20.0)
        assert message is not None, f"session died: {session.end_reason}"
        assert message.payload == payload
        assert not session.is_closed
    finally:
        session.close()


def test_a_chunk_slower_than_the_timeout_still_stalls():
    """The floor is real: progress has to arrive at some rate."""
    near, far = loopback_pair()
    session = Session(
        near, session_id=1, heartbeat_s=None, stall_timeout_s=0.3, rx_chunk_bytes=8192
    ).start()
    blob = encode(
        Frame(channel=Channel.CAMERA, seq=0, t_mono_ns=1, t_wall_ns=2, payload=b"\x5a" * 60_000)
    )

    def dribble():
        for offset in range(0, len(blob), 200):
            if session.is_closed:
                return
            try:
                far.send_all(blob[offset : offset + 200])
            except Exception:
                return
            time.sleep(0.05)

    threading.Thread(target=dribble, daemon=True).start()
    try:
        assert wait_until(lambda: session.is_closed, timeout=5.0)
        assert session.end_reason is SessionEndReason.STALLED
    finally:
        session.close()


# -- the confirmed timing parameters ----------------------------------------


def test_the_shipped_timing_defaults_match_the_spec():
    """Both numbers are in the cross-language contract, and the channel table
    gets a spec cross-check while these got nothing."""
    spec = (Path(__file__).resolve().parents[3] / "specs" / "transport_protocol.md").read_text()
    assert f"**{DEFAULT_HEARTBEAT_S:.1f} s**" in spec
    assert f"**{DEFAULT_STALL_TIMEOUT_S:.1f} s**" in spec
    assert DEFAULT_HEARTBEAT_S == 1.0
    assert DEFAULT_STALL_TIMEOUT_S == 5.0


def test_keepalives_go_out_at_the_interval_not_once_per_tick():
    """The timer wakes several times per interval. Sending on every wake would
    quietly multiply the keepalive rate on a metered link."""
    near, far = loopback_pair()
    drain = threading.Event()

    def keep_reading():
        while not drain.is_set():
            try:
                far.recv_exact(1)
            except Exception:
                return

    threading.Thread(target=keep_reading, daemon=True).start()
    session = Session(near, session_id=1, heartbeat_s=0.3, stall_timeout_s=None).start()
    try:
        time.sleep(1.05)
        sent = session.stats().heartbeats_sent
        assert 2 <= sent <= 6, f"{sent} keepalives in 1.05s at a 0.3s interval"
    finally:
        drain.set()
        session.close()


# -- signalling, not polling ------------------------------------------------


def test_a_message_is_delivered_far_faster_than_the_polling_tick():
    """Every wait in the session has a timeout, so a lost wakeup degrades to
    latency instead of a hang and no functional assertion can see it. This
    pins the signalling by measuring."""
    left, right = loopback_pair()
    sender = Session(left, session_id=1, heartbeat_s=1.0, stall_timeout_s=5.0).start()
    receiver = Session(right, session_id=2, heartbeat_s=1.0, stall_timeout_s=5.0).start()
    try:
        worst = 0.0
        for index in range(20):
            started = time.monotonic()
            sender.send(Channel.GPS, f"{index}".encode())
            message = receiver.recv(Channel.GPS, timeout=2.0)
            assert message is not None
            worst = max(worst, time.monotonic() - started)
        # The tick is 0.25s at these settings; a dropped notify would show up
        # as roughly a tick per hop.
        assert worst < 0.1, f"worst one-way delivery {worst * 1000:.0f} ms"
    finally:
        sender.close()
        receiver.close()


def test_no_session_thread_is_alive_the_moment_close_returns():
    """The acceptance criterion is that no thread outlives close(), which is
    stronger than eventually."""
    near, _ = loopback_pair()
    session = quiet_session(near, session_id=91)
    session.close()
    alive = [thread.name for thread in threading.enumerate() if "session91" in thread.name]
    assert alive == []


# -- who accounts for the frame in flight -----------------------------------


class HookedConnection:
    """Wraps a connection and runs a hook once, after a write completes.

    Turns a race into a schedule: the hook stands in for a shutdown landing in
    the window between send_all returning and the writer recording the
    outcome, which is what the timer does on a stall and the reader on a
    hangup.
    """

    def __init__(self, inner, hook) -> None:
        self._inner = inner
        self._hook = hook
        self._fired = False

    @property
    def peer(self) -> str:
        return self._inner.peer

    def send_all(self, data: bytes) -> None:
        self._inner.send_all(data)
        if not self._fired:
            self._fired = True
            self._hook()

    def recv_exact(self, n: int) -> bytes:
        return self._inner.recv_exact(n)

    def close(self) -> None:
        self._inner.close()


def test_a_frame_the_peer_received_is_counted_sent_not_abandoned():
    """The in-flight frame belongs to the writer, because it is the only thread
    that knows whether the write completed. Splitting that decision with
    shutdown and reconciling afterwards reported delivered frames as losses."""
    near, far = loopback_pair()
    holder: dict[str, Session] = {}

    def hook():
        closer = threading.Thread(
            target=lambda: holder["session"].close(SessionEndReason.STALLED), daemon=True
        )
        closer.start()
        closer.join(timeout=3.0)

    session = Session(
        HookedConnection(near, hook), session_id=1, heartbeat_s=None, stall_timeout_s=None
    )
    holder["session"] = session
    session.start()
    session.send(Channel.GPS, b"this-frame-reached-the-peer")
    assert drain_channels(far, 1)[0].payload == b"this-frame-reached-the-peer"

    def recorded():
        stats = session.stats().channels[Channel.GPS]
        return stats.sent + stats.abandoned_outbound >= 1

    assert wait_until(recorded, timeout=8.0)
    stats = session.stats().channels[Channel.GPS]
    assert stats.sent == 1, stats.to_record()
    assert stats.abandoned_outbound == 0, stats.to_record()
    assert stats.bytes_sent > 0
    assert stats.queued == stats.sent + stats.dropped_outbound + stats.abandoned_outbound


def test_sending_while_closing_never_orphans_a_message():
    """The deployment shape: sensor threads keep calling send() while the
    reader notices the phone hung up. A closed check outside the queue lock
    lets a message land after shutdown drained -- counted in `queued`, in
    nothing else, and never sent."""
    for _ in range(8):
        near, _far = loopback_pair(max_buffer_bytes=4096)
        session = quiet_session(near)
        stop = threading.Event()

        def spam(target=session):
            while not stop.is_set():
                try:
                    if not target.send(Channel.HERE, b"\x00" * 8192):
                        return
                except Exception:
                    return

        writer = threading.Thread(target=spam, daemon=True)
        writer.start()
        time.sleep(0.05)
        session.close()
        stop.set()
        writer.join(timeout=3.0)

        stats = session.stats().channels[Channel.HERE]
        assert stats.queued == (
            stats.sent + stats.dropped_outbound + stats.abandoned_outbound
        ), stats.to_record()
        assert session.outbound_pending(Channel.HERE) == 0


def test_a_backend_that_returns_empty_at_eof_ends_the_session():
    """recv_exact must raise, but returning b"" at EOF is the likeliest way to
    get it wrong -- it is what socket.recv does, and tasks 13 and 40 each write
    one. Treating an empty chunk as progress spun the reader forever while
    refreshing the very clock meant to notice."""

    class EmptyAtEof:
        peer = "empty-at-eof"

        def __init__(self, inner):
            self._inner = inner
            self.calls = 0

        def send_all(self, data):
            self._inner.send_all(data)

        def recv_exact(self, n):
            self.calls += 1
            try:
                return self._inner.recv_exact(n)
            except ConnectionClosed:
                return b""

        def close(self):
            self._inner.close()

    for payload_len in (1024, RX_CHUNK_BYTES * 4):
        near, far = loopback_pair()
        connection = EmptyAtEof(near)
        session = Session(connection, session_id=1, heartbeat_s=None, stall_timeout_s=0.3).start()
        try:
            blob = encode(
                Frame(
                    channel=Channel.CAMERA,
                    seq=0,
                    t_mono_ns=1,
                    t_wall_ns=2,
                    payload=b"\x5a" * payload_len,
                )
            )
            far.send_all(blob[:200])
            far.close()
            assert wait_until(lambda: session.is_closed, timeout=4.0), (
                f"payload {payload_len}: never ended after {connection.calls} calls"
            )
            assert connection.calls < 500, f"spun: {connection.calls} calls"
        finally:
            session.close()


def test_recv_wakes_promptly_when_the_session_ends():
    """A blocked reader is woken by shutdown rather than waiting out its tick."""
    near, _far = loopback_pair()
    session = Session(near, session_id=1, heartbeat_s=1.0, stall_timeout_s=5.0).start()
    woke: list[float] = []

    def wait_for_a_message():
        started = time.monotonic()
        session.recv(Channel.GPS, timeout=None)
        woke.append(time.monotonic() - started)

    waiter = threading.Thread(target=wait_for_a_message, daemon=True)
    waiter.start()
    time.sleep(0.1)
    session.close()
    waiter.join(timeout=3.0)
    assert woke, "recv never returned"
    assert woke[0] < 0.15, f"took {woke[0] * 1000:.0f} ms; the tick is 250 ms"


def test_a_peer_sending_one_small_frame_per_timeout_holds_the_session():
    """The floor the code actually enforces, rather than a bytes-per-second
    figure. The stamp lands per completed read and a frame needs at least two,
    so very little traffic keeps a session alive. That trade is accepted --
    displacement lets a healthy phone take the slot back -- but the spec has to
    state this floor and not a chunk-sized one."""
    near, far = loopback_pair()
    stall_s = 0.4
    session = Session(near, session_id=1, heartbeat_s=None, stall_timeout_s=stall_s).start()
    try:
        delivered = 0
        for index in range(6):
            time.sleep(stall_s * 0.8)
            if session.is_closed:
                break
            far.send_all(
                encode(
                    Frame(
                        channel=Channel.TELEMETRY,
                        seq=index,
                        t_mono_ns=1,
                        t_wall_ns=2,
                        payload=b"hi",
                    )
                )
            )
            delivered += 1
        assert not session.is_closed, f"closed after {delivered} frames"
        assert delivered == 6
    finally:
        session.close()


def test_start_declines_to_run_threads_on_an_already_closed_session():
    """A consumer can close a session the moment it is announced, which on the
    accept path is before start(). Starting anyway leaves threads behind a
    close() that already returned."""
    near, _ = loopback_pair()
    session = Session(near, session_id=93, heartbeat_s=None, stall_timeout_s=None)
    session.close()
    session.start()
    # Asserted on the thread list itself, not on threading.enumerate(): threads
    # started on a closed session exit within a tick on their own, so an
    # enumerate() check races their exit and passes either way.
    assert session._threads == []
    assert [t.name for t in threading.enumerate() if "session93" in t.name] == []
    assert session.end_reason is SessionEndReason.CLOSED_LOCAL


# -- a thread that dies must take the session with it -----------------------


class FailingWriteConnection:
    """A connection whose writes raise whatever it is handed."""

    peer = "failing-write"

    def __init__(self, inner, error) -> None:
        self._inner = inner
        self._error = error
        self.writes = 0

    def send_all(self, data: bytes) -> None:
        self.writes += 1
        raise self._error

    def recv_exact(self, n: int) -> bytes:
        return self._inner.recv_exact(n)

    def close(self) -> None:
        self._inner.close()


@pytest.mark.parametrize(
    "error",
    [RuntimeError("a wrapper bug"), MemoryError()],
    ids=["RuntimeError", "MemoryError"],
)
def test_an_unexpected_write_failure_ends_the_session(error):
    """Without a catch-all the writer thread dies and the session goes on
    claiming to be healthy: send() keeps returning True, the queues fill and
    begin counting drops for messages nobody ever attempted, and no
    SessionEnded is emitted, so the summary shows a session that transmitted
    nothing. A MemoryError on a 4 MiB write is the plausible route."""
    near, _far = loopback_pair()
    connection = FailingWriteConnection(near, error)
    session = Session(connection, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    try:
        with captured_thread_exceptions() as escaped:
            session.send(Channel.GPS, b"payload")
            assert wait_until(lambda: session.is_closed, timeout=3.0)
            # Waited on the exception, not just the flag: _shutdown runs before
            # the raise propagates, so the block can exit first and the
            # traceback then lands outside the capture.
            assert wait_until(lambda: bool(escaped), timeout=3.0)
        assert [type(exc) for exc in escaped] == [type(error)], escaped
        assert session.end_reason is SessionEndReason.TRANSPORT_ERROR
        stats = session.stats().channels[Channel.GPS]
        assert stats.abandoned_outbound == 1, stats.to_record()
        assert stats.queued == stats.sent + stats.dropped_outbound + stats.abandoned_outbound
        assert session.send(Channel.GPS, b"more") is False
    finally:
        session.close()


def test_a_reset_connection_is_accounted_for():
    """The OSError branch, which is how a real TCP write fails and which the
    loopback backend can never reach on its own."""
    near, _far = loopback_pair()
    connection = FailingWriteConnection(near, ConnectionResetError(104, "reset by peer"))
    session = Session(connection, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    try:
        with captured_thread_exceptions() as escaped:
            session.send(Channel.HERE, b"response")
            assert wait_until(lambda: session.is_closed, timeout=3.0)
        # A reset is an OSError, so it is an expected failure with its own
        # handler: the session ends and the frame is accounted for, and no
        # traceback escapes. Only the unforeseen kinds re-raise.
        assert escaped == [], escaped
        assert session.end_reason is SessionEndReason.TRANSPORT_ERROR
        stats = session.stats().channels[Channel.HERE]
        assert stats.abandoned_outbound == 1, stats.to_record()
        assert stats.queued == stats.sent + stats.dropped_outbound + stats.abandoned_outbound
    finally:
        session.close()


def test_start_is_idempotent():
    """The session is announced to consumers before its threads exist, so a
    consumer may reasonably call start() on one that is not running. A second
    set of threads puts two readers on one byte stream, and they split each
    other's frames -- reported as a framing error, indistinguishable from a
    corrupt link or a bad encoder on the phone."""
    near, far = loopback_pair()
    session = Session(near, session_id=900, heartbeat_s=None, stall_timeout_s=None)
    try:
        session.start()
        session.start()
        session.start()
        assert len(session._threads) == 3
        names = sorted(thread.name for thread in threading.enumerate() if "session900" in thread.name)
        assert names == ["session900-rx", "session900-timer", "session900-tx"], names

        # And one reader still parses the stream correctly.
        for seq in range(6):
            far.send_all(
                encode(
                    Frame(
                        channel=Channel.TELEMETRY,
                        seq=seq,
                        t_mono_ns=1,
                        t_wall_ns=2,
                        payload=b"payload-bytes",
                    )
                )
            )
        assert wait_until(
            lambda: session.stats().channels[Channel.TELEMETRY].received == 6, timeout=3.0
        )
        assert not session.is_closed
    finally:
        session.close()


class FailingReadConnection:
    """A connection whose reads raise something the reader does not expect."""

    peer = "failing-read"

    def __init__(self, inner, error) -> None:
        self._inner = inner
        self._error = error

    def send_all(self, data: bytes) -> None:
        self._inner.send_all(data)

    def recv_exact(self, n: int) -> bytes:
        raise self._error

    def close(self) -> None:
        self._inner.close()


def test_an_unexpected_read_failure_ends_the_session():
    """A dead reader would eventually be noticed by the peer's own stall timer,
    but only after the peer gives up on us -- a whole timeout of a drive spent
    recording nothing. End it here instead."""
    near, _far = loopback_pair()
    session = Session(
        FailingReadConnection(near, RuntimeError("a decoder wrapper bug")),
        session_id=1,
        heartbeat_s=None,
        stall_timeout_s=None,
    ).start()
    try:
        assert wait_until(lambda: session.is_closed, timeout=3.0)
        assert session.end_reason is SessionEndReason.TRANSPORT_ERROR
    finally:
        session.close()


def test_an_unexpected_timer_failure_ends_the_session():
    """The timer carries the keepalive and the stall watchdog together, so
    losing it silently removes the very thing that would have noticed."""
    near, _far = loopback_pair()

    def clock_that_fails_only_in_the_timer():
        if "timer" in threading.current_thread().name:
            raise RuntimeError("clock unavailable")
        return time.monotonic_ns()

    session = Session(
        near,
        session_id=1,
        heartbeat_s=0.05,
        stall_timeout_s=5.0,
        mono_clock=clock_that_fails_only_in_the_timer,
    )
    try:
        with captured_thread_exceptions() as escaped:
            session.start()
            assert wait_until(lambda: session.is_closed, timeout=3.0)
            assert wait_until(lambda: bool(escaped), timeout=3.0)
        assert [type(exc) for exc in escaped] == [RuntimeError], escaped
        assert session.end_reason is SessionEndReason.TRANSPORT_ERROR
    finally:
        session.close()


def test_a_consumer_callback_that_raises_does_not_decide_the_session_s_fate():
    """A raising on_end used to break two things at once: close() propagated
    before reaching join(), so threads outlived a close() that had already
    unwound, and inside a loop's catch-all the callback's exception replaced
    the one that actually killed the thread."""
    near, _far = loopback_pair()

    def raising_on_end(session, reason):
        raise RuntimeError("consumer bug in on_end")

    session = Session(
        near, session_id=41, heartbeat_s=None, stall_timeout_s=None, on_end=raising_on_end
    )
    session.start()
    session.close()
    assert session.end_reason is SessionEndReason.CLOSED_LOCAL
    assert session.on_end_failures == 1
    assert [t.name for t in threading.enumerate() if "session41" in t.name] == []


def test_a_raising_callback_does_not_mask_the_real_failure():
    near, _far = loopback_pair()

    def raising_on_end(session, reason):
        raise RuntimeError("consumer bug in on_end")

    session = Session(
        FailingWriteConnection(near, MemoryError()),
        session_id=42,
        heartbeat_s=None,
        stall_timeout_s=None,
        on_end=raising_on_end,
    )
    try:
        with captured_thread_exceptions() as escaped:
            session.start()
            session.send(Channel.GPS, b"payload")
            assert wait_until(lambda: session.is_closed, timeout=3.0)
            assert wait_until(lambda: bool(escaped), timeout=3.0)
        assert [type(exc) for exc in escaped] == [MemoryError], escaped
        assert session.on_end_failures == 1
        stats = session.stats().channels[Channel.GPS]
        assert stats.queued == stats.sent + stats.dropped_outbound + stats.abandoned_outbound
    finally:
        session.close()
