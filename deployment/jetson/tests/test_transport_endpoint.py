"""Sanity tests for the listener: accept, refuse, displace, and end sessions.

The phone opens connections and the Jetson accepts them, so these tests drive
a loopback client through a real handshake against a running listener.
"""

from __future__ import annotations

import threading
import time

import pytest

from transport.channels import Channel
from transport.endpoint import (
    SessionEnded,
    SessionRefused,
    SessionStarted,
    TransportListener,
)
from transport.connection import ConnectionClosed
from transport.frames import PROTOCOL_VERSION, Frame, encode, read_frame
from transport.handshake import Hello, Role, VersionMismatch, perform_handshake
from transport.loopback import LoopbackAcceptor
from transport.session import Session, SessionEndReason

JETSON = Hello(device_id="jetson-orin", role=Role.JETSON)


def wait_until(predicate, timeout=3.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def wait_for_event(listener, kind, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = listener.next_event(timeout=0.05)
        if isinstance(event, kind):
            return event
    return None


class Phone:
    """A loopback client that speaks the protocol, standing in for the app."""

    def __init__(self, acceptor, device_id="moto-g-power", version=PROTOCOL_VERSION):
        self.connection = acceptor.connect(device_id)
        self.handshake = perform_handshake(
            self.connection, Hello(device_id, Role.PHONE, protocol_version=version)
        )
        self.session = None

    def open_session(self, session_id=999, **kwargs):
        options = {"heartbeat_s": None, "stall_timeout_s": None}
        options.update(kwargs)
        self.session = Session(self.connection, session_id=session_id, **options).start()
        return self.session

    def close(self):
        if self.session is not None:
            self.session.close()
        else:
            self.connection.close()


def listener_for(acceptor, **kwargs):
    options = {"heartbeat_s": None, "stall_timeout_s": None, "accept_poll_s": 0.01}
    options.update(kwargs)
    return TransportListener(acceptor, JETSON, **options).start()


# -- accepting ---------------------------------------------------------------


def test_a_connection_becomes_a_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        phone = Phone(acceptor)
        event = wait_for_event(listener, SessionStarted)
        assert event is not None
        assert event.handshake.remote.device_id == "moto-g-power"
        assert event.handshake.remote.role is Role.PHONE
        assert listener.current_session is event.session
        assert listener.accepted == 1
        phone.close()
    finally:
        listener.stop()


def test_session_ids_start_at_one_and_increase():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    ids = []
    try:
        for _ in range(3):
            phone = Phone(acceptor)
            event = wait_for_event(listener, SessionStarted)
            assert event is not None
            ids.append(event.session.session_id)
            phone.close()
            assert wait_for_event(listener, SessionEnded) is not None
        assert ids == [1, 2, 3]
    finally:
        listener.stop()


def test_data_flows_over_an_accepted_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        phone = Phone(acceptor)
        started = wait_for_event(listener, SessionStarted)
        phone.open_session()
        phone.session.send(Channel.GPS, b"fix")
        message = started.session.recv(Channel.GPS, timeout=2.0)
        assert message is not None and message.payload == b"fix"

        started.session.send(Channel.ADVISORY, b"slow")
        reply = phone.session.recv(Channel.ADVISORY, timeout=2.0)
        assert reply is not None and reply.payload == b"slow"
        phone.close()
    finally:
        listener.stop()


def test_sequence_numbers_restart_on_a_fresh_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        seqs = []
        for _ in range(2):
            phone = Phone(acceptor)
            started = wait_for_event(listener, SessionStarted)
            assert started is not None
            started.session.send(Channel.ADVISORY, b"a")
            seqs.append(read_frame(phone.connection.recv_exact).seq)
            phone.close()
            wait_for_event(listener, SessionEnded)
        assert seqs == [0, 0]
    finally:
        listener.stop()


# -- refusing ----------------------------------------------------------------


def test_a_version_mismatch_is_refused_and_starts_no_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        with pytest.raises(VersionMismatch):
            Phone(acceptor, device_id="stale-app", version=PROTOCOL_VERSION + 1)
        event = wait_for_event(listener, SessionRefused)
        assert event is not None
        assert str(PROTOCOL_VERSION + 1) in event.error
        assert listener.refused == 1
        assert listener.accepted == 0
        assert listener.current_session is None
    finally:
        listener.stop()


def test_a_non_hello_first_frame_is_refused():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        connection = acceptor.connect("rude")
        connection.send_all(
            encode(Frame(channel=Channel.GPS, seq=0, t_mono_ns=1, t_wall_ns=2, payload=b"x"))
        )
        assert wait_for_event(listener, SessionRefused) is not None
        assert listener.accepted == 0
    finally:
        listener.stop()


def test_a_refusal_does_not_disturb_a_live_session():
    """Displacement happens only once the newcomer has proved it speaks the
    protocol; a garbage connection must not cost us a working one."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        good = Phone(acceptor)
        started = wait_for_event(listener, SessionStarted)
        assert started is not None

        with pytest.raises(VersionMismatch):
            Phone(acceptor, device_id="stale", version=99)
        assert wait_for_event(listener, SessionRefused) is not None

        assert listener.current_session is started.session
        assert not started.session.is_closed
        good.open_session()
        good.session.send(Channel.GPS, b"still here")
        message = started.session.recv(Channel.GPS, timeout=2.0)
        assert message is not None and message.payload == b"still here"
        good.close()
    finally:
        listener.stop()


# -- displacement ------------------------------------------------------------


def test_a_new_connection_displaces_the_live_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        first = Phone(acceptor, device_id="phone-a")
        first_started = wait_for_event(listener, SessionStarted)
        assert first_started is not None

        second = Phone(acceptor, device_id="phone-b")
        ended = wait_for_event(listener, SessionEnded)
        assert ended is not None
        assert ended.session_id == first_started.session.session_id
        assert ended.reason is SessionEndReason.DISPLACED
        assert listener.displaced == 1

        second_started = wait_for_event(listener, SessionStarted)
        assert second_started is not None
        assert second_started.session.session_id != first_started.session.session_id
        assert first_started.session.is_closed

        second.open_session()
        second.session.send(Channel.GPS, b"took over")
        message = second_started.session.recv(Channel.GPS, timeout=2.0)
        assert message is not None and message.payload == b"took over"
        second.close()
    finally:
        listener.stop()


def test_the_displaced_session_carries_its_reason_in_its_stats():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        Phone(acceptor, device_id="phone-a")
        first = wait_for_event(listener, SessionStarted)
        Phone(acceptor, device_id="phone-b")
        ended = wait_for_event(listener, SessionEnded)
        assert ended.stats.end_reason == "displaced"
        assert first.session.end_reason is SessionEndReason.DISPLACED
    finally:
        listener.stop()


def test_reconnecting_five_times_leaves_one_live_session():
    """A reconnect-looping phone must not accumulate sessions."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    phones = []
    try:
        for index in range(5):
            phones.append(Phone(acceptor, device_id=f"phone-{index}"))
            assert wait_for_event(listener, SessionStarted) is not None
        assert listener.accepted == 5
        assert listener.displaced == 4
        assert listener.current_session is not None
        assert listener.current_session.session_id == 5
        assert listener.current_session.stats().end_reason is None
    finally:
        for phone in phones:
            phone.close()
        listener.stop()


# -- ending ------------------------------------------------------------------


def test_a_phone_hangup_ends_the_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        phone = Phone(acceptor)
        assert wait_for_event(listener, SessionStarted) is not None
        phone.connection.close()
        ended = wait_for_event(listener, SessionEnded)
        assert ended is not None
        assert ended.reason is SessionEndReason.PEER_CLOSED
        assert listener.current_session is None
    finally:
        listener.stop()


def test_a_silent_phone_is_declared_stalled():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, stall_timeout_s=0.2)
    try:
        Phone(acceptor)  # connects, handshakes, then says nothing
        assert wait_for_event(listener, SessionStarted) is not None
        ended = wait_for_event(listener, SessionEnded, timeout=4.0)
        assert ended is not None
        assert ended.reason is SessionEndReason.STALLED
    finally:
        listener.stop()


def test_a_heartbeating_phone_is_not_declared_stalled():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, heartbeat_s=0.05, stall_timeout_s=0.5)
    try:
        phone = Phone(acceptor)
        started = wait_for_event(listener, SessionStarted)
        phone.open_session(heartbeat_s=0.05, stall_timeout_s=0.5)
        time.sleep(1.2)
        assert not started.session.is_closed
        assert started.session.stats().heartbeats_received > 0
        phone.close()
    finally:
        listener.stop()


def test_stopping_the_listener_closes_the_live_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    phone = Phone(acceptor)
    started = wait_for_event(listener, SessionStarted)
    assert started is not None
    listener.stop()
    assert started.session.is_closed
    assert started.session.end_reason is SessionEndReason.CLOSED_LOCAL
    phone.close()


def test_no_listener_thread_outlives_stop():
    before = {thread.name for thread in threading.enumerate()}
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    Phone(acceptor)
    assert wait_for_event(listener, SessionStarted) is not None
    listener.stop()
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if thread.name not in before and ("listener" in thread.name or "session" in thread.name)
        ]
    )


def test_events_are_ordered_end_before_the_replacement_start():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        Phone(acceptor, device_id="a")
        assert wait_for_event(listener, SessionStarted) is not None
        Phone(acceptor, device_id="b")
        collected = []
        deadline = time.monotonic() + 3.0
        while len(collected) < 2 and time.monotonic() < deadline:
            event = listener.next_event(timeout=0.05)
            if event is not None:
                collected.append(event)
        kinds = [type(event).__name__ for event in collected]
        assert kinds == ["SessionEnded", "SessionStarted"], kinds
    finally:
        listener.stop()


def test_next_event_returns_none_when_nothing_happened():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        assert listener.next_event(timeout=0.0) is None
        assert listener.next_event(timeout=0.05) is None
    finally:
        listener.stop()


# -- event ordering ---------------------------------------------------------


def test_a_session_that_dies_at_once_still_reports_started_first():
    """A consumer must never see an end for a session it was not told began.

    Session boundaries are the whole point of these events: clock offset,
    tracker state and the HERE cache are reset at one. An inverted pair has a
    consumer tear down state it never built, then build state for a session
    that is already dead.
    """
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        connection = acceptor.connect("dies-at-once")
        perform_handshake(connection, Hello("dies-at-once", Role.PHONE))
        # Valid prefix, header is a JSON array: a framing error on the first
        # frame after the handshake.
        connection.send_all(bytes([0, 0, 0, 0, 0, 2]) + b"[]")

        collected = []
        deadline = time.monotonic() + 3.0
        while len(collected) < 2 and time.monotonic() < deadline:
            event = listener.next_event(timeout=0.05)
            if event is not None:
                collected.append(event)
        kinds = [type(event).__name__ for event in collected]
        assert kinds == ["SessionStarted", "SessionEnded"], kinds
        assert collected[1].session_id == collected[0].session.session_id
        assert collected[1].reason is SessionEndReason.FRAMING_ERROR
    finally:
        listener.stop()


# -- a peer that connects and says nothing ----------------------------------


def test_a_connection_that_never_sends_a_hello_is_refused_not_tolerated():
    """The lockout displacement exists to prevent, one step earlier: a phone
    that completes the TCP handshake and loses signal before its hello would
    otherwise hold the accept loop forever, and the phone is the only party
    that can reconnect."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, handshake_timeout_s=0.3)
    try:
        acceptor.connect("mute")  # connects, sends nothing, ever
        event = wait_for_event(listener, SessionRefused, timeout=3.0)
        assert event is not None
        assert "hello" in event.error
        assert listener.refused == 1
        assert listener.accepted == 0
    finally:
        listener.stop()


def test_a_healthy_phone_still_connects_after_a_mute_one():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, handshake_timeout_s=0.3)
    try:
        acceptor.connect("mute")
        assert wait_for_event(listener, SessionRefused, timeout=3.0) is not None
        phone = Phone(acceptor, device_id="healthy")
        started = wait_for_event(listener, SessionStarted, timeout=3.0)
        assert started is not None
        assert started.handshake.remote.device_id == "healthy"
        phone.close()
    finally:
        listener.stop()


def test_stop_returns_and_leaves_no_thread_with_a_peer_mid_handshake():
    before = {thread.name for thread in threading.enumerate()}
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, handshake_timeout_s=30.0)
    acceptor.connect("mute")
    time.sleep(0.2)  # let the accept loop pick it up and block on the hello
    started = time.monotonic()
    listener.stop()
    assert time.monotonic() - started < 5.0
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if thread.name not in before
            and ("listener" in thread.name or "handshake" in thread.name)
        ],
        timeout=5.0,
    )


def test_a_refused_connection_is_closed():
    """Otherwise a rejected peer's socket is held until the process exits."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        connection = acceptor.connect("stale")
        with pytest.raises(VersionMismatch):
            perform_handshake(
                connection, Hello("stale", Role.PHONE, protocol_version=PROTOCOL_VERSION + 1)
            )
        assert wait_for_event(listener, SessionRefused) is not None
        # Bounded: a blocking read would hang rather than fail if the
        # connection were left open, and a hanging test reports nothing.
        outcome: list[str] = []

        def read_once():
            try:
                connection.recv_exact(1)
                outcome.append("data")
            except ConnectionClosed:
                outcome.append("closed")

        reader = threading.Thread(target=read_once, daemon=True)
        reader.start()
        reader.join(timeout=3.0)
        assert outcome == ["closed"], f"refused connection left open: {outcome}"
    finally:
        listener.stop()


# -- guards -----------------------------------------------------------------


def test_an_old_session_ending_late_does_not_clear_the_live_one():
    """A displaced session can finish ending after its replacement is
    installed. Without the identity check the late callback would clear
    current_session and emit an end after the new start."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    try:
        first = Phone(acceptor, device_id="phone-a")
        first_started = wait_for_event(listener, SessionStarted)
        second = Phone(acceptor, device_id="phone-b")
        assert wait_for_event(listener, SessionEnded) is not None
        second_started = wait_for_event(listener, SessionStarted)
        assert second_started is not None

        listener._on_session_end(first_started.session, SessionEndReason.PEER_CLOSED)
        assert listener.current_session is second_started.session
        first.close()
        second.close()
    finally:
        listener.stop()


def test_a_connection_arriving_after_stop_never_becomes_a_session():
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    listener.stop()
    try:
        acceptor.connect("late")
    except ConnectionClosed:
        pass  # the acceptor is closed, which is also correct
    time.sleep(0.2)
    assert listener.accepted == 0
    assert listener.current_session is None


def test_admit_drops_a_connection_that_arrives_as_the_listener_stops():
    """The accept loop can be sitting in a handshake when stop() is called.
    Without the check, a session would start after stop() had returned, and
    nothing would ever close it."""
    acceptor = LoopbackAcceptor()
    listener = TransportListener(
        acceptor, JETSON, heartbeat_s=None, stall_timeout_s=None, accept_poll_s=0.01
    )
    # Not started: _admit is driven directly, because that is where the race
    # lands and a scheduler cannot be asked to reproduce it on demand.
    connection = acceptor.connect("racer")
    phone = threading.Thread(
        target=lambda: perform_handshake(connection, Hello("racer", Role.PHONE)), daemon=True
    )
    phone.start()
    server_side = acceptor.accept(timeout=2.0)
    assert server_side is not None

    listener._stop.set()
    listener._admit(server_side)
    phone.join(timeout=2.0)

    assert listener.accepted == 0
    assert listener.current_session is None
    assert listener.next_event(timeout=0.1) is None


def test_a_healthy_phone_whose_hello_is_late_is_still_accepted():
    """The only case where the timeout's duration matters. The Phone helper
    handshakes before the listener accepts, so its hello is already buffered
    and a zero-length wait would look fine."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, handshake_timeout_s=2.0)
    try:
        connection = acceptor.connect("slow-to-speak")
        threading.Timer(
            0.4, lambda: perform_handshake(connection, Hello("slow-to-speak", Role.PHONE))
        ).start()
        started = wait_for_event(listener, SessionStarted, timeout=5.0)
        assert started is not None, "a healthy phone with a late hello was refused"
        assert started.handshake.remote.device_id == "slow-to-speak"
        assert listener.refused == 0
    finally:
        listener.stop()


def test_a_timed_out_handshake_closes_the_connection():
    """The timeout works by closing the connection under the blocked reader; if
    it did not close, the worker would never be released."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor, handshake_timeout_s=0.3)
    try:
        connection = acceptor.connect("mute")
        assert wait_for_event(listener, SessionRefused, timeout=3.0) is not None
        outcome: list[str] = []

        def read_until_closed():
            # The listener sends its own hello before blocking on ours, so
            # there are buffered bytes to consume before the close surfaces.
            try:
                for _ in range(4096):
                    connection.recv_exact(1)
                outcome.append("still open")
            except ConnectionClosed:
                outcome.append("closed")

        reader = threading.Thread(target=read_until_closed, daemon=True)
        reader.start()
        reader.join(timeout=3.0)
        assert outcome == ["closed"], f"timed-out handshake left the connection open: {outcome}"
        assert listener.handshake_workers_leaked == 0
    finally:
        listener.stop()


def test_stop_racing_the_session_install_leaves_nothing_running(monkeypatch):
    """stop() has to be a barrier against _admit, not a check _admit passed
    earlier. The window in real code spans displacement plus Session
    construction; the sleep here only widens it."""
    import transport.endpoint as endpoint_module

    class SlowSession(Session):
        def __init__(self, *args, **kwargs):
            time.sleep(0.4)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(endpoint_module, "Session", SlowSession)

    acceptor = LoopbackAcceptor()
    listener = TransportListener(
        acceptor, JETSON, heartbeat_s=None, stall_timeout_s=None, accept_poll_s=0.01
    )
    connection = acceptor.connect("racer")
    phone = threading.Thread(
        target=lambda: perform_handshake(connection, Hello("racer", Role.PHONE)), daemon=True
    )
    phone.start()
    server_side = acceptor.accept(timeout=2.0)
    assert server_side is not None

    before = {thread.name for thread in threading.enumerate()}
    admit = threading.Thread(target=lambda: listener._admit(server_side), daemon=True)
    admit.start()
    time.sleep(0.15)  # _admit is inside SlowSession.__init__
    listener.stop()
    admit.join(timeout=4.0)
    phone.join(timeout=2.0)

    assert listener.current_session is None
    assert listener.accepted == 0
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if thread.name not in before and "session" in thread.name
        ],
        timeout=3.0,
    )


def test_a_consumer_that_closes_on_started_gets_no_surviving_thread():
    """Rejecting a session by device id on SessionStarted is an obvious thing
    for the Jetson runtime to do, and it lands before start()."""
    acceptor = LoopbackAcceptor()
    listener = listener_for(acceptor)
    alive_at_return: list[str] = []
    original_put = listener._events.put

    def put_and_close(event):
        original_put(event)
        if isinstance(event, SessionStarted):
            event.session.close()
            alive_at_return.extend(
                thread.name
                for thread in threading.enumerate()
                if f"session{event.session.session_id}" in thread.name
            )

    listener._events.put = put_and_close
    try:
        phone = Phone(acceptor, device_id="unwanted")
        assert wait_for_event(listener, SessionStarted, timeout=3.0) is not None
        assert alive_at_return == [], f"threads outlived close(): {alive_at_return}"
        phone.close()
    finally:
        listener.stop()
