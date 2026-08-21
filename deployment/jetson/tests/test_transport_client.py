"""The dialling side: the retry schedule, the composition, and reconnection.

The schedule is tested as arithmetic rather than by waiting for it, and
reconnection is tested against a scripted peer rather than a real listener, so
that session lifetime is controlled rather than hoped for. The socket path has
its own coverage in test_transport_tcp.py.
"""

from __future__ import annotations

import threading
import time

import pytest

from transport.channels import Channel
from transport.connection import ConnectionClosed
from transport.endpoint import SessionStarted, TransportListener
from transport.handshake import Hello, Role, VersionMismatch, perform_handshake
from transport.loopback import loopback_pair
from transport.session import SessionEndReason
from transport.tcp import PEER_GONE_ERRNOS, TcpAcceptor, dial
from transport.client import (
    Backoff,
    ClientAttemptFailed,
    ClientGaveUp,
    ClientSessionEnded,
    ClientSessionStarted,
    SessionClient,
    connect_session,
)

PHONE = Hello(device_id="mac-standing-in-for-phone", role=Role.PHONE)
JETSON = Hello(device_id="jetson-orin", role=Role.JETSON)


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# -- the schedule ------------------------------------------------------------


def test_the_confirmed_schedule_is_what_the_code_computes():
    backoff = Backoff()
    assert backoff.initial_s == 0.25
    assert backoff.multiplier == 2.0
    assert backoff.cap_s == 5.0
    assert backoff.jitter == 0.25
    assert [backoff.base_delay_for(n) for n in range(1, 9)] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        5.0,
        5.0,
        5.0,
    ]


def test_the_delay_is_capped_however_far_it_goes():
    backoff = Backoff()
    assert backoff.base_delay_for(100) == backoff.cap_s


def test_jitter_spans_exactly_the_stated_band():
    backoff = Backoff()
    for attempt in range(1, 8):
        base = backoff.base_delay_for(attempt)
        low, high = backoff.bounds_for(attempt)
        assert backoff.delay_for(attempt, 0.0) == pytest.approx(low)
        assert backoff.delay_for(attempt, 1.0) == pytest.approx(high)
        assert backoff.delay_for(attempt, 0.5) == pytest.approx(base)
        assert low == pytest.approx(base * 0.75)
        assert high == pytest.approx(base * 1.25)


def test_a_random_unit_always_lands_inside_the_band():
    import random as random_module

    backoff = Backoff()
    generator = random_module.Random(20260820)
    for attempt in range(1, 8):
        low, high = backoff.bounds_for(attempt)
        for _ in range(200):
            delay = backoff.delay_for(attempt, generator.random())
            assert low <= delay <= high


def test_an_attempt_below_one_is_rejected():
    with pytest.raises(ValueError):
        Backoff().base_delay_for(0)


# -- composition -------------------------------------------------------------


class RecordingDialer:
    """Hands out a scripted connection and remembers what it handed out."""

    def __init__(self, factory):
        self._factory = factory
        self.connections = []

    def __call__(self, host, port, timeout=None):
        connection = self._factory(host, port)
        self.connections.append(connection)
        return connection


def scripted_peer(remote_hello=JETSON, lifetime_s=None, refuse=False):
    """A loopback pair whose far end handshakes and then optionally hangs up."""
    client_end, server_end = loopback_pair()

    def serve():
        hello = remote_hello
        if refuse:
            hello = Hello("stale-jetson", Role.JETSON, protocol_version=99)
        try:
            perform_handshake(server_end, hello)
        except Exception:
            return
        if lifetime_s is not None:
            time.sleep(lifetime_s)
            server_end.close()

    threading.Thread(target=serve, daemon=True).start()
    return client_end


def test_connect_session_returns_a_started_session():
    dialer = RecordingDialer(lambda host, port: scripted_peer())
    session, handshake = connect_session(
        "jetson", 1, local_hello=PHONE, heartbeat_s=None, stall_timeout_s=None, dial_fn=dialer
    )
    try:
        assert handshake.remote.device_id == "jetson-orin"
        assert session.send(Channel.GPS, b"fix") is True
        assert not session.is_closed
    finally:
        session.close()


def test_a_dial_failure_propagates_and_opens_nothing():
    def refuse(host, port, timeout=None):
        raise ConnectionRefusedError(61, "connection refused")

    with pytest.raises(OSError):
        connect_session("jetson", 1, local_hello=PHONE, dial_fn=refuse)


def test_a_handshake_failure_closes_the_connection_it_opened():
    dialer = RecordingDialer(lambda host, port: scripted_peer(refuse=True))
    with pytest.raises(VersionMismatch):
        connect_session("jetson", 1, local_hello=PHONE, dial_fn=dialer)
    assert len(dialer.connections) == 1
    with pytest.raises(ConnectionClosed):
        dialer.connections[0].recv_exact(1)


def test_a_session_start_failure_closes_the_connection_it_opened(monkeypatch):
    import transport.client as client_module

    class UnstartableSession(client_module.Session):
        def start(self):
            raise RuntimeError("no threads today")

    monkeypatch.setattr(client_module, "Session", UnstartableSession)
    dialer = RecordingDialer(lambda host, port: scripted_peer())
    with pytest.raises(RuntimeError):
        connect_session(
            "jetson", 1, local_hello=PHONE, heartbeat_s=None, stall_timeout_s=None, dial_fn=dialer
        )
    with pytest.raises(ConnectionClosed):
        dialer.connections[0].recv_exact(1)


# -- reconnection ------------------------------------------------------------


def fast_backoff():
    return Backoff(initial_s=0.02, multiplier=2.0, cap_s=0.08, jitter=0.25)


def wait_for_client_event(client, kind, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = client.next_event(timeout=0.05)
        if isinstance(event, kind):
            return event
    return None


def test_a_client_reconnects_after_the_peer_hangs_up():
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.15),
        poll_s=0.01,
    ).start()
    try:
        first = wait_for_client_event(client, ClientSessionStarted)
        assert first is not None
        ended = wait_for_client_event(client, ClientSessionEnded)
        assert ended is not None
        assert ended.reason is SessionEndReason.PEER_CLOSED
        second = wait_for_client_event(client, ClientSessionStarted)
        assert second is not None
        assert second.session.session_id != first.session.session_id
        assert client.reconnects >= 1
    finally:
        client.stop()


def test_a_failing_dial_reports_the_attempt_and_the_delay():
    def refuse(host, port, timeout=None):
        raise ConnectionRefusedError(61, "connection refused")

    backoff = fast_backoff()
    client = SessionClient(
        "jetson", 1, local_hello=PHONE, backoff=backoff, dial_fn=refuse, poll_s=0.01
    ).start()
    try:
        seen = []
        deadline = time.monotonic() + 5.0
        while len(seen) < 3 and time.monotonic() < deadline:
            event = client.next_event(timeout=0.05)
            if isinstance(event, ClientAttemptFailed):
                seen.append(event)
        assert len(seen) >= 3
        assert [event.attempt for event in seen[:3]] == [1, 2, 3]
        for event in seen:
            low, high = backoff.bounds_for(event.attempt)
            assert low <= event.retry_in_s <= high
            assert "refused" in event.error
    finally:
        client.stop()


def test_a_client_can_be_told_to_give_up():
    def refuse(host, port, timeout=None):
        raise ConnectionRefusedError(61, "connection refused")

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        dial_fn=refuse,
        max_attempts=3,
        poll_s=0.01,
    ).start()
    try:
        gave_up = wait_for_client_event(client, ClientGaveUp)
        assert gave_up is not None
        assert gave_up.attempts == 3
        assert client.failed_attempts == 3
    finally:
        client.stop()


def test_a_short_lived_session_does_not_reset_the_schedule():
    """Otherwise a link that connects and dies immediately reconnects at the
    initial delay forever, and the escalation does nothing."""
    backoff = fast_backoff()
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=backoff,
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.01),
        reset_after_s=10.0,  # nothing here will ever qualify as durable
        poll_s=0.01,
    ).start()
    try:
        attempts = []
        deadline = time.monotonic() + 6.0
        while len(attempts) < 3 and time.monotonic() < deadline:
            event = client.next_event(timeout=0.05)
            if isinstance(event, ClientSessionStarted):
                attempts.append(event.attempt)
        assert len(attempts) >= 3, attempts
        assert attempts == sorted(attempts)
        assert attempts[-1] > attempts[0], f"the schedule reset: {attempts}"
    finally:
        client.stop()


def test_a_durable_session_resets_the_schedule():
    backoff = fast_backoff()
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=backoff,
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.2),
        reset_after_s=0.05,  # everything here qualifies
        poll_s=0.01,
    ).start()
    try:
        attempts = []
        deadline = time.monotonic() + 6.0
        while len(attempts) < 3 and time.monotonic() < deadline:
            event = client.next_event(timeout=0.05)
            if isinstance(event, ClientSessionStarted):
                attempts.append(event.attempt)
        assert len(attempts) >= 3, attempts
        assert attempts == [1, 1, 1], attempts
    finally:
        client.stop()


def test_twenty_reconnects_accumulate_no_threads():
    before = threading.active_count()
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.005, multiplier=1.0, cap_s=0.005, jitter=0.0),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.02),
        reset_after_s=0.0,
        poll_s=0.005,
    ).start()
    try:
        assert wait_until(lambda: client.connected >= 20, timeout=20.0), client.connected
    finally:
        client.stop()
    assert wait_until(lambda: threading.active_count() <= before + 1, timeout=5.0), (
        f"threads grew from {before} to {threading.active_count()}"
    )


def test_stop_leaves_no_client_thread():
    before = {thread.name for thread in threading.enumerate()}
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(),
        poll_s=0.01,
    ).start()
    assert client.wait_for_session(timeout=5.0) is not None
    client.stop()
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if thread.name not in before and thread.name == "client"
        ]
    )


# -- against a real listener -------------------------------------------------


def test_the_client_and_the_listener_meet_over_a_real_socket():
    acceptor = TcpAcceptor("127.0.0.1", 0)
    listener = TransportListener(
        acceptor, JETSON, heartbeat_s=None, stall_timeout_s=None, accept_poll_s=0.02
    ).start()
    client = SessionClient(
        "127.0.0.1",
        acceptor.port,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        poll_s=0.01,
    ).start()
    try:
        session = client.wait_for_session(timeout=10.0)
        assert session is not None
        started = None
        deadline = time.monotonic() + 5.0
        while started is None and time.monotonic() < deadline:
            event = listener.next_event(timeout=0.05)
            if isinstance(event, SessionStarted):
                started = event
        assert started is not None

        session.send(Channel.CAMERA, b"\x5a" * 100_000, {"probe": 7})
        message = started.session.recv(Channel.CAMERA, timeout=10.0)
        assert message is not None
        assert len(message.payload) == 100_000
        assert message.extensions == {"probe": 7}

        started.session.send(Channel.ADVISORY, b"cap=25.0")
        back = session.recv(Channel.ADVISORY, timeout=10.0)
        assert back is not None and back.payload == b"cap=25.0"
    finally:
        client.stop()
        listener.stop()


def test_the_client_reconnects_when_the_listener_displaces_it():
    acceptor = TcpAcceptor("127.0.0.1", 0)
    listener = TransportListener(
        acceptor, JETSON, heartbeat_s=None, stall_timeout_s=None, accept_poll_s=0.02
    ).start()
    client = SessionClient(
        "127.0.0.1",
        acceptor.port,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        reset_after_s=0.0,
        poll_s=0.01,
    ).start()
    try:
        assert client.wait_for_session(timeout=10.0) is not None
        # A second dialler displaces the client's session; the client must
        # notice and come back rather than sit on a dead socket.
        intruder = dial("127.0.0.1", acceptor.port)
        perform_handshake(intruder, Hello("other-phone", Role.PHONE))
        assert wait_until(lambda: client.reconnects >= 1, timeout=10.0)
        intruder.close()
        assert client.wait_for_session(timeout=10.0) is not None
    finally:
        client.stop()
        listener.stop()
