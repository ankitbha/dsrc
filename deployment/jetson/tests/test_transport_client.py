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
from transport.session import Session, SessionEndReason
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


def assert_closed(connection, timeout=3.0):
    """Bounded: `pytest.raises` around a blocking read hangs rather than fails
    if the connection is still open, and a hung suite names no test."""
    outcome: list[str] = []

    def probe():
        try:
            connection.recv_exact(1)
            outcome.append("open")
        except ConnectionClosed:
            outcome.append("closed")

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout)
    assert outcome == ["closed"], f"connection not closed ({outcome or 'read still blocked'})"


def open_descriptor_count():
    import os

    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except OSError:
            continue
    return None


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
    assert_closed(dialer.connections[0])


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
    assert_closed(dialer.connections[0])


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


# -- the guards the round-1 audit found unpinned ----------------------------


def test_a_session_constructor_failure_closes_the_connection(monkeypatch):
    """The constructor used to sit between the two guarded blocks, so a failure
    there leaked the socket. Guarding start() alone covered the call, not the
    step."""
    import transport.client as client_module

    class UnconstructableSession(client_module.Session):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("constructor blew up")

    monkeypatch.setattr(client_module, "Session", UnconstructableSession)
    dialer = RecordingDialer(lambda host, port: scripted_peer())
    with pytest.raises(RuntimeError):
        connect_session(
            "jetson", 1, local_hello=PHONE, heartbeat_s=None, stall_timeout_s=None, dial_fn=dialer
        )
    assert_closed(dialer.connections[0])


def test_a_peer_that_never_sends_a_hello_is_given_up_on_and_retried():
    """A listener whose accept loop has died looks exactly like this: the TCP
    connection completes and nothing follows. dial's timeout covers only the
    connect, so without a handshake timeout the client wedges here forever --
    no retry, no attempt counted, no event."""

    def mute_peer(host, port, timeout=None):
        client_end, _server_end = loopback_pair()
        return client_end  # nobody ever handshakes on the far end

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        handshake_timeout_s=0.3,
        dial_fn=mute_peer,
        poll_s=0.01,
    ).start()
    try:
        first = wait_for_client_event(client, ClientAttemptFailed, timeout=5.0)
        assert first is not None
        assert "hello" in first.error
        assert wait_until(lambda: client.failed_attempts >= 2, timeout=6.0)
    finally:
        client.stop()


def test_stop_returns_with_no_live_session_and_no_thread():
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
    session = client.wait_for_session(timeout=5.0)
    assert session is not None
    client.stop()
    assert session.is_closed
    assert session.end_reason is SessionEndReason.CLOSED_LOCAL
    assert client.current_session is None
    assert [
        thread.name for thread in threading.enumerate() if thread.name not in before
    ] == []


def test_stop_interrupts_a_handshake_in_progress():
    """The handshake worker is blocked on a read, so the stop flag alone cannot
    reach it; closing the connection is what releases it."""
    before = {thread.name for thread in threading.enumerate()}

    def mute_peer(host, port, timeout=None):
        client_end, _server_end = loopback_pair()
        return client_end

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        handshake_timeout_s=30.0,  # far longer than the test will wait
        dial_fn=mute_peer,
        poll_s=0.01,
    ).start()
    time.sleep(0.3)  # let it get into the handshake
    started = time.monotonic()
    client.stop(timeout=5.0)
    assert time.monotonic() - started < 5.0
    assert wait_until(
        lambda: not [
            thread
            for thread in threading.enumerate()
            if thread.name not in before and thread.name in {"client", "client-handshake"}
        ],
        timeout=5.0,
    )


def test_current_session_never_hands_back_a_closed_session():
    """The class docstring says the event design exists so a consumer is not
    handed a silently-replaced object; handing back a dead one is the same
    failure."""
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.005, multiplier=1.0, cap_s=0.005, jitter=0.0),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.03),
        reset_after_s=0.0,
        poll_s=0.005,
    ).start()
    try:
        deadline = time.monotonic() + 3.0
        seen_closed = 0
        while time.monotonic() < deadline:
            session = client.current_session
            if session is not None and session.is_closed:
                seen_closed += 1
            time.sleep(0.001)
        assert client.connected >= 3, client.connected
        assert seen_closed == 0, f"handed back a closed session {seen_closed} times"
    finally:
        client.stop()


def test_current_session_filters_a_closed_session_it_still_holds():
    """The filter and the on_end clearing are redundant by design, so each hid
    the other's absence under mutation. This pins the filter alone."""
    near, _far = loopback_pair()
    client = SessionClient("jetson", 1, local_hello=PHONE)
    session = Session(near, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    session.close()
    client._current = session
    assert client.current_session is None


def test_the_session_reference_is_cleared_before_the_next_poll():
    """And this pins the clearing alone: with a deliberately slow poll, a
    reference cleared only by the maintain loop would linger for a whole
    interval after the session ended."""
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(),
        poll_s=2.0,
    ).start()
    try:
        session = client.wait_for_session(timeout=5.0)
        assert session is not None
        session.close()
        # Far shorter than poll_s, so only an immediate clear can satisfy it.
        assert wait_until(lambda: client._current is None, timeout=0.5), (
            "the reference survived the session, so only the property's filter "
            "was hiding it"
        )
    finally:
        client.stop()


def test_the_retry_delay_escalates_for_short_lived_sessions():
    """Pins the escalation's effect, not just the attempt numbering: the
    counter increments whatever delay is used."""
    backoff = fast_backoff()
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=backoff,
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.01),
        reset_after_s=10.0,
        random_unit=lambda: 0.5,  # no jitter, so the sequence is exact
        poll_s=0.01,
    ).start()
    try:
        delays = []
        deadline = time.monotonic() + 8.0
        while len(delays) < 3 and time.monotonic() < deadline:
            event = client.next_event(timeout=0.05)
            if isinstance(event, ClientSessionEnded):
                delays.append(event.retry_in_s)
        assert len(delays) >= 3, delays
        assert all(d is not None for d in delays), delays
        assert delays[:3] == [
            backoff.base_delay_for(1),
            backoff.base_delay_for(2),
            backoff.base_delay_for(3),
        ], delays
    finally:
        client.stop()


def test_a_durable_session_reports_no_retry_delay():
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(lifetime_s=0.1),
        reset_after_s=0.01,
        poll_s=0.01,
    ).start()
    try:
        ended = wait_for_client_event(client, ClientSessionEnded, timeout=6.0)
        assert ended is not None
        assert ended.retry_in_s is None
    finally:
        client.stop()


def test_the_connection_counters_are_exact():
    """`reconnects` is a headline number in the experiment's report, and every
    other assertion on it is >=, so an off-by-one would go unnoticed."""
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
        assert wait_until(lambda: client.connected >= 4, timeout=10.0)
    finally:
        client.stop()
    assert client.reconnects == client.connected - 1, (client.connected, client.reconnects)
    assert client.failed_attempts == 0


def test_the_default_reset_threshold_cannot_coincide_with_the_stall_timeout():
    """A stalled session lasts exactly the stall timeout. With the threshold at
    the backoff cap -- also 5.0 -- the commonest failure in a car landed on the
    resetting side of a >= test, so the escalation never engaged for it."""
    from transport.session import DEFAULT_STALL_TIMEOUT_S

    client = SessionClient("jetson", 1, local_hello=PHONE)
    assert client._reset_after_s > DEFAULT_STALL_TIMEOUT_S
    assert client._reset_after_s == max(Backoff().cap_s, DEFAULT_STALL_TIMEOUT_S * 2.0)

    without_stall = SessionClient("jetson", 1, local_hello=PHONE, stall_timeout_s=None)
    assert without_stall._reset_after_s == Backoff().cap_s


def test_giving_up_stops_the_client_rather_than_leaving_it_idle():
    def refuse(host, port, timeout=None):
        raise ConnectionRefusedError(61, "connection refused")

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.005, multiplier=1.0, cap_s=0.005, jitter=0.0),
        dial_fn=refuse,
        max_attempts=2,
        poll_s=0.005,
    ).start()
    try:
        assert wait_for_client_event(client, ClientGaveUp, timeout=5.0) is not None
        started = time.monotonic()
        assert client.wait_for_session(timeout=5.0) is None
        assert time.monotonic() - started < 1.0, "wait_for_session burned its whole timeout"
    finally:
        client.stop()


def test_twenty_reconnects_over_real_sockets_leak_no_descriptors():
    """The thread-only version of this ran against loopback, which has no file
    descriptors, so the fd half of the criterion was untested."""
    acceptor = TcpAcceptor("127.0.0.1", 0)
    listener = TransportListener(
        acceptor, JETSON, heartbeat_s=None, stall_timeout_s=None, accept_poll_s=0.01
    ).start()
    before_fds = open_descriptor_count()
    client = SessionClient(
        "127.0.0.1",
        acceptor.port,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.005, multiplier=1.0, cap_s=0.005, jitter=0.0),
        heartbeat_s=None,
        stall_timeout_s=None,
        reset_after_s=0.0,
        poll_s=0.005,
    ).start()
    try:
        for _ in range(20):
            session = client.wait_for_session(timeout=10.0)
            assert session is not None
            session.close()
        assert client.connected >= 20, client.connected
    finally:
        client.stop()
        listener.stop()
    if before_fds is not None:
        assert wait_until(
            lambda: open_descriptor_count() <= before_fds + 3, timeout=5.0
        ), f"descriptors grew from {before_fds} to {open_descriptor_count()}"
