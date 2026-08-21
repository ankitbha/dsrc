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


def test_a_partly_started_session_is_closed_on_the_way_out(monkeypatch):
    """The cleanup closes the session as well as the connection, and only a
    partial start shows why: Session.start() creates its threads one at a time,
    so a failure part-way leaves earlier threads running, and only close()
    joins them. A start() that fails before creating anything cannot tell the
    difference."""
    import transport.client as client_module

    started_threads: list[threading.Thread] = []

    class PartlyStartingSession(client_module.Session):
        def start(self):
            thread = threading.Thread(
                target=lambda: self._stop.wait(30.0), name="partial-session", daemon=True
            )
            thread.start()
            started_threads.append(thread)
            self._threads.append(thread)
            raise RuntimeError("failed on the second thread")

    monkeypatch.setattr(client_module, "Session", PartlyStartingSession)
    dialer = RecordingDialer(lambda host, port: scripted_peer())
    with pytest.raises(RuntimeError):
        connect_session(
            "jetson", 1, local_hello=PHONE, heartbeat_s=None, stall_timeout_s=None, dial_fn=dialer
        )
    assert_closed(dialer.connections[0])
    assert started_threads, "the fixture did not start a thread"
    assert wait_until(lambda: not started_threads[0].is_alive(), timeout=5.0), (
        "a thread from a partly started session outlived the failure"
    )


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


# -- a backend that ignores close(), on the dialling side -------------------


class DeafLoopbackConnection:
    """Honours the ByteConnection API and ignores its close() requirement.

    Wrong, and documented as wrong -- but it is what closing a POSIX socket
    without shutdown() does, so it is the mistake the next backend is most
    likely to make.
    """

    peer = "deaf-loopback"

    def __init__(self, registry=None):
        self._gate = threading.Event()
        self.closed = False
        if registry is not None:
            registry.append(self)

    def send_all(self, data):
        return None

    def recv_exact(self, n):
        self._gate.wait()  # never set; close() below does not release it
        return b""

    def close(self):
        self.closed = True

    def release(self):
        """Let the blocked reader go, so a deliberate leak does not outlive the
        test that made it. Session thread names are not unique across files, so
        a stray session1-rx breaks whatever else filters on that name."""
        self._gate.set()


def test_an_abandoned_handshake_worker_is_counted_not_hidden():
    """The escalation bounds the rate of this leak, not the total, and a leaked
    worker is not a connect failure -- so without a counter nothing notices at
    all. At the shipped defaults it is roughly one thread and one socket every
    ten seconds: several hundred over a drive, with failed_attempts climbing
    normally the whole time."""
    before = {thread.name for thread in threading.enumerate()}
    made: list[DeafLoopbackConnection] = []
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.01, multiplier=1.0, cap_s=0.01, jitter=0.0),
        handshake_timeout_s=0.1,
        dial_fn=lambda host, port, timeout=None: DeafLoopbackConnection(made),
        poll_s=0.01,
    ).start()
    try:
        assert wait_until(lambda: client.handshake_workers_leaked >= 2, timeout=10.0), (
            f"leaked {client.handshake_workers_leaked}, counted none"
        )
        failure = wait_for_client_event(client, ClientAttemptFailed, timeout=5.0)
        assert failure is not None
        assert "abandoned" in failure.error
    finally:
        client.stop()
    # The threads really are leaked; the point is that the count says so.
    leaked = [
        thread for thread in threading.enumerate()
        if thread.name not in before and thread.name == "client-handshake"
    ]
    assert leaked, "the fixture did not actually leak a worker"
    # Released only after the assertion, so the leak is proved and then cleaned
    # up rather than left for another test to trip over.
    for connection in made:
        connection.release()
    assert wait_until(
        lambda: not [
            t for t in threading.enumerate()
            if t.name not in before and t.name == "client-handshake"
        ],
        timeout=5.0,
    )


def test_a_handshake_timeout_closes_the_connection_to_release_the_worker():
    """Pins the close itself: HandshakeError is raised whether or not it
    happened, so the previous test asserted the message, not the release."""
    connections: list[object] = []

    def dial_recording(host, port, timeout=None):
        client_end, _server_end = loopback_pair()
        connections.append(client_end)
        return client_end

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.01, multiplier=1.0, cap_s=0.01, jitter=0.0),
        handshake_timeout_s=0.1,
        dial_fn=dial_recording,
        poll_s=0.01,
    ).start()
    try:
        assert wait_until(lambda: len(connections) >= 1, timeout=5.0)
        assert wait_until(lambda: client.failed_attempts >= 1, timeout=5.0)
        assert_closed(connections[0])
        assert client.handshake_workers_leaked == 0
    finally:
        client.stop()


# -- retry_in_s must never promise a retry that cannot happen ---------------


def test_a_session_ended_by_stop_reports_no_retry():
    """retry_in_s is the observable for the escalation, so a value meaning
    "intended" on some paths and "actual" on others is the one thing it must
    not be."""
    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=fast_backoff(),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=lambda host, port, timeout=None: scripted_peer(),
        reset_after_s=100.0,  # nothing here could ever be durable
        poll_s=0.01,
    ).start()
    assert client.wait_for_session(timeout=5.0) is not None
    client.stop()
    ended = [
        event for event in client.drain_events() if isinstance(event, ClientSessionEnded)
    ]
    assert ended, "no session-end event"
    assert ended[-1].reason is SessionEndReason.CLOSED_LOCAL
    assert ended[-1].retry_in_s is None, ended[-1]


def test_an_attempt_failure_racing_stop_reports_no_retry():
    """Reached by stopping during the dial, which is the only way in.

    The loop's own `while not stop` guard exits before a second attempt, so
    stopping between two attempts never reaches this branch -- the earlier
    version of this test assumed it did and asserted on an event that is never
    emitted.
    """
    holder: dict[str, SessionClient] = {}

    def refuse_after_stopping(host, port, timeout=None):
        holder["client"]._stop.set()  # as a stop() landing mid-dial would
        raise ConnectionRefusedError(61, "connection refused")

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=0.3, multiplier=1.0, cap_s=0.3, jitter=0.0),
        dial_fn=refuse_after_stopping,
        poll_s=0.01,
    )
    holder["client"] = client
    client.start()
    try:
        failure = wait_for_client_event(client, ClientAttemptFailed, timeout=5.0)
        assert failure is not None
        assert failure.retry_in_s is None, failure
        assert failure.attempt == 1
    finally:
        client.stop()


class DeafAfterHandshake:
    """Handshakes normally, then blocks its reader, and ignores close().

    Which makes session teardown slow for a real reason -- close() waits out
    its whole join timeout -- so it can show whether uptime includes that wait.
    """

    peer = "deaf-after-handshake"

    def __init__(self, inner, allow_reads=2):
        self._inner = inner
        self._allow_reads = allow_reads
        self._reads = 0
        self._gate = threading.Event()

    def send_all(self, data):
        self._inner.send_all(data)

    def recv_exact(self, n):
        if self._reads < self._allow_reads:
            self._reads += 1
            return self._inner.recv_exact(n)
        self._gate.wait()
        return b""

    def close(self):
        return None  # deliberately not honouring the contract

    def release(self):
        self._gate.set()
        self._inner.close()


def test_uptime_excludes_teardown_even_when_teardown_is_slow():
    """uptime decides durability, so join time inside it shifts the threshold.

    Only visible when teardown is slow: with a peer that ignores close(), the
    session's reader never exits and close() waits out its full join. A session
    that lived for milliseconds would then report seconds, and be judged
    durable.
    """
    connections: list[DeafAfterHandshake] = []

    def deaf_after_handshake(host, port, timeout=None):
        client_end, server_end = loopback_pair()
        threading.Thread(
            target=lambda: perform_handshake(server_end, JETSON), daemon=True
        ).start()
        wrapped = DeafAfterHandshake(client_end)
        connections.append(wrapped)
        return wrapped

    client = SessionClient(
        "jetson",
        1,
        local_hello=PHONE,
        backoff=Backoff(initial_s=5.0, multiplier=1.0, cap_s=5.0, jitter=0.0),
        heartbeat_s=None,
        stall_timeout_s=None,
        dial_fn=deaf_after_handshake,
        reset_after_s=1.0,
        poll_s=0.01,
    ).start()
    try:
        session = client.wait_for_session(timeout=10.0)
        assert session is not None
        time.sleep(0.05)
        session.close()

        ended = wait_for_client_event(client, ClientSessionEnded, timeout=10.0)
        assert ended is not None
        assert ended.uptime_s < 0.5, (
            f"uptime {ended.uptime_s:.2f}s includes the teardown wait"
        )
    finally:
        client.stop()
        for connection in connections:
            connection.release()


def test_the_connection_being_handshaked_is_released_after_success():
    """The finally clause: without it, stop() could close a connection that had
    already become a live session's."""
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
    try:
        session = client.wait_for_session(timeout=5.0)
        assert session is not None
        assert client._connecting is None, "the handshake reference was never cleared"
    finally:
        client.stop()


class IgnoresClose:
    """Honours the API and ignores close(), so a read outlives the timeout.

    The only shape in which a completed handshake exists to be salvaged: with a
    compliant backend the close ends the read and the late hello never arrives,
    so there is nothing to discard and the timeout is simply correct.
    """

    peer = "ignores-close"

    def __init__(self, inner):
        self._inner = inner

    def send_all(self, data):
        self._inner.send_all(data)

    def recv_exact(self, n):
        return self._inner.recv_exact(n)

    def close(self):
        return None

    def release(self):
        self._inner.close()


def test_a_handshake_completing_inside_the_grace_is_used_not_discarded():
    """Narrow but free: on this backend the timeout expires, the close does not
    release the reader, and the peer then answers inside the one-second grace.
    Discarding that success would cost a reconnect and a displacement on the far
    end, because on this side the timeout is the retry trigger."""
    from transport.client import _handshake_with_timeout
    from transport.clock import now_mono_ns, now_wall_ns

    client_end, server_end = loopback_pair()
    wrapped = IgnoresClose(client_end)

    def answer_late():
        try:
            perform_handshake(server_end, JETSON)
        except Exception:
            pass

    threading.Timer(0.2, answer_late).start()
    try:
        result = _handshake_with_timeout(
            wrapped,
            PHONE,
            timeout_s=0.05,
            mono_clock=now_mono_ns,
            wall_clock=now_wall_ns,
        )
        assert result.remote.device_id == JETSON.device_id
    finally:
        wrapped.release()


def test_a_compliant_backend_reports_the_timeout_rather_than_a_late_success():
    """The counterpart, and the reason the finding that prompted the salvage
    overstated itself: when close() is honoured, the read ends at the timeout
    and the late hello never arrives. There is no discarded success."""
    from transport.client import _handshake_with_timeout
    from transport.clock import now_mono_ns, now_wall_ns
    from transport.handshake import HandshakeError

    client_end, server_end = loopback_pair()

    def answer_late():
        try:
            perform_handshake(server_end, JETSON)
        except Exception:
            pass  # our side closed at the timeout; that is the point

    threading.Timer(0.3, answer_late).start()
    with pytest.raises(HandshakeError, match="no hello"):
        _handshake_with_timeout(
            client_end, PHONE, timeout_s=0.05, mono_clock=now_mono_ns, wall_clock=now_wall_ns
        )
