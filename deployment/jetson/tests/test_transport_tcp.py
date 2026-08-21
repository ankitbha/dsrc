"""Socket behaviour the in-process backend structurally cannot reproduce:
resets, real buffers, real accept semantics, and descriptor lifetime.

The generic requirements live in test_transport_backend_contract.py; this file
is what is specific to being a socket.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time

import pytest

from transport.channels import Channel
from transport.connection import ConnectionClosed
from transport.endpoint import SessionEnded, SessionStarted, TransportListener
from transport.handshake import Hello, Role, perform_handshake
from transport.session import Session, SessionEndReason
from transport.tcp import (
    ACCEPT_RETRY_ERRNOS,
    ACCEPT_TRANSIENT_ERRNOS,
    DEFAULT_PORT,
    PEER_GONE_ERRNOS,
    TcpAcceptor,
    TcpConnection,
    apply_socket_options,
    dial,
)
from transport.tcp import _translate

JETSON = Hello(device_id="jetson-orin", role=Role.JETSON)
PHONE = Hello(device_id="mac-standing-in-for-phone", role=Role.PHONE)


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def open_descriptor_count() -> int | None:
    """Best-effort open-fd count; None where it cannot be read."""
    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except OSError:
            continue
    return None


@pytest.fixture
def acceptor():
    made = TcpAcceptor("127.0.0.1", 0)
    try:
        yield made
    finally:
        made.close()


# -- binding and accepting ---------------------------------------------------


def test_an_ephemeral_port_is_bound_and_discoverable(acceptor):
    assert acceptor.host == "127.0.0.1"
    assert acceptor.port > 0
    assert acceptor.address == ("127.0.0.1", acceptor.port)


def test_the_default_port_is_distinct_from_the_v2v_beacon():
    """v2v/beacon.py already uses 47808 for UDP; sharing a number would make
    two unrelated things look related in a packet capture."""
    assert DEFAULT_PORT == 47811


def test_accept_returns_none_on_timeout(acceptor):
    """Bounded on a worker: a regression here would otherwise block forever and
    a hung suite names no test."""
    outcome: list[object] = []
    started = time.monotonic()
    worker = threading.Thread(
        target=lambda: outcome.append(acceptor.accept(timeout=0.2)), daemon=True
    )
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "accept did not return within 5s of a 0.2s timeout"
    assert outcome == [None]
    assert time.monotonic() - started >= 0.15


def test_accept_after_close_raises(acceptor):
    acceptor.close()
    with pytest.raises(ConnectionClosed):
        acceptor.accept(timeout=0.1)


def test_close_is_idempotent(acceptor):
    acceptor.close()
    acceptor.close()


def test_binding_a_port_already_in_use_raises_and_leaks_nothing(acceptor):
    before = open_descriptor_count()
    with pytest.raises(OSError):
        TcpAcceptor("127.0.0.1", acceptor.port)
    if before is not None:
        assert wait_until(lambda: open_descriptor_count() <= before + 1)


def test_dialling_a_closed_port_fails_as_an_oserror(acceptor):
    port = acceptor.port
    acceptor.close()
    time.sleep(0.05)
    with pytest.raises(OSError):
        dial("127.0.0.1", port, timeout=2.0)


def test_connections_are_accepted_in_order(acceptor):
    first = dial("127.0.0.1", acceptor.port)
    second = dial("127.0.0.1", acceptor.port)
    try:
        first.send_all(b"1")
        second.send_all(b"2")
        assert acceptor.accept(timeout=5.0).recv_exact(1) == b"1"
        assert acceptor.accept(timeout=5.0).recv_exact(1) == b"2"
    finally:
        first.close()
        second.close()


# -- socket options ----------------------------------------------------------


def test_the_applied_options_are_recorded_not_assumed():
    """The keepalive knobs have different names per platform and are missing on
    some. What was actually set is the evidence; the intent is not."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        applied = apply_socket_options(sock)
        assert applied["TCP_NODELAY"] == 1
        assert applied["SO_KEEPALIVE"] == 1
        assert set(applied) == {
            "TCP_NODELAY",
            "SO_KEEPALIVE",
            "keepalive_idle_s",
            "keepalive_interval_s",
            "keepalive_probes",
        }
        for key, value in applied.items():
            assert value is None or isinstance(value, int), (key, value)
    finally:
        sock.close()


def test_nodelay_is_actually_set_on_a_dialled_connection(acceptor):
    client = dial("127.0.0.1", acceptor.port)
    server = acceptor.accept(timeout=5.0)
    try:
        assert client.applied_options["TCP_NODELAY"] == 1
        assert server.applied_options["TCP_NODELAY"] == 1
    finally:
        client.close()
        server.close()


def test_a_dialled_socket_has_no_read_deadline():
    """create_connection leaves the connect timeout on the socket. Without the
    reset, every session inherits a read deadline equal to the connect timeout:
    an idle link dies as transport_error instead of stalled, and sendall can
    fail mid-write."""
    acceptor = TcpAcceptor("127.0.0.1", 0)
    client = dial("127.0.0.1", acceptor.port, timeout=0.3)
    server = acceptor.accept(timeout=5.0)
    try:
        assert client._sock.gettimeout() is None
        assert server._sock.gettimeout() is None
    finally:
        client.close()
        server.close()
        acceptor.close()


def test_options_can_be_skipped():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = TcpConnection(sock, peer="unset", set_options=False)
    try:
        assert connection.applied_options == {}
    finally:
        connection.close()


# -- a reset is the phone leaving, not our end breaking ----------------------


def reset_immediately(sock: socket.socket) -> None:
    """SO_LINGER with a zero timeout makes close() send RST instead of FIN."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()


def test_a_reset_reads_as_the_peer_going_away(acceptor):
    """ECONNRESET has to map to ConnectionClosed, so a phone that drops off the
    network reads as peer_closed rather than as a fault at our end."""
    raw = socket.create_connection(("127.0.0.1", acceptor.port), timeout=5.0)
    server = acceptor.accept(timeout=5.0)
    assert server is not None
    try:
        server.send_all(b"x" * 65536)  # ensure there is data to reset against
        reset_immediately(raw)
        with pytest.raises(ConnectionClosed):
            for _ in range(50):
                server.recv_exact(1)
                time.sleep(0.02)
    finally:
        server.close()


def test_a_reset_ends_a_session_as_peer_closed(acceptor):
    raw = socket.create_connection(("127.0.0.1", acceptor.port), timeout=5.0)
    server = acceptor.accept(timeout=5.0)
    session = Session(server, session_id=1, heartbeat_s=0.05, stall_timeout_s=None).start()
    try:
        reset_immediately(raw)
        assert wait_until(lambda: session.is_closed)
        assert session.end_reason is SessionEndReason.PEER_CLOSED
    finally:
        session.close()


def test_the_mapped_errno_set_is_exactly_the_confirmed_list():
    """A signed-off parameter, so the whole set is pinned. Three of the five
    could previously be deleted with the suite green, because only the two with
    behavioural tests were named here."""
    import errno as errno_module

    assert PEER_GONE_ERRNOS == frozenset(
        {
            errno_module.ECONNRESET,
            errno_module.EPIPE,
            errno_module.ECONNABORTED,
            errno_module.ENOTCONN,
            errno_module.ESHUTDOWN,
        }
    )


@pytest.mark.parametrize(
    "code",
    ["ECONNRESET", "EPIPE", "ECONNABORTED", "ENOTCONN", "ESHUTDOWN"],
)
def test_a_peer_gone_errno_becomes_connection_closed(code):
    import errno as errno_module

    number = getattr(errno_module, code)
    translated = _translate(OSError(number, "injected"), "recv")
    assert isinstance(translated, ConnectionClosed), (code, translated)


@pytest.mark.parametrize("code", ["ENOBUFS", "EHOSTUNREACH", "ENOMEM", "EACCES"])
def test_a_local_failure_stays_an_oserror(code):
    """Otherwise a Jetson out of buffers would look exactly like the phone
    hanging up, and transport_error would stop meaning anything."""
    import errno as errno_module

    number = getattr(errno_module, code)
    translated = _translate(OSError(number, "injected"), "send")
    assert not isinstance(translated, ConnectionClosed)
    assert isinstance(translated, OSError)


def test_retryable_accept_errnos_are_disjoint_from_fatal_ones():
    import errno as errno_module

    assert errno_module.ECONNABORTED in ACCEPT_RETRY_ERRNOS
    assert errno_module.EINTR in ACCEPT_RETRY_ERRNOS
    assert errno_module.EMFILE in ACCEPT_TRANSIENT_ERRNOS
    assert not (ACCEPT_RETRY_ERRNOS & ACCEPT_TRANSIENT_ERRNOS)


class FlakyServerSocket:
    """Wraps the listening socket and fails the first `failures` accepts."""

    def __init__(self, inner, error, failures=1):
        """failures=-1 fails forever."""
        self._inner = inner
        self._error = error
        self.remaining = failures

    def accept(self):
        if self.remaining != 0:
            if self.remaining > 0:
                self.remaining -= 1
            raise self._error
        return self._inner.accept()

    def settimeout(self, timeout):
        self._inner.settimeout(timeout)

    def shutdown(self, how):
        self._inner.shutdown(how)

    def close(self):
        self._inner.close()

    def getsockname(self):
        return self._inner.getsockname()


@pytest.mark.parametrize(
    "code", ["ECONNABORTED", "EINTR", "EMFILE"], ids=["ECONNABORTED", "EINTR", "EMFILE"]
)
def test_a_retryable_accept_failure_does_not_kill_the_acceptor(acceptor, code):
    """The only caller is an accept loop that treats any error as terminal, and
    it lives in a module this task must not change. ECONNABORTED is what a
    client resetting between SYN and accept produces -- the traffic an
    unbounded reconnect loop on a flaky link generates -- so one occurrence
    used to stop the Jetson accepting for the rest of the drive."""
    import errno as errno_module

    acceptor._server = FlakyServerSocket(
        acceptor._server, OSError(getattr(errno_module, code), "injected")
    )
    client = dial("127.0.0.1", acceptor.port)
    try:
        accepted = acceptor.accept(timeout=5.0)
        assert accepted is not None, "a retryable failure was treated as fatal"
        assert acceptor.transient_accept_errors == 1
        client.send_all(b"ping")
        assert accepted.recv_exact(4) == b"ping"
        accepted.close()
    finally:
        client.close()


def test_a_fatal_accept_failure_still_propagates_as_an_oserror(acceptor):
    """Retrying everything would hide a real fault behind an infinite loop."""
    acceptor._server = FlakyServerSocket(acceptor._server, OSError(9, "bad file descriptor"))
    with pytest.raises(OSError) as caught:
        acceptor.accept(timeout=1.0)
    assert not isinstance(caught.value, ConnectionClosed)


def test_a_retry_respects_the_accept_deadline(acceptor):
    """The retry loop must not outlive the timeout it was given."""
    import errno as errno_module

    # Never stops failing, so the deadline is the only way out. A finite count
    # would just be exhausted and the test would pass either way.
    acceptor._server = FlakyServerSocket(
        acceptor._server, OSError(errno_module.ECONNABORTED, "injected"), failures=-1
    )
    outcome: list[object] = []
    started = time.monotonic()
    worker = threading.Thread(
        target=lambda: outcome.append(acceptor.accept(timeout=0.3)), daemon=True
    )
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "the retry loop outlived the deadline it was given"
    assert outcome == [None]
    assert time.monotonic() - started < 3.0
    assert acceptor.transient_accept_errors > 0


# -- the whole transport over a real socket ----------------------------------


def listener_for(acceptor, **kwargs):
    options = {"heartbeat_s": None, "stall_timeout_s": None, "accept_poll_s": 0.02}
    options.update(kwargs)
    return TransportListener(acceptor, JETSON, **options).start()


def wait_for_event(listener, kind, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = listener.next_event(timeout=0.05)
        if isinstance(event, kind):
            return event
    return None


def test_a_session_runs_end_to_end_over_a_real_socket(acceptor):
    listener = listener_for(acceptor)
    try:
        connection = dial("127.0.0.1", acceptor.port)
        perform_handshake(connection, PHONE)
        started = wait_for_event(listener, SessionStarted)
        assert started is not None
        assert started.handshake.remote.device_id == PHONE.device_id

        phone = Session(connection, session_id=99, heartbeat_s=None, stall_timeout_s=None).start()
        try:
            phone.send(Channel.CAMERA, b"\x5a" * 200_000, {"probe": 1})
            message = started.session.recv(Channel.CAMERA, timeout=10.0)
            assert message is not None
            assert len(message.payload) == 200_000
            assert message.extensions == {"probe": 1}

            started.session.send(Channel.ADVISORY, b"cap=25.0")
            back = phone.recv(Channel.ADVISORY, timeout=10.0)
            assert back is not None and back.payload == b"cap=25.0"
        finally:
            phone.close()
    finally:
        listener.stop()


def test_a_silent_phone_is_reaped_over_a_real_socket(acceptor):
    listener = listener_for(acceptor, stall_timeout_s=0.4)
    try:
        connection = dial("127.0.0.1", acceptor.port)
        perform_handshake(connection, PHONE)
        assert wait_for_event(listener, SessionStarted) is not None
        started = time.monotonic()
        ended = wait_for_event(listener, SessionEnded, timeout=6.0)
        assert ended is not None
        assert ended.reason is SessionEndReason.STALLED
        # A 7.5x upper bound is not a measurement. The timer's own accuracy is
        # covered in test_transport_session.py; what this pins is that a real
        # socket does not add a multiple of it.
        elapsed = time.monotonic() - started
        assert 0.35 <= elapsed <= 1.2, f"reaped after {elapsed:.2f}s on a 0.4s timeout"
        connection.close()
    finally:
        listener.stop()


def test_displacement_works_over_a_real_socket(acceptor):
    listener = listener_for(acceptor)
    try:
        first = dial("127.0.0.1", acceptor.port)
        perform_handshake(first, Hello("phone-a", Role.PHONE))
        first_started = wait_for_event(listener, SessionStarted)
        assert first_started is not None

        second = dial("127.0.0.1", acceptor.port)
        perform_handshake(second, Hello("phone-b", Role.PHONE))
        ended = wait_for_event(listener, SessionEnded)
        assert ended is not None
        assert ended.reason is SessionEndReason.DISPLACED
        assert ended.session_id == first_started.session.session_id

        second_started = wait_for_event(listener, SessionStarted)
        assert second_started is not None
        first.close()
        second.close()
    finally:
        listener.stop()


# -- lifetime ----------------------------------------------------------------


def test_fifty_connect_disconnect_cycles_leak_no_descriptors(acceptor):
    before = open_descriptor_count()
    for _ in range(50):
        client = dial("127.0.0.1", acceptor.port)
        server = acceptor.accept(timeout=5.0)
        assert server is not None
        client.send_all(b"ping")
        assert server.recv_exact(4) == b"ping"
        client.close()
        server.close()
    if before is not None:
        assert wait_until(lambda: open_descriptor_count() <= before + 2), (
            f"descriptors grew from {before} to {open_descriptor_count()}"
        )


# -- the acceptor's own close, and its counters -----------------------------


class RecordingServerSocket:
    """Records shutdown/close order on the listening socket."""

    def __init__(self, address=("127.0.0.1", 1)):
        self.calls: list[str] = []
        self._address = address

    def settimeout(self, timeout):
        return None

    def shutdown(self, how):
        self.calls.append("shutdown")

    def close(self):
        self.calls.append("close")

    def getsockname(self):
        return self._address


def test_the_acceptor_shuts_its_socket_down_before_closing_it(acceptor):
    """Same requirement as TcpConnection, same platform blindness: on macOS
    close() alone already wakes a blocked accept, so only the call order can be
    asserted here. The stated reason for the change is Linux parity with
    LoopbackAcceptor on the documented Acceptor protocol."""
    recorder = RecordingServerSocket()
    acceptor._server = recorder
    acceptor.close()
    assert recorder.calls == ["shutdown", "close"], recorder.calls


def test_the_acceptor_still_closes_when_shutdown_is_refused(acceptor):
    class RefusingShutdown(RecordingServerSocket):
        def shutdown(self, how):
            self.calls.append("shutdown-failed")
            raise OSError(57, "socket is not connected")

    recorder = RefusingShutdown()
    acceptor._server = recorder
    acceptor.close()
    assert recorder.calls == ["shutdown-failed", "close"], recorder.calls


class AlwaysFailingServer(RecordingServerSocket):
    def __init__(self, error, address=("127.0.0.1", 1)):
        super().__init__(address)
        self.error = error
        self.attempts = 0

    def accept(self):
        self.attempts += 1
        raise self.error


def test_a_retry_storm_does_not_become_a_busy_spin(acceptor):
    """The module docstring names "spun at millions of calls per second" as the
    pathology it exists to prevent. Without a pause on the retry branch, the
    accept loop reached 2.2 million calls a second on a persistent
    ECONNABORTED -- the same pathology, one errno over."""
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.ECONNABORTED, "injected"))
    started = time.monotonic()
    assert acceptor.accept(timeout=0.2) is None
    elapsed = time.monotonic() - started
    rate = acceptor.transient_accept_errors / elapsed
    assert rate < 20_000, f"{rate:,.0f} retries/sec is a spin, not a retry"
    assert acceptor.transient_accept_errors > 0


@pytest.mark.parametrize("timeout", [0.01, 0.05])
def test_a_transient_pause_never_outlives_the_deadline(acceptor, timeout):
    """The pause was unconditional, so a 0.05s sleep overran a 0.01s deadline
    fivefold -- and under a storm it set the accept loop's cadence instead of
    the caller's poll interval."""
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.EMFILE, "injected"))
    started = time.monotonic()
    assert acceptor.accept(timeout=timeout) is None
    overshoot = time.monotonic() - started - timeout
    assert overshoot < 0.03, f"overshot its {timeout}s deadline by {overshoot:.3f}s"


def test_the_transient_branch_pauses_at_all(acceptor):
    """EMFILE retried flat out would spin the way the retry branch used to."""
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.EMFILE, "injected"))
    started = time.monotonic()
    acceptor.accept(timeout=0.25)
    elapsed = time.monotonic() - started
    rate = acceptor.transient_accept_errors / elapsed
    assert rate < 200, f"{rate:,.0f} EMFILE retries/sec means the pause is gone"


def test_retries_are_counted_per_errno_and_consecutively(acceptor):
    """One number cannot tell our own descriptor leak (EMFILE) from somebody
    else's (ENFILE), and retrying either indefinitely would otherwise turn our
    own leak into a passing weather condition."""
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.EMFILE, "injected"))
    acceptor.accept(timeout=0.2)
    stats = acceptor.stats()
    assert stats["accept_errors_by_errno"] == {"EMFILE": acceptor.transient_accept_errors}
    assert stats["max_consecutive_accept_errors"] == acceptor.transient_accept_errors
    assert stats["transient_accept_errors"] > 0
    assert stats["address"] == list(acceptor.address)


def test_a_successful_accept_resets_the_consecutive_count(acceptor):
    """The high water mark is per accept() call, not cumulative.

    Two separate calls of two failures each give 2, not 4. Written first as a
    test for a reset that does not exist: `consecutive` is a local, so it is
    re-initialised every call and no reset before the return could ever be
    observed. That line was dead and is gone.
    """
    import errno as errno_module

    error = OSError(errno_module.ECONNABORTED, "injected")
    accepted = []
    for _ in range(2):
        acceptor._server = FlakyServerSocket(acceptor._server, error, failures=2)
        client = dial("127.0.0.1", acceptor.port)
        try:
            connection = acceptor.accept(timeout=5.0)
            assert connection is not None
            accepted.append(connection)
        finally:
            client.close()
        # Unwrap for the next round, so each run starts from a clean socket.
        acceptor._server = acceptor._server._inner
    try:
        assert acceptor.transient_accept_errors == 4
        assert acceptor.max_consecutive_accept_errors == 2, (
            "the high water mark accumulated across accept() calls"
        )
        assert acceptor.stats()["accept_errors_by_errno"] == {"ECONNABORTED": 4}
    finally:
        for connection in accepted:
            connection.close()


def test_a_close_during_a_retry_ends_the_accept_cleanly(acceptor):
    """A close() landing mid-retry ends the accept as ConnectionClosed.

    The outcome is delivered by the exception handler's own closed check, not by
    the check at the top of the loop -- that one became redundant when
    settimeout moved inside the guard, and no test can distinguish them. It is
    kept as defence in depth, because the redundancy rests on CPython raising
    EBADF from settimeout on a closed socket, which is an implementation detail
    rather than a documented guarantee.
    """
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.EMFILE, "injected"))
    outcome: list[object] = []

    def do_accept():
        try:
            outcome.append(acceptor.accept(timeout=5.0))
        except ConnectionClosed:
            outcome.append("ConnectionClosed")
        except OSError as exc:
            outcome.append(f"OSError:{exc.errno}")

    worker = threading.Thread(target=do_accept, daemon=True)
    worker.start()
    time.sleep(0.15)
    acceptor.close()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert outcome == ["ConnectionClosed"], outcome


def test_a_settimeout_failure_is_classified_like_any_other(acceptor):
    """settimeout used to sit outside the guard, so a close() landing there
    escaped the errno classification the whole retry fix is about."""

    class ClosingSettimeout(RecordingServerSocket):
        """Fails the way a concurrent close() does: the flag flips and the call
        raises. Setting the flag before calling accept() instead would trip the
        check at the top of the method and never reach settimeout at all --
        which is what the first version of this test did."""

        def __init__(self, owner):
            super().__init__()
            self._owner = owner

        def settimeout(self, timeout):
            self._owner._closed = True
            raise OSError(9, "bad file descriptor")

        def accept(self):  # pragma: no cover - never reached
            raise AssertionError("accept should not be reached")

    acceptor._server = ClosingSettimeout(acceptor)
    with pytest.raises(ConnectionClosed):
        acceptor.accept(timeout=1.0)


def test_a_settimeout_failure_while_open_propagates_as_an_oserror(acceptor):
    class FailingSettimeout(RecordingServerSocket):
        def settimeout(self, timeout):
            raise OSError(22, "invalid argument")

        def accept(self):  # pragma: no cover
            raise AssertionError("accept should not be reached")

    acceptor._server = FailingSettimeout()
    with pytest.raises(OSError) as caught:
        acceptor.accept(timeout=1.0)
    assert not isinstance(caught.value, ConnectionClosed)


class SlowFailingServer(RecordingServerSocket):
    """Burns most of the accept window, then fails -- which is what a real
    accept that waits and then gets ECONNABORTED does."""

    def __init__(self, delay, error, address=("127.0.0.1", 1)):
        super().__init__(address)
        self.delay = delay
        self.error = error

    def accept(self):
        time.sleep(self.delay)
        raise self.error


@pytest.mark.parametrize("timeout,delay", [(0.05, 0.045), (0.10, 0.095)])
def test_a_late_arriving_failure_still_respects_the_deadline(acceptor, timeout, delay):
    """The pause was bounded against a `remaining` computed before accept()
    blocked, so a failure arriving late in the window still overshot by a whole
    pause -- 130% over a 50 ms deadline. The always-fails fixture cannot see
    this, because it raises instantly and the stale clock is still fresh."""
    import errno as errno_module

    acceptor._server = SlowFailingServer(delay, OSError(errno_module.EMFILE, "injected"))
    started = time.monotonic()
    assert acceptor.accept(timeout=timeout) is None
    overshoot = time.monotonic() - started - timeout
    assert overshoot < 0.02, f"overshot a {timeout}s deadline by {overshoot:.3f}s"


def test_the_error_span_is_recorded_not_just_the_high_water_mark(acceptor):
    """The high water mark is per accept() call, so it reads as the caller's
    poll interval divided by the pause -- the same number whether a storm
    lasted 200 ms or four hours. The span carries the duration."""
    import errno as errno_module

    acceptor._server = AlwaysFailingServer(OSError(errno_module.EMFILE, "injected"))
    acceptor.accept(timeout=0.2)
    stats = acceptor.stats()
    assert stats["first_accept_error_mono_ns"] is not None
    assert stats["last_accept_error_mono_ns"] >= stats["first_accept_error_mono_ns"]
    assert stats["accept_error_span_s"] >= 0.0
    assert stats["accept_error_span_s"] < 1.0


def test_stats_is_callable_while_accepts_are_failing(acceptor):
    """What this pins: stats() returns well-formed data under concurrent
    accepts, and never raises.

    What it cannot pin, stated so nobody trusts it to: sorted() over the live
    dict only raises when the dict changes *size* mid-iteration, and at most six
    keys can ever exist -- the two errno frozensets -- so the race is six events
    in a process lifetime. A test that reliably hit it would have to be flaky,
    which is worse than an untested one-word snapshot. The dict() copy stays as
    defence.
    """
    import errno as errno_module
    import itertools

    codes = itertools.cycle(
        [errno_module.EMFILE, errno_module.ENFILE, errno_module.ECONNABORTED, errno_module.EINTR]
    )

    class CyclingFailure(RecordingServerSocket):
        def accept(self):
            raise OSError(next(codes), "injected")

    acceptor._server = CyclingFailure()
    stop = threading.Event()

    def keep_failing():
        while not stop.is_set():
            try:
                acceptor.accept(timeout=0.05)
            except BaseException:
                return

    worker = threading.Thread(target=keep_failing, daemon=True)
    worker.start()
    try:
        for _ in range(400):
            snapshot = acceptor.stats()
            assert isinstance(snapshot["accept_errors_by_errno"], dict)
            assert snapshot["transient_accept_errors"] >= 0
    finally:
        stop.set()
        worker.join(timeout=3.0)


def test_close_is_noticed_within_one_internal_accept_poll(acceptor):
    """TCP's own bound. A timed accept cannot be woken by a close -- CPython
    waits in select() and closing the fd does not break that wait -- so the
    acceptor caps its own socket timeout and re-checks. Without that cap,
    accept(timeout=5) ignored close for the full five seconds."""
    from transport.tcp import INTERNAL_ACCEPT_POLL_S

    released = threading.Event()

    def wait_to_accept():
        try:
            acceptor.accept(timeout=5.0)
        except BaseException:
            pass
        released.set()

    waiter = threading.Thread(target=wait_to_accept, daemon=True)
    waiter.start()
    time.sleep(0.15)
    assert not released.is_set()
    started = time.monotonic()
    acceptor.close()
    assert released.wait(timeout=5.0)
    elapsed = time.monotonic() - started
    assert elapsed < INTERNAL_ACCEPT_POLL_S * 3, (
        f"released after {elapsed:.3f}s, more than three internal polls"
    )


def test_a_timed_accept_still_honours_its_own_timeout(acceptor):
    """The internal poll must not turn a bounded wait into an unbounded one."""
    started = time.monotonic()
    assert acceptor.accept(timeout=0.3) is None
    elapsed = time.monotonic() - started
    assert 0.28 <= elapsed < 0.6, f"a 0.3s accept took {elapsed:.3f}s"


def test_a_blocking_accept_is_released_by_close(acceptor):
    released = threading.Event()

    def wait_forever():
        try:
            acceptor.accept(timeout=None)
        except BaseException:
            pass
        released.set()

    waiter = threading.Thread(target=wait_forever, daemon=True)
    waiter.start()
    time.sleep(0.15)
    assert not released.is_set()
    acceptor.close()
    assert released.wait(timeout=5.0), "a blocking accept was never released"
