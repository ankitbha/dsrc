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
    DEFAULT_PORT,
    PEER_GONE_ERRNOS,
    TcpAcceptor,
    TcpConnection,
    apply_socket_options,
    dial,
)

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
    started = time.monotonic()
    assert acceptor.accept(timeout=0.2) is None
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


def test_every_mapped_errno_is_a_peer_gone_condition():
    import errno as errno_module

    assert errno_module.ECONNRESET in PEER_GONE_ERRNOS
    assert errno_module.EPIPE in PEER_GONE_ERRNOS
    # A local resource failure must not be mistaken for the peer leaving.
    assert errno_module.ENOBUFS not in PEER_GONE_ERRNOS
    assert errno_module.EHOSTUNREACH not in PEER_GONE_ERRNOS


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
        assert 0.3 <= time.monotonic() - started <= 3.0
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
    before_threads = threading.active_count()
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
    assert wait_until(lambda: threading.active_count() <= before_threads)
