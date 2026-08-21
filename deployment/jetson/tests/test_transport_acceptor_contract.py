"""The Acceptor protocol, run against every implementation.

`ByteConnection` got a conformance suite because backend divergence was the
recurring defect in this task. `Acceptor` has three implementations and had
none -- and immediately diverged: TcpAcceptor returned None from
`accept(timeout=0)` with a connection sitting in the backlog, while
LoopbackAcceptor returned the connection. `timeout=0.0` is this codebase's idiom
for "poll now" (`next_event`, `Session.recv`), so the two answers were both
plausible and only one can be right.

Task 40's USB acceptor is the next one written; it inherits this.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import pytest

from transport.connection import ConnectionClosed
from transport.loopback import LoopbackAcceptor
from transport.tcp import TcpAcceptor, dial


@dataclass
class AcceptorUnderTest:
    name: str
    acceptor: object
    connect: Callable[[], object]
    dispose: Callable[[], None]


def loopback_acceptor() -> AcceptorUnderTest:
    acceptor = LoopbackAcceptor()
    return AcceptorUnderTest(
        "loopback", acceptor, lambda: acceptor.connect("peer"), acceptor.close
    )


def tcp_acceptor() -> AcceptorUnderTest:
    acceptor = TcpAcceptor("127.0.0.1", 0)
    opened: list[object] = []

    def connect():
        connection = dial("127.0.0.1", acceptor.port)
        opened.append(connection)
        return connection

    def dispose():
        for connection in opened:
            connection.close()
        acceptor.close()

    return AcceptorUnderTest("tcp", acceptor, connect, dispose)


ACCEPTORS = {"loopback": loopback_acceptor, "tcp": tcp_acceptor}


@pytest.fixture(params=sorted(ACCEPTORS), ids=sorted(ACCEPTORS))
def under_test(request):
    made = ACCEPTORS[request.param]()
    try:
        yield made
    finally:
        made.dispose()


def test_a_waiting_connection_is_accepted(under_test):
    under_test.connect()
    assert under_test.acceptor.accept(timeout=5.0) is not None


def test_accept_returns_none_on_timeout_with_nothing_waiting(under_test):
    started = time.monotonic()
    assert under_test.acceptor.accept(timeout=0.15) is None
    assert time.monotonic() - started >= 0.1


def test_a_zero_timeout_polls_rather_than_declining(under_test):
    """Both answers were defensible; only one can be the protocol."""
    under_test.connect()
    # Give a real socket a moment to complete the TCP handshake.
    deadline = time.monotonic() + 2.0
    accepted = None
    while accepted is None and time.monotonic() < deadline:
        accepted = under_test.acceptor.accept(timeout=0)
        if accepted is None:
            time.sleep(0.01)
    assert accepted is not None, "a zero timeout declined a connection that was waiting"


def test_a_zero_timeout_returns_none_when_nothing_is_waiting(under_test):
    started = time.monotonic()
    assert under_test.acceptor.accept(timeout=0) is None
    assert time.monotonic() - started < 0.5, "a zero timeout blocked"


def test_accept_after_close_raises(under_test):
    under_test.acceptor.close()
    with pytest.raises(ConnectionClosed):
        under_test.acceptor.accept(timeout=0.1)


def test_close_is_idempotent(under_test):
    under_test.acceptor.close()
    under_test.acceptor.close()


def test_close_releases_a_waiting_accept(under_test):
    """A listener that cannot be stopped cannot be stopped.

    The shared bound is generous, because the two implementations release by
    genuinely different mechanisms: loopback is woken by a notify (0.000 s),
    while TCP notices within one internal accept poll (0.048 s) -- CPython
    waits in select() under a socket timeout and closing the fd does not break
    that wait, so there is nothing to be woken by. A single tight bound cannot
    hold both to their mechanism, so what this asserts is the property every
    implementation must have: close is noticed, not ignored. Before the internal
    poll existed, TCP took 4.892 s here, i.e. its caller's whole timeout.

    The tighter per-implementation bounds live with each implementation:
    test_transport_loopback.py holds loopback to its notify, and
    test_transport_tcp.py holds TCP to one internal poll.
    """
    released = threading.Event()

    def wait_to_accept():
        try:
            under_test.acceptor.accept(timeout=5.0)
        except BaseException:
            pass
        released.set()

    waiter = threading.Thread(target=wait_to_accept, daemon=True)
    waiter.start()
    time.sleep(0.1)
    assert not released.is_set(), "the accept did not block in the first place"
    started = time.monotonic()
    under_test.acceptor.close()
    assert released.wait(timeout=5.0), "close() did not release a waiting accept"
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, (
        f"released after {elapsed:.3f}s, which is the caller's timeout expiring "
        f"rather than close being noticed"
    )


def test_connections_are_accepted_in_order(under_test):
    first = under_test.connect()
    second = under_test.connect()
    first.send_all(b"1")
    second.send_all(b"2")
    accepted_first = under_test.acceptor.accept(timeout=5.0)
    accepted_second = under_test.acceptor.accept(timeout=5.0)
    assert accepted_first is not None and accepted_second is not None
    assert accepted_first.recv_exact(1) == b"1"
    assert accepted_second.recv_exact(1) == b"2"


def test_an_accepted_connection_satisfies_the_byte_connection_contract(under_test):
    """The two conformance suites were disjoint, so nothing checked that what
    accept() hands back is a compliant ByteConnection. A USB acceptor returning
    a connection that returns b"" at EOF would pass every test in this file."""
    client = under_test.connect()
    server = under_test.acceptor.accept(timeout=5.0)
    assert server is not None

    assert isinstance(server.peer, str) and server.peer
    assert server.recv_exact(0) == b""

    payload = bytes(range(256)) * 8
    sender = threading.Thread(target=lambda: client.send_all(payload), daemon=True)
    sender.start()
    assert server.recv_exact(len(payload)) == payload
    sender.join(timeout=5.0)

    # raises at EOF rather than returning short or empty
    client.close()
    with pytest.raises(ConnectionClosed):
        server.recv_exact(1)

    server.close()
    server.close()  # idempotent
