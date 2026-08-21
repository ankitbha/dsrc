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
    """A listener that cannot be stopped cannot be stopped."""
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
    under_test.acceptor.close()
    assert released.wait(timeout=5.0), "close() did not release a waiting accept"


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
