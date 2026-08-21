"""Unit tests for the loopback backend.

What is being pinned here is the ByteConnection contract every other backend
has to honor: exact reads or an exception, never a short read, and a close that
unblocks whoever is waiting.
"""

from __future__ import annotations

import threading
import time

import pytest

from transport.connection import ConnectionClosed
from transport.loopback import LoopbackAcceptor, loopback_pair


def test_bytes_flow_in_both_directions():
    left, right = loopback_pair()
    left.send_all(b"up")
    right.send_all(b"down")
    assert right.recv_exact(2) == b"up"
    assert left.recv_exact(4) == b"down"


def test_reads_are_reassembled_across_writes():
    left, right = loopback_pair()
    left.send_all(b"ab")
    left.send_all(b"cd")
    assert right.recv_exact(3) == b"abc"
    assert right.recv_exact(1) == b"d"


def test_zero_length_read_is_empty_and_does_not_block():
    left, _ = loopback_pair()
    assert left.recv_exact(0) == b""


def test_reader_blocks_until_data_arrives():
    left, right = loopback_pair()
    got = []

    reader = threading.Thread(target=lambda: got.append(right.recv_exact(3)), daemon=True)
    reader.start()
    time.sleep(0.05)
    assert got == []
    left.send_all(b"xyz")
    reader.join(timeout=2.0)
    assert got == [b"xyz"]


def test_close_surfaces_as_eof_to_the_peer():
    left, right = loopback_pair()
    left.close()
    with pytest.raises(ConnectionClosed):
        right.recv_exact(1)


def test_close_mid_read_raises_rather_than_returning_short():
    """A short read would hand a truncated header to the JSON parser."""
    left, right = loopback_pair()
    left.send_all(b"ab")
    error: list[Exception] = []

    def read_five():
        try:
            right.recv_exact(5)
        except ConnectionClosed as exc:
            error.append(exc)

    reader = threading.Thread(target=read_five, daemon=True)
    reader.start()
    time.sleep(0.05)
    left.close()
    reader.join(timeout=2.0)
    assert len(error) == 1
    assert "2 of 5" in str(error[0])


def test_close_is_idempotent():
    left, _ = loopback_pair()
    left.close()
    left.close()


def test_send_after_close_raises():
    left, right = loopback_pair()
    right.close()
    with pytest.raises(ConnectionClosed):
        left.send_all(b"x")


def test_capacity_applies_backpressure():
    """Without a bounded buffer the overflow policies can never trigger."""
    left, right = loopback_pair(max_buffer_bytes=8)
    done = threading.Event()

    def send_more_than_fits():
        left.send_all(b"0123456789ABCDEF")
        done.set()

    writer = threading.Thread(target=send_more_than_fits, daemon=True)
    writer.start()
    time.sleep(0.1)
    assert not done.is_set()
    assert left.unread_bytes == 8

    assert right.recv_exact(16) == b"0123456789ABCDEF"
    writer.join(timeout=2.0)
    assert done.is_set()


def test_a_read_larger_than_the_buffer_completes():
    """The whole transport rests on this. A frame payload is far larger than
    any socket buffer, so recv_exact has to accumulate across arrivals rather
    than wait for the whole request to be buffered at once -- which would
    deadlock, since the writer cannot deposit bytes the reader will not take."""
    left, right = loopback_pair(max_buffer_bytes=64)
    payload = bytes(range(256)) * 40  # 10240 bytes through a 64-byte buffer
    writer = threading.Thread(target=lambda: left.send_all(payload), daemon=True)
    writer.start()
    assert right.recv_exact(len(payload)) == payload
    writer.join(timeout=2.0)
    assert not writer.is_alive()


def test_interleaved_reads_and_writes_survive_a_tiny_buffer():
    left, right = loopback_pair(max_buffer_bytes=16)
    sent = [f"message-{index:04d}".encode() for index in range(50)]
    writer = threading.Thread(
        target=lambda: [left.send_all(chunk) for chunk in sent], daemon=True
    )
    writer.start()
    got = b"".join(right.recv_exact(len(chunk)) for chunk in sent)
    writer.join(timeout=5.0)
    assert got == b"".join(sent)


def test_unread_bytes_tracks_the_sender_side():
    left, right = loopback_pair()
    assert left.unread_bytes == 0
    left.send_all(b"abcd")
    assert left.unread_bytes == 4
    right.recv_exact(4)
    assert left.unread_bytes == 0


# -- acceptor ----------------------------------------------------------------


def test_acceptor_pairs_a_client_with_the_listener():
    acceptor = LoopbackAcceptor()
    client = acceptor.connect("phone")
    server = acceptor.accept(timeout=1.0)
    assert server is not None
    client.send_all(b"hi")
    assert server.recv_exact(2) == b"hi"
    server.send_all(b"yo")
    assert client.recv_exact(2) == b"yo"


def test_accept_returns_none_on_timeout():
    start = time.monotonic()
    assert LoopbackAcceptor().accept(timeout=0.15) is None
    assert time.monotonic() - start >= 0.1


def test_accept_waits_for_a_late_connection():
    acceptor = LoopbackAcceptor()
    threading.Timer(0.1, lambda: acceptor.connect("late")).start()
    assert acceptor.accept(timeout=2.0) is not None


def test_connections_queue_and_accept_in_order():
    acceptor = LoopbackAcceptor()
    first = acceptor.connect("one")
    second = acceptor.connect("two")
    first.send_all(b"1")
    second.send_all(b"2")
    assert acceptor.accept(timeout=1.0).recv_exact(1) == b"1"
    assert acceptor.accept(timeout=1.0).recv_exact(1) == b"2"


def test_closed_acceptor_refuses_both_sides():
    acceptor = LoopbackAcceptor()
    acceptor.close()
    with pytest.raises(ConnectionClosed):
        acceptor.connect()
    with pytest.raises(ConnectionClosed):
        acceptor.accept(timeout=0.1)


def test_close_unblocks_a_waiting_accept():
    acceptor = LoopbackAcceptor()
    error: list[Exception] = []

    def wait_to_accept():
        try:
            acceptor.accept(timeout=5.0)
        except ConnectionClosed as exc:
            error.append(exc)

    waiter = threading.Thread(target=wait_to_accept, daemon=True)
    waiter.start()
    time.sleep(0.05)
    acceptor.close()
    waiter.join(timeout=2.0)
    assert len(error) == 1


def test_close_wakes_a_waiting_accept_rather_than_letting_it_poll():
    """Loopback's own bound, tight enough to require the notify.

    The shared Acceptor contract can only assert a generous bound, because TCP
    releases by an internal poll and loopback by a notify. This is the half that
    holds loopback to its mechanism: without notify_all in close(), the waiting
    accept would be released by its own 50 ms condition wait instead.
    """
    acceptor = LoopbackAcceptor()
    released = threading.Event()

    def wait_to_accept():
        try:
            acceptor.accept(timeout=5.0)
        except BaseException:
            pass
        released.set()

    waiter = threading.Thread(target=wait_to_accept, daemon=True)
    waiter.start()
    time.sleep(0.1)
    assert not released.is_set()
    started = time.monotonic()
    acceptor.close()
    assert released.wait(timeout=5.0)
    elapsed = time.monotonic() - started
    assert elapsed < 0.02, f"released after {elapsed * 1000:.0f} ms: polled, not woken"
