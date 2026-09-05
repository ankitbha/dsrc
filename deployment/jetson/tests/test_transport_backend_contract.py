"""The contract every ByteConnection backend must satisfy, run against all of them.

These requirements were prose in connection.py, discovered while validating the
transport above it. Prose is not enough: each one is something a backend author
can plausibly get wrong, and two of them produced pathological failures when
they were got wrong -- a reader spinning at millions of calls per second, and a
thread abandoned per connection attempt.

Parametrizing over backends means the USB backend inherits the whole suite, and
it also cross-checks the loopback backend that shipped earlier: if loopback
fails a check here, the contract text and the shipped code disagree.

The `usb` backend is gated on a real device being attached (`attached_serial`),
skipping with a stated reason when not, because it is the one entry here whose
setup shells a real `adb reverse`. The bytes themselves are exchanged over a
local dial to the acceptor's bound port -- exactly what a phone's dial through
the reverse mapping lands as on this side -- so what the gate proves is that
`UsbAcceptor` really can establish and tear down a mapping against hardware;
the `ByteConnection` it hands back is a `TcpConnection`, proved generically by
this same suite under the `tcp` backend above.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import pytest

from transport.connection import ByteConnection, ConnectionClosed
from transport.loopback import loopback_pair
from transport.tcp import TcpAcceptor, dial
from transport.usb import UsbAcceptor, attached_serial


@dataclass
class Pair:
    name: str
    a: ByteConnection
    b: ByteConnection
    dispose: Callable[[], None]


def loopback_backend(**kwargs) -> Pair:
    left, right = loopback_pair(**kwargs)
    return Pair("loopback", left, right, lambda: (left.close(), right.close()))


def tcp_backend(**kwargs) -> Pair:
    acceptor = TcpAcceptor("127.0.0.1", 0, **kwargs)
    try:
        client = dial("127.0.0.1", acceptor.port)
        server = acceptor.accept(timeout=5.0)
        assert server is not None
    except BaseException:
        # `dispose` does not exist yet if `dial`/`accept`/the assert fails,
        # so nothing else would close this acceptor -- harmless for a local
        # socket, which dies with the process either way, but the pattern
        # is shared with `usb_backend` below, where it is not harmless
        # (B12, validation round 2).
        acceptor.close()
        raise

    def dispose():
        client.close()
        server.close()
        acceptor.close()

    return Pair("tcp", client, server, dispose)


def usb_backend(**kwargs) -> Pair:
    serial, reason = attached_serial()
    if serial is None:
        pytest.skip(f"no USB device attached: {reason}")
    acceptor = UsbAcceptor(serial, port=0, **kwargs)
    try:
        client = dial("127.0.0.1", acceptor.port)
        server = acceptor.accept(timeout=5.0)
        assert server is not None
    except BaseException:
        # B12: `dispose` is not defined until both calls above succeed, so
        # a failure here used to leave `adb reverse tcp:P tcp:P` on the
        # device permanently for this ephemeral P -- nothing enumerates or
        # sweeps a stray registered under a port nobody asks about again,
        # and `UsbAcceptor.close()` only ever removes its OWN spec.
        acceptor.close()
        raise

    def dispose():
        client.close()
        server.close()
        acceptor.close()

    return Pair("usb", client, server, dispose)


BACKENDS = {"loopback": loopback_backend, "tcp": tcp_backend, "usb": usb_backend}


@pytest.fixture(params=sorted(BACKENDS), ids=sorted(BACKENDS))
def pair(request):
    made = BACKENDS[request.param]()
    try:
        yield made
    finally:
        made.dispose()


# -- B12 (validation round 2): the backend factories close on setup failure --


def _this_module():
    import sys

    return sys.modules[__name__]


def test_usb_backend_closes_the_acceptor_if_dial_fails(monkeypatch):
    """A failure between constructing the acceptor and defining `dispose`
    must not leak the reverse mapping -- unlike a leaked TCP listener,
    which dies with the process, a leaked `adb reverse tcp:P tcp:P` sits on
    the device permanently for this ephemeral P, and nothing enumerates or
    sweeps a stray registered under a port nobody asks about again.
    """
    closed = []

    class FakeAcceptor:
        port = 47811

        def close(self):
            closed.append(True)

    module = _this_module()
    monkeypatch.setattr(module, "attached_serial", lambda: ("SERIALX", "ok"))
    monkeypatch.setattr(module, "UsbAcceptor", lambda *a, **k: FakeAcceptor())

    def raising_dial(host, port):
        raise ConnectionRefusedError("nothing is listening")

    monkeypatch.setattr(module, "dial", raising_dial)

    with pytest.raises(ConnectionRefusedError):
        usb_backend()
    assert closed == [True]


def test_usb_backend_closes_the_acceptor_if_accept_returns_none(monkeypatch):
    """The other half of the same failure window: `dial` succeeds but
    `accept` times out (`assert server is not None` fails) -- still before
    `dispose` exists."""
    closed = []

    class FakeAcceptor:
        port = 47811

        def accept(self, timeout=None):
            return None

        def close(self):
            closed.append(True)

    module = _this_module()
    monkeypatch.setattr(module, "attached_serial", lambda: ("SERIALX", "ok"))
    monkeypatch.setattr(module, "UsbAcceptor", lambda *a, **k: FakeAcceptor())
    monkeypatch.setattr(module, "dial", lambda host, port: object())

    with pytest.raises(AssertionError):
        usb_backend()
    assert closed == [True]


def test_tcp_backend_closes_the_acceptor_if_dial_fails(monkeypatch):
    """Same pattern, `tcp_backend`: lower stakes (a leaked local socket
    dies with the process) but the same shared code shape, fixed the same
    way."""
    real_acceptor = TcpAcceptor("127.0.0.1", 0)
    closed = []
    real_close = real_acceptor.close

    def tracking_close():
        closed.append(True)
        real_close()

    real_acceptor.close = tracking_close

    module = _this_module()
    monkeypatch.setattr(module, "TcpAcceptor", lambda *a, **k: real_acceptor)

    def raising_dial(host, port):
        raise ConnectionRefusedError("nothing is listening")

    monkeypatch.setattr(module, "dial", raising_dial)

    with pytest.raises(ConnectionRefusedError):
        tcp_backend()
    assert closed == [True]


# -- the basics the session assumes -----------------------------------------


def test_peer_is_a_non_empty_label(pair):
    assert isinstance(pair.a.peer, str) and pair.a.peer


def test_bytes_flow_in_both_directions(pair):
    pair.a.send_all(b"up")
    pair.b.send_all(b"down")
    assert pair.b.recv_exact(2) == b"up"
    assert pair.a.recv_exact(4) == b"down"


def test_a_zero_length_read_is_empty_and_does_not_block(pair):
    started = time.monotonic()
    assert pair.a.recv_exact(0) == b""
    assert time.monotonic() - started < 0.5


def test_reads_reassemble_across_separate_writes(pair):
    pair.a.send_all(b"ab")
    pair.a.send_all(b"cd")
    assert pair.b.recv_exact(3) == b"abc"
    assert pair.b.recv_exact(1) == b"d"


def test_a_large_payload_transfers_intact(pair):
    """Bigger than any plausible socket buffer, so send_all must be handling
    partial writes rather than assuming one call suffices."""
    payload = bytes(range(256)) * 8192  # 2 MiB
    sender = threading.Thread(target=lambda: pair.a.send_all(payload), daemon=True)
    sender.start()
    assert pair.b.recv_exact(len(payload)) == payload
    sender.join(timeout=10.0)
    assert not sender.is_alive()


# -- requirement 1: raise, never return short or empty ----------------------


def test_recv_raises_at_eof_rather_than_returning_empty(pair):
    """socket.recv returns b"" at EOF. A backend that passes that through makes
    the session reader treat it as progress, and it spins."""
    pair.a.close()
    with pytest.raises(ConnectionClosed):
        pair.b.recv_exact(1)


def test_recv_raises_rather_than_returning_short(pair):
    """A short read would hand a truncated header to the JSON parser."""
    pair.a.send_all(b"ab")
    error: list[BaseException] = []

    def read_five():
        try:
            pair.b.recv_exact(5)
        except BaseException as exc:
            error.append(exc)

    reader = threading.Thread(target=read_five, daemon=True)
    reader.start()
    time.sleep(0.1)
    pair.a.close()
    reader.join(timeout=5.0)
    assert not reader.is_alive(), "reader never returned"
    assert len(error) == 1 and isinstance(error[0], ConnectionClosed), error


def test_repeated_reads_after_eof_keep_raising(pair):
    pair.a.close()
    for _ in range(3):
        with pytest.raises(ConnectionClosed):
            pair.b.recv_exact(1)


# -- requirement 2: close() releases a blocked reader -----------------------


def test_close_from_another_thread_releases_a_blocked_reader(pair):
    """The session shutdown path and the listener's handshake timeout both work
    by closing under a blocked reader. A backend that does not honour this
    turns a timeout into an abandoned thread."""
    released = threading.Event()

    def read_forever():
        try:
            pair.b.recv_exact(1)
        except BaseException:
            pass
        released.set()

    reader = threading.Thread(target=read_forever, daemon=True)
    reader.start()
    time.sleep(0.1)
    assert not released.is_set(), "the read did not block in the first place"

    started = time.monotonic()
    pair.b.close()
    assert released.wait(timeout=5.0), "close() did not release the blocked reader"
    assert time.monotonic() - started < 5.0


def test_close_is_idempotent(pair):
    pair.a.close()
    pair.a.close()
    pair.a.close()


def test_close_is_safe_on_a_connection_whose_peer_already_went(pair):
    pair.a.close()
    with pytest.raises(ConnectionClosed):
        pair.b.recv_exact(1)
    pair.b.close()


# -- requirement 3: writes fail as the documented type ----------------------


def test_writing_to_a_departed_peer_raises_connection_closed(pair):
    """TCP will usually accept the first write into the local buffer and only
    fail once the reset comes back, so the check is that it fails as
    ConnectionClosed within a few attempts -- not that it fails immediately."""
    pair.a.close()
    for attempt in range(50):
        try:
            pair.b.send_all(b"x" * 1024)
        except ConnectionClosed:
            return
        except OSError as exc:  # pragma: no cover - would be a mapping failure
            pytest.fail(f"write failed as OSError rather than ConnectionClosed: {exc!r}")
        time.sleep(0.02)
    pytest.fail("writing to a departed peer never failed")


def test_writing_after_a_local_close_raises_connection_closed(pair):
    pair.a.close()
    with pytest.raises(ConnectionClosed):
        pair.a.send_all(b"x")


# -- requirement 2, by mechanism rather than by outcome ---------------------


class RecordingSocket:
    """Records the order of shutdown and close calls."""

    def __init__(self):
        self.calls: list[str] = []

    def settimeout(self, timeout):
        return None

    def setsockopt(self, *args):
        return None

    def shutdown(self, how):
        self.calls.append("shutdown")

    def close(self):
        self.calls.append("close")

    def recv(self, n):  # pragma: no cover - not exercised here
        raise AssertionError("not expected")

    def sendall(self, data):  # pragma: no cover
        raise AssertionError("not expected")


def test_close_shuts_the_socket_down_before_closing_it():
    """Asserted as a mechanism, because the outcome cannot discriminate here.

    macOS and BSD wake a blocked recv when the fd is closed; Linux does not.
    So on this machine a close() that skipped shutdown() would still release
    the reader -- down a different path, with a different exception -- and an
    outcome test would pass while the Jetson abandoned a thread per handshake
    timeout. The Linux evidence comes from running this suite on the Jetson.
    """
    from transport.tcp import TcpConnection

    recorder = RecordingSocket()
    connection = TcpConnection(recorder, peer="recorded", set_options=False)
    connection.close()
    assert recorder.calls == ["shutdown", "close"], recorder.calls


def test_close_still_closes_when_shutdown_is_refused():
    """A socket already torn down by the peer refuses shutdown; the fd still
    has to be released."""
    from transport.tcp import TcpConnection

    class RefusingShutdown(RecordingSocket):
        def shutdown(self, how):
            self.calls.append("shutdown-failed")
            raise OSError(57, "socket is not connected")

    recorder = RefusingShutdown()
    connection = TcpConnection(recorder, peer="recorded", set_options=False)
    connection.close()
    assert recorder.calls == ["shutdown-failed", "close"], recorder.calls


# -- the surface itself ------------------------------------------------------


# Members the protocol requires, and the per-backend extras that are allowed to
# exist. Anything else appearing on a connection is a deliberate act that has to
# update this list -- backend divergence was the defect class that recurred in
# every validation round of this task.
PROTOCOL_MEMBERS = {"peer", "send_all", "recv_exact", "close"}
ALLOWED_EXTRAS = {
    "loopback": {"unread_bytes"},
    "tcp": {"applied_options"},
    # The usb backend hands back a plain TcpConnection -- no new ByteConnection
    # class, per decision D1 -- so it carries exactly tcp's extras.
    "usb": {"applied_options"},
}


def test_a_connection_satisfies_the_protocol(pair):
    assert isinstance(pair.a, ByteConnection)
    for member in PROTOCOL_MEMBERS:
        assert hasattr(pair.a, member), member


def test_a_connection_carries_no_undeclared_surface(pair):
    """Including no liveness query: `is_closed` used to exist on TcpConnection
    and nowhere else, consumed by nothing, and a flag that is stale when read
    invites the check-then-act pattern that caused this task's two worst races.
    """
    public = {name for name in dir(pair.a) if not name.startswith("_")}
    unexpected = public - PROTOCOL_MEMBERS - ALLOWED_EXTRAS[pair.name]
    assert unexpected == set(), f"{pair.name} exposes undeclared surface: {sorted(unexpected)}"
    assert "is_closed" not in public
