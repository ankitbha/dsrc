"""TCP backend: the ByteConnection seam on a real socket.

The Jetson accepts and the phone dials, always -- the Jetson cannot originate
IP traffic to a tailnet peer, so there is a `TcpAcceptor` here and a `dial`, and
nothing that lets the listening side call out.

Three requirements this file exists to satisfy, all of them discovered while
validating the transport above it, and none of them provable against the
in-process loopback backend:

  recv_exact raises rather than returning short or empty. socket.recv returns
  b"" at EOF, so the empty case is the easy mistake, and the session reader
  treats an empty chunk as progress-free -- it spun at millions of calls per
  second before that was guarded.

  close() releases a read already in progress from another thread. That is what
  the session shutdown path and the listener's handshake timeout both rely on,
  and it needs shutdown() before close(): closing a socket does not release a
  thread already sitting in recv.

  A write failure surfaces as the documented exception type, because the
  session accounts for the frame in flight differently depending on which.

The errno mapping is the other half of that last point. A reset or a broken
pipe means the phone went away, which in a car is ordinary and is what
displacement exists to handle, so those read as `peer_closed`. Everything else
stays OSError and reads as `transport_error`, which keeps that reason meaning
"our end broke" rather than "someone drove into a tunnel".
"""

from __future__ import annotations

import errno
import socket
import threading
import time

from transport.connection import ConnectionClosed

DEFAULT_PORT = 47811
CONNECT_TIMEOUT_S = 10.0
LISTEN_BACKLOG = 4
RECV_CHUNK_BYTES = 65536

KEEPALIVE_IDLE_S = 30
KEEPALIVE_INTERVAL_S = 10
KEEPALIVE_PROBES = 3

# The peer is gone. Anything not in here is our end failing, and the session
# reports the two differently.
PEER_GONE_ERRNOS = frozenset(
    {
        errno.ECONNRESET,
        errno.EPIPE,
        errno.ECONNABORTED,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
    }
)


# accept() failures that say nothing about the listening socket. ECONNABORTED is
# what a client resetting between SYN and accept produces -- which is exactly
# what an unbounded reconnect loop on a flaky link generates -- so treating it
# as fatal would stop the Jetson accepting for the rest of a drive.
ACCEPT_RETRY_ERRNOS = frozenset({errno.ECONNABORTED, errno.EINTR})

# Local resource exhaustion. Also not fatal, but retrying flat out would spin,
# so these pause first. They may clear on their own; the listening socket is
# fine either way.
ACCEPT_TRANSIENT_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM})

# Even a retryable failure gets a breath. Without one, a persistent
# ECONNABORTED spins accept() at over two million calls a second -- the same
# pathology this file's docstring describes for the reader, on a different
# errno, on a device with six cores and other work to do.
# Cap on the socket timeout for a single accept attempt, so a close() is noticed
# within one of these however long the caller asked to wait.
#
# Needed because the platforms disagree, measured rather than assumed:
#   macOS  a blocking accept is released by close (0.00 s); a *timed* one is
#          not, and ran its caller's full 5 s.
#   Linux  both are released immediately (0.000 s).
# So the cap is what macOS needs and Linux does not; it costs Linux nothing.
INTERNAL_ACCEPT_POLL_S = 0.05

ACCEPT_RETRY_PAUSE_S = 0.001
ACCEPT_TRANSIENT_PAUSE_S = 0.05


def _translate(exc: OSError, context: str) -> BaseException:
    """The exception to raise for a socket failure."""
    if exc.errno in PEER_GONE_ERRNOS:
        return ConnectionClosed(f"{context}: {exc}")
    return exc


def apply_socket_options(
    sock: socket.socket,
    *,
    keepalive_idle_s: int = KEEPALIVE_IDLE_S,
    keepalive_interval_s: int = KEEPALIVE_INTERVAL_S,
    keepalive_probes: int = KEEPALIVE_PROBES,
) -> dict[str, int | None]:
    """Set what this platform allows, and report what actually took.

    The keepalive knobs live under different names on Linux and macOS and are
    absent on some platforms entirely. None means "not applied here", and that
    record is the evidence -- the intent is not worth much on its own.
    """
    applied: dict[str, int | None] = {}

    def attempt(name: str, level: int, option: int | None, value: int) -> None:
        if option is None:
            applied[name] = None
            return
        try:
            sock.setsockopt(level, option, value)
            applied[name] = value
        except OSError:
            applied[name] = None

    attempt("TCP_NODELAY", socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    attempt("SO_KEEPALIVE", socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # Linux spells idle time TCP_KEEPIDLE; macOS spells it TCP_KEEPALIVE.
    idle_option = getattr(socket, "TCP_KEEPIDLE", None) or getattr(
        socket, "TCP_KEEPALIVE", None
    )
    attempt("keepalive_idle_s", socket.IPPROTO_TCP, idle_option, keepalive_idle_s)
    attempt(
        "keepalive_interval_s",
        socket.IPPROTO_TCP,
        getattr(socket, "TCP_KEEPINTVL", None),
        keepalive_interval_s,
    )
    attempt(
        "keepalive_probes",
        socket.IPPROTO_TCP,
        getattr(socket, "TCP_KEEPCNT", None),
        keepalive_probes,
    )
    return applied


class TcpConnection:
    """One end of a TCP connection, as a ByteConnection."""

    def __init__(
        self,
        sock: socket.socket,
        peer: str,
        *,
        recv_chunk_bytes: int = RECV_CHUNK_BYTES,
        set_options: bool = True,
    ) -> None:
        self._sock = sock
        self._peer = peer
        self._recv_chunk_bytes = max(1, recv_chunk_bytes)
        self._closed = False
        self._close_lock = threading.Lock()
        self._sock.settimeout(None)  # blocking; the session's timer is the clock
        self.applied_options = apply_socket_options(sock) if set_options else {}

    @property
    def peer(self) -> str:
        return self._peer

    @property
    def is_closed(self) -> bool:
        return self._closed

    def send_all(self, data: bytes) -> None:
        if not data:
            return
        try:
            self._sock.sendall(data)
        except OSError as exc:
            if self._closed:
                raise ConnectionClosed(f"send on a locally closed socket: {exc}") from None
            raise _translate(exc, "send") from None

    def recv_exact(self, n: int) -> bytes:
        if n == 0:
            return b""
        out = bytearray()
        while len(out) < n:
            try:
                chunk = self._sock.recv(min(self._recv_chunk_bytes, n - len(out)))
            except OSError as exc:
                # A local close is the ordinary way this unblocks, and the
                # errno for it varies; report it as what it is rather than
                # letting EBADF read as a transport fault.
                if self._closed:
                    raise ConnectionClosed(
                        f"read released by a local close with {len(out)} of {n} bytes"
                    ) from None
                raise _translate(exc, "recv") from None
            if not chunk:
                raise ConnectionClosed(f"peer closed with {len(out)} of {n} bytes")
            out += chunk
        return bytes(out)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        # shutdown first: closing the socket does not release a thread already
        # sitting in recv, and releasing it is the whole point.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


class TcpAcceptor:
    """Listener side. Binds, accepts, and never dials."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        *,
        backlog: int = LISTEN_BACKLOG,
        recv_chunk_bytes: int = RECV_CHUNK_BYTES,
    ) -> None:
        self._recv_chunk_bytes = recv_chunk_bytes
        self._closed = False
        # Retryable accept failures, counted so they are visible rather than
        # silently absorbed. Per-errno as well as in total: EMFILE is our own
        # descriptor leak and ENFILE is somebody else's, and a single number
        # cannot tell them apart.
        self.transient_accept_errors = 0
        self.accept_errors_by_errno: dict[int, int] = {}
        self.max_consecutive_accept_errors = 0
        # A high water mark cannot carry duration; these can.
        self.first_accept_error_mono_ns: int | None = None
        self.last_accept_error_mono_ns: int | None = None
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind((host, port))
            self._server.listen(backlog)
        except OSError:
            self._server.close()
            raise
        # Resolved after bind, so port 0 is usable and discoverable.
        self.address: tuple[str, int] = self._server.getsockname()[:2]

    @property
    def host(self) -> str:
        return self.address[0]

    @property
    def port(self) -> int:
        return self.address[1]

    def stats(self) -> dict[str, object]:
        """The acceptor's own record, for a run report to carry.

        These numbers existed and nothing read them, which is how a listener
        that is alive and accepting nothing came to report perfect health.
        """
        # Snapshotted before sorting: the accept thread can insert a key
        # mid-iteration, and a diagnostic that raises is worse than useless.
        by_errno = dict(self.accept_errors_by_errno)
        return {
            "address": list(self.address),
            "transient_accept_errors": self.transient_accept_errors,
            # Per accept() call, so it reads as the caller's poll interval
            # divided by the pause -- 4 at a 0.2 s poll whether the storm
            # lasted 200 ms or four hours. Not a severity signal; the
            # cumulative count and the timestamps below answer that.
            "max_consecutive_accept_errors": self.max_consecutive_accept_errors,
            "first_accept_error_mono_ns": self.first_accept_error_mono_ns,
            "last_accept_error_mono_ns": self.last_accept_error_mono_ns,
            "accept_error_span_s": (
                None
                if self.first_accept_error_mono_ns is None
                else round(
                    (self.last_accept_error_mono_ns - self.first_accept_error_mono_ns) / 1e9, 3
                )
            ),
            "accept_errors_by_errno": {
                errno.errorcode.get(code, str(code)): count
                for code, count in sorted(by_errno.items())
            },
        }

    def accept(self, timeout: float | None = None) -> TcpConnection | None:
        """A new connection, None on timeout, ConnectionClosed once closed.

        Retryable failures are retried here rather than reported, because the
        only caller is an accept loop that treats any error as terminal -- and
        it lives in a module this task must not change. Converting every
        OSError into ConnectionClosed was worse than useless: it claimed the
        acceptor was closed when it was still listening, and one aborted
        connection stopped the Jetson accepting for the rest of the drive.

        Every retry pauses, and never past the caller's deadline. Retries are
        counted per errno, because retrying EMFILE indefinitely would otherwise
        turn our own descriptor leak into a passing weather condition.
        """
        if self._closed:
            raise ConnectionClosed("acceptor is closed")
        deadline = None if timeout is None else time.monotonic() + timeout
        # Per call: the high water mark below therefore means "worst run of
        # failures inside one accept()", which is the useful reading and the
        # only one a local can carry.
        consecutive = 0
        attempted = False
        while True:
            if self._closed:
                raise ConnectionClosed("acceptor is closed")
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if attempted:
                        return None
                    # timeout=0 means "poll now", not "do nothing":
                    # LoopbackAcceptor checks its queue and returns a waiting
                    # connection, and two backends behind one Acceptor protocol
                    # must not disagree about that.
                    remaining = 0.0
            # Set before the attempt, not between the two calls inside it: a
            # retryable errno raised by settimeout otherwise never sets it, and
            # with an expired deadline the loop then has no exit at all --
            # measured at 2.5 million iterations a second.
            if remaining is None:
                socket_timeout: float = INTERNAL_ACCEPT_POLL_S
            elif remaining == 0.0:
                socket_timeout = 0.0
            else:
                socket_timeout = min(remaining, INTERNAL_ACCEPT_POLL_S)
            attempted = True
            try:
                # settimeout is inside the guard so that a close() landing here
                # is classified like any other socket failure rather than
                # escaping as an unhandled OSError.
                self._server.settimeout(socket_timeout)
                sock, address = self._server.accept()
            except TimeoutError:
                # An internal poll expiring is not the caller's timeout; the
                # loop top re-checks both the closed flag and the deadline.
                continue
            except BlockingIOError:
                # settimeout(0) makes accept non-blocking; nothing waiting.
                return None
            except OSError as exc:
                if self._closed:
                    raise ConnectionClosed(f"acceptor closed: {exc}") from None
                retryable = exc.errno in ACCEPT_RETRY_ERRNOS
                transient = exc.errno in ACCEPT_TRANSIENT_ERRNOS
                if not (retryable or transient):
                    raise
                self.transient_accept_errors += 1
                self.accept_errors_by_errno[exc.errno] = (
                    self.accept_errors_by_errno.get(exc.errno, 0) + 1
                )
                consecutive += 1
                self.max_consecutive_accept_errors = max(
                    self.max_consecutive_accept_errors, consecutive
                )
                now_ns = time.monotonic_ns()
                if self.first_accept_error_mono_ns is None:
                    self.first_accept_error_mono_ns = now_ns
                self.last_accept_error_mono_ns = now_ns
                pause = ACCEPT_TRANSIENT_PAUSE_S if transient else ACCEPT_RETRY_PAUSE_S
                if deadline is not None:
                    # Against the clock now, not against the `remaining`
                    # computed before accept() blocked: a failure arriving late
                    # in the window otherwise still overshoots by a whole pause,
                    # which measured 130% over a 50 ms deadline.
                    pause = min(pause, max(0.0, deadline - time.monotonic()))
                if pause > 0:
                    time.sleep(pause)
                continue
            return TcpConnection(
                sock,
                peer=f"{address[0]}:{address[1]}",
                recv_chunk_bytes=self._recv_chunk_bytes,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # shutdown before close, for the same reason TcpConnection does it: on
        # Linux, closing a socket does not release a thread already blocked on
        # it, and LoopbackAcceptor does release a waiting accept.
        try:
            self._server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._server.close()
        except OSError:
            pass


def dial(
    host: str,
    port: int = DEFAULT_PORT,
    *,
    timeout: float = CONNECT_TIMEOUT_S,
    recv_chunk_bytes: int = RECV_CHUNK_BYTES,
) -> TcpConnection:
    """Open a connection. The dialling side is always the phone's role.

    No name discovery: MagicDNS does not resolve while the Nash VPN is up, and
    the Jetson's tailnet address has changed once already, so the caller
    supplies the host it means.
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    return TcpConnection(sock, peer=f"{host}:{port}", recv_chunk_bytes=recv_chunk_bytes)
