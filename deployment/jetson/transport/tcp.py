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

    def accept(self, timeout: float | None = None) -> TcpConnection | None:
        if self._closed:
            raise ConnectionClosed("acceptor is closed")
        self._server.settimeout(timeout)
        try:
            sock, address = self._server.accept()
        except TimeoutError:
            return None
        except OSError as exc:
            raise ConnectionClosed(f"acceptor closed: {exc}") from None
        return TcpConnection(
            sock,
            peer=f"{address[0]}:{address[1]}",
            recv_chunk_bytes=self._recv_chunk_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
