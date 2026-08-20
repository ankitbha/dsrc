"""In-process backend: a connected pair of byte streams, no sockets.

This ships rather than living in the tests because it is what keeps the whole
pipeline runnable with no phone and no Jetson -- for CI, for offline replay,
and for developing anything downstream of the transport.

`max_buffer_bytes` models a socket send buffer: once that much data is unread,
`send_all` blocks. That is the only way to exercise the overflow policies,
which otherwise never trigger because an unbounded in-memory pipe never
applies backpressure.

What loopback cannot do, and must not be mistaken for: partial writes,
reordering, RST, half-open state, or any timing resembling a real link. It
proves framing and policy correct. Socket handling is proved against a real
socket or not at all.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque

from transport.connection import ConnectionClosed

_ids = itertools.count(1)


class _Pipe:
    """One direction of bytes, with an optional capacity."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self._buffer = bytearray()
        self._cond = threading.Condition()
        self._closed = False
        self._max_bytes = max_bytes

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        with self._cond:
            while view:
                if self._closed:
                    raise ConnectionClosed("write to a closed pipe")
                if self._max_bytes is None:
                    space = len(view)
                else:
                    space = self._max_bytes - len(self._buffer)
                    if space <= 0:
                        self._cond.wait(0.05)
                        continue
                take = min(space, len(view))
                self._buffer += view[:take]
                view = view[take:]
                self._cond.notify_all()

    def read_exact(self, n: int) -> bytes:
        """Accumulate n bytes across as many arrivals as it takes.

        Taking whatever is buffered and coming back for the rest is what a
        socket recv_exact does, and it is required rather than merely tidy:
        waiting for all n bytes to be buffered at once would deadlock for any
        n larger than the buffer, and n is a whole frame payload.
        """
        if n == 0:
            return b""
        out = bytearray()
        with self._cond:
            while len(out) < n:
                if self._buffer:
                    take = min(n - len(out), len(self._buffer))
                    out += self._buffer[:take]
                    del self._buffer[:take]
                    self._cond.notify_all()
                    continue
                if self._closed:
                    raise ConnectionClosed(f"pipe closed with {len(out)} of {n} bytes available")
                self._cond.wait(0.05)
        return bytes(out)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def pending_bytes(self) -> int:
        with self._cond:
            return len(self._buffer)


class LoopbackConnection:
    """One end of a loopback pair."""

    def __init__(self, outbound: _Pipe, inbound: _Pipe, peer: str) -> None:
        self._outbound = outbound
        self._inbound = inbound
        self._peer = peer

    @property
    def peer(self) -> str:
        return self._peer

    def send_all(self, data: bytes) -> None:
        self._outbound.write(data)

    def recv_exact(self, n: int) -> bytes:
        return self._inbound.read_exact(n)

    def close(self) -> None:
        # Both directions: the peer sees EOF, and a local reader stops waiting.
        self._outbound.close()
        self._inbound.close()

    @property
    def unread_bytes(self) -> int:
        """Bytes this end has sent that the peer has not read. Test affordance."""
        return self._outbound.pending_bytes


def loopback_pair(
    max_buffer_bytes: int | None = None,
    labels: tuple[str, str] = ("phone", "jetson"),
) -> tuple[LoopbackConnection, LoopbackConnection]:
    """Two connected ends. Bytes written to one are read from the other."""
    a_to_b = _Pipe(max_buffer_bytes)
    b_to_a = _Pipe(max_buffer_bytes)
    left = LoopbackConnection(a_to_b, b_to_a, peer=f"loopback:{labels[1]}")
    right = LoopbackConnection(b_to_a, a_to_b, peer=f"loopback:{labels[0]}")
    return left, right


class LoopbackAcceptor:
    """Listener side of the loopback backend.

    Keeps the real asymmetry: the client calls `connect`, the listener calls
    `accept`. Nothing here lets the listener originate a connection, because
    on the actual hardware it cannot.
    """

    def __init__(self, max_buffer_bytes: int | None = None) -> None:
        self._max_buffer_bytes = max_buffer_bytes
        self._waiting: deque[LoopbackConnection] = deque()
        self._cond = threading.Condition()
        self._closed = False

    def connect(self, label: str | None = None) -> LoopbackConnection:
        """Client side. Returns its end; the listener's end waits in accept()."""
        with self._cond:
            if self._closed:
                raise ConnectionClosed("acceptor is closed")
            name = label or f"client-{next(_ids)}"
            client, server = loopback_pair(self._max_buffer_bytes, labels=(name, "jetson"))
            self._waiting.append(server)
            self._cond.notify_all()
            return client

    def accept(self, timeout: float | None = None) -> LoopbackConnection | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._waiting:
                if self._closed:
                    raise ConnectionClosed("acceptor is closed")
                if deadline is None:
                    self._cond.wait(0.05)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(min(remaining, 0.05))
            return self._waiting.popleft()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
