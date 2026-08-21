"""The one seam a transport backend has to implement.

Everything above this file -- framing, channels, overflow, sessions, the
handshake, the counters -- is backend-independent. A backend supplies a duplex
byte stream and nothing else, which is why the network backend and the USB
backend are each three methods rather than each a transport.

Two requirements a backend must meet, both of which are easy to get wrong and
neither of which the transport can work around:

`recv_exact` raises rather than returning short or empty. A frame reader that
accepts a short read has no way to distinguish "the rest is coming" from "the
peer is gone", and would hand a truncated header to the JSON parser. Returning
b"" at EOF -- which is what socket.recv does -- is the most likely mistake
here; the reader treats an empty chunk as a closed connection rather than as
progress, but a backend that does it is still wrong.

`close()` unblocks a read already in progress, from another thread. Both the
session shutdown path and the listener's handshake timeout work by closing the
connection under a blocked reader. On a POSIX socket that means shutdown()
before close(), because closing a socket does not release a thread already
sitting in recv. A backend that does not honour this turns a timeout into an
abandoned thread.

Measured on the Jetson (Linux 5.15 aarch64), which is where it matters:

    blocking recv, close() only        never released (still blocked at 6 s)
    blocking recv, shutdown()+close()  released in 0.001 s

macOS is more forgiving and releases on close alone, which is why this has to
be asserted as call order there and can only be verified as an outcome on
Linux.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ConnectionClosed(Exception):
    """The byte stream ended, from either side, for any reason."""


@runtime_checkable
class ByteConnection(Protocol):
    """A duplex byte stream. Not required to be safe for concurrent senders;
    a Session uses exactly one writer thread and one reader thread."""

    @property
    def peer(self) -> str:
        """Human-readable label for logs. Not an identity."""

    def send_all(self, data: bytes) -> None:
        """Write every byte, or raise ConnectionClosed."""

    def recv_exact(self, n: int) -> bytes:
        """Return exactly n bytes, or raise ConnectionClosed. n may be 0."""

    def close(self) -> None:
        """Idempotent. Unblocks a peer or a local reader waiting on this stream."""
