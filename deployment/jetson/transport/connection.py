"""The one seam a transport backend has to implement.

Everything above this file -- framing, channels, overflow, sessions, the
handshake, the counters -- is backend-independent. A backend supplies a duplex
byte stream and nothing else, which is why the network backend and the USB
backend are each three methods rather than each a transport.

`recv_exact` returning short is not an option. A frame reader that accepts a
short read has no way to distinguish "the rest is coming" from "the peer is
gone", and would hand a truncated header to the JSON parser. Backends raise
instead.
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
