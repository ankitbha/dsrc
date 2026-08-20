"""The listening side: accept one connection at a time, hand back sessions.

The phone opens connections and the Jetson accepts them, always, because the
Jetson cannot originate IP traffic to a tailnet peer -- its kernel is built
without the conntrack mark field Tailscale needs. So there is a listener here
and no dialer.

Displacement is the interesting rule. A new connection arriving while a session
is live ends the live one and takes over. The alternative -- refuse the
newcomer, keep what we have -- sounds safer and is worse: a half-open drop that
this side has not noticed yet would lock the phone out for the rest of the
drive, and the phone is the only party that can reconnect.

Sessions are surfaced as events rather than callbacks so that a consumer can
reset its own state deliberately at a boundary. Clock offset, tracker state and
the HERE cache all mean nothing across a reconnect, and a transport that hid
the seam would let them silently span it.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Protocol

from transport.clock import MonoClock, WallClock, now_mono_ns, now_wall_ns
from transport.connection import ByteConnection, ConnectionClosed
from transport.handshake import (
    HandshakeError,
    HandshakeResult,
    Hello,
    perform_handshake,
)
from transport.session import (
    DEFAULT_HEARTBEAT_S,
    DEFAULT_STALL_TIMEOUT_S,
    Session,
    SessionEndReason,
    SessionStats,
)


class Acceptor(Protocol):
    """Backend-side listener. Loopback, TCP and USB each supply one."""

    def accept(self, timeout: float | None = None) -> ByteConnection | None:
        """A new connection, or None on timeout. Raises ConnectionClosed once
        the acceptor itself is closed."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class SessionStarted:
    session: Session
    handshake: HandshakeResult


@dataclass(frozen=True)
class SessionEnded:
    session_id: int
    reason: SessionEndReason
    stats: SessionStats


@dataclass(frozen=True)
class SessionRefused:
    """A connection that never became a session: bad or absent hello, or a
    protocol version we do not speak."""

    peer: str
    error: str


SessionEvent = SessionStarted | SessionEnded | SessionRefused


class TransportListener:
    def __init__(
        self,
        acceptor: Acceptor,
        local_hello: Hello,
        *,
        heartbeat_s: float | None = DEFAULT_HEARTBEAT_S,
        stall_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
        mono_clock: MonoClock = now_mono_ns,
        wall_clock: WallClock = now_wall_ns,
        accept_poll_s: float = 0.05,
    ) -> None:
        self._acceptor = acceptor
        self._local_hello = local_hello
        self._heartbeat_s = heartbeat_s
        self._stall_timeout_s = stall_timeout_s
        self._mono = mono_clock
        self._wall = wall_clock
        self._accept_poll_s = accept_poll_s

        self._session_ids = itertools.count(1)
        self._events: Queue[SessionEvent] = Queue()
        self._lock = threading.Lock()
        self._current: Session | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.accepted = 0
        self.refused = 0
        self.displaced = 0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "TransportListener":
        self._thread = threading.Thread(target=self._accept_loop, name="listener", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._acceptor.close()
        except Exception:
            pass
        session = self.current_session
        if session is not None:
            session.close(SessionEndReason.CLOSED_LOCAL)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

    @property
    def current_session(self) -> Session | None:
        with self._lock:
            return self._current

    # -- events ----------------------------------------------------------

    def next_event(self, timeout: float | None = 0.0) -> SessionEvent | None:
        try:
            if timeout == 0:
                return self._events.get_nowait()
            return self._events.get(timeout=timeout)
        except Empty:
            return None

    def drain_events(self) -> list[SessionEvent]:
        events = []
        while True:
            event = self.next_event(0.0)
            if event is None:
                return events
            events.append(event)

    # -- accept loop -----------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                connection = self._acceptor.accept(timeout=self._accept_poll_s)
            except ConnectionClosed:
                return
            except OSError:
                return
            if connection is None:
                continue
            if self._stop.is_set():
                connection.close()
                return
            self._admit(connection)

    def _admit(self, connection: ByteConnection) -> None:
        try:
            handshake = perform_handshake(
                connection,
                self._local_hello,
                mono_clock=self._mono,
                wall_clock=self._wall,
            )
        except (HandshakeError, ConnectionClosed, OSError) as exc:
            self.refused += 1
            peer = getattr(connection, "peer", "?")
            try:
                connection.close()
            except Exception:
                pass
            self._events.put(SessionRefused(peer=peer, error=str(exc)))
            return

        # Only displace once the newcomer has proved it speaks the protocol.
        previous = self.current_session
        if previous is not None and not previous.is_closed:
            self.displaced += 1
            previous.close(SessionEndReason.DISPLACED)

        session = Session(
            connection,
            session_id=next(self._session_ids),
            heartbeat_s=self._heartbeat_s,
            stall_timeout_s=self._stall_timeout_s,
            mono_clock=self._mono,
            wall_clock=self._wall,
            on_end=self._on_session_end,
        )
        with self._lock:
            self._current = session
        self.accepted += 1
        session.start()
        self._events.put(SessionStarted(session=session, handshake=handshake))

    def _on_session_end(self, session: Session, reason: SessionEndReason) -> None:
        with self._lock:
            if self._current is session:
                self._current = None
        self._events.put(
            SessionEnded(session_id=session.session_id, reason=reason, stats=session.stats())
        )
