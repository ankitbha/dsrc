"""The dialling side: connect, handshake, run a session, and reconnect.

This is the phone's role. It ships as a supported component rather than a test
harness for two reasons: every section F task is developed with the phone
absent and something has to stand in for it, and the reconnection policy is far
cheaper to design and test here than in Kotlin on a drive.

Reconnection never gives up. A drive is long and a client that stops trying is
a drive lost. The cost is real and deliberate: each reconnect displaces the
listener's live session, so a client failing in a loop produces a chain of
short sessions rather than an error. That is visible in the listener's
`displaced` counter, and it is preferred over the alternative -- refusing the
newcomer would let one half-open drop lock the phone out for the rest of the
drive.

Which is exactly why the backoff schedule resets on a *durable* connection
rather than on any connection at all. A link that connects and dies immediately
would otherwise reset to the initial delay every time and reconnect several
times a second, turning the escalation into no escalation at all.
"""

from __future__ import annotations

import itertools
import random
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable

from transport.clock import MonoClock, WallClock, now_mono_ns, now_wall_ns
from transport.connection import ByteConnection
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
from transport.tcp import CONNECT_TIMEOUT_S, DEFAULT_PORT, dial


@dataclass(frozen=True)
class Backoff:
    """The retry schedule, as a pure function of the attempt number.

    Separating the schedule from the sleeping is what makes it testable: the
    jitter comes in as a unit value rather than from the random module, so the
    whole sequence can be checked without waiting for it.
    """

    initial_s: float = 0.25
    multiplier: float = 2.0
    cap_s: float = 5.0
    jitter: float = 0.25

    def base_delay_for(self, attempt: int) -> float:
        """Un-jittered delay before the attempt after this one. 1-based."""
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        growth = self.multiplier ** (attempt - 1)
        return min(self.cap_s, self.initial_s * growth)

    def delay_for(self, attempt: int, unit: float) -> float:
        """`unit` in [0, 1) supplies the jitter: 0 is the low edge, 1 the high."""
        base = self.base_delay_for(attempt)
        return base * (1.0 + self.jitter * (2.0 * unit - 1.0))

    def bounds_for(self, attempt: int) -> tuple[float, float]:
        base = self.base_delay_for(attempt)
        return base * (1.0 - self.jitter), base * (1.0 + self.jitter)


@dataclass(frozen=True)
class ClientSessionStarted:
    session: Session
    handshake: HandshakeResult
    attempt: int


@dataclass(frozen=True)
class ClientSessionEnded:
    session_id: int
    reason: SessionEndReason
    stats: SessionStats
    uptime_s: float
    # How long before the next attempt, and None when the session was durable
    # enough to reset the schedule. Reported so the escalation is observable:
    # the attempt counter increments whatever delay is used, so without this
    # a loop that always waited the initial delay looked identical.
    retry_in_s: float | None = None


@dataclass(frozen=True)
class ClientAttemptFailed:
    attempt: int
    error: str
    retry_in_s: float


@dataclass(frozen=True)
class ClientGaveUp:
    attempts: int


ClientEvent = ClientSessionStarted | ClientSessionEnded | ClientAttemptFailed | ClientGaveUp


def _handshake_with_timeout(
    connection: ByteConnection,
    local_hello: Hello,
    *,
    timeout_s: float | None,
    mono_clock: MonoClock,
    wall_clock: WallClock,
) -> HandshakeResult:
    """Handshake, but not forever.

    dial's timeout covers only the TCP connect; the hello read that follows has
    no clock behind it. A peer that accepts and then says nothing -- which is
    exactly what a listener whose accept loop has died looks like -- would
    otherwise wedge this thread permanently: never retrying, never counting an
    attempt, never emitting an event, and holding stop() open. The listener
    guards the mirror image of this case for the same stated reason.
    """
    if timeout_s is None:
        return perform_handshake(
            connection, local_hello, mono_clock=mono_clock, wall_clock=wall_clock
        )

    outcome: dict[str, object] = {}

    def attempt() -> None:
        try:
            outcome["result"] = perform_handshake(
                connection, local_hello, mono_clock=mono_clock, wall_clock=wall_clock
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=attempt, name="client-handshake", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        # Closing is what releases the worker's blocked read.
        try:
            connection.close()
        except Exception:
            pass
        worker.join(1.0)
        raise HandshakeError(f"no hello from {connection.peer} within {timeout_s}s")
    error = outcome.get("error")
    if error is not None:
        raise error
    return outcome["result"]  # type: ignore[return-value]


def connect_session(
    host: str,
    port: int = DEFAULT_PORT,
    *,
    local_hello: Hello,
    session_id: int = 1,
    heartbeat_s: float | None = DEFAULT_HEARTBEAT_S,
    stall_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
    connect_timeout_s: float = CONNECT_TIMEOUT_S,
    handshake_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
    mono_clock: MonoClock = now_mono_ns,
    wall_clock: WallClock = now_wall_ns,
    on_end: Callable[[Session, SessionEndReason], None] | None = None,
    dial_fn: Callable[..., ByteConnection] = dial,
    on_connection: Callable[[ByteConnection | None], None] | None = None,
) -> tuple[Session, HandshakeResult]:
    """Dial, handshake, and return a started session.

    Every caller wants these three in this order, and getting them wrong yields
    an unstarted session or a skipped handshake. A failure at any step closes
    what it opened rather than leaving a socket behind.
    """
    connection = dial_fn(host, port, timeout=connect_timeout_s)
    # Handed over as soon as it exists, so a caller that wants to interrupt a
    # handshake in progress has something to close. Cleared on the way out.
    if on_connection is not None:
        on_connection(connection)
    session: Session | None = None
    try:
        handshake = _handshake_with_timeout(
            connection,
            local_hello,
            timeout_s=handshake_timeout_s,
            mono_clock=mono_clock,
            wall_clock=wall_clock,
        )
        # Constructed inside the guard, not between the two: a failure here
        # left the socket open, and a guard around start() alone covered the
        # call rather than the step.
        session = Session(
            connection,
            session_id=session_id,
            heartbeat_s=heartbeat_s,
            stall_timeout_s=stall_timeout_s,
            mono_clock=mono_clock,
            wall_clock=wall_clock,
            on_end=on_end,
        )
        session.start()
    except BaseException:
        if session is not None:
            session.close()
        connection.close()
        raise
    finally:
        if on_connection is not None:
            on_connection(None)
    return session, handshake


class SessionClient:
    """Keeps one session up, reconnecting for as long as it is running.

    Session boundaries are surfaced as events rather than by swapping an object
    behind the caller's back, for the same reason the listener does it: clock
    offset, tracker state and the HERE cache all mean nothing across a
    reconnect, and a client that hid the seam would let them span it.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        local_hello: Hello,
        backoff: Backoff = Backoff(),
        heartbeat_s: float | None = DEFAULT_HEARTBEAT_S,
        stall_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
        handshake_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
        mono_clock: MonoClock = now_mono_ns,
        wall_clock: WallClock = now_wall_ns,
        dial_fn: Callable[..., ByteConnection] = dial,
        random_unit: Callable[[], float] = random.random,
        max_attempts: int | None = None,
        reset_after_s: float | None = None,
        poll_s: float = 0.02,
    ) -> None:
        self._host = host
        self._port = port
        self._local_hello = local_hello
        self._backoff = backoff
        self._heartbeat_s = heartbeat_s
        self._stall_timeout_s = stall_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._handshake_timeout_s = handshake_timeout_s
        self._mono = mono_clock
        self._wall = wall_clock
        self._dial = dial_fn
        self._random_unit = random_unit
        self._max_attempts = max_attempts
        # A session has to last this long to count as durable enough to reset
        # the schedule, so a link that dies quickly keeps escalating instead of
        # hammering. Derived from the stall timeout rather than fixed at the
        # backoff cap: those two were both 5.0 s, and a stalled session lasts
        # exactly the stall timeout, so the modal failure in a car landed on
        # the resetting side of a >= test and the escalation never engaged for
        # it. Deriving means they cannot coincide again if either changes.
        if reset_after_s is not None:
            self._reset_after_s = reset_after_s
        elif stall_timeout_s is None:
            self._reset_after_s = backoff.cap_s
        else:
            self._reset_after_s = max(backoff.cap_s, stall_timeout_s * 2.0)
        self._poll_s = poll_s

        self._session_ids = itertools.count(1)
        self._events: Queue[ClientEvent] = Queue()
        self._lock = threading.Lock()
        self._current: Session | None = None
        self._connecting: ByteConnection | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = 0
        self.failed_attempts = 0
        self.reconnects = 0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "SessionClient":
        self._thread = threading.Thread(target=self._maintain, name="client", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            session = self._current
            connecting = self._connecting
        if session is not None:
            session.close(SessionEndReason.CLOSED_LOCAL)
        # A handshake in progress is not interruptible by the flag: its worker
        # is blocked on a read. Closing the connection is what releases it, and
        # without this stop() returns while that thread is still waiting out
        # the handshake timeout.
        if connecting is not None:
            try:
                connecting.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

    @property
    def current_session(self) -> Session | None:
        """A session a caller can use, or None.

        Filtered as well as cleared. `_maintain` clears the reference when it
        notices the session ended, but "notices" is a poll, and between the
        session closing and the poll firing this used to hand back a dead
        session -- which is the same failure the event design exists to prevent,
        one step in.
        """
        with self._lock:
            session = self._current
        if session is None or session.is_closed:
            return None
        return session

    def wait_for_session(self, timeout: float = 10.0) -> Session | None:
        deadline = self._mono() + int(timeout * 1e9)
        while self._mono() < deadline:
            session = self.current_session
            if session is not None and not session.is_closed:
                return session
            if self._stop.is_set():
                return None
            self._stop.wait(self._poll_s)
        return None

    # -- events ----------------------------------------------------------

    def next_event(self, timeout: float | None = 0.0) -> ClientEvent | None:
        try:
            if timeout == 0:
                return self._events.get_nowait()
            return self._events.get(timeout=timeout)
        except Empty:
            return None

    def drain_events(self) -> list[ClientEvent]:
        events = []
        while True:
            event = self.next_event(0.0)
            if event is None:
                return events
            events.append(event)

    def _note_connecting(self, connection: ByteConnection | None) -> None:
        with self._lock:
            self._connecting = connection

    def _note_session_end(self, session: Session, reason: SessionEndReason) -> None:
        """Called from the session's own thread the instant it ends."""
        with self._lock:
            if self._current is session:
                self._current = None

    # -- the loop --------------------------------------------------------

    def _maintain(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            if self._max_attempts is not None and attempt > self._max_attempts:
                self._events.put(ClientGaveUp(attempts=attempt - 1))
                self._stop.set()
                return
            try:
                session, handshake = connect_session(
                    self._host,
                    self._port,
                    local_hello=self._local_hello,
                    session_id=next(self._session_ids),
                    heartbeat_s=self._heartbeat_s,
                    stall_timeout_s=self._stall_timeout_s,
                    connect_timeout_s=self._connect_timeout_s,
                    handshake_timeout_s=self._handshake_timeout_s,
                    mono_clock=self._mono,
                    wall_clock=self._wall,
                    dial_fn=self._dial,
                    on_connection=self._note_connecting,
                    on_end=self._note_session_end,
                )
            except Exception as exc:
                delay = self._backoff.delay_for(attempt, self._random_unit())
                self.failed_attempts += 1
                self._events.put(
                    ClientAttemptFailed(attempt=attempt, error=str(exc), retry_in_s=delay)
                )
                self._stop.wait(delay)
                continue

            started_ns = self._mono()
            with self._lock:
                self._current = session
            self.connected += 1
            if self.connected > 1:
                self.reconnects += 1
            self._events.put(
                ClientSessionStarted(session=session, handshake=handshake, attempt=attempt)
            )

            while not self._stop.is_set() and not session.is_closed:
                self._stop.wait(self._poll_s)

            session.close()
            uptime_s = (self._mono() - started_ns) / 1e9
            with self._lock:
                if self._current is session:
                    self._current = None
            durable = uptime_s >= self._reset_after_s
            # Short-lived: keep escalating rather than reconnecting at the
            # initial delay forever.
            delay = None if durable else self._backoff.delay_for(attempt, self._random_unit())
            self._events.put(
                ClientSessionEnded(
                    session_id=session.session_id,
                    reason=session.end_reason or SessionEndReason.CLOSED_LOCAL,
                    stats=session.stats(),
                    uptime_s=uptime_s,
                    retry_in_s=delay,
                )
            )
            if self._stop.is_set():
                return
            if durable:
                attempt = 0
                continue
            self._stop.wait(delay)
