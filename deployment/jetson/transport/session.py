"""One accepted connection: framing, priority, overflow, counters, lifetime.

Three threads per session, and the split matters:

  writer  picks the next message by priority and writes it whole. Priority is
          between frames, never inside one, so a high-priority message waits at
          most for the frame already going out. That bound is the frame size,
          which is why large payloads on a slow link are a rate-control problem
          rather than something the transport can fix.
  reader  reads frames and files them by channel. A framing error ends the
          session: the stream has no delimiter to resynchronize on, so a reader
          that has lost its place cannot get it back.
  timer   sends the keepalive and enforces the stall timeout. Without the
          timeout a half-open TCP connection looks healthy forever, which in a
          car means a drive that silently records nothing.

Messages are encoded in the caller's thread, in `send`. It costs a copy, and it
buys a synchronous error for a payload that is too large instead of a session
that dies later in a thread the caller cannot see.

Sequence numbers are assigned at enqueue, before the overflow decision, so a
gap the receiver sees is how it learns the sender dropped something. Gap
counting starts at the first frame observed on a channel -- a drop before your
first observation is not detectable, and pretending otherwise would report a
gap on every fresh session.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from transport.channels import (
    Channel,
    OverflowPolicy,
    channels_by_priority,
    policy_for,
)
from transport.clock import MonoClock, WallClock, now_mono_ns, now_wall_ns
from transport.connection import ByteConnection, ConnectionClosed
from transport.frames import (
    HEARTBEAT_KEY,
    Frame,
    FramingError,
    encode,
    read_frame,
)

DEFAULT_HEARTBEAT_S = 1.0
DEFAULT_STALL_TIMEOUT_S = 5.0


class SessionEndReason(str, Enum):
    CLOSED_LOCAL = "closed_local"
    PEER_CLOSED = "peer_closed"
    DISPLACED = "displaced"
    STALLED = "stalled"
    FRAMING_ERROR = "framing_error"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True)
class ReceivedMessage:
    frame: Frame
    t_recv_mono_ns: int

    @property
    def channel(self) -> Channel:
        return self.frame.channel

    @property
    def payload(self) -> bytes:
        return self.frame.payload

    @property
    def seq(self) -> int:
        return self.frame.seq

    @property
    def extensions(self) -> Mapping[str, Any]:
        return self.frame.extensions


@dataclass
class ChannelStats:
    channel: Channel
    queued: int = 0
    sent: int = 0
    dropped_outbound: int = 0
    bytes_sent: int = 0
    received: int = 0
    delivered: int = 0
    dropped_inbound: int = 0
    bytes_received: int = 0
    seq_gaps: int = 0
    missing_seqs: int = 0
    outbound_high_water: int = 0
    inbound_high_water: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "queued": self.queued,
            "sent": self.sent,
            "dropped_outbound": self.dropped_outbound,
            "bytes_sent": self.bytes_sent,
            "received": self.received,
            "delivered": self.delivered,
            "dropped_inbound": self.dropped_inbound,
            "bytes_received": self.bytes_received,
            "seq_gaps": self.seq_gaps,
            "missing_seqs": self.missing_seqs,
            "outbound_high_water": self.outbound_high_water,
            "inbound_high_water": self.inbound_high_water,
        }


@dataclass
class SessionStats:
    session_id: int
    peer: str
    end_reason: str | None
    heartbeats_sent: int
    heartbeats_received: int
    channels: dict[Channel, ChannelStats] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "peer": self.peer,
            "end_reason": self.end_reason,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_received": self.heartbeats_received,
            "channels": {c.value: s.to_record() for c, s in self.channels.items()},
        }


@dataclass
class _Outbound:
    frame: Frame
    encoded: bytes


class Session:
    """A live connection. Construct after the handshake, then `start()`."""

    def __init__(
        self,
        connection: ByteConnection,
        *,
        session_id: int,
        heartbeat_s: float | None = DEFAULT_HEARTBEAT_S,
        stall_timeout_s: float | None = DEFAULT_STALL_TIMEOUT_S,
        mono_clock: MonoClock = now_mono_ns,
        wall_clock: WallClock = now_wall_ns,
        on_end: Callable[["Session", SessionEndReason], None] | None = None,
        control_seq_start: int = 1,
    ) -> None:
        self._connection = connection
        self.session_id = session_id
        self._heartbeat_ns = None if heartbeat_s is None else int(heartbeat_s * 1e9)
        self._stall_ns = None if stall_timeout_s is None else int(stall_timeout_s * 1e9)
        self._mono = mono_clock
        self._wall = wall_clock
        self._on_end = on_end

        # The handshake already spent CONTROL seq 0 on the hello, so the
        # session's own control traffic continues from 1.
        self._seq = {
            channel: itertools.count(control_seq_start if channel is Channel.CONTROL else 0)
            for channel in Channel
        }

        self._tiers = channels_by_priority()
        self._rotation = {priority: 0 for priority, _ in self._tiers}
        self._outbound: dict[Channel, deque[_Outbound]] = {c: deque() for c in Channel}
        self._inbound: dict[Channel, deque[ReceivedMessage]] = {c: deque() for c in Channel}
        self._stats = {c: ChannelStats(channel=c) for c in Channel}
        self._last_rx_seq: dict[Channel, int] = {}

        self._out_cond = threading.Condition()
        self._in_cond = threading.Condition()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()

        self._end_reason: SessionEndReason | None = None
        self._heartbeats_sent = 0
        self._heartbeats_received = 0
        self._last_rx_mono_ns = self._mono()
        self._last_heartbeat_tx_ns = self._mono()
        self._threads: list[threading.Thread] = []

        interval_candidates = [v for v in (self._heartbeat_ns, self._stall_ns) if v]
        tick_ns = min(interval_candidates) / 4 if interval_candidates else 1e8
        self._tick_s = max(0.005, min(0.25, tick_ns / 1e9))

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "Session":
        for target, name in (
            (self._writer_loop, f"session{self.session_id}-tx"),
            (self._reader_loop, f"session{self.session_id}-rx"),
            (self._timer_loop, f"session{self.session_id}-timer"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def close(self, reason: SessionEndReason = SessionEndReason.CLOSED_LOCAL) -> None:
        self._shutdown(reason)
        self.join()

    def join(self, timeout: float = 2.0) -> None:
        current = threading.current_thread()
        for thread in self._threads:
            if thread is not current:
                thread.join(timeout=timeout)

    @property
    def is_closed(self) -> bool:
        return self._end_reason is not None

    @property
    def end_reason(self) -> SessionEndReason | None:
        return self._end_reason

    @property
    def peer(self) -> str:
        return self._connection.peer

    def _shutdown(self, reason: SessionEndReason) -> None:
        """First reason wins. Safe from any thread, and idempotent."""
        with self._state_lock:
            if self._end_reason is not None:
                return
            self._end_reason = reason
        self._stop.set()
        try:
            self._connection.close()
        except Exception:
            pass
        with self._out_cond:
            self._out_cond.notify_all()
        with self._in_cond:
            self._in_cond.notify_all()
        if self._on_end is not None:
            self._on_end(self, reason)

    # -- sending ---------------------------------------------------------

    def send(
        self,
        channel: Channel,
        payload: bytes = b"",
        extensions: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue a message. False means it was not queued: either the session
        has ended, or the queue was full and this message displaced nothing
        (which cannot happen -- a full queue always drops to make room, so a
        False here is always an ended session)."""
        if self.is_closed:
            return False
        policy = policy_for(channel)
        frame = Frame(
            channel=channel,
            seq=next(self._seq[channel]),
            t_mono_ns=self._mono(),
            t_wall_ns=self._wall(),
            payload=payload,
            extensions=dict(extensions or {}),
        )
        encoded = encode(frame)  # raises FramingError in the caller's thread

        stats = self._stats[channel]
        with self._out_cond:
            queue = self._outbound[channel]
            while len(queue) >= policy.depth:
                queue.popleft()
                stats.dropped_outbound += 1
            queue.append(_Outbound(frame=frame, encoded=encoded))
            stats.queued += 1
            stats.outbound_high_water = max(stats.outbound_high_water, len(queue))
            self._out_cond.notify_all()
        return True

    def _next_outbound(self) -> _Outbound | None:
        """Highest non-empty priority tier, round-robin within the tier."""
        for priority, channels in self._tiers:
            count = len(channels)
            start = self._rotation[priority]
            for offset in range(count):
                channel = channels[(start + offset) % count]
                queue = self._outbound[channel]
                if queue:
                    self._rotation[priority] = (start + offset + 1) % count
                    return queue.popleft()
        return None

    def _writer_loop(self) -> None:
        while True:
            with self._out_cond:
                item = None
                while True:
                    if self._stop.is_set():
                        return
                    item = self._next_outbound()
                    if item is not None:
                        break
                    self._out_cond.wait(self._tick_s)
            try:
                self._connection.send_all(item.encoded)
            except ConnectionClosed:
                self._shutdown(SessionEndReason.PEER_CLOSED)
                return
            except OSError:
                self._shutdown(SessionEndReason.TRANSPORT_ERROR)
                return
            stats = self._stats[item.frame.channel]
            stats.sent += 1
            stats.bytes_sent += len(item.encoded)
            if item.frame.extensions.get(HEARTBEAT_KEY):
                self._heartbeats_sent += 1

    # -- receiving -------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = read_frame(self._connection.recv_exact)
            except ConnectionClosed:
                self._shutdown(SessionEndReason.PEER_CLOSED)
                return
            except FramingError:
                self._shutdown(SessionEndReason.FRAMING_ERROR)
                return
            except OSError:
                self._shutdown(SessionEndReason.TRANSPORT_ERROR)
                return

            t_recv = self._mono()
            self._last_rx_mono_ns = t_recv
            self._record_inbound(frame, t_recv)

    def _record_inbound(self, frame: Frame, t_recv: int) -> None:
        stats = self._stats[frame.channel]
        stats.received += 1
        stats.bytes_received += len(frame.payload)

        previous = self._last_rx_seq.get(frame.channel)
        if previous is not None and frame.seq > previous + 1:
            stats.seq_gaps += 1
            stats.missing_seqs += frame.seq - previous - 1
        if previous is None or frame.seq > previous:
            self._last_rx_seq[frame.channel] = frame.seq

        # The transport generates keepalives, so it also consumes them rather
        # than making every caller filter them out of the control channel.
        if frame.extensions.get(HEARTBEAT_KEY):
            self._heartbeats_received += 1
            return

        policy = policy_for(frame.channel)
        with self._in_cond:
            queue = self._inbound[frame.channel]
            while len(queue) >= policy.depth:
                queue.popleft()
                stats.dropped_inbound += 1
            queue.append(ReceivedMessage(frame=frame, t_recv_mono_ns=t_recv))
            stats.delivered += 1
            stats.inbound_high_water = max(stats.inbound_high_water, len(queue))
            self._in_cond.notify_all()

    def recv(self, channel: Channel, timeout: float | None = 0.0) -> ReceivedMessage | None:
        """Oldest queued message on a channel, or None on timeout or once the
        session has ended. Callers check `is_closed` to tell those apart."""
        deadline = None if timeout is None else self._mono() + int(timeout * 1e9)
        with self._in_cond:
            while True:
                queue = self._inbound[channel]
                if queue:
                    return queue.popleft()
                if self._end_reason is not None:
                    return None
                if deadline is None:
                    self._in_cond.wait(self._tick_s)
                else:
                    remaining = (deadline - self._mono()) / 1e9
                    if remaining <= 0:
                        return None
                    self._in_cond.wait(min(remaining, self._tick_s))

    def pending(self, channel: Channel) -> int:
        with self._in_cond:
            return len(self._inbound[channel])

    def outbound_pending(self, channel: Channel) -> int:
        with self._out_cond:
            return len(self._outbound[channel])

    # -- timer -----------------------------------------------------------

    def _timer_loop(self) -> None:
        while not self._stop.wait(self._tick_s):
            now = self._mono()
            if self._heartbeat_ns and now - self._last_heartbeat_tx_ns >= self._heartbeat_ns:
                self._last_heartbeat_tx_ns = now
                self.send(Channel.CONTROL, extensions={HEARTBEAT_KEY: True})
            if self._stall_ns and now - self._last_rx_mono_ns > self._stall_ns:
                self._shutdown(SessionEndReason.STALLED)
                return

    # -- reporting -------------------------------------------------------

    def stats(self) -> SessionStats:
        return SessionStats(
            session_id=self.session_id,
            peer=self.peer,
            end_reason=None if self._end_reason is None else self._end_reason.value,
            heartbeats_sent=self._heartbeats_sent,
            heartbeats_received=self._heartbeats_received,
            channels={
                channel: ChannelStats(**vars(stats)) for channel, stats in self._stats.items()
            },
        )
