"""Everything between a socket and the two sensor backends, assembled once.

`PhoneCameraStream` and `PhoneGpsReader` take a router and a clock adapter. Getting
from "a phone will dial us" to those two objects means an acceptor, a listener, a
session, a router, a time-sync responder and an estimator, wired in one order and
torn down in the reverse. That sequence lived only inside
`scripts/run_loopback_pipeline.py`, so nothing but that harness could feed the
pipeline from a phone -- which is why `run_demo.py`, `replay_demo.py`,
`bench_latency.py` and `pipeline.py` mention `phone_source` zero times between them.

**The Jetson answers; it does not ask.** The harness had the Jetson run
`TimeSyncInitiator`, and its own docstring admitted that contradicts the spec and
needed sign-off. It works there because that harness owns both ends and
`transport/timebase.py` is role-symmetric. A real phone refuses: `Session`'s
`checkTimeSyncDirection` computes `wrongWay = if (role == ROLE_PHONE) !isPing else
isPing`, so a ping from us is dropped and counted as `unknown_value`, and the
estimate never converges. So the responder path here is not a variation on the
harness -- it is the only arrangement that works against the device.

Answering means we never learn when our pong landed, so we cannot build a
`TimeSyncSample`. `OneWayEstimator` is what a responder can build instead, and what
its bound does and does not cover is written on it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from sensors.here_feed import HereFeed
from sensors.phone_source import PhoneCameraStream, PhoneClockAdapter, PhoneGpsReader
from transport.channels import Channel
from transport.endpoint import SessionRefused, SessionStarted, TransportListener
from transport.handshake import Hello, Role
from transport.messages import MessageRouter
from transport.tcp import DEFAULT_PORT, TcpAcceptor
from transport.timebase import OneWayEstimator, OneWaySample, answer_ping


def sample_from(message: Any, receipt: Any) -> OneWaySample:
    """One arrival, stamped by the reader rather than by whoever got round to it.

    `receipt.t_recv_mono_ns` is taken by the session's reader at arrival. Reading a
    clock here instead would fold this thread's queue wait and the decode into the
    apparent delay -- up to a whole poll period -- and since the offset estimate is
    already the truth minus the delay, every microsecond of that goes straight into
    the error, in the one direction it already leans.

    Split out because the difference is invisible on a loopback, where the two
    stamps are microseconds apart: a test driving the thread cannot tell them
    apart, and this can be handed a receipt that is deliberately old.
    """
    return OneWaySample(
        exchange_id=message.exchange_id,
        t1_remote_send_ns=message.t_wire_mono_ns,
        t2_local_recv_ns=receipt.t_recv_mono_ns,
    )


def _end_reason_of(session: Any) -> str | None:
    """How a session ended, spelled as the rest of the repo spells it."""
    reason = getattr(session, "end_reason", None) if session is not None else None
    return getattr(reason, "value", None) if reason is not None else None


class PhoneLink:
    """One phone session, and the camera and GPS backends it feeds.

    Camera and GPS are not separable. They arrive on two channels of one session,
    so there is no arrangement where the camera comes from a phone and GPS does
    not, short of two sessions to the same handset. Callers select both or
    neither, and this class is the unit that gets selected.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        device_id: str = "jetson",
        heartbeat_s: float = 1.0,
        stall_timeout_s: float = 5.0,
        handshake_timeout_s: float = 5.0,
    ) -> None:
        self._acceptor = TcpAcceptor(host, port)
        self._listener = TransportListener(
            self._acceptor,
            Hello(device_id=device_id, role=Role.JETSON),
            heartbeat_s=heartbeat_s,
            stall_timeout_s=stall_timeout_s,
            handshake_timeout_s=handshake_timeout_s,
            accept_poll_s=0.2,
        )
        self.estimator = OneWayEstimator()
        self.adapter = PhoneClockAdapter(self.estimator)
        self.session: Any = None
        self.router: MessageRouter | None = None
        self.camera: PhoneCameraStream | None = None
        self.gps: PhoneGpsReader | None = None
        #: Filled from the `here` channel. Not a sensor backend: nothing polls it
        #: on a thread, because a query is answered against the caller's own
        #: position at the moment it asks, not at the moment the bytes arrived.
        self.here = HereFeed()
        self.here_failure: str | None = None
        self.peer_device_id: str | None = None
        self.pings_answered = 0
        #: Why a connection did not become a session. The diagnosis exists --
        #: "protocol version mismatch: local 2, remote 1" -- and was being pulled
        #: off the queue and dropped, so a phone that dialled in and was refused
        #: was reported to the operator as a phone that never dialled at all.
        self.refusals: list[str] = []
        self._stop = threading.Event()
        self._responder: threading.Thread | None = None
        self._here_reader: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._acceptor.host

    @property
    def port(self) -> int:
        return self._acceptor.port

    def wait_for_phone(self, timeout_s: float) -> bool:
        """Listen until a phone dials in, or give up and say so.

        The phone dials because the Jetson cannot: its Tegra kernel has
        `CONFIG_NF_CONNTRACK_MARK` unset, so Tailscale cannot install its connmark
        rules and the Jetson cannot originate traffic to tailnet peers. Waiting is
        therefore the only option, and a bounded wait that returns False beats one
        that blocks a drive forever on a phone nobody started.
        """
        self._listener.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            event = self._listener.next_event(timeout=0.1)
            if isinstance(event, SessionRefused):
                self.refusals.append(f"{event.peer}: {event.error}")
            if isinstance(event, SessionStarted):
                self.session = event.session
                self.peer_device_id = event.handshake.remote.device_id
                self._begin()
                return True
        return False

    def _begin(self) -> None:
        self.router = MessageRouter(self.session)
        self.camera = PhoneCameraStream(self.router, self.adapter).start()
        self.gps = PhoneGpsReader(self.router, self.adapter).start()
        self._responder = threading.Thread(
            target=self._answer_pings, name="phone-timesync", daemon=True
        )
        self._responder.start()
        self._here_reader = threading.Thread(
            target=self._read_here, name="phone-here", daemon=True
        )
        self._here_reader.start()

    def _answer_pings(self) -> None:
        """Answer every ping, and take the arrival as a one-way sample.

        Two jobs on one message and both matter. Answering is what the phone needs
        to build *its* estimate; sampling is what we need to build ours. Doing only
        the first would leave the phone converging while every stamp we convert
        went through the proxy path.
        """
        assert self.router is not None
        while not self._stop.is_set():
            received = self.router.recv_with_receipt(Channel.CONTROL, timeout=0.05)
            if received is None:
                # Same spin as the sensor readers: once the session has an end
                # reason `recv` returns at once and the timeout throttles nothing.
                if getattr(self.session, "is_closed", False):
                    return
                continue
            message, receipt = received
            ping = getattr(message, "t_peer_recv_mono_ns", None) is None
            if not ping:
                # A pong reaching a Jetson is a protocol error by the spec, and the
                # session already counts it. Ignored here rather than fed to the
                # estimator, where its stamps mean something else entirely.
                continue
            self.estimator.add(sample_from(message, receipt))
            self.router.send(answer_ping((message, receipt)))
            self.pings_answered += 1

    def _read_here(self) -> None:
        """Hand every HERE response to the feed, with its arrival stamp.

        The arrival is the reader's, converted like every other phone stamp, so
        response age is measured on the same clock the pipeline asks freshness
        questions on. Taking the time here instead would fold this thread's queue
        wait into the age of the traffic data.
        """
        assert self.router is not None
        while not self._stop.is_set():
            received = self.router.recv_with_receipt(Channel.HERE, timeout=0.05)
            if received is None:
                if getattr(self.session, "is_closed", False):
                    return
                continue
            message, receipt = received
            # `t_response_mono_ns`, not `t_capture_mono_ns`. The phone sets capture
            # to the moment it ISSUED the call -- `HerePipeline` passes
            # `call.requestMonoNs`, stamped before `openConnection` -- so using it
            # charged the whole HTTP round trip to the age of the traffic data. An
            # 8 s cellular fetch made a body that arrived 50 ms ago report an age
            # of 8.05 s, and ate most of the 30 s staleness limit. Both stamps exist
            # so a receiver can tell a slow road from a slow API without guessing.
            stamp = self.adapter.stamp(message.t_response_mono_ns, receipt.t_recv_mono_ns / 1e9)
            try:
                self.here.offer(
                    status=message.status,
                    body=message.body,
                    received_t_mono=stamp.t_capture_mono,
                    bound_s=stamp.bound_s,
                    proxy=stamp.proxy,
                )
            except Exception as exc:  # noqa: BLE001
                # Last resort. Every remote body is meant to end as a named outcome
                # inside the feed, and one that does not must still not take the
                # reader down for the rest of the drive -- which is what an
                # OverflowError out of `float()` did. Recorded so a dead-quiet feed
                # is visible rather than inferred.
                self.here_failure = f"{type(exc).__name__}: {exc}"


    def stop(self) -> None:
        """Reverse of coming up, and safe to call when it never came up."""
        self._stop.set()
        for worker in (self._responder, self._here_reader):
            if worker is not None:
                worker.join(timeout=2.0)
        for source in (self.camera, self.gps):
            if source is not None:
                source.stop()
        if self.session is not None:
            self.session.close()
        self._listener.stop()

    def to_record(self) -> dict[str, Any]:
        """Provenance for the run.

        A run where conversion worked and one where it silently did not are
        otherwise indistinguishable -- the loopback harness's own stated lesson --
        so which clock produced the stamps and how the offset was obtained is
        recorded rather than assumed.
        """
        return {
            "peer_device_id": self.peer_device_id,
            "session_id": None if self.session is None else self.session.session_id,
            # Accepted/displaced separate a run the phone left from one a second
            # device took over: with `--phone-host 0.0.0.0` any tailnet peer that
            # speaks the hello can displace a session, and `session_id` alone
            # cannot tell those apart afterwards.
            "sessions_accepted": self._listener.accepted,
            "sessions_displaced": self._listener.displaced,
            "sessions_refused": self._listener.refused,
            "refusals": list(self.refusals),
            # The wire's spelling, or a real null. `str()` recorded the string
            # "None" while the session was open and "SessionEndReason.PEER_CLOSED"
            # once it ended, so a reader filtering on the transport's own
            # `peer_closed` matched neither.
            "end_reason": _end_reason_of(self.session),
            "pings_answered": self.pings_answered,
            "timebase": self.estimator.to_record(),
            "clock": self.adapter.to_record(),
            "camera": None if self.camera is None else self.camera.to_record(),
            "gps": None if self.gps is None else self.gps.to_record(),
            "here": {**self.here.to_record(),
                     "reader_alive": bool(self._here_reader and self._here_reader.is_alive()),
                     "failure": self.here_failure},
        }
