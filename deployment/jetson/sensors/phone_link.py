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

**A phone that hangs up does not end the drive.** The listener keeps accepting, so a
redial is a new session on the same socket, and the supervisor tears the old one's
workers down and binds the new one in place. What it must not do is pretend the gap
did not happen: the timebase does NOT carry across. A new session is a new peer
clock, and `OneWayEstimator`'s samples from the old one are not comparable to the new
one's -- reattaching without resetting would let the first ticks of the second
session convert against the first session's offset and look perfectly healthy, which
is this module's own stated failure mode one level up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from sensors.here_feed import HereFeed
from sensors.phone_source import PhoneCameraStream, PhoneClockAdapter, PhoneGpsReader
from transport.channels import Channel
from transport.endpoint import SessionRefused, SessionStarted, TransportListener
from transport.handshake import Hello, Role
from transport.messages import MessageRouter, advisory_message_from_advisory
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


#: Distinguishes "the caller did not supply a session" from "the caller supplied
#: None", which is a real state: a link that never came up has no session at all.
_UNSET = object()


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
        acceptor: Any = None,
    ) -> None:
        # Injectable so a rebind can be driven end to end. Everything a redial
        # exercises -- the second handshake, the estimator reset, the workers coming
        # back up on a different session -- happens below the socket, and a test that
        # cannot supply a second connection cannot reach any of it.
        self._acceptor = acceptor if acceptor is not None else TcpAcceptor(host, port)
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
        #: The phone's last self-report. Read because the sensing controller backs
        #: off on thermal, and until now nothing on this side read the channel at
        #: all -- thermal_status, thermal_headroom and skin_temp_c arrived and were
        #: dropped on the floor, so the one input that argues for LOWER rates was
        #: the one the Jetson could not see.
        #: The report and the instant it arrived, as ONE value. Read through the
        #: `telemetry` and `telemetry_at_mono` properties.
        self._telemetry: tuple[Any, float] | None = None
        self.telemetry_received = 0
        #: The fourth modality. The phone streams `imu` at 50 Hz for the whole drive
        #: and nothing on this side read the channel, so the inbound deque sat
        #: permanently at its 256 depth and every later sample was discarded -- ~50 a
        #: second, each one waking all five reader threads through the shared
        #: condition. Drained and counted here so the queue does not thrash, and so a
        #: phone whose IMU never registered is distinguishable from one streaming.
        #:
        #: NOT yet an input to the controller. `plan_task29` names the IMU as half of
        #: "the free always-on tier", but `Inputs.ego_acceleration` still comes from a
        #: finite difference of 1 Hz GPS speed, so the `EVENT_ACCEL_MPS2` trigger has
        #: never seen an IMU sample. Wiring that changes what the controller reads,
        #: not what this class collects, and is left named rather than done.
        self._imu: tuple[Any, float] | None = None
        self.imu_received = 0
        #: Counted, not just named. `refused_by_reason` exists so refusals are
        #: counted rather than inferred, and the guard path was the one outcome
        #: countable only by subtraction -- one string cannot tell one bad body
        #: from six hundred.
        self.here_failures = 0
        self.peer_device_id: str | None = None
        self.pings_answered = 0
        #: What went DOWN the link. Until now nothing did: `run_demo.py --phone` took
        #: camera and GPS off the handset and sent nothing back, so the driver's panel
        #: stayed blank for the whole drive and the phone sampled at whatever it had
        #: last been set to by hand.
        self.advisories_sent = 0
        self.rate_commands_sent = 0
        #: Sends attempted with no session to send on. Counted rather than raised: a
        #: drive whose phone has dropped must keep ticking until the supervisor
        #: reattaches it, and a raise per tick would take the run down for the one
        #: condition the supervisor exists to survive.
        self.sends_without_a_session = 0
        #: Sends the session refused -- a closed or backed-up link. Distinct from
        #: having no session at all, because they mean different things about the run.
        self.sends_refused = 0
        #: One entry per session after the first: how long the link was down, and how
        #: the previous session ended. A drive that lost its phone and got it back is
        #: not the same drive as one that never lost it, and `session_id` alone cannot
        #: tell them apart afterwards.
        self.rebinds: list[dict[str, Any]] = []
        #: The finished sessions' records, in order. `to_record` reads live objects,
        #: and a rebind replaces them -- so a run whose first session proxied every
        #: frame recorded `proxied: 0` afterwards, which is precisely the failure
        #: this record's own docstring exists to prevent, reintroduced by the redial
        #: path. Kept rather than overwritten.
        self.sessions: list[dict[str, Any]] = []
        #: How the supervisor finished, or None while it is still watching. A run
        #: that gave up waiting and one still hoping look identical otherwise:
        #: no camera, no gps, no rebind entry.
        self.supervisor_ended: str | None = None
        #: Why a connection did not become a session. The diagnosis exists --
        #: "protocol version mismatch: local 2, remote 1" -- and was being pulled
        #: off the queue and dropped, so a phone that dialled in and was refused
        #: was reported to the operator as a phone that never dialled at all.
        #: Why a connection did not become a session, capped. Every other refusal
        #: account in this section is a bounded histogram over a closed vocabulary --
        #: `HereFeed.refused_by_reason`, `MessageRouter.errors_by_reason`,
        #: `TcpAcceptor.accept_errors_by_errno`, `PhoneClockAdapter.proxy_reasons`.
        #: This one kept every string, and nothing drains the listener's event queue
        #: during a live session, so a peer failing to handshake once a second put
        #: 10,800 entries and half a megabyte into a three-hour run's summary.json.
        #: The first `MAX_REFUSALS` are kept because the first ones are the
        #: diagnostic; the rest are counted.
        self.refusals: list[str] = []
        self.refusals_not_kept = 0
        self._stop = threading.Event()
        self._responder: threading.Thread | None = None
        self._here_reader: threading.Thread | None = None
        self._telemetry_reader: threading.Thread | None = None
        self._imu_reader: threading.Thread | None = None
        self._supervisor: threading.Thread | None = None
        #: How long the supervisor waits for a redial before giving up on the run.
        self.rebind_timeout_s = 120.0

    @property
    def telemetry(self) -> Any:
        """The phone's last self-report, or None."""
        held = self._telemetry
        return None if held is None else held[0]

    @property
    def telemetry_at_mono(self) -> float | None:
        """When that report arrived. Never None while `telemetry` is not."""
        held = self._telemetry
        return None if held is None else held[1]

    @property
    def imu(self) -> Any:
        held = self._imu
        return None if held is None else held[0]

    @property
    def imu_at_mono(self) -> float | None:
        held = self._imu
        return None if held is None else held[1]

    @property
    def host(self) -> str:
        return self._acceptor.host

    @property
    def port(self) -> int:
        return self._acceptor.port

    #: How many refusal strings to keep. The diagnosis is in the first few -- a
    #: version mismatch says the same thing the six hundredth time.
    MAX_REFUSALS = 50

    def _note_refusal(self, event: Any) -> None:
        if len(self.refusals) < self.MAX_REFUSALS:
            self.refusals.append(f"{event.peer}: {event.error}")
        else:
            self.refusals_not_kept += 1

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
                self._note_refusal(event)
            if isinstance(event, SessionStarted):
                self.session = event.session
                self.peer_device_id = event.handshake.remote.device_id
                self._begin()
                return True
        return False

    def _begin(self) -> bool:
        started = self._start_session_workers()
        # One supervisor for the whole run, not one per session: it outlives the
        # session it was started under, and starting another on every rebind would
        # leave a thread per redial all watching the same field.
        if self._supervisor is None:
            self._supervisor = threading.Thread(
                target=self._supervise, name="phone-supervisor", daemon=True
            )
            self._supervisor.start()
        return started

    def _start_session_workers(self) -> bool:
        """Everything that belongs to one session, in the order it comes up.

        False when a sensor could not be rebound, in which case the link is not
        usable and the caller must not pretend it is.

        The two sensor backends are built once and REBOUND afterwards. `run_demo`
        binds `camera` and `gps` at build time and the pipeline worker closes over
        both for the life of the run, so replacing the objects here reconnected the
        link to sensors nothing was reading.
        """
        self.router = MessageRouter(self.session)
        if self.camera is None or self.gps is None:
            self.camera = PhoneCameraStream(self.router, self.adapter).start()
            self.gps = PhoneGpsReader(self.router, self.adapter).start()
        else:
            # Both, and both checked. A source that refused to rebind is not a source
            # that rebound quietly: the caller used to go on to record a clean redial
            # and re-raise the redial-expected flag over it, so the run polled a dead
            # camera forever with `end_of_stream` False and nothing ever set `stop`.
            if not all((self.camera.rebind(self.router, self.adapter),
                        self.gps.rebind(self.router, self.adapter))):
                return False
        # This link supervises, so a closed session is a gap until the supervisor
        # says otherwise. Set here rather than at teardown: the reader sees the close
        # on a 5 ms poll and the supervisor on a 100 ms one, so a flag raised when
        # the session drops arrives after the stream has already ended.
        self.camera.expect_redial(True)
        self._responder = threading.Thread(
            target=self._answer_pings, name="phone-timesync", daemon=True
        )
        self._responder.start()
        self._here_reader = threading.Thread(
            target=self._read_here, name="phone-here", daemon=True
        )
        self._here_reader.start()
        self._telemetry_reader = threading.Thread(
            target=self._read_telemetry, name="phone-telemetry", daemon=True
        )
        self._telemetry_reader.start()
        self._imu_reader = threading.Thread(
            target=self._read_imu, name="phone-imu", daemon=True
        )
        self._imu_reader.start()
        return True

    def _stop_session_workers(self) -> bool:
        """Reverse of `_start_session_workers`, leaving the listener running.

        The readers return on their own once the session reports closed, so this
        joins rather than signals -- `self._stop` belongs to the run, and setting it
        here would take the supervisor down with the session it is supervising.

        False when a reader did not stop, and the caller must not rebind over it.
        Every reader re-reads `self.router` and `self.session` on each iteration, so
        one that outlived its join did not merely linger: it bound itself to the NEXT
        session and stayed there. Two readers on one channel, compounding per redial,
        splitting arrivals between two consumers of a single `HereFeed` whose counters
        are unguarded -- and on CONTROL, two `_answer_pings` threads, which is exactly
        the two-senders-on-one-channel arrangement that produces phantom sequence
        gaps. `to_record` tracks only the later thread, so the extra one is invisible.

        `_PhoneSource.rebind` guards precisely this case one layer down and calls it
        "a silent split brain". Same hazard, same answer.
        """
        stopped = True
        for worker in (self._responder, self._here_reader, self._telemetry_reader,
                       self._imu_reader):
            if worker is not None:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    stopped = False
        if self.session is not None:
            self.session.close()
        self._responder = self._here_reader = self._telemetry_reader = None
        self._imu_reader = None
        return stopped
        # The sensors are NOT stopped or dropped here. `rebind` stops and restarts
        # them, and the run is holding them by identity.

    def _supervise(self) -> None:
        """Rebind to the next phone rather than ending the run.

        A redial used to end the drive: `wait_for_phone` ran once, the readers
        returned for good on the first close, and the backends went quiet with
        nothing to restart them.
        """
        while not self._stop.is_set():
            if not getattr(self.session, "is_closed", False):
                self._stop.wait(0.1)
                continue
            if self._stop.is_set():
                break
            ended = _end_reason_of(self.session)
            down_from = time.monotonic()
            drained = self._stop_session_workers()
            if not drained:
                # A reader from the old session is still running and will follow the
                # rebind onto the next one. Ending the run beats doubling up on every
                # channel for the rest of the drive.
                self.supervisor_ended = "readers_would_not_stop"
                if self.camera is not None:
                    self.camera.expect_redial(False)
                return
            # Said out loud. The drive now stalls for up to `rebind_timeout_s` with a
            # camera that reports no frame and no end, which is the right behaviour
            # and an alarming silence to sit through without a word.
            logging.getLogger(__name__).warning(
                "phone link down (%s); waiting up to %.0fs for a redial",
                ended, self.rebind_timeout_s,
            )
            if not self._rebind(down_from=down_from, previous_end_reason=ended):
                # Falls through to the give-up branch below.
                # Now it really is over. Said out loud, and to the camera, because a
                # consumer that only ever sees "no frame right now" waits forever.
                # `or`, because `_rebind` may already have said something more
                # specific -- "rebind_failed" is a sensor that would not come back,
                # not a phone that never called, and overwriting it loses which.
                self.supervisor_ended = self.supervisor_ended or (
                    "stopped" if self._stop.is_set()
                    else f"gave_up_after_{self.rebind_timeout_s:g}s"
                )
                if self.camera is not None:
                    self.camera.expect_redial(False)
                return
        # Left the loop because the run is stopping, not because a redial failed.
        # Without this the field stays None after a clean stop, and its documented
        # meaning -- None while still watching -- is false for the whole teardown.
        self.supervisor_ended = self.supervisor_ended or "stopped"

    def _rebind(self, *, down_from: float, previous_end_reason: str | None) -> bool:
        """Wait for the next dial-in and bind it. False when none came."""
        deadline = down_from + self.rebind_timeout_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            event = self._listener.next_event(timeout=0.1)
            if isinstance(event, SessionRefused):
                self._note_refusal(event)
                continue
            if not isinstance(event, SessionStarted):
                continue
            # Everything the previous DEVICE said, kept before it is dropped.
            self.sessions.append(self._session_record())
            # A new session is a new peer clock. Carrying the old estimate over
            # would convert the new session's first stamps against the previous
            # phone's offset and report them as healthy.
            self.estimator = OneWayEstimator()
            self.adapter = PhoneClockAdapter(self.estimator)
            # And a new device is a new thermal state and a new position. The
            # estimator was reset on the argument that a new session is a new peer;
            # the same argument applies to everything else the peer reported. A
            # `nominal` reading from the phone that just hung up licensed full rates
            # on the handset that replaced it -- and a `critical` one cut a healthy
            # handset by 6.7x -- because the 10 s telemetry age gate does not cover a
            # rebind that takes seconds. `here` goes the same way: its readings
            # describe where the previous device was, for up to its 30 s window.
            self._telemetry = None
            # Reset because `_session_record` reads them straight. Left running they
            # reported the run total inside a dict labelled "what one session did",
            # so summing the sessions double-counted -- the same defect round 2 fixed
            # for the sensors' own counters and did not follow through on here.
            self.telemetry_received = 0
            self.pings_answered = 0
            self._imu = None
            self.imu_received = 0
            self.here = HereFeed()
            self.here_failure = None
            self.session = event.session
            self.peer_device_id = event.handshake.remote.device_id
            # Through `_begin`, not `_start_session_workers`. The single-supervisor
            # guard lives in `_begin`, and calling past it left the guard unreachable
            # on the only path that could ever need it -- so the test asserting one
            # supervisor was passing because `wait_for_phone` is called once.
            if not self._begin():
                # The sensors did not come back. Ended rather than left hanging: the
                # camera reports no frame and no end otherwise, so `run_demo`'s
                # worker loops forever and its teardown never runs.
                self.supervisor_ended = "rebind_failed"
                if self.camera is not None:
                    self.camera.expect_redial(False)
                return False
            # Appended AFTER the workers are up, so the entry means the rebind
            # finished. Appending first published it one statement into the camera's
            # rebind, before the gps had been touched or the reader thread existed --
            # which anything waiting on `len(rebinds)` then raced.
            self.rebinds.append({
                "down_s": time.monotonic() - down_from,
                "previous_end_reason": previous_end_reason,
                "peer_device_id": self.peer_device_id,
                "session_id": self.session.session_id,
            })
            logging.getLogger(__name__).warning(
                "phone link back after %.1fs on %s",
                self.rebinds[-1]["down_s"], self.peer_device_id,
            )
            return True
        return False

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
                self.here_failures += 1


    def _read_telemetry(self) -> None:
        """Keep the phone's latest self-report.

        Latest only, not a history: the controller asks "how hot is it now", and a
        queue of old thermal states would answer a question nobody asked while
        growing without bound on a long drive.
        """
        assert self.router is not None
        while not self._stop.is_set():
            received = self.router.recv_with_receipt(Channel.TELEMETRY, timeout=0.05)
            if received is None:
                if getattr(self.session, "is_closed", False):
                    return
                continue
            message, receipt = received
            # Stamped, like every other remote value here. Without an arrival time
            # a report from forty minutes ago is indistinguishable from one from
            # 200 ms ago, and the controller's "silence is not nominal" rule would
            # cover only a phone that never spoke -- not one that went quiet.
            # ONE store, not two. `here_feed`'s `_Snapshot` exists because "no
            # ordering of independent stores is safe", and this pair is worse than
            # the one that motivated it: a reader straddling two stores does not
            # merely misstate the age, it gets None -- and the controller's staleness
            # gate is `if age is not None`, so a None skips it. A 40 s old `critical`
            # would then cut a healthy handset by 6.7x with the gate never running.
            # Published as a single tuple, so a reader sees both or neither.
            self._telemetry = (message, receipt.t_recv_mono_ns / 1e9)
            self.telemetry_received += 1

    def send_advisory(self, advisory: Any, *, t_capture_mono_ns: int) -> bool:
        """Send the advisory back to the phone. False when there is nothing to send on.

        `t_capture_mono_ns` is the capture stamp of the frame this advisory is ABOUT,
        on our clock -- not the moment of sending. `AdvisoryMessage` carries no frame
        id, so that stamp is the only thing tying a recommendation to the frame that
        produced it, and a log in which every advisory claims to be about a frame
        captured at the instant it was sent joins to nothing afterwards.

        The phone does not read it: `AdvisoryHolder` expires on local arrival and says
        why -- "relating the two takes the timebase exchange, and a display that goes
        blank because a clock estimate wandered would be a fault invented by its own
        safety check". So this is provenance, and nothing here may come to depend on
        the phone reading it.
        """
        return self._send(advisory_message_from_advisory(advisory, t_capture_mono_ns),
                          counter="advisories_sent")

    def send_rate_command(self, command: Any) -> bool:
        """Send a rate command to the phone. False when there is nothing to send on."""
        return self._send(command, counter="rate_commands_sent")

    def _send(self, message: Any, *, counter: str) -> bool:
        """One place that knows there may be no session.

        Returning False rather than raising, and per reason. A drive whose phone has
        dropped keeps ticking while the supervisor reattaches it, so "no session" is
        an expected state for as long as the rebind takes -- but a session that
        REFUSED a message is a different fact about the run, and one counter cannot
        say which happened.
        """
        router = self.router
        if router is None or getattr(self.session, "is_closed", True):
            self.sends_without_a_session += 1
            return False
        # Nothing is caught here. `Session.send` returns False for a closed or full
        # link and raises only for our own mistakes -- InvalidMessage from the router
        # and FramingError from the codec, both by design so a caller cannot swallow
        # its own bug with the drop-and-count idiom meant for the peer's. A broad
        # `except Exception` here turned a KeyError on a malformed rates dict into a
        # link refusal, which is the same mistake one layer up.
        if not router.send(message):
            self.sends_refused += 1
            return False
        setattr(self, counter, getattr(self, counter) + 1)
        return True

    def _read_imu(self) -> None:
        """Drain the IMU channel, keeping the latest sample and the count.

        Latest only, like telemetry: a queue of old motion would answer a question
        nobody asks while growing without bound on a long drive. The reason to read
        at all is that nothing did -- see `self.imu`.
        """
        assert self.router is not None
        while not self._stop.is_set():
            received = self.router.recv_with_receipt(Channel.IMU, timeout=0.05)
            if received is None:
                if getattr(self.session, "is_closed", False):
                    return
                continue
            message, receipt = received
            self._imu = (message, receipt.t_recv_mono_ns / 1e9)
            self.imu_received += 1

    def stop(self) -> None:
        """Reverse of coming up, and safe to call when it never came up."""
        self._stop.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=3.0)
        if self._imu_reader is not None:
            self._imu_reader.join(timeout=2.0)
        # After the join, not before. `_rebind` can already be past its own `_stop`
        # check, and it finishes by calling `_on_reader_restarted` (which clears
        # `end_of_stream`) and `expect_redial(True)` -- so a clear taken first was
        # undone by a redial landing during teardown, leaving a stopped link
        # reporting an open stream. A deliberate stop is not a gap.
        if self.camera is not None:
            self.camera.expect_redial(False)
        for worker in (self._responder, self._here_reader, self._telemetry_reader):
            if worker is not None:
                worker.join(timeout=2.0)
        for source in (self.camera, self.gps):
            if source is not None:
                source.stop()
        if self.session is not None:
            self.session.close()
        self._listener.stop()

    def _wire_record(self, session: Any = _UNSET) -> dict[str, Any]:
        """What the transport itself did, per channel. Two blindnesses, one place.

        **Messages the channel threw away.** `Session._enqueue` evicts the oldest on
        overflow and returns True, so `_send` counts an enqueue as a send and
        `sends_refused` -- whose docstring says "a closed OR BACKED-UP link" -- could
        only ever see a closed one. A drive that lost eight rate commands to a
        16-deep queue and one that delivered all twenty-four wrote byte-identical
        records, on the channel whose policy is `reliable` precisely because no later
        message repeats what it said. `dropped_outbound` is a counted field, not a
        difference: this codebase's own rule is that deriving a loss by subtraction
        is how a counting bug hides.

        **Messages the decoder refused.** `MessageRouter` counts `decode_errors`,
        `errors_by_reason` and `last_error` per channel and nothing read them, so a
        phone whose every camera frame was rejected -- one missing header key, which
        is what a version skew produces -- was indistinguishable from a phone that
        sent nothing. `frames_received` counts only decodable arrivals, so it reads
        zero either way, and `decode_failures` is the JPEG counter one layer below
        the decoder that actually refused them.
        """
        record: dict[str, Any] = {}
        if self.router is not None:
            record["messages"] = self.router.to_record()
        # Read ONCE, and taken from the caller when it has already read it. Two loads
        # of `self.session` let a rebind land between them and pair one handset's peer
        # address with another handset's channel counters -- and the address is the
        # field that says which handset the counters belong to.
        #
        # Binding it inside this function alone was not enough: `to_record` read the
        # session three more times, so one returned record could carry two different
        # `session_id` values with nothing to say which was right -- and telling a row
        # taken before a redial from one taken after is what the field was added for.
        if session is _UNSET:
            session = self.session
        if session is not None:
            # The address the phone actually dialled from, off the accepted socket.
            # This is the only fact in the record that says which route the bytes
            # took: `127.0.0.1` means the phone dialled loopback and the data crossed
            # USB via `adb reverse`; a 100.x address means it crossed the tailnet.
            # `SessionStats.peer` carried it all along and this record dropped it,
            # iterating only the per-channel counters.
            stats = session.stats()
            record["peer"] = stats.peer
            # The term that closes the inbound account on the control channel. The
            # transport generates keepalives and also consumes them: `_record_inbound`
            # counts a heartbeat in `received` and returns before the queue, so it is
            # never delivered, never dropped and never abandoned. This record
            # published the other four terms and not this one, so a reader applying
            # the identity below found 241 received against 121 delivered on a run
            # that lost nothing, and could not tell that from a run that lost 120
            # messages. `SessionStats` carried both counters the whole time.
            record["heartbeats_received"] = stats.heartbeats_received
            record["heartbeats_sent"] = stats.heartbeats_sent
            # The session this row describes, so a reader can tell a record taken
            # before a redial from one taken after.
            record["session_id"] = stats.session_id
            record["channels"] = {
                channel.value: {
                    "sent": stats.sent,
                    "queued": stats.queued,
                    "dropped_outbound": stats.dropped_outbound,
                    "abandoned_outbound": stats.abandoned_outbound,
                    "received": stats.received,
                    "delivered": stats.delivered,
                    "dropped_inbound": stats.dropped_inbound,
                    # The term without which the inbound account cannot balance. On
                    # every channel but control the transport pins `received ==
                    # delivered + dropped_inbound + abandoned_inbound`; on control the
                    # identity carries `+ heartbeats_received`, published above,
                    # because the transport eats its own keepalives. This row
                    # published the first three, so
                    # a reader could not tell whether the difference was a message
                    # policy dropped or one nobody collected -- the very distinction
                    # `session.py` added the counter for. It is not derivable from
                    # the rest either, because `queued` was absent too.
                    #
                    # Reachable on the normal path, not a corner: `_rebind` snapshots
                    # this on a session that has ALREADY closed, so `_abandon_outbound`
                    # has run, and every per-session record in a drive with a redial
                    # has the property.
                    "abandoned_inbound": stats.abandoned_inbound,
                }
                for channel, stats in stats.channels.items()
            }
        return record

    def _session_record(self) -> dict[str, Any]:
        """What one session did, snapshotted before its objects are replaced.

        One read of `self.session` for the whole record, for the same reason
        `to_record` binds one: a record assembled from several reads can name two
        sessions. Only `_rebind` calls this, on the supervisor thread, which is the
        only writer of the field once the run is up -- so the disagreement is not
        reachable here today. The single read is what makes that a property of the
        code rather than of an argument about which threads run when.
        """
        session = self.session
        return {
            "session_id": None if session is None else session.session_id,
            "peer_device_id": self.peer_device_id,
            "end_reason": _end_reason_of(session),
            "pings_answered": self.pings_answered,
            "timebase": self.estimator.to_record(),
            "clock": self.adapter.to_record(),
            "camera": None if self.camera is None else self.camera.to_record(),
            "gps": None if self.gps is None else self.gps.to_record(),
            "here": self.here.to_record(),
            "telemetry_received": self.telemetry_received,
            "imu_received": self.imu_received,
            "wire": self._wire_record(session),
        }

    def to_record(self) -> dict[str, Any]:
        """Provenance for the run.

        A run where conversion worked and one where it silently did not are
        otherwise indistinguishable -- the loopback harness's own stated lesson --
        so which clock produced the stamps and how the offset was obtained is
        recorded rather than assumed.
        """
        # One read for the whole record, passed down, so every field in it describes
        # the same session.
        session = self.session
        return {
            "peer_device_id": self.peer_device_id,
            "session_id": None if session is None else session.session_id,
            # Accepted/displaced separate a run the phone left from one a second
            # device took over: with `--phone-host 0.0.0.0` any tailnet peer that
            # speaks the hello can displace a session, and `session_id` alone
            # cannot tell those apart afterwards.
            "sessions_accepted": self._listener.accepted,
            "sessions_displaced": self._listener.displaced,
            "sessions_refused": self._listener.refused,
            # `endpoint.py` says this count "is here so that a backend which does not
            # [honour the contract] shows up in the summary instead of quietly
            # accumulating threads across a drive". The only summary a drive produces
            # is this one, and it was not in it -- the counter was read by two bench
            # harnesses and nothing else.
            "handshake_workers_leaked": self._listener.handshake_workers_leaked,
            "refusals": list(self.refusals),
            # Counted, not silently truncated. A run that turned away six hundred
            # dial-ins and one that turned away fifty must not read alike.
            "refusals_not_kept": self.refusals_not_kept,
            # The wire's spelling, or a real null. `str()` recorded the string
            # "None" while the session was open and "SessionEndReason.PEER_CLOSED"
            # once it ended, so a reader filtering on the transport's own
            # `peer_closed` matched neither.
            "end_reason": _end_reason_of(session),
            "pings_answered": self.pings_answered,
            # Every session before the current one. The fields below read live
            # objects, and a rebind replaces them -- so without this a run whose
            # first session proxied every frame published `proxied: 0` afterwards.
            "sessions": list(self.sessions),
            # None while the supervisor is still watching. A run that gave up waiting
            # and one still hoping otherwise look identical: no camera, no gps, and
            # no rebind entry.
            "supervisor_ended": self.supervisor_ended,
            # What went down the link, and what could not. `sends_without_a_session`
            # being nonzero with an empty `rebinds` means the drive was sending into a
            # link that never came back, which reads nothing like a clean run.
            "sent": {
                "advisories": self.advisories_sent,
                "rate_commands": self.rate_commands_sent,
                "without_a_session": self.sends_without_a_session,
                "refused": self.sends_refused,
            },
            "rebinds": list(self.rebinds),
            "timebase": self.estimator.to_record(),
            "clock": self.adapter.to_record(),
            "camera": None if self.camera is None else self.camera.to_record(),
            "gps": None if self.gps is None else self.gps.to_record(),
            "telemetry": {
                "received": self.telemetry_received,
                # Absent rather than assumed nominal. A drive that never heard from
                # the phone must not read as a cool one.
                "thermal_status": getattr(self.telemetry, "thermal_status", None),
                "thermal_headroom": getattr(self.telemetry, "thermal_headroom", None),
                "skin_temp_c": getattr(self.telemetry, "skin_temp_c", None),
                "skin_temp_zone": getattr(self.telemetry, "skin_temp_zone", None),
                "at_mono": self.telemetry_at_mono,
                "reader_alive": bool(self._telemetry_reader and self._telemetry_reader.is_alive()),
            },
            # What the transport did, as opposed to what the readers made of it.
            "wire": self._wire_record(session),
            "imu": {
                "received": self.imu_received,
                "at_mono": self.imu_at_mono,
                "reader_alive": bool(self._imu_reader and self._imu_reader.is_alive()),
                # Named, because the count alone reads as though something consumes
                # it. Nothing does yet -- see `self.imu`.
                "feeds_the_controller": False,
            },
            "here": {**self.here.to_record(),
                     "reader_alive": bool(self._here_reader and self._here_reader.is_alive()),
                     "failure": self.here_failure,
                     "failures": self.here_failures},
        }
