"""Phone-fed sensor backends: the transport's messages behind the local interfaces.

`CameraStream` and `GpsReader` are what the pipeline consumes. These implement the
same consumer surface over a `MessageRouter`, so `pipeline.py`, `run_demo.py`,
`replay_demo.py` and `bench_latency.py` drive a phone exactly as they drive a USB
camera. `SimulatedGps` is the existing precedent for a synthetic backend behind
that seam.

**This module is the one place a phone clock becomes a Jetson clock.** That
matters more than it sounds. The pipeline decides whether a reading is fresh by
comparing it against the Jetson's `time.monotonic()`, and two devices' monotonic
clocks are not close -- measured on this pair, 67.57 hours apart, because they
count from their own boot. Unconverted, `gps_age` comes out at -243,264 s against
a 2.0 s staleness threshold, so `gps_fresh` is False on every tick of every
drive, ego speed silently falls back to its neutral value, and the loop goes on
producing advisories that look fine. Converting here, once, is what makes every
comparison downstream same-clock by construction rather than by review.

Two things are deliberately *not* converted. Frame-to-frame intervals inside the
phone's own stream -- what the tracker and the distance estimator consume -- are
better on the phone's own clock, because they carry capture-to-capture spacing
without the link's jitter in it; converting each stamp preserves those intervals
exactly, since conversion is affine. And `t_wall` is UTC epoch for log
correlation: both devices are NTP-locked, measured at 0.00 ppm slew on the Jetson
and a steady +12.00 ppm on the Mac, so the peer's wall stamp is usable directly.
It is not usable for latency math -- NTP accuracy is milliseconds and a wall clock
can step -- and nothing here uses it for that.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from sensors.camera_stream import Frame
from sensors.gps_reader import GpsDiagnostics, GpsFix
from sensors.time_sync import TimebaseStamp, now_mono
from transport.channels import Channel
from transport.timebase import TimebaseNotReady

# Proxying is normal for the first seconds of a drive and abnormal after that, so
# the reasons are counted separately rather than lumped into one number.


class PhoneClockAdapter:
    """Turns a peer capture stamp into a local one, or says it could not.

    Thread-safe: the camera and GPS backends convert from their own reader
    threads, and both share one adapter so the record covers the whole session.
    """

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self._lock = threading.Lock()
        self.converted = 0
        self.proxied = 0
        self.proxy_reasons: dict[str, int] = {}

    def stamp(self, t_capture_peer_ns: int, t_arrival_local_s: float) -> TimebaseStamp:
        """Convert, or fall back to arrival and say so.

        Arrival is a real upper bound on capture on this device's own clock,
        wrong by exactly the link segment -- 7.9 ms measured -- against staleness
        thresholds of 2 s. Marking it beats discarding sound perception because
        of a timing problem in the first seconds of every drive, and it beats
        refusing to tick, which would leave the advisory dark exactly when the
        driver has just set off.
        """
        try:
            # to_local, not to_remote: this is a PEER stamp arriving, and the
            # inverse direction. Getting it wrong is not subtle in its effect but
            # is invisible in its shape -- with the clocks 67.6 hours apart,
            # to_remote pushed every capture stamp 135 hours into the past, which
            # the estimator's extrapolation guard then refused, so every frame
            # silently took the proxy path and the run still produced advisories.
            converted = self._estimator.to_local(t_capture_peer_ns)
        except TimebaseNotReady as exc:
            with self._lock:
                self.proxied += 1
                self.proxy_reasons[exc.reason] = self.proxy_reasons.get(exc.reason, 0) + 1
            return TimebaseStamp(
                t_capture_mono=t_arrival_local_s,
                t_arrival_mono=t_arrival_local_s,
                bound_s=None,
                estimate_id=None,
                proxy=True,
            )
        with self._lock:
            self.converted += 1
        return TimebaseStamp(
            t_capture_mono=converted.t_remote_mono_ns / 1e9,
            t_arrival_mono=t_arrival_local_s,
            bound_s=converted.bound_ns / 1e9,
            estimate_id=converted.estimate_id,
            proxy=False,
        )

    def to_record(self) -> dict[str, Any]:
        with self._lock:
            total = self.converted + self.proxied
            return {
                "converted": self.converted,
                "proxied": self.proxied,
                "proxy_reasons": dict(sorted(self.proxy_reasons.items())),
                "proxy_fraction": None if total == 0 else round(self.proxied / total, 4),
            }


class _PhoneSource:
    """Shared reader-thread machinery for the two backends."""

    def __init__(self, router: Any, adapter: PhoneClockAdapter, channel: Channel,
                 name: str, poll_s: float = 0.005) -> None:
        self._router = router
        self._adapter = adapter
        self._channel = channel
        self._name = name
        self._poll_s = poll_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.messages_received = 0
        # Why the reader stopped, if it stopped for a reason. None while healthy.
        self.failure: str | None = None

    def start(self):
        # Idempotent. `PhoneLink` starts both sources as it builds them, and
        # `run_demo` then calls `start()` on whatever camera it was handed --
        # so an unguarded start put a SECOND reader thread on the same source,
        # splitting arrivals between two consumers of one router while `_thread`
        # tracked only the later one, leaving the first running after `stop()`.
        # A restart after the reader has ended still works: the guard is on the
        # thread being alive, not on having ever started.
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        # Cleared on restart. Without this a restarted source is permanently
        # "ended" with a live reader thread, and health() publishes the
        # contradictory pair reader_alive True beside end_of_stream True.
        self.failure = None
        self._on_reader_restarted()
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def rebind(self, router: Any, adapter: PhoneClockAdapter) -> bool:
        """Point this source at a new session, keeping its identity.

        The consumers hold these objects for the life of the run -- `run_demo`
        binds `camera` and `gps` once, at build time, and the pipeline worker
        closes over both. So a redial that CONSTRUCTED new sources reconnected
        the link to objects nobody was reading: the run went on polling the
        previous session's corpse while the new session's frames arrived at a
        camera with no consumer.

        Identity is the whole point of this method. Everything below the object
        changes; the object does not -- but everything the object counted about the
        PREVIOUS peer does, which is what `_on_rebound` clears. Preserving identity
        without clearing that state moves the failure one layer down instead of
        removing it.

        False when the old reader would not stop, in which case nothing was swapped.
        """
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            # `stop()` joins with a timeout and discards the result, and `start()`
            # refuses while a thread is alive -- so swapping the router here would
            # leave the old reader on the old session with `_router` pointing at the
            # new one and no live reader for it, silently.
            #
            # Reported rather than absorbed. Returning quietly let the caller go on
            # to record a clean redial over a dead source and re-raise the
            # redial-expected flag, so `end_of_stream` stayed False and the run
            # polled a source that would never produce a frame, forever -- trading a
            # silent split brain for a silent hang.
            self.failure = "reader did not stop in time for the rebind"
            self._on_reader_ended()
            return False
        self._router = router
        self._adapter = adapter
        self._on_rebound()
        self.start()
        return True

    def _on_rebound(self) -> None:
        """Forget what belonged to the previous peer. Default: the message count."""
        self.messages_received = 0

    def _loop(self) -> None:
        try:
            self._read_until_stopped()
        except BaseException as exc:  # noqa: BLE001 - see below
            # `transport/session.py` learned this the hard way and says so in
            # both its loops: a thread that dies outside its expected exception
            # types leaves the object reporting itself healthy -- here, a reader
            # with every counter at zero, `end_of_stream` False, and a consumer
            # blocking on `wait_for_fresh` forever with no way to learn why.
            # Recorded and surfaced rather than swallowed, and the source is
            # marked ended so a consumer's termination check actually fires.
            self.failure = f"{type(exc).__name__}: {exc}"
            self._on_reader_ended()
            raise

    def _session_ended(self) -> bool:
        """Whether the transport under us has finished.

        Read through the router rather than held separately, so a source cannot
        outlive the session it reads and cannot disagree with it.
        """
        session = getattr(self._router, "_session", None)
        return bool(getattr(session, "is_closed", False))

    def _read_until_stopped(self) -> None:
        while not self._stop.is_set():
            received = self._router.recv_with_receipt(self._channel, timeout=self._poll_s)
            if received is None:
                # A closed session is the phone's equivalent of a file source
                # running out, and it was the one ending that raised no signal.
                # `Session.recv` returns immediately once it has an end reason,
                # so the poll timeout stops throttling and this loop spins a core
                # at ~660k polls a second -- taken from the perception pipeline --
                # while `run_demo`'s worker, which breaks only on `end_of_stream`,
                # never ends the run at all: its `--duration-s` and `--max-ticks`
                # checks sit after a frame it will now never get.
                if self._session_ended():
                    self._on_reader_ended()
                    return
                continue
            message, receipt = received
            # Arrival from the transport's own reader, not a clock read here: a
            # stamp taken after recv returns folds the queue wait and the decode
            # into the link segment, which is the receive-side twin of the error
            # the wire stamp removes on the way out.
            arrival_s = receipt.t_recv_mono_ns / 1e9
            stamp = self._adapter.stamp(message.t_capture_mono_ns, arrival_s)
            self.messages_received += 1
            self._accept(message, receipt, stamp)

    def _accept(self, message, receipt, stamp: TimebaseStamp) -> None:  # pragma: no cover
        raise NotImplementedError

    def _on_reader_ended(self) -> None:
        """Hook for a subclass that has consumers to wake. Default: nothing."""

    def _on_reader_restarted(self) -> None:
        """The inverse hook. Default: nothing."""

    def health(self) -> dict[str, Any]:
        return {
            # A recorded failure counts as not alive even in the window between
            # the guard recording it and the thread actually unwinding.
            "reader_alive": bool(
                self._thread is not None and self._thread.is_alive()
                and self.failure is None
            ),
            "failure": self.failure,
        }


class PhoneCameraStream(_PhoneSource):
    """`CameraStream`'s consumer surface, fed from the camera channel."""

    def __init__(self, router: Any, adapter: PhoneClockAdapter, *,
                 decode: Any = None, poll_s: float = 0.005) -> None:
        super().__init__(router, adapter, Channel.CAMERA, "phone-camera", poll_s)
        self._decode = decode or _decode_jpeg
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._last_consumed_id = -1
        self._drop_counter = 0
        self.end_of_stream = False
        #: Set by `PhoneLink` for as long as a lost session may be replaced. The
        #: reader still ends -- the session under it is gone -- but the STREAM has
        #: not, and `run_demo`'s worker breaks on `end_of_stream`. Without this the
        #: drive terminated within one 5 ms poll of the phone hanging up, before the
        #: supervisor had even begun looking for the next dial-in, and the rebind
        #: then succeeded into a run that had already stopped.
        self.awaiting_link = False
        self.decode_failures = 0
        # Read by every camera consumer in the tree. `run_demo` reads
        # `file_recoveries` unconditionally in its summary block, so without it a
        # phone-fed run raised AttributeError after the work was done and before
        # the summary was written -- which made the claim that these backends
        # drive the existing entry points unchanged simply false.
        self.source = "phone:camera"
        self.file_recoveries = 0

    def _on_reader_ended(self) -> None:
        with self._cond:
            # Woken either way: a consumer blocked in `wait_for_fresh` must find out
            # that nothing is coming right now, whether or not anything will later.
            if not self.awaiting_link:
                self.end_of_stream = True
            self._cond.notify_all()

    def _on_reader_restarted(self) -> None:
        with self._cond:
            self.end_of_stream = False

    def _on_rebound(self) -> None:
        """Drop the previous phone's frames and its id high-water mark.

        Frame ids come from the peer -- `CameraPipeline` holds an `AtomicLong(0)`
        built per sensing service -- so a different handset, or the same one whose
        service restarted, counts from 1 again. `wait_for_fresh` refuses anything
        at or below `_last_consumed_id`, so a run that consumed frame 5000 from the
        first phone rejected every frame the second one ever sent, forever, with
        `reader_alive` True, `end_of_stream` False, frames arriving, and even the
        drop counter flat -- it fires on `frame_id > _last_consumed_id`, which is
        exactly the condition that is false here.

        `_drop_counter` goes too, and the reasoning that kept it was wrong. It IS a
        fact about the run -- but `to_record` is reused verbatim as the per-session
        record, so keeping one cumulative counter beside reset ones put two scopes in
        a dict labelled "what one session did": a session that received 2 frames
        reported dropping 4. Everything here is per-session now, and the run total is
        the sum over `sessions` plus the live one.
        """
        super()._on_rebound()
        with self._cond:
            self._latest = None
            self._last_consumed_id = -1
        self._drop_counter = 0
        self.decode_failures = 0

    def stop(self) -> None:
        super().stop()
        # A stopped source is an ended source. `run_demo` breaks on this, so
        # leaving it False turned a dead link into an infinite wait_for_fresh
        # loop where a file camera exits cleanly.
        #
        # `rebind` goes through here too, which is why it is `awaiting_link` and
        # not the caller that decides: during a redial the stop is a step in
        # reconnecting, not the end of anything.
        self._on_reader_ended()

    def expect_redial(self, expected: bool) -> None:
        """Whether a lost session will be replaced by another one.

        Standing state, set by whoever supervises the link -- NOT something to set
        when the session drops. The reader notices a closed session within one 5 ms
        poll and the supervisor sees it on a 100 ms one, so a flag raised at teardown
        loses that race every time and the stream ends before anyone can say it
        should not have.

        Clearing it ends the stream immediately, because by then the consumer is
        waiting on something that is not coming.
        """
        self.awaiting_link = expected
        if not expected:
            self._on_reader_ended()

    def _accept(self, message, receipt, stamp: TimebaseStamp) -> None:
        try:
            image = self._decode(message.jpeg)
        except Exception:
            # A frame we cannot decode is one frame lost, not a dead stream --
            # the same recoverability split the transport draws between a
            # malformed message and a malformed byte stream.
            self.decode_failures += 1
            return
        frame = Frame(
            image=image,
            frame_id=message.frame_id,
            t_mono=stamp.t_capture_mono,
            t_wall=receipt.frame.t_wall_ns / 1e9,
            timebase=stamp,
        )
        with self._cond:
            if self._latest is not None and self._latest.frame_id > self._last_consumed_id:
                self._drop_counter += 1
            self._latest = frame
            self._cond.notify_all()

    def wait_for_fresh(self, timeout: float = 1.0) -> Frame | None:
        deadline = now_mono() + timeout
        with self._cond:
            while self._latest is None or self._latest.frame_id <= self._last_consumed_id:
                remaining = deadline - now_mono()
                if remaining <= 0 or self.end_of_stream:
                    return None
                self._cond.wait(remaining)
            self._last_consumed_id = self._latest.frame_id
            return self._latest

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    @property
    def dropped_frames(self) -> int:
        return self._drop_counter

    def to_record(self) -> dict[str, Any]:
        return {
            "frames_received": self.messages_received,
            "frames_dropped_unconsumed": self._drop_counter,
            "decode_failures": self.decode_failures,
            "end_of_stream": self.end_of_stream,
            **self.health(),
        }


class PhoneGpsReader(_PhoneSource):
    """`GpsReader`'s consumer surface, fed from the gps channel."""

    def __init__(self, router: Any, adapter: PhoneClockAdapter, *,
                 stale_after_s: float = 2.0, poll_s: float = 0.005) -> None:
        super().__init__(router, adapter, Channel.GPS, "phone-gps", poll_s)
        self.stale_after_s = stale_after_s
        self.diagnostics = GpsDiagnostics(port_open=True, rate_configured=True)
        self._lock = threading.Lock()
        self._fix = GpsFix()

    def _accept(self, message, receipt, stamp: TimebaseStamp) -> None:
        self.diagnostics.sentences_parsed += 1
        fix = GpsFix(
            valid=message.valid,
            lat=_or_nan(message.lat),
            lon=_or_nan(message.lon),
            speed_mps=_or_nan(message.speed_mps),
            heading_deg=_or_nan(message.heading_deg),
            fix_quality=message.fix_quality,
            num_sats=message.num_sats,
            hdop=_or_nan(message.hdop),
            altitude_m=_or_nan(message.altitude_m),
            utc_epoch_s=(
                float("nan") if message.utc_epoch_ns is None
                else message.utc_epoch_ns / 1e9
            ),
            t_mono=stamp.t_capture_mono,
            t_wall=receipt.frame.t_wall_ns / 1e9,
            timebase=stamp,
        )
        with self._lock:
            self._fix = fix

    def _on_rebound(self) -> None:
        """Drop the previous handset's position.

        The camera clears its last frame; this held the last FIX and did not, so
        after a redial `latest()` served the previous device's position -- valid,
        fresh by its own age test, and stamped `measured` by the observation builder
        -- until the new phone's first record arrived. `messages_received` said the
        new phone had sent nothing while the fix said it was doing 27 m/s.

        The same argument the rebind already applies to thermal state and to the
        traffic feed: a new device is a new position. This was the one field it was
        not applied to, and the one the other two are derived from -- the V2V beacon
        gates on `fix.valid` alone with no age test, so it would have broadcast the
        previous device's position for as long as the new phone stayed quiet.
        """
        super()._on_rebound()
        with self._lock:
            self._fix = GpsFix()
        # The cumulative twin of `messages_received`, and an error string that
        # otherwise reads "reader stopped" beside a healthy rebound reader.
        self.diagnostics.sentences_parsed = 0
        self.diagnostics.last_error = None

    def latest(self) -> GpsFix:
        with self._lock:
            return self._fix

    def is_stale(self, t_mono_now: float | None = None) -> bool:
        """Symmetric, for the same reason the observation builder's gate is.

        A fix from this clock's future is not fresh, and a one-sided `>` called it
        fresh -- the dangerous half of the clock-mixing failure. The two freshness
        predicates in this codebase must not disagree about that.
        """
        age = self.latest().age_s(now_mono() if t_mono_now is None else t_mono_now)
        return not abs(age) <= self.stale_after_s

    def _on_reader_ended(self) -> None:
        self.diagnostics.last_error = self.failure or "reader stopped"

    def to_record(self) -> dict[str, Any]:
        return {"fixes_received": self.messages_received, **self.health()}


def _or_nan(value: float | None) -> float:
    """The transport's null becomes the pipeline's NaN.

    Both mean "no value" in their own layer, and the boundary between them is
    exactly here. The transport refuses to put a non-finite number on the wire,
    so a null arriving is the only way it can say this.
    """
    return float("nan") if value is None else float(value)


def _decode_jpeg(payload: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("jpeg payload did not decode")
    return image
