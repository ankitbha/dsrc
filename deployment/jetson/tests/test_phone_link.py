"""The assembly between a socket and the two backends.

Driven over a real loopback session pair rather than a mock router, because the
thing most worth pinning here is a direction the wire enforces: the Jetson answers
time-sync pings and never sends one. A mock would answer whatever it was asked.
"""

from __future__ import annotations

import threading
import time

import pytest

from policy.advisory import Advisory
from sensors.phone_link import PhoneLink
from transport.channels import Channel
from transport.handshake import Hello, Role
from transport.loopback import LoopbackAcceptor, loopback_pair
from transport.messages import GpsRecord, InvalidMessage, MessageRouter, RateCommand
from transport.session import Session
from transport.timebase import OneWaySample, now_mono_ns

NS = 1_000_000_000
PLANTED_OFFSET_NS = int(67.57 * 3600 * NS)
#: The fixture runs the phone's clock *behind* by the planted amount, so the
#: offset a receiver measures -- remote minus local -- is negative. Named rather
#: than negated inline, because a sign error here is the failure the whole
#: conversion exists to prevent and it should not hide in an expression.
TRUE_OFFSET_NS = -PLANTED_OFFSET_NS


def phone_and_jetson(offset_ns: int = PLANTED_OFFSET_NS):
    """Two sessions on one loopback pair, the phone's on a displaced clock."""
    phone_conn, jetson_conn = loopback_pair()
    phone = Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None,
                    mono_clock=lambda: now_mono_ns() - offset_ns).start()
    jetson = Session(jetson_conn, session_id=2, heartbeat_s=None,
                     stall_timeout_s=None).start()
    return phone, jetson, MessageRouter(phone), MessageRouter(jetson)


def attach(link: PhoneLink, session, router) -> None:
    """Put a link onto an existing session, skipping the accept.

    `wait_for_phone` binds a socket and waits for a dial-in; these tests are about
    what happens after that, so the session is supplied and the rest of the come-up
    runs exactly as it does in a real run.
    """
    link.session = session
    link.peer_device_id = "test-phone"
    link._begin()


class TestTimeSyncDirection:

    def test_the_jetson_answers_a_ping_and_never_sends_one(self):
        # The correctness property this module exists for. run_loopback_pipeline
        # had the Jetson run TimeSyncInitiator, which a real phone refuses:
        # checkTimeSyncDirection drops a ping arriving at a phone and counts it as
        # unknown_value, so the estimate never converges and every stamp takes the
        # proxy path. A mock router would have answered whatever it was asked.
        #
        # Asserted on the wire rather than through TimeSyncInitiator, so this pins
        # what WE send and not what the initiator makes of it.
        from transport.messages import TimeSyncMessage

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            up.send(TimeSyncMessage(t_capture_mono_ns=0, exchange_id=7))

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and link.pings_answered == 0:
                time.sleep(0.01)
            assert link.pings_answered == 1

            reply = up.recv(Channel.CONTROL, timeout=2.0)
            assert reply is not None, "the ping was taken and nothing came back"
            # A pong by the null convention, on the same exchange. A ping here --
            # `t_peer_recv_mono_ns` unset -- is the arrangement the phone refuses.
            assert reply.t_peer_recv_mono_ns is not None
            assert reply.exchange_id == 7
        finally:
            link.stop()
            phone.close()

    def test_answering_also_produces_our_own_sample(self):
        # Two jobs on one message. Answering is what the phone needs to build its
        # estimate; sampling the arrival is what we need to build ours. Doing only
        # the first leaves the phone converging while every stamp we convert goes
        # through the proxy path -- which looks like a working run.
        from transport.timebase import TimeSyncInitiator

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            initiator = TimeSyncInitiator(up)
            initiator.send_ping()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and link.estimator.samples_accepted == 0:
                initiator.pump(timeout=0.01)

            assert link.estimator.samples_accepted == 1
            estimate = link.estimator.estimate()
            assert estimate is not None
            # The planted offset, recovered from arrivals alone and sitting just
            # under it by the loopback's own delay.
            assert estimate.offset_ns <= TRUE_OFFSET_NS
            assert TRUE_OFFSET_NS - estimate.offset_ns < int(0.5 * NS)
        finally:
            link.stop()
            phone.close()

    def test_a_pong_is_not_taken_as_a_sample(self):
        # A pong's stamps mean something else entirely: its t_wire_mono_ns is the
        # responder's departure, not an initiator's send. Fed to the estimator it
        # would produce an offset built from the wrong pair of clocks.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            link.estimator.add(OneWaySample(1, t1_remote_send_ns=now_mono_ns() + TRUE_OFFSET_NS,
                                            t2_local_recv_ns=now_mono_ns()))
            before = link.estimator.samples_accepted

            # A pong by the null convention: `t_peer_recv_mono_ns` set is what
            # makes it one, so this needs no responder machinery to construct.
            from transport.messages import TimeSyncMessage
            up.send(TimeSyncMessage(
                t_capture_mono_ns=0, exchange_id=99,
                t_peer_recv_mono_ns=now_mono_ns(), t_peer_recv_wall_ns=now_mono_ns(),
                t_peer_wire_mono_ns=now_mono_ns(),
            ))

            time.sleep(0.4)
            assert link.estimator.samples_accepted == before
        finally:
            link.stop()
            phone.close()


class TestComeUpAndTeardown:

    def test_waiting_for_a_phone_that_never_dials_returns_false(self):
        # A run that asked for a phone and got none must say so. Falling back to a
        # simulator would produce a clean-looking run whose data came from nowhere
        # near a handset.
        link = PhoneLink(port=0)
        try:
            assert link.wait_for_phone(timeout_s=0.4) is False
            assert link.session is None
            assert link.camera is None and link.gps is None
        finally:
            link.stop()

    def test_stop_is_safe_when_the_phone_never_arrived(self):
        # Teardown runs on the failure path too, before anything was built.
        link = PhoneLink(port=0)
        link.wait_for_phone(timeout_s=0.1)
        link.stop()
        link.stop()

    def test_both_backends_come_up_together(self):
        # Camera and GPS share one session, so there is no arrangement where one
        # is present and the other is not. Selecting them separately is the bug
        # this asserts against.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            assert link.camera is not None
            assert link.gps is not None
        finally:
            link.stop()
            phone.close()


class TestProvenance:

    def test_the_record_says_the_offset_was_one_way(self):
        # A run where conversion worked and one where it silently did not are
        # otherwise indistinguishable. The record has to name which clock produced
        # the stamps and how the offset was obtained.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            record = link.to_record()

            assert record["timebase"]["one_way"] is True
            assert record["peer_device_id"] == "test-phone"
            assert record["session_id"] == jetson.session_id
            assert "clock" in record and "camera" in record and "gps" in record
        finally:
            link.stop()
            phone.close()


class TestArrivalStamping:
    """Which clock reading becomes t2.

    Invisible on a loopback -- the reader's stamp and a fresh read are microseconds
    apart -- so it is pinned on the helper, which can be handed a receipt that is
    deliberately old. A mutation replacing the receipt stamp with `now_mono_ns()`
    survives every test that drives the responder thread.
    """

    def test_the_readers_stamp_is_used_not_a_later_one(self):
        from sensors.phone_link import sample_from

        class Message:
            exchange_id = 3
            t_wire_mono_ns = 5_000_000_000

        class Receipt:
            # A whole 50 ms poll period behind: what a busy responder thread sees.
            t_recv_mono_ns = now_mono_ns() - 50_000_000

        sample = sample_from(Message(), Receipt())

        assert sample.t2_local_recv_ns == Receipt.t_recv_mono_ns
        # And it is genuinely earlier than reading the clock here would give, so
        # the assertion above is not satisfiable by a fresh read.
        assert sample.t2_local_recv_ns < now_mono_ns() - 40_000_000


class TestStartIsIdempotent:
    """PhoneLink starts both sources; run_demo then starts whatever it was handed.

    Unguarded, the second call put another reader thread on the same source:
    arrivals split between two consumers of one router, and `_thread` tracking only
    the later one so `stop()` left the first running for the life of the process.
    """

    def test_starting_an_already_running_source_does_not_add_a_reader(self):
        import threading

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            before = {t for t in threading.enumerate() if t.name == "phone-camera"}
            assert len(before) == 1

            link.camera.start()
            link.camera.start()

            after = {t for t in threading.enumerate() if t.name == "phone-camera"}
            assert after == before, "start() spawned another reader on a live source"
        finally:
            link.stop()
            phone.close()

    def test_a_source_whose_reader_has_ended_can_still_be_restarted(self):
        # The guard is on the thread being alive, not on having ever started, so
        # the reconnect path is untouched.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            link.camera.stop()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and link.camera._thread.is_alive():
                time.sleep(0.01)

            link.camera.start()
            assert link.camera._thread.is_alive()
        finally:
            link.stop()
            phone.close()


class TestSessionEndPropagates:
    """What happens when the phone goes away.

    `Session.recv` returns immediately once it has an end reason, so the poll
    timeouts stop throttling and every reader spins. Measured before the fix:
    ~660k polls a second across the three threads and a full core taken from the
    perception pipeline. Worse, nothing set `end_of_stream`, and `run_demo`'s
    worker breaks only on that -- its `--duration-s` and `--max-ticks` checks sit
    after a frame it will never get again -- so the run never ended at all.
    """

    def test_a_closed_session_ends_the_camera_stream_once_the_redial_is_given_up_on(self):
        # This used to assert the stream ended as soon as the session did. It no
        # longer does, and that is the point of the supervisor: a phone that hangs
        # up may redial, and ending the stream in the first 5 ms terminated the drive
        # before anyone looked. The concern behind the old assertion survives
        # unchanged -- the consumer must not wait FOREVER -- so it is now a bounded
        # wait, and the bound is the one the supervisor is working to.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        link.rebind_timeout_s = 0.3
        try:
            attach(link, jetson, down)
            assert link.camera.end_of_stream is False

            phone.close()
            # Not immediately: a redial is still possible for as long as the
            # supervisor is looking.
            time.sleep(0.1)
            assert link.camera.end_of_stream is False, \
                "the drive would have stopped before the redial was looked for"

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not link.camera.end_of_stream:
                time.sleep(0.02)

            assert link.camera.end_of_stream, "a dead session left the consumer waiting forever"
            assert link.supervisor_ended == "gave_up_after_0.3s"
        finally:
            link.stop()
            phone.close()

    def test_a_closed_session_stops_the_reader_thread_rather_than_spinning(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            reader = link.camera._thread
            phone.close()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and reader.is_alive():
                time.sleep(0.02)
            assert not reader.is_alive(), "the reader kept polling a dead session"
        finally:
            link.stop()
            phone.close()

    def test_a_closed_session_stops_the_here_reader(self):
        # The same defect F1 was about, in the thread task 27 added. Session.recv
        # returns at once once it has an end reason, so without this the reader
        # spins a core on a dead link -- and it is a new thread, so the tests that
        # cover the camera, gps and responder threads say nothing about it.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            reader = link._here_reader
            phone.close()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and reader.is_alive():
                time.sleep(0.02)
            assert not reader.is_alive(), "the here reader kept polling a dead session"
        finally:
            link.stop()
            phone.close()

    def test_a_closed_session_stops_the_responder_thread(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            responder = link._responder
            phone.close()

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and responder.is_alive():
                time.sleep(0.02)
            assert not responder.is_alive()
        finally:
            link.stop()
            phone.close()


class TestRefusalsAreReported:
    """A phone that dialled in and was turned away is not a phone that never called.

    `wait_for_phone` consumed SessionRefused events and dropped them, so a version
    mismatch -- whose diagnosis the event already carries -- reached the operator
    as "no phone dialled in", sending them to look at the network during exactly
    the bring-up where a mismatch is likeliest.
    """

    def test_a_refused_connection_is_kept_and_named(self):
        from transport.endpoint import SessionRefused

        link = PhoneLink(port=0)
        try:
            # Injected on the listener's own queue, which is where a real refusal
            # arrives; the loop must keep it rather than eat it.
            link._listener._events.put(
                SessionRefused(peer="100.75.142.126:5555",
                               error="protocol version mismatch: local 2, remote 1")
            )
            assert link.wait_for_phone(timeout_s=0.5) is False

            assert len(link.refusals) == 1
            assert "protocol version mismatch" in link.refusals[0]
            assert "100.75.142.126" in link.refusals[0]
        finally:
            link.stop()

    def test_the_record_separates_a_displaced_run_from_a_finished_one(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            # Values, not just keys. Asserting presence alone let a mutation
            # hardcode the counter to zero and still pass, which is precisely the
            # reading -- "nothing was displaced" -- that would be wrong.
            link._listener.accepted = 2
            link._listener.displaced = 1
            link._listener.refused = 3
            record = link.to_record()

            # session_id alone cannot say whether a second device took the socket.
            assert record["sessions_accepted"] == 2
            assert record["sessions_displaced"] == 1
            assert record["sessions_refused"] == 3
        finally:
            link.stop()
            phone.close()


class TestHereIngestion:
    """The `here` channel reaching the feed.

    Nothing on this side had ever opened a HERE body; the phone fetched and
    forwarded and the bytes stopped at the transport.
    """

    def test_a_here_response_crossing_the_wire_reaches_the_feed(self):
        import json as _json

        from transport.messages import HereResponse

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            shape = {"links": [{"points": [{"lat": 51.49, "lng": -0.20}]}]}
            payload = _json.dumps({"results": [
                {"location": {"shape": shape}, "currentFlow": {"jamFactor": 7.5}}
            ]}).encode()
            assert up.send(HereResponse(
                t_capture_mono_ns=now_mono_ns() + TRUE_OFFSET_NS,
                request_url="https://data.traffic.hereapi.com/v7/flow",
                status=200, content_type="application/json",
                query_lat=51.49, query_lon=-0.20, query_radius_m=1500.0,
                t_request_mono_ns=1, t_response_mono_ns=2, body=payload,
            ))

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and link.here.responses_parsed == 0:
                time.sleep(0.02)

            assert link.here.responses_parsed == 1
            assert link.to_record()["here"]["links_cached"] == 1
        finally:
            link.stop()
            phone.close()

    def test_the_record_carries_the_feed_even_before_anything_arrives(self):
        # A drive that received no traffic data must say so as a number rather
        # than by the key being absent.
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            here = link.to_record()["here"]
            assert here["responses_received"] == 0
            assert here["last_outcome"] == "no_response_yet"
            assert here["feed_lag_s"] is None
        finally:
            link.stop()
            phone.close()


class TestHereResponseAge:
    """Which of the phone's two stamps ages a HERE body.

    The phone sets `t_capture_mono_ns` to the moment it ISSUED the call --
    `HerePipeline` passes `call.requestMonoNs`, taken before `openConnection` -- so
    ageing from it charges the whole HTTP round trip to the traffic data. Both
    stamps are on the wire precisely so a receiver can tell a slow road from a slow
    API without guessing.

    Needs a converged timebase: on the proxy path the adapter ignores the peer
    stamp and returns arrival, so the two are indistinguishable and a mutation
    swapping them survives.
    """

    def test_the_age_comes_from_the_response_stamp_not_the_request(self):
        import json as _json

        from transport.messages import HereResponse
        from transport.timebase import MIN_OFFSET_SAMPLES, OneWaySample

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            # Converge the one-way estimator on the fixture's planted offset.
            for i in range(MIN_OFFSET_SAMPLES + 1):
                local = now_mono_ns()
                link.estimator.add(OneWaySample(
                    i, t1_remote_send_ns=local + TRUE_OFFSET_NS, t2_local_recv_ns=local))
            assert link.estimator.estimate() is not None

            # A fetch issued 8 s before the bytes came back.
            responded = now_mono_ns() + TRUE_OFFSET_NS
            requested = responded - 8 * NS
            shape = {"links": [{"points": [{"lat": 51.49, "lng": -0.20}]}]}
            payload = _json.dumps({"results": [
                {"location": {"shape": shape}, "currentFlow": {"jamFactor": 3.0}}
            ]}).encode()
            assert up.send(HereResponse(
                t_capture_mono_ns=requested,
                request_url="https://data.traffic.hereapi.com/v7/flow",
                status=200, content_type="application/json",
                query_lat=51.49, query_lon=-0.20, query_radius_m=1500.0,
                t_request_mono_ns=requested, t_response_mono_ns=responded, body=payload,
            ))

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and link.here.responses_parsed == 0:
                time.sleep(0.02)
            assert link.here.responses_parsed == 1

            from sensors.gps_reader import GpsFix
            reading = link.here.at(
                GpsFix(valid=True, lat=51.49, lon=-0.20, heading_deg=90.0, speed_mps=20.0),
                t_mono=now_mono_ns() / 1e9,
            )
            # Aged from the response: a second or two. From the request it would be
            # eight seconds older, which also eats most of the 30 s staleness limit.
            assert reading.response_age_s is not None
            assert reading.response_age_s < 4.0, (
                f"aged {reading.response_age_s:.1f}s -- the HTTP round trip was "
                "charged to the traffic data"
            )
        finally:
            link.stop()
            phone.close()


class TestTelemetryIngestion:
    """The channel the Jetson never read.

    thermal_status, thermal_headroom and skin_temp_c arrived and were dropped on the
    floor, so the one input that argues for LOWER rates was the one this side could
    not see.
    """

    def test_a_telemetry_frame_reaches_the_link(self):
        from transport.messages import PhoneTelemetry

        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            assert up.send(PhoneTelemetry(
                t_capture_mono_ns=now_mono_ns() + TRUE_OFFSET_NS,
                thermal_status="moderate", thermal_headroom=None,
                achieved={"camera_hz": 4.9, "gps_hz": 1.0, "imu_hz": 49.8, "here_hz": 0.2},
                dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0},
                here_calls=0, here_errors=0, skin_temp_c=41.5, skin_temp_zone="xo_therm",
            ))

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and link.telemetry_received == 0:
                time.sleep(0.02)

            assert link.telemetry_received == 1
            record = link.to_record()["telemetry"]
            assert record["thermal_status"] == "moderate"
            assert record["skin_temp_c"] == 41.5
        finally:
            link.stop()
            phone.close()

    def test_a_drive_that_heard_nothing_says_so_rather_than_reading_cool(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            record = link.to_record()["telemetry"]
            assert record["received"] == 0
            assert record["thermal_status"] is None
            assert record["skin_temp_c"] is None
        finally:
            link.stop()
            phone.close()

    def test_a_closed_session_stops_the_telemetry_reader(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            reader = link._telemetry_reader
            phone.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and reader.is_alive():
                time.sleep(0.02)
            assert not reader.is_alive()
        finally:
            link.stop()
            phone.close()


def test_telemetry_carries_its_arrival_time():
    # Every other remote value in this tree is stamped and age-gated
    # (`PhoneGpsReader.is_stale`, `HereFeed`'s response age). Telemetry was the one
    # that was not, so a report from forty minutes ago looked like a fresh one and
    # a phone that went quiet read as a cool phone for the rest of the drive.
    from transport.messages import PhoneTelemetry

    phone, jetson, up, down = phone_and_jetson()
    link = PhoneLink()
    try:
        attach(link, jetson, down)
        assert link.telemetry_at_mono is None
        up.send(PhoneTelemetry(
            t_capture_mono_ns=now_mono_ns() + TRUE_OFFSET_NS,
            thermal_status="nominal", thermal_headroom=None,
            achieved={"camera_hz": 1.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.2},
            dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0},
            here_calls=0, here_errors=0,
        ))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and link.telemetry_received == 0:
            time.sleep(0.02)

        assert link.telemetry_at_mono is not None
        assert link.to_record()["telemetry"]["at_mono"] == link.telemetry_at_mono
    finally:
        link.stop()
        phone.close()


class LoopbackPhone:
    """A loopback client that speaks the protocol, standing in for the app."""

    def __init__(self, acceptor, device_id="moto-g-power"):
        from transport.handshake import perform_handshake

        self.connection = acceptor.connect(device_id)
        self.handshake = perform_handshake(
            self.connection, Hello(device_id, Role.PHONE)
        )
        self.session = Session(self.connection, session_id=99, heartbeat_s=None,
                               stall_timeout_s=None).start()

    def hang_up(self):
        self.session.close()


def dial(acceptor, device_id):
    """Dial in from another thread, and hand back the phone once it is up.

    The handshake is synchronous on the client side, so calling `LoopbackPhone`
    inline before `wait_for_phone` deadlocks: nothing has started the listener that
    would answer it. Off-thread, the dial waits for the listener to come up the way
    a real handset does.
    """
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(LoopbackPhone, acceptor, device_id)
    pool.shutdown(wait=False)
    return future


def wait_until(predicate, timeout=3.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestTheReturnPath:
    """Until now nothing went down the link at all."""

    def test_an_advisory_carries_the_frames_capture_stamp_not_now(self):
        # `AdvisoryMessage` has no frame id, so `t_capture_mono_ns` is the only thing
        # tying a recommendation to the frame that produced it. Sending `now` instead
        # would make every advisory claim to be about a frame captured at the instant
        # it was sent, and the log would join to nothing.
        phone, jetson, phone_router, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            capture = 1_234_567_890
            assert link.send_advisory(_advisory(), t_capture_mono_ns=capture) is True
            received = phone_router.recv(Channel.ADVISORY, timeout=2.0)
            assert received is not None
            assert received.t_capture_mono_ns == capture
            assert link.advisories_sent == 1
        finally:
            link.stop()
            phone.close()

    def test_a_rate_command_reaches_the_phone(self):
        phone, jetson, phone_router, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            assert link.send_rate_command(_rate_command()) is True
            received = phone_router.recv(Channel.RATE_CMD, timeout=2.0)
            assert received is not None
            assert received.trigger == "idle"
            assert link.rate_commands_sent == 1
        finally:
            link.stop()
            phone.close()

    def test_sending_with_no_session_is_false_and_not_an_exception(self):
        # The drive keeps ticking while the supervisor reattaches the phone. A raise
        # per tick would take the run down for the one condition it exists to survive.
        link = PhoneLink(acceptor=LoopbackAcceptor())
        assert link.send_advisory(_advisory(), t_capture_mono_ns=1) is False
        assert link.send_rate_command(_rate_command()) is False
        assert link.sends_without_a_session == 2
        assert link.advisories_sent == 0 and link.rate_commands_sent == 0
        assert link.to_record()["sent"]["without_a_session"] == 2

    def test_a_closed_session_counts_as_having_none_rather_than_as_a_refusal(self):
        # Two different facts about a run: a link that is down between sessions, and
        # a session that turned a message away. One counter could not say which.
        phone, jetson, _, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            jetson.close()
            assert link.send_advisory(_advisory(), t_capture_mono_ns=1) is False
            assert link.sends_without_a_session == 1
            assert link.sends_refused == 0
        finally:
            link.stop()
            phone.close()

    def test_our_own_bad_message_is_raised_not_counted(self):
        # The router raises InvalidMessage precisely so a caller cannot swallow its
        # own bug with the drop-and-count idiom meant for the peer's mistakes.
        phone, jetson, _, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            with pytest.raises(InvalidMessage):
                link.send_rate_command(_rate_command(rates={
                    "camera_hz": 0.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.05}))
            assert link.sends_refused == 0
            assert link.rate_commands_sent == 0
        finally:
            link.stop()
            phone.close()


def _advisory(**over):
    fields = dict(recommended_speed_mps=13.4, recommended_speed_display=30.0,
                  current_speed_display=28.0, units="mph", headway_target_s=2.0,
                  lane_text="keep lane", merge_text="no merge", traffic_text="moderate",
                  confidence=0.8, confidence_label="high",
                  action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                          "lane_preference": "keep", "merge_mode": "normal"})
    fields.update(over)
    return Advisory(**fields)


def _rate_command(**over):
    fields = dict(t_capture_mono_ns=7, trigger="idle", shadow=False,
                  rates={"camera_hz": 5.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.05})
    fields.update(over)
    return RateCommand(**fields)


class TestRedial:
    """A phone that hangs up used to end the drive."""

    def test_a_second_phone_is_bound_without_restarting_the_run(self):
        # `wait_for_phone` ran once, the three readers returned for good on the first
        # close, and the backends went quiet with nothing to restart them. The
        # listener was still accepting the whole time.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            assert link.peer_device_id == "phone-one"
            first_session_id = link.session.session_id

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")

            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"
            assert link.peer_device_id == "phone-two"
            assert link.session.session_id != first_session_id
            # The workers came back up on the new session, not just the field.
            assert wait_until(lambda: link.camera is not None and link.gps is not None)
            assert link.router is not None and link.router.session is link.session
            second.hang_up()
        finally:
            link.stop()

    def test_the_timebase_does_not_carry_across_a_rebind(self):
        # A new session is a new peer clock. Samples from the old one are not
        # comparable to the new one's, so reattaching without resetting would let the
        # second session's first ticks convert against the first session's offset --
        # and look perfectly healthy while doing it, which is this module's own
        # stated failure mode one level up.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            link.estimator.add(OneWaySample(exchange_id=1, t1_remote_send_ns=10,
                                            t2_local_recv_ns=20))
            stale = link.estimator
            assert stale.to_record()["samples_accepted"] == 1

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.estimator is not stale, "the old estimator survived the rebind"
            assert link.estimator.to_record()["samples_accepted"] == 0
            # The adapter must follow it. Keeping the old one would leave the new
            # session's stamps converting through the previous phone's offset with a
            # fresh estimator sitting unused beside it.
            assert link.adapter._estimator is link.estimator
            second.hang_up()
        finally:
            link.stop()

    def test_the_rebind_names_how_the_previous_session_ended(self):
        # "The phone lost the link" and "the phone was displaced by another device"
        # are different drives, and `session_id` alone cannot tell them apart later.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            entry = link.rebinds[0]
            assert entry["previous_end_reason"] is not None
            assert entry["peer_device_id"] == "phone-two"
            assert entry["down_s"] >= 0.0
            assert link.to_record()["rebinds"] == link.rebinds
            second.hang_up()
        finally:
            link.stop()

    def test_only_one_supervisor_exists_however_many_rebinds(self):
        # One per session would leave a thread per redial, all watching one field and
        # all racing to tear the same session down.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            phones = [dialling.result(timeout=5.0)]
            supervisor = link._supervisor
            for n in range(2, 4):
                phones[-1].hang_up()
                phones.append(LoopbackPhone(acceptor, f"phone-{n}"))
                assert wait_until(lambda n=n: len(link.rebinds) == n - 1), f"rebind {n}"
            assert link._supervisor is supervisor
            assert sum(1 for t in threading.enumerate()
                       if t.name == "phone-supervisor") == 1
            phones[-1].hang_up()
        finally:
            link.stop()


class TestTheRunKeepsItsSensors:
    """A redial reconnects the link. It must reconnect the RUN.

    `run_demo.build_components` binds `camera = phone.camera` and `gps = phone.gps`
    once, and the pipeline worker closes over both for the life of the drive. A
    rebind that constructed new backends left the run polling the previous session's
    corpse while the new session's frames arrived at objects nobody read -- and every
    test in `TestRedial` passed either way, because none of them held a reference the
    way the run does.
    """

    def _rebound(self, acceptor, link):
        dialling = dial(acceptor, "phone-one")
        assert link.wait_for_phone(timeout_s=5.0) is True
        first = dialling.result(timeout=5.0)
        held_camera, held_gps = link.camera, link.gps
        first.hang_up()
        second = LoopbackPhone(acceptor, "phone-two")
        assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"
        return held_camera, held_gps, second

    def test_the_object_the_run_is_holding_is_the_one_that_gets_rebound(self):
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            camera, gps, second = self._rebound(acceptor, link)
            assert link.camera is camera, "the run's camera was replaced"
            assert link.gps is gps, "the run's gps was replaced"
            # And it is reading the NEW session, not still polling the dead one.
            assert wait_until(lambda: camera._router is link.router)
            assert gps._router is link.router
            assert camera.health()["reader_alive"] is True
            second.hang_up()
        finally:
            link.stop()

    def test_data_from_the_second_phone_reaches_the_object_the_run_holds(self):
        # Identity alone is not enough: a rebound object that never restarted its
        # reader is the same object and just as useless.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            _, gps, second = self._rebound(acceptor, link)
            before = gps.messages_received
            assert MessageRouter(second.session).send(GpsRecord(
                t_capture_mono_ns=now_mono_ns(), valid=True, fix_quality=1,
                num_sats=9, lat=40.744, lon=-74.032, speed_mps=27.0,
                heading_deg=90.0, hdop=0.9, altitude_m=10.0,
            ))
            assert wait_until(lambda: gps.messages_received > before), \
                "the second phone's fixes reached nothing the run can see"
            second.hang_up()
        finally:
            link.stop()

    def test_the_camera_does_not_end_the_stream_while_a_redial_is_expected(self):
        # `run_demo`'s worker breaks on `end_of_stream`, and the reader notices the
        # closed session within one 5 ms poll -- so the drive used to terminate
        # before the supervisor had even begun looking for the next dial-in.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            camera = link.camera

            first.hang_up()
            # Through the whole gap, not merely at the end of it.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                assert camera.end_of_stream is False, "the run would have stopped here"
                time.sleep(0.01)

            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"
            assert camera.end_of_stream is False
            second.hang_up()
        finally:
            link.stop()

    def test_giving_up_ends_the_stream_and_says_so(self):
        # The inverse. A consumer that only ever sees "no frame right now" waits for
        # a phone that is never coming, and the run hangs instead of finishing.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        link.rebind_timeout_s = 0.3
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            camera = link.camera
            first.hang_up()

            assert wait_until(lambda: link.supervisor_ended is not None), "never gave up"
            assert link.supervisor_ended == "gave_up_after_0.3s"
            assert camera.end_of_stream is True
            assert link.to_record()["supervisor_ended"] == "gave_up_after_0.3s"
        finally:
            link.stop()


class TestWhatANewDeviceInherits:
    """The estimator is reset because a new session is a new peer. So is everything
    else the peer reported."""

    def test_the_previous_phones_thermal_report_is_not_applied_to_the_new_one(self):
        # The controller's telemetry age gate is 10 s and a rebind takes seconds, so
        # a `nominal` reading from the handset that just hung up licensed full camera
        # and HERE rates on the one that replaced it -- and a `critical` one would
        # have cut a healthy handset by 6.7x.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            link.telemetry = type("T", (), {"thermal_status": "nominal",
                                            "skin_temp_c": 28.0})()
            link.telemetry_at_mono = time.monotonic()

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.telemetry is None, "phone-two inherited phone-one's temperature"
            assert link.telemetry_at_mono is None
            second.hang_up()
        finally:
            link.stop()

    def test_the_previous_phones_traffic_feed_is_not_kept(self):
        # Its readings describe where the OTHER device was, for up to the feed's
        # 30 s staleness window.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            stale_feed = link.here

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.here is not stale_feed, "phone-two inherited phone-one's road"
            second.hang_up()
        finally:
            link.stop()

    def test_the_finished_sessions_record_is_kept_not_overwritten(self):
        # `to_record` reads live objects and a rebind replaces them, so a run whose
        # first session proxied every frame published `proxied: 0` afterwards --
        # exactly the failure this record's docstring exists to prevent.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            link.estimator.add(OneWaySample(exchange_id=1, t1_remote_send_ns=10,
                                            t2_local_recv_ns=20))
            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            record = link.to_record()
            assert len(record["sessions"]) == 1
            kept = record["sessions"][0]
            assert kept["peer_device_id"] == "phone-one"
            assert kept["timebase"]["samples_accepted"] == 1
            assert kept["end_reason"] is not None
            # And the live fields describe the CURRENT session, not the old one.
            assert record["timebase"]["samples_accepted"] == 0
            assert record["peer_device_id"] == "phone-two"
            second.hang_up()
        finally:
            link.stop()
