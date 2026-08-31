"""The assembly between a socket and the two backends.

Every `PhoneLink` here is given a `LoopbackAcceptor`. The default binds
`0.0.0.0:47811`, and nothing in this file uses the socket -- each test either
attaches an existing loopback session or drives a loopback client -- but the bind
still happens, and four of these tests failed 6 times in 40 runs with
`OSError: Address already in use`. A concurrent pytest run, a leftover process, or a
real `run_demo.py --phone` on the same machine breaks them, and the noise lands in
mutation verdicts as false catches.

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
from transport.messages import (CameraFrame, GpsRecord, ImuSample, InvalidMessage,
                                MessageRouter, RateCommand)
from transport.session import Session
from transport.timebase import OneWaySample, now_mono_ns

NS = 1_000_000_000
PLANTED_OFFSET_NS = int(67.57 * 3600 * NS)
#: The fixture runs the phone's clock *ahead* by the planted amount, so the offset a
#: receiver measures -- remote minus local -- is positive. Named rather than written
#: inline, because a sign error here is the failure the whole conversion exists to
#: prevent and it should not hide in an expression.
#:
#: Ahead and not behind, because behind depends on the machine. The fixture used to
#: build the phone's clock as `now_mono_ns() - PLANTED_OFFSET_NS`, which is negative
#: whenever the process monotonic clock is below 67.57 hours -- and the transport
#: correctly refuses a negative `t_wire_mono_ns`, so two tests here failed on any
#: machine that had not been awake for three days. macOS excludes sleep from
#: `time.monotonic()`, so a laptop reads far below its wall uptime: measured at 12.39
#: hours against 1 day 12:51 of uptime, giving a phone clock of -55.18 hours.
#:
#: Adding keeps both clocks positive on every machine and leaves everything in the
#: real monotonic domain, so arrival stamps still compare against `time.monotonic()`
#: elsewhere. The negative direction is still covered, by planting a sample directly
#: rather than by displacing a whole session.
TRUE_OFFSET_NS = PLANTED_OFFSET_NS


def phone_and_jetson(offset_ns: int = PLANTED_OFFSET_NS):
    """Two sessions on one loopback pair, the phone's on a displaced clock."""
    phone_conn, jetson_conn = loopback_pair()
    phone = Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None,
                    mono_clock=lambda: now_mono_ns() + offset_ns).start()
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
        link = PhoneLink(acceptor=LoopbackAcceptor())
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
    link = PhoneLink(acceptor=LoopbackAcceptor())
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
            # Set as the pair, because that is the invariant: the report and its
            # arrival time are one value, so a reader cannot see a status without an
            # age -- which the controller's staleness gate would have skipped.
            link._telemetry = (type("T", (), {"thermal_status": "nominal",
                                              "skin_temp_c": 28.0})(),
                               time.monotonic())

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


def a_jpeg() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class TestTheSecondPhoneStartsClean:
    """What preserving the sensors' identity across a rebind carried with it."""

    def test_a_second_phone_low_frame_ids_are_not_refused_forever(self):
        # `wait_for_fresh` gates on `frame_id <= _last_consumed_id`, and frame ids
        # come from the peer -- `CameraPipeline` holds an AtomicLong(0) built per
        # sensing service, so a different handset counts from 1 again. Keeping the
        # camera object across a rebind, which is what makes the run survive a
        # redial, carried the previous phone's high-water mark with it: a run that
        # consumed frame 5000 refused every frame the second phone ever sent.
        #
        # Silent in every health field: reader_alive True, end_of_stream False,
        # frames arriving, and the drop counter flat -- it fires on
        # `frame_id > _last_consumed_id`, exactly the condition that is false here.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            camera = link.camera

            assert MessageRouter(first.session).send(CameraFrame(
                t_capture_mono_ns=now_mono_ns(), frame_id=5000, width=16, height=16,
                format="jpeg", quality=85, jpeg=a_jpeg()))
            assert wait_until(lambda: camera.wait_for_fresh(0.05) is not None), \
                "the first phone's frame never arrived"

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            # Nothing is served before phone-two speaks. Asserting only that SOME
            # frame arrives is satisfied by phone-one's, re-served: clearing the
            # high-water mark without clearing `_latest` hands back frame 5000, which
            # raises the mark to 5000 again and refuses frame 1 forever -- the whole
            # defect, past a green test.
            assert camera.wait_for_fresh(0.05) is None, \
                "a frame from the phone that hung up was served to the new session"

            assert MessageRouter(second.session).send(CameraFrame(
                t_capture_mono_ns=now_mono_ns(), frame_id=1, width=16, height=16,
                format="jpeg", quality=85, jpeg=a_jpeg()))
            served = None
            deadline = time.monotonic() + 3.0
            while served is None and time.monotonic() < deadline:
                served = camera.wait_for_fresh(0.05)
            assert served is not None, \
                "the second phone's frames were refused; the run goes blind and " \
                "never ends, because end_of_stream is False"
            assert served.frame_id == 1, f"served frame {served.frame_id}, not phone-two's"
            second.hang_up()
        finally:
            link.stop()

    def test_the_rebind_entry_means_the_rebind_finished(self):
        # Anything waiting on `len(rebinds)` is waiting for the link to be usable.
        # Appending before `_begin()` published it one statement into the camera's
        # rebind -- before the gps had been touched or a reader thread existed --
        # so every test using it raced, and one failed 3 times in 40 idle runs.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")

            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"
            # No further waiting. If the entry means anything, all of this holds now.
            assert link.camera._router is link.router
            assert link.gps._router is link.router
            assert link.camera.health()["reader_alive"] is True
            assert link.gps.health()["reader_alive"] is True
            second.hang_up()
        finally:
            link.stop()

    def test_each_sessions_record_counts_only_that_session(self):
        # The sensors' counters are cumulative on the object, and the object now
        # survives the rebind -- so a per-session record reading them straight
        # reported session two's frames as session one's plus session two's, next to
        # a clock that had counted only session two. An offline reader summing the
        # sessions double-counts.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            up = MessageRouter(first.session)
            for i in range(3):
                assert up.send(GpsRecord(
                    t_capture_mono_ns=now_mono_ns() + i, valid=True, fix_quality=1,
                    num_sats=9, lat=40.0, lon=-74.0, speed_mps=27.0,
                    heading_deg=90.0, hdop=0.9, altitude_m=10.0))
            assert wait_until(lambda: link.gps.messages_received == 3)

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.to_record()["sessions"][0]["gps"]["fixes_received"] == 3
            # And the new session starts from zero rather than inheriting the total.
            assert link.gps.messages_received == 0
            second.hang_up()
        finally:
            link.stop()


class TestTheGpsStartsCleanToo:
    """The camera's defect, in the sensor everything else is derived from."""

    def test_the_previous_handsets_position_is_not_served_to_the_new_session(self):
        # `PhoneCameraStream` clears its last frame on a rebind; the gps reader held
        # the last FIX and did not. After a redial `latest()` returned the previous
        # device's position -- `valid`, fresh by its own age test, and stamped
        # `measured` by the observation builder -- while `messages_received` said the
        # new phone had sent nothing. The V2V beacon gates on `fix.valid` alone with
        # no age test, so it would have broadcast that position for as long as the
        # new phone stayed quiet.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            gps = link.gps
            assert MessageRouter(first.session).send(GpsRecord(
                t_capture_mono_ns=now_mono_ns(), valid=True, fix_quality=1,
                num_sats=9, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
                hdop=0.9, altitude_m=10.0))
            assert wait_until(lambda: gps.latest().valid)

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            after = gps.latest()
            assert after.valid is False, (
                f"phone-two inherited phone-one's position: {after.lat}, {after.lon}"
            )
            assert gps.messages_received == 0
            second.hang_up()
        finally:
            link.stop()

    def test_the_new_session_does_not_inherit_the_old_ones_parse_count(self):
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            assert MessageRouter(first.session).send(GpsRecord(
                t_capture_mono_ns=now_mono_ns(), valid=True, fix_quality=1,
                num_sats=9, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
                hdop=0.9, altitude_m=10.0))
            assert wait_until(lambda: link.gps.diagnostics.sentences_parsed == 1)

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.gps.diagnostics.sentences_parsed == 0
            # And not still reporting why the PREVIOUS reader stopped.
            assert link.gps.diagnostics.last_error is None
            second.hang_up()
        finally:
            link.stop()


class TestOneRecordOneScope:
    """`to_record` is reused verbatim as the per-session record."""

    def test_a_sessions_dropped_frames_are_that_sessions(self):
        # Round 2 made `frames_received` per-session and deliberately kept
        # `_drop_counter` cumulative, so one dict labelled "what one session did"
        # carried two scopes: a session that received 2 frames reported dropping 4,
        # and summing the sessions double-counted the drops it had just stopped
        # double-counting for the frames.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            up = MessageRouter(first.session)
            # Never consumed, so each one displaces the last and counts as a drop.
            # Sent one at a time and waited for: `camera` is latest_wins at depth
            # one, so a burst is collapsed by the TRANSPORT and never reaches the
            # reader that does the counting.
            for i in range(4):
                assert up.send(CameraFrame(
                    t_capture_mono_ns=now_mono_ns() + i, frame_id=i, width=16,
                    height=16, format="jpeg", quality=85, jpeg=a_jpeg()))
                assert wait_until(lambda n=i: link.camera.messages_received == n + 1)
            dropped_in_session_one = link.camera.to_record()["frames_dropped_unconsumed"]
            assert dropped_in_session_one >= 1
            # And one body the decoder cannot read, so `decode_failures` is nonzero
            # going into the rebind rather than trivially already zero.
            assert up.send(CameraFrame(
                t_capture_mono_ns=now_mono_ns() + 9, frame_id=9, width=16,
                height=16, format="jpeg", quality=85, jpeg=b"not a jpeg at all"))
            assert wait_until(lambda: link.camera.decode_failures == 1)

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            record = link.to_record()
            assert record["sessions"][0]["camera"]["frames_dropped_unconsumed"] == \
                dropped_in_session_one
            live = link.camera.to_record()
            assert live["frames_received"] == 0
            assert live["frames_dropped_unconsumed"] == 0, \
                "the new session inherited the old session's drops"
            # Every counter in the dict, not the two that were noticed. This one
            # survived its own pin: nothing asserted it, in either direction.
            assert live["decode_failures"] == 0, \
                "the new session inherited the old session's decode failures"
            second.hang_up()
        finally:
            link.stop()

    def test_pings_and_telemetry_are_counted_per_session_too(self):
        # The same two-scopes-in-one-dict defect, two fields further on: both are
        # read straight into `_session_record`.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            link.pings_answered = 2
            link.telemetry_received = 5

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: len(link.rebinds) == 1), "never rebound"

            assert link.to_record()["sessions"][0]["pings_answered"] == 2
            assert link.to_record()["sessions"][0]["telemetry_received"] == 5
            assert link.pings_answered == 0
            assert link.telemetry_received == 0
            second.hang_up()
        finally:
            link.stop()


class TestARebindThatCouldNotHappen:

    def test_a_sensor_that_refuses_to_rebind_ends_the_run_rather_than_hanging_it(self):
        # The failed-stop guard used to return quietly, and the caller went on to
        # record a clean redial and re-raise the redial-expected flag over a dead
        # camera -- so `end_of_stream` stayed False, `run_demo`'s worker looped
        # forever, and its teardown never ran. A silent split brain traded for a
        # silent hang.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)
            camera = link.camera
            # A reader that will not stop, which is the only way into the branch.
            camera.rebind = lambda *a, **k: False

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")

            assert wait_until(lambda: link.supervisor_ended is not None), \
                "the supervisor neither rebound nor gave up"
            assert link.supervisor_ended == "rebind_failed"
            assert link.rebinds == [], "a failed rebind was recorded as a redial"
            assert camera.end_of_stream is True, "the run would hang here forever"
            second.hang_up()
        finally:
            link.stop()

    def test_a_clean_stop_says_the_supervisor_stopped(self):
        # Its documented meaning is "None while still watching", which was false for
        # the whole teardown: the field was only ever written on the give-up path.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        dialling = dial(acceptor, "phone-one")
        assert link.wait_for_phone(timeout_s=5.0) is True
        phone = dialling.result(timeout=5.0)
        assert link.supervisor_ended is None
        link.stop()
        assert link.supervisor_ended == "stopped"
        phone.hang_up()


class TestFailureDoesNotLookLikeSuccess:
    """Three records that could not tell a drive that failed from one that worked."""

    def test_a_rate_command_the_queue_threw_away_is_visible(self):
        # `Session._enqueue` evicts the oldest on overflow and returns True, so a send
        # into a backed-up link is counted as a send. `sends_refused` says "closed OR
        # backed-up" and could only ever see closed. On `rate_cmd` -- `reliable`,
        # depth 16, chosen precisely because no later message repeats what an earlier
        # one said -- a drive that lost commands wrote the same record as one that
        # did not.
        # A tiny buffer so the writer blocks and the queue actually BACKS UP. Closing
        # the far end instead would exercise the closed-link path, which the counter
        # could already see -- and would prove nothing about the one it could not.
        phone_conn, jetson_conn = loopback_pair(max_buffer_bytes=256)
        phone = Session(phone_conn, session_id=1, heartbeat_s=None,
                        stall_timeout_s=None).start()
        jetson = Session(jetson_conn, session_id=2, heartbeat_s=None,
                         stall_timeout_s=None).start()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            for i in range(200):
                link.send_rate_command(_rate_command(t_capture_mono_ns=i + 1))

            channels = link.to_record()["wire"]["channels"]
            dropped = channels[Channel.RATE_CMD.value]["dropped_outbound"]
            assert dropped > 0, (
                "the queue threw commands away and the record does not say so"
            )
            # Counted, not derived. Deriving the loss by subtraction is how a counting
            # bug hides, which is this transport's own stated rule.
            assert "dropped_outbound" in channels[Channel.RATE_CMD.value]
        finally:
            link.stop()
            phone.close()

    def test_a_phone_whose_frames_are_all_refused_is_not_a_silent_phone(self):
        # One missing header key -- what a version skew produces -- and every frame is
        # refused by the decoder. `frames_received` counts only decodable arrivals, so
        # it reads zero either way, and `decode_failures` is the JPEG counter one
        # layer BELOW the decoder that actually refused them.
        phone, jetson, up, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            for i in range(5):
                # A camera frame with `quality` absent rather than null: legal to
                # send, refused on arrival.
                extensions, payload = CameraFrame(
                    t_capture_mono_ns=now_mono_ns() + i, frame_id=i, width=16,
                    height=16, format="jpeg", quality=85, jpeg=a_jpeg()).to_wire()
                extensions.pop("quality")
                assert phone.send(Channel.CAMERA, payload, extensions)
            assert wait_until(
                lambda: link.to_record()["wire"]["messages"]
                .get(Channel.CAMERA.value, {}).get("decode_errors", 0) > 0
            ), "the refused frames reached no counter anything reads"

            camera = link.to_record()["wire"]["messages"][Channel.CAMERA.value]
            assert camera["decode_errors"] > 0
            assert camera["errors_by_reason"]
        finally:
            link.stop()
            phone.close()

    def test_the_imu_channel_is_drained_and_counted(self):
        # The phone streams it at 50 Hz for the whole drive and nothing read it, so
        # the inbound deque sat at its 256 depth and every later sample was discarded.
        phone, jetson, up, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            for i in range(5):
                assert up.send(ImuSample(
                    t_capture_mono_ns=now_mono_ns() + i,
                    ax=0.1, ay=0.2, az=9.8, gx=0.0, gy=0.0, gz=0.01))
            assert wait_until(lambda: link.imu_received == 5), \
                "nothing on this side reads the IMU channel"

            record = link.to_record()["imu"]
            assert record["received"] == 5
            assert record["reader_alive"] is True
            # Said out loud: the count alone reads as though something consumes it.
            assert record["feeds_the_controller"] is False
        finally:
            link.stop()
            phone.close()


class TestTheReportAndItsAgeAreOneValue:
    """Two stores could be read between, and the consumer treats absence as fresh."""

    def test_a_reader_can_never_see_a_report_without_its_arrival_time(self):
        # The dangerous half is asymmetric: `SensingController._thermal_scale` gates
        # on `if age is not None`, so a torn read does not merely misstate the age --
        # it removes it, and the staleness check never runs. A 40 s old `critical`
        # would then cut a healthy handset by 6.7x.
        #
        # `here_feed`'s `_Snapshot` exists for exactly this: "no ordering of
        # independent stores is safe". Asserted structurally, because the window is
        # one bytecode wide and racing for it is what this repo retired a pin for.
        link = PhoneLink(acceptor=LoopbackAcceptor())
        assert link.telemetry is None and link.telemetry_at_mono is None

        link._telemetry = (object(), 1234.5)
        assert link.telemetry is not None
        assert link.telemetry_at_mono == 1234.5

        # Neither half can be assigned on its own.
        for attribute in ("telemetry", "telemetry_at_mono", "imu", "imu_at_mono"):
            with pytest.raises(AttributeError):
                setattr(link, attribute, object())

    def test_the_imu_pair_is_published_the_same_way(self):
        link = PhoneLink(acceptor=LoopbackAcceptor())
        link._imu = (object(), 99.0)
        assert link.imu is not None and link.imu_at_mono == 99.0


class TestTeardownOrdering:

    def test_the_redial_flag_is_cleared_after_the_supervisor_is_joined(self):
        # Asserted on the source, because the behavioural version cannot be trusted:
        # whether a redial lands inside the window is decided by two unsynchronised
        # threads, so a test that races for it reports SURVIVED on one run and CAUGHT
        # on the next -- and this repo has already retired one pin for that.
        #
        # The ordering is load-bearing and the comment says why: `_rebind` can already
        # be past its own `_stop` check, and it finishes by restarting the readers and
        # setting `expect_redial(True)`, so a clear taken FIRST is undone by a redial
        # landing during teardown and the stopped link reports an open stream.
        import inspect

        source = inspect.getsource(PhoneLink.stop)
        join = source.index("self._supervisor.join(")
        clear = source.index("expect_redial(False)")
        assert join < clear, (
            "expect_redial(False) is taken before the supervisor join again, so a "
            "redial in flight undoes it and a stopped link reports an open stream"
        )

    def test_a_reader_that_will_not_stop_ends_the_run_rather_than_doubling_up(self):
        # Every reader re-reads `self.router` on each iteration, so one that outlived
        # its join binds itself to the NEXT session rather than exiting: two readers
        # on one channel, compounding per redial, and on CONTROL that is two senders
        # on one channel -- the precondition for a phantom sequence gap.
        acceptor = LoopbackAcceptor()
        link = PhoneLink(acceptor=acceptor)
        try:
            dialling = dial(acceptor, "phone-one")
            assert link.wait_for_phone(timeout_s=5.0) is True
            first = dialling.result(timeout=5.0)

            # A reader that cannot be joined, which is the only way into the branch.
            stuck = threading.Event()
            link._here_reader = threading.Thread(
                target=lambda: stuck.wait(30.0), name="stuck-here", daemon=True)
            link._here_reader.start()

            first.hang_up()
            second = LoopbackPhone(acceptor, "phone-two")
            assert wait_until(lambda: link.supervisor_ended is not None, timeout=10.0), \
                "the supervisor neither rebound nor refused"
            assert link.supervisor_ended == "readers_would_not_stop"
            assert link.rebinds == [], "it rebound over a reader that never stopped"
            stuck.set()
            second.hang_up()
        finally:
            link.stop()


class TestTheRecordBalancesAndIsBounded:

    def test_the_inbound_account_balances(self, ):
        # The transport pins `received == delivered + dropped_inbound +
        # abandoned_inbound` and this row published the first three, so a reader
        # could not tell a message policy dropped from one nobody collected -- the
        # distinction the counter was added for. Not derivable from the rest either,
        # because `queued` was absent too.
        phone, jetson, up, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            for i in range(6):
                assert up.send(GpsRecord(
                    t_capture_mono_ns=now_mono_ns() + i, valid=True, fix_quality=1,
                    num_sats=9, lat=40.0, lon=-74.0, speed_mps=27.0,
                    heading_deg=90.0, hdop=0.9, altitude_m=10.0))
            assert wait_until(lambda: link.gps.messages_received == 6)
            phone.close()
            wait_until(lambda: getattr(jetson, "is_closed", False))

            row = link.to_record()["wire"]["channels"][Channel.GPS.value]
            for term in ("queued", "dropped_outbound", "abandoned_outbound",
                         "received", "delivered", "dropped_inbound",
                         "abandoned_inbound"):
                assert term in row, f"{term} is missing, so the account cannot balance"
            assert row["received"] == (row["delivered"] + row["dropped_inbound"]
                                       + row["abandoned_inbound"]), row
        finally:
            link.stop()
            phone.close()

    def test_the_leaked_handshake_workers_reach_the_summary(self):
        # `endpoint.py` says the count exists so a misbehaving backend "shows up in
        # the summary instead of quietly accumulating threads across a drive". The
        # only summary a drive produces is this one.
        link = PhoneLink(acceptor=LoopbackAcceptor())
        try:
            assert "handshake_workers_leaked" in link.to_record()
        finally:
            link.stop()

    def test_the_refusal_list_is_bounded_and_says_what_it_dropped(self):
        # Every other refusal account here is a bounded histogram over a closed
        # vocabulary. This one kept every string, and nothing drains the listener's
        # events during a live session -- so one failed dial a second put 10,800
        # entries and half a megabyte into a three-hour run's summary.
        link = PhoneLink(acceptor=LoopbackAcceptor())
        try:
            refusal = type("E", (), {"peer": "1.2.3.4:5", "error": "version mismatch"})()
            for _ in range(link.MAX_REFUSALS + 25):
                link._note_refusal(refusal)

            record = link.to_record()
            assert len(record["refusals"]) == link.MAX_REFUSALS
            assert record["refusals_not_kept"] == 25, (
                "a run that turned away 75 dial-ins reads like one that turned away 50"
            )
        finally:
            link.stop()


class TestTheRecordSaysWhichRouteTheBytesTook:

    def test_the_sessions_own_peer_address_is_in_the_record(self):
        # The only fact in the record that distinguishes a run whose bytes crossed the
        # tailnet from one whose bytes crossed USB via `adb reverse`: the phone dials
        # 127.0.0.1 on the second and a 100.x address on the first. `SessionStats.peer`
        # carried it all along and `_wire_record` dropped it, iterating only the
        # per-channel counters.
        phone, jetson, _, _ = phone_and_jetson()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            record = link.to_record()["wire"]
            assert "peer" in record, (
                "the record cannot say where the phone dialled from"
            )
            assert record["peer"] == jetson.stats().peer
        finally:
            link.stop()
            phone.close()


class _SessionThatChangesUnderYou(PhoneLink):
    """A link whose `.session` yields a different session on each read.

    That is what the redial supervisor does: it replaces `self.session` while the
    rest of the object carries on. A record assembled from several independent
    reads can therefore describe two handsets at once, and no field in it says so.
    Rotating on every read makes the window certain instead of nanoseconds wide.
    """

    def install(self, first, second) -> None:
        self._rotation = [first, second]
        self._reads = 0

    @property
    def session(self):
        index = min(self._reads, len(self._rotation) - 1)
        self._reads += 1
        return self._rotation[index]

    @session.setter
    def session(self, value) -> None:
        self._rotation = [value]
        self._reads = 0


class TestOneRecordDescribesOneSession:

    def test_a_redial_mid_record_cannot_split_it_across_two_sessions(self):
        phone, jetson, _, _ = phone_and_jetson()
        # `phone_and_jetson` numbers every Jetson-side session 2, so the second one is
        # built here. A redial does raise the id, and if the two sessions shared one
        # the test would pass while reading two different objects.
        # Labelled, so the two sessions report different peer addresses. Both pairs
        # default to `loopback:phone`, and an assertion between two equal values
        # cannot detect the swap it is written to detect.
        other_phone_conn, other_jetson_conn = loopback_pair(
            labels=("second-phone", "jetson"))
        other_phone = Session(other_phone_conn, session_id=10, heartbeat_s=None,
                              stall_timeout_s=None).start()
        other_jetson = Session(other_jetson_conn, session_id=11, heartbeat_s=None,
                               stall_timeout_s=None).start()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        original = link.__class__
        try:
            link.__class__ = _SessionThatChangesUnderYou
            link.install(jetson, other_jetson)
            record = link.to_record()
            # Distinct by construction, so the two reads cannot agree by accident.
            assert jetson.session_id != other_jetson.session_id
            assert record["session_id"] == record["wire"]["session_id"], (
                "one record names two sessions: the top-level id is "
                f"{record['session_id']} and the wire block's is "
                f"{record['wire']['session_id']}"
            )
            # The peer address is the field that says which handset the channel
            # counters belong to, so it must come from the same session as the id.
            assert jetson.stats().peer != other_jetson.stats().peer
            assert record["wire"]["peer"] == (
                jetson.stats().peer
                if record["session_id"] == jetson.session_id
                else other_jetson.stats().peer
            )
        finally:
            link.__class__ = original
            link.session = jetson
            link.stop()
            phone.close()
            other_phone.close()


class TestTheSessionSnapshotDescribesOneSession:

    def test_the_snapshot_taken_at_a_rebind_names_one_session(self):
        # `_session_record` is the row `_rebind` writes before replacing a session,
        # and it read `self.session` three times. Only the supervisor thread writes
        # that field once a run is up, so the disagreement is not reachable today --
        # which is an argument about which threads run when, not a property of the
        # code. This asserts the property.
        phone, jetson, _, _ = phone_and_jetson()
        other_phone_conn, other_jetson_conn = loopback_pair(
            labels=("second-phone", "jetson"))
        other_phone = Session(other_phone_conn, session_id=20, heartbeat_s=None,
                              stall_timeout_s=None).start()
        other_jetson = Session(other_jetson_conn, session_id=21, heartbeat_s=None,
                               stall_timeout_s=None).start()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        original = link.__class__
        try:
            link.__class__ = _SessionThatChangesUnderYou
            link.install(jetson, other_jetson)
            record = link._session_record()
            assert jetson.session_id != other_jetson.session_id
            assert record["session_id"] == record["wire"]["session_id"], record
        finally:
            link.__class__ = original
            link.session = jetson
            link.stop()
            phone.close()
            other_phone.close()


class TestTheInboundAccountBalances:

    def test_a_consumed_heartbeat_is_published_not_just_subtracted(self):
        # `_record_inbound` counts a keepalive in `received` and returns before the
        # queue: never delivered, never dropped, never abandoned. The record published
        # the other four terms and not this one, so the first real run over the
        # tailnet reported 241 received against 121 delivered on the control channel
        # having lost nothing -- indistinguishable from a run that lost 120 messages.
        phone_conn, jetson_conn = loopback_pair()
        phone = Session(phone_conn, session_id=1, heartbeat_s=0.05,
                        stall_timeout_s=None).start()
        jetson = Session(jetson_conn, session_id=2, heartbeat_s=None,
                         stall_timeout_s=None).start()
        link = PhoneLink(acceptor=LoopbackAcceptor())
        attach(link, jetson, None)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if jetson.stats().heartbeats_received >= 3:
                    break
                time.sleep(0.02)
            wire = link.to_record()["wire"]
            control = wire["channels"][Channel.CONTROL.value]
            assert wire["heartbeats_received"] >= 3, (
                "the heartbeats never arrived, so this test proves nothing"
            )
            # Every inbound frame on the channel is in exactly one of these terms.
            assert control["received"] == (
                control["delivered"] + control["dropped_inbound"]
                + control["abandoned_inbound"] + wire["heartbeats_received"]
            ), control
            # And the gap is not zero, or the identity would hold with the term absent
            # and the test would pass against the code it was written to catch.
            assert control["received"] > control["delivered"]
        finally:
            link.stop()
            phone.close()
