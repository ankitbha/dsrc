"""The assembly between a socket and the two backends.

Driven over a real loopback session pair rather than a mock router, because the
thing most worth pinning here is a direction the wire enforces: the Jetson answers
time-sync pings and never sends one. A mock would answer whatever it was asked.
"""

from __future__ import annotations

import time

from sensors.phone_link import PhoneLink
from transport.channels import Channel
from transport.loopback import loopback_pair
from transport.messages import MessageRouter
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

    def test_a_closed_session_ends_the_camera_stream(self):
        phone, jetson, up, down = phone_and_jetson()
        link = PhoneLink()
        try:
            attach(link, jetson, down)
            assert link.camera.end_of_stream is False

            phone.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not link.camera.end_of_stream:
                time.sleep(0.02)

            assert link.camera.end_of_stream, "a dead session left the consumer waiting forever"
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
