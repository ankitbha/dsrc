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
