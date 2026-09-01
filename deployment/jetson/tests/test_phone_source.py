"""Phone-fed sensor backends: the clock boundary, and what it must not hide.

Machine-independent by design. Every latency budget lives in
`scripts/run_loopback_pipeline.py` instead, because an assertion on wall-clock
duration is the one kind that fails for whatever else the machine is doing rather
than for a defect.

The offset planted throughout is the real one: 67.57 hours, measured between this
Mac and the Jetson. It is not a stress value -- it is what two devices counting
from their own boots actually differ by, and it is what makes the failure this
module prevents a hard failure rather than a small error.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from perception.observation_builder import BuilderConfig, ObservationBuilder
from sensors.gps_reader import GpsFix
from sensors.phone_source import (
    PhoneCameraStream,
    PhoneClockAdapter,
    PhoneGpsReader,
    _or_nan,
)
from sensors.time_sync import TimebaseStamp
from transport.channels import Channel
from transport.clock import now_mono_ns
from transport.frames import WIRE_STAMP_KEY
from transport.frames import Frame as WireFrame
from transport.loopback import loopback_pair
from transport.messages import CameraFrame, GpsRecord, MessageRouter
from transport.session import ReceivedMessage, Session
from transport.timebase import (
    MIN_OFFSET_SAMPLES,
    OneWayEstimator,
    OneWaySample,
    TimebaseEstimator,
    TimebaseNotReady,
    TimeSyncSample,
)

# Measured between this Mac and the Jetson: 7.42 and 10.23 days of uptime.
PLANTED_OFFSET_NS = 243_264_000_000_000
BASE_NS = 4_000_000_000_000


class SteppedClock:
    def __init__(self, now_ns: int = BASE_NS) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def converged_estimator(offset_ns: int = PLANTED_OFFSET_NS, *, samples: int = 20):
    """An estimator that has seen a peer whose clock is `offset_ns` ahead.

    Built from samples rather than by setting a field, so the arithmetic under
    test is the arithmetic that runs in production.
    """
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    for index in range(samples):
        t1 = BASE_NS + index * 100_000_000
        up = 5_000_000
        t2 = t1 + up + offset_ns
        t3 = t2
        t4 = t1 + 2 * up
        clock.now_ns = t4
        estimator.add(TimeSyncSample(
            exchange_id=index, t1_local_send_ns=t1, t2_remote_recv_ns=t2,
            t3_remote_send_ns=t3, t4_local_recv_ns=t4,
        ))
    return estimator, clock


# -- the conversion itself ---------------------------------------------------


def test_a_planted_capture_stamp_converts_to_an_exact_local_value():
    """The off-by-1e9 test and the clock test at once: this boundary changes both
    the clock and the unit, and a 1e9 error looks like a plausible timestamp.

    Asserted as an exact value, not as "about right"."""
    estimator, _ = converged_estimator()
    adapter = PhoneClockAdapter(estimator)

    peer_capture_ns = BASE_NS + PLANTED_OFFSET_NS + 500_000_000
    arrival_s = (BASE_NS + 505_000_000) / 1e9
    stamp = adapter.stamp(peer_capture_ns, arrival_s)

    assert not stamp.proxy
    # The peer instant corresponds to BASE + 0.5 s on this clock, in SECONDS.
    assert stamp.t_capture_mono == pytest.approx((BASE_NS + 500_000_000) / 1e9, abs=1e-6)
    assert stamp.t_arrival_mono == arrival_s
    assert stamp.bound_s is not None and 0 < stamp.bound_s < 1.0
    assert adapter.converted == 1 and adapter.proxied == 0


def test_the_converted_value_is_seconds_not_nanoseconds():
    """Stated separately because it is the failure that hides: a value 1e9 too
    large is still a float, still monotonic, and still orders correctly."""
    estimator, _ = converged_estimator()
    stamp = PhoneClockAdapter(estimator).stamp(
        BASE_NS + PLANTED_OFFSET_NS, BASE_NS / 1e9
    )
    assert 1e3 < stamp.t_capture_mono < 1e7, (
        f"{stamp.t_capture_mono} is not a plausible monotonic value in seconds"
    )


def test_conversion_preserves_the_spacing_between_captures():
    """Frame-to-frame intervals are what the tracker and the distance estimator
    consume, and they must survive the boundary exactly -- conversion is affine,
    so it can. If it did not, every velocity in the pipeline would be wrong."""
    estimator, _ = converged_estimator()
    adapter = PhoneClockAdapter(estimator)
    spacing_ns = 100_000_000  # 10 Hz

    stamps = [
        adapter.stamp(BASE_NS + PLANTED_OFFSET_NS + i * spacing_ns, BASE_NS / 1e9)
        for i in range(5)
    ]
    gaps = [b.t_capture_mono - a.t_capture_mono for a, b in zip(stamps, stamps[1:])]
    for gap in gaps:
        assert gap == pytest.approx(spacing_ns / 1e9, abs=1e-6), gaps


def test_the_adapter_converts_incoming_stamps_not_outgoing_ones():
    """`to_remote` on an arriving peer stamp treats it as local and displaces it
    by twice the offset. Measured, that pushed every capture 135 hours into the
    past, the extrapolation guard refused all of them, and the run still produced
    a full set of ticks and advisories -- visible only in the provenance."""
    estimator, _ = converged_estimator()
    peer_capture_ns = BASE_NS + PLANTED_OFFSET_NS + 500_000_000

    local = estimator.to_local(peer_capture_ns).t_remote_mono_ns
    assert abs(local - (BASE_NS + 500_000_000)) < 10_000_000

    # The wrong direction is not merely inaccurate, it is refused -- which is why
    # the mistake showed up as a silent proxy rather than as a bad number.
    with pytest.raises(TimebaseNotReady):
        estimator.to_remote(peer_capture_ns)


# -- the proxy ---------------------------------------------------------------


def test_before_convergence_the_arrival_stamp_is_used_and_marked():
    estimator = TimebaseEstimator(mono_clock=SteppedClock())
    adapter = PhoneClockAdapter(estimator)
    arrival_s = 123.456

    stamp = adapter.stamp(BASE_NS + PLANTED_OFFSET_NS, arrival_s)
    assert stamp.proxy
    assert stamp.t_capture_mono == arrival_s
    assert stamp.t_arrival_mono == arrival_s
    assert stamp.bound_s is None, "a proxied stamp must not report a bound it does not have"
    assert stamp.estimate_id is None
    assert adapter.proxied == 1 and adapter.converted == 0
    assert adapter.to_record()["proxy_reasons"] == {"no samples": 1}


def test_the_proxy_lifts_once_the_estimator_converges():
    """A proxy that never lifts is the failure this pins: the loop would keep
    running and every freshness decision would rest on arrival time forever."""
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    adapter = PhoneClockAdapter(estimator)

    assert adapter.stamp(BASE_NS + PLANTED_OFFSET_NS, 1.0).proxy

    for index in range(MIN_OFFSET_SAMPLES + 3):
        t1 = BASE_NS + index * 100_000_000
        t2 = t1 + 5_000_000 + PLANTED_OFFSET_NS
        t4 = t1 + 10_000_000
        clock.now_ns = t4
        estimator.add(TimeSyncSample(
            exchange_id=index, t1_local_send_ns=t1, t2_remote_recv_ns=t2,
            t3_remote_send_ns=t2, t4_local_recv_ns=t4,
        ))

    after = adapter.stamp(BASE_NS + PLANTED_OFFSET_NS + 500_000_000, 1.0)
    assert not after.proxy, "the proxy did not lift after the estimator converged"
    assert after.bound_s is not None


def test_the_proxy_records_why_rather_than_only_that():
    """"Proxied 4,000 times" cannot say whether the link went bad or the run had
    just started, which is the same reason the transport counts drops per reason."""
    clock = SteppedClock()
    estimator = TimebaseEstimator(mono_clock=clock)
    adapter = PhoneClockAdapter(estimator)

    adapter.stamp(BASE_NS, 1.0)  # no samples
    estimator.add(TimeSyncSample(
        exchange_id=1, t1_local_send_ns=BASE_NS,
        t2_remote_recv_ns=BASE_NS + PLANTED_OFFSET_NS,
        t3_remote_send_ns=BASE_NS + PLANTED_OFFSET_NS,
        t4_local_recv_ns=BASE_NS + 1_000_000,
    ))
    clock.now_ns = BASE_NS + 1_000_000
    adapter.stamp(BASE_NS, 1.0)  # too few samples

    reasons = adapter.to_record()["proxy_reasons"]
    assert len(reasons) == 2, reasons
    assert "no samples" in reasons
    assert any("samples in the offset window" in r for r in reasons)


# -- the reason the task exists ----------------------------------------------


# The uptimes actually measured on this pair, so the magnitudes are the real
# ones rather than convenient small numbers.
JETSON_UPTIME_S = 884_285.2
PHONE_UPTIME_S = 641_020.6


@pytest.mark.parametrize(
    "phone_uptime_s,expected_age_sign",
    [(PHONE_UPTIME_S, "positive"), (JETSON_UPTIME_S + 200_000.0, "negative")],
    ids=["jetson-booted-later", "phone-booted-later"],
)
def test_an_unconverted_phone_fix_is_never_fresh_either_way_round(
    phone_uptime_s, expected_age_sign
):
    """The defect this module prevents, asserted rather than described -- and in
    both directions, because which one occurs depends on which device booted
    first and neither is the safe one.

    The two directions fail differently and only one of them is safe.

    With the Jetson up longer the age is hugely positive (+243,265 s measured),
    it fails the 2.0 s threshold directly, and ego speed falls back to neutral --
    degraded, but safe.

    With the phone up longer the stamp is in this clock's future and the age is
    negative. A one-sided `age <= 2.0` is *satisfied* by that, so an arbitrarily
    old fix read as current and the policy acted on it. That is the dangerous
    direction, it is the one the plan for this task got wrong, and it is why the
    gate is now bounded below as well as above.
    """
    builder = ObservationBuilder(BuilderConfig())
    unconverted = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                         t_mono=phone_uptime_s, t_wall=0.0)
    age = unconverted.age_s(JETSON_UPTIME_S)
    if expected_age_sign == "positive":
        assert age > 200_000, age
    else:
        assert age < -100_000, f"{age} is not in this clock's future"

    result = builder.build([], unconverted, JETSON_UPTIME_S, None)
    assert result.diagnostics["gps_fresh"] is False
    assert result.field_sources["ego_speed"] == "fallback_neutral"


def test_the_same_fix_converted_is_fresh():
    """The other half: with the stamp converted, the age is the link latency and
    the gate passes. Without this the test above would be satisfied by a
    conversion that broke everything equally."""
    offset_ns = int((JETSON_UPTIME_S - PHONE_UPTIME_S) * 1e9)
    clock = SteppedClock(int(JETSON_UPTIME_S * 1e9))
    estimator = TimebaseEstimator(mono_clock=clock)
    base = int(JETSON_UPTIME_S * 1e9) - 2_000_000_000
    for index in range(20):
        t1 = base + index * 100_000_000
        t2 = t1 + 5_000_000 - offset_ns          # the peer's clock is BEHIND ours
        t4 = t1 + 10_000_000
        clock.now_ns = t4
        estimator.add(TimeSyncSample(
            exchange_id=index, t1_local_send_ns=t1, t2_remote_recv_ns=t2,
            t3_remote_send_ns=t2, t4_local_recv_ns=t4,
        ))
    assert estimator.usable, estimator.why_not_usable()

    jetson_now = clock.now_ns / 1e9
    peer_capture_ns = clock.now_ns - offset_ns - 20_000_000  # captured 20 ms ago
    stamp = PhoneClockAdapter(estimator).stamp(peer_capture_ns, jetson_now)
    assert not stamp.proxy, "the estimator was converged; this should have converted"

    converted = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                       t_mono=stamp.t_capture_mono, t_wall=0.0, timebase=stamp)
    age = converted.age_s(jetson_now)
    assert 0.0 <= age < 1.0, f"converted age {age} s is not a link latency"
    result = ObservationBuilder(BuilderConfig()).build([], converted, jetson_now, None)
    assert result.diagnostics["gps_fresh"] is True
    assert result.field_sources["ego_speed"] == "measured_converted"


def test_provenance_distinguishes_converted_from_proxied_from_absent():
    """Three outcomes, not two. "measured" would hide the difference between a
    freshness established exactly and one resting on an arrival-time proxy."""
    builder_cfg = BuilderConfig()
    now = BASE_NS / 1e9

    local = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9, t_mono=now)
    assert ObservationBuilder(builder_cfg).build([], local, now, None).field_sources[
        "ego_speed"
    ] == "measured"

    proxied = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9, t_mono=now,
                     timebase=TimebaseStamp(now, now, None, None, True))
    assert ObservationBuilder(builder_cfg).build([], proxied, now, None).field_sources[
        "ego_speed"
    ] == "measured_arrival_proxy"

    stale = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                   t_mono=now - 60.0)
    assert ObservationBuilder(builder_cfg).build([], stale, now, None).field_sources[
        "ego_speed"
    ] == "fallback_neutral"


def test_a_missing_stamp_fails_towards_stale():
    """`age_s` returns inf for an unset stamp, so the failure direction of a
    missing timestamp is "not fresh" rather than "fresh". Worth pinning: the
    opposite default would make a dropped stamp look like a live reading."""
    assert GpsFix(valid=True).age_s(100.0) == float("inf")
    result = ObservationBuilder(BuilderConfig()).build([], GpsFix(valid=True), 100.0, None)
    assert result.diagnostics["gps_fresh"] is False


# -- over a real session -----------------------------------------------------


def a_jpeg() -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def phone_and_jetson(offset_ns: int = PLANTED_OFFSET_NS):
    """Two sessions on one loopback pair, the phone's on a displaced clock."""
    phone_conn, jetson_conn = loopback_pair()
    phone = Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None,
                    mono_clock=lambda: now_mono_ns() - offset_ns).start()
    jetson = Session(jetson_conn, session_id=2, heartbeat_s=None,
                     stall_timeout_s=None).start()
    return phone, jetson, MessageRouter(phone), MessageRouter(jetson)


def test_a_frame_crosses_the_transport_and_arrives_on_the_local_clock():
    phone, jetson, up, down = phone_and_jetson()
    estimator, _ = converged_estimator()
    adapter = PhoneClockAdapter(estimator)
    camera = PhoneCameraStream(down, adapter).start()
    try:
        capture_ns = BASE_NS + PLANTED_OFFSET_NS + 500_000_000
        assert up.send(CameraFrame(
            t_capture_mono_ns=capture_ns, frame_id=7, width=16, height=16,
            format="jpeg", quality=85, jpeg=a_jpeg(),
        ))
        frame = camera.wait_for_fresh(timeout=5.0)
        assert frame is not None
        assert frame.frame_id == 7
        assert frame.image.shape[0] == 16
        assert frame.timebase is not None and not frame.timebase.proxy
        assert frame.t_mono == pytest.approx((BASE_NS + 500_000_000) / 1e9, abs=1e-6)
        # Arrival is this device's own, so it is nowhere near the peer's clock.
        assert frame.timebase.t_arrival_mono > 0
    finally:
        camera.stop()
        phone.close()
        jetson.close()


def test_a_fix_crosses_and_its_nulls_become_nans():
    """The transport refuses to put a non-finite number on the wire, so `null` is
    the only way it can say "no value" -- and NaN is how the pipeline says it."""
    phone, jetson, up, down = phone_and_jetson()
    estimator, _ = converged_estimator()
    gps = PhoneGpsReader(down, PhoneClockAdapter(estimator)).start()
    try:
        assert up.send(GpsRecord(
            t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS, valid=False,
            fix_quality=0, num_sats=0,
        ))
        deadline_reached = False
        for _ in range(500):
            fix = gps.latest()
            if fix.t_mono != 0.0:
                deadline_reached = True
                break
            import time as clock
            clock.sleep(0.01)
        assert deadline_reached, "the fix never arrived"
        fix = gps.latest()
        assert fix.valid is False
        assert np.isnan(fix.lat) and np.isnan(fix.lon) and np.isnan(fix.speed_mps)
        assert fix.timebase is not None
    finally:
        gps.stop()
        phone.close()
        jetson.close()


def test_an_undecodable_frame_costs_one_frame_not_the_stream():
    """The same recoverability split the transport draws between a malformed
    message and a malformed byte stream."""
    phone, jetson, up, down = phone_and_jetson()
    estimator, _ = converged_estimator()
    adapter = PhoneClockAdapter(estimator)
    camera = PhoneCameraStream(down, adapter).start()
    try:
        import time as clock

        assert up.send(CameraFrame(
            t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS, frame_id=1, width=16,
            height=16, format="jpeg", quality=85, jpeg=b"not a jpeg at all",
        ))
        # Wait for it to be consumed before sending the next. `camera` is
        # LATEST_WINS at depth 1, so a second send would replace the first in the
        # outbound queue and the bad frame would never leave -- the test would
        # then pass with no decode attempted at all.
        deadline = clock.monotonic() + 5.0
        while clock.monotonic() < deadline and camera.decode_failures == 0:
            clock.sleep(0.01)
        assert camera.decode_failures == 1, "the undecodable frame never reached the decoder"

        assert up.send(CameraFrame(
            t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS + 1, frame_id=2, width=16,
            height=16, format="jpeg", quality=85, jpeg=a_jpeg(),
        ))
        frame = camera.wait_for_fresh(timeout=5.0)
        assert frame is not None and frame.frame_id == 2, "the good frame never arrived"
        assert camera.decode_failures == 1
        assert camera.to_record()["decode_failures"] == 1
    finally:
        camera.stop()
        phone.close()
        jetson.close()


def test_or_nan_maps_the_transports_null_to_the_pipelines_nan():
    assert np.isnan(_or_nan(None))
    assert _or_nan(0.0) == 0.0
    assert _or_nan(-3.5) == -3.5


def test_a_stamp_from_the_future_is_not_fresh():
    """The dangerous half of the unconverted case, pinned on its own.

    A one-sided freshness check treats a negative age as fresh, so an
    unconverted stamp from a peer that booted later than this device made an
    arbitrarily old reading read as current."""
    builder = ObservationBuilder(BuilderConfig())
    now = 884_285.2
    from_the_future = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                             t_mono=now + 200_000.0)
    assert from_the_future.age_s(now) < 0
    result = builder.build([], from_the_future, now, None)
    assert result.diagnostics["gps_fresh"] is False, (
        "a reading from this clock's future was accepted as fresh"
    )
    assert result.field_sources["ego_speed"] == "fallback_neutral"


def test_a_small_negative_age_is_ordinary_and_a_large_one_is_not():
    """The allowance is not zero, and it is the window already configured.

    A conversion may legitimately land slightly after the arrival it preceded --
    that is what its bound means -- and `now` is often sampled just before a
    reading is taken, so a hair-negative age is normal for a local sensor too.
    Rejecting those discards good data for being accurate. A large negative age
    is nonsense in either direction."""
    builder = ObservationBuilder(BuilderConfig())
    now = 884_285.2
    bound_s = 0.008
    stamp = TimebaseStamp(t_capture_mono=now + 0.004, t_arrival_mono=now,
                          bound_s=bound_s, estimate_id=1, proxy=False)
    slightly_ahead = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                            t_mono=now + 0.004, t_wall=0.0, timebase=stamp)
    result = builder.build([], slightly_ahead, now, None)
    assert result.diagnostics["gps_fresh"] is True, (
        "a conversion inside its own bound was rejected"
    )

    # And the allowance is the BOUND, not the staleness window. This case sits
    # between the two -- 0.5 s ahead, with an 8 ms bound, inside a 2 s window --
    # so it is the only kind that can tell them apart. Probing only inside the
    # bound and outside the window leaves both formulas passing.
    between = TimebaseStamp(t_capture_mono=now + 0.5, t_arrival_mono=now,
                            bound_s=bound_s, estimate_id=1, proxy=False)
    ahead = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                   t_mono=now + 0.5, t_wall=0.0, timebase=between)
    assert ObservationBuilder(BuilderConfig()).build(
        [], ahead, now, None
    ).diagnostics["gps_fresh"] is False, (
        "0.5 s into the future was accepted on the strength of an 8 ms bound"
    )

    far = TimebaseStamp(t_capture_mono=now + 5.0, t_arrival_mono=now,
                        bound_s=bound_s, estimate_id=1, proxy=False)
    too_far = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                     t_mono=now + 5.0, t_wall=0.0, timebase=far)
    assert ObservationBuilder(BuilderConfig()).build(
        [], too_far, now, None
    ).diagnostics["gps_fresh"] is False


def test_the_arrival_stamp_comes_from_the_transport_not_from_a_local_read():
    """Separated by giving the receiving session its own displaced clock, so the
    transport's arrival stamp and a local `now_mono()` cannot be confused.

    A stamp read after `recv` returns folds the inbound queue wait and the decode
    into the link segment -- the receive-side twin of the error the wire stamp
    removes on the way out -- and on a quiet loopback the two are microseconds
    apart, so nothing but a displaced clock can tell them apart.
    """
    jetson_shift_ns = 500_000_000_000  # 500 s, far outside any plausible jitter
    phone_conn, jetson_conn = loopback_pair()
    phone = Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None,
                    mono_clock=lambda: now_mono_ns() - PLANTED_OFFSET_NS).start()
    jetson = Session(jetson_conn, session_id=2, heartbeat_s=None, stall_timeout_s=None,
                     mono_clock=lambda: now_mono_ns() + jetson_shift_ns).start()
    up, down = MessageRouter(phone), MessageRouter(jetson)
    estimator, _ = converged_estimator()
    camera = PhoneCameraStream(down, PhoneClockAdapter(estimator)).start()
    try:
        assert up.send(CameraFrame(
            t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS, frame_id=1, width=16,
            height=16, format="jpeg", quality=85, jpeg=a_jpeg(),
        ))
        frame = camera.wait_for_fresh(timeout=5.0)
        assert frame is not None and frame.timebase is not None
        expected = (now_mono_ns() + jetson_shift_ns) / 1e9
        assert abs(frame.timebase.t_arrival_mono - expected) < 5.0, (
            f"arrival {frame.timebase.t_arrival_mono} is not on the session's clock "
            f"(expected near {expected}); it was read locally instead"
        )
    finally:
        camera.stop()
        phone.close()
        jetson.close()


def test_a_local_frame_reports_no_link_segment():
    """None, not zero. A local camera has no link, and a zero would read as a
    measured latency of zero rather than as an absent measurement."""
    import tempfile

    from pipeline import PerceptionPolicyPipeline
    from sensors.camera_stream import Frame

    pipeline = _a_pipeline()
    frame = Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), frame_id=1,
                  t_mono=_local_now() - 0.01, t_wall=0.0)
    tick = pipeline.step(frame, GpsFix(), None, detections_override=[])
    assert tick.link_ms is None
    assert tick.timebase is None
    assert tick.to_record()["link_ms"] is None
    # And with no link, the two segments coincide.
    assert tick.jetson_ms == pytest.approx(tick.e2e_ms, abs=1e-6)
    # The aggregate must report absence too. An empty series summarising as
    # zeros meant every local run published stats.link_ms = 0/0/0 for a segment
    # that does not exist -- a measured-looking zero.
    snapshot = pipeline.stats.snapshot()
    assert snapshot["link_ms"] is None, snapshot["link_ms"]
    assert snapshot["jetson_ms"] is not None and snapshot["jetson_ms"]["n"] == 1


def test_jetson_ms_is_measured_from_arrival_not_from_capture():
    """The two differ by the link segment, so measuring from capture puts the
    network inside a number that is supposed to be about this hardware."""
    from pipeline import PerceptionPolicyPipeline
    from sensors.camera_stream import Frame

    pipeline = _a_pipeline()
    now = _local_now()
    link_s = 0.250  # far larger than any real link, so the two cannot be confused
    stamp = TimebaseStamp(t_capture_mono=now - link_s, t_arrival_mono=now,
                          bound_s=0.008, estimate_id=1, proxy=False)
    frame = Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), frame_id=1,
                  t_mono=now - link_s, t_wall=0.0, timebase=stamp)
    tick = pipeline.step(frame, GpsFix(), None, detections_override=[])

    assert tick.link_ms == pytest.approx(link_s * 1000.0, abs=1.0)
    assert tick.jetson_ms < link_s * 1000.0, (
        f"jetson_ms {tick.jetson_ms} includes the {link_s * 1000} ms link segment"
    )
    assert tick.e2e_ms == pytest.approx(tick.jetson_ms + tick.link_ms, abs=1.0)


def _local_now() -> float:
    import time

    return time.monotonic()


def _a_pipeline():
    """The smoke test's stub pipeline, built once per call."""
    import tempfile

    from perception.detector import Detection  # noqa: F401
    from perception.distance import DistanceEstimator
    from perception.tracker import IouTracker
    from pipeline import PerceptionPolicyPipeline
    from policy.actor_runtime import ActorRuntime
    from policy.advisory import AdvisoryDecoder
    from policy.export_policy import build_random, export

    class _NoDetector:
        def infer(self, image):
            return []

        def warmup(self, iterations: int = 1) -> float:
            return 0.0

    tmp = tempfile.mkdtemp()
    prefix = str(pathlib.Path(tmp) / "actor_policy")
    actor, info = build_random(seed=0)
    export(actor, info, prefix)
    return PerceptionPolicyPipeline(
        detector=_NoDetector(),
        tracker=IouTracker(min_hits=2),
        distance=DistanceEstimator(fx_px=800.0, cx_px=640.0, horizon_y_px=360.0,
                                   camera_height_m=1.25, ema_alpha=0.6),
        builder=ObservationBuilder(BuilderConfig()),
        actor=ActorRuntime(prefix),
        advisory_decoder=AdvisoryDecoder(units="mph"),
    )


def test_a_fix_arrives_on_the_local_clock_with_an_exact_value():
    """The GPS twin of the camera test, and it was missing.

    Mutating `t_mono=stamp.t_capture_mono` to `stamp.t_arrival_mono` left the
    whole suite green: every fix would be arrival-stamped forever while
    `field_sources["ego_speed"]` still claimed `measured_converted`. A false
    provenance claim on the one field this task is about.

    The only session-level GPS test asserted `fix.t_mono != 0.0`, which is
    satisfied by either value.
    """
    phone, jetson, up, down = phone_and_jetson()
    estimator, _ = converged_estimator()
    gps = PhoneGpsReader(down, PhoneClockAdapter(estimator)).start()
    try:
        capture_ns = BASE_NS + PLANTED_OFFSET_NS + 500_000_000
        assert up.send(GpsRecord(
            t_capture_mono_ns=capture_ns, valid=True, fix_quality=1, num_sats=9,
            lat=40.744, lon=-74.032, speed_mps=27.0, heading_deg=90.0, hdop=0.9,
            altitude_m=10.0,
        ))
        import time as clock
        deadline = clock.monotonic() + 5.0
        while clock.monotonic() < deadline and gps.latest().t_mono == 0.0:
            clock.sleep(0.01)
        fix = gps.latest()
        assert fix.t_mono != 0.0, "the fix never arrived"
        assert fix.t_mono == pytest.approx((BASE_NS + 500_000_000) / 1e9, abs=1e-6), (
            "the fix was not stamped with its converted capture instant"
        )
        assert fix.timebase is not None and not fix.timebase.proxy
        # And it is emphatically not the arrival stamp, which is ~now.
        assert abs(fix.t_mono - fix.timebase.t_arrival_mono) > 1.0
    finally:
        gps.stop()
        phone.close()
        jetson.close()


def test_the_tick_record_carries_the_timebase_it_used():
    """Plan section 4's whole argument is that this record makes the proxy risk
    auditable after the fact. Setting `Tick.timebase = None` unconditionally left
    the suite green, and so did claiming `converted: true` on a proxied stamp."""
    import time as clock

    from sensors.camera_stream import Frame

    pipeline = _a_pipeline()
    now = clock.monotonic()
    converted_stamp = TimebaseStamp(t_capture_mono=now - 0.01, t_arrival_mono=now,
                                    bound_s=0.008, estimate_id=42, proxy=False)
    tick = pipeline.step(
        Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), frame_id=1,
              t_mono=now - 0.01, t_wall=0.0, timebase=converted_stamp),
        GpsFix(), None, detections_override=[],
    )
    assert tick.timebase is not None, "the tick did not record its timebase"
    assert tick.timebase["converted"] is True
    assert tick.timebase["proxy"] is False
    assert tick.timebase["estimate_id"] == 42
    assert tick.timebase["bound_ms"] == pytest.approx(8.0, abs=0.01)
    assert tick.to_record()["timebase"]["estimate_id"] == 42


def test_a_proxied_stamp_does_not_claim_to_have_been_converted():
    """The sharpest of the record mutations: a proxied stamp reporting
    `converted: true` and nothing noticing."""
    now = 1000.0
    proxied = TimebaseStamp(t_capture_mono=now, t_arrival_mono=now, bound_s=None,
                            estimate_id=None, proxy=True)
    record = proxied.to_record()
    assert record["converted"] is False
    assert record["proxy"] is True
    assert record["bound_ms"] is None
    assert record["estimate_id"] is None
    # And no link segment, because none was measured.
    assert record["link_ms"] is None
    assert proxied.link_s is None


def test_the_observation_diagnostics_carry_the_timebase_too():
    """`gps_timebase = None` unconditionally also left the suite green."""
    now = 884_285.2
    stamp = TimebaseStamp(t_capture_mono=now - 0.01, t_arrival_mono=now,
                          bound_s=0.008, estimate_id=7, proxy=False)
    fix = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                 t_mono=now - 0.01, t_wall=0.0, timebase=stamp)
    diagnostics = ObservationBuilder(BuilderConfig()).build(
        [], fix, now, None
    ).diagnostics
    assert diagnostics["gps_timebase"] is not None
    assert diagnostics["gps_timebase"]["estimate_id"] == 7
    # None for a local fix, because there is nothing to record.
    local = ObservationBuilder(BuilderConfig()).build(
        [], GpsFix(valid=True, speed_mps=27.0, t_mono=now), now, None
    ).diagnostics
    assert local["gps_timebase"] is None


def test_a_frame_is_delivered_once_and_drops_are_counted():
    """Nothing called `wait_for_fresh` twice on a live stream, so deleting the
    consumption bookkeeping returned the same frame forever -- a harness would
    tick on frame 1 indefinitely with a full tick count and an intact account.
    And nothing read `dropped_frames`."""
    phone, jetson, up, down = phone_and_jetson()
    estimator, _ = converged_estimator()
    camera = PhoneCameraStream(down, PhoneClockAdapter(estimator)).start()
    try:
        assert up.send(CameraFrame(
            t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS, frame_id=1, width=16,
            height=16, format="jpeg", quality=85, jpeg=a_jpeg(),
        ))
        first = camera.wait_for_fresh(timeout=5.0)
        assert first is not None and first.frame_id == 1
        # Nothing new has arrived, so the second call must not hand back frame 1.
        assert camera.wait_for_fresh(timeout=0.2) is None, (
            "the same frame was delivered twice"
        )

        # A frame replaced before it was consumed is a counted drop.
        before = camera.dropped_frames
        for frame_id in (2, 3):
            assert up.send(CameraFrame(
                t_capture_mono_ns=BASE_NS + PLANTED_OFFSET_NS + frame_id, frame_id=frame_id,
                width=16, height=16, format="jpeg", quality=85, jpeg=a_jpeg(),
            ))
            import time as clock
            clock.sleep(0.05)
        latest = camera.wait_for_fresh(timeout=5.0)
        assert latest is not None and latest.frame_id == 3, latest.frame_id
        assert camera.dropped_frames == before + 1, (
            f"frame 2 was replaced unconsumed but drops went {before} -> "
            f"{camera.dropped_frames}"
        )
    finally:
        camera.stop()
        phone.close()
        jetson.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_the_reader_surfaces_a_failure_instead_of_looking_healthy():
    """The transport's writer loop learned this and documented it against itself;
    this module reintroduced it one layer up. A dead reader used to report
    end_of_stream False, zeros in its record, and block a consumer forever."""
    import contextlib
    import io

    class _Exploding:
        def recv_with_receipt(self, channel, timeout=0.0):
            raise RuntimeError("boom")

    camera = PhoneCameraStream(_Exploding(), PhoneClockAdapter(TimebaseEstimator()))
    try:
        # The guard re-raises after recording, so the thread's traceback is not
        # swallowed; it goes to stderr and is expected here.
        with contextlib.redirect_stderr(io.StringIO()):
            camera.start()
            import time as clock
            deadline = clock.monotonic() + 5.0
            # Wait for the thread to be gone, not merely for the failure to be
            # recorded: the guard records and then re-raises, so `is_alive()` is
            # briefly true in between and a check there is a race.
            while clock.monotonic() < deadline and camera._thread.is_alive():
                clock.sleep(0.01)
        assert camera.failure is not None and "boom" in camera.failure
        assert camera.end_of_stream is True, "a dead reader did not end the stream"
        record = camera.to_record()
        assert record["reader_alive"] is False
        assert record["failure"] is not None
        # And a consumer is not left blocking with no way to learn why.
        assert camera.wait_for_fresh(timeout=5.0) is None
    finally:
        camera.stop()


def test_is_stale_refuses_a_fix_from_the_future():
    """The same one-sidedness the builder's gate closed, in the other freshness
    predicate. The two must not disagree about the dangerous direction."""

    class _Silent:
        def recv_with_receipt(self, channel, timeout=0.0):
            return None

    gps = PhoneGpsReader(_Silent(), PhoneClockAdapter(TimebaseEstimator()),
                         stale_after_s=2.0)
    now = 1000.0
    gps._fix = GpsFix(valid=True, speed_mps=1.0, t_mono=now + 60.0)
    assert gps.is_stale(now) is True, "a fix from this clock's future was called fresh"
    gps._fix = GpsFix(valid=True, speed_mps=1.0, t_mono=now - 0.5)
    assert gps.is_stale(now) is False
    gps._fix = GpsFix(valid=True, speed_mps=1.0, t_mono=now - 60.0)
    assert gps.is_stale(now) is True


# -- the allowance, at the widths that distinguish the formulas ---------------


def _fresh(ahead_s: float, bound_s: float | None, now: float = 884_285.2) -> dict:
    stamp = None if bound_s is None else TimebaseStamp(
        t_capture_mono=now + ahead_s, t_arrival_mono=now, bound_s=bound_s,
        estimate_id=1, proxy=False,
    )
    fix = GpsFix(valid=True, speed_mps=27.0, fix_quality=1, num_sats=9,
                 t_mono=now + ahead_s, t_wall=0.0, timebase=stamp)
    return ObservationBuilder(BuilderConfig()).build([], fix, now, None).diagnostics


def test_the_future_allowance_is_the_stamps_own_bound():
    """The case that separates the bound from the sampling epsilon.

    Both earlier tests probed where `max(epsilon, bound)` and `epsilon` alone
    agree -- 0.004 s ahead (under both) and 0.5 s ahead (over both) -- so
    deleting the bound lookup entirely left the suite green. 80 ms ahead is
    inside a 106 ms bound and outside a 50 ms epsilon, which is the only kind of
    case that can tell them apart. A 106 ms bound is what a 200 ms link
    produces, and 200 ms is exactly the round-trip ceiling the gate admits.
    """
    assert _fresh(0.080, 0.106)["gps_fresh"] is True, (
        "a stamp inside its own 106 ms bound was rejected"
    )
    assert _fresh(0.080, 0.008)["gps_fresh"] is False, (
        "80 ms into the future was accepted on the strength of an 8 ms bound"
    )


def test_a_bound_too_wide_to_decide_is_refused_not_indulged():
    """`max()` had no cap, so a stamp claiming a 10 s bound was granted 10 s of
    future tolerance -- more slack than the whole 2 s past window -- and read as
    measured nine seconds into this clock's future.

    The isolating case is a bound just past half the window on a fix that is
    otherwise perfectly current: every other clause passes it, so only the
    unresolved check can refuse it. Probing with 9 s ahead and a 10 s bound does
    not isolate anything -- the future-side clause refuses that on its own.
    """
    borderline = _fresh(0.0, 1.5)  # age 0, bound 1.5 s against a 2 s window
    assert borderline["gps_fresh"] is False, (
        "a bound wider than half the window decided freshness anyway"
    )
    assert borderline["gps_timebase_unresolved"] is True

    wide = _fresh(9.0, 10.0)
    assert wide["gps_fresh"] is False
    assert wide["gps_timebase_unresolved"] is True

    narrow = _fresh(0.0, 0.008)
    assert narrow["gps_timebase_unresolved"] is False
    assert narrow["gps_fresh"] is True


def test_the_bound_is_charged_on_the_past_side_too():
    """Otherwise the gate answers "possibly fresh" going back and "certainly
    fresh" going forward: a reading 1.9 s old with a 0.4 s uncertainty may really
    be 2.3 s old, which is outside a 2 s window."""
    assert _fresh(-1.9, 0.4)["gps_fresh"] is False
    assert _fresh(-1.0, 0.4)["gps_fresh"] is True
    # A local fix has no uncertainty to charge, so the window is the window.
    assert _fresh(-1.9, None)["gps_fresh"] is True


def test_the_camera_can_be_restarted_after_a_stop():
    """`stop()` marks the stream ended, and `start()` has to clear that or a
    restarted camera is permanently ended with a live reader thread -- and
    `health()` publishes the contradictory pair reader_alive True beside
    end_of_stream True. The local CameraStream has no such state, so this was a
    divergence in the one direction the docstring claims they are alike."""

    class _Silent:
        def recv_with_receipt(self, channel, timeout=0.0):
            return None

    camera = PhoneCameraStream(_Silent(), PhoneClockAdapter(TimebaseEstimator()))
    camera.start()
    camera.stop()
    assert camera.end_of_stream is True
    camera.start()
    try:
        assert camera.end_of_stream is False, "a restarted camera is still ended"
        assert camera.failure is None
        record = camera.to_record()
        assert record["reader_alive"] is True and record["end_of_stream"] is False
    finally:
        camera.stop()


def test_a_clean_stop_ends_the_stream_so_a_consumer_can_terminate():
    """`run_demo` breaks on `end_of_stream`. Only the failure path was pinned, so
    making a clean stop not end the stream left the suite green -- and a clean
    stop is the ordinary case."""

    class _Silent:
        def recv_with_receipt(self, channel, timeout=0.0):
            return None

    camera = PhoneCameraStream(_Silent(), PhoneClockAdapter(TimebaseEstimator())).start()
    assert camera.end_of_stream is False
    camera.stop()
    assert camera.end_of_stream is True, "a stopped stream did not report itself ended"
    assert camera.wait_for_fresh(timeout=5.0) is None


def test_reader_alive_accounts_for_a_recorded_failure():
    """The window between the guard recording a failure and the thread
    unwinding. With a dead thread the answer is False either way, so the test has
    to hold a LIVE thread with a failure recorded -- which is exactly the state
    `to_record()` could publish as reader_alive: True."""
    import threading

    class _Blocking:
        def recv_with_receipt(self, channel, timeout=0.0):
            import time as clock
            clock.sleep(0.01)
            return None

    camera = PhoneCameraStream(_Blocking(), PhoneClockAdapter(TimebaseEstimator())).start()
    try:
        assert camera._thread.is_alive()
        assert camera.health()["reader_alive"] is True
        # Simulate the guard having recorded a failure while the thread unwinds.
        camera.failure = "RuntimeError: boom"
        assert camera._thread.is_alive(), "the thread must still be live for this test"
        assert camera.health()["reader_alive"] is False, (
            "a reader with a recorded failure reported itself alive"
        )
        assert camera.to_record()["reader_alive"] is False
    finally:
        camera.failure = None
        camera.stop()


def test_a_zero_tick_run_can_still_print_its_summary():
    """The crash my own fix introduced. Making an empty series report absence
    rather than zeros meant run_demo's fallback needed every key the line below
    it reads; it carried only `mean`, so a zero-tick run raised KeyError inside
    the shutdown path and lost its summary. There is no test file for run_demo,
    which is why it was free to happen."""
    from pipeline import PipelineStats

    from run_demo import summary_line

    snapshot = PipelineStats().snapshot()
    assert snapshot["jetson_ms"] is None

    # run_demo's own function, not a copy of it. The first version of this test
    # reimplemented the format string, so mutating run_demo could not fail it --
    # a test that reimplements the code cannot catch a change to the code.
    rendered = summary_line({"ticks": 0, "stats": snapshot}, dropped_frames=0)
    assert "nan" in rendered
    assert "p50" in rendered and "p95" in rendered

    # And a populated series renders real numbers through the same path.
    populated = summary_line(
        {"ticks": 10, "stats": {"jetson_ms": {"n": 10, "mean": 1.5, "p50": 1.4,
                                              "p95": 2.0}}},
        dropped_frames=3,
    )
    assert "mean 1.5 ms" in populated and "dropped frames 3" in populated

    # And the dashboard's guard: get(k, default) does not apply the default for a
    # present-but-None key.
    assert (snapshot.get("link_ms") or {}).get("p50", 0) == 0


# -- the clock adapter's cascade ----------------------------------------------


def test_round_trip_is_tried_before_one_way():
    """Given both, the round-trip estimator wins: its error is bounded by half
    a round trip, the one-way estimator's only by a delay spread with an
    unobservable floor."""
    round_trip, _ = converged_estimator(offset_ns=1_000_000_000)
    one_way = OneWayEstimator(mono_clock=lambda: BASE_NS)
    for i in range(MIN_OFFSET_SAMPLES + 2):
        one_way.add(OneWaySample(
            exchange_id=i, t1_remote_send_ns=BASE_NS + 2_000_000_000,
            t2_local_recv_ns=BASE_NS,
        ))
    adapter = PhoneClockAdapter(one_way, round_trip=round_trip)

    stamp = adapter.stamp(BASE_NS + 1_000_000_000, BASE_NS / 1e9)
    assert not stamp.proxy
    assert stamp.source == "round_trip"
    assert adapter.to_record()["converted_by_source"] == {"round_trip": 1, "one_way": 0}


def test_one_way_is_used_when_round_trip_is_not_ready():
    """The fallback still fires, and says which source answered."""
    round_trip = TimebaseEstimator(mono_clock=lambda: BASE_NS)  # no samples, never usable
    one_way = OneWayEstimator(mono_clock=lambda: BASE_NS)
    for i in range(MIN_OFFSET_SAMPLES + 2):
        one_way.add(OneWaySample(
            exchange_id=i, t1_remote_send_ns=BASE_NS + 2_000_000_000,
            t2_local_recv_ns=BASE_NS,
        ))
    adapter = PhoneClockAdapter(one_way, round_trip=round_trip)

    stamp = adapter.stamp(BASE_NS + 2_000_000_000, BASE_NS / 1e9)
    assert not stamp.proxy
    assert stamp.source == "one_way"
    assert adapter.to_record()["converted_by_source"] == {"round_trip": 0, "one_way": 1}


def test_neither_ready_proxies_and_names_the_one_way_reason():
    """The last reason tried is the one recorded -- the one-way estimator's,
    since it is tried last -- and a stamp carries it, not only the aggregate."""
    round_trip = TimebaseEstimator(mono_clock=lambda: BASE_NS)
    one_way = OneWayEstimator(mono_clock=lambda: BASE_NS)
    adapter = PhoneClockAdapter(one_way, round_trip=round_trip)

    stamp = adapter.stamp(BASE_NS, BASE_NS / 1e9)
    assert stamp.proxy
    assert stamp.source == "proxy"
    assert stamp.proxy_reason == "no samples"
    assert adapter.to_record()["proxy_reasons"] == {"no samples": 1}


def test_a_single_estimator_caller_is_unaffected_by_the_cascade():
    """Every existing caller passes one estimator, of either kind, and gets it
    tried on its own -- the shape this class had before `round_trip` existed."""
    estimator, _ = converged_estimator()
    adapter = PhoneClockAdapter(estimator)
    stamp = adapter.stamp(BASE_NS + PLANTED_OFFSET_NS + 500_000_000, BASE_NS / 1e9)
    assert not stamp.proxy
    assert stamp.source == "one_way"
    assert adapter.converted == 1


# -- per-tick stage instrumentation on a phone-fed frame ----------------------


def a_receipt(*, t_mono_ns: int, t_recv_mono_ns: int, extensions: dict | None = None,
              t_wall_ns: int = 0) -> ReceivedMessage:
    frame = WireFrame(
        channel=Channel.CAMERA, seq=1, t_mono_ns=t_mono_ns, t_wall_ns=t_wall_ns,
        extensions=extensions or {},
    )
    return ReceivedMessage(frame=frame, t_recv_mono_ns=t_recv_mono_ns, t_recv_wall_ns=0)


def a_bare_camera_stream(adapter=None) -> PhoneCameraStream:
    """`_accept` is exercised directly, so the router is never touched."""
    return PhoneCameraStream(router=None, adapter=adapter or PhoneClockAdapter(TimebaseEstimator()))


def test_jpeg_decode_time_is_measured_and_never_negative():
    camera = a_bare_camera_stream()
    message = CameraFrame(
        t_capture_mono_ns=100, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    receipt = a_receipt(t_mono_ns=100, t_recv_mono_ns=110)
    camera._accept(message, receipt, stamp)

    frame = camera.latest()
    assert frame is not None
    assert frame.jpeg_decode_s is not None
    assert frame.jpeg_decode_s >= 0.0


def test_capture_is_an_instant_with_no_duration():
    camera = a_bare_camera_stream()
    message = CameraFrame(
        t_capture_mono_ns=100, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    camera._accept(message, a_receipt(t_mono_ns=100, t_recv_mono_ns=110), stamp)

    stages = camera.latest().phone_stages
    assert stages["capture"].basis == "instant"
    assert stages["capture"].ms == 0.0
    assert stages["capture"].clock == "phone"


def test_the_phone_dwell_segments_are_absent_without_encode_stamps():
    """An older phone build, or one that never instruments encode timing, must
    read as absent-with-a-reason -- never as a silent zero."""
    camera = a_bare_camera_stream()
    message = CameraFrame(
        t_capture_mono_ns=100, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    camera._accept(message, a_receipt(t_mono_ns=140, t_recv_mono_ns=150), stamp)

    stages = camera.latest().phone_stages
    for key in ("capture_to_encode_start", "encode", "encode_done_to_enqueue"):
        assert stages[key].basis == "absent", key
        assert stages[key].ms is None, key
        assert stages[key].reason, key


def test_the_phone_dwell_segments_are_measured_and_exact_with_encode_stamps():
    camera = a_bare_camera_stream()
    message = CameraFrame(
        t_capture_mono_ns=100, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
        t_encode_start_mono_ns=110, t_encode_done_mono_ns=130,
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    # enqueue (t_mono_ns) at 140, no wire stamp requested.
    camera._accept(message, a_receipt(t_mono_ns=140, t_recv_mono_ns=150), stamp)

    stages = camera.latest().phone_stages
    assert stages["capture_to_encode_start"].basis == "measured"
    assert stages["capture_to_encode_start"].ms == pytest.approx(10 / 1e6)
    assert stages["capture_to_encode_start"].clock == "phone"
    assert stages["encode"].ms == pytest.approx(20 / 1e6)
    assert stages["encode_done_to_enqueue"].ms == pytest.approx(10 / 1e6)
    # No wire stamp on this frame: the last leg and the whole transport segment
    # are absent, with a reason distinct from "no encode stamps".
    assert stages["enqueue_to_wire"].basis == "absent"
    assert stages["transport"].basis == "absent"
    assert stages["transport"].clock == "cross"


def test_a_span_whose_stamps_disagree_on_order_is_absent_not_negative():
    """A peer that reports `encode_done` before `encode_start` -- the current
    phone build never does, but nothing here trusts a peer to keep agreeing --
    must not have that disagreement reported as a negative duration."""
    camera = a_bare_camera_stream()
    message = CameraFrame(
        t_capture_mono_ns=100, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
        t_encode_start_mono_ns=130, t_encode_done_mono_ns=110,
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    camera._accept(message, a_receipt(t_mono_ns=140, t_recv_mono_ns=150), stamp)

    stage = camera.latest().phone_stages["encode"]
    assert stage.basis == "absent"
    assert stage.ms is None
    assert stage.reason == "stamps out of order"
    assert camera.out_of_order_phone_stages == 1
    assert camera.to_record()["phone_stages_out_of_order"] == 1


def test_transport_is_converted_when_the_frame_carries_a_wire_stamp():
    round_trip_estimator, _ = converged_estimator(offset_ns=0)
    adapter = PhoneClockAdapter(round_trip_estimator)
    camera = a_bare_camera_stream(adapter)
    message = CameraFrame(
        t_capture_mono_ns=BASE_NS, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    wire_ns = BASE_NS + 5_000_000
    receipt = a_receipt(
        t_mono_ns=BASE_NS, t_recv_mono_ns=BASE_NS + 20_000_000,
        extensions={WIRE_STAMP_KEY: wire_ns},
    )
    camera._accept(message, receipt, stamp)

    transport = camera.latest().phone_stages["transport"]
    assert transport.basis == "converted"
    assert transport.clock == "cross"
    assert transport.source == "one_way"
    assert transport.bound_ms is not None
    assert transport.estimate_id is not None
    assert transport.ms == pytest.approx(20.0 - 5.0, abs=1.0)
    # enqueue (t_mono_ns) at BASE_NS, wire stamp 5 ms later.
    enqueue_to_wire = camera.latest().phone_stages["enqueue_to_wire"]
    assert enqueue_to_wire.basis == "measured"
    assert enqueue_to_wire.ms == pytest.approx(5.0)


def test_transport_is_absent_when_the_wire_stamp_cannot_be_converted():
    """A frame that asked for the stamp but arrived before the estimator
    converged must not report a zero-length transport segment."""
    camera = a_bare_camera_stream()  # adapter with an unfed TimebaseEstimator
    message = CameraFrame(
        t_capture_mono_ns=BASE_NS, frame_id=1, width=16, height=16,
        format="jpeg", quality=85, jpeg=a_jpeg(),
    )
    stamp = TimebaseStamp(t_capture_mono=0.0, t_arrival_mono=0.0, bound_s=None,
                           estimate_id=None, proxy=True, source="proxy")
    receipt = a_receipt(
        t_mono_ns=BASE_NS, t_recv_mono_ns=BASE_NS + 20_000_000,
        extensions={WIRE_STAMP_KEY: BASE_NS + 5_000_000},
    )
    camera._accept(message, receipt, stamp)

    transport = camera.latest().phone_stages["transport"]
    assert transport.basis == "absent"
    assert transport.reason == "no samples"
