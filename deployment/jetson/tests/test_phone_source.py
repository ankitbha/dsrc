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
from transport.loopback import loopback_pair
from transport.messages import CameraFrame, GpsRecord, MessageRouter
from transport.session import Session
from transport.timebase import (
    MIN_OFFSET_SAMPLES,
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

    # But not beyond the staleness window.
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
