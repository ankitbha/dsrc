#!/usr/bin/env python3
"""Task 16: a synthetic phone driving the real transport into the real pipeline.

The two halves of this project have never been connected. Tasks 12-15 built a
transport that carries phone sensor data and returns advisories; `pipeline.py`
has run perception -> observation -> actor -> advisory since before any of it
existed. This runs one against the other, with the phone's clock deliberately
offset so the conversion is exercised rather than assumed.

**The offset is planted, and it is the real one.** The phone session runs on a
clock displaced by 67.57 hours -- measured between this Mac and the Jetson, both
counting from their own boot. Unconverted, `gps_age` comes out at -243,264 s
against a 2.0 s staleness threshold, so `gps_fresh` is False on every tick, ego
speed silently falls back to neutral, and the loop keeps producing advisories
that look fine. A run where the conversion works and a run where it does not are
distinguishable here only by the provenance table, which is the point.

**One assumption, flagged rather than buried.** Task 15's spec has the phone
initiate the time-sync exchange, so the phone holds the estimate. The converting
side is the Jetson -- it runs the pipeline and it is what consumes incoming
stamps -- so this runs the exchange the other way round: the Jetson initiates and
the phone answers. That needs no wire change and no code change, because both
roles are role-symmetric in `transport/timebase.py`; it does contradict one
sentence of `specs/transport_protocol.md`, which has not been edited and needs
sign-off. The alternative arrangements are worse: shipping the estimate needs a
new wire field, and having the phone convert its own stamps before sending
breaks the rule that a capture stamp is on its sender's clock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("DSRC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "deployment" / "jetson"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perception.detector import Detection  # noqa: E402
from perception.distance import DistanceEstimator  # noqa: E402
from perception.observation_builder import BuilderConfig, ObservationBuilder  # noqa: E402
from perception.tracker import IouTracker  # noqa: E402
from pipeline import PerceptionPolicyPipeline  # noqa: E402
from policy.actor_runtime import ActorRuntime  # noqa: E402
from policy.advisory import AdvisoryDecoder  # noqa: E402
from policy.export_policy import build_random, export  # noqa: E402
from sensors.phone_link import PhoneLink  # noqa: E402
from sensors.phone_source import PhoneCameraStream, PhoneClockAdapter, PhoneGpsReader  # noqa: E402
from transport.channels import Channel  # noqa: E402
from transport.clock import now_mono_ns, now_wall_ns  # noqa: E402
from transport.loopback import loopback_pair  # noqa: E402
from transport.messages import (  # noqa: E402
    CameraFrame,
    GpsRecord,
    InvalidMessage,
    MessageRouter,
    TimeSyncMessage,
)
from transport.client import Backoff, SessionClient  # noqa: E402
from transport.endpoint import SessionStarted, TransportListener  # noqa: E402
from transport.handshake import Hello, Role  # noqa: E402
from transport.session import Session  # noqa: E402
from transport.tcp import DEFAULT_PORT, TcpAcceptor  # noqa: E402
from transport.timebase import TimeSyncInitiator, answer_ping  # noqa: E402

from run_message_exercise import percentiles  # noqa: E402
from run_message_link import tailnet_path  # noqa: E402

# The measured difference between this Mac's and the Jetson's monotonic clocks.
# Planted rather than tuned: any value would exercise the conversion, and this
# one makes the failure mode concrete for a reader.
PHONE_CLOCK_OFFSET_NS = 243_264_000_000_000

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25
CAMERA_HZ, GPS_HZ = 10.0, 5.0
ADVISORY_ECHO_PREFIX = "frame:"
# The link segment is charged against the uncertainty the run itself reports,
# not against a constant. A flat 1000 ms ceiling was 2,600x the segment this link
# actually produces and 2,300x the bound the same run reports, so every realistic
# conversion bug -- a stale estimate, one term with the wrong sign, a
# systematically asymmetric link -- lived inside it. A conversion claiming +-0.4 ms
# has no business implying a 900 ms flight time.
LINK_FLOOR_MS = 5.0
LINK_BOUND_MULTIPLE = 10.0

# A run that converged and then FELL BACK for much of the drive is not a run that
# exercised the conversion -- plan section 9's stated risk is exactly the proxy
# being in use for longer than expected. Measured over the ticks after the first
# conversion, not over the whole run: the timebase needs ~1.1 s to converge, so a
# whole-run fraction rejected a perfectly healthy two-second run, and plan section
# 8.4 ("the first ten seconds, recorded deliberately") is short by design. A gate
# that fails a healthy rig is a false claim about the run.
MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE = 0.9

# How long the proxy may be in use before the timebase converges. Two clauses,
# not one replacing the other: measuring the fraction only after the first
# conversion fixed a healthy short run failing, but it made the prefix free at
# ANY length. Measured, a 0.25 Hz sync cadence converged at 16.1 s and 67% of the
# run's ego-speed decisions rested on the arrival proxy -- reported usable, and
# the whole-run fraction it replaced would have caught it.
#
# Convergence is 1.11 s measured at the default 4 Hz (MIN_OFFSET_SAMPLES = 5 at
# 4 Hz is 1.25 s), so this is generous by 4x. It is also the quantity plan
# section 8.4 exists to characterise, which makes the gated number the headline
# number rather than a second thing to reconcile.
CONVERGENCE_BUDGET_S = 5.0


class _FakeDetector:
    """The pipeline never calls this when detections are supplied, but the
    interface has to be complete. Scripted detections keep the loop running on a
    machine with no TensorRT and no GPU, which is what lets this run in CI."""

    def infer(self, image) -> list[Detection]:
        return []

    def warmup(self, iterations: int = 1) -> float:
        return 0.0


def project_box(z_m: float, x_m: float) -> np.ndarray:
    w_px = FX * 1.8 / z_m
    h_px = 0.85 * w_px
    u = CX + x_m * FX / z_m
    v_bottom = HORIZON + CAM_H * FX / z_m
    return np.array([u - w_px / 2, v_bottom - h_px, u + w_px / 2, v_bottom], dtype=np.float32)


def scene_detections(t_s: float) -> list[Detection]:
    """The same scripted scene the pipeline smoke test uses: a leader closing at
    2 m/s and one vehicle in each adjacent lane."""
    leader_z = max(10.0, 45.0 - 2.0 * t_s)
    boxes = [project_box(leader_z, 0.0), project_box(28.0, -3.7), project_box(60.0, 3.7)]
    return [Detection(xyxy=b, conf=0.9, cls=2) for b in boxes]


def build_pipeline(bundle_prefix: str) -> PerceptionPolicyPipeline:
    return PerceptionPolicyPipeline(
        detector=_FakeDetector(),
        tracker=IouTracker(min_hits=2),
        distance=DistanceEstimator(
            fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H, ema_alpha=0.6
        ),
        builder=ObservationBuilder(BuilderConfig()),
        actor=ActorRuntime(bundle_prefix),
        advisory_decoder=AdvisoryDecoder(units="mph"),
    )


def a_jpeg(width: int = 64, height: int = 48) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not encode the synthetic frame")
    return buffer.tobytes()


def phone_side(router, mono, stop, sent, sent_lock, advisories, jpeg,
               camera_hz=CAMERA_HZ, gps_hz=GPS_HZ) -> list[threading.Thread]:
    """The phone's threads: two senders, and one loop that answers pings and
    consumes advisories.

    Extracted so the same code drives the in-process pair and a real socket. Over
    a real link the clock difference is not planted -- it is whatever the two
    devices' uptimes give -- which is the version of this test that cannot be
    fooled by a planted constant.
    """

    def camera_sender():
        index = 0
        period = 1.0 / camera_hz
        next_at = time.monotonic()
        while not stop.is_set():
            index += 1
            try:
                if router.send(CameraFrame(
                    t_capture_mono_ns=mono(), frame_id=index, width=64, height=48,
                    format="jpeg", quality=85, jpeg=jpeg,
                )):
                    with sent_lock:
                        sent["frames"] += 1
            except InvalidMessage:
                with sent_lock:
                    sent["invalid"] += 1
            next_at += period
            delay = next_at - time.monotonic()
            stop.wait(delay) if delay > 0 else None
            if delay <= 0:
                next_at = time.monotonic()

    def gps_sender():
        period = 1.0 / gps_hz
        next_at = time.monotonic()
        while not stop.is_set():
            try:
                if router.send(GpsRecord(
                    t_capture_mono_ns=mono(), valid=True, fix_quality=1, num_sats=9,
                    lat=40.7440, lon=-74.0324, speed_mps=27.0, heading_deg=90.0,
                    hdop=0.9, altitude_m=10.0, utc_epoch_ns=now_wall_ns(),
                )):
                    with sent_lock:
                        sent["fixes"] += 1
            except InvalidMessage:
                with sent_lock:
                    sent["invalid"] += 1
            next_at += period
            delay = next_at - time.monotonic()
            stop.wait(delay) if delay > 0 else None
            if delay <= 0:
                next_at = time.monotonic()

    def responder():
        while not stop.is_set():
            arrived = router.recv_with_receipt(Channel.CONTROL, timeout=0.01)
            if arrived is not None and isinstance(arrived[0], TimeSyncMessage):
                try:
                    if router.send(answer_ping(arrived, mono_clock=mono)):
                        with sent_lock:
                            sent["pongs"] += 1
                except InvalidMessage:
                    with sent_lock:
                        sent["invalid"] += 1
            advisory = router.recv(Channel.ADVISORY, timeout=0.0)
            if advisory is not None:
                if advisory.lane_text.startswith(ADVISORY_ECHO_PREFIX):
                    advisories["received"] += 1
                    advisories["frame_ids"].add(
                        int(advisory.lane_text[len(ADVISORY_ECHO_PREFIX):])
                    )
                else:
                    advisories["unmatched"] += 1

    return [
        threading.Thread(target=camera_sender, name="phone-camera-tx", daemon=True),
        threading.Thread(target=gps_sender, name="phone-gps-tx", daemon=True),
        threading.Thread(target=responder, name="phone-responder", daemon=True),
    ]


def run(duration_s: float, offset_ns: int, sync_hz: float) -> dict:
    phone_conn, jetson_conn = loopback_pair()

    # The phone's whole session runs on a displaced clock, so every stamp it
    # produces -- captures, wire stamps, receive stamps -- is on the phone's
    # clock and nothing has to remember to offset them individually.
    def phone_mono() -> int:
        return now_mono_ns() - offset_ns

    phone = Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None,
                    mono_clock=phone_mono).start()
    jetson = Session(jetson_conn, session_id=2, heartbeat_s=None,
                     stall_timeout_s=None).start()
    phone_router, jetson_router = MessageRouter(phone), MessageRouter(jetson)

    # The Jetson initiates: it is the side that converts, so it is the side that
    # needs the estimate. See the module docstring.
    # In-process, so both halves are Python and `transport/timebase.py` is
    # role-symmetric: the Jetson can initiate here and the synthetic phone answers
    # without complaint. A real handset would not -- the spec has the phone
    # initiate and `Session.checkTimeSyncDirection` drops a ping arriving at a
    # phone -- so THIS mode does not exercise the direction rule and must not be
    # read as evidence about it. `--role jetson` does, on `PhoneLink`, which is
    # the assembly `run_demo --phone` ships.
    initiator = TimeSyncInitiator(jetson_router)
    adapter = PhoneClockAdapter(initiator.estimator)
    camera = PhoneCameraStream(jetson_router, adapter).start()
    gps = PhoneGpsReader(jetson_router, adapter).start()

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "actor_policy")
        actor, info = build_random(seed=0)
        export(actor, info, prefix)
        pipeline = build_pipeline(prefix)

        stop = threading.Event()
        sent = {"frames": 0, "fixes": 0, "pongs": 0, "invalid": 0}
        # `invalid` is written from three threads and `d[k] += 1` is not atomic.
        sent_lock = threading.Lock()
        advisories = {"received": 0, "unmatched": 0, "frame_ids": set()}
        jpeg = a_jpeg()

        def phone_camera_sender():
            index = 0
            period = 1.0 / CAMERA_HZ
            next_at = time.monotonic()
            while not stop.is_set():
                index += 1
                try:
                    if phone_router.send(CameraFrame(
                        t_capture_mono_ns=phone_mono(), frame_id=index,
                        width=64, height=48, format="jpeg", quality=85, jpeg=jpeg,
                    )):
                        sent["frames"] += 1
                except InvalidMessage:
                    with sent_lock:
                        sent["invalid"] += 1
                next_at += period
                delay = next_at - time.monotonic()
                stop.wait(delay) if delay > 0 else None
                if delay <= 0:
                    next_at = time.monotonic()

        def phone_gps_sender():
            period = 1.0 / GPS_HZ
            next_at = time.monotonic()
            while not stop.is_set():
                try:
                    if phone_router.send(GpsRecord(
                        t_capture_mono_ns=phone_mono(), valid=True, fix_quality=1,
                        num_sats=9, lat=40.7440, lon=-74.0324, speed_mps=27.0,
                        heading_deg=90.0, hdop=0.9, altitude_m=10.0,
                        utc_epoch_ns=now_wall_ns(),
                    )):
                        sent["fixes"] += 1
                except InvalidMessage:
                    with sent_lock:
                        sent["invalid"] += 1
                next_at += period
                delay = next_at - time.monotonic()
                stop.wait(delay) if delay > 0 else None
                if delay <= 0:
                    next_at = time.monotonic()

        def phone_responder():
            """Answers time-sync pings and consumes advisories."""
            while not stop.is_set():
                arrived = phone_router.recv_with_receipt(Channel.CONTROL, timeout=0.01)
                if arrived is not None and isinstance(arrived[0], TimeSyncMessage):
                    try:
                        # On the PHONE's clock. answer_ping's default is the
                        # host clock, which shipped a message with two devices'
                        # clocks inside it -- 67.57 hours apart. Inert today,
                        # because on_pong reads the wire and receipt stamps and
                        # never the pong's own capture stamp, but it falsified
                        # this harness's claim that every stamp the phone
                        # produces is on the phone's clock.
                        if phone_router.send(answer_ping(arrived, mono_clock=phone_mono)):
                            sent["pongs"] += 1
                    except InvalidMessage:
                        with sent_lock:
                            sent["invalid"] += 1
                advisory = phone_router.recv(Channel.ADVISORY, timeout=0.0)
                if advisory is not None:
                    if advisory.lane_text.startswith(ADVISORY_ECHO_PREFIX):
                        advisories["received"] += 1
                        advisories["frame_ids"].add(
                            int(advisory.lane_text[len(ADVISORY_ECHO_PREFIX):])
                        )
                    else:
                        advisories["unmatched"] += 1

        def jetson_syncer():
            period = 1.0 / sync_hz
            while not stop.is_set():
                initiator.send_ping()
                deadline = time.monotonic() + period
                while time.monotonic() < deadline and not stop.is_set():
                    initiator.pump(timeout=0.005)

        threads = [
            threading.Thread(target=phone_camera_sender, daemon=True),
            threading.Thread(target=phone_gps_sender, daemon=True),
            threading.Thread(target=phone_responder, daemon=True),
            threading.Thread(target=jetson_syncer, daemon=True),
        ]
        for thread in threads:
            thread.start()

        ticks: list[dict] = []
        started = time.monotonic()
        first_converted_at: float | None = None
        while time.monotonic() - started < duration_s:
            frame = camera.wait_for_fresh(timeout=0.5)
            if frame is None:
                continue
            fix = gps.latest()
            elapsed = time.monotonic() - started
            tick = pipeline.step(
                frame, fix, None, detections_override=scene_detections(elapsed)
            )
            if not (frame.timebase and frame.timebase.proxy) and first_converted_at is None:
                first_converted_at = elapsed
            ticks.append(_tick_row(tick, elapsed))
            jetson_router.send(_advisory_message(tick))

        stop.set()
        for thread in threads:
            thread.join(timeout=3.0)
        camera.stop()
        gps.stop()

        # Sampled while the sessions are still open, so the queue depths are the
        # real ones rather than a difference of counters.
        channels = (Channel.CAMERA, Channel.GPS)
        account = _account(
            phone.stats(), jetson.stats(),
            pending={c: phone.outbound_pending(c) for c in channels},
            pending_in={c: jetson.pending(c) for c in channels},
            decode_errors={
                c: jetson_router.stats()[c].decode_errors for c in channels
            },
        )
        report = _report(
            ticks, duration_s, offset_ns, sent, advisories, adapter, camera, gps,
            initiator, pipeline, first_converted_at,
            phone_stats=phone.stats(), jetson_stats=jetson.stats(), account=account,
        )
    phone.close()
    jetson.close()
    return report


def _tick_row(tick, elapsed: float) -> dict:
    """One trace row, shared by the loopback and split-role paths."""
    return {
                "t_s": round(elapsed, 3),
                "frame_id": tick.frame_id,
                "jetson_ms": round(tick.jetson_ms, 3),
                "link_ms": None if tick.link_ms is None else round(tick.link_ms, 3),
                "e2e_ms": round(tick.e2e_ms, 3),
                "timebase": tick.timebase,
                "gps_fresh": tick.obs_result.diagnostics.get("gps_fresh"),
                "gps_age_s": tick.obs_result.diagnostics.get("gps_age_s"),
                "ego_speed_source": tick.obs_result.field_sources.get("ego_speed"),
                "advisory": tick.advisory.one_line(),
    }


def _advisory_message(tick):
    """Task 14's bridge, with the frame id riding the lane text.

    The bridge is duck-typed and already exists, so it is used rather than
    reimplemented -- the hand-rolled version got the field names wrong, which is
    the argument for not having one. The frame id goes in `lane_text` for the
    same reason task 15's probe put an exchange id there: it is free-form display
    text, and a round trip that cannot be matched to its origin is two counters
    that happen to agree rather than a closed loop.
    """
    from dataclasses import replace

    from transport.messages import advisory_message_from_advisory

    marked = replace(tick.advisory, lane_text=f"{ADVISORY_ECHO_PREFIX}{tick.frame_id}")
    return advisory_message_from_advisory(marked, now_mono_ns())


def _report(ticks, duration_s, offset_ns, sent, advisories, adapter, camera, gps,
            initiator, pipeline, first_converted_at, phone_stats, jetson_stats,
            account) -> dict:
    converted = [t for t in ticks if t["timebase"] and t["timebase"]["converted"]]
    proxied = [t for t in ticks if t["timebase"] and t["timebase"]["proxy"]]
    fresh = [t for t in ticks if t["gps_fresh"]]
    converted_fresh = sum(1 for t in converted if t["gps_fresh"])
    link_summary = percentiles([int(t["link_ms"] * 1e6) for t in converted])
    # max and min, not p95. At n=91 a p95 leaves four values above it, so a
    # minority of grossly wrong conversions hid behind it entirely.
    link_max_ms = max((t["link_ms"] for t in converted), default=None)
    link_min_ms = min((t["link_ms"] for t in converted), default=None)
    bounds = [t["timebase"]["bound_ms"] for t in converted
              if t["timebase"].get("bound_ms") is not None]
    bound_p95_ms = percentiles([int(b * 1e6) for b in bounds]).get("p95") if bounds else None
    ceiling_ms = (
        None if bound_p95_ms is None
        else max(LINK_FLOOR_MS, LINK_BOUND_MULTIPLE * bound_p95_ms)
    )
    # Only the ticks from the first conversion onwards can be charged: before it
    # the estimator had no answer to give, which is expected rather than a fault.
    # Rounded on both sides. `t_s` is stored rounded to 3 dp and was compared
    # against an unrounded threshold, so the first converted tick fell out of
    # `after` -- harmless to the fraction, since it left numerator and
    # denominator together, but it made `ticks_before_convergence` wrong by one
    # in every run and turned a one-tick run's verdict into a rounding coin flip.
    converged_at = None if first_converted_at is None else round(first_converted_at, 3)
    after = [t for t in ticks if converged_at is not None and t["t_s"] >= converged_at]
    converted_after = [t for t in after if t["timebase"] and t["timebase"]["converted"]]
    converted_fraction = (len(converted_after) / len(after)) if after else 0.0
    return {
        "duration_s": duration_s,
        "planted_clock_offset_hours": round(offset_ns / 3.6e12, 3),
        "ticks": len(ticks),
        "advisories_returned": advisories["received"],
        "advisories_unmatched": advisories["unmatched"],
        # Matched by frame id, so this is a closed loop and not two counters
        # that happen to agree.
        "advisory_frame_ids_matched": len(advisories["frame_ids"] & {t["frame_id"] for t in ticks}),
        "phone_sent": dict(sent),
        # The transport's own counters, so the account can close. `sent` counts
        # send() returning True, which is ENQUEUED: camera is LATEST_WINS at
        # depth 1, so a replacement increments dropped_outbound and the two
        # numbers diverge with nothing else recording why.
        "transport": account,
        "camera": camera.to_record(),
        "gps": gps.to_record(),
        "adapter": adapter.to_record(),
        "timesync": initiator.to_record(),
        "latency": {
            "jetson_ms": percentiles([int(t["jetson_ms"] * 1e6) for t in ticks]),
            "link_ms": link_summary,
            "e2e_ms": percentiles([int(t["e2e_ms"] * 1e6) for t in ticks]),
        },
        "pipeline_stats": pipeline.stats.snapshot(),
        "conversion": {
            "ticks_converted": len(converted),
            "ticks_proxied": len(proxied),
            "first_converted_tick_at_s": first_converted_at,
            # The whole point of the task, stated rather than left to a reader:
            # with the clocks 67 hours apart, an unconverted stamp makes this
            # zero on every tick.
            "ticks_with_fresh_gps": len(fresh),
            "fresh_fraction": None if not ticks else round(len(fresh) / len(ticks), 4),
            "ego_speed_sources": _tally(t["ego_speed_source"] for t in ticks),
        },
        # One gate, and it has to charge the CONVERTED population.
        #
        # It used to ask only that some tick had a fresh fix, which the proxied
        # ticks satisfy by construction -- their capture stamp IS their arrival
        # stamp, so their age is ~0 whatever the conversion does. Measured
        # against a to_local replaced by the identity, the old gate returned
        # usable=true and exit 0 while reporting a 67.6-HOUR link segment and a
        # fresh fraction of 0.12. A gate that cannot fail on the defect the task
        # exists to prevent is not a gate.
        "usable": bool(
            ticks
            and advisories["received"] > 0
            and converted
            # Every converted tick must be fresh.
            and converted_fresh == len(converted)
            # Enough of the run actually exercised the conversion.
            and converted_fraction >= MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE
            # The worst segment must sit inside what the run's own uncertainty
            # allows -- not inside a constant that is three orders wider.
            and ceiling_ms is not None
            and link_max_ms is not None
            and link_max_ms <= ceiling_ms
            # And no further into the impossible direction than the bound
            # permits. Signed, not abs(): a negative flight time is not merely
            # large, it is the wrong sign, and abs() accepted -899 ms without
            # objection while leaving the only real check in another module.
            and link_min_ms is not None
            and link_min_ms >= -(bound_p95_ms or 0.0)
            and first_converted_at is not None
            # And it converged promptly. Without this the prefix is unbounded and
            # a run that spent most of itself proxying passes.
            and first_converted_at <= CONVERGENCE_BUDGET_S
        ),
        "gate_detail": {
            "converted_ticks": len(converted),
            "converted_and_fresh": converted_fresh,
            "converted_fraction_after_convergence": round(converted_fraction, 4),
            "min_converted_fraction_after_convergence": (
                MIN_CONVERTED_FRACTION_AFTER_CONVERGENCE
            ),
            "ticks_before_convergence": len(ticks) - len(after),
            "convergence_budget_s": CONVERGENCE_BUDGET_S,
            "link_max_ms": link_max_ms,
            "link_min_ms": link_min_ms,
            "bound_p95_ms": bound_p95_ms,
            "link_ceiling_ms": ceiling_ms,
            "first_converted_tick_at_s": first_converted_at,
        },
        "trace": ticks,
    }


def _account(phone_stats, jetson_stats, pending, pending_in, decode_errors) -> dict:
    """Reconciled from records, never by subtraction."""
    up = {}
    for channel in (Channel.CAMERA, Channel.GPS):
        out = phone_stats.channels[channel]
        inn = jetson_stats.channels[channel]
        up[channel.value] = {
            "queued": out.queued,
            # From the session's own queue depth, NOT derived as
            # `queued - accounted`. Derived, it made the identity below a
            # tautology: `accounted + (queued - accounted)` is `queued` for every
            # input, so the clause could not fail and the comment claiming it
            # verified something was false. Exhaustive sweep: 1170 of 1296 count
            # combinations "fired", none of them reachable.
            "in_flight": pending[channel],
            "sent": out.sent,
            "dropped_outbound": out.dropped_outbound,
            "abandoned_outbound": out.abandoned_outbound,
            "received": inn.received,
            "dropped_inbound": inn.dropped_inbound,
            "delivered": inn.delivered,
            # Still in the session's inbound queue when stats were sampled: the
            # receive-side twin of `in_flight`, and without it this identity is
            # flaky rather than wrong.
            "queued_in": pending_in[channel],
            "decode_errors": decode_errors.get(channel, 0),
            "seq_gaps": inn.seq_gaps,
            "missing_seqs": inn.missing_seqs,
        }
    gaps = []
    for channel, row in up.items():
        # `dropped_outbound` is a TERM of this identity, so the account closes by
        # construction exactly when a depth-1 LATEST_WINS channel drops. Closing
        # is therefore not the same as losing nothing, and the loss is named
        # separately rather than left for a reader to infer from a true flag.
        accounted = row["sent"] + row["dropped_outbound"] + row["abandoned_outbound"]
        if row["queued"] != accounted + row["in_flight"]:
            gaps.append(
                f"{channel}: queued {row['queued']} != sent {row['sent']} + dropped "
                f"{row['dropped_outbound']} + abandoned {row['abandoned_outbound']} "
                f"+ in flight {row['in_flight']}"
            )
        if row["sent"] != row["received"]:
            gaps.append(f"{channel}: {row['sent']} sent but {row['received']} received")
        # An arrived record that reached no consumer. This is NOT the
        # malformed-message check -- the session counts `delivered` when the
        # frame leaves it, and the router rejects an undecodable one afterwards,
        # so a run where every message failed to decode reconciles clean on this
        # clause alone. The router's own count is charged separately below.
        if row["received"] != row["delivered"] + row["dropped_inbound"] + row["queued_in"]:
            gaps.append(
                f"{channel}: received {row['received']} but delivered "
                f"{row['delivered']} + dropped inbound {row['dropped_inbound']} "
                f"+ still queued {row['queued_in']}"
            )
        # The malformed-message check, from the router that actually refuses them.
        if row["decode_errors"]:
            gaps.append(f"{channel}: {row['decode_errors']} records failed to decode")
    lost = {c: r["dropped_outbound"] + r["abandoned_outbound"] + r["dropped_inbound"]
            for c, r in up.items()}
    return {
        "upstream": up,
        "reconciliation": gaps,
        "reconciled": not gaps,
        # Stated outright. The account closing says the numbers are consistent,
        # not that nothing was lost.
        "lost_by_channel": lost,
        "lost_total": sum(lost.values()),
    }


def _tally(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def run_link_phone(args) -> dict:
    """The phone role over a real socket. It dials, per the binding constraint."""
    client = SessionClient(
        args.host, args.port,
        local_hello=Hello(device_id="phone-loopback-pipeline", role=Role.PHONE),
        backoff=Backoff(), heartbeat_s=1.0, stall_timeout_s=10.0,
        handshake_timeout_s=10.0,
    ).start()
    stop = threading.Event()
    sent = {"frames": 0, "fixes": 0, "pongs": 0, "invalid": 0}
    sent_lock = threading.Lock()
    advisories = {"received": 0, "unmatched": 0, "frame_ids": set()}
    jpeg = a_jpeg()
    threads: list[threading.Thread] = []
    started = time.monotonic()
    router = None
    while time.monotonic() - started < args.duration:
        session = client.current_session
        if session is not None and not session.is_closed and router is None:
            router = MessageRouter(session)
            # No offset argument: over a real link the clock difference is the
            # real one, which is the version of this that a planted constant
            # cannot fake.
            threads = phone_side(router, now_mono_ns, stop, sent, sent_lock,
                                 advisories, jpeg)
            for thread in threads:
                thread.start()
        stop.wait(0.1)
    stop.set()
    for thread in threads:
        thread.join(timeout=3.0)
    client.stop()
    return {
        "role": "phone",
        "duration_s": args.duration,
        "link_path": tailnet_path(args.host),
        "phone_sent": dict(sent),
        "advisories_seen": advisories["received"],
        "advisories_unmatched": advisories["unmatched"],
        "usable": bool(router is not None and sent["frames"] > 0
                       and advisories["received"] > 0),
    }


def run_link_jetson(args) -> dict:
    """The Jetson role: accept, then run the real pipeline on what arrives.

    Built on `PhoneLink`, which is the same assembly `run_demo --phone` uses, so
    this harness exercises the arrangement that ships rather than one of its own.

    It used to run `TimeSyncInitiator` here -- the Jetson asking and the phone
    answering. That works when this script owns both ends, because
    `transport/timebase.py` is role-symmetric, and it cannot work against a
    handset: the spec has the phone initiate, and `Session.checkTimeSyncDirection`
    drops a ping arriving at a phone as `unknown_value`. So the one end-to-end
    harness in the repo was exercising a path a real phone refuses, which is the
    opposite of what an end-to-end harness is for.
    """
    link = PhoneLink(
        host=args.host, port=args.port, device_id="jetson-loopback-pipeline",
        heartbeat_s=1.0, stall_timeout_s=10.0, handshake_timeout_s=10.0,
    )
    print(f"listening on {link.host}:{link.port} for {args.duration}s", flush=True)

    if not link.wait_for_phone(timeout_s=args.duration):
        for refusal in link.refusals:
            print(f"refused a connection -- {refusal}", flush=True)
        link.stop()
        return {"role": "jetson", "usable": False, "why": "no session was established",
                "refusals": list(link.refusals)}

    try:
        return _run_link_jetson_session(args, link)
    finally:
        link.stop()


def _run_link_jetson_session(args, link) -> dict:
    """The run itself, with the link's teardown guaranteed by the caller."""
    session = link.session
    print(f"session {session.session_id} from {link.peer_device_id}", flush=True)
    adapter = link.adapter
    # Dropped when this moved onto PhoneLink, while both uses stayed: the
    # advisory send in the tick loop and `router.stats()` in the accounting. So
    # every path that got as far as a frame -- or as far as the duration expiring
    # -- raised NameError, and the only path that returned was "no phone dialled
    # in". The harness could not produce a usable report at all.
    router = link.router
    camera, gps = link.camera, link.gps
    stop = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "actor_policy")
        actor, info = build_random(seed=0)
        export(actor, info, prefix)
        pipeline = build_pipeline(prefix)

        ticks: list[dict] = []
        advisories = {"received": 0, "unmatched": 0, "frame_ids": set()}
        first_converted_at: float | None = None
        started = time.monotonic()
        while time.monotonic() - started < args.duration and not session.is_closed:
            frame = camera.wait_for_fresh(timeout=0.5)
            if frame is None:
                continue
            elapsed = time.monotonic() - started
            tick = pipeline.step(frame, gps.latest(), None,
                                 detections_override=scene_detections(elapsed))
            if not (frame.timebase and frame.timebase.proxy) and first_converted_at is None:
                first_converted_at = elapsed
            ticks.append(_tick_row(tick, elapsed))
            router.send(_advisory_message(tick))
        stop.set()
        camera.stop()
        gps.stop()

        channels = (Channel.CAMERA, Channel.GPS)
        account = _account(
            session.stats(), session.stats(),
            pending={c: session.outbound_pending(c) for c in channels},
            pending_in={c: session.pending(c) for c in channels},
            decode_errors={c: router.stats()[c].decode_errors for c in channels},
        )
        report = _report(
            ticks, args.duration, 0, {"frames": 0}, advisories, adapter, camera, gps,
            link.estimator, pipeline, first_converted_at,
            phone_stats=session.stats(), jetson_stats=session.stats(), account=account,
        )
    # Over a real link the advisory return path is measured on the phone side,
    # so this role's gate does not charge it: the two reports are read together.
    report["role"] = "jetson"
    report["advisories_returned_note"] = "measured on the phone side"
    report["usable"] = bool(
        ticks and report["gate_detail"]["converted_ticks"] > 0
        and report["gate_detail"]["converted_and_fresh"]
        == report["gate_detail"]["converted_ticks"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("loopback", "phone", "jetson"),
                        default="loopback",
                        help="loopback runs both sides in-process; phone/jetson split "
                             "across a real socket, where the clock offset is real")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sync-hz", type=float, default=4.0)
    parser.add_argument("--clock-offset-hours", type=float, default=None,
                        help="override the planted phone/Jetson clock offset")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    offset = (
        PHONE_CLOCK_OFFSET_NS if args.clock_offset_hours is None
        else int(args.clock_offset_hours * 3.6e12)
    )
    if args.role == "phone":
        report = run_link_phone(args)
    elif args.role == "jetson":
        report = run_link_jetson(args)
    else:
        report = run(args.duration, offset, args.sync_hz)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
