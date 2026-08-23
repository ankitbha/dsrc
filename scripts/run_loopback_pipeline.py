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
from transport.session import Session  # noqa: E402
from transport.timebase import TimeSyncInitiator, answer_ping  # noqa: E402

from run_message_exercise import percentiles  # noqa: E402

# The measured difference between this Mac's and the Jetson's monotonic clocks.
# Planted rather than tuned: any value would exercise the conversion, and this
# one makes the failure mode concrete for a reader.
PHONE_CLOCK_OFFSET_NS = 243_264_000_000_000

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25
CAMERA_HZ, GPS_HZ = 10.0, 5.0
ADVISORY_ECHO_PREFIX = "frame:"


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
                        if phone_router.send(answer_ping(arrived)):
                            sent["pongs"] += 1
                    except InvalidMessage:
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
            ticks.append({
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
            })
            jetson_router.send(_advisory_message(tick))

        stop.set()
        for thread in threads:
            thread.join(timeout=3.0)
        camera.stop()
        gps.stop()

        report = _report(
            ticks, duration_s, offset_ns, sent, advisories, adapter, camera, gps,
            initiator, pipeline, first_converted_at,
        )
    phone.close()
    jetson.close()
    return report


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
            initiator, pipeline, first_converted_at) -> dict:
    converted = [t for t in ticks if t["timebase"] and t["timebase"]["converted"]]
    proxied = [t for t in ticks if t["timebase"] and t["timebase"]["proxy"]]
    fresh = [t for t in ticks if t["gps_fresh"]]
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
        "camera": camera.to_record(),
        "gps": gps.to_record(),
        "adapter": adapter.to_record(),
        "timesync": initiator.to_record(),
        "latency": {
            "jetson_ms": percentiles([int(t["jetson_ms"] * 1e6) for t in ticks]),
            "link_ms": percentiles([int(t["link_ms"] * 1e6) for t in converted]),
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
        # One gate, counted from records rather than derived by subtraction.
        "usable": bool(
            ticks
            and advisories["received"] > 0
            and len(converted) > 0
            and len(fresh) > 0
        ),
        "trace": ticks,
    }


def _tally(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    report = run(args.duration, offset, args.sync_hz)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
