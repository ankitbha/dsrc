"""A scripted drive with a phone at the other end, over the real transport.

    python3 scripts/run_phone_drive.py

Everything below the socket is real: a phone-role peer sends camera frames, GPS and
telemetry through the actual codec and session pair; this side runs the real
pipeline, the real sensing controller and the real return path. Mid-drive the phone
hangs up, stays away, and a second one dials in.

The tick loop is PACED, and that is not cosmetic. Run flat out, 240 ticks take 0.4 s
of wall time -- the heartbeat never fires, the vehicle covers 20 m per tick with no
time passing, and the send cadence it reports is an artefact of the harness rather
than a fact about a drive. At 30 Hz and 20 m/s the same code sends on a heartbeat
and almost never on a query move, which is what a real drive does.

**What this shows and what it does not.** It proves the wiring end to end: a decision
reaches the phone, every advisory the phone sees is about a frame this side actually
processed, the cadence rule holds against a channel that can refuse, and a redial
does not end the run. It says nothing about whether the rates are the RIGHT rates --
that is task 35, scoring shadow logs -- and nothing about a real network, which is
task 32. Note also that the run does not attempt a send during the outage: the camera
yields no frame, so the loop never reaches `on_tick`. The no-session path is covered
by unit tests, not by this.
"""
import json, sys, threading, time
sys.path.insert(0, "deployment/jetson")

import numpy as np

from perception.detector import Detection
from perception.distance import DistanceEstimator
from perception.observation_builder import BuilderConfig, ObservationBuilder
from perception.tracker import IouTracker
from pipeline import PerceptionPolicyPipeline
from policy.actor_runtime import ActorRuntime
from policy.advisory import AdvisoryDecoder
from policy.export_policy import build_random, export
from policy.sensing_loop import SensingLoop
from policy.shadow_mode import LIVE, SHADOW, ModeHolder
from sensors.phone_link import PhoneLink
from transport.channels import Channel
from transport.handshake import Hello, Role
from transport.loopback import LoopbackAcceptor
from transport.messages import (CameraFrame, GpsRecord, MessageRouter,
                                PhoneTelemetry, now_mono_ns)
from transport.session import Session

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25
TICKS = 600          # 20 s at 30 Hz
DROP_AT = 300
TICK_HZ = 30.0
SPEED_MPS = 20.0
GAP_S = 0.7          # the phone stays away this long before redialling


class FakeDetector:
    last_timings: dict = {}
    def infer(self, image): return []
    def warmup(self, iterations: int = 1): return 0.0


def project_box(z_m, x_m):
    w = FX * 1.8 / z_m
    u = CX + x_m * FX / z_m
    v = HORIZON + CAM_H * FX / z_m
    return np.array([u - w / 2, v - 0.85 * w, u + w / 2, v], dtype=np.float32)


def scene(t_s):
    lead = max(10.0, 45.0 - 2.0 * t_s)
    return [Detection(xyxy=project_box(lead, 0.0), conf=0.9, cls="car"),
            Detection(xyxy=project_box(28.0, -3.7), conf=0.8, cls="car")]


class Peer:
    """A phone-role client on the loopback acceptor."""
    def __init__(self, acceptor, device_id):
        from transport.handshake import perform_handshake
        self.conn = acceptor.connect(device_id)
        perform_handshake(self.conn, Hello(device_id, Role.PHONE))
        self.session = Session(self.conn, session_id=99, heartbeat_s=None,
                               stall_timeout_s=None).start()
        self.router = MessageRouter(self.session)
        self.advisories, self.commands = [], []
        self.advisory_stamps = []
        self.frame_id = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self):
        while not self._stop.is_set():
            for ch, sink in ((Channel.ADVISORY, self.advisories),
                             (Channel.RATE_CMD, self.commands)):
                # No try/except. `MessageRouter.recv` returns None on an empty queue,
                # on a closed session, and on a message it cannot decode -- which it
                # counts internally instead of raising. A counter fed by an
                # unreachable `except` reports zero on every run including the broken
                # ones, which is worse than not counting at all. The real figure is
                # `refused_by_the_decoder` below, read from the router itself.
                m = self.router.recv(ch, timeout=0.0)
                if m is not None:
                    sink.append(m)
                    if ch == Channel.ADVISORY:
                        self.advisory_stamps.append(m.t_capture_mono_ns)
            time.sleep(0.002)

    def send_tick(self, jpeg, lat, lon, speed):
        self.frame_id += 1
        ok = self.router.send(CameraFrame(
            t_capture_mono_ns=now_mono_ns(), frame_id=self.frame_id, width=16,
            height=16, format="jpeg", quality=85, jpeg=jpeg))
        self.router.send(GpsRecord(
            t_capture_mono_ns=now_mono_ns(), valid=True, fix_quality=1, num_sats=9,
            lat=lat, lon=lon, speed_mps=speed, heading_deg=90.0, hdop=0.9,
            altitude_m=10.0))
        return ok

    def send_telemetry(self, status, skin_c):
        return self.router.send(PhoneTelemetry(
            t_capture_mono_ns=now_mono_ns(), thermal_status=status,
            achieved={"camera_hz": 5.0, "gps_hz": 1.0, "imu_hz": 50.0, "here_hz": 0.05},
            dropped={"camera": 0, "gps": 0, "imu": 0, "here": 0},
            here_calls=1, here_errors=0,
            thermal_headroom=0.5, skin_temp_c=skin_c, skin_temp_zone="xo_therm"))

    def stop(self):
        self._stop.set(); self._t.join(timeout=2.0); self.session.close()


def main():
    import cv2, tempfile, pathlib
    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    jpeg = buf.tobytes()

    tmp = pathlib.Path(tempfile.mkdtemp())
    actor, info = build_random(seed=0)
    export(actor, info, str(tmp / "actor"))

    pipeline = PerceptionPolicyPipeline(
        detector=FakeDetector(), tracker=IouTracker(min_hits=2),
        distance=DistanceEstimator(fx_px=FX, cx_px=CX, horizon_y_px=HORIZON,
                                   camera_height_m=CAM_H, ema_alpha=0.6),
        builder=ObservationBuilder(BuilderConfig()),
        actor=ActorRuntime(str(tmp / "actor")),
        advisory_decoder=AdvisoryDecoder(units="mph"))

    acceptor = LoopbackAcceptor()
    link = PhoneLink(acceptor=acceptor)
    modes = ModeHolder(SHADOW)
    loop = SensingLoop(modes=modes, heartbeat_s=2.0)

    peer_box = {}
    def dial(name):
        peer_box[name] = Peer(acceptor, name)
    threading.Thread(target=dial, args=("phone-one",), daemon=True).start()
    assert link.wait_for_phone(timeout_s=10.0), "phone-one never dialled in"
    for _ in range(200):
        if "phone-one" in peer_box: break
        time.sleep(0.01)
    peer = peer_box["phone-one"]

    ticks, blind, sent_stamps = 0, 0, []
    ended_early = False
    lat, lon = 51.49, -0.20
    per_tick = 1.0 / TICK_HZ
    per_tick_m = SPEED_MPS / TICK_HZ
    gap_until = None
    t0 = time.monotonic()
    for i in range(TICKS):
        deadline = t0 + (i + 1) * per_tick
        if i == DROP_AT:
            # The phone goes away and stays away, so the gap is real and the run has
            # to survive ticks with no camera at all -- which is the whole claim.
            peer.stop()
            gap_until = time.monotonic() + GAP_S
        if gap_until is not None and time.monotonic() >= gap_until:
            gap_until = None
            threading.Thread(target=dial, args=("phone-two",), daemon=True).start()
            for _ in range(500):
                if len(link.rebinds) == 1 and "phone-two" in peer_box: break
                time.sleep(0.01)
            peer = peer_box.get("phone-two", peer)
        if i == 150:
            modes.flip_to(LIVE)
        if i == 450:
            peer.send_telemetry("severe", 46.0)
        elif i % 60 == 0:
            peer.send_telemetry("nominal", 32.0)

        lat += per_tick_m / 111_320.0
        if peer.session.is_closed is False:
            peer.send_tick(jpeg, lat, lon, SPEED_MPS)
        frame = link.camera.wait_for_fresh(timeout=0.05)
        if link.camera.end_of_stream:
            ended_early = True
            break
        if frame is None:
            blind += 1
            time.sleep(max(0.0, deadline - time.monotonic()))
            continue
        fix = link.gps.latest()
        feed = link.here.at(fix, time.monotonic())
        tick = pipeline.step(frame, fix, detections_override=scene(i / TICK_HZ),
                             feed=feed)
        outcome = loop.on_tick(tick, link)
        ticks += 1
        if outcome.advisory_sent:
            sent_stamps.append(int(tick.t_capture_mono * 1e9))
        time.sleep(max(0.0, deadline - time.monotonic()))

    time.sleep(0.3)
    elapsed = time.monotonic() - t0
    record = link.to_record()
    result = {
        "ticks_processed": ticks,
        "ticks_with_no_frame": blind,
        "wall_s": round(elapsed, 1),
        "survived_the_redial": len(record["rebinds"]) == 1,
        "rebind_down_s": None if not record["rebinds"] else round(record["rebinds"][0]["down_s"], 3),
        "run_ended_before_the_ticks_did": ended_early,
        "advisories": {
            "sent_by_jetson": record["sent"]["advisories"],
            # `advisory` is latest_wins at depth ONE, so the phone is meant to see
            # the newest and not all of them: a count below the sent count is the
            # channel working, not loss. What matters is that every advisory the
            # phone DID see is about a frame the Jetson actually processed.
            "seen_by_the_phone": len(peer.advisory_stamps),
            "matched_a_real_frame": sum(1 for s in peer.advisory_stamps
                                        if s in set(sent_stamps)),
            "unmatched": sum(1 for s in peer.advisory_stamps
                             if s not in set(sent_stamps)),
        },
        "rate_commands": {
            "sent": record["sent"]["rate_commands"],
            "per_tick_would_have_been": ticks,
            "ratio": None if not ticks else round(record["sent"]["rate_commands"] / ticks, 4),
            "by_reason": loop.sends_by_reason,
        },
        # From the router's own per-channel counters, which can be nonzero. The
        # figure this replaces came from an `except` around a call that does not
        # raise, so it was a constant dressed as a measurement.
        "refused_by_the_decoder": sum(
            v.get("decode_errors", 0) for v in peer.router.to_record().values()
        ),
        "sends_without_a_session": record["sent"]["without_a_session"],
        "sends_refused": record["sent"]["refused"],
        "sessions_recorded": len(record["sessions"]),
        "gap": {
            "requested_s": GAP_S,
            "ticks_with_no_camera": blind,
            "measured_down_s": None if not record["rebinds"] else round(record["rebinds"][0]["down_s"], 3),
        },
        "flips": loop.to_record()["mode"]["flips"],
        "commands_phone_two_saw_shadow_flags": sorted({c.shadow for c in peer.commands}),
    }
    peer.stop(); link.stop()
    print(json.dumps(result, indent=2))


main()
