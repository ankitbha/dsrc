#!/usr/bin/env python3
"""Task 14 on hardware: typed messages both ways over the real Tailscale link.

The loopback exercise shows the message layer works; it cannot show it works
over the transport it was built for. This runs the typed layer across the two
real devices, phone-initiating as the binding constraint requires, so the frames
cross a real MTU, a real TCP stack, and a real network rather than an in-process
pipe.

Latency is a ROUND TRIP on the phone's clock alone. Relating the two devices'
monotonic clocks is task 15's job and the protocol forbids comparing them
directly, so the Jetson answers each camera frame with a typed advisory that
carries the frame id back and nothing is subtracted across machines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(os.environ.get("DSRC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "deployment" / "jetson"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transport.channels import Channel  # noqa: E402
from transport.client import Backoff, SessionClient  # noqa: E402
from transport.clock import now_mono_ns  # noqa: E402
from transport.endpoint import SessionStarted, TransportListener  # noqa: E402
from transport.handshake import Hello, Role  # noqa: E402
from transport.messages import (  # noqa: E402
    AdvisoryMessage,
    InvalidMessage,
    MessageRouter,
)
from transport.tcp import DEFAULT_PORT, TcpAcceptor  # noqa: E402

from run_message_exercise import UPSTREAM_PLAN, build, percentiles  # noqa: E402

ECHO_PREFIX = "echo:"


def echo_advisory(frame_id: int) -> AdvisoryMessage:
    """A schema-legal advisory carrying the frame id in its lane text.

    The id has to ride a field that already exists -- adding one for the
    experiment would mean measuring a protocol nobody is shipping -- and
    lane_text is free-form display text, so this borrows it rather than
    inventing a field. Everything else is a realistic advisory.
    """
    return AdvisoryMessage(
        t_capture_mono_ns=now_mono_ns(), rec_speed_mps=11.18, rec_speed_display=25.0,
        current_speed_display=27.5, units="mph", headway_target_s=1.6,
        lane_text=f"{ECHO_PREFIX}{frame_id}", merge_text="Normal driving",
        traffic_text="Moderate", confidence=0.87, confidence_label="high",
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"},
    )


def run_jetson(args) -> dict:
    acceptor = TcpAcceptor(args.host, args.port)
    listener = TransportListener(
        acceptor, Hello(device_id=args.device_id, role=Role.JETSON),
        heartbeat_s=1.0, stall_timeout_s=10.0, handshake_timeout_s=10.0,
        accept_poll_s=0.2,
    ).start()
    print(f"listening on {acceptor.host}:{acceptor.port} for {args.duration}s", flush=True)

    stop = threading.Event()
    routers: list[MessageRouter] = []
    echoed = {"count": 0, "invalid": 0}

    def serve(session):
        router = MessageRouter(session)
        routers.append(router)
        while not stop.is_set() and not session.is_closed:
            frame = router.recv(Channel.CAMERA, timeout=0.25)
            if frame is not None:
                try:
                    if router.send(echo_advisory(frame.frame_id)):
                        echoed["count"] += 1
                except InvalidMessage:
                    echoed["invalid"] += 1
            for channel in (Channel.GPS, Channel.IMU, Channel.HERE, Channel.TELEMETRY):
                for _ in range(64):
                    if router.recv(channel, timeout=0.0) is None:
                        break

    workers: list[threading.Thread] = []
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        event = listener.next_event(timeout=0.1)
        if isinstance(event, SessionStarted):
            remote = event.handshake.remote
            print(f"session {event.session.session_id} from {remote.device_id}", flush=True)
            worker = threading.Thread(target=serve, args=(event.session,), daemon=True)
            worker.start()
            workers.append(worker)
    stop.set()
    for worker in workers:
        worker.join(timeout=5.0)

    # Counted from the routers' own records, never by subtraction: a run where
    # every session died has to read as a failure rather than as zero errors.
    report = {
        "role": "jetson",
        "sessions_served": len(routers),
        "echoes_sent": echoed["count"],
        "echoes_refused_as_invalid": echoed["invalid"],
        "accepted": listener.accepted,
        "refused": listener.refused,
        "displaced": listener.displaced,
        "by_channel": {},
    }
    for router in routers:
        for channel, stats in router.stats().items():
            if not (stats.delivered or stats.decode_errors or stats.send_rejected):
                continue
            report["by_channel"].setdefault(channel.value, {
                "delivered": 0, "decode_errors": 0, "errors_by_reason": {},
                "send_rejected": 0,
            })
            slot = report["by_channel"][channel.value]
            slot["delivered"] += stats.delivered
            slot["decode_errors"] += stats.decode_errors
            slot["send_rejected"] += stats.send_rejected
            for reason, count in stats.errors_by_reason.items():
                slot["errors_by_reason"][reason] = slot["errors_by_reason"].get(reason, 0) + count
    listener.stop()
    report["usable"] = bool(routers) and report["echoes_sent"] > 0
    return report


def tailnet_path(host: str) -> dict:
    """Whether this run went direct or through a relay, recorded at run time.

    An earlier run offered 40% of its commanded cadence and the cause could not
    be attributed afterwards, because nothing recorded which path the link took.
    A relayed link and a LAN link are different experiments, and the report has
    to say which one it was.
    """
    import shutil
    import subprocess

    binary = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    try:
        listing = subprocess.run([binary, "status"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"recorded": False, "why": str(exc)}
    for line in listing.stdout.splitlines():
        if host in line:
            tail = line.split(host, 1)[1]
            return {
                "recorded": True,
                "relayed": "relay" in tail,
                "direct": "direct" in tail,
                "detail": " ".join(tail.split()[-4:]),
            }
    return {"recorded": False, "why": f"{host} not in tailscale status"}


def run_phone(args) -> dict:
    client = SessionClient(
        args.host, args.port,
        local_hello=Hello(device_id=args.device_id, role=Role.PHONE),
        backoff=Backoff(), heartbeat_s=1.0, stall_timeout_s=10.0,
        handshake_timeout_s=10.0,
    ).start()

    stop = threading.Event()
    lock = threading.Lock()
    sent_at: dict[int, int] = {}
    round_trips: list[int] = []
    sent = {channel: 0 for channel in Channel}
    refused = {channel: 0 for channel in Channel}
    invalid = {channel: 0 for channel in Channel}
    unmatched = {"count": 0}
    send_costs = {channel: [] for channel in Channel}
    overshoot = {channel: [] for channel in Channel}
    routers: dict[int, MessageRouter] = {}

    def router_for(session):
        with lock:
            if session.session_id not in routers:
                routers[session.session_id] = MessageRouter(session)
            return routers[session.session_id]

    def sensor(channel, hz, size):
        period = 1.0 / hz
        index = 0
        next_at = time.monotonic()
        while not stop.is_set():
            session = client.current_session
            if session is not None and not session.is_closed:
                index += 1
                message = build(channel, index, size)
                if channel is Channel.CAMERA:
                    with lock:
                        sent_at[index] = now_mono_ns()
                started = now_mono_ns()
                try:
                    if router_for(session).send(message):
                        sent[channel] += 1
                    else:
                        refused[channel] += 1
                except InvalidMessage:
                    invalid[channel] += 1
                send_costs[channel].append(now_mono_ns() - started)
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                # Behind cadence. Recorded rather than silently absorbed: the
                # reset hides exactly the shortfall that makes an achieved rate
                # miss its commanded one, and a rate table alone cannot say
                # whether the cost was in send or in the scheduler.
                overshoot[channel].append(int(-delay * 1e9))
                next_at = time.monotonic()

    def reader():
        while not stop.is_set():
            session = client.current_session
            if session is None or session.is_closed:
                stop.wait(0.05)
                continue
            advisory = router_for(session).recv(Channel.ADVISORY, timeout=0.05)
            if advisory is None:
                continue
            if not advisory.lane_text.startswith(ECHO_PREFIX):
                unmatched["count"] += 1
                continue
            frame_id = int(advisory.lane_text[len(ECHO_PREFIX):])
            with lock:
                started = sent_at.pop(frame_id, None)
            if started is None:
                unmatched["count"] += 1
            else:
                # One clock, this machine's, from send to matched reply.
                round_trips.append(now_mono_ns() - started)

    threads = [threading.Thread(target=reader, daemon=True)]
    for channel, plan in UPSTREAM_PLAN.items():
        threads.append(threading.Thread(
            target=sensor, args=(channel, plan["hz"], plan["bytes"]), daemon=True))
    for thread in threads:
        thread.start()
    time.sleep(args.duration)
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)

    report = {
        "role": "phone",
        "host": args.host,
        "link_path": tailnet_path(args.host),
        "duration_s": args.duration,
        "sessions": sorted(routers),
        "round_trip_ms": percentiles(round_trips),
        "echoes_unmatched": unmatched["count"],
        "camera_frames_awaiting_reply_at_stop": len(sent_at),
        "upstream": {},
        "advisory_decode_errors": 0,
        "advisory_errors_by_reason": {},
    }
    for channel, plan in UPSTREAM_PLAN.items():
        report["upstream"][channel.value] = {
            "commanded_hz": plan["hz"],
            "achieved_hz": round(sent[channel] / args.duration, 2),
            "sent": sent[channel],
            "refused_by_overflow": refused[channel],
            "invalid_sends": invalid[channel],
            "send_ms": percentiles(send_costs[channel]),
            "behind_cadence_ms": percentiles(overshoot[channel]),
            "iterations_behind_cadence": len(overshoot[channel]),
        }
    for router in routers.values():
        stats = router.stats()[Channel.ADVISORY]
        report["advisory_decode_errors"] += stats.decode_errors
        for reason, count in stats.errors_by_reason.items():
            report["advisory_errors_by_reason"][reason] = (
                report["advisory_errors_by_reason"].get(reason, 0) + count
            )
        report.setdefault("advisories_delivered", 0)
        report["advisories_delivered"] += stats.delivered
    client.stop()
    # One gate, stated here rather than left to a reader: a run that connected,
    # sent, and got typed replies back is usable; anything less is not.
    report["usable"] = bool(
        routers
        and report["round_trip_ms"]["n"] > 0
        and sum(sent.values()) > 0
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("phone", "jetson"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.device_id = args.device_id or f"{args.role}-link-exercise"
    report = run_jetson(args) if args.role == "jetson" else run_phone(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if report.get("usable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
