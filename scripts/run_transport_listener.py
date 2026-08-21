#!/usr/bin/env python3
"""Jetson side of the transport: listen, accept, answer camera frames.

Stands in for the section F runtime. It perceives and infers nothing -- it
answers each camera frame with a synthetic advisory so the loop closes and the
far end can time a round trip on one clock.

The advisory carries the phone's own probe id back, because a one-way number
would need the two devices' monotonic clocks related to each other, and that is
task 15's job. This deliberately does not attempt it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from transport.channels import Channel  # noqa: E402
from transport.clock import now_mono_ns  # noqa: E402
from transport.endpoint import (  # noqa: E402
    SessionEnded,
    SessionRefused,
    SessionStarted,
    TransportListener,
)
from transport.handshake import Hello, Role  # noqa: E402
from transport.session import DEFAULT_HEARTBEAT_S, DEFAULT_STALL_TIMEOUT_S  # noqa: E402
from transport.tcp import DEFAULT_PORT, TcpAcceptor  # noqa: E402

DRAIN_BATCH = 128


def respond(session, stop: threading.Event) -> None:
    """Answer each camera frame at once, and drain the other channels.

    The recv is blocking on purpose. Polling non-blocking from the event loop
    below would add that loop's interval to every measurement, and the number
    would then describe this script rather than the transport.
    """
    while not stop.is_set() and not session.is_closed:
        message = session.recv(Channel.CAMERA, timeout=0.25)
        if message is not None:
            session.send(
                Channel.ADVISORY,
                b"cap=25.0",
                {
                    "echo_seq": message.seq,
                    "echo_probe": message.extensions.get("probe"),
                    "t_jetson_recv_mono_ns": message.t_recv_mono_ns,
                    "t_jetson_send_mono_ns": now_mono_ns(),
                },
            )
        for channel in (Channel.GPS, Channel.IMU, Channel.HERE, Channel.TELEMETRY):
            for _ in range(DRAIN_BATCH):
                if session.recv(channel, timeout=0.0) is None:
                    break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--device-id", default="jetson-orin")
    parser.add_argument("--heartbeat", type=float, default=DEFAULT_HEARTBEAT_S)
    parser.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT_S)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    acceptor = TcpAcceptor(args.host, args.port)
    listener = TransportListener(
        acceptor,
        Hello(device_id=args.device_id, role=Role.JETSON),
        heartbeat_s=args.heartbeat,
        stall_timeout_s=args.stall_timeout,
        accept_poll_s=0.2,
    ).start()

    log: dict = {"bound": list(acceptor.address), "events": [], "sessions": []}
    stop = threading.Event()
    responders: list[threading.Thread] = []
    print(f"listening on {acceptor.host}:{acceptor.port} for {args.duration}s", flush=True)

    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        event = listener.next_event(timeout=0.1)
        if isinstance(event, SessionStarted):
            remote = event.handshake.remote
            print(f"session {event.session.session_id} from {remote.device_id}", flush=True)
            log["events"].append(
                {
                    "kind": "started",
                    "session_id": event.session.session_id,
                    "device_id": remote.device_id,
                    "role": remote.role.value,
                    "applied_socket_options": getattr(
                        event.session._connection, "applied_options", None
                    ),
                    "handshake_round_trip_ns": event.handshake.clock.round_trip_ns,
                    "remote_mono_ns": event.handshake.clock.t_remote_mono_ns,
                    "remote_wall_ns": event.handshake.clock.t_remote_wall_ns,
                }
            )
            thread = threading.Thread(
                target=respond,
                args=(event.session, stop),
                daemon=True,
                name=f"respond-{event.session.session_id}",
            )
            thread.start()
            responders.append(thread)
        elif isinstance(event, SessionEnded):
            print(f"session {event.session_id} ended: {event.reason.value}", flush=True)
            log["events"].append(
                {"kind": "ended", "session_id": event.session_id, "reason": event.reason.value}
            )
            log["sessions"].append(event.stats.to_record())
        elif isinstance(event, SessionRefused):
            print(f"refused {event.peer}: {event.error}", flush=True)
            log["events"].append({"kind": "refused", "peer": event.peer, "error": event.error})

    live = listener.current_session
    if live is not None:
        log["sessions"].append(live.stats().to_record())
    stop.set()
    for thread in responders:
        thread.join(timeout=2.0)
    listener.stop()

    log["accepted"] = listener.accepted
    log["refused"] = listener.refused
    log["displaced"] = listener.displaced
    log["handshake_workers_leaked"] = listener.handshake_workers_leaked
    # The acceptor's own record. Without it, a listener that is alive and
    # accepting nothing produces a report indistinguishable from a clean run.
    log["acceptor"] = acceptor.stats()
    text = json.dumps(log, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", flush=True)
    acceptor_stats = acceptor.stats()
    print(
        json.dumps(
            {
                "accepted": listener.accepted,
                "refused": listener.refused,
                "displaced": listener.displaced,
                "handshake_workers_leaked": listener.handshake_workers_leaked,
                "transient_accept_errors": acceptor_stats["transient_accept_errors"],
                "accept_errors_by_errno": acceptor_stats["accept_errors_by_errno"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
