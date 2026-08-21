#!/usr/bin/env python3
"""Phone side of the transport: dial, push synthetic sensor traffic, reconnect.

Stands in for the Android app. Rates and payload sizes match what the four
phone sensors will produce, so the offered load is the real one.

Latency is reported as a ROUND TRIP measured entirely on this machine's clock.
A one-way figure needs the two devices' monotonic clocks related to each other,
which is task 15's job; the protocol says never to compare them directly and
this honours that.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from transport.channels import Channel  # noqa: E402
from transport.client import (  # noqa: E402
    Backoff,
    ClientAttemptFailed,
    ClientSessionEnded,
    ClientSessionStarted,
    SessionClient,
)
from transport.clock import now_mono_ns  # noqa: E402
from transport.handshake import Hello, Role  # noqa: E402
from transport.session import DEFAULT_HEARTBEAT_S, DEFAULT_STALL_TIMEOUT_S  # noqa: E402
from transport.tcp import DEFAULT_PORT  # noqa: E402

SENSOR_PLAN = {
    Channel.CAMERA: {"hz": 10.0, "bytes": 40_960},
    Channel.IMU: {"hz": 50.0, "bytes": 96},
    Channel.GPS: {"hz": 5.0, "bytes": 128},
    Channel.HERE: {"hz": 0.5, "bytes": 8_192},
    Channel.TELEMETRY: {"hz": 1.0, "bytes": 160},
}


def config_record(args) -> dict:
    """Every parameter that shapes the numbers in this report."""
    backoff = Backoff()
    return {
        "heartbeat_s": args.heartbeat,
        "stall_timeout_s": args.stall_timeout,
        "handshake_timeout_s": args.handshake_timeout,
        "backoff": {
            "initial_s": backoff.initial_s,
            "multiplier": backoff.multiplier,
            "cap_s": backoff.cap_s,
            "jitter": backoff.jitter,
        },
        "sensor_plan": {
            channel.value: dict(plan) for channel, plan in SENSOR_PLAN.items()
        },
    }


def maybe_round(value, digits=3):
    """retry_in_s is None whenever no retry will happen, so the report has to
    tolerate that. It did not: round(None) raised and destroyed the entire run
    report -- stdout empty, --out never written -- in exactly the case worth
    recording, a client reconnecting when the clock ran out."""
    return None if value is None else round(value, digits)


def percentiles(values, points=(50, 95, 99)):
    if not values:
        return {f"p{point}": None for point in points}
    ordered = sorted(values)
    return {
        f"p{point}": round(
            ordered[min(len(ordered) - 1, int(round(point / 100 * (len(ordered) - 1))))] / 1e6, 3
        )
        for point in points
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="literal address or resolvable name")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--device-id", default="mac-standing-in-for-phone")
    parser.add_argument("--heartbeat", type=float, default=DEFAULT_HEARTBEAT_S)
    parser.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT_S)
    parser.add_argument(
        "--handshake-timeout",
        type=float,
        default=DEFAULT_STALL_TIMEOUT_S,
        help="the hard cliff on the hello exchange; worth measuring against the real path",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    client = SessionClient(
        args.host,
        args.port,
        local_hello=Hello(device_id=args.device_id, role=Role.PHONE),
        backoff=Backoff(),
        heartbeat_s=args.heartbeat,
        stall_timeout_s=args.stall_timeout,
        handshake_timeout_s=args.handshake_timeout,
    ).start()

    stop = threading.Event()
    probes = itertools.count(1)
    sent_at: dict[int, int] = {}
    lock = threading.Lock()
    round_trips: list[int] = []
    counters = {"advisories": 0, "sends_refused": 0}
    # Five sensor threads increment these, and the report publishes them.
    counter_lock = threading.Lock()

    def sensor(channel, hz, size):
        payload = bytes(size)
        period = 1.0 / hz
        next_at = time.monotonic()
        while not stop.is_set():
            session = client.current_session
            if session is not None and not session.is_closed:
                extensions = {"t_cap": now_mono_ns()}
                if channel is Channel.CAMERA:
                    probe = next(probes)
                    with lock:
                        sent_at[probe] = now_mono_ns()
                    extensions["probe"] = probe
                if not session.send(channel, payload, extensions):
                    with counter_lock:
                        counters["sends_refused"] += 1
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_at = time.monotonic()

    def advisory_reader():
        while not stop.is_set():
            session = client.current_session
            if session is None:
                stop.wait(0.02)
                continue
            message = session.recv(Channel.ADVISORY, timeout=0.25)
            if message is None:
                continue
            with counter_lock:
                counters["advisories"] += 1
            probe = message.extensions.get("echo_probe")
            if probe is None:
                continue
            with lock:
                started = sent_at.pop(probe, None)
            if started is not None:
                round_trips.append(message.t_recv_mono_ns - started)

    # Bounded by the run for long runs, floored at 20s for short ones, and it
    # leaves a machine-readable record either way: a run that never connects
    # still saw attempt failures worth having.
    connect_deadline = max(20.0, min(args.duration, 60.0))
    if client.wait_for_session(timeout=connect_deadline) is None:
        failures = [e for e in client.drain_events() if isinstance(e, ClientAttemptFailed)]
        client.stop()
        report = {
            "host": args.host,
            "port": args.port,
            "duration_s": args.duration,
            "started_wall": started_wall,
            "connected": False,
            "config": config_record(args),
            "waited_s": connect_deadline,
            "failed_attempts": client.failed_attempts,
            "handshake_workers_leaked": client.handshake_workers_leaked,
            "attempt_failures": [
                {"attempt": e.attempt, "error": e.error, "retry_in_s": maybe_round(e.retry_in_s)}
                for e in failures
            ],
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            args.out.write_text(text + "\n")
        return 1

    threads = [threading.Thread(target=advisory_reader, daemon=True)]
    for channel, plan in SENSOR_PLAN.items():
        threads.append(
            threading.Thread(target=sensor, args=(channel, plan["hz"], plan["bytes"]), daemon=True)
        )
    for thread in threads:
        thread.start()

    time.sleep(args.duration)
    stop.set()
    for thread in threads:
        thread.join(timeout=3.0)

    # Drained after stop(), because the last session's end event is generated
    # by the stop itself -- and the last session is the one most likely to have
    # died interestingly.
    live = client.current_session
    applied_options = getattr(live._connection, "applied_options", None) if live else None
    client.stop()
    # Stats read after the stop, so the end reason is the one the session
    # actually ended with rather than None. The reference is kept from before,
    # because current_session correctly stops handing back a closed session.
    stats = live.stats() if live is not None else None
    events = client.drain_events()
    sessions = [event for event in events if isinstance(event, ClientSessionStarted)]
    ended = [event for event in events if isinstance(event, ClientSessionEnded)]
    failures = [event for event in events if isinstance(event, ClientAttemptFailed)]
    report = {
        "host": args.host,
        "port": args.port,
        "duration_s": args.duration,
        "started_wall": started_wall,
        "connected": True,
        # The parameters that produced the numbers. Without them a run cannot
        # answer whether the handshake timeout clears this path, and every
        # retry_in_s is uninterpretable.
        "config": config_record(args),
        "round_trip_ms": percentiles(round_trips),
        "round_trip_samples": len(round_trips),
        "advisories_received": counters["advisories"],
        "sends_refused_while_disconnected": counters["sends_refused"],
        "unmatched_probes_outstanding": len(sent_at),
        "connections": client.connected,
        "reconnects": client.reconnects,
        "failed_attempts": client.failed_attempts,
        "handshake_workers_leaked": client.handshake_workers_leaked,
        "applied_socket_options": applied_options,
        "sessions_started": [
            {
                "session_id": e.session.session_id,
                "attempt": e.attempt,
                # The client is the side that will measure the tailnet round
                # trip, and it was dropping the only copy it holds.
                "handshake_round_trip_ns": e.handshake.clock.round_trip_ns,
                "remote_device_id": e.handshake.remote.device_id,
            }
            for e in sessions
        ],
        "sessions_ended": [
            {
                "session_id": e.session_id,
                "reason": e.reason.value,
                "uptime_s": round(e.uptime_s, 2),
                "retry_in_s": maybe_round(e.retry_in_s),
            }
            for e in ended
        ],
        "attempt_failures": [
            {"attempt": e.attempt, "error": e.error, "retry_in_s": maybe_round(e.retry_in_s)}
            for e in failures
        ],
        "final_session": stats.to_record() if stats is not None else None,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
