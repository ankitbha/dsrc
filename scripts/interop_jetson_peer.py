"""A Jetson-side peer for the Kotlin interop test, using the real implementation.

Golden vectors pin bytes; they cannot pin behaviour. Handshake ordering, keepalive
cadence, stall detection, priority, overflow and drop accounting only interact under a
real peer, and a mock peer would agree with whatever the Kotlin side happens to do. So
this runs the actual `deployment/jetson/transport` Session and reports what it saw.

Protocol with the caller, over stdout, one JSON object per line:

    {"event": "listening", "port": N}   once the socket is bound
    {"event": "ready"}                  once the handshake has completed
    {"event": "summary", ...}           on shutdown, with the counters

The caller connects, exchanges frames, then closes; this prints the summary and exits.
Usage:

    python scripts/interop_jetson_peer.py [--seconds 30] [--expect-channel gps]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from transport.channels import Channel  # noqa: E402
from transport.handshake import Hello, Role, perform_handshake  # noqa: E402
from transport.session import Session  # noqa: E402
from transport.tcp import TcpConnection  # noqa: E402


def emit(payload: dict) -> None:
    # Flushed every time: the caller blocks on these lines, so a buffered stdout would
    # look exactly like a peer that never came up.
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson-side interop peer.")
    parser.add_argument("--seconds", type=float, default=30.0, help="give up after this long")
    parser.add_argument(
        "--echo-advisory",
        action="store_true",
        help="send one advisory per received frame, so the downlink is exercised too",
    )
    args = parser.parse_args(argv)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Port 0: the OS picks a free one and the caller reads it off stdout, so parallel
    # runs cannot collide on a hardcoded port.
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(args.seconds)
    emit({"event": "listening", "port": listener.getsockname()[1]})

    try:
        raw, _ = listener.accept()
    except TimeoutError:
        emit({"event": "summary", "error": "no client connected"})
        return 1
    finally:
        listener.close()

    connection = TcpConnection(raw, peer="interop-phone")
    local = Hello(protocol_version=1, device_id="interop-jetson", role=Role.JETSON)
    try:
        result = perform_handshake(connection, local)
    except Exception as exc:
        emit({"event": "summary", "error": f"handshake failed: {type(exc).__name__}: {exc}"})
        return 1

    received: list[dict] = []
    ended: list[str] = []

    session = Session(
        connection,
        session_id=1,
        on_end=lambda _session, reason: ended.append(reason.value),
    )
    session.start()
    emit(
        {
            "event": "ready",
            "peer_device_id": result.remote.device_id,
            "peer_role": result.remote.role.value,
            "peer_protocol_version": result.remote.protocol_version,
        }
    )

    stop = threading.Event()

    # Channels the phone sends on. recv() is per-channel, so the drain polls each in
    # turn rather than waiting on one and starving the rest.
    inbound_channels = [
        Channel.GPS,
        Channel.IMU,
        Channel.HERE,
        Channel.TELEMETRY,
        Channel.CAMERA,
        Channel.CONTROL,
    ]

    drain_error: list[str] = []

    def drain() -> None:
        try:
            drain_loop()
        except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
            # Without this the thread dies on stderr and the summary reports zero frames
            # received, which is indistinguishable from a peer that sent nothing. The
            # session's own counters showed five arrivals while the summary said none.
            drain_error.append(f"{type(exc).__name__}: {exc}")
            raise

    def drain_loop() -> None:
        while not stop.is_set():
            drained_any = False
            for channel in inbound_channels:
                message = session.recv(channel, timeout=0.0)
                if message is None:
                    continue
                drained_any = True
                received.append(
                    {
                        "channel": channel.value,
                        "seq": message.frame.seq,
                        "payload_len": len(message.payload),
                        "t_mono_ns": message.frame.t_mono_ns,
                        "t_recv_mono_ns": message.t_recv_mono_ns,
                        "t_capture_mono_ns": message.frame.extensions.get("t_capture_mono_ns"),
                    }
                )
                if args.echo_advisory:
                    echo_advisory(session)
            if not drained_any:
                # Nothing waiting anywhere: sleep briefly rather than spinning a core.
                time.sleep(0.01)

    def echo_advisory(session: Session) -> None:
        session.send(
            Channel.ADVISORY,
            extensions={
                "t_capture_mono_ns": time.monotonic_ns(),
                "rec_speed_mps": 13.4,
                "rec_speed_display": 30,
                "current_speed_display": 28,
                "units": "mph",
                "headway_target_s": 2.0,
                "lane_text": "keep",
                "merge_text": "normal",
                "traffic_text": "clear",
                "confidence": 0.87,
                "confidence_label": "high",
                "action": {
                    "desired_speed_bin": "nominal",
                    "desired_headway_bin": "normal",
                    "lane_preference": "keep",
                    "merge_mode": "normal",
                },
            },
        )

    reader = threading.Thread(target=drain, name="interop-drain", daemon=True)
    reader.start()

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and not ended:
        time.sleep(0.05)

    stop.set()
    reader.join(timeout=2.0)
    session.close()

    stats = session.stats()
    # to_record() is the implementation's own shape, so this cannot drift from whatever
    # ChannelStats happens to carry.
    record = stats.to_record()
    channels = {
        name: row
        for name, row in record["channels"].items()
        if any(v for v in row.values() if isinstance(v, int))
    }

    by_channel: dict[str, int] = {}
    sequences: dict[str, list[int]] = {}
    for entry in received:
        by_channel[entry["channel"]] = by_channel.get(entry["channel"], 0) + 1
        sequences.setdefault(entry["channel"], []).append(entry["seq"])

    emit(
        {
            "event": "summary",
            "drain_error": drain_error[0] if drain_error else None,
            "end_reason": ended[0] if ended else "closed_local",
            "frames_received": len(received),
            "by_channel": by_channel,
            "first_seq": {c: s[0] for c, s in sequences.items()},
            "last_seq": {c: s[-1] for c, s in sequences.items()},
            "distinct_seq": {c: len(set(s)) for c, s in sequences.items()},
            "monotonic_seq": {c: s == sorted(s) for c, s in sequences.items()},
            "heartbeats_sent": record["heartbeats_sent"],
            "heartbeats_received": record["heartbeats_received"],
            "channels": channels,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
