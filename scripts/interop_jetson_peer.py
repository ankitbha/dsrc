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
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from transport import frames  # noqa: E402
from transport.channels import Channel  # noqa: E402
from transport.handshake import Hello, Role, perform_handshake  # noqa: E402
from transport.messages import TimeSyncMessage  # noqa: E402
from transport.session import Session  # noqa: E402
from transport.timebase import answer_ping  # noqa: E402
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
    parser.add_argument(
        "--send-malformed",
        action="store_true",
        help="write one frame the phone's decoder must refuse, bypassing our own router so "
        "the phone's drop-and-count is exercised against a real malformed producer",
    )
    parser.add_argument(
        "--send-framing-error",
        action="store_true",
        help="write a frame whose header_len exceeds the cap, which must end the session on "
        "both sides rather than costing one message",
    )
    parser.add_argument(
        "--quiet-drain",
        action="store_true",
        help="do not record every frame; for the thousand-frame run, where the list would "
        "dominate the summary and the counters are the point",
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
    counted: dict[str, list[int]] = {}
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

    # Written directly through the connection, bypassing the session's own validation.
    # A message this side would refuse cannot be produced by session.send at all -- the
    # sender rule sees to that -- so provoking the phone's inbound path needs the raw
    # encoder. That is the point: without it, the phone's drop-and-count has no producer.
    if args.send_malformed:
        malformed = frames.Frame(
            channel=Channel.GPS,
            seq=0,
            t_mono_ns=time.monotonic_ns(),
            t_wall_ns=time.time_ns(),
            payload=b"",
            extensions={
                "t_capture_mono_ns": 1_000,
                # A valid fix must carry a position. This one claims validity with none,
                # which is out_of_range on both sides.
                "valid": True,
                "lat": None,
                "lon": None,
                "speed_mps": None,
                "heading_deg": None,
                "fix_quality": 1,
                "num_sats": 9,
                "hdop": None,
                "altitude_m": None,
                "utc_epoch_ns": None,
            },
        )
        connection.send_all(frames.encode(malformed))
        emit({"event": "sent_malformed"})

    if args.send_framing_error:
        # header_len past MAX_HEADER_BYTES. The spec makes this a framing error that ends
        # the session, unlike a malformed message, because the stream has desynchronised
        # and there is no delimiter to hunt for.
        oversize = (0).to_bytes(4, "big") + (65535).to_bytes(2, "big")
        connection.send_all(oversize + b"{" * 65535)
        emit({"event": "sent_framing_error"})

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
                # Each channel is drained until empty, not once per pass. One message per
                # channel per pass caps the drain at roughly one message per 10 ms sleep,
                # so a thousand-frame run overflowed the session's own inbound queue and
                # shed 621 of them -- a harness limit that reads exactly like the phone
                # sending too fast.
                while not stop.is_set():
                    message = session.recv(channel, timeout=0.0)
                    if message is None:
                        break
                    drained_any = True
                    if args.quiet_drain:
                        # Sequence numbers only. A thousand full records would dominate
                        # the summary line, and the counters are what the comparison reads.
                        counted.setdefault(channel.value, []).append(message.frame.seq)
                    else:
                        received.append(
                            {
                                "channel": channel.value,
                                "seq": message.frame.seq,
                                "payload_len": len(message.payload),
                                "t_mono_ns": message.frame.t_mono_ns,
                                "t_recv_mono_ns": message.t_recv_mono_ns,
                                "t_capture_mono_ns": message.frame.extensions.get(
                                    "t_capture_mono_ns"
                                ),
                            }
                        )
                    if channel is Channel.CONTROL:
                        answer_time_sync(session, message)
                    if args.echo_advisory:
                        echo_advisory(session)
            if not drained_any:
                # Nothing waiting anywhere: sleep briefly rather than spinning a core.
                time.sleep(0.01)

    def answer_time_sync(session: Session, message: object) -> None:
        """The Jetson's half of the timebase exchange.

        The responder existed in `timebase.py` and this peer never called it, so a ping
        from the phone was delivered to a drain loop that logged it and moved on. The phone
        could not tell that from a peer with no responder at all, which is exactly what it
        was.
        """
        ping = TimeSyncMessage.from_wire(message.frame.extensions, message.payload)
        if not ping.is_ping:
            # A pong reaching a responder is the wrong direction; the phone initiates.
            return
        receipt = SimpleNamespace(
            t_recv_mono_ns=message.t_recv_mono_ns,
            t_recv_wall_ns=getattr(message, "t_recv_wall_ns", time.time_ns()),
        )
        pong = answer_ping((ping, receipt))
        extensions, payload = pong.to_wire()
        # to_wire carries t_wire_mono_ns as the placeholder; the writer replaces it with
        # the real departure, which is what makes t3 a departure rather than an enqueue.
        session.send(
            Channel.CONTROL,
            extensions=extensions,
            payload=payload,
            allow_reserved=(frames.WIRE_STAMP_KEY,),
        )

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
    # Every channel, including the all-zero ones. Filtering them out made the summary
    # unusable for a field-by-field comparison: a counter that should have been non-zero
    # and was not simply vanished from the report, so the absence read as agreement.
    channels = record["channels"]

    by_channel: dict[str, int] = {}
    sequences: dict[str, list[int]] = dict(counted)
    for entry in received:
        by_channel[entry["channel"]] = by_channel.get(entry["channel"], 0) + 1
        sequences.setdefault(entry["channel"], []).append(entry["seq"])
    for name, seqs in counted.items():
        by_channel[name] = by_channel.get(name, 0) + len(seqs)

    emit(
        {
            "event": "summary",
            "drain_error": drain_error[0] if drain_error else None,
            "end_reason": ended[0] if ended else "closed_local",
            "frames_received": len(received) + sum(len(v) for v in counted.values()),
            "by_channel": by_channel,
            "first_seq": {c: s[0] for c, s in sequences.items()},
            "last_seq": {c: s[-1] for c, s in sequences.items()},
            "distinct_seq": {c: len(set(s)) for c, s in sequences.items()},
            "monotonic_seq": {c: s == sorted(s) for c, s in sequences.items()},
            "heartbeats_sent": record["heartbeats_sent"],
            "heartbeats_received": record["heartbeats_received"],
            "session_end_reason": record["end_reason"],
            "channels": channels,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
