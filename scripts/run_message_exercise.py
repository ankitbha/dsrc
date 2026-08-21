#!/usr/bin/env python3
"""Task 14 experiment: every message type through the transport at real rates.

Runs the typed layer over loopback at the rates the four phone sensors will
actually use, and reports what the counters see. One process, one monotonic
clock, so the capture-to-enqueue latency is directly measurable -- which is the
one latency this protocol permits without the offset estimator, and the reason
every message carries its own capture stamp.

What this establishes: every message type survives sustained traffic through a
real Session, the per-reason drop counters reconcile against what the transport
delivered, and the null convention holds on the wire under load rather than only
in a unit test.

What it cannot establish: anything about a second device. Cross-device timing
waits for task 15, deliberately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import os

REPO = Path(os.environ.get("DSRC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "deployment" / "jetson"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transport.channels import Channel  # noqa: E402
from transport.clock import now_mono_ns  # noqa: E402
from transport.loopback import loopback_pair  # noqa: E402
from transport.frames import MAX_HEADER_BYTES, encode_header  # noqa: E402
from transport.messages import (  # noqa: E402
    CAPTURE_KEY,
    check_reserved,
    decode_message,
    AdvisoryMessage,
    CameraFrame,
    GpsRecord,
    HereResponse,
    ImuSample,
    MessageRouter,
    PhoneTelemetry,
    RateCommand,
)
from transport.session import Session  # noqa: E402

# The commanded rates from the sensor plan, and the payload each produces.
UPSTREAM_PLAN = {
    Channel.CAMERA: {"hz": 10.0, "bytes": 40_960},
    Channel.IMU: {"hz": 50.0, "bytes": 0},
    Channel.GPS: {"hz": 5.0, "bytes": 0},
    Channel.HERE: {"hz": 0.5, "bytes": 8_192},
    Channel.TELEMETRY: {"hz": 1.0, "bytes": 0},
}


def percentiles(values, points=(50, 95, 99)):
    """Reported as a distribution with its n and both ends. A median alone is
    not a distributional claim, and a real-time budget is spent in the tail."""
    if not values:
        return {"n": 0, **{f"p{p}": None for p in points}}
    ordered = sorted(values)

    def at(point):
        return ordered[min(len(ordered) - 1, int(round(point / 100 * (len(ordered) - 1))))]

    summary = {"n": len(ordered), "min_ms": round(ordered[0] / 1e6, 3)}
    summary.update({f"p{point}": round(at(point) / 1e6, 3) for point in points})
    summary["max_ms"] = round(ordered[-1] / 1e6, 3)
    summary["negative"] = sum(1 for value in ordered if value < 0)
    return summary


def audit(samples: int):
    """Two costs the run itself cannot separate, plus one hard limit.

    The header budget: every channel's encoded header must fit MAX_HEADER_BYTES,
    and the run would only show that as a working session rather than as margin.

    The send-validation cost: send now decodes what it built, holding every
    message to the same conditions its own decoder applies. That is a second
    pass per message on a 50 Hz channel, so the price belongs in the record
    next to the guarantee.
    """
    header_bytes = {}
    encode_ns = {}
    validate_ns = {}
    for channel, plan in list(UPSTREAM_PLAN.items()) + [(Channel.ADVISORY, {"bytes": 0})]:
        widest = 0
        encode_samples = []
        validate_samples = []
        for index in range(1, samples + 1):
            message = (
                an_advisory_for_audit()
                if channel is Channel.ADVISORY
                else build(channel, index, plan["bytes"])
            )
            started = now_mono_ns()
            extensions, payload = message.to_wire()
            encoded = encode_header(extensions)
            after_encode = now_mono_ns()
            check_reserved(extensions)
            decode_message(channel, extensions, payload)
            after_validate = now_mono_ns()
            widest = max(widest, len(encoded))
            encode_samples.append(after_encode - started)
            validate_samples.append(after_validate - after_encode)
        header_bytes[channel.value] = {
            "max_encoded_header_bytes": widest,
            "limit": MAX_HEADER_BYTES,
            "headroom_bytes": MAX_HEADER_BYTES - widest,
            "fraction_of_limit": round(widest / MAX_HEADER_BYTES, 4),
        }
        encode_ns[channel.value] = percentiles(encode_samples)
        validate_ns[channel.value] = percentiles(validate_samples)
    return {
        "samples_per_channel": samples,
        "header_budget": header_bytes,
        "encode_ms": encode_ns,
        "send_validation_ms": validate_ns,
    }


def an_advisory_for_audit():
    return AdvisoryMessage(
        t_capture_mono_ns=now_mono_ns(), rec_speed_mps=11.18, rec_speed_display=25.0,
        current_speed_display=27.5, units="mph", headway_target_s=1.6,
        lane_text="Keep lane", merge_text="Normal driving", traffic_text="Moderate",
        confidence=0.87, confidence_label="high",
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"},
    )


def idle_poll_cost(samples: int):
    """What recv costs when there is nothing there -- the call a control loop
    makes most often, and the one the tightened test bound rests on. Measured
    here under whatever scheduling this machine actually has."""
    phone_conn, jetson_conn = loopback_pair()
    session = Session(jetson_conn, session_id=9, heartbeat_s=None, stall_timeout_s=None).start()
    router = MessageRouter(session)
    costs = []
    try:
        for _ in range(samples):
            started = now_mono_ns()
            router.recv(Channel.IMU, timeout=0.0)
            costs.append(now_mono_ns() - started)
    finally:
        session.close()
        phone_conn.close()
    return percentiles(costs)


def build(channel: Channel, index: int, size: int):
    """A message of the right type, with an unavailable field every tenth one so
    the null convention is exercised under load rather than only in a unit test."""
    now = now_mono_ns()
    absent = index % 10 == 0
    if channel is Channel.CAMERA:
        return CameraFrame(
            t_capture_mono_ns=now, frame_id=index, width=1280, height=720,
            format="jpeg", quality=None if absent else 85, jpeg=bytes(size),
        )
    if channel is Channel.IMU:
        return ImuSample(
            t_capture_mono_ns=now, ax=0.1, ay=-0.2, az=9.79, gx=0.0, gy=0.0, gz=-0.02,
            accuracy=None if absent else 3,
        )
    if channel is Channel.GPS:
        return GpsRecord(
            t_capture_mono_ns=now, valid=not absent, fix_quality=1, num_sats=9,
            lat=None if absent else 51.5074, lon=None if absent else -0.1278,
            speed_mps=None if absent else 13.4, heading_deg=None, hdop=0.9,
            altitude_m=None if absent else 35.0, utc_epoch_ns=None,
        )
    if channel is Channel.HERE:
        return HereResponse(
            t_capture_mono_ns=now, request_url="https://data.traffic.hereapi.com/v7/flow",
            status=200, content_type=None if absent else "application/json",
            query_lat=51.5, query_lon=-0.1278, query_radius_m=1500.0,
            t_request_mono_ns=now - 50_000_000, t_response_mono_ns=now, body=bytes(size),
        )
    return PhoneTelemetry(
        t_capture_mono_ns=now, thermal_status="nominal",
        thermal_headroom=None if absent else 0.42,
        achieved={"camera_hz": 9.9, "gps_hz": 5.0, "imu_hz": 49.8, "here_hz": 0.5},
        dropped={"camera": index, "gps": 0, "imu": 0, "here": 0},
        here_calls=index, here_errors=0,
    )


def run(duration_s: float, advisory_hz: float, malformed_every: int):
    phone_conn, jetson_conn = loopback_pair()
    phone = Session(phone_conn, session_id=1, heartbeat_s=1.0, stall_timeout_s=5.0).start()
    jetson = Session(jetson_conn, session_id=2, heartbeat_s=1.0, stall_timeout_s=5.0).start()
    up = MessageRouter(phone)
    down = MessageRouter(jetson)

    stop = threading.Event()
    latencies = {channel: [] for channel in Channel}
    send_costs = {channel: [] for channel in Channel}
    sent = {channel: 0 for channel in Channel}
    malformed_sent = {channel: 0 for channel in Channel}
    malformed_kinds = {
        channel: {"missing_field": 0, "wrong_type": 0, "wrong_type_capture": 0}
        for channel in Channel
    }

    def sensor(channel, hz, size):
        period = 1.0 / hz
        next_at = time.monotonic()
        index = 0
        while not stop.is_set():
            index += 1
            # Every Nth message is deliberately malformed, so the drop counters
            # and their reasons are exercised by the run rather than asserted.
            if malformed_every and index % malformed_every == 0:
                extensions, payload = build(channel, index, size).to_wire()
                # Rotated through three corruption kinds, so the per-reason
                # breakdown has to distinguish causes rather than just count.
                # A single kind would have shown the counter working and the
                # breakdown doing nothing.
                victim = next(k for k in extensions if k != "t_capture_mono_ns")
                kind = (index // malformed_every) % 3
                if kind == 0:
                    extensions.pop(victim)
                    malformed_kinds[channel]["missing_field"] += 1
                elif kind == 1:
                    extensions[victim] = ["not", "a", "scalar"]
                    malformed_kinds[channel]["wrong_type"] += 1
                else:
                    extensions[CAPTURE_KEY] = 1.5
                    malformed_kinds[channel]["wrong_type_capture"] += 1
                phone.send(channel, payload, extensions)
                malformed_sent[channel] += 1
            else:
                message = build(channel, index, size)
                started = now_mono_ns()
                accepted = up.send(message)
                send_costs[channel].append(now_mono_ns() - started)
                if accepted:
                    sent[channel] += 1
            next_at += period
            delay = next_at - time.monotonic()
            stop.wait(delay) if delay > 0 else None
            if delay <= 0:
                next_at = time.monotonic()

    def jetson_reader():
        while not stop.is_set() or any(jetson.pending(c) for c in Channel):
            idle = True
            for channel in UPSTREAM_PLAN:
                message = down.recv(channel, timeout=0.0)
                if message is None:
                    continue
                idle = False
                latencies[channel].append(now_mono_ns() - message.t_capture_mono_ns)
            if idle:
                stop.wait(0.001)

    def advisor():
        period = 1.0 / advisory_hz
        while not stop.is_set():
            down.send(AdvisoryMessage(
                t_capture_mono_ns=now_mono_ns(), rec_speed_mps=11.18,
                rec_speed_display=25.0, current_speed_display=27.5, units="mph",
                headway_target_s=1.6, lane_text="Keep lane", merge_text="Normal driving",
                traffic_text="Moderate", confidence=0.87, confidence_label="high",
                action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                        "lane_preference": "keep", "merge_mode": "normal"},
            ))
            sent[Channel.ADVISORY] += 1
            stop.wait(period)

    def commander():
        while not stop.is_set():
            stop.wait(0.5)
            if stop.is_set():
                break
            down.send(RateCommand(
                t_capture_mono_ns=now_mono_ns(),
                rates={"camera_hz": 5.0, "gps_hz": 5.0, "imu_hz": 50.0, "here_hz": 0.5},
                trigger="advisory_bin_boundary", shadow=True,
            ))
            sent[Channel.RATE_CMD] += 1

    def phone_reader():
        while not stop.is_set() or any(phone.pending(c) for c in (Channel.ADVISORY, Channel.RATE_CMD)):
            idle = True
            for channel in (Channel.ADVISORY, Channel.RATE_CMD):
                message = up.recv(channel, timeout=0.0)
                if message is not None:
                    idle = False
                    latencies[channel].append(now_mono_ns() - message.t_capture_mono_ns)
            if idle:
                stop.wait(0.001)

    threads = [threading.Thread(target=jetson_reader, daemon=True),
               threading.Thread(target=phone_reader, daemon=True),
               threading.Thread(target=advisor, daemon=True),
               threading.Thread(target=commander, daemon=True)]
    for channel, plan in UPSTREAM_PLAN.items():
        threads.append(threading.Thread(
            target=sensor, args=(channel, plan["hz"], plan["bytes"]), daemon=True))
    for thread in threads:
        thread.start()
    time.sleep(duration_s)
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)

    up_stats, down_stats = up.stats(), down.stats()
    transport_up, transport_down = phone.stats(), jetson.stats()
    phone.close()
    jetson.close()

    report = {
        "duration_s": duration_s,
        "malformed_every": malformed_every,
        "upstream": {},
        "downstream": {},
        "reconciliation": [],
    }
    for channel, plan in UPSTREAM_PLAN.items():
        message_stats = down_stats[channel]
        transport = transport_down.channels[channel]
        report["upstream"][channel.value] = {
            "commanded_hz": plan["hz"],
            "achieved_hz": round(sent[channel] / duration_s, 2),
            "sent_well_formed": sent[channel],
            "sent_malformed": malformed_sent[channel],
            "malformed_by_kind": {k: v for k, v in malformed_kinds[channel].items() if v},
            "transport_delivered": transport.delivered,
            "messages_delivered": message_stats.delivered,
            "decode_errors": message_stats.decode_errors,
            "errors_by_reason": message_stats.errors_by_reason,
            "capture_to_read_ms": percentiles(latencies[channel]),
            "send_ms": percentiles(send_costs[channel]),
            "invalid_sends": message_stats.send_rejected,
        }
        accounted = message_stats.delivered + message_stats.decode_errors
        if accounted != transport.delivered:
            report["reconciliation"].append({
                "channel": channel.value,
                "issue": (f"transport delivered {transport.delivered} but the message layer "
                          f"accounted for {accounted}"),
            })
    for channel in (Channel.ADVISORY, Channel.RATE_CMD):
        message_stats = up_stats[channel]
        report["downstream"][channel.value] = {
            "sent": sent[channel],
            "messages_delivered": message_stats.delivered,
            "decode_errors": message_stats.decode_errors,
            "capture_to_read_ms": percentiles(latencies[channel]),
        }
    report["message_counters"] = down.to_record()

    # Stated, not left to a reader diffing counter tables: on a clean run every
    # per-reason bucket must be empty, and nothing may be attributed to a
    # refusal that did not happen. A report where failure looks like success is
    # the failure mode this project has already paid for once.
    dirty_reasons = {
        channel.value: dict(stats.errors_by_reason)
        for channel, stats in list(down_stats.items()) + list(up_stats.items())
        if stats.errors_by_reason
    }
    invalid_sends = {
        channel.value: dict(stats.rejected_by_reason)
        for channel, stats in list(down_stats.items()) + list(up_stats.items())
        if stats.send_rejected
    }
    report["clean_run"] = {
        "malformed_injected": malformed_every != 0,
        "errors_by_reason_nonempty": dirty_reasons,
        "invalid_sends": invalid_sends,
        "verdict": (
            "clean"
            if malformed_every == 0 and not dirty_reasons and not invalid_sends
            else "corruption injected on purpose"
            if malformed_every
            else "UNEXPECTED: refusals on a run with nothing corrupted"
        ),
    }
    report["reconciled"] = not report["reconciliation"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--advisory-hz", type=float, default=10.0)
    parser.add_argument(
        "--malformed-every",
        type=int,
        default=0,
        help="deliberately corrupt every Nth message, to exercise the drop counters",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--audit-samples", type=int, default=2000)
    parser.add_argument("--skip-run", action="store_true",
                        help="the header/validation audit only, with no traffic")
    args = parser.parse_args()
    report = {"machine": {"platform": sys.platform, "python": sys.version.split()[0]}}
    report["audit"] = audit(args.audit_samples)
    report["idle_poll_ms"] = idle_poll_cost(args.audit_samples)
    if not args.skip_run:
        report.update(run(args.duration, args.advisory_hz, args.malformed_every))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
