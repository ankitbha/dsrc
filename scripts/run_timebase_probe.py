#!/usr/bin/env python3
"""Task 15 experiment: the shared timebase over the real link, both roles.

Runs the time-sync exchange against the sensor load it will actually compete
with -- a 10 Hz camera stream at 40 KB a frame is what makes the difference
between an enqueue stamp and a departure stamp worth having -- and records what
the estimator concludes over a drive-length run.

`ASSUMED_SKEW_PPM` is the sole bound on the true relative skew of the two
monotonic clocks, so the whole guarantee rests on it. Measuring that quantity is
this experiment's second job, and the first way of doing it does not work.

**The cross-device wall pair is not an instrument for it.** Running the same
midpoint arithmetic on the two wall clocks and differencing the slopes cancels
the quantity of interest. With each device's own wall-versus-monotonic slew
written `s`, and the true monotonic skew `sigma`:

    mono-pair slope  =  sigma
    wall-pair slope  =  sigma + (s_remote - s_local)
    difference       =  s_local - s_remote          <- sigma is gone

So it reports the difference of the two NTP slew rates and says nothing about the
skew. Measured on this pair, it returned +10.4 ppm and +75.8 ppm on two runs the
same evening -- the first being roughly the slew difference and the second mostly
noise from the delay asymmetry in the midpoint.

**What does work needs no network at all.** Each device measures its own
`wall - monotonic` slew locally, which is exact and needs no delay model; both
wall clocks are NTP-locked to UTC, so each local slew states how far that
device's monotonic clock runs from UTC, and the difference of the two is the
skew. Both roles record it here and the analysis differences them.

Nothing here compares a phone monotonic value with a Jetson one directly. The
exchange offsets are differences within one clock on each side, and the slews are
each within one device.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(os.environ.get("DSRC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "deployment" / "jetson"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transport.channels import Channel  # noqa: E402
from transport.frames import WIRE_STAMP_KEY  # noqa: E402
from transport.client import Backoff, SessionClient  # noqa: E402
from transport.clock import now_mono_ns, now_wall_ns  # noqa: E402
from transport.endpoint import SessionStarted, TransportListener  # noqa: E402
from transport.handshake import Hello, Role  # noqa: E402
from transport.messages import InvalidMessage, MessageRouter  # noqa: E402
from transport.tcp import DEFAULT_PORT, TcpAcceptor  # noqa: E402
from transport.timebase import (  # noqa: E402
    ASSUMED_SKEW_PPM,
    NS_PER_S,
    TimebaseNotReady,
    TimeSyncInitiator,
    answer_ping,
)

from run_message_exercise import UPSTREAM_PLAN, build, percentiles  # noqa: E402
from run_message_link import tailnet_path  # noqa: E402


def local_slew_ppm(duration_s: float, stop: threading.Event) -> dict:
    """This device's own wall-versus-monotonic slew, sampled while the run goes on.

    No network, no delay model, no cross-device arithmetic -- so unlike the wall
    pair this is not contaminated by asymmetry, and it is the term that actually
    carries the skew. Reported with its halves so a reader can see whether it was
    a steady slew or an NTP correction event.
    """
    rows: list[tuple[int, int]] = []
    start_m, start_w = now_mono_ns(), now_wall_ns()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not stop.is_set():
        mono, wall = now_mono_ns(), now_wall_ns()
        rows.append((mono - start_m, (wall - start_w) - (mono - start_m)))
        stop.wait(2.0)
    if len(rows) < 6:
        return {"measured": False, "samples": len(rows)}
    half = len(rows) // 2
    steps = [
        round((b[1] - a[1]) / 1e6, 3)
        for a, b in zip(rows, rows[1:]) if abs(b[1] - a[1]) > 1_000_000
    ]
    return {
        "measured": True,
        "samples": len(rows),
        "span_s": round(rows[-1][0] / 1e9, 1),
        "slew_ppm": slope_ppm(rows),
        "slew_ppm_first_half": slope_ppm(rows[:half]),
        "slew_ppm_second_half": slope_ppm(rows[half:]),
        "total_drift_ms": round(rows[-1][1] / 1e6, 3),
        "steps_over_1ms": steps[:8],
        "step_count": len(steps),
    }


def slope_ppm(points: list[tuple[int, int]]) -> float | None:
    """Least squares of offset against local time, in ppm.

    Centred on the mean for the same reason the estimator's fit is: raw
    monotonic nanoseconds are ~1e15 and squaring them loses the variation.
    """
    if len(points) < 3:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return (sxy / sxx) * 1e6


def run_jetson(args) -> dict:
    acceptor = TcpAcceptor(args.host, args.port)
    listener = TransportListener(
        acceptor, Hello(device_id=args.device_id, role=Role.JETSON),
        heartbeat_s=1.0, stall_timeout_s=10.0, handshake_timeout_s=10.0,
        accept_poll_s=0.2,
    ).start()
    print(f"listening on {acceptor.host}:{acceptor.port} for {args.duration}s", flush=True)

    stop = threading.Event()
    served = {"pings": 0, "invalid": 0, "wrong_direction": 0}
    # What write-time stamping actually bought, measured where it is visible: the
    # sender's enqueue stamp and its departure stamp both ride the frame, so the
    # gap between them is exactly the queueing delay t1 would otherwise have
    # carried. Only the receiver can see it, which is why it is measured here.
    enqueue_to_wire: list[int] = []
    routers: list[MessageRouter] = []
    slew_box: dict[str, dict] = {}

    def measure_slew():
        slew_box["local"] = local_slew_ppm(args.duration, stop)

    slew_thread = threading.Thread(target=measure_slew, daemon=True)
    slew_thread.start()

    def serve(session):
        router = MessageRouter(session)
        routers.append(router)
        while not stop.is_set() and not session.is_closed:
            # recv_with_receipt, not recv: t2 must be the stamp this session's
            # reader took, because t3 comes from this session's writer clock and
            # the initiator subtracts one from the other.
            arrived = router.recv_with_receipt(Channel.CONTROL, timeout=0.05)
            if arrived is not None:
                frame = arrived[1].frame
                departed = frame.extensions.get(WIRE_STAMP_KEY)
                if departed:
                    enqueue_to_wire.append(departed - frame.t_mono_ns)
                try:
                    if router.send(answer_ping(arrived)):
                        served["pings"] += 1
                except InvalidMessage:
                    served["invalid"] += 1
                except Exception:  # a pong reaching the responder
                    served["wrong_direction"] += 1
            for channel in (Channel.CAMERA, Channel.GPS, Channel.IMU,
                            Channel.HERE, Channel.TELEMETRY):
                for _ in range(64):
                    if router.recv(channel, timeout=0.0) is None:
                        break

    workers: list[threading.Thread] = []
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        event = listener.next_event(timeout=0.1)
        if isinstance(event, SessionStarted):
            print(f"session {event.session.session_id} from "
                  f"{event.handshake.remote.device_id}", flush=True)
            worker = threading.Thread(target=serve, args=(event.session,), daemon=True)
            worker.start()
            workers.append(worker)
    stop.set()
    for worker in workers:
        worker.join(timeout=5.0)

    slew_thread.join(timeout=10.0)
    report = {
        "role": "jetson",
        "local_slew": slew_box.get("local", {"measured": False}),
        "peer_enqueue_to_wire_ms": percentiles(enqueue_to_wire),
        "sessions_served": len(routers),
        "pings_answered": served["pings"],
        "answers_refused_as_invalid": served["invalid"],
        "pongs_received_in_error": served["wrong_direction"],
        "accepted": listener.accepted,
        "by_channel": {},
    }
    for router in routers:
        for channel, stats in router.stats().items():
            if not (stats.delivered or stats.decode_errors or stats.send_rejected):
                continue
            report["by_channel"][channel.value] = {
                "delivered": stats.delivered,
                "decode_errors": stats.decode_errors,
                "errors_by_reason": dict(stats.errors_by_reason),
                "send_rejected": stats.send_rejected,
            }
    listener.stop()
    report["usable"] = bool(routers) and served["pings"] > 0
    return report


def run_phone(args) -> dict:
    client = SessionClient(
        args.host, args.port,
        local_hello=Hello(device_id=args.device_id, role=Role.PHONE),
        backoff=Backoff(), heartbeat_s=1.0, stall_timeout_s=10.0,
        handshake_timeout_s=10.0,
    ).start()

    stop = threading.Event()
    routers: dict[int, MessageRouter] = {}
    lock = threading.Lock()
    initiator_box: dict[str, TimeSyncInitiator] = {}
    # (local mono instant, offset) for each estimator, on both clock pairs.
    mono_track: list[tuple[int, int]] = []
    wall_track: list[tuple[int, int]] = []
    samples_seen = 0
    send_to_arrival: list[int] = []
    trace: list[dict] = []
    sensor_sent = {channel: 0 for channel in Channel}

    def router_for(session):
        with lock:
            if session.session_id not in routers:
                routers[session.session_id] = MessageRouter(session)
                initiator_box["current"] = TimeSyncInitiator(routers[session.session_id])
            return routers[session.session_id], initiator_box["current"]

    def sensor(channel, hz, size):
        period = 1.0 / hz
        index = 0
        next_at = time.monotonic()
        while not stop.is_set():
            session = client.current_session
            if session is not None and not session.is_closed:
                index += 1
                router, _ = router_for(session)
                try:
                    if router.send(build(channel, index, size)):
                        sensor_sent[channel] += 1
                except InvalidMessage:
                    pass
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_at = time.monotonic()

    def syncer():
        """Ping on the shipped cadence and match each pong to its own exchange.

        Two things the first version got wrong. It looped as fast as the link
        would answer -- 300 exchanges a second against a designed 4 Hz then 1 Hz
        -- so it measured a system nobody is shipping and put 300x the intended
        load on the control channel. And it paired the wall stamp it took before
        one ping with whatever pong `pump` returned, which is the oldest queued
        one and can belong to an earlier exchange; on loopback that mismatch
        alone manufactured a 51 ppm slope out of a link with no skew at all.

        Both stamps are now keyed by exchange id, so a wall pair is always the
        two ends of one round trip.
        """
        sent_wall_by_id: dict[int, int] = {}
        while not stop.is_set():
            session = client.current_session
            if session is None or session.is_closed:
                stop.wait(0.05)
                continue
            _, initiator = router_for(session)
            period = initiator.period_s
            sent_wall_by_id[initiator.next_exchange_id()] = now_wall_ns()
            initiator.send_ping()

            # Drain whatever has arrived, within this period.
            deadline = time.monotonic() + period
            while time.monotonic() < deadline and not stop.is_set():
                sample = initiator.pump(timeout=0.01)
                if sample is None:
                    continue
                nonlocal samples_seen
                samples_seen += 1
                mono_track.append((sample.t_local_mid_ns, sample.offset_ns))
                sent_wall = sent_wall_by_id.pop(sample.exchange_id, None)
                if (
                    sent_wall is not None
                    and sample.t2_remote_recv_wall_ns is not None
                    and sample.t4_local_recv_wall_ns is not None
                ):
                    # The same midpoint arithmetic on the two wall clocks, for
                    # this exchange and no other, with both arrival stamps taken
                    # by the readers rather than at handling time.
                    wall_track.append((
                        sample.t_local_mid_ns,
                        sample.t2_remote_recv_wall_ns
                        - (sent_wall + sample.t4_local_recv_wall_ns) // 2,
                    ))
                    # The one residual asymmetry left in the pair: the local send
                    # stamp is taken before the enqueue while both arrival stamps
                    # come from the readers. A constant one cancels in the slope;
                    # a drifting one is what would corrupt the number, so it is
                    # bounded here rather than left unattributable.
                    send_to_arrival.append(
                        sample.t4_local_recv_wall_ns - sent_wall
                    )
            # Anything unanswered for far longer than the cadence is not coming.
            horizon = now_wall_ns() - int(60 * NS_PER_S)
            for key in [k for k, at in list(sent_wall_by_id.items()) if at < horizon]:
                sent_wall_by_id.pop(key, None)

    def sampler():
        while not stop.is_set():
            stop.wait(args.trace_every)
            initiator = initiator_box.get("current")
            if initiator is None:
                continue
            estimate = initiator.estimator.estimate
            row = {
                "t_mono_ns": now_mono_ns(),
                "usable": initiator.estimator.usable,
                "why_not_usable": initiator.estimator.why_not_usable(),
                "samples": 0 if estimate is None else estimate.offset_samples,
            }
            if estimate is not None:
                row.update({
                    "offset_ns": estimate.offset_ns,
                    "rtt_min_ns": estimate.rtt_min_ns,
                    "skew_ppm": estimate.skew_ppm,
                    "skew_stderr_ppm": estimate.skew_stderr_ppm,
                    "skew_uncertainty_ppm": estimate.skew_uncertainty_ppm,
                    "bound_ns": estimate.bound_ns_at(row["t_mono_ns"]),
                })
            try:
                converted = initiator.estimator.to_remote(now_mono_ns())
                row["converted_bound_ns"] = converted.bound_ns
                row["converted_estimate_id"] = converted.estimate_id
            except TimebaseNotReady as exc:
                row["conversion_refused"] = str(exc)
            trace.append(row)

    slew_box: dict[str, dict] = {}

    def measure_slew():
        slew_box["local"] = local_slew_ppm(args.duration, stop)

    threads = [threading.Thread(target=syncer, daemon=True),
               threading.Thread(target=sampler, daemon=True),
               threading.Thread(target=measure_slew, daemon=True)]
    if not args.no_sensor_load:
        for channel, plan in UPSTREAM_PLAN.items():
            threads.append(threading.Thread(
                target=sensor, args=(channel, plan["hz"], plan["bytes"]), daemon=True))
    for thread in threads:
        thread.start()
    time.sleep(args.duration)
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)

    initiator = initiator_box.get("current")
    mono_slope = slope_ppm(mono_track)
    wall_slope = slope_ppm(wall_track)
    empirical = None
    if mono_slope is not None and wall_slope is not None:
        empirical = mono_slope - wall_slope

    report = {
        "role": "phone",
        "host": args.host,
        "duration_s": args.duration,
        "sensor_load": not args.no_sensor_load,
        "link_path": tailnet_path(args.host),
        "sessions": sorted(routers),
        "sync_samples": samples_seen,
        "sensor_sent": {c.value: n for c, n in sensor_sent.items() if n},
        "initiator": None if initiator is None else initiator.to_record(),
        "trace": trace,
        "local_slew": slew_box.get("local", {"measured": False}),
        "wall_cross_check": {
            # Counted, so a cross-check that silently measured nothing reads as
            # zero coverage rather than as agreement.
            "mono_pairs": len(mono_track),
            "wall_pairs": len(wall_track),
            "mono_pair_slope_ppm": mono_slope,
            "wall_pair_slope_ppm": wall_slope,
            "empirical_true_skew_ppm": empirical,
            "assumed_skew_ppm": ASSUMED_SKEW_PPM,
            "slew_difference_ppm": empirical,
            # Recorded, not assumed away: the local send stamp is taken before
            # the enqueue while the arrival stamps come from the readers, so this
            # is the one residual asymmetry left in the pair. A constant one
            # cancels in the slope; a drifting one is the thing that would
            # corrupt the number, so it is bounded rather than left unattributable.
            "local_send_to_arrival_ms": percentiles(send_to_arrival),
            "INVALID_FOR_SKEW": (
                "Retained as evidence, not as a measurement. This difference is "
                "s_local - s_remote, the two NTP slew rates: the true skew cancels "
                "out of it algebraically. Use local_slew from both roles instead."
            ),
        },
    }
    usable_rows = [row for row in trace if row.get("usable")]
    if usable_rows:
        report["bound_ms"] = percentiles([row["bound_ns"] for row in usable_rows])
        report["rtt_min_ms"] = percentiles([row["rtt_min_ns"] for row in usable_rows])
        offsets = [row["offset_ns"] for row in usable_rows]
        report["offset_spread_ms"] = round((max(offsets) - min(offsets)) / 1e6, 3)
        skews = [row["skew_ppm"] for row in usable_rows if row.get("skew_ppm") is not None]
        report["skew_ppm_reported"] = (
            None if not skews
            else {"n": len(skews), "median": round(statistics.median(skews), 3),
                  "min": round(min(skews), 3), "max": round(max(skews), 3)}
        )
    # A cross-check that produced no pairs is not a cross-check. Stated rather
    # than left as a null: the first version of this read an attribute that did
    # not exist, so it would have reported None forever and looked like a
    # measurement nobody could fault.
    report["wall_cross_check"]["measured"] = bool(wall_track) and empirical is not None
    report["trace_rows"] = len(trace)
    report["trace_rows_usable"] = len(usable_rows)
    # One gate, counted from records rather than derived by subtraction.
    report["usable"] = bool(
        routers and samples_seen > 0 and len(usable_rows) > 0
    )
    client.stop()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("phone", "jetson"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--trace-every", type=float, default=10.0)
    parser.add_argument("--no-sensor-load", action="store_true",
                        help="run the exchange alone, to isolate what the load costs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.device_id = args.device_id or f"{args.role}-timebase-probe"
    report = run_jetson(args) if args.role == "jetson" else run_phone(args)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        summary = {k: v for k, v in report.items() if k != "trace"}
        print(json.dumps(summary, indent=2))
    else:
        print(text)
    return 0 if report.get("usable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
