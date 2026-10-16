#!/usr/bin/env python3
"""The camera-against-feed case table on the two drives with HERE responses (jam plan
section 3).

    .venv/bin/python scripts/drive_lead_cases.py

Writes `results/jam/drive_lead_cases.json`. An event is a HERE flow item within 60 m of
the vehicle at a response's receipt whose jam factor rose by at least 2 to at least 4
between the consecutive versions the vehicle received. For each event the table gives, on
the run's own clock (the Jetson monotonic clock, through `metadata.jsonl`'s wall-clock
anchor): the version's stamp, its first receipt, the time the vehicle's GPS speed first
fell below 5 m/s, and the camera's first report from the re-processed detections (when
the frame records cover the window): the first frame whose `k_hat` reaches `0.9 k_c` or
whose own-lane leader's estimated speed (GPS speed plus the tracked closing rate) is below
half of its value 30 s earlier.

This is a description for the author, not a lead-time estimate: both roads are signalised
arterials and the vehicle reached each queue once, so the order of the two first reports
is set by when the vehicle arrived. Only the recorded HERE bodies are read; no request is
made, and no request URL is copied.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRIVES = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")
RUNS = ("run_20260908_170849", "run_20260908_190548")
NEAR_M = 60.0
MIN_RISE = 2.0
MIN_JAM = 4.0
K_CRITICAL = 0.178
STOP_MPS = 5.0
WINDOW_S = 300.0
LEADER_LOOKBACK_S = 30.0
OWN_HALF_WIDTH_M = 1.85
ADJACENT_OUTER_M = 5.55
RANGE_M = 80.0
OCCUPANCY_LENGTH_M = 4.5


@dataclass(frozen=True)
class FlowItem:
    description: str
    length: float
    points: list[dict[str, float]]
    jam_factor: float | None


@dataclass(frozen=True)
class Response:
    received_t: float
    source_updated: str
    items: list[FlowItem]


@dataclass(frozen=True)
class GpsFix:
    t: float
    lat: float
    lon: float
    speed_mps: float


@dataclass(frozen=True)
class Event:
    description: str
    length: float
    previous_version: str
    version: str
    jam_before: float
    jam_after: float
    received_t: float
    distance_m: float


def _distance_m(lat1, lon1, lat2, lon2) -> float:
    k = 111320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


def _nearest_fix(track: list[GpsFix], t: float) -> GpsFix:
    times = np.array([f.t for f in track])
    return track[int(np.argmin(np.abs(times - t)))]


def find_events(responses: list[Response], track: list[GpsFix]) -> list[Event]:
    """Items near the vehicle whose jam factor rose by >= 2 to >= 4 between versions."""
    last: dict[tuple, tuple[str, float]] = {}
    events = []
    for response in sorted(responses, key=lambda r: r.received_t):
        fix = _nearest_fix(track, response.received_t)
        for item in response.items:
            if item.jam_factor is None:
                continue
            key = (item.description, item.length)
            previous = last.get(key)
            if previous is not None and previous[0] != response.source_updated:
                distance = min(_distance_m(fix.lat, fix.lon, p["lat"], p["lng"])
                               for p in item.points)
                if (item.jam_factor >= previous[1] + MIN_RISE and item.jam_factor >= MIN_JAM
                        and distance <= NEAR_M):
                    events.append(Event(item.description, item.length, previous[0],
                                        response.source_updated, previous[1], item.jam_factor,
                                        response.received_t, distance))
            if previous is None or previous[0] != response.source_updated:
                last[key] = (response.source_updated, item.jam_factor)
    return events


def feed_first_report(responses: list[Response], version: str) -> float | None:
    times = [r.received_t for r in responses if r.source_updated == version]
    return min(times) if times else None


def gps_first_stop(track: list[GpsFix], t_from: float, t_to: float) -> float | None:
    for fix in sorted(track, key=lambda f: f.t):
        if t_from <= fix.t <= t_to and fix.speed_mps < STOP_MPS:
            return fix.t
    return None


def gps_stop_onsets(track: list[GpsFix], t_from: float, t_to: float) -> list[float]:
    """Every instant in the window where the speed falls from at least 5 m/s to below it;
    the first stop in a window can belong to an earlier signal than the event's."""
    out, previous = [], None
    for fix in sorted(track, key=lambda f: f.t):
        if t_from <= fix.t <= t_to and previous is not None:
            if previous >= STOP_MPS > fix.speed_mps:
                out.append(fix.t)
        previous = fix.speed_mps
    return out


def camera_first_report(records: list[dict], track: list[GpsFix], t_from: float,
                        t_to: float) -> dict[str, Any] | None:
    """The first frame in the window where the camera's local density reaches `0.9 k_c`
    or the own-lane leader's estimated speed falls below half of its value 30 s earlier.

    Lanes from the lateral offset: own within 1.85 m of the axis, adjacent out to 5.55 m;
    `k_hat` is the mean over the three lanes of `L/(g + L)` (own lane) and `n L / 80 m`.
    """
    by_frame: dict[float, list[dict]] = {}
    for r in records:
        if r.get("type") == "det" and t_from <= r["t"] <= t_to:
            by_frame.setdefault(float(r["t"]), []).append(r)
    if not by_frame or not track:
        return None
    leader_speed: list[tuple[float, float]] = []
    for t in sorted(by_frame):
        rows = by_frame[t]
        own = [r for r in rows if abs(r["lateral_m"]) < OWN_HALF_WIDTH_M]
        left = [r for r in rows if -ADJACENT_OUTER_M < r["lateral_m"] <= -OWN_HALF_WIDTH_M
                and r["ema_m"] <= RANGE_M]
        right = [r for r in rows if OWN_HALF_WIDTH_M <= r["lateral_m"] < ADJACENT_OUTER_M
                 and r["ema_m"] <= RANGE_M]
        k_own = 0.0
        leader = min(own, key=lambda r: r["ema_m"]) if own else None
        if leader is not None and leader["ema_m"] <= RANGE_M:
            k_own = OCCUPANCY_LENGTH_M / (max(leader["ema_m"], 0.0) + OCCUPANCY_LENGTH_M)
        k_hat = (k_own + (len(left) + len(right)) * OCCUPANCY_LENGTH_M / RANGE_M) / 3.0
        if k_hat >= 0.9 * K_CRITICAL:
            return {"t": t, "criterion": "density", "k_hat": k_hat}
        if leader is not None and leader.get("closing_valid"):
            speed = _nearest_fix(track, t).speed_mps + float(leader["closing_rate_mps"])
            earlier = [s for u, s in leader_speed if u <= t - LEADER_LOOKBACK_S + 1e-9]
            if earlier and speed < 0.5 * earlier[-1]:
                return {"t": t, "criterion": "leader_speed", "leader_speed_mps": speed,
                        "leader_speed_30s_earlier_mps": earlier[-1], "k_hat": k_hat}
            leader_speed.append((t, speed))
    return None


# ------------------------------------------------------------------- the drives


def _stamp(iso: str) -> float:
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def load_drive(run_dir: Path) -> tuple[float, list[GpsFix], float]:
    """`(wall minus run clock offset, GPS fixes on the run clock, first valid fix)`."""
    anchors, fixes = [], []
    with (run_dir / "metadata.jsonl").open() as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r.get("t_mono"), (int, float)) and "t_wall" in r:
                anchors.append(r["t_wall"] - r["t_mono"])
            gps = r.get("gps") if r.get("type") == "tick" else None
            if gps and gps.get("valid"):
                fixes.append((r["t_wall"], gps["lat"], gps["lon"], gps["speed_mps"]))
    offset = statistics.median(anchors)
    track = [GpsFix(w - offset, lat, lon, v) for w, lat, lon, v in fixes]
    return offset, track, track[0].t


def load_responses(run_dir: Path) -> list[Response]:
    out = []
    for line in (run_dir / "here_index.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") != 200:
            continue
        body = json.loads((run_dir / row["file"]).read_text())
        items = [FlowItem(it["location"].get("description"), it["location"].get("length"),
                          [p for link in it["location"]["shape"]["links"] for p in link["points"]],
                          (it.get("currentFlow") or {}).get("jamFactor"))
                 for it in body.get("results", [])]
        out.append(Response(float(row["received_t_mono"]), body["sourceUpdated"], items))
    return out


def _detections(frames_dir: Path, run: str) -> list[dict]:
    path = frames_dir / run / "detections_640.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines()[1:]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DRIVES))
    parser.add_argument("--frames-dir", default="outputs/jam/frames")
    parser.add_argument("--out", default="results/jam/drive_lead_cases.json")
    args = parser.parse_args(argv)
    frames_dir = Path(args.frames_dir)
    frames_dir = frames_dir if frames_dir.is_absolute() else REPO_ROOT / frames_dir
    cases = []
    for run in RUNS:
        run_dir = Path(args.data_dir) / run
        offset, track, first = load_drive(run_dir)
        responses = load_responses(run_dir)
        detections = _detections(frames_dir, run)
        for event in find_events(responses, track):
            received = feed_first_report(responses, event.version)
            stamp = _stamp(event.version) - offset
            window = (received - WINDOW_S, received + WINDOW_S)
            camera = camera_first_report(detections, track, *window)
            stop = gps_first_stop(track, *window)
            covered = [r["t"] for r in detections
                       if r.get("type") == "frame" and window[0] <= r["t"] <= window[1]]
            cases.append({
                "run": run, **asdict(event), "stamp_t": stamp, "first_receipt_t": received,
                "received_after_first_tick_s": received - first,
                "stamp_to_receipt_s": received - stamp,
                "gps_first_below_5mps_t": stop,
                "gps_stop_onsets_t": gps_stop_onsets(track, *window),
                "camera_frames_in_window": len(covered),
                "camera_coverage_t": [min(covered), max(covered)] if covered else None,
                "camera_first_report": camera,
                "camera_note": (None if detections else
                                "no re-processed frame records for this drive"),
                "camera_minus_feed_s": (camera["t"] - received) if camera else None,
            })
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rule": f"item within {NEAR_M:g} m whose jam factor rose by >= {MIN_RISE:g} to "
                f">= {MIN_JAM:g} between consecutive versions",
        "clock": "the run's Jetson monotonic clock; first tick = first valid GPS fix",
        "note": "a case description, not a lead-time estimate (plan section 3)",
        "events": cases}, indent=1))
    print(f"wrote {out}: {len(cases)} events")
    for c in cases:
        print(f"  {c['run']} {c['description']!r}: {c['jam_before']} -> {c['jam_after']} in "
              f"{c['version']}, received {c['received_after_first_tick_s']:.0f} s after the "
              f"first fix, {c['distance_m']:.0f} m away")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
