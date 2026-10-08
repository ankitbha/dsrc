"""Synthetic phone log, HERE index/bodies and video index written to a temp directory."""
from __future__ import annotations

import json
from pathlib import Path


def gps_line(seq, utc_s, lat, lon, speed=10.0, heading=90.0, wall_offset=1.2, valid=True):
    r = {"ch": "gps", "seq": seq, "valid": valid, "lat": lat if valid else None, "lon": lon,
         "speed_mps": speed, "heading_deg": heading,
         "utc_epoch_ns": int(utc_s * 1e9), "t_wall_ns": int((utc_s + wall_offset) * 1e9),
         "t_mono_ns": 5, "t_capture_mono_ns": 4}
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def here_line(seq, wall_s, lat, lon, resp_mono_s=None):
    r = {"ch": "here", "seq": seq, "query_lat": lat, "query_lon": lon, "t_wall_ns": int(wall_s * 1e9), "status": 200}
    if resp_mono_s is not None:
        r["t_response_mono_ns"] = int(resp_mono_s * 1e9)
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def camera_line(frame_id, wall_s, mono_gap_s=0.05):
    r = {"ch": "camera", "frame_id": frame_id, "t_wall_ns": int(wall_s * 1e9),
         "t_mono_ns": int(100e9 + mono_gap_s * 1e9), "t_capture_mono_ns": int(100e9)}
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def body(source_updated, results):
    return {"sourceUpdated": source_updated, "results": results}


def result(desc, length, pts, speed=10.0, free=12.0, jam=1.0, subs=None, fc=2):
    """pts: list of (lat, lng). One link per consecutive pair."""
    links = [{"points": [{"lat": a[0], "lng": a[1]}, {"lat": b[0], "lng": b[1]}], "length": 1.0, "functionalClass": fc}
             for a, b in zip(pts, pts[1:])]
    flow = {"speed": speed, "freeFlow": free, "jamFactor": jam, "confidence": 0.99}
    if subs:
        flow["subSegments"] = subs
    return {"location": {"description": desc, "length": length, "shape": {"links": links}}, "currentFlow": flow}


def write_run(root: Path, run: str, phone_lines, index, bodies, video_ids):
    """index: list of dicts for here_index.jsonl; bodies: filename -> body dict."""
    d = root / run
    (d / "here").mkdir(parents=True)
    with (d / "here_index.jsonl").open("w") as f:
        for rec in index:
            f.write(json.dumps(rec) + "\n")
    for name, b in bodies.items():
        (d / name).write_text(json.dumps(b))
    with (d / "video_index.jsonl").open("w") as f:
        for pos, fid in enumerate(video_ids):
            f.write(json.dumps({"pos": pos, "frame_id": fid}) + "\n")
    phone = root / "phone.jsonl"
    phone.write_text("\n".join(phone_lines) + "\n")
    return d, phone


# ---- planar helpers: build roads in metres and convert to latitude/longitude --------------
import math  # noqa: E402

from here_error import load  # noqa: E402
from here_error.models import GpsFix, HereBody  # noqa: E402

LAT0, LON0 = 40.0, -74.0
R = 6371000.0


def to_latlon(x, y):
    return LAT0 + math.degrees(y / R), LON0 + math.degrees(x / (R * math.cos(math.radians(LAT0))))


def road(desc, xy_points, stated_length=None, **kw):
    """A HERE flow result for a polyline given in metres east/north of (LAT0, LON0)."""
    pts = [to_latlon(x, y) for x, y in xy_points]
    length = stated_length
    if length is None:
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy_points, xy_points[1:]))
    return result(desc, round(length), pts, **kw)


def segment(desc, xy_points, **kw):
    return load.parse_segment(road(desc, xy_points, **kw))


def make_body(run, seq, segments, data_time_s, response_utc_s=None):
    return HereBody(run, seq, f"here/{seq:06d}.json", data_time_s,
                    data_time_s + 60 if response_utc_s is None else response_utc_s, LAT0, LON0, tuple(segments))


def fix(t, x, y, speed=10.0, heading=90.0):
    lat, lon = to_latlon(x, y)
    return GpsFix(utc_s=float(t), wall_s=float(t) + 1.2, lat=lat, lon=lon, speed_mps=speed, heading_deg=heading)


def drive_east(t0, x0, x1, step=10.0, speed=10.0, y=0.0):
    """One fix per second moving east from x0 to x1 inclusive."""
    n = int(round((x1 - x0) / step))
    return [fix(t0 + i, x0 + i * step, y, speed=speed, heading=90.0) for i in range(n + 1)]
