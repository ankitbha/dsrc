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


def here_line(seq, wall_s, lat, lon):
    r = {"ch": "here", "seq": seq, "query_lat": lat, "query_lon": lon, "t_wall_ns": int(wall_s * 1e9), "status": 200}
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
