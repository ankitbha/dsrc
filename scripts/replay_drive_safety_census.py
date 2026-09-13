#!/usr/bin/env python3
"""Rebuild each drive's observation from its stored frames, with the rotation fix applied.

The 2026-09-08 drives recorded frames rotated ninety degrees, so the detector saw
almost nothing: 16 detection-bearing ticks across 22,929, and an empty vehicle list
even on those. The fix landed 2026-09-09, one day after the collection, so no drive
in the corpus carries a tracked leader. Every safety rule that needs one therefore
reads `not_evaluable` on every recorded tick, and `score_safety.py` over the drives
reports the same census it reports over the bench runs.

The frames survive. This re-runs detection, tracking, distance and the observation
builder over them with the rotation applied and the post-rotation intrinsics, and
writes a `metadata.jsonl` that `score_safety.py` consumes unchanged. What comes out
is what the gate would have been able to evaluate had the drives run with the fix.

FOUR THINGS THIS SUBSTITUTES, EACH OF WHICH LIMITS THE CLAIM.

1. The detector is not the deployed one. The device runs a YOLOv8n TensorRT FP16
   engine; neither TensorRT nor CUDA nor any weights file exists off-device, so this
   runs ultralytics YOLOv8n in FP32 on the CPU. Same architecture and same
   confidence, IoU and vehicle-class settings read from the same config, but a
   different implementation and a different numeric precision. Marginal detections
   near the confidence threshold will differ. Whether a leader is tracked at all,
   which is what the census turns on, is robust to that difference.

2. The action is the one the drive recorded, not one re-inferred from the new
   observation. The actor bundle is gitignored and absent off-device, and the half
   of the pipeline the rotation defect broke is the perception half. Holding the
   action fixed changes one variable. `score_safety.py` reports two forced-action
   arms beside the recorded one, which is where action sensitivity is covered.

3. Timing is the replay's, not the drive's. The tracker and the relative-speed
   window key off a monotonic clock, and this walks the recorded frame order at the
   recorded spacing rather than reproducing the live loop's jitter. The live
   pipeline also drops frames under load, latest-value-wins, so a live tick may have
   used a different frame than the one paired here by index.

4. Distance rests on an asserted calibration. The post-rotation intrinsics in
   `config.yaml` were measured over forty mid-drive frames of one run. Gap and
   time-to-collision inherit whatever error is in that, and nothing in the corpus
   independently validates it.

So this establishes what the gate could have evaluated, not what it did, and not
what the deployed engine would have detected to the last box. Nothing is written
into the drive-data directory.

    python3 scripts/replay_drive_safety_census.py <run_dir> [<run_dir> ...] \
        --out-dir <scratch> [--rotate-cw-deg N] [--limit N]

`--rotate-cw-deg 0` is the control: it reproduces the drive's own zero-detection
result from the same frames through the same code, which is what makes the rotated
run's count evidence rather than an assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JETSON = REPO / "deployment" / "jetson"
sys.path.insert(0, str(JETSON))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception.detector import Detection  # noqa: E402
from perception.distance import DistanceEstimator  # noqa: E402
from perception.observation_builder import BuilderConfig, ObservationBuilder  # noqa: E402
from perception.tracker import IouTracker  # noqa: E402
from sensors.gps_reader import GpsFix  # noqa: E402
from sensors.phone_source import rotate_frame  # noqa: E402
from run_demo import load_config  # noqa: E402


def tick_records(path: Path) -> list[dict]:
    """The run's tick records, in order. The log interleaves other record types."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and "obs" in rec and "field_sources" in rec:
            out.append(rec)
    return out


def fix_from_record(record: dict, t_mono: float) -> GpsFix:
    """The recorded fix, restamped onto the replay clock so freshness is about
    this pass rather than about a clock that stopped a year ago."""
    gps = record.get("gps") or {}
    valid = bool(gps.get("valid"))

    def num(key):
        v = gps.get(key)
        return float("nan") if v is None else float(v)

    return GpsFix(
        valid=valid,
        lat=num("lat"),
        lon=num("lon"),
        speed_mps=num("speed_mps"),
        heading_deg=num("heading_deg"),
        fix_quality=1 if valid else 0,
        num_sats=int(gps.get("num_sats") or 0),
        hdop=num("hdop"),
        t_mono=t_mono,
        t_wall=float(record.get("t_wall") or 0.0),
    )


def detections_from(result, classes: set[int], conf: float) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    out = []
    for box, c, k in zip(xyxy, confs, clss):
        if int(k) in classes and float(c) >= conf:
            out.append(Detection(xyxy=box.astype(np.float32), conf=float(c), cls=int(k)))
    return out


def replay_run(run_dir: Path, out_dir: Path, config: dict, model, rotate_cw_deg: int,
               limit: int) -> dict:
    video_path = run_dir / "video.avi"
    meta_path = run_dir / "metadata.jsonl"
    records = tick_records(meta_path)
    cam = config["camera"]
    det_cfg = config["detector"]
    trk = config["tracker"]
    dist_cfg = config["distance"]
    classes = {int(c) for c in det_cfg["vehicle_classes"]}

    tracker = IouTracker(
        iou_match_threshold=trk["iou_match_threshold"],
        max_age_frames=trk["max_age_frames"],
        min_hits=trk["min_hits"],
    )
    distance = DistanceEstimator(
        fx_px=cam["fx_px"],
        cx_px=cam["cx_px"],
        horizon_y_px=cam["horizon_y_px"],
        camera_height_m=cam["camera_height_m"],
        method=dist_cfg["method"],
        ema_alpha=dist_cfg["ema_alpha"],
        rel_speed_window=dist_cfg["rel_speed_window"],
        rel_speed_min_span_s=dist_cfg["rel_speed_min_span_s"],
        class_widths_m={int(k): float(v) for k, v in dist_cfg["class_widths_m"].items()},
        max_range_m=dist_cfg["max_range_m"],
        contact_cutoff_y_px=(cam.get("hood_line_y_px") or (cam["height"] - 3)),
    )
    builder = ObservationBuilder(BuilderConfig.from_full_config(config))

    video = cv2.VideoCapture(str(video_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metadata.jsonl"
    n_written = n_frames = n_det_ticks = n_leader_ticks = 0
    t0 = time.monotonic()
    with out_path.open("w") as handle:
        i = 0
        while i < len(records):
            ok, image = video.read()
            if not ok:
                break
            if limit and i >= limit:
                break
            rec = records[i]
            if rotate_cw_deg:
                image = rotate_frame(image, rotate_cw_deg)
            # A synthetic clock at the recorded spacing: the tracker's age counter
            # and the relative-speed window both read it, and the wall clock the
            # drive recorded is not monotonic across a replay.
            t_mono = t0 + i * 0.2
            result = model.predict(image, verbose=False, conf=det_cfg["conf_threshold"],
                                   iou=det_cfg["iou_threshold"], imgsz=det_cfg["input_size"])[0]
            dets = detections_from(result, classes, float(det_cfg["conf_threshold"]))
            tracks = tracker.update(dets, t_mono)
            vehicles = distance.update(tracks, t_mono)
            obs = builder.build(vehicles, fix_from_record(rec, t_mono), t_mono)
            leader_gap = obs.obs.get("leader_gap")
            if dets:
                n_det_ticks += 1
            if leader_gap is not None and np.isfinite(leader_gap):
                n_leader_ticks += 1
            handle.write(json.dumps({
                "type": "tick",
                "tick_id": rec.get("tick_id", i),
                "frame_id": i,
                "t_wall": rec.get("t_wall", 0.0),
                "n_detections": len(dets),
                "vehicles": [
                    {"track_id": v.track_id, "distance_m": v.distance_m,
                     "lateral_m": v.lateral_m, "rel_speed_mps": v.rel_speed_mps,
                     "cls": v.cls}
                    for v in vehicles
                ],
                "obs": obs.obs,
                "field_sources": obs.field_sources,
                "obs_diagnostics": obs.diagnostics,
                # Held fixed from the drive: see the docstring's point 2.
                "action": rec.get("action"),
            }) + "\n")
            n_written += 1
            n_frames += 1
            i += 1
    video.release()
    return {
        "run": run_dir.name,
        "rotate_cw_deg": rotate_cw_deg,
        "live_tick_records": len(records),
        "frames_replayed": n_frames,
        "ticks_written": n_written,
        "ticks_with_a_detection": n_det_ticks,
        "ticks_with_a_finite_leader_gap": n_leader_ticks,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rotate-cw-deg", type=int, default=None,
                    help="defaults to the deployed config's value; 0 is the control")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    config = load_config(str(JETSON / "config.yaml"))
    rotate = (args.rotate_cw_deg if args.rotate_cw_deg is not None
              else int((config.get("camera") or {}).get("rotate_cw_deg", 0) or 0))
    model = YOLO(args.weights)
    print(f"[replay] rotation {rotate} deg cw, detector {args.weights} on cpu, "
          f"conf {config['detector']['conf_threshold']}, "
          f"intrinsics cx={config['camera']['cx_px']} cy={config['camera']['cy_px']} "
          f"horizon={config['camera']['horizon_y_px']}")

    summaries = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser()
        if not (run_dir / "video.avi").exists():
            print(f"[replay] {run_dir.name}: no video, skipped")
            continue
        out = args.out_dir.expanduser() / run_dir.name
        s = replay_run(run_dir, out, config, model, rotate, args.limit)
        summaries.append(s)
        print(f"[replay] {s['run']}: {s['ticks_written']} ticks, "
              f"{s['ticks_with_a_detection']} with a detection, "
              f"{s['ticks_with_a_finite_leader_gap']} with a leader gap, "
              f"{s['seconds']}s")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
