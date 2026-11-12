#!/usr/bin/env python3
"""Re-process the drive frames for the camera error model (jam plan block F, section 2.2).

    .venv/bin/python scripts/process_drive_frames.py                  # all six drives, 640 and 1,280 px
    .venv/bin/python scripts/process_drive_frames.py --imgsz 640 --runs run_20260908_190548 \\
        --limit 50 --start-pos 4000 --out-dir outputs/jam/smoke/frames   # the S7 smoke

    .venv/bin/python scripts/process_drive_frames.py --rebuild-from-raw   # detections only

For each drive and image size, two files under `outputs/jam/frames/<run>/`:

* `raw_<imgsz>.jsonl`: a header (weights sha256, detector settings, ultralytics and torch
  versions, the frame counts: indexed, planned, without a tick, duplicate ids,
  unparseable index lines), then one line per frame with its capture time from the
  matched tick and its boxes `[x1, y1, x2, y2, conf, cls]` on the upright image.
* `detections_<imgsz>.jsonl`: a header with the drive's horizon and camera-height fits
  (all usable boxes, and the boxes near the centre column as a diagnostic), then the
  deployment tracker's output, one `frame` line per frame and one `det` line per tracked
  vehicle per frame, with both of `DistanceEstimator`'s methods: the width prior (`fx`
  and the class widths only) and the ground plane on the all-box fit.
  `--rebuild-from-raw` rewrites this file from `raw_<imgsz>.jsonl` without the detector.

The weights are the official Ultralytics `yolov8n.pt`, fetched once into
`outputs/models/yolov8n.pt` (gitignored). Their sha256 is recorded in
`results/jam/weights.json` and checked at every run; `--record-weights` writes that
record when it does not exist yet. The drive data directory is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import camera_error as ce  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "outputs" / "models" / "yolov8n.pt"
DEFAULT_RECORD = REPO_ROOT / "results" / "jam" / "weights.json"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_weights(weights: Path, record: Path, write: bool) -> str:
    """The weights' sha256, which must equal the recorded one."""
    if not weights.exists():
        raise SystemExit(f"{weights} does not exist; fetch the official yolov8n.pt once: "
                         f"curl -sSL --fail -o {weights} {WEIGHTS_URL}")
    sha = _sha256(weights)
    if not record.exists():
        if not write:
            raise SystemExit(f"{record} does not exist; pass --record-weights to record "
                             f"{weights} (sha256 {sha})")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({
            "path": str(weights.relative_to(REPO_ROOT)) if weights.is_relative_to(REPO_ROOT)
            else str(weights), "sha256": sha, "bytes": weights.stat().st_size,
            "source_url": WEIGHTS_URL, "recorded_wall": time.time(),
            "note": "official Ultralytics release asset, fetched once; kept out of git"},
            indent=1))
    recorded = json.loads(record.read_text())["sha256"]
    if recorded != sha:
        raise SystemExit(f"{weights} has sha256 {sha}, but {record} records {recorded}")
    return sha


def _versions() -> dict[str, str | None]:
    out = {}
    for name in ("ultralytics", "torch", "cv2", "numpy"):
        try:
            out[name] = getattr(__import__(name), "__version__", None)
        except ImportError:
            out[name] = None
    return out


def process(run: str, imgsz: int, model, weights_sha: str, *, data_dir: Path, out_dir: Path,
            start_pos: int | None, limit: int | None) -> dict:
    run_dir = data_dir / run
    plan = ce.frame_plan(run_dir)
    frames = [f for f in plan.frames if start_pos is None or f.pos >= start_pos]
    if limit is not None:
        frames = frames[:limit]
    target = out_dir / run
    target.mkdir(parents=True, exist_ok=True)
    detected: list[ce.FrameDetections] = []
    walls, shape = [], None
    raw_path = target / f"raw_{imgsz}.jsonl"
    started = time.perf_counter()
    with raw_path.open("w") as handle:
        lines = []
        for frame, image in ce.iter_frames(run_dir, frames):
            shape = list(image.shape)
            t0 = time.perf_counter()
            boxes = ce.detect(model, image, imgsz)
            walls.append(time.perf_counter() - t0)
            detected.append(ce.FrameDetections(frame.frame_id, frame.pos, frame.t, boxes))
            lines.append(json.dumps({"type": "frame", "frame_id": frame.frame_id,
                                     "pos": frame.pos, "t": frame.t,
                                     "boxes": [[*[round(float(v), 2) for v in b], round(c, 4), k]
                                               for b, c, k in boxes]}))
        header = {"type": "header", "run": run, "imgsz": imgsz, "weights_sha256": weights_sha,
                  "confidence": ce.CONFIDENCE, "nms_iou": ce.NMS_IOU,
                  "classes": list(ce.VEHICLE_CLASSES), "hood_line_y_px": ce.HOOD_LINE_Y_PX,
                  "rotation": "cv2.ROTATE_90_CLOCKWISE", "upright_shape": shape,
                  "frame_time": "matched tick t_capture_mono_ns", **plan.describe(),
                  "processed_frames": len(detected), "start_pos": start_pos, "limit": limit,
                  "versions": _versions(), "detect_ms_mean": 1000 * float(np.mean(walls)) if walls else None}
        handle.write(json.dumps(header) + "\n")
        handle.write("\n".join(lines) + ("\n" if lines else ""))
    summary = write_detections(target, run, imgsz, weights_sha, detected)
    return {"run": run, "imgsz": imgsz, "frames": len(detected), **plan.describe(),
            "boxes": sum(len(f.boxes) for f in detected), **summary,
            "wall_s": time.perf_counter() - started,
            "detect_ms_mean": 1000 * float(np.mean(walls)) if walls else None}


#: The centre-column band of the diagnostic horizon fit (orchestrator finding O-1).
CENTRE_FRACTION = 0.15


def write_detections(target: Path, run: str, imgsz: int, weights_sha: str,
                     detected: list, rebuilt: bool = False) -> dict:
    """`detections_<imgsz>.jsonl` from a drive's boxes: both horizon fits as diagnostics,
    then the tracker's records with both distance methods.

    The ground-plane estimator uses the all-box fit; the width-prior estimator, whose
    distances the camera model's primary tables are built on, uses neither fit.
    """
    fit = ce.horizon_fit(b for f in detected for b in f.boxes)
    centre = ce.horizon_fit((b for f in detected for b in f.boxes),
                            centre_fraction=CENTRE_FRACTION)
    records = ce.track_and_measure(detected, fit["horizon_y_px"], fit["camera_height_m"])
    with (target / f"detections_{imgsz}.jsonl").open("w") as handle:
        handle.write(json.dumps({"type": "header", "run": run, "imgsz": imgsz,
                                 "weights_sha256": weights_sha, "horizon": fit,
                                 "horizon_centre_boxes": centre,
                                 "ground_plane_uses": "horizon (all usable boxes)",
                                 "distance_methods": ["width_prior", "ground_plane"],
                                 "fx_px": ce.FX_PX, "cx_px": ce.CX_PX,
                                 "hood_line_y_px": ce.HOOD_LINE_Y_PX,
                                 "frames": len(detected), "rebuilt_from_raw": rebuilt}) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return {"tracked_records": sum(1 for r in records if r["type"] == "det"),
            "horizon": fit, "horizon_centre_boxes": centre}


def rebuild(run: str, imgsz: int, weights_sha: str, out_dir: Path) -> dict:
    """Rewrite `detections_<imgsz>.jsonl` from an existing `raw_<imgsz>.jsonl` without the
    detector: the raw boxes do not depend on the horizon fit. The raw file must have run
    the recorded weights."""
    target = out_dir / run
    raw_path = target / f"raw_{imgsz}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist; run the detector first")
    lines = raw_path.read_text().splitlines()
    header = json.loads(lines[0])
    if header.get("weights_sha256") != weights_sha:
        raise SystemExit(f"{raw_path} ran weights {header.get('weights_sha256')}; the record "
                         f"names {weights_sha}")
    detected = []
    for line in lines[1:]:
        row = json.loads(line)
        if row.get("type") != "frame":
            continue
        detected.append(ce.FrameDetections(
            row["frame_id"], row["pos"], row["t"],
            [(np.asarray(b[:4], dtype=np.float32), float(b[4]), int(b[5])) for b in row["boxes"]]))
    started = time.perf_counter()
    summary = write_detections(target, run, imgsz, weights_sha, detected, rebuilt=True)
    return {"run": run, "imgsz": imgsz, "frames": len(detected),
            "boxes": sum(len(f.boxes) for f in detected), **summary,
            "wall_s": time.perf_counter() - started, "detect_ms_mean": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(ce.INDEXED_RUNS))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    parser.add_argument("--limit", type=int, default=None, help="frames per drive (smoke)")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="skip indexed frames before this AVI position (smoke)")
    parser.add_argument("--data-dir", default=str(ce.DRIVE_DATA))
    parser.add_argument("--out-dir", default="outputs/jam/frames")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--weights-record", default=str(DEFAULT_RECORD))
    parser.add_argument("--record-weights", action="store_true")
    parser.add_argument("--rebuild-from-raw", action="store_true",
                        help="rewrite detections_<imgsz>.jsonl from the existing raw_<imgsz>.jsonl "
                             "without running the detector (its weights sha256 is checked)")
    args = parser.parse_args(argv)

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    out_dir = resolve(args.out_dir)
    if args.rebuild_from_raw:
        record = resolve(args.weights_record)
        if not record.exists():
            raise SystemExit(f"{record} does not exist")
        sha = json.loads(record.read_text())["sha256"]
        model = None
    else:
        weights = resolve(args.weights)
        sha = check_weights(weights, resolve(args.weights_record), args.record_weights)
        from ultralytics import YOLO

        model = YOLO(str(weights))
    summaries = []
    for run in args.runs:
        for imgsz in args.imgsz:
            summary = (rebuild(run, imgsz, sha, out_dir) if args.rebuild_from_raw else
                       process(run, imgsz, model, sha, data_dir=Path(args.data_dir),
                               out_dir=out_dir, start_pos=args.start_pos, limit=args.limit))
            summaries.append(summary)
            print(f"{run} {imgsz} px: {summary['frames']} frames, {summary['boxes']} boxes, "
                  f"{summary['tracked_records']} tracked records, horizon "
                  f"{summary['horizon']['horizon_y_px']:.1f} px ({summary['horizon']['source']}), "
                  f"{summary['detect_ms_mean'] or 0:.1f} ms/frame, {summary['wall_s']:.1f} s",
                  flush=True)
    (out_dir / ("summary_rebuilt.json" if args.rebuild_from_raw else "summary.json")).write_text(json.dumps(
        {"weights_sha256": sha, "runs": summaries, "command": argv or sys.argv}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
