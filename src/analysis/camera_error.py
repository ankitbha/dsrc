"""The camera error model from the re-processed drive frames (plan section 2).

The device's own detector saw almost nothing on the drives because the phone delivered its
frames rotated ninety degrees. This module re-processes the recorded frames: each indexed
frame is rotated clockwise to a 720 x 1,280 upright image, timed by its tick's capture
instant (never by the AVI's 30 fps tag; the frames were captured 0.20 s or 0.67 s apart),
run through YOLOv8n, then through the deployment's own `IouTracker` and
`DistanceEstimator` (`deployment/jetson/perception/`), unchanged. Section 2.3's tables are
then measured from those records (`detection_rate_table` and the rest, step S8).

The drive data directory is only read.
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JETSON = REPO_ROOT / "deployment" / "jetson"
DRIVE_DATA = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")
#: The six drives with a frame index (plan section 2.2).
INDEXED_RUNS: tuple[str, ...] = tuple(f"run_20260908_{t}" for t in (
    "155910", "161422", "170849", "173809", "183538", "190548"))
#: Camera geometry of the upright frame (section 2.2): `fx` is the config's assumed value,
#: the principal point moves to the upright image's centre, the hood starts at 1,010 px.
FX_PX: float = 800.0
CX_PX: float = 360.0
HOOD_LINE_Y_PX: float = 1010.0
UPRIGHT_SHAPE: tuple[int, int, int] = (1280, 720, 3)
VEHICLE_CLASSES: tuple[int, ...] = (2, 3, 5, 7)
CONFIDENCE: float = 0.30
NMS_IOU: float = 0.5
#: The deployment's class width priors (`config.yaml distance.class_widths_m`).
CLASS_WIDTHS_M: dict[int, float] = {2: 1.8, 3: 0.8, 5: 2.55, 7: 2.5}
#: Used for a drive whose detections are too few to fit (fewer than 30 usable boxes).
FALLBACK_HORIZON_Y_PX: float = 825.0
FALLBACK_CAMERA_HEIGHT_M: float = 1.25


def _perception():
    if str(JETSON) not in sys.path:
        sys.path.insert(0, str(JETSON))
    from perception.detector import Detection
    from perception.distance import DistanceEstimator
    from perception.tracker import IouTracker

    return Detection, DistanceEstimator, IouTracker


# ------------------------------------------------------------- index and times


@dataclass
class FrameIndex:
    positions: dict[int, int]
    lines: int
    duplicates: int
    unparseable: int


def read_index(run_dir: Path) -> FrameIndex:
    """`frame_id -> AVI position`, the first position of a duplicated id kept."""
    positions: dict[int, int] = {}
    lines = duplicates = unparseable = 0
    for line in (Path(run_dir) / "video_index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            row = json.loads(line)
            frame_id, pos = int(row["frame_id"]), int(row["pos"])
        except (ValueError, KeyError, TypeError):
            unparseable += 1
            continue
        if frame_id in positions:
            duplicates += 1
            continue
        positions[frame_id] = pos
    return FrameIndex(positions, lines, duplicates, unparseable)


def frame_times(run_dir: Path) -> dict[int, float]:
    """`frame_id -> capture time (s)` from the first tick naming the frame."""
    times: dict[int, float] = {}
    with (Path(run_dir) / "metadata.jsonl").open() as handle:
        for line in handle:
            if '"tick"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") != "tick" or row.get("frame_id") is None:
                continue
            frame_id = int(row["frame_id"])
            if frame_id not in times and row.get("t_capture_mono_ns") is not None:
                times[frame_id] = float(row["t_capture_mono_ns"]) / 1e9
    return times


@dataclass(frozen=True)
class PlannedFrame:
    frame_id: int
    pos: int
    t: float


@dataclass
class FramePlan:
    frames: list[PlannedFrame]
    indexed: int
    no_tick: int
    duplicates: int
    unparseable: int
    index_lines: int

    def describe(self) -> dict[str, int]:
        return {"indexed_frames": self.indexed, "planned_frames": len(self.frames),
                "no_tick": self.no_tick, "duplicate_ids": self.duplicates,
                "unparseable_index_lines": self.unparseable, "index_lines": self.index_lines}


def frame_plan(run_dir: Path) -> FramePlan:
    """The indexed frames that have a tick, in AVI order, with their capture times."""
    index = read_index(run_dir)
    times = frame_times(run_dir)
    frames, missing = [], 0
    for frame_id, pos in sorted(index.positions.items(), key=lambda item: item[1]):
        if frame_id not in times:
            missing += 1
            continue
        frames.append(PlannedFrame(frame_id, pos, times[frame_id]))
    return FramePlan(frames, len(index.positions), missing, index.duplicates,
                     index.unparseable, index.lines)


def rotate(raw: np.ndarray) -> np.ndarray:
    """The landscape buffer turned ninety degrees clockwise to the upright scene."""
    import cv2

    return cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)


def iter_frames(run_dir: Path, frames: Sequence[PlannedFrame]) -> Iterator[tuple[PlannedFrame, np.ndarray]]:
    """Each planned frame with its upright image, reading the AVI forward."""
    import cv2

    capture = cv2.VideoCapture(str(Path(run_dir) / "video.avi"))
    try:
        wanted = sorted(frames, key=lambda f: f.pos)
        if not wanted:
            return
        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0].pos)
        position = wanted[0].pos
        for frame in wanted:
            while position < frame.pos:
                if not capture.grab():
                    return
                position += 1
            ok, raw = capture.read()
            position += 1
            if not ok:
                return
            yield frame, rotate(raw)
    finally:
        capture.release()


# -------------------------------------------------------------------- detection


Box = tuple[np.ndarray, float, int]  # (xyxy, confidence, class)


@dataclass
class FrameDetections:
    frame_id: int
    pos: int
    t: float
    boxes: list[Box] = field(default_factory=list)


def drop_hood_boxes(boxes: Iterable[Box], hood_line_y: float = HOOD_LINE_Y_PX) -> list[Box]:
    """The device detector's rule: a box whose centre lies below the hood line is the ego
    vehicle's hood or a reflection on it, and is dropped."""
    return [b for b in boxes if (b[0][1] + b[0][3]) / 2.0 < hood_line_y]


def detect(model, image: np.ndarray, imgsz: int, conf: float = CONFIDENCE,
           iou: float = NMS_IOU) -> list[Box]:
    """YOLOv8n on CPU through ultralytics, vehicle classes only, hood boxes dropped."""
    result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou,
                           classes=list(VEHICLE_CLASSES), device="cpu", verbose=False)[0]
    boxes = []
    for b in result.boxes:
        cls = int(b.cls.item())
        if cls in VEHICLE_CLASSES:
            boxes.append((b.xyxy[0].cpu().numpy().astype(np.float32), float(b.conf.item()), cls))
    return drop_hood_boxes(boxes)


# ------------------------------------------------------------ track and measure


def ground_plane_m(box: np.ndarray, horizon_y: float, camera_height: float,
                   fx: float = FX_PX, hood: float = HOOD_LINE_Y_PX,
                   max_range: float = 120.0) -> float | None:
    """`DistanceEstimator`'s ground-plane distance, or None where it is not valid."""
    bottom = float(box[3])
    dv = bottom - horizon_y
    if bottom >= hood or dv <= 2.0:
        return None
    z = camera_height * fx / dv
    return z if 1.0 <= z <= max_range else None


def width_prior_m(box: np.ndarray, cls: int, fx: float = FX_PX) -> float:
    return fx * CLASS_WIDTHS_M.get(int(cls), 1.8) / max(float(box[2] - box[0]), 1.0)


def track_and_measure(frames: Sequence[FrameDetections], horizon_y: float,
                      camera_height: float, fx: float = FX_PX, cx: float = CX_PX,
                      hood: float = HOOD_LINE_Y_PX) -> list[dict[str, Any]]:
    """The deployment's tracker and distance estimator over one drive's detections, each
    frame at its own capture time. Returns a `frame` record per frame and a `det` record
    per tracked vehicle per frame (the tracker reports confirmed, visible tracks only)."""
    Detection, DistanceEstimator, IouTracker = _perception()
    tracker = IouTracker()
    estimator = DistanceEstimator(fx_px=fx, cx_px=cx, horizon_y_px=horizon_y,
                                  camera_height_m=camera_height, method="ground_plane",
                                  contact_cutoff_y_px=hood)
    out: list[dict[str, Any]] = []
    for frame in frames:
        detections = [Detection(xyxy=np.asarray(b, dtype=np.float32), conf=float(c), cls=int(k))
                      for b, c, k in frame.boxes]
        tracks = tracker.update(detections, frame.t)
        vehicles = estimator.update(tracks, frame.t)
        out.append({"type": "frame", "frame_id": frame.frame_id, "pos": frame.pos,
                    "t": frame.t, "detections": len(detections), "tracked": len(vehicles)})
        for v in vehicles:
            box = np.asarray(v.xyxy, dtype=np.float32)
            out.append({
                "type": "det", "frame_id": frame.frame_id, "t": frame.t,
                "track_id": int(v.track_id), "cls": int(v.cls), "conf": float(v.conf),
                "box": [round(float(x), 2) for x in box],
                "raw_ground_m": ground_plane_m(box, horizon_y, camera_height, fx, hood),
                "width_prior_m": width_prior_m(box, v.cls, fx),
                "ema_m": float(v.distance_m), "method": v.method,
                "closing_rate_mps": float(v.rel_speed_mps),
                "closing_valid": bool(v.rel_speed_valid), "lateral_m": float(v.lateral_m)})
    return out


# ------------------------------------------------------------------- horizon fit


def horizon_fit(boxes: Iterable[Box], frame_height: int = UPRIGHT_SHAPE[0],
                frame_width: int = UPRIGHT_SHAPE[1], min_conf: float = 0.5,
                min_boxes: int = 30) -> dict[str, Any]:
    """Horizon row and camera height by `calibration/auto_horizon.py`'s method.

    `v_bottom = y_horizon + h_cam * (w_px / W_class)`: a Theil-Sen line of the box bottom
    on the width over the class width prior; the intercept is the horizon row and the
    slope the camera height. Boxes are filtered as `auto_horizon.py` filters them (small,
    touching the frame border, in the top quarter, below the confidence). With fewer than
    `min_boxes` usable boxes the fit is refused and the fallback values are returned with
    `source = "fallback"`.
    """
    if str(JETSON) not in sys.path:
        sys.path.insert(0, str(JETSON))
    from calibration.auto_horizon import theil_sen

    xs, vs = [], []
    for box, conf, cls in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if conf < min_conf or w < 14 or h < 12 or cls not in CLASS_WIDTHS_M:
            continue
        if x1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2 or y2 < 0.25 * frame_height:
            continue
        xs.append(w / CLASS_WIDTHS_M[cls])
        vs.append(y2)
    if len(xs) < min_boxes:
        return {"source": "fallback", "horizon_y_px": FALLBACK_HORIZON_Y_PX,
                "camera_height_m": FALLBACK_CAMERA_HEIGHT_M, "boxes": len(xs),
                "reason": f"{len(xs)} usable boxes, fewer than {min_boxes}"}
    x, v = np.asarray(xs), np.asarray(vs)
    height, horizon = theil_sen(x, v)
    residual = v - (horizon + height * x)
    return {"source": "fit", "horizon_y_px": float(horizon), "camera_height_m": float(height),
            "boxes": len(xs), "residual_mad_px": float(np.median(np.abs(residual))),
            "plausible_height": bool(0.8 <= height <= 2.5)}
