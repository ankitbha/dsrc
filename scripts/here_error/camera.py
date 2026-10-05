"""Stage 3: offline vehicle detection on the recorded video.

The recorded frames are rotated 90 degrees anticlockwise, so each is rotated clockwise before
detection. Results are appended one line per frame so an interrupted run resumes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from here_error import params
from here_error.models import Frame, FrameDetections


class CameraError(ValueError):
    pass


@dataclass(frozen=True)
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


#: Takes an upright BGR image of shape (height, width, 3) and returns boxes in pixel coordinates.
Detector = Callable[[np.ndarray], Sequence[Box]]


def upright(image: np.ndarray) -> np.ndarray:
    """Rotate a recorded frame clockwise: a (720, 1280, 3) frame becomes (1280, 720, 3)."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def is_ego_hood(b: Box, width: int, height: int) -> bool:
    """The recording car's own bonnet: a wide box that reaches the bottom of the upright image."""
    return ((b.x2 - b.x1) >= params.EGO_HOOD_MIN_WIDTH_SHARE * width
            and b.y2 >= (1.0 - params.EGO_HOOD_BOTTOM_SHARE) * height)


def vehicle_boxes(boxes: Sequence[Box], width: int, height: int) -> list[Box]:
    """Boxes of vehicle classes at or above the confidence threshold, leaving out the recording car's own bonnet."""
    return [b for b in boxes
            if b.cls in params.VEHICLE_CLASSES and b.conf >= params.YOLO_CONF and not is_ego_hood(b, width, height)]


def has_leader(boxes: Sequence[Box], width: int, height: int) -> bool:
    """A vehicle whose box centre is in the middle LEADER_CENTRE_SHARE of the width and whose box is tall enough."""
    half = params.LEADER_CENTRE_SHARE * width / 2.0
    for b in vehicle_boxes(boxes, width, height):
        cx = (b.x1 + b.x2) / 2.0
        if abs(cx - width / 2.0) <= half and (b.y2 - b.y1) >= params.LEADER_MIN_HEIGHT_SHARE * height:
            return True
    return False


def analyse_frame(raw: np.ndarray, detector: Detector) -> tuple[int, bool, float]:
    img = upright(raw)
    boxes = detector(img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return len(vehicle_boxes(boxes, w, h)), has_leader(boxes, w, h), float(gray.mean())


def _cut_partial_last_line(path: Path) -> None:
    """Drop bytes after the last newline: a killed process can leave half a line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        path.write_bytes(data[:cut])


def load_detections(path: Path | str) -> list[FrameDetections]:
    path = Path(path)
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(FrameDetections(r["run"], r["pos"], r["frame_id"], r["utc_s"], r["n_vehicles"],
                                       r["leader"], r["brightness"]))
    return out


@dataclass(frozen=True)
class CameraStats:
    n_new: int
    n_skipped: int
    seconds: float


def run_camera(
    run: str,
    video_path: Path | str,
    frames: Sequence[Frame],
    detector: Detector,
    out_path: Path | str,
    limit: int | None = None,
) -> CameraStats:
    """Detect on frames in video order, appending to `out_path`; positions already there are skipped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _cut_partial_last_line(out_path)
    done = {d.pos for d in load_detections(out_path)}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CameraError(f"cannot open {video_path}")
    try:
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_video != len(frames):
            raise CameraError(f"{video_path} holds {n_video} frames but the index lists {len(frames)}")
        last = len(frames) if limit is None else min(limit, len(frames))
        n_new = n_skip = 0
        t0 = time.monotonic()
        with out_path.open("a") as out:
            for pos in range(last):
                if pos in done:
                    cap.grab()
                    n_skip += 1
                    continue
                ok, raw = cap.read()
                if not ok:
                    raise CameraError(f"{video_path}: could not decode frame {pos}")
                n, leader, bright = analyse_frame(raw, detector)
                f = frames[pos]
                out.write(json.dumps({
                    "run": run, "pos": f.pos, "frame_id": f.frame_id, "utc_s": f.capture_utc_s,
                    "n_vehicles": n, "leader": leader, "brightness": bright,
                }, sort_keys=True) + "\n")
                out.flush()
                n_new += 1
        return CameraStats(n_new, n_skip, time.monotonic() - t0)
    finally:
        cap.release()


def make_yolo_detector(weights: Path | str, device: str | None = None) -> Detector:
    """A YOLO detector over the vehicle classes. Weights are fetched to `weights` when absent."""
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    weights = Path(weights)
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(weights))
    if not weights.exists():
        raise CameraError(f"weights not found and not downloadable at {weights}")
    model = YOLO(str(weights))

    def detect(img: np.ndarray) -> list[Box]:
        res = model.predict(img, conf=params.YOLO_CONF, classes=list(params.VEHICLE_CLASSES),
                            device=device, verbose=False)[0]
        return [Box(int(c), float(p), *map(float, xyxy))
                for c, p, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]

    return detect
