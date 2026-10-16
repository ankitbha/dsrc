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


# ------------------------------------------------------------- the error tables


#: Section 2.3: 10 m bins of EMA distance, to the estimator's 120 m maximum range.
BIN_M: float = 10.0
N_BINS: int = 12
#: `R_max` is the largest bin with at least this many observations and this rate.
R_MAX_MIN_OBSERVATIONS: int = 50
R_MAX_MIN_P: float = 0.2
NOISE_WINDOW: int = 5
SLOPE_WINDOW: int = 8
MAD_SCALE: float = 1.4826
#: A drive whose median frame gap is below this is in the 0.20 s group.
INTERVAL_SPLIT_S: float = 0.4


def _bin(d: float, bin_m: float = BIN_M, n_bins: int = N_BINS) -> int | None:
    if d is None or not np.isfinite(d) or d < 0:
        return None
    index = int(d // bin_m)
    return index if index < n_bins else None


def _tracks(records: Iterable[dict]) -> dict[tuple, list[dict]]:
    """Detections grouped by `(run, track_id)`, in time order."""
    out: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("type") == "det":
            out.setdefault((r.get("run"), r["track_id"]), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["t"])
    return out


def _frame_times_by_run(records: Iterable[dict]) -> dict[Any, np.ndarray]:
    out: dict[Any, list[float]] = {}
    for r in records:
        if r.get("type") == "frame":
            out.setdefault(r.get("run"), []).append(float(r["t"]))
    return {run: np.array(sorted(ts)) for run, ts in out.items()}


def detection_rate_table(records: Sequence[dict]) -> dict[str, list]:
    """Per bin, detected and missed frames of the tracks the tracker held.

    For a track with at least 3 detections every frame strictly between its first and
    last detection is an opportunity; the first and last are detections by construction
    and would bias the rate up, so they are not counted. A missed frame takes the distance
    interpolated linearly in time between the detections around it. A vehicle never
    detected is invisible here, and
    the tracker's 10-frame maximum age splits longer gaps, so `p` is the detection rate of a
    vehicle the tracker holds.
    """
    detected = [0] * N_BINS
    missed = [0] * N_BINS
    frames = _frame_times_by_run(records)
    for (run, _), rows in _tracks(records).items():
        if len(rows) < 3:
            continue
        times = np.array([r["t"] for r in rows])
        dist = np.array([r["ema_m"] for r in rows])
        for d in dist[1:-1]:
            b = _bin(d)
            if b is not None:
                detected[b] += 1
        seen = set(np.round(times, 6))
        all_frames = frames.get(run, np.array([]))
        between = all_frames[(all_frames > times[0]) & (all_frames < times[-1])]
        for t in between:
            if round(float(t), 6) in seen:
                continue
            b = _bin(float(np.interp(t, times, dist)))
            if b is not None:
                missed[b] += 1
    opportunities = [d + m for d, m in zip(detected, missed)]
    return {"detected": detected, "missed": missed, "opportunities": opportunities,
            "p": [d / n if n else None for d, n in zip(detected, opportunities)]}


def _local_residuals(t: np.ndarray, d: np.ndarray,
                     window: int = NOISE_WINDOW) -> list[tuple[int, float]]:
    """`(centre index, standardised residual)` for every window of `window` consecutive
    values that are all finite: the centre's residual about the window's least-squares
    line divided by `sqrt(1 - h)`, `h` the centre's leverage, so that it has the noise's
    variance (an unscaled centre residual of a 5-point line has 0.8 of it)."""
    out = []
    half = window // 2
    for i in range(half, len(d) - half):
        tt, dd = t[i - half:i + half + 1], d[i - half:i + half + 1]
        if not np.all(np.isfinite(dd)):
            continue
        tc = tt - tt.mean()
        sxx = float((tc * tc).sum())
        if sxx <= 0:
            continue
        slope = float((tc * (dd - dd.mean())).sum() / sxx)
        fitted = dd.mean() + slope * tc[half]
        leverage = 1.0 / window + tc[half] ** 2 / sxx
        out.append((i, float((d[i] - fitted) / np.sqrt(1.0 - leverage))))
    return out


def _mad_sigma(values: list[float]) -> float | None:
    if not values:
        return None
    v = np.asarray(values)
    return float(MAD_SCALE * np.median(np.abs(v - np.median(v))))


def distance_noise_table(records: Sequence[dict]) -> dict[str, list]:
    """Per bin of EMA distance, `1.4826 MAD` of the raw ground-plane and of the EMA
    distances' residuals about a local linear fit over 5 consecutive detections."""
    raw: list[list[float]] = [[] for _ in range(N_BINS)]
    ema: list[list[float]] = [[] for _ in range(N_BINS)]
    for rows in _tracks(records).values():
        t = np.array([r["t"] for r in rows])
        e = np.array([r["ema_m"] for r in rows], dtype=float)
        g = np.array([np.nan if r.get("raw_ground_m") is None else r["raw_ground_m"]
                      for r in rows], dtype=float)
        for target, series in ((ema, e), (raw, g)):
            for centre, residual in _local_residuals(t, series):
                b = _bin(e[centre])
                if b is not None:
                    target[b].append(residual)
    return {"n_raw": [len(v) for v in raw], "sigma_raw_m": [_mad_sigma(v) for v in raw],
            "n_ema": [len(v) for v in ema], "sigma_ema_m": [_mad_sigma(v) for v in ema]}


def closing_rate_noise_table(records: Sequence[dict]) -> dict[str, list]:
    """Per bin, `1.4826 x median |difference of successive window slopes| / sqrt(2)`.

    The closing rate of a track is the least-squares slope of its EMA distance over
    non-overlapping windows of 8 detections; a pair of successive windows is binned at the
    mean of their mean distances. Real relative acceleration between windows also enters
    the difference, so this bounds the estimator's noise from above.
    """
    diffs: list[list[float]] = [[] for _ in range(N_BINS)]
    for rows in _tracks(records).values():
        t = np.array([r["t"] for r in rows])
        e = np.array([r["ema_m"] for r in rows], dtype=float)
        slopes, means = [], []
        for start in range(0, len(e) - SLOPE_WINDOW + 1, SLOPE_WINDOW):
            tt, ee = t[start:start + SLOPE_WINDOW], e[start:start + SLOPE_WINDOW]
            tc = tt - tt.mean()
            sxx = float((tc * tc).sum())
            if sxx <= 0:
                continue
            slopes.append(float((tc * (ee - ee.mean())).sum() / sxx))
            means.append(float(ee.mean()))
        for k in range(1, len(slopes)):
            b = _bin((means[k] + means[k - 1]) / 2.0)
            if b is not None:
                diffs[b].append(slopes[k] - slopes[k - 1])
    sigma = [None if not v else float(MAD_SCALE * np.median(np.abs(v)) / np.sqrt(2.0))
             for v in diffs]
    return {"n": [len(v) for v in diffs], "sigma_v_mps": sigma}


def confidence_table(records: Sequence[dict]) -> dict[str, list]:
    """Every detection's confidence, by bin of EMA distance, stored whole for resampling."""
    values: list[list[float]] = [[] for _ in range(N_BINS)]
    for r in records:
        if r.get("type") == "det":
            b = _bin(r["ema_m"])
            if b is not None:
                values[b].append(round(float(r["conf"]), 4))
    return {"n": [len(v) for v in values], "values": values}


def bias_ratio_table(records: Sequence[dict]) -> dict[str, list]:
    """Per bin, the median ratio of ground-plane to width-prior distance. Both scale with
    `fx`, so this checks the horizon and height fit against the 1.8 m car prior, not the
    absolute scale."""
    ratios: list[list[float]] = [[] for _ in range(N_BINS)]
    for r in records:
        if r.get("type") == "det" and r.get("raw_ground_m") and r.get("width_prior_m"):
            b = _bin(r["ema_m"])
            if b is not None:
                ratios[b].append(float(r["raw_ground_m"]) / float(r["width_prior_m"]))
    return {"n": [len(v) for v in ratios],
            "median": [float(np.median(v)) if v else None for v in ratios]}


def first_detection_table(records: Sequence[dict]) -> dict[str, Any]:
    """First-detection EMA distances of the tracks whose distance decreases over their life."""
    first = []
    for rows in _tracks(records).values():
        if len(rows) >= 2 and rows[-1]["ema_m"] < rows[0]["ema_m"]:
            first.append(float(rows[0]["ema_m"]))
    hist = np.histogram(first, bins=N_BINS, range=(0.0, BIN_M * N_BINS))[0].tolist()
    return {"n": len(first), "median_m": float(np.median(first)) if first else None,
            "p90_m": float(np.percentile(first, 90)) if first else None, "histogram": hist}


def r_max(detection: dict[str, list], bin_m: float = BIN_M) -> float | None:
    """The upper edge of the largest bin with at least 50 observations and `p >= 0.2`."""
    best = None
    for b, (n, p) in enumerate(zip(detection["opportunities"], detection["p"])):
        if n >= R_MAX_MIN_OBSERVATIONS and p is not None and p >= R_MAX_MIN_P:
            best = b
    return None if best is None else (best + 1) * bin_m


def interval_group(times: Sequence[float]) -> str:
    """"0.20" or "0.67": the drive's frame-interval group by its median gap."""
    gaps = np.diff(np.asarray(sorted(times), dtype=float))
    return "0.20" if len(gaps) and float(np.median(gaps)) < INTERVAL_SPLIT_S else "0.67"


def error_tables(records: Sequence[dict]) -> dict[str, Any]:
    """Every table of section 2.3 for one group of drives' records."""
    detection = detection_rate_table(records)
    return {
        "bins_m": [[b * BIN_M, (b + 1) * BIN_M] for b in range(N_BINS)],
        "detection": detection, "distance_noise": distance_noise_table(records),
        "closing_rate_noise": closing_rate_noise_table(records),
        "confidence": confidence_table(records), "bias_ratio": bias_ratio_table(records),
        "first_detection": first_detection_table(records), "R_max_m": r_max(detection),
    }
