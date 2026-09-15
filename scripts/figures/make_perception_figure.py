"""Rebuild the on-vehicle perception view from a recorded drive frame.

The drive delivered camera frames rotated ninety degrees, so the on-device
detector returned almost nothing (paper Table IV). This reproduces what the
perception stage would have produced on those frames: rotate clockwise, detect,
then run the deployment's own tracker and distance estimator over a short run-up
so track ids and the distance EMA are warm at the display frame.

Detection here is YOLOv8n on CPU through ultralytics rather than the device's
TensorRT FP16 engine. Same weights, different executor; the boxes are the
detector's, and the ids, distances and lane assignment are the deployment's own
code (perception/tracker.py, perception/distance.py).

Ego speed and the end-to-end latency in the panel are read from the tick the
device actually recorded for this frame. No advisory speed is drawn: the
campaign ran an untrained policy, so its recommendation is not a policy output.

Usage: make_teaser.py <run> <frame_id> [horizon_y]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")
DATA = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")
sys.path.insert(0, str(REPO / "deployment" / "jetson"))

from perception.detector import Detection  # noqa: E402
from perception.distance import DistanceEstimator  # noqa: E402
from perception.tracker import IouTracker  # noqa: E402

OUT = Path(__file__).resolve().parent

# run_config.yaml, adjusted for the rotated frame. The buffer is 1280x720 with
# the scene turned ninety degrees, so the upright image is 720 wide by 1280 tall:
# fx is unchanged (square pixels) and cx moves to the new image centre. The
# horizon row is measured off the upright frame, where the config's 360 belongs
# to the landscape buffer that never held an upright scene.
FX_PX = 800.0
CX_PX = 360.0
CAMERA_HEIGHT_M = 1.25
HOOD_LINE_Y_PX = 1010.0
CONF = 0.30
IMGSZ = 1280
VEHICLE_CLASSES = (2, 3, 5, 7)
LANE_WIDTH_M = 3.7
LABEL_SCALE = 0.95
LABEL_THICK = 2
MAX_LABELS = 3
RUN_UP = 14

AMBER = (60, 190, 245)
CYAN = (235, 200, 90)
WHITE = (240, 240, 240)
RED = (70, 70, 235)
DIM = (170, 170, 170)


def frame_positions(rd: Path) -> dict[int, int]:
    return {r["frame_id"]: r["pos"] for r in map(json.loads, (rd / "video_index.jsonl").open())}


def tick_for_frame(rd: Path, frame_id: int) -> dict | None:
    for line in (rd / "metadata.jsonl").open():
        rec = json.loads(line)
        if rec.get("type") == "tick" and rec.get("frame_id") == frame_id:
            return rec
    return None


def detect(model, image: np.ndarray) -> list[Detection]:
    res = model.predict(image, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    return [
        Detection(xyxy=b.xyxy[0].cpu().numpy().astype(np.float32),
                  conf=float(b.conf.item()), cls=int(b.cls.item()))
        for b in res.boxes
        if int(b.cls.item()) in VEHICLE_CLASSES
    ]


def build(rd: Path, frame_id: int, horizon: float):
    from ultralytics import YOLO

    model = YOLO(str(OUT / "yolov8n.pt"))
    pos = frame_positions(rd)[frame_id]
    cap = cv2.VideoCapture(str(rd / "video.avi"))
    tracker = IouTracker()
    dist = DistanceEstimator(
        fx_px=FX_PX, cx_px=CX_PX, horizon_y_px=horizon,
        camera_height_m=CAMERA_HEIGHT_M, method="ground_plane",
        contact_cutoff_y_px=HOOD_LINE_Y_PX,
    )
    upright, vehicles = None, []
    for i, p in enumerate(range(pos - RUN_UP, pos + 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, raw = cap.read()
        if not ok:
            continue
        upright = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        t = i / 30.0
        vehicles = dist.update(tracker.update(detect(model, upright), t), t)
    cap.release()
    return upright, vehicles


def render(image: np.ndarray, vehicles, tick: dict, horizon: float):
    """Crop to the road, then place the panel and the labels in that frame.

    Cropping first is what lets the panel reserve its own rectangle: the label
    placer treats it as occupied, so a label can never land underneath it.
    Sizes here are chosen for a half-column panel, where the earlier 0.52 text
    scale printed at about 2 pt.
    """
    top = max(0, int(horizon) - 265)
    bottom = min(image.shape[0], int(HOOD_LINE_Y_PX) + 30)
    out = image[top:bottom, :].copy()

    leader = None
    for v in vehicles:
        if abs(v.lateral_m) < LANE_WIDTH_M / 2 and (leader is None or v.distance_m < leader.distance_m):
            leader = v

    g = tick.get("gps") or {}
    ego = g.get("speed_mps")
    e2e = tick.get("e2e_ms")
    lines = [
        (f"ego {ego * 2.23694:.0f} mph" if ego else "ego --", WHITE, 1.15),
        (f"tracked {len(vehicles)}", DIM, 0.78),
        (f"leader gap {leader.distance_m:.0f} m" if leader else "no leader in lane", DIM, 0.78),
        (f"end to end {e2e:.0f} ms" if e2e else "", DIM, 0.78),
    ]
    lines = [ln for ln in lines if ln[0]]
    pad, lh = 16, 34
    panel_h = pad * 2 + int(sum(lh * (1.35 if sc > 0.9 else 1.0) for _, _, sc in lines)) + 26
    panel_w = 340
    # The panel sits over the hood, which is the only part of the frame that
    # carries no road. Anywhere higher would cover the traffic it describes.
    py = out.shape[0] - panel_h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, py), (panel_w, out.shape[0]), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
    y = py + pad + 34
    for text, colour, scale in lines:
        cv2.putText(out, text, (pad, y), cv2.FONT_HERSHEY_DUPLEX, scale, colour, 2, cv2.LINE_AA)
        y += int(lh * (1.35 if scale > 0.9 else 1.0))
    cv2.putText(out, "ADVISORY ONLY - NOT FOR VEHICLE CONTROL", (pad, out.shape[0] - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, RED, 1, cv2.LINE_AA)

    placed: list[tuple[int, int, int, int]] = [(0, py, panel_w, out.shape[0])]

    def free_slot(x: int, y: int, w: int, h: int) -> int:
        for _ in range(16):
            box = (x, y - h, x + w, y)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                return y
            y -= h + 4
        return y

    nearest = sorted(vehicles, key=lambda v: v.distance_m)[:MAX_LABELS]
    if leader is not None and leader not in nearest:
        nearest = [leader] + nearest[:-1]

    for v in sorted(vehicles, key=lambda v: -v.distance_m):
        x1, y1, x2, y2 = [int(round(c)) for c in v.xyxy]
        y1, y2 = y1 - top, y2 - top
        colour = AMBER if v is leader else CYAN
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 4 if v is leader else 3, cv2.LINE_AA)
        if v not in nearest:
            continue
        label = f"#{v.track_id} {v.distance_m:.0f} m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, LABEL_SCALE, LABEL_THICK)
        lw, lh_ = tw + 14, th + 14
        lx = min(x1, out.shape[1] - lw - 2)          # never run off the right edge
        ly = free_slot(lx, max(lh_, y1), lw, lh_)
        placed.append((lx, ly - lh_, lx + lw, ly))
        if ly < y1 - 2:
            cv2.line(out, (lx + lw // 2, ly), (lx + lw // 2, y1), colour, 2, cv2.LINE_AA)
        cv2.rectangle(out, (lx, ly - lh_), (lx + lw, ly), colour, -1)
        cv2.putText(out, label, (lx + 7, ly - 7), cv2.FONT_HERSHEY_DUPLEX, LABEL_SCALE,
                    (25, 25, 25), LABEL_THICK, cv2.LINE_AA)
    return out, leader


if __name__ == "__main__":
    run = sys.argv[1]
    fid = int(sys.argv[2])
    horizon = float(sys.argv[3]) if len(sys.argv) > 3 else 825.0
    rd = DATA / run
    img, vehicles = build(rd, fid, horizon)
    tick = tick_for_frame(rd, fid) or {}
    print(f"{run} frame {fid} horizon={horizon}: {len(vehicles)} tracked")
    for v in sorted(vehicles, key=lambda v: v.distance_m):
        print(f"  #{v.track_id} cls={v.cls} conf={v.conf:.2f} d={v.distance_m:6.1f} m "
              f"lat={v.lateral_m:+6.2f} m {v.method}")
    g = tick.get("gps") or {}
    print(f"  recorded ego={g.get('speed_mps')} m/s e2e={tick.get('e2e_ms')} ms "
          f"device_detections={tick.get('n_detections')}")
    out, leader = render(img, vehicles, tick, horizon)
    dest = OUT / f"annotated_{run}_{fid}.jpg"
    cv2.imwrite(str(dest), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("wrote", dest)
