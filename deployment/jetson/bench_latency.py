#!/usr/bin/env python3
"""End-to-end latency benchmark (plan_deployment.md Task P8 numbers).

Runs the full pipeline - real TensorRT detector included - over a video
file or synthetic frames, and prints/saves the per-stage latency table
used in the paper's deployment section.

Synthetic mode notes: YOLO will (correctly) not detect the painted
rectangles in synthetic frames, so detector timing is realistic but the
downstream stages would idle. --fake-detections therefore injects a
scripted traffic scene (leader closing at -2 m/s plus adjacent-lane
vehicles, geometry produced by the same pinhole model the distance
estimator inverts), exercising tracking/observation/policy on every
tick while the detector still runs on the frame for timing. With a real
dashcam clip via --source file:..., omit --fake-detections.

  python3 bench_latency.py --ticks 300
  python3 bench_latency.py --source file:dashcam.mp4 --no-fake-detections
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception.detector import Detection  # noqa: E402
from run_demo import build_components, load_config  # noqa: E402
from sensors.camera_stream import Frame  # noqa: E402
from sensors.gps_reader import GpsFix  # noqa: E402
from sensors.time_sync import now_mono, now_wall  # noqa: E402


class SyntheticScenario:
    """Deterministic traffic scene projected through the pinhole model.

    Vehicles are (initial_distance_m, lateral_m, closing_rate_mps); boxes
    are generated exactly where the ground-plane distance estimator will
    invert them back, so synthetic runs also sanity-check the geometry.
    """

    DEFAULT_VEHICLES = (
        (40.0, 0.0, -2.0),   # leader, ego lane, closing
        (25.0, -3.7, 0.5),   # left lane
        (55.0, 3.7, -1.0),   # right lane
        (70.0, 0.0, 0.0),    # far ego lane
    )

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fx: float = 800.0,
        cx: float = 640.0,
        horizon_y: float = 360.0,
        camera_height: float = 1.25,
        vehicles=DEFAULT_VEHICLES,
    ) -> None:
        self.width, self.height = width, height
        self.fx, self.cx = fx, cx
        self.horizon_y = horizon_y
        self.camera_height = camera_height
        self.vehicles = [list(v) for v in vehicles]
        self._background = self._make_background()

    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[: int(self.horizon_y)] = (150, 120, 90)      # sky
        img[int(self.horizon_y):] = (60, 60, 60)         # road
        for lane_edge in (-1.5 * 3.7, -0.5 * 3.7, 0.5 * 3.7, 1.5 * 3.7):
            p_near = self._project(6.0, lane_edge)
            p_far = self._project(120.0, lane_edge)
            cv2.line(img, p_near, p_far, (200, 200, 200), 2)
        return img

    def _project(self, z_m: float, x_m: float) -> tuple[int, int]:
        u = self.cx + x_m * self.fx / z_m
        v = self.horizon_y + self.camera_height * self.fx / z_m
        return int(u), int(v)

    def bbox(self, z_m: float, x_m: float, veh_width_m: float = 1.8) -> np.ndarray:
        w_px = self.fx * veh_width_m / z_m
        h_px = 0.85 * w_px
        u, v_bottom = self._project(z_m, x_m)
        return np.array(
            [u - w_px / 2, v_bottom - h_px, u + w_px / 2, v_bottom], dtype=np.float32
        )

    def step(self, dt_s: float) -> tuple[np.ndarray, list[Detection]]:
        frame = self._background.copy()
        detections = []
        for veh in self.vehicles:
            veh[0] = max(8.0, veh[0] + veh[2] * dt_s)
            box = self.bbox(veh[0], veh[1])
            x1, y1, x2, y2 = (int(c) for c in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 30, 30), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
            detections.append(Detection(xyxy=box, conf=0.9, cls=2))
        return frame, detections


def bench_gps_fix() -> GpsFix:
    return GpsFix(
        valid=True, lat=40.7440, lon=-74.0324, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=10.0,
        utc_epoch_s=now_wall(), t_mono=now_mono(), t_wall=now_wall(),
    )


def markdown_table(stats: dict[str, dict[str, float]], fps: float, ticks: int) -> str:
    lines = [
        f"| stage | mean (ms) | p50 (ms) | p95 (ms) |",
        f"|---|---|---|---|",
    ]
    order = ["detect_ms", "track_ms", "observe_ms", "policy_ms", "jetson_ms", "e2e_ms"]
    label = {
        "detect_ms": "detection (TRT FP16, incl. pre/post)",
        "track_ms": "tracking + distance",
        "observe_ms": "observation build + encode",
        "policy_ms": "actor + advisory decode",
        "jetson_ms": "ON-JETSON (arrival -> advisory)",
        "e2e_ms": "END-TO-END (capture -> advisory)",
    }
    for key in order:
        s = stats[key]
        lines.append(f"| {label[key]} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |")
    lines.append(f"\npipeline throughput: **{fps:.1f} FPS** over {ticks} ticks")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(JETSON_DIR / "config.yaml"))
    parser.add_argument("--source", help="file:<path> to bench on a real clip (default: synthetic)")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--fake-detections", dest="fake", action="store_true", default=True)
    parser.add_argument("--no-fake-detections", dest="fake", action="store_false")
    parser.add_argument("--out", default=str(JETSON_DIR / "models" / "bench_results"))
    args = parser.parse_args()

    config = load_config(args.config)
    # bench never opens real sensors
    camera, gps, pipeline, actor = build_components(config, args.source or "file:/dev/null", use_gps=False)
    print(f"[bench] warmup: {pipeline.detector.warmup():.1f} ms/frame "
          f"(trained policy: {actor.is_trained})")

    cam_cfg = config["camera"]
    scenario = SyntheticScenario(
        width=cam_cfg["width"], height=cam_cfg["height"], fx=cam_cfg["fx_px"],
        cx=cam_cfg["cx_px"], horizon_y=cam_cfg["horizon_y_px"],
        camera_height=cam_cfg["camera_height_m"],
    )

    video = None
    if args.source:
        video = cv2.VideoCapture(args.source[len("file:"):])
        if not video.isOpened():
            print(f"[bench] cannot open {args.source}")
            return 1

    t_start = time.monotonic()
    for i in range(args.ticks):
        if video is not None:
            ok, image = video.read()
            if not ok:
                print(f"[bench] video ended at tick {i}")
                break
            _, fake_dets = scenario.step(1 / 30)
        else:
            image, fake_dets = scenario.step(1 / 30)
        frame = Frame(image=image, frame_id=i, t_mono=now_mono(), t_wall=now_wall())
        pipeline.step(
            frame,
            bench_gps_fix(),
            detections_override=fake_dets if args.fake else None,
            run_detector_with_override=True,
        )
    wall = time.monotonic() - t_start

    ticks = pipeline._tick_counter
    fps = ticks / wall if wall > 0 else 0.0
    stats = pipeline.stats.snapshot()
    table = markdown_table(stats, fps, ticks)
    print("\n" + table + "\n")

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_prefix) + ".json", "w") as f:
        json.dump({"stats": stats, "fps": fps, "ticks": ticks,
                   "source": args.source or "synthetic",
                   "fake_detections": args.fake}, f, indent=2)
    with open(str(out_prefix) + ".md", "w") as f:
        f.write(table + "\n")
    print(f"[bench] wrote {out_prefix}.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
