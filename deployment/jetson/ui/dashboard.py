"""Driver-facing dashboard: annotated camera view + advisory panel.

Rendering is decoupled from the pipeline: the pipeline thread publishes
the latest Tick into a slot and this window (on the main thread, where
OpenCV GUI calls must live) draws whatever is newest at its own pace.
A slow display can therefore never add latency to the advisory loop.

ADVISORY ONLY: this display is for demo/logging visibility. It must not
be followed while driving, and the system is never connected to vehicle
actuation (see plan_deployment.md safety constraints).
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from perception.detector import COCO_VEHICLE_NAMES
from pipeline import Tick

PANEL_W = 380
GREEN = (80, 220, 80)
YELLOW = (60, 200, 240)
GRAY = (160, 160, 160)
RED = (60, 60, 230)
WHITE = (235, 235, 235)
BLUE = (220, 160, 60)


class TickSlot:
    """Latest-value handoff between the pipeline thread and the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tick: Tick | None = None
        self._image: np.ndarray | None = None

    def publish(self, tick: Tick, image: np.ndarray) -> None:
        with self._lock:
            self._tick = tick
            self._image = image

    def latest(self) -> tuple[Tick | None, np.ndarray | None]:
        with self._lock:
            return self._tick, self._image


def annotate_frame(
    image: np.ndarray,
    tick: Tick,
    horizon_y: float | None = None,
    lane_width_px_hint: float | None = None,
) -> np.ndarray:
    out = image.copy()
    if horizon_y is not None and 0 < horizon_y < out.shape[0]:
        cv2.line(out, (0, int(horizon_y)), (out.shape[1], int(horizon_y)), BLUE, 1, cv2.LINE_AA)
    for v in tick.vehicles:
        x1, y1, x2, y2 = (int(c) for c in v.xyxy)
        lane = round(v.lateral_m / 3.7)
        color = GREEN if lane == 0 else (YELLOW if abs(lane) == 1 else GRAY)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        rel = f" {v.rel_speed_mps:+.1f}m/s" if v.rel_speed_valid else ""
        label = f"#{v.track_id} {COCO_VEHICLE_NAMES.get(v.cls, '?')} {v.distance_m:.0f}m{rel}"
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def render_dashboard(
    image: np.ndarray,
    tick: Tick,
    stats: dict[str, dict[str, float]],
    policy_trained: bool,
    extra_lines: list[str] | None = None,
    horizon_y: float | None = None,
) -> np.ndarray:
    view = annotate_frame(image, tick, horizon_y)
    h = view.shape[0]
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)

    y = 30

    def put(text: str, color=WHITE, scale=0.55, dy=26, thick=1) -> None:
        nonlocal y
        cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += dy

    adv = tick.advisory
    obs = tick.obs_result.obs
    put("DSRC EDGE PROTOTYPE", BLUE, 0.6, 24, 2)
    put("ADVISORY ONLY - DO NOT FOLLOW", RED, 0.45, 26)
    if not policy_trained:
        put("[ UNTRAINED POLICY - BRING-UP ]", RED, 0.5, 28, 2)
    y += 6
    put(f"Recommended: {adv.recommended_speed_display:5.0f} {adv.units}", GREEN, 0.8, 36, 2)
    put(f"Current:     {adv.current_speed_display:5.0f} {adv.units}", WHITE, 0.8, 38, 2)
    put(f"Lane:    {adv.lane_text}", YELLOW, 0.55, 26)
    put(f"Merge:   {adv.merge_text}", WHITE, 0.5, 24)
    put(f"Headway: {adv.headway_target_s:.1f} s target", WHITE, 0.5, 28)
    y += 4
    put(f"Traffic: {adv.traffic_text}   vehicles: {obs['active_vehicle_count_local']}", WHITE, 0.5, 24)
    lg = obs["leader_gap"]
    put(f"Leader:  {'%.0f m' % lg if np.isfinite(lg) else '--'}", WHITE, 0.5, 24)
    put(f"Confidence: {adv.confidence_label} ({tick.policy.confidence:.2f})", WHITE, 0.5, 28)
    y += 4
    # `or {}`, not `get(..., {})`: the key is present and None for an absent
    # series, so the default never applies. link_ms is None on every local run.
    e2e = stats.get("e2e_ms") or {}
    put(f"FPS {tick.fps:4.1f}   e2e {tick.e2e_ms:5.1f} ms", GREEN if tick.e2e_ms < 200 else RED, 0.5, 24)
    put(f"e2e p50/p95 {e2e.get('p50', 0):5.1f}/{e2e.get('p95', 0):5.1f} ms", GRAY, 0.45, 22)
    put(f"det {tick.stage_ms['detect']:4.1f}  obs {tick.stage_ms['observe']:4.2f}  pol {tick.stage_ms['policy_advisory']:4.2f} ms", GRAY, 0.45, 26)
    gps = tick.gps
    gps_ok = gps.valid
    put(
        f"GPS {'FIX' if gps_ok else 'NO FIX'}  sats {gps.num_sats}"
        + (f"  {gps.speed_mps * 2.237:.0f} mph" if gps_ok and np.isfinite(gps.speed_mps) else ""),
        GREEN if gps_ok else RED,
        0.5,
        24,
    )
    if tick.n_peers:
        put(f"V2V peers: {tick.n_peers}", BLUE, 0.5, 24)
    for line in extra_lines or []:
        put(line, GRAY, 0.45, 22)

    return np.hstack([view, panel])


class DashboardWindow:
    WINDOW = "dsrc-advisory"

    def __init__(self, fullscreen: bool = False) -> None:
        self.fullscreen = fullscreen
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            self._set_fullscreen(True)

    def _set_fullscreen(self, on: bool) -> None:
        cv2.setWindowProperty(
            self.WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL,
        )
        self.fullscreen = on

    def show(self, canvas: np.ndarray) -> str | None:
        """Display and poll keys. Returns 'quit' on q/ESC."""
        cv2.imshow(self.WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("f"):
            self._set_fullscreen(not self.fullscreen)
        return None

    def close(self) -> None:
        cv2.destroyAllWindows()
