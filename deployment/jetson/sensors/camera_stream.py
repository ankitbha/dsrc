"""Threaded camera capture with a latest-frame-wins policy.

Latency design: the capture thread always overwrites a single slot; the
pipeline asks for the newest frame and frames that were never consumed
are silently dropped. Queueing frames would only add stale latency -
an advisory computed on an old frame is worse than a skipped frame.

Sources
  "0", "1", ...      USB UVC camera via V4L2 (/dev/videoN)
  "file:<path>"      video file (replay / benchmarks); optionally paced
                     to the file's native FPS so timing behaves like live
  "csi:<id>"         CSI camera via nvarguscamerasrc GStreamer pipeline.
                     Requires OpenCV with GStreamer support (NOT the pip
                     wheel). Kept here so the config story is complete.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from sensors.time_sync import TimebaseStamp, now_mono, now_wall


@dataclass
class Frame:
    image: np.ndarray  # BGR
    frame_id: int
    t_mono: float      # capture time (monotonic), basis for all latency math
    t_wall: float
    # Set only when the capture happened on another device and t_mono above is a
    # converted or proxied value. None means this device captured it and t_mono
    # is exact, which is the case for every local camera.
    timebase: "TimebaseStamp | None" = None


def reopen_past_failure(capture: cv2.VideoCapture, open_fn) -> cv2.VideoCapture | None:
    """Recover a file capture whose decoder state was poisoned mid-stream
    (e.g. one bad VP9 cluster): reopen via open_fn and seek one frame past
    the failure. Returns the fresh capture, or None when the failure
    position looks like genuine end of stream (caller should stop)."""
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if total <= 0 or pos <= 0 or pos >= total - 5:
        return None
    capture.release()
    fresh = open_fn()
    if fresh is None or not fresh.isOpened():
        return None
    fresh.set(cv2.CAP_PROP_POS_FRAMES, pos + 1)
    return fresh


def _csi_gstreamer_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )


class CameraStream:
    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fourcc: str = "MJPG",
        pace_file_source: bool = True,
        max_file_recoveries: int = 64,
    ) -> None:
        self.source = str(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.pace_file_source = pace_file_source
        self.max_file_recoveries = max_file_recoveries
        self.is_file = self.source.startswith("file:")

        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._running = False
        self._frame_counter = 0
        self._drop_counter = 0
        self._file_recoveries = 0
        self._last_consumed_id = -1
        self.end_of_stream = False

    # -- lifecycle -------------------------------------------------------

    def start(self) -> "CameraStream":
        self._capture = self._open()
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(
                f"camera source '{self.source}' failed to open "
                "(check cable, /dev/video*, or the file path)"
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()

    def _open(self) -> cv2.VideoCapture:
        if self.is_file:
            return cv2.VideoCapture(self.source[len("file:"):])
        if self.source.startswith("csi:"):
            pipeline = _csi_gstreamer_pipeline(
                int(self.source[len("csi:"):]), self.width, self.height, self.fps
            )
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(int(self.source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the V4L2 queue shallow
        return cap

    # -- capture thread ---------------------------------------------------

    def _loop(self) -> None:
        assert self._capture is not None
        file_dt = None
        if self.is_file and self.pace_file_source:
            src_fps = self._capture.get(cv2.CAP_PROP_FPS) or self.fps
            file_dt = 1.0 / max(src_fps, 1.0)
        while self._running:
            tick = time.monotonic()
            ok, image = self._capture.read()
            if not ok:
                if self.is_file:
                    if self._recover_file_stream():
                        continue
                    self.end_of_stream = True
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.05)  # transient USB hiccup; retry
                continue
            frame = Frame(
                image=image,
                frame_id=self._frame_counter,
                t_mono=now_mono(),
                t_wall=now_wall(),
            )
            with self._cond:
                if self._latest is not None and self._latest.frame_id > self._last_consumed_id:
                    self._drop_counter += 1
                self._latest = frame
                self._frame_counter += 1
                self._cond.notify_all()
            if file_dt is not None:
                remaining = file_dt - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)

    def _recover_file_stream(self) -> bool:
        """Bounded reopen-past-failure for file sources, so a truly
        truncated file still terminates (each retry advances >= 1 frame)."""
        assert self._capture is not None
        if self._file_recoveries >= self.max_file_recoveries:
            return False
        fresh = reopen_past_failure(self._capture, self._open)
        if fresh is None:
            return False
        self._file_recoveries += 1
        self._capture = fresh
        return True

    # -- consumers ---------------------------------------------------------

    def wait_for_fresh(self, timeout: float = 1.0) -> Frame | None:
        """Block until a frame newer than the last consumed one arrives."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while (
                self._latest is None or self._latest.frame_id <= self._last_consumed_id
            ) and not self.end_of_stream:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._latest is None or self._latest.frame_id <= self._last_consumed_id:
                return None  # end of stream
            self._last_consumed_id = self._latest.frame_id
            return self._latest

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    @property
    def dropped_frames(self) -> int:
        return self._drop_counter

    @property
    def file_recoveries(self) -> int:
        return self._file_recoveries
