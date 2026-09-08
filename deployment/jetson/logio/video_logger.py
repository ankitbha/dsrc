"""Raw frame recording for offline replay.

Writes the frames exactly as the pipeline consumed them (pre-annotation),
so replay_demo.py can re-run perception on identical input.

**Video position is NOT tick index, and this file used to claim it was.** The
previous docstring said "video frame index == tick index == metadata record
index", which the drop path five lines below can break: the queue is bounded and
`write` discards on `queue.Full`. After one drop every later frame is off by one,
silently, and every offline re-analysis keyed on position is wrong from there --
detections attributed to the wrong GPS fix, distances to the wrong speed.

So position is not relied on. Each written frame appends a line to
`video_index.jsonl` carrying its video position and the `frame_id` it came from,
written by the same thread that writes the frame, so a dropped frame produces no
line and no claim. Alignment is then a fact in the artifact rather than an
inference from a counter, and a drop is visible as a gap in `frame_id` even to a
reader who never sees this code.

MJPG/AVI is used instead of H.264 because the pip OpenCV wheel has no
hardware encoder access; MJPG encode of 720p costs ~4-6 ms on a worker
thread, which never blocks the pipeline.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VideoLogger:
    #: The sidecar naming which `frame_id` each video position holds.
    INDEX_NAME = "video_index.jsonl"

    #: How long `close` waits to hand the writer thread its sentinel before giving
    #: up on it. Bounded because the queue is: a blocking put against a full queue
    #: whose drainer has died never returns, and `close` is on the teardown path
    #: that writes `summary.json`. A hung teardown loses the whole drive's record,
    #: which is a far worse outcome than a video missing its last frames.
    CLOSE_TIMEOUT_S = 5.0

    def __init__(self, run_dir: Path, fps: float, fourcc: str = "MJPG") -> None:
        self.path = str(Path(run_dir) / "video.avi")
        self.index_path = str(Path(run_dir) / self.INDEX_NAME)
        self.fps = max(1.0, fps)
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        #: `(frame, frame_id)`; `None` is the shutdown sentinel.
        self._queue: queue.Queue[tuple[np.ndarray, object] | None] = queue.Queue(maxsize=60)
        self.dropped_frames = 0
        self.written_frames = 0
        #: Every `frame_id` this logger was handed but discarded. Kept, not just
        #: counted: knowing three frames were lost is far less use than knowing
        #: WHICH, because a reader can then exclude those ticks rather than
        #: distrust the whole run.
        self.dropped_frame_ids: list[object] = []
        #: Set when the writer thread stopped draining and `close` had to give up on
        #: it, or when the loop itself failed. Either means the video is short and
        #: the index is the only trustworthy account of what it holds.
        self.close_blocked = False
        self.writer_error: str | None = None
        self._index = open(self.index_path, "a", buffering=1)
        self._thread = threading.Thread(target=self._loop, name="video-log", daemon=True)
        self._thread.start()

    def write(self, frame_bgr: np.ndarray, frame_id: object = None) -> None:
        try:
            self._queue.put_nowait((frame_bgr, frame_id))
        except queue.Full:
            self.dropped_frames += 1
            self.dropped_frame_ids.append(frame_id)

    def to_record(self) -> dict[str, object]:
        """What a reader needs to trust, or distrust, position-based alignment.

        Surfaced because the old counter was incremented and read by nothing: a
        run could drop frames and say so nowhere, leaving "no drops" and "drops
        not reported" indistinguishable.
        """
        return {
            "written_frames": self.written_frames,
            "dropped_frames": self.dropped_frames,
            "dropped_frame_ids": list(self.dropped_frame_ids),
            "index": self.INDEX_NAME,
            "position_is_tick_index": self.dropped_frames == 0,
            "close_blocked": self.close_blocked,
            "writer_error": self.writer_error,
        }

    def close(self) -> None:
        # Bounded, for the reason on CLOSE_TIMEOUT_S. Found by a test that retired
        # the drainer and filled the queue: it hung, which is exactly what a run
        # would have done at teardown with a dead writer thread.
        try:
            self._queue.put(None, timeout=self.CLOSE_TIMEOUT_S)
        except queue.Full:
            self.close_blocked = True
        self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()
        try:
            self._index.close()
        except Exception:
            pass

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, frame_id = item
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
                )
            try:
                self._writer.write(frame)
            except Exception as exc:  # noqa: BLE001 - the thread must not die
                # Recorded and kept draining. A drainer that dies fills the queue,
                # and a full queue is what turns a video problem into a lost
                # summary.json. Dropping frames is survivable; wedging teardown is
                # not.
                self.writer_error = f"{type(exc).__name__}: {exc}"
                continue
            # The index line is written by the thread that wrote the frame, and
            # after the write, so a line exists only for a frame that is really in
            # the file. Writing it in `write()` would name frames that were then
            # dropped, which is the failure this sidecar exists to prevent.
            self._index.write(json.dumps({"pos": self.written_frames, "frame_id": frame_id}) + "\n")
            self.written_frames += 1
