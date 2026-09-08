"""The video logger's alignment record.

There was no test file for this class at all, and the thing it now asserts is what
the class previously got wrong: its docstring claimed "video frame index == tick
index == metadata record index" while the drop path below it could break that
silently, and the drop counter was read by nothing. A drive on 2026-09-08 recorded
1,632 ticks whose frames could only be tied to telemetry by position, with no way to
check whether that position was trustworthy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from logio.video_logger import VideoLogger


def _frame(value: int = 0) -> np.ndarray:
    return np.full((16, 24, 3), value % 256, dtype=np.uint8)


def _index_lines(run_dir: Path) -> list[dict]:
    path = run_dir / VideoLogger.INDEX_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _settle(logger: VideoLogger, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and logger.written_frames < expected:
        time.sleep(0.01)


class TestIndexNamesWhatIsInTheFile:

    def test_each_written_frame_gets_a_line_naming_its_frame_id(self, tmp_path):
        logger = VideoLogger(tmp_path, fps=5.0)
        try:
            for fid in (7, 8, 9):
                logger.write(_frame(fid), frame_id=fid)
            _settle(logger, 3)
        finally:
            logger.close()

        lines = _index_lines(tmp_path)
        assert [ln["frame_id"] for ln in lines] == [7, 8, 9]
        # Position is the video's own index, which is what a reader seeks by.
        assert [ln["pos"] for ln in lines] == [0, 1, 2]

    def test_a_dropped_frame_leaves_no_line_and_is_named(self, tmp_path):
        # The case the old code could not report. The drainer is stopped first so the
        # queue genuinely fills, rather than racing the writer thread and hoping.
        logger = VideoLogger(tmp_path, fps=5.0)
        logger._queue.put(None)          # retire the writer thread
        logger._thread.join(timeout=5.0)
        assert not logger._thread.is_alive()

        capacity = logger._queue.maxsize
        for fid in range(capacity):
            logger.write(_frame(fid), frame_id=fid)
        assert logger.dropped_frames == 0

        logger.write(_frame(999), frame_id=999)
        assert logger.dropped_frames == 1
        # WHICH frame was lost, not merely how many: a reader can exclude that tick
        # instead of distrusting the whole drive.
        assert logger.dropped_frame_ids == [999]

        record = logger.to_record()
        assert record["dropped_frames"] == 1
        assert record["dropped_frame_ids"] == [999]
        assert record["position_is_tick_index"] is False

        # The defect this test found: `close` used a blocking put on a bounded
        # queue, so with the drainer gone it never returned -- the run would hang
        # at teardown and never write summary.json. It must now give up and say so.
        logger.CLOSE_TIMEOUT_S = 0.2
        started = time.monotonic()
        logger.close()
        assert time.monotonic() - started < 5.0, "close() hung on a full queue"
        assert logger.close_blocked is True
        assert logger.to_record()["close_blocked"] is True

    def test_position_stays_contiguous_when_frame_ids_are_not(self, tmp_path):
        # The whole reason the sidecar exists. A gap in `frame_id` beside contiguous
        # `pos` is what tells an offline reader that position cannot be used as a
        # tick index -- visible in the artifact, without reading any code.
        logger = VideoLogger(tmp_path, fps=5.0)
        try:
            for fid in (1, 2, 5, 6):     # 3 and 4 never reached the logger
                logger.write(_frame(fid), frame_id=fid)
            _settle(logger, 4)
        finally:
            logger.close()

        lines = _index_lines(tmp_path)
        assert [ln["pos"] for ln in lines] == [0, 1, 2, 3]
        assert [ln["frame_id"] for ln in lines] == [1, 2, 5, 6]

    def test_a_clean_run_says_position_is_the_tick_index(self, tmp_path):
        logger = VideoLogger(tmp_path, fps=5.0)
        try:
            for fid in range(4):
                logger.write(_frame(fid), frame_id=fid)
            _settle(logger, 4)
            record = logger.to_record()
        finally:
            logger.close()
        assert record["dropped_frames"] == 0
        assert record["position_is_tick_index"] is True
        assert record["written_frames"] == 4

    def test_a_frame_id_is_optional_so_old_callers_still_work(self, tmp_path):
        logger = VideoLogger(tmp_path, fps=5.0)
        try:
            logger.write(_frame(1))
            _settle(logger, 1)
        finally:
            logger.close()
        assert _index_lines(tmp_path) == [{"pos": 0, "frame_id": None}]
