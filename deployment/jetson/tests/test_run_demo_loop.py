"""The tick loop's stopping conditions, which nothing tested.

`run_demo.py` had no test file at all, and it has now produced two defects for that
reason: a `build_components` NameError that killed live mode, and a deadline that
governed only the iterations which got a frame. This covers the loop's termination
directly, with everything below it stubbed, because the defect is in the loop's
control flow and not in what it drives.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import pytest

import run_demo


class SilentCamera:
    """Connected, healthy, and producing nothing. Not end-of-stream."""

    def __init__(self) -> None:
        self.end_of_stream = False
        self.dropped_frames = 0
        self.file_recoveries = 0
        self.source = "stub"
        self.asked = 0

    def wait_for_fresh(self, timeout: float = 1.0):
        self.asked += 1
        time.sleep(min(timeout, 0.01))
        return None

    def stop(self) -> None:
        pass


def loop_for(camera, *, duration_s: float, max_ticks: int = 0) -> float:
    """Run the loop's termination logic against a camera, and time it.

    A transcription of `run_live`'s `_tick_loop` down to its stopping conditions.
    Not the function itself -- constructing it needs a detector, a policy bundle and
    a config -- so this asserts the conditions rather than the whole run, and the
    mutation pin is what ties the two together.
    """
    import threading

    stop = threading.Event()
    deadline = time.monotonic() + duration_s if duration_s else None
    started = time.monotonic()
    while not stop.is_set():
        if deadline and time.monotonic() >= deadline:
            break
        frame = camera.wait_for_fresh(timeout=1.0)
        if frame is None:
            if camera.end_of_stream:
                break
            continue
    return time.monotonic() - started


class TestTheDriveEnds:

    def test_the_deadline_governs_a_camera_that_produces_nothing(self):
        # The state the fix is about: connected, `end_of_stream` False, no frames.
        # Reachable from an undecodable JPEG (swallowed per frame, so permanent), a
        # dead camera pipeline on a healthy session, and the redial gap.
        camera = SilentCamera()
        elapsed = loop_for(camera, duration_s=0.3)
        assert elapsed < 3.0, "the drive did not end; --duration-s never fired"
        assert camera.asked > 0, "the loop never polled, so it proved nothing"

    def test_the_source_reads_the_deadline_before_the_frame_not_after(self):
        # The transcription above is only worth what its fidelity is worth, so the
        # ordering is asserted against the real file: the deadline check must come
        # BEFORE `wait_for_fresh`, which is the whole difference.
        import inspect

        source = inspect.getsource(run_demo.run_live)
        check = source.index("if deadline and time.monotonic() >= deadline:")
        wait = source.index("frame = camera.wait_for_fresh(")
        assert check < wait, (
            "the deadline is checked after the frame again, so a silent camera "
            "runs the drive forever"
        )

    def test_end_of_stream_still_ends_the_drive_immediately(self):
        # The other exit, which must not have been traded away for the first.
        camera = SilentCamera()
        camera.end_of_stream = True
        elapsed = loop_for(camera, duration_s=60.0)
        assert elapsed < 3.0
