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


class _FakeEstimate:
    def __init__(self, estimate_id: int) -> None:
        self.estimate_id = estimate_id

    def to_record(self) -> dict:
        return {"estimate_id": self.estimate_id, "offset_ns": 123}


class _FakeRoundTripEstimator:
    """`TimebaseEstimator.estimate` is a property; the fake mirrors that shape
    with a plain attribute, since a test double has no need of the real
    property machinery to be read the same way."""

    def __init__(self, estimate=None, *, usable: bool = True, why_not_usable: str | None = None) -> None:
        self.estimate = estimate
        self.usable = usable
        self._why_not_usable = why_not_usable

    def why_not_usable(self) -> str | None:
        return self._why_not_usable


class _FakeOneWayEstimator:
    """`OneWayEstimator.estimate()` is a method, not a property -- the
    asymmetry `_log_timebase_estimates` itself has to read across."""

    def __init__(self, estimate=None, *, usable: bool = True, why_not_usable: str | None = None) -> None:
        self._estimate = estimate
        self.usable = usable
        self._why_not_usable = why_not_usable

    def estimate(self):
        return self._estimate

    def why_not_usable(self) -> str | None:
        return self._why_not_usable


class _FakeSession:
    def __init__(self, session_id) -> None:
        self.session_id = session_id


class _FakePhone:
    def __init__(self, *, round_trip=None, one_way=None, session_id="s1") -> None:
        self.round_trip_estimator = _FakeRoundTripEstimator(round_trip)
        self.estimator = _FakeOneWayEstimator(one_way)
        self.session = None if session_id is None else _FakeSession(session_id)


class _FakeLogger:
    def __init__(self) -> None:
        self.lines: list[dict] = []

    def write(self, record: dict) -> None:
        self.lines.append(record)


class TestTimebaseEstimateLogging:
    """A `timebase_estimate` line per source, written only when the estimate
    actually changes, so an offline reader can re-derive a conversion against
    the estimate that was current at the time."""

    def test_no_estimate_on_either_side_writes_nothing(self):
        logger = _FakeLogger()
        phone = _FakePhone()
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))
        assert logger.lines == []

    def test_a_new_estimate_on_each_side_is_logged_once_each(self):
        logger = _FakeLogger()
        phone = _FakePhone(round_trip=_FakeEstimate(1), one_way=_FakeEstimate(9))
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        assert len(logger.lines) == 2
        by_source = {line["source"]: line for line in logger.lines}
        assert by_source["round_trip"]["type"] == "timebase_estimate"
        assert by_source["round_trip"]["estimate_id"] == 1
        assert by_source["one_way"]["estimate_id"] == 9
        assert "t_wall" in by_source["round_trip"]
        assert ids == {"round_trip": 1, "one_way": 9}

    def test_the_same_estimate_id_is_not_logged_twice(self):
        logger = _FakeLogger()
        phone = _FakePhone(round_trip=_FakeEstimate(1))
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))
        assert len(logger.lines) == 1

    def test_a_changed_estimate_id_is_logged_again(self):
        logger = _FakeLogger()
        phone = _FakePhone(round_trip=_FakeEstimate(1))
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        phone.round_trip_estimator.estimate = _FakeEstimate(2)
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        assert len(logger.lines) == 2
        assert logger.lines[-1]["estimate_id"] == 2

    def test_usable_and_why_not_usable_come_from_the_estimator_not_the_estimate(self):
        # `_FakeEstimate.to_record()` carries no `usable` field at all -- if
        # the line got these from the estimate, the key would be missing.
        logger = _FakeLogger()
        phone = _FakePhone(
            round_trip=_FakeEstimate(1), one_way=None,
        )
        phone.round_trip_estimator.usable = False
        phone.round_trip_estimator._why_not_usable = "only 1 samples in the offset window"
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        line = logger.lines[0]
        assert line["usable"] is False
        assert line["why_not_usable"] == "only 1 samples in the offset window"

    def test_the_session_id_is_carried_onto_the_line(self):
        logger = _FakeLogger()
        phone = _FakePhone(round_trip=_FakeEstimate(1), session_id="peer-7")
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        assert logger.lines[0]["session_id"] == "peer-7"

    def test_no_session_is_a_null_session_id_not_a_crash(self):
        logger = _FakeLogger()
        phone = _FakePhone(round_trip=_FakeEstimate(1), session_id=None)
        ids: dict[str, int | None] = {"round_trip": None, "one_way": None}
        run_demo._log_timebase_estimates(logger, phone, ids, run_demo.tick_session_id(phone))

        assert logger.lines[0]["session_id"] is None


class TestTheSensingRecordIsBuiltByTickOutcome:
    """`record["sensing"]` used to be assembled inline in `run_live`, which is
    what left task 34's round-1 defect -- the emitted shape was the one thing
    no test read. `on_tick`'s return value is exercised directly in
    `test_sensing_loop.py`; this pins that `run_live` actually calls it rather
    than reverting to its own copy of the shape, which needs none of the
    camera/policy/config machinery a full run does.
    """

    def test_run_live_builds_the_sensing_record_from_outcome_to_record(self):
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert 'record["sensing"] = outcome.to_record()' in source


class TestThermalSamplerIntegration:
    """`run_live`'s four thermal integration points: constructing the sampler
    under its config gate, writing the tick block, writing the summary block,
    and stopping it in the `finally`. Like `TestTheSensingRecordIsBuiltByTickOutcome`
    above, none of these need the detector/policy bundle/config/camera
    machinery a full run does, so this asserts the literal wiring rather than
    driving one.
    """

    def test_the_sampler_is_constructed_under_its_config_gate(self):
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert 'if config["logio"]["thermal"]:' in source
        assert "thermal_sampler = ThermalSampler(" in source

    def test_the_tick_writes_the_samplers_own_block(self):
        # Also the boundary of the mutation named "the tick's thermal block
        # reads sensing.reference instead of the sampler": that substitution
        # can only be written here, where `outcome` is in scope -- thermal.py
        # itself has no `sensing` concept to read from.
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert 'record["thermal"] = thermal_sampler.latest()' in source

    def test_the_summary_writes_the_samplers_own_record(self):
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert 'summary["thermal"] = thermal_sampler.to_record(' in source

    def test_the_sampler_is_stopped_before_its_record_is_read(self):
        # Stopped before `to_record()`, not after: reading the snapshot while
        # the thread might still be mid-write would undercount the
        # `thermal_sample` lines already on disk by one.
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert "thermal_sampler.stop()" in source
        assert source.index("thermal_sampler.stop()") < source.index(
            'summary["thermal"] = thermal_sampler.to_record('
        )


class TestTickSessionId:
    """`run_live` used to read `phone.session.session_id` twice -- once for the
    tick record, once inside `_log_timebase_estimates` -- and a rebind between
    the two reads could put a different session's id on each. The fix pulls
    the read out into this one function, called once per tick, so there is
    only one place left for it to disagree with itself. Testing it directly,
    rather than through `run_live`, needs none of the detector/policy
    bundle/config/camera/GPS machinery a full run does.
    """

    def test_reads_the_current_sessions_id(self):
        phone = _FakePhone(session_id="peer-7")
        assert run_demo.tick_session_id(phone) == "peer-7"

    def test_no_session_on_the_phone_is_none(self):
        phone = _FakePhone(session_id=None)
        assert run_demo.tick_session_id(phone) is None

    def test_no_phone_at_all_is_none(self):
        assert run_demo.tick_session_id(None) is None
