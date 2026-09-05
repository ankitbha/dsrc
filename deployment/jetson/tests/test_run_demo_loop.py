"""The tick loop's stopping conditions, which nothing tested.

`run_demo.py` had no test file at all, and it has now produced two defects for that
reason: a `build_components` NameError that killed live mode, and a deadline that
governed only the iterations which got a frame. This covers the loop's termination
directly, with everything below it stubbed, because the defect is in the loop's
control flow and not in what it drives.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from types import SimpleNamespace

import pytest

import run_demo
from logio.failure_log import BLIND_EPISODE_MIN_S


# ---------------------------------------------------------------------------
# AST helpers for the two structural pins below (M6): a character-window
# check ("is `raise` within 80 characters of this call") passes a guard that
# keeps the word `raise` nearby without ever reaching it -- `if 0: raise` --
# because the window has no idea what actually executes. These walk the real
# parse tree instead, so what is asserted is control flow, not text.
# ---------------------------------------------------------------------------


def _call_name(node: ast.AST) -> str | None:
    """The attribute or bare name a `Call` node invokes, or `None`."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _calls(stmt: ast.AST, name: str) -> bool:
    """Whether `name` is called anywhere inside `stmt` (at any nesting)."""
    return any(_call_name(node) == name for node in ast.walk(stmt))


def _find_except_handler(tree: ast.AST, exception_name: str) -> ast.ExceptHandler | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) \
                and node.type.id == exception_name:
            return node
    return None


def _is_frame_is_none(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
        and test.left.id == "frame" and len(test.ops) == 1 and isinstance(test.ops[0], ast.Is)
        and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
    )


def _is_camera_end_of_stream(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Attribute) and test.attr == "end_of_stream"
        and isinstance(test.value, ast.Name) and test.value.id == "camera"
    )


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
    """`run_live`'s thermal integration points into the tick loop and its own
    `finally`: constructing the sampler under its config gate, writing the
    tick block, and handing the sampler to `_thermal_summary` for the summary
    block. Like `TestTheSensingRecordIsBuiltByTickOutcome` above, none of
    these need the detector/policy bundle/config/camera machinery a full run
    does, so this asserts the literal wiring rather than driving one.
    `_thermal_summary`'s own stop-before-read behaviour is driven directly,
    below, rather than by where its two calls sit in the source.
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

    def test_the_summary_is_assigned_from_thermal_summary(self):
        import inspect

        source = inspect.getsource(run_demo.run_live)
        assert "thermal_summary = _thermal_summary(thermal_sampler, stats_sampler)" in source
        assert 'summary["thermal"] = thermal_summary' in source


class _OrderRecordingSampler:
    """A stand-in that records which of its two methods ran, and in which
    order -- the runtime fact `_thermal_summary` exists to guarantee, where a
    pin on the position of the two calls' text in the source file cannot tell
    a real reordering from a second, redundant `stop()` placed earlier in the
    file that changes nothing about which call actually runs first.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stop(self) -> None:
        self.calls.append("stop")

    def to_record(self, *, jtop_available: bool | None = None) -> dict:
        self.calls.append("to_record")
        return {"jtop_available": jtop_available, "calls_seen_by_to_record": list(self.calls)}


class TestThermalSummaryStopsBeforeReading:
    """`_thermal_summary` is `run_live`'s `finally`-block logic pulled out so
    it can be driven directly: stopping the sampler joins its thread, closing
    the window where a pass has counted itself under its lock but not yet
    written its `thermal_sample` line, which reading the record first would
    leave open.
    """

    def test_no_sampler_is_a_null_summary(self):
        assert run_demo._thermal_summary(None, None) is None

    def test_stop_runs_before_to_record(self):
        sampler = _OrderRecordingSampler()
        result = run_demo._thermal_summary(sampler, None)
        assert sampler.calls == ["stop", "to_record"]
        assert result["calls_seen_by_to_record"] == ["stop", "to_record"]

    def test_jtop_available_comes_from_the_stats_sampler_when_present(self):
        sampler = _OrderRecordingSampler()
        stats_sampler = SimpleNamespace(available=True)
        result = run_demo._thermal_summary(sampler, stats_sampler)
        assert result["jtop_available"] is True

    def test_jtop_available_is_none_with_no_stats_sampler(self):
        sampler = _OrderRecordingSampler()
        result = run_demo._thermal_summary(sampler, None)
        assert result["jtop_available"] is None


class _RecordingFailures:
    """Stands in for a `FailureSampler` where only the two direct-notification
    entry points matter -- what `worker()` and `_tick_loop()` actually call."""

    def __init__(self) -> None:
        self.no_frame_calls: list[bool] = []
        self.exceptions: list[BaseException] = []

    def note_no_frame(self, *, end_of_stream: bool = False) -> None:
        self.no_frame_calls.append(end_of_stream)

    def note_pipeline_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)


def worker_for(tick_loop, failures, stop) -> None:
    """A transcription of `run_live`'s `worker()`, down to the exception
    handling -- constructing the real one needs the whole run. The mutation
    pin is what ties the two together, as `loop_for` above already does for
    the deadline.
    """
    try:
        tick_loop()
    except BaseException as exc:  # noqa: BLE001
        if failures is not None:
            failures.note_pipeline_exception(exc)
        raise
    finally:
        stop.set()


class TestWorkerRecordsAndReraises:
    """Behaviour change 1: `worker()` gains an `except BaseException` that
    records and re-raises. What must NOT change: the exception still reaches
    the caller, and `stop.set()` still runs in the `finally` -- both
    asserted together, because recording it and swallowing it are different
    defects and a test that only checked one would miss the other."""

    def test_the_exception_still_propagates_and_stop_is_still_set(self):
        import threading

        stop = threading.Event()
        failures = _RecordingFailures()

        def raising_tick_loop():
            raise RuntimeError("pipeline.step blew up")

        with pytest.raises(RuntimeError, match="pipeline.step blew up"):
            worker_for(raising_tick_loop, failures, stop)

        assert stop.is_set(), "the finally must still run stop.set() on a raise"
        assert len(failures.exceptions) == 1
        assert isinstance(failures.exceptions[0], RuntimeError)

    def test_a_clean_loop_records_nothing(self):
        import threading

        stop = threading.Event()
        failures = _RecordingFailures()
        worker_for(lambda: None, failures, stop)
        assert failures.exceptions == []
        assert stop.is_set()

    def test_no_failures_sampler_still_lets_the_exception_through(self):
        # `failures` can be None (the sampler is disabled in config) --
        # `worker()` must not crash trying to record onto nothing.
        import threading

        stop = threading.Event()
        with pytest.raises(RuntimeError):
            worker_for(lambda: (_ for _ in ()).throw(RuntimeError("x")), None, stop)
        assert stop.is_set()

    def test_run_live_wires_the_except_and_the_reraise(self):
        # M6: the previous version of this test asserted `"raise" in
        # source[note_at:note_at+80]` -- a character window, not a
        # behaviour. A guard that keeps the word `raise` inside that window
        # without ever reaching it (`if 0: raise`) swallows the exception and
        # still passes it. This parses `run_live` and asserts the handler's
        # own LAST statement is a bare `raise`, which that guard fails: its
        # last statement is the `if 0: raise`, not a `Raise` node.
        import inspect

        source = inspect.getsource(run_demo.run_live)
        tree = ast.parse(source)
        handler = _find_except_handler(tree, "BaseException")
        assert handler is not None, "no `except BaseException` handler in run_live"
        assert _calls(handler, "note_pipeline_exception"), (
            "the handler must call failures.note_pipeline_exception"
        )
        last = handler.body[-1]
        assert isinstance(last, ast.Raise) and last.exc is None, (
            "the handler's last statement must be a bare `raise`, immediately "
            f"reraising what it just recorded -- got {ast.dump(last)}"
        )


class TestBlindTickWiring:
    """Behaviour change 2: the tick loop counts the branch it already takes.
    No new predicate -- `frame is None` and `camera.end_of_stream` are both
    evaluated on this path already."""

    def test_run_live_calls_note_no_frame_before_the_end_of_stream_check(self):
        # M6: same treatment as the exception handler above -- this asserts
        # the actual statement order inside the `if frame is None:` block's
        # AST, not the order the three anchor strings happen to appear in
        # the source text.
        import inspect

        source = inspect.getsource(run_demo.run_live)
        tree = ast.parse(source)
        frame_none_if = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.If) and _is_frame_is_none(n.test)),
            None,
        )
        assert frame_none_if is not None, "no `if frame is None:` block in run_live"

        note_index = next(
            (i for i, stmt in enumerate(frame_none_if.body) if _calls(stmt, "note_no_frame")),
            None,
        )
        eos_index = next(
            (i for i, stmt in enumerate(frame_none_if.body)
             if isinstance(stmt, ast.If) and _is_camera_end_of_stream(stmt.test)),
            None,
        )
        assert note_index is not None, "note_no_frame is not called inside `if frame is None:`"
        assert eos_index is not None, "no `if camera.end_of_stream:` inside `if frame is None:`"
        assert note_index < eos_index


class TestWriteLogHealth:
    """`log_health.json`: the metadata logger's own final state, written
    after `close()` so `writer_failure` and `dropped_records` are whatever
    they will ever be (D16)."""

    def test_a_healthy_close_reports_no_failure(self, tmp_path):
        from logio.metadata_logger import MetadataLogger

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logger = MetadataLogger(run_dir)
        logger.write({"type": "tick", "tick_id": 0})
        logger.close()

        run_demo._write_log_health(logger, run_dir)
        health = json.loads((run_dir / "log_health.json").read_text())
        assert health["writer_failure"] is None
        assert health["dropped_records"] == 0
        assert health["thread_alive_at_close"] is False
        assert health["path"] == "metadata.jsonl"
        assert health["bytes_on_disk"] > 0

    def test_a_dead_writer_thread_is_named_not_hidden(self, tmp_path):
        # The field's own failure mode: a full card, a read-only mount --
        # `MetadataLogger._loop` catches `OSError` from the file handle.
        from logio.metadata_logger import MetadataLogger

        class BrokenFile:
            def write(self, _):
                raise OSError(28, "No space left on device")

            def flush(self):
                pass

            def close(self):
                pass

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logger = MetadataLogger(run_dir)
        logger._file = BrokenFile()
        for i in range(10):
            logger.write({"type": "tick", "tick_id": i})
        logger.close()

        run_demo._write_log_health(logger, run_dir)
        health = json.loads((run_dir / "log_health.json").read_text())
        assert health["writer_failure"] is not None
        assert "No space left" in health["writer_failure"]
        assert health["dropped_records"] > 0


class _FiniteCamera:
    """A real `run_live` drive needs a camera; this produces `n_frames` real
    frames -- each with a small sleep standing in for capture latency, so
    the failure sampler's own background thread gets scheduled between them
    -- then signals `end_of_stream` so the drive ends on its own rather than
    on a `--duration-s` deadline."""

    def __init__(self, n_frames: int, *, frame_delay_s: float = 0.05) -> None:
        import numpy as np

        self.dropped_frames = 0
        self.file_recoveries = 0
        self.source = "test"
        self.end_of_stream = False
        self._remaining = n_frames
        self._frame_delay_s = frame_delay_s
        self._image = np.zeros((64, 64, 3), dtype="uint8")
        self._frame_id = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait_for_fresh(self, timeout: float = 1.0):
        from sensors.camera_stream import Frame

        if self._remaining <= 0:
            self.end_of_stream = True
            return None
        time.sleep(self._frame_delay_s)
        self._remaining -= 1
        frame = Frame(
            image=self._image, frame_id=self._frame_id,
            t_mono=time.monotonic(), t_wall=time.time(),
        )
        self._frame_id += 1
        return frame


class _SilentCamera:
    """Never delivers a frame and never sets `end_of_stream`, so the only way
    a `run_live` drive against this camera ends is the `--duration-s`
    deadline. Exercises `note_no_frame` through the real tick loop, not the
    `worker_for` transcription."""

    def __init__(self) -> None:
        self.dropped_frames = 0
        self.file_recoveries = 0
        self.source = "test"
        self.end_of_stream = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait_for_fresh(self, timeout: float = 1.0):
        time.sleep(0.02)
        return None


def _build_real_pipeline(tmp_path):
    """A real `PerceptionPolicyPipeline` with a stub detector, built once for
    every test in this file that drives `run_live` itself rather than a
    transcription of the code inside it."""
    from perception.distance import DistanceEstimator
    from perception.observation_builder import BuilderConfig, ObservationBuilder
    from perception.tracker import IouTracker
    from pipeline import PerceptionPolicyPipeline
    from policy.actor_runtime import ActorRuntime
    from policy.advisory import AdvisoryDecoder
    from policy.export_policy import build_random, export

    class FakeDetector:
        def infer(self, image):
            return []

        def warmup(self, iterations: int = 1) -> float:
            return 0.0

    bundle_prefix = tmp_path / "bundle" / "actor_policy"
    bundle_prefix.parent.mkdir()
    actor_obj, info = build_random(seed=0)
    export(actor_obj, info, str(bundle_prefix))
    actor = ActorRuntime(str(bundle_prefix))

    pipeline = PerceptionPolicyPipeline(
        detector=FakeDetector(),
        tracker=IouTracker(min_hits=2),
        distance=DistanceEstimator(
            fx_px=800.0, cx_px=640.0, horizon_y_px=360.0, camera_height_m=1.25, ema_alpha=0.6,
        ),
        builder=ObservationBuilder(BuilderConfig()),
        actor=actor,
        advisory_decoder=AdvisoryDecoder(units="mph"),
    )
    return pipeline, actor


def _real_drive_config(tmp_path):
    config = run_demo.load_config(str(run_demo.JETSON_DIR / "config.yaml"))
    config["telemetry"]["enabled"] = False
    config["v2v"]["enabled"] = False
    config["ui"]["display"] = False
    config["logio"]["video"] = False
    config["logio"]["system_stats"] = False
    config["logio"]["thermal"] = False
    config["logio"]["nmea"] = False
    config["logio"]["metadata"] = True
    config["logio"]["failures"] = True
    config["logio"]["failure_interval_s"] = 0.02
    config["paths"]["log_dir"] = str(tmp_path / "logs")
    return config


def _real_drive_args(**overrides):
    base = dict(
        duration_s=0.0, headless=True, live_rates=False, max_ticks=0,
        no_gps=True, no_log=False, phone=False, phone_host="0.0.0.0",
        phone_port=0, phone_wait_s=0.0, print_every=1.0, rate_heartbeat_s=1.0,
        require_gps=False, sim_gps=None, source=None, usb=False, usb_serial=None,
        rebind_timeout_s=120.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_hand_built_args_cover_every_option_run_live_can_read():
    """The fixture above stands in for parsed arguments, and drifts silently.

    Adding `--rebind-timeout-s` broke five tests at once, and the symptom was an
    `AttributeError` raised deep inside `run_live` rather than anything naming the
    fixture. This compares the fixture against the real parser, so the next option
    fails here, once, with a message that says which field is missing.

    Only options `run_live` can reach are required: `--config`, `--selfcheck` and
    `--scenario` are consumed by `main` before `run_live` is called.
    """
    consumed_by_main = {"config", "selfcheck", "scenario"}
    parsed = vars(run_demo.build_parser().parse_args([]))
    missing = set(parsed) - consumed_by_main - set(vars(_real_drive_args()))
    assert not missing, f"_real_drive_args is missing: {sorted(missing)}"


class TestRunLiveFailureSamplerIntegration:
    """M3: five of `run_live`'s six failure-sampler wiring points had no
    test executing them with a real sampler -- only the literal source-text
    pins above. This drives `run_live` itself, the way
    `TestThermalSummaryStopsBeforeReading` drives the thermal sampler's own
    teardown, with everything upstream of the camera and policy bundle
    faked or built the way `test_pipeline_smoke.py` already does without a
    GPU.
    """

    def test_a_real_drive_writes_a_failure_summary_tick_block_and_log_health(self, tmp_path, monkeypatch):
        pipeline, actor = _build_real_pipeline(tmp_path)
        camera = _FiniteCamera(n_frames=6)

        monkeypatch.setattr(run_demo, "build_components", lambda *a, **k: (camera, None, pipeline, actor))

        config = _real_drive_config(tmp_path)
        args = _real_drive_args()

        rc = run_demo.run_live(config, args)
        assert rc == 0

        run_dirs = list((tmp_path / "logs").iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        summary = json.loads((run_dir / "summary.json").read_text())
        # Wiring point 1 & 2: the sampler was actually constructed and
        # started -- a summary with no passes means it never ran.
        assert summary["failures"]["scan"]["passes"] > 0

        records = [
            json.loads(line) for line in (run_dir / "metadata.jsonl").read_text().splitlines() if line.strip()
        ]
        ticks = [r for r in records if r.get("type") == "tick"]
        assert len(ticks) == 6, "the camera's six real frames should be six ticks"
        # Wiring point 3: the tick record carries the sampler's own block.
        assert all("failures" in t for t in ticks)
        assert ticks[0]["failures"]["basis"] is not None

        # Wiring point 4: `summary["failures"]` came from the sampler's own
        # `to_record()`, not a dropped or default-shaped stand-in.
        assert "sources" in summary["failures"]
        assert summary["failures"]["scan"]["sources_n"] > 0

        # Wiring point 5: `_write_log_health` ran after `close()`.
        health = json.loads((run_dir / "log_health.json").read_text())
        assert health["writer_failure"] is None
        assert health["path"] == "metadata.jsonl"


class TestWorkerExceptionReachesTheRealThread:
    """M3: `test_run_live_wires_the_except_and_the_reraise` above proves a
    shape -- that `note_pipeline_exception` appears somewhere in the handler's
    subtree and that the handler's last statement is a bare `raise` -- and
    `TestWorkerRecordsAndReraises` proves a behaviour, but only on
    `worker_for`, a hand transcription of `run_live`'s own `worker()`. Neither
    one runs if `run_live`'s actual handler is mutated to keep both shapes
    while breaking what they mean: wrapping the `note_pipeline_exception`
    call in `if 0:` still has the call in the AST, and inserting `if True:
    return` before the final `raise` leaves that `raise` exactly where the
    AST check looks for it, without ever reaching it. This drives `run_live`
    for real, so both mutants fail here.
    """

    def test_the_exception_reaches_the_threads_own_hook_and_is_recorded(self, tmp_path, monkeypatch):
        import threading

        pipeline, actor = _build_real_pipeline(tmp_path)

        def raising_step(*args, **kwargs):
            raise RuntimeError("perception blew up")

        monkeypatch.setattr(pipeline, "step", raising_step)
        camera = _FiniteCamera(n_frames=3)
        monkeypatch.setattr(run_demo, "build_components", lambda *a, **k: (camera, None, pipeline, actor))

        caught: list[BaseException] = []
        # `threading.excepthook` is what an uncaught exception on a
        # background thread actually reaches -- the default one prints to
        # stderr and nothing programmatic ever sees it. Capturing it here
        # is the only way to observe, from outside `run_live`, whether the
        # `raise` in `worker()`'s `except` clause was actually reached
        # rather than swallowed by a mutation that keeps the statement but
        # returns before it.
        monkeypatch.setattr(threading, "excepthook", lambda args: caught.append(args.exc_value))

        config = _real_drive_config(tmp_path)
        args = _real_drive_args()
        rc = run_demo.run_live(config, args)
        assert rc == 0

        assert len(caught) == 1, "the exception must reach the thread's own excepthook exactly once"
        assert isinstance(caught[0], RuntimeError)
        assert str(caught[0]) == "perception blew up"

        run_dir = next((tmp_path / "logs").iterdir())
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["failures"]["pipeline_exception"] == "RuntimeError"

        records = [
            json.loads(line) for line in (run_dir / "metadata.jsonl").read_text().splitlines() if line.strip()
        ]
        opens = [
            r for r in records
            if r.get("type") == "failure_event" and r.get("phase") == "open"
            and r.get("source") == "pipeline.exception"
        ]
        assert len(opens) == 1


class TestSilentCameraNoteNoFrameReachesTheRealLoop:
    """M3's other half: `test_run_live_calls_note_no_frame_before_the_end_of_stream_check`
    checks statement order inside `run_live`'s own `if frame is None:` block,
    and `ast.walk` finds a call inside a dead `if 0:` branch or an
    uninvoked lambda just as readily as a reachable one. This drives a camera
    that never yields a frame through the real tick loop and asserts the
    sampler actually recorded it."""

    def test_a_silent_camera_is_recorded_through_the_real_tick_loop(self, tmp_path, monkeypatch):
        pipeline, actor = _build_real_pipeline(tmp_path)
        camera = _SilentCamera()
        monkeypatch.setattr(run_demo, "build_components", lambda *a, **k: (camera, None, pipeline, actor))

        config = _real_drive_config(tmp_path)
        # No `end_of_stream` is coming from this camera -- the deadline is the
        # only thing that ends the drive. The duration has to clear
        # `BLIND_EPISODE_MIN_S` for real, or the blind streak below never runs
        # long enough to become an episode -- this drives the real clock, not
        # a fake one, so the run genuinely takes this long.
        args = _real_drive_args(duration_s=BLIND_EPISODE_MIN_S + 0.5)
        rc = run_demo.run_live(config, args)
        assert rc == 0

        run_dir = next((tmp_path / "logs").iterdir())
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["failures"]["blind_ticks"] > 0

        records = [
            json.loads(line) for line in (run_dir / "metadata.jsonl").read_text().splitlines() if line.strip()
        ]
        # A drive the camera never delivered a single frame on writes no tick
        # records at all -- the failure log is the only record of it.
        assert not any(r.get("type") == "tick" for r in records)
        opens = [
            r for r in records
            if r.get("type") == "failure_event" and r.get("source") == "camera.blind_ticks"
        ]
        assert len(opens) >= 1


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


class TestUsbAcceptorCloseIsRegisteredWithAtexit:
    """B2 (validation round 1): `adb reverse` is external state on the
    device, not a socket that dies with the process. `UsbAcceptor.close()`
    was only reachable from the `finally` at the end of `run_live` that
    tears down `camera`/`gps`/etc., and that `finally` cannot exist until
    those objects are built -- so anything raising between constructing the
    acceptor and reaching it (build_components, camera warmup, the worker
    thread) left the mapping on the device. The fix registers `close()`
    with `atexit` at construction, which runs on any exception that reaches
    the top of the process (verified separately: an uncaught exception and
    a KeyboardInterrupt both still ran a registered atexit callback).

    Exercised by making `wait_for_phone` fail immediately after the
    registration should have happened, so nothing past it (build_components,
    the camera, the detector engine) needs to run at all.
    """

    def test_registers_close_before_waiting_for_the_phone(self, monkeypatch, tmp_path):
        registered = []
        monkeypatch.setattr(run_demo.atexit, "register", lambda fn: registered.append(fn))

        closed = []

        class FakeUsbAcceptor:
            port = 47811

            def __init__(self, serial, port=47811):
                self.serial = serial

            def close(self):
                closed.append(self.serial)

        monkeypatch.setattr("transport.usb.UsbAcceptor", FakeUsbAcceptor)

        class FakePhoneLink:
            def __init__(self, **kwargs):
                self.host = "127.0.0.1"
                self.port = 47811
                self.refusals = []
                self.session = None

            def wait_for_phone(self, timeout_s):
                return False

            def stop(self):
                pass

        monkeypatch.setattr("sensors.phone_link.PhoneLink", FakePhoneLink)

        args = _real_drive_args(
            phone=True, usb=True, usb_serial="ZY227VV4XC", phone_wait_s=0.01, no_gps=False,
        )
        config = _real_drive_config(tmp_path)
        rc = run_demo.run_live(config, args)

        assert rc == 2, "no phone dialled in (FakePhoneLink.wait_for_phone returns False)"
        assert len(registered) == 1
        # The registered callable is the fake acceptor's own bound close --
        # calling it proves it is the SAME object atexit would call, not a
        # copy or a no-op standing in for it.
        registered[0]()
        assert closed == ["ZY227VV4XC"]

    def test_does_not_register_when_usb_is_not_used(self, monkeypatch, tmp_path):
        registered = []
        monkeypatch.setattr(run_demo.atexit, "register", lambda fn: registered.append(fn))

        class FakePhoneLink:
            def __init__(self, **kwargs):
                self.host = "0.0.0.0"
                self.port = 47811
                self.refusals = []
                self.session = None

            def wait_for_phone(self, timeout_s):
                return False

            def stop(self):
                pass

        monkeypatch.setattr("sensors.phone_link.PhoneLink", FakePhoneLink)

        args = _real_drive_args(phone=True, usb=False, phone_wait_s=0.01, no_gps=False)
        config = _real_drive_config(tmp_path)
        rc = run_demo.run_live(config, args)

        assert rc == 2
        assert registered == []


class TestUsbSigtermHandler:
    """C5 (validation round 2, upgraded to required): a bare `kill`
    (SIGTERM) bypasses `atexit` entirely -- Python's default disposition
    for it terminates the process immediately, a different path from an
    uncaught exception reaching the top of the interpreter (which DOES
    still run atexit). Left unhandled, the leaked `adb reverse
    tcp:47811 tcp:47811` holds a device-side listener on that exact port,
    which `ImuWireTest.kt` binds a `ServerSocket` on
    (`LinkConfig.DEFAULT_PORT`), so an instrumented run after a SIGTERM'd
    drive fails with `EADDRINUSE` for an unrelated reason.
    """

    def _install_fakes(self, monkeypatch):
        installed_handlers: dict[int, object] = {}
        monkeypatch.setattr(
            run_demo.signal, "signal",
            lambda sig, handler: installed_handlers.__setitem__(sig, handler),
        )

        closed = []

        class FakeUsbAcceptor:
            port = 47811

            def __init__(self, serial, port=47811):
                self.serial = serial

            def close(self):
                closed.append(self.serial)

        monkeypatch.setattr("transport.usb.UsbAcceptor", FakeUsbAcceptor)

        class FakePhoneLink:
            def __init__(self, **kwargs):
                self.host = "127.0.0.1"
                self.port = 47811
                self.refusals = []
                self.session = None

            def wait_for_phone(self, timeout_s):
                return False

            def stop(self):
                pass

        monkeypatch.setattr("sensors.phone_link.PhoneLink", FakePhoneLink)
        return installed_handlers, closed

    def test_installs_a_sigterm_handler_when_usb_is_used(self, monkeypatch, tmp_path):
        installed_handlers, _closed = self._install_fakes(monkeypatch)
        args = _real_drive_args(
            phone=True, usb=True, usb_serial="ZY227VV4XC", phone_wait_s=0.01, no_gps=False,
        )
        config = _real_drive_config(tmp_path)
        rc = run_demo.run_live(config, args)

        assert rc == 2
        assert run_demo.signal.SIGTERM in installed_handlers

    def test_the_handler_raises_systemexit_which_still_runs_the_atexit_close(
        self, monkeypatch, tmp_path,
    ):
        """Not just that a handler is installed: that invoking it actually
        unwinds through SystemExit (verified separately: that still runs a
        registered atexit callback, unlike the SIGTERM default) rather than
        calling `close()` directly and swallowing the signal -- which would
        leave the process running with a signal handler that silently ate
        a termination request.
        """
        registered = []
        monkeypatch.setattr(run_demo.atexit, "register", lambda fn: registered.append(fn))
        installed_handlers, closed = self._install_fakes(monkeypatch)

        args = _real_drive_args(
            phone=True, usb=True, usb_serial="ZY227VV4XC", phone_wait_s=0.01, no_gps=False,
        )
        config = _real_drive_config(tmp_path)
        run_demo.run_live(config, args)

        handler = installed_handlers[run_demo.signal.SIGTERM]
        with pytest.raises(SystemExit) as exc_info:
            handler(run_demo.signal.SIGTERM, None)
        assert exc_info.value.code == 128 + run_demo.signal.SIGTERM
        # The handler itself does not call close() -- atexit (registered
        # separately, above) is what runs it once SystemExit is unhandled.
        assert closed == []
        registered[0]()
        assert closed == ["ZY227VV4XC"]

    def test_does_not_install_a_sigterm_handler_when_usb_is_not_used(self, monkeypatch, tmp_path):
        installed_handlers = {}
        monkeypatch.setattr(
            run_demo.signal, "signal",
            lambda sig, handler: installed_handlers.__setitem__(sig, handler),
        )

        class FakePhoneLink:
            def __init__(self, **kwargs):
                self.host = "0.0.0.0"
                self.port = 47811
                self.refusals = []
                self.session = None

            def wait_for_phone(self, timeout_s):
                return False

            def stop(self):
                pass

        monkeypatch.setattr("sensors.phone_link.PhoneLink", FakePhoneLink)
        args = _real_drive_args(phone=True, usb=False, phone_wait_s=0.01, no_gps=False)
        config = _real_drive_config(tmp_path)
        rc = run_demo.run_live(config, args)

        assert rc == 2
        assert installed_handlers == {}


#: Verbatim `adb -s ZY227VV4XC shell dumpsys package com.dsrc.phone` (this
#: session, 2026-09-05) -- the same real capture
#: `tests/test_record_installed_apk.py`'s `REAL_DUMPSYS_OUTPUT` uses.
_REAL_DUMPSYS_OUTPUT = (
    "    versionCode=1 minSdk=29 targetSdk=35\n"
    "    versionName=0.1\n"
    "    splits=[base]\n"
    "    firstInstallTime=2026-09-05 00:58:05\n"
    "    lastUpdateTime=2026-09-05 00:58:05\n"
)


class TestLiveApkLastUpdateTime:
    """A5 (validation round 3): `_build_provenance`'s own live re-check of
    the phone's CURRENT `lastUpdateTime`, queried at run time rather than
    trusted from the sidecar alone."""

    def test_parses_the_real_device_output(self, monkeypatch):
        def fake_run(args, *, capture_output, text, timeout):
            class Result:
                stdout = _REAL_DUMPSYS_OUTPUT

            return Result()

        monkeypatch.setattr(run_demo.subprocess, "run", fake_run)
        assert run_demo._live_apk_last_update_time("ZY227VV4XC") == "2026-09-05 00:58:05"

    def test_is_none_on_unrecognised_output(self, monkeypatch):
        def fake_run(args, *, capture_output, text, timeout):
            class Result:
                stdout = "nothing relevant here"

            return Result()

        monkeypatch.setattr(run_demo.subprocess, "run", fake_run)
        assert run_demo._live_apk_last_update_time("ZY227VV4XC") is None

    def test_is_none_when_adb_is_unreachable(self, monkeypatch):
        def raising_run(args, *, capture_output, text, timeout):
            raise FileNotFoundError("no adb")

        monkeypatch.setattr(run_demo.subprocess, "run", raising_run)
        assert run_demo._live_apk_last_update_time("ZY227VV4XC") is None


class TestLiveSourceTreeSha256:
    """B21 (validation round 4): the same hash `scripts/
    record_deployed_commit.py`'s `source_tree_sha256` computes, recomputed
    against a real tree at run time."""

    def test_is_none_when_the_tree_does_not_exist(self, tmp_path):
        assert run_demo._live_source_tree_sha256(tmp_path / "nope") is None

    def test_is_stable_across_two_calls_on_an_unchanged_tree(self, tmp_path):
        (tmp_path / "a.py").write_text("print(1)\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("print(2)\n")
        first = run_demo._live_source_tree_sha256(tmp_path)
        second = run_demo._live_source_tree_sha256(tmp_path)
        assert first == second
        assert first is not None

    def test_changes_when_a_files_content_changes(self, tmp_path):
        (tmp_path / "a.py").write_text("print(1)\n")
        before = run_demo._live_source_tree_sha256(tmp_path)
        (tmp_path / "a.py").write_text("print(2)\n")
        after = run_demo._live_source_tree_sha256(tmp_path)
        assert before != after

    def test_changes_when_a_file_is_renamed_with_identical_content(self, tmp_path):
        """Content alone is not enough -- a rename is a real change to
        what was deployed, and must not hash identically to the file it
        replaced."""
        (tmp_path / "a.py").write_text("print(1)\n")
        before = run_demo._live_source_tree_sha256(tmp_path)
        (tmp_path / "a.py").rename(tmp_path / "b.py")
        after = run_demo._live_source_tree_sha256(tmp_path)
        assert before != after

    def test_ignores_files_under_excluded_directories(self, tmp_path):
        (tmp_path / "a.py").write_text("print(1)\n")
        before = run_demo._live_source_tree_sha256(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "deployed_commit.json").write_text('{"commit": "x"}')
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"bytecode")
        after = run_demo._live_source_tree_sha256(tmp_path)
        assert before == after

    def test_matches_record_deployed_commits_own_function_on_the_real_tree(self):
        """The two must agree byte-for-byte on the real `deployment/
        jetson/` tree -- a divergence here is exactly how B21's own check
        would report a false mismatch on an unchanged deploy."""
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "scripts"))
        import record_deployed_commit as rdc  # noqa: E402

        assert (
            run_demo._live_source_tree_sha256(run_demo.JETSON_DIR)
            == rdc.source_tree_sha256(run_demo.JETSON_DIR)
        )


class TestBuildProvenance:
    """A4 (validation round 2): given a run directory alone, none of the
    policy bundle, the detector engine, or the code revision was
    recoverable -- `summary.json`'s only model field was `policy_trained:
    false`. `_build_provenance` fills `summary["build"]`.
    """

    def _config(self, tmp_path, **overrides):
        config = {
            "policy": {"bundle": str(tmp_path / "actor_policy")},
            "detector": {"engine": str(tmp_path / "yolov8n.engine")},
        }
        config.update(overrides)
        return config

    def test_reads_the_real_git_commit(self, tmp_path):
        # This is the real `git rev-parse HEAD` against run_demo.JETSON_DIR
        # -- not a fake -- so this deliberately reads correctly in BOTH of
        # this task's own two environments: a real git checkout (the Mac)
        # and an ad-hoc rsync copy with no `.git` at all (this task's own
        # `~/dsrc-task40` on jetson-orin, exit 128) -- the exact case A4
        # names and the one an earlier version of this test did not handle,
        # asserting `None == ""` there instead of `None == None`.
        #
        # B16 (validation round 3) added a second real source on top of
        # git: `run_demo.DEPLOYED_COMMIT_RECORD` (also read for real, not
        # monkeypatched, for the same reason as `git rev-parse` above) --
        # present for real on the jetson tree once a deploy step has
        # written it, absent on a Mac checkout where git already answers.
        # `expected` follows `_build_provenance`'s own precedence so this
        # test reads correctly in all three states (git works; git fails,
        # sidecar present; git fails, no sidecar) rather than assuming
        # whichever one happened to be true when it was written -- the
        # exact failure this docstring already names once.
        import subprocess

        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(run_demo.JETSON_DIR),
            capture_output=True, text=True,
        )
        expected = probe.stdout.strip() if probe.returncode == 0 else None
        if expected is None and run_demo.DEPLOYED_COMMIT_RECORD.exists():
            try:
                expected = json.loads(run_demo.DEPLOYED_COMMIT_RECORD.read_text()).get("commit")
            except ValueError:
                expected = None
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] == expected
        if expected is not None:
            assert len(expected) == 40

    def test_commit_is_none_when_git_rev_parse_fails(self, tmp_path, monkeypatch):
        """The exact shape `~/dsrc-task40` (this task's own rsync copy, not
        a git checkout) produces -- named rather than silently omitted.
        `DEPLOYED_COMMIT_RECORD` is pointed at a path that does not exist
        (B16, validation round 3): this test's own intent is "git fails
        AND there is no other source", which a REAL sidecar left on
        whichever machine runs this suite would otherwise silently supply
        -- the jetson always carries one once a deploy has run."""
        def failing_run(args, *, cwd, capture_output, text, timeout):
            class Result:
                returncode = 128
                stdout = ""

            return Result()

        monkeypatch.setattr(run_demo.subprocess, "run", failing_run)
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] is None

    def test_commit_is_none_when_git_is_not_on_path(self, tmp_path, monkeypatch):
        def raising_run(args, *, cwd, capture_output, text, timeout):
            raise FileNotFoundError("no git")

        monkeypatch.setattr(run_demo.subprocess, "run", raising_run)
        # Same isolation as the test above, same reason.
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] is None

    # -- B16 (validation round 3): deployed_commit.json fallback ------------

    def _failing_git_rev_parse(self, monkeypatch):
        def failing_run(args, *, cwd, capture_output, text, timeout):
            class Result:
                returncode = 128
                stdout = ""

            return Result()

        monkeypatch.setattr(run_demo.subprocess, "run", failing_run)

    def test_commit_falls_back_to_the_deployed_commit_record_when_git_fails(self, tmp_path, monkeypatch):
        """The exact case this fix is for: `git rev-parse` fails (the
        jetson's rsync copy), but the SOURCE machine wrote this sidecar
        before the rsync that stripped `.git` -- the commit survives
        where `git rev-parse` alone cannot reach it."""
        self._failing_git_rev_parse(monkeypatch)
        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text(json.dumps({"commit": "a" * 40, "dirty": False}))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] == "a" * 40

    def test_commit_is_none_when_git_fails_and_no_deployed_commit_record_exists(self, tmp_path, monkeypatch):
        self._failing_git_rev_parse(monkeypatch)
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] is None

    def test_commit_is_none_when_the_deployed_commit_record_is_malformed(self, tmp_path, monkeypatch):
        self._failing_git_rev_parse(monkeypatch)
        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text("not json")
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] is None

    def test_commit_prefers_a_real_git_rev_parse_over_the_deployed_commit_record(self, tmp_path, monkeypatch):
        """On a machine where `git rev-parse` genuinely works (a real git
        checkout -- the Mac, not the jetson's own rsync copy, which has no
        `.git` at all and is exactly what the fallback tests above cover),
        a stale sidecar left over from some earlier deploy must not
        override the real, current HEAD -- the fallback is for when git
        itself cannot answer, not a general override. Skips rather than
        assuming which of the two this is: `test_reads_the_real_git_commit`
        already established that this suite runs in both.
        """
        import subprocess

        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(run_demo.JETSON_DIR),
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            pytest.skip("this tree is not a git checkout; the fallback tests above cover it")
        real_commit = probe.stdout.strip()

        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text(json.dumps({"commit": "stale" * 8, "dirty": False}))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] == real_commit
        assert result["commit"] != "stale" * 8

    def test_reads_the_policy_bundle_sidecar_as_a_copy(self, tmp_path):
        bundle_json = {
            "trained": False, "sim_commit": "d477dba",
            "contract_fingerprint": "918ec57cf2f2e1db",
        }
        (tmp_path / "actor_policy.json").write_text(json.dumps(bundle_json))
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["policy_bundle"] == bundle_json

    def test_policy_bundle_is_none_when_the_sidecar_is_absent(self, tmp_path):
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["policy_bundle"] is None

    def test_hashes_the_detector_engine(self, tmp_path):
        engine_bytes = b"a fake tensorrt engine, just bytes to hash"
        (tmp_path / "yolov8n.engine").write_bytes(engine_bytes)
        import hashlib

        expected = hashlib.sha256(engine_bytes).hexdigest()
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["detector_engine_sha256"] == expected

    def test_detector_engine_sha256_is_none_when_the_engine_is_absent(self, tmp_path):
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["detector_engine_sha256"] is None

    def test_reads_apk_sha256_from_the_installed_apk_record(self, tmp_path, monkeypatch):
        record_path = tmp_path / "installed_apk.json"
        record_path.write_text(json.dumps({"sha256": "abc123", "version_code": 1}))
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["apk_sha256"] == "abc123"

    def test_apk_sha256_is_none_when_no_install_record_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["apk_sha256"] is None

    # -- B22 (validation round 4): the sidecar copied whole, not cherry-picked --

    def test_installed_apk_record_is_the_whole_sidecar(self, tmp_path, monkeypatch):
        """`dirty`, `apk_name`, `version_code`, `version_name` and
        `local_apk_sha256` must all survive, not just `sha256` -- a bare
        64-character hash cannot be resolved to a build once the APK
        archive itself is gone."""
        record_path = tmp_path / "installed_apk.json"
        whole = {
            "sha256": "abc123", "local_apk_sha256": "def456",
            "last_update_time": "2026-09-05 00:58:05", "apk_name": "app-debug.apk",
            "version_code": 1, "version_name": "0.1", "serial": "ZY227VV4XC",
        }
        record_path.write_text(json.dumps(whole))
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["installed_apk_record"] == whole

    def test_installed_apk_record_is_none_when_no_install_record_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["installed_apk_record"] is None

    # -- A5 (validation round 3): live lastUpdateTime re-check ---------------
    # -- B20 (validation round 4): a closed vocabulary, not a bare bool | None --

    def _install_record(self, tmp_path, monkeypatch, **overrides):
        record_path = tmp_path / "installed_apk.json"
        record = {"sha256": "abc123", "last_update_time": "2026-09-05 00:58:05"}
        record.update(overrides)
        record_path.write_text(json.dumps(record))
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", record_path)

    def test_reads_apk_last_update_time_from_the_installed_apk_record(self, tmp_path, monkeypatch):
        self._install_record(tmp_path, monkeypatch)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["apk_last_update_time"] == "2026-09-05 00:58:05"

    def test_apk_last_update_time_is_none_when_no_install_record_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["apk_last_update_time"] is None

    def test_apk_last_update_time_check_is_no_serial_without_a_serial(self, tmp_path, monkeypatch):
        """The tailnet path -- 'not applicable', not 'applicable but the
        device would not answer' (B20's own two-nulls-that-differ)."""
        self._install_record(tmp_path, monkeypatch)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["apk_last_update_time_check"] == run_demo.APK_TIMESTAMP_NO_SERIAL
        assert result["apk_last_update_time_check"] in run_demo.APK_TIMESTAMP_VOCABULARY

    def test_apk_last_update_time_check_is_matched_when_the_live_device_agrees(self, tmp_path, monkeypatch):
        self._install_record(tmp_path, monkeypatch)
        monkeypatch.setattr(run_demo, "_live_apk_last_update_time", lambda serial: "2026-09-05 00:58:05")
        result = run_demo._build_provenance(self._config(tmp_path), serial="ZY227VV4XC")
        assert result["apk_last_update_time_check"] == run_demo.APK_TIMESTAMP_MATCHED

    def test_apk_last_update_time_check_is_mismatched_when_the_phone_was_reinstalled(self, tmp_path, monkeypatch):
        """The core of A5: `adb install -r other.apk` without re-running
        `record_installed_apk.py` leaves the sidecar's `sha256` describing
        an install that is no longer on the phone -- this is the live
        re-check catching exactly that, at run time rather than at write
        time."""
        self._install_record(tmp_path, monkeypatch)
        monkeypatch.setattr(run_demo, "_live_apk_last_update_time", lambda serial: "2026-09-05 09:00:00")
        result = run_demo._build_provenance(self._config(tmp_path), serial="ZY227VV4XC")
        assert result["apk_last_update_time_check"] == run_demo.APK_TIMESTAMP_MISMATCHED

    def test_apk_last_update_time_check_is_device_did_not_answer_when_the_live_device_is_unreachable(
        self, tmp_path, monkeypatch,
    ):
        """B20's other null: a serial WAS given (a USB drive), but adb
        could not answer -- a fault, distinct from 'not applicable'
        above, and must not read the same as it."""
        self._install_record(tmp_path, monkeypatch)
        monkeypatch.setattr(run_demo, "_live_apk_last_update_time", lambda serial: None)
        result = run_demo._build_provenance(self._config(tmp_path), serial="NOSUCHSERIAL")
        assert result["apk_last_update_time_check"] == run_demo.APK_TIMESTAMP_DEVICE_DID_NOT_ANSWER

    def test_apk_last_update_time_check_is_sidecar_has_no_timestamp(
        self, tmp_path, monkeypatch,
    ):
        """A record written before A5, or with no device reachable at
        write time either -- nothing to compare a live value against, so
        this must not call `_live_apk_last_update_time` at all."""
        record_path = tmp_path / "installed_apk.json"
        record_path.write_text(json.dumps({"sha256": "abc123"}))  # no last_update_time key
        monkeypatch.setattr(run_demo, "INSTALLED_APK_RECORD", record_path)

        def unexpected(serial):
            raise AssertionError("must not be called with nothing to compare against")

        monkeypatch.setattr(run_demo, "_live_apk_last_update_time", unexpected)
        result = run_demo._build_provenance(self._config(tmp_path), serial="ZY227VV4XC")
        assert result["apk_last_update_time"] is None
        assert result["apk_last_update_time_check"] == run_demo.APK_TIMESTAMP_SIDECAR_HAS_NO_TIMESTAMP

    # -- B21 (validation round 4): the deployed commit's own staleness signal --

    def _real_git_rev_parse_succeeds(self) -> bool:
        """Probes `JETSON_DIR` directly, real `git`, no mock -- this repo
        runs in two real environments (a Mac checkout with `.git`; the
        jetson's own rsync copy without one), and `result["commit"] is
        not None` cannot distinguish "git succeeded" from "the fallback
        sidecar supplied it" now that B16/B21 gave `commit` a second
        source -- exactly the trap `test_reads_the_real_git_commit`
        already names once for this same ambiguity."""
        import subprocess

        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(run_demo.JETSON_DIR),
            capture_output=True, text=True,
        )
        return probe.returncode == 0

    def test_source_tree_check_is_not_applicable_when_git_rev_parse_succeeds(self, tmp_path):
        """On a real git checkout (this repo, on the Mac), the sidecar
        staleness question does not even arise -- git itself is the live
        source of truth."""
        if not self._real_git_rev_parse_succeeds():
            pytest.skip("this tree is not a git checkout; the fallback tests above cover it")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["source_tree_check"] == run_demo.SOURCE_TREE_NOT_APPLICABLE_GIT_CHECKOUT

    def test_source_tree_check_is_no_commit_recorded(self, tmp_path, monkeypatch):
        self._failing_git_rev_parse(monkeypatch)
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", tmp_path / "nope.json")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] is None
        assert result["source_tree_check"] == run_demo.SOURCE_TREE_NO_COMMIT_RECORDED

    def test_source_tree_check_is_no_sidecar_hash_on_a_pre_b21_record(self, tmp_path, monkeypatch):
        """A `deployed_commit.json` written before B21 has no
        `source_tree_sha256` key at all -- nothing to compare against,
        distinct from a real mismatch."""
        self._failing_git_rev_parse(monkeypatch)
        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text(json.dumps({"commit": "a" * 40, "dirty": False}))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["source_tree_check"] == run_demo.SOURCE_TREE_NO_SIDECAR_HASH

    def test_source_tree_check_is_matched_when_the_live_tree_agrees(self, tmp_path, monkeypatch):
        self._failing_git_rev_parse(monkeypatch)
        real_hash = run_demo._live_source_tree_sha256(run_demo.JETSON_DIR)
        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text(json.dumps({
            "commit": "a" * 40, "dirty": False, "source_tree_sha256": real_hash,
        }))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["source_tree_check"] == run_demo.SOURCE_TREE_MATCHED

    def test_source_tree_check_is_mismatched_when_a_redeploy_forgot_the_script(self, tmp_path, monkeypatch):
        """B21's own core case: deploy A's sidecar (`deployed_commit.json`,
        no `--delete` behind the rsync) sitting beside deploy B's code --
        the exact scenario a stale, un-refreshed `source_tree_sha256`
        must be caught by, since `git rev-parse` always fails on this
        tree and `commit` alone would report deploy A with full
        confidence."""
        self._failing_git_rev_parse(monkeypatch)
        record_path = tmp_path / "deployed_commit.json"
        record_path.write_text(json.dumps({
            "commit": "a" * 40, "dirty": False,
            "source_tree_sha256": "0" * 64,  # deliberately wrong
        }))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["commit"] == "a" * 40, "the stale commit is still reported, just flagged"
        assert result["source_tree_check"] == run_demo.SOURCE_TREE_MISMATCHED

    def test_deployed_commit_record_is_the_whole_sidecar(self, tmp_path, monkeypatch):
        """B22: `dirty` and `source_tree_sha256` must survive, not just
        `commit` alone."""
        self._failing_git_rev_parse(monkeypatch)
        record_path = tmp_path / "deployed_commit.json"
        whole = {"commit": "a" * 40, "dirty": True, "source_tree_sha256": "b" * 64}
        record_path.write_text(json.dumps(whole))
        monkeypatch.setattr(run_demo, "DEPLOYED_COMMIT_RECORD", record_path)
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["deployed_commit_record"] == whole

    def test_deployed_commit_record_is_none_when_git_rev_parse_succeeds(self, tmp_path):
        if not self._real_git_rev_parse_succeeds():
            pytest.skip("this tree is not a git checkout; the fallback tests above cover it")
        result = run_demo._build_provenance(self._config(tmp_path))
        assert result["deployed_commit_record"] is None

    def test_a_real_drive_writes_the_build_block(self, tmp_path, monkeypatch):
        """End to end through run_live's own teardown, not a transcription:
        the same real pipeline TestRunLiveFailureSamplerIntegration drives."""
        pipeline, actor = _build_real_pipeline(tmp_path)
        camera = _FiniteCamera(n_frames=3)
        monkeypatch.setattr(run_demo, "build_components", lambda *a, **k: (camera, None, pipeline, actor))

        config = _real_drive_config(tmp_path)
        args = _real_drive_args()
        rc = run_demo.run_live(config, args)
        assert rc == 0

        run_dirs = list((tmp_path / "logs").iterdir())
        summary = json.loads((run_dirs[0] / "summary.json").read_text())
        assert "build" in summary
        assert set(summary["build"]) == {
            "commit", "source_tree_check", "deployed_commit_record",
            "policy_bundle", "detector_engine_sha256",
            "apk_sha256", "apk_last_update_time", "apk_last_update_time_check",
            "installed_apk_record",
        }
