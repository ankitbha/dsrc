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
        require_gps=False, sim_gps=None, source=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


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
