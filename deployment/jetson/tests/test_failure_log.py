"""The failure sampler: a time axis, an episode, and a reader for failures this
repository already detects.

Task 37's lesson carried forward explicitly: a fixture's failure mode has to be
the field's. There, a deleted file and a denied permission both raised `OSError`
while the real device raised `TypeError`, and every test passed while the
feature did nothing on the only machine it was written for. This task's own
equivalent hazard is sharper, because every source reads through an accessor
into a real object -- a fake counter cannot fail the way the real one does.
`TestRegistryAccessorsResolve` and `TestNotEvaluableIsNotQuiet` exist for
exactly that reason and construct real objects, not fakes.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from logio import failure_log
from logio.failure_log import (
    MAX_EPISODES_PER_SOURCE,
    MISSING,
    OUTCOME_OPEN_AT_END,
    OUTCOME_RECOVERED,
    OUTCOME_UNOBSERVABLE,
    OUTCOMES,
    REGISTRY,
    FailureSampler,
    SourceSnapshot,
    reason_is_valid,
)
from logio.metadata_logger import MetadataLogger
from policy import sensing_controller
from sensors import time_sync
from sensors.camera_stream import CameraStream
from sensors.gps_reader import GpsReader
from sensors.phone_link import PhoneLink
from sensors.phone_source import PhoneCameraStream
from transport.messages import MessageRouter
from transport.session import Session
from transport.tcp import TcpAcceptor


class Sink:
    """Records every line written, in order, without touching a disk."""

    def __init__(self) -> None:
        self.lines: list[dict] = []

    def write(self, record: dict) -> None:
        self.lines.append(record)


def sampler(sink=None, *, now: list[float] | None = None, **kwargs) -> FailureSampler:
    clock = now or [0.0]
    return FailureSampler(
        sink if sink is not None else Sink(),
        clock=lambda: clock[0], wall_clock=lambda: clock[0],
        **kwargs,
    )


def real_phone_pair():
    """A real `PhoneLink` bound to a real socket, plus a real session pair on
    a loopback connection, attached to it. Not a mock router: the transport's
    own real objects, so a renamed field or method fails here.
    """
    acceptor = TcpAcceptor(host="127.0.0.1", port=0)
    link = PhoneLink(acceptor=acceptor, device_id="jetson")
    from transport.loopback import loopback_pair

    phone_conn, jetson_conn = loopback_pair()
    jetson_session = Session(jetson_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    Session(phone_conn, session_id=1, heartbeat_s=None, stall_timeout_s=None).start()
    link.session = jetson_session
    link.peer_device_id = "test-phone"
    link._begin()
    return link


# ---------------------------------------------------------------------------
# 1. Every registry accessor resolves against the real object.
# ---------------------------------------------------------------------------


class TestRegistryAccessorsResolve:
    """A renamed counter fails here and nowhere else -- this is the test that
    would have caught the task-37 class of defect at plan time.

    The old version of this test asserted only "readable, or unreadable with
    a named reason" -- and MISSING accepts six different words, so a rename
    that flips a source from readable to not_evaluable still satisfies it.
    This asserts an explicit expected-readable set instead, so a rename shows
    up as that set changing rather than as a still-passing test.
    """

    def test_every_accessor_returns_a_readable_or_named_missing_snapshot(self, tmp_path):
        gps = GpsReader()
        camera = CameraStream("file:/does/not/exist.mp4")
        logger = MetadataLogger(tmp_path)
        phone = real_phone_pair()
        # `phone.router` is a real `MessageRouter` over a real `Session` pair
        # once `_begin()` has run -- constructed by `real_phone_pair()`,
        # exercised here rather than duplicated.
        assert isinstance(phone.router, MessageRouter)
        try:
            ctx = failure_log._Context(
                phone=phone, camera=camera, gps=gps,
                pass_session_id=phone.session.session_id,
            )
            registry = failure_log.build_registry(camera=camera)
            # Four of the thirty rows are not readable against this fixture,
            # and each for a fact this fixture states rather than one it
            # forgot: `camera` is a local `CameraStream`, which has no
            # `.decode_failures` or `.failure` (only a phone-fed
            # `PhoneCameraStream` does -- see the test below); and `phone`
            # has never received a telemetry frame, so `phone.telemetry` is
            # still `None`.
            not_readable_here = {
                "camera.decode_failures", "camera.reader_failure",
                "phone.dropped", "phone.here_errors",
            }
            expected_readable = {s.name for s in registry} - not_readable_here
            actually_readable = set()
            for source in registry:
                snap = source.read(ctx)
                assert isinstance(snap, SourceSnapshot), source.name
                if snap.readable:
                    actually_readable.add(source.name)
                    for value in snap.by_reason.values():
                        assert isinstance(value, int), source.name
                else:
                    assert snap.missing in MISSING, source.name
            assert actually_readable == expected_readable
        finally:
            phone.stop()
            logger.close()

    def test_the_phone_fed_camera_sources_are_readable_against_a_real_phonecamerastream(self):
        # The two sources `hasattr`-gated out above are D4's own point: a
        # typed accessor still has to be exercised against the real object it
        # names, not just against the one backend that lacks the field. Never
        # started -- every field these two accessors read is set in
        # `PhoneCameraStream.__init__`, so no reader thread is needed.
        phone = real_phone_pair()
        try:
            camera = PhoneCameraStream(phone.router, phone.adapter)
            registry = failure_log.build_registry(camera=camera)
            by_name = {s.name: s for s in registry}
            ctx = failure_log._Context(
                phone=None, camera=camera, gps=None, pass_session_id=None,
            )
            for name in ("camera.decode_failures", "camera.reader_failure"):
                snap = by_name[name].read(ctx)
                assert snap.readable, name
            # And the scope this camera earns for the source it changes:
            # `hasattr(camera, "decode_failures")` is True here, so
            # `camera.dropped_unconsumed` must come out session-scoped (C2),
            # not the "run" scope a local `CameraStream` gets.
            assert by_name["camera.dropped_unconsumed"].scope == "session"
        finally:
            phone.stop()


# ---------------------------------------------------------------------------
# 2 & 3. not_evaluable vs quiet are different records, on a source whose
# object is entirely absent.
# ---------------------------------------------------------------------------


class TestNotEvaluableIsNotQuiet:

    def test_an_absent_phone_reports_not_evaluable_with_missing_named(self):
        s = sampler(phone=None, camera=None, gps=None)
        s.sample_once()
        record = s.to_record()
        for name in ("here.refused", "phone.dropped", "link.down"):
            row = record["sources"][name]
            assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE, name
            assert row["passes_readable"] == 0
            assert row["passes_attempted"] == 1
            assert "missing" in row and row["missing"], name

    def test_a_readable_quiet_source_is_a_different_record_from_not_evaluable(self):
        s = sampler(phone=None, camera=None, gps=GpsReader())
        s.sample_once()
        record = s.to_record()
        quiet = record["sources"]["gps.parse_errors"]
        not_evaluable = record["sources"]["here.refused"]
        assert quiet["status"] == sensing_controller.RULE_QUIET
        assert "missing" not in quiet
        assert quiet["episodes"] == 0
        assert not_evaluable["status"] == sensing_controller.RULE_NOT_EVALUABLE
        assert not_evaluable["episodes"] == 0
        assert "missing" in not_evaluable


class TestMissingIsExhaustive:
    """`OUTCOMES`' own exhaustiveness is pinned by
    `test_all_three_outcomes_are_reachable_and_distinct` -- a `seen ==
    OUTCOMES` comparison built from driving every real code path, which
    catches a member added or removed on either side. `MISSING` had no
    equivalent: its only use is a membership check that passes whether it
    has five, six or seven words in it, and grew a sixth member
    (`MISSING_ACCESSOR_RAISED`) with nothing to notice if a seventh arrived
    -- or if one were quietly dropped. This drives every real cause of an
    unreadable pass and asserts the observed set is exactly `MISSING`."""

    def test_every_missing_reason_is_reachable_and_the_set_is_exact(self):
        seen: set[str] = set()

        def _collect(record: dict) -> None:
            for row in record["sources"].values():
                if row["status"] == sensing_controller.RULE_NOT_EVALUABLE:
                    seen.update(row.get("missing") or [])

        # MISSING_NO_PHONE and MISSING_NO_SOURCE: nothing is wired up at all.
        s1 = sampler(phone=None, camera=None, gps=None)
        s1.sample_once()
        _collect(s1.to_record())

        # MISSING_NO_SESSION: a phone exists, nothing is bound yet.
        phone = FakePhone()
        phone.session = None
        s2 = sampler(phone=phone)
        s2.sample_once()
        _collect(s2.to_record())

        # MISSING_NO_TELEMETRY: a session is bound, no telemetry frame has
        # arrived yet -- `FakePhone.telemetry` defaults to `None`.
        s3 = sampler(phone=FakePhone())
        s3.sample_once()
        _collect(s3.to_record())

        # MISSING_SESSION_MOVED: the pass straddles a rebind (D15), the same
        # drive as `test_a_pass_that_straddles_a_rebind_is_not_evaluable`.
        s4 = sampler(phone=FakePhone(session_id="s1"))
        s4.sample_once()
        real_read = failure_log._read_here_refused

        def torn_read(ctx):
            snap = real_read(ctx)
            return SourceSnapshot(readable=True, by_reason=dict(snap.by_reason), session_id="a-different-session")

        here_source = s4._by_name["here.refused"]
        object.__setattr__(here_source, "read", torn_read)
        try:
            s4.sample_once()
        finally:
            object.__setattr__(here_source, "read", real_read)
        _collect(s4.to_record())

        # MISSING_ACCESSOR_RAISED: the accessor's own bug.
        s5 = sampler(gps=FakeGps())

        def boom(ctx):
            raise RuntimeError("accessor bug")

        parse_errors_source = s5._by_name["gps.parse_errors"]
        object.__setattr__(parse_errors_source, "read", boom)
        try:
            s5.sample_once()
        finally:
            object.__setattr__(parse_errors_source, "read", failure_log._read_gps_parse_errors)
        _collect(s5.to_record())

        assert seen == MISSING


# ---------------------------------------------------------------------------
# 4 & 9 & 10. Episodes, counted from movement, not from occurrences.
# ---------------------------------------------------------------------------


class FakeGpsDiagnostics:
    def __init__(self) -> None:
        self.parse_errors = 0
        self.ingest_errors = 0
        self.last_error = ""
        self.rate_configured = True


class FakeFix:
    def __init__(self, valid: bool = True, t_mono: float = 1.0) -> None:
        self.valid = valid
        self.t_mono = t_mono

    def age_s(self, now: float) -> float:
        return now - self.t_mono if self.t_mono > 0 else float("inf")


class FakeGps:
    """Stands in for `GpsReader` only where the real one needs a live serial
    port to move at all (`diagnostics.parse_errors`): a counter this repo
    already owns, read through the same accessor either way."""

    def __init__(self) -> None:
        self.diagnostics = FakeGpsDiagnostics()

    def is_stale(self) -> bool:
        return False

    def latest(self):
        return FakeFix()


class FakeCamera:
    """Stands in for `PhoneCameraStream` wherever a test only needs to move
    `decode_failures`. Carries every mandatory attribute the registry's
    camera accessors now read directly (`dropped_frames`, `file_recoveries`,
    `end_of_stream`, `failure`) -- a bare object with only `decode_failures`
    set used to work by accident, back when `_read_camera_dropped_unconsumed`
    read through a defaulting `getattr`; direct attribute access (M1) means a
    test double has to carry the real shape."""

    def __init__(self, *, decode_failures: int = 0) -> None:
        self.decode_failures = decode_failures
        self.dropped_frames = 0
        self.file_recoveries = 0
        self.end_of_stream = False
        self.failure: str | None = None


class TestEpisodesFromMovement:

    def test_a_counter_advancing_over_three_passes_is_one_episode(self):
        gps = FakeGps()
        s = sampler(gps=gps)
        s.sample_once()
        gps.diagnostics.parse_errors = 5
        s.sample_once()
        gps.diagnostics.parse_errors = 8
        s.sample_once()
        gps.diagnostics.parse_errors = 15
        s.sample_once()
        s.stop()
        row = s.to_record()["sources"]["gps.parse_errors"]
        assert row["episodes"] == 1
        assert row["total"] == 15

        opens = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "open"]
        # `gps.parse_errors` carries no `event_records` flag (not one of the
        # task's named failures), so no `failure_event` line exists for it --
        # only the summary row proves the episode. Assert that instead.
        assert not any(o["source"] == "gps.parse_errors" for o in opens)

    def test_first_pass_n_is_the_movement_on_the_opening_pass(self):
        gps = FakeGps()
        s = sampler(gps=gps, quiet_passes_to_close=3)
        # Use camera.decode_failures instead: it IS event_records=True, so
        # the open record is observable directly.
        camera = FakeCamera()
        s2 = sampler(camera=camera)
        s2.sample_once()
        camera.decode_failures = 5
        s2.sample_once()
        camera.decode_failures = 8
        s2.sample_once()
        s2.stop()
        opens = [
            l for l in s2._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "open" and l["source"] == "camera.decode_failures"
        ]
        assert len(opens) == 1
        assert opens[0]["first_pass_n"] == 5
        closes = [
            l for l in s2._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "close" and l["source"] == "camera.decode_failures"
        ]
        assert closes[0]["n"] == 8


# ---------------------------------------------------------------------------
# 5 & 6. Close threshold: consecutive quiet passes, derived from interval_s.
# ---------------------------------------------------------------------------


class TestCloseThreshold:

    def test_a_still_gap_shorter_than_the_threshold_does_not_split_the_episode(self):
        camera = FakeCamera()
        s = sampler(camera=camera, quiet_passes_to_close=3)
        s.sample_once()
        camera.decode_failures = 3
        s.sample_once()          # moved
        s.sample_once()          # still 1
        camera.decode_failures = 7
        s.sample_once()          # moved again, before the 3rd still pass
        s.stop()
        closes = [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "close"
        ]
        assert len(closes) == 1
        assert closes[0]["n"] == 7

    def test_the_close_threshold_is_derived_from_interval_s(self):
        camera = FakeCamera()
        now = [0.0]
        s = sampler(camera=camera, now=now, interval_s=0.2, quiet_passes_to_close=3)
        s.sample_once()
        camera.decode_failures = 1
        now[0] += 0.2
        s.sample_once()   # opens
        for _ in range(2):
            now[0] += 0.2
            s.sample_once()   # 2 still passes: not yet closed
        assert s._state["camera.decode_failures"].open_episode is not None
        now[0] += 0.2
        s.sample_once()   # 3rd still pass: closes
        assert s._state["camera.decode_failures"].open_episode is None

        # The still window it took to get here, as REPORTED on the close
        # record, must be the derived 3 x 0.2 s -- not a typed 3.0 s that
        # would happen to look right at the default interval_s=1.0 and wrong
        # everywhere else. This is what MAX_EVIDENCE_GAP_S's own failure mode
        # looks like: a bound that stops tracking the value it was derived
        # from.
        closes = [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "close"
        ]
        assert closes[0]["close_after_s"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 7 & 8. A source that becomes unreadable closes as unobservable.
# ---------------------------------------------------------------------------


class TestUnobservable:

    def test_a_source_that_stops_being_readable_closes_as_unobservable(self):
        camera = FakeCamera()
        t = [0.0]
        s = FailureSampler(Sink(), camera=camera, clock=lambda: t[0], wall_clock=lambda: t[0])
        s.sample_once()                 # t=0: readable, quiet
        t[0] = 1.0
        camera.decode_failures = 4
        s.sample_once()                 # t=1: readable, opens the episode
        t[0] = 2.0
        s.sample_once()                 # t=2: readable, quiet (still open)
        t[0] = 3.0
        s._camera = None                # the source vanishes mid-episode
        s.sample_once()                 # t=3: not readable -- closes
        row = s.to_record()["sources"]["camera.decode_failures"]
        assert row["episodes"] == 1
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["outcome"] == OUTCOME_UNOBSERVABLE
        # Duration measures to the LAST READABLE pass (t=2), never to the
        # pass that noticed the gap (t=3) and never to teardown.
        assert closes[0]["t_mono"] == 2.0
        assert closes[0]["duration_s"] == 1.0

    def test_all_three_outcomes_are_reachable_and_distinct(self):
        # M2: `recovered` and `open_at_end` are adjacent, same-typed string
        # literals whose only call sites are the two `_close_episode` calls
        # this test drives -- asserting only `seen == OUTCOMES` lets a swap
        # between them through, since both memberships are still hit and the
        # SET is unchanged either way. Each construction's outcome is
        # asserted by name for exactly that reason.
        seen = set()

        # recovered
        camera = FakeCamera()
        s = sampler(camera=camera, quiet_passes_to_close=1)
        s.sample_once()
        camera.decode_failures = 1
        s.sample_once()
        s.sample_once()  # one quiet pass closes it (threshold=1)
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["outcome"] == OUTCOME_RECOVERED
        seen.add(closes[0]["outcome"])

        # unobservable
        camera2 = FakeCamera()
        s2 = FailureSampler(Sink(), camera=camera2)
        s2.sample_once()
        camera2.decode_failures = 1
        s2.sample_once()
        s2._camera = None
        s2.sample_once()
        closes2 = [l for l in s2._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes2[0]["outcome"] == OUTCOME_UNOBSERVABLE
        seen.add(closes2[0]["outcome"])

        # open_at_end
        camera3 = FakeCamera()
        s3 = FailureSampler(Sink(), camera=camera3)
        s3.sample_once()
        camera3.decode_failures = 1
        s3.sample_once()
        s3.stop()
        closes3 = [l for l in s3._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes3[0]["outcome"] == OUTCOME_OPEN_AT_END
        seen.add(closes3[0]["outcome"])

        assert seen == OUTCOMES


# ---------------------------------------------------------------------------
# 9 & 11. Session-scoped counters are not diffed across a redial.
# ---------------------------------------------------------------------------


class FakeHere:
    def __init__(self) -> None:
        self.refused_by_reason: dict[str, int] = {}
        self.proxied_stamps = 0


class FakePhone:
    def __init__(self, session_id: str = "s1") -> None:
        self.here = FakeHere()
        self.here_failure = None
        self.here_failures = 0
        self.telemetry = None
        self.rebinds: list[dict] = []
        self.supervisor_ended = None
        self.refusals: list[str] = []
        self.refusals_not_kept = 0
        self.sends_without_a_session = 0
        self.sends_refused = 0
        self.adapter = type("A", (), {"proxy_reasons": {}})()
        self.router = None

        class _Listener:
            displaced = 0
            handshake_workers_leaked = 0

        self._listener = _Listener()
        self._acceptor = None
        self.session = _fake_session(session_id)


class _FakeSessionStats:
    def __init__(self) -> None:
        self.channels: dict = {}


def _fake_session(session_id: str):
    return type("S", (), {
        "session_id": session_id, "is_closed": False,
        "stats": lambda self: _FakeSessionStats(),
    })()


class TestSessionScopedResetOnRedial:

    def test_a_session_scoped_counter_is_not_diffed_across_a_redial(self):
        phone = FakePhone(session_id="s1")
        s = sampler(phone=phone)
        phone.here.refused_by_reason = {"unparseable": 40}
        s.sample_once()

        # A new session, counting from zero -- HereFeed is rebuilt whole at
        # `_rebind`, so this is the honest state of a redial.
        phone.session = _fake_session("s2")
        phone.here = FakeHere()
        phone.here.refused_by_reason = {"unparseable": 3}
        s.sample_once()
        s.stop()

        record = s.to_record()
        row = record["sources"]["here.refused"]
        # No episode of size -37, and the run total is 40 + 3 = 43.
        assert row["total"] == 43
        assert not record["counter_went_backwards"]
        outcomes = record["outcomes"]
        # The old session's open episode is closed as unobservable at the
        # rebind boundary, not attributed a negative size.
        assert outcomes.get(OUTCOME_UNOBSERVABLE, 0) >= 1

    def test_camera_dropped_unconsumed_is_not_diffed_across_a_redial(self):
        # M2: the previous pin for this fix
        # (`test_the_phone_fed_camera_sources_are_readable_against_a_real_phonecamerastream`)
        # asserted only the source's declared `scope` string. `scope ==
        # "session"` stayed true even while `_read_camera_dropped_unconsumed`
        # attached no `session_id` at all, so the gate in `_process_source`
        # (`source.scope in ("session", "mixed") and snap.session_id is not
        # None`) never reset the baseline on a redial, and this source
        # reported `counter_went_backwards` on every one -- exactly the
        # output the declared scope was supposed to prevent. This drives an
        # actual redial and pins the behaviour instead of the declaration.
        phone = FakePhone(session_id="s1")
        camera = FakeCamera(decode_failures=0)  # hasattr(camera, "decode_failures") -> session scope
        camera.dropped_frames = 40
        s = sampler(phone=phone, camera=camera)
        s.sample_once()

        phone.session = _fake_session("s2")
        camera.dropped_frames = 3
        s.sample_once()
        s.stop()

        record = s.to_record()
        row = record["sources"]["camera.dropped_unconsumed"]
        assert row["total"] == 43
        assert "camera.dropped_unconsumed" not in record["counter_went_backwards"]

    def test_a_run_cumulative_counter_that_decreases_is_recorded_not_clamped(self):
        phone = FakePhone()
        s = sampler(phone=phone)
        phone.here_failures = 5
        s.sample_once()
        phone.here_failures = 2   # a run-scoped source going backwards is a real anomaly
        s.sample_once()
        s.stop()
        record = s.to_record()
        assert "here.reader_failures" in record["counter_went_backwards"]
        occurrences = record["counter_went_backwards"]["here.reader_failures"]
        assert len(occurrences) == 1
        assert occurrences[0]["from"] == 5 and occurrences[0]["to"] == 2

    def test_repeated_backwards_steps_accumulate_instead_of_overwriting(self):
        # C2: `_backwards[name] = {...}` used to overwrite, so N occurrences
        # left one entry and nothing counted how many times it happened.
        phone = FakePhone()
        s = sampler(phone=phone)
        phone.here_failures = 5
        s.sample_once()
        phone.here_failures = 2
        s.sample_once()
        phone.here_failures = 9
        s.sample_once()
        phone.here_failures = 1
        s.sample_once()
        s.stop()
        occurrences = s.to_record()["counter_went_backwards"]["here.reader_failures"]
        assert [(o["from"], o["to"]) for o in occurrences] == [(5, 2), (9, 1)]

    def test_a_pass_that_straddles_a_rebind_is_not_evaluable(self):
        phone = FakePhone(session_id="s1")
        s = sampler(phone=phone)
        s.sample_once()

        real_read = failure_log._read_here_refused

        def torn_read(ctx):
            snap = real_read(ctx)
            # Reports a DIFFERENT session id than the pass snapshot took --
            # the exact race D15 exists to catch.
            return SourceSnapshot(readable=True, by_reason=dict(snap.by_reason), session_id="a-different-session")

        # Patched on the SAMPLER's own registry, not the module-level
        # `REGISTRY`: `FailureSampler` builds its own copy from the camera it
        # was given (C2's `build_registry`), so mutating a `Source` in the
        # module tuple would patch an object nothing here reads.
        here_source = s._by_name["here.refused"]
        object.__setattr__(here_source, "read", torn_read)
        try:
            s.sample_once()
        finally:
            object.__setattr__(here_source, "read", real_read)
        row = s.to_record()["sources"]["here.refused"]
        assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE
        # M10: the plan's test 11 specifies the reason, not just the status
        # word -- any unreadable cause satisfied the status-only assertion.
        assert row["missing"] == [failure_log.MISSING_SESSION_MOVED]


# ---------------------------------------------------------------------------
# 12 & 13. The scan record exists on a driveless pass, and ticks_seen is a
# measurement, not a detection.
# ---------------------------------------------------------------------------


class TestScanRecordSurvivesNoTicks:

    def test_the_scan_record_is_written_with_no_pipeline_at_all(self):
        sink = Sink()
        s = sampler(sink)
        s.sample_once()
        scans = [l for l in sink.lines if l["type"] == "failure_scan"]
        assert len(scans) == 1
        assert scans[0]["ticks_seen"] is None
        assert scans[0]["sources_n"] == len(REGISTRY)

    def test_ticks_seen_falls_to_zero_and_recovers_with_no_threshold(self):
        class Pipeline:
            _tick_counter = 0

        pipeline = Pipeline()
        sink = Sink()
        s = sampler(sink, pipeline=pipeline)
        s.sample_once()  # first pass: baseline only
        pipeline._tick_counter = 5
        s.sample_once()
        pipeline._tick_counter = 5  # the tick loop stalled
        s.sample_once()
        pipeline._tick_counter = 12
        s.sample_once()
        scans = [l for l in sink.lines if l["type"] == "failure_scan"]
        seen = [sc["ticks_seen"] for sc in scans]
        assert seen == [0, 5, 0, 7]
        # No event was raised for the stall -- it is data, not a detection.
        assert not any(l["type"] == "failure_event" for l in sink.lines)


# ---------------------------------------------------------------------------
# 14 & 15. The tick block does not depend on sensing, and staleness is a
# real basis, not collapsed to absent.
# ---------------------------------------------------------------------------


class TestTickBlock:

    def test_the_tick_block_is_complete_with_no_pipeline_or_phone(self):
        s = sampler()
        s.sample_once()
        block = s.latest()
        assert block["basis"] == failure_log.FAILURE_BASIS_MEASURED
        assert block["open"] == []
        assert block["reason"] is None

    def test_a_stale_scan_is_stale_not_absent(self):
        now = [0.0]
        s = sampler(now=now, interval_s=1.0)
        s.sample_once()
        now[0] += 600.0
        block = s.latest(now=now[0])
        assert block["basis"] == failure_log.FAILURE_BASIS_STALE
        assert block["scan_age_s"] == 600.0

    def test_no_sample_yet_is_absent_not_stale(self):
        s = sampler()
        # Simulates the thread having started without spawning one: `latest()`
        # asks `_running` to tell "stopped, or never started" apart from
        # "started, first pass not finished yet" -- the same distinction
        # `ThermalSampler._latest_jetson` draws.
        s._running = True
        block = s.latest()
        assert block["basis"] == failure_log.FAILURE_BASIS_ABSENT
        assert block["reason"] == failure_log.FAILURE_ABSENT_NO_SAMPLE_YET

    def test_a_sampler_never_started_reports_sampler_stopped(self):
        s = sampler()
        block = s.latest()
        assert block["basis"] == failure_log.FAILURE_BASIS_ABSENT
        assert block["reason"] == failure_log.FAILURE_ABSENT_SAMPLER_STOPPED


# ---------------------------------------------------------------------------
# 16 & 17. The imported vocabularies are the controller's and time_sync's own
# objects, by identity.
# ---------------------------------------------------------------------------


class TestImportedVocabulariesAreTheRealObjects:

    def test_rule_words_are_the_controllers_own(self):
        assert failure_log.RULE_FIRED is sensing_controller.RULE_FIRED
        assert failure_log.RULE_QUIET is sensing_controller.RULE_QUIET
        assert failure_log.RULE_NOT_EVALUABLE is sensing_controller.RULE_NOT_EVALUABLE

    def test_basis_words_are_time_syncs_own(self):
        assert failure_log.FAILURE_BASIS_MEASURED is time_sync.STAGE_BASIS_MEASURED
        assert failure_log.FAILURE_BASIS_CONVERTED is time_sync.STAGE_BASIS_CONVERTED
        assert failure_log.FAILURE_BASIS_ABSENT is time_sync.STAGE_BASIS_ABSENT


# ---------------------------------------------------------------------------
# 18. Every emitted reason is a member of its source's declared vocabulary.
# ---------------------------------------------------------------------------


class TestReasonVocabulary:

    def test_every_source_declares_a_reason_shape(self):
        # A source with neither `reason` nor `vocabulary` and no explicit
        # freeform intent is exactly how a fifth vocabulary gets built by
        # accident -- this asserts every registry row picked one of the
        # documented shapes.
        for source in REGISTRY:
            assert source.reason is not None or source.vocabulary is None or callable(source.vocabulary) \
                or isinstance(source.vocabulary, frozenset)

    def test_fixed_reason_sources_only_ever_emit_their_own_word(self):
        for source in REGISTRY:
            if source.reason is not None:
                assert reason_is_valid(source, source.reason)
                assert not reason_is_valid(source, "some_other_word")

    def test_closed_vocabulary_sources_reject_a_foreign_word(self):
        # Both shapes of a declared vocabulary -- a frozenset and a predicate
        # function (`here.refused`'s prefix check against `Outcome`,
        # `link.session_end`'s against `SessionEndReason`, `link.supervisor_ended`'s
        # pattern match) -- must refuse a word that is not theirs. A source with
        # neither `vocabulary` nor `reason` set (freeform) has nothing to check.
        for source in REGISTRY:
            if source.vocabulary is not None:
                assert not reason_is_valid(source, "not-a-real-reason-xyz"), source.name

    def test_a_driven_corpus_never_emits_a_reason_outside_its_sources_vocabulary(self):
        phone = FakePhone()
        phone.here.refused_by_reason = {"unparseable": 1, "http_error:status 429": 2}
        phone.telemetry = type("T", (), {"dropped": {"camera": 1, "gps": 0, "imu": 0, "here": 0}, "here_errors": 1})()
        camera = FakeCamera(decode_failures=1)
        camera.dropped_frames = 1
        camera.file_recoveries = 1
        gps = FakeGps()
        gps.diagnostics.parse_errors = 1
        s = sampler(phone=phone, camera=camera, gps=gps)
        s.sample_once()
        s.stop()
        by_name = {src.name: src for src in REGISTRY}
        for record in s._sink.lines:
            if record.get("type") == "failure_event" and record["phase"] == "open":
                source = by_name[record["source"]]
                assert reason_is_valid(source, record["reason"]), (record["source"], record["reason"])


# ---------------------------------------------------------------------------
# 19. The episode cap counts what it does not keep.
# ---------------------------------------------------------------------------


class TestEpisodeCap:

    def test_the_cap_counts_what_it_does_not_keep(self):
        camera = FakeCamera()
        s = sampler(camera=camera, quiet_passes_to_close=1)
        total = 0
        for i in range(150):
            total += 1
            camera.decode_failures = total
            s.sample_once()   # opens (or continues) the episode
            s.sample_once()   # one quiet pass: closes it (threshold=1)
        s.stop()
        row = s.to_record()["sources"]["camera.decode_failures"]
        assert row["episodes"] == MAX_EPISODES_PER_SOURCE
        assert row["episodes_not_kept"] == 150 - MAX_EPISODES_PER_SOURCE

    def test_one_continuous_outage_past_the_cap_is_one_not_kept_episode_not_thirty(self):
        # C3: this fixture's own episode is exactly one movement pass, so it
        # cannot tell "episodes counted" from "movement passes counted" --
        # `test_the_cap_counts_what_it_does_not_keep` above passes on both the
        # broken and the fixed code for that reason. Here, ONE outage moves on
        # every pass for 30 passes straight (no quiet gap), which the bug
        # reported as 30 discarded episodes instead of the one it actually is.
        camera = FakeCamera()
        s = sampler(camera=camera, quiet_passes_to_close=1)
        total = 0
        for i in range(MAX_EPISODES_PER_SOURCE):
            total += 1
            camera.decode_failures = total
            s.sample_once()
            s.sample_once()  # closes at the cap, kept
        assert s.to_record()["sources"]["camera.decode_failures"]["episodes"] == MAX_EPISODES_PER_SOURCE

        for _ in range(30):
            total += 1
            camera.decode_failures = total
            s.sample_once()  # one continuous outage, past the cap
        s.stop()
        row = s.to_record()["sources"]["camera.decode_failures"]
        assert row["episodes"] == MAX_EPISODES_PER_SOURCE
        assert row["episodes_not_kept"] == 1
        assert row["suppressed"] == 30
        assert row["total"] == MAX_EPISODES_PER_SOURCE + 30
        assert row["kept_total"] + row["suppressed"] + row["below_episode_threshold"] == row["total"]


# ---------------------------------------------------------------------------
# M7: bound_s, events_written, and detail truncation actually reach a record.
# ---------------------------------------------------------------------------


class TestDetailCapping:

    def test_short_text_is_not_truncated(self):
        text, truncated = failure_log._capped("short")
        assert text == "short"
        assert truncated is False

    def test_none_stays_none(self):
        assert failure_log._capped(None) == (None, False)

    def test_long_text_is_cut_to_the_cap_and_the_flag_is_set(self):
        text, truncated = failure_log._capped("x" * (failure_log.DETAIL_MAX_LEN + 50))
        assert len(text) == failure_log.DETAIL_MAX_LEN
        assert truncated is True

    def test_a_truncated_detail_is_counted_on_the_source_row(self):
        # M7: `_capped`'s truncation flag was discarded at all six call
        # sites, so no record anywhere said a detail had been cut. Drives
        # `_read_camera_reader_failure`, one of the six, through a real pass.
        camera = FakeCamera()
        camera.failure = "x" * (failure_log.DETAIL_MAX_LEN + 10)
        s = sampler(camera=camera)
        s.sample_once()
        row = s.to_record()["sources"]["camera.reader_failure"]
        assert row["truncated_details"] == 1

        opens = [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "open" and l["source"] == "camera.reader_failure"
        ]
        assert len(opens[0]["detail"]) == failure_log.DETAIL_MAX_LEN


class TestEventsWrittenCountsRecords:

    def test_events_written_counts_the_open_and_close_lines(self):
        camera = FakeCamera()
        s = sampler(camera=camera, quiet_passes_to_close=1)
        s.sample_once()
        camera.decode_failures = 3
        s.sample_once()   # opens -- event_records=True for camera.decode_failures
        s.sample_once()   # closes (threshold=1)
        row = s.to_record()["sources"]["camera.decode_failures"]
        events = [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["source"] == "camera.decode_failures"
        ]
        assert len(events) == 2
        assert row["events_written"] == 2

    def test_a_source_with_no_event_records_never_writes_one_and_never_counts_one(self):
        # `gps.parse_errors` carries `event_records=False` -- an episode still
        # opens and closes for it, but `events_written` must stay 0.
        gps = FakeGps()
        s = sampler(gps=gps)
        s.sample_once()
        gps.diagnostics.parse_errors = 4
        s.sample_once()
        s.stop()
        row = s.to_record()["sources"]["gps.parse_errors"]
        assert row["episodes"] == 1
        assert row["events_written"] == 0
        assert not any(l.get("source") == "gps.parse_errors" for l in s._sink.lines if l.get("type") == "failure_event")


class TestARaisingAccessorCostsOnlyItsOwnSource:
    """M8: latent today (no accessor is known to raise), fixed anyway because
    the structure -- one raise stops the whole registry loop, freezing every
    later source's reading at whatever its last successful pass left it with
    -- makes the next accessor added a silent kill switch."""

    def test_a_raising_source_reads_not_evaluable_and_later_sources_still_run(self):
        gps = FakeGps()
        s = sampler(gps=gps)

        def boom(ctx):
            raise RuntimeError("accessor bug")

        # `gps.parse_errors` sits before `gps.rate_unconfigured` in the
        # registry -- breaking the first proves the second still ran.
        parse_errors_source = s._by_name["gps.parse_errors"]
        object.__setattr__(parse_errors_source, "read", boom)
        try:
            s.sample_once()
        finally:
            object.__setattr__(parse_errors_source, "read", failure_log._read_gps_parse_errors)

        record = s.to_record()
        broken = record["sources"]["gps.parse_errors"]
        assert broken["status"] == sensing_controller.RULE_NOT_EVALUABLE
        assert broken["missing"] == [failure_log.MISSING_ACCESSOR_RAISED]
        later = record["sources"]["gps.rate_unconfigured"]
        assert later["passes_attempted"] == 1
        assert later["passes_readable"] == 1

    def test_the_scan_record_is_still_written_after_a_raising_source(self):
        sink = Sink()
        gps = FakeGps()
        s = sampler(sink, gps=gps)

        def boom(ctx):
            raise RuntimeError("accessor bug")

        object.__setattr__(s._by_name["gps.parse_errors"], "read", boom)
        s.sample_once()
        scans = [l for l in sink.lines if l["type"] == "failure_scan"]
        assert len(scans) == 1
        assert scans[0]["seq"] == 1


class TestOpenRecordBoundS:

    def test_bound_s_equals_the_samplers_own_interval(self):
        camera = FakeCamera()
        s = sampler(camera=camera, interval_s=0.5)
        s.sample_once()
        camera.decode_failures = 1
        s.sample_once()
        opens = [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "open" and l["source"] == "camera.decode_failures"
        ]
        assert opens[0]["bound_s"] == 0.5
        assert opens[0]["basis"] == failure_log.FAILURE_BASIS_MEASURED


# ---------------------------------------------------------------------------
# 20. Inputs is unchanged -- the contract this task must not touch.
# ---------------------------------------------------------------------------


class TestInputsUnchanged:

    def test_inputs_still_has_its_seventeen_fields_in_order(self):
        from policy.sensing_controller import Inputs

        names = [f.name for f in fields(Inputs)]
        assert len(names) == 17, names


# ---------------------------------------------------------------------------
# 27. A blind tick is counted and a second one opens an episode.
# ---------------------------------------------------------------------------


class TestBlindTicks:

    def test_a_single_blind_tick_is_credited_immediately_but_opens_no_episode(self):
        # C1: a lone blind tick used to go uncredited until a second one
        # arrived to pair with, so `blind_ticks` (40) and this source's own
        # `total` (0) disagreed and `status` read `quiet` on a drive the
        # camera went blind on. Every call is now credited the instant it is
        # seen; a streak still has to run for `BLIND_EPISODE_MIN_S` before it
        # is worth an episode.
        s = sampler()
        s.note_no_frame()
        record = s.to_record()
        assert record["blind_ticks"] == 1
        row = record["sources"]["camera.blind_ticks"]
        assert row["total"] == 1
        assert row["episodes"] == 0
        assert row["status"] == sensing_controller.RULE_FIRED

    def test_a_single_blind_tick_resolved_by_a_frame_is_below_the_episode_threshold(self):
        # Recovery is only ever reported when a frame is actually observed
        # (`note_frame`) -- there is no quiet timer left that can resolve a
        # short streak on its own, because the tick loop always calls either
        # `note_no_frame` or `note_frame` on every tick; nothing goes silent
        # without one of them saying so.
        now = [0.0]
        s = sampler(now=now)
        s.note_no_frame()
        now[0] += 0.5
        s.note_frame()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        assert row["total"] == 1
        assert row["episodes"] == 0
        assert row["below_episode_threshold"] == 1
        # The reading rule this row makes checkable (C1's fix): every
        # occurrence in `total` lands in exactly one of the three.
        assert row["kept_total"] + row["suppressed"] + row["below_episode_threshold"] == row["total"]

    def test_end_of_stream_resolves_a_pending_tick_immediately(self):
        # `end_of_stream` used to be accepted and never read. The tick loop's
        # last call for a drive passes it -- no further call is coming, so a
        # tick left pending by THIS call must not wait on a quiet timer that
        # assumes one might still arrive.
        s = sampler()
        s.note_no_frame(end_of_stream=True)
        row = s.to_record()["sources"]["camera.blind_ticks"]
        assert row["total"] == 1
        assert row["below_episode_threshold"] == 1
        assert row["episodes"] == 0

    def test_end_of_stream_closes_an_open_episode_immediately(self):
        now = [0.0]
        s = sampler(now=now)
        s.note_no_frame()
        now[0] = failure_log.BLIND_EPISODE_MIN_S + 1.0
        s.note_no_frame()  # the streak has now run long enough to open the episode
        now[0] += 5.0
        s.note_no_frame(end_of_stream=True)
        record = s.to_record()
        row = record["sources"]["camera.blind_ticks"]
        assert row["total"] == 3
        assert row["episodes"] == 1
        assert record["outcomes"].get(OUTCOME_OPEN_AT_END) == 1
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["outcome"] == OUTCOME_OPEN_AT_END
        assert closes[0]["n"] == 3

    def test_a_pending_tick_at_stop_is_resolved_not_dropped(self):
        # The run ends some other way (a tick-loop exception, say) while a
        # tick is still pending -- `stop()` must not lose it either.
        s = sampler()
        s.note_no_frame()
        s.stop()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        assert row["total"] == 1
        assert row["below_episode_threshold"] == 1

    def test_a_streak_already_past_threshold_at_stop_is_promoted_then_closed(self):
        # The run ends (some other way than `end_of_stream`, e.g. the tick
        # loop's own deadline) while a blind streak is under way and has
        # already run long enough to be an episode, but nothing has called
        # `note_no_frame` again since to notice. `stop()` must promote it
        # before closing it, not silently drop it into below-threshold.
        now = [0.0]
        s = sampler(now=now)
        s.note_no_frame()
        now[0] = failure_log.BLIND_EPISODE_MIN_S + 5.0
        s.stop()
        record = s.to_record()
        row = record["sources"]["camera.blind_ticks"]
        assert row["episodes"] == 1
        assert row["below_episode_threshold"] == 0
        assert record["outcomes"].get(OUTCOME_OPEN_AT_END) == 1
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["n"] == 1

    def test_two_blind_ticks_an_hour_apart_do_not_pair_into_one_episode(self):
        # Each isolated tick is resolved by its own `note_frame` call, the
        # way the real tick loop resolves one -- a later, unrelated tick must
        # start its own streak rather than being folded into the first.
        now = [0.0]
        s = sampler(now=now)
        s.note_no_frame()
        now[0] += 0.1
        s.note_frame()  # resolves the first tick as below_episode_threshold
        now[0] += 3600.0
        s.note_no_frame()  # an unrelated, later tick -- starts its own streak
        now[0] += 0.1
        s.note_frame()
        record = s.to_record()
        row = record["sources"]["camera.blind_ticks"]
        assert row["episodes"] == 0
        assert row["below_episode_threshold"] == 2
        assert row["total"] == 2

    def test_four_consecutive_blind_ticks_spanning_the_threshold_are_one_episode_of_four(self):
        # The rule is duration, not tick count: four ticks whose span reaches
        # `BLIND_EPISODE_MIN_S` are one episode of four, back-dated to the
        # first of them.
        now = [0.0]
        s = sampler(now=now)
        period = failure_log.BLIND_EPISODE_MIN_S / 3.0
        for i in range(4):
            now[0] = i * period
            s.note_no_frame()
        s.stop()
        record = s.to_record()
        assert record["blind_ticks"] == 4
        assert record["sources"]["camera.blind_ticks"]["total"] == 4
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["n"] == 4

    @pytest.mark.parametrize("period", [0.2, 1.0, 3.0, 3.5, 4.001, 5.0])
    def test_one_continuous_blackout_gives_exactly_one_episode_at_every_failed_read_period(self, period):
        # The acceptance test the redesign exists to pass: a single 82 s
        # blackout must read as one episode no matter how far apart the
        # camera's own failed-read notifications land -- including 4.001 s,
        # the period a real drive's log shows, which is wider than
        # `close_after_s` (3.0 s) and used to fragment into several episodes
        # under the old quiet-timer close.
        blackout_s = 82.0
        scan_interval = 1.0
        now = [0.0]
        s = sampler(now=now, interval_s=scan_interval)
        t = 0.0
        next_scan = scan_interval
        next_note = 0.0
        while t < blackout_s:
            t = min(next_scan, next_note)
            now[0] = t
            if abs(t - next_note) < 1e-9:
                s.note_no_frame()
                next_note += period
            if abs(t - next_scan) < 1e-9:
                s.sample_once()
                next_scan += scan_interval
        now[0] = blackout_s + 0.01
        s.note_no_frame(end_of_stream=True)
        row = s.to_record()["sources"]["camera.blind_ticks"]
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert row["episodes"] == 1
        assert len(closes) == 1
        assert closes[0]["outcome"] == OUTCOME_OPEN_AT_END
        assert closes[0]["duration_s"] == pytest.approx(blackout_s, abs=1.0)

    def test_a_camera_that_recovers_mid_run_reads_recovered_not_unobservable(self):
        # The other half of the acceptance test: a real recovery must still
        # read `recovered`. The old design got this direction right and the
        # outage direction wrong; a fix that only ever says `unobservable`
        # would get this direction wrong instead.
        now = [0.0]
        s = sampler(now=now)
        for i in range(21):
            now[0] = float(i)
            s.note_no_frame()
            s.sample_once()
        now[0] = 21.0
        s.note_frame()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert row["episodes"] == 1
        assert closes[0]["outcome"] == OUTCOME_RECOVERED
        assert closes[0]["duration_s"] == pytest.approx(20.0, abs=0.5)

    def test_silence_with_no_observed_frame_closes_as_unobservable_not_recovered(self):
        # An open episode that goes quiet for `BLIND_EPISODE_MIN_S` with no
        # further notification and no `note_frame` call is not evidence the
        # camera recovered -- the drive says nothing about that window.
        now = [0.0]
        s = sampler(now=now)
        for i in range(21):
            now[0] = float(i)
            s.note_no_frame()
        now[0] = 21.0 + failure_log.BLIND_EPISODE_MIN_S
        s.sample_once()  # nothing further arrives; the quiet bound alone closes it
        row = s.to_record()["sources"]["camera.blind_ticks"]
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert row["episodes"] == 1
        assert closes[0]["outcome"] == OUTCOME_UNOBSERVABLE

    def test_an_unobservable_close_reports_when_blindness_was_last_confirmed_not_the_last_scan(self):
        # A16: `last_readable_t_mono` advances on every ordinary scan pass
        # regardless of this pseudo-source's real state -- its accessor
        # always reports readable=True (PSEUDO_SOURCES) -- so it carries no
        # information about the camera at all. Using it as the unobservable
        # close's own instant credited every scan pass that ran after the
        # tick loop had already died with no more evidence, overstating the
        # observed blindness by however long the sampler kept scanning past
        # the last real notification.
        now = [0.0]
        s = sampler(now=now)
        # Notifications every 1.0 s from t=2 to t=16 (15 of them); the
        # episode promotes once the streak reaches BLIND_EPISODE_MIN_S at
        # t=12, back-dated to t=2.
        for t in range(2, 17):
            now[0] = float(t)
            s.note_no_frame()
        # The tick loop then dies -- no more note_no_frame, no
        # end_of_stream -- but the sampler's own 1 Hz scan thread keeps
        # running regardless, independent of the tick loop.
        for t in range(17, 27):
            now[0] = float(t)
            s.sample_once()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert row["episodes"] == 1
        assert closes[0]["outcome"] == OUTCOME_UNOBSERVABLE
        assert closes[0]["n"] == 15
        # The blindness was confirmed from t=2 (first notification) to
        # t=16 (last notification): 14.0 s, not 24.0 s (to t=26, the last
        # scan pass before the quiet bound tripped).
        assert closes[0]["duration_s"] == pytest.approx(14.0)

    def test_the_tick_loop_exception_is_recorded(self):
        s = sampler()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            s.note_pipeline_exception(exc)
        record = s.to_record()
        assert record["pipeline_exception"] == "ValueError"
        opens = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["source"] == "pipeline.exception"]
        assert opens[0]["reason"] == "ValueError"

    def test_three_consecutive_pipeline_exceptions_accumulate_in_one_episode(self):
        # M1: `note_pipeline_exception` credited `run_total` on every call but
        # only grew the open episode's own `n` on the first one, since there
        # was no `else` branch to match `note_no_frame`'s. Three exceptions
        # used to give `total == 3` against `kept_total == 1`.
        s = sampler()
        for msg in ("a", "b", "c"):
            try:
                raise ValueError(msg)
            except ValueError as exc:
                s.note_pipeline_exception(exc)
        s.stop()
        row = s.to_record()["sources"]["pipeline.exception"]
        assert row["total"] == 3
        assert row["episodes"] == 1
        assert row["kept_total"] + row["suppressed"] + row["below_episode_threshold"] == row["total"]
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["n"] == 3


# ---------------------------------------------------------------------------
# M1: the accounting invariant, walked across every registry row rather than
# the two single-source spots that pinned it before.
# ---------------------------------------------------------------------------


class TestAccountingInvariantAcrossAllSources:

    def test_the_invariant_holds_on_every_registry_row(self):
        phone = FakePhone()
        phone.here.refused_by_reason = {"unparseable": 3}
        phone.telemetry = type(
            "T", (), {"dropped": {"camera": 2, "gps": 1, "imu": 0, "here": 0}, "here_errors": 1},
        )()
        camera = FakeCamera(decode_failures=4)
        camera.dropped_frames = 2
        camera.file_recoveries = 1
        gps = FakeGps()
        gps.diagnostics.parse_errors = 3
        s = sampler(phone=phone, camera=camera, gps=gps)
        s.sample_once()
        s.note_no_frame()
        s.note_no_frame()
        try:
            raise ValueError("x")
        except ValueError as exc:
            s.note_pipeline_exception(exc)
            s.note_pipeline_exception(exc)
            s.note_pipeline_exception(exc)
        s.sample_once()
        s.stop()

        record = s.to_record()
        assert len(record["sources"]) == len(REGISTRY)
        for name, row in record["sources"].items():
            assert row["kept_total"] + row["suppressed"] + row["below_episode_threshold"] == row["total"], name

        # The specific defect this walk exists to catch: three exceptions
        # must land as three occurrences, not collapse to the first one.
        assert record["sources"]["pipeline.exception"]["total"] == 3
        assert record["sources"]["pipeline.exception"]["kept_total"] == 3


class TestPipelineExceptionIsCumulative:
    """m1: `pipeline.exception` counts one real occurrence per call, the same
    way `camera.blind_ticks` does -- both are direct-notification pseudo-
    sources with no counter of their own to diff. `camera.blind_ticks`
    declares `cumulative=True`; `pipeline.exception` declared `False`, which
    disagreed with what `total` actually counts and made `report.md`
    describe three exceptions as "3 passes with the condition active"."""

    def test_the_registry_declares_it_cumulative(self):
        by_name = {s.name: s for s in REGISTRY}
        assert by_name["pipeline.exception"].cumulative is True
        assert by_name["pipeline.exception"].cumulative == by_name["camera.blind_ticks"].cumulative


# ---------------------------------------------------------------------------
# D1: a direct-notification pseudo-source's episode is governed by its own
# timer, never by the generic scan's quiet-streak path.
# ---------------------------------------------------------------------------


class TestPseudoSourceEpisodesAreNotClosedByTheGenericScan:
    """A continuously blind camera interleaves the tick loop's direct
    `note_no_frame` calls with the sampler's own independent 1 Hz scan
    thread. `camera.blind_ticks`' accessor always reports an empty
    `by_reason` -- there is no counter on the scan path for it to read --
    so a scan pass used to read that emptiness as one quiet pass, and three
    of them (the default `quiet_passes_to_close`) closed the episode as
    `recovered` on a fixed schedule tied to the number of scan passes since
    it opened, regardless of how many direct calls proved the outage was
    still live in between. On the real drive, an 82 s outage this way
    became 21 separate `recovered` episodes instead of one open outage."""

    def _closes(self, s, source_name):
        return [
            l for l in s._sink.lines
            if l.get("type") == "failure_event" and l["phase"] == "close" and l["source"] == source_name
        ]

    def test_scan_passes_do_not_close_an_open_episode_while_direct_calls_keep_arriving(self):
        min_s = failure_log.BLIND_EPISODE_MIN_S
        now = [0.0]
        s = sampler(now=now)
        now[0] = 0.0
        s.note_no_frame()  # pending
        now[0] = min_s + 0.1
        s.note_no_frame()  # the streak has run long enough now: opens, n=2, back-dated to 0.0
        st = s._state["camera.blind_ticks"]
        assert st.open_episode is not None
        opened_episode_id = st.open_episode.episode_id

        # Three scan passes elapse, each well inside `BLIND_EPISODE_MIN_S` of
        # the last direct call -- not enough silence to close the episode,
        # even though a scan pass has nothing of its own to read for this
        # source and the outage itself never actually paused.
        for dt in (2.0, 4.0, 6.0):
            now[0] = min_s + 0.1 + dt
            s.sample_once()
        assert st.open_episode is not None
        assert st.open_episode.episode_id == opened_episode_id
        assert not self._closes(s, "camera.blind_ticks")

        # More direct calls, still well inside the outage, then more scan
        # passes -- the episode must still be the same one open episode.
        now[0] = min_s + 6.5
        s.note_no_frame()
        now[0] = min_s + 7.4
        s.note_no_frame()
        for dt in (1.6, 3.6, 5.6):
            now[0] = min_s + 7.4 + dt
            s.sample_once()
        assert st.open_episode is not None
        assert st.open_episode.episode_id == opened_episode_id
        assert st.open_episode.n == 4
        assert not self._closes(s, "camera.blind_ticks")

        s.stop()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        assert row["episodes"] == 1
        assert row["total"] == 4
        closes = self._closes(s, "camera.blind_ticks")
        assert len(closes) == 1
        assert closes[0]["episode_id"] == opened_episode_id
        assert closes[0]["outcome"] == OUTCOME_OPEN_AT_END
        assert closes[0]["n"] == 4

    def test_pipeline_exceptions_open_episode_is_not_closed_by_a_scan_pass_either(self):
        # `pipeline.exception` is the other direct-notification pseudo-
        # source and shares the same accessor shape (`_read_pipeline_
        # exception` always reports an empty `by_reason`), so it is exposed
        # to the identical hazard even though the drive's own defect
        # (D1) surfaced through the camera.
        now = [0.0]
        s = sampler(now=now)
        try:
            raise ValueError("boom")
        except ValueError as exc:
            s.note_pipeline_exception(exc)
        st = s._state["pipeline.exception"]
        opened_episode_id = st.open_episode.episode_id
        for t in (1.0, 2.0, 3.0, 4.0):
            now[0] = t
            s.sample_once()
        assert st.open_episode is not None
        assert st.open_episode.episode_id == opened_episode_id
        assert not self._closes(s, "pipeline.exception")
        s.stop()
        row = s.to_record()["sources"]["pipeline.exception"]
        assert row["episodes"] == 1
        closes = self._closes(s, "pipeline.exception")
        assert len(closes) == 1
        assert closes[0]["episode_id"] == opened_episode_id


# ---------------------------------------------------------------------------
# D7: `camera.blind_ticks.passes_attempted` counts scan passes only, the way
# it does for every other of the thirty registry rows -- not scans plus
# direct notifications combined.
# ---------------------------------------------------------------------------


class TestPseudoSourcePassesAttemptedCountsScanPassesOnly:

    def test_camera_blind_ticks_passes_attempted_ignores_direct_notifications(self):
        now = [0.0]
        s = sampler(now=now)
        for t in (0.0, 1.0, 2.0):
            now[0] = t
            s.sample_once()
        for _ in range(10):
            s.note_no_frame()
        s.stop()
        row = s.to_record()["sources"]["camera.blind_ticks"]
        assert row["passes_attempted"] == 3
        assert row["passes_readable"] == 3
        assert row["total"] == 10
        assert row["status"] == sensing_controller.RULE_FIRED

    def test_pipeline_exception_passes_attempted_ignores_direct_notifications(self):
        now = [0.0]
        s = sampler(now=now)
        for t in (0.0, 1.0):
            now[0] = t
            s.sample_once()
        for msg in ("a", "b", "c"):
            try:
                raise ValueError(msg)
            except ValueError as exc:
                s.note_pipeline_exception(exc)
        s.stop()
        row = s.to_record()["sources"]["pipeline.exception"]
        assert row["passes_attempted"] == 2
        assert row["passes_readable"] == 2
        assert row["total"] == 3


# ---------------------------------------------------------------------------
# D2: `missing` names the reason from the last unreadable pass, and survives
# the source becoming readable again before teardown.
# ---------------------------------------------------------------------------


class TestMissingSurvivesTheSourceBecomingReadableAgain:
    """`missing` used to be cleared on every readable pass, so a source
    unreadable mid-drive and readable again by teardown reported
    `not_evaluable` naming nothing -- on every drive with a redial, since a
    redial is exactly a source going unreadable and then readable again."""

    def test_a_source_unreadable_mid_drive_and_readable_at_teardown_still_names_missing(self):
        phone = FakePhone()
        s = sampler(phone=phone)
        s.sample_once()  # readable: session present
        phone.session = None
        s.sample_once()  # not evaluable this pass: no session
        phone.session = _fake_session("s1")
        s.sample_once()  # readable again, well before teardown
        s.stop()
        row = s.to_record()["sources"]["wire.dropped"]
        assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE
        assert row["passes_attempted"] == 3
        assert row["passes_readable"] == 2
        assert row.get("missing") == [failure_log.MISSING_NO_SESSION]


# ---------------------------------------------------------------------------
# D4: the two phone-telemetry sources go `not_evaluable`, not `quiet`, while
# the link is down -- `PhoneLink.telemetry` is not cleared on session loss.
# ---------------------------------------------------------------------------


class TestPhoneTelemetrySourcesAreNotEvaluableWhileTheLinkIsDown:
    """`_read_phone_dropped` and `_read_phone_here_errors` read `PhoneLink.
    telemetry`, which is cleared on a rebind (`_rebind`), not on session
    loss. Without a session gate, a stale pre-outage snapshot was read as
    current for as long as the link stayed down, and both sources reported
    the phone's failure counters as unchanged -- and therefore `quiet` --
    through a window this drive never actually observed. `link.down` is
    supposed to make every phone-side source honest about exactly this."""

    def test_phone_dropped_goes_not_evaluable_when_the_session_drops(self):
        phone = FakePhone()
        phone.telemetry = type(
            "T", (), {"dropped": {"camera": 0, "gps": 0, "imu": 0, "here": 0}, "here_errors": 0},
        )()
        s = sampler(phone=phone)
        s.sample_once()
        assert s.to_record()["sources"]["phone.dropped"]["status"] == sensing_controller.RULE_QUIET

        phone.session = None  # the link goes down; the stale telemetry object is untouched
        s.sample_once()
        row = s.to_record()["sources"]["phone.dropped"]
        assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE
        assert row["missing"] == [failure_log.MISSING_NO_SESSION]

    def test_phone_here_errors_goes_not_evaluable_when_the_session_drops(self):
        phone = FakePhone()
        phone.telemetry = type("T", (), {"dropped": {}, "here_errors": 0})()
        s = sampler(phone=phone)
        s.sample_once()
        assert s.to_record()["sources"]["phone.here_errors"]["status"] == sensing_controller.RULE_QUIET

        phone.session = None
        s.sample_once()
        row = s.to_record()["sources"]["phone.here_errors"]
        assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE
        assert row["missing"] == [failure_log.MISSING_NO_SESSION]


# ---------------------------------------------------------------------------
# D5: `by_reason_total` credits every reason that moved on a pass, not only
# the single largest-moving one.
# ---------------------------------------------------------------------------


class TestByReasonCreditsEveryReasonThatMovedInAPass:
    """`clock.proxied`'s own counters across a drive gave `{no_samples: 3,
    only_1: 11, only_2: 11, only_3: 13, only_4: 13}`; the failure log
    reported `{only_1: 11, only_2: 11, only_3: 11, only_4: 18}`. The reason
    named on an episode's own record is, and stays, the single dominant one
    (D9) -- but the source's own `by_reason_total` answers a different
    question, how many times each reason occurred, and crediting only the
    dominant reason with a whole pass's total delta silently dropped every
    other reason that moved in the same pass. The total was right and the
    breakdown was wrong, which the plan calls worse than an obviously wrong
    total."""

    def test_a_pass_where_two_reasons_move_together_credits_both(self):
        phone = FakePhone()
        s = sampler(phone=phone)
        phone.adapter.proxy_reasons = {"no_samples": 1, "only_4": 5}
        s.sample_once()
        phone.adapter.proxy_reasons = {"no_samples": 1, "only_4": 10, "only_3": 2}
        s.sample_once()
        s.stop()
        row = s.to_record()["sources"]["clock.proxied"]
        assert row["by_reason"] == {"no_samples": 1, "only_4": 10, "only_3": 2}
        assert row["total"] == 13
        assert sum(row["by_reason"].values()) == row["total"]

    def test_the_drives_own_numbers_reconcile_exactly(self):
        # The exact counters the drive reported, replayed as two passes:
        # everything moves together in the first pass, and the remaining
        # `only_3` and `only_4` movement lands in the second.
        phone = FakePhone()
        s = sampler(phone=phone)
        phone.adapter.proxy_reasons = {
            "no_samples": 3, "only_1": 11, "only_2": 11, "only_3": 11, "only_4": 13,
        }
        s.sample_once()
        phone.adapter.proxy_reasons = {
            "no_samples": 3, "only_1": 11, "only_2": 11, "only_3": 13, "only_4": 13,
        }
        s.sample_once()
        s.stop()
        row = s.to_record()["sources"]["clock.proxied"]
        assert row["by_reason"] == {
            "no_samples": 3, "only_1": 11, "only_2": 11, "only_3": 13, "only_4": 13,
        }
        assert row["total"] == 51
