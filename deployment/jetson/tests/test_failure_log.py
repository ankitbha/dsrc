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
    would have caught the task-37 class of defect at plan time."""

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
                blind_ticks_total=0, pipeline_exception=None,
                pass_session_id=phone.session.session_id,
            )
            for source in REGISTRY:
                snap = source.read(ctx)
                assert isinstance(snap, SourceSnapshot), source.name
                if snap.readable:
                    for value in snap.by_reason.values():
                        assert isinstance(value, int), source.name
                else:
                    assert snap.missing in MISSING, source.name
        finally:
            phone.stop()
            logger.close()


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
        camera = type("C", (), {"decode_failures": 0})()
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
        camera = type("C", (), {"decode_failures": 0})()
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
        camera = type("C", (), {"decode_failures": 0})()
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
        camera = type("C", (), {"decode_failures": 0})()
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
        seen = set()

        # recovered
        camera = type("C", (), {"decode_failures": 0})()
        s = sampler(camera=camera, quiet_passes_to_close=1)
        s.sample_once()
        camera.decode_failures = 1
        s.sample_once()
        s.sample_once()  # one quiet pass closes it (threshold=1)
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        seen.add(closes[0]["outcome"])

        # unobservable
        camera2 = type("C", (), {"decode_failures": 0})()
        s2 = FailureSampler(Sink(), camera=camera2)
        s2.sample_once()
        camera2.decode_failures = 1
        s2.sample_once()
        s2._camera = None
        s2.sample_once()
        closes2 = [l for l in s2._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        seen.add(closes2[0]["outcome"])

        # open_at_end
        camera3 = type("C", (), {"decode_failures": 0})()
        s3 = FailureSampler(Sink(), camera=camera3)
        s3.sample_once()
        camera3.decode_failures = 1
        s3.sample_once()
        s3.stop()
        closes3 = [l for l in s3._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
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
        step = record["counter_went_backwards"]["here.reader_failures"]
        assert step["from"] == 5 and step["to"] == 2

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

        here_source = next(src for src in REGISTRY if src.name == "here.refused")
        object.__setattr__(here_source, "read", torn_read)
        try:
            s.sample_once()
        finally:
            object.__setattr__(here_source, "read", real_read)
        row = s.to_record()["sources"]["here.refused"]
        assert row["status"] == sensing_controller.RULE_NOT_EVALUABLE


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
        camera = type("C", (), {"decode_failures": 1, "dropped_frames": 1, "file_recoveries": 1,
                                 "end_of_stream": False, "failure": None})()
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
        camera = type("C", (), {"decode_failures": 0})()
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

    def test_a_single_blind_tick_opens_nothing(self):
        s = sampler()
        s.note_no_frame()
        record = s.to_record()
        assert record["blind_ticks"] == 1
        assert record["sources"]["camera.blind_ticks"]["total"] == 0

    def test_four_consecutive_blind_ticks_are_one_episode_of_four(self):
        s = sampler()
        for _ in range(4):
            s.note_no_frame()
        s.stop()
        record = s.to_record()
        assert record["blind_ticks"] == 4
        assert record["sources"]["camera.blind_ticks"]["total"] == 4
        closes = [l for l in s._sink.lines if l.get("type") == "failure_event" and l["phase"] == "close"]
        assert closes[0]["n"] == 4

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
