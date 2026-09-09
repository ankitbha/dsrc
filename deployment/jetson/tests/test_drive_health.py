"""The drive watcher's verdicts.

The point of these tests is the third outcome. This project has a drive whose `usb2`
reported `connected` through a total network outage, and a `camera.blind_ticks:
quiet` on a drive with 40 blind episodes -- both cases of an unmeasured condition
taking the code path meant for a measured negative. So the assertions below spend
most of their effort on the difference between "checked and fine" and "could not
check", because that is the distinction the instrument exists to preserve.

Thresholds are asserted against the measurements they came from, so a later edit to
a number has to argue with the drive that produced it.
"""
from __future__ import annotations

import pytest

from drive_health import (
    BROKEN,
    DEGRADED,
    LINK_DEGRADED_MS,
    OK,
    TICK_STALL_S,
    UNKNOWN,
    assess,
    check_clock,
    check_gps,
    check_link,
    check_thermal,
    check_ticks,
    check_video,
    worst_of,
)

HEALTHY = {
    "tick_count": 200, "last_tick_age_s": 0.2, "link_ms": 52.0,
    "thermal_status": "nominal", "skin_temp_c": 31.0,
    "gps_valid": True, "gps_fix_fraction": 1.0,
    "video_bytes": 9_000_000, "index_lines": 97, "video_enabled": True,
    "disk_free_gb": 640.0, "ntp_synchronised": True,
}


class TestCouldNotCheckIsNotFine:
    """The whole reason this module is tri-state."""

    @pytest.mark.parametrize("check,args", [
        (check_ticks, (None, None)),
        (check_link, (None,)),
        (check_thermal, (None, None)),
        (check_gps, (None, None)),
        (check_clock, (None,)),
    ])
    def test_a_missing_fact_is_unknown_never_ok(self, check, args):
        result = check(*args)
        assert result.verdict == UNKNOWN, result
        assert result.detail, "an UNKNOWN with no reason is unattributable"

    def test_unknown_outranks_ok_in_the_summary(self):
        # A run where one thing could not be checked must not report OK, or the
        # headline launders the gap away.
        assert worst_of([OK, UNKNOWN]) == UNKNOWN
        assert worst_of([OK, OK]) == OK
        assert worst_of([OK, UNKNOWN, DEGRADED]) == DEGRADED
        assert worst_of([DEGRADED, BROKEN]) == BROKEN

    def test_an_empty_verdict_list_is_unknown_not_ok(self):
        # The fall-through case. Nothing checked means nothing known.
        assert worst_of([]) == UNKNOWN

    def test_a_drive_with_nothing_readable_is_not_healthy(self):
        health = assess({})
        assert health.verdict in (BROKEN, UNKNOWN)
        assert health.verdict != OK
        # And it says which checks it could not make, by name.
        assert "could not check" in health.to_record()["line"]


class TestTheFailuresThatActuallyHappened:

    def test_the_thermal_death_is_caught_by_link(self):
        # The observable on the tick where the 2026-09-08 session died. No gate in
        # the system looks at link_ms; this is the only thing that would have said.
        result = check_link(135_359.0)
        assert result.verdict == BROKEN
        assert "keeping up" in result.detail

    def test_a_normal_link_passes(self):
        assert check_link(52.0).verdict == OK
        assert check_link(110.0).verdict == OK      # the measured p95

    def test_the_link_threshold_sits_between_the_p95_and_the_death(self):
        # Asserted so a future edit has to disagree with the drive, not with taste.
        assert 110.0 < LINK_DEGRADED_MS < 135_359.0

    def test_severe_thermal_is_broken_and_moderate_is_only_degraded(self):
        # severe is where a drive died. moderate is what the backoff is for, and it
        # handled it, so calling it broken would cry wolf on correct behaviour.
        assert check_thermal("severe", 51.6).verdict == BROKEN
        assert check_thermal("moderate", 40.2).verdict == DEGRADED
        assert check_thermal("nominal", 31.0).verdict == OK

    def test_a_stalled_tick_stream_is_broken(self):
        assert check_ticks(1313, 0.2).verdict == OK
        assert check_ticks(1313, TICK_STALL_S + 1).verdict == BROKEN
        # The largest legitimate gap measured on a drive was 2.6 s, so that must pass.
        assert check_ticks(1313, 2.6).verdict == OK

    def test_no_ticks_at_all_says_the_phone_never_dialled(self):
        result = check_ticks(0, 0.0)
        assert result.verdict == BROKEN
        assert "dialled" in result.detail

    def test_frame_logging_off_is_broken_because_the_drive_cannot_explain_itself(self):
        # 2026-09-08: 1,632 ticks of frames decoded, detected on and discarded, and
        # the drive's central question was unanswerable from its own record.
        result = check_video(None, None, video_enabled=False)
        assert result.verdict == BROKEN
        assert "re-processed" in result.detail

    def test_no_gps_fix_is_broken(self):
        assert check_gps(True, 1.0).verdict == OK
        assert check_gps(False, 0.0).verdict == BROKEN
        assert check_gps(True, 0.5).verdict == DEGRADED

    def test_an_unsynced_clock_is_flagged(self):
        # Task 65: t_wall steps mid-run if the drive starts before NTP lands.
        assert check_clock(False).verdict == DEGRADED
        assert check_clock(True).verdict == OK


class TestTheHealthyCaseStillPasses:
    """A check that cannot pass is as useless as one that cannot fail."""

    def test_a_good_drive_reports_ok(self):
        health = assess(HEALTHY)
        assert health.verdict == OK, health.to_record()["line"]
        assert "all checks passed" in health.to_record()["line"]

    def test_every_check_is_named_in_the_record(self):
        record = assess(HEALTHY).to_record()
        assert set(record["checks"]) == {
            "ticks", "link", "thermal", "gps", "video", "disk", "clock"}
        for name, entry in record["checks"].items():
            assert entry["verdict"] in (OK, DEGRADED, BROKEN, UNKNOWN), name
            assert entry["detail"], f"{name} has no reason"

    def test_one_bad_check_names_itself_in_the_line(self):
        health = assess({**HEALTHY, "thermal_status": "severe", "skin_temp_c": 51.6})
        line = health.to_record()["line"]
        assert health.verdict == BROKEN
        assert "thermal" in line and "severe" in line


class TestItWatchesTheRunThatIsActuallyBeingWritten:
    """Both defects below shipped, and both were found by the same drive.

    On 2026-09-08 the watcher reported a stall on a healthy drive because it was
    reading a run that had already ended. Nothing in the twenty tests above could
    have caught it: they all hand `assess` a facts dict and never ask which run the
    facts came from.
    """

    def test_a_finished_run_does_not_outrank_the_live_one(self, tmp_path):
        import os

        from drive_health import newest_run

        # The live run was created first and is still appending to metadata.
        live = tmp_path / "run_20260908_182926"
        (live / "here").mkdir(parents=True)
        (live / "metadata.jsonl").write_text('{"type": "tick"}\n')

        # The finished run was created later, wrote less, and then had summary.json
        # created at teardown -- which is what bumps a DIRECTORY's mtime. Its name
        # also sorts above the live run's, so name ordering cannot save us either.
        dead = tmp_path / "run_20260908_183538"
        dead.mkdir()
        (dead / "metadata.jsonl").write_text('{"type": "tick"}\n')
        os.utime(dead / "metadata.jsonl", (1000, 1000))
        (dead / "summary.json").write_text("{}")

        # Live metadata written most recently; dead directory touched most recently.
        os.utime(live / "metadata.jsonl", (9000, 9000))
        os.utime(live, (1, 1))
        os.utime(dead, (9999, 9999))

        assert newest_run(str(tmp_path)) == str(live), (
            "picked the run whose directory was touched last, not the one being written"
        )

    def test_a_short_run_does_not_lose_its_first_tick(self, tmp_path):
        from drive_health import _tail_ticks

        path = tmp_path / "metadata.jsonl"
        # Well under the 400 KB tail window, so the seek starts at 0 and every line
        # is a complete record.
        path.write_text("".join(
            '{"type": "tick", "tick_id": %d}\n' % i for i in range(3)
        ))
        ticks = _tail_ticks(str(path))
        assert [t["tick_id"] for t in ticks] == [0, 1, 2]

    def test_a_long_run_still_drops_the_partial_first_record(self, tmp_path):
        from drive_health import _tail_ticks

        path = tmp_path / "metadata.jsonl"
        # Past the window, so the seek lands mid-record and the leading fragment
        # must still be discarded rather than parsed.
        filler = '{"type": "tick", "tick_id": -1, "pad": "%s"}\n' % ("x" * 900)
        body = filler * 500 + '{"type": "tick", "tick_id": 7}\n'
        path.write_text(body)
        ticks = _tail_ticks(str(path))
        assert ticks, "the tail window produced nothing"
        assert ticks[-1]["tick_id"] == 7
