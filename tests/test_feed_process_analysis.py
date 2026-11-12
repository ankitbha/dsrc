"""The feed-process analysis: synthetic polling records, then the real drive logs.

The drive-log tests read `/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908`
(the logs of 2026-09-08) and check the numbers the plan's appendix reports.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import analyze_feed_process as afp  # noqa: E402

from src.sumo.feed import FeedProcess  # noqa: E402

DRIVES = REPO_ROOT.parent / "dsrc_drive_data_20260908"
BASE = 1_800_000_000.0  # a wall-clock origin on a whole minute


def _stamp(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _publications(gaps, first=BASE + 36.0):
    times = [first]
    for gap in gaps:
        times.append(times[-1] + gap)
    return times


def _records(publications, lag, query_times, area="nj", rtt=0.0):
    """Polling records: a query sent at q returns the newest version available at q."""
    out = []
    for seq, q in enumerate(sorted(query_times)):
        available = [p for p in publications if p + lag <= q]
        out.append({"type": "request", "seq": seq, "area": area, "t_send_wall": q,
                    "t_recv_wall": q + rtt, "status": 200,
                    "sourceUpdated": _stamp(available[-1]) if available else None,
                    "rtt_s": rtt, "Age": "0"})
    return out


GAPS = [60, 26, 34, 60, 26, 34, 60, 60]


class TestAnalyzeRecords:
    def test_queries_straddling_each_availability_recover_gaps_lag_and_zero_jitter(self):
        pubs = _publications(GAPS)
        # One query a millisecond before and one at each availability instant.
        queries = sorted([p + 45.0 - 0.001 for p in pubs] + [p + 45.0 for p in pubs])
        out = afp.analyze_records(_records(pubs, 45.0, queries))["nj"]
        assert out["gaps"] == [float(g) for g in GAPS]
        assert out["gap_counts"] == {"26": 2, "34": 2, "60": 4}
        assert out["phase_counts"] == {"2": 2, "36": 7}
        assert all(v["L_up"] == pytest.approx(45.0) for v in out["per_version"])
        assert out["L_bar"] == pytest.approx(45.0)
        assert out["J"] == pytest.approx(0.0, abs=0.002)
        assert [s[0] for s in out["sequence"]] == [float(g) for g in GAPS]
        assert all(s[1] == pytest.approx(45.0) for s in out["sequence"])

    def test_five_second_polling_brackets_the_lag_to_within_one_interval(self):
        pubs = _publications(GAPS)
        queries = np.arange(BASE, pubs[-1] + 120.0, 5.0)
        out = afp.analyze_records(_records(pubs, 45.0, queries))["nj"]
        for version in out["per_version"][1:]:
            assert version["L_lo"] < 45.0 <= version["L_up"]
            assert version["L_up"] - version["L_lo"] == pytest.approx(5.0)
        assert 5.0 <= out["J"] < 10.0

    def test_the_first_version_of_a_session_has_no_lower_bound(self):
        pubs = _publications(GAPS)
        queries = np.arange(pubs[0] + 50.0, pubs[-1] + 120.0, 5.0)
        out = afp.analyze_records(_records(pubs, 45.0, queries))["nj"]
        assert out["per_version"][0]["L_lo"] is None
        assert out["L_low"] == min(v["L_lo"] for v in out["per_version"][1:])

    def test_failed_and_older_responses_do_not_make_versions(self):
        pubs = _publications([60, 60])
        records = _records(pubs, 45.0, np.arange(BASE + 40.0, BASE + 260.0, 5.0))
        records[5]["status"] = 503
        records[5]["sourceUpdated"] = None
        # A cache serving the previous version after the new one was seen.
        stale = dict(records[-1], seq=999, t_send_wall=records[-1]["t_send_wall"] + 1,
                     t_recv_wall=records[-1]["t_recv_wall"] + 1,
                     sourceUpdated=_stamp(pubs[0]))
        out = afp.analyze_records(records + [stale])["nj"]
        assert out["versions"] == 3
        assert out["regressions"] == 1

    def test_rtt_and_age_statistics(self):
        pubs = _publications([60])
        out = afp.analyze_records(_records(pubs, 45.0, np.arange(BASE + 40, BASE + 160, 5.0),
                                           rtt=0.25))["nj"]
        assert out["rtt_s"]["max"] == pytest.approx(0.25)
        assert out["r_bar"] == pytest.approx(0.25)
        assert out["age_header_s"]["max"] == 0.0

    def test_two_areas_are_kept_apart(self):
        pubs = _publications([60, 26])
        records = (_records(pubs, 45.0, np.arange(BASE, BASE + 300, 5.0), area="nj")
                   + _records(pubs, 30.0, np.arange(BASE + 2.5, BASE + 300, 5.0), area="mainz"))
        out = afp.analyze_records(records)
        assert set(out) == {"nj", "mainz"}
        assert out["mainz"]["L_bar"] < out["nj"]["L_bar"]


class TestPsiStar:
    def test_the_second_farthest_from_the_support(self):
        psi, margin = afp.psi_star([10.0, 20.0], r_bar=0.0)
        assert psi == 45 and margin == pytest.approx(25.0)

    def test_the_window_width_counts(self):
        psi, margin = afp.psi_star([0.0], r_bar=10.0)
        assert psi == 25 and margin == pytest.approx(25.0)


@pytest.fixture(scope="module")
def drive():
    return afp.analyze_drive_logs(DRIVES)


@pytest.mark.skipif(not DRIVES.exists(), reason="drive logs are not on this machine")
class TestTheDriveLogs:
    def test_sixty_versions_in_123_responses(self, drive):
        assert drive["overall"]["responses"] == 123
        assert drive["overall"]["versions"] == 60
        runs = {name: (s["responses"], s["versions"]) for name, s in drive["sessions"].items()}
        assert runs == {"drive_run_20260908_170849": (93, 30),
                        "drive_run_20260908_183538": (2, 2),
                        "drive_run_20260908_190548": (28, 28)}

    def test_the_first_receipt_lag_matches_the_appendix(self, drive):
        lag = drive["overall"]["L_up"]
        assert round(lag["min"], 1) == 33.8
        assert round(lag["p50"], 1) == 62.8
        assert round(lag["max"], 1) == 109.7

    def test_cadence_and_phase_match_the_appendix(self, drive):
        assert drive["overall"]["gap_counts"] == {"26": 4, "34": 4, "60": 48, "120": 1}
        assert drive["overall"]["phase_counts"] == {"2": 22, "36": 38}

    def test_a_lower_end_below_zero_is_clipped(self, drive):
        # run_20260908_190548 skipped 23:06:02: 23:07:02 arrived 49.8 s after its stamp
        # and the query before it 60 s earlier, so the raw lower end is -10.2 s.
        versions = {v["sourceUpdated"]: v
                    for v in drive["sessions"]["drive_run_20260908_190548"]["per_version"]}
        assert versions["2026-09-08T23:07:02Z"]["L_lo"] == 0.0
        assert all(v["L_lo"] is None or v["L_lo"] >= 0.0
                   for s in drive["sessions"].values() for v in s["per_version"])

    def test_the_consecutive_brackets_the_appendix_quotes(self, drive):
        versions = {v["sourceUpdated"]: v
                    for v in drive["sessions"]["drive_run_20260908_170849"]["per_version"]}
        first = versions["2026-09-08T21:13:36Z"]
        second = versions["2026-09-08T21:14:36Z"]
        # The appendix took the previous query as 16.7 s earlier; the logged receipts
        # put the lower ends at 50.53 and 23.84 s, within 0.3 s of its 50.4 and 23.6.
        assert first["L_lo"] == pytest.approx(50.4, abs=0.3)
        assert round(first["L_up"], 1) == 67.1
        assert second["L_lo"] == pytest.approx(23.6, abs=0.3)
        assert round(second["L_up"], 1) == 40.3
        # The two brackets do not overlap: the lag moved by more than 10 s.
        assert first["L_lo"] - second["L_up"] > 10.0

    def test_the_artefact_is_a_placeholder_that_replays(self, drive, tmp_path):
        path = tmp_path / "feed_process.json"
        artefact = afp.write_artefact(path, drive=drive, polling=None)
        assert artefact["placeholder"] is True
        assert artefact["primary"]["default_session"] == "drive_run_20260908_170849"
        session = "drive_run_20260908_170849"
        sequence = artefact["sessions"][session]["sequence"]
        # The placeholder session's 29 gaps span 1,526 s, shorter than an episode:
        # it replays only with wrapping allowed, and keeps every stamp's second.
        with pytest.raises(ValueError, match="too short"):
            FeedProcess.from_artefact(path, session, 16, 60.0, 2600.0)
        feed = FeedProcess.from_artefact(path, session, 16, 60.0, 2600.0, allow_wrap=True)
        assert feed.meta["placeholder"] is True and feed.meta["wraps"] >= 1
        start = feed.meta["start_index"]
        assert [t % 60 for t in feed.publish_s] == \
            [sequence[(start + k) % len(sequence)][2] for k in range(feed.count)]
        # A 1,240 s horizon fits inside the session and does not wrap.
        short = FeedProcess.from_artefact(path, session, 16, 60.0, 1300.0)
        assert short.meta["wraps"] == 0


class TestR114:
    """Amendment 1, A10: mutants A3, A6 (send versus receipt) and A4 (window gaps)."""

    def test_a_round_trip_puts_l_up_at_the_receipt_and_l_lo_at_the_previous_send(self):
        pubs = _publications([60, 60])
        queries = np.arange(BASE + 40.0, BASE + 260.0, 5.0)
        records = _records(pubs, 45.0, queries, rtt=0.4)
        out = afp.analyze_records(records)["nj"]
        for version in out["per_version"][1:]:
            T = version["T"]
            first = min(q for q in queries if q >= T + 45.0)
            assert version["L_up"] == pytest.approx(first + 0.4 - T, abs=1e-6)
            assert version["L_lo"] == pytest.approx(first - 5.0 - T, abs=1e-6)
        assert out["L_bar"] == pytest.approx(max(v["L_up"] for v in out["per_version"]))

    def test_the_smallest_availability_window_includes_the_lag_change(self):
        stamps = [BASE + 36.0, BASE + 62.0]  # a 26 s gap
        records = []
        for seq, (q, stamp) in enumerate([(stamps[0] + 40.0, stamps[0]),
                                          (stamps[1] + 30.0, stamps[1])]):
            records.append({"type": "request", "seq": seq, "area": "nj", "t_send_wall": q,
                            "t_recv_wall": q, "status": 200, "sourceUpdated": _stamp(stamp),
                            "rtt_s": 0.0, "Age": "0"})
        out = afp.analyze_records(records)["nj"]
        assert [v["L_up"] for v in out["per_version"]] == pytest.approx([40.0, 30.0])
        # 26 s gap + 30 s lag - 40 s lag: version 1 is returned for 16 s only.
        assert out["min_availability_gap"] == pytest.approx(16.0)
