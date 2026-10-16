"""The drive case table (plan section 3, step S9).

Unit tests use synthetic responses, GPS tracks and detection records. The real smoke at
the bottom scans the HERE bodies recorded on the two drives; the drive data is only read
and no HERE request is made.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import drive_lead_cases as dlc  # noqa: E402

LAT, LON = 40.35, -74.62
M_PER_DEG = 111320.0


def _point(east_m: float):
    return {"lat": LAT, "lng": LON + east_m / (M_PER_DEG * math.cos(math.radians(LAT)))}


def _response(t, version, items):
    return dlc.Response(received_t=t, source_updated=version,
                        items=[dlc.FlowItem(desc, 400.0, [_point(e), _point(e + 50.0)], jf)
                               for desc, e, jf in items])


TRACK = [dlc.GpsFix(t=float(t), lat=LAT, lon=LON, speed_mps=15.0) for t in range(0, 400)]


class TestEvents:
    def test_a_jump_from_3_8_to_8_1_within_60_m_is_one_event(self):
        responses = [_response(100.0, "v1", [("Main St", 40.0, 3.8)]),
                     _response(160.0, "v2", [("Main St", 40.0, 8.1)])]
        events = dlc.find_events(responses, TRACK)
        assert len(events) == 1
        e = events[0]
        assert (e.description, e.jam_before, e.jam_after) == ("Main St", 3.8, 8.1)
        assert e.version == "v2" and e.received_t == 160.0
        assert e.distance_m == pytest.approx(40.0, abs=1.0)

    def test_the_same_jump_70_m_away_is_none(self):
        responses = [_response(100.0, "v1", [("Main St", 70.0, 3.8)]),
                     _response(160.0, "v2", [("Main St", 70.0, 8.1)])]
        assert dlc.find_events(responses, TRACK) == []

    def test_a_rise_below_two_or_ending_below_four_is_none(self):
        small = [_response(100.0, "v1", [("A", 10.0, 4.0)]),
                 _response(160.0, "v2", [("A", 10.0, 5.9)])]
        low = [_response(100.0, "v1", [("A", 10.0, 1.0)]),
               _response(160.0, "v2", [("A", 10.0, 3.9)])]
        assert dlc.find_events(small, TRACK) == [] and dlc.find_events(low, TRACK) == []

    def test_a_repeated_version_is_not_a_new_one(self):
        responses = [_response(100.0, "v1", [("A", 10.0, 3.0)]),
                     _response(110.0, "v1", [("A", 10.0, 3.0)]),
                     _response(160.0, "v2", [("A", 10.0, 5.5)])]
        events = dlc.find_events(responses, TRACK)
        assert len(events) == 1 and events[0].previous_version == "v1"

    def test_the_first_receipt_of_the_version_is_the_feed_report(self):
        responses = [_response(100.0, "v1", [("A", 10.0, 3.0)]),
                     _response(160.0, "v2", [("A", 10.0, 7.0)]),
                     _response(170.0, "v2", [("A", 10.0, 7.0)])]
        event = dlc.find_events(responses, TRACK)[0]
        assert dlc.feed_first_report(responses, event.version) == 160.0


def _det(t, track, gap, lateral=0.0, closing=0.0, valid=True):
    return {"type": "det", "t": t, "track_id": track, "ema_wp_m": gap, "lateral_wp_m": lateral,
            "closing_wp_mps": closing, "closing_wp_valid": valid}


class TestReports:
    def test_the_gps_stop_is_the_first_fix_below_5_m_s_in_the_window(self):
        track = [dlc.GpsFix(float(t), LAT, LON, 15.0 if t < 250 else 3.0) for t in range(400)]
        assert dlc.gps_first_stop(track, 100.0, 400.0) == 250.0
        assert dlc.gps_first_stop(TRACK, 100.0, 400.0) is None

    def test_every_stop_onset_in_the_window_is_listed(self):
        speeds = [15.0 if (t < 150 or 200 <= t < 300) else 2.0 for t in range(400)]
        track = [dlc.GpsFix(float(t), LAT, LON, v) for t, v in enumerate(speeds)]
        assert dlc.gps_stop_onsets(track, 100.0, 400.0) == [150.0, 300.0]

    def test_camera_density_criterion(self):
        # A queue builds in the own lane and both adjacent lanes from 200 s.
        records = []
        for k in range(100):
            t = 150.0 + k
            gap = 30.0 if t < 200 else 3.0
            records.append(_det(t, 1, gap))
            if t >= 200:
                records += [_det(t, 2, 5.0, lateral=-3.7), _det(t, 3, 6.0, lateral=-3.7),
                            _det(t, 4, 5.0, lateral=3.7), _det(t, 5, 6.0, lateral=3.7)]
        track = [dlc.GpsFix(float(t), LAT, LON, 10.0) for t in range(400)]
        report = dlc.camera_first_report(records, track, 100.0, 400.0)
        assert report["t"] == 200.0 and report["criterion"] == "density"
        assert report["k_hat"] >= 0.9 * 0.178

    def test_camera_leader_speed_criterion(self):
        # Ego at 15 m/s; the own-lane leader closes at 0 then at -10 m/s from 260 s, so its
        # estimated speed falls from 15 to 5 m/s, below half its value 30 s earlier.
        records = [_det(float(t), 1, 40.0, closing=0.0 if t < 260 else -10.0)
                   for t in range(200, 320)]
        track = [dlc.GpsFix(float(t), LAT, LON, 15.0) for t in range(400)]
        report = dlc.camera_first_report(records, track, 100.0, 400.0)
        assert report["t"] == 260.0 and report["criterion"] == "leader_speed"

    def test_no_records_in_the_window_give_none(self):
        assert dlc.camera_first_report([], TRACK, 0.0, 10.0) is None


# ------------------------------------------------------------------ real smoke


DRIVES = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")


@pytest.mark.skipif(not DRIVES.exists(), reason="the drive data is not on this machine")
def test_the_two_section_3_events_are_found_on_the_drives(tmp_path):
    out = tmp_path / "drive_lead_cases.json"
    assert dlc.main(["--out", str(out), "--frames-dir", str(tmp_path / "no_frames")]) == 0
    cases = json.loads(out.read_text())
    found = {(c["run"], c["description"], c["jam_before"], c["jam_after"], c["version"])
             for c in cases["events"]}
    assert ("run_20260908_170849", "Scudders Mill Rd", 3.8, 8.1, "2026-09-08T21:18:36Z") in found
    assert any(run == "run_20260908_190548" and "Deans" in desc and before == 3.2
               and after == 8.1 and version == "2026-09-08T23:16:36Z"
               for run, desc, before, after, version in found)
    by_run = {c["run"]: c for c in cases["events"] if c["jam_after"] == 8.1}
    assert by_run["run_20260908_170849"]["received_after_first_tick_s"] == pytest.approx(533, abs=2)
    assert by_run["run_20260908_190548"]["received_after_first_tick_s"] == pytest.approx(719, abs=2)
    # No request URL (which could carry a key) is stored.
    assert "apiKey" not in out.read_text() and "request_url" not in out.read_text()
