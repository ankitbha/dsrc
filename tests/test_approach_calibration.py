"""The approach travel-time calibration and the boundary checks (approach plan sections
2.2 and 6, step S3).

Unit tests use synthetic step sequences and schedules. The sanity tests at the bottom read
the committed calibration and run SUMO on the approach boundary, the last one for both
training seeds' arrival and burst checks (about two minutes); run this file in its own
process.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_mainz_scenario as bms  # noqa: E402
import calibrate_approach_travel as cat  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"


class TestCrossings:
    def test_the_first_step_on_the_vehicle_s_own_entry_link(self):
        steps = [(0.5, {"v1": "a140_2"}),
                 (1.0, {"v1": "140", "v2": "a255_2"}),
                 (1.5, {"v1": "140", "v2": "255", "v3": "182"}),
                 (2.0, {"v3": "182", "v4": "a293_1"})]
        entry_of = {"v1": "140", "v2": "255", "v3": "140", "v4": "293"}
        assert cat.entry_crossings(steps, entry_of) == {"v1": 1.0, "v2": 1.5}

    def test_a_vehicle_leaving_and_returning_keeps_its_first_crossing(self):
        steps = [(3.0, {"v1": "140"}), (3.5, {}), (4.0, {"v1": "140"})]
        assert cat.entry_crossings(steps, {"v1": "140"}) == {"v1": 3.0}


class TestStats:
    def test_on_a_hand_list(self):
        stats = cat.travel_stats([1.0, 2.0, 3.0, 4.0, 10.0])
        assert stats["n"] == 5 and stats["mean_s"] == 4.0 and stats["median_s"] == 3.0
        assert stats["sd_s"] == pytest.approx(12.5 ** 0.5)
        assert stats["p5_s"] == pytest.approx(1.2) and stats["p95_s"] == pytest.approx(8.8)

    def test_an_empty_entry_has_no_statistics(self):
        assert cat.travel_stats([]) == {"n": 0, "mean_s": None, "median_s": None,
                                        "sd_s": None, "p5_s": None, "p95_s": None}


def _rows(entry="140", count=200, start=240.0, headway=7.5, W=240.0):
    return [{"id": f"v{entry}_{k}", "entry": entry, "depart": start + k * headway - 190.0,
             "entry_time": start + k * headway} for k in range(count)]


class TestArrivalCheck:
    def test_arrivals_on_schedule_pass(self):
        rows = _rows()
        crossings = {r["id"]: r["entry_time"] + (1.0 if k % 2 else -1.0)
                     for k, r in enumerate(rows)}
        out = cat.arrival_check(rows, crossings, 240.0, 240.0 + 1800.0)
        assert out["passes"] and out["entries"]["140"]["mean_lag_s"] == pytest.approx(0.0)
        assert out["windows"] == 5 and out["missing"] == []

    def test_a_mean_lag_over_3_s_fails(self):
        rows = _rows()
        crossings = {r["id"]: r["entry_time"] + 3.5 for r in rows}
        out = cat.arrival_check(rows, crossings, 240.0, 240.0 + 1800.0)
        assert not out["passes"] and not out["entries"]["140"]["mean_lag_ok"]

    def test_a_window_short_by_more_than_max_6_and_15_percent_fails(self):
        rows = _rows()
        # 40 scheduled per 300 s window; drop 7 arrivals from the second window (6 is the
        # floor, 15% of 40 is 6), shifting none of the others.
        dropped = {r["id"] for r in rows if 540.0 <= r["entry_time"] < 840.0}
        dropped = set(sorted(dropped)[:7])
        crossings = {r["id"]: r["entry_time"] + (300.0 if r["id"] in dropped else 0.0)
                     for r in rows}
        out = cat.arrival_check(rows, crossings, 240.0, 240.0 + 1800.0)
        failures = out["entries"]["140"]["window_failures"]
        assert not out["passes"] and failures[0]["window"] == [540.0, 840.0]
        assert failures[0]["scheduled"] - failures[0]["arrived"] == 7

    def test_a_due_vehicle_that_never_arrives_fails(self):
        rows = _rows()
        crossings = {r["id"]: r["entry_time"] for r in rows[1:]}
        out = cat.arrival_check(rows, crossings, 240.0, 240.0 + 1800.0)
        assert not out["passes"] and out["missing"] == [rows[0]["id"]]


class TestBurstCheck:
    def test_an_entry_short_by_more_than_10_percent_fails_unless_its_approach_queues(self):
        rows = _rows("140", count=100, start=1140.0, headway=3.0) + _rows(
            "295", count=100, start=1140.0, headway=3.0)
        crossings = {r["id"]: r["entry_time"] for r in rows}
        for r in rows[100:125]:
            crossings[r["id"]] = 2000.0  # 25 of 295's burst vehicles arrive after the window
        queued = cat.burst_check(rows, crossings, {"140": [0.0], "295": [227.0]}, 240.0)
        assert queued["passes"] and not queued["entries"]["295"]["tested"]
        assert queued["entries"]["295"]["difference_share"] == pytest.approx(-0.25)
        free = cat.burst_check(rows, crossings, {"140": [0.0], "295": [0.0]}, 240.0)
        assert not free["passes"]


def test_seeds_other_than_1_and_2_are_refused():
    with pytest.raises(SystemExit):
        cat.main(["--seed", "16"])
    with pytest.raises(SystemExit):
        cat.main(["--check", "--seeds", "1", "3"])


# ----------------------------------------------------------------- committed files


@pytest.fixture(scope="module")
def travel():
    return json.loads((DATA / "mainz_approach_travel.json").read_text())


@pytest.fixture(scope="module")
def meta():
    return json.loads((DATA / "mainz_approach.json").read_text())


def test_t_a_is_the_calibration_s_per_entry_mean_on_this_net(travel, meta):
    assert meta["calibrated"] is True and travel["seed"] == 1
    assert travel["net_sha256"] == meta["net"]["sha256"]
    assert meta["T_A"] == {e: travel["entries"][e]["mean_s"] for e in meta["entries"]}
    assert meta["W"] == bms.approach_offset(max(meta["T_A"].values()))
    assert meta["travel"]["sha256"] == cat._sha256(DATA / "mainz_approach_travel.json")


def test_the_raw_crossings_reproduce_the_means(travel):
    by_entry: dict[str, list[float]] = {}
    for vehicle, entry, depart, crossing in travel["vehicles"]:
        if crossing is not None:
            by_entry.setdefault(entry, []).append(crossing - depart)
    for entry, stats in travel["entries"].items():
        assert stats["mean_s"] == pytest.approx(statistics.fmean(by_entry[entry]), abs=1e-9)
        assert stats["n"] == len(by_entry[entry])
    assert travel["pending_max"] == 0 and travel["late_insertions"] == 0
    assert travel["colliding_vehicles"] == 0 and travel["collision_teleports"] == 0


def test_a_1440_s_run_reproduces_each_entry_s_mean_travel_time(meta):
    W = meta["W"]
    out = cat.run("mainz_bg_A", 1, 1200.0 + W)
    assert out["pending_max"] == 0 and out["colliding_vehicles"] == 0
    for entry in meta["entries"]:
        times = [out["crossings"][r["id"]] - r["depart"] for r in out["rows"]
                 if r["entry"] == entry and r["id"] in out["crossings"]]
        assert len(times) > 20
        assert statistics.fmean(times) == pytest.approx(meta["T_A"][entry], abs=3.0)


def test_the_arrival_and_burst_checks_pass_on_seeds_1_and_2(tmp_path):
    report = cat.check((1, 2), tmp_path / "checks.json")
    assert report["passes"], json.dumps({k: report[k] for k in ("arrivals", "burst")})[:2000]
    for seed in ("1", "2"):
        assert report["arrivals"][seed]["pending_max"] == 0
        assert len(report["burst"][seed]["entries"]) == 8
        assert all(v["mean_lag_s"] is not None for v in report["burst"][seed]["entries"].values())
