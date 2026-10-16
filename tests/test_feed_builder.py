"""The scenario builder's demand-only path, its named outputs and its burst demand.

Unit tests need no SUMO. The sanity tests at the bottom start SUMO on the two drain
scenarios the feed program adds (`mainz_surge`, `mainz_burst`), so this file runs in
its own pytest process.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_mainz_scenario as bms  # noqa: E402
import train_mainz_src as tms  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A copy of the inputs `--demand-only` reads, so nothing under data/ is written."""
    for name in ("Mainz20base.inpx", "rl_links_mainz.txt", "mainz.net.xml",
                 "mainz_segments.json"):
        shutil.copy(DATA / name, tmp_path / name)
    return tmp_path


class TestBurst:
    def test_two_entries_get_three_intervals_at_the_stated_rates(self):
        volumes = {"1": [(0.0, 1000.0, 100.0)], "2": [(0.0, 1000.0, 50.0)]}
        opening = {"1": 400.0, "2": 200.0}

        shaped = bms.burst(volumes, 300.0, 500.0, 0.5, opening)

        assert shaped == {
            "1": [(0.0, 300.0, 100.0), (300.0, 500.0, 200.0), (500.0, 1000.0, 100.0)],
            "2": [(0.0, 300.0, 50.0), (300.0, 500.0, 100.0), (500.0, 1000.0, 50.0)],
        }

    def test_without_an_opening_map_the_first_interval_is_the_opening_volume(self):
        shaped = bms.burst({"7": [(0.0, 100.0, 60.0)]}, 10.0, 20.0, 2.0)
        assert shaped == {"7": [(0.0, 10.0, 60.0), (10.0, 20.0, 120.0),
                                (20.0, 100.0, 60.0)]}

    def test_a_burst_past_the_demand_end_still_runs_its_own_interval(self):
        # Demand stops at 100 s and the burst is [80, 150): the background piece after
        # the burst is empty and is dropped rather than written as a negative interval.
        shaped = bms.burst({"1": [(0.0, 100.0, 10.0)]}, 80.0, 150.0, 1.0)
        assert shaped == {"1": [(0.0, 80.0, 10.0), (80.0, 150.0, 10.0)]}

    def test_departure_count_of_a_burst_on_a_background(self, tmp_path: Path):
        # 1,000 s at 3,600 veh/h with a 100 s burst at 7,200 veh/h: 900 + 200 = 1,100.
        volumes = bms.burst({"1": [(0.0, 1000.0, 3600.0)]}, 400.0, 500.0, 2.0)
        count = bms.write_routes(tmp_path / "x.rou.xml", {"1": ["1", "2"]}, volumes,
                                 1000.0, 1.0, 1, name="x")
        assert count == 1100
        schedule = json.loads((tmp_path / "x_schedule.json").read_text())
        in_burst = [r for r in schedule if 400.0 <= r["depart"] < 500.0]
        assert len(in_burst) == 200


class TestNamedOutputs:
    def test_out_name_writes_the_three_named_files_and_no_mainz_routes(self, data_dir):
        before = {p.name for p in data_dir.iterdir()}
        assert bms.main(["--demand-only", "--out-name", "probe", "--duration-s", "300",
                         "--demand-duration-s", "300", "--demand-scale", "0.25"],
                        data_dir=data_dir) == 0
        written = {p.name for p in data_dir.iterdir()} - before
        assert written == {"probe.rou.xml", "probe_routes.rou.xml", "probe_schedule.json"}
        assert not (data_dir / "mainz.rou.xml").exists()

    def test_demand_only_leaves_the_network_byte_identical(self, data_dir):
        before = _sha(data_dir / "mainz.net.xml")
        bms.main(["--demand-only", "--out-name", "probe", "--duration-s", "300"],
                 data_dir=data_dir)
        assert _sha(data_dir / "mainz.net.xml") == before
        assert before == _sha(DATA / "mainz.net.xml")

    def test_demand_only_reproduces_the_committed_sustained_scenario(self, data_dir):
        # The committed files were built with netconvert in the loop; the demand-only
        # path must give the same three files from the committed net, or the drain
        # scenarios are not the same network and demand model as the published one.
        bms.main(["--demand-only", "--out-name", "again", "--duration-s", "2500",
                  "--demand-duration-s", "2500", "--demand-scale", "0.25"],
                 data_dir=data_dir)
        assert _sha(data_dir / "again.rou.xml") == _sha(DATA / "mainz.rou.xml")
        assert _sha(data_dir / "again_routes.rou.xml") == _sha(DATA / "mainz_routes.rou.xml")
        assert _sha(data_dir / "again_schedule.json") == _sha(DATA / "mainz_schedule.json")

    def test_burst_and_rush_together_are_refused(self, data_dir):
        with pytest.raises(SystemExit):
            bms.main(["--demand-only", "--out-name", "probe", "--demand-rush-s", "100",
                      "--demand-burst", "10", "20", "0.5"], data_dir=data_dir)


class TestTheDrainScenarios:
    """The two files the drain block reads, built by the commands in plan section 3.4."""

    def test_surge_is_the_published_demand_for_1200_s(self):
        schedule = json.loads((DATA / "mainz_surge_schedule.json").read_text())
        assert len(schedule) == 6000
        assert max(r["depart"] for r in schedule) < 1200.0

    def test_burst_rates_are_3572_then_5000_then_3572(self):
        # The committed file was rebuilt at a 3,572 veh/h background (feed-agreement plan
        # Amendment 2, B4: --demand-scale 0.198444); it carries 892, 834 and 10,418
        # departures on the three intervals.
        schedule = json.loads((DATA / "mainz_burst_schedule.json").read_text())
        departs = [r["depart"] for r in schedule]
        before = sum(1 for d in departs if d < 900.0)
        during = sum(1 for d in departs if 900.0 <= d < 1500.0)
        after = sum(1 for d in departs if 1500.0 <= d < 12000.0)
        # Each entry's headway is constant inside an interval, so counts are the rate
        # times the length to within one vehicle per entry (eight entries).
        assert abs(before - 3572.0 * 900.0 / 3600.0) <= 8
        assert abs(during - 5000.0 * 600.0 / 3600.0) <= 8
        assert abs(after - 3572.0 * 10500.0 / 3600.0) <= 8

    def test_scenarios_are_registered_beside_the_shared_network(self):
        for name in ("mainz_surge", "mainz_burst"):
            paths = tms.SCENARIOS[name]
            assert paths["net"] == "data/mainz/mainz.net.xml"
            assert paths["segments"] == "data/mainz/mainz_segments.json"
            for key in ("routes", "gated_routes", "schedule"):
                assert (REPO_ROOT / paths[key]).exists(), (name, key)
                assert Path(paths[key]).name.startswith(name)

    def test_the_committed_sustained_scenario_is_untouched(self):
        assert tms.SCENARIOS["mainz"]["routes"] == "data/mainz/mainz.rou.xml"


@pytest.mark.parametrize("scenario", ["mainz_surge", "mainz_burst"])
def test_a_400_s_no_control_episode_starts_and_inserts(scenario):
    from src.sumo.mainz import HERE_FEATURES

    env = tms.make_env(tms.SCENARIOS[scenario], seed=16, duration_s=400.0,
                       features=HERE_FEATURES, step_length=0.5, gate_entries=False)
    try:
        env.reset()
        env.release()
        env._advance(100.0)
        metrics = env.metrics()
    finally:
        env.close()
    assert metrics["time_s"] == pytest.approx(400.0)
    assert metrics["running"] > 0
    assert metrics["arrived"] > 0
