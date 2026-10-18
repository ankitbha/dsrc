"""The approach demand (approach plan section 2, step S2).

Unit tests use hand-made rows and the committed demand files and need no SUMO. The
sanity tests at the bottom load every `_A` file in SUMO and run an 840 s BG episode; run
this file in its own process.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_mainz_scenario as bms  # noqa: E402
from src.sumo.mainz import controlled_index  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
NAMES = ("mainz_bg", "mainz_b7500", "mainz_b9000", "mainz_burst", "mainz")
ROW_COUNTS = {"mainz_bg": 7144, "mainz_b7500": 7468, "mainz_b9000": 7594,
              "mainz_burst": 12144, "mainz": 3124}
ENTRIES = ("277", "289", "231", "255", "140", "182", "293", "295")

ROWS = [{"id": "v140_0", "route": "r140", "type": "av", "depart": 3.02, "entry": "140"},
        {"id": "v255_1", "route": "r255", "type": "human", "depart": 4.5, "entry": "255"},
        {"id": "v140_2", "route": "r140", "type": "av", "depart": 9.0, "entry": "140"}]
T_A = {"140": 190.32, "255": 194.67}


class TestOffset:
    def test_w_is_the_next_multiple_of_60_s(self):
        assert bms.approach_offset(194.67) == 240.0
        assert bms.approach_offset(240.5) == 300.0
        assert bms.approach_offset(240.0) == 240.0
        assert bms.approach_offset(180.0) == 180.0
        assert bms.approach_offset(0.5) == 60.0


class TestRows:
    def test_far_end_departure_is_depart_plus_w_minus_t_a(self):
        rows = bms.approach_rows(ROWS, T_A, 240.0)
        assert [r["depart"] for r in rows] == [round(3.02 + 240.0 - 190.32, 2),
                                               round(4.5 + 240.0 - 194.67, 2),
                                               round(9.0 + 240.0 - 190.32, 2)]
        assert [r["entry_time"] for r in rows] == [243.02, 244.5, 249.0]

    def test_order_ids_types_routes_and_entries_are_kept(self):
        rows = bms.approach_rows(ROWS, T_A, 240.0)
        for new, old in zip(rows, ROWS):
            assert {k: new[k] for k in ("id", "route", "type", "entry")} == {
                k: old[k] for k in ("id", "route", "type", "entry")}

    def test_a_negative_far_end_departure_is_refused(self):
        with pytest.raises(ValueError):
            bms.approach_rows(ROWS, T_A, 180.0)

    def test_the_route_file_is_sorted_by_far_end_departure(self, tmp_path):
        # Entry 255's longer travel time puts its vehicle first at the far end.
        (tmp_path / "x_schedule.json").write_text(json.dumps(ROWS))
        (tmp_path / "x_routes.rou.xml").write_text(
            '<routes>\n  <vType id="human" maxSpeed="16.67"/>\n'
            '  <route id="r140" edges="140 9"/>\n  <route id="r255" edges="255 9"/>\n'
            "</routes>\n")
        meta = {"approach": {e: {"edges": [f"a{e}_0", f"a{e}_1", f"a{e}_2"]}
                             for e in ("140", "255")}}
        count = bms.write_approach_demand("x", T_A, 240.0, data_dir=tmp_path, metadata=meta)
        assert count == 3
        text = (tmp_path / "x_A.rou.xml").read_text()
        ids = re.findall(r'<vehicle id="([^"]+)"', text)
        departs = [float(v) for v in re.findall(r'depart="([0-9.]+)"', text)]
        assert ids == ["v255_1", "v140_0", "v140_2"] and departs == sorted(departs)
        assert 'departSpeed="max" departLane="best"' in text
        routes = (tmp_path / "x_A_routes.rou.xml").read_text()
        assert '<route id="r140" edges="a140_0 a140_1 a140_2 140 9"/>' in routes
        assert '<vType id="human" maxSpeed="16.67"/>' in routes
        assert "<vehicle" not in routes
        schedule = json.loads((tmp_path / "x_A_schedule.json").read_text())
        assert [r["id"] for r in schedule] == [r["id"] for r in ROWS]


# ------------------------------------------------------------ committed files


@pytest.fixture(scope="module")
def meta():
    return json.loads((DATA / "mainz_approach.json").read_text())


@pytest.mark.parametrize("name", NAMES)
def test_row_counts_ids_and_far_end_departures(meta, name):
    gated = json.loads((DATA / f"{name}_schedule.json").read_text())
    approach = json.loads((DATA / f"{name}_A_schedule.json").read_text())
    assert len(gated) == len(approach) == ROW_COUNTS[name]
    W = meta["W"]
    assert W % 60 == 0 and W == bms.approach_offset(max(meta["T_A"].values()))
    for old, new in zip(gated, approach):
        assert new["id"] == old["id"] and new["type"] == old["type"]
        assert new["route"] == old["route"] and new["entry"] == old["entry"]
        assert new["depart"] == round(old["depart"] + W - meta["T_A"][old["entry"]], 2)
        assert new["entry_time"] == round(old["depart"] + W, 2)
        assert new["depart"] >= 0.0


@pytest.mark.parametrize("name", NAMES)
def test_the_controlled_set_is_the_gated_one(name):
    gated = json.loads((DATA / f"{name}_schedule.json").read_text())
    approach = json.loads((DATA / f"{name}_A_schedule.json").read_text())
    for p in (0.5, 0.75, 1.0):
        a = {r["id"] for i, r in enumerate(gated) if controlled_index(i, p)}
        b = {r["id"] for i, r in enumerate(approach) if controlled_index(i, p)}
        assert a == b


@pytest.mark.parametrize("name", NAMES)
def test_the_route_file_holds_the_schedule_s_vehicles_in_departure_order(name):
    text = (DATA / f"{name}_A.rou.xml").read_text()
    vehicles = re.findall(r'<vehicle id="([^"]+)" type="([^"]+)" route="([^"]+)" '
                          r'depart="([0-9.]+)"', text)
    schedule = {r["id"]: r for r in json.loads((DATA / f"{name}_A_schedule.json").read_text())}
    assert len(vehicles) == len(schedule)
    departs = [float(v[3]) for v in vehicles]
    assert departs == sorted(departs)
    for vid, kind, route, depart in vehicles:
        row = schedule[vid]
        assert (kind, route, float(depart)) == (row["type"], row["route"], row["depart"])


@pytest.mark.parametrize("name", NAMES)
def test_every_route_starts_with_its_three_approach_edges(meta, name):
    from src.analysis.gains import load_routes

    gated = load_routes(DATA / f"{name}_routes.rou.xml")
    approach = load_routes(DATA / f"{name}_A_routes.rou.xml")
    assert set(gated) == set(approach)
    for route_id, edges in gated.items():
        assert approach[route_id] == meta["approach"][edges[0]]["edges"] + edges
    vtypes = lambda p: [line for line in p.read_text().splitlines() if "<vType" in line]
    assert vtypes(DATA / f"{name}_A_routes.rou.xml") == vtypes(DATA / f"{name}_routes.rou.xml")


def test_every_approach_file_set_is_registered():
    from train_mainz_src import SCENARIOS

    for name in NAMES:
        paths = SCENARIOS[f"{name}_A"]
        assert paths["net"] == "data/mainz/mainz_approach.net.xml"
        assert paths["routes"] == f"data/mainz/{name}_A.rou.xml"
        assert paths["gated_routes"] == f"data/mainz/{name}_A_routes.rou.xml"
        assert paths["schedule"] == f"data/mainz/{name}_A_schedule.json"
        assert paths["segments"] == "data/mainz/mainz_segments.json"
        assert paths["approach"] == "data/mainz/mainz_approach.json"
        for relative in paths.values():
            assert (REPO_ROOT / relative).exists() or relative.endswith("_travel.json")


def test_the_metadata_names_every_demand_file_s_hash(meta):
    import hashlib

    for name in NAMES:
        for suffix in ("_A.rou.xml", "_A_routes.rou.xml", "_A_schedule.json"):
            path = DATA / f"{name}{suffix}"
            assert meta["demand"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------- SUMO sanity


@pytest.mark.parametrize("name", NAMES)
def test_each_approach_route_file_loads_in_sumo(name):
    from sumolib import checkBinary

    from src.sumo.binding import _sumo

    _sumo.start([checkBinary("sumo"), "-n", str(DATA / "mainz_approach.net.xml"),
                 "-r", str(DATA / f"{name}_A.rou.xml"), "--step-length", "0.5",
                 "--no-step-log", "true", "--no-warnings", "true",
                 "--max-depart-delay", "-1", "--time-to-teleport", "-1"])
    try:
        for _ in range(240):
            _sumo.simulationStep()
        assert _sumo.vehicle.getIDCount() > 0
    finally:
        _sumo.close()


def test_an_840_s_bg_episode_inserts_every_vehicle_at_a_far_end_with_nothing_pending():
    from src.sumo.binding import _sumo
    from src.sumo.mainz import MainzEnv

    env = MainzEnv(seed=1, duration_s=840.0, warmup_s=0.0, step_length_s=0.5,
                   gate_entries=False, net_file=DATA / "mainz_approach.net.xml",
                   route_file=DATA / "mainz_bg_A.rou.xml",
                   schedule_file=DATA / "mainz_bg_A_schedule.json")
    far_ends = {f"a{e}_0" for e in ENTRIES}
    departed, pending_max = 0, 0
    try:
        env.reset()
        while env.time_s() < 840.0:
            env._step()
            for vehicle in _sumo.simulation.getDepartedIDList():
                assert _sumo.vehicle.getRoadID(vehicle) in far_ends, vehicle
                assert _sumo.vehicle.getLanePosition(vehicle) < 20.0
                departed += 1
            pending_max = max(pending_max, env.pending())
    finally:
        env.close()
    schedule = json.loads((DATA / "mainz_bg_A_schedule.json").read_text())
    due = sum(1 for r in schedule if r["depart"] < 840.0 - 0.5)
    assert pending_max == 0
    assert departed >= due
