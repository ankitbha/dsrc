"""The approach roads in front of the eight entry links (approach plan section 1, step S1).

Unit tests run `approach_geometry` on a synthetic net and need no netconvert. The sanity
tests at the bottom read the built `data/mainz/mainz_approach.net.xml`; run this file in
its own process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_mainz_scenario as bms  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
ENTRIES = ("277", "289", "231", "255", "140", "182", "293", "295")


class _Node:
    def __init__(self, node_id, x, y):
        self._id, self._xy = node_id, (x, y)

    def getID(self):
        return self._id

    def getCoord(self):
        return self._xy


class _Lane:
    def __init__(self, width):
        self._width = width

    def getWidth(self):
        return self._width


class _Edge:
    def __init__(self, start, shape, widths, speed):
        self._start, self._shape, self._widths, self._speed = start, shape, widths, speed

    def getFromNode(self):
        return self._start

    def getShape(self):
        return self._shape

    def getLaneNumber(self):
        return len(self._widths)

    def getLanes(self):
        return [_Lane(w) for w in self._widths]

    def getSpeed(self):
        return self._speed


class _Net:
    def __init__(self, edges):
        self._edges = edges

    def getEdge(self, edge_id):
        return self._edges[edge_id]


def _synthetic():
    """Entry `e` starts at (0, 0) heading +x with lanes of 3.00 and 3.25 m at 16.67 m/s; a
    second entry `f` starts at (10, 10) heading +y with one 3.5 m lane."""
    return _Net({
        "e": _Edge(_Node("e-begin", 0.0, 0.0), [(0.0, 0.0), (50.0, 0.0), (90.0, 5.0)],
                   [3.0, 3.25], 16.67),
        "f": _Edge(_Node("f-begin", 10.0, 10.0), [(10.0, 10.0), (10.0, 60.0)], [3.5], 13.9),
    })


# A synthetic 12-segment partition: entry e in segment 0, entry f in segment 8.
SEGMENTS = [["e", "x0"]] + [[f"x{i}"] for i in range(1, 8)] + [["f", "x8"]] + [
    [f"x{i}"] for i in range(9, 12)]


class TestGeometry:
    def test_nodes_lie_upstream_along_the_first_shape_segment(self):
        g = bms.approach_geometry(_synthetic(), ("e", "f"), 3000.0, 1000.0, segments=SEGMENTS)
        nodes = {n["id"]: (n["x"], n["y"]) for n in g["nodes"]}
        assert nodes["ae_n0"] == pytest.approx((-3000.0, 0.0))
        assert nodes["ae_n1"] == pytest.approx((-2000.0, 0.0))
        assert nodes["ae_n2"] == pytest.approx((-1000.0, 0.0))
        assert nodes["af_n0"] == pytest.approx((10.0, -2990.0))
        assert all(n["type"] == "priority" for n in g["nodes"])

    def test_edges_copy_the_entry_s_lanes_widths_and_speed_and_end_at_its_start(self):
        g = bms.approach_geometry(_synthetic(), ("e", "f"), 3000.0, 1000.0, segments=SEGMENTS)
        edges = {e["id"]: e for e in g["edges"]}
        assert [e["id"] for e in g["edges"]] == ["ae_0", "ae_1", "ae_2", "af_0", "af_1", "af_2"]
        assert (edges["ae_0"]["from"], edges["ae_0"]["to"]) == ("ae_n0", "ae_n1")
        assert (edges["ae_2"]["from"], edges["ae_2"]["to"]) == ("ae_n2", "e-begin")
        assert edges["ae_1"]["widths"] == [3.0, 3.25] and edges["ae_1"]["lanes"] == 2
        assert edges["ae_1"]["speed"] == 16.67
        assert edges["af_2"]["widths"] == [3.5] and edges["af_2"]["speed"] == 13.9

    def test_connections_join_lane_i_to_lane_i_and_the_last_edge_to_the_entry(self):
        g = bms.approach_geometry(_synthetic(), ("e", "f"), 3000.0, 1000.0, segments=SEGMENTS)
        pairs = {(c["from"], c["to"], c["fromLane"], c["toLane"]) for c in g["connections"]}
        assert pairs == {("ae_0", "ae_1", 0, 0), ("ae_0", "ae_1", 1, 1),
                         ("ae_1", "ae_2", 0, 0), ("ae_1", "ae_2", 1, 1),
                         ("ae_2", "e", 0, 0), ("ae_2", "e", 1, 1),
                         ("af_0", "af_1", 0, 0), ("af_1", "af_2", 0, 0), ("af_2", "f", 0, 0)}

    def test_indices_are_12_plus_3j_plus_i(self):
        g = bms.approach_geometry(_synthetic(), ("e", "f"), 3000.0, 1000.0, segments=SEGMENTS)
        meta = g["metadata"]
        assert meta["approach"]["e"]["segments"] == [12, 13, 14]
        assert meta["approach"]["f"]["segments"] == [15, 16, 17]
        assert meta["approach"]["e"]["entry_segment"] == 0
        assert meta["approach"]["f"]["entry_segment"] == 8
        assert meta["approach_segments"] == [["ae_0"], ["ae_1"], ["ae_2"], ["af_0"], ["af_1"],
                                             ["af_2"]]

    def test_plan_row_and_downstream(self):
        g = bms.approach_geometry(_synthetic(), ("e", "f"), 3000.0, 1000.0, segments=SEGMENTS)
        meta = g["metadata"]
        assert meta["plan_row"] == list(range(12)) + [0, 0, 0, 8, 8, 8]
        down = meta["downstream"]
        assert down[0] == list(range(1, 12)) and down[11] == []
        assert down[12] == [13, 14] + list(range(0, 12))
        assert down[14] == list(range(0, 12))
        assert down[17] == [8, 9, 10, 11]

    def test_the_mainz_examples_of_section_4_1(self):
        # a277_0 (index 12) and a293_2 (index 32), on the committed segments.
        segments = json.loads((DATA / "mainz_segments.json").read_text())
        meta = bms.approach_tables(ENTRIES, {"277": 0, "289": 0, "231": 2, "255": 2, "140": 5,
                                             "182": 5, "293": 8, "295": 8}, 3, len(segments))
        assert meta["downstream"][12] == [13, 14] + list(range(12))
        assert meta["downstream"][32] == [8, 9, 10, 11]
        assert meta["plan_row"][12:] == [0] * 6 + [2] * 6 + [5] * 6 + [8] * 6
        assert len(meta["plan_row"]) == len(meta["downstream"]) == 36

    def test_a_negative_or_fractional_length_is_refused(self):
        with pytest.raises(ValueError):
            bms.approach_geometry(_synthetic(), ("e",), -3000.0, 1000.0, segments=SEGMENTS)
        with pytest.raises(ValueError):
            bms.approach_geometry(_synthetic(), ("e",), 3000.0, -1000.0, segments=SEGMENTS)
        with pytest.raises(ValueError):
            bms.approach_geometry(_synthetic(), ("e",), 2500.0, 1000.0, segments=SEGMENTS)

    def test_an_entry_in_no_segment_is_refused(self):
        with pytest.raises(ValueError):
            bms.approach_geometry(_synthetic(), ("e",), 3000.0, 1000.0,
                                  segments=[["x0"]] * 12)

    def test_the_patch_files_carry_the_geometry(self):
        g = bms.approach_geometry(_synthetic(), ("e",), 3000.0, 1000.0, segments=SEGMENTS)
        nod, edg, con = bms.approach_patch_xml(g)
        assert 'id="ae_n0"' in nod and 'type="priority"' in nod
        assert '<lane index="1" width="3.25"/>' in edg and 'numLanes="2"' in edg
        assert 'from="ae_2" to="e" fromLane="1" toLane="1"' in con


class TestNetComparison:
    def _net(self, tmp_path, name, shape="0.00,0.00 10.00,0.00", junction_type="dead_end"):
        path = tmp_path / name
        path.write_text(
            '<net>\n'
            f'<edge id="a" from="n0" to="n1"><lane id="a_0" index="0" speed="16.67" '
            f'length="10.00" width="3.75" shape="{shape}"/></edge>\n'
            '<junction id="n0" type="' + junction_type + '" x="0.00" y="0.00"/>\n'
            '<junction id="n1" type="dead_end" x="10.00" y="0.00"/>\n'
            '<connection from="a" to="b" fromLane="0" toLane="0" dir="s" state="M"/>\n'
            '</net>\n')
        return path

    def test_a_signed_zero_is_not_a_change(self, tmp_path):
        a = self._net(tmp_path, "a.xml")
        b = self._net(tmp_path, "b.xml", shape="-0.00,0.00 10.00,0.00")
        assert bms.net_changes(a, b) == {"edges": [], "connections": [], "junctions": [],
                                         "missing": []}

    def test_a_moved_shape_and_a_changed_junction_are_reported(self, tmp_path):
        a = self._net(tmp_path, "a.xml")
        b = self._net(tmp_path, "b.xml", shape="0.00,0.10 10.00,0.00", junction_type="priority")
        changes = bms.net_changes(a, b)
        assert changes["edges"] == ["a"] and changes["junctions"] == ["n0"]


# ----------------------------------------------------------------- built net


BUILT = DATA / "mainz_approach.net.xml"
META = DATA / "mainz_approach.json"


@pytest.fixture(scope="module")
def built():
    if not BUILT.exists():
        pytest.fail(f"{BUILT} has not been built (build_mainz_scenario.py --approach)")
    return json.loads(META.read_text())


def test_the_original_network_is_unchanged(built):
    changes = bms.net_changes(DATA / "mainz.net.xml", BUILT)
    assert changes["edges"] == [] and changes["connections"] == [] and changes["missing"] == []
    assert sorted(changes["junctions"]) == sorted(f"{e}-begin" for e in ENTRIES)


def test_the_original_files_keep_their_hashes(built):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(DATA / "mainz.net.xml").startswith("037de791")
    assert sha(DATA / "mainz_segments.json").startswith("a6f82397")
    assert built["source_net"]["sha256"] == sha(DATA / "mainz.net.xml")
    assert built["net"]["sha256"] == sha(BUILT)


def test_every_approach_edge_is_1000_m_with_the_entry_s_lanes(built):
    import sumolib

    net = sumolib.net.readNet(str(BUILT))
    for entry in ENTRIES:
        source = net.getEdge(entry)
        for edge_id in built["approach"][entry]["edges"]:
            edge = net.getEdge(edge_id)
            assert edge.getLaneNumber() == source.getLaneNumber()
            for lane, original in zip(edge.getLanes(), source.getLanes()):
                assert lane.getLength() == pytest.approx(1000.0, abs=0.05)
                assert lane.getWidth() == original.getWidth()
                assert lane.getSpeed() == original.getSpeed()
        assert built["approach"][entry]["edge_lengths_m"] == pytest.approx([1000.0] * 3,
                                                                           abs=0.05)


def test_every_route_is_connected_through_its_approach(built):
    from src.analysis.gains import load_routes

    links = bms.outgoing(BUILT)
    for route_id, edges in load_routes(DATA / "mainz_routes.rou.xml").items():
        entry = edges[0]
        path = built["approach"][entry]["edges"] + edges
        assert bms.check_connected(path, links) == [], route_id


def test_the_approach_segments_file_lists_24_one_edge_segments_in_index_order(built):
    segments = json.loads((DATA / "mainz_approach_segments.json").read_text())
    assert len(segments) == 24 and all(len(s) == 1 for s in segments)
    expected = [[f"a{e}_{i}"] for e in ENTRIES for i in range(3)]
    assert segments == expected
    assert built["plan_row"][12 + 3 * 7 + 2] == 8
    assert built["downstream"][32] == [8, 9, 10, 11]


def test_every_route_visits_the_original_segments_in_increasing_contiguous_order():
    # Section 4.1's premise for `downstream[s] = [s + 1, ..., 11]`.
    from src.analysis.gains import load_routes

    segments = json.loads((DATA / "mainz_segments.json").read_text())
    segment_of = {e: i for i, s in enumerate(segments) for e in s}
    for route_id, edges in load_routes(DATA / "mainz_routes.rou.xml").items():
        visited = []
        for edge in edges:
            if edge in segment_of and (not visited or visited[-1] != segment_of[edge]):
                visited.append(segment_of[edge])
        assert visited == list(range(visited[0], 12)), (route_id, visited)


def test_a_5000_m_rebuild_gives_five_edges_per_approach_and_w_300(tmp_path):
    """Block Q's fallback (approach plan section 7.1), built into a copy of the data."""
    import shutil

    names = ["mainz.net.xml", "mainz_segments.json"] + [
        f"{name}{suffix}" for name in bms.APPROACH_DEMANDS
        for suffix in (".rou.xml", "_routes.rou.xml", "_schedule.json")]
    for name in names:
        shutil.copy(DATA / name, tmp_path / name)
    assert bms.build_approach(tmp_path, 5000.0, 1000.0) == 0
    meta = json.loads((tmp_path / "mainz_approach.json").read_text())
    assert meta["edges_per_approach"] == 5 and len(meta["approach_segments"]) == 40
    assert meta["approach"]["295"]["segments"] == list(range(12 + 35, 12 + 40))
    assert meta["T_A"] == {e: 300.0 for e in ENTRIES} and meta["W"] == 300.0
    assert meta["calibrated"] is False
    assert len(meta["plan_row"]) == len(meta["downstream"]) == 52
    lengths = [v for entry in meta["approach"].values() for v in entry["edge_lengths_m"]]
    assert min(lengths) >= 999.95 and max(lengths) <= 1000.05
    for name in names:  # the copies of the committed files are left as they were
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()
