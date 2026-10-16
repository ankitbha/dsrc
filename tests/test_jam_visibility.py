"""Which leaders a dashboard camera can see (plan section 1.2, step S3).

Unit tests build synthetic lane snapshots and need no SUMO. The sanity tests at the bottom
run a 600 s SUMO episode and compare the vectorised candidate table with a brute-force
per-vehicle computation; run this file in its own process.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.sumo.sensing import (  # noqa: E402
    FOV_HALF_TAN, VEHICLE_WIDTH_M, CandidateTable, EdgeGeometry, Ego, LaneSnapshot,
    closed_form_occluded, occluded, successors, visible_candidates,
)

W = 3.7  # the plan's worked numbers use 3.7 m lanes
GEOMETRY = {"A": EdgeGeometry(lanes=3, length_m=500.0, widths=(W, W, W)),
            "B": EdgeGeometry(lanes=2, length_m=300.0, widths=(W, W))}
SUCCESSOR = {"A": "B"}
L = 4.5


def _leader(vid, edge, lane, gap_from, x, length=L, speed=10.0):
    """A vehicle whose back is `gap_from` metres ahead of front position `x`."""
    return {"id": vid, "edge": edge, "lane": lane, "front_m": x + gap_from + length,
            "length_m": length, "speed_mps": speed}


def _snapshot(vehicles, ego_lane=1, x=100.0):
    ego = {"id": "ego", "edge": "A", "lane": ego_lane, "front_m": x, "length_m": L,
           "speed_mps": 10.0}
    return LaneSnapshot.from_vehicles([ego, *vehicles], GEOMETRY, SUCCESSOR, t=0.0)


def _visible(snapshot, R=80.0):
    c = visible_candidates(snapshot, Ego("ego", "A", 1, 100.0), R)
    return sorted((round(float(d), 6), int(dl)) for d, dl, v in zip(c.d, c.dlane, c.visible) if v)


class TestScenes:
    def test_own_lane_leaders_at_10_20_30_give_one_visible(self):
        snap = _snapshot([_leader(f"o{g}", "A", 1, g, 100.0) for g in (10, 20, 30)])
        assert _visible(snap) == [(10.0, 0)]

    def test_adjacent_leaders_at_10_12_30_give_10_and_30(self):
        # 12 <= 1.32 x 10, so 12 is hidden; 30 > 1.32 x 12 and it stays visible.
        snap = _snapshot([_leader(f"a{g}", "A", 2, g, 100.0) for g in (10, 12, 30)])
        assert _visible(snap) == [(10.0, 1), (30.0, 1)]

    def test_adjacent_at_50_with_own_leader_at_10_is_occluded(self):
        snap = _snapshot([_leader("own", "A", 1, 10, 100.0), _leader("adj", "A", 0, 50, 100.0)])
        assert _visible(snap) == [(10.0, 0)]

    def test_adjacent_at_40_with_own_leader_at_10_is_visible(self):
        snap = _snapshot([_leader("own", "A", 1, 10, 100.0), _leader("adj", "A", 0, 40, 100.0)])
        assert _visible(snap) == [(10.0, 0), (40.0, -1)]

    def test_a_leader_20_m_past_the_edge_end_appears_through_the_successor(self):
        ego = {"id": "ego", "edge": "A", "lane": 1, "front_m": 490.0, "length_m": L,
               "speed_mps": 10.0}
        far = {"id": "f", "edge": "B", "lane": 1, "front_m": 20.0 + L, "length_m": L,
               "speed_mps": 3.0}
        snap = LaneSnapshot.from_vehicles([ego, far], GEOMETRY, SUCCESSOR, t=0.0)
        c = visible_candidates(snap, Ego("ego", "A", 1, 490.0), 80.0)
        assert list(c.ids) == ["f"]
        assert c.d[0] == pytest.approx(30.0)
        assert c.dlane[0] == 0 and bool(c.visible[0])
        assert c.speed[0] == pytest.approx(3.0)

    def test_the_successor_lane_is_clamped_to_its_lane_count(self):
        # Ego on lane 2 of the three-lane A; B has two lanes, so j = 1 and B's lane 1 is
        # straight ahead (offset 0) and lane 0 is offset -1.
        ego = {"id": "ego", "edge": "A", "lane": 2, "front_m": 490.0, "length_m": L,
               "speed_mps": 10.0}
        ahead = {"id": "b1", "edge": "B", "lane": 1, "front_m": 30.0, "length_m": L,
                 "speed_mps": 3.0}
        left = {"id": "b0", "edge": "B", "lane": 0, "front_m": 60.0, "length_m": L,
                "speed_mps": 3.0}
        snap = LaneSnapshot.from_vehicles([ego, ahead, left], GEOMETRY, SUCCESSOR, t=0.0)
        c = visible_candidates(snap, Ego("ego", "A", 2, 490.0), 80.0)
        assert dict(zip(c.ids, c.dlane.tolist())) == {"b1": 0, "b0": -1}

    def test_the_field_of_view_removes_an_adjacent_vehicle_at_6_m(self):
        snap = _snapshot([_leader("adj", "A", 2, 6, 100.0)])
        c = visible_candidates(snap, Ego("ego", "A", 1, 100.0), 80.0)
        assert list(c.ids) == ["adj"]
        assert not bool(c.in_fov[0]) and not bool(c.visible[0])
        assert W / 6.0 > FOV_HALF_TAN

    def test_the_field_of_view_bound_is_360_over_800(self):
        # At 7 m an adjacent vehicle sits at |y|/d = 0.53: outside 0.45, inside 0.60 (kills
        # a wider field of view, M25); at 9 m (0.41) it is in view.
        for gap, inside in ((7.0, False), (9.0, True)):
            c = visible_candidates(_snapshot([_leader("adj", "A", 2, gap, 100.0)]),
                                   Ego("ego", "A", 1, 100.0), 80.0)
            assert bool(c.in_fov[0]) is inside, gap
        assert FOV_HALF_TAN == pytest.approx(0.45)

    def test_candidates_are_limited_to_the_range_and_to_leaders(self):
        # Ego on lane 0 of three: lanes 0 and 1 are in view, lane 2 is two lanes over.
        def vehicle(vid, lane, front):
            return {"id": vid, "edge": "A", "lane": lane, "front_m": front, "length_m": L,
                    "speed_mps": 10.0}

        snap = LaneSnapshot.from_vehicles(
            [vehicle("ego", 0, 100.0), vehicle("two_over", 2, 120.0),
             _leader("near", "A", 0, 79.0, 100.0), _leader("far", "A", 0, 81.0, 100.0),
             vehicle("behind", 0, 95.0), vehicle("alongside", 1, 102.0)],
            GEOMETRY, SUCCESSOR, t=0.0)
        c = visible_candidates(snap, Ego("ego", "A", 0, 100.0), 80.0)
        assert list(c.ids) == ["near"]

    def test_the_gap_is_to_the_leader_s_back(self):
        snap = _snapshot([_leader("truck", "A", 1, 15.0, 100.0, length=12.0)])
        c = visible_candidates(snap, Ego("ego", "A", 1, 100.0), 80.0)
        assert c.d[0] == pytest.approx(15.0)

    def test_lateral_offset_uses_the_ego_lane_width(self):
        geometry = {"A": EdgeGeometry(lanes=3, length_m=500.0, widths=(3.5, 3.2, 3.75)),
                    "B": GEOMETRY["B"]}
        snap = LaneSnapshot.from_vehicles(
            [{"id": "ego", "edge": "A", "lane": 1, "front_m": 100.0, "length_m": L,
              "speed_mps": 10.0}, _leader("adj", "A", 2, 20.0, 100.0)],
            geometry, SUCCESSOR, t=0.0)
        c = visible_candidates(snap, Ego("ego", "A", 1, 100.0), 80.0)
        assert c.y[0] == pytest.approx(3.2)


class TestOcclusionRules:
    def test_general_computation_on_hand_cases(self):
        # Own lane: the nearer leader covers all of the farther one.
        assert occluded(np.array([10.0, 20.0]), np.array([0.0, 0.0])).tolist() == [False, True]
        # Adjacent lane: hidden exactly when d2 <= y/(y - 0.9) d1.
        ratio = W / (W - 0.9)
        assert occluded(np.array([10.0, 10.0 * ratio - 1e-6]), np.array([W, W])).tolist() == [
            False, True]
        assert occluded(np.array([10.0, 10.0 * ratio + 1e-3]), np.array([W, W])).tolist() == [
            False, False]
        # By the own-lane leader: hidden exactly when d2 >= y/0.9 d0.
        own = W / 0.9
        assert occluded(np.array([10.0, 10.0 * own + 1e-6]), np.array([0.0, W])).tolist() == [
            False, True]
        assert occluded(np.array([10.0, 10.0 * own - 1e-3]), np.array([0.0, W])).tolist() == [
            False, False]
        assert round(ratio, 2) == 1.32 and round(own, 2) == 4.11

    def test_closed_form_rules_against_the_interval_computation_on_1000_scenes(self):
        rng = np.random.default_rng(7)
        disagreements, rule_says_hidden_general_visible = 0, 0
        for _ in range(1000):
            n = int(rng.integers(1, 9))
            d = rng.uniform(1.0, 80.0, n)
            dlane = rng.integers(-1, 2, n)
            general = occluded(d, dlane * W, VEHICLE_WIDTH_M)
            closed = closed_form_occluded(d, dlane, W, VEHICLE_WIDTH_M)
            rule_says_hidden_general_visible += int(np.sum(closed & ~general))
            for k in np.flatnonzero(closed != general):
                disagreements += 1
                # Every disagreement is one hidden only by the own-lane leader and the
                # next nearer vehicle in its own lane together, neither alone covering half.
                assert dlane[k] != 0
                nearer_own = d[(dlane == 0) & (d < d[k])]
                nearer_same = d[(dlane == dlane[k]) & (d < d[k])]
                assert len(nearer_own) and len(nearer_same)
        # The rules never hide a vehicle the general computation shows.
        assert rule_says_hidden_general_visible == 0
        # The combined case is rare; the general computation is the definition.
        assert disagreements < 0.02 * 1000 * 4.5

    def test_the_vectorised_table_equals_the_general_computation_on_1000_scenes(self):
        rng = np.random.default_rng(11)
        for scene in range(1000):
            vehicles = [{"id": "ego", "edge": "A", "lane": 1, "front_m": 100.0,
                         "length_m": L, "speed_mps": 10.0}]
            for k in range(int(rng.integers(1, 12))):
                lane = int(rng.integers(0, 3))
                vehicles.append(_leader(f"v{k}", "A", lane, float(rng.uniform(0.5, 85.0)),
                                        100.0, length=float(rng.choice([4.5, 5.0, 12.0]))))
            snap = LaneSnapshot.from_vehicles(vehicles, GEOMETRY, SUCCESSOR, t=0.0)
            c = visible_candidates(snap, Ego("ego", "A", 1, 100.0), 80.0)
            reference = occluded(c.d, c.y, VEHICLE_WIDTH_M)
            assert c.occluded.tolist() == reference.tolist(), scene


class TestSuccessors:
    def test_each_edge_maps_to_its_one_route_successor(self):
        routes = {"r1": ["a", "b", "c"], "r2": ["x", "b", "c"]}
        assert successors(routes) == {"a": "b", "b": "c", "x": "b"}

    def test_two_successors_are_refused(self):
        with pytest.raises(ValueError):
            successors({"r1": ["a", "b"], "r2": ["a", "c"]})


class TestTable:
    def test_rows_are_ordered_by_ego_id_then_distance(self):
        vehicles = [
            {"id": "z", "edge": "A", "lane": 1, "front_m": 100.0, "length_m": L, "speed_mps": 1.0},
            {"id": "a", "edge": "A", "lane": 0, "front_m": 200.0, "length_m": L, "speed_mps": 1.0},
            _leader("l2", "A", 1, 30.0, 100.0), _leader("l1", "A", 1, 10.0, 100.0),
            _leader("m1", "A", 0, 5.0, 200.0),
        ]
        snap = LaneSnapshot.from_vehicles(vehicles, GEOMETRY, SUCCESSOR, t=0.0)
        table = CandidateTable.build(snap, [Ego("z", "A", 1, 100.0), Ego("a", "A", 0, 200.0)],
                                     80.0)
        assert table.ego_ids == ["a", "z"]
        rows = [(table.ego_ids[e], snap.ids[g]) for e, g in zip(table.ego, table.vehicle)]
        assert rows == [("a", "m1"), ("z", "l1"), ("z", "l2")]
        assert table.counts().tolist() == [1, 2]


# ----------------------------------------------------------------- SUMO sanity


def _brute_force_counts(R: float) -> dict[str, int]:
    """Per controlled vehicle on a segment edge, candidates by the definition, read
    vehicle by vehicle from SUMO."""
    from src.sumo.binding import _sumo

    everyone = {}
    for v in _sumo.vehicle.getIDList():
        road = _sumo.vehicle.getRoadID(v)
        if road.startswith(":"):
            continue
        everyone[v] = (road, _sumo.vehicle.getLaneIndex(v), _sumo.vehicle.getLanePosition(v),
                       _sumo.vehicle.getLength(v))
    return everyone


def test_the_table_matches_a_brute_force_count_and_one_pass_is_fast():
    import train_mainz_src as tms
    from src.analysis.gains import load_routes
    from src.sumo.binding import _sumo
    from src.sumo.mainz import HERE_FEATURES

    paths = tms.SCENARIOS["mainz"]
    env = tms.make_env(paths, seed=1, duration_s=600.0, features=HERE_FEATURES,
                       step_length=0.5, gate_entries=False)
    succ = successors(load_routes(REPO_ROOT / paths["gated_routes"]))
    segment_edges = {e for s in env.segments for e in s}
    R = 80.0
    walls, sizes, compared = [], [], 0
    try:
        env.reset()
        snapshot_builder = LaneSnapshot.builder(env, succ)
        while env.time_s() < 600.0:
            env.release()
            env.advance_ticks(60.0, tick_s=5.0, on_tick=lambda t: None)
            for check in range(1):
                started = time.perf_counter()
                snap = snapshot_builder.build(env.time_s())
                egos = snap.egos(env.controlled, segment_edges)
                table = CandidateTable.build(snap, egos, R)
                walls.append(time.perf_counter() - started)
                sizes.append(len(egos))
            if env.time_s() in (420.0, 540.0, 600.0):
                everyone = _brute_force_counts(R)
                geometry = snap.geometry
                counts = dict(zip(table.ego_ids, table.counts().tolist()))
                for ego in egos:
                    edge, lane, x = ego.edge, ego.lane, ego.x
                    n_lanes = geometry[edge].lanes
                    offsets = {o for o in (-1, 0, 1) if 0 <= lane + o < n_lanes}
                    nxt = succ.get(edge)
                    expected = 0
                    for v, (road, li, pos, length) in everyone.items():
                        if v == ego.vehicle_id:
                            continue
                        if road == edge and (li - lane) in offsets:
                            gap = pos - length - x
                        elif (road == nxt and x + R > geometry[edge].length_m
                              and road in geometry):
                            j = min(lane, geometry[nxt].lanes - 1)
                            if (li - j) not in offsets:
                                continue
                            gap = geometry[edge].length_m - x + pos - length
                        else:
                            continue
                        if 0.0 < gap <= R:
                            expected += 1
                    assert counts.get(ego.vehicle_id, 0) == expected, (ego, env.time_s())
                    compared += 1
    finally:
        env.close()
    assert compared > 300
    # Plan section 1.4 measured 3.0 ms for the counting pass at about 800 vehicles.
    assert float(np.mean(walls)) < 0.010, (np.mean(walls), max(sizes))
    _ = _sumo
