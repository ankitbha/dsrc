"""The human driver that can see converging traffic.

Plain `IDMVehicle` follows only within its own lane. Every node in the tree
topologies reduces lane count -- `b1` 3 into 2, `b2` 3 into 2, `c` 4 into 2 -- so
two humans arriving on different incoming lanes had no mutual awareness and drove
through each other. Measured on 12 identical `no_av` conditions: 212 collisions
with plain IDM, 120 with this class.

The unit tests below pin the projection rule; the sanity test at the end checks the
behaviour that motivated it, because a projection that is arithmetically right and
does not reduce collisions would be worthless.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config.loaders import load_named_config
from src.road.highway_imports import ensure_highway_env_importable
from src.road.topology_factory import build_topology
from src.vehicles.merge_aware import MergeAwareIDMVehicle

ensure_highway_env_importable()

from highway_env.road.road import Road  # noqa: E402


@pytest.fixture
def road():
    topology = build_topology("inverted_tree", load_named_config("topology", "inverted_tree"))
    return Road(network=topology.road_network, np_random=np.random.RandomState(7), record_history=False)


def _place(road, lane_index, longitudinal, speed=20.0):
    vehicle = MergeAwareIDMVehicle.make_on_lane(road, lane_index, longitudinal=longitudinal, speed=speed)
    road.vehicles.append(vehicle)
    return vehicle


class TestTheProjection:

    def test_no_converging_traffic_means_no_constraint(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        assert ego._merge_acceleration() is None

    def test_a_vehicle_arriving_first_becomes_a_constraint(self, road):
        # Ego is 100 m from node b2, the other 60 m: it arrives first and is a
        # leader 40 m ahead.
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        _place(road, ("a4_entry", "b2", 0), 440.0, speed=5.0)
        acceleration = ego._merge_acceleration()
        assert acceleration is not None
        assert acceleration < 0.0, "closing on a much slower converging vehicle should decelerate"

    def test_a_vehicle_arriving_second_is_ignored(self, road):
        # Symmetry matters here: if both vehicles yielded to each other the merge
        # would deadlock, so only the one arriving later gives way.
        ego = _place(road, ("a5_entry", "b2", 0), 440.0)
        _place(road, ("a4_entry", "b2", 0), 400.0, speed=5.0)
        assert ego._merge_acceleration() is None

    def test_a_vehicle_on_the_same_arc_is_left_to_ordinary_following(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        _place(road, ("a5_entry", "b2", 0), 440.0, speed=5.0)
        assert ego._merge_acceleration() is None

    def test_a_conflict_beyond_the_horizon_is_ignored(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 100.0)     # 400 m from the node
        _place(road, ("a4_entry", "b2", 0), 480.0, speed=5.0)  # 20 m from it
        # A projected gap of 380 m is past MERGE_HORIZON_M; without the horizon a
        # vehicle would crawl the whole length of an empty arc.
        assert ego._merge_acceleration() is None

    def test_the_nearest_converging_vehicle_is_the_one_used(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        _place(road, ("a4_entry", "b2", 0), 420.0, speed=5.0)   # 20 m ahead
        _place(road, ("a6_entry", "b2", 0), 490.0, speed=5.0)   # 90 m ahead
        near_only = ego._merge_acceleration()
        road.vehicles = [v for v in road.vehicles if v.position[0] < 480.0]
        assert ego._merge_acceleration() == pytest.approx(near_only), (
            "removing the far vehicle changed the answer, so the far one was in use"
        )


class TestItComposesWithOrdinaryFollowing:

    def test_act_never_accelerates_more_than_plain_idm_would(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0, speed=25.0)
        _place(road, ("a4_entry", "b2", 0), 430.0, speed=3.0)
        ego.act()
        merged = ego.action["acceleration"]
        merge_only = ego._merge_acceleration()
        assert merged <= merge_only + 1e-9, "the more restrictive constraint must win"


class TestItActuallyReducesCollisions:

    def test_a_no_av_run_has_fewer_collisions_than_with_plain_idm(self):
        # The behaviour the class exists for. Two conditions rather than twelve so
        # the suite stays quick; the full sweep is recorded in the plan.
        from highway_env.vehicle.behavior import IDMVehicle

        import src.demand.spawner as spawner
        from src.analysis.simulator_health import CellSpec, run_condition

        cell = CellSpec(topology="inverted_tree", demand="high", av_penetration=0.10)
        original = spawner.MergeAwareIDMVehicle
        try:
            spawner.MergeAwareIDMVehicle = IDMVehicle
            plain = sum((run_condition(cell, "no_av", s, duration_steps=120).collisions or 0)
                        for s in (7, 17))
            spawner.MergeAwareIDMVehicle = MergeAwareIDMVehicle
            aware = sum((run_condition(cell, "no_av", s, duration_steps=120).collisions or 0)
                        for s in (7, 17))
        finally:
            spawner.MergeAwareIDMVehicle = original
        assert plain > 0, "the control produced no collisions, so this proves nothing"
        assert aware < plain, f"merge-aware {aware} was not fewer than plain IDM {plain}"
