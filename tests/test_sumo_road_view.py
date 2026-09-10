"""A SUMO-backed view of the road, so the sensing model needs no change.

`lane_gap_context` reads `topology.road_network` for lane lengths, lane geometry and
the successor of a lane, and `topology.segment_for_lane` for the segment id. Those
are the only things it wants from a road. Supplying them from SUMO keeps the sensing
model, its route-aware gap search and its merge projection exactly as they are and
as they were validated.

The alternative was to keep building the `highway_env` topology alongside the SUMO
one purely for geometry. That would have been quicker and wrong: the two roads have
different shapes -- `highway_env` used sine curves where SUMO uses straight edges --
so every gap computed from one and applied to the other would be slightly off, which
is the undiagnosable class of defect the generated network exists to avoid.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sumo.network import SumoNetwork
from src.sumo.road_view import SumoTopologyView


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    network = SumoNetwork.build(
        "inverted_tree",
        load_named_config("topology", "inverted_tree"),
        tmp_path_factory.mktemp("view"),
    )
    return SumoTopologyView(network)


class TestItAnswersWhatTheSensingModelAsks:

    def test_every_lane_is_listed(self, view):
        lanes = view.road_network.lanes_dict()
        # 6 single-lane entries, 2 two-lane middles, 1 two-lane trunk = 12.
        assert len(lanes) == 18, sorted(lanes)

    def test_lane_lengths_match_the_spec(self, view):
        network = view.road_network
        for index in network.lanes_dict():
            length = network.get_lane(index).length
            if index[0].startswith("a"):
                assert length == pytest.approx(500.0, abs=1.0), index
            elif index[1] == "c":
                assert length == pytest.approx(600.0, abs=1.0), index
            else:
                assert length == pytest.approx(900.0, abs=1.0), index

    def test_a_lane_has_a_successor_where_the_road_continues(self, view):
        network = view.road_network
        entry = ("a1", "b1", 0)
        lane = network.get_lane(entry)
        successor = network.next_lane(entry, route=None,
                                      position=lane.position(lane.length, 0))
        assert successor is not None
        assert successor[0] == "b1", f"a1->b1 continues into {successor}"

    def test_the_exit_lane_has_no_successor(self, view):
        network = view.road_network
        exit_lane = ("c", "exit", 0)
        lane = network.get_lane(exit_lane)
        assert network.next_lane(exit_lane, route=None,
                                 position=lane.position(lane.length, 0)) is None

    def test_position_and_local_coordinates_round_trip(self, view):
        # The sensing model converts between the two constantly; a mismatch would
        # move every gap it computes.
        #
        # The lane's declared length is 500 m while its drawn shape measures
        # 457.60 m, because netconvert shortens a lane at a junction and the node
        # geometry is a diagonal. Distances are expressed in the declared length and
        # mapped onto the shape, so this round trip is what pins that mapping.
        lane = view.road_network.get_lane(("a1", "b1", 0))
        for distance in (0.0, 120.0, 499.0):
            x, y = lane.position(distance, 0)
            back, lateral = lane.local_coordinates((x, y))
            assert back == pytest.approx(distance, abs=1.0)
            assert lateral == pytest.approx(0.0, abs=1.0)

    def test_the_graph_reports_the_merges(self, view):
        graph = view.road_network.graph
        # Three entries into each of b1 and b2 is the shape that matters: it is what
        # the merge projection keys on.
        into_b1 = [origin for origin, dests in graph.items() if "b1" in dests]
        into_b2 = [origin for origin, dests in graph.items() if "b2" in dests]
        assert len(into_b1) == 3, into_b1
        assert len(into_b2) == 3, into_b2

    def test_segments_are_reported_for_every_lane(self, view):
        for index in view.road_network.lanes_dict():
            assert view.segment_for_lane(index) is not None, index


class TestItWorksWithTheRealSensingModel:

    def test_lane_gap_context_runs_against_this_view(self, view):
        # The point of the whole adapter: the unchanged sensing model accepts it.
        import numpy as np

        from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot

        def snapshot(vehicle_id, lane_index, longitudinal, role="human", speed=20.0):
            lane = view.road_network.get_lane(lane_index)
            x, y = lane.position(longitudinal, 0)
            return VehicleSnapshot(
                vehicle_id=vehicle_id, role=role,
                segment_id=view.segment_for_lane(lane_index),
                lane_index=lane_index, lane_id=lane_index[2],
                position=(float(x), float(y)), longitudinal_m=longitudinal,
                speed_mps=speed, acceleration_mps2=0.0, free_flow_speed_mps=30.0,
            )

        ego = snapshot("av_0", ("a1", "b1", 0), 300.0, role="av")
        leader = snapshot("h_1", ("a1", "b1", 0), 340.0, speed=8.0)
        builder = LocalObservationBuilder(SensingConfig())
        context = builder.lane_gap_context(
            ego_id="av_0", time_s=0.0, topology=view, snapshots=[ego, leader],
            target_lane=None, rng=np.random.RandomState(7),
        )
        assert context.leader_gap_m == pytest.approx(40.0, abs=2.0)
        assert context.leader_relative_speed_mps == pytest.approx(-12.0, abs=0.5)


class TestTheLaneDropIsSensed:
    """`bottleneck_segments` returned an empty tuple whatever the network, so the
    bottleneck variant sensed identically to the plain one although netconvert had
    built the lane drop. Two sensing fields read it: the distance to the downstream
    bottleneck, and the branch deciding whether cooperation is scored at all.
    """

    def test_the_plain_tree_has_no_lane_drop(self, tmp_path):
        network = SumoNetwork.build(
            "inverted_tree", load_named_config("topology", "inverted_tree"), tmp_path)
        assert SumoTopologyView(network).bottleneck_segments == ()

    def test_the_bottleneck_variant_reports_its_lane_drop(self, tmp_path):
        network = SumoNetwork.build(
            "inverted_tree_bottleneck",
            load_named_config("topology", "inverted_tree_bottleneck"), tmp_path)
        view = SumoTopologyView(network)
        assert view.bottleneck_segments == ("tree_bottleneck_d",)
        # Derived from the built lane counts rather than declared, so it agrees with
        # the road by construction: the drop is from the two-lane trunk to one lane.
        assert view.lane_counts["tree_trunk_c"] == 2
        assert view.lane_counts["tree_bottleneck_d"] == 1

    def test_a_vehicle_on_the_lane_drop_senses_zero_distance_to_it(self, tmp_path):
        from src.sumo.env import SumoTopologyEnv

        env = SumoTopologyEnv("inverted_tree_bottleneck", {
            "topology": load_named_config("topology", "inverted_tree_bottleneck"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 200, "dt": 1.0, "warmup_steps": 200,
            "work_dir": str(tmp_path)})
        env.reset(seed=5)
        try:
            on_drop = elsewhere = 0
            for _ in range(200):
                observations, _, _, _, _ = env.step({})
                segment_of = {s.vehicle_id: s.segment_id for s in env.vehicle_snapshots()}
                for agent_id, observation in observations.items():
                    distance = observation["distance_to_downstream_bottleneck"]
                    if segment_of.get(agent_id) == "tree_bottleneck_d":
                        assert distance == 0.0
                        on_drop += 1
                    else:
                        elsewhere += 1
            assert on_drop > 0, "no AV was ever observed on the lane drop"
            assert elsewhere > 0
        finally:
            env.close()
