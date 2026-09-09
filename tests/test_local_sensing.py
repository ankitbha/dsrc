from __future__ import annotations

import math

import numpy as np
import pytest

from src.road.topology_factory import build_topology
from src.safety import SafetyConstraints, SafetyState
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot


def snapshot(
    vehicle_id: str,
    *,
    role: str = "av",
    lane_id: int = 1,
    longitudinal_m: float = 100.0,
    speed_mps: float = 20.0,
    segment_id: str = "straight_upstream",
) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id=vehicle_id,
        role=role,
        segment_id=segment_id,
        lane_index=("s0", "s1", lane_id),
        lane_id=lane_id,
        position=(longitudinal_m, lane_id * 4.0),
        longitudinal_m=longitudinal_m,
        speed_mps=speed_mps,
        acceleration_mps2=0.0,
        free_flow_speed_mps=30.0,
    )


def build_obs(
    snapshots,
    *,
    config: SensingConfig | None = None,
    current_av_ids=("av_0",),
    time_s: float = 0.0,
    rng_seed: int = 7,
):
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(config)
    return builder.build_all(
        time_s=time_s,
        topology=topology,
        snapshots=snapshots,
        current_av_ids=list(current_av_ids),
        safety_states={agent_id: SafetyState() for agent_id in current_av_ids},
        target_headways={agent_id: 1.6 for agent_id in current_av_ids},
        target_lanes={agent_id: ("s0", "s1", 1) for agent_id in current_av_ids},
        segment_metrics={"straight_upstream": {"density": 4.0, "jam_fraction": 0.0}},
        constraints=SafetyConstraints(),
        rng=np.random.RandomState(rng_seed),
    )


def test_local_observation_contains_public_schema_sensor_and_cooperation() -> None:
    obs = build_obs([snapshot("av_0")])["av_0"]

    for key in (
        "is_active",
        "ego_speed",
        "ego_acceleration",
        "ego_lane",
        "ego_headway_s",
        "target_headway_s",
        "current_segment",
        "leader_gap",
        "follower_gap",
        "local_density_bin",
        "local_mean_speed_bin",
        "active_vehicle_count_local",
        "active_av_count_local",
        "nearby_av_count",
        "nearby_av_density",
        "nearby_av_mean_speed",
        "nearby_av_lane_distribution",
    ):
        assert key in obs
    assert obs["sensor"] == {
        "range_m": 150.0,
        "latency_s": 0.0,
        "position_noise_std": 0.0,
        "speed_noise_std": 0.0,
    }
    assert obs["cooperation"] == {
        "segment_target_speed": 30.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
    }


def test_local_counts_exclude_ego_and_respect_range() -> None:
    obs = build_obs(
        [
            snapshot("av_0", longitudinal_m=100.0),
            snapshot("av_1", longitudinal_m=130.0),
            snapshot("human_0", role="human", longitudinal_m=190.0),
            snapshot("av_2", longitudinal_m=260.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["active_vehicle_count_local"] == 2
    assert obs["active_av_count_local"] == 1
    assert obs["nearby_av_count"] == 1
    assert obs["nearby_av_density"] == pytest.approx(5.0)
    assert obs["nearby_av_mean_speed"] == pytest.approx(20.0)


def test_neutral_cooperation_fallback_when_no_nearby_avs() -> None:
    obs = build_obs(
        [
            snapshot("av_0", speed_mps=18.0),
            snapshot("human_0", role="human", longitudinal_m=130.0, speed_mps=12.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["nearby_av_count"] == 0
    assert obs["nearby_av_density"] == 0.0
    assert obs["nearby_av_mean_speed"] == pytest.approx(30.0)
    assert obs["nearby_av_lane_distribution"] == {}
    assert obs["cooperation"]["segment_target_speed"] == pytest.approx(30.0)
    assert obs["cooperation"]["merge_pressure"] == 0.0
    assert obs["cooperation"]["downstream_congestion_estimate"] == 0.0


def test_nearby_av_aggregates_hide_neighbor_identities() -> None:
    obs = build_obs(
        [
            snapshot("av_0", lane_id=1, longitudinal_m=100.0),
            snapshot("av_1", lane_id=0, longitudinal_m=130.0, speed_mps=10.0),
            snapshot("av_2", lane_id=2, longitudinal_m=140.0, speed_mps=20.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["nearby_av_count"] == 2
    assert obs["nearby_av_mean_speed"] == pytest.approx(15.0)
    assert obs["nearby_av_lane_distribution"] == {"0": 0.5, "2": 0.5}
    assert "av_1" not in str(obs["nearby_av_lane_distribution"])
    assert "av_2" not in str(obs["cooperation"])


def test_lane_relative_gaps_are_range_limited() -> None:
    obs = build_obs(
        [
            snapshot("av_0", lane_id=1, longitudinal_m=100.0, speed_mps=20.0),
            snapshot("human_front", role="human", lane_id=1, longitudinal_m=130.0, speed_mps=15.0),
            snapshot("human_rear", role="human", lane_id=1, longitudinal_m=80.0, speed_mps=25.0),
            snapshot("human_left", role="human", lane_id=0, longitudinal_m=120.0),
            snapshot("human_right", role="human", lane_id=2, longitudinal_m=90.0),
            snapshot("human_far", role="human", lane_id=1, longitudinal_m=400.0),
        ],
        config=SensingConfig(range_m=100.0),
    )["av_0"]

    assert obs["leader_gap"] == pytest.approx(30.0)
    assert obs["leader_relative_speed"] == pytest.approx(-5.0)
    assert obs["follower_gap"] == pytest.approx(20.0)
    assert obs["follower_relative_speed"] == pytest.approx(5.0)
    assert obs["left_lane_front_gap"] == pytest.approx(20.0)
    assert math.isinf(obs["left_lane_rear_gap"])
    assert math.isinf(obs["right_lane_front_gap"])
    assert obs["right_lane_rear_gap"] == pytest.approx(10.0)
    assert obs["leader_gap"] < 300.0


def test_density_speed_bins_and_queue_estimate() -> None:
    obs = build_obs(
        [
            snapshot("av_0", longitudinal_m=100.0, speed_mps=20.0),
            snapshot("human_0", role="human", longitudinal_m=110.0, speed_mps=4.0),
            snapshot("human_1", role="human", longitudinal_m=120.0, speed_mps=6.0),
            snapshot("human_2", role="human", longitudinal_m=130.0, speed_mps=8.0),
        ],
        config=SensingConfig(range_m=50.0, density_bin_edges_veh_per_km=(12.0, 30.0), mean_speed_bin_edges_mps=(5.0, 10.0)),
    )["av_0"]

    assert obs["local_density_bin"] == 2
    assert obs["local_mean_speed_bin"] == 1
    assert obs["local_queue_estimate"] == 1


def test_noise_is_seed_reproducible_and_changes_continuous_measurements() -> None:
    snapshots = [
        snapshot("av_0", longitudinal_m=100.0, speed_mps=20.0),
        snapshot("human_front", role="human", longitudinal_m=130.0, speed_mps=15.0),
    ]
    noisy = SensingConfig(position_noise_std=2.0, speed_noise_std=1.0)

    obs_a = build_obs(snapshots, config=noisy, rng_seed=17)["av_0"]
    obs_b = build_obs(snapshots, config=noisy, rng_seed=17)["av_0"]
    obs_clean = build_obs(snapshots, config=SensingConfig(), rng_seed=17)["av_0"]

    assert obs_a["leader_gap"] == pytest.approx(obs_b["leader_gap"])
    assert obs_a["leader_relative_speed"] == pytest.approx(obs_b["leader_relative_speed"])
    assert obs_a["leader_gap"] != pytest.approx(obs_clean["leader_gap"])


def test_latency_uses_oldest_frame_until_buffer_is_warm_then_delayed_frame() -> None:
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(SensingConfig(latency_s=1.0))
    kwargs = {
        "topology": topology,
        "current_av_ids": ["av_0"],
        "safety_states": {"av_0": SafetyState()},
        "target_headways": {"av_0": 1.6},
        "target_lanes": {"av_0": ("s0", "s1", 1)},
        "segment_metrics": {"straight_upstream": {"density": 1.0, "jam_fraction": 0.0}},
        "constraints": SafetyConstraints(),
    }

    first = builder.build_all(
        time_s=0.0,
        snapshots=[snapshot("av_0", speed_mps=10.0)],
        rng=np.random.RandomState(3),
        **kwargs,
    )["av_0"]
    delayed = builder.build_all(
        time_s=1.0,
        snapshots=[snapshot("av_0", speed_mps=20.0)],
        rng=np.random.RandomState(3),
        **kwargs,
    )["av_0"]

    assert first["ego_speed"] == pytest.approx(10.0)
    assert delayed["ego_speed"] == pytest.approx(10.0)


def test_lane_gap_context_reuses_frame_recorded_by_build_all() -> None:
    topology = build_topology("straight_multilane")
    builder = LocalObservationBuilder(SensingConfig())
    snapshots = [
        snapshot("av_0", longitudinal_m=100.0),
        snapshot("human_rear", role="human", longitudinal_m=80.0),
    ]
    kwargs = {
        "topology": topology,
        "current_av_ids": ["av_0"],
        "safety_states": {"av_0": SafetyState()},
        "target_headways": {"av_0": 1.6},
        "target_lanes": {"av_0": ("s0", "s1", 1)},
        "segment_metrics": {"straight_upstream": {"density": 1.0, "jam_fraction": 0.0}},
        "constraints": SafetyConstraints(),
    }

    builder.build_all(time_s=2.0, snapshots=snapshots, rng=np.random.RandomState(3), **kwargs)
    frame_count = len(builder.buffer._frames)
    builder.lane_gap_context(
        ego_id="av_0",
        time_s=2.0,
        topology=topology,
        snapshots=snapshots,
        target_lane=("s0", "s1", 1),
        rng=np.random.RandomState(3),
    )

    assert len(builder.buffer._frames) == frame_count


# ---------------------------------------------------------------------------
# Gaps across arc boundaries.
#
# `_lane_gaps` matched neighbours on an exact `lane_index`, and `longitudinal_m`
# restarts at every arc, so a vehicle metres ahead on the next arc was invisible
# while one hundreds of metres ahead on the ego's own arc was reported as the
# leader. Measured on inverted_tree: 51 terminating collisions, of which 27 were
# with a vehicle on a sibling arc feeding the same node and 18 were same-lane
# collisions where the leader first became visible at a mean of 74 m while
# needing 75 m to stop. The safety layer brakes correctly for what it is told.
# ---------------------------------------------------------------------------

def _tree():
    from src.config.loaders import load_named_config

    return build_topology("inverted_tree", load_named_config("topology", "inverted_tree"))


def _on_lane(topology, vehicle_id, lane_index, longitudinal_m, *, speed_mps=20.0,
             role="human", crashed=False):
    """A snapshot placed at real coordinates on a named lane.

    Position has to be genuine because `_sensed_neighbors` filters on Euclidean
    range before any lane reasoning happens.
    """
    lane = topology.road_network.get_lane(lane_index)
    x, y = lane.position(longitudinal_m, 0)
    return VehicleSnapshot(
        vehicle_id=vehicle_id,
        role=role,
        segment_id=topology.segment_for_lane(lane_index),
        lane_index=lane_index,
        lane_id=lane_index[2],
        position=(float(x), float(y)),
        longitudinal_m=float(longitudinal_m),
        speed_mps=speed_mps,
        acceleration_mps2=0.0,
        free_flow_speed_mps=30.0,
        crashed=crashed,
    )


def _gap_context(topology, snapshots, ego_id="av_0", config=None):
    builder = LocalObservationBuilder(config or SensingConfig())
    return builder.lane_gap_context(
        ego_id=ego_id,
        time_s=0.0,
        topology=topology,
        snapshots=snapshots,
        target_lane=None,
        rng=np.random.RandomState(7),
    )


class TestLeaderIsFoundAcrossAnArcBoundary:

    def test_a_leader_on_the_next_arc_is_reported(self) -> None:
        topology = _tree()
        # 50 m from the end of a 500 m entry arc, with a vehicle 20 m onto the arc
        # it feeds: a true gap of 70 m. The successor of ('a5_entry','b2',0) is
        # ('b2','c',1), NOT ordinal 0 -- the b2 arcs end at y = -14 and lane 1 is
        # the one that starts there.
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        lead = _on_lane(topology, "h_1", ("b2", "c", 1), 20.0, speed_mps=8.0)
        ctx = _gap_context(topology, [ego, lead])
        assert ctx.leader_gap_m == pytest.approx(70.0, abs=2.0), (
            f"leader on the successor arc reported as {ctx.leader_gap_m}"
        )
        assert ctx.leader_relative_speed_mps == pytest.approx(8.0 - 20.0, abs=0.5)

    def test_a_vehicle_in_a_lane_the_ego_never_enters_is_not_a_leader(self) -> None:
        # The control for the test above, and the defect it replaces: an
        # ordinal-preserving rule reported ('b2','c',0) as the leader, which is a
        # lane this ego never drives on, while the vehicle it was about to meet on
        # ('b2','c',1) read as infinitely far away.
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        wrong_lane = _on_lane(topology, "h_1", ("b2", "c", 0), 20.0, speed_mps=8.0)
        ctx = _gap_context(topology, [ego, wrong_lane])
        assert ctx.leader_gap_m == float("inf")

    def test_the_successor_ordinal_is_taken_from_the_network(self) -> None:
        # Pins the fact the rule depends on, so a topology change that alters it
        # fails here rather than silently blinding the ego.
        topology = _tree()
        network = topology.road_network
        for entry, expected in (
            (("a1_entry", "b1", 0), ("b1", "c", 0)),
            (("a5_entry", "b2", 0), ("b2", "c", 1)),
        ):
            lane = network.get_lane(entry)
            assert network.next_lane(entry, route=None,
                                     position=lane.position(lane.length, 0)) == expected

    def test_a_nearer_leader_on_the_next_arc_beats_a_far_one_on_this_arc(self) -> None:
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 100.0, role="av")
        far = _on_lane(topology, "h_far", ("a5_entry", "b2", 0), 240.0)
        # 400 m to the end of the ego's arc plus 10 m = 410 m: further away, so
        # the same-arc vehicle at 140 m must still win. This is the control for
        # the test above -- the fix must not simply prefer the other arc.
        near = _on_lane(topology, "h_next", ("b2", "c", 1), 10.0)
        ctx = _gap_context(topology, [ego, far, near])
        assert ctx.leader_gap_m == pytest.approx(140.0, abs=2.0)

    def test_a_follower_on_the_previous_arc_is_reported(self) -> None:
        topology = _tree()
        # ('a5_entry','b2',0) feeds ('b2','c',1), so that is the lane a follower
        # from it appears behind.
        ego = _on_lane(topology, "av_0", ("b2", "c", 1), 30.0, role="av")
        behind = _on_lane(topology, "h_1", ("a5_entry", "b2", 0), 480.0)
        ctx = _gap_context(topology, [ego, behind])
        assert ctx.follower_gap_m == pytest.approx(50.0, abs=2.0)

    def test_range_m_still_bounds_the_search(self) -> None:
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 380.0, role="av")
        # 120 m to the arc end plus 60 m = 180 m, beyond a 100 m range.
        lead = _on_lane(topology, "h_1", ("b2", "c", 1), 60.0)
        ctx = _gap_context(topology, [ego, lead], config=SensingConfig(range_m=100.0))
        assert ctx.leader_gap_m == float("inf"), (
            "a vehicle beyond range_m must stay invisible; the fix must not make "
            "the AV clairvoyant"
        )


class TestMergingTrafficIsVisible:

    def test_a_vehicle_on_a_sibling_arc_produces_a_merge_conflict(self) -> None:
        topology = _tree()
        # a4_entry and a5_entry both feed b2. Neither is on the other's route,
        # so neither is a leader, but they converge and 53% of the measured
        # collisions were exactly this pair.
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        other = _on_lane(topology, "h_1", ("a4_entry", "b2", 0), 460.0)
        ctx = _gap_context(topology, [ego, other])
        assert ctx.distance_to_next_merge_m == pytest.approx(50.0, abs=2.0)
        # Projected onto the shared node: the ego is 50 m from it, the other 40 m,
        # so the other arrives first and is effectively 10 m ahead.
        assert ctx.merge_conflict_gap_m == pytest.approx(10.0, abs=2.0)

    def test_the_nearest_converging_vehicle_wins_not_the_one_nearest_the_node(self) -> None:
        # The defect this replaces picked the vehicle closest to the node, which is
        # the one furthest ahead. Measured cost: a conflict reported 85.7 m away
        # while the vehicle actually struck was 4.1 m away, alongside.
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        # Chosen so the two rules give different answers: the old one reports the
        # far vehicle's 5 m to the node, the projection reports the near vehicle's
        # 20 m of following distance. Equal numbers would prove nothing.
        alongside = _on_lane(topology, "h_near", ("a4_entry", "b2", 0), 470.0)
        far_ahead = _on_lane(topology, "h_far", ("a4_entry", "b2", 0), 495.0)
        ctx = _gap_context(topology, [ego, alongside, far_ahead])
        assert ctx.merge_conflict_gap_m == pytest.approx(20.0, abs=2.0)

    def test_a_vehicle_the_ego_beats_to_the_node_is_not_a_conflict(self) -> None:
        # It arrives behind, so it is not a leader and braking for it would invent
        # an obstacle.
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 460.0, role="av")
        behind = _on_lane(topology, "h_1", ("a4_entry", "b2", 0), 430.0)
        ctx = _gap_context(topology, [ego, behind])
        assert ctx.merge_conflict_gap_m == float("inf")

    def test_an_arc_feeding_a_different_node_is_not_a_conflict(self) -> None:
        topology = _tree()
        # a1_entry feeds b1, not b2: these two never meet at this node.
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        other = _on_lane(topology, "h_1", ("a1_entry", "b1", 0), 450.0)
        ctx = _gap_context(topology, [ego, other])
        assert ctx.merge_conflict_gap_m == float("inf")

    def test_distance_to_next_merge_decreases_as_the_ego_approaches(self) -> None:
        topology = _tree()
        seen = []
        for longitudinal in (300.0, 380.0, 450.0, 490.0):
            ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), longitudinal, role="av")
            other = _on_lane(topology, "h_1", ("a4_entry", "b2", 0), longitudinal + 5.0)
            seen.append(_gap_context(topology, [ego, other]).distance_to_next_merge_m)
        assert seen == sorted(seen, reverse=True), f"not monotonic: {seen}"
        assert seen[-1] == pytest.approx(10.0, abs=2.0)


class TestASingleLaneReachableBothWays:
    """On a closed loop a lane is both ahead and behind, and the nearer wins.

    Taking the forward distance regardless reported a follower 75 m back as
    absent in both slots -- `leader_gap_m` infinite because the forward route to
    it exceeded `range_m`, and `follower_gap_m` infinite because the forward
    branch had already claimed the lane. `inverted_tree` is acyclic, so nothing
    there can exercise this and it needs the ring.
    """

    def test_a_near_follower_beats_a_far_forward_route(self) -> None:
        from src.config.loaders import load_named_config

        topology = build_topology("ring", load_named_config("topology", "ring"))
        # Four 65 m arcs. Ego 5 m onto ('r0','r1',0); the other 60 m onto
        # ('r2','r3',0), which is 185 m ahead around the loop -- past the 150 m
        # range -- but only 75 m behind. The nearer reading is the one that exists.
        ego = _on_lane(topology, "av_0", ("r0", "r1", 0), 5.0, role="av")
        behind = _on_lane(topology, "h_1", ("r2", "r3", 0), 60.0)
        ctx = _gap_context(topology, [ego, behind])
        assert not (ctx.leader_gap_m == float("inf")
                    and ctx.follower_gap_m == float("inf")), (
            "a vehicle 75 m behind and inside sensing range was reported as absent "
            "in both slots"
        )
        assert ctx.follower_gap_m == pytest.approx(75.0, abs=3.0)
        assert ctx.leader_gap_m == float("inf")


class TestTheSensingSideMergeFilters:
    """The same two filters as the vehicle, on the half that feeds the safety layer.

    Both were added in two places. The vehicle copy carries four tests and two
    controls; this copy carried none, and it is the one whose output reaches
    `SafetyContext` and therefore the controller under study.
    """

    def test_a_crashed_converging_vehicle_is_not_a_conflict(self) -> None:
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        wreck = _on_lane(topology, "h_1", ("a4_entry", "b2", 0), 460.0, crashed=True)
        ctx = _gap_context(topology, [ego, wreck])
        assert ctx.merge_conflict_gap_m == float("inf")

    def test_a_live_converging_vehicle_still_is(self) -> None:
        # The control: without it, disabling merge detection entirely would pass
        # the test above.
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        live = _on_lane(topology, "h_1", ("a4_entry", "b2", 0), 460.0)
        ctx = _gap_context(topology, [ego, live])
        assert ctx.merge_conflict_gap_m == pytest.approx(10.0, abs=2.0)

    def test_lanes_with_different_successors_are_not_a_conflict(self) -> None:
        topology = _tree()
        network = topology.road_network

        def successor(index):
            lane = network.get_lane(index)
            return network.next_lane(index, route=None, position=lane.position(lane.length, 0))

        first, second = ("b1", "c", 0), ("b2", "c", 1)
        assert successor(first) != successor(second), "fixture assumption broken"
        ego = _on_lane(topology, "av_0", first, 550.0, role="av")
        other = _on_lane(topology, "h_1", second, 560.0)
        ctx = _gap_context(topology, [ego, other])
        assert ctx.merge_conflict_gap_m == float("inf")


def _tree_observation(topology, snapshots, ego_id="av_0", config=None):
    """One encoded-observation dict from the tree topology."""
    builder = LocalObservationBuilder(config or SensingConfig())
    result = builder.build_all(
        time_s=0.0,
        topology=topology,
        snapshots=snapshots,
        current_av_ids=[ego_id],
        safety_states={ego_id: SafetyState()},
        target_headways={ego_id: 1.6},
        target_lanes={ego_id: None},
        segment_metrics={},
        constraints=SafetyConstraints(),
        rng=np.random.RandomState(7),
    )
    return result[ego_id]


class TestTheObservationSeesWhatTheSafetyLayerSees:
    """The policy's input had neither of the two fixes the safety layer got.

    `lane_gap_context` feeds `SafetyContext`; `build_one` builds the observation the
    actor is trained on. Only the first was given route-aware gaps, and
    `distance_to_next_merge` was still the literal 0.0 task 5 recorded as one of
    two structurally uninformative fields. So the MAPPO policy trained in task 68
    could see neither a leader across an arc boundary nor that a merge existed --
    on a topology whose every node is a merge.
    """

    def test_the_observation_reports_a_leader_on_the_next_arc(self) -> None:
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        lead = _on_lane(topology, "h_1", ("b2", "c", 1), 20.0, speed_mps=8.0)
        obs = _tree_observation(topology, [ego, lead])
        assert obs["leader_gap"] == pytest.approx(70.0, abs=2.0), (
            f"the observation reported {obs['leader_gap']} for a leader 70 m ahead"
        )

    def test_the_observation_keeps_merge_distance_at_sim_parity(self) -> None:
        # Deliberately 0.0 and asserted so, because `deployment/jetson`'s
        # observation builder has no map matching and sets the same. The real
        # distance IS computed -- `lane_gap_context` reports it -- but putting it
        # in the observation would train the policy on information the deployed
        # vehicle cannot measure, and task 47's parity ledger failed immediately
        # when it was tried.
        topology = _tree()
        ego = _on_lane(topology, "av_0", ("a5_entry", "b2", 0), 450.0, role="av")
        obs = _tree_observation(topology, [ego])
        assert obs["distance_to_next_merge"] == 0.0
        # And the sensing model does know the real value, so the information is
        # available to anything that may legitimately use it.
        ctx = _gap_context(topology, [ego])
        assert ctx.distance_to_next_merge_m == pytest.approx(50.0, abs=2.0)
