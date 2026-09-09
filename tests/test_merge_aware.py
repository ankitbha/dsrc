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


class TestWhatItDoesAndDoesNotBuy:
    """Measured on the 108-run grid, and the result is mixed.

    Measured on 81 DISTINCT runs against a same-run plain-IDM control: AV episodes
    complete 43/81 to 57/81, +17 points, and collisions fall 155 to 138 over the 9
    distinct `no_av` conditions at penetration 0.10.

    Two earlier readings of this are superseded and should not be quoted. A 43%
    collision cut was an artifact of unbounded braking -- mean_speed reached
    -2.6e11 m/s. A later -19% still included vehicles reversing out of trouble at
    up to -7.5 m/s. Both are gone, and so is the `burst` condition, which is
    byte-identical to `medium` at 120 steps and made every total count `medium`
    twice.
    """

    def test_av_episodes_complete_more_often_than_with_plain_idm(self):
        from highway_env.vehicle.behavior import IDMVehicle

        import src.demand.spawner as spawner
        from src.analysis.simulator_health import CellSpec, run_condition

        cell = CellSpec(topology="inverted_tree", demand="high", av_penetration=0.20)
        original = spawner.MergeAwareIDMVehicle

        def completed(cls):
            spawner.MergeAwareIDMVehicle = cls
            return sum(
                run_condition(cell, "backpressure", s, duration_steps=120).completed
                for s in (7, 17, 27)
            )

        try:
            plain = completed(IDMVehicle)
            aware = completed(MergeAwareIDMVehicle)
        finally:
            spawner.MergeAwareIDMVehicle = original
        assert plain < 3, "the control completed everything, so this proves nothing"
        assert aware >= plain, f"merge-aware completed {aware}, plain IDM {plain}"


class TestItCannotProduceAnAbsurdAcceleration:
    """IDM's gap term is `(desired_gap / d)^2`, which diverges as the gap closes.

    Real vehicles never reach a sub-metre gap because a collision is detected
    first, but a PROJECTED leader can be placed arbitrarily close, and the first
    version of this class did exactly that: a `no_av` run reached a mean_speed of
    -2.6e11 m/s at step 28, against 28.1 for plain IDM on the same seed. The score
    the trainer maximises is `mean_speed - jam_fraction`, so that number goes
    straight into training.
    """

    def test_a_vanishing_projected_gap_stays_within_braking_limits(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0, speed=25.0)
        # 1 cm ahead in the projection: the worst case the filter allows.
        _place(road, ("a4_entry", "b2", 0), 400.01, speed=0.0)
        acceleration = ego._merge_acceleration()
        assert acceleration is not None
        assert acceleration >= -abs(ego.ACC_MAX) - 1e-6, (
            f"projected gap of 1 cm produced {acceleration} m/s^2"
        )

    def test_a_full_run_keeps_mean_speed_physical(self):
        from src.analysis.simulator_health import SampleSpec, build_env_config
        from src.baselines.registry import make_baseline
        from src.envs.topology_env import HighwayTopologyEnv

        spec = SampleSpec(topology="inverted_tree", controller="no_av", seed=7,
                          duration_steps=120, demand="medium", human_model="normal",
                          av_penetration=0.0)
        env = HighwayTopologyEnv("inverted_tree", build_env_config(spec))
        policy = make_baseline("no_av")
        policy.reset(env_metadata={"topology_id": "inverted_tree"}, seed=7)
        obs, _ = env.reset(seed=7)
        terminated = truncated = False
        worst = 0.0
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(policy.act(obs, global_state=None))
            speed = (info.get("metrics") or {}).get("mean_speed", 0.0)
            if abs(speed) > abs(worst):
                worst = speed
        assert abs(worst) < 100.0, f"mean_speed reached {worst} m/s"


class TestWhoCountsAsAConflict:
    """Two filters the first version lacked, both measured on a single run.

    21% of 17,639 phantom leaders were vehicles that had already crashed: a wreck
    is decelerated to a stop and then sits at the node forever, and everything
    upstream yielded to it indefinitely.

    48% of the conflicts computed at node `c` were between lanes that never meet.
    `b1->c` lane 0 and `b2->c` lane 0 both feed `c->exit` lane 0 and really do
    converge; the cross pairs feed different exit lanes and cannot collide.
    """

    def test_a_crashed_vehicle_is_not_yielded_to(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        wreck = _place(road, ("a4_entry", "b2", 0), 440.0, speed=0.0)
        wreck.crashed = True
        assert ego._merge_acceleration() is None

    def test_a_live_vehicle_in_the_same_place_still_counts(self, road):
        # The control: without it the test above would pass on any change that
        # disabled merging altogether.
        ego = _place(road, ("a5_entry", "b2", 0), 400.0)
        _place(road, ("a4_entry", "b2", 0), 440.0, speed=0.0)
        assert ego._merge_acceleration() is not None

    def test_lanes_that_feed_different_exits_are_not_a_conflict(self, road):
        # At node c, ('b1','c',0) and ('b2','c',1) do not share a successor.
        network = road.network
        first = ("b1", "c", 0)
        second = ("b2", "c", 1)
        def successor(index):
            lane = network.get_lane(index)
            return network.next_lane(index, route=None, position=lane.position(lane.length, 0))
        assert successor(first) != successor(second), "fixture assumption broken"
        ego = _place(road, first, 500.0)
        _place(road, second, 540.0, speed=0.0)
        assert ego._merge_acceleration() is None

    def test_lanes_that_feed_the_same_exit_are_a_conflict(self, road):
        network = road.network
        def successor(index):
            lane = network.get_lane(index)
            return network.next_lane(index, route=None, position=lane.position(lane.length, 0))
        pair = [
            (a, b)
            for a in (("b1", "c", 0), ("b1", "c", 1))
            for b in (("b2", "c", 0), ("b2", "c", 1))
            if successor(a) == successor(b)
        ]
        assert pair, "no converging pair at node c; fixture assumption broken"
        first, second = pair[0]
        ego = _place(road, first, 500.0)
        _place(road, second, 540.0, speed=0.0)
        assert ego._merge_acceleration() is not None


class TestItCannotDriveBackwards:
    """The clamp bounds the acceleration; it does not bound the speed.

    IDM's own braking vanishes as a vehicle stops, because its desired gap scales
    with the ego's speed. A PROJECTED gap is a difference of distances-to-node and
    does not shrink with speed, so the full -6 m/s^2 can be commanded at a
    standstill: one 1 s step from 2.5 m/s reaches -3.5 m/s, and highway_env only
    floors speed at -40. Measured on demand=medium, seed 7: minimum vehicle speed
    -7.473 m/s, 11 vehicle-steps in reverse.

    `test_a_full_run_keeps_mean_speed_physical` does not catch it -- the mean over
    ~40 vehicles reads 28.1 on that same run. That guard is on the aggregate that
    blew up before, not on the quantity that is wrong.
    """

    def test_braking_vanishes_as_the_vehicle_stops(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0, speed=2.0)
        _place(road, ("a4_entry", "b2", 0), 400.5, speed=0.0)
        acceleration = ego._merge_acceleration()
        assert acceleration is not None
        # One second of this must not reverse the vehicle.
        assert acceleration >= -2.0 - 1e-6, (
            f"a vehicle at 2.0 m/s was told to decelerate at {acceleration} m/s^2"
        )

    def test_a_stopped_vehicle_is_not_braked_further(self, road):
        ego = _place(road, ("a5_entry", "b2", 0), 400.0, speed=0.0)
        _place(road, ("a4_entry", "b2", 0), 400.5, speed=0.0)
        acceleration = ego._merge_acceleration()
        assert acceleration is None or acceleration >= -1e-6

    def test_hard_braking_is_still_available_at_speed(self, road):
        # The control: the fix must not sedate the yield rule at road speed, or it
        # would trade one defect for another.
        ego = _place(road, ("a5_entry", "b2", 0), 400.0, speed=25.0)
        _place(road, ("a4_entry", "b2", 0), 400.5, speed=0.0)
        acceleration = ego._merge_acceleration()
        assert acceleration is not None
        assert acceleration <= -3.0, f"only {acceleration} m/s^2 at a 0.5 m projected gap"

    def test_no_vehicle_reverses_over_a_full_run(self):
        from src.analysis.simulator_health import SampleSpec, build_env_config
        from src.baselines.registry import make_baseline
        from src.envs.topology_env import HighwayTopologyEnv

        spec = SampleSpec(topology="inverted_tree", controller="no_av", seed=7,
                          duration_steps=120, demand="medium", human_model="normal",
                          av_penetration=0.0)
        env = HighwayTopologyEnv("inverted_tree", build_env_config(spec))
        policy = make_baseline("no_av")
        policy.reset(env_metadata={"topology_id": "inverted_tree"}, seed=7)
        obs, _ = env.reset(seed=7)
        terminated = truncated = False
        worst = 0.0
        while not (terminated or truncated):
            obs, _, terminated, truncated, _ = env.step(policy.act(obs, global_state=None))
            for vehicle in env.road.vehicles:
                worst = min(worst, float(vehicle.speed))
        assert worst > -0.5, f"a vehicle reached {worst} m/s"
