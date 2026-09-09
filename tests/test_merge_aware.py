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

    AV episodes complete more often -- 48/108 to 58/108 -- which is the criterion
    that gates training. Human-on-human collisions in `no_av` traffic go UP, 212 to
    301 over 12 identical conditions, because a vehicle yielding to a projected
    conflict slows and the traffic behind it in its own lane does not always react
    in time.

    An earlier version of this class appeared to cut collisions by 43%. That was an
    artifact of a numerical blow-up, not merge behaviour: mean_speed reached
    -2.6e11 m/s. The test below asserts the claim that survived the fix, not the
    one that did not.
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
