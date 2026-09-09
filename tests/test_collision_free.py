"""Collisions are impossible by construction, not penalised after the fact.

PTV Vissim and SUMO both guarantee this in their car-following models. This
simulator did not: at 120 steps the AV arms completed 30 of 54, giving a per-step
survival of 0.995114 and an expected 205 steps -- about 3.4 minutes -- to the first
AV crash. The chance of an hour-long run reaching its duration was 2.2e-8, so an
hour of simulated driving was unmeasurable.

**Each vehicle enforces the bound on itself, from what it can perceive**, rather
than the environment capping everything centrally. A referee reaching into every
vehicle is not a model of driver behaviour, and it would make every vehicle equally
safe regardless of its sensing -- erasing the property this project measures. Both
vehicle kinds mix in `CollisionFreeMixin` and so obey the same physics while
perceiving independently.

It is geometric rather than lane-based, which is what makes it cover all three
measured collision classes at once -- 27 of 51 were between vehicles on sibling
arcs, 18 same-lane, 6 in an adjacent lane of the same arc. A lane-based leader
search sees only the second.
"""
from __future__ import annotations

import math

import pytest

from src.analysis.simulator_health import CellSpec, run_condition
from src.vehicles import safe_following
from src.vehicles.safe_following import safe_speed


class TestTheSafeVelocityBound:

    def test_a_larger_gap_permits_a_higher_speed(self):
        safe = safe_speed
        speeds = [safe(gap, 0.0) for gap in (5.0, 20.0, 50.0, 100.0)]
        assert speeds == sorted(speeds)
        assert speeds[0] < speeds[-1]

    def test_a_zero_gap_permits_no_speed(self):
        assert safe_speed(0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_a_moving_leader_permits_more_than_a_stopped_one(self):
        safe = safe_speed
        assert safe(30.0, 20.0) > safe(30.0, 0.0)

    def test_the_bound_is_the_krauss_expression(self):
        # Pinned so a later edit has to argue with the closed form rather than
        # quietly changing what "safe" means.
        b = safe_following.SAFE_DECEL_MPS2
        tau = safe_following.SAFE_REACTION_S
        gap, lead = 40.0, 15.0
        expected = -b * tau + math.sqrt((b * tau) ** 2 + lead ** 2 + 2 * b * gap)
        assert safe_speed(gap, lead) == pytest.approx(expected)

    def test_stopping_distance_fits_in_the_gap(self):
        # The property the bound exists for: from v_safe, braking at b, the
        # vehicle stops within the gap plus the leader's own stopping distance.
        b = safe_following.SAFE_DECEL_MPS2
        tau = safe_following.SAFE_REACTION_S
        for gap in (2.0, 10.0, 45.0):
            for lead in (0.0, 8.0, 22.0):
                v = safe_speed(gap, lead)
                travelled = v * tau + v * v / (2 * b)
                available = gap + lead * lead / (2 * b)
                assert travelled <= available + 1e-6, (
                    f"gap {gap}, lead {lead}: needs {travelled:.2f} m, has {available:.2f} m"
                )


class TestWhatTheBoundAchievesAndWhatItCannot:
    """98 collisions to 6 over three seeds, and mean speed 10.81 to 16.62 m/s.

    Flow improves BECAUSE collisions stop: a crash halts a vehicle permanently and
    everything behind it queues.

    It is not zero, and the reason is not car-following. All six residual events are
    side-by-side, at lateral separations of 1.9 to 3.7 m, four of them below
    1.3 m/s. They are caused by the topology: the arcs do not join, so a vehicle
    crossing a node is translated sideways -- 4.00 m at b1, 2.00 m at c, and 10.00 m
    from ('b2','c',1) to ('c','exit',1). No speed bound can prevent a collision
    caused by a vehicle being moved laterally into occupied space, which is why
    tightening MOBIL's braking parameter from 2.0 to 0.05 changed the count not at
    all. Task 75 covers the geometry.
    """

    #: Not zero yet, and this number is a measurement rather than a tolerance.
    #: With the arcs joined, `no_av` at high/0.20 produces 2 collisions and
    #: `backpressure` produces 0. The remaining pair is recorded as open in task 77;
    #: it must not be raised to accommodate a regression.
    RESIDUAL_ALLOWANCE = 2

    @pytest.mark.parametrize("controller", ["no_av", "backpressure"])
    def test_collisions_are_at_most_the_recorded_geometric_residual(self, controller):
        cell = CellSpec(topology="inverted_tree", demand="high", av_penetration=0.20)
        run = run_condition(cell, controller, 7, duration_steps=120)
        assert run.collisions <= self.RESIDUAL_ALLOWANCE, (
            f"{controller} produced {run.collisions} collisions, above the "
            f"{self.RESIDUAL_ALLOWANCE} attributable to the node geometry"
        )

    def test_a_human_only_run_reaches_its_duration(self):
        # Stated for `no_av` rather than an AV arm on purpose. The bound is a clear
        # win for human traffic and mixed for AVs: cooperative_smoothing at
        # high/0.20 goes 0/3 to 2/3 completed and 13 to 2 collisions, but
        # backpressure at the same cell stays 0/3 and goes 6 to 14 collisions. A
        # slower AV spends longer in the node transitions that teleport vehicles
        # sideways, so the residual geometry hurts it more. Asserting AV completion
        # here would be asserting something that is not yet true.
        cell = CellSpec(topology="inverted_tree", demand="high", av_penetration=0.20)
        run = run_condition(cell, "no_av", 7, duration_steps=120)
        assert run.completed is True
        assert run.steps == 120


class TestTheArcsJoin:
    """The geometry that made collision-freedom unreachable, now fixed.

    A vehicle crossing node `c` from ('b2','c',1) was moved TEN METRES sideways,
    and several other transitions moved it 2 to 4 m. Two causes: the SineLanes used
    `phase=pi/2`, putting the sine at full amplitude AT the arc ends, and the
    nominal endpoints did not match their successors' starts. Both fixed, and the
    dropped lane of the bottleneck now tapers instead of ending beside the lane it
    merges into.
    """

    def test_every_transition_is_continuous(self):
        import numpy as np

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology(
            "inverted_tree", load_named_config("topology", "inverted_tree")
        ).road_network
        jumps = {}
        for index in sorted(network.lanes_dict()):
            lane = network.get_lane(index)
            end = np.array(lane.position(lane.length, 0))
            try:
                nxt = network.next_lane(index, route=None, position=end)
            except Exception:
                continue
            if nxt is None or nxt == index:
                continue
            start = np.array(network.get_lane(nxt).position(0.0, 0))
            jumps[(index, nxt)] = float(np.linalg.norm(end - start))
        assert jumps, "no transitions found; the fixture assumption is broken"
        worst = max(jumps.values())
        assert worst <= 0.5, (
            f"a transition jumps {worst:.2f} m laterally; a vehicle is 2 m wide, so "
            "it is being placed into occupied space and no car-following bound can "
            "prevent the resulting collision"
        )

    def test_traffic_still_moves(self):
        # The control. A cap that stops everything is trivially collision-free and
        # useless, and this is the failure mode to watch for.
        cell = CellSpec(topology="inverted_tree", demand="medium", av_penetration=0.10)
        run = run_condition(cell, "no_av", 7, duration_steps=120)
        assert run.mean_speed is not None and run.mean_speed > 5.0, (
            f"mean speed collapsed to {run.mean_speed} m/s"
        )


class TestConvergingPathsAreSeen:
    """The forward window could not see a conflict beside the vehicle.

    `perceived_safe_speed` caps on the nearest vehicle ahead within a 2.6 m lateral
    window. Measured at the saturating demand: AV arms die at steps 171, 225, 290
    and 319 of 600, and `no_av` seed 47 has 9 human-human collisions with no AVs
    present. The collisions are side-by-side, at 1.9 to 3.7 m lateral -- outside or
    at the edge of that window -- so a forward test cannot reach them.

    Widening the window is the wrong fix: at 4 m every adjacent-lane vehicle becomes
    a longitudinal constraint and multi-lane sections over-brake. What distinguishes
    a real conflict from parallel traffic is not current separation but whether the
    two paths converge, which is what closest-point-of-approach measures.
    """

    def _pair(self, road, ego_state, other_state):
        """Two vehicles with explicit positions, headings and speeds."""
        from src.vehicles.merge_aware import MergeAwareIDMVehicle

        made = []
        for position, heading, speed in (ego_state, other_state):
            vehicle = MergeAwareIDMVehicle(road, position, heading=heading, speed=speed)
            road.vehicles.append(vehicle)
            made.append(vehicle)
        return made

    def test_parallel_traffic_in_the_next_lane_is_not_a_conflict(self):
        # The control, and the reason a wider window is wrong. Two vehicles 3 m
        # apart laterally at the same speed and heading never meet, and must not
        # constrain each other at all.
        import numpy as np
        from highway_env.road.road import Road

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology("straight_multilane",
                                 load_named_config("topology", "straight_multilane")).road_network
        road = Road(network=network, np_random=np.random.RandomState(7), record_history=False)
        ego, _ = self._pair(road, ((0.0, 0.0), 0.0, 25.0), ((5.0, 3.0), 0.0, 25.0))
        assert ego.perceived_safe_speed() == float("inf")

    def test_converging_paths_outside_the_lateral_window_are_a_conflict(self):
        # 3.2 m apart laterally, so invisible to the 2.6 m window, but closing
        # laterally at 2 m/s: they arrive at the same point.
        import math

        import numpy as np
        from highway_env.road.road import Road

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology("straight_multilane",
                                 load_named_config("topology", "straight_multilane")).road_network
        road = Road(network=network, np_random=np.random.RandomState(7), record_history=False)
        # Closing in BOTH axes: 30 m ahead at 15 m/s against the ego's 25, so the
        # ego gains 10 m/s longitudinally, while drifting inward at 1 m/s across
        # the 3 m of lateral offset. Predicted closest approach is 0.01 m at 2.99 s.
        # A fixture converging only laterally never meets, which is what my first
        # attempt at this test described.
        converging = math.atan2(-1.0, 15.0)
        ego, _ = self._pair(road, ((0.0, 0.0), 0.0, 25.0),
                            ((30.0, 3.0), converging, 15.0))
        limit = ego.perceived_safe_speed()
        assert limit < 25.0, (
            f"a vehicle converging from 3.2 m lateral imposed no limit (got {limit})"
        )

    def test_a_separating_vehicle_is_not_a_conflict(self):
        # Diverging rather than converging. Without this the rule would brake for
        # anything nearby regardless of where it is going.
        import math

        import numpy as np
        from highway_env.road.road import Road

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology("straight_multilane",
                                 load_named_config("topology", "straight_multilane")).road_network
        road = Road(network=network, np_random=np.random.RandomState(7), record_history=False)
        diverging = math.atan2(1.0, 15.0)
        ego, _ = self._pair(road, ((0.0, 0.0), 0.0, 25.0),
                            ((30.0, 3.0), diverging, 15.0))
        assert ego.perceived_safe_speed() == float("inf")

    def test_a_conflict_beyond_the_horizon_is_ignored(self):
        import math

        import numpy as np
        from highway_env.road.road import Road

        from src.config.loaders import load_named_config
        from src.road.topology_factory import build_topology

        network = build_topology("straight_multilane",
                                 load_named_config("topology", "straight_multilane")).road_network
        road = Road(network=network, np_random=np.random.RandomState(7), record_history=False)
        # The same convergence as the test above, 200 m away instead of 30, so
        # closest approach is 19.8 s out. Genuinely a conflict, genuinely too far
        # to brake for now.
        far = math.atan2(-1.0, 15.0)
        ego, _ = self._pair(road, ((0.0, 0.0), 0.0, 25.0),
                            ((200.0, 3.0), far, 15.0))
        assert ego.perceived_safe_speed() == float("inf")
