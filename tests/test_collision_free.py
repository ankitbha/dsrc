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

    RESIDUAL_ALLOWANCE = 6

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


class TestTheResidualIsGeometric:
    """Pins the cause, so the residual allowance above cannot quietly become a
    tolerance for a car-following defect."""

    def test_arcs_do_not_join_at_the_nodes(self):
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
        assert worst > 2.0, (
            "the arcs now join within a vehicle width, so the geometric residual "
            "should be gone and RESIDUAL_ALLOWANCE should drop to 0"
        )

    def test_traffic_still_moves(self):
        # The control. A cap that stops everything is trivially collision-free and
        # useless, and this is the failure mode to watch for.
        cell = CellSpec(topology="inverted_tree", demand="medium", av_penetration=0.10)
        run = run_condition(cell, "no_av", 7, duration_steps=120)
        assert run.mean_speed is not None and run.mean_speed > 5.0, (
            f"mean speed collapsed to {run.mean_speed} m/s"
        )
