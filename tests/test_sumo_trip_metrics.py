"""Travel time, delay, stopping and jerk, and the per-agent neighbourhood.

None of these existed before task 103. The environment reported throughput, speed
and jam fraction, so the axes the predecessor paper's largest gains were on --
delay and flow smoothness -- could not be measured at all, and a per-agent reward
had nothing local to be priced from.

Every test here is paired with something that would move if the quantity were
computed the obvious wrong way: a level instead of a difference, one step instead
of a window, the agent's own segment instead of its neighbourhood.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sensing import VehicleSnapshot
from src.sumo.env import SumoTopologyEnv


def _env(tmp_path, *, demand="medium", **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", demand),
        # The calibrated Wiedemann-99 fleet, which is what the experiments run, and
        # which these metrics need. Under SUMO's default Krauss model a saturated
        # network creeps rather than stopping: measured at 8000 veh/h with 859
        # vehicles on the road and a mean speed of 1.51 m/s, `stopped_fraction` was
        # 0.006 under Krauss and 0.517 under W99. A test of stopping written
        # against Krauss asserts on a quantity that model does not produce.
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": 60,
        "dt": 1.0,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    return SumoTopologyEnv("inverted_tree", config)


def _snapshot(vehicle_id, acceleration, segment_id="tree_trunk_c"):
    return VehicleSnapshot(
        vehicle_id=vehicle_id,
        role="av",
        segment_id=segment_id,
        lane_index=("b", "c", 0),
        lane_id=0,
        position=(0.0, 0.0),
        longitudinal_m=0.0,
        speed_mps=10.0,
        acceleration_mps2=acceleration,
        free_flow_speed_mps=30.0,
    )


class TestFreeFlowTimes:

    def test_they_come_from_the_road_and_are_not_zero(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            times = env._route_free_flow_s
            assert times, "no route was measured"
            road = load_named_config("topology", "inverted_tree")["road"]
            limit = float(road["speed_limit_mps"])
            lengths = road["segment_lengths"]
            # inverted_tree is leaf -> middle -> trunk, so the free-flow time of
            # every route is the same three segments over the same limit.
            expected = sum(
                float(lengths[name]) for name in ("tree_leaf_a1", "tree_middle_b1", "tree_trunk_c")
            ) / limit
            for entry, value in times.items():
                assert value == pytest.approx(expected, rel=0.05), entry
            # And it is a real duration, not a defaulted zero.
            assert expected > 30.0
        finally:
            env.close()


class TestTravelTimeAndDelay:

    def test_every_completed_trip_is_timed(self, tmp_path):
        # A cross-check between two counters maintained independently: `_arrivals`
        # is extended by `getArrivedNumber`, `_completed_trips` by iterating
        # `getArrivedIDList` and looking up a departure time. If a departure time
        # is ever missing the second is short of the first, and the mean travel
        # time then reads low rather than absent.
        env = _env(tmp_path, duration_steps=120, warmup_steps=60)
        try:
            env.reset(seed=7)
            for _ in range(120):
                _, _, _, _, info = env.step({})
            metrics = info["metrics"]
            assert metrics["throughput_recent"] > 0, "nothing completed; the test proves nothing"
            assert len(env._completed_trips) == metrics["throughput_recent"]
            assert metrics["mean_travel_time_recent"] > 0.0
        finally:
            env.close()

    def test_a_trip_that_began_during_warm_up_keeps_its_departure_time(self, tmp_path):
        # `_branch_completed` and `arrived_total` are deliberately reset at the end
        # of warm-up. `_departure_time` must NOT be, for the same reason
        # `_origin_of` is not: a vehicle that departed during the fill and arrives
        # in the episode is only timeable from the departure recorded then. Cleared,
        # its arrival would carry no travel time and the early-episode mean would
        # be built from the few trips that began after the fill.
        env = _env(tmp_path, duration_steps=40, warmup_steps=90)
        try:
            env.reset(seed=7)
            for _ in range(40):
                _, _, _, _, info = env.step({})
            metrics = info["metrics"]
            assert metrics["throughput_recent"] > 0
            assert len(env._completed_trips) == metrics["throughput_recent"]
            # A route whose free-flow time is about 66 s cannot be driven in the 40
            # observed steps, so every one of these trips started during warm-up.
            free_flow = min(env._route_free_flow_s.values())
            assert metrics["mean_travel_time_recent"] >= free_flow
        finally:
            env.close()

    def test_delay_is_travel_time_above_the_route_free_flow_time(self, tmp_path):
        env = _env(tmp_path, duration_steps=120, warmup_steps=60)
        try:
            env.reset(seed=7)
            for _ in range(120):
                _, _, _, _, info = env.step({})
            metrics = info["metrics"]
            travel = [t[1] for t in env._completed_trips]
            expected = [max(0.0, t[1] - t[2]) for t in env._completed_trips if t[2] > 0.0]
            assert expected, "no trip had a known route, so delay was not measured"
            assert metrics["mean_delay_recent"] == pytest.approx(sum(expected) / len(expected))
            # Delay is strictly below travel time, which is what distinguishes it
            # from a copy of travel time under a different name.
            assert metrics["mean_delay_recent"] < sum(travel) / len(travel)
        finally:
            env.close()


class TestStopping:

    def test_stopped_fraction_can_be_zero_and_can_be_positive(self, tmp_path):
        # Both halves are the point. A metric that is positive under congestion but
        # cannot reach zero is measuring something other than stopping, and one
        # that is zero everywhere cannot charge a controller anything.
        free = _env(tmp_path / "free", demand="medium", duration_steps=40)
        try:
            free.reset(seed=7)
            for _ in range(40):
                _, _, _, _, info = free.step({})
            assert info["metrics"]["active_vehicle_count"] > 0
            uncongested = info["metrics"]["stopped_fraction"]
        finally:
            free.close()

        # 8000 veh/h, not the config's 3000. At 3000 the two-lane road holds the
        # excess on the carriageway rather than stopping it: 400 s in, mean speed
        # is 2.0 m/s and NOTHING is below 0.1 m/s, so the test asserted a
        # difference the run could not produce. The rate is raised rather than the
        # run lengthened because the assertion is about the metric's range, not
        # about any operating point.
        gridlock = dict(load_named_config("demand", "sumo_oversaturated"))
        gridlock["total_vehicles_per_hour"] = 8000.0
        jammed = _env(tmp_path / "jam", duration_steps=200, warmup_steps=300)
        jammed.config["demand"] = gridlock
        try:
            jammed.reset(seed=7)
            for _ in range(200):
                _, _, _, _, info = jammed.step({})
            congested = info["metrics"]["stopped_fraction"]
        finally:
            jammed.close()

        assert uncongested == pytest.approx(0.0, abs=0.02)
        assert congested > uncongested
        assert 0.0 < congested <= 1.0
    

class TestJerk:

    def test_it_is_a_change_in_acceleration_and_not_a_level(self, tmp_path):
        env = _env(tmp_path)
        env.config["dt"] = 0.1
        env._last_accel = {}
        # First sight of a vehicle: nothing to difference against.
        assert env._mean_abs_jerk([_snapshot("v1", 2.0)]) == 0.0
        # Same acceleration held: a level would report 2.0 here.
        assert env._mean_abs_jerk([_snapshot("v1", 2.0)]) == 0.0
        # A change of 1.0 m/s^2 over a 0.1 s step is 10 m/s^3.
        assert env._mean_abs_jerk([_snapshot("v1", 3.0)]) == pytest.approx(10.0)
        # And the sign does not cancel.
        assert env._mean_abs_jerk([_snapshot("v1", 2.0)]) == pytest.approx(10.0)

    def test_a_vehicle_absent_last_step_contributes_nothing(self, tmp_path):
        # A vehicle inside a junction has no edge and so no segment, so it drops out
        # of the snapshots for a few steps. Differencing across that gap would
        # report a jerk for a vehicle that was simply not being watched.
        env = _env(tmp_path)
        env.config["dt"] = 0.1
        env._last_accel = {}
        env._mean_abs_jerk([_snapshot("v1", 0.0)])
        env._mean_abs_jerk([])                              # v1 crosses a junction
        assert env._mean_abs_jerk([_snapshot("v1", 4.0)]) == 0.0
        assert env._mean_abs_jerk([_snapshot("v1", 4.0)]) == 0.0

    def test_it_is_measured_on_a_running_simulation(self, tmp_path):
        env = _env(tmp_path, demand="sumo_oversaturated", duration_steps=120,
                   warmup_steps=120, dt=0.5)
        try:
            env.reset(seed=7)
            values = []
            for _ in range(120):
                _, _, _, _, info = env.step({})
                values.append(info["metrics"]["mean_abs_jerk"])
            assert max(values) > 0.0, "no vehicle ever changed its acceleration"
        finally:
            env.close()


class TestNeighbourhood:

    def test_it_averages_the_agents_segment_with_the_segments_downstream(self, tmp_path):
        env = _env(tmp_path, demand="sumo_oversaturated", duration_steps=200,
                   warmup_steps=200)
        try:
            env.reset(seed=7)
            for _ in range(200):
                _, _, _, _, info = env.step({})
            neighbourhood = info["neighbourhood"]
            assert neighbourhood, "no agent was on a segment"
            segments = env.get_segment_metrics()
            downstream = env.view.downstream_segments()
            own_of = {s.vehicle_id: s.segment_id for s in env.vehicle_snapshots()}

            differs = 0
            for agent_id, values in neighbourhood.items():
                own = own_of[agent_id]
                names = [own, *downstream.get(own, ())]
                present = [segments[name] for name in names if name in segments]
                expected = sum(float(m["jam_fraction"]) for m in present) / len(present)
                assert values["jam_fraction"] == pytest.approx(expected)
                if len(present) > 1 and abs(
                        float(segments[own]["jam_fraction"]) - expected) > 1e-9:
                    differs += 1
            # Without this the assertion above passes on an own-segment-only
            # implementation: it only discriminates where the downstream segment is
            # in a different state from the agent's own.
            assert differs > 0, (
                "no agent had a downstream segment in a different state, so this "
                "test could not tell a neighbourhood mean from an own-segment value"
            )
        finally:
            env.close()

    def test_outflow_recent_is_a_window_and_not_one_step(self, tmp_path):
        env = _env(tmp_path, demand="sumo_oversaturated", duration_steps=200,
                   warmup_steps=200)
        try:
            env.reset(seed=7)
            for _ in range(200):
                _, _, _, _, info = env.step({})
            values = [v["outflow_recent"] for v in info["neighbourhood"].values()]
            assert values
            # A single step's outflow from one segment is 0 or a small integer; a
            # 60 s window at this demand is tens of vehicles.
            assert max(values) > 2.0
        finally:
            env.close()

    def test_the_warm_up_fills_the_outflow_window(self, tmp_path):
        # Paired against a run with no warm-up. Without the tail of the warm-up
        # recording flows, the window starts empty and the same traffic prices
        # lower on the first steps of the episode than in the middle of it.
        warmed = _env(tmp_path / "warm", demand="sumo_oversaturated",
                      duration_steps=2, warmup_steps=120)
        try:
            warmed.reset(seed=7)
            _, _, _, _, info = warmed.step({})
            with_warmup = max(
                [v["outflow_recent"] for v in info["neighbourhood"].values()] or [0.0])
        finally:
            warmed.close()

        cold = _env(tmp_path / "cold", demand="sumo_oversaturated",
                    duration_steps=2, warmup_steps=0)
        try:
            cold.reset(seed=7)
            _, _, _, _, info = cold.step({})
            without_warmup = max(
                [v["outflow_recent"] for v in info["neighbourhood"].values()] or [0.0])
        finally:
            cold.close()

        assert with_warmup > 0.0
        assert without_warmup == pytest.approx(0.0)


class TestTheStepInfoCarriesThem:

    def test_all_four_metrics_are_reported_every_step(self, tmp_path):
        env = _env(tmp_path, duration_steps=20)
        try:
            env.reset(seed=7)
            _, _, _, _, info = env.step({})
            for key in ("mean_travel_time_recent", "mean_delay_recent",
                        "stopped_fraction", "mean_abs_jerk"):
                assert key in info["metrics"], key
                assert isinstance(info["metrics"][key], float)
        finally:
            env.close()
