"""Observations out of the SUMO env, and AV commands into it.

The sensing model, the safety layer and the encoders are unchanged; this is the
wiring that feeds them SUMO's state and applies what they decide. The measurements
that matter are that an AV commanded slower actually slows, and that the metrics the
reward reads are populated from SUMO rather than defaulted.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sumo.env import SumoTopologyEnv


def _env(tmp_path, **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "saturating"),
        "duration_steps": 200,
        "dt": 1.0,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    return SumoTopologyEnv("inverted_tree", config)


class TestObservations:

    def test_each_av_gets_an_observation(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            observations = {}
            for _ in range(120):
                observations, _, _, _, _ = env.step({})
                if observations:
                    break
            assert observations, "no AV observation after 120 steps"
            assert set(observations) == set(env.agent_ids)
            for observation in observations.values():
                assert observation["ego_speed"] >= 0.0
                assert observation["is_active"] is True
                assert "leader_gap" in observation
        finally:
            env.close()

    def test_a_leader_shortens_the_reported_gap(self, tmp_path):
        # The control that the gap is measured rather than defaulted: with traffic
        # present at a congesting demand, at least one AV must see a finite leader.
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            finite = 0
            for _ in range(200):
                observations, _, _, _, _ = env.step({})
                finite += sum(
                    1 for o in observations.values()
                    if o["leader_gap"] != float("inf")
                )
            assert finite > 0, "no AV ever reported a finite leader gap"
        finally:
            env.close()


class TestActuation:

    def test_an_av_commanded_slower_actually_slows(self, tmp_path):
        # Commanded in m/s rather than through a speed bin. At a congesting demand
        # an AV is already slow, so `slow` -- which decodes to the allowed speed
        # minus 10 -- would often be ABOVE its current speed and the test would
        # pass or fail on the traffic state rather than on the actuation.
        env = _env(tmp_path, demand=load_named_config("demand", "low"))
        try:
            env.reset(seed=7)
            for _ in range(150):
                env.step({})
                if env.agent_ids and env.av_speed(env.agent_ids[0]) > 8.0:
                    break
            if not env.agent_ids:
                pytest.skip("no AV appeared at low demand within 150 steps")
            target = env.agent_ids[0]
            before = env.av_speed(target)
            for _ in range(12):
                if target not in env.agent_ids:
                    pytest.skip("the AV left the network before the command landed")
                env.command_speed(target, 2.0)
                env.step({})
            after = env.av_speed(target) if target in env.agent_ids else 0.0
            assert after < before - 1.0, (
                f"commanded 2 m/s and speed went {before:.2f} -> {after:.2f} m/s"
            )
        finally:
            env.close()

    def test_an_uncommanded_av_is_left_to_sumo(self, tmp_path):
        # The control for the test above. If the env clamped speed regardless of
        # the command, the previous test would pass for the wrong reason.
        env = _env(tmp_path, demand=load_named_config("demand", "low"))
        try:
            env.reset(seed=7)
            for _ in range(120):
                env.step({})
                if env.agent_ids:
                    break
            target = env.agent_ids[0]
            speeds = []
            for _ in range(15):
                env.step({})
                if target in env.agent_ids:
                    speeds.append(env.av_speed(target))
            assert speeds, "the AV left the network"
            assert max(speeds) > 1.0, (
                "an uncommanded AV never exceeded 1 m/s, so something is holding it"
            )
        finally:
            env.close()


class TestMetrics:

    def test_the_reward_metrics_are_populated_from_sumo(self, tmp_path):
        env = _env(tmp_path, duration_steps=300)
        try:
            env.reset(seed=7)
            info = {}
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            metrics = info["metrics"]
            for field in ("mean_speed", "throughput_recent", "jam_fraction",
                          "collision_count", "queue_length_total"):
                assert field in metrics, f"{field} missing from metrics"
            assert metrics["mean_speed"] > 0.0
            assert metrics["collision_count"] == 0
        finally:
            env.close()

    def test_throughput_counts_arrivals(self, tmp_path):
        env = _env(tmp_path, duration_steps=300)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            # Over 300 s at a congesting demand vehicles must be completing, or the
            # throughput term in the reward is measuring nothing.
            assert env.arrived_total > 0, "no vehicle completed its route in 300 steps"
            assert info["metrics"]["throughput_recent"] >= 0
        finally:
            env.close()


class TestTheTwoVehicleTypesDifferOnlyInIdentity:
    """Penetration must not change the fleet, only who is controllable.

    An earlier version gave the human type a speedFactor spread and the AV type
    none, so it defaulted to 1.0. Raising penetration then made the fleet more
    homogeneous, which reduces speed variance and congestion: with no controller
    acting at all, mean speed went 6.15 m/s at 10% penetration to 11.29 at 40% and
    arrivals 169 to 224. Reported as a control effect, that would have been wrong.
    """

    def test_both_types_carry_the_same_speed_factor(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            routes = (env.work_dir / "demand.rou.xml").read_text()
            # The DISTRIBUTION members, which is what the flows draw from. An
            # earlier version of this test excluded every line containing
            # `probability` -- that is, exactly those two -- and checked the two
            # top-level vTypes that no flow references. The test written to stop
            # the speedFactor confound returning could not fail.
            types = [line for line in routes.splitlines()
                     if "<vType " in line and "probability" in line]
            assert len(types) == 2, f"expected two distribution members, got {types}"
            factors = [line.split('speedFactor="')[1].split('"')[0] for line in types]
            assert factors[0] == factors[1], (
                f"the two vTypes have different speed factors: {factors}"
            )
            maxima = [line.split('maxSpeed="')[1].split('"')[0] for line in types]
            assert maxima[0] == maxima[1], maxima
        finally:
            env.close()

    def test_raising_penetration_alone_does_not_change_flow(self, tmp_path):
        # The measurement the confound would have corrupted: with no actions
        # applied, penetration must not move mean speed much, because an
        # uncontrolled AV is just a vehicle.
        results = {}
        for pen in (0.10, 0.40):
            demand = dict(load_named_config("demand", "sumo_saturating"))
            demand["av_penetration"] = pen
            env = _env(tmp_path / f"p{int(pen*100)}", demand=demand, duration_steps=600)
            try:
                env.reset(seed=7)
                terminated = truncated = False
                info = {}
                while not (terminated or truncated):
                    _, _, terminated, truncated, info = env.step({})
                results[pen] = info["metrics"]["mean_speed"]
            finally:
                env.close()
        low, high = results[0.10], results[0.40]
        assert abs(high - low) < 0.5 * max(low, 1.0), (
            f"penetration alone moved mean speed {low:.2f} -> {high:.2f} m/s, so the "
            "two vehicle types still differ in something other than controllability"
        )


class TestTheDemandLastsTheWholeEpisode:
    """The flow must cover the warm-up as well as the episode.

    It previously ended at `duration_steps * dt`, while the simulation runs
    `warmup + duration` steps, so demand stopped before the episode did. With a
    300-step warm-up and a 600-step episode that left a third of the run draining,
    and the final step reported mean_speed 0.000 on an empty network.
    """

    def test_vehicles_are_still_present_at_the_final_step(self, tmp_path):
        env = _env(tmp_path, duration_steps=600, warmup_steps=300)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            assert info["metrics"]["active_vehicle_count"] > 0, (
                "the network was empty at the final step, so demand ran out early"
            )
            assert info["metrics"]["mean_speed"] > 0.0
        finally:
            env.close()

    def test_the_flow_end_covers_warmup_plus_duration(self, tmp_path):
        env = _env(tmp_path, duration_steps=600, warmup_steps=300)
        try:
            env.reset(seed=7)
            routes = (env.work_dir / "demand.rou.xml").read_text()
            ends = {line.split('end="')[1].split('"')[0]
                    for line in routes.splitlines() if "<flow" in line}
            assert ends == {"900.0"}, f"flow end is {ends}, expected 900.0"
        finally:
            env.close()
