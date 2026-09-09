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
