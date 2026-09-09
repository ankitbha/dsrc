"""The SUMO-backed environment, at the seam the rest of the project already uses.

`VehicleSnapshot` is the boundary between the simulator and everything this project
built: the sensing model, the safety layer, the etiquette filters, the encoders, the
policy and the trainers all consume snapshots and none of them knows what produces
them. So replacing the simulator means producing snapshots from TraCI instead of
from highway_env, and nothing above the seam changes.

The collision assertions are not decoration. A silent assumption that SUMO cannot
collide is exactly the kind of unmeasured premise this project has been caught by
three times, so the count is read every step and checked.
"""
from __future__ import annotations

import pytest

from src.config.loaders import load_named_config
from src.sensing import VehicleSnapshot
from src.sumo.env import SumoTopologyEnv


def _env(tmp_path, **overrides):
    config = {
        "topology": load_named_config("topology", "inverted_tree"),
        "demand": load_named_config("demand", "medium"),
        "duration_steps": 60,
        "dt": 1.0,
        "work_dir": str(tmp_path),
    }
    config.update(overrides)
    return SumoTopologyEnv("inverted_tree", config)


class TestItRunsAndCloses:

    def test_reset_returns_observations_and_step_advances(self, tmp_path):
        env = _env(tmp_path)
        try:
            observations, info = env.reset(seed=7)
            assert isinstance(observations, dict)
            for _ in range(10):
                observations, rewards, terminated, truncated, info = env.step({})
            assert env.step_count == 10
            assert terminated is False
        finally:
            env.close()

    def test_the_episode_truncates_at_its_duration(self, tmp_path):
        env = _env(tmp_path, duration_steps=15)
        try:
            env.reset(seed=7)
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step({})
                steps += 1
                assert steps <= 20, "the episode did not truncate"
            assert truncated is True
            assert terminated is False, "SUMO produced a termination, which means a collision"
            assert steps == 15
        finally:
            env.close()

    def test_two_envs_do_not_share_a_connection(self, tmp_path):
        # libsumo is process-global, so two live envs would silently drive one
        # simulation. Whatever the implementation does about that, it must not
        # produce a network with vehicles from both.
        first = _env(tmp_path / "a")
        first.reset(seed=7)
        for _ in range(5):
            first.step({})
        first_count = len(first.vehicle_snapshots())
        first.close()
        second = _env(tmp_path / "b")
        second.reset(seed=7)
        try:
            assert len(second.vehicle_snapshots()) <= first_count + 5
        finally:
            second.close()


class TestSnapshotsAreUsableBySensing:

    def test_snapshots_have_every_field_the_sensing_model_reads(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            for _ in range(30):
                env.step({})
            snapshots = env.vehicle_snapshots()
            assert snapshots, "no vehicles after 30 steps; the demand did not insert any"
            for snapshot in snapshots:
                assert isinstance(snapshot, VehicleSnapshot)
                assert snapshot.vehicle_id
                assert snapshot.role in {"av", "human"}
                assert snapshot.segment_id is not None
                assert snapshot.lane_index is not None
                assert snapshot.longitudinal_m >= 0.0
                assert snapshot.speed_mps >= 0.0
                assert snapshot.free_flow_speed_mps > 0.0
                assert snapshot.crashed is False
        finally:
            env.close()

    def test_longitudinal_position_advances_along_a_lane(self, tmp_path):
        env = _env(tmp_path)
        try:
            env.reset(seed=7)
            for _ in range(10):
                env.step({})
            first = {s.vehicle_id: s for s in env.vehicle_snapshots()}
            for _ in range(5):
                env.step({})
            second = {s.vehicle_id: s for s in env.vehicle_snapshots()}
            moved = [
                vid for vid in first.keys() & second.keys()
                if second[vid].lane_index == first[vid].lane_index
                and second[vid].longitudinal_m > first[vid].longitudinal_m
            ]
            assert moved, "no vehicle advanced along its own lane over 5 steps"
        finally:
            env.close()


class TestNoCollisionsEver:

    def test_a_full_run_records_zero_collisions(self, tmp_path):
        env = _env(tmp_path, duration_steps=120)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step({})
            assert env.collision_count == 0, (
                f"SUMO reported {env.collision_count} collisions, which contradicts "
                "the premise this migration rests on"
            )
        finally:
            env.close()

    def test_the_collision_counter_is_actually_read(self, tmp_path):
        # The control. A counter that is never incremented reads zero whatever
        # happens, so assert the env queried SUMO rather than defaulting.
        env = _env(tmp_path, duration_steps=20)
        try:
            env.reset(seed=7)
            for _ in range(20):
                env.step({})
            assert env.collision_checks == 20, (
                f"the collision count was read {env.collision_checks} times in 20 steps"
            )
        finally:
            env.close()
