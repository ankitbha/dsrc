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

    def test_a_second_live_env_is_refused(self, tmp_path):
        # The previous version of this test closed the first env before building
        # the second, so it never exercised the path its name described and could
        # not fail. Measured before the fix: A stepped 30 steps, B.reset() rewound
        # the shared clock to 0.0 while A stayed `_running`, and A.step() then
        # advanced B's simulation and returned B's vehicles, with A's own step
        # count, arrivals and collision total continuing across both. Nothing
        # raised.
        first = _env(tmp_path / "a")
        first.reset(seed=7)
        try:
            for _ in range(5):
                first.step({})
            second = _env(tmp_path / "b")
            with pytest.raises(RuntimeError, match="process-global"):
                second.reset(seed=7)
            # And the first env is untouched by the refusal.
            before = first.step_count
            first.step({})
            assert first.step_count == before + 1
        finally:
            first.close()

    def test_the_connection_is_released_on_close(self, tmp_path):
        # The control: after closing, a second env must be able to start. Without
        # it the refusal above could be satisfied by never releasing at all.
        first = _env(tmp_path / "a")
        first.reset(seed=7)
        for _ in range(5):
            first.step({})
        first.close()
        second = _env(tmp_path / "b")
        second.reset(seed=7)
        try:
            second.step({})
            assert second.step_count == 1
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

    def test_the_collision_count_reflects_what_sumo_reports(self, tmp_path, monkeypatch):
        # The previous version asserted `collision_checks == 20`, but that counter
        # is incremented by a separate statement, so replacing the collision read
        # with `+= 0` left it passing. The guard on this migration's central
        # premise was defeated by a one-token change.
        #
        # This drives the count from SUMO's side instead: if the env stops reading
        # the reported number, the assertion fails.
        import src.sumo.env as env_module

        env = _env(tmp_path, duration_steps=10)
        env.reset(seed=7)
        try:
            reported = [0, 3, 0, 2, 0, 0, 1, 0, 0, 0]
            calls = iter(reported)
            monkeypatch.setattr(
                env_module._sumo.simulation, "getCollidingVehiclesNumber",
                lambda: next(calls, 0),
            )
            for _ in range(10):
                env.step({})
            assert env.collision_count == sum(reported), (
                f"SUMO reported {sum(reported)} collisions and the env counted "
                f"{env.collision_count}"
            )
            assert env.collision_checks == 10
        finally:
            env.close()

    def test_collisions_are_reported_not_removed(self, tmp_path):
        # `--collision.action remove` would delete colliding vehicles and the count
        # would read 0, making every "zero collisions" claim in this migration
        # vacuous. Pin the option.
        env = _env(tmp_path, duration_steps=5)
        env.reset(seed=7)
        try:
            assert env.collision_action == "warn", (
                f"collisions are handled with {env.collision_action!r}; anything that "
                "removes the vehicles makes the collision count meaningless"
            )
        finally:
            env.close()
