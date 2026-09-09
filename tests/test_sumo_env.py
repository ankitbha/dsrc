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
        # The first version asserted `collision_checks == 20`, but that counter is
        # incremented by a separate statement, so replacing the collision read with
        # `+= 0` left it passing: the guard on this migration's central premise was
        # defeated by a one-token change. This drives the count from SUMO's side.
        #
        # It also pins the counting unit. `--collision.action warn` leaves colliding
        # vehicles on the network, so adding up how many are colliding each step
        # counted vehicle-steps rather than events: the sequence below has four
        # vehicles enter a collision and would read 8 when summed.
        import src.sumo.env as env_module

        env = _env(tmp_path, duration_steps=10)
        env.reset(seed=7)
        try:
            reported = [
                (),                       # nothing
                ("v1", "v2"),             # two vehicles enter a collision: 2 events
                ("v1", "v2"),             # the same collision persists: 0 events
                ("v1", "v2", "v3"),       # a third joins: 1 event
                (),                       # cleared
                ("v1",),                  # v1 collides again: 1 event
                (), (), (), (),
            ]
            calls = iter(reported)
            monkeypatch.setattr(
                env_module._sumo.simulation, "getCollidingVehiclesIDList",
                lambda: next(calls, ()),
            )
            per_step = []
            for _ in range(10):
                env.step({})
                per_step.append(env.new_collisions_last_step)
            assert per_step == [0, 2, 0, 1, 0, 1, 0, 0, 0, 0], per_step
            assert env.collision_count == 4, (
                f"four vehicles entered a collision and the env counted "
                f"{env.collision_count}; summing the per-step totals would give 8"
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


class TestGeneratedFilesStayOutOfTheRepository:
    """`work_dir` defaulted to `"."`.

    Any caller that did not set it wrote the generated network and demand files
    into its working directory, and a `sumo_inverted_tree/` directory appeared in
    the repository root. Generated artefacts do not belong in the repo.
    """

    def test_the_default_work_dir_is_not_the_working_directory(self):
        from pathlib import Path

        from src.config.loaders import load_named_config

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "medium"),
            "duration_steps": 5, "dt": 1.0,
        })
        assert Path.cwd() not in env.work_dir.parents, (
            f"generated files would land under the working directory: {env.work_dir}"
        )

    def test_the_default_work_dir_is_private_to_this_process(self):
        # A fixed shared path let two processes running the same topology -- an
        # evaluation sweep, or a pytest-xdist worker -- overwrite each other's
        # demand file between the write and the read, so a run could silently use
        # another run's penetration or duration.
        import os

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating")})
        assert str(os.getpid()) in str(env.work_dir), (
            f"the default work_dir is shared between processes: {env.work_dir}"
        )

    def test_an_explicit_work_dir_is_honoured(self, tmp_path):
        # The control: the default must not be ignoring the setting entirely.
        env = _env(tmp_path)
        assert tmp_path in env.work_dir.parents or env.work_dir.parent == tmp_path


class TestTheProcessLockIsReleasedOnFailure:
    """`_LIVE` was set after the simulation started and cleared only by `close`, so
    an exception between the two left it set for the life of the process and every
    later `reset` raised. One failing test cascaded into every SUMO test after it.
    """

    def _config(self, tmp_path):
        return {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 5, "dt": 1.0, "warmup_steps": 0,
            "work_dir": str(tmp_path),
        }

    def test_a_reset_that_raises_does_not_lock_the_process(self, tmp_path):
        from src.sumo import env as env_module

        failing = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
        failing._warm_up = lambda: (_ for _ in ()).throw(RuntimeError("simulated"))
        with pytest.raises(RuntimeError, match="simulated"):
            failing.reset(seed=1)
        assert env_module._LIVE is None, "the failed env still holds the connection"

        survivor = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
        try:
            survivor.reset(seed=1)
            survivor.step({})
        finally:
            survivor.close()

    def test_a_dropped_env_does_not_block_the_next_one(self, tmp_path):
        from src.sumo import env as env_module

        dropped = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
        dropped.reset(seed=1)
        dropped.close()
        # A closed env that is still referenced by the marker holds no connection.
        env_module._LIVE = dropped
        successor = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
        try:
            successor.reset(seed=1)
            assert env_module._LIVE is successor
        finally:
            successor.close()

    def test_a_live_env_still_refuses_a_second_one(self, tmp_path):
        # The control: the guard must still do the job it exists for.
        first = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
        first.reset(seed=1)
        try:
            second = SumoTopologyEnv("inverted_tree", self._config(tmp_path))
            with pytest.raises(RuntimeError, match="already holds the SUMO connection"):
                second.reset(seed=1)
        finally:
            first.close()


class TestTheTopologysSafetyBlockReachesTheObservation:
    """`_safety_constraints` reads the topology's declared limits instead of
    passing a bare `SafetyConstraints()`. Nothing could tell the two apart: the
    four values `inverted_tree` declares are numerically equal to the library
    defaults, and the only two fields the SUMO sensing path consumes --
    `uncongested_density_threshold_veh_per_km` and `low_speed_free_flow_delta_mps`
    -- are not declared at all. The plumbing was correct and inert, so a mutant
    reverting it survived. These tests give it a premise that is actually active.
    """

    def _config(self, tmp_path, safety_overrides=None):
        topology = dict(load_named_config("topology", "inverted_tree"))
        if safety_overrides is not None:
            topology["safety"] = {**(topology.get("safety") or {}), **safety_overrides}
        return {
            "topology": topology,
            "demand": load_named_config("demand", "sumo_saturating"),
            "duration_steps": 30, "dt": 1.0, "warmup_steps": 0,
            "work_dir": str(tmp_path),
        }

    def test_a_declared_value_overrides_the_library_default(self, tmp_path):
        env = SumoTopologyEnv("inverted_tree", self._config(
            tmp_path, {"uncongested_density_threshold_veh_per_km": 999.0,
                       "lane_change_dwell_s": 42.0}))
        constraints = env._safety_constraints()
        assert constraints.uncongested_density_threshold_veh_per_km == 999.0
        assert constraints.lane_change_dwell_s == 42.0
        # A field the topology does not declare keeps the library default, so the
        # test above is not satisfied by a constructor that takes everything.
        assert constraints.emergency_decel_mps2 == 6.0

    def test_an_undeclared_key_does_not_reach_the_constraints(self, tmp_path):
        # SafetyConstraints is a dataclass, so an unknown key would raise rather
        # than be ignored; a topology carrying an unrelated safety key must still
        # build.
        env = SumoTopologyEnv("inverted_tree", self._config(
            tmp_path, {"not_a_constraint_field": 1.0}))
        assert env._safety_constraints().lane_change_dwell_s == 15.0

    def test_the_declared_threshold_changes_the_observations(self, tmp_path):
        # End to end: the two constraints the sensing path consumes decide
        # `uncongested_low_speed_flag`. Setting the density threshold to zero makes
        # every segment count as congested, so the flag cannot be raised anywhere.
        def flags(overrides):
            env = SumoTopologyEnv("inverted_tree", self._config(tmp_path, overrides))
            env.reset(seed=11)
            try:
                seen = []
                for _ in range(30):
                    observations, _, _, _, _ = env.step({})
                    seen += [o["uncongested_low_speed_flag"] for o in observations.values()]
                return seen
            finally:
                env.close()

        permissive = flags({"uncongested_density_threshold_veh_per_km": 1000.0,
                            "low_speed_free_flow_delta_mps": 0.1})
        strict = flags({"uncongested_density_threshold_veh_per_km": 0.0})
        assert permissive, "no AV observations produced, so the comparison is empty"
        assert any(permissive), "the permissive threshold raised the flag nowhere"
        assert not any(strict), "the strict threshold still raised the flag"


class TestTheDemandsSpeedDistributionReachesSumo:
    """`_write_routes` read only `max_mps` and hardcoded the desired-speed spread as
    `normc(1,0.1,0.8,1.2)` on the lane limit, so a config declaring a 24 m/s mean
    produced a 30 m/s fleet: the measured operating point belonged to a fleet no
    config described. `mean_mps`, `std_mps` and `min_mps` now map onto the factor.
    """

    def _sampled_speeds(self, tmp_path, mean_mps):
        demand = dict(load_named_config("demand", "sumo_saturating"))
        demand["total_vehicles_per_hour"] = 300.0  # free flow: desired speed is visible
        demand["speed_distribution"] = {**demand["speed_distribution"],
                                        "mean_mps": mean_mps}
        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": demand, "duration_steps": 400, "dt": 1.0,
            "warmup_steps": 100, "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            per_vehicle = {}
            for _ in range(400):
                env.step({})
                for snapshot in env.vehicle_snapshots():
                    per_vehicle[snapshot.vehicle_id] = snapshot.free_flow_speed_mps
            return list(per_vehicle.values())
        finally:
            env.close()

    def test_the_declared_mean_is_what_the_fleet_drives(self, tmp_path):
        speeds = self._sampled_speeds(tmp_path, 24.0)
        assert len(speeds) > 20
        mean = sum(speeds) / len(speeds)
        assert 22.5 < mean < 25.5, (
            f"the config declares a 24.0 m/s mean and the fleet drives {mean:.2f}; "
            "the lane limit is 30.0, so an unmapped distribution reads near that"
        )

    def test_changing_the_declared_mean_moves_the_fleet(self, tmp_path):
        # The control. Under the hardcoded factor both configurations produced the
        # same fleet, so a test of one value alone would pass on a config that is
        # never read.
        fast = self._sampled_speeds(tmp_path, 24.0)
        slow = self._sampled_speeds(tmp_path, 18.0)
        fast_mean = sum(fast) / len(fast)
        slow_mean = sum(slow) / len(slow)
        assert fast_mean - slow_mean > 4.0, (
            f"declaring 24.0 gave {fast_mean:.2f} and declaring 18.0 gave "
            f"{slow_mean:.2f}; the declared mean is not reaching SUMO"
        )
