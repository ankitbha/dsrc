"""The trainer building a SUMO environment.

One flag selects the simulator, so a training config says which road its policy was
learned on. That matters because the two simulators produce different observation
distributions -- SUMO's road is collision-free and junction-limited, the old one was
neither -- and a checkpoint carries no record of which it saw otherwise.

Defaulting to `highway_env` keeps every existing config and its results unchanged;
the SUMO configs opt in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rl.ppo import PPOConfig
from src.rl.trainers import TrainingConfig, make_trainer


def _config(**overrides):
    base = {
        "algorithm": "mappo",
        "topology": "inverted_tree",
        "demand": "sumo_saturating",
        "duration_steps": 120,
        # Without a warm-up a 64-step rollout starting at t = 0 sees an empty
        # network and collects zero transitions, which is how the warm-up was
        # found.
        "warmup_steps": 300,
    }
    base.update(overrides)
    return base


class TestTheSimulatorIsSelectable:

    def test_it_defaults_to_highway_env(self):
        assert TrainingConfig.from_mapping(_config()).simulator == "highway_env"

    def test_a_config_can_ask_for_sumo(self):
        assert TrainingConfig.from_mapping(_config(simulator="sumo")).simulator == "sumo"

    def test_an_unknown_simulator_is_refused_by_name(self):
        trainer = make_trainer(
            TrainingConfig.from_mapping(_config(simulator="carla")),
            PPOConfig.from_mapping({}),
        )
        with pytest.raises(ValueError, match="carla"):
            trainer.build_env()

    def test_the_sumo_trainer_builds_a_sumo_env(self, tmp_path):
        from src.sumo.env import SumoTopologyEnv

        trainer = make_trainer(
            TrainingConfig.from_mapping(_config(simulator="sumo", work_dir=str(tmp_path))),
            PPOConfig.from_mapping({}),
        )
        env = trainer.build_env()
        try:
            assert isinstance(env, SumoTopologyEnv)
        finally:
            env.close()

    def test_the_default_trainer_still_builds_the_old_env(self):
        from src.envs.topology_env import HighwayTopologyEnv

        trainer = make_trainer(
            TrainingConfig.from_mapping(_config()), PPOConfig.from_mapping({})
        )
        assert isinstance(trainer.build_env(), HighwayTopologyEnv)


class TestARolloutRunsOnSumo:

    def test_a_rollout_collects_transitions_without_collisions(self, tmp_path):
        # `collect_rollout` rather than `train`, so the assertion is about the
        # environment and the transitions rather than about checkpoint writing.
        trainer = make_trainer(
            TrainingConfig.from_mapping(_config(
                simulator="sumo", work_dir=str(tmp_path), duration_steps=400,
                total_updates=1,
            )),
            PPOConfig.from_mapping({"rollout_steps": 64}),
        )
        buffer, metrics = trainer.collect_rollout(seed=7)
        assert len(buffer) > 0, "the rollout collected no transitions"
        assert float(metrics.get("collision_count", 0) or 0) == 0.0, (
            "SUMO reported a collision during a rollout, which contradicts the "
            "premise of the migration"
        )
        assert float(metrics.get("mean_speed", 0) or 0) > 0.0

    def test_a_full_update_writes_a_checkpoint(self, tmp_path):
        trainer = make_trainer(
            TrainingConfig.from_mapping(_config(
                simulator="sumo", work_dir=str(tmp_path / "sim"), duration_steps=400,
                total_updates=1, output_root=str(tmp_path / "out"),
            )),
            PPOConfig.from_mapping({"rollout_steps": 64}),
        )
        result = trainer.train()
        assert result["updates"] == 1
        assert (Path(result["output_dir"]) / "actor.pt").exists()


class TestTheCrashPenaltyPathRuns:
    """The trainer's crash-penalty branch, which a rollout without a penalty skips.

    My first rollout test passed while this path was never entered, because its
    PPOConfig had no `crash_penalty`. The branch reached into `env._av_vehicles`,
    which exists on only one of the two simulators, so it failed the moment the
    real config was used.
    """

    def test_a_rollout_with_a_crash_penalty_runs_on_sumo(self, tmp_path):
        trainer = make_trainer(
            TrainingConfig.from_mapping(_config(
                simulator="sumo", work_dir=str(tmp_path), duration_steps=400,
                total_updates=1,
            )),
            PPOConfig.from_mapping({"rollout_steps": 64, "crash_penalty": 5.0}),
        )
        buffer, metrics = trainer.collect_rollout(seed=7)
        assert len(buffer) > 0
        assert float(metrics.get("collision_count", 0) or 0) == 0.0

    def test_both_simulators_answer_the_question(self, tmp_path):
        from src.envs.topology_env import HighwayTopologyEnv
        from src.sumo.env import SumoTopologyEnv

        for simulator, expected in (("highway_env", HighwayTopologyEnv),
                                    ("sumo", SumoTopologyEnv)):
            trainer = make_trainer(
                TrainingConfig.from_mapping(_config(
                    simulator=simulator, work_dir=str(tmp_path / simulator))),
                PPOConfig.from_mapping({}),
            )
            env = trainer.build_env()
            try:
                assert isinstance(env, expected)
                assert env.crashed_agent_ids() == []
            finally:
                if hasattr(env, "close"):
                    env.close()
