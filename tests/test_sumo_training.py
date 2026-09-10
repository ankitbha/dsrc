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


class TestTheTrainingConfigsEpisodeLengthIsDeclared:
    """`mappo_sumo.yaml` declared no `duration_steps` and took the trainer's default
    of 120. The effect the configuration exists to learn does not exist at that
    length: AVs holding 10 m/s produce 19.7 arrivals against 22.0 uncommanded over
    120 steps, and 143.0 against 111.8 over 600. The reward ranking inverts with it,
    so training at the default would have optimised against the effect under study.
    """

    def test_the_sumo_config_declares_an_episode_long_enough_to_show_the_effect(self):
        from src.config.loaders import load_named_config

        config = load_named_config("training", "mappo_sumo")
        assert "duration_steps" in config, (
            "mappo_sumo.yaml does not declare duration_steps, so it silently takes "
            "the trainer's 120-step default"
        )
        assert int(config["duration_steps"]) >= 300, (
            f"{config['duration_steps']}-step episodes are too short: the throughput "
            "effect is absent at 120 steps and present at 300 and 600"
        )

    def test_the_declared_length_reaches_the_environment(self, tmp_path):
        # The control on the assertion above: a declared value that the trainer
        # ignores would leave the episode at 120 whatever the config says.
        from src.config.loaders import load_named_config

        config = dict(load_named_config("training", "mappo_sumo"))
        config["duration_steps"] = 7
        config["work_dir"] = str(tmp_path)
        config["warmup_steps"] = 0
        trainer = make_trainer(
            TrainingConfig.from_mapping(config), PPOConfig.from_mapping({}))
        assert trainer.config.duration_steps == 7
        env = trainer.build_env()
        try:
            env.reset(seed=1)
            truncated = [env.step({})[3] for _ in range(7)]
            assert truncated == [False] * 6 + [True], truncated
        finally:
            env.close()


class TestTheDeploymentSensingModelIsActive:
    """The paper's claim is that MAPPO works under the sensing model measured on
    the deployment, so the noise has to actually reach the observations. A
    defaulted field the constructor never passes reads as a measured zero.
    """

    def _observations(self, tmp_path, sensing):
        from src.config.loaders import load_named_config
        from src.sumo.env import SumoTopologyEnv

        env = SumoTopologyEnv("inverted_tree", {
            "topology": load_named_config("topology", "inverted_tree"),
            "demand": load_named_config("demand", "sumo_saturating"),
            "human_model": load_named_config("human_model", "w99_calibrated"),
            "duration_steps": 60, "dt": 1.0, "warmup_steps": 300,
            "sensing": sensing, "work_dir": str(tmp_path)})
        env.reset(seed=7)
        try:
            seen = {}
            for step in range(60):
                observations, _, _, _, _ = env.step({})
                for agent_id, observation in observations.items():
                    gap = float(observation["leader_gap"])
                    if gap < 1e6:
                        seen[(step, agent_id)] = gap
            return seen
        finally:
            env.close()

    def test_the_configs_noise_reaches_the_observations(self, tmp_path):
        import statistics

        from src.config.loaders import load_named_config

        sensing = dict(load_named_config("training", "mappo_sumo")["sensing"])
        assert sensing["position_noise_std"] > 0.0
        assert sensing["speed_noise_std"] > 0.0

        clean = self._observations(tmp_path, {**sensing, "position_noise_std": 0.0,
                                              "speed_noise_std": 0.0})
        noisy = self._observations(tmp_path, sensing)
        shared = set(clean) & set(noisy)
        assert len(shared) > 100, f"only {len(shared)} paired observations"
        error_sd = statistics.stdev(noisy[k] - clean[k] for k in shared)
        # The observed spread is a little below the configured value because a gap
        # is clipped at zero, so this checks the order rather than the exact figure.
        assert error_sd > sensing["position_noise_std"] / 2, (
            f"the configured {sensing['position_noise_std']} m of position noise "
            f"produced an observed spread of {error_sd:.2f} m"
        )


class TestTheCommandLineDoesNotOverrideTheConfig:
    """`train_policy.py` built its `env` block from argparse defaults, and
    `TrainingConfig.from_mapping` prefers that block over everything else, so the
    literal defaults -- topology "ring", demand "medium", duration_steps 120 --
    silently replaced whatever the training config declared. A run of
    `mappo_sumo`, which declares 9000-step episodes at dt 0.1 on `sumo_burst`,
    resolved to 120 steps at dt 1.0 on `medium`: the wrong scenario, the wrong
    episode length, and the step size the throughput result was retracted at.
    """

    def _namespace(self, **overrides):
        import argparse

        base = dict(training="mappo_sumo", topology=None, demand=None,
                    human_model=None, seed=7, total_updates=None, rollout_steps=None,
                    duration_steps=None, dt=None, controlled_vehicles=None,
                    initial_human_vehicles=None, output_root="outputs/checkpoints")
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_the_configs_declarations_survive_an_unqualified_run(self):
        from scripts.train_policy import load_training_bundle

        config = TrainingConfig.from_mapping(load_training_bundle(self._namespace()))
        assert config.dt == 0.1
        assert config.duration_steps == 6000
        assert config.warmup_steps == 3000
        # ONE DECISION PER SIMULATED SECOND, so `rollout_steps` counts decisions and
        # `duration_steps` counts simulation steps, and the two are not comparable
        # without it.
        assert config.decision_interval_s == 1.0
        decisions_per_episode = config.duration_steps / (config.decision_interval_s / config.dt)
        # Three episodes per update. One episode per update was 1/50th of the
        # gradient quality Flow's benchmarks use, and the run that used it never
        # left its initialisation.
        assert config.rollout_steps == 3 * decisions_per_episode
        assert config.topology == "inverted_tree"
        # A steady demand above capacity, at the rate whose collapse happens inside
        # the episode rather than during the warm-up.
        assert config.demand == "sumo_capacity_drop"
        # Half of each agent's reward comes from its own neighbourhood, which is the
        # change task 103 exists to test.
        assert config.local_reward_weights
        # The calibrated driving model, without which the fundamental diagram
        # has no capacity drop and there is nothing to recover.
        assert config.human_model == "w99_calibrated"
        # speed_headway, not full: the lane and merge heads act through an
        # uncalibrated lane-change model, and dropping them cuts the action space
        # from 81 combinations to 9.
        assert config.action_profile == "speed_headway"

    def test_an_explicit_argument_still_overrides(self):
        # The control: the fix must not have made the command line inert.
        from scripts.train_policy import load_training_bundle

        config = TrainingConfig.from_mapping(load_training_bundle(
            self._namespace(dt=0.5, duration_steps=300, demand="sumo_saturating")))
        assert config.dt == 0.5
        assert config.duration_steps == 300
        assert config.demand == "sumo_saturating"


class TestTheCheckpointIsSelectedOnTheObjective:
    """`actor.pt` was selected on `mean_speed - jam_fraction`, which is neither the
    reward the policy maximises nor a monotone function of it. A recorded run whose
    objective improved 30% showed a score change of -0.067, so the saved policy was
    chosen by a quantity nothing was optimising.
    """

    def test_the_score_is_the_team_reward(self):
        from src.rl.rewards import build_team_reward

        weights = {"throughput_recent": 0.10, "jam_fraction": -2.0}
        # Two rollouts, ordered oppositely by the old proxy and the objective: the
        # first has the higher mean speed, the second far more throughput.
        slow_but_productive = {"mean_speed": 5.0, "jam_fraction": 0.4,
                               "throughput_recent": 40.0}
        fast_but_empty = {"mean_speed": 20.0, "jam_fraction": 0.0,
                          "throughput_recent": 1.0}
        proxy = lambda m: m["mean_speed"] - m["jam_fraction"]  # noqa: E731
        assert proxy(fast_but_empty) > proxy(slow_but_productive)
        assert (build_team_reward(slow_but_productive, weights)
                > build_team_reward(fast_but_empty, weights)), (
            "the objective must rank these the other way, or the test proves nothing"
        )

    def test_a_short_run_reports_the_objective_as_its_score(self, tmp_path):
        from src.config.loaders import load_named_config
        from src.rl.ppo import PPOConfig

        config = dict(load_named_config("training", "mappo_sumo"))
        config["duration_steps"] = 60
        config["warmup_steps"] = 300
        config["rollout_steps"] = 60
        config["total_updates"] = 1
        config["work_dir"] = str(tmp_path / "sumo")
        config["output_root"] = str(tmp_path / "out")
        trainer = make_trainer(TrainingConfig.from_mapping(config),
                               PPOConfig.from_mapping(config))
        result = trainer.train()
        rows = list((Path(result["output_dir"]) / "training_metrics.csv").read_text().splitlines())
        assert len(rows) >= 2
        header = rows[0].split(",")
        values = dict(zip(header, rows[1].split(",")))
        from src.rl.rewards import build_team_reward

        metrics = {k: float(v) for k, v in values.items()
                   if k not in ("update", "score") and _is_number(v)}
        assert float(values["score"]) == pytest.approx(
            build_team_reward(metrics, trainer.config.reward_weights), abs=1e-6)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
