from __future__ import annotations

import pytest
import torch

from src.envs.wrappers import validate_action_mapping
from src.rl.actions import ActionSpec, action_to_indices, indices_to_action
from src.rl.controller import LearnedPolicyController
from src.rl.encoders import (
    encode_global_state,
    encode_local_observation,
    encode_physical_global_state,
    global_state_dim,
    local_obs_dim,
    physical_global_state_dim,
)
from src.rl.models import GlobalCritic, LocalCritic, MultiCategoricalActor
from src.rl.ppo import PPOConfig, ppo_update
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainers import IPPOTrainer, MAPPOTrainer, SharedPPOTrainer, TrainingConfig, aggregate_rollout_metrics


def local_obs(**overrides):
    obs = {
        "is_active": True,
        "ego_speed": 20.0,
        "ego_acceleration": 0.0,
        "ego_lane": 0,
        "ego_headway_s": 2.0,
        "target_headway_s": 1.6,
        "leader_gap": 50.0,
        "leader_relative_speed": 0.0,
        "local_density_bin": 0,
        "local_mean_speed_bin": 2,
        "local_queue_estimate": 0,
        "nearby_av_count": 0,
        "nearby_av_mean_speed": 30.0,
        "nearby_av_lane_distribution": {"0": 1.0},
        "cooperation": {
            "segment_target_speed": 30.0,
            "merge_pressure": 0.0,
            "downstream_congestion_estimate": 0.0,
        },
    }
    obs.update(overrides)
    return obs


def global_state():
    return {
        "time": 1.0,
        "active_vehicle_count": 2,
        "active_av_count": 1,
        "completed_vehicle_count": 0,
        "segment_state": {
            "seg": {
                "vehicle_count": 2,
                "av_count": 1,
                "mean_speed": 20.0,
                "density": 4.0,
                "queue_length": 0,
                "jam_fraction": 0.0,
            }
        },
        "demand_state": {"current_vehicles_per_hour": 1000.0, "av_penetration": 0.1},
        "branch_state": {"per_branch_spawned": {"main": 2}, "per_branch_completed": {"main": 0}},
        "previous_step_metrics": {"mean_speed": 10.0, "completed_vehicle_count": 0},
    }


def test_encoders_have_stable_dimensions() -> None:
    local = encode_local_observation(local_obs())
    global_encoded = encode_global_state(global_state())
    assert local.shape == (local_obs_dim(),)
    assert global_encoded.shape == (global_state_dim(),)
    assert torch.isfinite(local).all()
    assert torch.isfinite(global_encoded).all()


def test_encoder_bounds_nonfinite_local_observation_values() -> None:
    encoded = encode_local_observation(
        local_obs(
            ego_headway_s=float("inf"),
            time_since_last_lane_change=float("inf"),
            distance_to_downstream_bottleneck=float("inf"),
            leader_gap=float("inf"),
            follower_gap=float("-inf"),
        )
    )

    assert torch.isfinite(encoded).all()
    assert encoded.abs().max() <= 5.0


@pytest.mark.parametrize("profile", ["speed_only", "speed_headway", "full"])
def test_actor_emits_valid_v2_actions_for_profiles(profile: str) -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec(profile))  # type: ignore[arg-type]
    obs = torch.stack([encode_local_observation(local_obs()), encode_local_observation(local_obs(ego_speed=10.0))])
    actions, indices, log_probs, entropies = actor.sample(obs, deterministic=True)
    action_map = {f"av_{index}": action for index, action in enumerate(actions)}
    validate_action_mapping(action_map, expected_agent_ids=action_map.keys())
    assert indices.shape == (2, 4)
    assert log_probs.shape == (2,)
    assert entropies.shape == (2,)
    if profile == "speed_only":
        assert all(action["desired_headway_bin"] == "normal" for action in actions)
        assert all(action["lane_preference"] == "keep" for action in actions)
    if profile == "speed_headway":
        assert all(action["lane_preference"] == "keep" for action in actions)


def test_actor_sampling_and_evaluation_are_consistent_without_pre_filtering() -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec("full"))
    obs = torch.stack([encode_local_observation(local_obs()), encode_local_observation(local_obs(ego_speed=10.0))])

    actions, indices, log_probs, entropies = actor.sample(obs, deterministic=True)
    evaluated_log_probs, evaluated_entropies = actor.evaluate_actions(obs, indices)

    validate_action_mapping({str(index): action for index, action in enumerate(actions)})
    assert torch.allclose(log_probs, evaluated_log_probs)
    assert torch.allclose(entropies, evaluated_entropies)


def test_action_index_round_trip() -> None:
    spec = ActionSpec("full")
    action = {
        "desired_speed_bin": "fast",
        "desired_headway_bin": "largest",
        "lane_preference": "prefer_right_if_safe",
        "merge_mode": "create_gap",
    }
    assert indices_to_action(action_to_indices(action), spec) == action


def test_rollout_buffer_handles_interleaved_agents_and_finite_gae() -> None:
    buffer = RolloutBuffer()
    obs = encode_local_observation(local_obs())
    action = torch.tensor([1, 0, 0, 0])
    for agent_id, reward, value in (
        ("av_0", 1.0, 0.2),
        ("av_1", 0.5, 0.1),
        ("av_0", 1.0, 0.3),
        ("av_1", 0.5, 0.2),
    ):
        buffer.add(
            observation=obs,
            action=action,
            log_prob=torch.tensor(-1.0),
            reward=reward,
            value=torch.tensor(value),
            done=False,
            agent_id=agent_id,
        )
    batch = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    assert batch.observations.shape[0] == 4
    assert torch.isfinite(batch.advantages).all()
    assert torch.isfinite(batch.returns).all()


def test_interleaved_rollout_gae_matches_isolated_agent_trajectories() -> None:
    obs = encode_local_observation(local_obs())
    action = torch.tensor([1, 0, 0, 0])
    interleaved = RolloutBuffer()
    isolated: dict[str, RolloutBuffer] = {"av_0": RolloutBuffer(), "av_1": RolloutBuffer()}
    transitions = (
        ("av_0", 1.0, 0.2),
        ("av_1", 0.5, 0.1),
        ("av_0", 1.5, 0.3),
        ("av_1", 0.7, 0.2),
    )
    for agent_id, reward, value in transitions:
        kwargs = {
            "observation": obs,
            "action": action,
            "log_prob": torch.tensor(-1.0),
            "reward": reward,
            "value": torch.tensor(value),
            "done": False,
            "agent_id": agent_id,
        }
        interleaved.add(**kwargs)
        isolated[agent_id].add(**kwargs)
    bootstrap = {"av_0": 0.4, "av_1": 0.25}
    interleaved.set_bootstrap_values(bootstrap)
    for agent_id, buffer in isolated.items():
        buffer.set_bootstrap_values({agent_id: bootstrap[agent_id]})

    batch = interleaved.compute_returns_and_advantages(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
    av0 = isolated["av_0"].compute_returns_and_advantages(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
    av1 = isolated["av_1"].compute_returns_and_advantages(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)

    assert batch.returns[[0, 2]].tolist() == pytest.approx(av0.returns.tolist())
    assert batch.returns[[1, 3]].tolist() == pytest.approx(av1.returns.tolist())


def test_rollout_buffer_bootstraps_nonterminal_final_transition() -> None:
    buffer = RolloutBuffer()
    obs = encode_local_observation(local_obs())
    action = torch.tensor([1, 0, 0, 0])
    buffer.add(
        observation=obs,
        action=action,
        log_prob=torch.tensor(-1.0),
        reward=1.0,
        value=torch.tensor(0.25),
        done=False,
        agent_id="av_0",
    )
    buffer.set_bootstrap_values({"av_0": 0.75})

    batch = buffer.compute_returns_and_advantages(gamma=0.5, gae_lambda=1.0, normalize_advantages=False)

    assert batch.returns[0] == pytest.approx(1.375)


def test_shared_ppo_uses_shared_advantages_and_bootstrap_key() -> None:
    obs = encode_local_observation(local_obs())
    action = torch.tensor([1, 0, 0, 0])
    buffer = RolloutBuffer()
    for agent_id, reward, value in (
        ("av_0", 1.0, 0.2),
        ("av_1", 0.5, 0.1),
    ):
        buffer.add(
            observation=obs,
            action=action,
            log_prob=torch.tensor(-1.0),
            reward=reward,
            value=torch.tensor(value),
            done=False,
            agent_id=agent_id,
        )
    buffer.set_bootstrap_values({"av_0": 10.0, "av_1": 20.0, "__shared__": 30.0})

    shared = buffer.compute_returns_and_advantages(
        gamma=0.5,
        gae_lambda=1.0,
        normalize_advantages=False,
        group_by_agent=False,
    )
    per_agent = buffer.compute_returns_and_advantages(
        gamma=0.5,
        gae_lambda=1.0,
        normalize_advantages=False,
        group_by_agent=True,
    )

    assert SharedPPOTrainer.advantage_group_by_agent is False
    assert IPPOTrainer.advantage_group_by_agent is True
    assert shared.returns.tolist() != pytest.approx(per_agent.returns.tolist())
    assert shared.returns[-1] == pytest.approx(15.5)


def test_ppo_update_runs_one_minibatch() -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec("speed_only"))
    critic = LocalCritic(local_obs_dim(), hidden_sizes=(16,))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3)
    buffer = RolloutBuffer()
    obs = encode_local_observation(local_obs())
    with torch.no_grad():
        _, action_indices, log_probs, _ = actor.sample(obs.unsqueeze(0))
        value = critic(obs.unsqueeze(0))[0]
    buffer.add(
        observation=obs,
        action=action_indices[0],
        log_prob=log_probs[0],
        reward=1.0,
        value=value,
        done=True,
        agent_id="av_0",
    )
    batch = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95, normalize_advantages=False)
    stats = ppo_update(actor=actor, critic=critic, optimizer=optimizer, batch=batch, config=PPOConfig(update_epochs=1), device=torch.device("cpu"))
    assert "loss" in stats
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_ppo_update_rejects_nonfinite_old_log_probs() -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec("speed_only"))
    critic = LocalCritic(local_obs_dim(), hidden_sizes=(16,))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3)
    obs = encode_local_observation(local_obs())
    batch = RolloutBuffer()
    batch.add(
        observation=obs,
        action=torch.tensor([1, 0, 0, 0]),
        log_prob=torch.tensor(float("-inf")),
        reward=1.0,
        value=torch.tensor(0.0),
        done=True,
        agent_id="av_0",
    )

    with pytest.raises(ValueError, match="non-finite old log"):
        ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            batch=batch.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95, normalize_advantages=False),
            config=PPOConfig(update_epochs=1),
            device=torch.device("cpu"),
        )


def test_ppo_update_clamps_extreme_finite_log_ratios() -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec("speed_only"))
    critic = LocalCritic(local_obs_dim(), hidden_sizes=(16,))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3)
    obs = encode_local_observation(local_obs())
    buffer = RolloutBuffer()
    buffer.add(
        observation=obs,
        action=torch.tensor([1, 0, 0, 0]),
        log_prob=torch.tensor(-1.0e6),
        reward=1.0,
        value=torch.tensor(0.0),
        done=True,
        agent_id="av_0",
    )
    batch = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95, normalize_advantages=False)

    stats = ppo_update(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        batch=batch,
        config=PPOConfig(update_epochs=1),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(list(stats.values()))).all()
    assert all(torch.isfinite(parameter).all().item() for parameter in actor.parameters())
    assert all(torch.isfinite(parameter).all().item() for parameter in critic.parameters())


def test_collect_rollout_fills_horizon_across_short_episodes() -> None:
    trainer = SharedPPOTrainer(
        TrainingConfig(
            algorithm="shared_ppo",
            action_profile="speed_only",
            hidden_sizes=(8,),
            rollout_steps=5,
            duration_steps=2,
            controlled_vehicles=1,
            initial_human_vehicles=0,
            topology="ring",
        ),
        PPOConfig(update_epochs=1, minibatch_size=4),
        device="cpu",
    )

    rollout, metrics = trainer.collect_rollout(seed=17)

    assert len(rollout) == 5
    assert metrics


def test_mappo_critic_uses_global_state_but_controller_rejects_global_state() -> None:
    config = TrainingConfig(algorithm="mappo", action_profile="speed_only", hidden_sizes=(16,))
    trainer = MAPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    local = torch.stack([encode_local_observation(local_obs())])
    value_obs = trainer.value_observation_tensor(global_state(), local, 1)
    assert value_obs.shape == (1, physical_global_state_dim() + local_obs_dim())
    controller = LearnedPolicyController(trainer.actor)
    action = controller.act({"av_0": local_obs()})
    validate_action_mapping(action, expected_agent_ids=["av_0"])
    with pytest.raises(ValueError):
        controller.act({"av_0": local_obs()}, global_state=global_state())


def test_learned_policy_controller_does_not_need_pre_filter_metadata() -> None:
    actor = MultiCategoricalActor(local_obs_dim(), hidden_sizes=(16,), action_spec=ActionSpec("speed_only"))
    controller = LearnedPolicyController(actor)

    actions = controller.act({"av_0": local_obs()})

    validate_action_mapping(actions, expected_agent_ids=["av_0"])
    assert actions["av_0"]["desired_headway_bin"] == "normal"
    assert actions["av_0"]["lane_preference"] == "keep"


def test_physical_global_state_excludes_cumulative_counters() -> None:
    baseline = global_state()
    changed = {
        **baseline,
        "completed_vehicle_count": 999,
        "demand_state": {
            **baseline["demand_state"],
            "spawned_vehicle_count": 100,
            "completed_vehicle_count": 90,
            "skipped_spawn_count": 12,
        },
        "branch_state": {
            "per_branch_spawned": {"main": 100},
            "per_branch_completed": {"main": 90},
            "branch_travel_time_mean": {"main": 42.0},
        },
        "previous_step_metrics": {"mean_speed": 1.0, "completed_vehicle_count": 777},
    }

    assert torch.equal(encode_physical_global_state(baseline), encode_physical_global_state(changed))
    assert not torch.equal(encode_global_state(baseline), encode_global_state(changed))


def test_aggregate_rollout_metrics_averages_numeric_fields_for_score() -> None:
    metrics = aggregate_rollout_metrics(
        [
            {"mean_speed": 10.0, "jam_fraction": 0.2, "branch_throughput": {"main": 1}},
            {"mean_speed": 20.0, "jam_fraction": 0.4, "branch_throughput": {"main": 2}},
        ]
    )

    assert metrics["mean_speed"] == pytest.approx(15.0)
    assert metrics["jam_fraction"] == pytest.approx(0.3)
    assert metrics["branch_throughput"] == {"main": 2}


def test_trainer_state_resume_selects_best_or_latest_and_rejects_incompatible_config(tmp_path) -> None:
    config = TrainingConfig(algorithm="shared_ppo", action_profile="speed_only", hidden_sizes=(16,))
    trainer = SharedPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    best_state = {key: value.detach().clone() for key, value in trainer.actor.state_dict().items()}
    trainer.save_checkpoint(tmp_path, best_score=1.5)
    with torch.no_grad():
        for parameter in trainer.actor.parameters():
            parameter.add_(1.0)
    latest_state = {key: value.detach().clone() for key, value in trainer.actor.state_dict().items()}
    trainer.save_checkpoint(tmp_path, best_score=1.5, actor_filename="latest_actor.pt", critic_filename="latest_critic.pt")
    trainer.save_trainer_state(tmp_path, completed_update=3, best_score=1.5, metrics_rows=[{"update": 3, "score": 1.5}])

    resumed = SharedPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    start_update, best_score, rows = resumed.load_training_state(tmp_path, resume_latest=False)

    assert start_update == 4
    assert best_score == pytest.approx(1.5)
    assert rows == [{"update": 3, "score": 1.5}]
    for key, value in resumed.actor.state_dict().items():
        assert torch.equal(value, best_state[key])
    assert not (tmp_path / ".actor.pt.tmp").exists()

    resumed_latest = SharedPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    resumed_latest.load_training_state(tmp_path, resume_latest=True)
    for key, value in resumed_latest.actor.state_dict().items():
        assert torch.equal(value, latest_state[key])

    incompatible = SharedPPOTrainer(
        TrainingConfig(algorithm="shared_ppo", action_profile="full", hidden_sizes=(16,)),
        PPOConfig(update_epochs=1),
        device="cpu",
    )
    with pytest.raises(ValueError, match="action_profile"):
        incompatible.load_training_state(tmp_path, resume_latest=True)


def test_trainer_state_resume_warns_on_device_mismatch(tmp_path) -> None:
    config = TrainingConfig(algorithm="shared_ppo", action_profile="speed_only", hidden_sizes=(16,))
    trainer = SharedPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    trainer.save_checkpoint(tmp_path, best_score=1.0)
    trainer.save_trainer_state(tmp_path, completed_update=1, best_score=1.0, metrics_rows=[])
    state = torch.load(tmp_path / "trainer_state.pt", map_location="cpu", weights_only=False)
    state["device_type"] = "cuda"
    torch.save(state, tmp_path / "trainer_state.pt")

    with pytest.warns(RuntimeWarning, match="RNG replay may differ"):
        trainer.load_training_state(tmp_path, resume_latest=False)


def test_training_config_sensing_passthrough_reaches_env_config() -> None:
    bundle = {
        "training": {
            "algorithm": "shared_ppo",
            "action_profile": "speed_only",
            "sensing": {"range_m": 100.0, "position_noise_std": 1.5, "speed_noise_std": 0.15},
        },
        "env": {"topology": "ring"},
    }
    config = TrainingConfig.from_mapping(bundle)
    assert config.sensing == {"range_m": 100.0, "position_noise_std": 1.5, "speed_noise_std": 0.15}
    trainer = SharedPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    env_config = trainer.env_config()
    assert env_config["sensing"] == config.sensing

    clean = TrainingConfig.from_mapping({"training": {"algorithm": "shared_ppo"}})
    assert clean.sensing is None
    clean_trainer = SharedPPOTrainer(clean, PPOConfig(update_epochs=1), device="cpu")
    assert "sensing" not in clean_trainer.env_config()


def test_ppo_config_reads_crash_penalty_from_optimization_block() -> None:
    config = PPOConfig.from_mapping({"optimization": {"crash_penalty": 2.0, "entropy_coef": 0.005}})
    assert config.crash_penalty == 2.0
    assert config.entropy_coef == 0.005
    assert PPOConfig.from_mapping({}).crash_penalty == 0.0
