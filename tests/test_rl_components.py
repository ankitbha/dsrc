from __future__ import annotations

import pytest
import torch

from src.envs.wrappers import validate_action_mapping
from src.rl.actions import ActionSpec, action_to_indices, indices_to_action
from src.rl.controller import LearnedPolicyController
from src.rl.encoders import encode_global_state, encode_local_observation, global_state_dim, local_obs_dim
from src.rl.models import GlobalCritic, LocalCritic, MultiCategoricalActor
from src.rl.ppo import PPOConfig, ppo_update
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.trainers import MAPPOTrainer, TrainingConfig


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
    }


def test_encoders_have_stable_dimensions() -> None:
    local = encode_local_observation(local_obs())
    global_encoded = encode_global_state(global_state())
    assert local.shape == (local_obs_dim(),)
    assert global_encoded.shape == (global_state_dim(),)
    assert torch.isfinite(local).all()
    assert torch.isfinite(global_encoded).all()


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


def test_mappo_critic_uses_global_state_but_controller_rejects_global_state() -> None:
    config = TrainingConfig(algorithm="mappo", action_profile="speed_only", hidden_sizes=(16,))
    trainer = MAPPOTrainer(config, PPOConfig(update_epochs=1), device="cpu")
    local = torch.stack([encode_local_observation(local_obs())])
    value_obs = trainer.value_observation_tensor(global_state(), local, 1)
    assert value_obs.shape == (1, global_state_dim())
    controller = LearnedPolicyController(trainer.actor)
    action = controller.act({"av_0": local_obs()})
    validate_action_mapping(action, expected_agent_ids=["av_0"])
    with pytest.raises(ValueError):
        controller.act({"av_0": local_obs()}, global_state=global_state())
