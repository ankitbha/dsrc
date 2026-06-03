from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.config.loaders import load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger
from src.rl.actions import ActionSpec
from src.rl.encoders import encode_global_state, encode_local_batch, global_state_dim, local_obs_dim
from src.rl.models import GlobalCritic, LocalCritic, MultiCategoricalActor
from src.rl.ppo import PPOConfig, ppo_update
from src.rl.rewards import build_team_reward, safety_penalty_for_agent
from src.rl.rollout_buffer import RolloutBuffer


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "shared_ppo"
    action_profile: str = "speed_only"
    total_updates: int = 1
    rollout_steps: int = 32
    seed: int = 7
    topology: str = "ring"
    demand: str = "medium"
    human_model: str = "normal"
    controlled_vehicles: int = 2
    initial_human_vehicles: int = 12
    duration_steps: int = 120
    dt: float = 1.0
    output_root: str = "outputs/checkpoints"
    hidden_sizes: tuple[int, ...] = (128, 128)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TrainingConfig:
        training = dict(config.get("training", config))
        env = dict(config.get("env", {}))
        actor_cfg = dict(training.get("actor", {})) if isinstance(training.get("actor", {}), Mapping) else {}
        hidden = actor_cfg.get("hidden_sizes", (128, 128))
        return cls(
            algorithm=str(training.get("algorithm", "shared_ppo")),
            action_profile=str(training.get("action_profile", actor_cfg.get("action_profile", "speed_only"))),
            total_updates=int(training.get("total_updates", 1)),
            rollout_steps=int(training.get("rollout_steps", 32)),
            seed=int(config.get("seed", training.get("seed", 7))),
            topology=str(env.get("topology", config.get("topology", "ring"))),
            demand=str(env.get("demand", config.get("demand", "medium"))),
            human_model=str(env.get("human_model", config.get("human_model", "normal"))),
            controlled_vehicles=int(env.get("controlled_vehicles", config.get("controlled_vehicles", 2))),
            initial_human_vehicles=int(env.get("initial_human_vehicles", config.get("initial_human_vehicles", 12))),
            duration_steps=int(env.get("duration_steps", config.get("duration_steps", 120))),
            dt=float(env.get("dt", config.get("dt", 1.0))),
            output_root=str(config.get("output_root", training.get("output_root", "outputs/checkpoints"))),
            hidden_sizes=tuple(int(value) for value in hidden),
        )


class BasePPOTrainer:
    critic_scope = "local"
    advantage_group_by_agent = True

    def __init__(
        self,
        config: TrainingConfig,
        ppo_config: PPOConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.ppo_config = ppo_config
        self.device = torch.device(device)
        self.action_spec = ActionSpec(config.action_profile)  # type: ignore[arg-type]
        self.actor = MultiCategoricalActor(
            local_obs_dim(),
            hidden_sizes=config.hidden_sizes,
            action_spec=self.action_spec,
        ).to(self.device)
        critic_input_dim = global_state_dim() if self.critic_scope == "global" else local_obs_dim()
        critic_cls = GlobalCritic if self.critic_scope == "global" else LocalCritic
        self.critic = critic_cls(critic_input_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=ppo_config.learning_rate,
        )

    @property
    def experiment_id(self) -> str:
        return f"{self.config.algorithm}_{self.config.topology}_{self.config.action_profile}_seed{self.config.seed}"

    def train(self) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        output_dir = Path(self.config.output_root) / self.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "training_metrics.csv"
        rows: list[dict[str, Any]] = []
        best_score = float("-inf")
        for update in range(1, self.config.total_updates + 1):
            rollout, episode_metrics = self.collect_rollout(seed=self.config.seed + update)
            if len(rollout) == 0:
                raise RuntimeError("rollout collected no active AV transitions")
            batch = rollout.compute_returns_and_advantages(
                gamma=self.ppo_config.gamma,
                gae_lambda=self.ppo_config.gae_lambda,
                group_by_agent=self.advantage_group_by_agent,
            )
            stats = ppo_update(
                actor=self.actor,
                critic=self.critic,
                optimizer=self.optimizer,
                batch=batch,
                config=self.ppo_config,
                device=self.device,
            )
            score = float(episode_metrics.get("mean_speed", 0.0)) - float(episode_metrics.get("jam_fraction", 0.0))
            best_score = max(best_score, score)
            row = {"update": update, "score": score, **stats, **episode_metrics}
            rows.append(row)
            self.save_checkpoint(output_dir, best_score=best_score)
            write_training_metrics(metrics_path, rows)
        write_resolved_config(output_dir / "config_resolved.yaml", self.config, self.ppo_config)
        return {"output_dir": str(output_dir), "updates": self.config.total_updates, "best_score": best_score}

    def collect_rollout(self, *, seed: int) -> tuple[RolloutBuffer, dict[str, Any]]:
        env = HighwayTopologyEnv(self.config.topology, self.env_config())
        observations, _ = env.reset(seed=seed)
        buffer = RolloutBuffer()
        episode_metrics: dict[str, Any] = {}
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < self.config.rollout_steps:
            agent_ids, obs_tensor = encode_local_batch(observations)
            if not agent_ids:
                observations, _, terminated, truncated, info = env.step({})
                episode_metrics = dict(info.get("metrics", {}))
                steps += 1
                continue
            obs_tensor = obs_tensor.to(self.device)
            with torch.no_grad():
                actions, action_indices, log_probs, _ = self.actor.sample(obs_tensor)
                value_obs = self.value_observation_tensor(env.get_global_state(), obs_tensor, len(agent_ids))
                values = self.critic(value_obs)
            action_map = {agent_id: action for agent_id, action in zip(agent_ids, actions, strict=True)}
            next_observations, _, terminated, truncated, info = env.step(action_map)
            team_reward = build_team_reward(info.get("metrics", {})) * self.ppo_config.reward_scale
            done = bool(terminated or truncated)
            for index, agent_id in enumerate(agent_ids):
                reward = team_reward - safety_penalty_for_agent(info, agent_id)
                reward = max(-self.ppo_config.reward_clip, min(self.ppo_config.reward_clip, reward))
                buffer.add(
                    observation=obs_tensor[index],
                    action=action_indices[index],
                    log_prob=log_probs[index],
                    reward=reward,
                    value=values[index],
                    done=done or agent_id not in next_observations,
                    value_observation=value_obs[index],
                    agent_id=agent_id,
                )
            observations = next_observations
            episode_metrics = dict(info.get("metrics", {}))
            steps += 1
        return buffer, episode_metrics

    def value_observation_tensor(self, global_state: Mapping[str, Any], obs_tensor: torch.Tensor, agent_count: int) -> torch.Tensor:
        return obs_tensor

    def env_config(self) -> dict[str, Any]:
        topology_cfg = load_named_config("topology", self.config.topology)
        demand_cfg = load_named_config("demand", self.config.demand)
        human_cfg = load_named_config("human_model", self.config.human_model)
        return {
            "topology": topology_cfg,
            "demand": demand_cfg,
            "human_model": human_cfg,
            "controller": {"name": self.config.algorithm, "family": "rl", "safety_mode": "integrated_rl"},
            "controlled_vehicles": self.config.controlled_vehicles,
            "initial_human_vehicles": self.config.initial_human_vehicles if self.config.topology == "ring" else 0,
            "duration_steps": self.config.duration_steps,
            "dt": self.config.dt,
        }

    def save_checkpoint(self, output_dir: Path, *, best_score: float) -> None:
        actor_payload = {
            "state_dict": self.actor.state_dict(),
            "metadata": self.actor.checkpoint_metadata(),
            "hidden_sizes": self.config.hidden_sizes,
            "best_score": best_score,
        }
        critic_payload = {
            "state_dict": self.critic.state_dict(),
            "input_dim": self.critic.input_dim,
            "scope": self.critic_scope,
            "hidden_sizes": self.config.hidden_sizes,
            "best_score": best_score,
        }
        torch.save(actor_payload, output_dir / "actor.pt")
        torch.save(critic_payload, output_dir / "critic.pt")


class SharedPPOTrainer(BasePPOTrainer):
    critic_scope = "local"
    advantage_group_by_agent = False


class IPPOTrainer(BasePPOTrainer):
    critic_scope = "local"
    advantage_group_by_agent = True


class MAPPOTrainer(BasePPOTrainer):
    critic_scope = "global"
    advantage_group_by_agent = False

    def value_observation_tensor(self, global_state: Mapping[str, Any], obs_tensor: torch.Tensor, agent_count: int) -> torch.Tensor:
        encoded = encode_global_state(global_state).to(self.device)
        return encoded.unsqueeze(0).repeat(agent_count, 1)


def make_trainer(config: TrainingConfig, ppo_config: PPOConfig, *, device: str | torch.device = "cpu") -> BasePPOTrainer:
    algorithm = config.algorithm.lower()
    if algorithm == "shared_ppo":
        return SharedPPOTrainer(config, ppo_config, device=device)
    if algorithm == "ippo":
        return IPPOTrainer(config, ppo_config, device=device)
    if algorithm == "mappo":
        return MAPPOTrainer(config, ppo_config, device=device)
    raise ValueError(f"unsupported RL algorithm '{config.algorithm}'")


def write_training_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_resolved_config(path: Path, config: TrainingConfig, ppo_config: PPOConfig) -> None:
    path.write_text(yaml.safe_dump({"training": config.__dict__, "ppo": ppo_config.__dict__}, sort_keys=True))
