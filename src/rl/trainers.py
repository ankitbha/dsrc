from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.config.loaders import load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger
from src.rl.actions import ActionSpec
from src.rl.encoders import (
    encode_action_mask_batch,
    encode_local_batch,
    encode_physical_global_state,
    local_obs_dim,
    physical_global_state_dim,
)
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
        critic_input_dim = self.critic_input_dim()
        critic_cls = GlobalCritic if self.critic_scope == "global" else LocalCritic
        self.critic = critic_cls(critic_input_dim, hidden_sizes=config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=ppo_config.learning_rate,
        )

    @property
    def experiment_id(self) -> str:
        return f"{self.config.algorithm}_{self.config.topology}_{self.config.action_profile}_seed{self.config.seed}"

    def train(self, *, resume_from: str | Path | None = None, resume_latest: bool = False) -> dict[str, Any]:
        seed_everything(self.config.seed)
        output_dir = Path(self.config.output_root) / self.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "training_metrics.csv"
        rows: list[dict[str, Any]] = []
        best_score = float("-inf")
        start_update = 1
        if resume_from is not None:
            start_update, best_score, rows = self.load_training_state(Path(resume_from), resume_latest=resume_latest)
        for update in range(start_update, self.config.total_updates + 1):
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
            is_best = score > best_score
            best_score = max(best_score, score)
            row = {"update": update, "score": score, **stats, **episode_metrics}
            rows.append(row)
            self.save_checkpoint(
                output_dir,
                best_score=best_score,
                actor_filename="latest_actor.pt",
                critic_filename="latest_critic.pt",
            )
            if is_best:
                self.save_checkpoint(output_dir, best_score=best_score)
            self.save_trainer_state(
                output_dir,
                completed_update=update,
                best_score=best_score,
                metrics_rows=rows,
            )
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
            action_masks = encode_action_mask_batch(observations, agent_ids, self.action_spec).to(self.device)
            with torch.no_grad():
                actions, action_indices, log_probs, _ = self.actor.sample(obs_tensor, action_masks=action_masks)
                value_obs = self.value_observation_tensor(env.get_global_state(), obs_tensor, len(agent_ids))
                values = self.critic(value_obs)
            action_map = {agent_id: action for agent_id, action in zip(agent_ids, actions, strict=True)}
            next_observations, _, terminated, truncated, info = env.step(action_map)
            team_reward = build_team_reward(info.get("metrics", {})) * self.ppo_config.reward_scale
            for index, agent_id in enumerate(agent_ids):
                reward = team_reward - safety_penalty_for_agent(info, agent_id)
                reward = max(-self.ppo_config.reward_clip, min(self.ppo_config.reward_clip, reward))
                buffer.add(
                    observation=obs_tensor[index],
                    action=action_indices[index],
                    log_prob=log_probs[index],
                    reward=reward,
                    value=values[index],
                    done=bool(terminated or agent_id not in next_observations),
                    value_observation=value_obs[index],
                    action_mask=action_masks[index],
                    agent_id=agent_id,
                )
            observations = next_observations
            episode_metrics = dict(info.get("metrics", {}))
            steps += 1
        self._set_bootstrap_values(buffer, env, observations, terminated)
        return buffer, episode_metrics

    def critic_input_dim(self) -> int:
        return physical_global_state_dim() if self.critic_scope == "global" else local_obs_dim()

    def value_observation_tensor(self, global_state: Mapping[str, Any], obs_tensor: torch.Tensor, agent_count: int) -> torch.Tensor:
        return obs_tensor

    def _set_bootstrap_values(
        self,
        buffer: RolloutBuffer,
        env: HighwayTopologyEnv,
        observations: Mapping[str, Mapping[str, Any]],
        terminated: bool,
    ) -> None:
        if terminated or not observations:
            return
        agent_ids, obs_tensor = encode_local_batch(observations)
        if not agent_ids:
            return
        obs_tensor = obs_tensor.to(self.device)
        with torch.no_grad():
            value_obs = self.value_observation_tensor(env.get_global_state(), obs_tensor, len(agent_ids))
            values = self.critic(value_obs)
        bootstrap_values = {
            agent_id: float(values[index].detach().cpu().item())
            for index, agent_id in enumerate(agent_ids)
        }
        if not self.advantage_group_by_agent and bootstrap_values:
            bootstrap_values["__shared__"] = float(values[-1].detach().cpu().item())
        buffer.set_bootstrap_values(bootstrap_values)

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

    def save_checkpoint(
        self,
        output_dir: Path,
        *,
        best_score: float,
        actor_filename: str = "actor.pt",
        critic_filename: str = "critic.pt",
    ) -> None:
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
        torch.save(actor_payload, output_dir / actor_filename)
        torch.save(critic_payload, output_dir / critic_filename)

    def save_trainer_state(
        self,
        output_dir: Path,
        *,
        completed_update: int,
        best_score: float,
        metrics_rows: list[dict[str, Any]],
    ) -> None:
        torch.save(
            {
                "completed_update": int(completed_update),
                "best_score": float(best_score),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": self.config.__dict__,
                "ppo_config": self.ppo_config.__dict__,
                "algorithm": self.config.algorithm,
                "action_profile": self.config.action_profile,
                "critic_scope": self.critic_scope,
                "critic_input_dim": self.critic.input_dim,
                "hidden_sizes": self.config.hidden_sizes,
                "metrics_rows": list(metrics_rows),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            output_dir / "trainer_state.pt",
        )

    def load_training_state(
        self,
        checkpoint_dir: Path,
        *,
        resume_latest: bool,
    ) -> tuple[int, float, list[dict[str, Any]]]:
        state_path = checkpoint_dir / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"trainer state not found: {state_path}")
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        self._validate_training_state(state)

        actor_path = checkpoint_dir / ("latest_actor.pt" if resume_latest or (checkpoint_dir / "latest_actor.pt").exists() else "actor.pt")
        critic_path = checkpoint_dir / ("latest_critic.pt" if resume_latest or (checkpoint_dir / "latest_critic.pt").exists() else "critic.pt")
        actor_payload = torch.load(actor_path, map_location=self.device, weights_only=False)
        critic_payload = torch.load(critic_path, map_location=self.device, weights_only=False)
        self._validate_model_payloads(actor_payload, critic_payload)
        self.actor.load_state_dict(actor_payload["state_dict"])
        self.critic.load_state_dict(critic_payload["state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._restore_rng_state(state.get("rng_state", {}))
        rows = list(state.get("metrics_rows", []))
        return int(state["completed_update"]) + 1, float(state.get("best_score", float("-inf"))), rows

    def _validate_training_state(self, state: Mapping[str, Any]) -> None:
        expected = {
            "algorithm": self.config.algorithm,
            "action_profile": self.config.action_profile,
            "critic_scope": self.critic_scope,
            "critic_input_dim": self.critic.input_dim,
            "hidden_sizes": self.config.hidden_sizes,
        }
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise ValueError(f"checkpoint {key}={state.get(key)!r} does not match expected {expected_value!r}")
        saved_config = state.get("training_config", {})
        if isinstance(saved_config, Mapping):
            for key in ("topology", "demand", "human_model"):
                if saved_config.get(key) != getattr(self.config, key):
                    raise ValueError(
                        f"checkpoint {key}={saved_config.get(key)!r} does not match expected {getattr(self.config, key)!r}"
                    )

    def _validate_model_payloads(self, actor_payload: Mapping[str, Any], critic_payload: Mapping[str, Any]) -> None:
        metadata = actor_payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        if int(metadata.get("input_dim", -1)) != local_obs_dim():
            raise ValueError("actor checkpoint input_dim does not match current local observation dimension")
        if str(metadata.get("action_profile", "")) != self.config.action_profile:
            raise ValueError("actor checkpoint action_profile does not match current config")
        if str(critic_payload.get("scope", "")) != self.critic_scope:
            raise ValueError("critic checkpoint scope does not match current trainer")
        if int(critic_payload.get("input_dim", -1)) != self.critic.input_dim:
            raise ValueError("critic checkpoint input_dim does not match current critic")

    def _restore_rng_state(self, rng_state: Mapping[str, Any]) -> None:
        if not isinstance(rng_state, Mapping):
            return
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])


class SharedPPOTrainer(BasePPOTrainer):
    critic_scope = "local"
    advantage_group_by_agent = False


class IPPOTrainer(BasePPOTrainer):
    critic_scope = "local"
    advantage_group_by_agent = True


class MAPPOTrainer(BasePPOTrainer):
    critic_scope = "global"
    advantage_group_by_agent = False

    def critic_input_dim(self) -> int:
        return physical_global_state_dim() + local_obs_dim()

    def value_observation_tensor(self, global_state: Mapping[str, Any], obs_tensor: torch.Tensor, agent_count: int) -> torch.Tensor:
        encoded = encode_physical_global_state(global_state).to(self.device)
        return torch.cat([encoded.unsqueeze(0).repeat(agent_count, 1), obs_tensor], dim=1)


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
