from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Mapping
import warnings

import numpy as np
import torch
import yaml

from src.config.loaders import load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger
from src.rl.actions import ActionSpec
from src.rl.encoders import (
    encode_local_batch,
    encode_neighbourhood_batch,
    encode_physical_global_state,
    local_obs_dim,
    neighbourhood_feature_dim,
    physical_global_state_dim,
)
from src.rl.models import GlobalCritic, LocalCritic, MultiCategoricalActor
from src.rl.ppo import PPOConfig, ppo_update
from src.rl.rewards import (
    blend_rewards,
    build_local_reward,
    build_team_reward,
    safety_penalty_for_agent,
)
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
    sensing: Mapping[str, Any] | None = None
    #: Overrides for `DEFAULT_REWARD_WEIGHTS`. None leaves the library
    #: defaults, which are speed-led; a config that wants a throughput-led
    #: objective says so here rather than editing the library and changing
    #: every other experiment at the same time.
    reward_weights: Mapping[str, float] | None = None
    #: Which simulator the policy is learned on. Defaults to `highway_env` so every
    #: existing config and its recorded results are unchanged; the SUMO configs opt
    #: in. Worth being explicit about rather than inferring, because the two roads
    #: produce different observation distributions -- SUMO's is collision-free and
    #: junction-limited, the other was neither -- and a checkpoint otherwise carries
    #: no record of which one it saw.
    simulator: str = "highway_env"
    #: Where a SUMO run writes its generated network and demand files.
    work_dir: str | None = None
    #: Steps run before the episode is observed, so a rollout does not begin on an
    #: empty network. Only meaningful for SUMO.
    warmup_steps: int = 0
    #: Overrides for `DEFAULT_LOCAL_REWARD_WEIGHTS`, the per-agent term. Priced by
    #: the config for the same reason the team weights are: the environment
    #: measures the neighbourhood and the config decides what it is worth.
    local_reward_weights: Mapping[str, float] | None = None
    #: Whether the CENTRALIZED critic is given each agent's own neighbourhood
    #: metrics -- the same quantities the per-agent reward is priced from, unnoised.
    #: Default False, so every configuration written before task 104 has the critic
    #: input dimension it had, and a checkpoint from one still loads.
    #:
    #: It exists because the per-agent reward alone did not raise the actor's
    #: policy-gradient norm: measured on paired trajectories, local over team was
    #: 0.962. The reward raised the cross-agent variance as much as the signal, and
    #: the critic could not centre it -- the global-state vector carries every
    #: segment in a fixed order and nothing in it says which is this agent's.
    critic_privileged_neighbourhood: bool = False
    #: How long one action is held, in simulated seconds. None means one decision
    #: per simulation step, which is what every configuration written before task
    #: 103 does. At dt 0.1 that is ten decisions a second, and a single one of them
    #: changes the next observation almost not at all, so the credit for an action
    #: is diluted by the step size before the team reward divides it among agents.
    decision_interval_s: float | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TrainingConfig:
        training = dict(config.get("training", config))
        env = dict(config.get("env", {}))
        actor_cfg = dict(training.get("actor", {})) if isinstance(training.get("actor", {}), Mapping) else {}
        hidden = actor_cfg.get("hidden_sizes", (128, 128))
        sensing_cfg = training.get("sensing", env.get("sensing"))
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
            sensing=dict(sensing_cfg) if isinstance(sensing_cfg, Mapping) and sensing_cfg else None,
            reward_weights=(
                {str(k): float(v) for k, v in weights_cfg.items()}
                if isinstance(weights_cfg := training.get("reward_weights"), Mapping) and weights_cfg
                else None
            ),
            simulator=str(training.get("simulator", "highway_env")),
            work_dir=(str(training["work_dir"]) if training.get("work_dir") else None),
            warmup_steps=int(training.get("warmup_steps", 0)),
            local_reward_weights=(
                {str(k): float(v) for k, v in local_cfg.items()}
                if isinstance(local_cfg := training.get("local_reward_weights"), Mapping) and local_cfg
                else None
            ),
            critic_privileged_neighbourhood=bool(
                training.get("critic_privileged_neighbourhood", False)),
            decision_interval_s=(
                float(training["decision_interval_s"])
                if training.get("decision_interval_s") is not None
                else None
            ),
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
            # The objective, not a proxy for it. This was
            # `mean_speed - jam_fraction`, which is neither the reward the policy
            # maximises nor a monotone function of it: a run whose objective
            # improved 30% recorded a score change of -0.067, so `actor.pt` was
            # selected on a quantity nothing was optimising. `episode_metrics` is
            # already the mean over the rollout and the reward is linear in the
            # metrics, so this is the mean team reward per step.
            score = build_team_reward(episode_metrics, self.config.reward_weights)
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

    def action_repeat(self) -> int:
        """How many simulation steps one action is held for.

        1 unless `decision_interval_s` is set, so every configuration written
        before task 103 decides once per simulation step exactly as it did.
        """
        interval = self.config.decision_interval_s
        if interval is None:
            return 1
        dt = float(self.config.dt) or 1.0
        return max(1, int(round(float(interval) / dt)))

    def hold_action(
        self,
        env: Any,
        action_map: Mapping[str, Any],
        repeat: int,
    ) -> tuple[dict[str, Any], bool, bool, list[Mapping[str, Any]], set[str]]:
        """Advance the simulation `repeat` steps, applying the action on the first.

        The action is applied ONCE. `setSpeed` and `setTau` persist in SUMO until
        they are changed, so re-issuing them every step would change nothing about the
        traffic while multiplying the actuation counters -- `lane_change_requests`
        exists to tell an inert head from an unused one, and it would read ten
        times the number of decisions.

        Returns every step's info, because the reward is the sum over the interval
        rather than the value at the end of it: a decision that causes a queue to
        clear halfway through the second should be credited with the whole of it.
        """
        infos: list[Mapping[str, Any]] = []
        crashed: set[str] = set()
        observations: dict[str, Any] = {}
        terminated = truncated = False
        for index in range(max(1, repeat)):
            observations, _, terminated, truncated, info = env.step(action_map if index == 0 else {})
            infos.append(info)
            if self.ppo_config.crash_penalty:
                # Asked of the environment rather than read out of its internals. The
                # previous form reached into `_av_vehicles`, which only exists on one
                # of the two simulators, so it failed the moment a second one appeared.
                #
                # Collected over every step of the interval, not only the last: a
                # collision the simulator resolves within the second is still a
                # collision the action is answerable for.
                crashed |= set(env.crashed_agent_ids())
            if terminated or truncated:
                break
        return observations, terminated, truncated, infos, crashed

    def collect_rollout(self, *, seed: int) -> tuple[RolloutBuffer, dict[str, Any]]:
        env = self.build_env()
        repeat = self.action_repeat()
        scale = self.ppo_config.reward_scale
        local_weight = self.ppo_config.local_reward_weight
        try:
            observations, _ = env.reset(seed=seed)
            buffer = RolloutBuffer()
            metric_history: list[dict[str, Any]] = []
            terminated = False
            truncated = False
            episode_index = 0
            steps = 0
            while steps < self.config.rollout_steps:
                if terminated or truncated:
                    episode_index += 1
                    observations, _ = env.reset(seed=seed + episode_index)
                    terminated = False
                    truncated = False
                agent_ids, obs_tensor = encode_local_batch(observations)
                if not agent_ids:
                    observations, terminated, truncated, infos, _ = self.hold_action(env, {}, repeat)
                    metric_history.extend(dict(info.get("metrics", {})) for info in infos)
                    steps += 1
                    continue
                obs_tensor = obs_tensor.to(self.device)
                privileged = self.privileged_features(env, agent_ids)
                with torch.no_grad():
                    actions, action_indices, log_probs, _ = self.actor.sample(obs_tensor)
                    value_obs = self.value_observation_tensor(
                        env.get_global_state(), obs_tensor, len(agent_ids), privileged)
                    values = self.critic(value_obs)
                action_map = {agent_id: action for agent_id, action in zip(agent_ids, actions, strict=True)}
                next_observations, terminated, truncated, infos, crashed_agents = self.hold_action(
                    env, action_map, repeat)
                metric_history.extend(dict(info.get("metrics", {})) for info in infos)
                # Summed over the interval the action was held for, so a longer
                # interval carries proportionally more reward. `reward_scale` is
                # per simulation step and unchanged: at dt 0.1 with a 1 s interval
                # the value targets have the same magnitude as one decision per
                # step at a gamma ten times closer to 1.
                team_reward = scale * sum(
                    build_team_reward(info.get("metrics", {}), self.config.reward_weights)
                    for info in infos
                )
                local_reward = self._local_rewards(agent_ids, infos, scale)
                for index, agent_id in enumerate(agent_ids):
                    reward = blend_rewards(team_reward, local_reward[agent_id], local_weight)
                    # From the last step of the interval. Inert on SUMO, where the
                    # safety layer does not run, and the previous form read the
                    # single step there was.
                    reward -= safety_penalty_for_agent(infos[-1], agent_id)
                    if agent_id in crashed_agents:
                        # the dense speed reward otherwise dominates the one-step
                        # collision term and argmax collapses to constant "fast"
                        reward -= self.ppo_config.crash_penalty
                    reward = max(-self.ppo_config.reward_clip, min(self.ppo_config.reward_clip, reward))
                    buffer.add(
                        observation=obs_tensor[index],
                        action=action_indices[index],
                        log_prob=log_probs[index],
                        reward=reward,
                        value=values[index],
                        done=bool(terminated or agent_id not in next_observations),
                        value_observation=value_obs[index],
                        agent_id=agent_id,
                    )
                observations = next_observations
                steps += 1
            self._set_bootstrap_values(buffer, env, observations, terminated)
            return buffer, aggregate_rollout_metrics(metric_history)
        finally:
            # libsumo is process-global and one connection is held at a time,
            # so a rollout that does not release it leaves a live simulation
            # behind and the next environment cannot start.
            if hasattr(env, "close"):
                env.close()

    def _local_rewards(
        self,
        agent_ids: list[str],
        infos: list[Mapping[str, Any]],
        scale: float,
    ) -> dict[str, float]:
        """Each agent's own reward, summed over the interval and scaled.

        Zero for an agent the environment reported no neighbourhood for, which is
        an agent between segments rather than an agent in clear traffic. Summed
        over the same interval as the team reward, so the two are comparable and
        the blend weight means what it says.
        """
        totals = {agent_id: 0.0 for agent_id in agent_ids}
        if not self.ppo_config.local_reward_weight:
            return totals
        for info in infos:
            neighbourhood = info.get("neighbourhood") or {}
            if not isinstance(neighbourhood, Mapping):
                continue
            for agent_id in agent_ids:
                values = neighbourhood.get(agent_id)
                if isinstance(values, Mapping):
                    totals[agent_id] += build_local_reward(values, self.config.local_reward_weights)
        return {agent_id: scale * total for agent_id, total in totals.items()}

    def critic_input_dim(self) -> int:
        return physical_global_state_dim() if self.critic_scope == "global" else local_obs_dim()

    def privileged_features(self, env: Any, agent_ids: list[str]) -> torch.Tensor | None:
        """Per-agent inputs for the critic that the actor never sees.

        None unless a trainer asks for them. `BasePPOTrainer` has a local critic
        that reads the same observation the actor does, so there is nothing
        privileged to add.
        """
        return None

    def value_observation_tensor(
        self,
        global_state: Mapping[str, Any],
        obs_tensor: torch.Tensor,
        agent_count: int,
        privileged: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        privileged = self.privileged_features(env, agent_ids)
        with torch.no_grad():
            value_obs = self.value_observation_tensor(
                env.get_global_state(), obs_tensor, len(agent_ids), privileged)
            values = self.critic(value_obs)
        bootstrap_values = {
            agent_id: float(values[index].detach().cpu().item())
            for index, agent_id in enumerate(agent_ids)
        }
        if not self.advantage_group_by_agent:
            bootstrap_values = {"__shared__": float(values.detach().mean().cpu().item())}
        buffer.set_bootstrap_values(bootstrap_values)

    def build_env(self) -> Any:
        """The environment this config asks for.

        An unknown name is refused by name rather than falling back to a default:
        a silent fallback would train on a road the config did not ask for and the
        checkpoint would not say so.
        """
        name = str(self.config.simulator).lower()
        if name == "highway_env":
            return HighwayTopologyEnv(self.config.topology, self.env_config())
        if name == "sumo":
            from src.sumo.env import SumoTopologyEnv

            config = self.env_config()
            if self.config.work_dir:
                config["work_dir"] = self.config.work_dir
            config.setdefault("warmup_steps", self.config.warmup_steps)
            return SumoTopologyEnv(self.config.topology, config)
        raise ValueError(
            f"unsupported simulator {self.config.simulator!r}; expected "
            "'highway_env' or 'sumo'"
        )

    def env_config(self) -> dict[str, Any]:
        topology_cfg = load_named_config("topology", self.config.topology)
        demand_cfg = load_named_config("demand", self.config.demand)
        human_cfg = load_named_config("human_model", self.config.human_model)
        config = {
            "topology": topology_cfg,
            "demand": demand_cfg,
            "human_model": human_cfg,
            "controller": {"name": self.config.algorithm, "family": "rl", "safety_mode": "integrated_rl"},
            "controlled_vehicles": self.config.controlled_vehicles,
            "initial_human_vehicles": self.config.initial_human_vehicles if self.config.topology == "ring" else 0,
            "duration_steps": self.config.duration_steps,
            "dt": self.config.dt,
        }
        if self.config.sensing:
            config["sensing"] = dict(self.config.sensing)
        return config

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
            "device_type": self.device.type,
        }
        critic_payload = {
            "state_dict": self.critic.state_dict(),
            "input_dim": self.critic.input_dim,
            "scope": self.critic_scope,
            "hidden_sizes": self.config.hidden_sizes,
            "best_score": best_score,
            "device_type": self.device.type,
        }
        atomic_torch_save(actor_payload, output_dir / actor_filename)
        atomic_torch_save(critic_payload, output_dir / critic_filename)

    def save_trainer_state(
        self,
        output_dir: Path,
        *,
        completed_update: int,
        best_score: float,
        metrics_rows: list[dict[str, Any]],
    ) -> None:
        atomic_torch_save(
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
                "device_type": self.device.type,
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

        actor_path = checkpoint_dir / ("latest_actor.pt" if resume_latest else "actor.pt")
        critic_path = checkpoint_dir / ("latest_critic.pt" if resume_latest else "critic.pt")
        actor_payload = torch.load(actor_path, map_location=self.device, weights_only=False)
        critic_payload = torch.load(critic_path, map_location=self.device, weights_only=False)
        self._warn_device_mismatch(state, actor_payload, critic_payload)
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
            for key in ("topology", "demand", "human_model", "sensing"):
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

    def _warn_device_mismatch(
        self,
        state: Mapping[str, Any],
        actor_payload: Mapping[str, Any],
        critic_payload: Mapping[str, Any],
    ) -> None:
        saved_device = state.get("device_type") or actor_payload.get("device_type") or critic_payload.get("device_type")
        if saved_device is not None and str(saved_device) != self.device.type:
            warnings.warn(
                f"resuming checkpoint saved on device type {saved_device!r} on {self.device.type!r}; RNG replay may differ",
                RuntimeWarning,
                stacklevel=2,
            )

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
    advantage_group_by_agent = True

    def critic_input_dim(self) -> int:
        extra = (neighbourhood_feature_dim()
                 if self.config.critic_privileged_neighbourhood else 0)
        return physical_global_state_dim() + local_obs_dim() + extra

    def privileged_features(self, env: Any, agent_ids: list[str]) -> torch.Tensor | None:
        """Each agent's own neighbourhood, unnoised, for the critic alone.

        Asked of the environment rather than reconstructed from the global state,
        because the global state carries every segment in a fixed order and nothing
        in it says which segment is this agent's -- which is precisely why the
        critic could not centre the per-agent reward.

        The environment need not offer it: `highway_env` has no
        `neighbourhood_metrics`, so a MAPPO run there keeps the input it had.
        """
        if not self.config.critic_privileged_neighbourhood:
            return None
        getter = getattr(env, "neighbourhood_metrics", None)
        neighbourhood = getter() if callable(getter) else {}
        return encode_neighbourhood_batch(agent_ids, neighbourhood or {}).to(self.device)

    def value_observation_tensor(
        self,
        global_state: Mapping[str, Any],
        obs_tensor: torch.Tensor,
        agent_count: int,
        privileged: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = encode_physical_global_state(global_state).to(self.device)
        parts = [encoded.unsqueeze(0).repeat(agent_count, 1), obs_tensor]
        if self.config.critic_privileged_neighbourhood:
            # Zeros rather than a shorter vector when the environment offers
            # nothing, so the critic's input width is a property of the config and
            # not of what a particular step happened to report.
            parts.append(privileged if privileged is not None
                         else torch.zeros((agent_count, neighbourhood_feature_dim()),
                                          device=self.device))
        return torch.cat(parts, dim=1)


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


def aggregate_rollout_metrics(metric_history: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_history:
        return {}
    result = dict(metric_history[-1])
    numeric_fields = {
        key
        for row in metric_history
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for key in numeric_fields:
        values = [
            float(row[key])
            for row in metric_history
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        ]
        if values:
            result[key] = sum(values) / len(values)
    return result


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def write_resolved_config(path: Path, config: TrainingConfig, ppo_config: PPOConfig) -> None:
    path.write_text(yaml.safe_dump({"training": config.__dict__, "ppo": ppo_config.__dict__}, sort_keys=True))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
