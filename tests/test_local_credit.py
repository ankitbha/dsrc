"""The per-agent reward term and the action repeat, added in task 103.

Both exist for one reason: two SUMO training runs produced no learning, and the
diagnosis was the credit-assignment signal. A team reward divided among about
twelve agents moves less than its own noise when one agent changes what it does,
and at dt 0.1 a single action changes the next observation almost not at all.

Every assertion here is paired with the previous behaviour, because the sign-off
condition for this task is that an unconfigured trainer is unchanged.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.rl.ppo import PPOConfig
from src.rl.rewards import (
    DEFAULT_LOCAL_REWARD_WEIGHTS,
    blend_rewards,
    build_local_reward,
)
from src.rl.trainers import SharedPPOTrainer, TrainingConfig


LOCAL_WEIGHTS = {"outflow_recent": 0.1, "mean_speed": 0.05, "jam_fraction": -1.0}


def _observation(**overrides):
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


class StubEnv:
    """Two agents whose neighbourhoods differ, and a record of every step.

    `fast` sits on a clear segment and `slow` on a jammed one, so a reward that
    reads the neighbourhood separates them and a team reward cannot.
    """

    def __init__(self, *, duration_steps: int = 1000) -> None:
        self.duration_steps = duration_steps
        self.steps = 0
        self.applied: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None):
        self.steps = 0
        return self._observations(), {}

    def _observations(self):
        return {"fast": _observation(), "slow": _observation(ego_speed=2.0)}

    def step(self, actions):
        self.applied.append(dict(actions or {}))
        self.steps += 1
        info = {
            "metrics": {"mean_speed": 10.0, "throughput_recent": 2.0},
            "neighbourhood": {
                "fast": {"outflow_recent": 20.0, "mean_speed": 25.0, "jam_fraction": 0.0},
                "slow": {"outflow_recent": 2.0, "mean_speed": 3.0, "jam_fraction": 0.9},
            },
            "safety": {"penalties": {}, "layer_ran": False},
        }
        return self._observations(), {}, False, self.steps >= self.duration_steps, info

    def get_global_state(self):
        return {"time": float(self.steps), "active_vehicle_count": 2, "active_av_count": 2,
                "segment_state": {}, "demand_state": {}}

    def crashed_agent_ids(self):
        return []

    def close(self):
        pass


def _trainer(env: StubEnv, **ppo) -> SharedPPOTrainer:
    trainer = SharedPPOTrainer(
        TrainingConfig(
            algorithm="shared_ppo",
            action_profile="speed_only",
            hidden_sizes=(8,),
            rollout_steps=ppo.pop("rollout_steps", 4),
            dt=0.1,
            local_reward_weights=ppo.pop("local_reward_weights", None),
            decision_interval_s=ppo.pop("decision_interval_s", None),
        ),
        PPOConfig(update_epochs=1, minibatch_size=4, reward_scale=1.0, **ppo),
        device="cpu",
    )
    trainer.build_env = lambda: env  # type: ignore[method-assign]
    return trainer


class TestTheDefaultsAreTheOldBehaviour:

    def test_every_local_weight_defaults_to_zero(self):
        # A working default would change the objective of every experiment in this
        # project the moment `local_reward_weight` became non-zero anywhere.
        assert set(DEFAULT_LOCAL_REWARD_WEIGHTS.values()) == {0.0}

    def test_an_unset_blend_weight_gives_the_team_reward_exactly(self):
        assert blend_rewards(0.91, 1.02, 0.0) == 0.91
        assert blend_rewards(0.91, 1.02, 1.0) == 1.02
        assert blend_rewards(0.0, 1.0, 0.25) == 0.25
        # Out-of-range weights are clamped rather than extrapolated, which would
        # make the reward a difference of the two rather than a mixture.
        assert blend_rewards(0.0, 1.0, 2.0) == 1.0
        assert blend_rewards(0.0, 1.0, -1.0) == 0.0

    def test_an_unset_decision_interval_is_one_step(self):
        assert _trainer(StubEnv()).action_repeat() == 1

    def test_both_agents_get_the_same_reward_without_the_local_term(self):
        env = StubEnv()
        trainer = _trainer(env, local_reward_weights=LOCAL_WEIGHTS)
        buffer, _ = trainer.collect_rollout(seed=7)
        rewards = _rewards_by_agent(buffer)
        assert rewards["fast"] == rewards["slow"], (
            "the agents differed with local_reward_weight at its default, so the "
            "test below cannot attribute a difference to the local term"
        )


class TestTheLocalTermSeparatesAgents:

    def test_an_agent_on_a_clear_segment_is_rewarded_above_one_in_a_jam(self):
        env = StubEnv()
        trainer = _trainer(env, local_reward_weights=LOCAL_WEIGHTS, local_reward_weight=0.5)
        buffer, _ = trainer.collect_rollout(seed=7)
        rewards = _rewards_by_agent(buffer)
        assert rewards["fast"] > rewards["slow"]

        # And by exactly the blend the config asked for, not by some other amount.
        team = 0.05 * 10.0 + 0.02 * 2.0          # library defaults for these two keys
        fast = build_local_reward(
            {"outflow_recent": 20.0, "mean_speed": 25.0, "jam_fraction": 0.0}, LOCAL_WEIGHTS)
        slow = build_local_reward(
            {"outflow_recent": 2.0, "mean_speed": 3.0, "jam_fraction": 0.9}, LOCAL_WEIGHTS)
        assert rewards["fast"] == pytest.approx(blend_rewards(team, fast, 0.5))
        assert rewards["slow"] == pytest.approx(blend_rewards(team, slow, 0.5))

    def test_an_agent_with_no_neighbourhood_scores_zero_locally(self):
        # An agent between segments has no neighbourhood. Its local term is zero,
        # which is a measurement gap rather than a claim that the road is clear.
        assert build_local_reward({}, LOCAL_WEIGHTS) == 0.0


class TestActionRepeat:

    def test_the_interval_is_read_in_seconds_against_the_step(self):
        assert _trainer(StubEnv(), decision_interval_s=1.0).action_repeat() == 10
        assert _trainer(StubEnv(), decision_interval_s=0.1).action_repeat() == 1
        # Below one step it is still one step: a decision cannot be finer than the
        # simulation it acts on.
        assert _trainer(StubEnv(), decision_interval_s=0.01).action_repeat() == 1

    def test_the_simulation_advances_ten_steps_per_decision(self):
        env = StubEnv()
        trainer = _trainer(env, decision_interval_s=1.0, rollout_steps=3)
        buffer, _ = trainer.collect_rollout(seed=7)
        assert env.steps == 30, "the interval did not reach the simulation"
        assert len(buffer) == 6, "one transition per agent per decision, not per step"

    def test_the_action_is_applied_once_and_then_held(self):
        # `setSpeed` and `setTau` persist until they are changed, so re-issuing
        # them would change nothing about the traffic while multiplying the
        # actuation counters that exist to tell an inert head from an unused one.
        env = StubEnv()
        trainer = _trainer(env, decision_interval_s=1.0, rollout_steps=2)
        trainer.collect_rollout(seed=7)
        assert len(env.applied) == 20
        assert [index for index, applied in enumerate(env.applied) if applied] == [0, 10]

    def test_the_reward_is_summed_over_the_interval_and_not_sampled_at_its_end(self):
        held = _rewards_by_agent(
            _trainer(StubEnv(), decision_interval_s=1.0, rollout_steps=2).collect_rollout(seed=7)[0])
        per_step = _rewards_by_agent(
            _trainer(StubEnv(), rollout_steps=2).collect_rollout(seed=7)[0])
        assert held["fast"] == pytest.approx(10.0 * per_step["fast"])

    def test_an_episode_that_truncates_mid_interval_stops_there(self):
        env = StubEnv(duration_steps=4)
        trainer = _trainer(env, decision_interval_s=1.0, rollout_steps=1)
        trainer.collect_rollout(seed=7)
        assert env.steps == 4, "the simulation ran past the end of its episode"


def _rewards_by_agent(buffer) -> dict[str, float]:
    entries = list(zip(buffer.agent_ids, buffer.rewards, strict=True))
    return {agent_id: reward for agent_id, reward in entries}
