from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_REWARD_WEIGHTS = {
    "mean_speed": 0.05,
    "throughput_recent": 0.02,
    "speed_std": -0.02,
    "jam_fraction": -1.0,
    "queue_length_total": -0.02,
    "collision_count": -5.0,
    "hard_braking_count": -0.1,
    "rolling_roadblock_score": -2.0,
    "fairness_jain": 0.5,
}


def build_team_reward(metrics: Mapping[str, Any], weights: Mapping[str, float] | None = None) -> float:
    reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
    reward_weights.update(dict(weights or {}))
    reward = 0.0
    for key, weight in reward_weights.items():
        metric_key = "new_collision_count" if key == "collision_count" and "new_collision_count" in metrics else key
        value = _float(metrics.get(metric_key), 0.0)
        reward += float(weight) * value
    return float(reward)


#: Weights for the per-agent term, over the agent's own segment and the segments
#: downstream of it. `outflow_recent` is the local analogue of
#: `throughput_recent`, `mean_speed` and `jam_fraction` are the same quantities the
#: team reward reads, measured near the agent instead of over the network.
#:
#: Default 0.0 on every key, so the term is inert unless a config asks for it. The
#: alternative -- shipping working defaults -- would change the objective of every
#: experiment in this project the moment `local_reward_weight` became non-zero
#: anywhere, and the weights that suit one operating point do not suit another.
DEFAULT_LOCAL_REWARD_WEIGHTS = {
    "outflow_recent": 0.0,
    "mean_speed": 0.0,
    "jam_fraction": 0.0,
    "queue_length": 0.0,
}


def build_local_reward(
    neighbourhood: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> float:
    """One agent's reward from the state of the road around it.

    **Why this exists.** Two training runs produced no learning, and the diagnosis
    was the credit-assignment signal rather than a defect: with a team reward
    shared among about twelve agents, one agent's action moves the reward by far
    less than the noise in it over the advantage horizon. A term measured over the
    agent's own neighbourhood responds to that agent's action by a much larger
    fraction, so the advantage carries information about the action.

    **It does not break decentralised execution.** The term is used only to compute
    a reward during training. The actor still reads the local observation alone,
    and nothing about what the deployed vehicle can sense changes.

    **The neighbourhood includes the segments downstream of the agent**, which is
    the environment's decision and is documented at
    `SumoTopologyEnv.neighbourhood_metrics`. Speed metering means holding the own
    segment below free flow to relieve the next one, so an own-segment-only term
    would penalise the behaviour this project exists to study.
    """
    local_weights = dict(DEFAULT_LOCAL_REWARD_WEIGHTS)
    local_weights.update(dict(weights or {}))
    reward = 0.0
    for key, weight in local_weights.items():
        reward += float(weight) * _float(neighbourhood.get(key), 0.0)
    return float(reward)


def blend_rewards(team: float, local: float, local_weight: float) -> float:
    """A convex combination of the team reward and the agent's own.

    `local_weight` 0.0 is the team reward exactly, which is what every
    configuration written before task 103 gets. 1.0 discards the team reward and
    makes the problem fully local, which is a different algorithm; the value in
    between is what says how much of the objective each agent is answerable for.
    """
    weight = min(1.0, max(0.0, float(local_weight)))
    return float((1.0 - weight) * float(team) + weight * float(local))


def safety_penalty_for_agent(info: Mapping[str, Any], agent_id: str) -> float:
    safety = info.get("safety", {})
    if not isinstance(safety, Mapping):
        return 0.0
    penalties = safety.get("penalties", {})
    if not isinstance(penalties, Mapping):
        return 0.0
    agent_penalties = penalties.get(agent_id, {})
    if not isinstance(agent_penalties, Mapping):
        return 0.0
    return sum(_float(value, 0.0) for value in agent_penalties.values())


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
