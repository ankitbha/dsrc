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
        value = _float(metrics.get(key), 0.0)
        reward += float(weight) * value
    return float(reward)


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
