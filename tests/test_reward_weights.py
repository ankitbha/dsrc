"""Reward weights come from the config, and throughput leads them.

Measured over the first 100-update run, the shipped defaults gave `mean_speed`
69.3% of the positive reward (weight 0.05 against a mean of 20.86) and
`throughput_recent` 1.3% (weight 0.02 against a mean of 0.95) -- a ratio of 55 to
1, because the weights do not normalise for scale. The agent optimised mean speed
because that is where the reward was, and task 69 measured exactly that: +1.5 m/s
of speed and no throughput gain.

The weights were also not configurable at all: `build_team_reward` accepts a
`weights` argument and the trainer never passed one, so the only way to change the
objective was to edit the library default and change it for every experiment.
"""
from __future__ import annotations

import pytest

from src.rl.rewards import DEFAULT_REWARD_WEIGHTS, build_team_reward
from src.rl.trainers import TrainingConfig

#: The operating point the weights are balanced at: inverted_tree at the
#: `saturating` demand of 2000 veh/h, measured on 600-step runs.
SATURATING = {
    "mean_speed": 10.70,
    "throughput_recent": 11.5,
    "jam_fraction": 0.1083,
    "fairness_jain": 0.885,
    "speed_std": 3.48,
    "queue_length_total": 0.71,
    "new_collision_count": 0.0,
    "hard_braking_count": 0.92,
    "rolling_roadblock_score": 0.05,
}


def _shares(weights):
    contributions = {k: weights.get(k, 0.0) * v for k, v in SATURATING.items()}
    positive = sum(c for c in contributions.values() if c > 0)
    return {k: (c / positive if positive else 0.0) for k, c in contributions.items()}


class TestTheWeightsAreConfigurable:

    def test_a_config_can_override_them(self):
        config = TrainingConfig.from_mapping({
            "algorithm": "mappo",
            "reward_weights": {"throughput_recent": 0.10},
        })
        assert config.reward_weights is not None
        assert config.reward_weights["throughput_recent"] == 0.10

    def test_an_override_reaches_the_reward(self):
        metrics = {"mean_speed": 10.0, "throughput_recent": 10.0}
        base = build_team_reward(metrics, {"mean_speed": 0.05, "throughput_recent": 0.02})
        heavy = build_team_reward(metrics, {"mean_speed": 0.05, "throughput_recent": 0.20})
        assert heavy > base

    def test_absent_weights_fall_back_to_the_library_defaults(self):
        assert TrainingConfig.from_mapping({"algorithm": "mappo"}).reward_weights is None
        # And a None override must leave build_team_reward on its defaults.
        metrics = {"mean_speed": 10.0}
        assert build_team_reward(metrics, None) == pytest.approx(
            DEFAULT_REWARD_WEIGHTS["mean_speed"] * 10.0
        )


class TestThroughputLeadsAtTheOperatingPoint:

    def test_the_shipped_defaults_are_speed_led(self):
        # The control, and its numbers correct an earlier claim of mine. The 55-to-1
        # ratio reported from the first training run was measured at `medium`
        # demand, where throughput_recent averaged 0.95. At the saturating operating
        # point it averages 11.5, so the shipped weights already give it 19% there
        # against mean_speed's 44%. The imbalance is real but far smaller than the
        # figure I first quoted, and it depends on the demand.
        shares = _shares(DEFAULT_REWARD_WEIGHTS)
        assert shares["mean_speed"] == pytest.approx(0.44, abs=0.03)
        assert shares["throughput_recent"] == pytest.approx(0.19, abs=0.03)
        assert shares["mean_speed"] > shares["throughput_recent"]

    def test_the_deployment_config_is_throughput_led(self):
        import yaml

        config = yaml.safe_load(open("configs/training/mappo_deploysense.yaml"))
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        weights.update(config["reward_weights"])
        shares = _shares(weights)
        assert shares["throughput_recent"] > shares["mean_speed"], (
            f"throughput {shares['throughput_recent']:.1%} does not lead "
            f"mean_speed {shares['mean_speed']:.1%}"
        )
        assert shares["throughput_recent"] > 0.4

    def test_speed_still_carries_a_dense_signal(self):
        # Not zeroed. Throughput is sparse -- a count of completions in a 60 s
        # window, which is 0 on most steps early in a run -- so removing the speed
        # term would leave long stretches with no gradient at all.
        import yaml

        config = yaml.safe_load(open("configs/training/mappo_deploysense.yaml"))
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        weights.update(config["reward_weights"])
        assert _shares(weights)["mean_speed"] > 0.1
