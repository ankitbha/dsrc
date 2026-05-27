#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loaders import compose_experiment_config
from src.controllers.base import BaseController, ControllerMetadata
from src.envs.base_ctde_env import AVObservationMap
from src.envs.wrappers import validate_action_mapping


class DummyController(BaseController):
    def __init__(self) -> None:
        super().__init__(
            ControllerMetadata(
                name="dummy_controller",
                family="baseline",
                requires_global_state=False,
            )
        )

    def act(self, local_obs: AVObservationMap, global_state=None):
        return {
            agent_id: {
                "desired_speed": 20.0,
                "desired_lane": "keep",
            }
            for agent_id in local_obs
        }


def main() -> int:
    sample_actions = validate_action_mapping(
        {
            "av_0": {"desired_speed": 22.0, "desired_lane": "keep"},
            "av_1": {"desired_speed": 18.5, "desired_lane": "left"},
        },
        expected_agent_ids=["av_0", "av_1"],
    )
    assert sample_actions["av_0"]["desired_lane"] == "keep"
    assert sample_actions["av_1"]["desired_speed"] == 18.5

    bundle = compose_experiment_config("exp_ring_wave_damping")
    assert bundle["topology"]["id"] == "ring"
    assert bundle["demand"]["id"] == "medium"
    assert bundle["human_model"]["id"] == "heterogeneous"
    assert bundle["training"]["id"] == "mappo"
    assert bundle["experiment"]["id"] == "exp_ring_wave_damping"
    assert bundle["outputs"]["episode_summary"].endswith("episode_summary.json")

    controller = DummyController()
    local_obs = {
        "av_0": {"ego_speed": 21.0},
        "av_1": {"ego_speed": 18.0},
    }
    action_map = controller.act(local_obs)
    validated = validate_action_mapping(action_map, expected_agent_ids=local_obs.keys())
    assert set(validated) == set(local_obs)
    assert controller.name == "dummy_controller"

    print("project interface validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
