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
from src.envs.wrappers import HEADWAY_BINS, LANE_PREFERENCES, MERGE_MODES, SPEED_BINS, validate_action, validate_action_mapping


def sample_v2_action(
    speed: str = "nominal",
    headway: str = "normal",
    lane: str = "keep",
    merge: str = "normal",
) -> dict[str, str]:
    return {
        "desired_speed_bin": speed,
        "desired_headway_bin": headway,
        "lane_preference": lane,
        "merge_mode": merge,
    }


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
            agent_id: sample_v2_action()
            for agent_id in local_obs
        }


def validate_topology_ladder_if_available() -> None:
    try:
        from src.envs.topology_env import HighwayTopologyEnv
        from src.road.topology_factory import TOPOLOGY_IDS, build_topology
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        print(f"topology ladder validation skipped; missing optional simulation dependency: {missing}")
        return

    for topology_id in TOPOLOGY_IDS:
        topology = build_topology(topology_id)
        assert topology.topology_id == topology_id
        assert all(topology.segment_lengths[segment_id] > 0 for segment_id in topology.segment_ids)
        assert set(topology.detector_locations).issubset(set(topology.segment_ids))

    env = HighwayTopologyEnv("ring", {"controlled_vehicles": 2, "duration_steps": 2})
    obs, info = env.reset(seed=7)
    assert info["topology_id"] == "ring"
    assert set(obs) == {"av_0", "av_1"}
    actions = {agent_id: sample_v2_action() for agent_id in obs}
    next_obs, rewards, terminated, truncated, step_info = env.step(actions)
    assert set(next_obs) == set(rewards)
    assert terminated is False
    assert truncated is False
    assert step_info["topology_id"] == "ring"
    assert env.get_global_state()["topology_id"] == "ring"
    assert set(env.get_segment_metrics()) == {"ring_main"}


def main() -> int:
    assert SPEED_BINS == ("slow", "nominal", "fast")
    assert HEADWAY_BINS == ("normal", "larger", "largest")
    assert LANE_PREFERENCES == ("keep", "prefer_left_if_safe", "prefer_right_if_safe")
    assert MERGE_MODES == ("normal", "create_gap", "hold_lane")
    for speed_bin in SPEED_BINS:
        assert validate_action(sample_v2_action(speed=speed_bin))["desired_speed_bin"] == speed_bin
    for headway_bin in HEADWAY_BINS:
        assert validate_action(sample_v2_action(headway=headway_bin))["desired_headway_bin"] == headway_bin
    for lane_preference in LANE_PREFERENCES:
        assert validate_action(sample_v2_action(lane=lane_preference))["lane_preference"] == lane_preference
    for merge_mode in MERGE_MODES:
        assert validate_action(sample_v2_action(merge=merge_mode))["merge_mode"] == merge_mode

    for bad_action in (
        {"desired_speed": 20.0, "desired_lane": "keep"},
        sample_v2_action(lane="left"),
        sample_v2_action(lane="right"),
        sample_v2_action(merge="block_lanes"),
        {"desired_speed_bin": "nominal", "lane_preference": "keep", "merge_mode": "normal"},
    ):
        try:
            validate_action(bad_action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_action} should not be accepted")

    sample_actions = validate_action_mapping(
        {
            "av_0": sample_v2_action(),
            "av_1": sample_v2_action(speed="slow", headway="larger", lane="prefer_left_if_safe", merge="create_gap"),
        },
        expected_agent_ids=["av_0", "av_1"],
    )
    assert sample_actions["av_0"]["lane_preference"] == "keep"
    assert sample_actions["av_1"]["desired_speed_bin"] == "slow"
    try:
        validate_action_mapping(
            {
                "av_0": sample_v2_action(),
                "av_inactive": sample_v2_action(lane="prefer_right_if_safe"),
            },
            expected_agent_ids=["av_0"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("inactive/unexpected AV ids should not be accepted")

    bundle = compose_experiment_config("exp_ring_wave_damping")
    assert bundle["topology"]["id"] == "ring"
    assert bundle["demand"]["id"] == "medium"
    assert bundle["human_model"]["id"] == "heterogeneous"
    assert bundle["training"]["id"] == "mappo"
    assert bundle["experiment"]["id"] == "exp_ring_wave_damping"
    assert bundle["outputs"]["episode_summary"].endswith("episode_summary.json")
    assert bundle["outputs"]["step_metrics"].endswith("step_metrics.csv")
    assert bundle["outputs"]["segment_metrics"].endswith("segment_metrics.csv")

    controller = DummyController()
    local_obs = {
        "av_0": {"ego_speed": 21.0},
        "av_1": {"ego_speed": 18.0},
    }
    action_map = controller.act(local_obs)
    validated = validate_action_mapping(action_map, expected_agent_ids=local_obs.keys())
    assert set(validated) == set(local_obs)
    assert controller.name == "dummy_controller"
    assert controller.metadata.cooperation_mode == "none"
    assert controller.metadata.safety_mode == "external_filter"
    assert controller.metadata.supports_fallback_individual is True

    ctde_metadata = ControllerMetadata(
        name="mappo",
        family="rl",
        requires_global_state=True,
        safety_mode="integrated_rl",
    )
    assert ctde_metadata.cooperation_mode == "global_state"
    cooperative_metadata = ControllerMetadata(
        name="cooperative_mappo",
        family="rl",
        cooperation_mode="local_aggregate",
        safety_mode="integrated_rl",
        supports_fallback_individual=True,
    )
    assert cooperative_metadata.cooperation_mode == "local_aggregate"
    assert cooperative_metadata.supports_fallback_individual is True
    for bad_metadata_kwargs in (
        {"cooperation_mode": "v2v"},
        {"safety_mode": "unsafe_direct"},
    ):
        try:
            ControllerMetadata(name="bad", family="test", **bad_metadata_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"metadata should reject {bad_metadata_kwargs}")

    validate_topology_ladder_if_available()

    print("project interface validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
