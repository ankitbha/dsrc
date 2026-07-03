from __future__ import annotations

import pytest

from src.envs.topology_env import HighwayTopologyEnv
from src.road.topology_factory import TOPOLOGY_IDS, build_topology


EXPECTED_SEGMENTS = {
    "ring": ("ring_main",),
    "straight_single_lane": ("straight_upstream", "straight_mid", "straight_downstream"),
    "straight_multilane": ("straight_upstream", "straight_mid", "straight_downstream"),
    "merge": ("merge_main_upstream", "merge_ramp", "merge_weave", "merge_trunk"),
    "inverted_tree": (
        "tree_leaf_a1",
        "tree_leaf_a2",
        "tree_leaf_a3",
        "tree_leaf_a4",
        "tree_leaf_a5",
        "tree_leaf_a6",
        "tree_middle_b1",
        "tree_middle_b2",
        "tree_trunk_c",
    ),
    "inverted_tree_bottleneck": (
        "tree_leaf_a1",
        "tree_leaf_a2",
        "tree_leaf_a3",
        "tree_leaf_a4",
        "tree_leaf_a5",
        "tree_leaf_a6",
        "tree_middle_b1",
        "tree_middle_b2",
        "tree_trunk_c",
        "tree_bottleneck_d",
    ),
}


def safe_action() -> dict[str, str]:
    return {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }


@pytest.mark.parametrize("topology_id", TOPOLOGY_IDS)
def test_topology_factory_metadata(topology_id: str) -> None:
    spec = build_topology(topology_id)
    assert spec.topology_id == topology_id
    assert spec.segment_ids == EXPECTED_SEGMENTS[topology_id]
    assert set(spec.entry_segments).issubset(spec.segment_ids)
    assert set(spec.exit_segments).issubset(spec.segment_ids)
    assert set(spec.lane_counts) == set(spec.segment_ids)
    assert spec.lane_segments
    for segment_id in spec.segment_ids:
        assert spec.segment_lengths[segment_id] > 0
        assert spec.segment_edges[segment_id]
    for segment_id, locations in spec.detector_locations.items():
        assert segment_id in spec.segment_ids
        for location in locations:
            assert 0 <= location <= spec.segment_lengths[segment_id]


def test_expected_lane_counts_and_merge_nodes() -> None:
    assert build_topology("ring").lane_counts == {"ring_main": 1}
    assert build_topology("straight_single_lane").lane_counts["straight_upstream"] == 1
    assert build_topology("straight_multilane").lane_counts["straight_upstream"] == 3
    merge = build_topology("merge")
    assert merge.merge_nodes == ("merge_0",)
    assert merge.lane_counts["merge_main_upstream"] == 2
    assert merge.lane_counts["merge_ramp"] == 1
    tree = build_topology("inverted_tree")
    assert tree.merge_nodes == ("tree_merge_b1", "tree_merge_b2", "tree_merge_c")
    assert tree.segment_lengths["tree_trunk_c"] == 900.0
    assert tree.lane_counts["tree_trunk_c"] == 2
    assert tree.exit_segments == ("tree_trunk_c",)
    assert tree.bottleneck_segments == ()
    bottleneck_tree = build_topology("inverted_tree_bottleneck")
    assert bottleneck_tree.merge_nodes == ("tree_merge_b1", "tree_merge_b2", "tree_merge_c")
    assert bottleneck_tree.segment_lengths["tree_trunk_c"] == 600.0
    assert bottleneck_tree.lane_counts["tree_bottleneck_d"] == 1
    assert bottleneck_tree.exit_segments == ("tree_bottleneck_d",)
    assert bottleneck_tree.bottleneck_segments == ("tree_bottleneck_d",)


@pytest.mark.parametrize("topology_id", TOPOLOGY_IDS)
def test_highway_topology_env_smoke(topology_id: str) -> None:
    env = HighwayTopologyEnv(topology_id, {"controlled_vehicles": 2, "duration_steps": 4})
    obs, info = env.reset(seed=7)
    assert info["topology_id"] == topology_id
    assert set(obs) == {"av_0", "av_1"}
    for local_obs in obs.values():
        assert "sensor" in local_obs
        assert "cooperation" in local_obs
        assert local_obs["is_active"] is True
    assert env.get_global_state()["topology_id"] == topology_id
    assert set(env.get_segment_metrics()) == set(EXPECTED_SEGMENTS[topology_id])

    actions = {agent_id: safe_action() for agent_id in obs}
    next_obs, rewards, terminated, truncated, step_info = env.step(actions)
    assert set(next_obs) == set(rewards)
    assert set(next_obs).issubset(set(env.agent_ids))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["topology_id"] == topology_id

    with pytest.raises(ValueError):
        env.step({"av_inactive": safe_action()})


def test_integrated_rl_safety_mode_bounds_av_acceleration() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 1,
            "duration_steps": 2,
            "initial_speed_mps": 5.0,
            "controller": {"safety_mode": "integrated_rl"},
            "safety": {"max_accel_mps2": 0.25},
        },
    )
    obs, _ = env.reset(seed=17)
    _, _, _, _, _ = env.step({agent_id: safe_action() | {"desired_speed_bin": "fast"} for agent_id in obs})

    acceleration = env._active_vehicle_records()[0]["acceleration"]
    assert acceleration <= 0.25 + 1e-9


def test_nested_topology_safety_config_bounds_av_acceleration() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 1,
            "duration_steps": 2,
            "initial_speed_mps": 5.0,
            "controller": {"safety_mode": "integrated_rl"},
            "topology": {"safety": {"max_accel_mps2": 0.1}},
        },
    )
    obs, _ = env.reset(seed=17)
    _, _, _, _, _ = env.step({agent_id: safe_action() | {"desired_speed_bin": "fast"} for agent_id in obs})

    acceleration = env._active_vehicle_records()[0]["acceleration"]
    assert acceleration <= 0.1 + 1e-9


def test_top_level_safety_config_overrides_nested_topology_safety_config() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 1,
            "duration_steps": 2,
            "initial_speed_mps": 5.0,
            "controller": {"safety_mode": "integrated_rl"},
            "topology": {"safety": {"max_accel_mps2": 0.1}},
            "safety": {"max_accel_mps2": 0.3},
        },
    )
    obs, _ = env.reset(seed=17)
    _, _, _, _, _ = env.step({agent_id: safe_action() | {"desired_speed_bin": "fast"} for agent_id in obs})

    acceleration = env._active_vehicle_records()[0]["acceleration"]
    assert acceleration <= 0.3 + 1e-9
    assert acceleration > 0.1


def test_human_crash_does_not_terminate_without_av_crash() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 0,
            "initial_human_vehicles": 1,
            "duration_steps": 2,
        },
    )
    obs, _ = env.reset(seed=17)
    assert obs == {}
    for vehicle in env._human_vehicles.values():
        vehicle.crashed = True

    _, _, terminated, _, info = env.step({})

    assert terminated is False
    assert info["metrics"]["collision_count"] == 1


def test_simulator_default_safety_mode_skips_dsrc_penalties_for_baseline_avs() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 1,
            "duration_steps": 2,
            "initial_speed_mps": 5.0,
            "controller": {"safety_mode": "simulator_default"},
            "safety": {"max_accel_mps2": 0.01},
        },
    )
    obs, _ = env.reset(seed=17)
    _, _, _, _, info = env.step({agent_id: safe_action() | {"desired_speed_bin": "fast"} for agent_id in obs})

    assert info["safety"]["mode"] == "simulator_default"
    assert info["safety"]["penalties"] == {}
    assert info["diagnostics"]["external_safety_override"] == []


def test_ring_initial_spawn_spreads_full_loop() -> None:
    import math

    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 2,
            "initial_human_vehicles": 12,
            "duration_steps": 30,
            "controller": {"safety_mode": "integrated_rl"},
        },
    )
    env.reset(seed=11)
    road = env.road
    assert road is not None
    assert len(road.vehicles) == 14
    assert not any(vehicle.crashed for vehicle in road.vehicles)

    radius = float(env.topology.metadata["radius_m"])
    circumference = float(env.topology.metadata["circumference_m"])
    arc_positions = sorted(
        (math.atan2(float(v.position[1]), float(v.position[0])) % (2.0 * math.pi)) * radius
        for v in road.vehicles
    )
    gaps = [b - a for a, b in zip(arc_positions, arc_positions[1:])]
    gaps.append(circumference - arc_positions[-1] + arc_positions[0])
    spacing = circumference / 14.0
    assert min(gaps) > 10.0, f"vehicles overlap at spawn: min gap {min(gaps):.1f} m"
    assert max(gaps) < 2.0 * spacing, f"spawn clustered on part of the ring: max gap {max(gaps):.1f} m"


def test_ring_moderate_density_episode_survives_slow_commands() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 2,
            "initial_human_vehicles": 8,
            "duration_steps": 40,
            "controller": {"safety_mode": "integrated_rl"},
        },
    )
    observations, _ = env.reset(seed=11)
    info: dict = {}
    for step in range(25):
        actions = {agent_id: safe_action() | {"desired_speed_bin": "slow"} for agent_id in observations}
        observations, _, terminated, truncated, info = env.step(actions)
        metrics = info.get("metrics", {})
        assert not terminated, f"episode terminated at step {step + 1}"
        assert not truncated
        assert float(metrics.get("new_collision_count", 0.0)) == 0.0
    assert float(info["metrics"]["mean_speed"]) > 5.0


def test_ring_av_slots_interleaved_across_loop() -> None:
    import math

    env = HighwayTopologyEnv(
        "ring",
        {
            "controlled_vehicles": 2,
            "initial_human_vehicles": 12,
            "duration_steps": 5,
            "controller": {"safety_mode": "integrated_rl"},
        },
    )
    env.reset(seed=3)
    road = env.road
    assert road is not None
    radius = float(env.topology.metadata["radius_m"])
    circumference = float(env.topology.metadata["circumference_m"])
    av_arcs = [
        (math.atan2(float(v.position[1]), float(v.position[0])) % (2.0 * math.pi)) * radius
        for v in env._av_vehicles.values()
    ]
    assert len(av_arcs) == 2
    separation = abs(av_arcs[0] - av_arcs[1])
    separation = min(separation, circumference - separation)
    spacing = circumference / 14.0
    assert separation > circumference / 2.0 - 2.0 * spacing
