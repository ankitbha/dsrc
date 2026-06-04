from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.demand import BranchRoute, DemandProfile, DemandSpawner, build_route_plan, load_demand_profile
from src.demand.route_sampler import RoutePlan, road_route_to_destination
from src.envs.base_ctde_env import (
    AVActionMap,
    AVObservationMap,
    AVAction,
    EpisodeSummary,
    GlobalState,
    InfoDict,
    RewardMap,
    SegmentMetrics,
)
from src.envs.base_ctde_env import BaseCTDEEnv
from src.envs.wrappers import decode_speed_bin, default_agent_ids, lane_preference_to_action, validate_action_mapping
from src.metrics import MetricThresholds, compute_segment_metrics, compute_step_metrics, metric_thresholds_from_config
from src.metrics.safety_metrics import rear_ttc
from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec
from src.road.topology_factory import build_topology
from src.safety import SafetyConstraints, SafetyContext, SafetyState, apply_safety_layer
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot
from src.vehicles import HumanBehaviorModel, apply_human_behavior_profile, load_human_behavior_model

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex, Road
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle


@dataclass(frozen=True)
class VehicleMeta:
    vehicle_id: str
    role: str
    branch_id: str
    entry_segment: str
    behavior_profile: str | None = None


@dataclass
class VehicleRuntime:
    vehicle_id: str
    role: str
    branch_id: str
    spawn_time_s: float
    previous_speed_mps: float
    previous_lane_index: LaneIndex | None
    previous_segment_id: str | None
    distance_traveled_m: float = 0.0
    lane_change_times_s: list[float] = field(default_factory=list)
    lane_changed_this_step: bool = False
    acceleration_mps2: float = 0.0
    exit_time_s: float | None = None


class HighwayTopologyEnv(BaseCTDEEnv):
    """Minimal DSRC wrapper around project-owned HighwayEnv topology builders."""

    def __init__(
        self,
        topology_id: str,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.topology_id = topology_id
        self.config = dict(config or {})
        self.topology: TopologySpec = build_topology(topology_id, self._road_config())
        self.agent_ids = default_agent_ids(int(self.config.get("controlled_vehicles", 2)))
        self.road: Road | None = None
        self._av_vehicles: dict[str, ControlledVehicle] = {}
        self._human_vehicles: dict[str, IDMVehicle] = {}
        self._vehicle_meta: dict[int, VehicleMeta] = {}
        self._vehicle_runtime: dict[str, VehicleRuntime] = {}
        self._safety_states: dict[str, SafetyState] = {}
        self._target_headways: dict[str, float] = {}
        self._completed_vehicle_count = 0
        self._completed_travel_times: list[float] = []
        self._completed_travel_times_by_branch: dict[str, list[float]] = {}
        self._recent_completion_times: list[float] = []
        self._spawned_vehicle_count = 0
        self._spawned_av_count = 0
        self._spawned_human_count = 0
        self._spawned_human_by_profile: dict[str, int] = {}
        self._completed_human_by_profile: dict[str, int] = {}
        self._skipped_spawn_count = 0
        self._per_branch_spawned: dict[str, int] = {}
        self._per_branch_completed: dict[str, int] = {}
        self._per_branch_skipped_spawn: dict[str, int] = {}
        self._step_inflow: dict[str, int] = {}
        self._step_outflow: dict[str, int] = {}
        self._last_spawn_events: list[dict[str, Any]] = []
        self._last_skipped_spawn_events: list[dict[str, Any]] = []
        self._next_av_index = 0
        self._next_human_index = 0
        self._route_plan: RoutePlan = build_route_plan(self.topology)
        self._demand_profile: DemandProfile = load_demand_profile(None, enabled=False)
        self._human_behavior_model: HumanBehaviorModel = load_human_behavior_model(None)
        self._demand_spawner: DemandSpawner | None = None
        self._sensing = LocalObservationBuilder(SensingConfig.from_config(self.config))
        self._last_step_metrics: dict[str, Any] = {}
        self._step_count = 0
        self._time = 0.0

    def reset(
        self,
        config: Mapping[str, Any] | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[AVObservationMap, InfoDict]:
        if config:
            self.config.update(config)
            self.topology = build_topology(self.topology_id, self._road_config())
        if options and "config" in options and isinstance(options["config"], Mapping):
            self.config.update(options["config"])
            self.topology = build_topology(self.topology_id, self._road_config())

        self.agent_ids = default_agent_ids(int(self.config.get("controlled_vehicles", len(self.agent_ids))))
        self._route_plan = build_route_plan(self.topology)
        self.road = Road(
            network=self.topology.road_network,
            np_random=np.random.RandomState(seed),
            record_history=False,
        )
        self._av_vehicles = {}
        self._human_vehicles = {}
        self._vehicle_meta = {}
        self._vehicle_runtime = {}
        self._safety_states = {}
        self._target_headways = {}
        self._completed_vehicle_count = 0
        self._completed_travel_times = []
        self._completed_travel_times_by_branch = {}
        self._recent_completion_times = []
        self._spawned_vehicle_count = 0
        self._spawned_av_count = 0
        self._spawned_human_count = 0
        self._spawned_human_by_profile = {}
        self._completed_human_by_profile = {}
        self._skipped_spawn_count = 0
        self._per_branch_spawned = {branch.branch_id: 0 for branch in self._route_plan.branches}
        self._per_branch_completed = {branch.branch_id: 0 for branch in self._route_plan.branches}
        self._per_branch_skipped_spawn = {branch.branch_id: 0 for branch in self._route_plan.branches}
        self._step_inflow = {segment_id: 0 for segment_id in self.topology.segment_ids}
        self._step_outflow = {segment_id: 0 for segment_id in self.topology.segment_ids}
        self._last_spawn_events = []
        self._last_skipped_spawn_events = []
        self._next_av_index = 0
        self._next_human_index = 0
        self._sensing.reset(SensingConfig.from_config(self.config))
        self._last_step_metrics = {}
        self._step_count = 0
        self._time = 0.0
        self._human_behavior_model = load_human_behavior_model(self._human_model_config())
        self._configure_demand_spawner()
        if not self._uses_continuous_demand():
            self._spawn_controlled_vehicles()
            self._spawn_initial_human_vehicles()
        else:
            self.agent_ids = []
        return self.get_local_observations(), {
            "topology_id": self.topology_id,
            "segment_ids": self.topology.segment_ids,
            "demand": self._demand_state(),
            "routes": self._route_metadata(),
        }

    def step(
        self,
        av_actions: AVActionMap,
    ) -> tuple[AVObservationMap, RewardMap, bool, bool, InfoDict]:
        if self.road is None:
            raise RuntimeError("environment must be reset before step")

        self._step_inflow = {segment_id: 0 for segment_id in self.topology.segment_ids}
        self._step_outflow = {segment_id: 0 for segment_id in self.topology.segment_ids}
        self._last_spawn_events = []
        self._last_skipped_spawn_events = []
        self._reset_step_runtime_flags()

        active_agent_ids = list(self._av_vehicles)
        normalized_actions = validate_action_mapping(av_actions, expected_agent_ids=active_agent_ids)
        diagnostics: dict[str, list[dict[str, Any]]] = {
            "safety_masked_action": [],
            "etiquette_blocked_action": [],
            "follower_disruption_blocked": [],
            "external_safety_override": [],
            "simulator_blocked_action": [],
        }
        physical_accelerations: dict[str, float] = {}
        safety_penalties: dict[str, dict[str, float]] = {}
        for agent_id, action in normalized_actions.items():
            vehicle = self._av_vehicles[agent_id]
            if self._uses_integrated_safety():
                safety_decision = apply_safety_layer(
                    action,
                    self._safety_states[agent_id],
                    self._safety_context_for_vehicle(vehicle),
                    self._safety_constraints(),
                    agent_id=agent_id,
                )
                for key, events in safety_decision.diagnostics.items():
                    diagnostics.setdefault(key, []).extend(events)
                safety_penalties[agent_id] = dict(safety_decision.penalty_terms)
                vehicle.target_speed = np.clip(
                    safety_decision.target_speed_mps,
                    float(self.config.get("min_speed_mps", 0.0)),
                    float(self.config.get("max_speed_mps", 40.0)),
                )
                physical_accelerations[agent_id] = safety_decision.acceleration_mps2
                self._target_headways[agent_id] = safety_decision.target_headway_s
                if self.topology.supports_lane_change and safety_decision.lane_action is not None:
                    self._apply_lane_action(agent_id, vehicle, safety_decision.lane_action, diagnostics)
            else:
                self._apply_simulator_default_av_action(agent_id, vehicle, action, diagnostics)

        self.road.act()
        for agent_id, acceleration in physical_accelerations.items():
            vehicle = self._av_vehicles.get(agent_id)
            if vehicle is not None:
                vehicle.action["acceleration"] = float(acceleration)
        self.road.step(float(self.config.get("dt", 1.0)))
        self._step_count += 1
        self._time += float(self.config.get("dt", 1.0))
        self._update_vehicle_runtime_after_step()
        self._update_safety_distances()
        self._clear_exited_vehicles()
        self._spawn_demand_vehicles()
        self.agent_ids = list(self._av_vehicles)
        segment_metrics = self.get_segment_metrics()
        self._last_step_metrics = self._compute_step_metrics(segment_metrics, diagnostics)

        observations = self.get_local_observations()
        rewards = {agent_id: self._reward_for_vehicle(vehicle) for agent_id, vehicle in self._av_vehicles.items()}
        terminated = any(vehicle.crashed for vehicle in self._av_vehicles.values())
        truncated = self._step_count >= int(self.config.get("duration_steps", 120))
        info: InfoDict = {
            "topology_id": self.topology_id,
            "time": self._time,
            "diagnostics": diagnostics,
            "metrics": dict(self._last_step_metrics),
            "safety": {
                "mode": self._safety_mode(),
                "penalties": safety_penalties,
            },
            "demand": {
                "spawned": self._last_spawn_events,
                "skipped": self._last_skipped_spawn_events,
            },
        }
        return observations, rewards, terminated, truncated, info

    def get_local_observations(self) -> AVObservationMap:
        if self.road is None:
            return {}
        return self._sensing.build_all(
            time_s=self._time,
            topology=self.topology,
            snapshots=self._vehicle_snapshots(),
            current_av_ids=list(self._av_vehicles),
            safety_states=self._safety_states,
            target_headways=self._target_headways,
            target_lanes={agent_id: vehicle.target_lane_index for agent_id, vehicle in self._av_vehicles.items()},
            segment_metrics=self.get_segment_metrics(),
            constraints=self._safety_constraints(),
            rng=self.road.np_random,
        )

    def get_global_state(self) -> GlobalState:
        return {
            "time": self._time,
            "topology_id": self.topology_id,
            "active_vehicle_count": len(self._active_vehicles()),
            "active_av_count": len(self._av_vehicles),
            "completed_vehicle_count": self._completed_vehicle_count,
            "segment_state": self.get_segment_metrics(),
            "branch_state": {
                "per_branch_spawned": dict(self._per_branch_spawned),
                "per_branch_completed": dict(self._per_branch_completed),
                "per_branch_skipped_spawn": dict(self._per_branch_skipped_spawn),
                "branch_travel_time_mean": self._branch_travel_time_mean(),
                "fairness_jain": self._last_step_metrics.get("fairness_jain", 1.0),
            },
            "demand_state": self._demand_state(),
            "vehicle_role_state": self._vehicle_role_state(),
            "step_metrics": dict(self._last_step_metrics),
        }

    def get_segment_metrics(self) -> SegmentMetrics:
        return compute_segment_metrics(
            segment_ids=self.topology.segment_ids,
            segment_lengths_m=self.topology.segment_lengths,
            lane_counts=self.topology.lane_counts,
            active_vehicle_records=self._active_vehicle_records(),
            step_inflow=self._step_inflow,
            step_outflow=self._step_outflow,
            thresholds=self._metric_thresholds(),
        )

    def get_episode_summary(self) -> EpisodeSummary:
        return {
            "topology_id": self.topology_id,
            "steps": self._step_count,
            "time": self._time,
            "active_vehicle_count": len(self._active_vehicles()),
            "active_av_count": len(self._av_vehicles),
            "completed_vehicle_count": self._completed_vehicle_count,
            "travel_time_mean": float(np.mean(self._completed_travel_times)) if self._completed_travel_times else 0.0,
            "travel_time_count": len(self._completed_travel_times),
            "branch_travel_time_mean": self._branch_travel_time_mean(),
            "fairness_jain": self._last_step_metrics.get("fairness_jain", 1.0),
            **self._demand_state(),
        }

    def _spawn_controlled_vehicles(self) -> None:
        if self.road is None:
            raise RuntimeError("road is not initialized")
        if not self.agent_ids:
            return
        spawn_lanes, destination = self._spawn_lanes_and_destination()
        for index, agent_id in enumerate(self.agent_ids):
            lane_index = spawn_lanes[index % len(spawn_lanes)]
            lane = self.road.network.get_lane(lane_index)
            longitudinal = min(20.0 + 35.0 * index, max(5.0, lane.length - 10.0))
            vehicle = ControlledVehicle.make_on_lane(
                self.road,
                lane_index,
                longitudinal=longitudinal,
                speed=float(self.config.get("initial_speed_mps", min(lane.speed_limit or 24.0, 24.0))),
            )
            vehicle.route = road_route_to_destination(lane_index, destination, self.topology)
            self.road.vehicles.append(vehicle)
            self._register_existing_av(agent_id, vehicle, "initial", self.topology.segment_for_lane(lane_index) or "")

    def _spawn_initial_human_vehicles(self) -> None:
        if self.road is None:
            raise RuntimeError("road is not initialized")
        human_count = int(self.config.get("initial_human_vehicles", 0))
        if human_count <= 0:
            return
        spawn_lanes, destination = self._spawn_lanes_and_destination()
        for index in range(human_count):
            lane_index = spawn_lanes[index % len(spawn_lanes)]
            lane = self.road.network.get_lane(lane_index)
            longitudinal = (20.0 + 35.0 * index) % max(lane.length, 1.0)
            speed = float(self.config.get("initial_speed_mps", min(lane.speed_limit or 24.0, 24.0)))
            vehicle = IDMVehicle.make_on_lane(self.road, lane_index, longitudinal=longitudinal, speed=speed)
            vehicle.enable_lane_change = self.topology.supports_lane_change
            profile_id = self._human_behavior_model.sample_profile_id(self.road.np_random)
            profile = self._human_behavior_model.profile_for(profile_id)
            apply_human_behavior_profile(
                vehicle,
                profile,
                base_target_speed_mps=speed,
                min_speed_mps=float(self.config.get("min_speed_mps", 0.0)),
                max_speed_mps=float(self.config.get("max_speed_mps", 40.0)),
                lane_speed_limit_mps=lane.speed_limit,
            )
            vehicle.route = road_route_to_destination(lane_index, destination, self.topology)
            self.road.vehicles.append(vehicle)
            self._register_existing_human(vehicle, "initial", self.topology.segment_for_lane(lane_index) or "", profile_id)

    def _spawn_lanes_and_destination(self) -> tuple[list[LaneIndex], str]:
        inverted_tree_entries = [
            ("a1_entry", "b1", 0),
            ("a2_entry", "b1", 0),
            ("a3_entry", "b1", 0),
            ("a4_entry", "b2", 0),
            ("a5_entry", "b2", 0),
            ("a6_entry", "b2", 0),
        ]
        lanes_by_topology: dict[str, tuple[list[LaneIndex], str]] = {
            "ring": ([( "r0", "r1", 0)], "r0"),
            "straight_single_lane": ([("s0", "s1", 0)], "s3"),
            "straight_multilane": ([("s0", "s1", 0), ("s0", "s1", 1), ("s0", "s1", 2)], "s3"),
            "merge": ([("m0", "m1", 0), ("m0", "m1", 1), ("r0", "m1", 0)], "m3"),
            "inverted_tree": (inverted_tree_entries, "exit"),
            "inverted_tree_bottleneck": (inverted_tree_entries, "exit"),
        }
        return lanes_by_topology[self.topology_id]

    def _clear_exited_vehicles(self) -> None:
        if self.road is None or self.topology_id == "ring":
            return
        active_av: dict[str, ControlledVehicle] = {}
        active_human: dict[str, IDMVehicle] = {}
        active_vehicle_objects: set[int] = set()

        for agent_id, vehicle in self._av_vehicles.items():
            if self._has_exited(vehicle):
                self._record_vehicle_exit(vehicle)
            else:
                active_av[agent_id] = vehicle
                active_vehicle_objects.add(id(vehicle))
        for vehicle_id, vehicle in self._human_vehicles.items():
            if self._has_exited(vehicle):
                self._record_vehicle_exit(vehicle)
            else:
                active_human[vehicle_id] = vehicle
                active_vehicle_objects.add(id(vehicle))

        self._av_vehicles = active_av
        self._human_vehicles = active_human
        self._safety_states = {agent_id: state for agent_id, state in self._safety_states.items() if agent_id in active_av}
        self._target_headways = {agent_id: headway for agent_id, headway in self._target_headways.items() if agent_id in active_av}
        self._vehicle_meta = {vehicle_key: meta for vehicle_key, meta in self._vehicle_meta.items() if vehicle_key in active_vehicle_objects}
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if id(vehicle) in active_vehicle_objects]
        self.agent_ids = list(self._av_vehicles)

    def _has_exited(self, vehicle: ControlledVehicle) -> bool:
        if vehicle.lane_index is None:
            return True
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        if segment_id not in self.topology.exit_segments:
            return False
        lane = self.road.network.get_lane(vehicle.lane_index) if self.road is not None else None
        if lane is None:
            return True
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        return longitudinal >= lane.length - vehicle.LENGTH

    @staticmethod
    def _reward_for_vehicle(vehicle: ControlledVehicle) -> float:
        return 0.0 if vehicle.crashed else float(vehicle.speed)

    def _safety_constraints(self) -> SafetyConstraints:
        topology_cfg = self.config.get("topology", {})
        nested_cfg = topology_cfg.get("safety", {}) if isinstance(topology_cfg, Mapping) else {}
        cfg = self.config.get("safety", nested_cfg)
        if not isinstance(cfg, Mapping):
            cfg = {}
        return SafetyConstraints(
            lane_change_dwell_s=float(cfg.get("lane_change_dwell_s", 15.0)),
            max_lane_changes_per_km=float(cfg.get("max_lane_changes_per_km", 2.0)),
            max_follower_braking_mps2=float(cfg.get("max_follower_braking_mps2", 2.5)),
            comfortable_decel_mps2=float(cfg.get("comfortable_decel_mps2", 3.0)),
            max_accel_mps2=float(cfg.get("max_accel_mps2", 2.0)),
            max_decel_mps2=float(cfg.get("max_decel_mps2", 3.0)),
            emergency_decel_mps2=float(cfg.get("emergency_decel_mps2", 6.0)),
            min_front_gap_m=float(cfg.get("min_front_gap_m", 5.0)),
            min_rear_gap_m=float(cfg.get("min_rear_gap_m", 5.0)),
            min_forward_ttc_s=float(cfg.get("min_forward_ttc_s", 2.0)),
            min_lane_change_ttc_s=float(cfg.get("min_lane_change_ttc_s", 2.5)),
            speed_control_kp=float(cfg.get("speed_control_kp", 0.6)),
        )

    def _safety_mode(self) -> str:
        controller_cfg = self.config.get("controller", {})
        if isinstance(controller_cfg, Mapping) and "safety_mode" in controller_cfg:
            return str(controller_cfg["safety_mode"])
        safety_cfg = self.config.get("safety", {})
        if isinstance(safety_cfg, Mapping) and "mode" in safety_cfg:
            return str(safety_cfg["mode"])
        return "integrated_rl"

    def _uses_integrated_safety(self) -> bool:
        return self._safety_mode() == "integrated_rl"

    def _apply_lane_action(
        self,
        agent_id: str,
        vehicle: ControlledVehicle,
        lane_action: str,
        diagnostics: dict[str, list[dict[str, Any]]],
    ) -> None:
        before = vehicle.target_lane_index
        vehicle.act(lane_action)
        if vehicle.target_lane_index == before:
            diagnostics["simulator_blocked_action"].append({"agent_id": agent_id, "lane_action": lane_action})
        else:
            state = self._safety_states[agent_id]
            state.last_lane_change_time_s = self._time
            state.lane_changes_last_km += 1
            state.last_lane_index = vehicle.target_lane_index

    def _apply_simulator_default_av_action(
        self,
        agent_id: str,
        vehicle: ControlledVehicle,
        action: AVAction,
        diagnostics: dict[str, list[dict[str, Any]]],
    ) -> None:
        vehicle.target_speed = np.clip(
            decode_speed_bin(
                action["desired_speed_bin"],
                free_flow_speed_mps=self._free_flow_speed_for_vehicle(vehicle),
                min_contextual_speed_mps=float(self.config.get("min_contextual_speed_mps", 12.0)),
            ),
            float(self.config.get("min_speed_mps", 0.0)),
            float(self.config.get("max_speed_mps", 40.0)),
        )
        self._target_headways[agent_id] = self._target_headways.get(agent_id, 1.6)
        lane_action = None if action["merge_mode"] == "hold_lane" else lane_preference_to_action(action["lane_preference"])
        if self.topology.supports_lane_change and lane_action is not None:
            self._apply_lane_action(agent_id, vehicle, lane_action, diagnostics)

    def _safety_context_for_vehicle(self, vehicle: ControlledVehicle) -> SafetyContext:
        if self.road is None:
            raise RuntimeError("environment must be reset before safety context is requested")
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        metrics = self.get_segment_metrics().get(segment_id, {}) if segment_id else {}
        free_flow_speed = self._free_flow_speed_for_vehicle(vehicle)
        agent_id = self._agent_id_for_vehicle(vehicle)
        gap_context = None
        if agent_id is not None:
            gap_context = self._sensing.lane_gap_context(
                ego_id=agent_id,
                time_s=self._time,
                topology=self.topology,
                snapshots=self._vehicle_snapshots(),
                target_lane=vehicle.target_lane_index,
                rng=self.road.np_random,
            )
        return SafetyContext(
            time_s=self._time,
            ego_speed_mps=float(vehicle.speed),
            free_flow_speed_mps=free_flow_speed,
            min_contextual_speed_mps=float(self.config.get("min_contextual_speed_mps", 12.0)),
            local_density_veh_per_km=(
                gap_context.local_density_veh_per_km if gap_context is not None else float(metrics.get("density", 0.0))
            ),
            downstream_congested=False,
            leader_gap_m=gap_context.leader_gap_m if gap_context is not None else float("inf"),
            leader_relative_speed_mps=gap_context.leader_relative_speed_mps if gap_context is not None else 0.0,
            follower_gap_m=gap_context.follower_gap_m if gap_context is not None else float("inf"),
            follower_relative_speed_mps=gap_context.follower_relative_speed_mps if gap_context is not None else 0.0,
            target_lane_exists=gap_context.target_lane_exists if gap_context is not None else True,
            target_lane_front_gap_m=gap_context.target_lane_front_gap_m if gap_context is not None else float("inf"),
            target_lane_front_relative_speed_mps=(
                gap_context.target_lane_front_relative_speed_mps if gap_context is not None else 0.0
            ),
            target_lane_rear_gap_m=gap_context.target_lane_rear_gap_m if gap_context is not None else float("inf"),
            target_lane_rear_relative_speed_mps=(
                gap_context.target_lane_rear_relative_speed_mps if gap_context is not None else 0.0
            ),
            target_lane_rear_required_decel_mps2=(
                gap_context.target_lane_rear_required_decel_mps2 if gap_context is not None else 0.0
            ),
            all_lanes_av_occupied=gap_context.all_lanes_av_occupied if gap_context is not None else False,
            av_mean_speed_mps=gap_context.nearby_av_mean_speed_mps if gap_context is not None else float(metrics.get("mean_speed", free_flow_speed)),
            local_mean_speed_mps=gap_context.local_mean_speed_mps if gap_context is not None else float(metrics.get("mean_speed", free_flow_speed)),
            near_merge=bool(segment_id and ("merge" in segment_id or segment_id in self.topology.bottleneck_segments)),
        )

    def _free_flow_speed_for_vehicle(self, vehicle: ControlledVehicle) -> float:
        if self.road is None or vehicle.lane_index is None:
            return 30.0
        lane = self.road.network.get_lane(vehicle.lane_index)
        return float(lane.speed_limit or 30.0)

    @staticmethod
    def _headway_s(vehicle: ControlledVehicle, leader: Any | None) -> float:
        if leader is None or vehicle.speed <= 0:
            return float("inf")
        return max(0.0, float(vehicle.lane_distance_to(leader)) / max(float(vehicle.speed), 1e-6))

    def _agent_id_for_vehicle(self, vehicle: ControlledVehicle) -> str | None:
        for agent_id, candidate in self._av_vehicles.items():
            if candidate is vehicle:
                return agent_id
        return None

    def _vehicle_snapshots(self) -> list[VehicleSnapshot]:
        if self.road is None:
            return []
        snapshots: list[VehicleSnapshot] = []
        for vehicle in self._active_vehicles():
            meta = self._vehicle_meta.get(id(vehicle))
            if meta is None:
                continue
            runtime = self._vehicle_runtime.get(meta.vehicle_id)
            lane_index = vehicle.lane_index
            segment_id = self.topology.segment_for_lane(lane_index)
            lane_id = int(lane_index[2]) if lane_index is not None else -1
            longitudinal = 0.0
            if lane_index is not None:
                lane = self.road.network.get_lane(lane_index)
                longitudinal = float(lane.local_coordinates(vehicle.position)[0])
            snapshots.append(
                VehicleSnapshot(
                    vehicle_id=meta.vehicle_id,
                    role=meta.role,
                    segment_id=segment_id,
                    lane_index=lane_index,
                    lane_id=lane_id,
                    position=(float(vehicle.position[0]), float(vehicle.position[1])),
                    longitudinal_m=longitudinal,
                    speed_mps=float(vehicle.speed),
                    acceleration_mps2=float(
                        runtime.acceleration_mps2 if runtime is not None else vehicle.action.get("acceleration", 0.0)
                    ),
                    free_flow_speed_mps=self._free_flow_speed_for_vehicle(vehicle),
                    crashed=bool(vehicle.crashed),
                )
            )
        return snapshots

    def _update_safety_distances(self) -> None:
        dt = float(self.config.get("dt", 1.0))
        for agent_id, vehicle in self._av_vehicles.items():
            state = self._safety_states[agent_id]
            state.distance_since_window_start_m += max(0.0, float(vehicle.speed)) * dt
            if state.distance_since_window_start_m >= 1000.0:
                state.distance_since_window_start_m = 0.0
                state.lane_changes_last_km = 0

    def _active_vehicles(self) -> list[ControlledVehicle]:
        return [*self._av_vehicles.values(), *self._human_vehicles.values()]

    def _road_config(self) -> Mapping[str, Any] | None:
        topology_config = self.config.get("topology")
        if isinstance(topology_config, Mapping):
            return topology_config.get("road", topology_config)
        road_config = self.config.get("road")
        return road_config if isinstance(road_config, Mapping) else None

    def _demand_config(self) -> Mapping[str, Any] | None:
        demand_config = self.config.get("demand")
        return demand_config if isinstance(demand_config, Mapping) else None

    def _human_model_config(self) -> Mapping[str, Any] | None:
        human_model_config = self.config.get("human_model")
        return human_model_config if isinstance(human_model_config, Mapping) else None

    def _uses_continuous_demand(self) -> bool:
        return self._demand_spawner is not None and self._demand_profile.enabled and self._route_plan.enabled

    def _configure_demand_spawner(self) -> None:
        demand_config = self._demand_config()
        demand_enabled = demand_config is not None and self.topology_id != "ring"
        self._demand_profile = load_demand_profile(demand_config, enabled=demand_enabled)
        if self.road is None or not self._demand_profile.enabled or not self._route_plan.enabled:
            self._demand_spawner = None
            return
        self._demand_spawner = DemandSpawner(
            self._demand_profile,
            self._route_plan,
            self.topology,
            self.road.np_random,
            self._human_behavior_model,
        )

    def _spawn_demand_vehicles(self) -> None:
        if self.road is None or self._demand_spawner is None:
            return
        result = self._demand_spawner.spawn_due(
            self.road,
            time_s=self._time,
            dt_s=float(self.config.get("dt", 1.0)),
            register_vehicle=self._register_spawned_vehicle,
        )
        self._last_spawn_events = result.spawned
        self._last_skipped_spawn_events = result.skipped
        for event in result.skipped:
            branch_id = str(event["branch_id"])
            self._skipped_spawn_count += 1
            self._per_branch_skipped_spawn[branch_id] = self._per_branch_skipped_spawn.get(branch_id, 0) + 1

    def _register_spawned_vehicle(
        self,
        role: str,
        branch_id: str,
        vehicle: ControlledVehicle,
        branch: BranchRoute,
        behavior_profile: str | None = None,
    ) -> str:
        if self.road is None:
            raise RuntimeError("road is not initialized")
        if role == "av":
            vehicle_id = f"av_{self._next_av_index}"
            self._next_av_index += 1
            self._av_vehicles[vehicle_id] = vehicle
            self._safety_states[vehicle_id] = SafetyState(last_lane_index=vehicle.lane_index)
            self._target_headways[vehicle_id] = 1.6
            self._spawned_av_count += 1
        elif role == "human":
            vehicle_id = f"human_{self._next_human_index}"
            self._next_human_index += 1
            self._human_vehicles[vehicle_id] = vehicle  # type: ignore[assignment]
            self._spawned_human_count += 1
            if behavior_profile is not None:
                self._spawned_human_by_profile[behavior_profile] = self._spawned_human_by_profile.get(behavior_profile, 0) + 1
        else:
            raise ValueError(f"unsupported vehicle role '{role}'")

        self.road.vehicles.append(vehicle)
        self._vehicle_meta[id(vehicle)] = VehicleMeta(
            vehicle_id=vehicle_id,
            role=role,
            branch_id=branch_id,
            entry_segment=branch.entry_segment,
            behavior_profile=behavior_profile,
        )
        self._vehicle_runtime[vehicle_id] = self._runtime_for_vehicle(self._vehicle_meta[id(vehicle)], vehicle)
        self._spawned_vehicle_count += 1
        self._per_branch_spawned[branch_id] = self._per_branch_spawned.get(branch_id, 0) + 1
        self._step_inflow[branch.entry_segment] = self._step_inflow.get(branch.entry_segment, 0) + 1
        self.agent_ids = list(self._av_vehicles)
        return vehicle_id

    def _register_existing_av(
        self,
        agent_id: str,
        vehicle: ControlledVehicle,
        branch_id: str,
        entry_segment: str,
    ) -> None:
        self._av_vehicles[agent_id] = vehicle
        self._vehicle_meta[id(vehicle)] = VehicleMeta(
            vehicle_id=agent_id,
            role="av",
            branch_id=branch_id,
            entry_segment=entry_segment,
            behavior_profile=None,
        )
        self._vehicle_runtime[agent_id] = self._runtime_for_vehicle(self._vehicle_meta[id(vehicle)], vehicle)
        self._safety_states[agent_id] = SafetyState(last_lane_index=vehicle.lane_index)
        self._target_headways[agent_id] = 1.6
        try:
            self._next_av_index = max(self._next_av_index, int(agent_id.split("_", 1)[1]) + 1)
        except (IndexError, ValueError):
            pass

    def _register_existing_human(
        self,
        vehicle: IDMVehicle,
        branch_id: str,
        entry_segment: str,
        behavior_profile: str | None,
    ) -> None:
        vehicle_id = f"human_{self._next_human_index}"
        self._next_human_index += 1
        self._human_vehicles[vehicle_id] = vehicle
        self._vehicle_meta[id(vehicle)] = VehicleMeta(
            vehicle_id=vehicle_id,
            role="human",
            branch_id=branch_id,
            entry_segment=entry_segment,
            behavior_profile=behavior_profile,
        )
        self._vehicle_runtime[vehicle_id] = self._runtime_for_vehicle(self._vehicle_meta[id(vehicle)], vehicle)
        self._spawned_vehicle_count += 1
        self._spawned_human_count += 1
        if behavior_profile is not None:
            self._spawned_human_by_profile[behavior_profile] = self._spawned_human_by_profile.get(behavior_profile, 0) + 1
        self._per_branch_spawned[branch_id] = self._per_branch_spawned.get(branch_id, 0) + 1

    def _record_vehicle_exit(self, vehicle: ControlledVehicle) -> None:
        meta = self._vehicle_meta.get(id(vehicle))
        self._completed_vehicle_count += 1
        if meta is not None:
            self._per_branch_completed[meta.branch_id] = self._per_branch_completed.get(meta.branch_id, 0) + 1
            runtime = self._vehicle_runtime.get(meta.vehicle_id)
            if runtime is not None:
                runtime.exit_time_s = self._time
                travel_time = max(0.0, self._time - runtime.spawn_time_s)
                self._completed_travel_times.append(travel_time)
                self._completed_travel_times_by_branch.setdefault(meta.branch_id, []).append(travel_time)
                self._recent_completion_times.append(self._time)
            if meta.role == "human" and meta.behavior_profile is not None:
                self._completed_human_by_profile[meta.behavior_profile] = (
                    self._completed_human_by_profile.get(meta.behavior_profile, 0) + 1
                )
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        if segment_id is not None:
            self._step_outflow[segment_id] = self._step_outflow.get(segment_id, 0) + 1

    def _demand_state(self) -> dict[str, Any]:
        return {
            "enabled": self._uses_continuous_demand(),
            "profile_id": self._demand_profile.profile_id,
            "current_vehicles_per_hour": self._demand_profile.vehicles_per_hour_at(self._time),
            "av_penetration": self._demand_profile.av_penetration,
            "spawned_vehicle_count": self._spawned_vehicle_count,
            "spawned_av_count": self._spawned_av_count,
            "spawned_human_count": self._spawned_human_count,
            "completed_vehicle_count": self._completed_vehicle_count,
            "per_branch_spawned": dict(self._per_branch_spawned),
            "per_branch_completed": dict(self._per_branch_completed),
            "per_branch_travel_time_mean": self._branch_travel_time_mean(),
            "branch_split": dict(self._demand_profile.branch_split),
            "skipped_spawn_count": self._skipped_spawn_count,
            "per_branch_skipped_spawn": dict(self._per_branch_skipped_spawn),
            "active_human_by_profile": self._active_human_by_profile(),
            "spawned_human_by_profile": dict(self._spawned_human_by_profile),
            "completed_human_by_profile": dict(self._completed_human_by_profile),
        }

    def _vehicle_role_state(self) -> dict[str, Any]:
        return {
            "human_model_id": self._human_behavior_model.model_id,
            "active_av_count": len(self._av_vehicles),
            "active_human_count": len(self._human_vehicles),
            "active_human_by_profile": self._active_human_by_profile(),
            "spawned_human_by_profile": dict(self._spawned_human_by_profile),
            "completed_human_by_profile": dict(self._completed_human_by_profile),
        }

    def _active_human_by_profile(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for vehicle in self._human_vehicles.values():
            meta = self._vehicle_meta.get(id(vehicle))
            if meta is None or meta.behavior_profile is None:
                continue
            counts[meta.behavior_profile] = counts.get(meta.behavior_profile, 0) + 1
        return counts

    def _route_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self._route_plan.enabled,
            "destination": self._route_plan.destination,
            "branches": {
                branch.branch_id: {
                    "entry_edge": branch.entry_edge,
                    "entry_segment": branch.entry_segment,
                    "lane_count": branch.lane_count,
                }
                for branch in self._route_plan.branches
            },
        }

    def _metric_thresholds(self) -> MetricThresholds:
        return metric_thresholds_from_config(self.config)

    def _reset_step_runtime_flags(self) -> None:
        for runtime in self._vehicle_runtime.values():
            runtime.lane_changed_this_step = False
            runtime.acceleration_mps2 = 0.0

    def _update_vehicle_runtime_after_step(self) -> None:
        dt = max(float(self.config.get("dt", 1.0)), 1e-9)
        for vehicle in self._active_vehicles():
            meta = self._vehicle_meta.get(id(vehicle))
            if meta is None:
                continue
            runtime = self._vehicle_runtime.get(meta.vehicle_id)
            if runtime is None:
                runtime = self._runtime_for_vehicle(meta, vehicle)
                self._vehicle_runtime[meta.vehicle_id] = runtime
            speed = float(vehicle.speed)
            runtime.acceleration_mps2 = (speed - runtime.previous_speed_mps) / dt
            runtime.distance_traveled_m += max(0.0, speed) * dt
            lane_changed = vehicle.lane_index != runtime.previous_lane_index
            runtime.lane_changed_this_step = lane_changed
            if lane_changed:
                runtime.lane_change_times_s.append(self._time)
            runtime.previous_speed_mps = speed
            runtime.previous_lane_index = vehicle.lane_index
            runtime.previous_segment_id = self.topology.segment_for_lane(vehicle.lane_index)

    def _runtime_for_vehicle(self, meta: VehicleMeta, vehicle: ControlledVehicle) -> VehicleRuntime:
        return VehicleRuntime(
            vehicle_id=meta.vehicle_id,
            role=meta.role,
            branch_id=meta.branch_id,
            spawn_time_s=self._time,
            previous_speed_mps=float(vehicle.speed),
            previous_lane_index=vehicle.lane_index,
            previous_segment_id=self.topology.segment_for_lane(vehicle.lane_index),
        )

    def _active_vehicle_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for vehicle in self._active_vehicles():
            meta = self._vehicle_meta.get(id(vehicle))
            if meta is None:
                continue
            runtime = self._vehicle_runtime.get(meta.vehicle_id)
            segment_id = self.topology.segment_for_lane(vehicle.lane_index)
            lane_id = vehicle.lane_index[2] if vehicle.lane_index is not None else 0
            records.append(
                {
                    "vehicle_id": meta.vehicle_id,
                    "role": meta.role,
                    "branch_id": meta.branch_id,
                    "behavior_profile": meta.behavior_profile,
                    "segment_id": segment_id,
                    "lane_id": int(lane_id),
                    "speed": float(vehicle.speed),
                    "acceleration": float(runtime.acceleration_mps2 if runtime is not None else vehicle.action.get("acceleration", 0.0)),
                    "distance_traveled_m": float(runtime.distance_traveled_m if runtime is not None else 0.0),
                    "lane_changed_this_step": bool(runtime.lane_changed_this_step if runtime is not None else False),
                    "lane_change_times_s": list(runtime.lane_change_times_s if runtime is not None else []),
                    "free_flow_speed_mps": self._free_flow_speed_for_vehicle(vehicle),
                    "segment_density": self._segment_density(segment_id),
                    "crashed": bool(vehicle.crashed),
                }
            )
        return records

    def _segment_density(self, segment_id: str | None) -> float:
        if segment_id is None:
            return 0.0
        length_km = self.topology.segment_lengths[segment_id] / 1000.0
        count = sum(1 for vehicle in self._active_vehicles() if self.topology.segment_for_lane(vehicle.lane_index) == segment_id)
        return count / max(length_km, 1e-9)

    def _compute_step_metrics(
        self,
        segment_metrics: SegmentMetrics,
        diagnostics: Mapping[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        thresholds = self._metric_thresholds()
        active_records = self._active_vehicle_records()
        hard_braking_count = sum(
            1
            for record in active_records
            if float(record.get("acceleration", 0.0)) <= thresholds.hard_braking_mps2
        )
        hard_brakes_caused_by_av = sum(
            1
            for record in active_records
            if record.get("role") == "av" and float(record.get("acceleration", 0.0)) <= thresholds.hard_braking_mps2
        )
        lane_change_dwell_times = self._lane_change_dwell_times()
        self._recent_completion_times = [
            completion_time
            for completion_time in self._recent_completion_times
            if self._time - completion_time <= thresholds.throughput_window_s
        ]
        return compute_step_metrics(
            time_s=self._time,
            active_vehicle_records=active_records,
            segment_metrics=segment_metrics,
            diagnostics=diagnostics,
            completed_vehicle_count=self._completed_vehicle_count,
            recent_completion_times=self._recent_completion_times,
            completed_travel_times=self._completed_travel_times,
            branch_completed=self._per_branch_completed,
            branch_travel_times=self._completed_travel_times_by_branch,
            lane_change_dwell_times=lane_change_dwell_times,
            hard_braking_count=hard_braking_count,
            hard_brakes_caused_by_av=hard_brakes_caused_by_av,
            follower_delay_imposed_by_av=float(len(diagnostics.get("follower_disruption_blocked", []))),
            rear_ttc_after_av_lane_change_min=self._rear_ttc_after_av_lane_change_min(),
            thresholds=thresholds,
        )

    def _rear_ttc_after_av_lane_change_min(self) -> float:
        snapshots = self._vehicle_snapshots()
        by_id = {snapshot.vehicle_id: snapshot for snapshot in snapshots}
        values: list[float] = []
        for runtime in self._vehicle_runtime.values():
            if runtime.role != "av" or not runtime.lane_changed_this_step:
                continue
            ego = by_id.get(runtime.vehicle_id)
            if ego is None or ego.lane_index is None:
                continue
            rear_candidates = [
                snapshot
                for snapshot in snapshots
                if snapshot.vehicle_id != ego.vehicle_id
                and snapshot.lane_index == ego.lane_index
                and snapshot.longitudinal_m < ego.longitudinal_m
            ]
            if not rear_candidates:
                continue
            rear = max(rear_candidates, key=lambda snapshot: snapshot.longitudinal_m)
            rear_gap_m = max(0.0, ego.longitudinal_m - rear.longitudinal_m)
            values.append(
                rear_ttc(
                    rear_gap_m=rear_gap_m,
                    rear_relative_speed_mps=rear.speed_mps - ego.speed_mps,
                )
            )
        return float(min(values)) if values else float("inf")

    def _lane_change_dwell_times(self) -> list[float]:
        dwell_times: list[float] = []
        for runtime in self._vehicle_runtime.values():
            times = runtime.lane_change_times_s
            if len(times) < 2:
                continue
            dwell_times.extend([later - earlier for earlier, later in zip(times, times[1:])])
        return dwell_times

    def _branch_travel_time_mean(self) -> dict[str, float]:
        branch_ids = sorted(set(self._per_branch_completed) | set(self._completed_travel_times_by_branch))
        return {
            branch_id: float(np.mean(self._completed_travel_times_by_branch.get(branch_id, [])))
            if self._completed_travel_times_by_branch.get(branch_id)
            else 0.0
            for branch_id in branch_ids
        }
