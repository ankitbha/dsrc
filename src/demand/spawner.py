from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from src.demand.demand_profiles import DemandProfile
from src.demand.route_sampler import BranchRoute, RoutePlan, road_route_to_destination
from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec
from src.vehicles import HumanBehaviorModel, load_human_behavior_model, apply_human_behavior_profile

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex, Road
from highway_env.vehicle.behavior import IDMVehicle
from src.vehicles.merge_aware import MergeAwareIDMVehicle
from src.vehicles.safe_controlled import SafeControlledVehicle
from highway_env.vehicle.controller import ControlledVehicle


VehicleFactory = Callable[[str, str, ControlledVehicle, BranchRoute, str | None], str]


@dataclass(frozen=True)
class DemandSpawnResult:
    spawned: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


class DemandSpawner:
    def __init__(
        self,
        profile: DemandProfile,
        route_plan: RoutePlan,
        topology: TopologySpec,
        rng: np.random.RandomState,
        human_behavior_model: HumanBehaviorModel | None = None,
    ) -> None:
        self.profile = profile
        self.route_plan = route_plan
        self.topology = topology
        self.rng = rng
        self.human_behavior_model = human_behavior_model or load_human_behavior_model(None)
        self._lane_cursor: dict[str, int] = {branch.branch_id: 0 for branch in route_plan.branches}

    def reset(self) -> None:
        self._lane_cursor = {branch.branch_id: 0 for branch in self.route_plan.branches}

    def spawn_due(
        self,
        road: Road,
        *,
        time_s: float,
        dt_s: float,
        register_vehicle: VehicleFactory,
    ) -> DemandSpawnResult:
        if not self.profile.enabled or not self.route_plan.enabled:
            return DemandSpawnResult(spawned=[], skipped=[])

        rate_per_step = self.profile.vehicles_per_hour_at(time_s) * dt_s / 3600.0
        count = int(self.rng.poisson(rate_per_step))
        spawned: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for _ in range(count):
            branch = self._sample_branch()
            role = "av" if self.rng.uniform() < self.profile.av_penetration else "human"
            vehicle = self._try_spawn_on_branch(road, branch, role)
            if vehicle is None:
                skipped.append({"branch_id": branch.branch_id, "reason": "entry_gap"})
                continue
            behavior_profile = getattr(vehicle, "behavior_profile", None) if role == "human" else None
            vehicle_id = register_vehicle(role, branch.branch_id, vehicle, branch, behavior_profile)
            spawned.append(
                {
                    "vehicle_id": vehicle_id,
                    "role": role,
                    "behavior_profile": behavior_profile,
                    "branch_id": branch.branch_id,
                    "entry_segment": branch.entry_segment,
                    "lane_index": vehicle.lane_index,
                }
            )
        return DemandSpawnResult(spawned=spawned, skipped=skipped)

    def _sample_branch(self) -> BranchRoute:
        branches = {branch.branch_id: branch for branch in self.route_plan.branches}
        weights = self._weights_for_available_branches(branches)
        branch_ids = tuple(weights)
        probabilities = tuple(weights[branch_id] for branch_id in branch_ids)
        return branches[str(self.rng.choice(branch_ids, p=probabilities))]

    def _weights_for_available_branches(self, branches: Mapping[str, BranchRoute]) -> dict[str, float]:
        raw = {branch_id: self.profile.branch_split.get(branch_id, 0.0) for branch_id in branches}
        if sum(raw.values()) <= 0:
            raw = {branch_id: 1.0 for branch_id in branches}
        total = sum(raw.values())
        return {branch_id: weight / total for branch_id, weight in raw.items()}

    def _try_spawn_on_branch(self, road: Road, branch: BranchRoute, role: str) -> ControlledVehicle | None:
        start_cursor = self._lane_cursor.get(branch.branch_id, 0)
        for offset in range(branch.lane_count):
            lane_id = (start_cursor + offset) % branch.lane_count
            lane_index = branch.lane_index(lane_id)
            if not self._lane_has_spawn_gap(road, lane_index):
                continue
            self._lane_cursor[branch.branch_id] = (lane_id + 1) % branch.lane_count
            return self._make_vehicle(road, branch, lane_index, role)
        self._lane_cursor[branch.branch_id] = (start_cursor + 1) % max(branch.lane_count, 1)
        return None

    def _lane_has_spawn_gap(self, road: Road, lane_index: LaneIndex) -> bool:
        min_gap = self.profile.spawn_min_gap_m
        for vehicle in road.vehicles:
            if vehicle.lane_index != lane_index:
                continue
            longitudinal, _ = road.network.get_lane(lane_index).local_coordinates(vehicle.position)
            if -vehicle.LENGTH <= longitudinal <= min_gap:
                return False
        return True

    def _make_vehicle(self, road: Road, branch: BranchRoute, lane_index: LaneIndex, role: str) -> ControlledVehicle:
        speed = self._sample_speed()
        # Merge-aware: plain IDM follows only within its own lane, and every node
        # in the tree topologies reduces lane count, so converging humans had no
        # mutual awareness at all. `isinstance(vehicle, IDMVehicle)` below still
        # holds, since this is a subclass.
        vehicle_cls = SafeControlledVehicle if role == "av" else MergeAwareIDMVehicle
        vehicle = vehicle_cls.make_on_lane(road, lane_index, longitudinal=0.0, speed=speed)
        vehicle.route = road_route_to_destination(lane_index, branch.destination, self.topology)
        if isinstance(vehicle, IDMVehicle):
            vehicle.enable_lane_change = self.topology.supports_lane_change
            profile_id = self.human_behavior_model.sample_profile_id(self.rng)
            profile = self.human_behavior_model.profile_for(profile_id)
            lane = road.network.get_lane(lane_index)
            apply_human_behavior_profile(
                vehicle,
                profile,
                base_target_speed_mps=speed,
                min_speed_mps=self.profile.speed_distribution.min_mps,
                max_speed_mps=self.profile.speed_distribution.max_mps,
                lane_speed_limit_mps=lane.speed_limit,
            )
        return vehicle

    def _sample_speed(self) -> float:
        speed_cfg = self.profile.speed_distribution
        if speed_cfg.std_mps == 0:
            speed = speed_cfg.mean_mps
        else:
            speed = float(self.rng.normal(speed_cfg.mean_mps, speed_cfg.std_mps))
        return float(np.clip(speed, speed_cfg.min_mps, speed_cfg.max_mps))
