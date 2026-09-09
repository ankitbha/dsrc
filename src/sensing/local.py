from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec
from src.safety import SafetyState
from src.safety.etiquette import is_low_speed_uncongested

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex


@dataclass(frozen=True)
class SensingConfig:
    range_m: float = 150.0
    latency_s: float = 0.0
    position_noise_std: float = 0.0
    speed_noise_std: float = 0.0
    density_bin_edges_veh_per_km: tuple[float, ...] = (12.0, 30.0)
    mean_speed_bin_edges_mps: tuple[float, ...] = (8.0, 18.0)
    queue_speed_mps: float = 5.0
    #: Present only what the deployed rig can sense. The parity ledger classifies
    #: six of the 39 observation slots as `structurally_absent` -- every rear-facing
    #: field, because the live vehicle list is forward-camera derived and there is
    #: no rear sensor -- and seven more as `substituted`, where the rig fills in a
    #: constant. With this off, an actor learns from 13 of 39 inputs the vehicle
    #: cannot produce, which is the largest single reason a policy would not
    #: transfer.
    #:
    #: Defaults off deliberately. Turning it on everywhere would change every
    #: existing measurement, and section C's sufficiency work needs the full vector
    #: to ablate against -- this flag IS that ablation. `mappo_deploysense.yaml`
    #: sets it, being the config that describes the deployed profile.
    deployed_fidelity: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> SensingConfig:
        cfg = config.get("sensing", config) if isinstance(config, Mapping) else {}
        if not isinstance(cfg, Mapping):
            cfg = {}
        return cls(
            range_m=max(0.0, float(cfg.get("range_m", 150.0))),
            latency_s=max(0.0, float(cfg.get("latency_s", 0.0))),
            position_noise_std=max(0.0, float(cfg.get("position_noise_std", 0.0))),
            speed_noise_std=max(0.0, float(cfg.get("speed_noise_std", 0.0))),
            density_bin_edges_veh_per_km=_float_tuple(cfg.get("density_bin_edges_veh_per_km", (12.0, 30.0))),
            mean_speed_bin_edges_mps=_float_tuple(cfg.get("mean_speed_bin_edges_mps", (8.0, 18.0))),
            queue_speed_mps=max(0.0, float(cfg.get("queue_speed_mps", 5.0))),
            deployed_fidelity=bool(cfg.get("deployed_fidelity", False)),
        )


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: str
    role: str
    segment_id: str | None
    lane_index: LaneIndex | None
    lane_id: int
    position: tuple[float, float]
    longitudinal_m: float
    speed_mps: float
    acceleration_mps2: float
    free_flow_speed_mps: float
    crashed: bool = False


@dataclass(frozen=True)
class LaneGapContext:
    leader_gap_m: float
    leader_relative_speed_mps: float
    follower_gap_m: float
    follower_relative_speed_mps: float
    target_lane_exists: bool
    target_lane_front_gap_m: float
    target_lane_front_relative_speed_mps: float
    target_lane_rear_gap_m: float
    target_lane_rear_relative_speed_mps: float
    target_lane_rear_required_decel_mps2: float
    local_vehicle_count: int
    local_av_count: int
    local_density_veh_per_km: float
    local_mean_speed_mps: float
    nearby_av_mean_speed_mps: float
    all_lanes_av_occupied: bool
    #: Distance along the route to the next node where two or more arcs join.
    #: A property of the road, not of any vehicle, so it is NOT limited by
    #: `range_m`: a driver knows a merge is coming without seeing anyone on it.
    distance_to_next_merge_m: float = float("inf")
    #: Distance to that same merge point of the nearest vehicle approaching it on
    #: a different arc. This IS a vehicle observation and is range-limited.
    #: Infinite when nobody is converging.
    merge_conflict_gap_m: float = float("inf")
    merge_conflict_relative_speed_mps: float = 0.0


@dataclass(frozen=True)
class SensingFrame:
    time_s: float
    snapshots: tuple[VehicleSnapshot, ...]


class SensingBuffer:
    def __init__(self) -> None:
        self._frames: deque[SensingFrame] = deque(maxlen=256)

    def reset(self) -> None:
        self._frames.clear()

    def record(self, time_s: float, snapshots: Sequence[VehicleSnapshot]) -> None:
        frame = SensingFrame(float(time_s), tuple(snapshots))
        if self._frames and self._frames[-1].time_s == frame.time_s:
            self._frames[-1] = frame
        else:
            self._frames.append(frame)

    def has_frame(self, time_s: float, snapshots: Sequence[VehicleSnapshot]) -> bool:
        frame = SensingFrame(float(time_s), tuple(snapshots))
        return bool(self._frames and self._frames[-1] == frame)

    def frame_for_latency(self, time_s: float, latency_s: float) -> SensingFrame | None:
        if not self._frames:
            return None
        target_time = float(time_s) - max(0.0, float(latency_s))
        selected = self._frames[0]
        for frame in self._frames:
            if frame.time_s <= target_time:
                selected = frame
            else:
                break
        return selected


class LocalObservationBuilder:
    def __init__(self, config: SensingConfig | None = None) -> None:
        self.config = config or SensingConfig()
        self.buffer = SensingBuffer()

    def reset(self, config: SensingConfig | None = None) -> None:
        if config is not None:
            self.config = config
        self.buffer.reset()

    def build_all(
        self,
        *,
        time_s: float,
        topology: TopologySpec,
        snapshots: Sequence[VehicleSnapshot],
        current_av_ids: Sequence[str],
        safety_states: Mapping[str, SafetyState],
        target_headways: Mapping[str, float],
        target_lanes: Mapping[str, LaneIndex | None],
        segment_metrics: Mapping[str, Mapping[str, Any]],
        constraints: Any,
        rng: np.random.RandomState,
    ) -> dict[str, dict[str, Any]]:
        self.buffer.record(time_s, snapshots)
        frame = self.buffer.frame_for_latency(time_s, self.config.latency_s)
        if frame is None:
            return {}
        by_id = {snapshot.vehicle_id: snapshot for snapshot in frame.snapshots}
        observations: dict[str, dict[str, Any]] = {}
        for agent_id in current_av_ids:
            ego = by_id.get(agent_id)
            if ego is None:
                continue
            target_lane = target_lanes.get(agent_id) or ego.lane_index
            observations[agent_id] = self.build_one(
                time_s=frame.time_s,
                ego=ego,
                snapshots=frame.snapshots,
                topology=topology,
                safety_state=safety_states.get(agent_id, SafetyState()),
                target_headway_s=float(target_headways.get(agent_id, 1.6)),
                target_lane=target_lane,
                segment_metrics=segment_metrics,
                constraints=constraints,
                rng=rng,
            )
        return observations

    def build_one(
        self,
        *,
        time_s: float,
        ego: VehicleSnapshot,
        snapshots: Sequence[VehicleSnapshot],
        topology: TopologySpec,
        safety_state: SafetyState,
        target_headway_s: float,
        target_lane: LaneIndex | None,
        segment_metrics: Mapping[str, Mapping[str, Any]],
        constraints: Any,
        rng: np.random.RandomState,
    ) -> dict[str, Any]:
        sensed = self._sensed_neighbors(ego, snapshots)
        measured = [self._measure_neighbor(ego, neighbor, rng) for neighbor in sensed]
        local_vehicle_count = len(measured)
        local_av = [neighbor for neighbor in measured if neighbor.snapshot.role == "av"]
        local_av_count = len(local_av)
        local_density = self._local_density(local_vehicle_count)
        local_speeds = [neighbor.speed_mps for neighbor in measured]
        local_mean_speed = _mean(local_speeds) if local_speeds else ego.speed_mps
        local_queue_estimate = sum(1 for neighbor in measured if neighbor.speed_mps < self.config.queue_speed_mps)

        # The same route-aware gaps the safety layer gets. Without this the
        # observation the policy trains on was still arc-blind while
        # `lane_gap_context` was not, so the actor and the safety layer disagreed
        # about where the traffic was.
        route_deltas = self._route_deltas(ego, measured, topology)
        same_lane = self._lane_gaps(ego, measured, ego.lane_index, route_deltas=route_deltas)
        left_lane = self._lane_gaps(ego, measured, _adjacent_lane(ego.lane_index, -1, topology))
        right_lane = self._lane_gaps(ego, measured, _adjacent_lane(ego.lane_index, 1, topology))
        target_lane_exists = target_lane is not None and target_lane in topology.road_network.lanes_dict()
        target_gaps = self._lane_gaps(ego, measured, target_lane if target_lane_exists else None)
        target_rear_decel = self._required_rear_decel(ego, target_gaps.rear_neighbor, target_gaps.rear_gap_m)

        nearby_av_density = self._local_density(local_av_count)
        nearby_av_mean_speed = _mean([neighbor.speed_mps for neighbor in local_av]) if local_av else ego.free_flow_speed_mps
        lane_distribution = _lane_distribution(local_av)
        segment_metric = segment_metrics.get(ego.segment_id or "", {})
        downstream_congestion = _downstream_congestion_estimate(segment_metric)
        merge_pressure = _merge_pressure(ego, topology, local_queue_estimate, local_vehicle_count)
        segment_target_speed = ego.free_flow_speed_mps if not local_av else nearby_av_mean_speed
        if not local_av:
            downstream_congestion = 0.0
            merge_pressure = 0.0
            segment_target_speed = ego.free_flow_speed_mps

        time_since_last_lane_change = (
            float("inf")
            if safety_state.last_lane_change_time_s is None
            else max(0.0, float(time_s) - safety_state.last_lane_change_time_s)
        )
        segment_density = float(segment_metric.get("density", 0.0))
        uncongested_low_speed = is_low_speed_uncongested(
            ego.speed_mps,
            ego.free_flow_speed_mps,
            segment_density,
            constraints,
        )
        ego_headway = (
            float("inf")
            if same_lane.front_gap_m == float("inf") or ego.speed_mps <= 0
            else max(0.0, same_lane.front_gap_m / max(ego.speed_mps, 1e-6))
        )
        cooperation = {
            "segment_target_speed": float(segment_target_speed),
            "merge_pressure": float(merge_pressure),
            "downstream_congestion_estimate": float(downstream_congestion),
        }
        observation = {
            "is_active": True,
            "ego_speed": float(ego.speed_mps),
            "ego_acceleration": float(ego.acceleration_mps2),
            "ego_lane": int(ego.lane_id),
            "ego_headway_s": ego_headway,
            "target_headway_s": float(target_headway_s),
            "time_since_last_lane_change": time_since_last_lane_change,
            "lane_changes_last_km": int(safety_state.lane_changes_last_km),
            "current_segment": ego.segment_id,
            # Deliberately 0.0, matching the live side, NOT an oversight. The
            # real distance is computed -- `_merge_context` returns it and the
            # safety layer could take it -- but `deployment/jetson`'s observation
            # builder has no map matching and sets this to 0.0 for sim parity, so
            # putting a real value here would train the policy on information the
            # deployed vehicle cannot measure. Task 47's parity ledger caught the
            # divergence immediately when it was tried.
            "distance_to_next_merge": 0.0,
            "distance_to_downstream_bottleneck": 0.0 if ego.segment_id in topology.bottleneck_segments else float("inf"),
            "leader_gap": same_lane.front_gap_m,
            "leader_relative_speed": same_lane.front_relative_speed_mps,
            "follower_gap": same_lane.rear_gap_m,
            "follower_relative_speed": same_lane.rear_relative_speed_mps,
            "left_lane_front_gap": left_lane.front_gap_m,
            "left_lane_rear_gap": left_lane.rear_gap_m,
            "right_lane_front_gap": right_lane.front_gap_m,
            "right_lane_rear_gap": right_lane.rear_gap_m,
            "target_lane_front_gap": target_gaps.front_gap_m,
            "target_lane_rear_gap": target_gaps.rear_gap_m,
            "target_lane_rear_required_decel": target_rear_decel,
            "downstream_congestion_estimate": cooperation["downstream_congestion_estimate"],
            "merge_pressure": cooperation["merge_pressure"],
            "segment_target_speed": cooperation["segment_target_speed"],
            "uncongested_low_speed_flag": uncongested_low_speed,
            "local_density_bin": _bin(local_density, self.config.density_bin_edges_veh_per_km),
            "local_mean_speed_bin": _bin(local_mean_speed, self.config.mean_speed_bin_edges_mps),
            "local_queue_estimate": int(local_queue_estimate),
            "active_vehicle_count_local": int(local_vehicle_count),
            "active_av_count_local": int(local_av_count),
            "nearby_av_count": int(local_av_count),
            "nearby_av_density": float(nearby_av_density if local_av else 0.0),
            "nearby_av_mean_speed": float(nearby_av_mean_speed),
            "nearby_av_lane_distribution": lane_distribution,
            "sensor": {
                "range_m": float(self.config.range_m),
                "latency_s": float(self.config.latency_s),
                "position_noise_std": float(self.config.position_noise_std),
                "speed_noise_std": float(self.config.speed_noise_std),
            },
            "cooperation": cooperation,
        }
        # The rig cannot sense some of the above, and this is where the
        # simulator stops pretending otherwise. Off by default; see the flag.
        if self.config.deployed_fidelity:
            return self._apply_deployed_fidelity(observation)
        return observation

    def lane_gap_context(
        self,
        *,
        ego_id: str,
        time_s: float,
        topology: TopologySpec,
        snapshots: Sequence[VehicleSnapshot],
        target_lane: LaneIndex | None,
        rng: np.random.RandomState,
    ) -> LaneGapContext:
        if not self.buffer.has_frame(time_s, snapshots):
            self.buffer.record(time_s, snapshots)
        by_id = {snapshot.vehicle_id: snapshot for snapshot in snapshots}
        ego = by_id[ego_id]
        measured = [self._measure_neighbor(ego, neighbor, rng) for neighbor in self._sensed_neighbors(ego, snapshots)]
        route_deltas = self._route_deltas(ego, measured, topology)
        same_lane = self._lane_gaps(ego, measured, ego.lane_index, route_deltas=route_deltas)
        distance_to_merge, merge_gap, merge_relative_speed = self._merge_context(ego, measured, topology)
        target_lane_exists = target_lane is not None and target_lane in topology.road_network.lanes_dict()
        target_gaps = self._lane_gaps(ego, measured, target_lane if target_lane_exists else None)
        local_av = [neighbor for neighbor in measured if neighbor.snapshot.role == "av"]
        local_speeds = [neighbor.speed_mps for neighbor in measured]
        local_mean_speed = _mean(local_speeds) if local_speeds else ego.speed_mps
        nearby_av_mean_speed = _mean([neighbor.speed_mps for neighbor in local_av]) if local_av else ego.free_flow_speed_mps
        return LaneGapContext(
            leader_gap_m=same_lane.front_gap_m,
            leader_relative_speed_mps=same_lane.front_relative_speed_mps,
            follower_gap_m=same_lane.rear_gap_m,
            follower_relative_speed_mps=same_lane.rear_relative_speed_mps,
            target_lane_exists=target_lane_exists,
            target_lane_front_gap_m=target_gaps.front_gap_m,
            target_lane_front_relative_speed_mps=target_gaps.front_relative_speed_mps,
            target_lane_rear_gap_m=target_gaps.rear_gap_m,
            target_lane_rear_relative_speed_mps=target_gaps.rear_relative_speed_mps,
            target_lane_rear_required_decel_mps2=self._required_rear_decel(ego, target_gaps.rear_neighbor, target_gaps.rear_gap_m),
            local_vehicle_count=len(measured),
            local_av_count=len(local_av),
            local_density_veh_per_km=self._local_density(len(measured)),
            local_mean_speed_mps=local_mean_speed,
            nearby_av_mean_speed_mps=nearby_av_mean_speed,
            all_lanes_av_occupied=_all_lanes_av_occupied(ego, local_av, topology),
            distance_to_next_merge_m=distance_to_merge,
            merge_conflict_gap_m=merge_gap,
            merge_conflict_relative_speed_mps=merge_relative_speed,
        )

    def _sensed_neighbors(
        self,
        ego: VehicleSnapshot,
        snapshots: Sequence[VehicleSnapshot],
    ) -> list[VehicleSnapshot]:
        return [
            snapshot
            for snapshot in snapshots
            if snapshot.vehicle_id != ego.vehicle_id
            and _euclidean_m(ego.position, snapshot.position) <= self.config.range_m
        ]

    def _measure_neighbor(
        self,
        ego: VehicleSnapshot,
        neighbor: VehicleSnapshot,
        rng: np.random.RandomState,
    ) -> MeasuredNeighbor:
        longitudinal_delta = neighbor.longitudinal_m - ego.longitudinal_m
        lateral_delta = 0.0
        if ego.lane_index is not None and neighbor.lane_index is not None and ego.lane_index[:2] == neighbor.lane_index[:2]:
            lateral_delta = float(neighbor.lane_id - ego.lane_id)
        if self.config.position_noise_std > 0:
            longitudinal_delta += float(rng.normal(0.0, self.config.position_noise_std))
        speed = neighbor.speed_mps
        if self.config.speed_noise_std > 0:
            speed += float(rng.normal(0.0, self.config.speed_noise_std))
        return MeasuredNeighbor(
            snapshot=neighbor,
            longitudinal_delta_m=longitudinal_delta,
            lateral_delta_lanes=lateral_delta,
            speed_mps=max(0.0, speed),
        )

    def _route_deltas(
        self,
        ego: VehicleSnapshot,
        measured: Sequence["MeasuredNeighbor"],
        topology: TopologySpec,
    ) -> dict[str, float]:
        """Signed distance to each neighbour that lies on the ego's own route.

        Positive ahead, negative behind. A neighbour on an arc the ego will drive
        onto, or has just driven off, gets a real distance here; one on a sibling
        arc or an adjacent lane is left out, because it is not a leader and
        treating it as one would corrupt headway control.
        """
        if ego.lane_index is None:
            return {}
        network = topology.road_network
        lanes = network.lanes_dict()
        if ego.lane_index not in lanes:
            return {}
        limit = float(self.config.range_m)
        forward = self._forward_lane_offsets(topology, ego.lane_index, ego.longitudinal_m, limit)
        backward = self._backward_lane_offsets(topology, ego.lane_index, ego.longitudinal_m, limit)
        deltas: dict[str, float] = {}
        for neighbor in measured:
            lane_index = neighbor.snapshot.lane_index
            if lane_index is None or lane_index == ego.lane_index or lane_index not in lanes:
                continue
            # Membership in one of the walks IS the route test: they follow the
            # network's successors, so a lane in neither is one the ego does not
            # drive on, and an adjacent lane is excluded for free.
            ahead = forward.get(lane_index)
            behind = backward.get(lane_index)
            length = float(network.get_lane(lane_index).length)
            # The measurement error this neighbour was already given. Reusing the
            # draw rather than taking another keeps the same error on the same
            # vehicle however its geometry was derived, and consumes no extra rng
            # so seeded runs stay reproducible.
            #
            # Without it a cross-arc gap came out noiseless while a same-arc gap
            # spread 3.3 m at the training configs' 1.5 m, so the actor's leader_gap
            # had a noise level that depended on which side of an arc boundary the
            # leader happened to be -- and every vehicle here crosses two per episode.
            noise = (float(neighbor.longitudinal_delta_m)
                     - (float(neighbor.snapshot.longitudinal_m) - float(ego.longitudinal_m)))
            candidates = []
            if ahead is not None:
                candidates.append(ahead + float(neighbor.snapshot.longitudinal_m) + noise)
            if behind is not None:
                candidates.append(-(behind + (length - float(neighbor.snapshot.longitudinal_m)) - noise))
            if not candidates:
                continue
            # A short circuit can put one lane both ahead and behind. Whichever is
            # nearer is the one the ego will meet, and taking `forward` regardless
            # reported a follower 75 m back as absent on the ring.
            deltas[neighbor.snapshot.vehicle_id] = min(candidates, key=abs)
        return deltas

    def _next_lane(self, network, lane_index):
        """The lane a vehicle on `lane_index` actually drives onto next.

        Asking the network rather than assuming the lane ordinal carries over. It
        does not: on `inverted_tree`, `a1..a3_entry` continue into `('b1','c',0)`
        but `a4..a6_entry` continue into `('b2','c',1)`, because the b2 arcs end at
        y = -14 and lane 1 of `b2->c` is the one that starts there. An
        ordinal-preserving rule made the ego blind to the lane it was about to
        enter on three of the six entry arcs, and handed it a headway target in a
        lane it never occupies.
        """
        try:
            lane = network.get_lane(lane_index)
            return network.next_lane(lane_index, route=None, position=lane.position(lane.length, 0))
        except Exception:  # noqa: BLE001 - a lane with no successor is normal at an exit
            return None

    #: What the rig puts in each field it cannot sense, from
    #: `deployment/jetson/perception/observation_builder.py`. Kept as data rather
    #: than scattered conditionals so a change on the rig is a change to one table.
    RIG_UNSENSED = {
        "ego_lane": 1,                              # cfg.assumed_lane
        "time_since_last_lane_change": float("inf"),
        "lane_changes_last_km": 0,
        "distance_to_downstream_bottleneck": float("inf"),
        "follower_gap": float("inf"),                # no rear sensing
        "follower_relative_speed": 0.0,
        "left_lane_rear_gap": float("inf"),
        "right_lane_rear_gap": float("inf"),
        "target_lane_rear_gap": float("inf"),
        "target_lane_rear_required_decel": 0.0,
    }

    def _apply_deployed_fidelity(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Replace what the rig cannot sense with what the rig reports instead.

        Only fields the parity ledger classifies as `structurally_absent` or
        `substituted` are touched. Everything the forward camera genuinely measures
        -- the leader gap, the ego's own speed and acceleration, the headway -- is
        left exactly as the simulator computed it, because the rig measures those
        too.
        """
        for field, value in self.RIG_UNSENSED.items():
            if field in observation:
                observation[field] = value
        return observation

    def _forward_lane_offsets(
        self,
        topology: TopologySpec,
        lane_index: LaneIndex,
        longitudinal_m: float,
        limit_m: float,
    ) -> dict[LaneIndex, float]:
        """Distance from the ego to the START of each lane ahead on its own route.

        `longitudinal_m` restarts at every arc, so a raw subtraction across a
        boundary is meaningless and the gap search used to require an identical
        `lane_index` -- which made a vehicle metres ahead on the next arc invisible
        while one hundreds of metres ahead on the ego's own arc was reported as the
        leader.

        Following the network's own successor gives one chain rather than a search,
        so there is no traversal order to get wrong and no distance-keyed visited
        set to round.
        """
        network = topology.road_network
        lanes = network.lanes_dict()
        if lane_index not in lanes:
            return {}
        offsets: dict[LaneIndex, float] = {}
        current = lane_index
        distance = float(network.get_lane(current).length) - float(longitudinal_m)
        seen = {current}
        while distance <= limit_m:
            nxt = self._next_lane(network, current)
            if nxt is None or nxt in seen or nxt not in lanes:
                break
            offsets[nxt] = distance
            seen.add(nxt)
            distance += float(network.get_lane(nxt).length)
            current = nxt
        return offsets

    def _backward_lane_offsets(
        self,
        topology: TopologySpec,
        lane_index: LaneIndex,
        longitudinal_m: float,
        limit_m: float,
    ) -> dict[LaneIndex, float]:
        """Distance from the END of each lane behind, to the ego.

        A predecessor is a lane whose own successor is the lane we are standing on,
        which is the same question as `_forward_lane_offsets` asks, reversed. Where
        several arcs feed one lane -- three do at each inner node here -- all of
        them are genuinely behind and all are recorded at that distance.
        """
        network = topology.road_network
        lanes = network.lanes_dict()
        if lane_index not in lanes:
            return {}
        offsets: dict[LaneIndex, float] = {}
        current = lane_index
        distance = float(longitudinal_m)
        seen = {current}
        while distance <= limit_m:
            predecessors = [
                li for li in lanes
                if li[1] == current[0] and self._next_lane(network, li) == current
            ]
            predecessors = [li for li in predecessors if li not in seen]
            if not predecessors:
                break
            for li in predecessors:
                if li not in offsets or distance < offsets[li]:
                    offsets[li] = distance
            seen.update(predecessors)
            current = predecessors[0]
            distance += float(network.get_lane(current).length)
        return offsets

    def _merge_context(
        self,
        ego: VehicleSnapshot,
        measured: Sequence["MeasuredNeighbor"],
        topology: TopologySpec,
    ) -> tuple[float, float, float]:
        """Distance to the next joining node, and the nearest vehicle converging on it.

        A vehicle on a sibling arc is on nobody's route: it is neither ahead nor
        behind, and reporting it as a leader would corrupt headway control, which
        is a different quantity. It still collides. On `inverted_tree` three entry
        arcs feed each of `b1` and `b2`, and 27 of 51 measured terminating
        collisions were between two vehicles on sibling arcs approaching the same
        node.
        """
        network = topology.road_network
        lanes = network.lanes_dict()
        if ego.lane_index is None or ego.lane_index not in lanes:
            return float("inf"), float("inf"), 0.0
        node = ego.lane_index[1]
        incoming_arcs = {(li[0], li[1]) for li in lanes if li[1] == node}
        ego_to_merge = float(network.get_lane(ego.lane_index).length) - float(ego.longitudinal_m)
        if len(incoming_arcs) < 2:
            # A node only one arc enters is not a merge, however close it is.
            return float("inf"), float("inf"), 0.0
        own_successor = self._next_lane(network, ego.lane_index)
        nearest_gap = float("inf")
        nearest_speed = 0.0
        for neighbor in measured:
            snapshot = neighbor.snapshot
            if snapshot.lane_index is None or snapshot.lane_index not in lanes:
                continue
            if snapshot.crashed:
                continue  # a wreck sits at the node forever; yielding to it never ends
            if snapshot.lane_index[1] != node:
                continue
            if snapshot.lane_index[0] == ego.lane_index[0]:
                continue  # same arc: it is a leader or a follower, not a conflict
            # Sharing a node is not converging: lanes that feed different successor
            # lanes never meet, and 48% of the conflicts computed at node c were of
            # that kind.
            if own_successor is not None and self._next_lane(network, snapshot.lane_index) != own_successor:
                continue
            to_merge = float(network.get_lane(snapshot.lane_index).length) - float(snapshot.longitudinal_m)
            if to_merge < 0:
                continue
            # Project both onto the shared node. A smaller distance to it means it
            # arrives first, so it is a leader at this following distance. Choosing
            # by distance-to-node instead picks the vehicle FURTHEST ahead, which is
            # what the first version of this did: it reported a conflict 85.7 m away
            # while the vehicle actually struck was 4.1 m away, alongside.
            projected_gap = ego_to_merge - to_merge
            if projected_gap <= 0:
                continue  # arrives behind the ego, so not a leader
            if projected_gap < nearest_gap:
                nearest_gap = projected_gap
                nearest_speed = float(neighbor.speed_mps) - float(ego.speed_mps)
        return ego_to_merge, nearest_gap, nearest_speed

    def _lane_gaps(
        self,
        ego: VehicleSnapshot,
        measured: Sequence[MeasuredNeighbor],
        lane_index: LaneIndex | None,
        route_deltas: Mapping[str, float] | None = None,
    ) -> LaneGaps:
        if lane_index is None:
            return LaneGaps()
        front: MeasuredNeighbor | None = None
        rear: MeasuredNeighbor | None = None
        front_gap = float("inf")
        rear_gap = float("inf")
        for neighbor in measured:
            if neighbor.snapshot.lane_index != lane_index:
                # On another arc. `route_deltas` holds a signed distance for the
                # ones that lie on the ego's route; anything absent is on a
                # sibling arc or an adjacent lane and is not a leader.
                if route_deltas is None:
                    continue
                delta = route_deltas.get(neighbor.snapshot.vehicle_id)
                if delta is None:
                    continue
                gap = delta
            else:
                gap = neighbor.longitudinal_delta_m
            if abs(gap) > self.config.range_m:
                # A route walk can reach past the sensing horizon; the horizon
                # still governs. Without this the AV would be clairvoyant across
                # arc boundaries but not within one, which is worse than either.
                continue
            if gap >= 0 and (front is None or gap < front_gap):
                front = neighbor
                front_gap = gap
            if gap < 0 and (rear is None or gap > -rear_gap):
                rear = neighbor
                rear_gap = -gap
        front_gap = max(0.0, front_gap) if front is not None else float("inf")
        rear_gap = max(0.0, rear_gap) if rear is not None else float("inf")
        return LaneGaps(
            front_gap_m=float(front_gap),
            front_relative_speed_mps=float(front.speed_mps - ego.speed_mps) if front is not None else 0.0,
            rear_gap_m=float(rear_gap),
            rear_relative_speed_mps=float(rear.speed_mps - ego.speed_mps) if rear is not None else 0.0,
            rear_neighbor=rear,
        )

    def _required_rear_decel(
        self,
        ego: VehicleSnapshot,
        rear: MeasuredNeighbor | None,
        rear_gap_m: float,
    ) -> float:
        if rear is None or not np.isfinite(rear_gap_m) or rear_gap_m <= 0:
            return 0.0
        closing_speed = max(0.0, rear.speed_mps - ego.speed_mps)
        if closing_speed <= 0:
            return 0.0
        return float((closing_speed**2) / max(2.0 * rear_gap_m, 1e-6))

    def _local_density(self, count: int) -> float:
        if self.config.range_m <= 0:
            return 0.0
        return float(count) / max((2.0 * self.config.range_m) / 1000.0, 1e-9)


@dataclass(frozen=True)
class MeasuredNeighbor:
    snapshot: VehicleSnapshot
    longitudinal_delta_m: float
    lateral_delta_lanes: float
    speed_mps: float


@dataclass(frozen=True)
class LaneGaps:
    front_gap_m: float = float("inf")
    front_relative_speed_mps: float = 0.0
    rear_gap_m: float = float("inf")
    rear_relative_speed_mps: float = 0.0
    rear_neighbor: MeasuredNeighbor | None = None


def _float_tuple(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    return tuple(sorted(float(value) for value in raw))


def _euclidean_m(position_a: tuple[float, float], position_b: tuple[float, float]) -> float:
    return float(((position_a[0] - position_b[0]) ** 2 + (position_a[1] - position_b[1]) ** 2) ** 0.5)


def _adjacent_lane(
    lane_index: LaneIndex | None,
    delta: int,
    topology: TopologySpec,
) -> LaneIndex | None:
    if lane_index is None:
        return None
    candidate = (lane_index[0], lane_index[1], lane_index[2] + delta)
    return candidate if candidate in topology.road_network.lanes_dict() else None


def _bin(value: float, edges: Sequence[float]) -> int:
    return int(sum(float(value) >= edge for edge in edges))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _lane_distribution(neighbors: Sequence[MeasuredNeighbor]) -> dict[str, float]:
    if not neighbors:
        return {}
    counts: dict[str, int] = {}
    for neighbor in neighbors:
        lane_id = str(neighbor.snapshot.lane_id)
        counts[lane_id] = counts.get(lane_id, 0) + 1
    total = len(neighbors)
    return {lane_id: count / total for lane_id, count in counts.items()}


def _downstream_congestion_estimate(segment_metric: Mapping[str, Any]) -> float:
    return float(max(0.0, min(1.0, float(segment_metric.get("jam_fraction", 0.0)))))


def _merge_pressure(
    ego: VehicleSnapshot,
    topology: TopologySpec,
    local_queue_estimate: int,
    local_vehicle_count: int,
) -> float:
    if not topology.merge_nodes and ego.segment_id not in topology.bottleneck_segments:
        return 0.0
    return float(local_queue_estimate / max(local_vehicle_count, 1))


def _all_lanes_av_occupied(
    ego: VehicleSnapshot,
    local_av: Sequence[MeasuredNeighbor],
    topology: TopologySpec,
) -> bool:
    if ego.segment_id is None:
        return False
    lane_count = int(topology.lane_counts.get(ego.segment_id, 1))
    occupied = {ego.lane_id}
    occupied.update(neighbor.snapshot.lane_id for neighbor in local_av if neighbor.snapshot.segment_id == ego.segment_id)
    return all(lane_id in occupied for lane_id in range(lane_count))
