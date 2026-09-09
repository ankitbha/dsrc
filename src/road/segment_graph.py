from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

from src.road.highway_imports import ensure_highway_env_importable

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex, RoadNetwork


DetectorLocations: TypeAlias = Mapping[str, tuple[float, ...]]
SegmentEdges: TypeAlias = Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class TopologySpec:
    """Project-owned topology metadata plus the HighwayEnv road network."""

    topology_id: str
    road_network: RoadNetwork
    segment_ids: tuple[str, ...]
    segment_lengths: Mapping[str, float]
    segment_edges: SegmentEdges
    entry_segments: tuple[str, ...]
    exit_segments: tuple[str, ...]
    merge_nodes: tuple[str, ...]
    detector_locations: DetectorLocations
    lane_counts: Mapping[str, int]
    bottleneck_segments: tuple[str, ...] = ()
    lane_segments: Mapping[LaneIndex, str] | None = None
    supports_lane_change: bool = True
    metadata: Mapping[str, Any] | None = None

    def segment_for_lane(self, lane_index: LaneIndex | None) -> str | None:
        if lane_index is None:
            return None
        if self.lane_segments and lane_index in self.lane_segments:
            return self.lane_segments[lane_index]
        edge = f"{lane_index[0]}->{lane_index[1]}"
        for segment_id, edges in self.segment_edges.items():
            if edge in edges:
                return segment_id
        return None

    def downstream_segments(self) -> Mapping[str, tuple[str, ...]]:
        """The segments a vehicle can reach directly from each segment.

        Derived from `segment_edges` rather than declared, so it cannot disagree
        with the road: an edge is written "from->to", so segment Y is downstream of
        segment X when some edge of X ends at the node some edge of Y starts from.

        Read by the rolling-roadblock metric, which excuses AVs holding a clear
        segment slow when the next one is congested. Without it that metric cannot
        tell speed metering from obstruction.
        """
        starts: dict[str, list[str]] = {}
        ends: dict[str, list[str]] = {}
        for segment_id, edges in self.segment_edges.items():
            for edge in edges:
                if "->" not in edge:
                    continue
                origin, destination = edge.split("->", 1)
                starts.setdefault(origin, []).append(segment_id)
                ends.setdefault(destination, []).append(segment_id)
        downstream: dict[str, set[str]] = {segment_id: set() for segment_id in self.segment_ids}
        for node, upstream_segments in ends.items():
            for upstream_segment in upstream_segments:
                for candidate in starts.get(node, ()):
                    if candidate != upstream_segment:
                        downstream.setdefault(upstream_segment, set()).add(candidate)
        return {segment_id: tuple(sorted(values)) for segment_id, values in downstream.items()}

    def validate(self) -> None:
        missing_lengths = [segment_id for segment_id in self.segment_ids if segment_id not in self.segment_lengths]
        if missing_lengths:
            raise ValueError(f"missing segment lengths for {missing_lengths}")
        non_positive = [segment_id for segment_id, length in self.segment_lengths.items() if length <= 0]
        if non_positive:
            raise ValueError(f"segments must have positive lengths: {non_positive}")
        unknown_detector_segments = [segment_id for segment_id in self.detector_locations if segment_id not in self.segment_ids]
        if unknown_detector_segments:
            raise ValueError(f"detectors reference unknown segments: {unknown_detector_segments}")
        for segment_id, locations in self.detector_locations.items():
            length = self.segment_lengths[segment_id]
            bad = [location for location in locations if location < 0 or location > length]
            if bad:
                raise ValueError(f"detectors for {segment_id} outside [0, {length}]: {bad}")


def edge_id(start: str, end: str) -> str:
    return f"{start}->{end}"


def lane_segment_map(
    road_network: RoadNetwork,
    edge_segments: Mapping[tuple[str, str], str],
) -> dict[LaneIndex, str]:
    lane_segments: dict[LaneIndex, str] = {}
    for start, end_segments in road_network.graph.items():
        for end, lanes in end_segments.items():
            segment_id = edge_segments[(start, end)]
            for lane_id in range(len(lanes)):
                lane_segments[(start, end, lane_id)] = segment_id
    return lane_segments

