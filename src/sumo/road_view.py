"""A SUMO network presented the way the sensing model already asks for a road.

`lane_gap_context` wants four things from `topology.road_network` -- the set of
lanes, a lane's length and geometry, a lane's successor, and the node graph -- plus
`topology.segment_for_lane`. Supplying those from SUMO keeps the sensing model, its
route-aware gap search and its merge projection exactly as written and as validated.

The alternative was to keep building the `highway_env` topology alongside the SUMO
one purely for its geometry. That would have been quicker and wrong: `highway_env`
used sine curves where SUMO uses straight edges, so a gap computed on one road and
applied to positions from the other is wrong by an amount that varies along the
arc -- the undiagnosable class of defect the generated network exists to avoid.

Lane identity is `(from_junction, to_junction, ordinal)`, the same tuple
`highway_env` used, because a SUMO edge also runs between two junctions.

**Correction 2026-09-09.** An earlier version of this docstring claimed a wrong
mapping would "silently move every gap the sensing model computes". That is false on
this path, and measuring it is what showed so: over a 200-step episode `position` is
called 1384 times, `local_coordinates` 0 times and `heading_at` 0 times, and every
`position` call comes from one line of `lane_gap_context` which passes the result
only as the `position=` argument of `next_lane` -- which ignores it. What the sensing
model reads from a lane is its `length`, and from the network, `next_lane`. The
geometry here is correct, with a round-trip error of 0.000 m at both ends of every
arc, and it is currently unused.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.sumo.network import SumoNetwork

LaneIndex = tuple[str, str, int]


@dataclass(frozen=True)
class SumoLane:
    """One lane, answering the geometry questions the sensing model asks.

    Geometry comes from the lane's shape rather than from its endpoints, so a lane
    that `netconvert` shortened at a junction reports the length vehicles actually
    drive.
    """

    index: LaneIndex
    length: float
    _shape: tuple[tuple[float, float], ...]

    @property
    def _shape_length(self) -> float:
        """How long the drawn shape is, which is not the lane's length.

        `netconvert` shortens a lane where it meets a junction, and the node
        geometry is a diagonal, so the shape of `a1->b1` measures 457.60 m against
        a declared 500.00 m. Distances are therefore expressed in the lane's
        declared length and mapped onto the shape proportionally.

        Using the shape length as the lane length instead would make every entry
        arc 8% shorter than the spec says, and capacity is proportional to length.
        """
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(self._shape, self._shape[1:])
        ) or 1.0

    def position(self, longitudinal: float, lateral: float = 0.0) -> tuple[float, float]:
        """The point `longitudinal` metres along the lane, offset `lateral` across it."""
        scale = self._shape_length / self.length if self.length > 0 else 1.0
        along = min(max(float(longitudinal), 0.0), self.length) * scale
        travelled = 0.0
        for (x0, y0), (x1, y1) in zip(self._shape, self._shape[1:]):
            segment = math.hypot(x1 - x0, y1 - y0)
            if segment <= 0.0:
                continue
            if travelled + segment >= along:
                fraction = (along - travelled) / segment
                x = x0 + (x1 - x0) * fraction
                y = y0 + (y1 - y0) * fraction
                if lateral:
                    nx, ny = -(y1 - y0) / segment, (x1 - x0) / segment
                    x += nx * lateral
                    y += ny * lateral
                return (x, y)
            travelled += segment
        return self._shape[-1]

    def local_coordinates(self, point: tuple[float, float]) -> tuple[float, float]:
        """Distance along the lane and offset across it, for a point near it.

        The inverse of `position`, and the sensing model converts between the two
        constantly, so a mismatch here would move every gap it computes.
        """
        px, py = float(point[0]), float(point[1])
        best_along = 0.0
        best_lateral = 0.0
        best_distance = float("inf")
        travelled = 0.0
        for (x0, y0), (x1, y1) in zip(self._shape, self._shape[1:]):
            dx, dy = x1 - x0, y1 - y0
            segment = math.hypot(dx, dy)
            if segment <= 0.0:
                continue
            # Closest point on this segment, clamped to its ends so a point beyond
            # the lane projects onto the nearest end rather than off the line.
            along_segment = ((px - x0) * dx + (py - y0) * dy) / (segment * segment)
            along_segment = min(max(along_segment, 0.0), 1.0)
            cx, cy = x0 + dx * along_segment, y0 + dy * along_segment
            distance = math.hypot(px - cx, py - cy)
            if distance < best_distance:
                best_distance = distance
                best_along = travelled + segment * along_segment
                best_lateral = (-(px - x0) * dy + (py - y0) * dx) / segment
            travelled += segment
        # Back into the lane's declared length, the units every gap is expressed in.
        scale = self.length / self._shape_length if self._shape_length > 0 else 1.0
        return (best_along * scale, best_lateral)

    def heading_at(self, longitudinal: float) -> float:
        scale = self._shape_length / self.length if self.length > 0 else 1.0
        along = min(max(float(longitudinal), 0.0), self.length) * scale
        travelled = 0.0
        for (x0, y0), (x1, y1) in zip(self._shape, self._shape[1:]):
            segment = math.hypot(x1 - x0, y1 - y0)
            if segment <= 0.0:
                continue
            if travelled + segment >= along:
                return math.atan2(y1 - y0, x1 - x0)
            travelled += segment
        x0, y0 = self._shape[-2]
        x1, y1 = self._shape[-1]
        return math.atan2(y1 - y0, x1 - x0)

    @property
    def speed_limit(self) -> float | None:
        return None


class SumoRoadNetwork:
    """The lane set, geometry, successors and node graph, read from a SUMO net."""

    def __init__(self, network: SumoNetwork) -> None:
        import sumolib

        self._net = sumolib.net.readNet(str(network.net_file))
        self._lanes: dict[LaneIndex, SumoLane] = {}
        self._edge_of_lane: dict[LaneIndex, str] = {}
        self.graph: dict[str, dict[str, list[Any]]] = {}
        for edge in self._net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            self.graph.setdefault(from_node, {})[to_node] = []
            for lane in edge.getLanes():
                index: LaneIndex = (from_node, to_node, lane.getIndex())
                self._lanes[index] = SumoLane(
                    index=index,
                    length=float(lane.getLength()),
                    _shape=tuple((float(x), float(y)) for x, y in lane.getShape()),
                )
                self._edge_of_lane[index] = edge_id

    def lanes_dict(self) -> Mapping[LaneIndex, SumoLane]:
        return dict(self._lanes)

    def lanes_list(self) -> list[SumoLane]:
        return list(self._lanes.values())

    def get_lane(self, index: LaneIndex) -> SumoLane:
        return self._lanes[tuple(index)]  # type: ignore[index]

    def edge_of_lane(self, index: LaneIndex) -> str | None:
        return self._edge_of_lane.get(tuple(index))  # type: ignore[arg-type]

    def next_lane(self, index: LaneIndex, route: Any = None,
                  position: Any = None, np_random: Any = None) -> LaneIndex | None:
        """The lane a vehicle drives onto next, or None at an exit.

        Taken from SUMO's own connections rather than inferred from lane ordinals.
        On the previous simulator an ordinal-preserving rule was wrong on half the
        topology and left the ego blind to the lane it was about to enter, so this
        asks the network instead of assuming.
        """
        edge_id = self.edge_of_lane(index)
        if edge_id is None:
            return None
        edge = self._net.getEdge(edge_id)
        ordinal = int(index[2])
        try:
            outgoing = edge.getLanes()[ordinal].getOutgoing()
        except (IndexError, AttributeError):
            outgoing = []
        for connection in outgoing:
            target = connection.getToLane()
            target_edge = target.getEdge()
            if target_edge.getID().startswith(":"):
                continue
            return (target_edge.getFromNode().getID(),
                    target_edge.getToNode().getID(),
                    target.getIndex())
        return None

    def all_side_lanes(self, index: LaneIndex) -> list[LaneIndex]:
        return [other for other in self._lanes if other[:2] == tuple(index)[:2]]

    def side_lanes(self, index: LaneIndex) -> list[LaneIndex]:
        return [other for other in self.all_side_lanes(index)
                if abs(other[2] - int(index[2])) == 1]


class SumoTopologyView:
    """What the sensing model receives in place of a `TopologySpec`."""

    def __init__(self, network: SumoNetwork) -> None:
        self.network = network
        self.topology_id = network.topology_id
        self.road_network = SumoRoadNetwork(network)
        self.supports_lane_change = True

    def segment_for_lane(self, index: LaneIndex) -> str | None:
        edge_id = self.road_network.edge_of_lane(index)
        if edge_id is None:
            return None
        return self.network.segment_for_edge(edge_id)

    @property
    def bottleneck_segments(self) -> tuple[str, ...]:
        """Segments with fewer lanes than every segment feeding them.

        Derived from the built network for the same reason `merge_nodes` is: a
        lane drop is a property of the road, and reading it from the config would
        let the two disagree. On `inverted_tree_bottleneck` this is
        `tree_bottleneck_d`, one lane below the two-lane trunk; on `inverted_tree`
        there is no lane drop and the result is empty.

        Two sensing fields read it -- `distance_to_downstream_bottleneck`, and the
        branch that decides whether cooperation is scored at all -- so returning an
        empty tuple unconditionally, as an earlier version did, made the bottleneck
        variant sense identically to the plain one.
        """
        counts = self.lane_counts
        upstream: dict[str, set[str]] = {}
        for segment, followers in self.downstream_segments().items():
            for follower in followers:
                upstream.setdefault(follower, set()).add(segment)
        bottlenecks = [
            segment for segment, feeders in upstream.items()
            if feeders and all(counts.get(segment, 0) < counts.get(f, 0) for f in feeders)
        ]
        return tuple(sorted(bottlenecks))

    def downstream_segments(self) -> dict[str, tuple[str, ...]]:
        """The segments a vehicle can reach directly from each segment.

        Taken from SUMO's own lane connections, the same source `next_lane` uses,
        so it cannot disagree with the road the vehicles drive on. The other
        simulator derives the same relation from its `segment_edges`; both feed the
        rolling-roadblock metric, which excuses AVs holding a clear segment slow
        when the next one is congested, and the two must agree on what "next" means.
        """
        following: dict[str, set[str]] = {}
        for index in self.road_network.lanes_dict():
            segment = self.segment_for_lane(index)
            if segment is None:
                continue
            following.setdefault(segment, set())
            successor = self.road_network.next_lane(index)
            if successor is None:
                continue
            downstream = self.segment_for_lane(successor)
            if downstream is not None and downstream != segment:
                following[segment].add(downstream)
        return {segment: tuple(sorted(values)) for segment, values in following.items()}

    @property
    def merge_nodes(self) -> tuple[str, ...]:
        """Junctions that more than one edge enters.

        Derived from the built network rather than read from the config, so it
        cannot disagree with the road vehicles actually drive on. On
        `inverted_tree` this is b1, b2 and c: three entries into each of the first
        two and two middles into the trunk.
        """
        incoming: dict[str, int] = {}
        for (_, to_node, ordinal) in self.road_network.lanes_dict():
            if ordinal != 0:
                continue  # count edges, not lanes
            incoming[to_node] = incoming.get(to_node, 0) + 1
        return tuple(sorted(node for node, count in incoming.items() if count > 1))

    @property
    def lane_counts(self) -> Mapping[str, int]:
        """Lanes per segment, as `netconvert` built them.

        Read by `_all_lanes_av_occupied`, which asks whether cooperating AVs hold
        every lane of a segment. Taken from the built network rather than the
        requested config, for the same reason the network module reads them back: a
        lane count that silently differs changes capacity.
        """
        counts: dict[str, int] = {}
        for index in self.road_network.lanes_dict():
            segment = self.segment_for_lane(index)
            if segment is None:
                continue
            counts[segment] = max(counts.get(segment, 0), int(index[2]) + 1)
        return counts
