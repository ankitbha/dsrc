from __future__ import annotations

import math
from typing import Mapping

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec, edge_id, lane_segment_map

ensure_highway_env_importable()

from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import RoadNetwork


DEFAULT_LEAF_M = 500.0
DEFAULT_MIDDLE_M = 600.0
DEFAULT_TRUNK_M = 600.0
DEFAULT_BOTTLENECK_M = 300.0
DEFAULT_SPEED_LIMIT_MPS = 30.0


def _add_sine(net: RoadNetwork, start: str, end: str, p0: list[float], p1: list[float], length_hint: float, speed_limit: float) -> None:
    net.add_lane(
        start,
        end,
        SineLane(
            p0,
            p1,
            amplitude=2.0,
            pulsation=math.pi / max(length_hint, 1.0),
            # Phase 0, not pi/2. At pi/2 the sine is at plus or minus its full
            # amplitude AT the arc ends, so a lane finished 2 m off the point it was
            # built to reach and did not meet its successor. At 0 the offset is zero
            # at both ends and the arc still bulges 2 m in the middle, which is what
            # the sine was for.
            phase=0.0,
            line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
            speed_limit=speed_limit,
        ),
    )


def build_inverted_tree_topology(config: Mapping | None = None, *, bottleneck: bool = False) -> TopologySpec:
    cfg = dict(config or {})
    leaf = float(cfg.get("leaf_m", DEFAULT_LEAF_M))
    middle = float(cfg.get("middle_m", DEFAULT_MIDDLE_M))
    trunk = float(cfg.get("trunk_m", DEFAULT_TRUNK_M))
    bottleneck_length = float(cfg.get("bottleneck_m", DEFAULT_BOTTLENECK_M))
    speed_limit = float(cfg.get("speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS))

    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    leaf_y = {
        "a1": 24.0,
        "a2": 18.0,
        "a3": 12.0,
        "a4": -12.0,
        "a5": -18.0,
        "a6": -24.0,
    }
    lane_width = StraightLane.DEFAULT_WIDTH
    b1_starts = (12.0, 12.0 + lane_width)
    b2_starts = (-12.0, -12.0 - lane_width)
    # The middle arcs end exactly where the trunk lanes begin, so lane k of both
    # b1 and b2 feeds trunk lane k. Previously they ended at (4, 8) and (0, -4)
    # against trunk lanes at 0 and 4, so two of the four had nowhere to go and a
    # vehicle leaving ('b2','c',1) was placed ten metres sideways.
    trunk_ys = (0.0, lane_width)
    b1_ends = trunk_ys
    b2_ends = trunk_ys

    # Each leaf aims at an actual lane start of the node it feeds, alternating
    # between the two so three leaves distribute over two lanes rather than all
    # converging on one point. Aiming them at a single y put three arcs within
    # 0.01 m of each other, which is a collision by geometry rather than a merge.
    for index, (leaf_id, y) in enumerate(leaf_y.items(), start=1):
        merge_node = "b1" if index <= 3 else "b2"
        starts = b1_starts if index <= 3 else b2_starts
        target_y = starts[(index - 1) % 2]
        _add_sine(net, f"{leaf_id}_entry", merge_node, [0.0, y], [leaf, target_y], leaf, speed_limit)

    for lane_id in range(2):
        net.add_lane(
            "b1",
            "c",
            SineLane(
                [leaf, b1_starts[lane_id]],
                [leaf + middle, b1_ends[lane_id]],
                amplitude=2.0,
                pulsation=math.pi / middle,
                phase=0.0,  # zero lateral offset at both ends; see _add_sine
                line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                speed_limit=speed_limit,
            ),
        )
        net.add_lane(
            "b2",
            "c",
            SineLane(
                [leaf, b2_starts[lane_id]],
                [leaf + middle, b2_ends[lane_id]],
                amplitude=2.0,
                pulsation=math.pi / middle,
                phase=0.0,  # zero lateral offset at both ends; see _add_sine
                line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                speed_limit=speed_limit,
            ),
        )

    trunk_end = leaf + middle + trunk
    if bottleneck:
        for lane_id, y in enumerate((0.0, lane_width)):
            # The dropped lane TAPERS to the surviving lane rather than ending
            # beside it. `d->exit` is a single lane at y = 0, so lane 1 ending at
            # y = 4 left a vehicle to be moved 4 m sideways at the node -- which is
            # how a lane drop was being modelled, and it is not how one is built.
            # Tapering makes the drop a merge the vehicles have to negotiate, which
            # is the phenomenon the bottleneck exists to create.
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [leaf + middle, y],
                    [trunk_end, 0.0 if lane_id else y],
                    line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                    speed_limit=speed_limit,
                ),
            )
        net.add_lane(
            "d",
            "exit",
            StraightLane(
                [trunk_end, 0.0],
                [trunk_end + bottleneck_length, 0.0],
                line_types=[c, c],
                speed_limit=0.8 * speed_limit,
            ),
        )
        trunk_segment_length = trunk
        edge_segments_extra = {
            ("c", "d"): "tree_trunk_c",
            ("d", "exit"): "tree_bottleneck_d",
        }
        exit_segments = ("tree_bottleneck_d",)
        bottleneck_segments = ("tree_bottleneck_d",)
    else:
        trunk_segment_length = trunk + bottleneck_length
        for lane_id, y in enumerate((0.0, lane_width)):
            net.add_lane(
                "c",
                "exit",
                StraightLane(
                    [leaf + middle, y],
                    [leaf + middle + trunk_segment_length, y],
                    line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                    speed_limit=speed_limit,
                ),
            )
        edge_segments_extra = {("c", "exit"): "tree_trunk_c"}
        exit_segments = ("tree_trunk_c",)
        bottleneck_segments = ()

    edge_segments: dict[tuple[str, str], str] = {}
    for leaf_id in leaf_y:
        edge_segments[(f"{leaf_id}_entry", "b1" if leaf_id in {"a1", "a2", "a3"} else "b2")] = f"tree_leaf_{leaf_id}"
    edge_segments.update(
        {
            ("b1", "c"): "tree_middle_b1",
            ("b2", "c"): "tree_middle_b2",
            **edge_segments_extra,
        }
    )
    base_segment_ids = (
        "tree_leaf_a1",
        "tree_leaf_a2",
        "tree_leaf_a3",
        "tree_leaf_a4",
        "tree_leaf_a5",
        "tree_leaf_a6",
        "tree_middle_b1",
        "tree_middle_b2",
        "tree_trunk_c",
    )
    segment_ids = (*base_segment_ids, "tree_bottleneck_d") if bottleneck else base_segment_ids
    spec = TopologySpec(
        topology_id="inverted_tree_bottleneck" if bottleneck else "inverted_tree",
        road_network=net,
        segment_ids=segment_ids,
        segment_lengths={
            **{f"tree_leaf_{leaf_id}": leaf for leaf_id in leaf_y},
            "tree_middle_b1": middle,
            "tree_middle_b2": middle,
            "tree_trunk_c": trunk_segment_length,
            **({"tree_bottleneck_d": bottleneck_length} if bottleneck else {}),
        },
        segment_edges={
            segment: tuple(edge_id(*edge) for edge, edge_segment in edge_segments.items() if edge_segment == segment)
            for segment in segment_ids
        },
        entry_segments=tuple(f"tree_leaf_{leaf_id}" for leaf_id in leaf_y),
        exit_segments=exit_segments,
        merge_nodes=("tree_merge_b1", "tree_merge_b2", "tree_merge_c"),
        detector_locations={
            **{f"tree_leaf_{leaf_id}": (leaf,) for leaf_id in leaf_y},
            "tree_middle_b1": (middle,),
            "tree_middle_b2": (middle,),
            "tree_trunk_c": (trunk_segment_length / 2.0,),
            **({"tree_bottleneck_d": (bottleneck_length,)} if bottleneck else {}),
        },
        lane_counts={
            **{f"tree_leaf_{leaf_id}": 1 for leaf_id in leaf_y},
            "tree_middle_b1": 2,
            "tree_middle_b2": 2,
            "tree_trunk_c": 2,
            **({"tree_bottleneck_d": 1} if bottleneck else {}),
        },
        bottleneck_segments=bottleneck_segments,
        lane_segments=lane_segment_map(net, edge_segments),
        supports_lane_change=True,
        metadata={
            "leaf_m": leaf,
            "middle_m": middle,
            "trunk_m": trunk,
            "bottleneck_m": bottleneck_length,
            "speed_limit_mps": speed_limit,
        },
    )
    spec.validate()
    return spec
