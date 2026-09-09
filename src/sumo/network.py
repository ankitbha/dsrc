"""Generate a SUMO network from the topology spec this project already has.

Generated rather than hand-written so `inverted_tree` has one definition. A second,
hand-maintained `.net.xml` would drift from the spec, and the drift would surface as
every gap the sensing model computes being slightly wrong -- which is close to
undiagnosable, and is the shape of a defect this project has already paid for once
in the arc geometry.

**Edge ids are the segment ids.** `segment_for_edge` is then an identity on the
named edges rather than a lookup table, which removes a whole class of mapping bug.
Junction-internal edges, which SUMO names with a leading colon, are the exception
and map to no segment.

**Lengths are set explicitly** with the edge `length` attribute rather than left to
the node geometry, so an edge is exactly as long as the spec says whatever the
diagonal between its nodes happens to be. Capacity is the quantity the replication
turns on, and it is proportional to length and lane count.
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sumolib import checkBinary

#: Where each node sits. Chosen only to be unambiguous for `netconvert`; the
#: distances vehicles actually travel come from the explicit edge lengths.
_NODE_XY = {
    "a1": (-500.0, 70.0), "a2": (-500.0, 30.0), "a3": (-500.0, -10.0),
    "a4": (-500.0, 10.0), "a5": (-500.0, -30.0), "a6": (-500.0, -70.0),
    "b1": (0.0, 30.0), "b2": (0.0, -30.0),
    "c": (600.0, 0.0), "d": (1200.0, 0.0), "exit": (1500.0, 0.0),
}

#: leaf -> the node it feeds. Three into each of b1 and b2, matching the spec's
#: `merge_nodes`.
_LEAF_TO_MIDDLE = {"a1": "b1", "a2": "b1", "a3": "b1",
                   "a4": "b2", "a5": "b2", "a6": "b2"}


@dataclass(frozen=True)
class SumoNetwork:
    """A built SUMO network plus the mapping back to this project's segment ids."""

    topology_id: str
    net_file: Path
    #: edge id -> segment id. Edge ids ARE segment ids, so this is an identity on
    #: the keys; it exists to make the direction explicit and to answer for
    #: junction-internal edges.
    edge_to_segment: Mapping[str, str]
    #: edge id -> lane count, read back from what netconvert actually built rather
    #: than from what was requested.
    _lane_counts: Mapping[str, int] = field(default_factory=dict)
    _entry_edges: frozenset[str] = frozenset()
    _middle_edges: frozenset[str] = frozenset()
    exit_edge: str = ""

    @classmethod
    def build(cls, topology_id: str, config: Mapping[str, Any], out_dir: Path) -> "SumoNetwork":
        road = config.get("road", config)
        lane_counts = road.get("lane_counts", {})
        leaves_lanes = int(lane_counts.get("leaves", 1))
        middle_lanes = int(lane_counts.get("middle", 2))
        trunk_lanes = int(lane_counts.get("trunk", 2))
        speed = float(road.get("speed_limit_mps", 30.0))
        lengths = road.get("segment_lengths", {})

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        wanted = set(_NODE_XY)
        if "tree_bottleneck_d" not in (road.get("segment_ids") or []):
            wanted.discard("d")   # no bottleneck: `d` would be an unreachable node
        nodes = "\n".join(
            f'  <node id="{name}" x="{x}" y="{y}"/>'
            for name, (x, y) in _NODE_XY.items() if name in wanted
        )
        (out_dir / "net.nod.xml").write_text(f"<nodes>\n{nodes}\n</nodes>\n")

        edges: list[str] = []
        entry: set[str] = set()
        middle: set[str] = set()
        for leaf, dest in _LEAF_TO_MIDDLE.items():
            segment = f"tree_leaf_{leaf}"
            entry.add(segment)
            edges.append(
                f'  <edge id="{segment}" from="{leaf}" to="{dest}" numLanes="{leaves_lanes}"'
                f' speed="{speed}" length="{float(lengths.get(segment, 500.0))}"/>'
            )
        for node in ("b1", "b2"):
            segment = f"tree_middle_{node}"
            middle.add(segment)
            edges.append(
                f'  <edge id="{segment}" from="{node}" to="c" numLanes="{middle_lanes}"'
                f' speed="{speed}" length="{float(lengths.get(segment, 600.0))}"/>'
            )
        # The trunk, and -- when the spec declares one -- a lane-dropping
        # bottleneck after it. Ignoring the layout here meant
        # `inverted_tree_bottleneck` built the plain network with no error, so
        # comparing the two topologies would have compared one with itself.
        layout = str(road.get("layout", topology_id))
        if layout not in {"inverted_tree", "inverted_tree_bottleneck"}:
            raise ValueError(
                f"unsupported layout {layout!r}: this generator builds inverted_tree "
                "and inverted_tree_bottleneck"
            )
        has_bottleneck = "tree_bottleneck_d" in (road.get("segment_ids") or [])
        trunk = "tree_trunk_c"
        trunk_to = "d" if has_bottleneck else "exit"
        edges.append(
            f'  <edge id="{trunk}" from="c" to="{trunk_to}" numLanes="{trunk_lanes}"'
            f' speed="{speed}" length="{float(lengths.get(trunk, 900.0))}"/>'
        )
        exit_edge = trunk
        if has_bottleneck:
            bottleneck = "tree_bottleneck_d"
            bottleneck_lanes = int(lane_counts.get("bottleneck", 1))
            edges.append(
                f'  <edge id="{bottleneck}" from="d" to="exit"'
                f' numLanes="{bottleneck_lanes}" speed="{speed}"'
                f' length="{float(lengths.get(bottleneck, 300.0))}"/>'
            )
            exit_edge = bottleneck
        (out_dir / "net.edg.xml").write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n")

        net_file = out_dir / "net.net.xml"
        result = subprocess.run(
            [checkBinary("netconvert"),
             "-n", str(out_dir / "net.nod.xml"),
             "-e", str(out_dir / "net.edg.xml"),
             "-o", str(net_file),
             # Deterministic output, so building twice gives the same file and a
             # network change is visible as a diff rather than as noise.
             "--no-warnings", "true",
             "--offset.disable-normalization", "true"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not net_file.exists():
            raise RuntimeError(f"netconvert failed: {result.stderr.strip()[:400]}")

        built = cls._read_lane_counts(net_file)
        segments = {edge: edge for edge in built}
        return cls(
            topology_id=topology_id,
            net_file=net_file,
            edge_to_segment=segments,
            _lane_counts=built,
            _entry_edges=frozenset(entry),
            _middle_edges=frozenset(middle),
            exit_edge=exit_edge,
        )

    @staticmethod
    def _read_lane_counts(net_file: Path) -> dict[str, int]:
        """Lane counts as netconvert built them, not as they were requested.

        Reading back rather than trusting the request, because a lane count that
        silently differs from the spec changes capacity, and capacity is what the
        replication measures.
        """
        import sumolib

        net = sumolib.net.readNet(str(net_file))
        return {
            edge.getID(): len(edge.getLanes())
            for edge in net.getEdges()
            if not edge.getID().startswith(":")
        }

    def entry_edges(self) -> frozenset[str]:
        return self._entry_edges

    def middle_edges(self) -> frozenset[str]:
        return self._middle_edges

    def lane_counts(self) -> Mapping[str, int]:
        return dict(self._lane_counts)

    def segment_for_edge(self, edge_id: str) -> str | None:
        """The project's segment id for a SUMO edge, or None for junction internals."""
        if edge_id.startswith(":"):
            return None
        return self.edge_to_segment.get(edge_id)

    def route_to_exit(self, from_edge: str) -> tuple[str, ...]:
        """Edges from `from_edge` to the exit, or an empty tuple if unreachable.

        Checked rather than assumed: SUMO discards a vehicle whose route cannot be
        completed, and it does so without failing the run, so an unreachable entry
        would show up only as demand quietly lower than configured.
        """
        import sumolib

        net = sumolib.net.readNet(str(self.net_file))
        try:
            start = net.getEdge(from_edge)
            target = net.getEdge(self.exit_edge)
        except Exception:  # noqa: BLE001 - an unknown edge id is a caller error
            return ()
        path, _ = net.getOptimalPath(start, target)
        if not path:
            return ()
        return tuple(edge.getID() for edge in path if not edge.getID().startswith(":"))
