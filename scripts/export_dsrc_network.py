#!/usr/bin/env python3
"""Generate specs/dsrc_network_mainz.json from the Mainz SUMO network.

Run once, on a machine with `sumolib` (no running SUMO instance needed --
`sumolib.net.readNet` parses the XML statically). The file it writes is a
frozen, committed artifact: `deployment/jetson/perception/segment_state.py`
loads it at runtime and never regenerates it, the way
`specs/transport_golden_frames.json` is read by `test_transport_golden.py`
but only ever written by `scripts/generate_transport_golden_frames.py`.

Per super-segment, in the order `data/mainz/mainz_segments.json` gives them:
the segment id, its ordered edge ids, each edge's lane count / length /
speed limit (read with `sumolib`, mirroring `MainzEnv._load_static`), each
edge's polyline, and the segment-level statics
(`speed_limit_kmh`/`length_km`/`lanes`) computed the same way
`MainzEnv.observe` aggregates them (`_nanmean`, unweighted, for the first and
third; summed then divided by 1000 for the second) so this file's own
numbers are the ones `test_export_dsrc_network.py` checks against
section 1.3 of the plan.

**The polylines are not a real-world location.** `mainz.net.xml`'s
`<location>` tag carries `projParameter="!"` -- SUMO's own marker for "no
geo-projection available" -- so `sumolib`'s `convertXY2LonLat` refuses this
network outright (`RuntimeError: Network does not provide geo-projection`).
Mainz is a Vissim import positioned in an arbitrary local Cartesian frame,
never given a real-world anchor at any point in this project. There is
therefore no way to compute "the WGS84 polyline of each edge" as a
conversion; what follows is a placement, not a measurement.

This script picks ANCHOR_LAT/ANCHOR_LON -- the same New Jersey-area point
already used for GPS fixtures elsewhere in this deployment
(`tests/test_pipeline_smoke.py`'s `test_fuse_reports_the_builders_own_
measurement` uses `lat=40.0, lon=-74.0`) -- and maps the network's own
local x/y onto it with a flat equirectangular tangent-plane projection
(`geo.py`'s own conversion, inverted), centered on the network's bounding
box. This gives every edge a stable, internally-consistent WGS84 polyline
that the matching code in `segment_state.py` can run against, but it
carries no claim that Mainz sits at that point on Earth, and matching these
polylines against real HERE links collected elsewhere measures the
matcher's behaviour on synthetic geometry, not a real correspondence. See
`plans/plan_task145_dsrc_policy_runtime.md` risk 4, which already names the
weaker version of this problem (no HERE response was ever collected over
Mainz); the network's own coordinates being fabricated is the fuller
version, recorded here because the plan text ("the WGS84 polyline of each
edge") reads as a straightforward extraction and is not one.

Usage:
  python3 scripts/export_dsrc_network.py
  python3 scripts/export_dsrc_network.py --out specs/dsrc_network_mainz.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from geo import DEG_M  # noqa: E402

import sumolib  # noqa: E402

NETWORK_ID = "mainz"
NET_FILE = REPO_ROOT / "data" / "mainz" / "mainz.net.xml"
SEGMENT_FILE = REPO_ROOT / "data" / "mainz" / "mainz_segments.json"
DEFAULT_OUT = REPO_ROOT / "specs" / "dsrc_network_mainz.json"

#: See the module docstring: Mainz carries no real geo-reference, so this is
#: a placement, chosen to sit in the same New Jersey area already used for
#: GPS fixtures elsewhere in `deployment/jetson/tests`, not a measurement.
ANCHOR_LAT = 40.0
ANCHOR_LON = -74.0

FEATURE_NAMES = ("speed", "free_flow", "jam_factor", "lanes", "length_km")
ACTION_FRACTIONS = (0.5, 0.75, 1.0)


def _local_to_lonlat(
    x: float, y: float, origin_x: float, origin_y: float,
) -> tuple[float, float]:
    """Invert `geo.point_to_segment_m`'s local tangent-plane projection.

    `geo.py`'s own `xy()` closure maps (lat, lon) to metres east/north of an
    anchor with `((lon - lon0) * cos(lat0) * DEG_M, (lat - lat0) * DEG_M)`.
    This is that map's inverse, centered on `(origin_x, origin_y)` rather
    than on zero, so the network's own bounding-box centre lands on the
    anchor.
    """
    lat = ANCHOR_LAT + (y - origin_y) / DEG_M
    lon = ANCHOR_LON + (x - origin_x) / (DEG_M * math.cos(math.radians(ANCHOR_LAT)))
    return lat, lon


def build_network_definition(net_file: Path, segment_file: Path) -> dict:
    net = sumolib.net.readNet(str(net_file))
    xmin, ymin, xmax, ymax = net.getBoundary()
    origin_x, origin_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    segments: list[list[str]] = json.loads(segment_file.read_text())
    segment_records = []
    for index, edge_ids in enumerate(segments):
        edge_records = []
        for edge_id in edge_ids:
            edge = net.getEdge(edge_id)
            lane0 = edge.getLanes()[0]
            polyline = [
                list(_local_to_lonlat(x, y, origin_x, origin_y))
                for x, y in edge.getShape()
            ]
            edge_records.append({
                "edge_id": edge_id,
                "lanes": float(edge.getLaneNumber()),
                "length_m": float(lane0.getLength()),
                "speed_limit_kmh": float(lane0.getSpeed()) * 3.6,
                "polyline": polyline,
            })
        lanes = _nanmean([e["lanes"] for e in edge_records])
        speed_limit_kmh = _nanmean([e["speed_limit_kmh"] for e in edge_records])
        length_km = sum(e["length_m"] for e in edge_records) / 1000.0
        segment_records.append({
            "segment_id": str(index),
            "edge_ids": list(edge_ids),
            "lanes": lanes,
            "speed_limit_kmh": speed_limit_kmh,
            "length_km": length_km,
            "edges": edge_records,
        })

    return {
        "network_id": NETWORK_ID,
        "source_net_file": str(net_file.relative_to(REPO_ROOT)),
        "source_segment_file": str(segment_file.relative_to(REPO_ROOT)),
        "feature_names": list(FEATURE_NAMES),
        "action_fractions": list(ACTION_FRACTIONS),
        "geo_reference": {
            "kind": "synthetic_local_tangent_plane",
            "anchor_lat": ANCHOR_LAT,
            "anchor_lon": ANCHOR_LON,
            "note": (
                "mainz.net.xml carries projParameter='!' (no real geo-projection); "
                "these coordinates are a fabricated placement anchored in the New "
                "Jersey area already used by this deployment's test fixtures, not a "
                "measurement of Mainz's real-world location. See this script's "
                "module docstring and plan_task145 risk 4."
            ),
        },
        "segments": segment_records,
    }


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-file", type=Path, default=NET_FILE)
    parser.add_argument("--segment-file", type=Path, default=SEGMENT_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    definition = build_network_definition(args.net_file, args.segment_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(definition, indent=2) + "\n")
    print(f"wrote {args.out} ({len(definition['segments'])} segments)")


if __name__ == "__main__":
    main()
