#!/usr/bin/env python3
"""Sweep the segment/link match tolerance and report the match-count curve.

plan_task145_dsrc_policy_runtime.md step 3 asks to "run the matcher over
the 123 New Jersey HERE bodies; report how many of the 915 segments match
at each tolerance, and pick from the curve." That corpus does not exist
anywhere in this checkout: `plans/implementation_records.md` item 71
records it as 123 bodies collected on 2026-09-08, but the run directory
they were written to was never committed (this repo's `.gitignore` excludes
`logs/`, `outputs/` and `device-test-*/` other than one named exception),
and no copy of it is reachable from this machine.

There is a second reason the sweep this step asks for could not have been
informative even with the real corpus in hand: `data/mainz/mainz.net.xml`
carries `projParameter="!"` (see `scripts/export_dsrc_network.py`'s
docstring), so Mainz has no real-world geo-reference at all. Its polylines,
as exported, are a placement anchored in the New Jersey area, not a
measurement of where Mainz actually is. Matching real New Jersey HERE links
against Mainz's fabricated New-Jersey-area polylines would have measured
how often an arbitrary placement happens to sit near real roads -- not
"which HERE link belongs to which Mainz super-segment", because no such
correspondence exists to be measured (plan risk 4 names the weaker version
of this same problem).

What this script actually does, honestly labelled as provisional: it builds
a small SYNTHETIC stand-in corpus with the same v7-flow-body generator
idiom `deployment/jetson/tests/test_here_feed.py` already uses (`body`,
`stretch`, `offset`), scattered around Mainz's own synthetic segment
polylines at a range of controlled distances, and reports how many of the
12 Mainz segments acquire at least one matching link at each of several
tolerances. This shows the matcher's own sensitivity to the tolerance
parameter -- useful for picking a number that is not absurdly tight or
absurdly loose -- but it is not a calibration against real geometry, and
`perception.segment_state.SEGMENT_MATCH_TOLERANCE_M`'s own comment
reuses `sensors.here_feed.ASSOCIATION_RADIUS_M` (60.0 m) rather than any
value this sweep produces, because the physical reasoning that constant
already states (a link further than this describes a different road)
applies here directly and needs no synthetic sweep to justify it.

Usage:
  python3 scripts/measure_dsrc_match_tolerance.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deployment" / "jetson"))

from perception.segment_state import SegmentStateBuilder  # noqa: E402
from sensors.here_feed import FlowLink, FlowReading, Outcome  # noqa: E402

NETWORK_DEFINITION = REPO_ROOT / "specs" / "dsrc_network_mainz.json"
#: Offsets (metres, perpendicular to a segment's own polyline) at which a
#: synthetic link is placed, one per Mainz segment, cycling through this
#: list so different segments probe different distances.
OFFSETS_M = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 100.0, 150.0, 250.0, 400.0, 600.0, 900.0)
TOLERANCES_M = (15.0, 30.0, 45.0, 60.0, 90.0, 150.0, 250.0)


def offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lat + dlat, lon + dlon)


def synthetic_links(builder: SegmentStateBuilder) -> tuple[FlowLink, ...]:
    """One link per segment, offset a known distance from that segment's own
    first polyline point, cycling through OFFSETS_M so the corpus spans a
    range of distances rather than sitting at one."""
    links = []
    for index, points in enumerate(builder._segment_points):  # noqa: SLF001
        lat0, lon0 = points[0]
        lat1, lon1 = points[-1] if len(points) > 1 else points[0]
        d_lat, d_lon = offset(lat0, lon0, OFFSETS_M[index % len(OFFSETS_M)], 0.0)
        links.append(FlowLink(
            points=((d_lat, d_lon), (lat1 + (d_lat - lat0), lon1 + (d_lon - lon0))),
            speed_mps=20.0, free_flow_mps=25.0, jam_factor=3.0,
            confidence=0.9, traversability="open", length_m=200.0,
        ))
    return tuple(links)


def sweep() -> dict[float, int]:
    counts = {}
    for tolerance in TOLERANCES_M:
        builder = SegmentStateBuilder(NETWORK_DEFINITION, match_tolerance_m=tolerance)
        links = synthetic_links(builder)
        reading = FlowReading(outcome=Outcome.OK, response_age_s=1.0)
        result = builder.build(links, reading, t_mono=0.0)
        counts[tolerance] = sum(1 for count in result.matched_links if count > 0)
    return counts


def main() -> None:
    counts = sweep()
    print("tolerance_m  segments_matched_of_12   (synthetic corpus -- see module docstring)")
    for tolerance, matched in counts.items():
        print(f"{tolerance:>10.0f}   {matched:>18d}")


if __name__ == "__main__":
    main()
