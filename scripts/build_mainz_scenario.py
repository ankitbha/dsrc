#!/usr/bin/env python3
"""Build the SUMO Mainz scenario from the SRC paper's Vissim network.

    .venv/bin/python scripts/build_mainz_scenario.py

Reads `data/mainz/Mainz20base.inpx` and `data/mainz/rl_links_mainz.txt` and writes
`mainz.net.xml`, `mainz.rou.xml` and `mainz_segments.json` beside them.

Three things about the conversion are decisions rather than mechanics, and each is
visible in the output so it can be checked:

**Speed.** Every edge imports at 13.9 m/s (50 km/h), and the SRC action set tops out
at 60 km/h. In Vissim a desired speed may exceed the link's nominal speed; in SUMO
`setSpeed` is capped by `getAllowedSpeed`, so the top action would silently clamp and
become indistinguishable from releasing the vehicle. `--speed.factor 1.2` lifts the
network to 16.67 m/s so all three actions are expressible, and makes the released
state equal to the fastest action exactly as `def_speed = speeds[-1]` does in SRC.

**Split links.** A Vissim link with a connector attaching part-way along becomes two
SUMO edges, `N[0]` and `N[1]`. Link 206 is the only one here, and it lies on every
route, so a name-for-name mapping loses all eight of them. Link ids are expanded to
their SUMO parts everywhere.

**Dropped links.** netconvert refuses links 24 and 68. Both sit in super-segment 8 and
neither appears on any demand route, so they are dropped from that segment's
membership rather than silently counted as present.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "mainz"

# 50 km/h * 1.2 = 60 km/h, the fastest action in the SRC action set.
SPEED_FACTOR = 1.2
# Vissim numbers connectors from 10000 up in this file; they become SUMO's internal
# edges and never appear as named ones.
CONNECTOR_ID_BASE = 10000


def run_netconvert(inpx: Path, out: Path) -> None:
    binary = REPO_ROOT / ".venv" / "bin" / "netconvert"
    command = [
        str(binary if binary.exists() else "netconvert"),
        "--vissim-file", str(inpx),
        # Without this the imported z-offsets produce edge grades up to 923%, which
        # SUMO applies to acceleration.
        "--flatten",
        "--speed.factor", str(SPEED_FACTOR),
        "-o", str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"netconvert failed with code {result.returncode}")
    refused = re.findall(r"Will not build edge '(\d+)'", result.stderr)
    if refused:
        print(f"  netconvert refused {len(refused)} edges: {sorted(refused, key=int)}")


def edge_ids(net_file: Path) -> set[str]:
    root = ET.parse(net_file).getroot()
    return {
        edge.get("id")
        for edge in root.findall("edge")
        if edge.get("function") != "internal"
    }


def expand(link_no: str, present: set[str]) -> list[str]:
    """A Vissim link id as the SUMO edges it became, in order.

    Returns the empty list for a link netconvert did not build, so a caller that
    drops it does so knowingly rather than by producing a route with a hole in it.
    """
    if link_no in present:
        return [link_no]
    parts = sorted(
        (e for e in present if re.fullmatch(rf"{re.escape(link_no)}\[\d+\]", e)),
        key=lambda e: int(e.split("[")[1].rstrip("]")),
    )
    return parts


def outgoing(net_file: Path) -> dict[str, set[str]]:
    """Edge to the edges reachable from it in one step, via connections."""
    root = ET.parse(net_file).getroot()
    links: dict[str, set[str]] = {}
    for connection in root.findall("connection"):
        source, target = connection.get("from"), connection.get("to")
        if source and target and not source.startswith(":"):
            links.setdefault(source, set()).add(target)
    return links


def demand(inpx_root: ET.Element) -> dict[str, list[tuple[float, float, float]]]:
    """Entry link to its `(start_s, end_s, veh/h)` intervals.

    The demand is NOT constant and reading only its first interval more than doubles
    it: every entry runs at its stated volume for the first 1,200 s and at zero
    afterwards, so the episode is a twenty-minute surge followed by a drain. Holding
    the opening volume for the whole 2,500 s offers 12,500 vehicles where the scenario
    offers 6,000.
    """
    schedule: dict[str, list[tuple[float, float, float]]] = {}
    for vehicle_input in inpx_root.find("vehicleInputs").findall("vehicleInput"):
        points = sorted(
            (float(entry.get("timeInt").split()[1]) / 1000.0, float(entry.get("volume")))
            for entry in vehicle_input.find("timeIntVehVols").findall("timeIntervalVehVolume")
        )
        if not any(volume > 0.0 for _, volume in points):
            continue
        intervals = []
        for index, (start, volume) in enumerate(points):
            end = points[index + 1][0] if index + 1 < len(points) else float("inf")
            if volume > 0.0:
                intervals.append((start, end, volume))
        schedule[vehicle_input.get("link")] = intervals
    return schedule


def routes(inpx_root: ET.Element, present: set[str], entries: set[str]) -> dict[str, list[str]]:
    """Entry link to its route as SUMO edges.

    Every route in this network ends at link 218, so one route per entry describes the
    demand completely and no route choice has to be modelled.
    """
    built: dict[str, list[str]] = {}
    decisions = inpx_root.find("vehicleRoutingDecisionsStatic")
    for decision in decisions.findall("vehicleRoutingDecisionStatic"):
        entry = decision.get("link")
        if entry not in entries or entry in built:
            continue
        static = decision.find("vehRoutSta")
        if static is None:
            continue
        for route in static.findall("vehicleRouteStatic"):
            # THE ROUTE STARTS AT THE ENTRY LINK. A Vissim routing decision sits ON a
            # link at some position along it and its `linkSeq` is the path onward from
            # there, so the sequence alone omits the link the vehicles are generated
            # on. Dropping it inserted traffic directly onto the next edge, which for
            # three of the eight entries is a single lane where the entry link has two
            # or three, throttling insertion and leaving thousands of vehicles queued
            # outside a network that never filled.
            sequence = [entry] + [
                ref.get("key") for ref in route.find("linkSeq").findall("intObjectRef")
            ]
            path = [edge for key in sequence for edge in expand(key, present)]
            if path:
                built[entry] = path
                break
    return built


def check_connected(path: list[str], links: dict[str, set[str]]) -> list[tuple[str, str]]:
    return [
        (path[i], path[i + 1])
        for i in range(len(path) - 1)
        if path[i + 1] not in links.get(path[i], set())
    ]


def vtype_lines() -> list[str]:
    """Vehicle types carrying the paper's own Vissim calibration.

    SUMO's default Krauss model computes a collision-free safe speed exactly and
    recovers from a disturbance immediately, so it produces no capacity drop and
    roughly twice the capacity this network should have. `w99_calibrated` transcribes
    Appendix A of the SRC paper, which is the calibration its own Vissim runs used, so
    using it is part of reproducing the paper rather than a tuning choice of ours.
    """
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "human_models" / "w99_calibrated.yaml").read_text())
    sumo = config["sumo"]
    attributes = " ".join(
        f'{name}="{sumo[name]}"' for name in
        ("cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9")
    )
    common = (f'carFollowModel="{sumo["car_following_model"]}" '
              f'minGap="{sumo["min_gap_m"]}" {attributes} maxSpeed="16.67"')
    return [
        f'  <vType id="human" {common}/>',
        f'  <vType id="av" {common} color="1,0,0"/>',
    ]


def write_routes(out_path: Path, built: dict[str, list[str]], volumes: dict[str, float],
                 duration_s: float, av_fraction: float, seed: int) -> int:
    """One explicit departure per vehicle, evenly spaced, sorted by time.

    Explicit vehicles rather than `<flow>` so a vehicle's id encodes whether it is an
    AV: SRC selects its controlled subset by vehicle number, and reproducing that here
    keeps the penetration deterministic and inspectable instead of resting on a draw.
    """
    lines = ["<routes>"] + vtype_lines()
    for entry, edges in sorted(built.items(), key=lambda kv: int(kv[0])):
        lines.append(f'  <route id="r{entry}" edges="{" ".join(edges)}"/>')
    departures: list[tuple[float, str, str]] = []
    for entry, intervals in sorted(volumes.items(), key=lambda kv: int(kv[0])):
        if entry not in built:
            continue
        for start, end, rate in intervals:
            headway = 3600.0 / rate
            time = start + headway / 2.0
            limit = min(end, duration_s)
            while time < limit:
                departures.append((time, entry, f"v{entry}_{len(departures)}"))
                time += headway
    departures.sort()
    for index, (time, entry, vehicle_id) in enumerate(departures):
        kind = "av" if (index % round(1.0 / av_fraction) == 0) else "human"
        lines.append(
            f'  <vehicle id="{vehicle_id}" type="{kind}" route="r{entry}" '
            f'depart="{time:.2f}" departSpeed="max" departLane="best"/>'
        )
    lines.append("</routes>")
    out_path.write_text("\n".join(lines) + "\n")
    return len(departures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=2500.0,
                        help="SRC's episode length; End_of_simulation in its train.py")
    # The published RL.py controls every second vehicle, but that script produces the
    # paper's ABLATION row, not its main table. The main table is at full penetration,
    # so matching the headline result means 1.0 and reading the code would have given
    # the wrong answer.
    parser.add_argument("--av-fraction", type=float, default=1.0,
                        help="1.0 for the paper's main table; its ablation uses 0.5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--demand-duration-s", type=float, default=None,
                        help="hold the opening volume for this long instead of using "
                             "the .inpx intervals; the published scenario surges for "
                             "1,200 s and then stops, which leaves the rest of the "
                             "episode draining rather than at an operating point")
    parser.add_argument("--demand-scale", type=float, default=1.0,
                        help="multiply every entry volume, to place the network at a "
                             "chosen point on its flow-density curve")
    args = parser.parse_args()

    inpx = DATA / "Mainz20base.inpx"
    net_file = DATA / "mainz.net.xml"
    if not inpx.exists():
        raise SystemExit(f"missing {inpx}; see data/mainz in .gitignore for its origin")

    print("building the network")
    run_netconvert(inpx, net_file)
    present = edge_ids(net_file)
    links = outgoing(net_file)
    print(f"  {len(present)} named edges")

    root = ET.parse(inpx).getroot()
    volumes = demand(root)

    built = routes(root, present, set(volumes))
    for entry, path in sorted(built.items(), key=lambda kv: int(kv[0])):
        breaks = check_connected(path, links)
        status = "ok" if not breaks else f"BREAKS {breaks[:2]}"
        print(f"  route from {entry}: {len(path)} edges, {status}")
    if len(built) != len(volumes):
        raise SystemExit(f"only {len(built)} of {len(volumes)} entries have a route")

    if args.demand_duration_s is not None:
        volumes = {entry: [(0.0, args.demand_duration_s, intervals[0][2])]
                   for entry, intervals in volumes.items()}
    if args.demand_scale != 1.0:
        volumes = {entry: [(a, b, v * args.demand_scale) for a, b, v in intervals]
                   for entry, intervals in volumes.items()}
    opening = sum(intervals[0][2] for intervals in volumes.values())
    last_end = min(max(i[1] for intervals in volumes.values() for i in intervals),
                   args.duration_s)
    print(f"  demand: {len(volumes)} entries, {opening:g} veh/h until {last_end:g} s, "
          f"then zero")
    count = write_routes(DATA / "mainz.rou.xml", built, volumes,
                         args.duration_s, args.av_fraction, args.seed)
    print(f"  {count} vehicle departures over {args.duration_s:g} s")

    groups = eval((DATA / "rl_links_mainz.txt").read_text())
    segments, dropped = [], []
    for group in groups:
        members: list[str] = []
        for link_no in group:
            parts = expand(str(link_no), present)
            if parts:
                members.extend(parts)
            else:
                dropped.append(str(link_no))
        segments.append(members)
    (DATA / "mainz_segments.json").write_text(json.dumps(segments, indent=1))
    covered = sum(len(s) for s in segments)
    print(f"  {len(segments)} super-segments over {covered} edges; "
          f"dropped {sorted(set(dropped), key=int)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
