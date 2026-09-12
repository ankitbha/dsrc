#!/usr/bin/env python3
"""Build `inverted_tree_bottleneck` in the four files `MainzEnv` drives.

    .venv/bin/python scripts/build_tree_scenario.py

`SumoTopologyEnv` already generates this network, but with the observation and action
model of the earlier MAPPO leg: per-vehicle agents, not the per-super-segment Q-learning
SRC uses. Rather than port the controller a second time, this writes the tree into the
same artifacts `build_mainz_scenario.py` writes for Mainz -- a net, a route file, a
routes-only file, a departure schedule and a super-segment grouping -- so the same env
and the same training driver run on both networks and the two results are comparable.

The fleet is `eidm_reaction`, for the reason in that config's header: the paper's W99
calibration has no capacity drop on SUMO, and without one no controller can gain
throughput. The gate confirms it for this network specifically -- served flow peaks at
1,199 +/- 5 veh/h and falls to 1,140 +/- 20, a resolved 4.9%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loaders import load_named_config  # noqa: E402
from src.sumo.network import SumoNetwork  # noqa: E402

DATA = REPO_ROOT / "data" / "tree"


def vtype_lines(model: str) -> list[str]:
    """The same two types the Mainz builder writes, from the same config."""
    import yaml

    sumo = yaml.safe_load(
        (REPO_ROOT / "configs" / "human_models" / f"{model}.yaml").read_text())["sumo"]
    attributes = " ".join(
        f'{name}="{value}"' for name, value in sumo.items()
        if name not in ("car_following_model", "min_gap_m"))
    common = (f'carFollowModel="{sumo["car_following_model"]}" '
              f'minGap="{sumo["min_gap_m"]}" {attributes}')
    return [f'  <vType id="human" {common}/>',
            f'  <vType id="av" {common} color="1,0,0"/>']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="inverted_tree_bottleneck")
    parser.add_argument("--human-model", default="eidm_reaction")
    parser.add_argument("--veh-per-hour", type=float, default=1500.0,
                        help="offered demand summed over the entries. The gate puts "
                             "capacity between 1,200 and 1,500, so 1,500 is just past "
                             "breakdown, where there is a drop to prevent rather than "
                             "one nothing could avoid")
    parser.add_argument("--duration-s", type=float, default=2500.0)
    parser.add_argument("--av-fraction", type=float, default=1.0)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    topology = load_named_config("topology", args.topology)
    network = SumoNetwork.build(args.topology, topology, DATA)
    net_file = Path(network.net_file)
    print(f"building {args.topology}")
    print(f"  net {net_file.name}")

    # Super-segments, in the topology's own order, holding only edges the net has.
    order = list(topology["road"]["segment_ids"])
    grouped: dict[str, list[str]] = {name: [] for name in order}
    for edge in sorted(network.lane_counts()):
        segment = network.segment_for_edge(edge)
        if segment in grouped:
            grouped[segment].append(edge)
    segments = [grouped[name] for name in order if grouped[name]]
    (DATA / "tree_segments.json").write_text(json.dumps(segments))
    print(f"  {len(segments)} super-segments over "
          f"{sum(len(s) for s in segments)} edges")

    entries = sorted(network.entry_edges())
    routes = {entry: network.route_to_exit(entry) for entry in entries}
    unreachable = [e for e, path in routes.items() if not path]
    if unreachable:
        raise SystemExit(f"entries with no route to the exit: {unreachable}")

    header = ["<routes>"] + vtype_lines(args.human_model)
    for entry in entries:
        header.append(f'  <route id="r{entry}" edges="{" ".join(routes[entry])}"/>')

    # One explicit departure per vehicle, evenly spaced per entry and sorted by time,
    # so a vehicle's index decides whether it is an AV. SRC picks its controlled subset
    # by vehicle number, and reproducing that keeps penetration deterministic.
    per_entry = args.veh_per_hour / len(entries)
    headway = 3600.0 / per_entry
    departures: list[tuple[float, str]] = []
    for entry in entries:
        time = headway / 2.0
        while time < args.duration_s:
            departures.append((time, entry))
            time += headway
    departures.sort()

    lines, schedule = list(header), []
    for index, (time, entry) in enumerate(departures):
        kind = "av" if (index % round(1.0 / args.av_fraction) == 0) else "human"
        vehicle_id = f"v{entry}_{index}"
        lines.append(f'  <vehicle id="{vehicle_id}" type="{kind}" route="r{entry}" '
                     f'depart="{time:.2f}" departSpeed="max" departLane="best"/>')
        schedule.append({"id": vehicle_id, "route": f"r{entry}", "type": kind,
                         "depart": round(time, 2), "entry": entry})
    lines.append("</routes>")
    (DATA / "tree.rou.xml").write_text("\n".join(lines) + "\n")
    (DATA / "tree_routes.rou.xml").write_text("\n".join(header + ["</routes>"]) + "\n")
    (DATA / "tree_schedule.json").write_text(json.dumps(schedule))
    print(f"  {len(entries)} entries, {args.veh_per_hour:g} veh/h, "
          f"{len(departures)} departures over {args.duration_s:g} s")
    print(f"  fleet {args.human_model}, {args.av_fraction:.0%} AV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
