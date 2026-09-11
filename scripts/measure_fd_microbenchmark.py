#!/usr/bin/env python3
"""The fundamental diagram on the simplest geometry there is.

    .venv/bin/python scripts/measure_fd_microbenchmark.py --model w99

A flow-density curve is a hill: flow rises with density to a maximum at a critical
density and falls beyond it toward jam. That shape has been measured on real roads
many times, so a simulator that does not produce it is misconfigured, and no result
taken from it means anything. This checks the car-following model on one road with
nothing else in the way -- no junctions, no merges, no routing, one vehicle type.

**Two geometries, because a straight road cannot show the whole curve.** With free
outflow, demand above capacity queues at the entry and the road itself stays near
critical density, so a demand sweep traces the rising branch and the peak and never
the congested branch. Density has to be set independently of demand to see the fall,
and a ring does that: N vehicles on a closed loop of known length IS a density.

Flow is computed as density times mean speed, which at steady state on a ring equals
the count past any point and avoids putting a detector in an arbitrary place.
"""
from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sumo.env import _sumo  # noqa: E402

#: Metres of road one vehicle occupies bumper to bumper, which fixes jam density.
#: SUMO's default car is 5 m long; SRC's density uses 4.5 m.
VEHICLE_LENGTH_M = 5.0
LANES = 2
SPEED_LIMIT_MPS = 27.78          # 100 km/h, a highway link
RING_CIRCUMFERENCE_M = 2000.0
STRAIGHT_LENGTH_M = 3000.0


def netconvert(nodes: str, edges: str, out: Path) -> None:
    directory = out.parent
    (directory / "n.nod.xml").write_text(nodes)
    (directory / "e.edg.xml").write_text(edges)
    binary = REPO_ROOT / ".venv" / "bin" / "netconvert"
    result = subprocess.run(
        [str(binary if binary.exists() else "netconvert"),
         "-n", str(directory / "n.nod.xml"), "-e", str(directory / "e.edg.xml"),
         "--no-turnarounds", "true",
         # Otherwise each corner of the polygon that approximates the ring imposes a
         # turn-speed limit, and the ring's free-flow speed becomes an artifact of how
         # many segments were used to draw it.
         "--junctions.limit-turn-speed", "-1",
         "-o", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit("netconvert failed")


def build_straight(directory: Path) -> Path:
    nodes = ('<nodes>'
             '<node id="a" x="0" y="0"/>'
             f'<node id="b" x="{STRAIGHT_LENGTH_M}" y="0"/>'
             '</nodes>')
    edges = ('<edges>'
             f'<edge id="road" from="a" to="b" numLanes="{LANES}" '
             f'speed="{SPEED_LIMIT_MPS}"/>'
             '</edges>')
    out = directory / "straight.net.xml"
    netconvert(nodes, edges, out)
    return out


def build_ring(directory: Path, segments: int = 32) -> tuple[Path, list[str]]:
    radius = RING_CIRCUMFERENCE_M / (2.0 * math.pi)
    points = [(radius * math.cos(2 * math.pi * i / segments),
               radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    nodes = "<nodes>" + "".join(
        f'<node id="n{i}" x="{x:.3f}" y="{y:.3f}"/>' for i, (x, y) in enumerate(points)
    ) + "</nodes>"
    edges = "<edges>" + "".join(
        f'<edge id="r{i}" from="n{i}" to="n{(i + 1) % segments}" '
        f'numLanes="{LANES}" speed="{SPEED_LIMIT_MPS}"/>' for i in range(segments)
    ) + "</edges>"
    out = directory / "ring.net.xml"
    netconvert(nodes, edges, out)
    return out, [f"r{i}" for i in range(segments)]


def vtype(model: str) -> str:
    """One vehicle type, the model under test, everything else SUMO's default."""
    if model == "krauss":
        return '<vType id="car" carFollowModel="Krauss"/>'
    import yaml

    sumo = yaml.safe_load(
        (REPO_ROOT / "configs" / "human_models" / "w99_calibrated.yaml").read_text())["sumo"]
    cc = " ".join(f'{k}="{sumo[k]}"' for k in
                  ("cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9"))
    return f'<vType id="car" carFollowModel="W99" minGap="{sumo["min_gap_m"]}" {cc}/>'


def start(net: Path, routes: Path, step: float, seed: int) -> None:
    from sumolib import checkBinary

    _sumo.start([checkBinary("sumo"), "-n", str(net), "-r", str(routes),
                 "--step-length", str(step), "--seed", str(seed),
                 "--no-step-log", "true", "--no-warnings", "true",
                 "--time-to-teleport", "-1", "--max-depart-delay", "-1"])


def straight_point(rate: float, model: str, directory: Path, step: float,
                   seed: int, warmup_s: float, measure_s: float) -> dict:
    """One demand on the straight road. Density and flow on the middle third."""
    net = build_straight(directory)
    headway = 3600.0 / rate
    total = warmup_s + measure_s
    departures = "".join(
        f'<vehicle id="v{i}" type="car" route="r" depart="{headway * i:.2f}" '
        f'departSpeed="max" departLane="best"/>'
        for i in range(int(total / headway)))
    routes = directory / "straight.rou.xml"
    routes.write_text(f'<routes>{vtype(model)}<route id="r" edges="road"/>'
                      f'{departures}</routes>')
    start(net, routes, step, seed)
    try:
        densities, speeds = [], []
        for index in range(int(total / step)):
            _sumo.simulationStep()
            if index * step < warmup_s or index % int(round(10.0 / step)):
                continue
            # Vehicles in the middle third only, so entry and exit effects are excluded.
            low, high = STRAIGHT_LENGTH_M / 3.0, 2.0 * STRAIGHT_LENGTH_M / 3.0
            here = [v for v in _sumo.vehicle.getIDList()
                    if low <= _sumo.vehicle.getLanePosition(v) <= high]
            if not here:
                continue
            section = (high - low) * LANES
            densities.append(len(here) / section)
            speeds.append(statistics.fmean(_sumo.vehicle.getSpeed(v) for v in here))
    finally:
        _sumo.close()
    if not densities:
        return {"density": 0.0, "speed": 0.0, "flow": 0.0}
    density = statistics.fmean(densities)
    speed = statistics.fmean(speeds)
    return {"density": density, "speed": speed, "flow": density * speed * 3600.0}


def ring_point(count: int, model: str, directory: Path, step: float, seed: int,
               warmup_s: float, measure_s: float) -> dict:
    """N vehicles on the ring, which IS a density. Flow is density times speed."""
    net, loop = build_ring(directory)
    laps = 60
    route = " ".join(loop * laps)
    departures = "".join(
        f'<vehicle id="v{i}" type="car" route="r" depart="0" '
        f'departPos="{(RING_CIRCUMFERENCE_M / count) * i:.2f}" '
        f'departSpeed="0" departLane="{i % LANES}"/>'
        for i in range(count))
    routes = directory / "ring.rou.xml"
    routes.write_text(f'<routes>{vtype(model)}<route id="r" edges="{route}"/>'
                      f'{departures}</routes>')
    start(net, routes, step, seed)
    try:
        speeds, counts = [], []
        for index in range(int((warmup_s + measure_s) / step)):
            _sumo.simulationStep()
            if index * step < warmup_s or index % int(round(5.0 / step)):
                continue
            live = _sumo.vehicle.getIDList()
            if live:
                # DENSITY COMES FROM THE VEHICLES ACTUALLY ON THE RING, not from the
                # number asked for. Above jam density SUMO cannot insert them all, and
                # dividing a requested count by the ring length then reports a density
                # no road can hold -- 275 veh/km/lane against a jam of 200 -- while the
                # speed is that of the few vehicles present, so the product is a flow
                # of 19,000 veh/h/lane. The two have to describe the same vehicles.
                counts.append(len(live))
                speeds.append(statistics.fmean(_sumo.vehicle.getSpeed(v) for v in live))
        present = len(_sumo.vehicle.getIDList())
    finally:
        _sumo.close()
    if not counts:
        return {"density": 0.0, "speed": 0.0, "flow": 0.0, "present": 0, "asked": count}
    density = statistics.fmean(counts) / (RING_CIRCUMFERENCE_M * LANES)
    speed = statistics.fmean(speeds)
    return {"density": density, "speed": speed, "flow": density * speed * 3600.0,
            "present": present, "asked": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("w99", "krauss"), default="w99")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-s", type=float, default=300.0)
    parser.add_argument("--measure-s", type=float, default=300.0)
    parser.add_argument("--out", default="outputs/fd_microbenchmark")
    args = parser.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jam = 1000.0 / VEHICLE_LENGTH_M
    print(f"model {args.model}, {LANES} lanes, {SPEED_LIMIT_MPS:.1f} m/s limit, "
          f"dt {args.step}, jam density {jam:.0f} veh/km/lane\n")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)

        print("STRAIGHT ROAD, sweeping demand (rising branch only, by construction)")
        print(f"  {'offered veh/h':>14} {'density veh/km/ln':>18} {'speed km/h':>11} "
              f"{'flow veh/h/ln':>14}")
        straight = []
        for rate in (600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 6000, 8000):
            point = straight_point(rate, args.model, directory, args.step, args.seed,
                                   args.warmup_s, args.measure_s)
            k = point["density"] * 1000.0
            print(f"  {rate:>14} {k:>18.1f} {point['speed'] * 3.6:>11.1f} "
                  f"{point['flow']:>14.0f}", flush=True)
            straight.append((k, point["flow"]))

        print("\nRING, sweeping vehicle count (density set directly, whole curve)")
        jam_capacity = int(RING_CIRCUMFERENCE_M * LANES / VEHICLE_LENGTH_M)
        print(f"  ring holds {jam_capacity} vehicles bumper to bumper")
        print(f"  {'asked':>6} {'on ring':>8} {'density veh/km/ln':>18} "
              f"{'speed km/h':>11} {'flow veh/h/ln':>14}")
        ring = []
        for count in (8, 16, 32, 48, 64, 80, 100, 120, 160, 200, 250, 300,
                      360, 440, 520, 600, 680, 760):
            point = ring_point(count, args.model, directory, args.step, args.seed,
                               args.warmup_s, args.measure_s)
            k = point["density"] * 1000.0
            print(f"  {count:>6} {point['present']:>8} {k:>18.1f} "
                  f"{point['speed'] * 3.6:>11.1f} {point['flow']:>14.0f}", flush=True)
            ring.append((k, point["flow"]))

    peak = max(ring, key=lambda p: p[1])
    after = [f for k, f in ring if k > peak[0]]
    print(f"\n  ring peak flow {peak[1]:.0f} veh/h/lane at {peak[0]:.1f} veh/km/lane")
    if after:
        print(f"  lowest flow past the peak {min(after):.0f} veh/h/lane, "
              f"a fall of {100 * (peak[1] - min(after)) / peak[1]:.0f}%")
        print("  A hill needs a clear rise to a peak and a fall beyond it.")
    else:
        print("  the peak is the last point, so the grid does not reach the congested branch")

    plot(straight, ring, args.model, out)
    return 0


def plot(straight, ring, model: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot([k for k, _ in ring], [f for _, f in ring], "o-", label="ring (density set)")
    axes.plot([k for k, _ in straight], [f for _, f in straight], "s--",
              label="straight road (demand swept)")
    axes.set_xlabel("density (veh/km/lane)")
    axes.set_ylabel("flow (veh/h/lane)")
    axes.set_title(f"Fundamental diagram, {model}, {LANES} lanes")
    axes.grid(alpha=0.3)
    axes.legend()
    path = out / f"fd_{model}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n  plot written to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
