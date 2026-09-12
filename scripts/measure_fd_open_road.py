#!/usr/bin/env python3
"""A fundamental diagram on a plain straight road: which driving model, or what boundary.

    .venv/bin/python scripts/measure_fd_open_road.py

The road is two lanes and nothing else. No signal, no lane drop, no merge, no junction:
every earlier attempt here produced its congested branch from added infrastructure, and
the level of that branch turned out to be a property of the infrastructure rather than of
the road. Measured on the signalised version, changing only the signal's cycle moved the
congested branch from 419 to 810 veh/h/lane while the peak stayed at 984 to 1,015.

WHAT A STRAIGHT ROAD CAN AND CANNOT SHOW. In steady state, flow through every
cross-section of a road equals its throughput. So a boundary that releases at a constant
rate gives a rising branch and then a FLAT line at that rate, whatever the density behind
it. A congested branch that falls needs one of two things, and this script tests both:

* **A string-unstable car-following model.** If small disturbances amplify as they pass
  upstream, stop-and-go waves form on their own at high density, and a fixed detector
  passes through free flow and jam as they arrive. No bottleneck is required, and this is
  what a real freeway detector records. `--restriction none` tests this: the road drains
  freely and the only thing that can create congestion is the fleet itself.

* **An outflow whose rate falls once a queue forms.** `--restriction meter` holds the
  boundary to `--outflow-veh-h` by admitting a vehicle past a gate only when the headway
  since the last one has elapsed. It is a backpressure function, not a signal: there is no
  cycle and no red phase, so nothing is imposed on the road periodically.

Density and speed are read on `measure`, a 1 km section in the middle, and flow is their
product. Samples are binned over `--interval-s`, which is what a loop detector reports.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sumo.env import _sumo  # noqa: E402

LANES = 2
FEED_M, MEASURE_M, DRAIN_M = 1000.0, 1000.0, 1000.0
#: Where on `drain` the outflow gate sits, far enough in that a queue at it does not
#: immediately cover the detector.
GATE_M = 900.0


def build(directory: Path, speed_mps: float) -> Path:
    (directory / "n.nod.xml").write_text(
        "<nodes>"
        '<node id="a" x="0" y="0"/>'
        f'<node id="b" x="{FEED_M}" y="0"/>'
        f'<node id="c" x="{FEED_M + MEASURE_M}" y="0"/>'
        f'<node id="d" x="{FEED_M + MEASURE_M + DRAIN_M}" y="0"/>'
        "</nodes>")
    (directory / "e.edg.xml").write_text(
        "<edges>"
        f'<edge id="feed" from="a" to="b" numLanes="{LANES}" speed="{speed_mps}"/>'
        f'<edge id="measure" from="b" to="c" numLanes="{LANES}" speed="{speed_mps}"/>'
        f'<edge id="drain" from="c" to="d" numLanes="{LANES}" speed="{speed_mps}"/>'
        "</edges>")
    out = directory / "road.net.xml"
    binary = REPO_ROOT / ".venv" / "bin" / "netconvert"
    result = subprocess.run(
        [str(binary if binary.exists() else "netconvert"),
         "-n", str(directory / "n.nod.xml"), "-e", str(directory / "e.edg.xml"),
         "--no-turnarounds", "true", "-o", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit("netconvert failed")
    return out


def vtype(model: str) -> str:
    """One `<vType id="car">`, for each model this compares."""
    import yaml

    sumo = yaml.safe_load(
        (REPO_ROOT / "configs" / "human_models" / "w99_calibrated.yaml").read_text())["sumo"]
    w99 = {k: sumo[k] for k in
           ("cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9")}
    if model == "w99":
        pass
    elif model == "w99_oscillating":
        # CC2 is the following-distance oscillation and CC4/CC5 the thresholds at which
        # a driver reacts to closing or opening. The paper's appendix names these as the
        # ones that "help introduce stop-and-go dynamics" and "influence capacity drop
        # effects", so widening them is the calibration's own lever for instability.
        w99.update(cc2=40.0, cc4=-2.0, cc5=2.0)
    elif model == "w99_slow_reaction":
        # A longer reaction and gentler acceleration is the other route to instability:
        # a driver who responds late overshoots, and the overshoot grows upstream.
        w99.update(cc1=3.0, cc7=0.25, cc8=0.5, cc9=0.4)
    elif model == "krauss":
        return f'<vType id="car" carFollowModel="Krauss" minGap="{sumo["min_gap_m"]}"/>'
    elif model == "krauss_imperfect":
        # sigma is Krauss's driver imperfection: the share of the safe speed a driver
        # randomly gives up. At 0 the model is exactly string-stable.
        return (f'<vType id="car" carFollowModel="Krauss" '
                f'minGap="{sumo["min_gap_m"]}" sigma="0.9" tau="1.4"/>')
    elif model == "idm":
        return f'<vType id="car" carFollowModel="IDM" minGap="{sumo["min_gap_m"]}"/>'
    else:
        raise SystemExit(f"unknown model {model!r}")
    attributes = " ".join(f'{k}="{v}"' for k, v in w99.items())
    return (f'<vType id="car" carFollowModel="W99" '
            f'minGap="{sumo["min_gap_m"]}" {attributes}/>')


def meter(rate_veh_h: float, released: dict) -> None:
    """Let a vehicle past the gate only when the headway since the last one has elapsed.

    A backpressure function rather than a signal: the boundary serves at a steady rate
    with no cycle, so nothing periodic is imposed on the road. A vehicle that arrives
    early is held at the gate and the ones behind it queue, which is the only way the
    road's density can exceed what its own inflow sets.
    """
    now = float(_sumo.simulation.getTime())
    headway = 3600.0 / rate_veh_h
    for vehicle in _sumo.edge.getLastStepVehicleIDs("drain"):
        if vehicle in released["passed"]:
            continue
        if float(_sumo.vehicle.getLanePosition(vehicle)) < GATE_M:
            continue
        if now >= released["next_at"]:
            released["passed"].add(vehicle)
            released["next_at"] = now + headway
            _sumo.vehicle.setSpeed(vehicle, -1.0)
        else:
            _sumo.vehicle.setSpeed(vehicle, 0.0)


def run(args, model: str, rate: float) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        net = build(directory, args.speed_mps)
        headway = 3600.0 / rate
        departures = []
        time = headway / 2.0
        while time < args.duration_s:
            departures.append(
                f'<vehicle id="v{len(departures)}" type="car" route="r" '
                f'depart="{time:.2f}" departSpeed="desired" departLane="free"/>')
            time += headway
        routes = directory / "road.rou.xml"
        routes.write_text(f'<routes>{vtype(model)}'
                          f'<route id="r" edges="feed measure drain"/>'
                          f'{"".join(departures)}</routes>')

        from sumolib import checkBinary
        _sumo.start([checkBinary("sumo"), "-n", str(net), "-r", str(routes),
                     "--step-length", str(args.step), "--seed", str(args.seed),
                     "--no-step-log", "true", "--no-warnings", "true",
                     "--time-to-teleport", "-1", "--max-depart-delay", "-1"])
        lane_metres = MEASURE_M * LANES
        released = {"passed": set(), "next_at": 0.0}
        samples, counts, speeds = [], [], []
        try:
            for tick in range(int(args.duration_s / args.step)):
                if args.restriction == "meter":
                    meter(args.outflow_veh_h, released)
                _sumo.simulationStep()
                if (tick + 1) * args.step < args.warmup_s:
                    continue
                counts.append(int(_sumo.edge.getLastStepVehicleNumber("measure")))
                speeds.append(float(_sumo.edge.getLastStepMeanSpeed("measure")))
                if (tick + 1) % int(round(args.interval_s / args.step)):
                    continue
                occupancy = statistics.fmean(counts)
                speed = statistics.fmean([s for s, c in zip(speeds, counts) if c] or [0.0])
                counts.clear(); speeds.clear()
                density = occupancy / lane_metres
                samples.append({"k": density * 1000.0, "v": speed * 3.6,
                                "q": density * speed * 3600.0})
        finally:
            _sumo.close()
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["w99", "w99_oscillating", "w99_slow_reaction",
                                 "krauss", "krauss_imperfect", "idm"])
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[1200, 1800, 2400, 3000, 4200])
    parser.add_argument("--restriction", choices=("none", "meter"), default="none")
    parser.add_argument("--outflow-veh-h", type=float, default=1500.0)
    parser.add_argument("--speed-mps", type=float, default=16.67)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=2400.0)
    parser.add_argument("--warmup-s", type=float, default=600.0)
    parser.add_argument("--interval-s", type=float, default=20.0)
    parser.add_argument("--profile", action="store_true",
                        help="print mean flow per density band, which is what tells a "
                             "congested branch set by the road from one set by the "
                             "boundary: the second is flat at the boundary's rate")
    args = parser.parse_args()

    print(f"plain {LANES}-lane straight road, {args.speed_mps:.2f} m/s, dt {args.step}, "
          f"restriction: {args.restriction}"
          + (f" at {args.outflow_veh_h:.0f} veh/h" if args.restriction == "meter" else "")
          + f", {args.interval_s:.0f} s bins\n")
    for model in args.models:
        gathered = []
        for rate in args.rates:
            gathered += run(args, model, rate)
        usable = [s for s in gathered if s["k"] > 0.5]
        if not usable:
            print(f"  {model:>18}: no traffic")
            continue
        peak = max(usable, key=lambda s: s["q"])
        beyond = [s for s in usable if s["k"] > peak["k"]]
        note = "no samples past the peak density"
        if beyond:
            worst = min(beyond, key=lambda s: s["q"])
            note = (f"falls to {worst['q']:>4.0f} at k={worst['k']:>5.1f} "
                    f"({100.0 * (peak['q'] - worst['q']) / peak['q']:>4.1f}%)")
        print(f"  {model:>18}: peak {peak['q']:>4.0f} veh/h/lane at k={peak['k']:>5.1f}, "
              f"k reaches {max(s['k'] for s in usable):>5.1f}, {note}")
        if args.profile:
            # Mean flow per density band. A congested branch set by the boundary is
            # FLAT at the boundary's rate; one set by the road falls with density.
            bands: dict[int, list[float]] = {}
            for sample in usable:
                bands.setdefault(int(sample["k"] // 10) * 10, []).append(sample)
            for low in sorted(bands):
                rows = bands[low]
                print(f"      k {low:>3}-{low + 10:>3}: "
                      f"q {statistics.fmean(r['q'] for r in rows):>5.0f} veh/h/lane   "
                      f"v {statistics.fmean(r['v'] for r in rows):>5.1f} km/h   "
                      f"({len(rows):>3} bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
