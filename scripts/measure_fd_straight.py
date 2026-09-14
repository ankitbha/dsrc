#!/usr/bin/env python3
"""The fundamental diagram on a straight road, measured as real highways measure it.

    .venv/bin/python scripts/measure_fd_straight.py --model w99

Empirical flow-density curves come from loop detectors on straight highway sections,
not from rings. The congested branch appears because the detector sits UPSTREAM OF A
BOTTLENECK: when demand exceeds what the bottleneck discharges, the queue grows back
past the detector, which then records high density at low flow. Over a rush that builds
and clears, one detector passes through free flow, capacity and congestion, and the
scatter of its samples is the hill.

Two things are therefore needed and a plain road with free outflow has neither:

* **A downstream restriction that periodically stops the road.** A signal at the far
  end. A lane drop was tried first and is wrong for this purpose twice over: a section
  immediately upstream of a bottleneck cannot exceed the BOTTLENECK's capacity, so it
  never shows its own; and it sits in the merge turbulence, which collapsed speed to
  67 km/h at a density of 6.6 veh/km/lane, where vehicles are 150 m apart and cannot
  be interacting at all. A signal restricts without a merge, and its queue sweeps the
  detector through every state from free flow to jam and back on every cycle.
* **A demand profile that rises and falls.** A single constant rate gives one operating
  point. A rush traces the curve, and the loading and unloading halves should fall on
  the same branches if the measurement is sound.

Geometry, all at one speed limit:

    feed (2 lanes, 1 km) -> measure (2 lanes, 0.5 km) -> exit (2 lanes, 0.6 km, signal)

The detector is 600 m clear of the signal so it is never inside the discharge fan.

Density and speed are read on `measure`, and flow is their product, which is the
standard hydrodynamic relation q = k v.
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

from src.sumo.binding import _sumo  # noqa: E402

APPROACH_LANES = 2
SPEED_LIMIT_MPS = 27.78          # 100 km/h, overridable with --speed-mps
FEED_M, MEASURE_M, NECK_M = 1000.0, 500.0, 600.0
VEHICLE_LENGTH_M = 5.0


def build(directory: Path, cycle_s: float) -> Path:
    nodes = ("<nodes>"
             '<node id="a" x="0" y="0"/>'
             f'<node id="b" x="{FEED_M}" y="0"/>'
             f'<node id="c" x="{FEED_M + MEASURE_M}" y="0"/>'
             f'<node id="d" x="{FEED_M + MEASURE_M + NECK_M}" y="0" type="traffic_light"/>'
             f'<node id="e" x="{FEED_M + MEASURE_M + NECK_M + 500.0}" y="0"/>'
             "</nodes>")
    edges = ("<edges>"
             f'<edge id="feed" from="a" to="b" numLanes="{APPROACH_LANES}" speed="{SPEED_LIMIT_MPS}"/>'
             f'<edge id="measure" from="b" to="c" numLanes="{APPROACH_LANES}" speed="{SPEED_LIMIT_MPS}"/>'
             f'<edge id="exit" from="c" to="d" numLanes="{APPROACH_LANES}" speed="{SPEED_LIMIT_MPS}"/>'
             f'<edge id="sink" from="d" to="e" numLanes="{APPROACH_LANES}" speed="{SPEED_LIMIT_MPS}"/>'
             "</edges>")
    (directory / "n.nod.xml").write_text(nodes)
    (directory / "e.edg.xml").write_text(edges)
    out = directory / "road.net.xml"
    binary = REPO_ROOT / ".venv" / "bin" / "netconvert"
    result = subprocess.run(
        [str(binary if binary.exists() else "netconvert"),
         "-n", str(directory / "n.nod.xml"), "-e", str(directory / "e.edg.xml"),
         "--no-turnarounds", "true",
         # The cycle belongs to the network, not the run: its red phase is what backs
         # a queue over the detector and sweeps it through every traffic state.
         "--tls.cycle.time", str(int(cycle_s)),
         "-o", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit("netconvert failed")
    return out


def vtype(model: str, cc1: float | None) -> str:
    if model == "krauss":
        return '<vType id="car" carFollowModel="Krauss"/>'
    import yaml

    sumo = yaml.safe_load(
        (REPO_ROOT / "configs" / "human_models" / "w99_calibrated.yaml").read_text())["sumo"]
    values = {k: sumo[k] for k in
              ("cc1", "cc2", "cc3", "cc4", "cc5", "cc6", "cc7", "cc8", "cc9")}
    if cc1 is not None:
        values["cc1"] = cc1
    cc = " ".join(f'{k}="{v}"' for k, v in values.items())
    return f'<vType id="car" carFollowModel="W99" minGap="{sumo["min_gap_m"]}" {cc}/>'


def profile(peak: float, duration_s: float) -> list[tuple[float, float]]:
    """A rush: demand ramps from a quarter of peak up to peak and back down.

    Both halves are traced so the loading and unloading legs can be compared; if they
    do not lie on the same curve the measurement is picking up a transient rather than
    a property of the road.
    """
    steps = 12
    half = duration_s / 2.0
    rates = []
    for index in range(steps):
        fraction = index / (steps - 1)
        rates.append((half * fraction, peak * (0.25 + 0.75 * fraction)))
    for index in range(1, steps):
        fraction = index / (steps - 1)
        rates.append((half + half * fraction, peak * (1.0 - 0.75 * fraction)))
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("w99", "krauss"), default="w99")
    parser.add_argument("--cc1", type=float, default=None,
                        help="override W99's desired time headway, which sets capacity")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--peak-veh-per-h", type=float, default=4800.0)
    parser.add_argument("--interval-s", type=float, default=20.0)
    parser.add_argument("--cycle-s", type=float, default=90.0)
    parser.add_argument("--speed-mps", type=float, default=None,
                        help="free-flow speed; critical density depends on it, so it "
                             "must match the network the threshold is being set for")
    parser.add_argument("--out", default="outputs/fd_straight")
    args = parser.parse_args()

    global SPEED_LIMIT_MPS
    if args.speed_mps is not None:
        SPEED_LIMIT_MPS = args.speed_mps
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        net = build(directory, args.cycle_s)
        departures, index, time = [], 0, 0.0
        schedule = profile(args.peak_veh_per_h, args.duration_s)
        for position, (start, rate) in enumerate(schedule):
            end = (schedule[position + 1][0] if position + 1 < len(schedule)
                   else args.duration_s)
            headway = 3600.0 / rate
            time = max(time, start)
            while time < end:
                departures.append(f'<vehicle id="v{index}" type="car" route="r" '
                                  f'depart="{time:.2f}" departSpeed="desired" '
                                  f'departLane="free"/>')
                index += 1
                time += headway
        routes = directory / "road.rou.xml"
        routes.write_text(f'<routes>{vtype(args.model, args.cc1)}'
                          f'<route id="r" edges="feed measure exit sink"/>'
                          f'{"".join(departures)}</routes>')

        from sumolib import checkBinary
        _sumo.start([checkBinary("sumo"), "-n", str(net), "-r", str(routes),
                     "--step-length", str(args.step), "--seed", str(args.seed),
                     "--no-step-log", "true", "--no-warnings", "true",
                     "--time-to-teleport", "-1", "--max-depart-delay", "-1"])

        lane_metres = MEASURE_M * APPROACH_LANES
        samples, counts, speeds = [], [], []
        try:
            for tick in range(int(args.duration_s / args.step)):
                _sumo.simulationStep()
                counts.append(int(_sumo.edge.getLastStepVehicleNumber("measure")))
                speeds.append(float(_sumo.edge.getLastStepMeanSpeed("measure")))
                if (tick + 1) % int(round(args.interval_s / args.step)):
                    continue
                occupancy = statistics.fmean(counts)
                speed = statistics.fmean([s for s, c in zip(speeds, counts) if c] or [0.0])
                counts.clear(); speeds.clear()
                density = occupancy / lane_metres
                samples.append({
                    "t": (tick + 1) * args.step,
                    "k": density * 1000.0,
                    "v": speed * 3.6,
                    "q": density * speed * 3600.0,
                })
        finally:
            _sumo.close()

    jam = 1000.0 / VEHICLE_LENGTH_M
    label = f"{args.model}" + (f", cc1={args.cc1}" if args.cc1 is not None else "")
    print(f"{label}, {APPROACH_LANES} lanes, signalised end, {SPEED_LIMIT_MPS:.1f} m/s, dt {args.step}, "
          f"jam {jam:.0f} veh/km/lane, peak demand {args.peak_veh_per_h:.0f} veh/h\n")
    print(f"  {'t':>6} {'k veh/km/ln':>12} {'v km/h':>8} {'q veh/h/ln':>11}")
    for sample in samples:
        print(f"  {sample['t']:>6.0f} {sample['k']:>12.1f} {sample['v']:>8.1f} "
              f"{sample['q']:>11.0f}")

    usable = [s for s in samples if s["k"] > 0.5]
    peak = max(usable, key=lambda s: s["q"])
    congested = [s for s in usable if s["k"] > peak["k"]]
    print(f"\n  peak flow {peak['q']:.0f} veh/h/lane at {peak['k']:.1f} veh/km/lane "
          f"({peak['v']:.0f} km/h)")
    if congested:
        worst = min(congested, key=lambda s: s["q"])
        print(f"  congested branch reaches {max(s['k'] for s in congested):.1f} "
              f"veh/km/lane at {worst['q']:.0f} veh/h/lane")
        print(f"  fall from the peak: {100 * (peak['q'] - worst['q']) / peak['q']:.0f}%")
        print("  A hill: flow rises to a peak and falls beyond it.")
    else:
        print("  NO congested branch: nothing sampled past the peak density, so the")
        print("  bottleneck never backed the queue over the measurement section.")

    plot(samples, peak, label, out, args)
    return 0


def plot(samples, peak, label: str, out: Path, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = args.duration_s / 2.0
    loading = [s for s in samples if s["t"] <= half and s["k"] > 0.5]
    unloading = [s for s in samples if s["t"] > half and s["k"] > 0.5]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter([s["k"] for s in loading], [s["q"] for s in loading],
                    label="loading", s=42)
    axes[0].scatter([s["k"] for s in unloading], [s["q"] for s in unloading],
                    label="unloading", s=42, marker="^")
    axes[0].scatter([peak["k"]], [peak["q"]], color="red", s=90, zorder=5, label="peak")
    axes[0].set_xlabel("density k (veh/km/lane)")
    axes[0].set_ylabel("flow q (veh/h/lane)")
    axes[0].set_title(f"Fundamental diagram, {label}")
    axes[1].scatter([s["k"] for s in loading], [s["v"] for s in loading], s=42)
    axes[1].scatter([s["k"] for s in unloading], [s["v"] for s in unloading],
                    s=42, marker="^")
    axes[1].set_xlabel("density k (veh/km/lane)")
    axes[1].set_ylabel("speed v (km/h)")
    axes[1].set_title("Speed-density")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].legend()
    name = f"fd_straight_{args.model}" + (f"_cc1_{args.cc1}" if args.cc1 else "") + ".png"
    figure.savefig(out / name, dpi=120, bbox_inches="tight")
    print(f"\n  plot written to {(out / name).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
