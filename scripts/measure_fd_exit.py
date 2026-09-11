#!/usr/bin/env python3
"""The exit super-segment's flow-density curve, measured as the straight road's was.

    .venv/bin/python scripts/measure_fd_exit.py

Mainz's outflow was never restricted. Routes end on link 218, and a vehicle reaching
the end of its route leaves instantly, so 218's own capacity of 3,899 veh/h was the
only cap -- above the 2,880 veh/h the network delivers, so it never bound. Nothing
could queue at the exit, and the exit super-segment therefore never left free flow.

`build_mainz_scenario.py` now continues the network into an exit road of 218's own
width and heading with a signal 300 m along it, so the discharge is a rate we set.
This measures what the exit super-segment does once that rate binds, using the
instrument that reproduced the curve on a straight road:

* **The restriction is intermittent, not a narrowing.** A section immediately upstream
  of a permanent bottleneck cannot exceed the bottleneck's capacity, so it never shows
  its own. A signal lets the section run at its own capacity during green and fills it
  during red, which sweeps the measurement through every state on every cycle.
* **Demand rises and falls.** `--demand-rush-s` ramps every entry from a quarter of its
  peak to the peak and back, so the queue from the meter grows back over the segment
  and then clears. The halves are compared over the densities they both reach: a queue
  lags the demand that caused it, so the two halves cover different density ranges and
  only their overlap is a check. Where they overlap they must agree on flow; if they do
  not, the curve is a transient rather than a property of the road.
* **Bins, not instants.** Counts and speeds are averaged over `--interval-s`, which is
  what a loop detector reports.

Density and speed are read on the exit super-segment's occupied edges. Segment 11 is
`218 219 517 31`, of which 219 and 517 are the opposite carriageway and hold no vehicle
at any demand here; counting their lane-metres would halve the reported density. Flow
is k * v, the hydrodynamic relation, per lane.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The super-segment link 218 belongs to, which every route leaves the network through.
EXIT_SEGMENT = 11
#: The .inpx demand, which `--peak-veh-per-h` is expressed against.
BASE_VEH_PER_HOUR = 18000.0


def build(peak: float, green: float, cycle_s: float, duration_s: float) -> None:
    result = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"),
         str(REPO_ROOT / "scripts" / "build_mainz_scenario.py"),
         "--av-fraction", "1.0",
         "--duration-s", f"{duration_s:.0f}",
         "--demand-scale", f"{peak / BASE_VEH_PER_HOUR:.6f}",
         "--demand-rush-s", f"{duration_s:.0f}",
         "--exit-green-fraction", f"{green:.4f}",
         "--exit-cycle-s", f"{cycle_s:.1f}"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit("building the scenario failed")
    return result.stdout


def measure(args) -> tuple[list[dict], list[str], float]:
    import libsumo as traffic

    import src.sumo.mainz as mainz

    env = mainz.MainzEnv(seed=args.seed, duration_s=args.duration_s, warmup_s=0.0,
                         step_length_s=args.step)
    ticks = int(round(args.duration_s / args.step))
    per_bin = int(round(args.interval_s / args.step))
    try:
        env.reset()
        edges = env.segments[EXIT_SEGMENT]
        counts: dict[str, list[int]] = {edge: [] for edge in edges}
        speeds: dict[str, list[float]] = {edge: [] for edge in edges}
        for _ in range(ticks):
            env._advance(args.step)
            for edge in edges:
                counts[edge].append(int(traffic.edge.getLastStepVehicleNumber(edge)))
                speeds[edge].append(float(traffic.edge.getLastStepMeanSpeed(edge)))
        discharge = env.window_flow_veh_per_h()
        held_back = len(traffic.simulation.getPendingVehicles())
    finally:
        env.close()

    # An edge that never holds a vehicle is not part of the road being measured, and
    # including its lane-metres would report a density about half the real one.
    carrying = [edge for edge in edges if sum(counts[edge]) > 0]
    lane_metres = sum(env._static[edge]["length_m"] * env._static[edge]["lanes"]
                      for edge in carrying)

    samples = []
    for start in range(0, ticks - per_bin + 1, per_bin):
        window = range(start, start + per_bin)
        present = [sum(counts[edge][tick] for edge in carrying) for tick in window]
        # Space-mean speed: each edge's mean weighted by how many vehicles it held.
        weighted = [(counts[edge][tick], speeds[edge][tick])
                    for edge in carrying for tick in window if counts[edge][tick]]
        vehicles = sum(number for number, _ in weighted)
        speed = (sum(number * value for number, value in weighted) / vehicles
                 if vehicles else 0.0)
        density = statistics.fmean(present) / lane_metres
        samples.append({
            "t": (start + per_bin) * args.step,
            "k": density * 1000.0,
            "v": speed * 3.6,
            "q": density * speed * 3600.0,
        })
    return samples, carrying, discharge, held_back


def report(samples: list[dict], duration_s: float) -> dict:
    usable = [sample for sample in samples if sample["k"] > 0.5]
    peak = max(usable, key=lambda sample: sample["q"])
    congested = [sample for sample in usable if sample["k"] > peak["k"]]
    print(f"\n  peak flow {peak['q']:.0f} veh/h/lane at {peak['k']:.1f} veh/km/lane "
          f"({peak['v']:.1f} km/h, t = {peak['t']:.0f} s)")
    summary = {"peak": peak, "hill": False}
    if not congested:
        print("  NO congested branch: nothing sampled past the peak density, so the")
        print("  meter never backed a queue over the exit super-segment.")
        return summary
    worst = min(congested, key=lambda sample: sample["q"])
    fall = 100.0 * (peak["q"] - worst["q"]) / peak["q"]
    print(f"  congested branch reaches {max(s['k'] for s in congested):.1f} "
          f"veh/km/lane at {worst['q']:.0f} veh/h/lane")
    print(f"  fall from the peak: {fall:.0f}%")
    print("  A hill: flow rises to a peak and falls beyond it.")
    half = duration_s / 2.0
    legs = {name: [s for s in usable if (s["t"] <= half) == rising]
            for name, rising in (("loading", True), ("unloading", False))}
    for name, leg in legs.items():
        if leg:
            top = max(leg, key=lambda sample: sample["q"])
            print(f"    {name:<9} peak {top['q']:>5.0f} veh/h/lane at "
                  f"{top['k']:>5.1f} veh/km/lane, densities "
                  f"{min(s['k'] for s in leg):.1f}-{max(s['k'] for s in leg):.1f}")
    # A queue lags the demand that raised it, so the two halves of the rush reach
    # different densities and only the band both reach compares like with like.
    summary.update(hill=True, fall_percent=fall, congested=worst)
    if all(legs.values()):
        low = max(min(s["k"] for s in leg) for leg in legs.values())
        high = min(max(s["k"] for s in leg) for leg in legs.values())
        if high > low:
            means = {name: statistics.fmean(
                        [s["q"] for s in leg if low <= s["k"] <= high] or [0.0])
                     for name, leg in legs.items()}
            gap = abs(means["loading"] - means["unloading"])
            reference = max(means.values()) or 1.0
            print(f"    both halves reach {low:.1f}-{high:.1f} veh/km/lane; mean flow "
                  f"there {means['loading']:.0f} loading against "
                  f"{means['unloading']:.0f} unloading, a gap of "
                  f"{100.0 * gap / reference:.0f}%")
            summary["overlap_gap_percent"] = 100.0 * gap / reference
        else:
            print("    the halves reach no common density, so they cannot be compared")
    return summary


def plot(samples: list[dict], peak: dict, out: Path, label: str, duration_s: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = duration_s / 2.0
    legs = (("loading", [s for s in samples if s["t"] <= half], "tab:blue"),
            ("unloading", [s for s in samples if s["t"] > half], "tab:orange"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, leg, colour in legs:
        axes[0].scatter([s["k"] for s in leg], [s["q"] for s in leg],
                        s=18, alpha=0.75, color=colour, label=name)
        axes[1].scatter([s["k"] for s in leg], [s["v"] for s in leg],
                        s=18, alpha=0.75, color=colour, label=name)
    axes[0].axvline(peak["k"], color="grey", linestyle="--", linewidth=1)
    axes[0].annotate(f"peak {peak['q']:.0f} at k={peak['k']:.1f}",
                     (peak["k"], peak["q"]), textcoords="offset points",
                     xytext=(8, 6), fontsize=9)
    axes[0].set_xlabel("density k (veh/km/lane)")
    axes[0].set_ylabel("flow q (veh/h/lane)")
    axes[0].set_title(f"Mainz exit super-segment\n{label}")
    axes[1].set_xlabel("density k (veh/km/lane)")
    axes[1].set_ylabel("speed (km/h)")
    axes[1].set_title("Speed-density")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(out / "fd_exit.png", dpi=120, bbox_inches="tight")
    print(f"\n  plot written to {(out / 'fd_exit.png').relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--green", type=float, default=0.85,
                        help="share of the exit meter's cycle that is green. It sets "
                             "the discharge the queue builds against. It must be "
                             "under the peak demand, so a queue forms, and over the "
                             "rush's average, so the queue clears again")
    parser.add_argument("--cycle-s", type=float, default=90.0)
    parser.add_argument("--peak-veh-per-h", type=float, default=4200.0,
                        help="demand at the top of the rush, summed over the eight "
                             "entries; the rush starts and ends at a quarter of it")
    parser.add_argument("--duration-s", type=float, default=5400.0)
    parser.add_argument("--interval-s", type=float, default=20.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--out", default="outputs/fd_exit")
    args = parser.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    built = build(args.peak_veh_per_h, args.green, args.cycle_s, args.duration_s)
    for line in built.splitlines():
        if "exit:" in line or "demand:" in line or "departures" in line:
            print(" " + line.strip())
    samples, carrying, discharge, held_back = measure(args)

    label = (f"{args.green:.2f} green on a {args.cycle_s:.0f} s cycle, "
             f"rush peaking at {args.peak_veh_per_h:.0f} veh/h")
    print(f"\n  measured on {' '.join(carrying)}; network discharge past the meter "
          f"{discharge:.0f} veh/h, {held_back} still waiting to enter")
    print(f"\n  {'t':>6} {'k veh/km/ln':>12} {'v km/h':>8} {'q veh/h/ln':>11}")
    for sample in samples:
        print(f"  {sample['t']:>6.0f} {sample['k']:>12.1f} {sample['v']:>8.1f} "
              f"{sample['q']:>11.0f}")
    summary = report(samples, args.duration_s)
    (out / "samples.json").write_text(json.dumps(
        {"label": label, "carrying": carrying, "discharge_veh_per_h": discharge,
         "samples": samples}, indent=1))
    plot(samples, summary["peak"], out, label, args.duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
