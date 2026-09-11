#!/usr/bin/env python3
"""Mainz's flow-density curve, measured the way `inverted_tree`'s was.

    .venv/bin/python scripts/measure_mainz_fundamental_diagram.py

Task 100 measured `inverted_tree` with the paper's W99 calibration and found served
flow peaking at 1,298 veh/h and falling 24% -- a capacity drop, which is the
inefficiency a density-holding controller recovers. An earlier coarse check on Mainz
suggested only 1.2%, but it averaged flow over the whole episode including the drain
and sampled neither the peak nor the critical point, so it is a lower bound from a
blunt instrument rather than a measurement.

This measures properly:

* **dt 0.1.** At 1 s the sign of the effect flips, so 1 s is too coarse to resolve the
  stop-and-go oscillation a capacity drop comes from.
* **A steady window.** Mainz's demand is a surge for 1,200 s and then nothing, so flow
  is counted between `--window-start` and the end of the surge, after the fill and
  before the drain. Averaging across the drain is what made the earlier number blunt.
* **A fine grid through the onset.** Every vehicle arrives at 3,600 veh/h offered and
  the network saturates by 5,400, so the peak and any drop lie between them.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The .inpx demand, which every scale is expressed against.
BASE_VEH_PER_HOUR = 18000.0
#: Demand ends here in the published scenario.
SURGE_END_S = 1200.0


def build(scale: float) -> None:
    subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"),
         str(REPO_ROOT / "scripts" / "build_mainz_scenario.py"),
         "--av-fraction", "1.0", "--demand-scale", f"{scale:.6f}"],
        cwd=REPO_ROOT, capture_output=True, check=True)


def run(rate: float, seed: int, dt: float, window_start: float) -> dict:
    import numpy as np

    import src.sumo.mainz as mainz

    build(rate / BASE_VEH_PER_HOUR)
    env = mainz.MainzEnv(seed=seed, duration_s=SURGE_END_S, warmup_s=0.0,
                         step_length_s=dt)
    densities, speeds = [], []
    start_arrived = None
    try:
        env.reset()
        sample_every = int(round(10.0 / dt))
        steps = int(round(SURGE_END_S / dt))
        for step in range(steps):
            env._advance(dt)
            now = (step + 1) * dt
            if start_arrived is None and now >= window_start:
                start_arrived = env.arrived_total
            if now >= window_start and step % sample_every == 0:
                segment = np.nan_to_num(env.densities())
                occupied = segment[segment > 0.0]
                if occupied.size:
                    densities.append(float(occupied.mean()))
                speeds.append(env.metrics()["mean_speed_kmh"])
        window_s = SURGE_END_S - window_start
        served = (env.arrived_total - (start_arrived or 0)) * 3600.0 / window_s
    finally:
        env.close()
    return {
        "served": served,
        "density": statistics.fmean(densities) if densities else 0.0,
        "speed": statistics.fmean(speeds) if speeds else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--window-start", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[16])
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200])
    args = parser.parse_args()

    print(f"Mainz, W99 calibrated, dt {args.dt}, flow counted over "
          f"{args.window_start:.0f}-{SURGE_END_S:.0f} s, seeds {tuple(args.seeds)}\n")
    print(f"  {'offered':>8} {'density':>9} {'speed':>8} {'served/h':>9} {'served/offered':>15}")
    peak = (None, -1.0)
    rows = []
    for rate in args.rates:
        runs = [run(rate, seed, args.dt, args.window_start) for seed in args.seeds]
        served = statistics.fmean(r["served"] for r in runs)
        density = statistics.fmean(r["density"] for r in runs)
        speed = statistics.fmean(r["speed"] for r in runs)
        rows.append((rate, density, served))
        if served > peak[1]:
            peak = (rate, served)
        print(f"  {rate:>8.0f} {density:>9.3f} {speed:>8.2f} {served:>9.0f} "
              f"{served / rate:>14.2f}", flush=True)

    after = [s for r, _, s in rows if r > peak[0]]
    print(f"\n  peak served flow {peak[1]:.0f} veh/h at {peak[0]:.0f} veh/h offered")
    if after:
        worst = min(after)
        print(f"  lowest served flow past the peak: {worst:.0f} veh/h, "
              f"a drop of {100 * (peak[1] - worst) / peak[1]:.1f}%")
        print(f"  inverted_tree under the same calibration drops 24% (task 100)")
    else:
        print("  the peak is the last point measured, so the grid does not reach the drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
