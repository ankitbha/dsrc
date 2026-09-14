#!/usr/bin/env python3
"""Stage 1 acceptance: does the ported Mainz network congest?

    .venv/bin/python scripts/check_mainz_baseline.py

Mainz has no traffic signals -- `signalController` count is zero in the .inpx -- so in
this network the junctions are the control, and that behaviour lives in 60 Vissim
conflict areas which netconvert does not import. SUMO infers right-of-way from geometry
instead. If the inferred priorities do not reproduce how Mainz congests then nothing
trained on this network means anything, and the fix is in the junction definitions
rather than in the learner.

This runs the network with no control and prints the trajectory. It is an acceptance
check on the conversion, not an experiment: what it has to show is that demand above
capacity produces a queue that grows and a speed that falls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sumo.mainz import DECISION_INTERVAL_S, MainzEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=2500.0)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    args = parser.parse_args()

    env = MainzEnv(seed=args.seed, duration_s=args.duration_s, warmup_s=args.warmup_s)
    env.reset()
    print(f"{'t(s)':>6} {'arrived':>8} {'running':>8} {'queued':>7} "
          f"{'speed km/h':>11} {'over rho*':>10} {'max rho':>8}")
    try:
        while True:
            env.release()
            env._advance(DECISION_INTERVAL_S)
            m = env.metrics()
            print(f"{m['time_s']:>6.0f} {m['arrived']:>8} {m['running']:>8} "
                  f"{m['waiting_to_enter']:>7} {m['mean_speed_kmh']:>11.2f} "
                  f"{m['segments_over_critical']:>10} {m['max_density']:>8.3f}",
                  flush=True)
            if m["time_s"] >= args.duration_s:
                break
        # Whether the reward's congestion term can fire at all is a precondition for
        # training, not a result: SRC's reward is -100*1[rho>0.3] + 0.2*speed, and a
        # threshold never crossed leaves only the speed term.
        import math as _math
        edges = env._per_edge()
        # An empty edge now carries no density rather than a zero, so it is absent
        # from these statistics rather than counted as uncongested road.
        values = sorted(v["density"] for v in edges.values()
                        if not _math.isnan(v["density"]))
        over = [e for e, v in edges.items()
                if not _math.isnan(v["density"]) and v["density"] > 0.3]
        n = len(values)
        print(f"\nper-EDGE density at t={env.metrics()['time_s']:.0f}s over {n} OCCUPIED edges:")
        print(f"  median {values[n//2]:.3f}  p90 {values[int(0.9*n)]:.3f}  "
              f"max {values[-1]:.3f}")
        print(f"  occupied edges over 0.3: {len(over)} of {n}")
        seg = env.densities()
        print(f"per-SEGMENT density: max {seg.max():.3f}, over 0.3: "
              f"{int((seg > 0.3).sum())} of {len(seg)}")
    finally:
        env.close()

    print("\nThe conversion is acceptable if the queue grows and the speed falls: the")
    print("demand is 18,000 veh/h and no free-flow network carries that. A network that")
    print("stays fast and empty means the junction priorities did not survive the port.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
