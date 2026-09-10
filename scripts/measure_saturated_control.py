"""Under a demand above capacity, does control raise throughput and lower latent demand?

The accounting is complete: every scheduled vehicle is served, on the road, or
latent (waiting to enter). A controller has to move the first up and the third
down; moving vehicles from the road into the queue is not an improvement.

It also tests a structural hypothesis about this topology. The six entry leaves are
SINGLE LANE, so an AV that slows there cannot be overtaken and is a rolling
roadblock rather than a meter -- ramp metering works because the meter sits beside
the road, not in it. `--leaf-lanes 2` rebuilds the leaves with two lanes, which is
the smallest change that lets an AV hold back its own lane without blocking the
others.

    .venv/bin/python scripts/measure_saturated_control.py --leaf-lanes 1 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loaders import load_named_config  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

SEEDS = (7, 17, 27, 37, 47)


def run(*, arm, speed, seed, leaf_lanes, dt, duration_steps, warmup_steps, demand_name,
        work_dir, control_interval_s=1.0):
    topology = dict(load_named_config("topology", "inverted_tree"))
    road = dict(topology["road"])
    road["lane_counts"] = {**road["lane_counts"], "leaves": int(leaf_lanes)}
    topology["road"] = road
    demand = dict(load_named_config("demand", demand_name))
    demand["av_penetration"] = 0.0 if arm == "no_av" else 0.2
    config = {
        "topology": topology, "demand": demand,
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": duration_steps, "dt": dt, "warmup_steps": warmup_steps,
    }
    if work_dir:
        # One directory PER RUN. The runs go in parallel and each builds its own
        # network and route file, so a shared directory has them overwriting each
        # other's net.net.xml while another process is reading it -- which surfaces
        # as an XML parse error from netconvert output that was never finished.
        config["work_dir"] = str(
            Path(work_dir) / f"{arm}_{speed:g}_{leaf_lanes}lane_seed{seed}")
    env = SumoTopologyEnv("inverted_tree", config)
    env.reset(seed=seed)
    try:
        downstream = env.view.downstream_segments()
        latent = []
        # A control decision every `control_interval_s`, not every simulation step.
        # A metering heuristic does not need 10 Hz, and the per-step version spent
        # most of its time rebuilding the vehicle snapshot: under saturation that is
        # about 480 vehicles times several TraCI calls times 18000 steps, which made
        # a controlled run cost 110 s against 20 s uncontrolled.
        every = max(1, int(round(control_interval_s / dt)))
        for step in range(duration_steps):
            if arm != "no_av" and step % every == 0:
                segments = env.get_segment_metrics()
                busy = {name for name, m in segments.items()
                        if m["jam_fraction"] > 0.25 or m["queue_length"] > 0}
                for snapshot in env.vehicle_snapshots():
                    if snapshot.role != "av" or snapshot.vehicle_id not in env.agent_ids:
                        continue
                    if arm == "metering" and not (
                            set(downstream.get(snapshot.segment_id, ())) & busy):
                        continue
                    env.command_speed(snapshot.vehicle_id, speed)
            _, _, _, _, info = env.step({})
            latent.append(info["metrics"]["latent_demand"])
        return {"served": env.arrived_total, "latent": latent[-1],
                "latent_mean": statistics.fmean(latent)}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", default="sumo_oversaturated")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration-steps", type=int, default=6000)
    # Long enough that the road is already saturated and latent demand is growing
    # when the episode starts: at 3000 veh/h that takes about 1200 s.
    parser.add_argument("--warmup-steps", type=int, default=12000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--leaf-lanes", type=int, nargs="+", default=[1])
    parser.add_argument("--speeds", type=float, nargs="+", default=[10.0, 16.0])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent simulations; libsumo allows one per process")
    parser.add_argument("--control-interval-s", type=float, default=1.0,
                        help="how often the controller acts, in simulated seconds")
    # The driver re-invokes this file with --one for each run.
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", default="no_av", help=argparse.SUPPRESS)
    parser.add_argument("--speed", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        print(json.dumps(run(
            arm=args.arm, speed=args.speed, seed=args.seed,
            leaf_lanes=args.leaf_lanes[0], dt=args.dt,
            duration_steps=args.duration_steps, warmup_steps=args.warmup_steps,
            demand_name=args.demand, work_dir=args.work_dir,
            control_interval_s=args.control_interval_s)))
        return 0

    kw = dict(dt=args.dt, duration_steps=args.duration_steps,
              warmup_steps=args.warmup_steps, demand_name=args.demand,
              work_dir=args.work_dir, control_interval_s=args.control_interval_s)

    def across_seeds(**fixed):
        """One run per seed, each in its own interpreter.

        NOT a ProcessPoolExecutor: libsumo keeps the simulation in module-level
        global state, and the pool forks children that inherit a copy of it, which
        hangs. Independent subprocesses each import libsumo fresh, which is also how
        the training runs achieve parallelism.
        """
        commands = []
        for seed in args.seeds:
            command = [sys.executable, __file__, "--one",
                       "--arm", str(fixed["arm"]), "--speed", str(fixed["speed"]),
                       "--seed", str(seed), "--leaf-lanes", str(fixed["leaf_lanes"]),
                       "--demand", args.demand, "--dt", str(args.dt),
                       "--duration-steps", str(args.duration_steps),
                       "--warmup-steps", str(args.warmup_steps),
                       "--control-interval-s", str(args.control_interval_s)]
            if args.work_dir:
                command += ["--work-dir", args.work_dir]
            commands.append(command)
        running = [subprocess.Popen(c, stdout=subprocess.PIPE, text=True)
                   for c in commands]
        out = []
        for process in running:
            stdout, _ = process.communicate()
            if process.returncode != 0:
                raise SystemExit(f"a run failed: {' '.join(commands[0])}")
            out.append(json.loads(stdout.strip().splitlines()[-1]))
        return out
    print(f"{args.demand}, dt {args.dt}, {args.duration_steps * args.dt:.0f} s episodes "
          f"after {args.warmup_steps * args.dt:.0f} s of warm-up, seeds {tuple(args.seeds)}.")
    print("Served should rise and latent should fall. Both, or neither counts.\n")
    for leaf_lanes in args.leaf_lanes:
        print(f"  leaves with {leaf_lanes} lane(s):")
        print(f"    {'arm':>22} {'served':>8} {'sd':>6} {'vs no_av':>9} "
              f"{'latent':>8} {'vs no_av':>9}")
        base = across_seeds(arm="no_av", speed=0.0, leaf_lanes=leaf_lanes)
        bs = [r["served"] for r in base]
        bl = [r["latent"] for r in base]
        print(f"    {'no_av':>22} {statistics.fmean(bs):>8.1f} "
              f"{statistics.stdev(bs):>6.1f} {'-':>9} {statistics.fmean(bl):>8.0f} {'-':>9}")
        for arm in ("metering", "always"):
            for speed in args.speeds:
                rows = across_seeds(arm=arm, speed=speed, leaf_lanes=leaf_lanes)
                served = [r["served"] for r in rows]
                lat = [r["latent"] for r in rows]
                ds = statistics.fmean(v - b for v, b in zip(served, bs))
                dl = statistics.fmean(v - b for v, b in zip(lat, bl))
                print(f"    {f'{arm} at {speed:g} m/s':>22} {statistics.fmean(served):>8.1f} "
                      f"{statistics.stdev(served):>6.1f} {ds:>+9.1f} "
                      f"{statistics.fmean(lat):>8.0f} {dl:>+9.0f}", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
