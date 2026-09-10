"""Does the NETWORK outcome depend on the fleet-wide action mix?

The question this answers, raised by Ankit: a shared policy can represent the
centralized behaviour, so if a centralized controller helps, the behaviour is in the
search space and the problem is the search rather than the space. Each agent samples
independently from the shared policy, so the fleet-wide mix concentrates -- at 42
agents and p = 1/3 the per-step mix has a standard deviation of 0.073, and averaged
over a 150-decision rollout, 0.006. The policy therefore never samples the fleet
configurations the oracle arms tested. Each agent's own share of the signal is
dJ/dp / N, which is what sits below the gradient noise floor.

So dJ/dp could be large and invisible. This measures dJ/dp directly.

THE SWEEP IS ONE DIMENSIONAL AND CLEANLY INTERPRETABLE. At every decision, each AV
independently issues a metering command with probability p and otherwise RELEASES to
SUMO's car-following via setSpeed(-1). p = 0 is therefore exactly the uncommanded
fleet and p = 1 is every AV metered at every decision. Without the release a vehicle
commanded once would stay commanded, and p would mean "fraction ever commanded"
rather than "fraction of decisions".

TWO DEFINITIONS OF THE COMMAND, because one of them is known not to bite. `shipped`
is what the action contract gives: max(12, allowed - 10), which is 20 m/s at a 30 m/s
limit and binds on 6% of decisions at this operating point. `binding` is 0.6 times
the vehicle's own current speed, which binds on 29%. Without both, a flat result
cannot be told from a command that does nothing.

Every arm at a given seed sees the SAME traffic: penetration is fixed at the config's
value in every arm, so the vTypeDistribution draws and therefore the fleet are
identical, and the arms differ only in what is commanded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc")

from src.config.loaders import load_named_config  # noqa: E402

SEEDS = (7, 17, 27, 37, 47)
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
SCHEMES = ("shipped", "binding", "headway", "harmonize")


def run_one(*, mix, scheme, seed, work_dir, dt, warmup_steps, duration_steps,
            demand_name, decision_interval_s, topology_name):
    import numpy as np
    from src.sumo.env import SumoTopologyEnv
    import src.sumo.env as env_module

    sumo = env_module._sumo
    config = {
        "topology": load_named_config("topology", topology_name),
        "demand": load_named_config("demand", demand_name),
        "human_model": load_named_config("human_model", "w99_calibrated"),
        "duration_steps": duration_steps, "dt": dt, "warmup_steps": warmup_steps,
        "work_dir": work_dir,
    }
    env = SumoTopologyEnv(topology_name, config)
    env.reset(seed=seed)
    # Its own generator, so the command draw does not consume the simulator's
    # randomness and every arm sees the same traffic.
    rng = np.random.RandomState(seed)
    # Read rather than assumed, so the release restores what the vType actually had.
    default_tau = (float(sumo.vehicle.getTau(env.agent_ids[0]))
                   if env.agent_ids else 1.0)
    every = max(1, int(round(decision_interval_s / dt)))
    series = {k: [] for k in ("mean_speed", "throughput_recent", "mean_delay_recent",
                              "stopped_fraction", "mean_abs_jerk", "jam_fraction",
                              "latent_demand", "active_av_count")}
    commanded = released = 0
    try:
        for step in range(duration_steps):
            if step % every == 0:
                for agent_id in list(env.agent_ids):
                    if rng.random_sample() < mix:
                        if scheme == "headway":
                            # The other action head. `desired_headway_bin` maps onto
                            # setTau, and tau is the Krauss/IDM desired time headway
                            # while W99 follows on cc1 -- so it was not obvious this
                            # reaches the model at all. Measured paired on one seed: it
                            # does, taking the mean AV gap from 78.05 m to 83.51 m and
                            # arrivals from 42 to 40. `largest` is 3.0 s.
                            sumo.vehicle.setTau(agent_id, 3.0)
                        elif scheme == "harmonize":
                            # SPEED HARMONIZATION, a different intervention from
                            # metering: the target is the speed the traffic around
                            # the vehicle is already holding, so a vehicle faster
                            # than its neighbours slows and a slower one speeds up,
                            # within SUMO's safe-speed bound. It reduces the VARIANCE
                            # of speed rather than its mean, which is the mechanism
                            # the ring-road results damp stop-and-go waves with.
                            # Every other arm here only ever slows a vehicle, so none
                            # of them tests it.
                            segment = env.network.segment_for_edge(
                                sumo.vehicle.getRoadID(agent_id))
                            metrics = env.get_segment_metrics().get(segment or "", {})
                            target = float(metrics.get("mean_speed", 0.0) or 0.0)
                            if target > 0.0:
                                env.command_speed(agent_id, target)
                        elif scheme == "shipped":
                            allowed = float(sumo.vehicle.getAllowedSpeed(agent_id))
                            env.command_speed(agent_id, max(12.0, allowed - 10.0))
                        elif scheme == "binding":
                            env.command_speed(
                                agent_id, max(0.0, 0.6 * float(sumo.vehicle.getSpeed(agent_id))))
                        else:
                            # NOT a fall-through. The previous version sent every
                            # unrecognised scheme to the binding command, so a new
                            # scheme name ran the old intervention under a new label
                            # and produced numbers that looked like a measurement of
                            # something never executed.
                            raise ValueError(f"unknown scheme {scheme!r}")
                        commanded += 1
                    else:
                        if scheme == "headway":
                            sumo.vehicle.setTau(agent_id, default_tau)
                        else:
                            # -1 hands the vehicle back to the car-following model.
                            # Without it a vehicle commanded on one decision stays
                            # commanded.
                            sumo.vehicle.setSpeed(agent_id, -1.0)
                        released += 1
            _, _, _, _, info = env.step({})
            for key in series:
                series[key].append(float(info["metrics"].get(key, 0.0) or 0.0))
        return {"mix": mix, "scheme": scheme, "seed": seed,
                "arrivals": env.arrived_total,
                "commanded": commanded, "released": released,
                **{key: statistics.fmean(values) for key, values in series.items()}}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", default="sumo_capacity_drop")
    # The merge tree is junction-limited: throughput there is set by gap acceptance
    # at the merge, and slowing upstream traffic does not create junction capacity.
    # `inverted_tree_bottleneck` has a 2-to-1 lane drop, which is the setting the
    # capacity-drop literature demonstrates speed metering on. A null on one says
    # nothing about the other.
    parser.add_argument("--topology", default="inverted_tree")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--duration-steps", type=int, default=6000)
    parser.add_argument("--decision-interval-s", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--work-dir", default="/tmp/dsrc_mix")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mix", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--scheme", default="shipped", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=7, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        print(json.dumps(run_one(
            mix=args.mix, scheme=args.scheme, seed=args.seed,
            work_dir=f"{args.work_dir}/{args.scheme}_{args.mix:g}_{args.seed}",
            dt=args.dt, warmup_steps=args.warmup_steps,
            duration_steps=args.duration_steps, demand_name=args.demand,
            decision_interval_s=args.decision_interval_s,
            topology_name=args.topology)))
        return 0

    # p = 0 issues no command, so it is the same arm under both schemes and is run
    # once as the shared reference.
    jobs = [(0.0, "reference", seed) for seed in args.seeds]
    jobs += [(mix, scheme, seed) for scheme in SCHEMES
             for mix in MIXES if mix > 0.0 for seed in args.seeds]

    rows = []
    for start in range(0, len(jobs), args.workers):
        processes = []
        for mix, scheme, seed in jobs[start:start + args.workers]:
            processes.append(((mix, scheme, seed), subprocess.Popen(
                [sys.executable, __file__, "--one", "--mix", str(mix),
                 "--scheme", ("shipped" if scheme == "reference" else scheme),
                 "--seed", str(seed), "--demand", args.demand, "--dt", str(args.dt),
                 "--warmup-steps", str(args.warmup_steps),
                 "--duration-steps", str(args.duration_steps),
                 "--decision-interval-s", str(args.decision_interval_s),
                 "--topology", args.topology, "--work-dir", args.work_dir],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        for (mix, scheme, seed), process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"  FAILED {scheme} p={mix} seed={seed}: {stderr[-500:]}", flush=True)
                continue
            row = json.loads(stdout.strip().splitlines()[-1])
            row["scheme"] = scheme
            rows.append(row)
            print(f"  {scheme:>9} p={mix:<5g} seed {seed:>3}: arrivals {row['arrivals']:>4} "
                  f"speed {row['mean_speed']:.2f} delay {row['mean_delay_recent']:.0f} "
                  f"stopped {row['stopped_fraction']:.3f} "
                  f"cmd {row['commanded']}/{row['commanded'] + row['released']}", flush=True)

    reference = [r["arrivals"] for r in rows if r["scheme"] == "reference"]
    print(f"\n{'scheme':>9} {'p':>5} {'arrivals':>16} {'paired vs p=0':>16} "
          f"{'speed':>6} {'delay':>7} {'stopped':>8}")
    by_seed = {r["seed"]: r["arrivals"] for r in rows if r["scheme"] == "reference"}
    print(f"{'reference':>9} {0.0:>5g} {statistics.fmean(reference):>9.1f} +/- "
          f"{statistics.stdev(reference):>4.1f} {'':>16} "
          f"{statistics.fmean(r['mean_speed'] for r in rows if r['scheme'] == 'reference'):>6.2f} "
          f"{statistics.fmean(r['mean_delay_recent'] for r in rows if r['scheme'] == 'reference'):>7.0f} "
          f"{statistics.fmean(r['stopped_fraction'] for r in rows if r['scheme'] == 'reference'):>8.3f}")
    for scheme in SCHEMES:
        for mix in MIXES:
            if mix == 0.0:
                continue
            group = [r for r in rows if r["scheme"] == scheme and r["mix"] == mix]
            if not group:
                continue
            arrivals = [r["arrivals"] for r in group]
            paired = [r["arrivals"] - by_seed[r["seed"]] for r in group if r["seed"] in by_seed]
            se = (statistics.stdev(paired) / len(paired) ** 0.5) if len(paired) > 1 else 0.0
            verdict = "real" if abs(statistics.fmean(paired)) > 2 * se else "no effect"
            print(f"{scheme:>9} {mix:>5g} {statistics.fmean(arrivals):>9.1f} +/- "
                  f"{(statistics.stdev(arrivals) if len(arrivals) > 1 else 0.0):>4.1f} "
                  f"{statistics.fmean(paired):>+9.1f} +/- {se:>4.1f} "
                  f"{statistics.fmean(r['mean_speed'] for r in group):>6.2f} "
                  f"{statistics.fmean(r['mean_delay_recent'] for r in group):>7.0f} "
                  f"{statistics.fmean(r['stopped_fraction'] for r in group):>8.3f}  {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
