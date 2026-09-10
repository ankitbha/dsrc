"""Evaluate a trained policy against no control, per the pre-registration.

The metrics, seeds, comparators and the bar for calling a difference real were
fixed in `plans/task_list.md` BEFORE any policy was trained -- task 93 for the burst
scenario, task 99 for the corrected road, task 103 for the steady demand above
capacity that the shipped config now uses. This script implements that and nothing
else: it does not choose a metric, a seed or an arm.

    .venv/bin/python scripts/evaluate_burst_scenario.py --checkpoint-root <dir>

The name is historical: it reads the scenario out of the training config, so it runs
whatever demand the policy trained on. The burst-only metric, `recovery_s`, is None
under a demand that declares no burst rather than being computed from a shock that
did not happen.

The learned arm uses `latest_actor.pt`, the FINAL policy, not `actor.pt`, the one
the trainer scored highest. Selecting a checkpoint is a choice made after seeing
results, and the point of the pre-registration is that no such choice is made.

**Penetration is NOT zeroed for the `no_av` arm**, and that is deliberate. The two
vTypes are identical except for their id and colour, so an uncommanded AV is a human
driver; but the fleet is drawn from one vTypeDistribution per vehicle, so changing
the penetration changes the random draw and therefore the traffic realisation. Every
arm sees the same traffic and the arms differ only in whether a controller commands
anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.registry import make_baseline  # noqa: E402
from src.config.loaders import load_named_config  # noqa: E402
from src.rl.controller import LearnedPolicyController  # noqa: E402
from src.sumo.env import SumoTopologyEnv  # noqa: E402

SEEDS = (7, 17, 27, 37, 47)
#: The queue is "cleared" below this many vehicles; the burst peaks it near 40.
CLEARED_QUEUE = 10
#: Metrics reported as a mean over the episode's steps. `mean_delay_recent`,
#: `stopped_fraction` and `mean_abs_jerk` were added in task 103 and are the axes
#: the predecessor paper's gains were largest on; without them an evaluation can
#: only see throughput and speed.
EPISODE_MEANS = ("mean_speed", "speed_std", "throughput_recent",
                 "mean_delay_recent", "stopped_fraction", "mean_abs_jerk")
#: What is paired against `no_av` and judged on the two-standard-error bar.
PAIRED_KEYS = ("arrivals", "mean_delay_recent", "stopped_fraction",
               "mean_abs_jerk", "trough_speed")


def decision_interval_steps(training) -> int:
    """How many simulation steps one action is held for, from the training config.

    A policy trained at one decision per simulated second and evaluated at ten
    decisions per second is not the policy that was trained: it would be asked for
    an action in states it never saw itself in, and the actuation rate -- which is
    what the reward was earned at -- would differ by a factor of ten. Read from the
    same config the environment is built from, for the same reason `human_model`
    is.
    """
    interval = training.get("decision_interval_s")
    if interval is None:
        return 1
    dt = float(training["dt"]) or 1.0
    return max(1, int(round(float(interval) / dt)))


def run_one(*, controller, seed, training, work_dir):
    """One episode. Returns the pre-registered metrics and nothing else."""
    dt = float(training["dt"])
    duration = int(training["duration_steps"])
    demand = load_named_config("demand", str(training["demand"]))
    burst_end_s = float((demand.get("burst") or {}).get("end_s", 0.0))
    config = {
        "topology": load_named_config("topology", str(training["topology"])),
        "demand": demand, "duration_steps": duration, "dt": dt,
        "warmup_steps": int(training["warmup_steps"]),
        "sensing": dict(training.get("sensing") or {}),
        "work_dir": work_dir,
    }
    # The driving model MUST come from the training config. Without this the policy
    # was evaluated under SUMO's default Krauss while it had been trained under the
    # calibrated Wiedemann-99: a road with no capacity drop, which is not the road
    # it learned on, and a comparison between two different environments.
    if training.get("human_model"):
        config["human_model"] = load_named_config(
            "human_model", str(training["human_model"]))
    env = SumoTopologyEnv(str(training["topology"]), config)
    every = decision_interval_steps(training)
    has_burst = bool((demand.get("burst") or {}).get("enabled", False))
    observations, _ = env.reset(seed=seed)
    try:
        speeds, queues, roadblock = [], [], 0.0
        series = {key: [] for key in EPISODE_MEANS}
        latent = 0
        recovered_at = None
        actions: dict = {}
        for step in range(duration):
            # The action is chosen once per decision interval and held, which is
            # what the trainer does. `setSpeed` and `setTau` persist in SUMO, so
            # holding means passing nothing rather than re-issuing.
            if step % every == 0:
                actions = controller.act(observations, global_state=None) if controller else {}
            else:
                actions = {}
            observations, _, _, _, info = env.step(actions)
            metrics = info.get("metrics", {}) or {}
            for key in EPISODE_MEANS:
                series[key].append(float(metrics.get(key, 0.0) or 0.0))
            speeds.append(float(metrics.get("mean_speed", 0.0)))
            queues.append(int(metrics.get("queue_length_total", 0)))
            roadblock += float(metrics.get("rolling_roadblock_score", 0.0))
            latent = int(metrics.get("latent_demand", 0) or 0)
            now = step * dt
            if (has_burst and recovered_at is None and now > burst_end_s
                    and queues[-1] < CLEARED_QUEUE):
                recovered_at = now - burst_end_s
        window = max(1, int(round(60.0 / dt)))
        rolling = [statistics.fmean(speeds[i:i + window])
                   for i in range(0, len(speeds) - window + 1)]
        row = {
            "seed": seed,
            "arrivals": env.arrived_total,
            "trough_speed": min(rolling) if rolling else 0.0,
            # None under a demand that declares no burst: there is no shock for the
            # queue to recover from, and a number computed anyway would be the time
            # the queue first fell below ten vehicles for an unrelated reason.
            "recovery_s": recovered_at,
            "roadblock_sum": roadblock,
            "collisions": env.collision_count,
            "latent_end": latent,
        }
        row.update({key: statistics.fmean(values) if values else 0.0
                    for key, values in series.items()})
        return row
    finally:
        env.close()


def paired_verdict(arm_rows, reference_rows, key):
    """The pre-registered bar: a difference counts only above two standard errors."""
    by_seed = {row["seed"]: row for row in reference_rows}
    paired = [(row[key] - by_seed[row["seed"]][key])
              for row in arm_rows if row["seed"] in by_seed
              and row[key] is not None and by_seed[row["seed"]][key] is not None]
    if len(paired) < 2:
        return None, None, "too few pairs"
    mean = statistics.fmean(paired)
    standard_error = statistics.stdev(paired) / len(paired) ** 0.5
    verdict = "real" if abs(mean) > 2 * standard_error else "no effect"
    return mean, standard_error, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="mappo_sumo")
    parser.add_argument("--checkpoint-root", required=True,
                        help="directory holding mappo_<topology>_<profile>_seed<N>/")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="traffic seeds every arm is evaluated on")
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=list(SEEDS),
                        help="the seeds the policies were TRAINED with; each policy is "
                             "run on every traffic seed and averaged, because a policy "
                             "and a traffic realisation are independent choices")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    training = load_named_config("training", args.training)
    root = Path(args.checkpoint_root)
    rows: dict[str, list[dict[str, Any]]] = {}

    for arm in ("no_av", "density_lookup", "mappo"):
        rows[arm] = []
        for seed in args.seeds:
            if arm == "mappo":
                # Every trained policy is run on this traffic seed and the results
                # averaged, so the arm's value at a seed is "what a policy from this
                # training procedure does on this traffic" rather than the result of
                # one policy. Pairing against no_av then compares like with like on
                # the same traffic. The previous form looked for a checkpoint whose
                # seed equalled the traffic seed, which finds nothing as soon as the
                # evaluation seeds are chosen disjoint from the training seeds.
                per_policy = []
                for policy_seed in args.policy_seeds:
                    actor = (root / f"mappo_{training['topology']}_"
                                    f"{training['action_profile']}_seed{policy_seed}"
                             / "latest_actor.pt")
                    if not actor.exists():
                        print(f"  missing {actor}", flush=True)
                        continue
                    controller = LearnedPolicyController.from_checkpoint(
                        actor, device=args.device)
                    if hasattr(controller, "reset"):
                        controller.reset(env_metadata={"topology_id": training["topology"]},
                                         seed=seed)
                    per_policy.append(run_one(controller=controller, seed=seed,
                                              training=training, work_dir=args.work_dir))
                if not per_policy:
                    continue
                merged = {"seed": seed, "policies": len(per_policy)}
                for key in ("arrivals", "trough_speed", "roadblock_sum", "collisions",
                            "latent_end", *EPISODE_MEANS):
                    merged[key] = statistics.fmean(r[key] for r in per_policy)
                recovered = [r["recovery_s"] for r in per_policy if r["recovery_s"] is not None]
                merged["recovery_s"] = statistics.fmean(recovered) if recovered else None
                rows[arm].append(merged)
            else:
                controller = None if arm == "no_av" else make_baseline(arm)
                if controller is not None and hasattr(controller, "reset"):
                    controller.reset(env_metadata={"topology_id": training["topology"]},
                                     seed=seed)
                rows[arm].append(run_one(controller=controller, seed=seed,
                                         training=training, work_dir=args.work_dir))
            print(f"  {arm} seed {seed}: {rows[arm][-1]}", flush=True)

    print(f"\n{'arm':>16} {'arrivals':>16} {'delay s':>9} {'stopped':>8} "
          f"{'jerk':>7} {'trough m/s':>11} {'latent':>7} {'roadblock':>10} "
          f"{'collisions':>11}")
    for arm in ("no_av", "density_lookup", "mappo"):
        if not rows[arm]:
            continue
        arrivals = [r["arrivals"] for r in rows[arm]]
        print(f"{arm:>16} {statistics.fmean(arrivals):>9.1f} +/- "
              f"{statistics.stdev(arrivals) if len(arrivals) > 1 else 0.0:>4.1f} "
              f"{statistics.fmean(r['mean_delay_recent'] for r in rows[arm]):>9.1f} "
              f"{statistics.fmean(r['stopped_fraction'] for r in rows[arm]):>8.3f} "
              f"{statistics.fmean(r['mean_abs_jerk'] for r in rows[arm]):>7.3f} "
              f"{statistics.fmean(r['trough_speed'] for r in rows[arm]):>11.2f} "
              f"{statistics.fmean(r['latent_end'] for r in rows[arm]):>7.1f} "
              f"{statistics.fmean(r['roadblock_sum'] for r in rows[arm]):>10.1f} "
              f"{sum(r['collisions'] for r in rows[arm]):>11}")

    print("\nPaired against no_av on the same seeds; the bar is two standard errors.")
    for arm in ("density_lookup", "mappo"):
        for key in PAIRED_KEYS:
            mean, se, verdict = paired_verdict(rows[arm], rows["no_av"], key)
            if mean is None:
                continue
            print(f"  {arm:>15} {key:>13}: {mean:+8.2f} +/- {se:5.2f}  -> {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\n  rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
