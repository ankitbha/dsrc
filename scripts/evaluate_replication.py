#!/usr/bin/env python3
"""Task 69: trained MAPPO against `no_av` for throughput on one topology.

This is the paper's only flow-level number, so the two things that would quietly
invalidate it are both enforced here rather than left to the operator.

**The sensing block comes from the checkpoint.** `src/envs/topology_env.py` builds
every agent observation through `SensingConfig`, so the block is the actor's input
distribution. A policy trained under one block and scored under another measures
the mismatch, not the controller. `scripts/evaluate_policy.py` cannot express this
at all -- it has no sensing argument -- which is why this exists beside it.

**AV population is set through `demand.av_penetration`.** `--controlled-vehicles`
is inert on every topology except ring: `HighwayTopologyEnv` clears `agent_ids` and
defers to the demand spawner whenever continuous demand is active. Using it on
`inverted_tree` yields an evaluation with no AVs in it, which reads as a controller
that changes nothing.

    python3 scripts/evaluate_replication.py --checkpoint <dir> [--seeds 7 17 27 ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.analysis.observation_audit import SampleSpec, build_env_config  # noqa: E402
from src.baselines.registry import make_baseline  # noqa: E402
from src.envs.topology_env import HighwayTopologyEnv  # noqa: E402
from src.rl.controller import LearnedPolicyController  # noqa: E402


def sensing_from_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    """The block the actor was trained under, or a refusal.

    Refusing beats defaulting: a silent fallback to the library defaults is exactly
    the mismatch this script exists to prevent, and it would not show up anywhere
    in the output.
    """
    resolved = checkpoint_dir / "config_resolved.yaml"
    if not resolved.exists():
        raise SystemExit(f"{resolved} missing: cannot tell what sensing this actor was trained under")
    config = yaml.safe_load(resolved.read_text()) or {}
    sensing = (config.get("training") or {}).get("sensing")
    if not sensing:
        raise SystemExit(
            f"{resolved} records no sensing block, so this checkpoint cannot be "
            "evaluated under the profile it was trained on"
        )
    return dict(sensing)


#: `build_env_config` resolves the safety mode by looking the controller name up in
#: the baseline registry, which does not contain the learned policy. This stands in
#: for it while the config is built: it is a real registry entry declaring the same
#: `integrated_rl` mode the trained actor runs behind, and the controller block is
#: overwritten immediately afterwards so the name never reaches the environment.
_SAFETY_MODE_PROXY = "backpressure"


def run_one(topology: str, demand: str, penetration: float, seed: int,
            sensing: dict[str, Any], controller: Any, controller_name: str,
            duration_steps: int, simulator: str = "highway_env",
            warmup_steps: int = 0, work_dir: str | None = None) -> dict[str, Any]:
    """One evaluation run, on whichever simulator was asked for.

    The simulator is named rather than inferred, because the two produce different
    observation distributions and a row that does not say which road it came from
    cannot be compared with anything.
    """
    is_learned = controller_name != "no_av"
    spec = SampleSpec(topology=topology,
                      controller=_SAFETY_MODE_PROXY if is_learned else "no_av",
                      seed=seed, duration_steps=duration_steps, demand=demand,
                      human_model="normal", av_penetration=penetration)
    config = build_env_config(spec)
    config["sensing"] = dict(sensing)
    if is_learned:
        config["controller"] = {"name": controller_name, "family": "rl",
                                "safety_mode": "integrated_rl"}
    if simulator == "sumo":
        from src.sumo.env import SumoTopologyEnv

        config["warmup_steps"] = warmup_steps
        if work_dir:
            config["work_dir"] = work_dir
        env: Any = SumoTopologyEnv(topology, config)
    elif simulator == "highway_env":
        env = HighwayTopologyEnv(topology, config)
    else:
        raise ValueError(f"unsupported simulator {simulator!r}")
    try:
        observations, _ = env.reset(seed=seed)
        terminated = truncated = False
        metrics: dict[str, Any] = {}
        while not (terminated or truncated):
            actions = controller.act(observations, global_state=None)
            observations, _, terminated, truncated, info = env.step(actions)
            metrics = info.get("metrics", {}) or {}
        arrived = getattr(env, "arrived_total", None)
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "seed": seed, "demand": demand, "penetration": penetration,
        "simulator": simulator,
        "completed": bool(truncated),
        "throughput": metrics.get("throughput_recent"),
        # Arrivals over the whole run, which is the throughput measure that does not
        # depend on where the 60 s rolling window happens to fall at the final step.
        "arrived": arrived,
        "mean_speed": metrics.get("mean_speed"),
        "jam_fraction": metrics.get("jam_fraction"),
        "collisions": metrics.get("collision_count"),
    }


def summarise(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["completed"]]
    take = lambda key: [float(r[key]) for r in done if r.get(key) is not None]
    out = {"arm": label, "runs": len(rows), "completed": len(done)}
    for key in ("throughput", "mean_speed", "jam_fraction"):
        values = take(key)
        out[key] = statistics.mean(values) if values else None
        out[f"{key}_n"] = len(values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="training output directory")
    parser.add_argument("--topology", default="inverted_tree")
    parser.add_argument("--demands", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--penetrations", nargs="+", type=float, default=[0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--duration-steps", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="write the rows as JSON here")
    parser.add_argument("--simulator", default="highway_env",
                        choices=["highway_env", "sumo"],
                        help="which simulator to evaluate on; recorded in every row")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="SUMO only: steps run before the episode is observed")
    parser.add_argument("--work-dir", default=None,
                        help="SUMO only: where generated network and demand files go")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    sensing = sensing_from_checkpoint(checkpoint_dir)
    print(f"  sensing block from the checkpoint: {sensing}")

    actor_path = checkpoint_dir / "actor.pt"
    if not actor_path.exists():
        raise SystemExit(f"{actor_path} missing")
    learned = LearnedPolicyController.from_checkpoint(actor_path, device=args.device)

    rows: list[dict[str, Any]] = []
    for arm, controller, name in (("mappo", learned, "learned_policy"),
                                  ("no_av", make_baseline("no_av"), "no_av")):
        if hasattr(controller, "reset"):
            controller.reset(env_metadata={"topology_id": args.topology}, seed=args.seeds[0])
        for demand in args.demands:
            for penetration in args.penetrations:
                for seed in args.seeds:
                    row = run_one(args.topology, demand, penetration, seed, sensing,
                                  controller, name, args.duration_steps,
                                  simulator=args.simulator,
                                  warmup_steps=args.warmup_steps,
                                  work_dir=args.work_dir)
                    row["arm"] = arm
                    rows.append(row)
        print(f"  {arm}: {sum(1 for r in rows if r['arm'] == arm)} runs done", flush=True)

    print()
    for arm in ("no_av", "mappo"):
        s = summarise(arm, [r for r in rows if r["arm"] == arm])
        thr = f"{s['throughput']:.2f}" if s["throughput"] is not None else "-"
        spd = f"{s['mean_speed']:.2f}" if s["mean_speed"] is not None else "-"
        print(f"  {arm:<8} completed {s['completed']:>3}/{s['runs']:<3} "
              f"throughput {thr:>7} (n={s['throughput_n']})  mean_speed {spd:>6}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\n  rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
