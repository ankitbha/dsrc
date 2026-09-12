#!/usr/bin/env python3
"""Evaluate the committed Mainz checkpoints on seeds 16-30, paired against no control.

    .venv/bin/python scripts/evaluate_mainz_checkpoints.py \\
        > results/evaluation/mainz_paired_seeds.log

Reads `results/checkpoints/mainz_{here,src}_best.pt` and never writes them. For each
arm and each seed it runs one episode, then reports the paired gain over no control:
each seed is one traffic realisation that every arm ran, so the seed's contribution
to the level of flow is common to every arm and cancels when the comparison is a
per-seed difference rather than a difference of separately-computed means. The bar is
`2 * stdev(differences) / sqrt(n)` with the sample standard deviation, `n - 1`; see
`paired_statistics`.

`train_mainz_src.py`'s own `TEST_SEEDS` is `range(16, 21)`, five seeds, read once at
the end of training. This script reads seeds 16-30 against a checkpoint already on
disk, any number of times, which is a different job: re-running the driver to widen
the read would retrain and would keep a different checkpoint, because selection is by
validation return over a run that is not seeded identically end to end. `run_episode`,
`evaluate_no_control` and `SCENARIOS` are imported from `train_mainz_src` rather than
re-implemented, so the two never drift apart on what they measure.

The no-control baseline is recomputed here, per seed, in the same process: it does
not exist for seeds 21-30 anywhere, and pairing is only defined against the same
seed. `gate_for_arm` derives the entry-gate setting from the arm name in one place --
on for `here` and `src`, never for `no_control` -- and `_check_gate_asymmetry` checks
the run actually came out that way rather than assuming it.

The step length, episode duration and throughput-window start below are not this
script's defaults so much as a recovered fact: no command line was ever written down
for the checkpoints this reads, and `train_mainz_src.py`'s own argparse defaults
(`--step-length 1.0 --window-start-s 0.0`) reproduce neither the committed
`no_control` block nor a plausible reading of the training logs. `0.5s / 2500s / 900s`
does reproduce the committed `no_control` block on all seven of its metrics, and
`mean_vehicles = 858.0370370370371 = 115835/(5*27)` pins the window start
independently: 27 accumulation samples after ten 60s decisions past a 300s warmup is
t=900s. Both are recorded in every run's own `config` block so this does not have to
be recovered a second time.

This runs the action mapping that is actually in the tree, `SPEED_ACTION_FRACTIONS`
(a fraction of each edge's own speed limit), not the `SPEED_ACTIONS_KMH` literal
30/45/60 km/h mapping the committed checkpoints were originally read under. Every
controlled edge on Mainz has a free-flow speed of 60.012 km/h rather than 60.000, so
the two mappings differ by 0.02% in commanded speed -- enough to move a five-seed
mean flow by 8.4 veh/h. The stored `test` blocks in `results/checkpoints/*_result.json`
are therefore artifacts of a mapping this repository no longer contains; this script
does not carry a flag to reproduce it, because the only reason to would be to match a
number the code in the tree cannot produce.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_mainz_src import SCENARIOS, evaluate_no_control, run_episode  # noqa: E402
from src.rl.src_q import SrcQNetwork  # noqa: E402
from src.sumo.mainz import HERE_FEATURES, MainzEnv, SRC_FEATURES  # noqa: E402

#: A trained arm's checkpoint file and its feature tuple, bound together under one
#: name so a checkpoint can never be pointed at the wrong feature tuple by passing
#: the two as independent flags.
CHECKPOINT_FEATURES: dict[str, tuple[str, ...]] = {"here": HERE_FEATURES, "src": SRC_FEATURES}

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(16, 31))
DEFAULT_ARMS: tuple[str, ...] = ("no_control", "here", "src")
DEFAULT_STEP_LENGTH_S: float = 0.5
DEFAULT_DURATION_S: float = 2500.0
DEFAULT_WINDOW_START_S: float = 900.0

#: `MainzEnv`'s own default, read from the class rather than copied as a literal so
#: this cannot go stale the way the run's command line itself once did.
WARMUP_S: float = inspect.signature(MainzEnv).parameters["warmup_s"].default

#: The per-episode fields carried into the JSONL record and the per-seed table.
EPISODE_FIELDS: tuple[str, ...] = (
    "flow", "arrived", "return", "mean_speed_kmh", "space_mean_speed_kmh",
    "mean_vehicles", "held_at_entry",
)


# --------------------------------------------------------------------- checkpoints


def _num_segments(paths: dict) -> int:
    return len(json.loads((REPO_ROOT / paths["segments"]).read_text()))


def _build_and_load(checkpoint_path: Path, features: tuple[str, ...],
                    num_segments: int) -> SrcQNetwork:
    """Construct the network for `features` and load `checkpoint_path` into it.

    `load_state_dict` fails on a shape mismatch by itself, but the message it raises
    names tensor shapes rather than segments and features, so the width is checked
    directly first and the error names what actually disagrees.
    """
    model = SrcQNetwork(num_segments, len(features), 3)
    state_dict = torch.load(checkpoint_path, weights_only=True)
    expected_width = num_segments * len(features)
    checkpoint_width = state_dict["stack.0.weight"].shape[1]
    if checkpoint_width != expected_width:
        raise ValueError(
            f"{checkpoint_path} was built for an input width of {checkpoint_width}, "
            f"but {len(features)} features over {num_segments} segments need "
            f"{expected_width}; this checkpoint and this feature tuple do not match")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_arm_model(name: str, checkpoint_dir: Path,
                   num_segments: int) -> tuple[SrcQNetwork, tuple[str, ...]]:
    """Bind a trained arm's checkpoint and feature tuple, and load it."""
    if name not in CHECKPOINT_FEATURES:
        raise ValueError(f"{name!r} has no checkpoint; only here/src are trained arms")
    features = CHECKPOINT_FEATURES[name]
    checkpoint_path = checkpoint_dir / f"mainz_{name}_best.pt"
    model = _build_and_load(checkpoint_path, features, num_segments)
    return model, features


def gate_for_arm(name: str) -> bool:
    """Whether the entry gate is on for this arm: every trained arm, and only those.

    The gate stands in for the policy acting on a link above each entry link, which
    this scenario does not simulate (`src/sumo/mainz.py`'s `_admit`). A run with no
    policy has nothing acting on that link either, so gating its entries would credit
    it with a control action it never takes. This is derived from the arm's name in
    this one place rather than accepted as a caller-supplied flag.
    """
    return name != "no_control"


def _zero_output_head(model: SrcQNetwork) -> None:
    """Zero the output layer, so every Q-value is 0 and argmax picks index 0 always.

    With every controlled segment always choosing action 0, the policy becomes a
    standing order to drive every controlled vehicle at half its road's free-flow
    speed. Used to prove this evaluator reports an obviously wrong policy as wrong
    before its numbers on the real checkpoints are believed.
    """
    with torch.no_grad():
        model.stack[2].weight.zero_()
        model.stack[2].bias.zero_()


# ---------------------------------------------------------------------- statistics


def paired_statistics(arm_flows: list[float], baseline_flows: list[float]) -> dict:
    """The paired gain and its bar, seed by seed.

    `d_s = arm_flows[s] - baseline_flows[s]` for each seed, then the gain is the mean
    of `d_s` and the bar is `2 * stdev(d_s) / sqrt(n)` with the SAMPLE standard
    deviation (`statistics.stdev`, `n - 1`), not the population one
    (`statistics.pstdev`, `n`), and not `2 * sqrt(se_arm**2 + se_baseline**2)`, which
    is the same two numbers read as two independent means instead of one paired
    sample. Each seed is one traffic realisation that both the arm and the baseline
    ran, so the seed's own contribution to the level of flow is common to both and
    cancels in `d_s`; computing the bar from each side's separate spread does not
    cancel it and reports a wider interval that describes the wrong quantity.
    """
    if len(arm_flows) != len(baseline_flows):
        raise ValueError(
            f"paired statistics need one baseline reading per arm reading, got "
            f"{len(arm_flows)} and {len(baseline_flows)}")
    n = len(arm_flows)
    if n < 2:
        raise ValueError(f"a sample standard deviation needs at least two seeds, got {n}")
    differences = [a - b for a, b in zip(arm_flows, baseline_flows)]
    gain = statistics.fmean(differences)
    two_se = 2.0 * statistics.stdev(differences) / math.sqrt(n)
    baseline_mean = statistics.fmean(baseline_flows)
    return {
        "differences": differences,
        "gain": gain,
        "two_se": two_se,
        "resolved": abs(gain) > two_se,
        "percent": 100.0 * gain / baseline_mean if baseline_mean else float("nan"),
    }


def per_arm_two_se(values: list[float]) -> float:
    """`2 * stdev(values) / sqrt(n)` for one arm's own readings, not the paired bar.

    Reported alongside the paired gain because the source documents quote it, but it
    describes one arm's spread across seeds and is never a substitute for
    `paired_statistics`'s bar.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"a sample standard deviation needs at least two seeds, got {n}")
    return 2.0 * statistics.stdev(values) / math.sqrt(n)


# ------------------------------------------------------------------------- the run


def _demand_veh_per_h(paths: dict, duration_s: float) -> float:
    """The schedule's own arrival rate, informational only: it is fixed by the
    schedule file and does not change with how the episode is configured to read it.
    """
    rows = json.loads((REPO_ROOT / paths["schedule"]).read_text())
    return len(rows) / duration_s * 3600.0


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _compact_metrics(metrics: dict) -> dict:
    """The serialisable subset of a `run_episode` / `evaluate_no_control` result.

    `run_episode` also returns `states`, `actions` and `rewards` -- the trajectory
    tensors `td_loss` trains on -- which this evaluator never trains with and must
    not carry into a JSON file.
    """
    return {field: metrics[field] for field in EPISODE_FIELDS}


def _episode_record(arm: str, seed: int, metrics: dict) -> dict:
    return {"arm": arm, "seed": seed, **_compact_metrics(metrics)}


def _check_gate_asymmetry(per_seed: dict[str, dict[int, dict]]) -> None:
    """Confirm the entry gate came out asymmetric rather than assuming it did.

    Every `no_control` episode must show `held_at_entry == 0`; if any trained arm ran,
    at least one of its episodes must show `held_at_entry > 0`. Either direction of
    failure means the gate was applied to the wrong arm and the comparison this script
    exists to make is not the comparison that actually ran.
    """
    if "no_control" in per_seed:
        for seed, metrics in per_seed["no_control"].items():
            if metrics["held_at_entry"] != 0:
                raise AssertionError(
                    f"no_control seed {seed} held {metrics['held_at_entry']} vehicles "
                    "at entry; the gate must never apply to an uncontrolled run")
    trained_metrics = [
        metrics for name, seeds in per_seed.items() if name != "no_control"
        for metrics in seeds.values()
    ]
    if trained_metrics and not any(m["held_at_entry"] > 0 for m in trained_metrics):
        raise AssertionError(
            "no trained-arm episode held any vehicle at entry; the entry gate does "
            "not appear to be active")


def _build_summary(*, commit: str, config: dict,
                   per_seed: dict[str, dict[int, dict]]) -> dict:
    arms_summary = {}
    for name, seed_metrics in per_seed.items():
        rows = list(seed_metrics.values())
        flows = [row["flow"] for row in rows]
        arms_summary[name] = {
            "flow_mean": statistics.fmean(flows),
            "flow_2se": per_arm_two_se(flows) if len(flows) >= 2 else None,
            "return_mean": statistics.fmean(row["return"] for row in rows),
            "arrived_mean": statistics.fmean(row["arrived"] for row in rows),
            "mean_speed_kmh_mean": statistics.fmean(row["mean_speed_kmh"] for row in rows),
            "space_mean_speed_kmh_mean":
                statistics.fmean(row["space_mean_speed_kmh"] for row in rows),
            "mean_vehicles_mean": statistics.fmean(row["mean_vehicles"] for row in rows),
            "held_at_entry_mean": statistics.fmean(row["held_at_entry"] for row in rows),
        }

    paired = {}
    baseline_seed_metrics = per_seed.get("no_control")
    if baseline_seed_metrics:
        for name, seed_metrics in per_seed.items():
            if name == "no_control":
                continue
            common_seeds = sorted(set(seed_metrics) & set(baseline_seed_metrics))
            if not common_seeds:
                continue
            arm_flows = [seed_metrics[seed]["flow"] for seed in common_seeds]
            baseline_flows = [baseline_seed_metrics[seed]["flow"] for seed in common_seeds]
            paired[name] = paired_statistics(arm_flows, baseline_flows)

    return {"commit": commit, "config": config, "per_seed": per_seed,
            "arms": arms_summary, "paired": paired}


def run(*, arms: tuple[str, ...] = DEFAULT_ARMS, seeds: tuple[int, ...] = DEFAULT_SEEDS,
        step_length: float = DEFAULT_STEP_LENGTH_S,
        duration_s: float = DEFAULT_DURATION_S,
        window_start_s: float = DEFAULT_WINDOW_START_S,
        checkpoint_dir: Path, out_dir: Path, scenario: str = "mainz") -> dict:
    """Run every (arm, seed) episode once, sequentially, and write both artifacts.

    Sequential and strictly one environment at a time: `libsumo` is process-global
    (`src/sumo/binding.py`), so a second episode may only start after the first has
    closed its own, which `run_episode` and `evaluate_no_control` each guarantee with
    a `finally`.
    """
    paths = SCENARIOS[scenario]
    num_segments = _num_segments(paths)

    models: dict[str, SrcQNetwork] = {}
    features_by_arm: dict[str, tuple[str, ...]] = {}
    checkpoints_used: dict[str, str] = {}
    for name in arms:
        if name == "no_control":
            continue
        model, features = load_arm_model(name, checkpoint_dir, num_segments)
        models[name] = model
        features_by_arm[name] = features
        checkpoints_used[name] = str(checkpoint_dir / f"mainz_{name}_best.pt")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "mainz_paired_seeds.jsonl"
    per_seed: dict[str, dict[int, dict]] = {name: {} for name in arms}

    with jsonl_path.open("w") as handle:
        for name in arms:
            for seed in seeds:
                if name == "no_control":
                    # `features` only reaches `observe()`, and the no-control loop
                    # inside `evaluate_no_control` never calls it -- every vehicle is
                    # left to the car-following model, so no observation is ever
                    # built. HERE_FEATURES is passed because the parameter is
                    # positional, not because it does anything here.
                    metrics = evaluate_no_control((seed,), HERE_FEATURES, duration_s,
                                                 step_length, window_start_s, paths)
                else:
                    metrics = run_episode(models[name], seed, features_by_arm[name],
                                          duration_s, 0.0, None, step_length,
                                          window_start_s, True, paths)
                record = _episode_record(name, seed, metrics)
                per_seed[name][seed] = _compact_metrics(metrics)
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    _check_gate_asymmetry(per_seed)

    config = {
        "seeds": list(seeds),
        "duration_s": duration_s,
        "step_length_s": step_length,
        "window_start_s": window_start_s,
        "warmup_s": WARMUP_S,
        "demand_veh_per_h": _demand_veh_per_h(paths, duration_s),
        "scenario": scenario,
        "action_set": "SPEED_ACTION_FRACTIONS",
        "checkpoints": checkpoints_used,
        "gate_entries": {name: gate_for_arm(name) for name in arms},
    }
    summary = _build_summary(commit=_git_commit(), config=config, per_seed=per_seed)

    # Whole-then-rename: a reader never sees a partially written summary.
    summary_path = out_dir / "mainz_paired_seeds.json"
    tmp_path = out_dir / "mainz_paired_seeds.json.tmp"
    tmp_path.write_text(json.dumps(summary, indent=1))
    tmp_path.rename(summary_path)

    return summary


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="mainz")
    parser.add_argument("--arms", nargs="+", choices=("no_control", "here", "src"),
                        default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="paired against no_control seed for seed; the committed "
                             "checkpoints were selected on seeds 11-15 and never see "
                             "these during training")
    parser.add_argument("--step-length", type=float, default=DEFAULT_STEP_LENGTH_S)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--window-start-s", type=float, default=DEFAULT_WINDOW_START_S,
                        help="flow is counted only from here; the ramp-up before it "
                             "is excluded from every reported rate")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--out", default="results/evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = REPO_ROOT / args.out
    checkpoint_dir = REPO_ROOT / args.checkpoint_dir

    print(f"scenario {args.scenario}  arms {args.arms}  "
          f"seeds {args.seeds[0]}-{args.seeds[-1]} ({len(args.seeds)} total)")
    print(f"step-length {args.step_length}s  duration {args.duration_s}s  "
          f"window-start {args.window_start_s}s  warmup {WARMUP_S}s")

    started = time.time()
    summary = run(arms=tuple(args.arms), seeds=tuple(args.seeds),
                  step_length=args.step_length, duration_s=args.duration_s,
                  window_start_s=args.window_start_s, checkpoint_dir=checkpoint_dir,
                  out_dir=out_dir, scenario=args.scenario)
    elapsed = time.time() - started

    print()
    for name, stats in summary["arms"].items():
        bar = f" +/- {stats['flow_2se']:.1f}" if stats["flow_2se"] is not None else ""
        print(f"  {name:<10} flow {stats['flow_mean']:>8.2f} veh/h{bar}")
    for name, stats in summary["paired"].items():
        status = "resolved" if stats["resolved"] else "NOT resolved"
        print(f"  {name} vs no_control: paired gain {stats['gain']:+.2f} "
              f"+/- {stats['two_se']:.1f} veh/h ({stats['percent']:+.1f}%) -- {status}")

    print(f"\nwrote {out_dir / 'mainz_paired_seeds.json'} and "
          f"{out_dir / 'mainz_paired_seeds.jsonl'} in {elapsed:.0f}s, at commit "
          f"{summary['commit'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
