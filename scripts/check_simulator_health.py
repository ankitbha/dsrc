#!/usr/bin/env python3
"""Classify each simulator cell against four health criteria.

The simulator is the only source of a flow-level claim, so this establishes
whether it can produce one and where. A cell is healthy only if congestion is
reachable, baselines separate, episodes complete, and throughput does not
collapse. Reports the recommended operating point, or states that none exists.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.simulator_health import (
    DEFAULT_MIN_COMPLETED_SEEDS,
    DEFAULT_MIN_JAM_FRACTION,
    DEFAULT_MIN_SPEED_SEPARATION_MPS,
    DEFAULT_MIN_THROUGHPUT_RATIO,
    CellSpec,
    HealthVerdict,
    assess_cell,
    run_condition,
    select_operating_points,
)
from src.baselines import BASELINE_NAMES

TOPOLOGIES = (
    "ring",
    "straight_single_lane",
    "straight_multilane",
    "merge",
    "inverted_tree",
    "inverted_tree_bottleneck",
)
DEMANDS = ("low", "medium", "high", "burst")
AV_PENETRATIONS = (0.05, 0.10, 0.20)
CONTROLLERS = ("no_av", "cooperative_smoothing", "backpressure")
SEEDS = (7, 17, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    parser.add_argument("--demands", nargs="+", default=list(DEMANDS))
    parser.add_argument("--av-penetrations", nargs="+", type=float, default=list(AV_PENETRATIONS))
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=list(BASELINE_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--duration-steps", type=int, default=120,
                        help="congestion needs ~90+ steps to develop; shorter sweeps give false negatives")
    parser.add_argument("--min-jam-fraction", type=float, default=DEFAULT_MIN_JAM_FRACTION)
    parser.add_argument("--min-speed-separation", type=float, default=DEFAULT_MIN_SPEED_SEPARATION_MPS)
    parser.add_argument("--min-completed-seeds", type=int, default=DEFAULT_MIN_COMPLETED_SEEDS)
    parser.add_argument("--min-throughput-ratio", type=float, default=DEFAULT_MIN_THROUGHPUT_RATIO)
    parser.add_argument("--output-root", default="outputs/validation/simulator_health")
    return parser.parse_args()


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def cell_row(verdict: HealthVerdict) -> str:
    mark = lambda ok: "yes" if ok else "NO"
    failed = ", ".join(verdict.failed_criteria) or "-"
    return (
        f"| `{verdict.cell.cell_id}` "
        f"| {mark(verdict.congestion_reachable)} ({verdict.reference_jam_fraction:.3f}) "
        f"| {mark(verdict.baselines_separate)} ({verdict.best_speed_separation:.2f}) "
        f"| {mark(verdict.episodes_complete)} ({verdict.min_completed_seeds}) "
        f"| {mark(verdict.throughput_holds)} ({verdict.worst_throughput_ratio:.2f}) "
        f"| {'HEALTHY' if verdict.healthy else failed} |"
    )


def write_report(verdicts: list[HealthVerdict], operating: tuple[HealthVerdict, ...],
                 output_root: Path, args: argparse.Namespace) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "cells": [
            {
                "cell": dataclasses.asdict(v.cell),
                "healthy": v.healthy,
                "failed_criteria": list(v.failed_criteria),
                "congestion_reachable": v.congestion_reachable,
                "reference_jam_fraction": v.reference_jam_fraction,
                "baselines_separate": v.baselines_separate,
                "best_speed_separation": v.best_speed_separation,
                "best_controller": v.best_controller,
                "episodes_complete": v.episodes_complete,
                "min_completed_seeds": v.min_completed_seeds,
                "throughput_holds": v.throughput_holds,
                "worst_throughput_ratio": v.worst_throughput_ratio,
                "by_controller": {k: dataclasses.asdict(s) for k, s in v.by_controller.items()},
            }
            for v in verdicts
        ],
        "operating_points": [v.cell.cell_id for v in operating],
    }
    json_path = output_root / "simulator_health.json"
    json_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))

    lines = [
        "# Simulator health",
        "",
        f"{len(verdicts)} cells, {args.duration_steps} steps, seeds {args.seeds}.",
        f"Thresholds: jam >= {args.min_jam_fraction:g}, separation >= {args.min_speed_separation:g} m/s, "
        f"completed seeds >= {args.min_completed_seeds}, throughput ratio >= {args.min_throughput_ratio:g}.",
        "",
    ]
    if operating:
        lines += ["## Recommended operating point", ""]
        lines += [f"- `{v.cell.cell_id}` — separation {v.best_speed_separation:.2f} m/s "
                  f"via `{v.best_controller}`" for v in operating]
    else:
        lines += [
            "## No usable operating point",
            "",
            "No cell passes all four criteria. On this evidence the simulator cannot",
            "currently support a flow-level claim, and repairing it is a prerequisite",
            "for any sufficiency study.",
        ]
    lines += [
        "",
        "## All cells",
        "",
        "| cell | congestion (jam) | separation (m/s) | complete (seeds) | throughput (ratio) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    lines += [cell_row(v) for v in sorted(verdicts, key=lambda v: v.cell.cell_id)]
    md_path = output_root / "simulator_health.md"
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": json_path, "markdown": md_path}


def main() -> int:
    args = parse_args()
    cells = [
        CellSpec(topology=t, demand=d, av_penetration=p)
        for t in args.topologies for d in args.demands for p in args.av_penetrations
    ]
    total = len(cells) * len(args.controllers) * len(args.seeds)
    print(f"assessing {len(cells)} cells ({total} runs)...", flush=True)

    verdicts = []
    for index, cell in enumerate(cells, 1):
        runs = [
            run_condition(cell, controller, seed, duration_steps=args.duration_steps)
            for controller in args.controllers
            for seed in args.seeds
        ]
        verdicts.append(
            assess_cell(
                cell, runs,
                min_jam_fraction=args.min_jam_fraction,
                min_speed_separation=args.min_speed_separation,
                min_completed_seeds=args.min_completed_seeds,
                min_throughput_ratio=args.min_throughput_ratio,
            )
        )
        print(f"  [{index}/{len(cells)}] {cell.cell_id}: "
              f"{'HEALTHY' if verdicts[-1].healthy else ','.join(verdicts[-1].failed_criteria)}", flush=True)

    operating = select_operating_points(verdicts)
    paths = write_report(verdicts, operating, Path(args.output_root), args)
    print(f"\nhealthy cells: {len(operating)}/{len(verdicts)}")
    if operating:
        print(f"recommended: {operating[0].cell.cell_id} (separation {operating[0].best_speed_separation:.2f} m/s)")
    else:
        print("NO usable operating point: no cell passes all four criteria")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
