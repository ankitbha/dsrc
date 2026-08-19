#!/usr/bin/env python3
"""Audit which encoded observation fields carry information.

Ablating a field that never varied reports it as unnecessary, which is an
artifact of the simulator rather than a finding about sensing. This produces the
list of genuinely informative fields, broken down by topology and AV count
because inertness depends on both.
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

from src.analysis.observation_audit import (
    DEFAULT_NEAR_CONSTANT_VARIANCE,
    AuditResult,
    FieldAudit,
    SampleSpec,
    run_audit,
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
CONTROLLERS = ("no_av", "cooperative_smoothing", "random_av")
AV_COUNTS = (2, 8, 20)
SEEDS = (7, 17, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=list(BASELINE_NAMES))
    parser.add_argument("--av-counts", nargs="+", type=int, default=list(AV_COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--duration-steps", type=int, default=120)
    parser.add_argument(
        "--near-constant-variance",
        type=float,
        default=DEFAULT_NEAR_CONSTANT_VARIANCE,
        help="normalised variance below which a field counts as near-constant",
    )
    parser.add_argument("--output-root", default="outputs/validation/observation_audit")
    return parser.parse_args()


def json_safe(value: object) -> object:
    """Replace non-finite floats with null.

    `constant_value` can legitimately be an infinity, and json.dumps writes that
    as the bare token `Infinity`, which strict parsers reject.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def verdict(audit: FieldAudit) -> str:
    if audit.n_samples == 0:
        return "no-coverage"
    if audit.is_strictly_constant:
        return "constant"
    if audit.is_near_constant:
        return "near-constant"
    return "informative"


def markdown_table(title: str, audits: tuple[FieldAudit, ...]) -> str:
    lines = [f"### {title}", "", "| field | n | uniq | variance | verdict | never left fallback |", "|---|---|---|---|---|---|"]
    for audit in audits:
        fallback = "n/a" if audit.never_left_fallback is None else str(audit.never_left_fallback).lower()
        lines.append(
            f"| `{audit.field}` | {audit.n_samples} | {audit.unique_values} "
            f"| {'n/a' if audit.variance is None else f'{audit.variance:.3e}'} "
            f"| {verdict(audit)} | {fallback} |"
        )
    return "\n".join(lines) + "\n"


def write_report(result: AuditResult, output_root: Path, args: argparse.Namespace) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "samples_by_condition": dict(result.samples_by_condition),
        "pooled": [dataclasses.asdict(a) for a in result.per_field_pooled],
        "by_topology": {k: [dataclasses.asdict(a) for a in v] for k, v in result.per_field_by_topology.items()},
        "by_av_count": {str(k): [dataclasses.asdict(a) for a in v] for k, v in result.per_field_by_av_count.items()},
    }
    json_path = output_root / "observation_audit.json"
    # allow_nan=False so a non-finite value raises here rather than silently
    # producing a file that strict parsers reject.
    json_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))

    total = sum(result.samples_by_condition.values())
    sections = [
        "# Observation field audit",
        "",
        f"Conditions: {len(result.samples_by_condition)}; total samples: {total}.",
        f"Near-constant threshold: {args.near_constant_variance:g} normalised variance.",
        "",
        markdown_table("Pooled across all conditions", result.per_field_pooled),
    ]
    for topology in sorted(result.per_field_by_topology):
        sections.append(markdown_table(f"Topology: {topology}", result.per_field_by_topology[topology]))
    for count in sorted(result.per_field_by_av_count):
        sections.append(markdown_table(f"AV count: {count}", result.per_field_by_av_count[count]))
    sections.append("### Samples per condition\n")
    sections.append("| condition | samples |\n|---|---|")
    for condition in sorted(result.samples_by_condition):
        sections.append(f"| `{condition}` | {result.samples_by_condition[condition]} |")
    md_path = output_root / "observation_audit.md"
    md_path.write_text("\n".join(sections) + "\n")
    return {"json": json_path, "markdown": md_path}


def main() -> int:
    args = parse_args()
    specs = [
        SampleSpec(
            topology=topology,
            controller=controller,
            seed=seed,
            duration_steps=args.duration_steps,
            controlled_vehicles=av_count,
        )
        for topology in args.topologies
        for controller in args.controllers
        for av_count in args.av_counts
        for seed in args.seeds
    ]
    print(f"auditing {len(specs)} conditions...", flush=True)
    result = run_audit(specs, near_constant_variance=args.near_constant_variance)
    paths = write_report(result, Path(args.output_root), args)

    counts = {"informative": 0, "near-constant": 0, "constant": 0, "no-coverage": 0}
    for audit in result.per_field_pooled:
        counts[verdict(audit)] += 1
    print(f"pooled verdicts: {counts}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
