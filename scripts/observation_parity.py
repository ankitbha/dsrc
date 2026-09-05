#!/usr/bin/env python3
"""Task 47: build the live-vs-simulator observation parity ledger and write it.

"The observation vector produced live matches the simulator's sensing model
field for field" does not hold -- at least ten of the 39 encoded slots
differ by construction, six of them because the vehicle has no rear sensor.
This produces the slot-by-slot ledger that states which, and why, instead
of a check written to pass on the 29 that agree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.observation_parity import build_ledger, check_run_against_ledger  # noqa: E402


def json_safe(value: object) -> object:
    """Replace non-finite floats with null, the same convention
    scripts/audit_observation_fields.py uses: `json.dumps` writes a bare
    `Infinity` token for one, which strict parsers reject, and several
    slots here are genuinely `inf` by design (a leader gap with no leader)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def markdown_report(ledger: dict) -> str:
    counts = {"identical": 0, "approximated": 0, "substituted": 0, "structurally_absent": 0}
    for row in ledger["slots"]:
        counts[row["class"]] += 1
    lines = [
        "# Observation parity ledger (task 47)",
        "",
        f"Scenes: {', '.join(ledger['scenes'])}. Tolerance: atol={ledger['atol']:g}.",
        "",
        f"identical {counts['identical']} | approximated {counts['approximated']} | "
        f"substituted {counts['substituted']} | structurally_absent {counts['structurally_absent']} "
        f"| total {len(ledger['slots'])}",
        "",
        "| slot | class | always equal | class holds | constant holds | provenance ok | mechanism |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ledger["slots"]:
        const = "n/a" if row["constant_holds_across_scenes"] is None else str(row["constant_holds_across_scenes"])
        lines.append(
            f"| `{row['slot']}` | {row['class']} | {row['always_equal_across_scenes']} | "
            f"{row['matches_claimed_class']} | {const} | {row['provenance_in_substituted_partition']} | "
            f"{row['mechanism']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/validation/observation_parity")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="also check a real drive's own metadata.jsonl against the ledger (47.8/47.9) "
             "and write observation_run_check.json into it -- B6, validation round 1: "
             "47.8/47.9 had no committed command and left no artifact",
    )
    args = parser.parse_args()

    ledger = build_ledger()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "observation_parity.json"
    json_path.write_text(json.dumps(json_safe(ledger), indent=2, sort_keys=True, allow_nan=False))

    md_path = output_root / "observation_parity.md"
    md_path.write_text(markdown_report(ledger))

    mismatches = [r["slot"] for r in ledger["slots"] if not r["matches_claimed_class"]]
    prov_mismatches = [r["slot"] for r in ledger["slots"] if not r["provenance_in_substituted_partition"]]
    print(f"{len(ledger['slots'])} slots classified across {len(ledger['scenes'])} scenes")
    print(f"class mismatches: {mismatches or 'none'}")
    print(f"provenance mismatches: {prov_mismatches or 'none'}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    ledger_ok = not mismatches and not prov_mismatches

    run_check_ok = True
    if args.run_dir is not None:
        run_check = check_run_against_ledger(args.run_dir, ledger)
        run_check_path = args.run_dir / "observation_run_check.json"
        run_check_path.write_text(
            json.dumps(json_safe(run_check), indent=2, sort_keys=True, allow_nan=False)
        )
        print(
            f"run check: {run_check['ticks_checked']} ticks, "
            f"reencode mismatches {len(run_check['reencode_mismatches'])}, "
            f"constant mismatches {list(run_check['constant_mismatches'])}"
        )
        print(f"run check json: {run_check_path}")
        run_check_ok = run_check["ok"]

    return 0 if ledger_ok and run_check_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
