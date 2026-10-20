#!/usr/bin/env python3
"""Re-score committed approach-boundary records on the network alone (2026-10-20).

    .venv/bin/python scripts/rescore_network.py --source results/jam_approach \
        --traces outputs/jam_approach/run_5cea9d3/traces --out outputs/jam_network/run

The approach roads are a boundary that paces the demand into the network, so a result
measures the network's 12 super-segments only. This rebuilds each episode record of blocks
M, L, P and K from its own trace with the runner's `approach_metrics`, which now makes the
network's jam the record's `jam` and adds `storage.network_min`. Nothing is simulated
again: an episode is a deterministic function of its spec, and the trace is its output.

Every number the old record held must come back under its new name, or the record is
refused: the old `jam` (all 36) is the new `jam_all`, the old `jam_network` is the new
`jam`, the approach part and the two storage measures are unchanged. Headers and footers are
copied with a `rescored` note. The output JSONL is what `run_jam_experiments.py --resume`
then adds the gated rule's episodes to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_jam_experiments as rje  # noqa: E402

BLOCKS = ("M", "L", "P", "K")
#: The metric fields the runner writes on the approach boundary, old and new.
OLD_FIELDS = ("jam", "jam_network", "jam_approach", "jam_30kmh", "jam_30kmh_network",
              "jam_30kmh_approach", "storage", "flags", "approach_short")
#: Old field -> the new field that must equal it.
RENAMED = {"jam": "jam_all", "jam_network": "jam", "jam_approach": "jam_approach",
           "jam_30kmh": "jam_30kmh_all", "jam_30kmh_network": "jam_30kmh",
           "jam_30kmh_approach": "jam_30kmh_approach", "flags": "flags",
           "approach_short": "approach_short"}


def spec_of(record: dict) -> rje.Spec:
    return rje.Spec(record["block"], record["arm"], record["variant"], record["scenario"],
                    int(record["seed"]), float(record["duration_s"]), m=float(record["m"]),
                    penetration=float(record["penetration"]), bias=float(record["bias"]),
                    params=tuple(sorted((record.get("params") or {}).items())),
                    pair_sigma_scale=float(record.get("pair_sigma_scale", 1.0)))


def rescore(record: dict, trace_path: Path, v_jam: float, boundary) -> dict[str, Any]:
    """The record with its metric fields recomputed on the network; refused when any old
    number does not come back under its new name."""
    from src.sumo.jam_episode import SCENARIOS

    spec = spec_of(record)
    if spec.key != record["key"]:
        raise SystemExit(f"{record['key']}: its fields rebuild the key {spec.key}")
    stored = json.loads(trace_path.read_text())
    if stored.get("key") != record["key"]:
        raise SystemExit(f"{trace_path} holds {stored.get('key')}, not {record['key']}")
    new = rje.approach_metrics(stored["trace"], SCENARIOS[spec.scenario], spec, v_jam,
                               boundary)
    for old, name in RENAMED.items():
        if new[name] != record[old]:
            raise SystemExit(f"{record['key']}: the new {name} is not the old {old}")
    for name in ("approach_excess_min", "system_excess_min"):
        if new["storage"][name] != record["storage"][name]:
            raise SystemExit(f"{record['key']}: storage {name} changed on re-scoring")
    out = {k: v for k, v in record.items() if k not in OLD_FIELDS}
    out.update(new)
    out["trace_file"] = str(trace_path)
    out["rescored"] = {"from_trace_file": record.get("trace_file")}
    return out


def rescore_block(block: str, source: Path, traces: Path, out_dir: Path, note: dict) -> int:
    from src.analysis.jam_metrics import load_v_jam

    v_jam, _ = load_v_jam()
    boundary = None
    lines_out, count = [], 0
    for line in (source / f"{block}.jsonl").read_text().splitlines():
        record = json.loads(line)
        kind = record.get("type")
        if kind == "header":
            if record.get("boundary") != "approach":
                raise SystemExit(f"{source / block}.jsonl has a header that did not run the "
                                 "approach boundary")
            # The run worktrees the headers name are gone; the repository's approach
            # metadata is used, and must be the file the run read.
            if rje._sha256(rje.Context().approach_file) != record["approach"]["approach_sha256"]:
                raise SystemExit(f"{block}: the header ran approach metadata "
                                 f"{record['approach']['approach_sha256']}, the repository "
                                 f"holds {rje._sha256(rje.Context().approach_file)}")
            if boundary is None:
                boundary = rje.Context(boundary="approach").boundary_object()
            lines_out.append({**record, "rescored": note})
        elif kind == "episode":
            name = rje.Spec(block, record["arm"], record["variant"], record["scenario"],
                            record["seed"], record["duration_s"]).trace_name()
            lines_out.append(rescore(record, traces / block / name, v_jam, boundary))
            count += 1
        else:
            lines_out.append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{block}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists; move it away first")
    target.write_text("".join(json.dumps(r, default=float) + "\n" for r in lines_out))
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="results/jam_approach")
    parser.add_argument("--traces", default="outputs/jam_approach/run_5cea9d3/traces")
    parser.add_argument("--out", default="outputs/jam_network/run")
    parser.add_argument("--block", nargs="+", choices=BLOCKS, default=list(BLOCKS))
    args = parser.parse_args(argv)
    source, traces, out_dir = (rje._resolve(p) for p in (args.source, args.traces, args.out))
    provenance = rje._provenance()
    note = {"by": "scripts/rescore_network.py", "source": str(source), "traces": str(traces),
            "commit": provenance.get("commit"), "dirty": provenance.get("dirty")}
    for block in args.block:
        print(f"block {block}: re-scored {rescore_block(block, source, traces, out_dir, note)} "
              f"episodes into {out_dir / f'{block}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
