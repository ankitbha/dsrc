#!/usr/bin/env python3
"""Pair every approach-boundary result with its gated-boundary result, seed by seed.

    .venv/bin/python scripts/compare_boundaries.py --gated results/jam \\
        --approach results/jam_approach

Approach plan 2026-10-17 section 7.1, step 4. For blocks M, L, P and K, per arm, variant
and scenario, `boundary_comparison.json` (written beside the approach results) holds:

* the gated boundary's jam time (its 12 super-segments) and the approach boundary's jam
  time over all 36 super-segments, the network's 12 and the approaches' 24;
* the episodes with no 120 s jam spell (prevented) on each;
* served flow at the network's exit on each;
* the storage upstream of the network: held plus pending vehicle-minutes on the gated
  boundary against approach excess plus pending vehicle-minutes on the approach boundary;
* system excess vehicle-minutes on both. The gated records do not carry it, so it is
  computed from their traces (`running + held + pending`, `N_ff = 0`) over the same window.

Each difference (approach minus gated) is paired by seed, and a seed present on one
boundary only is listed rather than dropped. The two boundaries run the same seeds but not
the same trajectories, so these differences describe the two boundaries; they are not a
controlled comparison of one mechanism. On the approach boundary alone, the paper's three
arms are contrasted over seeds 16-30: the two-layer controller (the selection's best
model) against no control and against the lookahead, for jam time over all 36 and over the
network's 12 super-segments, served flow and system excess vehicle-minutes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.analysis.jam_metrics import Paired, system_excess_minutes  # noqa: E402

BLOCKS = ("M", "L", "P", "K")
DEFAULT_GATED_TRACES = REPO_ROOT / "outputs" / "jam" / "run_c03cca5" / "traces"
NOTE = ("The two boundaries run the same seeds but not the same trajectories, so a "
        "difference paired by seed describes the two boundaries; it is not a controlled "
        "comparison of one mechanism. The thresholds and the best model were selected on "
        "the gated boundary and were not re-selected on the approach boundary.")


def pair_rows(gated: Mapping[int, float], approach: Mapping[int, float]) -> dict[str, Any]:
    """`approach - gated` paired by seed, with the seeds either side lacks listed."""
    paired = Paired.of(dict(approach), dict(gated))
    return {**paired.as_dict(),
            "seeds_only_gated": sorted(set(gated) - set(approach)),
            "seeds_only_approach": sorted(set(approach) - set(gated))}


def _read(results_dir: Path, block: str) -> tuple[list[dict], list[dict]]:
    import run_jam_experiments as rje

    return rje.read_block(results_dir, block)


def _cells(episodes: list[dict]) -> dict[str, dict[int, dict]]:
    cells: dict[str, dict[int, dict]] = {}
    for r in episodes:
        cells.setdefault(f"{r['arm']}/{r['variant']}/{r['scenario']}", {})[int(r["seed"])] = r
    return cells


def _values(records: Mapping[int, dict], value: Callable[[dict], float | None]) -> dict[int, float]:
    out = {}
    for seed, record in records.items():
        v = value(record)
        if v is not None:
            out[seed] = float(v)
    return out


def _mean(values: Mapping[int, float]) -> float | None:
    return statistics.fmean(values.values()) if values else None


def gated_system_excess(record: dict, block: str, traces_dir: Path) -> float | None:
    """The gated episode's system excess vehicle-minutes from its trace, or None when the
    trace cannot be found."""
    from src.sumo.jam_episode import SCENARIOS

    name = f"{record['arm']}_{record['variant']}_{record['scenario']}_{record['seed']}.json"
    candidates = [Path(traces_dir) / block / name, Path(record.get("trace_file") or "")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    trace = json.loads(path.read_text())["trace"]
    return system_excess_minutes(trace, start_s=SCENARIOS[record["scenario"]].start_s,
                                 cap=float(record["duration_s"]))


def compare_cell(block: str, gated: Mapping[int, dict], approach: Mapping[int, dict],
                 traces_dir: Path) -> dict[str, Any]:
    """One arm, variant and scenario on both boundaries."""
    g_jam = _values(gated, lambda r: r["jam"]["jam_time_min"])
    g_flow = _values(gated, lambda r: r["flow"])
    g_store = _values(gated, lambda r: r["jam"]["held_min"] + r["jam"]["pending_min"])
    g_system = _values(gated, lambda r: gated_system_excess(r, block, traces_dir))
    a_all = _values(approach, lambda r: r["jam"]["jam_time_min"])
    a_net = _values(approach, lambda r: r["jam_network"]["jam_time_min"])
    a_app = _values(approach, lambda r: r["jam_approach"]["jam_time_min"])
    a_flow = _values(approach, lambda r: r["flow"])
    a_store = _values(approach, lambda r: r["storage"]["approach_excess_min"]
                      + r["jam"]["pending_min"])
    a_system = _values(approach, lambda r: r["storage"]["system_excess_min"])
    return {
        "gated": {"seeds": sorted(gated), "jam_time_min_mean": _mean(g_jam),
                  "prevented": sum(1 for r in gated.values() if r["jam"]["prevented"]),
                  "flow_mean": _mean(g_flow), "held_plus_pending_min_mean": _mean(g_store),
                  "system_excess_min_mean": _mean(g_system),
                  "system_excess_seeds_without_trace": sorted(set(gated) - set(g_system))},
        "approach": {"seeds": sorted(approach), "jam_all_time_min_mean": _mean(a_all),
                     "jam_network_time_min_mean": _mean(a_net),
                     "jam_approach_time_min_mean": _mean(a_app),
                     "prevented_all": sum(1 for r in approach.values() if r["jam"]["prevented"]),
                     "prevented_network": sum(1 for r in approach.values()
                                              if r["jam_network"]["prevented"]),
                     "prevented_approach": sum(1 for r in approach.values()
                                               if r["jam_approach"]["prevented"]),
                     "flow_mean": _mean(a_flow),
                     "approach_excess_plus_pending_min_mean": _mean(a_store),
                     "system_excess_min_mean": _mean(a_system),
                     "approach_short_episodes": sum(1 for r in approach.values()
                                                    if r.get("approach_short"))},
        "approach_minus_gated": {
            "network_jam_time_min": pair_rows(g_jam, a_net),
            "all_jam_time_min_against_gated": pair_rows(g_jam, a_all),
            "flow": pair_rows(g_flow, a_flow),
            "storage_min": pair_rows(g_store, a_store),
            "system_excess_min": pair_rows(g_system, a_system),
        },
    }


def contrasts(episodes: list[dict], best: str, lookahead: str) -> dict[str, Any]:
    """The paper's three arms on the approach boundary, per scenario: the two-layer
    controller (`best`, `arm/variant`) minus no control and minus the lookahead, paired
    over seeds."""
    cells = _cells(episodes)
    metrics = {"jam_all_time_min": lambda r: r["jam"]["jam_time_min"],
               "jam_network_time_min": lambda r: r["jam_network"]["jam_time_min"],
               "flow": lambda r: r["flow"],
               "system_excess_min": lambda r: r["storage"]["system_excess_min"]}
    out: dict[str, Any] = {}
    for scenario in sorted({r["scenario"] for r in episodes}):
        controller = cells.get(f"{best}/{scenario}")
        if controller is None:
            continue
        per: dict[str, Any] = {}
        for other in ("no_control/-", lookahead):
            against = cells.get(f"{other}/{scenario}")
            if against is None:
                continue
            per[f"{best} - {other}"] = {
                name: {**Paired.of(_values(controller, f), _values(against, f)).as_dict(),
                       "seeds_only_controller": sorted(set(controller) - set(against)),
                       "seeds_only_other": sorted(set(against) - set(controller))}
                for name, f in metrics.items()}
        out[scenario] = per
    return out


def compare(gated_dir: Path, approach_dir: Path, traces_dir: Path) -> dict[str, Any]:
    import run_jam_experiments as rje

    out: dict[str, Any] = {"note": NOTE, "gated": str(gated_dir), "approach": str(approach_dir),
                           "gated_traces": str(traces_dir), "blocks": {}}
    for block in BLOCKS:
        g_headers, g_episodes = _read(gated_dir, block)
        a_headers, a_episodes = _read(approach_dir, block)
        if not a_episodes:
            continue
        if any(h.get("boundary") != "approach" for h in a_headers):
            raise SystemExit(f"{approach_dir / block}.jsonl has a header that did not run the "
                             "approach boundary")
        if any(h.get("boundary") == "approach" for h in g_headers):
            raise SystemExit(f"{gated_dir / block}.jsonl has an approach-boundary header")
        g_cells, a_cells = _cells(g_episodes), _cells(a_episodes)
        rows = {name: compare_cell(block, g_cells.get(name, {}), cells, traces_dir)
                for name, cells in sorted(a_cells.items())}
        out["blocks"][block] = {
            "W": a_headers[-1].get("W"), "T_A": (a_headers[-1].get("approach") or {}).get("T_A"),
            "rows": rows,
            "cells_only_gated": sorted(set(g_cells) - set(a_cells)),
        }
        if block == "M":
            selection = (a_headers[-1].get("config") or {}).get("selection") or {}
            best = f"{selection['best_arm']}/{rje._variant(selection['best_arm'], selection['best_params'])}"
            lookahead = f"lookahead/{rje._variant('lookahead', {'J_th': selection['J_th']})}"
            out["contrasts"] = {"two_layer_controller": best, "lookahead": lookahead,
                                "scenarios": contrasts(a_episodes, best, lookahead)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gated", default="results/jam")
    parser.add_argument("--approach", default="results/jam_approach")
    parser.add_argument("--gated-traces", default=str(DEFAULT_GATED_TRACES),
                        help="where the gated run's traces are, as <dir>/<block>/<name>.json")
    parser.add_argument("--out", default=None,
                        help="default <approach>/boundary_comparison.json")
    args = parser.parse_args(argv)
    resolve = lambda p: Path(p) if Path(p).is_absolute() else REPO_ROOT / p
    approach = resolve(args.approach)
    result = compare(resolve(args.gated), approach, resolve(args.gated_traces))
    out = resolve(args.out) if args.out else approach / "boundary_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    blocks = ", ".join(f"{b} ({len(v['rows'])} rows)" for b, v in result["blocks"].items())
    print(f"wrote {out}: {blocks or 'no approach block'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
