#!/usr/bin/env python3
"""The jam-avoidance figures and table, from the artefacts and nothing else.

    .venv/bin/python scripts/figures/jam_figures.py

Plan section 9. Reads `results/jam/camera_error_model.json`, `results/jam/leads.json`, and
the summaries `results/jam/{S,M,L}_summary.json`; writes PDFs and the table's rows to
`results/jam/figures/`. Each `fig_*`/`table_*` function takes the loaded JSON and returns
the series it plots, so a test can check them against the JSON; the `plot_*` functions
only draw those series. No number in this file is a result.

The paper shows three arms: no control, the network-wide feed lookahead at its selected
threshold, and the best model of `S_summary.json`.

* Fig. 1: (a) the camera error model, detection rate and distance noise against distance,
  640 px with the 1,280 px curve thin; (b) per scenario, `W_feed` and `W_cam` on the first
  segment to jam in block M's no-control episodes, with the band `D + [0, 60] s`.
* Fig. 2: per scenario, ordered by no control's median onset in block M: the share of seeds
  with no 120 s jam spell, and jammed segment-minutes with paired two-standard-error bars
  against the plan alone.
* Fig. 3: jammed segment-minutes against the lag multiplier 0, 1, 2, 4 (Amendment 1) on B9
  and I_exit for the three arms; `m = 1` is block M's, the rest block L's.
* Table 1: the three arms by scenario (median onset, prevented of n, jam time, fraction
  drained and median drain, served flow, held vehicle-minutes); BG: active share and served
  flow against the plan alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RESULTS = REPO_ROOT / "results" / "jam"
OUT = RESULTS / "figures"
#: Amendment 1: Fig. 3's lag axis.
LAG_AXIS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
LAG_SCENARIOS: tuple[str, ...] = ("B9", "I_exit")
PERTURBATIONS: tuple[str, ...] = ("B5", "B7.5", "B9", "I_exit", "I_mid")
DECISION_INTERVAL_S: float = 60.0


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run the step that writes it first")
    return json.loads(path.read_text())


def paper_arms(s_summary: dict) -> dict[str, str]:
    """`label -> "arm/variant"` for the three printed arms, read from the selection."""
    from run_jam_experiments import _variant

    best = s_summary.get("best")
    look = (s_summary.get("selected") or {}).get("lookahead")
    if not best or not look:
        raise ValueError("S_summary.json has no selection; the paper's arms are undefined")
    return {"no control": "no_control/-",
            "feed lookahead": f"lookahead/{_variant('lookahead', look)}",
            "best model": f"{best['arm']}/{_variant(best['arm'], best['params'])}"}


# --------------------------------------------------------------------- Fig. 1


def fig_warning(camera: dict, leads: dict, D: float) -> dict[str, Any]:
    bins = camera["tables"][camera["sumo_table"]]["bins_m"]
    centres = [(a + b) / 2.0 for a, b in bins]
    primary = camera["tables"][camera["sumo_table"]]
    sensitivity = ((camera.get("sensitivity") or {}).get("1280") or {}).get("tables", {}) \
        .get(camera["sumo_table"])
    panel_a = {"distance_m": centres, "p_640": primary["detection"]["p"],
               "sigma_640": primary["distance_noise"]["sigma_ema_m"],
               "p_1280": sensitivity["detection"]["p"] if sensitivity else None,
               "sigma_1280": sensitivity["distance_noise"]["sigma_ema_m"] if sensitivity else None,
               "R_max_m": primary["R_max_m"]}
    by_scenario: dict[str, dict[str, list]] = {}
    for e in leads["episodes"]:
        entry = by_scenario.setdefault(e["scenario"], {"seed": [], "W_feed": [], "W_cam": []})
        entry["seed"].append(e["seed"])
        entry["W_feed"].append(e["W_feed"])
        entry["W_cam"].append(e["W_cam"])
    return {"a": panel_a, "b": {"scenarios": by_scenario,
                                "band_s": [D, D + DECISION_INTERVAL_S]}}


def plot_warning(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a = series["a"]
    ax.plot(a["distance_m"], _nan(a["p_640"]), "o-", label="detection rate, 640 px")
    if a["p_1280"]:
        ax.plot(a["distance_m"], _nan(a["p_1280"]), "-", lw=0.7, label="1,280 px")
    ax2 = ax.twinx()
    ax2.plot(a["distance_m"], _nan(a["sigma_640"]), "s--", color="C3", label="distance noise")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("detection rate")
    ax2.set_ylabel("distance noise sigma (m)")
    ax.legend(fontsize=6, loc="lower left")
    b = series["b"]
    names = sorted(b["scenarios"])
    for i, name in enumerate(names):
        values = b["scenarios"][name]
        bx.scatter([i - 0.1] * len(values["W_feed"]), _nan(values["W_feed"]), s=8, c="C0",
                   label="W_feed" if i == 0 else None)
        bx.scatter([i + 0.1] * len(values["W_cam"]), _nan(values["W_cam"]), s=8, c="C1",
                   label="W_cam" if i == 0 else None)
    bx.axhspan(*b["band_s"], color="0.85", label="D + [0, 60] s")
    bx.set_xticks(range(len(names)), names, fontsize=7)
    bx.set_ylabel("warning before onset (s)")
    bx.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _nan(values):
    return [np.nan if v is None else v for v in values]


# --------------------------------------------------------------------- Fig. 2


def fig_onset_prevention(m_summary: dict, s_summary: dict) -> dict[str, Any]:
    arms = paper_arms(s_summary)
    rows = m_summary["rows"]
    scenarios = [s for s in PERTURBATIONS if f"no_control/-/{s}" in rows]

    def onset_key(s):
        median = rows[f"no_control/-/{s}"]["km_onset_median_s"]
        return (median is None, median if median is not None else 0.0)

    scenarios.sort(key=onset_key)
    out: dict[str, Any] = {"scenarios": scenarios, "arms": {}}
    for label, name in arms.items():
        entry = {"prevented_share": [], "jam_time_min": [], "two_se_vs_plan": []}
        for s in scenarios:
            row = rows.get(f"{name}/{s}")
            if row is None:
                for key in entry:
                    entry[key].append(None)
                continue
            entry["prevented_share"].append(row["prevented"] / row["n"])
            entry["jam_time_min"].append(row["jam_time_min_mean"])
            paired = row.get("paired_jam_time_vs_plan")
            entry["two_se_vs_plan"].append(paired["two_se"] if paired else None)
        out["arms"][label] = entry
    return out


def plot_onset_prevention(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(series["scenarios"]))
    for k, (label, entry) in enumerate(series["arms"].items()):
        offset = (k - 1) * 0.25
        ax.bar(x + offset, _nan(entry["prevented_share"]), width=0.25, label=label)
        bx.errorbar(x + offset, _nan(entry["jam_time_min"]), yerr=_nan(entry["two_se_vs_plan"]),
                    fmt="o", ms=3, capsize=2, label=label)
    for axis in (ax, bx):
        axis.set_xticks(x, series["scenarios"], fontsize=7)
    ax.set_ylabel("share of seeds with no 120 s spell")
    bx.set_ylabel("jammed segment-minutes")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- Fig. 3


def fig_lag(l_summary: dict, m_summary: dict, s_summary: dict) -> dict[str, Any]:
    arms = paper_arms(s_summary)
    out: dict[str, Any] = {"lag_axis": list(LAG_AXIS), "scenarios": {}}
    for scenario in LAG_SCENARIOS:
        per_arm = {}
        for label, name in arms.items():
            arm, variant = name.split("/", 1)
            values = []
            for m in LAG_AXIS:
                if arm == "no_control" or m == 1.0:
                    # No control has no feed; m = 1 is block M's own run.
                    row = m_summary["rows"].get(f"{name}/{scenario}")
                else:
                    row = l_summary["rows"].get(f"{arm}/{variant}_lag{m:g}/{scenario}")
                values.append(row["jam_time_min_mean"] if row else None)
            per_arm[label] = values
        out["scenarios"][scenario] = per_arm
    return out


def plot_lag(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(series["scenarios"]), figsize=(7.0, 2.4), sharey=True)
    for ax, (scenario, per_arm) in zip(np.atleast_1d(axes), series["scenarios"].items()):
        for label, values in per_arm.items():
            ax.plot(series["lag_axis"], _nan(values), "o-", label=label)
        ax.set_title(scenario, fontsize=8)
        ax.set_xticks(series["lag_axis"])
        ax.set_xlabel("lag multiplier m")
    np.atleast_1d(axes)[0].set_ylabel("jammed segment-minutes")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# -------------------------------------------------------------------- Table 1


def table_main(m_summary: dict, s_summary: dict) -> dict[str, Any]:
    arms = paper_arms(s_summary)
    rows = m_summary["rows"]
    out: dict[str, Any] = {"rows": [], "bg": []}
    for scenario in PERTURBATIONS + ("S45",):
        for label, name in arms.items():
            row = rows.get(f"{name}/{scenario}")
            if row is None:
                continue
            out["rows"].append({
                "scenario": scenario, "arm": label,
                "km_onset_median_s": row["km_onset_median_s"],
                "prevented": row["prevented"], "n": row["n"],
                "jam_time_min": row["jam_time_min_mean"],
                "fraction_drained": row["fraction_drained"],
                "km_drain_median_s": row["km_drain_median_s"],
                "flow": row["flow_mean"], "held_min": row["held_min_mean"]})
    plan = rows.get("plan/-/BG")
    for label, name in arms.items():
        row = rows.get(f"{name}/BG")
        if row is None:
            continue
        out["bg"].append({"arm": label, "active_share": row["active_share_mean"],
                          "active_changing_share": row.get("active_changing_share_mean"),
                          "flow": row["flow_mean"],
                          "flow_minus_plan": (row["flow_mean"] - plan["flow_mean"]
                                              if plan and row["flow_mean"] is not None else None)})
    return out


def render_all(out_dir: Path, *, camera: dict, leads: dict, D: float, s: dict, m: dict,
               l: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"fig1": plot_warning(fig_warning(camera, leads, D), out_dir / "fig1_warning.pdf"),
             "fig2": plot_onset_prevention(fig_onset_prevention(m, s),
                                           out_dir / "fig2_onset_prevention.pdf"),
             "fig3": plot_lag(fig_lag(l, m, s), out_dir / "fig3_lag.pdf")}
    table = out_dir / "table1_main.json"
    table.write_text(json.dumps(table_main(m, s), indent=1))
    paths["table1"] = table
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    results = Path(args.results_dir)
    m = load(results / "M_summary.json")
    D = (m.get("feed_artefact") or {}).get("D")
    if D is None:
        raise ValueError("M_summary.json records no feed delay D")
    paths = render_all(Path(args.out_dir) if args.out_dir else results / "figures",
                       camera=load(results / "camera_error_model.json"),
                       leads=load(results / "leads.json"), D=float(D),
                       s=load(results / "S_summary.json"), m=m,
                       l=load(results / "L_summary.json"))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
