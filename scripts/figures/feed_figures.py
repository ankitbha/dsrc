#!/usr/bin/env python3
"""The feed-agreement figures and table, from the artefacts and nothing else.

    .venv/bin/python scripts/figures/feed_figures.py

Plan section 6. Reads `results/feed/feed_process.json` and the block summaries
`results/feed/{A,B,C,D,E}_summary.json`, writes PDFs and the table's rows to
`results/feed/figures/`. Each `fig_*`/`table_*` function takes the loaded JSON and
returns the series it plots, so a test can check them against the JSON; the `plot_*`
functions only draw those series. No number in this file is a result.

* Fig. 1: (a) each version's second of the minute against time, with its availability-lag
  bracket; (b) disagreement against `D - L_bar` for the version-keyed rule: Lemma 1 (b)'s
  cycle average as a line, block A's measured points, and the other rules as horizontal
  marks.
* Fig. 2: (a) paired throughput loss against the measured fraction of vehicles on the
  previous snapshot, the penetration loss at `1 - p` on the same axis, and the line
  through the origin; (b) loss against staleness with its line.
* Fig. 3: Kaplan-Meier drain curves per arm and scenario, censoring marks at the cap,
  fraction drained and median, under the primary threshold.
* Table 1: block A's rules, and the penetration levels with block C's flow and block E's
  certificate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results" / "feed"
OUT = RESULTS / "figures"


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run the block that writes it first")
    return json.loads(path.read_text())


# --------------------------------------------------------------------- Fig. 1


def fig_feed_and_rule(feed_process: dict, a_summary: dict,
                      sessions: list[str] | None = None) -> dict[str, Any]:
    from src.sumo.feed import lemma_cycle_average

    names = sessions or feed_process["primary"]["sessions"]
    panel_a = []
    for name in names:
        versions = feed_process["sessions"][name]["per_version"]
        origin = versions[0]["T"] if versions else 0.0
        panel_a.append({"session": name,
                        "t_min": [(v["T"] - origin) / 60.0 for v in versions],
                        "phase": [v["T"] % 60 for v in versions],
                        "L_lo": [v["L_lo"] for v in versions],
                        "L_up": [v["L_up"] for v in versions]})
    lemma = a_summary.get("lemma", [])
    feed = a_summary.get("feed", {})
    bound_points = [(r["D_minus_L_bar"], r["bound_cycle_average_g_min"]) for r in lemma]
    measured = [(r["D_minus_L_bar"], r["measured_delta_mean"]) for r in lemma
                if r["measured_delta_mean"] is not None]
    line_x, line_y = [], []
    if lemma and feed:
        offsets = [r["D_minus_L_bar"] for r in lemma]
        line_x = list(np.linspace(min(offsets), max(offsets), 201))
        line_y = [lemma_cycle_average(feed["L_bar"] + x, feed["L_bar"], feed["P_q"],
                                      feed["g_min"]) for x in line_x]
    rows = a_summary.get("rows", {})

    def reference(prefix: str):
        match = [row["delta_mean"] for key, row in rows.items() if key.startswith(prefix)]
        return match[0] if match else None

    return {"panel_a": panel_a,
            "panel_b": {"bound_points": bound_points, "measured": measured,
                        "line_x": line_x, "line_y": line_y,
                        "references": {"no_rule_60": reference("no_rule/P60"),
                                       "no_rule_10": reference("no_rule/P10"),
                                       "fixed_phase": reference("fixed_phase/")},
                        "placeholder": a_summary.get("placeholder")}}


def plot_feed_and_rule(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for line in series["panel_a"]:
        ax_a.plot(line["t_min"], line["phase"], "o", ms=3, label=line["session"])
        lows = [lo if lo is not None else up for lo, up in zip(line["L_lo"], line["L_up"])]
        ax_a.vlines(line["t_min"], [(p + lo) % 60 for p, lo in zip(line["phase"], lows)],
                    [(p + up) % 60 for p, up in zip(line["phase"], line["L_up"])],
                    alpha=0.3)
    ax_a.set_xlabel("time since first version (min)")
    ax_a.set_ylabel("second of the minute")
    ax_a.legend(fontsize=5)
    b = series["panel_b"]
    ax_b.plot(b["line_x"], b["line_y"], "-", label="Lemma 1 (b) bound")
    if b["measured"]:
        ax_b.plot(*zip(*b["measured"]), "o", label="block A")
    for name, value in b["references"].items():
        if value is not None:
            ax_b.axhline(value, ls=":", lw=0.8)
            ax_b.annotate(name, (ax_b.get_xlim()[0], value), fontsize=5)
    ax_b.set_xlabel("D - L_bar (s)")
    ax_b.set_ylabel("time-averaged disagreement")
    ax_b.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- Fig. 2


def fig_flow_consequence(b_summary: dict, c_summary: dict) -> dict[str, Any]:
    def loss(row):
        paired = row.get("paired_vs_published") or {}
        if "gain" not in paired:
            return None, None
        return -paired["gain"], paired["two_se"]

    delta, stale, penetration = [], [], []
    for row in b_summary.get("rows", {}).values():
        value, bar = loss(row)
        if value is None:
            continue
        if row["arm"] == "inject_delta" and row.get("previous_fraction_mean") is not None:
            # Amendment 1, A6: placed at the measured fraction on the previous snapshot.
            delta.append((row["previous_fraction_mean"], value, bar))
        elif row["arm"] == "inject_stale":
            stale.append((float(row["variant"][1:]), value, bar))
    for row in c_summary.get("rows", {}).values():
        value, bar = loss(row)
        if value is not None:
            penetration.append((1.0 - row["penetration"], value, bar))
    fit_delta = b_summary.get("fit_loss_vs_delta") or {}
    fit_stale = b_summary.get("fit_loss_vs_staleness") or {}
    return {"panel_a": {"delta": sorted(delta), "penetration": sorted(penetration),
                        "delta_label": "fraction of vehicles on the previous snapshot",
                        "slope": fit_delta.get("slope"),
                        "quadratic_resolved": fit_delta.get("quadratic_resolved")},
            "panel_b": {"staleness": sorted(stale), "slope": fit_stale.get("slope"),
                        "quadratic_resolved": fit_stale.get("quadratic_resolved")}}


def plot_flow_consequence(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a = series["panel_a"]
    for points, label in ((a["delta"], a["delta_label"]),
                          (a["penetration"], "1 - penetration")):
        if points:
            x, y, bar = zip(*points)
            ax_a.errorbar(x, y, yerr=bar, fmt="o", ms=3, capsize=2, label=label)
    if a["slope"] is not None:
        ax_a.plot([0.0, 1.0], [0.0, a["slope"]], "-", lw=0.8, label="line through origin")
    ax_a.set_xlabel("fraction")
    ax_a.set_ylabel("paired flow loss (veh/h)")
    ax_a.legend(fontsize=6)
    b = series["panel_b"]
    if b["staleness"]:
        x, y, bar = zip(*b["staleness"])
        ax_b.errorbar(x, y, yerr=bar, fmt="o", ms=3, capsize=2)
        if b["slope"] is not None:
            ax_b.plot([0.0, max(x)], [0.0, b["slope"] * max(x)], "-", lw=0.8)
    ax_b.set_xlabel("snapshot staleness (s)")
    ax_b.set_ylabel("paired flow loss (veh/h)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- Fig. 3


def fig_drain(d_summary: dict, rule: str | None = None) -> dict[str, Any]:
    out = {}
    for scenario, entry in d_summary.get("scenarios", {}).items():
        if "drain" not in entry:
            continue
        # Without a network threshold (refused, or not yet fitted) the fixed 30 km/h rule
        # is drawn, and the panel says so.
        key = rule or entry.get("primary") or "fixed_30/H120"
        result = entry["drain"][key]
        panel: dict[str, Any] = {"rule": key, "cap_minus_t0": entry["cap_s"] - entry["t_0"]}
        for arm, data in result["arms"].items():
            panel[arm] = {"times": data["km"]["times"], "survival": data["km"]["survival"],
                          "median": data["km"]["median"],
                          "fraction_drained": data["fraction_drained"],
                          "censored": [row["time_s"] for row in data["per_seed"]
                                       if not row["drained"]]}
        panel["sensitivities"] = {
            name: {arm: data["fraction_drained"] for arm, data in r["arms"].items()}
            for name, r in entry["drain"].items()}
        out[scenario] = panel
    return out


def plot_drain(series: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, max(len(series), 1), figsize=(7.0, 2.6), squeeze=False)
    for ax, (scenario, panel) in zip(axes[0], series.items()):
        for arm in ("no_control", "here"):
            if arm not in panel:
                continue
            data = panel[arm]
            times = [0.0] + list(data["times"])
            survival = [1.0] + list(data["survival"])
            median = "> cap" if data["median"] is None else f"{data['median'] / 60:.0f} min"
            ax.step([t / 60 for t in times], survival, where="post",
                    label=f"{arm}: {data['fraction_drained']:.2f} drained, median {median}")
            if data["censored"]:
                ax.plot([t / 60 for t in data["censored"]],
                        [survival[-1]] * len(data["censored"]), "|", ms=8)
        ax.set_title(f"{scenario} ({panel['rule']})", fontsize=7)
        ax.set_xlabel("minutes after demand stops")
        ax.set_ylabel("share not yet drained")
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# -------------------------------------------------------------------- Table 1


def table_certificate_and_rules(a_summary: dict, c_summary: dict,
                                e_summary: dict) -> dict[str, Any]:
    rules = []
    for key, row in a_summary.get("rows", {}).items():
        paired = row.get("paired_vs_published") or {}
        rules.append({"rule": key, "delta": row.get("delta_mean"),
                      "command_delta": row.get("command_delta_mean"),
                      "mean_age_s": row.get("age_mean_s"), "flow_gain": paired.get("gain"),
                      "flow_gain_two_se": paired.get("two_se")})
    p_min = (e_summary.get("p_min") or {}).get("p_min")
    flows = {row["penetration"]: row for row in c_summary.get("rows", {}).values()}
    levels = []
    for level, entry in e_summary.get("levels", {}).items():
        p = float(level)
        c_row = flows.get(p)
        paired = (c_row or {}).get("paired_vs_published") or {}
        levels.append({"penetration": p,
                       "flow_gain": 0.0 if p == 1.0 else paired.get("gain"),
                       "rho_hat": entry.get("rho_hat"), "rho_UCB": entry.get("rho_UCB"),
                       "certificate": entry.get("certificate"),
                       "p_min": p_min is not None and p == p_min})
    return {"rules": rules, "penetration": sorted(levels, key=lambda r: -r["penetration"]),
            "placeholder": a_summary.get("placeholder")}


# ----------------------------------------------------------------------- main


def render_all(out_dir: Path, *, feed_process: dict, a: dict, b: dict, c: dict, d: dict,
               e: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        plot_feed_and_rule(fig_feed_and_rule(feed_process, a), out_dir / "fig1_feed_and_rule.pdf"),
        plot_flow_consequence(fig_flow_consequence(b, c), out_dir / "fig2_flow_consequence.pdf"),
        plot_drain(fig_drain(d), out_dir / "fig3_drain.pdf"),
    ]
    table = out_dir / "table1.json"
    table.write_text(json.dumps(table_certificate_and_rules(a, c, e), indent=1))
    return written + [table]


def main() -> int:
    written = render_all(
        OUT, feed_process=load(RESULTS / "feed_process.json"),
        a=load(RESULTS / "A_summary.json"), b=load(RESULTS / "B_summary.json"),
        c=load(RESULTS / "C_summary.json"), d=load(RESULTS / "D_summary.json"),
        e=load(RESULTS / "E_summary.json"))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
