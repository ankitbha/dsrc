"""Figures for the jam-avoidance letter, from the committed results only.

    .venv/bin/python scripts/figures/paper_jam_figs.py --results-dir results/jam_approach \
        --out paper/figs

Reads `{M.jsonl,M_summary.json,leads.json}` and, when present, `L.jsonl` from the results
directory, and the selection (`S_summary.json`) from `--selection`, which defaults to
`results/jam/S_summary.json`: the thresholds were selected once, on the gated boundary, and
the approach results carry no selection of their own. It writes three PDFs, each with a
300 dpi PNG beside it for checking:

* `fig_warning.pdf`: per trigger, the feed's speed warning and the camera's density warning
  before onset on the first segment to jam (block M's no-control runs), median and
  interquartile range, with the plan's switch delay D drawn as a line. Half a column wide.
* `fig_jams.pdf`: per trigger, ordered by no control's median onset, the jammed
  segment-minutes, mean and two standard errors over the 15 evaluation seeds, for no control,
  the feed lookahead and the two-layer controller. The seeds with no jam of 120 s or more are
  in the paper's table, so the figure does not repeat them; they stay in the series JSON.
  Half a column wide, beside `fig_warning.pdf`.
* `fig_lag.pdf` (only with block L): jammed segment-minutes against the feed delay D
  on the largest burst and the exit block, for the lookahead and the two-layer controller
  (no control reads no feed, so D does not apply to it).
* `fig_penetration.pdf` (only with block P): jammed segment-minutes against the share of
  vehicles that follow, on the largest burst and the exit block, for the plan alone, the
  two-layer controller and, while the controller runs no gate, the rule with Section VI-F's
  gate; with `--mechanism-selection` the third line is the rule without its gate instead
  (the system before block N2). No control is a line. Block P gives 0.5 and 0.75, the sweep in `sweep/p*/P.jsonl`
  (2026-10-21) the shares between, and block M 1.0.

The three paper arms are no control, the lookahead at its selected threshold and the best
model, both read from `S_summary.json`; with `--gate-selection` (block N's `N_summary.json`,
2026-10-20) the best model is the rule with its selected feed gate, and with
`--mechanism-selection` (block N2's `N2_summary.json`, 2026-10-25) the rule with its selected
queue release and gates. Every jam number is the
record's `jam`, which on the network-measured results is the network's 12 super-segments. Colours follow the entity: the camera and the
two-layer controller blue, the feed and the lookahead orange, no control neutral grey.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "jam"
SELECTION = RESULTS / "S_summary.json"

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
#: The penetration figure's two other series; with blue they pass the CVD check for all pairs.
GREEN, PINK = "#008300", "#e87ba4"
#: The feed gate Section VI-F reports: block N's candidate at 0.5, which the selection did not
#: choose and which ran on the evaluation seeds on Ankit's decision (2026-10-20).
GATE_SHOWN = 0.5
INK, MUTED = "#2b2b29", "#6b6b66"
COLUMN_IN = 3.45
#: The warning and jam figures sit side by side in one column, so they are drawn at half its width.
HALF_IN = 1.65

LABELS = {"B5": "Burst\n5,000", "B7.5": "Burst\n7,500", "B9": "Burst\n9,000",
          "I_exit": "Exit\nblock", "I_mid": "Mid\nblock"}
TRIGGERS = ["B5", "B7.5", "B9", "I_exit", "I_mid"]


def _style() -> None:
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _boxed(legend):
    """A thin rectangular frame, so the legend reads apart from the data."""
    frame = legend.get_frame()
    frame.set_edgecolor(MUTED)
    frame.set_linewidth(0.6)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    return legend


def _records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("type") not in ("header", "footer")]


def paper_arms(s_summary: dict, n_summary: dict | None = None
               ) -> list[tuple[str, str | None, str, str, str]]:
    """(label, arm, variant, colour, marker) for the three arms the paper shows."""
    best = s_summary["best"]
    look = s_summary["selected"]["lookahead"]
    look_variant = f"J{look['J_th']:g}"
    best_variant = best["name"].split("/", 1)[1] if "/" in best["name"] else "-"
    if n_summary is not None:
        chosen = n_summary["selected"]
        if best["arm"] != "rule" or float(chosen["m_on"]) != float(best["params"]["m_on"]):
            raise SystemExit("the gate selection is for another rule than the best model")
        best_variant = chosen["variant"]
    return [("No control", "no_control", "-", GREY, "^"),
            ("Feed lookahead", "lookahead", look_variant, ORANGE, "s"),
            ("Global-local", best["arm"], best_variant, BLUE, "o")]


def _jam_by_seed(records: list[dict], arm: str, variant: str, scenario: str,
                 m: float | None = None) -> dict[int, float]:
    """Jam time by seed for one arm and variant; with `m` (block L), the variant run at that
    lag multiplier, `<variant>_lag<m>`. Variants match exactly: the rule with and without
    the feed gate (`m0_g...` and `m0`) are different arms."""
    want = variant if m is None else f"{variant}_lag{m:g}"
    out = {}
    for r in records:
        if r["arm"] != arm or r["scenario"] != scenario or str(r["variant"]) != want:
            continue
        if m is not None and abs(float(r.get("m", 1.0)) - m) > 1e-9:
            continue
        if int(r["seed"]) in out:
            raise SystemExit(f"two records of {arm}/{want}/{scenario}/{r['seed']}")
        out[int(r["seed"])] = float(r["jam"]["jam_time_min"])
    return out


def _mean_2se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    return mean, (2.0 * statistics.stdev(values) / math.sqrt(len(values))
                  if len(values) > 1 else 0.0)


def fig_warning(leads: dict, D: float, out: Path) -> dict:
    series = {}
    for s in TRIGGERS:
        eps = [e for e in leads["episodes"] if e["scenario"] == s and e.get("onset") is not None]
        feed = sorted(e["W_feed"] for e in eps if e.get("W_feed") is not None)
        cam = sorted(e["W_cam"] for e in eps if e.get("W_cam") is not None)
        series[s] = {"feed": feed, "camera": cam}
    fig, ax = plt.subplots(figsize=(HALF_IN, 0.78))
    order = sorted(TRIGGERS, key=lambda s: statistics.median(series[s]["feed"] or [0]))
    for i, s in enumerate(order):
        for dx, key, colour, marker, label in ((-0.15, "feed", ORANGE, "s", "Feed speed"),
                                               (0.15, "camera", BLUE, "o", "Camera density")):
            v = series[s][key]
            if not v:
                continue
            q1, med, q3 = statistics.quantiles(v, n=4) if len(v) > 1 else (v[0], v[0], v[0])
            ax.vlines(i + dx, max(q1, 5), q3, color=colour, linewidth=1.2)
            ax.plot(i + dx, max(med, 5), marker=marker, color=colour, markersize=3.8,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=label if i == 0 else None, linestyle="none")
    ax.axhline(D, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(-0.3, D * 1.15, f"D = {D:.1f} s", color=MUTED, fontsize=6, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(5, 4000)
    ax.set_xlim(-0.45, len(order) - 0.55)
    ax.set_xticks(range(len(order)), [LABELS[s] for s in order])
    ax.tick_params(labelsize=6)
    # Above the plot rather than along it: rotated, the label would be taller than the axes.
    ax.set_title("warning before onset (s)", loc="left", fontsize=6.5, color=INK, pad=3)
    # Lower right, below D, where no trigger's warning falls.
    _boxed(ax.legend(loc="lower right", frameon=True, fancybox=False, fontsize=6,
                     handletextpad=0.2, handlelength=1.0, borderaxespad=0.15, borderpad=0.3,
                     labelspacing=0.25))
    fig.savefig(out / "fig_warning.pdf")
    fig.savefig(out / "fig_warning.png", dpi=300)
    plt.close(fig)
    return series


def fig_jams(records: list[dict], m_summary: dict, arms, out: Path) -> dict:
    rows = m_summary["rows"]

    def row(arm, variant, s):
        for r in rows.values():
            if r["arm"] == arm and r["scenario"] == s and (variant == "-" or str(r["variant"]) == variant):
                return r
        raise KeyError((arm, variant, s))

    onset = {s: row("no_control", "-", s)["km_onset_median_s"] or 1e9 for s in TRIGGERS}
    order = sorted(TRIGGERS, key=lambda s: onset[s])
    series = {}
    fig, bottom = plt.subplots(figsize=(HALF_IN, 0.95))
    width = 0.27
    for j, (label, arm, variant, colour, marker) in enumerate(arms):
        dx = (j - 1) * width
        free, means, ses = [], [], []
        for s in order:
            r = row(arm, variant, s)
            free.append(r["prevented"])
            mean, se = _mean_2se(list(_jam_by_seed(records, arm, variant, s).values()))
            means.append(mean)
            ses.append(se)
        series[label] = {"jam_free": dict(zip(order, free)), "jam_mean": dict(zip(order, means)),
                         "jam_2se": dict(zip(order, ses))}
        xs = [i + dx for i in range(len(order))]
        bottom.errorbar(xs, means, yerr=ses, color=colour, marker=marker, markersize=3.5,
                        linestyle="none", elinewidth=1.0, capsize=0, markeredgecolor="white",
                        markeredgewidth=0.5, label=label)
    # Above the plot, since inside it would cover a no-control point at this width; reordered
    # so the rows read no control and the lookahead, then the global-local advisory.
    handles, labels = bottom.get_legend_handles_labels()
    _boxed(fig.legend([handles[k] for k in (0, 2, 1)], [labels[k] for k in (0, 2, 1)],
                      loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True,
                      fancybox=False, fontsize=6, handlelength=0.8, handletextpad=0.2,
                      columnspacing=0.8, borderaxespad=0.0, borderpad=0.3, labelspacing=0.2))
    bottom.set_title("jammed segment-min", loc="left", fontsize=6.5, color=INK, pad=3)
    bottom.set_ylim(bottom=0)
    bottom.set_xlim(-0.45, len(order) - 0.55)
    bottom.set_xticks(range(len(order)), [LABELS[s] for s in order])
    bottom.tick_params(labelsize=6)
    fig.savefig(out / "fig_jams.pdf")
    fig.savefig(out / "fig_jams.png", dpi=300)
    plt.close(fig)
    return {"order": order, "series": series}


def fig_lag(records_l: list[dict], records_m: list[dict], m_summary: dict, arms,
            out: Path) -> dict:
    D1 = float(m_summary["feed_artefact"]["D"])
    lbar, pq = D1 - 10.0, 10.0
    scenarios = ["B9", "I_exit"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.4), sharey=False)
    series = {}
    for ax, s in zip(axes, scenarios):
        for label, arm, variant, colour, marker in arms:
            if arm == "no_control":
                # No control reads no feed, so D does not apply; the caption gives its value.
                continue
            pts = []
            for mult in (0.0, 1.0, 2.0, 4.0):
                from_m = mult == 1.0
                src = records_m if from_m else records_l
                vals = list(_jam_by_seed(src, arm, variant, s, m=None if from_m else mult).values())
                if not vals:
                    continue
                mean, se = _mean_2se(vals)
                D = 0.0 if mult == 0.0 else mult * lbar + pq
                pts.append((D, mean, se))
            if not pts:
                continue
            series[(s, label)] = pts
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                        color=colour, marker=marker, markersize=4, linewidth=1.2, elinewidth=1.0,
                        capsize=0, markeredgecolor="white", markeredgewidth=0.6, label=label)
        ax.set_title(LABELS[s].replace("\n", " "), fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("feed delay D (s)")
        ax.set_xticks([0.0, D1, 2 * lbar + pq, 4 * lbar + pq],
                      ["0", f"{D1:.0f}", f"{2 * lbar + pq:.0f}", f"{4 * lbar + pq:.0f}"])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("jammed segment-min")
    # One legend for both panels, above their titles and centred over the pair.
    left, right = axes[0].get_position().x0, axes[1].get_position().x1
    handles, labels = axes[0].get_legend_handles_labels()
    _boxed(fig.legend(handles, labels, loc="lower center", bbox_to_anchor=((left + right) / 2, 1.0),
                      ncol=2, frameon=True, fancybox=False, handlelength=1.6,
                      columnspacing=1.2, borderaxespad=0.0))
    fig.savefig(out / "fig_lag.pdf")
    fig.savefig(out / "fig_lag.png", dpi=300)
    plt.close(fig)
    return {f"{k[0]}|{k[1]}": v for k, v in series.items()}


def fig_penetration(records_p: list[dict], records_m: list[dict], arms, out: Path,
                    earlier_rule: bool = False) -> dict:
    """Jam time against the share that follows. Every point must hold no control's seeds."""
    _, nc_arm, nc_variant, _, _ = arms[0]
    _, best_arm, best_variant, _, _ = arms[2]
    lines = [("Plan alone", "plan", "-", GREEN, "s", "-"),
             ("Global-local", best_arm, best_variant, BLUE, "o", "-")]
    if earlier_rule:
        lines.append(("Global-local, no gate", "rule", "m0", PINK, "D", (0, (3, 1.5))))
    elif best_arm == "rule" and "_g" not in best_variant:
        lines.append(("Global-local with feed gate", "rule", f"{best_variant}_g{GATE_SHOWN:g}",
                      PINK, "D", (0, (3, 1.5))))
    shares = sorted({float(r["penetration"]) for r in records_p} | {1.0})
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.3))
    series = {}
    for ax, s in zip(axes, ("B9", "I_exit")):
        nc = _jam_by_seed(records_m, nc_arm, nc_variant, s)
        ax.axhline(statistics.fmean(nc.values()), color=GREY, linewidth=1.0,
                   linestyle=(0, (1, 1.5)), label="No control")
        for j, (label, arm, variant, colour, marker, style) in enumerate(lines):
            pts = []
            for share in shares:
                src = records_m if share == 1.0 else records_p
                by_seed = _jam_by_seed(src, arm, variant if share == 1.0
                                       else f"{variant}_p{share:g}", s)
                if set(by_seed) != set(nc):
                    raise SystemExit(f"{arm}/{variant} at {share:g} on {s} has seeds "
                                     f"{sorted(by_seed)}, not no control's {sorted(nc)}")
                pts.append((100 * share, *_mean_2se(list(by_seed.values()))))
            series[f"{s}|{label}"] = pts
            dx = (j - (len(lines) - 1) / 2) * 0.9
            ax.errorbar([p[0] + dx for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                        color=colour, marker=marker, markersize=3.2, linewidth=1.1,
                        linestyle=style, elinewidth=0.8, capsize=0, markeredgecolor="white",
                        markeredgewidth=0.5, label=label)
        series[f"{s}|No control"] = statistics.fmean(nc.values())
        ax.set_title(LABELS[s].replace("\n", " "), fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("vehicles following (%)")
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        top = max([statistics.fmean(nc.values())]
                  + [p[1] + p[2] for k, v in series.items() if k.startswith(f"{s}|")
                     and isinstance(v, list) for p in v])
        ax.set_ylim(0, 1.08 * top)
    axes[0].set_ylabel("jammed segment-min")
    # One row, flush with the right panel: centred, it is wider than the panels and widens the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    _boxed(fig.legend(handles, labels, loc="lower right",
                      bbox_to_anchor=(axes[1].get_position().x1, 1.0), ncol=4, frameon=True,
                      fancybox=False, fontsize=6, handlelength=1.2, handletextpad=0.3,
                      columnspacing=0.6, borderpad=0.3, borderaxespad=0.0))
    fig.savefig(out / "fig_penetration.pdf")
    fig.savefig(out / "fig_penetration.png", dpi=300)
    plt.close(fig)
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--selection", default=str(SELECTION))
    parser.add_argument("--gate-selection", default=None,
                        help="block N's N_summary.json: the rule's selected feed gate")
    parser.add_argument("--mechanism-selection", default=None,
                        help="block N2's N2_summary.json: the rule's selected queue release "
                             "and gates")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    results, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    m_summary = json.loads((results / "M_summary.json").read_text())
    s_summary = json.loads(Path(args.selection).read_text())
    leads = json.loads((results / "leads.json").read_text())
    records_m = _records(results / "M.jsonl")
    if args.gate_selection and args.mechanism_selection:
        raise SystemExit("--mechanism-selection replaces block N's gate; pass one of them")
    chosen = args.mechanism_selection or args.gate_selection
    n_summary = json.loads(Path(chosen).read_text()) if chosen else None
    arms = paper_arms(s_summary, n_summary)
    D = float(m_summary["feed_artefact"]["D"])
    written = {"warning": fig_warning(leads, D, out),
               "jams": fig_jams(records_m, m_summary, arms, out)}
    if (results / "L.jsonl").exists():
        written["lag"] = fig_lag(_records(results / "L.jsonl"), records_m, m_summary, arms, out)
    if (results / "P.jsonl").exists():
        records_p = _records(results / "P.jsonl")
        for path in sorted((results / "sweep").glob("p*/P.jsonl")):
            records_p += _records(path)
        written["penetration"] = fig_penetration(
            records_p, records_m, arms, out,
            earlier_rule=bool(args.mechanism_selection) and arms[2][2] != "m0")
    (out / "paper_jam_figs_series.json").write_text(json.dumps(written, indent=1, default=str))
    print("wrote", ", ".join(sorted(written)), "to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
