"""Column-width figures for the jam-avoidance letter, from the committed results only.

    .venv/bin/python scripts/figures/paper_jam_figs.py --results-dir results/jam_approach \
        --out paper/figs

Reads `{M.jsonl,M_summary.json,leads.json}` and, when present, `L.jsonl` from the results
directory, and the selection (`S_summary.json`) from `--selection`, which defaults to
`results/jam/S_summary.json`: the thresholds were selected once, on the gated boundary, and
the approach results carry no selection of their own. It writes three PDFs, each with a
300 dpi PNG beside it for checking:

* `fig_warning.pdf`: per trigger, the feed's speed warning and the camera's density warning
  before onset on the first segment to jam (block M's no-control runs), median and
  interquartile range, with the plan's switch delay D drawn as a line.
* `fig_jams.pdf`: per trigger, ordered by no control's median onset, the jammed
  segment-minutes, mean and two standard errors over the 15 evaluation seeds, for no control,
  the feed lookahead and the two-layer controller. The seeds with no jam of 120 s or more are
  in the paper's table, so the figure does not repeat them; they stay in the series JSON.
* `fig_lag.pdf` (only with block L): jammed segment-minutes against the feed delay D
  on the largest burst and the exit block, for the lookahead and the two-layer controller
  (no control reads no feed, so D does not apply to it).

The three paper arms are no control, the lookahead at its selected threshold and the best
model, both read from `S_summary.json`. Colours follow the entity: the camera and the
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
INK, MUTED = "#2b2b29", "#6b6b66"
COLUMN_IN = 3.45

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


def _records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("type") not in ("header", "footer")]


def paper_arms(s_summary: dict) -> list[tuple[str, str | None, str, str, str]]:
    """(label, arm, variant, colour, marker) for the three arms the paper shows."""
    best = s_summary["best"]
    look = s_summary["selected"]["lookahead"]
    look_variant = f"J{look['J_th']:g}"
    best_variant = best["name"].split("/", 1)[1] if "/" in best["name"] else "-"
    return [("No control", "no_control", "-", GREY, "^"),
            ("Feed lookahead", "lookahead", look_variant, ORANGE, "s"),
            ("Global-local", best["arm"], best_variant, BLUE, "o")]


def _jam_by_seed(records: list[dict], arm: str, variant: str, scenario: str,
                 m: float | None = None) -> dict[int, float]:
    out = {}
    for r in records:
        if r["arm"] != arm or r["scenario"] != scenario:
            continue
        if variant != "-" and not str(r["variant"]).startswith(variant):
            continue
        if m is not None and abs(float(r.get("m", 1.0)) - m) > 1e-9:
            continue
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
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.3))
    order = sorted(TRIGGERS, key=lambda s: statistics.median(series[s]["feed"] or [0]))
    for i, s in enumerate(order):
        for dx, key, colour, marker, label in ((-0.12, "feed", ORANGE, "s", "Feed speed warning"),
                                               (0.12, "camera", BLUE, "o", "Camera density warning")):
            v = series[s][key]
            if not v:
                continue
            q1, med, q3 = statistics.quantiles(v, n=4) if len(v) > 1 else (v[0], v[0], v[0])
            ax.vlines(i + dx, max(q1, 5), q3, color=colour, linewidth=1.5)
            ax.plot(i + dx, max(med, 5), marker=marker, color=colour, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=label if i == 0 else None, linestyle="none")
    ax.axhline(D, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(0.55, D * 1.12, f"feed delay D = {D:.1f} s", color=MUTED,
            fontsize=6.5, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(5, 4000)
    ax.set_xticks(range(len(order)), [LABELS[s] for s in order])
    ax.set_ylabel("warning before onset (s)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
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
    fig, bottom = plt.subplots(figsize=(COLUMN_IN, 1.35))
    width = 0.26
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
        bottom.errorbar(xs, means, yerr=ses, color=colour, marker=marker, markersize=4.5,
                        linestyle="none", elinewidth=1.2, capsize=0, markeredgecolor="white",
                        markeredgewidth=0.6, label=label)
    bottom.legend(loc="upper right", ncol=1, frameon=False, handlelength=1.0,
                  borderaxespad=0.1)
    bottom.set_ylabel("jammed segment-min")
    bottom.set_ylim(bottom=0)
    bottom.set_xticks(range(len(order)), [LABELS[s] for s in order])
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
    # Headroom above the largest error bar, so the legend sits above the data.
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.2, borderaxespad=0.1)
    fig.savefig(out / "fig_lag.pdf")
    fig.savefig(out / "fig_lag.png", dpi=300)
    plt.close(fig)
    return {f"{k[0]}|{k[1]}": v for k, v in series.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--selection", default=str(SELECTION))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    results, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    m_summary = json.loads((results / "M_summary.json").read_text())
    s_summary = json.loads(Path(args.selection).read_text())
    leads = json.loads((results / "leads.json").read_text())
    records_m = _records(results / "M.jsonl")
    arms = paper_arms(s_summary)
    D = float(m_summary["feed_artefact"]["D"])
    written = {"warning": fig_warning(leads, D, out),
               "jams": fig_jams(records_m, m_summary, arms, out)}
    if (results / "L.jsonl").exists():
        written["lag"] = fig_lag(_records(results / "L.jsonl"), records_m, m_summary, arms, out)
    (out / "paper_jam_figs_series.json").write_text(json.dumps(written, indent=1, default=str))
    print("wrote", ", ".join(sorted(written)), "to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
