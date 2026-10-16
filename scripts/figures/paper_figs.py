"""The RA-L paper's three result figures, drawn at IEEE column width.

    .venv/bin/python scripts/figures/paper_figs.py paper/figs agreement flow drain

Every plotted value comes from `feed_figures.py`'s series functions over
`results/feed/`; this script only draws them, so no number in it is a result.
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "figures"))
import feed_figures as ff
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "legend.fontsize": 6, "legend.frameon": False,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})
R = REPO / "results" / "feed"
def load(n): return json.loads((R / n).read_text())
W = 3.45  # IEEE column width, inches

def fig_agreement():
    s = ff.fig_feed_and_rule(load("feed_process.json"), load("A_summary.json"))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W, 1.55), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (line, mk) in enumerate(zip(s["panel_a"], ["o", "s", "^", "D"])):
        ax_a.plot(line["t_min"], line["phase"], mk, ms=2.2, mfc="none", mew=0.6, color=str(0.15 + 0.3 * i), label=f"drive {i + 1}")
    ax_a.legend(loc="upper center", fontsize=5.5, handletextpad=0.1, ncol=3, columnspacing=0.6, bbox_to_anchor=(0.55, 1.0))
    ax_a.set_xlabel("(a) time since first version (min)"); ax_a.set_ylabel("publication second")
    ax_a.set_yticks([0, 15, 30, 45, 60]); ax_a.set_ylim(-3, 63)
    b = s["panel_b"]
    ax_b.plot(b["line_x"], b["line_y"], "-", color="0.5", label="Lemma 1(b) bound")
    xs, ys = zip(*b["measured"]); ax_b.plot(xs, ys, "o", ms=3, color="k", label="Algorithm 1")
    refs = b["references"]
    ax_b.axhline(refs["no_rule_60"], ls="--", lw=0.7, color="k"); ax_b.text(10, refs["no_rule_60"] - 0.035, "no rule, 60 s", fontsize=5.5, ha="right")
    ax_b.axhline(refs["no_rule_10"], ls=":", lw=0.8, color="k"); ax_b.text(10, refs["no_rule_10"] + 0.012, "no rule, 10 s", fontsize=5.5, ha="right")
    ax_b.set_xlabel(r"(b) $D-\bar L$ (s)"); ax_b.set_ylabel("disagreement")
    ax_b.set_ylim(-0.01, 0.45); ax_b.legend(loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(OUT / "fig_agreement.pdf"); plt.close(fig)
    return s

if __name__ == "__main__":
    what = sys.argv[2:] or ["agreement"]
    if "agreement" in what: s = fig_agreement(); print("agreement refs", s["panel_b"]["references"], "n versions", [len(l["t_min"]) for l in s["panel_a"]])

def fig_flow():
    s = ff.fig_flow_consequence(load("B_summary.json"), load("C_summary.json"))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W, 1.55), gridspec_kw={"width_ratios": [1.1, 1]})
    a = s["panel_a"]
    for pts, mk, lab in ((a["delta"], "o", "on previous version"), (a["penetration"], "s", "not following ($1-p$)")):
        x, y, e = zip(*pts)
        ax_a.errorbar(x, y, yerr=e, fmt=mk, ms=2.8, mfc="none", mew=0.7, capsize=1.5, elinewidth=0.6, color="k", label=lab)
    ax_a.axhline(0, color="0.6", lw=0.5)
    ax_a.set_xlabel("fraction of vehicles"); ax_a.set_ylabel("flow loss (veh/h)"); ax_a.set_ylim(-90, 360)
    ax_a.legend(loc="upper left", handletextpad=0.3, fontsize=5.5)
    b = s["panel_b"]
    x, y, e = zip(*b["staleness"])
    ax_b.errorbar(x, y, yerr=e, fmt="o", ms=2.8, mfc="none", mew=0.7, capsize=1.5, elinewidth=0.6, color="k")
    ax_b.axhline(0, color="0.6", lw=0.5)
    ax_b.set_xlabel("snapshot age added (s)"); ax_b.set_ylabel("flow loss (veh/h)")
    ax_a.text(0.97, 0.04, "(a)", transform=ax_a.transAxes, va="bottom", ha="right", fontsize=7)
    ax_b.text(0.97, 0.97, "(b)", transform=ax_b.transAxes, va="top", ha="right", fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(OUT / "fig_flow.pdf"); plt.close(fig)
    return s

if __name__ == "__main__" and "flow" in sys.argv[2:]:
    s = fig_flow(); print("flow series", json.dumps(s)[:600])

def fig_drain():
    s = ff.fig_drain(load("D_summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.5))
    names = {"mainz_surge": "(a) surge: min after demand stops", "mainz_burst": "(b) burst: min after it ends"}
    for ax, scen in zip(axes, ("mainz_surge", "mainz_burst")):
        p = s[scen]
        for arm, ls, lab in (("no_control", "--", "no control"), ("here", "-", "deployed policy")):
            d = p[arm]
            t = [0.0] + [x / 60 for x in d["times"]]; y = [1.0] + list(d["survival"])
            t.append(p["cap_minus_t0"] / 60); y.append(y[-1])
            ax.step(t, y, where="post", ls=ls, color="k", lw=0.9, label=lab)
        ax.set_ylim(-0.03, 1.05); ax.set_xlim(0, p["cap_minus_t0"] / 60)
        ax.set_xlabel(names[scen], fontsize=6.5); ax.set_ylabel("not yet drained")
    axes[1].legend(loc="lower left", fontsize=5.5)
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(OUT / "fig_drain.pdf"); plt.close(fig)
    return s

if __name__ == "__main__" and "drain" in sys.argv[2:]:
    s = fig_drain(); print({k: {a: (v[a]["fraction_drained"], v[a]["median"]) for a in ("no_control", "here")} for k, v in s.items()})
