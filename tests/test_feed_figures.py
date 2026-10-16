"""The feed figures: each function returns the series it plots, equal to the summaries.

The summaries are produced by the runner's own summarize functions on synthetic
episodes, so the figures are tested against the schema the runner writes. No SUMO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "figures"))
import feed_figures as ff  # noqa: E402
import run_feed_experiments as rfe  # noqa: E402

ARTEFACT = REPO_ROOT / "results" / "feed" / "feed_process.json"
SEEDS = (16, 17, 18)


@pytest.fixture(scope="module")
def feed_process():
    return json.loads(ARTEFACT.read_text())


@pytest.fixture(scope="module")
def a_summary():
    context = rfe.Context()
    context.feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
    episodes = []
    for arm, variant, flow, delta in (("published", "-", 3750.0, 0.0),
                                      ("no_rule", "P60", 3740.0, 0.25),
                                      ("no_rule", "P10", 3745.0, 0.04),
                                      ("fixed_phase", "psi46", 3748.0, 0.02),
                                      ("version_keyed", "D-20", 3741.0, 0.3),
                                      ("version_keyed", "D+0", 3744.0, 0.05),
                                      ("version_keyed", "D+10", 3747.0, 0.0)):
        for seed in SEEDS:
            episodes.append({"arm": arm, "variant": variant, "seed": seed,
                             "flow": flow + seed % 2, "delta_mean": delta,
                             "command_delta_mean": delta / 10, "age_mean_s": 90.0 + delta})
    return rfe.summarize_a(episodes, [], context)


@pytest.fixture(scope="module")
def b_summary():
    references = {"here": {s: 3750.0 for s in SEEDS}, "no_control": {s: 3700.0 for s in SEEDS}}
    episodes = []
    for d in (0.1, 0.25, 0.5, 0.75):
        for seed in SEEDS:
            episodes.append({"arm": "inject_delta", "variant": f"delta{d:g}", "seed": seed,
                             "flow": 3750.0 - 100.0 * d - seed % 2, "delta_mean": 0.9 * d})
    for s in (30.0, 60.0, 120.0):
        for seed in SEEDS:
            episodes.append({"arm": "inject_stale", "variant": f"s{s:g}", "seed": seed,
                             "flow": 3750.0 - 0.2 * s - seed % 2})
    return rfe.summarize_b(episodes, references)


@pytest.fixture(scope="module")
def c_summary():
    references = {"here": {s: 3750.0 for s in SEEDS}, "no_control": {s: 3700.0 for s in SEEDS}}
    episodes = [{"arm": "penetration", "variant": f"p{p:g}", "seed": seed,
                 "flow": 3700.0 + 50.0 * p + seed % 2}
                for p in (0.75, 0.5, 0.25, 0.1) for seed in SEEDS]
    return rfe.summarize_c(episodes, references)


@pytest.fixture(scope="module")
def d_summary(tmp_path_factory):
    """Jams end at 2,600 s without control (never, on seed 18) and 1,500 s with it."""
    tmp = tmp_path_factory.mktemp("d")
    episodes = []
    for arm in ("no_control", "here"):
        for seed in SEEDS:
            jam_until = 1500 if arm == "here" else (9000 if seed == 18 else 2600)
            trace = [{"t": float(t), "count": [3] * 12, "held": 0, "pending": 0,
                      "speed_kmh": [20.0 if 600 <= t < jam_until else 50.0] * 12,
                      "arrived": t} for t in range(10, 3001, 10)]
            path = tmp / f"{arm}_{seed}.json"
            path.write_text(json.dumps({"trace": trace}))
            episodes.append({"arm": arm, "seed": seed, "scenario": "mainz_surge",
                             "duration_s": 3000.0, "trace_file": str(path), "flow": 1.0})
    # The sustained episodes the network threshold is fitted on (Amendment 1, A4).
    rng = np.random.default_rng(3)
    for arm in ("no_control", "here"):
        trace = [{"t": float(t), "count": [3] * 12, "held": 0, "pending": 0, "arrived": t,
                  "speed_kmh": [float(rng.normal(25 if rng.random() < 0.25 else 52, 3))
                                for _ in range(12)]} for t in range(10, 2521, 10)]
        path = tmp / f"{arm}_mainz.json"
        path.write_text(json.dumps({"trace": trace}))
        episodes.append({"arm": arm, "seed": 16, "scenario": "mainz", "duration_s": 2500.0,
                         "trace_file": str(path), "flow": 1.0})
    return rfe.summarize_d(episodes)


@pytest.fixture(scope="module")
def e_summary(tmp_path_factory):
    return {"block": "E", "levels": {
        "1": {"rho_hat": 0.4, "rho_UCB": 0.6, "certificate": True},
        "0.5": {"rho_hat": 0.8, "rho_UCB": 1.3, "certificate": False},
        "0.25": {"note": "fewer than two complete seeds: no bound"}},
        "p_min": {"p_min": 1.0, "first_not_met": 0.5}}


class TestFeedAndRule:
    def test_panel_a_carries_each_version_s_phase_and_bracket(self, feed_process, a_summary):
        series = ff.fig_feed_and_rule(feed_process, a_summary)
        for line in series["panel_a"]:
            session = feed_process["sessions"][line["session"]]
            assert line["phase"] == [v["T"] % 60 for v in session["per_version"]]
            assert line["L_up"] == [v["L_up"] for v in session["per_version"]]
            assert line["L_lo"] == [v["L_lo"] for v in session["per_version"]]

    def test_panel_b_is_the_summary_s_lemma_and_measured_points(self, feed_process, a_summary):
        series = ff.fig_feed_and_rule(feed_process, a_summary)
        assert series["panel_b"]["bound_points"] == [
            (r["D_minus_L_bar"], r["bound_cycle_average_g_min"]) for r in a_summary["lemma"]]
        assert series["panel_b"]["measured"] == [
            (r["D_minus_L_bar"], r["measured_delta_mean"]) for r in a_summary["lemma"]
            if r["measured_delta_mean"] is not None]
        assert series["panel_b"]["references"]["no_rule_60"] == \
            a_summary["rows"]["no_rule/P60"]["delta_mean"]
        assert series["panel_b"]["references"]["fixed_phase"] == \
            a_summary["rows"]["fixed_phase/psi46"]["delta_mean"]
        # The line passes through the summary's bound at each D.
        x, y = np.array(series["panel_b"]["line_x"]), np.array(series["panel_b"]["line_y"])
        for offset, bound in series["panel_b"]["bound_points"]:
            assert np.interp(offset, x, y) == pytest.approx(bound, abs=1e-3)


class TestFlowConsequence:
    def test_the_loss_points_are_the_paired_gains_negated(self, b_summary, c_summary):
        series = ff.fig_flow_consequence(b_summary, c_summary)
        for x, loss, bar in series["panel_a"]["delta"]:
            row = b_summary["rows"][f"inject_delta/delta{x:g}"]
            assert loss == -row["paired_vs_published"]["gain"]
            assert bar == row["paired_vs_published"]["two_se"]
        for x, loss, bar in series["panel_a"]["penetration"]:
            row = c_summary["rows"][f"penetration/p{1 - x:g}"]
            assert loss == -row["paired_vs_published"]["gain"]
        assert series["panel_a"]["slope"] == b_summary["fit_loss_vs_delta"]["slope"]
        assert series["panel_b"]["slope"] == b_summary["fit_loss_vs_staleness"]["slope"]
        assert [p[0] for p in series["panel_b"]["staleness"]] == [30.0, 60.0, 120.0]


class TestDrain:
    def test_the_primary_rule_is_plotted_by_default(self, d_summary):
        entry = d_summary["scenarios"]["mainz_surge"]
        assert ff.fig_drain(d_summary)["mainz_surge"]["rule"] == entry["primary"]

    def test_km_curves_and_censoring_come_from_the_named_rule(self, d_summary):
        series = ff.fig_drain(d_summary, rule="fixed_30/H120")
        entry = d_summary["scenarios"]["mainz_surge"]
        primary = entry["drain"]["fixed_30/H120"]
        panel = series["mainz_surge"]
        for arm in ("no_control", "here"):
            assert panel[arm]["times"] == primary["arms"][arm]["km"]["times"]
            assert panel[arm]["survival"] == primary["arms"][arm]["km"]["survival"]
            assert panel[arm]["fraction_drained"] == primary["arms"][arm]["fraction_drained"]
            assert panel[arm]["median"] == primary["arms"][arm]["km"]["median"]
        # Seed 18 without control is still jammed at the cap: one censoring mark.
        assert panel["no_control"]["censored"] == [entry["cap_s"] - entry["t_0"]]
        assert panel["sensitivities"]


class TestTable:
    def test_rows_carry_the_summary_values(self, a_summary, c_summary, e_summary):
        table = ff.table_certificate_and_rules(a_summary, c_summary, e_summary)
        rules = {row["rule"]: row for row in table["rules"]}
        assert rules["no_rule/P60"]["delta"] == a_summary["rows"]["no_rule/P60"]["delta_mean"]
        assert rules["version_keyed/D+0"]["flow_gain"] == \
            a_summary["rows"]["version_keyed/D+0"]["paired_vs_published"]["gain"]
        levels = {row["penetration"]: row for row in table["penetration"]}
        assert levels[1.0]["rho_UCB"] == 0.6 and levels[1.0]["p_min"] is True
        assert levels[0.5]["certificate"] is False
        assert levels[0.5]["flow_gain"] == c_summary["rows"]["penetration/p0.5"][
            "paired_vs_published"]["gain"]
        assert levels[0.25]["rho_UCB"] is None


def test_a_missing_artefact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="A_summary.json"):
        ff.load(tmp_path / "A_summary.json")


def test_the_figures_render_to_pdf(tmp_path, feed_process, a_summary, b_summary, c_summary,
                                   d_summary, e_summary):
    written = ff.render_all(tmp_path, feed_process=feed_process, a=a_summary, b=b_summary,
                            c=c_summary, d=d_summary, e=e_summary)
    assert {p.name for p in written} == {"fig1_feed_and_rule.pdf", "fig2_flow_consequence.pdf",
                                         "fig3_drain.pdf", "table1.json"}
    assert all(p.stat().st_size > 0 for p in written)


def test_the_script_holds_no_result_literal():
    """No number from a summary is typed into the figure script (attack point 15)."""
    source = (REPO_ROOT / "scripts" / "figures" / "feed_figures.py").read_text()
    for literal in ("3751", "62.8", "109.7", "67.08", "0.25)", "1562"):
        assert literal not in source


def test_without_a_network_threshold_the_drain_figure_draws_the_fixed_rule(d_summary):
    stripped = json.loads(json.dumps(d_summary))
    stripped["scenarios"]["mainz_surge"]["primary"] = None
    assert ff.fig_drain(stripped)["mainz_surge"]["rule"] == "fixed_30/H120"
