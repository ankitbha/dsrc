"""The jam figures: each function returns the series it plots, equal to the summaries.

The summaries are produced by the runner's own `summarize_eval` on synthetic episodes, so
the figures are tested against the schema the runner writes. No SUMO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "figures"))
import jam_figures as jf  # noqa: E402
import run_jam_experiments as rje  # noqa: E402

SEEDS = (16, 17, 18)
S_SUMMARY = {"selected": {"lookahead": {"J_th": 4.0}, "rule": {"m_on": 0.1},
                          "learned": {"theta": 0.45}},
             "best": {"arm": "rule", "params": {"m_on": 0.1}}}


def _episode(arm, variant, scenario, seed, jam_time, onset, flow=3500.0, active=None):
    return {"arm": arm, "variant": variant, "scenario": scenario, "seed": seed,
            "flow": flow + seed, "active_share": active, "return": 1.0, "wall_s": 1.0,
            "jam": {"start_s": 900.0, "cap_s": 7200.0, "onset_s": onset,
                    "prevented": onset is None, "jam_time_min": jam_time + seed,
                    "jam_time_min_900_3600": jam_time, "held_min": 2.0, "pending_min": 0.0,
                    # A censored run reports cap - t_0, as drain.drain_time does.
                    "drained": onset is not None, "drain_s": 600.0 if onset else 6000.0}}


@pytest.fixture(scope="module")
def m_summary():
    episodes = []
    arms = (("no_control", "-", 60.0, 1500.0), ("lookahead", "J4", 40.0, 1700.0),
            ("plan", "-", 30.0, 1800.0), ("rule", "m0.1", 20.0, None))
    for scenario, shift in (("B9", 0.0), ("I_exit", -800.0), ("B5", 300.0), ("BG", 0.0)):
        for arm, variant, minutes, onset in arms:
            for seed in SEEDS:
                o = None if (onset is None or scenario == "BG") else onset + shift
                episodes.append(_episode(arm, variant, scenario, seed, minutes, o,
                                         active=0.1 if arm == "rule" else None))
    return rje.summarize_eval("M", episodes)


@pytest.fixture(scope="module")
def l_summary():
    episodes = []
    for m, extra in ((0.0, -5.0), (2.0, 5.0), (4.0, 12.0)):
        for arm, variant, minutes in (("lookahead", "J4", 40.0), ("plan", "-", 30.0),
                                      ("rule", "m0.1", 20.0)):
            for scenario in ("B9", "I_exit"):
                for seed in SEEDS:
                    episodes.append(_episode(arm, f"{variant}_lag{m:g}", scenario, seed,
                                             minutes + extra * (1 if arm != "rule" else 0.5),
                                             1600.0))
    return rje.summarize_eval("L", episodes)


CAMERA = {"sumo_table": "0.20",
          "tables": {"0.20": {"bins_m": [[0, 10], [10, 20], [20, 30]],
                              "detection": {"p": [0.9, 0.7, None]},
                              "distance_noise": {"sigma_ema_m": [0.3, 0.9, None]},
                              "R_max_m": 20.0}},
          "sensitivity": {"1280": {"tables": {"0.20": {
              "detection": {"p": [0.95, 0.8, 0.4]},
              "distance_noise": {"sigma_ema_m": [0.2, 0.6, 1.4]}}}}}}
LEADS = {"episodes": [{"scenario": "B9", "seed": 16, "W_feed": 150.0, "W_cam": 400.0},
                      {"scenario": "B9", "seed": 17, "W_feed": None, "W_cam": 380.0},
                      {"scenario": "I_exit", "seed": 16, "W_feed": 30.0, "W_cam": 40.0}]}


class TestFigures:
    def test_fig1_is_the_camera_table_and_the_leads(self):
        series = jf.fig_warning(CAMERA, LEADS, D=99.49)
        assert series["a"]["distance_m"] == [5.0, 15.0, 25.0]
        assert series["a"]["p_640"] == [0.9, 0.7, None]
        assert series["a"]["sigma_1280"] == [0.2, 0.6, 1.4]
        assert series["a"]["R_max_m"] == 20.0
        assert series["b"]["scenarios"]["B9"] == {"seed": [16, 17], "W_feed": [150.0, None],
                                                  "W_cam": [400.0, 380.0]}
        assert series["b"]["band_s"] == [99.49, 159.49]

    def test_fig2_orders_by_no_control_onset_and_reads_the_rows(self, m_summary):
        series = jf.fig_onset_prevention(m_summary, S_SUMMARY)
        assert series["scenarios"] == ["I_exit", "B9", "B5"]
        best = series["arms"]["best model"]
        assert best["prevented_share"] == [1.0, 1.0, 1.0]
        assert series["arms"]["no control"]["prevented_share"] == [0.0, 0.0, 0.0]
        row = m_summary["rows"]["lookahead/J4/B9"]
        look = series["arms"]["feed lookahead"]
        assert look["jam_time_min"][1] == row["jam_time_min_mean"]
        assert look["two_se_vs_plan"][1] == row["paired_jam_time_vs_plan"]["two_se"]

    def test_fig3_puts_block_m_at_m_1_and_block_l_at_0_2_4(self, m_summary, l_summary):
        series = jf.fig_lag(l_summary, m_summary, S_SUMMARY)
        assert series["lag_axis"] == [0.0, 1.0, 2.0, 4.0]
        look = series["scenarios"]["B9"]["feed lookahead"]
        assert look == [l_summary["rows"]["lookahead/J4_lag0/B9"]["jam_time_min_mean"],
                        m_summary["rows"]["lookahead/J4/B9"]["jam_time_min_mean"],
                        l_summary["rows"]["lookahead/J4_lag2/B9"]["jam_time_min_mean"],
                        l_summary["rows"]["lookahead/J4_lag4/B9"]["jam_time_min_mean"]]
        nc = series["scenarios"]["I_exit"]["no control"]
        assert len(set(nc)) == 1 and nc[0] == m_summary["rows"]["no_control/-/I_exit"][
            "jam_time_min_mean"]
        assert series["scenarios"]["B9"]["best model"][3] == l_summary["rows"][
            "rule/m0.1_lag4/B9"]["jam_time_min_mean"]

    def test_table1_reads_the_rows_and_bg_against_the_plan(self, m_summary):
        table = jf.table_main(m_summary, S_SUMMARY)
        first = next(r for r in table["rows"] if r["scenario"] == "B9" and r["arm"] == "best model")
        row = m_summary["rows"]["rule/m0.1/B9"]
        assert first["prevented"] == row["prevented"] == 3 and first["n"] == 3
        assert first["jam_time_min"] == row["jam_time_min_mean"]
        assert first["held_min"] == row["held_min_mean"]
        bg = {r["arm"]: r for r in table["bg"]}
        assert bg["best model"]["active_share"] == pytest.approx(0.1)
        assert bg["best model"]["flow_minus_plan"] == pytest.approx(0.0)

    def test_the_paper_arms_need_a_selection(self):
        with pytest.raises(ValueError):
            jf.paper_arms({"best": None, "selected": {}})

    def test_a_missing_artefact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            jf.load(tmp_path / "M_summary.json")
        with pytest.raises(FileNotFoundError):
            jf.main(["--results-dir", str(tmp_path)])

    def test_render_all_writes_three_pdfs_and_the_table(self, tmp_path, m_summary, l_summary):
        paths = jf.render_all(tmp_path, camera=CAMERA, leads=LEADS, D=99.49, s=S_SUMMARY,
                              m=m_summary, l=l_summary)
        assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in paths.values())
        assert {Path(p).suffix for p in paths.values()} == {".pdf", ".json"}
