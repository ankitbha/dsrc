"""The rule's feed gate and the network-only selection block N (2026-10-20).

Unit tests need no SUMO. The sanity tests at the bottom run three 840 s B9 episodes on the
approach boundary (about a minute); run this file in its own process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_jam_experiments as rje  # noqa: E402
from src.sumo.correction import (  # noqa: E402
    RuleCorrection, advised_jam_factor, gate_signal, rule_margin,
)
from src.sumo.jam_episode import JamArm  # noqa: E402

SELECTION = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                          best_params={"m_on": 0.0})
GATED_SELECTION = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                                best_params={"m_on": 0.0}, gate_J=1.5)


class TestSignal:
    def test_the_advisory_s_own_slowing_reads_zero(self):
        # Free flow; a plan of 0.75 met at 45 km/h (J 2.5) and at one step below, 30 km/h
        # (J 5); a plan of 0.5 met at 30 km/h: none is congestion.
        jam = np.array([0.0, 2.5, 5.0, 5.0])
        plan = np.array([1.0, 0.75, 0.75, 0.5])
        assert advised_jam_factor(jam, plan).tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_traffic_slower_than_one_step_below_the_plan_reads_its_shortfall(self):
        # J 4 under a plan of 1.0 is 36 of 60 km/h against 45: 10 (1 - 0.6 / 0.75) = 2.
        # J 6 under 0.5 is 24 against 30: 2. J 7.5 under 0.75 is 15 against 30: 5.
        # A stopped segment reads 10 whatever the plan.
        jam = np.array([4.0, 6.0, 7.5, 10.0, 10.0])
        plan = np.array([1.0, 0.5, 0.75, 1.0, 0.5])
        assert advised_jam_factor(jam, plan) == pytest.approx([2.0, 2.0, 5.0, 10.0, 10.0])

    def test_the_signal_is_the_largest_over_the_segments(self):
        assert gate_signal(np.array([4.0, 7.5, 0.0]), np.array([1.0, 0.75, 1.0])) == \
            pytest.approx(5.0)
        assert gate_signal(np.zeros(12), np.ones(12)) == 0.0


class TestGatedRule:
    def test_without_a_mask_the_rule_is_unchanged(self):
        rng = np.random.default_rng(3)
        plain, masked = RuleCorrection(0.0), RuleCorrection(0.0)
        ids = [f"v{i}" for i in range(20)]
        for tick in range(60):
            k = rng.uniform(0.0, 0.3, size=20)
            s = rng.normal(0.0, 0.005, size=20)
            assert (plain.update(ids, 5.0 * tick, k, s).tolist()
                    == masked.update(ids, 5.0 * tick, k, s, allow=np.ones(20, bool)).tolist())

    def test_a_closed_gate_stops_the_switch_on(self):
        rule = RuleCorrection(0.0)
        k, s = np.array([0.2]), np.array([0.0])
        assert rule_margin(k, s)[0] < 0.0
        assert rule.update(["a"], 0.0, k, s, allow=np.array([False])).tolist() == [False]
        assert rule.update(["a"], 5.0, k, s, allow=np.array([True])).tolist() == [True]

    def test_a_vehicle_on_is_released_after_the_hysteresis_once_the_gate_closes(self):
        # On at 0 s; the gate closes from 5 s while the road stays full. The hysteresis
        # holds it on for 30 s, so it is on at 25 s and off at 30 s.
        rule = RuleCorrection(0.0)
        k, s = np.array([0.2]), np.array([0.0])
        assert rule.update(["a"], 0.0, k, s, allow=np.array([True])).tolist() == [True]
        states = [rule.update(["a"], t, k, s, allow=np.array([False]))[0]
                  for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)]
        assert states == [True, True, True, True, True, False]

    def test_the_gate_is_refused_on_an_arm_without_the_rule_s_switch(self):
        assert JamArm("rule", m_on=0.0, gate_J=1.0).gate_J == 1.0
        assert JamArm("oracle", m_on=0.0, gate_J=1.0).gate_J == 1.0
        for arm, params in (("lookahead", {"J_th": 2.0}), ("plan", {}),
                            ("learned", {"theta": 0.75}), ("take_the_slower", {})):
            with pytest.raises(ValueError):
                JamArm(arm, gate_J=1.0, **params)


class TestBlockN:
    def test_block_n_runs_the_grid_on_the_selection_seeds(self):
        specs = rje.block_specs("N", selection=SELECTION, oracle_window=60.0)
        variants = sorted({(s.arm, s.variant) for s in specs})
        assert variants == [("no_control", "-"), ("plan", "-"), ("rule", "m0"),
                            ("rule", "m0_g0.5"), ("rule", "m0_g1"), ("rule", "m0_g1.5"),
                            ("rule", "m0_g2")]
        assert len(specs) == 7 * 6 * 5
        assert {s.seed for s in specs} == set(range(11, 16))
        assert {s.scenario for s in specs} == {"B5", "B7.5", "B9", "I_exit", "I_mid", "BG"}
        assert {s.duration_s for s in specs} == {3600.0}
        gated = next(s for s in specs if s.variant == "m0_g1.5")
        assert gated.jam_arm().gate_J == 1.5 and gated.jam_arm().m_on == 0.0
        with pytest.raises(SystemExit):
            rje.block_specs("N", seeds=[16], selection=SELECTION, oracle_window=60.0)

    def test_block_n_runs_on_the_approach_boundary_only(self):
        rje.check_block_boundary("N", "approach")
        with pytest.raises(SystemExit):
            rje.check_block_boundary("N", "gated")

    def test_without_a_gate_the_evaluation_blocks_are_unchanged(self):
        for block in ("M", "L", "P", "K"):
            keys = [s.key for s in rje.block_specs(block, selection=SELECTION,
                                                   oracle_window=60.0)]
            assert not any("_g" in k.split("/")[2] for k in keys), block

    def test_the_selected_gate_reaches_the_rule_in_m_to_k(self):
        def variants(block):
            return sorted({s.variant for s in rje.block_specs(block, selection=GATED_SELECTION,
                                                               oracle_window=60.0)
                           if s.arm == "rule"})
        # Block M keeps the rule without the gate beside it, for the no-trigger contrast.
        assert variants("M") == ["m0", "m0_g1.5"]
        assert variants("L") == ["m0_g1.5_lag0", "m0_g1.5_lag2", "m0_g1.5_lag4"]
        assert variants("P") == ["m0_g1.5_p0.5", "m0_g1.5_p0.75"]
        assert variants("K") == ["m0_g1.5_b0.8", "m0_g1.5_b1.25", "m0_g1.5_b1_ps0.5"]
        spec = next(s for s in rje.block_specs("P", selection=GATED_SELECTION,
                                               oracle_window=60.0) if s.arm == "rule")
        assert spec.jam_arm().gate_J == 1.5
        # The oracle keeps the rule's switch without the gate.
        oracle = next(s for s in rje.block_specs("M", selection=GATED_SELECTION,
                                                 oracle_window=60.0) if s.arm == "oracle")
        assert oracle.jam_arm().gate_J is None


def _n_record(arm, variant, scenario, seed, jam, flow=3400.0, collide=0, params=None):
    return {"arm": arm, "variant": variant, "scenario": scenario, "seed": seed, "flow": flow,
            "params": params or {}, "jam": {"jam_time_min_900_3600": jam, "prevented": jam == 0},
            "storage": {"network_min": 1000.0}, "colliding_vehicles": collide,
            "collision_teleports": 0, "writes_above_plan": 0, "gate_open_share": None}


def _n_block(collide_g1=False):
    # C(rule m0) = (10 - 50) / 100 = -0.4; C(g1) = -0.395, within C_TIE; C(g2) = -0.2.
    jams = {"no_control": 100.0, "plan": 50.0, "m0": 10.0, "m0_g1": 10.5, "m0_g2": 30.0}
    bg_flow = {"no_control": 3570.0, "plan": 3450.0, "m0": 3300.0, "m0_g1": 3400.0,
               "m0_g2": 3500.0}
    records = []
    for scenario in ("B5", "B7.5", "B9", "I_exit", "I_mid", "BG"):
        for seed in range(11, 16):
            for name, jam in jams.items():
                arm = name if name in ("no_control", "plan") else "rule"
                variant = "-" if arm != "rule" else name
                params = ({"m_on": 0.0} if name == "m0" else
                          {"m_on": 0.0, "gate_J": float(name[-1])} if arm == "rule" else {})
                records.append(_n_record(
                    arm, variant, scenario, seed, 0.0 if scenario == "BG" else jam,
                    flow=bg_flow[name] if scenario == "BG" else 3500.0,
                    collide=int(collide_g1 and name == "m0_g1" and seed == 12),
                    params=params))
    return records


class TestSummaryN:
    def test_the_lowest_c_wins_and_a_tie_goes_to_the_flow_with_no_trigger(self):
        summary = rje.summarize_n(_n_block())
        c = {k: v["C"] for k, v in summary["candidates"].items()}
        assert c["rule/m0"] == pytest.approx(-0.4) and c["rule/m0_g1"] == pytest.approx(-0.395)
        assert c["rule/m0_g2"] == pytest.approx(-0.2)
        assert summary["selected"] == {"label": "rule/m0_g1", "variant": "m0_g1",
                                       "m_on": 0.0, "gate_J": 1.0}
        assert summary["candidates"]["rule/m0_g1"]["bg_flow_mean"] == 3400.0
        assert summary["reference"]["no_control"]["bg_flow_mean"] == 3570.0

    def test_a_candidate_with_a_collision_is_excluded(self):
        summary = rje.summarize_n(_n_block(collide_g1=True))
        assert summary["candidates"]["rule/m0_g1"]["excluded"] is True
        assert summary["selected"]["label"] == "rule/m0"
        assert summary["selected"]["gate_J"] is None

    def test_seeds_outside_the_selection_are_refused(self):
        records = _n_block()
        records[0]["seed"] = 16
        with pytest.raises(SystemExit):
            rje.summarize_n(records)

    def test_some_of_the_selection_seeds_select_nothing(self):
        summary = rje.summarize_n([r for r in _n_block() if r["seed"] == 11])
        assert summary["selected"] is None and summary["decision"].startswith("incomplete")


class TestReadGateSelection:
    def _write(self, tmp_path, **changes):
        data = {"block": "N", "seeds": list(range(11, 16)),
                "selected": {"label": "rule/m0_g1.5", "variant": "m0_g1.5", "m_on": 0.0,
                             "gate_J": 1.5}}
        data.update(changes)
        path = tmp_path / "N_summary.json"
        path.write_text(json.dumps(data))
        return path

    def test_the_gate_is_read_with_its_sha256(self, tmp_path):
        path = self._write(tmp_path)
        out = rje.read_gate_selection(path, SELECTION)
        assert out.gate_J == 1.5 and out.m_on == 0.0 and out.gate_path == str(path)
        assert len(out.gate_sha256) == 64

    def test_a_selection_without_a_gate_runs_the_rule_without_one(self, tmp_path):
        path = self._write(tmp_path, selected={"label": "rule/m0", "variant": "m0",
                                               "m_on": 0.0, "gate_J": None})
        assert rje.read_gate_selection(path, SELECTION).gate_J is None

    @pytest.mark.parametrize("changes", [{"block": "S"}, {"seeds": [11, 12]},
                                         {"selected": None},
                                         {"selected": {"m_on": 0.1, "gate_J": 1.5}}])
    def test_a_wrong_summary_is_refused(self, tmp_path, changes):
        with pytest.raises(SystemExit):
            rje.read_gate_selection(self._write(tmp_path, **changes), SELECTION)

    def test_the_gated_boundary_refuses_a_gate_selection(self, tmp_path):
        with pytest.raises(SystemExit):
            rje.main(["--block", "M", "--gate-selection", str(self._write(tmp_path)),
                      "--results-dir", str(tmp_path), "--summarize-only"])


# ----------------------------------------------------------------- SUMO sanity


@pytest.fixture(scope="module")
def gate_runs():
    import evaluate_mainz_checkpoints as emc

    import src.sumo.jam_episode as je
    from src.sumo.sensing import CameraModel

    model = emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                                emc.CHECKPOINT_FEATURES["here"], 12)
    camera = CameraModel.from_artefact(REPO_ROOT / "results/jam/camera_error_model.json")
    feed = je.FeedSetting(REPO_ROOT / "results/jam/feed_process_afternoon.json",
                          "afternoon_20261016_mainz", m=1.0)
    out = {}
    for name, arm, cam in (("rule", JamArm("rule", m_on=0.0), camera),
                           ("open", JamArm("rule", m_on=0.0, gate_J=0.0), camera),
                           ("shut", JamArm("rule", m_on=0.0, gate_J=10.5), camera),
                           ("plan", JamArm("plan"), None)):
        engines = []
        result = je.run_jam_episode(model, 1, "B9", arm, feed=feed, duration_s=600.0,
                                    window_start_s=300.0, camera=cam, check_max_speed=True,
                                    boundary=je.approach_boundary(), on_engine=engines.append)
        out[name] = (result, engines[0])
    return out


SAME = ("flow", "arrived", "writes", "writes_stepped_down", "corrected_vehicle_ticks",
        "active_share", "active_changing_share", "running", "colliding_vehicles")


def test_a_gate_that_is_always_open_is_the_rule_without_one(gate_runs):
    rule, _ = gate_runs["rule"]
    open_, _ = gate_runs["open"]
    assert rule["writes_stepped_down"] > 0
    for key in SAME:
        assert open_[key] == rule[key], key
    assert open_["trace"] == rule["trace"]
    assert open_["gate_open_share"] == 1.0 and rule["gate_open_share"] is None


def test_a_gate_that_never_opens_is_the_plan_alone(gate_runs):
    shut, _ = gate_runs["shut"]
    plan, _ = gate_runs["plan"]
    assert shut["writes_stepped_down"] == 0 and shut["active_share"] == 0.0
    assert shut["gate_open_share"] == 0.0
    for key in ("flow", "arrived", "writes", "running"):
        assert shut[key] == plan[key], key
    assert [s["speed_kmh"] for s in shut["trace"]] == [s["speed_kmh"] for s in plan["trace"]]


def test_the_recorded_signal_is_each_version_s_network_rows_against_its_plan(gate_runs):
    from src.sumo.mainz import SPEED_ACTION_FRACTIONS

    out, engine = gate_runs["open"]
    fractions = np.asarray(SPEED_ACTION_FRACTIONS)
    assert out["gate_signal_by_version"]
    for k, value in out["gate_signal_by_version"].items():
        version = int(k)
        expected = gate_signal(engine.states[version][:, 2], fractions[engine.vectors[version]])
        assert value == pytest.approx(round(expected, 4), abs=1e-9)
        assert engine.states[version].shape == (12, 5)
