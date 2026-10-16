"""The jam experiment runner (plan sections 8.1 to 8.3, step S12).

Unit tests drive the runner with a stub episode function and need no SUMO. The sanity
tests at the bottom run `--smoke` end to end; run this file in its own process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_jam_experiments as rje  # noqa: E402

SELECTION = rje.Selection(J_th=4.0, m_on=0.1, theta=0.45, best_arm="rule",
                          best_params={"m_on": 0.1})


class TestSpecs:
    def test_block_sizes_match_the_matrix(self):
        assert len(rje.block_specs("G")) == 20
        assert len(rje.block_specs("G", grid=True)) == 70  # the base setting is shared
        assert len(rje.block_specs("T")) == 60
        assert len(rje.block_specs("T", dagger=True)) == 60
        assert len(rje.block_specs("S")) == 390
        assert len(rje.block_specs("M", selection=SELECTION)) == 525 + 105 + 105 + 30
        # Amendment 1: m in {0, 2, 4} for the lookahead, the plan and the best model.
        assert len(rje.block_specs("L", selection=SELECTION)) == 3 * 3 * 2 * 15
        assert len(rje.block_specs("P", selection=SELECTION)) == 120
        assert len(rje.block_specs("K", selection=SELECTION)) == 60

    def test_block_l_runs_the_three_lag_levels(self):
        specs = rje.block_specs("L", selection=SELECTION)
        assert sorted({s.m for s in specs}) == [0.0, 2.0, 4.0]
        assert {s.arm for s in specs} == {"lookahead", "plan", "rule"}
        assert {s.scenario for s in specs} == {"B9", "I_exit"}

    def test_k_is_empty_when_the_best_model_reads_no_camera(self):
        plan_best = rje.Selection(4.0, 0.1, 0.45, "plan", {})
        assert rje.block_specs("K", selection=plan_best) == []
        # L and P then hold the plan once.
        assert {s.arm for s in rje.block_specs("P", selection=plan_best)} == {"plan"}

    def test_durations_follow_j21(self):
        durations = {(s.block, s.scenario): s.duration_s
                     for b in ("G", "S") for s in rje.block_specs(b)}
        assert set(durations.values()) == {3600.0}
        m = {s.scenario: s.duration_s for s in rje.block_specs("M", selection=SELECTION)}
        assert m == {"B5": 7200.0, "B7.5": 7200.0, "B9": 7200.0, "I_exit": 7200.0,
                     "I_mid": 7200.0, "BG": 3600.0, "S45": 2500.0}

    def test_seeds_outside_a_block_are_refused(self):
        with pytest.raises(SystemExit):
            rje.block_specs("S", seeds=[16])
        with pytest.raises(SystemExit):
            rje.block_specs("T", seeds=[11])
        with pytest.raises(SystemExit):
            rje.block_specs("M", seeds=[15], selection=SELECTION)
        with pytest.raises(SystemExit):
            rje.block_specs("G", seeds=[6])
        assert {s.seed for s in rje.block_specs("G", seeds=[2, 3])} == {2, 3}

    def test_arms_share_identical_seed_lists(self):
        for block in ("G", "S", "M"):
            specs = rje.block_specs(block, selection=SELECTION)
            by_arm = {}
            for s in specs:
                by_arm.setdefault((s.arm, s.variant, s.scenario), set()).add(s.seed)
            assert len({tuple(sorted(v)) for v in by_arm.values()}) == 1

    def test_the_oracle_in_m_uses_the_selected_rule_threshold_and_a_wide_variant(self):
        specs = rje.block_specs("M", selection=SELECTION)
        oracle = {s.variant: s for s in specs if s.arm == "oracle"}
        assert set(oracle) == {"tau10_m0.1", "tau10_m0.1_R200"}
        assert dict(oracle["tau10_m0.1_R200"].params)["oracle_R"] == 200.0
        assert {s.scenario for s in specs if s.variant == "tau10_m0.1_R200"} == {"B9", "I_exit"}

    def test_block_t_samples_features(self):
        assert {s.sample_fraction for s in rje.block_specs("T")} == {0.05}
        dagger = rje.block_specs("T", dagger=True)
        assert {(s.arm, dict(s.params)["theta"]) for s in dagger} == {("learned", 0.5)}

    def test_smoke_runs_one_episode_per_arm_on_seed_1(self):
        specs = rje.smoke_subset(rje.block_specs("S", smoke=True))
        assert [s.arm for s in specs] == ["no_control", "plan", "lookahead", "take_the_slower",
                                          "rule", "learned"]
        assert {(s.seed, s.duration_s) for s in specs} == {(1, 600.0)}


def _stub_result(spec):
    return {"flow": 1000.0 + spec.seed, "arrived": 10, "trace": [{"t": 0.0}],
            "write_log": None, "samples": None, "colliding_vehicles": 0,
            "collision_teleports": 0, "writes_above_plan": 0, "sensor": "none",
            "jam": {"jam_time_min": 1.0, "prevented": True, "onset_s": None}}


def _header(**identity):
    base = {"type": "header", "checkpoint": {"sha256": "c"},
            "feed_artefact": {"artefact_sha256": "f"}, "camera_model": None, "predictor": None,
            "commit": None, "dirty": None}
    base.update(identity)
    return base


class TestRunBlock:
    def test_resume_skips_present_keys(self, tmp_path):
        specs = rje.block_specs("G")[:4]
        ran = []

        def episode(spec):
            ran.append(spec.key)
            return _stub_result(spec)

        rje.run_block("G", specs[:2], episode, results_dir=tmp_path, traces_dir=tmp_path,
                      header_record=_header())
        out = rje.run_block("G", specs, episode, results_dir=tmp_path, traces_dir=tmp_path,
                            header_record=_header(), resume=True)
        assert ran == [s.key for s in specs[:2]] + [s.key for s in specs[2:]]
        assert out["skipped"] == [s.key for s in specs[:2]]

    def test_an_existing_block_needs_resume(self, tmp_path):
        specs = rje.block_specs("G")[:1]
        rje.run_block("G", specs, _stub_result, results_dir=tmp_path, traces_dir=tmp_path,
                      header_record=_header())
        with pytest.raises(SystemExit):
            rje.run_block("G", specs, _stub_result, results_dir=tmp_path, traces_dir=tmp_path,
                          header_record=_header())

    def test_the_pause_file_stops_before_the_next_episode(self, tmp_path):
        specs = rje.block_specs("G")[:3]
        pause = tmp_path / "PAUSE"
        calls = []

        def episode(spec):
            calls.append(spec.key)
            pause.write_text("")
            return _stub_result(spec)

        out = rje.run_block("G", specs, episode, results_dir=tmp_path, traces_dir=tmp_path,
                            header_record=_header(), pause_file=pause)
        assert out["status"] == "paused" and calls == [specs[0].key]
        state = json.loads((tmp_path / "G_state.json").read_text())
        assert state["next_key"] == specs[1].key and state["completed"] == 1

    def test_resume_refuses_another_feed_artefact(self, tmp_path):
        specs = rje.block_specs("G")[:2]
        rje.run_block("G", specs[:1], _stub_result, results_dir=tmp_path, traces_dir=tmp_path,
                      header_record=_header())
        with pytest.raises(SystemExit):
            rje.run_block("G", specs, _stub_result, results_dir=tmp_path, traces_dir=tmp_path,
                          header_record=_header(feed_artefact={"artefact_sha256": "other"}),
                          resume=True)

    def test_records_carry_their_identity_and_trace_file(self, tmp_path):
        specs = rje.block_specs("G")[:1]
        rje.run_block("G", specs, _stub_result, results_dir=tmp_path, traces_dir=tmp_path,
                      header_record=_header())
        _, episodes = rje.read_block(tmp_path, "G")
        assert episodes[0]["identity"] == ["c", "f", None, None]
        assert Path(episodes[0]["trace_file"]).exists()
        assert "trace" not in episodes[0]


class TestHeader:
    def test_the_header_carries_every_artefact_hash(self, tmp_path):
        camera = tmp_path / "camera.json"
        camera.write_text("{}")
        predictor = tmp_path / "pred.json"
        predictor.write_text("{}")
        weights = tmp_path / "weights.json"
        weights.write_text(json.dumps({"sha256": "w" * 64}))
        context = rje.Context(camera_model=camera, predictor=predictor, weights_record=weights)
        from src.sumo.jam_episode import FeedSetting

        context.feed = FeedSetting(rje.DEFAULT_FEED_ARTEFACT, rje.DEFAULT_FEED_SESSION).parameters()
        specs = rje.block_specs("S")
        h = rje.header("S", context, specs, argv=["--block", "S"])
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert h["checkpoint"]["sha256"] == sha(rje.DEFAULT_CHECKPOINT)
        assert h["feed_artefact"]["artefact_sha256"] == sha(rje.DEFAULT_FEED_ARTEFACT)
        assert h["feed_artefact"]["session"] == "afternoon_20261016_mainz"
        assert h["feed_artefact"]["placeholder"] is False
        assert h["feed_artefact"]["D"] == pytest.approx(h["feed_artefact"]["L_bar"] + 10.0)
        assert h["camera_model"]["sha256"] == sha(camera)
        assert h["predictor"]["sha256"] == sha(predictor)
        assert h["weights_sha256"] == "w" * 64
        assert h["drain_thresholds"]["sha256"]
        assert "mainz_b9000:schedule" in h["scenario_files"]

    def test_block_g_needs_no_camera_or_predictor(self):
        context = rje.Context()
        context.feed = {"artefact_sha256": "f"}
        h = rje.header("G", context, rje.block_specs("G"), argv=[])
        assert h["camera_model"] is None and h["predictor"] is None


def _selection_file(tmp_path, **overrides):
    data = {"seeds": [11, 12, 13, 14, 15], "selected": {"lookahead": {"J_th": 4.0},
                                                       "rule": {"m_on": 0.1},
                                                       "learned": {"theta": 0.45}},
            "best": {"arm": "rule", "params": {"m_on": 0.1}},
            "feed_artefact_sha256": "f", "camera_model_sha256": "c", "predictor_sha256": "p"}
    data.update(overrides)
    path = tmp_path / "S_summary.json"
    path.write_text(json.dumps(data))
    return path


class TestSelection:
    def test_reads_thresholds_and_records_its_hash(self, tmp_path):
        path = _selection_file(tmp_path)
        s = rje.read_selection(path, feed_sha="f", camera_sha="c", predictor_sha="p")
        assert (s.J_th, s.m_on, s.theta, s.best_arm) == (4.0, 0.1, 0.45, "rule")
        assert s.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_m_refuses_a_selection_built_on_another_feed_artefact(self, tmp_path):
        path = _selection_file(tmp_path)
        with pytest.raises(SystemExit):
            rje.read_selection(path, feed_sha="other", camera_sha="c", predictor_sha="p")
        with pytest.raises(SystemExit):
            rje.read_selection(path, feed_sha="f", camera_sha="other", predictor_sha="p")

    def test_a_selection_on_evaluation_seeds_is_refused(self, tmp_path):
        path = _selection_file(tmp_path, seeds=[11, 16])
        with pytest.raises(SystemExit):
            rje.read_selection(path, feed_sha="f", camera_sha="c", predictor_sha="p")

    def test_the_selector_refuses_a_seed_16_record(self):
        record = {"arm": "plan", "variant": "-", "scenario": "B9", "seed": 16, "flow": 1.0,
                  "jam": {"jam_time_min_900_3600": 1.0, "prevented": True, "onset_s": None}}
        with pytest.raises(SystemExit):
            rje.summarize_s([record], rje.Context())


def _jam(minutes, prevented=False):
    return {"jam_time_min": minutes, "prevented": prevented}


class TestBlockGRule:
    def test_a_clear_reduction_passes(self):
        plan = {"B9": {s: _jam(20.0 + s) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0) for s in range(1, 6)}}
        oracle = {"B9": {s: _jam(10.0 + s * 0.9) for s in range(1, 6)},
                  "I_exit": {s: _jam(15.0 + (0.1 if s % 2 else -0.1)) for s in range(1, 6)}}
        out = rje.g_pass(oracle, plan)
        assert out["scenarios"]["B9"]["below_by_2se"]
        assert not out["scenarios"]["I_exit"]["above_by_2se"]
        assert out["passes"]

    def test_prevention_on_two_more_b9_seeds_passes(self):
        plan = {"B9": {s: _jam(20.0) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        oracle = {"B9": {s: _jam(0.0 if s <= 2 else 21.0 + s, prevented=s <= 2)
                         for s in range(1, 6)},
                  "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        out = rje.g_pass(oracle, plan)
        assert out["scenarios"]["B9"]["prevention_rule"] and out["passes"]

    def test_a_clear_increase_in_either_scenario_fails(self):
        plan = {"B9": {s: _jam(20.0 + s) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        oracle = {"B9": {s: _jam(5.0 + s) for s in range(1, 6)},
                  "I_exit": {s: _jam(25.0 + s * 1.1) for s in range(1, 6)}}
        out = rje.g_pass(oracle, plan)
        assert out["scenarios"]["I_exit"]["above_by_2se"] and not out["passes"]

    def test_no_difference_fails(self):
        plan = {"B9": {s: _jam(20.0 + s) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        oracle = {k: {s: _jam(v["jam_time_min"] + (0.5 if s % 2 else -0.5))
                      for s, v in rows.items()} for k, rows in plan.items()}
        assert not rje.g_pass(oracle, plan)["passes"]

    def test_unpaired_seeds_are_refused(self):
        with pytest.raises(SystemExit):
            rje.g_pass({"B9": {1: _jam(1.0)}}, {"B9": {2: _jam(1.0)}})

    def test_the_decision_names_the_grid_after_a_failing_base(self):
        records = []
        for arm, variant, minutes in (("plan", "-", 20.0), ("oracle", "tau10_m0.1", 20.0)):
            for scenario in ("B9", "I_exit"):
                for seed in range(1, 6):
                    records.append({"arm": arm, "variant": variant, "scenario": scenario,
                                    "seed": seed,
                                    "jam": _jam(minutes + (0.3 if seed % 2 else -0.3))})
        assert rje.summarize_g(records)["decision"].startswith("base setting fails")


# ----------------------------------------------------------------- SUMO sanity


def test_block_g_smoke_runs_end_to_end(tmp_path):
    results, traces = tmp_path / "results", tmp_path / "traces"
    assert rje.main(["--block", "G", "--smoke", "--results-dir", str(results),
                     "--traces-dir", str(traces)]) == 0
    summary = json.loads((results / "G_summary.json").read_text())
    assert summary["episodes"] == 2 and summary["smoke"] is True
    assert summary["controlled_episodes_with_collisions"] == 0
    assert summary["writes_above_plan_total"] == 0
    assert summary["max_speed_mismatched_total"] == 0
    assert summary["placeholder"] is False
    assert summary["feed_artefact"]["session"] == "afternoon_20261016_mainz"
    headers, episodes = rje.read_block(results, "G")
    assert {e["arm"] for e in episodes} == {"oracle", "plan"}
    assert summary["decision"] != "incomplete: the base oracle setting has not run"
    assert {e["sensor"] for e in episodes} == {"oracle", "none"}
    assert all(e["seed"] == 1 and e["duration_s"] == 600.0 for e in episodes)
    for e in episodes:
        trace = json.loads(Path(e["trace_file"]).read_text())
        assert trace["trace"] and trace["write_log"]
        assert e["jam"]["cap_s"] == 600.0
    assert headers[0]["camera_model"] is None
