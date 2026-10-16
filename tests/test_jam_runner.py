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
        assert len(rje.block_specs("M", selection=SELECTION, oracle_window=60.0)) == 525 + 105 + 105 + 30
        # Amendment 1: m in {0, 2, 4} for the lookahead, the plan and the best model.
        assert len(rje.block_specs("L", selection=SELECTION)) == 3 * 3 * 2 * 15
        assert len(rje.block_specs("P", selection=SELECTION)) == 120
        # Two biases at the stored pair spread, and half the spread at b = 1 (Amendment 3).
        assert len(rje.block_specs("K", selection=SELECTION)) == 90

    def test_block_l_runs_the_three_lag_levels(self):
        specs = rje.block_specs("L", selection=SELECTION)
        assert sorted({s.m for s in specs}) == [0.0, 2.0, 4.0]
        assert {s.arm for s in specs} == {"lookahead", "plan", "rule"}
        assert {s.scenario for s in specs} == {"B9", "I_exit"}

    def test_block_k_runs_half_the_stored_pair_spread(self):
        specs = rje.block_specs("K", selection=SELECTION)
        cells = {(s.bias, s.pair_sigma_scale) for s in specs}
        assert cells == {(0.8, 1.0), (1.25, 1.0), (1.0, 0.5)}
        half = [s for s in specs if s.pair_sigma_scale == 0.5]
        assert len(half) == 30 and {s.scenario for s in half} == {"B9", "I_exit"}
        assert all(s.arm == "rule" and s.variant.endswith("_ps0.5") for s in half)

    def test_the_episode_runner_hands_each_spec_its_bias_and_pair_spread(self, tmp_path,
                                                                         monkeypatch):
        import src.sumo.jam_episode as je

        artefact = {"schema": "camera_error_model/1", "bin_m": 10.0, "sumo_table": "0.20",
                    "method": "width_prior",
                    "tables": {"0.20": {"detection": {"p": [0.9]},
                                        "distance_noise": {"sigma_ema_m": [0.5]},
                                        "closing_rate_noise": {"sigma_v_mps": [0.2]},
                                        "confidence": {"values": [[0.9]]}, "R_max_m": 10.0,
                                        "persistent_error": {"relative_spread": 0.28}}}}
        path = tmp_path / "camera.json"
        path.write_text(json.dumps(artefact))
        seen = []

        def fake_episode(model, seed, scenario, arm, **kwargs):
            seen.append(kwargs["camera"])
            trace = [{"t": float(t), "speed_kmh": [50.0] * 12, "count": [1] * 12,
                      "held": 0, "pending": 0} for t in range(0, 3610, 10)]
            return {"trace": trace}

        monkeypatch.setattr(je, "run_jam_episode", fake_episode)
        context = rje.Context(camera_model=path)
        runner = rje.episode_runner(None, context)
        specs = rje.block_specs("K", selection=SELECTION)
        for spec in (next(s for s in specs if s.bias == 0.8),
                     next(s for s in specs if s.pair_sigma_scale == 0.5)):
            runner(spec)
        assert (seen[0].bias, seen[0].pair_sigma_log) == (0.8, 0.28)
        assert (seen[1].bias, seen[1].pair_sigma_log) == (1.0, pytest.approx(0.14))

    def test_the_camera_goes_to_the_episodes_that_need_it_in_any_order(self, tmp_path,
                                                                        monkeypatch):
        # Validation R2-6: block M's no-control episodes sense passively (section 1.5);
        # they come first in the block, and the camera was loaded only by a camera arm.
        import src.sumo.jam_episode as je

        path = tmp_path / "camera.json"
        path.write_text(json.dumps({
            "schema": "camera_error_model/1", "bin_m": 10.0, "sumo_table": "0.20",
            "tables": {"0.20": {"detection": {"p": [0.9]}, "distance_noise": {"sigma_ema_m": [0.5]},
                                "closing_rate_noise": {"sigma_v_mps": [0.2]},
                                "confidence": {"values": [[0.9]]}, "R_max_m": 10.0}}}))
        given = {}

        def fake_episode(model, seed, scenario, arm, **kwargs):
            given.setdefault(arm.arm, set()).add(kwargs["camera"] is not None)
            trace = [{"t": float(t), "speed_kmh": [50.0] * 12, "count": [1] * 12,
                      "held": 0, "pending": 0} for t in range(0, 7210, 10)]
            return {"trace": trace}

        monkeypatch.setattr(je, "run_jam_episode", fake_episode)
        specs = [s for s in rje.block_specs("M", selection=SELECTION, oracle_window=60.0)
                 if s.seed == 16 and s.arm != "learned"]
        for order in (specs, list(reversed(specs))):
            given.clear()
            runner = rje.episode_runner(None, rje.Context(camera_model=path))
            for spec in order:
                runner(spec)
            assert given == {"no_control": {True}, "lookahead": {False}, "plan": {False},
                             "take_the_slower": {True}, "rule": {True}, "oracle": {False}}
        assert rje.needs_camera([s for s in specs if s.arm == "no_control"])
        s_no_control = [s for s in rje.block_specs("S") if s.arm == "no_control"]
        assert not rje.needs_camera(s_no_control)

    def test_k_is_empty_when_the_best_model_reads_no_camera(self):
        plan_best = rje.Selection(4.0, 0.1, 0.45, "plan", {})
        assert rje.block_specs("K", selection=plan_best) == []
        # L and P then hold the plan once.
        assert {s.arm for s in rje.block_specs("P", selection=plan_best)} == {"plan"}

    def test_durations_follow_j21(self):
        durations = {(s.block, s.scenario): s.duration_s
                     for b in ("G", "S") for s in rje.block_specs(b)}
        assert set(durations.values()) == {3600.0}
        m = {s.scenario: s.duration_s for s in rje.block_specs("M", selection=SELECTION, oracle_window=60.0)}
        assert m == {"B5": 7200.0, "B7.5": 7200.0, "B9": 7200.0, "I_exit": 7200.0,
                     "I_mid": 7200.0, "BG": 3600.0, "S45": 2500.0}

    def test_seeds_outside_a_block_are_refused(self):
        with pytest.raises(SystemExit):
            rje.block_specs("S", seeds=[16])
        with pytest.raises(SystemExit):
            rje.block_specs("T", seeds=[11])
        with pytest.raises(SystemExit):
            rje.block_specs("M", seeds=[15], selection=SELECTION, oracle_window=60.0)
        with pytest.raises(SystemExit):
            rje.block_specs("G", seeds=[6])
        assert {s.seed for s in rje.block_specs("G", seeds=[2, 3])} == {2, 3}

    def test_arms_share_identical_seed_lists(self):
        for block in ("G", "S", "M"):
            specs = rje.block_specs(block, selection=SELECTION, oracle_window=60.0)
            by_arm = {}
            for s in specs:
                by_arm.setdefault((s.arm, s.variant, s.scenario), set()).add(s.seed)
            assert len({tuple(sorted(v)) for v in by_arm.values()}) == 1

    def test_the_oracle_in_m_uses_the_selected_rule_threshold_and_a_wide_variant(self):
        specs = rje.block_specs("M", selection=SELECTION, oracle_window=60.0)
        oracle = {s.variant: s for s in specs if s.arm == "oracle"}
        assert set(oracle) == {"tau10_m0.1_R60", "tau10_m0.1_R200"}
        assert dict(oracle["tau10_m0.1_R200"].params)["oracle_R"] == 200.0
        assert {s.scenario for s in specs if s.variant == "tau10_m0.1_R200"} == {"B9", "I_exit"}

    def test_from_block_s_on_the_oracle_reads_the_camera_s_window(self):
        # Amendment 3 (R2-5): min(80 m, R_max), so the oracle and the camera arms see the
        # same window; oracle_wide stays at 200 m and block G keeps its 80 m oracle.
        specs = rje.block_specs("M", selection=SELECTION, oracle_window=60.0)
        windows = {s.variant: dict(s.params)["oracle_R"] for s in specs if s.arm == "oracle"}
        assert windows == {"tau10_m0.1_R60": 60.0, "tau10_m0.1_R200": 200.0}
        assert all("oracle_R" not in dict(s.params) for s in rje.block_specs("G")
                   if s.arm == "oracle")
        assert {s.jam_arm().oracle_R for s in rje.block_specs("G") if s.arm == "oracle"} == {80.0}
        with pytest.raises(SystemExit):
            rje.block_specs("M", selection=SELECTION)

    def test_the_window_is_min_80_m_and_r_max_from_the_camera_model(self, tmp_path):
        def artefact(r_max):
            path = tmp_path / f"camera_{r_max:g}.json"
            path.write_text(json.dumps({
                "schema": "camera_error_model/1", "bin_m": 10.0, "sumo_table": "0.20",
                "tables": {"0.20": {"detection": {"p": [0.9] * 12},
                                    "distance_noise": {"sigma_ema_m": [0.5] * 12},
                                    "closing_rate_noise": {"sigma_v_mps": [0.2] * 12},
                                    "confidence": {"values": [[0.9]] * 12},
                                    "R_max_m": r_max}}}))
            return path

        assert rje.oracle_window(artefact(60.0)) == 60.0
        assert rje.oracle_window(artefact(100.0)) == 80.0
        with pytest.raises(SystemExit):
            rje.oracle_window(tmp_path / "missing.json")

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

    def test_a_reduction_within_two_standard_errors_does_not_pass(self):
        # Paired differences with mean -1 and two standard errors of 3.1: below zero, not
        # below by two standard errors (kills "below by 2 SE" -> "below zero", M31).
        plan = {"B9": {s: _jam(20.0) for s in range(1, 6)},
                "I_exit": {s: _jam(20.0) for s in range(1, 6)}}
        steps = {1: -5.5, 2: -1.0, 3: 3.5, 4: -3.0, 5: 1.0}
        oracle = {"B9": {s: _jam(20.0 + d) for s, d in steps.items()},
                  "I_exit": {s: _jam(20.0 + d) for s, d in steps.items()}}
        out = rje.g_pass(oracle, plan)
        paired = out["scenarios"]["B9"]["paired_jam_time"]
        assert paired["mean"] == pytest.approx(-1.0) and paired["two_se"] > 1.0
        assert not out["scenarios"]["B9"]["below_by_2se"] and not out["passes"]

    def test_one_more_prevented_b9_seed_does_not_pass(self):
        # Kills a prevention margin of 1 seed (M32).
        plan = {"B9": {s: _jam(20.0) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        oracle = {"B9": {s: _jam(0.0 if s == 1 else 21.0 + s, prevented=s == 1)
                         for s in range(1, 6)},
                  "I_exit": {s: _jam(15.0 + s) for s in range(1, 6)}}
        out = rje.g_pass(oracle, plan)
        assert out["scenarios"]["B9"]["prevented_oracle"] == 1
        assert not out["scenarios"]["B9"]["prevention_rule"] and not out["passes"]

    def test_prevention_on_i_exit_alone_does_not_pass(self):
        # The prevention rule is B9's alone (kills it applying to any scenario, M33).
        plan = {"B9": {s: _jam(20.0 + s) for s in range(1, 6)},
                "I_exit": {s: _jam(15.0) for s in range(1, 6)}}
        oracle = {"B9": {s: _jam(20.0 + s) for s in range(1, 6)},
                  "I_exit": {s: _jam(0.0 if s <= 3 else 16.0 + 3 * s, prevented=s <= 3)
                             for s in range(1, 6)}}
        out = rje.g_pass(oracle, plan)
        assert out["scenarios"]["I_exit"]["prevented_oracle"] == 3
        assert not out["scenarios"]["I_exit"]["prevention_rule"] and not out["passes"]

    def _g_records(self, seeds=range(1, 6), scenarios=("B9", "I_exit"),
                   oracle_minutes=5.0):
        records = []
        for arm, variant, minutes in (("plan", "-", 20.0),
                                      ("oracle", "tau10_m0.1", oracle_minutes)):
            for scenario in scenarios:
                for seed in seeds:
                    records.append({"arm": arm, "variant": variant, "scenario": scenario,
                                    "seed": seed,
                                    "jam": _jam(minutes + (0.3 if seed % 2 else -0.3))})
        return records

    def test_a_complete_base_that_passes_is_go(self):
        assert rje.summarize_g(self._g_records())["decision"] == "go"

    def test_a_missing_seed_or_scenario_makes_the_decision_incomplete(self):
        out = rje.summarize_g(self._g_records(seeds=range(1, 5)))
        assert out["decision"].startswith("incomplete") and out["plan_missing_seeds"]
        out = rje.summarize_g(self._g_records(scenarios=("B9",)))
        assert out["decision"].startswith("incomplete")
        assert out["plan_missing_seeds"] == {"I_exit": [1, 2, 3, 4, 5]}
        records = [r for r in self._g_records()
                   if not (r["arm"] == "oracle" and r["seed"] == 5 and r["scenario"] == "B9")]
        out = rje.summarize_g(records)
        assert out["decision"].startswith("incomplete: the base oracle setting lacks")
        assert out["settings"]["tau10_m0.1"]["passes"] is None

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


SMOKE_CAMERA = REPO_ROOT / "outputs/jam/smoke/camera_error_model.json"
SMOKE_PREDICTOR = REPO_ROOT / "outputs/jam/smoke/results/onset_predictor.json"


@pytest.mark.skipif(not (SMOKE_CAMERA.exists() and SMOKE_PREDICTOR.exists()),
                    reason="the S8 and S11 smoke artefacts have not been written")
def test_block_m_smoke_senses_passively_in_no_control(tmp_path):
    results = tmp_path / "results"
    assert rje.main(["--block", "M", "--smoke", "--camera-model", str(SMOKE_CAMERA),
                     "--predictor", str(SMOKE_PREDICTOR), "--results-dir", str(results),
                     "--traces-dir", str(tmp_path / "traces")]) == 0
    _, episodes = rje.read_block(results, "M")
    no_control = [e for e in episodes if e["arm"] == "no_control"]
    assert no_control and all(e["camera_model"] is not None for e in no_control)
    for e in no_control:
        trace = json.loads(Path(e["trace_file"]).read_text())["trace"]
        assert any(v is not None for sample in trace if sample["t"] >= 300.0
                   for v in sample["k_hat_mean"])
    leads = json.loads((results / "leads.json").read_text())
    assert leads["episodes"] and all(entry["k_hat_available"] for entry in leads["episodes"])
    assert all("W_cam" in entry for entry in leads["episodes"])


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
    # One seed is not block G: no decision is made from it.
    assert summary["decision"].startswith("incomplete")
    assert {e["sensor"] for e in episodes} == {"oracle", "none"}
    from src.sumo.jam_episode import SCENARIOS

    for e in episodes:
        # The metric clock is the scenario's (kills a fixed 300 s, M39) and the threshold
        # the network fit's 19.0 km/h (kills 30 km/h, M46).
        assert e["jam"]["start_s"] == SCENARIOS[e["scenario"]].start_s == 900.0
        assert e["jam"]["v_jam"] == 19.0 and e["jam_30kmh"]["v_jam"] == 30.0
    assert all(e["seed"] == 1 and e["duration_s"] == 600.0 for e in episodes)
    for e in episodes:
        trace = json.loads(Path(e["trace_file"]).read_text())
        assert trace["trace"] and trace["write_log"]
        assert e["jam"]["cap_s"] == 600.0
    assert headers[0]["camera_model"] is None


def _s_records(jam=None, bg_flow=None, collide=None, above=None, seeds=None, drop=None):
    """One record for every block S spec (seeds 11-15, six scenarios), as summarize_s reads
    them. `jam(name)` gives a candidate's jam time on [900, 3600] s; no control 40, plan 20."""
    jam = jam or (lambda name: 19.5)
    out = []
    for s in rje.block_specs("S", seeds=seeds):
        name = s.arm if s.variant == "-" else f"{s.arm}/{s.variant}"
        if drop and drop(s):
            continue
        j = {"no_control": 40.0, "plan": 20.0}.get(name, jam(name))
        flow = (bg_flow or {}).get(name, 3000.0) if s.scenario == "BG" else 3000.0
        out.append({"arm": s.arm, "variant": s.variant, "scenario": s.scenario, "seed": s.seed,
                    "flow": flow, "active_share": 0.1, "return": 1.0,
                    "jam": {"jam_time_min_900_3600": j, "jam_time_min": j, "prevented": False,
                            "onset_s": 1000.0, "start_s": 900.0, "cap_s": 3600.0,
                            "held_min": 0.0, "pending_min": 0.0, "drained": True,
                            "drain_s": 1.0},
                    "colliding_vehicles": 1 if collide == name and s.seed == 13 else 0,
                    "collision_teleports": 0,
                    "writes_above_plan": 1 if above == name and s.seed == 12 else 0})
    return out


class TestSelectionCompleteness:
    """Validation R2-1: block S selects only on seeds 11-15 of every arm on all six scenarios."""

    def _ctx(self):
        context = rje.Context(camera_model=None, predictor=None)
        context.feed = {"artefact_sha256": "f"}
        return context

    def test_complete_data_selects(self):
        out = rje.summarize_s(_s_records(), self._ctx())
        assert out["decision"] == "selected" and out["best"] is not None

    def test_a_seed_subset_selects_nothing(self):
        out = rje.summarize_s(_s_records(seeds=[11, 12]), self._ctx())
        assert out["decision"].startswith("incomplete") and out["best"] is None
        assert out["selected"] == {"lookahead": None, "rule": None, "learned": None}
        assert "plan/-/B5" in out["missing"] and out["missing"]["plan/-/B5"] == [13, 14, 15]

    def test_a_missing_scenario_selects_nothing(self):
        out = rje.summarize_s(_s_records(drop=lambda s: s.scenario == "I_mid"), self._ctx())
        assert out["decision"].startswith("incomplete") and out["best"] is None
        assert out["missing"]["no_control/-/I_mid"] == [11, 12, 13, 14, 15]

    def test_a_candidate_missing_from_one_scenario_selects_nothing(self):
        out = rje.summarize_s(_s_records(drop=lambda s: s.arm == "rule" and s.scenario == "B5"),
                              self._ctx())
        assert out["best"] is None and out["missing"]["rule/m0.1/B5"] == [11, 12, 13, 14, 15]

    def test_read_selection_needs_seeds_11_to_15_and_a_best(self, tmp_path):
        with pytest.raises(SystemExit):
            rje.read_selection(_selection_file(tmp_path, seeds=[11, 12, 13, 14]), feed_sha="f",
                               camera_sha="c", predictor_sha="p")
        with pytest.raises(SystemExit):
            rje.read_selection(_selection_file(tmp_path, best=None), feed_sha="f",
                               camera_sha="c", predictor_sha="p")


class TestSelectionProvenance:
    """Validation R2-2: S_summary.json takes its hashes from block S's own headers."""

    def _block(self, tmp_path, camera_sha, predictor_sha):
        header = {"type": "header", "checkpoint": {"sha256": "k"},
                  "feed_artefact": {"artefact_sha256": "f"},
                  "camera_model": {"sha256": camera_sha}, "predictor": {"sha256": predictor_sha},
                  "config": {"smoke": False}}
        lines = [json.dumps(header)] + [json.dumps({"type": "episode", "key": f"k{n}", **r})
                                        for n, r in enumerate(_s_records())]
        (tmp_path / "S.jsonl").write_text("\n".join(lines) + "\n")

    def _files(self, tmp_path, camera_text, predictor_text):
        camera, predictor = tmp_path / "cam.json", tmp_path / "pred.json"
        camera.write_text(camera_text)
        predictor.write_text(predictor_text)
        return camera, predictor

    def test_the_summary_records_the_hashes_block_s_ran_under(self, tmp_path):
        camera, predictor = self._files(tmp_path, "camera", "predictor")
        self._block(tmp_path, hashlib.sha256(b"camera").hexdigest(),
                    hashlib.sha256(b"predictor").hexdigest())
        context = rje.Context(camera_model=camera, predictor=predictor)
        context.feed = {"artefact_sha256": "f"}
        summary = rje.summarize("S", tmp_path, context)
        assert summary["camera_model_sha256"] == hashlib.sha256(b"camera").hexdigest()
        assert summary["predictor_sha256"] == hashlib.sha256(b"predictor").hexdigest()
        assert summary["feed_artefact_sha256"] == "f"

    def test_a_camera_model_changed_since_block_s_is_refused(self, tmp_path):
        camera, predictor = self._files(tmp_path, "camera refitted", "predictor")
        self._block(tmp_path, hashlib.sha256(b"camera").hexdigest(),
                    hashlib.sha256(b"predictor").hexdigest())
        context = rje.Context(camera_model=camera, predictor=predictor)
        context.feed = {"artefact_sha256": "f"}
        with pytest.raises(SystemExit):
            rje.summarize("S", tmp_path, context)
