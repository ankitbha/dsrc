"""The experiment runner: resume, pause, header and seed lists with stubbed episodes,
then one real 600 s episode of every block on seed 16 into temporary directories.

Run this file in its own process; the last test starts SUMO five times.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_feed_experiments as rfe  # noqa: E402

ARTEFACT = REPO_ROOT / "results" / "feed" / "feed_process.json"


def _specs(n_seeds: int = 3) -> list[rfe.Spec]:
    return [rfe.Spec("B", "inject_delta", variant, seed, mode_args=(0.1,))
            for variant in ("delta0.1", "delta0.25") for seed in range(16, 16 + n_seeds)]


def _stub(calls: list[str], on_call=None):
    def run(spec):
        calls.append(spec.key)
        if on_call:
            on_call(len(calls))
        return {"flow": 3000.0 + spec.seed, "arrived": 10, "trace": [{"t": 10.0}],
                "snapshots": [], "late_draws": [], "delta_mean": 0.1}
    return run


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestResumeAndPause:
    def test_resume_skips_the_two_keys_already_written_and_runs_the_other_four(self, tmp_path):
        specs = _specs()
        jsonl = tmp_path / "B.jsonl"
        jsonl.write_text("".join(json.dumps({"type": "episode", "key": s.key, "arm": s.arm,
                                             "variant": s.variant, "seed": s.seed,
                                             "flow": 1.0}) + "\n" for s in specs[:2]))
        calls: list[str] = []
        result = rfe.run_block("B", specs, _stub(calls), results_dir=tmp_path,
                               traces_dir=tmp_path / "traces", header_record={"type": "header"},
                               resume=True)
        assert calls == [s.key for s in specs[2:]]
        assert result["status"] == "complete"
        assert result["skipped"] == [s.key for s in specs[:2]]
        episodes = [r for r in _jsonl(jsonl) if r["type"] == "episode"]
        assert sorted(r["key"] for r in episodes) == sorted(s.key for s in specs)
        assert rfe.resume_keys(jsonl) == {s.key for s in specs}

    def test_an_existing_block_without_resume_is_refused(self, tmp_path):
        (tmp_path / "B.jsonl").write_text("")
        with pytest.raises(SystemExit):
            rfe.run_block("B", _specs(), _stub([]), results_dir=tmp_path,
                          traces_dir=tmp_path, header_record={"type": "header"})

    def test_a_pause_file_before_episode_3_stops_after_episode_2(self, tmp_path):
        specs = _specs()
        pause = tmp_path / "PAUSE"
        calls: list[str] = []
        result = rfe.run_block("B", specs, _stub(calls, lambda n: n == 2 and pause.touch()),
                               results_dir=tmp_path, traces_dir=tmp_path / "traces",
                               header_record={"type": "header"}, pause_file=pause)
        assert calls == [specs[0].key, specs[1].key]
        assert result["status"] == "paused" and result["next_key"] == specs[2].key
        state = json.loads((tmp_path / "B_state.json").read_text())
        assert state["next_key"] == specs[2].key and state["completed"] == 2
        # Remove the file, resume, and the rest runs; the state file goes away.
        pause.unlink()
        rest: list[str] = []
        rfe.run_block("B", specs, _stub(rest), results_dir=tmp_path,
                      traces_dir=tmp_path / "traces", header_record={"type": "header"},
                      resume=True, pause_file=pause)
        assert rest == [s.key for s in specs[2:]]
        assert not (tmp_path / "B_state.json").exists()

    def test_main_exits_zero_when_paused(self, tmp_path):
        (tmp_path / "PAUSE").touch()
        code = rfe.main(["--block", "C", "--seeds", "16", "17", "--results-dir", str(tmp_path),
                         "--traces-dir", str(tmp_path / "t")],
                        run_episode_factory=lambda model, context: _stub([]))
        assert code == 0
        assert json.loads((tmp_path / "C_state.json").read_text())["next_key"] == \
            "C/penetration/p0.75/16"

    def test_traces_go_to_their_own_file_and_not_into_the_jsonl(self, tmp_path):
        rfe.run_block("B", _specs(1)[:1], _stub([]), results_dir=tmp_path,
                      traces_dir=tmp_path / "traces", header_record={"type": "header"})
        record = [r for r in _jsonl(tmp_path / "B.jsonl") if r["type"] == "episode"][0]
        assert "trace" not in record and "snapshots" not in record
        trace = json.loads(Path(record["trace_file"]).read_text())
        assert trace["trace"] == [{"t": 10.0}]


class TestHeaderAndSpecs:
    def test_the_header_carries_versions_and_digests(self):
        context = rfe.Context(seeds=(16, 17))
        context.feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
        record = rfe.header("A", context,
                            REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt", ["x"])
        assert record["sumo_version"].startswith("Eclipse SUMO sumo")
        assert record["libsumo_version"]
        assert len(record["checkpoint"]["sha256"]) == 64
        assert record["feed_artefact"]["sha256"] == rfe._sha256(ARTEFACT)
        assert record["feed_artefact"]["placeholder"] is True
        assert record["config"]["feed"]["session"] == record["feed_artefact"]["session"]
        assert record["scenario_files"]["mainz:net"]["sha256"]

    def test_block_a_refuses_the_placeholder_without_the_flag(self):
        with pytest.raises(SystemExit, match="placeholder"):
            rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=False)

    def test_block_a_reads_l_bar_and_psi_from_the_replayed_session(self):
        feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
        session = json.loads(ARTEFACT.read_text())["sessions"][feed["session"]]
        assert feed["L_bar"] == session["L_bar"]
        assert feed["D_values"] == [session["L_bar"] + o for o in (-20, -5, 0, 5, 10)]
        assert feed["psi_star"] == session["psi_star"]

    @pytest.mark.parametrize("block", rfe.BLOCKS)
    def test_every_arm_shares_the_identical_seed_list(self, block):
        context = rfe.Context()
        if block == "A":
            context.feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
        specs = rfe.block_specs(block, context)
        by_arm: dict[tuple, list[int]] = {}
        for spec in specs:
            by_arm.setdefault((spec.arm, spec.variant, spec.pulse and spec.pulse[1]), []).append(
                spec.seed)
        lists = {tuple(v) for (arm, *_), v in by_arm.items() if arm != "pulse30"}
        assert lists == {tuple(range(16, 31))}
        assert len({s.key for s in specs}) == len(specs)

    def test_the_matrix_sizes(self):
        context = rfe.Context()
        context.feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
        sizes = {b: len(rfe.block_specs(b, context)) for b in rfe.BLOCKS}
        assert sizes == {"A": 150, "B": 135, "C": 60, "D": 90, "E": 1080}


class TestSummaries:
    @staticmethod
    def _pilot_trace(jam_from=None, jam_to=None, end=12000, pending_until=0):
        return [{"t": float(t), "arrived": t, "held": 0,
                 "pending": 1 if t < pending_until else 0, "count": [3, 3],
                 "speed_kmh": [10.0 if jam_from is not None and jam_from <= t < jam_to
                               else 50.0, 50.0]}
                for t in range(10, end + 1, 10)]

    def test_the_pilot_rule_reads_a_jam_spell_of_h_after_t0(self):
        drains = self._pilot_trace(1000, 1600)
        broken = self._pilot_trace(1680, 9000)
        decision = rfe.pilot_decision(broken, drains, 12000.0, 20.0)
        assert decision["breaks_no_control_down"] is True
        assert decision["controlled_drains_before_cap"] is True
        assert decision["action"] == "keep the scenario"

    def test_a_spell_shorter_than_h_is_not_a_breakdown(self):
        brief = self._pilot_trace(2000, 2110)  # 11 samples, 110 s
        decision = rfe.pilot_decision(brief, self._pilot_trace(), 12000.0, 20.0)
        assert decision["breaks_no_control_down"] is False
        assert "6,000" in decision["action"]

    def test_a_jam_before_t0_only_is_not_a_breakdown(self):
        early = self._pilot_trace(600, 1500)
        assert rfe.pilot_decision(early, self._pilot_trace(), 12000.0, 20.0)[
            "breaks_no_control_down"] is False

    def test_a_controlled_arm_that_jams_again_does_not_drain(self):
        broken = self._pilot_trace(1680, 9000)
        relapse = self._pilot_trace(8000, 9000)
        decision = rfe.pilot_decision(broken, relapse, 12000.0, 20.0)
        assert decision["controlled_drains_before_cap"] is True  # clear at the end
        stuck = self._pilot_trace(1500, 12001)
        decision = rfe.pilot_decision(broken, stuck, 12000.0, 20.0)
        assert decision["controlled_drains_before_cap"] is False
        assert "3,600" in decision["action"]

    def test_no_threshold_no_decision(self):
        decision = rfe.pilot_decision(self._pilot_trace(), self._pilot_trace(), 12000.0, None)
        assert decision["breaks_no_control_down"] is None

    def test_the_origin_fit_and_its_quadratic_term(self):
        linear = [(x, 2.0 * x) for x in (0.1, 0.25, 0.5, 0.75) for _ in range(3)]
        fit = rfe.origin_fit(linear)
        assert fit["slope"] == pytest.approx(2.0)
        curved = [(x, 2.0 * x + 8.0 * x * x + 0.01 * k) for x in (0.1, 0.25, 0.5, 0.75)
                  for k in range(3)]
        assert rfe.origin_fit(curved)["quadratic_resolved"] is True


@pytest.mark.parametrize("block", rfe.BLOCKS)
def test_one_600_s_episode_of_each_block_writes_a_line_a_trace_and_a_summary(block, tmp_path):
    # The cap checkpoints do not exist until Amendment 1's retraining has run; the smoke
    # only proves the entry point, so it reads the committed checkpoint.
    argv = ["--block", block, "--smoke", "--results-dir", str(tmp_path / "results"),
            "--traces-dir", str(tmp_path / "traces"),
            "--checkpoint", str(REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt")]
    if block == "A":
        argv.append("--allow-placeholder")
    assert rfe.main(argv) == 0
    lines = _jsonl(tmp_path / "results" / f"{block}.jsonl")
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "footer"
    episodes = [r for r in lines if r["type"] == "episode"]
    assert len(episodes) == 1 and episodes[0]["seed"] == 16
    assert episodes[0]["arrived"] > 0
    trace = json.loads(Path(episodes[0]["trace_file"]).read_text())["trace"]
    assert trace and trace[-1]["t"] >= 600.0
    summary = json.loads((tmp_path / "results" / f"{block}_summary.json").read_text())
    assert summary["episodes"] == 1 and summary["smoke"] is True
    assert episodes[0]["actuation"] == "max_speed"
    assert summary["controlled_episodes_with_collisions"] == 0
    assert "colliding_vehicles" in trace[-1] and "collision_teleports" in trace[-1]


def test_a_smoke_run_never_writes_into_the_real_results(tmp_path, monkeypatch):
    seen = {}

    def fake_run_block(block, specs, run_episode, *, results_dir, traces_dir, **kwargs):
        seen["results"], seen["traces"] = results_dir, traces_dir
        return {"status": "paused", "ran": [], "skipped": []}

    monkeypatch.setattr(rfe, "run_block", fake_run_block)
    rfe.main(["--block", "B", "--smoke"], run_episode_factory=lambda m, c: _stub([]))
    assert seen["results"] == REPO_ROOT / "outputs" / "feed" / "smoke" / "results"
    assert seen["traces"] == REPO_ROOT / "outputs" / "feed" / "smoke" / "traces"


def _write_trace(path: Path, trace: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trace": trace}))
    return str(path)


class TestSummariesOnSyntheticTraces:
    @staticmethod
    def _sustained(tmp_path, rng):
        """Sustained-scenario traces: free flow near 50 km/h, slow spells near 25."""
        episodes = []
        for arm in ("no_control", "here"):
            for seed in (16, 17):
                trace = [{"t": float(t), "count": [3] * 12, "held": 0, "pending": 0,
                          "arrived": t,
                          "speed_kmh": [float(rng.normal(25 if rng.random() < 0.25 else 52, 3))
                                        for _ in range(12)]}
                         for t in range(10, 2521, 10)]
                path = _write_trace(tmp_path / "t" / f"{arm}_mainz_{seed}.json", trace)
                episodes.append({"arm": arm, "seed": seed, "scenario": "mainz",
                                 "duration_s": 2500.0, "trace_file": path, "flow": 1.0})
        return episodes

    def test_block_d_uses_the_network_threshold_and_reports_every_rule(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(5)
        episodes = self._sustained(tmp_path, rng)
        for arm, jam_until in (("no_control", 2600), ("here", 1500)):
            for seed in (16, 17):
                trace = []
                for t in range(10, 3001, 10):
                    jam = 600 <= t < jam_until
                    speeds = [float(rng.normal(12 if jam else 50, 2)) for _ in range(12)]
                    trace.append({"t": float(t), "speed_kmh": speeds, "count": [3] * 12,
                                  "held": 0, "pending": 0, "arrived": t})
                path = _write_trace(tmp_path / "t" / f"{arm}_{seed}.json", trace)
                episodes.append({"arm": arm, "seed": seed, "scenario": "mainz_surge",
                                 "duration_s": 3000.0, "trace_file": path, "flow": 1.0})
        summary = rfe.summarize_d(episodes, results_dir=tmp_path)
        network = summary["network_threshold"]
        assert network["episodes"] == 4 and network["t_from"] == 300.0
        assert network["t_to"] == 2500.0
        assert network["pooled"]["refused"] is False
        v_network = network["pooled"]["v_jam"]
        assert 15.0 < v_network < 30.0
        entry = summary["scenarios"]["mainz_surge"]
        assert entry["t_0"] == 1200.0 and entry["cap_s"] == 3000.0
        assert entry["primary"] == "network/H120"
        assert entry["rules"]["network"] == v_network
        assert set(entry["drain"]) == {f"{r}/H{h}" for r in
                                       ("network", "network_per_segment", "scenario",
                                        "fixed_30") for h in (120, 60, 300)}
        written = json.loads((tmp_path / "drain_thresholds.json").read_text())
        assert written["network"]["pooled"]["v_jam"] == v_network
        assert written["scenarios"]["mainz_surge"] == entry["scenario_threshold"]
        # Jams (12 km/h) end at 2,600 s without control and 1,500 s with it; the
        # last jammed samples are 2,590 and 1,490, plus the 120 s hold and one sample.
        primary = entry["drain"]["network/H120"]
        assert primary["arms"]["no_control"]["per_seed"][0]["time_s"] == 2720 - 1200
        assert primary["arms"]["here"]["per_seed"][0]["time_s"] == 1620 - 1200
        assert primary["arms"]["here"]["fraction_drained"] == 1.0
        assert primary["paired"]["here"]["differences"] == [-1100.0, -1100.0]
        # The scenario's own pool is the jam itself: its knots bracket the jam mode,
        # so its threshold finds no jam (A4's reason for the network threshold).
        assert entry["drain"]["scenario/H120"]["arms"]["no_control"]["per_seed"][0][
            "time_s"] == 120.0
        # Each hold sensitivity is computed with its own H (mutant R5).
        assert entry["drain"]["network/H60"]["arms"]["here"]["per_seed"][0]["time_s"] == \
            1560 - 1200
        assert entry["drain"]["network/H300"]["arms"]["here"]["per_seed"][0]["time_s"] == \
            1800 - 1200

    def test_a_refused_network_fit_leaves_no_primary_and_says_why(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(6)
        episodes = []
        for arm in ("no_control", "here"):
            speeds = np.concatenate([rng.uniform(0, 3.5, 300), rng.uniform(3.5, 25, 400),
                                     rng.uniform(25, 40, 200)])
            trace = [{"t": float(10 * (i + 30)), "count": [3] * 12, "held": 0, "pending": 0,
                      "arrived": i, "speed_kmh": [float(speeds[i % len(speeds)])] * 12}
                     for i in range(220)]
            path = _write_trace(tmp_path / f"{arm}.json", trace)
            episodes.append({"arm": arm, "seed": 16, "scenario": "mainz", "duration_s": 2500.0,
                             "trace_file": path, "flow": 1.0})
        summary = rfe.summarize_d(episodes)
        assert summary["network_threshold"]["pooled"]["refused"] is True
        assert "refused" in summary["primary_unavailable_reason"]

    def test_block_e_estimates_the_certificate_and_checks_the_pulse_size(self, tmp_path):
        import numpy as np

        traces_dir, results_dir = tmp_path / "traces", tmp_path / "results"
        results_dir.mkdir()
        a = np.full(12, 0.02)
        B = np.zeros((12, 12))
        B[np.arange(11), np.arange(1, 12)] = 0.005
        times = list(range(1100, 1901, 10))
        episodes, baselines = [], []
        rng = np.random.default_rng(9)
        for seed in (16, 17, 18):
            base = [{"t": float(t), "density": [0.1] * 12} for t in times]
            _write_trace(traces_dir / "E" / f"baseline_p1_{seed}.json", base)
            baselines.append({"type": "episode", "key": f"E/baseline/p1/{seed}",
                              "arm": "baseline", "variant": "p1", "seed": seed,
                              "penetration": 1.0, "duration_s": 2500.0, "arrived": 2000,
                              "running": 900, "pending": 0, "held_at_entry": 50})
            for site in range(12):
                y = np.zeros(12)
                trace = []
                for t in times:
                    if t == 1200:
                        y[site] = 0.05
                    elif t > 1200:
                        y = y + 10.0 * (-a * y + B @ y) + rng.normal(0, 1e-5, 12)
                    trace.append({"t": float(t), "density": list(0.1 + y)})
                path = _write_trace(traces_dir / "E" / f"pulse60_{seed}_{site}.json", trace)
                episodes.append({"key": f"E/pulse60/p1_site{site}/{seed}", "arm": "pulse60",
                                 "variant": f"p1_site{site}", "seed": seed,
                                 "penetration": 1.0, "duration_s": 2500.0,
                                 "pulse": [site, 60, 1200.0, 60.0], "trace_file": path,
                                 "arrived": 2040, "running": 920, "pending": 0,
                                 "held_at_entry": 50})
        (results_dir / "E.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in baselines))
        # One pulsed run ends with a vehicle mid-teleport (accounted), one is one short.
        episodes[0].update(arrived=2039, in_teleport=1)
        episodes[1].update(arrived=2039, in_teleport=0)
        summary = rfe.summarize_e(episodes, traces_dir, results_dir)
        level = summary["levels"]["1"]
        assert level["seeds_used"] == [16, 17, 18]
        assert np.allclose(level["a_hat"], 0.02, rtol=0.1)
        assert level["rho_UCB"] >= level["rho_hat"]
        assert summary["p_min"]["p_min"] == 1.0
        assert summary["inserted_difference"] == {
            "checked": 36, "without_baseline": 0,
            "mismatched": [{"key": episodes[1]["key"], "difference": 59}]}
        floor = level["noise_floor"]
        assert floor["noise_samples"] == 3 * 12 * 12 * 11  # seeds x sites x segments x samples
        assert floor["peak_self_deviation_mean"] == pytest.approx(0.05)
        assert floor["resolved"] is True
        assert summary["regression_window"].startswith("from the injection's end")
        assert (results_dir / "certificate.json").exists()

    def test_block_a_records_the_lemma_condition_and_does_not_claim_zero_when_it_fails(self):
        feed = rfe.load_feed_parameters(ARTEFACT, None, allow_placeholder=True)
        context = rfe.Context()
        context.feed = feed
        episodes = [{"arm": "published", "variant": "-", "seed": s, "flow": 3700.0 + s}
                    for s in (16, 17)]
        episodes += [{"arm": "version_keyed", "variant": "D+0", "seed": s, "flow": 3690.0 + s,
                      "delta_mean": 0.01} for s in (16, 17)]
        summary = rfe.summarize_a(episodes, [], context)
        condition = summary["lemma_condition"]
        assert condition["g_min"] == feed["g_min"] and condition["P_q"] == 10.0
        # The drive brackets give J far above 16 s, so part (a) is not claimed.
        assert condition["P_q_le_g_min_minus_J_measured"] is False
        assert condition["zero_disagreement_claimed"] is False
        assert summary["rows"]["version_keyed/D+0"]["paired_vs_published"]["gain"] == -10.0
        assert summary["placeholder"] is True
        assert [row["D_minus_L_bar"] for row in summary["lemma"]] == [-20, -5, 0, 5, 10]


def test_every_summary_counts_the_controlled_episodes_that_collided(tmp_path):
    rows = [{"type": "episode", "key": f"C/penetration/p0.5/{seed}", "arm": "penetration",
             "variant": "p0.5", "seed": seed, "flow": 3000.0, "actuation": "max_speed",
             "colliding_vehicles": c, "collision_teleports": t}
            for seed, c, t in ((16, 0, 0), (17, 4, 2), (18, 0, 1))]
    (tmp_path / "C.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = rfe.summarize("C", tmp_path, tmp_path, rfe.Context())
    assert summary["controlled_episodes_with_collisions"] == 2
    assert summary["controlled_episodes"] == 3
    assert summary["colliding_vehicles_total"] == 4
    assert summary["collision_teleports_total"] == 3
    assert summary["actuations"] == ["max_speed"]


def test_no_control_episodes_are_not_counted_as_controlled():
    counts = rfe.collision_counts([
        {"arm": "no_control", "colliding_vehicles": 2, "collision_teleports": 1},
        {"arm": "here", "colliding_vehicles": 0, "collision_teleports": 0}])
    assert counts["controlled_episodes_with_collisions"] == 0
    assert counts["controlled_episodes"] == 1
    assert counts["actuations"] == ["unrecorded"]


def test_the_default_checkpoint_is_the_cap_one_and_a_missing_one_is_refused(tmp_path):
    assert rfe.DEFAULT_CHECKPOINT.name == "mainz_here_cap_best.pt"
    with pytest.raises(SystemExit, match="does not exist"):
        rfe.main(["--block", "C", "--smoke", "--results-dir", str(tmp_path / "r"),
                  "--traces-dir", str(tmp_path / "t"),
                  "--checkpoint", str(tmp_path / "missing_best.pt")])


class TestBlockB:
    def test_the_fit_uses_the_measured_fraction_and_the_least_squares_slope(self):
        """Mutants R1 (loss sign) and R4 (slope as sum y / sum x)."""
        import numpy as np

        references = {"here": {16: 3750.0, 17: 3760.0, 18: 3740.0}, "no_control": {}}
        measured = {("delta0.1", 16): 0.08, ("delta0.1", 17): 0.11, ("delta0.1", 18): 0.09,
                    ("delta0.5", 16): 0.42, ("delta0.5", 17): 0.47, ("delta0.5", 18): 0.40}
        losses = {("delta0.1", 16): 9.0, ("delta0.1", 17): 14.0, ("delta0.1", 18): 7.0,
                  ("delta0.5", 16): 45.0, ("delta0.5", 17): 52.0, ("delta0.5", 18): 38.0}
        episodes = [{"arm": "inject_delta", "variant": v, "seed": seed,
                     "flow": references["here"][seed] - losses[(v, seed)],
                     "previous_fraction_mean": measured[(v, seed)]}
                    for (v, seed) in measured]
        summary = rfe.summarize_b(episodes, references)
        x = np.array([measured[k] for k in measured])
        y = np.array([losses[k] for k in measured])
        assert sorted(summary["loss_points_delta"]) == sorted(zip(x.tolist(), y.tolist()))
        assert summary["fit_loss_vs_delta"]["slope"] == pytest.approx((x * y).sum() / (x * x).sum())
        assert summary["fit_loss_vs_delta"]["slope"] > 0  # flow falls, loss is positive
        assert summary["fit_loss_vs_delta"]["slope"] != pytest.approx(y.sum() / x.sum())

    def test_the_block_runs_at_or_below_one_half(self):
        assert rfe.DELTAS == (0.05, 0.1, 0.25, 0.5)

    def test_the_reference_is_the_cap_evaluation_and_its_identity_is_checked(self, tmp_path):
        path = tmp_path / "ref.jsonl"
        path.write_text(json.dumps({"type": "header", "config": {
            "actuation": "max_speed", "checkpoints": {"here": {"sha256": "abc"}}}}) + "\n"
            + json.dumps({"arm": "here", "seed": 16, "flow": 3600.0}) + "\n"
            + json.dumps({"arm": "no_control", "seed": 16, "flow": 3700.0}) + "\n")
        references = rfe.committed_references(path)
        assert references["here"] == {16: 3600.0} and references["no_control"] == {16: 3700.0}
        check = rfe._reference_check(references, [{"checkpoint": {"sha256": "abc"},
                                                   "config": {"actuation": "max_speed"}}])
        assert check["same_checkpoint"] is True and check["same_actuation"] is True
        check = rfe._reference_check(references, [{"checkpoint": {"sha256": "zzz"},
                                                   "config": {"actuation": "max_speed"}}])
        assert check["same_checkpoint"] is False
        assert rfe.REFERENCE_EVALUATION.name == "mainz_paired_seeds_cap.jsonl"


def test_the_regression_starts_when_the_injection_ends():
    record = {"pulse": [7, 60, 1200.0, 60.0]}
    assert rfe._regression_window(record) == (1260.0, 1800.0)
    assert rfe._pulse_start(record) == 1200.0
