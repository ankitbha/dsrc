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
        record = rfe.header("A", context, rfe.DEFAULT_CHECKPOINT, ["x"])
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
        assert sizes == {"A": 150, "B": 135, "C": 60, "D": 60, "E": 1080}


class TestSummaries:
    def test_the_pilot_rule(self):
        def trace(rows):
            return [{"t": t, "arrived": a, "pending": p, "held": h} for t, a, p, h in rows]

        broken = trace([(1500, 1000, 50, 0), (3000, 2400, 300, 0)])  # 3,360 veh/h, growing
        drains = trace([(1500, 1000, 0, 20), (2000, 1500, 0, 0)])
        assert rfe.pilot_decision(broken, drains, 12000.0)["action"] == "keep the scenario"
        healthy = trace([(1500, 1000, 50, 0), (3000, 2600, 10, 0)])
        assert "6,000" in rfe.pilot_decision(healthy, drains, 12000.0)["action"]
        stuck = trace([(1500, 1000, 0, 20), (2000, 1500, 3, 0)])
        assert "3,600" in rfe.pilot_decision(broken, stuck, 12000.0)["action"]

    def test_the_origin_fit_and_its_quadratic_term(self):
        linear = [(x, 2.0 * x) for x in (0.1, 0.25, 0.5, 0.75) for _ in range(3)]
        fit = rfe.origin_fit(linear)
        assert fit["slope"] == pytest.approx(2.0)
        curved = [(x, 2.0 * x + 8.0 * x * x + 0.01 * k) for x in (0.1, 0.25, 0.5, 0.75)
                  for k in range(3)]
        assert rfe.origin_fit(curved)["quadratic_resolved"] is True


@pytest.mark.parametrize("block", rfe.BLOCKS)
def test_one_600_s_episode_of_each_block_writes_a_line_a_trace_and_a_summary(block, tmp_path):
    argv = ["--block", block, "--smoke", "--results-dir", str(tmp_path / "results"),
            "--traces-dir", str(tmp_path / "traces")]
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
    def test_block_d_fits_thresholds_and_reports_drain_under_every_rule(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(5)
        episodes = []
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
        summary = rfe.summarize_d(episodes)
        entry = summary["scenarios"]["mainz_surge"]
        assert entry["t_0"] == 1200.0 and entry["cap_s"] == 3000.0
        assert set(entry["drain"]) == {f"{r}/H{h}" for r in ("pooled", "per_segment", "fixed_30")
                                       for h in (120, 60, 300)}
        assert entry["primary"] == "pooled/H120"
        pooled = entry["thresholds"]["pooled"]
        assert pooled["v_jam"] == (pooled["s1"] + pooled["s2"]) / 4.0
        # With the jam mode holding two thirds of the samples the fit's knees bracket
        # the jam mode itself (9.5 and 14.5 km/h), so the pooled threshold is 6 km/h
        # and finds no jam here; the fixed 30 km/h threshold does.
        fixed = entry["drain"]["fixed_30/H120"]
        # Jams end at 2,600 s without control and 1,500 s with it, plus the 120 s hold.
        assert fixed["arms"]["no_control"]["per_seed"][0]["time_s"] == 2600 + 120 - 1200
        assert fixed["arms"]["here"]["per_seed"][0]["time_s"] == 1500 + 120 - 1200
        assert fixed["arms"]["here"]["fraction_drained"] == 1.0
        assert fixed["paired"]["here"]["differences"] == [-1100.0, -1100.0]
        assert entry["drain"]["pooled/H120"]["arms"]["no_control"]["per_seed"][0][
            "time_s"] == 120.0

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
        summary = rfe.summarize_e(episodes, traces_dir, results_dir)
        level = summary["levels"]["1"]
        assert level["seeds_used"] == [16, 17, 18]
        assert np.allclose(level["a_hat"], 0.02, rtol=0.1)
        assert level["rho_UCB"] >= level["rho_hat"]
        assert summary["p_min"]["p_min"] == 1.0
        assert summary["inserted_difference"] == {"checked": 36, "mismatched": [],
                                                  "without_baseline": 0}
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
