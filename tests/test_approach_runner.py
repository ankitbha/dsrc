"""The runner on the approach boundary (approach plan sections 7 and 8, step S7).

Unit tests drive the runner with stubs and need no SUMO. The sanity test at the bottom
runs `--smoke --boundary approach --block Q M L P K` end to end (about a minute); run this
file in its own process.
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

SELECTION = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                          best_params={"m_on": 0.0})
COMMITTED_SELECTION = REPO_ROOT / "results" / "jam" / "S_summary.json"
DATA = REPO_ROOT / "data" / "mainz"


class TestBlockQ:
    def test_sixty_episodes_on_training_seeds_1_and_2(self):
        specs = rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
        assert len(specs) == 60
        assert {s.seed for s in specs} == {1, 2}
        assert {s.scenario for s in specs} == {"B9", "I_exit", "S45"}
        assert {(s.scenario, s.duration_s) for s in specs} == {
            ("B9", 7200.0), ("I_exit", 7200.0), ("S45", 2500.0)}

    def test_the_ten_settings(self):
        specs = rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
        settings = {(s.arm, s.variant, s.penetration, s.m) for s in specs}
        assert settings == {
            ("no_control", "-", 1.0, 1.0), ("lookahead", "J2", 1.0, 1.0),
            ("plan", "-", 1.0, 1.0), ("take_the_slower", "-", 1.0, 1.0),
            ("rule", "m0", 1.0, 1.0), ("learned", "th0.75", 1.0, 1.0),
            ("oracle", "tau10_m0_R60", 1.0, 1.0), ("plan", "-_p0.5", 0.5, 1.0),
            ("rule", "m0_p0.5", 0.5, 1.0), ("lookahead", "J2_lag0", 1.0, 0.0)}
        assert all(dict(s.params)["oracle_R"] == 60.0 for s in specs if s.arm == "oracle")

    def test_q_refuses_seed_3_and_m_refuses_seed_1(self):
        with pytest.raises(SystemExit):
            rje.block_specs("Q", seeds=[3], selection=SELECTION, oracle_window=60.0)
        with pytest.raises(SystemExit):
            rje.block_specs("M", seeds=[1], selection=SELECTION, oracle_window=60.0)

    def test_q_no_control_senses_passively_as_in_block_m(self):
        specs = rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
        assert all(rje.uses_camera(s) for s in specs if s.arm == "no_control")


class TestPenetrationSweep:
    def test_block_p_runs_the_given_shares_for_the_plan_and_the_best(self):
        specs = rje.block_specs("P", selection=SELECTION, oracle_window=60.0,
                                penetrations=(0.55, 0.9))
        assert len(specs) == 2 * 2 * 2 * 15
        assert {(s.arm, s.variant, s.penetration) for s in specs} == {
            ("plan", "-_p0.55", 0.55), ("rule", "m0_p0.55", 0.55),
            ("plan", "-_p0.9", 0.9), ("rule", "m0_p0.9", 0.9)}
        assert {s.seed for s in specs} == set(range(16, 31))
        assert {s.scenario for s in specs} == {"B9", "I_exit"}

    def test_the_gated_best_carries_its_gate_in_the_variant(self):
        gated = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                              best_params={"m_on": 0.0}, gate_J=0.5, gate_selected=False)
        specs = rje.block_specs("P", selection=gated, oracle_window=60.0,
                                penetrations=(0.6,))
        assert {s.variant for s in specs} == {"-_p0.6", "m0_g0.5_p0.6"}

    def test_without_shares_block_p_keeps_its_two(self):
        specs = rje.block_specs("P", selection=SELECTION, oracle_window=60.0)
        assert {s.penetration for s in specs} == set(rje.PENETRATIONS_P)

    @pytest.mark.parametrize("shares", [(1.0,), (0.0,), (0.6, 0.6)])
    def test_a_share_outside_zero_one_or_repeated_is_refused(self, shares):
        with pytest.raises(SystemExit):
            rje.block_specs("P", selection=SELECTION, oracle_window=60.0, penetrations=shares)

    def test_shares_are_refused_for_another_block(self):
        with pytest.raises(SystemExit):
            rje.block_specs("M", selection=SELECTION, oracle_window=60.0, penetrations=(0.6,))


class TestBoundaryRules:
    @pytest.mark.parametrize("block", ["G", "T", "S"])
    def test_selection_blocks_are_refused_on_the_approach_boundary(self, block):
        with pytest.raises(SystemExit):
            rje.check_block_boundary(block, "approach")
        rje.check_block_boundary(block, "gated")

    def test_q_is_refused_on_the_gated_boundary(self):
        with pytest.raises(SystemExit):
            rje.check_block_boundary("Q", "gated")
        rje.check_block_boundary("Q", "approach")
        rje.check_block_boundary("M", "approach")

    def test_main_refuses_g_and_q_on_the_wrong_boundary(self, tmp_path):
        with pytest.raises(SystemExit):
            rje.main(["--block", "G", "--boundary", "approach", "--results-dir",
                      str(tmp_path / "r"), "--traces-dir", str(tmp_path / "t")])
        with pytest.raises(SystemExit):
            rje.main(["--block", "Q", "--results-dir", str(tmp_path / "r"),
                      "--traces-dir", str(tmp_path / "t")])

    def test_another_selection_is_refused_on_the_approach_boundary(self, tmp_path):
        data = json.loads(COMMITTED_SELECTION.read_text())
        other = tmp_path / "S_summary.json"
        other.write_text(json.dumps(data, indent=2))  # the same content, another file
        assert hashlib.sha256(other.read_bytes()).hexdigest() != rje.APPROACH_SELECTION_SHA256
        with pytest.raises(SystemExit):
            rje.read_selection(other, feed_sha=None, camera_sha=None, predictor_sha=None,
                               required_sha=rje.APPROACH_SELECTION_SHA256)
        selection = rje.read_selection(COMMITTED_SELECTION, feed_sha=None, camera_sha=None,
                                       predictor_sha=None,
                                       required_sha=rje.APPROACH_SELECTION_SHA256)
        assert (selection.J_th, selection.m_on, selection.theta) == (2.0, 0.0, 0.75)

    def test_results_are_never_written_under_results_jam(self, tmp_path):
        with pytest.raises(SystemExit):
            rje.main(["--block", "Q", "--boundary", "approach", "--smoke",
                      "--results-dir", "results/jam", "--traces-dir", str(tmp_path / "t")])
        with pytest.raises(SystemExit):
            rje.main(["--block", "M", "--boundary", "approach",
                      "--results-dir", str(REPO_ROOT / "results" / "jam" / "x"),
                      "--traces-dir", str(tmp_path / "t")])

    def test_approach_directories_default_under_outputs_jam_approach(self):
        args = rje.build_parser().parse_args(["--block", "M", "--boundary", "approach"])
        rje.resolve_directories(args)
        assert args.results_dir == "outputs/jam_approach/run"
        assert args.traces_dir == "outputs/jam_approach/traces"
        smoke = rje.build_parser().parse_args(["--block", "M", "--boundary", "approach",
                                               "--smoke"])
        rje.resolve_directories(smoke)
        assert smoke.results_dir == "outputs/jam_approach/smoke/results"
        assert smoke.traces_dir == "outputs/jam_approach/smoke/traces"
        gated = rje.build_parser().parse_args(["--block", "M"])
        rje.resolve_directories(gated)
        assert gated.results_dir == "results/jam" and gated.traces_dir == "outputs/jam/traces"


def _header(**identity):
    base = {"type": "header", "checkpoint": {"sha256": "c"},
            "feed_artefact": {"artefact_sha256": "f"}, "camera_model": None, "predictor": None,
            "commit": None, "dirty": None}
    base.update(identity)
    return base


def _stub(spec):
    return {"flow": 1000.0, "arrived": 10, "trace": [{"t": 0.0}], "write_log": None,
            "samples": None, "colliding_vehicles": 0, "collision_teleports": 0,
            "writes_above_plan": 0, "sensor": "none",
            "jam": {"jam_time_min": 1.0, "prevented": True, "onset_s": None}}


class TestIdentity:
    def test_the_gated_identity_keeps_its_four_elements(self):
        assert rje.header_identity(_header()) == ("c", "f", None, None)

    def test_the_approach_identity_adds_the_approach_net(self):
        h = _header(approach={"net_sha256": "n"})
        assert rje.header_identity(h) == ("c", "f", None, None, "n")

    def test_resume_refuses_a_jsonl_that_ran_the_gated_boundary(self, tmp_path):
        specs = rje.block_specs("M", selection=SELECTION, oracle_window=60.0)[:2]
        rje.run_block("M", specs[:1], _stub, results_dir=tmp_path, traces_dir=tmp_path,
                      header_record=_header())
        with pytest.raises(SystemExit):
            rje.run_block("M", specs, _stub, results_dir=tmp_path, traces_dir=tmp_path,
                          header_record=_header(approach={"net_sha256": "n"}), resume=True)


class TestHeader:
    def test_the_approach_header_carries_the_boundary(self):
        from src.sumo.jam_episode import FeedSetting

        context = rje.Context(boundary="approach")
        context.feed = FeedSetting(rje.DEFAULT_FEED_ARTEFACT, rje.DEFAULT_FEED_SESSION).parameters()
        specs = rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
        h = rje.header("Q", context, specs, argv=["--block", "Q"])
        meta = json.loads((DATA / "mainz_approach.json").read_text())
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert h["boundary"] == "approach" and h["W"] == 240.0
        approach = h["approach"]
        assert approach["T_A"] == meta["T_A"]
        assert approach["length_m"] == 3000.0 and approach["segment_m"] == 1000.0
        assert approach["net_sha256"] == sha(DATA / "mainz_approach.net.xml")
        assert approach["approach_sha256"] == sha(DATA / "mainz_approach.json")
        assert approach["segments_sha256"] == sha(DATA / "mainz_approach_segments.json")
        assert approach["travel_sha256"] == sha(DATA / "mainz_approach_travel.json")
        files = h["scenario_files"]
        for key in ("mainz_b9000_A:routes", "mainz_b9000_A:gated_routes",
                    "mainz_b9000_A:schedule", "mainz_bg_A:schedule", "mainz_A:routes"):
            assert files[key]["sha256"] == sha(REPO_ROOT / files[key]["path"])
        assert rje.header_identity(h)[4] == approach["net_sha256"]

    def test_the_gated_header_is_unchanged(self):
        context = rje.Context()
        context.feed = {"artefact_sha256": "f"}
        h = rje.header("G", context, rje.block_specs("G"), argv=[])
        assert "boundary" not in h and "approach" not in h
        assert len(rje.header_identity(h)) == 4


def _trace36(cap, jam_columns=(), queue=0.0, pending=0, on=100):
    out = []
    t = 0.0
    while t <= cap + 1e-9:
        out.append({"t": t, "speed_kmh": [10.0 if s in jam_columns and t >= 1200.0 else 50.0
                                          for s in range(36)],
                    "density": [0.05] * 36, "count": [1] * 36, "held": 0, "pending": pending,
                    "running": 1000, "on_approach": on, "queue_m": [0.0] * 7 + [queue],
                    "slow_near_insertion": [0] * 8, "k_hat_mean": [None] * 36,
                    "k_true_mean": [None] * 36})
        t += 10.0
    return out


class TestEpisodeRunner:
    def test_the_approach_record_carries_three_parts_storage_and_flags(self, monkeypatch):
        import src.sumo.jam_episode as je

        seen = {}

        def fake_episode(model, seed, scenario, arm, **kwargs):
            seen["boundary"] = kwargs["boundary"]
            seen["duration_s"] = kwargs["duration_s"]
            return {"trace": _trace36(7440.0, jam_columns=(3, 20), queue=150.0)}

        monkeypatch.setattr(je, "run_jam_episode", fake_episode)
        context = rje.Context(boundary="approach", camera_model=None, predictor=None)
        spec = next(s for s in rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
                    if s.arm == "plan" and s.scenario == "B9")
        result = rje.episode_runner(None, context)(spec)
        assert seen["boundary"].W == 240.0 and seen["duration_s"] == 7200.0
        # 2026-10-20: the network is what is measured; all 36 and the approach 24 are
        # diagnostics. Column 3 is on the network, column 20 on an approach road.
        network, every, approach = result["jam"], result["jam_all"], result["jam_approach"]
        assert every["jam_time_min"] == pytest.approx(network["jam_time_min"]
                                                      + approach["jam_time_min"])
        assert network["jam_time_min"] > 0.0
        assert network["jam_time_min"] == pytest.approx(approach["jam_time_min"])
        assert network["columns"] == list(range(12)) and network["offset_s"] == 240.0
        assert every["columns"] is None and approach["columns"] == list(range(12, 36))
        assert network["start_s"] == 1140.0 and network["cap_s"] == 7440.0
        assert "jam_network" not in result
        assert result["jam_30kmh"]["columns"] == list(range(12))
        assert "jam_30kmh_all" in result and "jam_30kmh_approach" in result
        assert set(result["storage"]) == {"network_min", "approach_excess_min",
                                          "system_excess_min"}
        assert result["flags"]["queue_max_m"][7] == 150.0
        assert result["approach_short"] is False

    def test_a_pending_vehicle_flags_the_episode(self, monkeypatch):
        import src.sumo.jam_episode as je

        monkeypatch.setattr(je, "run_jam_episode",
                            lambda *a, **k: {"trace": _trace36(7440.0, pending=1)})
        context = rje.Context(boundary="approach", camera_model=None, predictor=None)
        spec = next(s for s in rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
                    if s.arm == "plan" and s.scenario == "B9")
        assert rje.episode_runner(None, context)(spec)["approach_short"] is True


class TestRunnerStorageAndFlags:
    """Validation R1-3 and R1-5: the runner's storage arguments, its flag limit and the
    leads baseline on the approach boundary."""

    def _spec(self):
        return next(s for s in rje.block_specs("Q", selection=SELECTION, oracle_window=60.0)
                    if s.arm == "plan" and s.scenario == "B9")

    def test_the_storage_measures_use_the_approach_schedule_and_the_shifted_window(
            self, monkeypatch):
        # Mutants RJ3 (approach excess on the gated window) and RJ4 (system excess without
        # N_ff). The free-flow occupancy of the B9 schedule differs between [900, 1,140) s
        # and (7,200, 7,440] s, so each mutant changes its number.
        import src.analysis.jam_metrics as jm
        import src.sumo.jam_episode as je

        trace = _trace36(7440.0, on=250)
        monkeypatch.setattr(je, "run_jam_episode", lambda *a, **k: {"trace": trace})
        context = rje.Context(boundary="approach", camera_model=None, predictor=None)
        storage = rje.episode_runner(None, context)(self._spec())["storage"]
        rows = json.loads((DATA / "mainz_b9000_A_schedule.json").read_text())
        T_A = json.loads((DATA / "mainz_approach.json").read_text())["T_A"]
        excess = jm.approach_excess_minutes(trace, rows, T_A, start_s=900.0, cap=7200.0,
                                            offset_s=240.0)
        system = jm.system_excess_minutes(trace, start_s=900.0, cap=7200.0, offset_s=240.0,
                                          rows=rows, travel_s=T_A)
        assert storage["approach_excess_min"] == pytest.approx(excess, rel=1e-12)
        assert storage["system_excess_min"] == pytest.approx(system, rel=1e-12)
        # 750 vehicles on the network (1,000 running, 250 on the approaches) at every 10 s
        # sample of [1,140, 7,440] s: 631 samples x 10 s x 750 / 60.
        assert storage["network_min"] == pytest.approx(631 * 10.0 * 750.0 / 60.0, rel=1e-12)
        unshifted = jm.approach_excess_minutes(trace, rows, T_A, start_s=900.0, cap=7200.0)
        without_n_ff = jm.system_excess_minutes(trace, start_s=900.0, cap=7200.0,
                                                offset_s=240.0)
        assert abs(unshifted - excess) > 100.0 and abs(without_n_ff - system) > 1000.0

    def test_a_queue_reaching_the_insertion_edge_flags_the_episode(self, monkeypatch):
        # Mutant RJ1: the limit is the approach length minus one edge, 2,000 m, not 3,000 m.
        import src.sumo.jam_episode as je

        monkeypatch.setattr(je, "run_jam_episode",
                            lambda *a, **k: {"trace": _trace36(7440.0, queue=2500.0)})
        context = rje.Context(boundary="approach", camera_model=None, predictor=None)
        result = rje.episode_runner(None, context)(self._spec())
        assert result["flags"]["limit_m"] == 2000.0
        assert result["approach_short"] is True

    def test_the_leads_baseline_moves_by_w(self, tmp_path):
        # Mutant RJ5. Segment 3 runs at 50 km/h until 900 s and at 30 km/h from 900 s, so
        # its mean is 50 km/h on [600, 900) s and 34 km/h on [840, 1,140) s. It runs at
        # 25 km/h from 1,200 s and jams at 10 km/h from 1,500 s. Against the shifted
        # baseline (80% of 34 = 27.2 km/h) the speed precursor starts at 1,200 s; against the
        # unshifted one (40 km/h) it would start at the clock start, 1,140 s.
        def speed(t):
            if t < 900.0:
                return 50.0
            if t < 1200.0:
                return 30.0
            if t < 1500.0:
                return 25.0
            return 10.0 if t < 1800.0 else 50.0

        trace = []
        t = 0.0
        while t <= 7440.0:
            trace.append({"t": t, "speed_kmh": [speed(t) if s == 3 else 50.0 for s in range(36)],
                          "density": [0.05] * 36, "count": [1] * 36, "held": 0, "pending": 0,
                          "k_hat_mean": [None] * 36, "k_true_mean": [None] * 36})
            t += 10.0
        path = tmp_path / "trace.json"
        path.write_text(json.dumps({"trace": trace}))
        record = {"arm": "no_control", "scenario": "B9", "seed": 1, "duration_s": 7200.0,
                  "trace_file": str(path)}
        out = rje.write_leads([record], tmp_path, offset_s=240.0, network_columns=list(range(12)))
        entry = out["episodes"][0]
        assert entry["onset"] == 1500.0 and entry["segment"] == 3
        assert entry["W_feed"] == 300.0
        assert entry["all"]["onset"] == 1500.0 and entry["all"]["W_feed"] == 300.0
        assert out["measured_columns"] == list(range(12))

    def test_the_leads_are_taken_on_the_network_not_an_approach_road(self, tmp_path):
        # 2026-10-20: an approach super-segment (20) jams at 1,300 s and network segment 3
        # at 1,600 s. The leads' onset is the network's; the all-segment onset is kept
        # under `all` only.
        trace = []
        t = 0.0
        while t <= 7440.0:
            speeds = [50.0] * 36
            if 1300.0 <= t < 2000.0:
                speeds[20] = 10.0
            if 1600.0 <= t < 2000.0:
                speeds[3] = 10.0
            trace.append({"t": t, "speed_kmh": speeds, "density": [0.05] * 36,
                          "count": [1] * 36, "held": 0, "pending": 0,
                          "k_hat_mean": [None] * 36, "k_true_mean": [None] * 36})
            t += 10.0
        path = tmp_path / "trace.json"
        path.write_text(json.dumps({"trace": trace}))
        record = {"arm": "no_control", "scenario": "B9", "seed": 1, "duration_s": 7200.0,
                  "trace_file": str(path)}
        entry = rje.write_leads([record], tmp_path, offset_s=240.0,
                                network_columns=list(range(12)))["episodes"][0]
        assert entry["onset"] == 1600.0 and entry["segment"] == 3
        assert entry["all"]["onset"] == 1300.0 and entry["all"]["segment"] == 20


def _record(arm, variant, scenario, seed, *, jam_net, jam_app, flagged=False, collide=0,
            system=1000.0, flow=None, params=None):
    def part(minutes, prevented):
        return {"jam_time_min": minutes, "jam_time_min_900_3600": minutes,
                "prevented": prevented, "onset_s": None if prevented else 1500.0,
                "start_s": 1140.0, "cap_s": 7440.0, "held_min": 0.0, "pending_min": 0.0,
                "drained": True, "drain_s": 300.0}
    return {"arm": arm, "variant": variant, "scenario": scenario, "seed": seed,
            "flow": 3000.0 + seed if flow is None else flow, "params": params or {},
            "jam": part(jam_net, jam_net == 0),
            "jam_all": part(jam_net + jam_app, jam_net + jam_app == 0),
            "jam_approach": part(jam_app, jam_app == 0),
            "storage": {"network_min": 900.0 + seed, "approach_excess_min": 50.0 * seed,
                        "system_excess_min": system + seed},
            "flags": {"queue_max_m": [0.0] * 7 + [100.0 * seed], "pending_max": 0,
                      "approach_short": flagged, "limit_m": 2000.0,
                      "slow_near_insertion_max": [0] * 8},
            "approach_short": flagged, "colliding_vehicles": collide, "collision_teleports": 0,
            "writes_above_plan": 0, "W": 240.0}


class TestSummaries:
    def _block(self, flag_seed=None, collide_seed=None):
        records = []
        for arm, variant in (("no_control", "-"), ("plan", "-"), ("rule", "m0")):
            for seed in (1, 2):
                records.append(_record(arm, variant, "B9", seed, jam_net=10.0, jam_app=4.0,
                                       flagged=(arm == "rule" and seed == flag_seed),
                                       collide=int(arm == "no_control" and seed == collide_seed)))
        return records

    def test_a_stub_block_summarises_three_parts_and_counts_flags(self):
        summary = rje.summarize_eval("Q", self._block(flag_seed=2))
        row = summary["rows"]["rule/m0/B9"]
        assert row["jam_time_min_mean"] == 10.0
        assert row["jam_all_time_min_mean"] == 14.0
        assert row["jam_approach_time_min_mean"] == 4.0
        assert row["network_min_mean"] == 901.5
        assert row["approach_excess_min_mean"] == 75.0
        assert row["system_excess_min_mean"] == 1001.5
        assert row["approach_short_episodes"] == 1
        assert row["queue_max_m"] == [0.0] * 7 + [200.0]
        assert row["by_seed"]["2"]["jam_time_min"] == 10.0
        assert row["by_seed"]["2"]["jam_all_time_min"] == 14.0
        assert row["by_seed"]["2"]["network_min"] == 902.0
        assert summary["approach_short_total"] == 1
        assert "gated boundary" in summary["thresholds_note"]
        assert row["paired_system_excess_vs_plan"]["mean"] == 0.0
        assert row["paired_network_min_vs_plan"]["mean"] == 0.0

    def test_the_q_rule_passes_only_without_flags_and_collisions(self):
        assert rje.qualification(self._block())["passes"] is True
        flagged = rje.qualification(self._block(flag_seed=1))
        assert flagged["passes"] is False and flagged["flagged"] == ["Q/rule/m0/B9/1"]
        collided = rje.qualification(self._block(collide_seed=2))
        assert collided["passes"] is False and collided["collisions"] == ["Q/no_control/-/B9/2"]

    def test_collisions_are_counted_over_every_arm(self):
        counts = rje.collision_counts(self._block(collide_seed=1))
        assert counts["episodes_with_collisions"] == 1
        assert counts["controlled_episodes_with_collisions"] == 0

    def test_a_gated_summary_has_no_approach_fields(self):
        records = [{k: v for k, v in r.items()
                    if k not in ("jam_all", "jam_approach", "storage", "flags",
                                 "approach_short", "W")} for r in self._block()]
        summary = rje.summarize_eval("M", records)
        assert "jam_all_time_min_mean" not in summary["rows"]["plan/-/B9"]
        assert "approach_short_total" not in summary


# ----------------------------------------------------------------- SUMO sanity


def test_the_approach_smoke_runs_every_block_end_to_end(tmp_path):
    results, traces = tmp_path / "results", tmp_path / "traces"
    assert rje.main(["--block", "Q", "M", "L", "P", "K", "--smoke", "--boundary", "approach",
                     "--results-dir", str(results), "--traces-dir", str(traces),
                     "--training-dir", str(tmp_path / "training")]) == 0
    arms = {"Q": 7, "M": 7, "L": 3, "P": 2, "K": 1}
    for block, count in arms.items():
        headers, episodes = rje.read_block(results, block)
        assert len(episodes) == count, block
        assert headers[0]["boundary"] == "approach" and headers[0]["W"] == 240.0
        assert headers[0]["selection"]["sha256"] == rje.APPROACH_SELECTION_SHA256
        for e in episodes:
            assert e["seed"] == 1 and e["W"] == 240.0 and e["gate_entries"] is False
            assert e["colliding_vehicles"] == 0 and e["collision_teleports"] == 0
            assert e["writes_above_plan"] == 0 and e["max_speed_mismatched"] == 0
            assert e["admitted"] == -1 and e["held_at_entry"] == 0
            assert e["flags"]["pending_max"] == 0 and e["approach_short"] is False
            assert e["jam"]["cap_s"] == 840.0 and e["jam"]["start_s"] == 1140.0
            trace = json.loads(Path(e["trace_file"]).read_text())["trace"]
            assert trace[-1]["t"] == 840.0 and len(trace[-1]["count"]) == 36
            assert all(s["held"] == 0 for s in trace)
        summary = json.loads((results / f"{block}_summary.json").read_text())
        assert summary["episodes"] == count and summary["smoke"] is True
        assert summary["episodes_with_collisions"] == 0
    q = json.loads((results / "Q_summary.json").read_text())
    assert q["qualification"]["passes"] is True
    assert (results / "leads.json").exists()


def test_the_approach_boundary_runs_the_committed_models():
    # Approach plan section 4.5 and attack point 9: nothing is re-selected or refitted.
    sha = lambda p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
    assert sha("results/jam/S_summary.json") == rje.APPROACH_SELECTION_SHA256
    assert sha("results/jam/onset_predictor.json").startswith("3b2256cd")
    assert sha("results/jam/camera_error_model.json").startswith("fdf527cd")
    assert sha("results/checkpoints/mainz_here_best.pt").startswith("941ac210")
    assert sha("results/jam/feed_process_afternoon.json").startswith("8f5430bf")
    assert rje.DEFAULT_PREDICTOR == REPO_ROOT / "results/jam/onset_predictor.json"
    assert rje.DEFAULT_CAMERA == REPO_ROOT / "results/jam/camera_error_model.json"
    assert rje.DEFAULT_CHECKPOINT == REPO_ROOT / "results/checkpoints/mainz_here_best.pt"
