"""The engines on the approach boundary (approach plan sections 2.4, 3 and 4, step S5).

Unit tests drive the engines against fakes and need no SUMO. The sanity tests at the
bottom run SUMO: the gated-boundary regression against two committed block M records
(about 70 s) and an 840 s B9 smoke of every block M arm on the approach boundary; run
this file in its own process.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.sumo.jam_episode as je  # noqa: E402
from src.rl import onset_predictor as op  # noqa: E402
from src.sumo.correction import LookaheadRule  # noqa: E402
from src.sumo.feed import FeedProcess, VersionKeyed  # noqa: E402
from src.sumo.feed_episode import Mode  # noqa: E402
from src.sumo.jam_episode import GATED, JamArm, JamEngine, approach_boundary  # noqa: E402
from src.sumo.mainz import SPEED_ACTION_FRACTIONS  # noqa: E402
from src.sumo.sensing import CandidateTable, EdgeGeometry, Ego, LaneSnapshot  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
META = json.loads((DATA / "mainz_approach.json").read_text())
MEASURED = REPO_ROOT / "results" / "jam" / "feed_process_afternoon.json"
SESSION = "afternoon_20261016_mainz"


class TestLookaheadChain:
    def test_the_chain_rule_equals_the_index_rule_on_the_twelve(self):
        rng = np.random.default_rng(0)
        down = [list(range(s + 1, 12)) for s in range(12)]
        for _ in range(1000):
            jam = rng.uniform(0.0, 10.0, size=12) * (rng.random(12) < 0.4)
            J_th = float(rng.choice([2.0, 4.0, 6.0]))
            rule = LookaheadRule(J_th)
            expected = rule.fractions(jam, np.arange(12)).tolist()
            assert [rule.fraction_chain(jam, down[s]) for s in range(12)] == expected

    def test_an_approach_vehicle_reads_its_own_chain(self):
        # Every approach super-segment below J_th, segment 5 (entries 140 and 182) above.
        jam = np.zeros(36)
        jam[5] = 2.0
        rule = LookaheadRule(2.0)
        index = {name: 12 + k for k, (name,) in enumerate(META["approach_segments"])}
        assert rule.fraction_chain(jam, META["downstream"][index["a140_2"]]) == 0.5
        assert rule.fraction_chain(jam, META["downstream"][index["a140_1"]]) == 0.75
        assert rule.fraction_chain(jam, META["downstream"][index["a140_0"]]) == 1.0
        jam[index["a140_2"]] = 5.0  # a queue on the next approach edge
        assert rule.fraction_chain(jam, META["downstream"][index["a140_1"]]) == 0.5
        assert rule.fraction_chain(jam, META["downstream"][index["a140_0"]]) == 0.75
        assert rule.fraction_chain(jam, []) == 1.0


class FakeEnv:
    """The approach tables of the committed metadata; `placement(t)` lists `(vehicle,
    edge, super-segment)`."""

    def __init__(self, placement, approach=True):
        self.placement = placement
        self.segments = [[f"e{i}"] for i in range(12)]
        self.approach_segments = [list(s) for s in META["approach_segments"]] if approach else []
        self.plan_row = list(META["plan_row"]) if approach else list(range(12))
        self.downstream = ([list(c) for c in META["downstream"]] if approach
                           else [list(range(s + 1, 12)) for s in range(12)])
        self.t = 0.0
        self.writes = []
        self.schedule_size = 10
        self.arrived_total = 0
        self.held = 0
        self.jam = np.zeros(24, dtype=np.float32)

    @property
    def all_segments(self):
        return self.segments + self.approach_segments

    def controlled_on_segments(self):
        return list(self.placement(self.t))

    def controlled(self, vid):
        return True

    def observe(self):
        x = np.zeros((12, 5), dtype=np.float32)
        x[:, 1] = 60.0
        return x

    def observe_approaches(self):
        x = np.zeros((len(self.approach_segments), 5), dtype=np.float32)
        x[:, 1] = 60.0
        x[:, 2] = self.jam[:len(self.approach_segments)]
        return x

    def command_vehicles(self, mapping, edge_of=None):
        for vid, fraction in mapping.items():
            self.writes.append((self.t, vid, fraction))
        return len(mapping)

    def segment_state(self):
        n = len(self.all_segments)
        return {"speed_kmh": [50.0] * n, "density": [0.1] * n, "count": [1] * n}

    def held_at_entry(self):
        return self.held

    def pending(self):
        return 0

    def running(self):
        return len(self.placement(self.t))

    def position(self, vid):
        return int(vid[1:])

    def approach_report(self):
        return {"on_approach": 3, "queue_m": [0.0] * 7 + [120.0],
                "slow_near_insertion": [0] * 8}


FEED = FeedProcess.synthetic(60.0 * np.arange(1, 12), np.zeros(11))


def _engine(env, arm, policy, **kw):
    return JamEngine(env, policy, Mode("rule", VersionKeyed(10.0, 10.0), name="vk"), FEED,
                     seed=1, control_start_s=0.0, window_start_s=0.0, trace_every_s=10.0,
                     end_s=700.0, decision_instants=[], record_trace=False, arm=arm,
                     sense=False, **kw)


def _drive(engine, env, end):
    for t in range(1, end + 1):
        env.t = float(t)
        engine.on_tick(float(t))


def _by_segment(x):
    """Action 0 on segment 5, 1 on segment 8, 2 elsewhere."""
    actions = [2] * 12
    actions[5], actions[8] = 0, 1
    return actions


class TestPlanOnApproaches:
    def test_an_approach_vehicle_is_written_its_entry_segment_s_plan(self):
        # a182_0 is super-segment 27 (entry 182, j = 5), whose plan row is segment 5.
        env = FakeEnv(lambda t: [("v1", "a182_0", 27), ("v2", "a295_2", 35), ("v3", "e5", 5)])
        engine = _engine(env, JamArm("plan"), _by_segment)
        _drive(engine, env, 100)
        last = {v: f for _, v, f in env.writes}
        assert last == {"v1": 0.5, "v2": 0.75, "v3": 0.5}
        assert engine._fraction(1, 27) == SPEED_ACTION_FRACTIONS[engine.vectors[1][5]]
        assert engine.counter.above_base == 0

    def test_command_disagreement_reads_the_plan_row(self):
        env = FakeEnv(lambda t: [("v1", "a182_0", 27), ("v3", "e5", 5)])
        engine = _engine(env, JamArm("plan"), _by_segment)
        _drive(engine, env, 100)
        assert engine._measure(100.0, env.controlled_on_segments(), {"v1": 1, "v3": 1})[1] == 0.0

    def test_the_lookahead_reads_the_approach_feed(self):
        env = FakeEnv(lambda t: [("v1", "a140_1", 25), ("v2", "a140_0", 24), ("v3", "e4", 4)])
        env.jam[26 - 12] = 5.0  # a140_2 jammed in every snapshot
        engine = _engine(env, JamArm("lookahead", J_th=2.0), lambda x: [2] * 12)
        _drive(engine, env, 100)
        last = {v: f for _, v, f in env.writes}
        assert last == {"v1": 0.5, "v2": 0.75, "v3": 1.0}
        assert engine.approach_states[1].shape == (24, 5)

    def test_the_trace_has_36_columns_and_the_approach_report(self):
        env = FakeEnv(lambda t: [("v1", "a182_0", 27)])
        engine = _engine(env, JamArm("plan"), _by_segment, assert_no_gate=True)
        _drive(engine, env, 100)
        sample = engine.jam_trace[-1]
        assert len(sample["plan_mean"]) == len(sample["shown_mean"]) == 36
        assert sample["plan_mean"][27] == 0.5
        assert sample["on_approach"] == 3 and sample["queue_m"][7] == 120.0
        assert len(sample["slow_near_insertion"]) == 8

    def test_a_held_vehicle_on_the_approach_boundary_stops_the_episode(self):
        env = FakeEnv(lambda t: [("v1", "a182_0", 27)])
        env.held = 1
        engine = _engine(env, JamArm("plan"), _by_segment, assert_no_gate=True)
        with pytest.raises(AssertionError):
            _drive(engine, env, 20)

    def test_without_approaches_the_trace_keeps_its_gated_fields(self):
        env = FakeEnv(lambda t: [("v3", "e5", 5)], approach=False)
        engine = _engine(env, JamArm("plan"), _by_segment)
        _drive(engine, env, 100)
        sample = engine.jam_trace[-1]
        assert len(sample["plan_mean"]) == 12 and "queue_m" not in sample
        assert engine.approach_states == {}

    def test_the_snapshot_must_be_the_q_network_s_12_by_5(self):
        env = FakeEnv(lambda t: [])
        env.observe = lambda: np.zeros((36, 5), dtype=np.float32)
        engine = _engine(env, JamArm("plan"), lambda x: [2] * 12)
        with pytest.raises(ValueError):
            _drive(engine, env, 100)


def _gated_engine_features(engine, t, index, table, est, s_hat, v_slope, versions, segments):
    """`engine_features` as it was before the approach boundary (commit 5c17f86)."""
    if not hasattr(engine, "_remaining_tail"):
        engine._remaining_tail = op._remaining_tail(table.snapshot, engine._segment_of)
    tail = engine._remaining_tail
    geometry = table.snapshot.geometry
    rows = []
    for i in np.asarray(index, dtype=np.int64):
        k = int(versions[i])
        e = int(segments[i])
        nxt = min(e + 1, op.LAST_SEGMENT)
        x = engine.states[k]
        free_mps = float(x[e, 1]) / 3.6
        vector = engine.vectors[k]
        edge = table.ego_edges[i]
        g = est.g_hat[i]
        rows.append([
            est.k_hat[i], s_hat[i], est.n_hat[i, 0], est.n_hat[i, 1], est.n_hat[i, 2],
            float(table.lanes_visible[i].sum()),
            op.G_NONE_M if np.isnan(g) else g,
            free_mps if np.isnan(est.own_speed[i]) else est.own_speed[i],
            free_mps if np.isnan(est.mean_speed[i]) else est.mean_speed[i],
            v_slope[i],
            0.0 if np.isnan(est.conf_mean[i]) else est.conf_mean[i],
            float(est.n_det[i]),
            t - engine.feed.publish_of(k),
            SPEED_ACTION_FRACTIONS[vector[e]], SPEED_ACTION_FRACTIONS[vector[nxt]],
            x[e, 0], x[e, 2], x[nxt, 0], x[nxt, 2], x[e, 4], x[e, 3],
            geometry[edge].length_m - float(table.ego_x[i]) + tail.get(edge, 0.0),
        ])
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(op.FEATURES))


def _sensing(edges, egos, extra=()):
    from src.sumo.sensing import CameraModel, estimate

    geometry = {e: EdgeGeometry(2, 1000.0, (3.75, 3.75)) for e in edges}
    vehicles = [{"id": v, "edge": e, "lane": 0, "front_m": x, "length_m": 5.0, "speed_mps": 8.0}
                for v, e, x in list(egos) + list(extra)]
    successor = {a: b for a, b in zip(edges, edges[1:])}
    snap = LaneSnapshot.from_vehicles(vehicles, geometry, successor, t=100.0)
    table = CandidateTable.build(snap, [Ego(v, e, 0, x) for v, e, x in egos], 80.0)
    camera = CameraModel.from_tables(bin_m=10.0, p=np.ones(8), sigma_m=np.zeros(8),
                                     sigma_v_mps=np.zeros(8), confidences=[np.array([0.7])] * 8,
                                     R_max=80.0)
    est = estimate(table, camera.observe(table, np.random.default_rng(0)), camera.R_eff)
    return table, est


class TestLearnedFeatures:
    def test_on_the_twelve_the_features_are_the_gated_function_s(self):
        rng = np.random.default_rng(3)
        edges = [f"e{i}" for i in range(12)]
        egos = [(f"v{i}", f"e{i}", 300.0) for i in range(12)]
        table, est = _sensing(edges, egos, extra=[(f"h{i}", f"e{i}", 330.0) for i in range(12)])
        x = rng.uniform(0.0, 60.0, size=(12, 5)).astype(np.float32)
        vector = [int(a) for a in rng.integers(0, 3, size=12)]
        stub = dict(_segment_of={e: i for i, e in enumerate(edges)}, states={1: x},
                    vectors={1: vector}, feed=SimpleNamespace(publish_of=lambda k: 60.0))
        old = SimpleNamespace(**stub)
        new = SimpleNamespace(**stub, plan_row=list(range(12)),
                              downstream=[list(range(s + 1, 12)) for s in range(12)],
                              approach_states={})
        index = np.arange(12)
        versions = np.ones(12, dtype=np.int64)
        segments = np.array([int(e[1:]) for e in table.ego_edges])
        s_hat, v_slope = rng.normal(size=12), rng.normal(size=12)
        expected = _gated_engine_features(old, 100.0, index, table, est, s_hat, v_slope,
                                          versions, segments)
        got = op.engine_features(new, 100.0, index, table, est, s_hat, v_slope, versions,
                                 segments)
        assert np.array_equal(got, expected)
        # An engine without the tables (an older caller) reads the same.
        bare = SimpleNamespace(**stub)
        assert np.array_equal(op.engine_features(bare, 100.0, index, table, est, s_hat, v_slope,
                                                 versions, segments), expected)

    def test_an_approach_vehicle_reads_its_approach_feed_and_its_entry_s_plan(self):
        # Ego on a140_1 (super-segment 25): next is a140_2 (26), plan rows are segment 5.
        edges = ["a140_1", "a140_2"]
        table, est = _sensing(edges, [("v1", "a140_1", 400.0)])
        x = np.zeros((12, 5), dtype=np.float32)
        x[:, 1] = 60.0
        x[5] = [30.0, 60.0, 5.0, 3.0, 1.685]
        approach = np.zeros((24, 5), dtype=np.float32)
        approach[:, 1] = 60.0
        approach[:, 3:] = [3.0, 1.0]
        approach[25 - 12, [0, 2]] = [40.0, 3.3]
        approach[26 - 12, [0, 2]] = [10.0, 8.3]
        vector = [2] * 12
        vector[5] = 0
        engine = SimpleNamespace(_segment_of={"a140_1": 25, "a140_2": 26}, states={1: x},
                                 approach_states={1: approach}, vectors={1: vector},
                                 plan_row=list(META["plan_row"]),
                                 downstream=[list(c) for c in META["downstream"]],
                                 feed=SimpleNamespace(publish_of=lambda k: 60.0))
        row = op.engine_features(engine, 100.0, np.array([0]), table, est, np.zeros(1),
                                 np.zeros(1), np.array([1]), np.array([25]))[0]
        f = dict(zip(op.FEATURES, row.tolist()))
        assert f["plan_fraction_own"] == 0.5 and f["plan_fraction_next"] == 0.5
        assert f["feed_speed_own_kmh"] == 40.0 and f["feed_jam_factor_own"] == pytest.approx(3.3)
        assert f["feed_speed_next_kmh"] == 10.0 and f["feed_jam_factor_next"] == pytest.approx(8.3)
        assert f["segment_length_km"] == 1.0 and f["segment_lanes"] == 3.0
        assert f["remaining_in_segment_m"] == 600.0


def test_a_camera_near_the_end_of_an_approach_sees_across_onto_the_entry_link():
    from src.analysis.gains import load_routes
    from src.sumo.sensing import successors

    successor = successors(load_routes(DATA / "mainz_b9000_A_routes.rou.xml"))
    assert successor["a140_0"] == "a140_1" and successor["a140_2"] == "140"
    geometry = {"a140_2": EdgeGeometry(3, 1000.0, (3.75,) * 3),
                "140": EdgeGeometry(3, 161.0, (3.75,) * 3)}
    snap = LaneSnapshot.from_vehicles(
        [{"id": "ego", "edge": "a140_2", "lane": 1, "front_m": 980.0, "length_m": 5.0,
          "speed_mps": 8.0},
         {"id": "lead", "edge": "140", "lane": 1, "front_m": 30.0, "length_m": 5.0,
          "speed_mps": 4.0}], geometry, successor, t=10.0)
    table = CandidateTable.build(snap, [Ego("ego", "a140_2", 1, 980.0)], 80.0)
    assert [snap.ids[v] for v in table.vehicle] == ["lead"]
    assert table.d[0] == pytest.approx(45.0)


class TestBoundary:
    def test_the_gated_boundary_moves_nothing(self):
        assert GATED.W == 0.0 and not GATED.is_approach
        assert GATED.files("mainz_b9000") == "mainz_b9000"

    def test_the_approach_boundary_reads_w_from_its_metadata(self):
        boundary = approach_boundary()
        assert boundary.W == META["W"] == 240.0 and boundary.is_approach
        assert boundary.files("mainz_b9000") == "mainz_b9000_A"
        assert boundary.describe()["W"] == 240.0

    def test_an_uncalibrated_boundary_is_refused(self, tmp_path):
        path = tmp_path / "approach.json"
        path.write_text(json.dumps({**META, "calibrated": False}))
        with pytest.raises(SystemExit):
            approach_boundary(path)

    @pytest.mark.parametrize("arm", [JamArm("no_control"), JamArm("plan"),
                                     JamArm("rule", m_on=0.0)])
    def test_every_arm_runs_without_the_gate_on_the_full_route_file(self, arm):
        env = je._env_for(je.SCENARIOS["B9"], seed=1, duration_s=840.0, arm=arm,
                          penetration=0.5, step_length_s=0.5, boundary=approach_boundary())
        assert env.gate_entries is False
        assert env.route_file == DATA / "mainz_b9000_A.rou.xml"
        assert env.schedule_file == DATA / "mainz_b9000_A_schedule.json"
        assert env.net_file == DATA / "mainz_approach.net.xml"
        assert env.warmup_s == 300.0 + 240.0
        assert len(env.all_segments) == 36
        assert env.penetration == (1.0 if arm.arm == "no_control" else 0.5)

    def test_the_gated_env_is_unchanged(self):
        env = je._env_for(je.SCENARIOS["B9"], seed=1, duration_s=600.0, arm=JamArm("plan"),
                          penetration=1.0, step_length_s=0.5)
        assert env.gate_entries is True and env.warmup_s == 300.0
        assert env.route_file == DATA / "mainz_b9000_routes.rou.xml"
        assert env.approach_segments == []


@pytest.mark.parametrize("seed", range(16, 31))
def test_the_feed_replay_moves_by_exactly_w(seed):
    gated = FeedProcess.from_artefact(MEASURED, SESSION, seed, 60.0, 7380.0, allow_wrap=True)
    moved = FeedProcess.from_artefact(MEASURED, SESSION, seed, 300.0, 7620.0, allow_wrap=True)
    assert moved.count == gated.count
    assert np.array_equal(moved.publish_s - gated.publish_s, np.full(gated.count, 240.0))
    assert np.array_equal(moved.lag_s, gated.lag_s)
    assert moved.meta["start_index"] == gated.meta["start_index"]
    assert moved.meta["wraps"] == gated.meta["wraps"]


def test_the_feed_setting_starts_the_replay_where_it_is_told():
    setting = je.FeedSetting(MEASURED, SESSION, m=1.0)
    gated, _, _ = setting.process(16, 7380.0)
    moved, _, _ = setting.process(16, 7620.0, start_s=je.FEED_START_S + 240.0)
    assert np.array_equal(moved.publish_s, gated.publish_s + 240.0)


# ----------------------------------------------------------------- SUMO sanity


def _committed(key: str) -> dict:
    for line in (REPO_ROOT / "results" / "jam" / "M.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("key") == key:
            return record
    raise AssertionError(f"no committed record {key}")


@pytest.fixture(scope="module")
def model():
    import evaluate_mainz_checkpoints as emc

    return emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                               emc.CHECKPOINT_FEATURES["here"], 12)


@pytest.fixture(scope="module")
def gated_runner(model):
    import run_jam_experiments as rje

    context = rje.Context(check_max_speed=True)
    context.selection = rje.read_selection(
        REPO_ROOT / "results/jam/S_summary.json", feed_sha=None, camera_sha=None,
        predictor_sha=None)
    return rje, rje.episode_runner(model, context), context


REGRESSION_KEYS = ("flow", "arrived", "return", "space_mean_speed_kmh", "held_at_entry",
                   "pending", "running", "admitted", "colliding_vehicles",
                   "collision_teleports", "in_teleport", "writes", "writes_above_plan",
                   "writes_stepped_down", "corrected_vehicle_ticks", "active_share",
                   "active_changing_share", "max_speed_checked", "max_speed_mismatched",
                   "incident_candidate_rows", "sense_passes", "versions_published",
                   "switches_by_version", "jam", "jam_30kmh", "incidents")


@pytest.mark.parametrize("key, expected", [
    ("M/rule/m0/B9/16", {"flow": 3676.0, "arrived": 6685, "writes": 1215428}),
    ("M/no_control/-/B9/16", {"flow": 3509.1428571428573, "arrived": 6442}),
])
def test_the_gated_boundary_reproduces_committed_block_m_records(gated_runner, key, expected):
    rje, runner, context = gated_runner
    window = rje.oracle_window(context.camera_model)
    spec = next(s for s in rje.block_specs("M", selection=context.selection,
                                           oracle_window=window) if s.key == key)
    result = runner(spec)
    record = _committed(key)
    for name, value in expected.items():
        assert result[name] == value
    for name in REGRESSION_KEYS:
        assert result[name] == record[name], name
    if key.startswith("M/rule"):
        assert result["jam"]["jam_time_min"] == pytest.approx(5.6667, abs=1e-4)
        assert result["active_share"] == pytest.approx(0.309509, abs=1e-6)
    else:
        assert result["jam"]["jam_time_min"] == 185.5
        assert result["jam"]["pending_min"] == 6374.0


APPROACH_ARMS = [JamArm("no_control"), JamArm("lookahead", J_th=2.0), JamArm("plan"),
                 JamArm("take_the_slower"), JamArm("rule", m_on=0.0),
                 JamArm("learned", theta=0.75), JamArm("oracle", m_on=0.0, oracle_R=60.0)]


@pytest.fixture(scope="module")
def approach_smoke(model):
    from src.rl.onset_predictor import load
    from src.sumo.sensing import CameraModel

    camera = CameraModel.from_artefact(REPO_ROOT / "results/jam/camera_error_model.json")
    predictor = load(REPO_ROOT / "results/jam/onset_predictor.json")
    feed = je.FeedSetting(MEASURED, SESSION, m=1.0)
    out = {}
    for arm in APPROACH_ARMS:
        engines = []
        result = je.run_jam_episode(
            model, 1, "B9", arm, feed=feed, duration_s=600.0, window_start_s=300.0,
            camera=camera if arm.arm in ("no_control", "take_the_slower", "rule", "learned")
            else None, predictor=predictor, check_max_speed=True, write_log=True,
            boundary=approach_boundary(), on_engine=engines.append)
        out[arm.arm] = (result, engines[0])
    return out


@pytest.mark.parametrize("arm", [a.arm for a in APPROACH_ARMS])
def test_an_840_s_b9_episode_per_arm_on_the_approach_boundary(approach_smoke, arm):
    out, engine = approach_smoke[arm]
    assert out["boundary"]["name"] == "approach" and out["W"] == 240.0
    assert out["gate_entries"] is False and out["admitted"] == -1
    assert out["colliding_vehicles"] == 0 and out["collision_teleports"] == 0
    assert out["writes_above_plan"] == 0 and out["max_speed_mismatched"] == 0
    assert out["held_at_entry"] == 0 and out["pending"] == 0
    trace = out["trace"]
    assert trace[-1]["t"] == 840.0 and len(trace[-1]["count"]) == 36
    assert all(s["held"] == 0 and s["pending"] == 0 for s in trace)
    assert all(len(s["queue_m"]) == 8 for s in trace)
    # Sensing every 5 s from the warm-up's end at 540 s: 540, 545, ..., 840.
    assert out["sense_passes"] == 61
    if arm == "no_control":
        assert out["writes"] == 0
        return
    log = engine.writes_log
    approach_writes = [row for row in log if row[2].startswith("a")]
    assert approach_writes, "no approach vehicle was written"
    segment_of = engine._segment_of
    for vehicle, tick, edge, version, plan, shown in log:
        s = segment_of[edge]
        assert plan == SPEED_ACTION_FRACTIONS[engine.vectors[version][META["plan_row"][s]]]
        if arm == "lookahead":
            jam = np.concatenate([engine.states[version][:, 2],
                                  engine.approach_states[version][:, 2]])
            assert shown == engine.lookahead.fraction_chain(jam, META["downstream"][s])
        else:
            assert shown <= plan + 1e-12


@pytest.mark.parametrize("arm", ["take_the_slower", "rule", "learned", "no_control"])
def test_camera_egos_include_approach_vehicles(approach_smoke, arm):
    out, _ = approach_smoke[arm]
    assert any(v is not None for s in out["trace"] for v in s["k_hat_mean"][12:])


def test_the_oracle_senses_approach_vehicles(approach_smoke):
    out, _ = approach_smoke["oracle"]
    assert any(v is not None for s in out["trace"] for v in s["k_true_mean"][12:])


def test_feed_stamps_and_decisions_move_by_w(approach_smoke, model):
    out, engine = approach_smoke["plan"]
    gated = je.run_jam_episode(model, 1, "B9", JamArm("plan"),
                               feed=je.FeedSetting(MEASURED, SESSION, m=1.0), duration_s=600.0,
                               window_start_s=300.0, sense=False, record_trace=False)
    assert out["feed"]["replay"]["start_index"] == gated["feed"]["replay"]["start_index"]
    for version, entry in out["switches_by_version"].items():
        if version in gated["switches_by_version"]:
            assert entry["T_k"] == pytest.approx(
                gated["switches_by_version"][version]["T_k"] + 240.0, abs=1e-9)
    assert min(engine._decision_ticks) == 540 or not engine._decision_ticks
    assert engine.control_start_s == 540.0 and engine.window_start_s == 540.0
