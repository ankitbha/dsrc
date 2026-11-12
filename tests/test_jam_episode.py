"""The jam episode engine (plan section 5.1, step S6).

Unit tests drive `JamEngine.on_tick` by hand against a fake env, with synthetic lane
snapshots where a test needs sensing. The sanity tests at the bottom run SUMO; run this
file in its own process.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.sumo.jam_episode as je  # noqa: E402
from src.sumo.correction import step_down  # noqa: E402
from src.sumo.feed import FeedProcess, Synchronous, VersionKeyed  # noqa: E402
from src.sumo.feed_episode import FeedEngine, Mode  # noqa: E402
from src.sumo.jam_episode import FeedSetting, JamArm, JamEngine  # noqa: E402
from src.sumo.mainz import SPEED_ACTION_FRACTIONS  # noqa: E402
from src.sumo.sensing import CameraModel, EdgeGeometry, LaneSnapshot  # noqa: E402

MEASURED = REPO_ROOT / "results" / "jam" / "feed_process_afternoon.json"
MAINZ_SESSION = "afternoon_20261016_mainz"
PLACEHOLDER = REPO_ROOT / "results" / "feed" / "feed_process.json"


class FakeEnv:
    """Two one-edge segments; `placement(t)` lists `(vehicle, segment)` on them at `t`."""

    def __init__(self, placement):
        self.placement = placement
        self.segments = [["e0"], ["e1"]]
        self.t = 0.0
        self.writes: list[tuple[float, str, float]] = []
        self.schedule_size = 10
        self.arrived_total = 0
        self.colliding_vehicles = 0
        self.collision_teleports = 0

    def controlled_on_segments(self):
        return [(vid, f"e{seg}", seg) for vid, seg in self.placement(self.t)]

    def controlled(self, vid):
        return vid.startswith("v")

    def observe(self):
        x = np.zeros((2, 5), dtype=np.float32)
        x[:, 0] = self.t
        return x

    def command_vehicles(self, mapping, edge_of=None):
        for vid, fraction in mapping.items():
            self.writes.append((self.t, vid, fraction))
        return len(mapping)

    def segment_state(self):
        return {"speed_kmh": [50.0, 50.0], "density": [0.1, 0.1], "count": [1, 1]}

    def held_at_entry(self):
        return 0

    def pending(self):
        return 0

    def running(self):
        return len(self.placement(self.t))

    def position(self, vid):
        return int(vid[1:])

    def time_s(self):
        return self.t


def by_minute(x):
    """Action 2, 1, 0, 2, ... by minute of snapshot time, on both segments."""
    return [2 - int(x[0][0] // 60) % 3] * 2


FEED = FeedProcess.synthetic(60.0 * np.arange(1, 12), np.full(11, 5.0))


def _engine(env, arm, *, rule=None, feed=FEED, policy=by_minute, sense=False, **kw):
    mode = Mode("rule", rule or VersionKeyed(10.0, 20.0), name="vk")
    return JamEngine(env, policy, mode, feed, seed=1, control_start_s=0.0, window_start_s=0.0,
                     trace_every_s=10.0, end_s=700.0, decision_instants=[],
                     record_trace=False, arm=arm, sense=sense, **kw)


def _drive(engine, env, end):
    for t in range(1, end + 1):
        env.t = float(t)
        engine.on_tick(float(t))


class TestLagMultiplier:
    def test_two_doubles_every_lag_and_keeps_the_stamps(self):
        feed = FeedProcess.synthetic([60.0, 120.0, 180.0], [40.0, 55.0, 70.0])
        doubled = feed.with_lag_multiplier(2)
        assert doubled.publish_s.tolist() == feed.publish_s.tolist()
        assert doubled.lag_s.tolist() == [80.0, 110.0, 140.0]
        assert doubled.meta["lag_multiplier"] == 2.0

    def test_zero_is_no_lag_and_negative_is_refused(self):
        feed = FeedProcess.synthetic([60.0, 120.0], [40.0, 55.0])
        assert feed.with_lag_multiplier(0).available_s.tolist() == [60.0, 120.0]
        with pytest.raises(ValueError):
            feed.with_lag_multiplier(-1)


class TestFeedSetting:
    def test_measured_artefact_gives_D_from_its_session(self):
        params = FeedSetting(MEASURED, MAINZ_SESSION, m=1.0).parameters()
        data = json.loads(MEASURED.read_text())
        L_bar = data["sessions"][MAINZ_SESSION]["L_bar"]
        assert params["L_bar"] == L_bar and params["D"] == pytest.approx(L_bar + 10.0)
        assert params["placeholder"] is False and params["rule"] == "version_keyed"
        assert FeedSetting(MEASURED, MAINZ_SESSION, m=4.0).parameters()["D"] == pytest.approx(
            4 * L_bar + 10.0)

    def test_m_zero_is_the_synchronous_rule(self):
        feed, rule, params = FeedSetting(MEASURED, MAINZ_SESSION, m=0.0).process(1, 3780.0)
        assert isinstance(rule, Synchronous) and params["D"] is None
        assert np.all(feed.lag_s == 0.0)

    def test_the_placeholder_needs_permission(self):
        with pytest.raises(SystemExit):
            FeedSetting(PLACEHOLDER, "drive_run_20260908_170849").parameters()
        assert FeedSetting(PLACEHOLDER, "drive_run_20260908_170849",
                           allow_placeholder=True).parameters()["placeholder"] is True

    def test_a_7200_s_episode_wraps_the_measured_session_and_counts_it(self):
        feed, _, _ = FeedSetting(MEASURED, MAINZ_SESSION).process(16, 7380.0)
        assert feed.meta["wraps"] >= 1
        assert feed.publish_s[-1] <= 7380.0 and feed.publish_s[-1] > 7200.0
        # Every stamp keeps the session's recorded second of the minute.
        phase = json.loads(MEASURED.read_text())["sessions"][MAINZ_SESSION]["sequence"][0][2]
        assert set(np.round(feed.publish_s % 60.0, 6)) == {float(phase)}


class TestArms:
    def test_parameters_are_required(self):
        with pytest.raises(ValueError):
            JamArm("rule")
        with pytest.raises(ValueError):
            JamArm("lookahead")
        with pytest.raises(ValueError):
            JamArm("bogus")
        assert JamArm("oracle", m_on=0.1).sensor == "oracle"
        assert JamArm("rule", m_on=0.1).sensor == "camera"
        assert JamArm("plan").sensor == "none"

    def test_a_camera_arm_without_a_camera_is_refused(self):
        env = FakeEnv(lambda t: [])
        with pytest.raises(ValueError):
            _engine(env, JamArm("rule", m_on=0.1))


class TestWrites:
    def test_refresh_writes_every_listed_vehicle_every_5_s(self):
        env = FakeEnv(lambda t: [("v0", 0), ("v1", 1)])
        engine = _engine(env, JamArm("plan"))
        _drive(engine, env, 200)
        for vid in ("v0", "v1"):
            ticks = {t for t, v, _ in env.writes if v == vid}
            # Nothing before the first switch at T_1 + D = 80 s, then every 5 s.
            assert min(ticks) == 80.0
            assert {5.0 * k for k in range(16, 41)} <= ticks
            assert len([t for t, v, _ in env.writes if v == vid]) == len(ticks)
        assert engine.counter.above_base == 0

    def test_every_write_is_the_applied_version_s_plan(self):
        env = FakeEnv(lambda t: [("v0", 0)])
        engine = _engine(env, JamArm("plan"))
        _drive(engine, env, 300)
        for t, vid, fraction in env.writes:
            version = max(k for k in range(1, 12) if 60.0 * k + 20.0 <= t) if t >= 80 else 0
            if version:
                assert fraction == SPEED_ACTION_FRACTIONS[by_minute(
                    np.full((2, 5), 60.0 * version))[0]]

    def test_without_refresh_the_writes_are_feed_engine_s(self):
        placement = lambda t: [] if 140 <= t <= 142 else [("v0", 0), ("v1", 1)]
        env_jam, env_feed = FakeEnv(placement), FakeEnv(placement)
        jam = _engine(env_jam, JamArm("plan", refresh=False))
        feed = FeedEngine(env_feed, by_minute, Mode("rule", VersionKeyed(10.0, 20.0), name="vk"),
                          FEED, seed=1, control_start_s=0.0, window_start_s=0.0,
                          trace_every_s=10.0, end_s=700.0, decision_instants=[],
                          record_trace=False)
        _drive(jam, env_jam, 400)
        _drive(feed, env_feed, 400)
        assert env_jam.writes == env_feed.writes
        assert env_jam.writes

    def test_lookahead_writes_its_own_fraction(self):
        def jam_policy(x):
            return [2, 2]

        class JamEnv(FakeEnv):
            def observe(self):
                x = np.zeros((2, 5), dtype=np.float32)
                x[1, 2] = 7.0  # segment 1 jammed in every snapshot
                return x

        env = JamEnv(lambda t: [("v0", 0), ("v1", 1)])
        engine = _engine(env, JamArm("lookahead", J_th=4.0), policy=jam_policy)
        _drive(engine, env, 120)
        last = {v: f for _, v, f in env.writes}
        assert last == {"v0": 0.5, "v1": 1.0}
        assert engine.counter.above_base == 0


GEOMETRY = {"e0": EdgeGeometry(3, 500.0, (3.75,) * 3), "e1": EdgeGeometry(3, 500.0, (3.75,) * 3)}


def _dense_snapshot(t):
    """v0 on e0 lane 1 at 100 m with leaders every 7 m on its three lanes: occupancy
    near 0.5 within 80 m. v1 on e1 alone."""
    vehicles = [{"id": "v0", "edge": "e0", "lane": 1, "front_m": 100.0, "length_m": 4.5,
                 "speed_mps": 5.0},
                {"id": "v1", "edge": "e1", "lane": 1, "front_m": 100.0, "length_m": 4.5,
                 "speed_mps": 15.0}]
    for lane in range(3):
        for k in range(10):
            vehicles.append({"id": f"h{lane}_{k}", "edge": "e0", "lane": lane,
                             "front_m": 100.0 + 7.0 * (k + 1) + 4.5, "length_m": 4.5,
                             "speed_mps": 3.0})
    return LaneSnapshot.from_vehicles(vehicles, GEOMETRY, {"e0": "e1"}, t=t)


class TestCorrectedWrites:
    def test_the_oracle_steps_a_dense_vehicle_down_and_never_above_the_plan(self):
        env = FakeEnv(lambda t: [("v0", 0), ("v1", 1)])
        engine = _engine(env, JamArm("oracle", m_on=0.1), policy=lambda x: [2, 2],
                         snapshot_source=_dense_snapshot)
        _drive(engine, env, 200)
        late = [(v, f) for t, v, f in env.writes if t >= 100.0]
        assert {f for v, f in late if v == "v0"} == {step_down(1.0)}
        assert {f for v, f in late if v == "v1"} == {1.0}
        assert engine.counter.above_base == 0 and engine.counter.stepped_down > 0
        assert engine.jam_summary()["active_share"] == pytest.approx(0.5, abs=0.05)

    def test_a_max_in_place_of_min_is_counted_as_writes_above_the_plan(self, monkeypatch):
        monkeypatch.setattr(je, "shown_fraction", lambda plan, corr: max(plan, corr))
        env = FakeEnv(lambda t: [("v0", 0), ("v1", 1)])
        # The plan is 0.75 everywhere; an inactive correction is 1.0, above it.
        engine = _engine(env, JamArm("oracle", m_on=0.1), policy=lambda x: [1, 1],
                         snapshot_source=_dense_snapshot)
        _drive(engine, env, 200)
        assert engine.counter.above_base > 0

    def test_a_rule_correction_on_a_camera_that_sees_nothing_never_fires(self):
        blind = CameraModel.from_tables(bin_m=10.0, p=np.zeros(8), sigma_m=np.zeros(8),
                                        sigma_v_mps=np.zeros(8),
                                        confidences=[np.array([0.5])] * 8, R_max=80.0)
        env = FakeEnv(lambda t: [("v0", 0), ("v1", 1)])
        engine = _engine(env, JamArm("rule", m_on=0.1), policy=lambda x: [2, 2],
                         snapshot_source=_dense_snapshot, camera=blind)
        _drive(engine, env, 200)
        assert {f for t, v, f in env.writes if t >= 100.0} == {1.0}
        assert engine.counter.stepped_down == 0


def _leaders(count_at, gap0=6.0, step=6.0, calls=None):
    """A snapshot source: `v0` on lane 1 of the three-lane `e0` at 100 m, with
    `count_at(t)` leaders at gaps `gap0 + step j` on lanes `j % 3`. At 80 m over three
    visible lanes each leader adds 4.5 / 240 = 0.01875 of true occupancy."""

    def source(t):
        if calls is not None:
            calls.append(t)
        vehicles = [{"id": "v0", "edge": "e0", "lane": 1, "front_m": 100.0, "length_m": 4.5,
                     "speed_mps": 10.0}]
        for j in range(int(count_at(t))):
            vehicles.append({"id": f"h{j}", "edge": "e0", "lane": j % 3,
                             "front_m": 100.0 + gap0 + step * j + 4.5, "length_m": 4.5,
                             "speed_mps": 5.0})
        return LaneSnapshot.from_vehicles(vehicles, GEOMETRY, {"e0": "e1"}, t=t)

    return source


def _first_step_down(env, vehicle="v0"):
    return min((t for t, v, f in env.writes if v == vehicle and f < 1.0), default=None)


def test_the_episode_seed_sets_each_pair_s_distance_factor():
    # Amendment 3 (R2-4): with no other camera error, two episode seeds on one scene differ
    # only by the pair factors, so the engine must hand the camera its own seed.
    camera = CameraModel.from_tables(bin_m=10.0, p=np.ones(8), sigma_m=np.zeros(8),
                                     sigma_v_mps=np.zeros(8), confidences=[np.array([0.8])] * 8,
                                     R_max=80.0).with_pair_sigma(0.3)
    k_hat = {}
    for seed in (1, 2):
        env = FakeEnv(lambda t: [("v0", 0)])
        mode = Mode("rule", VersionKeyed(10.0, 20.0), name="vk")
        engine = JamEngine(env, lambda x: [2, 2], mode, FEED, seed=seed, control_start_s=0.0,
                           window_start_s=0.0, trace_every_s=10.0, end_s=700.0,
                           decision_instants=[], record_trace=False, arm=JamArm("plan"),
                           sense=True, camera=camera, snapshot_source=_leaders(lambda t: 3))
        _drive(engine, env, 10)
        k_hat[seed] = float(engine._sensed["k_hat"][0])
    assert k_hat[1] != k_hat[2]


class TestOracleInputs:
    def test_the_oracle_fires_through_its_trend_before_the_static_trigger(self):
        # One more leader every 5 s from 65 s: k_true = 0.01875 n, rising 0.00375 /s. With
        # the 10 s horizon the margin reaches 0.1 at 95 s (k = 0.131 < 0.160); without the
        # trend it would wait for k >= 0.178 x 0.9 at 105 s (kills s_true x 0, M18).
        env = FakeEnv(lambda t: [("v0", 0)])
        engine = _engine(env, JamArm("oracle", m_on=0.1), policy=lambda x: [2, 2],
                         snapshot_source=_leaders(lambda t: max(0, (t - 60) // 5)))
        _drive(engine, env, 200)
        assert _first_step_down(env) == 95.0

    def test_the_oracle_acts_on_the_true_occupancy_itself(self):
        # Constant scenes either side of the static trigger 0.1602: nine leaders (0.169)
        # fire at the first tick with a version, eight (0.150) never do (kills an oracle
        # fed half of k_true, M19).
        above, below = FakeEnv(lambda t: [("v0", 0)]), FakeEnv(lambda t: [("v0", 0)])
        for env, n in ((above, 9), (below, 8)):
            engine = _engine(env, JamArm("oracle", m_on=0.1), policy=lambda x: [2, 2],
                             snapshot_source=_leaders(lambda t, n=n: n))
            _drive(engine, env, 200)
        assert _first_step_down(above) == 80.0
        assert _first_step_down(below) is None

    def test_sensing_and_decisions_run_every_5_s(self):
        calls = []
        env = FakeEnv(lambda t: [("v0", 0)])
        engine = _engine(env, JamArm("oracle", m_on=0.1), policy=lambda x: [2, 2],
                         snapshot_source=_leaders(lambda t: 3, calls=calls))
        _drive(engine, env, 200)
        # From control start at 0 s; v0 has a version from 80 s, so it is decided on at
        # the 25 ticks 80, 85, ..., 200 (kills a 10 s cadence, M40).
        assert calls == [5.0 * k for k in range(1, 41)]
        assert engine.counter.vehicle_ticks == 25
        assert engine.jam_summary()["sense_passes"] == 40

    def test_an_active_tick_on_a_plan_already_at_0_5_changes_nothing_and_is_counted_apart(self):
        # Validation R1-7: the same dense scene under a plan of 0.5 and of 1.0. Both are
        # active on every decided tick; only the 1.0 plan's step-down changes the speed.
        slow, fast = FakeEnv(lambda t: [("v0", 0)]), FakeEnv(lambda t: [("v0", 0)])
        engines = {}
        for env, action in ((slow, 0), (fast, 2)):
            engines[action] = _engine(env, JamArm("oracle", m_on=0.1),
                                      policy=lambda x, a=action: [a, a],
                                      snapshot_source=_leaders(lambda t: 12))
            _drive(engines[action], env, 200)
        slow_summary, fast_summary = engines[0].jam_summary(), engines[2].jam_summary()
        assert slow_summary["active_share"] == fast_summary["active_share"] == 1.0
        assert slow_summary["active_changing_share"] == 0.0
        assert fast_summary["active_changing_share"] == 1.0
        assert slow_summary["writes_stepped_down"] == 0

    def test_oracle_wide_decides_on_its_200_m_occupancy(self):
        # 22 leaders between 85 and 190 m: none within 80 m, 0.165 over 200 m and three
        # lanes. The 200 m oracle fires, the 80 m one never (kills oracle_wide reading
        # the 80 m k_true, M42).
        wide, narrow = FakeEnv(lambda t: [("v0", 0)]), FakeEnv(lambda t: [("v0", 0)])
        for env, R in ((wide, 200.0), (narrow, 80.0)):
            engine = _engine(env, JamArm("oracle", m_on=0.1, oracle_R=R),
                             policy=lambda x: [2, 2],
                             snapshot_source=_leaders(lambda t: 22, gap0=85.0, step=5.0))
            _drive(engine, env, 200)
        assert _first_step_down(wide) == 80.0
        assert _first_step_down(narrow) is None


# ----------------------------------------------------------------- SUMO sanity


def _model():
    import evaluate_mainz_checkpoints as emc

    return emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                               emc.CHECKPOINT_FEATURES["here"], 12)


def _camera():
    return CameraModel.from_tables(
        bin_m=10.0, p=np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        sigma_m=np.linspace(0.3, 6.0, 10), sigma_v_mps=np.linspace(0.2, 2.0, 10),
        confidences=[np.array([0.8, 0.6, 0.4])] * 10, R_max=100.0)


def _committed_here_row(seed: int = 16) -> dict:
    path = REPO_ROOT / "results/evaluation/mainz_paired_seeds_deployed_cap.jsonl"
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("arm") == "here" and record.get("seed") == seed:
            return record
    raise AssertionError("no committed here row")


@pytest.fixture(scope="module")
def model():
    return _model()


def test_refresh_off_zero_lag_reproduces_the_committed_deployed_row(model):
    row = _committed_here_row()
    out = je.run_jam_episode(model, 16, "S45", JamArm("plan", refresh=False), feed=None,
                             duration_s=2500.0, aligned_feed=True, sense=False,
                             record_trace=False)
    assert out["arrived"] == row["arrived"] == 1936
    assert out["flow"] == row["flow"]
    assert out["held_at_entry"] == row["held_at_entry"]
    assert out["writes_above_plan"] == 0


def test_passive_sensing_leaves_flow_and_arrivals_unchanged(model):
    row = _committed_here_row()
    out = je.run_jam_episode(model, 16, "S45", JamArm("plan", refresh=False), feed=None,
                             duration_s=2500.0, aligned_feed=True, sense=True, camera=_camera())
    assert out["arrived"] == row["arrived"] and out["flow"] == row["flow"]
    assert out["sense_passes"] > 400
    assert any(v is not None for s in out["trace"] for v in s["k_hat_mean"])


SMOKE_ARMS = [JamArm("no_control"), JamArm("lookahead", J_th=4.0), JamArm("plan"),
              JamArm("take_the_slower"), JamArm("rule", m_on=0.1), JamArm("oracle", m_on=0.1)]


@pytest.fixture(scope="module")
def smoke(model):
    feed = FeedSetting(MEASURED, MAINZ_SESSION, m=1.0)
    return {arm.arm: je.run_jam_episode(model, 1, "B9", arm, feed=feed, duration_s=600.0,
                                        camera=_camera(), window_start_s=300.0,
                                        check_max_speed=True, write_log=True)
            for arm in SMOKE_ARMS}


@pytest.mark.parametrize("arm", [a.arm for a in SMOKE_ARMS])
def test_a_600_s_b9_episode_per_arm_has_no_collision_and_no_write_above_the_plan(smoke, arm):
    out = smoke[arm]
    assert out["colliding_vehicles"] == 0 and out["collision_teleports"] == 0
    assert out["writes_above_plan"] == 0
    assert out["max_speed_mismatched"] == 0
    assert out["arrived"] > 0
    # Sensing every 5 s from the warm-up's end: 300, 305, ..., 600 s.
    assert out["sense_passes"] == 61
    if arm != "no_control":
        assert out["feed"]["artefact_sha256"] and out["feed"]["placeholder"] is False
        assert out["feed"]["D"] == pytest.approx(out["feed"]["L_bar"] + 10.0)
        assert out["writes"] > 0
    else:
        assert out["writes"] == 0


def test_the_per_write_log_never_exceeds_the_plan(smoke):
    for arm in ("take_the_slower", "rule", "oracle"):
        log = smoke[arm]["write_log"]
        assert log and all(shown <= plan + 1e-12 for *_, plan, shown in log)


def test_the_oracle_steps_vehicles_down_through_set_max_speed(smoke):
    out = smoke["oracle"]
    assert out["writes_stepped_down"] > 0
    assert out["max_speed_checked"] == out["writes_stepped_down"]
    lower = [(s["shown_mean"][e], s["plan_mean"][e]) for s in out["trace"]
             for e in range(12) if s["shown_mean"][e] is not None]
    assert any(shown < plan for shown, plan in lower)


def test_switches_land_at_stamp_plus_D(smoke):
    out = smoke["plan"]
    D = out["feed"]["D"]
    checked = 0
    for version, entry in out["switches_by_version"].items():
        # A vehicle fetches every 10 s, so it switches at T_k + D or on receipt if later.
        assert entry["first_tick"] >= entry["T_k"] + D - 1.0
        assert entry["first_tick"] <= entry["T_k"] + D + 10.0 + 1.0
        checked += 1
    assert checked >= 3


def test_the_camera_arm_and_the_oracle_trace_differently(smoke):
    rule, oracle = smoke["rule"], smoke["oracle"]
    assert rule["sensor"] == "camera" and oracle["sensor"] == "oracle"
    assert [s["k_hat_mean"] for s in rule["trace"]] != [s["k_true_mean"] for s in rule["trace"]]
    assert rule["trace"] != oracle["trace"]


def test_the_exit_block_is_seen_as_a_candidate_and_its_metrics_compute(model):
    from src.analysis.jam_metrics import episode_metrics, load_v_jam

    # The block stands 100 m into exit_road, past the end of edge 218 (219.67 m), so its
    # back is more than 95 m from any vehicle still on a segment edge: outside the 80 m
    # window, inside oracle_wide's 200 m.
    feed = FeedSetting(MEASURED, MAINZ_SESSION, m=1.0)
    out = je.run_jam_episode(model, 1, "I_exit", JamArm("oracle", m_on=0.1, oracle_R=200.0),
                             feed=feed, duration_s=1300.0, check_max_speed=True)
    incident = out["incidents"][0]
    assert 900.0 <= incident["inserted_s"] <= 930.0
    assert incident["removed_s"] == incident["inserted_s"] + 300.0 and incident["standing_s"] == 300.0
    assert out["incident_candidate_rows"] > 0
    assert out["colliding_vehicles"] == 0 and out["collision_teleports"] == 0
    assert out["writes_above_plan"] == 0 and out["max_speed_mismatched"] == 0
    v_jam, _ = load_v_jam()
    metrics = episode_metrics(out["trace"], v_jam=v_jam, start_s=900.0, end_s=1200.0,
                              cap=1300.0)
    assert metrics["jam_time_min"] > 0.0


@pytest.mark.parametrize("m", [0.0, 2.0, 4.0])
def test_each_lag_multiplier_switches_at_its_own_delay(model, m):
    """Block L's levels (Amendment 1): `T_k + m L_bar + P_q`, and at m = 0 the stamp."""
    feed = FeedSetting(MEASURED, MAINZ_SESSION, m=m)
    out = je.run_jam_episode(model, 1, "B9", JamArm("plan"), feed=feed, duration_s=600.0,
                             window_start_s=300.0, sense=False)
    params = feed.parameters()
    assert out["feed"]["m"] == m
    assert out["feed"]["replay"]["lag_multiplier"] == m
    switches = out["switches_by_version"]
    assert switches
    for entry in switches.values():
        if m == 0.0:
            assert out["feed"]["rule"] == "synchronous"
            assert entry["first_tick"] == entry["last_tick"] == math.ceil(entry["T_k"] - 1e-9)
        else:
            D = params["D"]
            assert D == pytest.approx(m * params["L_bar"] + 10.0)
            assert entry["T_k"] + D - 1.0 <= entry["first_tick"] <= entry["T_k"] + D + 11.0
