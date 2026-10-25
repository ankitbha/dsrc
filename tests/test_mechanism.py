"""Queue release, camera gate, the camera alone, driver response and blocks N2 and D
(plan 2026-10-25).

Unit tests need no SUMO. The sanity tests at the bottom run seven 600 s B9 episodes on the
approach boundary (a few minutes); run this file in its own process.
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
from src.sumo.correction import RuleCorrection, discharging, rule_margin  # noqa: E402
from src.sumo.driver import DRIVERS, DriverResponse, DriverSetting, traits  # noqa: E402
from src.sumo.jam_episode import JamArm  # noqa: E402
from src.sumo.mainz import DENSITY_CRITICAL  # noqa: E402

K_C = DENSITY_CRITICAL
SELECTION = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                          best_params={"m_on": 0.0})
MECHANISM = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                          best_params={"m_on": 0.0}, gate_J=0.5, release_discharge=True,
                          camera_gate=0.8, mechanism_path="N2_summary.json")


class TestQueueRelease:
    def test_a_full_road_that_is_emptying_is_discharging(self):
        k = np.array([K_C, K_C + 0.1, K_C + 0.1, K_C - 0.01, K_C + 0.1])
        s = np.array([-0.001, -0.01, 0.0, -0.01, 0.01])
        assert discharging(k, s).tolist() == [True, True, False, False, False]

    def test_without_the_release_the_rule_is_unchanged(self):
        rng = np.random.default_rng(5)
        plain, explicit = RuleCorrection(0.0), RuleCorrection(0.0, release_discharge=False)
        ids = [f"v{i}" for i in range(20)]
        for tick in range(60):
            k = rng.uniform(0.0, 0.3, size=20)
            s = rng.normal(0.0, 0.005, size=20)
            assert (plain.update(ids, 5.0 * tick, k, s).tolist()
                    == explicit.update(ids, 5.0 * tick, k, s).tolist())

    def test_a_discharging_road_does_not_switch_the_rule_on(self):
        k, falling, rising = np.array([0.25]), np.array([-0.002]), np.array([0.002])
        assert rule_margin(k, falling)[0] < 0.0
        released = RuleCorrection(0.0, release_discharge=True)
        assert released.update(["a"], 0.0, k, falling).tolist() == [False]
        assert released.update(["a"], 5.0, k, rising).tolist() == [True]
        # The same road read by the rule without the release switches on at once.
        assert RuleCorrection(0.0).update(["a"], 0.0, k, falling).tolist() == [True]

    def test_a_vehicle_on_is_released_through_the_hysteresis_once_the_queue_empties(self):
        # On at 0 s while the road fills; from 5 s it is full and emptying. The hysteresis
        # keeps it on for 30 s, so it is on at 25 s and off at 30 s. Without the release it
        # stays on, because the margin stays below zero.
        k = np.array([0.25])
        rule, plain = (RuleCorrection(0.0, release_discharge=True), RuleCorrection(0.0))
        for r in (rule, plain):
            assert r.update(["a"], 0.0, k, np.array([0.002])).tolist() == [True]
        states = [rule.update(["a"], t, k, np.array([-0.002]))[0]
                  for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)]
        assert states == [True, True, True, True, True, False]
        assert all(plain.update(["a"], t, k, np.array([-0.002]))[0]
                   for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0))


class TestCameraGate:
    def test_the_vehicle_s_own_occupancy_opens_a_closed_feed_gate(self):
        k, s = np.array([0.8 * K_C + 0.001, 0.8 * K_C - 0.02]), np.array([0.01, 0.01])
        assert (rule_margin(k, s) < 0.0).all()
        shut = np.array([False, False])
        gated = RuleCorrection(0.0, camera_gate=0.8)
        assert gated.update(["a", "b"], 0.0, k, s, allow=shut).tolist() == [True, False]
        assert RuleCorrection(0.0).update(["a", "b"], 0.0, k, s, allow=shut).tolist() == \
            [False, False]

    def test_without_a_feed_gate_the_camera_gate_does_nothing(self):
        rng = np.random.default_rng(9)
        plain, camera = RuleCorrection(0.0), RuleCorrection(0.0, camera_gate=1.0)
        ids = [f"v{i}" for i in range(15)]
        for tick in range(40):
            k = rng.uniform(0.0, 0.3, size=15)
            s = rng.normal(0.0, 0.005, size=15)
            assert (plain.update(ids, 5.0 * tick, k, s).tolist()
                    == camera.update(ids, 5.0 * tick, k, s).tolist())


class TestArm:
    def test_the_options_belong_to_the_rule_and_the_oracle(self):
        assert JamArm("rule", m_on=0.0, release_discharge=True).release_discharge
        assert JamArm("oracle", m_on=0.0, gate_J=0.5, camera_gate=1.0).camera_gate == 1.0
        for arm, params in (("plan", {}), ("lookahead", {"J_th": 2.0}),
                            ("learned", {"theta": 0.75}), ("take_the_slower", {})):
            for extra in ({"release_discharge": True}, {"no_plan": True}):
                with pytest.raises(ValueError):
                    JamArm(arm, **params, **extra)

    def test_a_camera_gate_needs_a_feed_gate_and_no_plan_has_none(self):
        with pytest.raises(ValueError):
            JamArm("rule", m_on=0.0, camera_gate=1.0)
        with pytest.raises(ValueError):
            JamArm("rule", m_on=0.0, no_plan=True, gate_J=0.5)

    def test_the_default_arm_describes_itself_as_before(self):
        assert JamArm("rule", m_on=0.0).describe() == {
            "arm": "rule", "J_th": None, "m_on": 0.0, "tau_l": 10.0, "theta": None,
            "oracle_R": JamArm("rule", m_on=0.0).oracle_R, "refresh": True, "gate_J": None,
            "sensor": "camera"}
        assert JamArm("rule", m_on=0.0, release_discharge=True).describe()["release_discharge"]
        assert RuleCorrection(0.0).describe() == {
            "kind": "rule", "m_on": 0.0, "tau_l": 10.0, "k_c": K_C, "release_offset": 0.05,
            "min_on_s": 30.0, "release_ticks": 2}


class TestVariants:
    def test_earlier_variants_keep_their_names(self):
        assert rje._variant("rule", {"m_on": 0.0}) == "m0"
        assert rje._variant("rule", {"m_on": 0.0, "gate_J": 0.5}) == "m0_g0.5"
        assert rje._variant("oracle", {"m_on": 0.0, "tau_l": 10.0, "oracle_R": 60.0}) == \
            "tau10_m0_R60"

    def test_the_mechanism_is_named_in_a_fixed_order(self):
        assert rje._variant("rule", {"m_on": 0.0, "camera_gate": 0.8, "gate_J": 0.5,
                                     "release_discharge": True}) == "m0_r_g0.5_c0.8"
        assert rje._variant("rule", {"m_on": 0.0, "release_discharge": True,
                                     "no_plan": True}) == "m0_r_local"
        assert rje._variant("oracle", {"m_on": 0.0, "tau_l": 10.0, "oracle_R": 60.0,
                                       "gate_J": 0.5, "camera_gate": 1.0}) == \
            "tau10_m0_R60_g0.5_c1"


class TestBlocks:
    def test_block_n2_runs_the_grid_and_its_references_on_the_selection_seeds(self):
        specs = rje.block_specs("N2", selection=SELECTION)
        variants = sorted({(s.arm, s.variant) for s in specs})
        assert variants == [("no_control", "-"), ("plan", "-"), ("rule", "m0_g0.5_c0.8"),
                            ("rule", "m0_g0.5_c1"), ("rule", "m0_r"), ("rule", "m0_r_g0.5"),
                            ("rule", "m0_r_g0.5_c0.8"), ("rule", "m0_r_g0.5_c1")]
        assert len(specs) == 8 * 6 * 5
        assert {s.seed for s in specs} == set(range(11, 16))
        assert {s.duration_s for s in specs} == {3600.0}
        arm = next(s for s in specs if s.variant == "m0_r_g0.5_c0.8").jam_arm()
        assert (arm.release_discharge, arm.gate_J, arm.camera_gate) == (True, 0.5, 0.8)
        rje.check_block_boundary("N2", "approach")
        with pytest.raises(SystemExit):
            rje.check_block_boundary("N2", "gated")

    def test_block_m_adds_the_system_its_oracle_and_the_camera_alone(self):
        specs = rje.block_specs("M", selection=MECHANISM, oracle_window=60.0)
        variants = sorted({(s.arm, s.variant) for s in specs if s.arm in ("rule", "oracle")})
        assert variants == [("oracle", "tau10_m0_R200"), ("oracle", "tau10_m0_R60"),
                            ("oracle", "tau10_m0_R60_r_g0.5_c0.8"), ("rule", "m0"),
                            ("rule", "m0_r_g0.5_c0.8"), ("rule", "m0_r_local")]
        local = next(s for s in specs if s.variant == "m0_r_local").jam_arm()
        assert local.no_plan and local.release_discharge and local.gate_J is None
        # Every earlier block M key is still there, so a resume runs only the new arms.
        earlier = {s.key for s in rje.block_specs("M", selection=SELECTION, oracle_window=60.0)}
        assert earlier <= {s.key for s in specs}
        assert len(specs) - len(earlier) == 3 * 7 * 15

    def test_the_selected_mechanism_reaches_blocks_l_p_k_and_d(self):
        def variants(block):
            return sorted({s.variant for s in rje.block_specs(block, selection=MECHANISM,
                                                               oracle_window=60.0)})
        assert "m0_r_g0.5_c0.8_lag2" in variants("L")
        assert "m0_r_g0.5_c0.8_p0.5" in variants("P")
        assert "m0_r_g0.5_c0.8_b0.8" in variants("K")
        assert variants("D") == ["-_drvlax", "-_drvnominal", "m0_r_g0.5_c0.8_drvlax",
                                 "m0_r_g0.5_c0.8_drvnominal"]
        specs = rje.block_specs("D", selection=MECHANISM, oracle_window=60.0)
        assert len(specs) == 2 * 2 * 2 * 15
        assert {s.driver for s in specs} == {"nominal", "lax"}
        assert {s.scenario for s in specs} == {"B9", "I_exit"}


def _rec(arm, variant, scenario, seed, jam, flow=3400.0, block="N2", params=None,
         arrived=1000, collide=0):
    return {"arm": arm, "variant": variant, "scenario": scenario, "seed": seed, "flow": flow,
            "block": block, "arrived": arrived, "params": params or {},
            "jam": {"jam_time_min_900_3600": jam, "prevented": jam == 0},
            "storage": {"network_min": 1000.0}, "colliding_vehicles": collide,
            "collision_teleports": 0, "writes_above_plan": 0}


def _blocks(jams: dict[str, float], bg_flows: dict[str, float], differ=False):
    """Blocks N and N2 with every cell. `jams[label]` is each candidate's jam time on every
    trigger (no control 10, plan 5); `bg_flows[label]` its flow with no trigger (plan 3,400,
    no control 3,500, so the floor is 3,365)."""
    n, n2 = [], []
    for scenario in ("B5", "B7.5", "B9", "I_exit", "I_mid", "BG"):
        for seed in range(11, 16):
            bg = scenario == "BG"
            for block, out in (("N", n), ("N2", n2)):
                jam_nc = 0.0 if bg else 10.0
                out.append(_rec("no_control", "-", scenario, seed, jam_nc, 3500.0, block))
                plan_flow = 3400.0 + (1.0 if differ and block == "N2" and bg else 0.0)
                out.append(_rec("plan", "-", scenario, seed, 0.0 if bg else 5.0, plan_flow,
                                block))
            for g in (None,) + rje.GATE_GRID:
                params = {"m_on": 0.0, **({"gate_J": g} if g is not None else {})}
                label = rje._variant("rule", params)
                n.append(_rec("rule", label, scenario, seed, 0.0 if bg else jams.get(label, 6.0),
                              bg_flows.get(label, 3300.0) if bg else 3400.0, "N", params))
            for extra in rje.MECHANISM_GRID:
                params = {"m_on": 0.0, **extra}
                label = rje._variant("rule", params)
                n2.append(_rec("rule", label, scenario, seed,
                               0.0 if bg else jams.get(label, 6.0),
                               bg_flows.get(label, 3300.0) if bg else 3400.0, "N2", params))
    return n2, n


class TestSummarizeN2:
    def test_the_lowest_c_above_the_no_trigger_floor_wins(self):
        # m0 has the lowest jam but fails the floor; m0_r_g0.5 is the best that meets it.
        n2, n = _blocks({"m0": 1.0, "m0_r_g0.5": 2.0, "m0_g0.5": 3.0},
                        {"m0": 3300.0, "m0_r_g0.5": 3380.0, "m0_g0.5": 3390.0})
        out = rje.summarize_n2(n2, n)
        assert out["bg_floor"] == pytest.approx(3365.0)
        assert out["candidates"]["rule/m0"]["meets_bg_floor"] is False
        assert out["selected"]["variant"] == "m0_r_g0.5"
        assert out["selected"]["release_discharge"] is True
        assert out["selected"]["gate_J"] == 0.5 and out["selected"]["camera_gate"] is None

    def test_a_tie_goes_to_the_flow_with_no_trigger(self):
        n2, n = _blocks({"m0_r_g0.5": 2.0, "m0_g0.5": 2.05},
                        {"m0_r_g0.5": 3370.0, "m0_g0.5": 3395.0})
        assert rje.summarize_n2(n2, n)["selected"]["variant"] == "m0_g0.5"

    def test_no_candidate_above_the_floor_selects_nothing(self):
        n2, n = _blocks({}, {})
        out = rje.summarize_n2(n2, n)
        assert out["selected"] is None and "nothing selected" in out["decision"]

    def test_references_that_moved_since_block_n_select_nothing(self):
        n2, n = _blocks({"m0_g0.5": 2.0}, {"m0_g0.5": 3390.0}, differ=True)
        out = rje.summarize_n2(n2, n)
        assert out["selected"] is None and len(out["differ_from_block_n"]) == 5

    def test_a_missing_episode_selects_nothing(self):
        n2, n = _blocks({"m0_g0.5": 2.0}, {"m0_g0.5": 3390.0})
        out = rje.summarize_n2(n2[1:], n)
        assert out["selected"] is None and out["missing"]


class TestMechanismSelection:
    def _write(self, tmp_path, **changes):
        data = {"block": "N2", "seeds": [11, 12, 13, 14, 15],
                "selected": {"m_on": 0.0, "gate_J": 0.5, "release_discharge": True,
                             "camera_gate": 1.0}, **changes}
        path = tmp_path / "N2_summary.json"
        path.write_text(json.dumps(data))
        return path

    def test_the_selection_carries_the_mechanism(self, tmp_path):
        gated = rje.Selection(J_th=2.0, m_on=0.0, theta=0.75, best_arm="rule",
                              best_params={"m_on": 0.0}, gate_J=1.5, gate_path="N")
        out = rje.read_mechanism_selection(self._write(tmp_path), gated)
        assert (out.gate_J, out.release_discharge, out.camera_gate) == (0.5, True, 1.0)
        assert out.gate_path is None and out.mechanism_sha256
        assert out.mechanism_params() == {"gate_J": 0.5, "release_discharge": True,
                                          "camera_gate": 1.0}

    @pytest.mark.parametrize("changes", [{"block": "N"}, {"seeds": [11, 12]},
                                         {"selected": None}])
    def test_a_wrong_summary_is_refused(self, tmp_path, changes):
        with pytest.raises(SystemExit):
            rje.read_mechanism_selection(self._write(tmp_path, **changes), SELECTION)

    def test_block_d_needs_the_mechanism_selection(self, tmp_path):
        with pytest.raises(SystemExit):
            rje.main(["--boundary", "approach", "--block", "D", "--results-dir",
                      str(tmp_path / "r"), "--traces-dir", str(tmp_path / "t")],
                     run_episode_factory=lambda model, context: None)


SETTING = DriverSetting("test", 2.0, 0.0, 0.1, 0.0, 1.5)


class _Road:
    def __init__(self):
        self.caps: dict[str, list[tuple[float, float]]] = {}
        self.t = 0.0

    def apply(self, vehicle, mps):
        self.caps.setdefault(vehicle, []).append((self.t, mps))

    @staticmethod
    def speed(vehicle):
        return 30.0


def _run(driver, road, until, present=("a",)):
    for t in range(int(road.t) + 1, until + 1):
        road.t = float(t)
        driver.advance(float(t), set(present))


class TestDriver:
    def test_a_driver_s_traits_depend_on_the_seed_and_the_vehicle_only(self):
        a = traits(DRIVERS["nominal"], 16, "veh_12")
        assert traits(DRIVERS["nominal"], 16, "veh_12") == a
        assert traits(DRIVERS["nominal"], 17, "veh_12") != a
        assert traits(DRIVERS["nominal"], 16, "veh_13") != a
        draws = [traits(DRIVERS["lax"], 1, f"v{i}") for i in range(2000)]
        reactions = np.array([d[0] for d in draws])
        offsets = np.array([d[1] for d in draws])
        assert np.median(reactions) == pytest.approx(4.0, rel=0.05)
        assert offsets.mean() == pytest.approx(0.15, abs=0.01)
        assert offsets.min() >= -0.1 and offsets.max() <= 0.5

    def test_a_change_reaches_the_cap_after_the_reaction_time_with_the_offset(self):
        road = _Road()
        driver = DriverResponse(SETTING, 1, road.apply, road.speed)
        driver.submit(0.0, "a", 40.0)
        _run(driver, road, 1)
        assert "a" not in road.caps
        _run(driver, road, 2)
        # First cap: the goal 44 m/s is above the vehicle's 30 m/s, so it applies at once.
        assert road.caps["a"] == [(2.0, pytest.approx(44.0))]

    def test_a_lower_cap_is_approached_at_the_deceleration(self):
        road = _Road()
        driver = DriverResponse(SETTING, 1, road.apply, road.speed)
        driver.submit(0.0, "a", 20.0)  # goal 22 m/s, from the vehicle's 30 m/s
        _run(driver, road, 10)
        values = [v for _, v in road.caps["a"]]
        assert values == pytest.approx([28.5, 27.0, 25.5, 24.0, 22.5, 22.0])
        assert driver.cap["a"] == pytest.approx(22.0)

    def test_an_unchanged_advice_is_not_a_change_and_a_newer_one_replaces_a_pending_one(self):
        road = _Road()
        driver = DriverResponse(SETTING, 1, road.apply, road.speed)
        driver.submit(0.0, "a", 40.0)
        driver.submit(1.0, "a", 40.0)
        assert driver.changes == 1
        driver.submit(1.0, "a", 50.0)
        _run(driver, road, 2)
        assert "a" not in road.caps
        _run(driver, road, 3)
        assert road.caps["a"] == [(3.0, pytest.approx(55.0))]

    def test_a_vehicle_that_left_is_forgotten(self):
        road = _Road()
        driver = DriverResponse(SETTING, 1, road.apply, road.speed)
        driver.submit(0.0, "a", 20.0)
        _run(driver, road, 3)
        _run(driver, road, 6, present=())
        assert road.caps["a"][-1][0] == 3.0
        assert not driver.cap and not driver._goal


# ----------------------------------------------------------------- SUMO sanity


@pytest.fixture(scope="module")
def runs():
    import evaluate_mainz_checkpoints as emc

    import src.sumo.jam_episode as je
    from src.sumo.sensing import CameraModel

    model = emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                                emc.CHECKPOINT_FEATURES["here"], 12)
    camera = CameraModel.from_artefact(REPO_ROOT / "results/jam/camera_error_model.json")
    feed = je.FeedSetting(REPO_ROOT / "results/jam/feed_process_afternoon.json",
                          "afternoon_20261016_mainz", m=1.0)
    out = {}
    for name, arm, cam, driver in (
            ("rule", JamArm("rule", m_on=0.0), camera, None),
            ("camera_open", JamArm("rule", m_on=0.0, gate_J=10.5, camera_gate=0.0), camera, None),
            ("camera_shut", JamArm("rule", m_on=0.0, gate_J=10.5, camera_gate=100.0), camera,
             None),
            ("release", JamArm("rule", m_on=0.0, release_discharge=True), camera, None),
            ("local", JamArm("rule", m_on=0.0, no_plan=True), camera, None),
            ("plan", JamArm("plan"), None, None),
            ("rule_driver", JamArm("rule", m_on=0.0), camera, DRIVERS["lax"])):
        engines = []
        result = je.run_jam_episode(model, 1, "B9", arm, feed=feed, duration_s=600.0,
                                    window_start_s=300.0, camera=cam, check_max_speed=True,
                                    boundary=je.approach_boundary(), driver=driver,
                                    on_engine=engines.append)
        out[name] = (result, engines[0])
    return out


SAME = ("flow", "arrived", "writes", "writes_stepped_down", "corrected_vehicle_ticks",
        "active_share", "active_changing_share", "running", "colliding_vehicles")


def test_a_camera_gate_that_is_always_open_is_the_rule_without_a_gate(runs):
    rule, _ = runs["rule"]
    open_, _ = runs["camera_open"]
    assert rule["writes_stepped_down"] > 0
    for key in SAME:
        assert open_[key] == rule[key], key
    assert open_["trace"] == rule["trace"]


def test_a_camera_gate_that_never_opens_is_the_plan_alone(runs):
    shut, _ = runs["camera_shut"]
    plan, _ = runs["plan"]
    assert shut["writes_stepped_down"] == 0 and shut["active_share"] == 0.0
    for key in ("flow", "arrived", "writes", "running"):
        assert shut[key] == plan[key], key


def test_the_release_runs_cleanly_and_switches_vehicles_on(runs):
    out, _ = runs["release"]
    assert out["writes_stepped_down"] > 0
    assert out["colliding_vehicles"] == 0 and out["writes_above_plan"] == 0
    assert out["max_speed_mismatched"] == 0
    assert out["correction"]["release_discharge"] is True


def test_the_camera_alone_shows_the_limit_or_one_step_below_it(runs):
    out, _ = runs["local"]
    assert out["writes_stepped_down"] > 0 and out["writes_above_plan"] == 0
    plans = [p for s in out["trace"] for p in s["plan_mean"] if p is not None]
    shown = [p for s in out["trace"] for p in s["shown_mean"] if p is not None]
    assert plans and set(plans) == {1.0}
    assert min(shown) >= 0.75 - 1e-9 and min(shown) < 1.0


def test_the_driver_writes_the_caps_it_reports(runs):
    out, engine = runs["rule_driver"]
    rule, _ = runs["rule"]
    assert out["driver"]["name"] == "lax" and rule["driver"] is None
    assert out["driver"]["changes"] > 0 and out["driver"]["caps_checked"] > 0
    assert out["driver"]["caps_mismatched"] == 0
    assert out["colliding_vehicles"] == 0 and out["writes_above_plan"] == 0
    # The advice itself is unchanged by the driver: what the display shows never exceeds
    # the plan, and the driver's goal is the shown speed times its own offset.
    for vehicle, goal in engine.driver_response._goal.items():
        _, offset = engine.driver_response.traits_of(vehicle)
        assert goal / (1.0 + offset) <= 40.0
