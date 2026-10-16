"""The camera error model and the local estimates (plan sections 1.2 and 1.3, step S4).

Unit tests build synthetic snapshots and camera tables and need no SUMO. The sanity test at
the bottom runs a 600 s SUMO episode under the plan; run this file in its own process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.sumo.sensing import (  # noqa: E402
    CameraModel, CandidateTable, EdgeGeometry, Ego, LaneSnapshot, OracleSensor,
    TrendTracker, estimate,
)

W = 3.75
L = 4.5
GEOMETRY = {"A": EdgeGeometry(lanes=3, length_m=2000.0, widths=(W, W, W)),
            "B": EdgeGeometry(lanes=2, length_m=300.0, widths=(W, W))}
SUCCESSOR = {"A": "B"}


def model(p=1.0, sigma=0.0, sigma_v=0.0, bias=1.0, R_max=100.0, bins=12, conf=0.8):
    return CameraModel.from_tables(
        bin_m=10.0, p=np.full(bins, p), sigma_m=np.full(bins, sigma),
        sigma_v_mps=np.full(bins, sigma_v), confidences=[np.array([conf])] * bins,
        R_max=R_max, bias=bias)


def vehicle(vid, lane, front, speed=10.0, edge="A"):
    return {"id": vid, "edge": edge, "lane": lane, "front_m": front, "length_m": L,
            "speed_mps": speed}


def table_for(vehicles, egos, R=100.0):
    snap = LaneSnapshot.from_vehicles(vehicles, GEOMETRY, SUCCESSOR, t=0.0)
    return CandidateTable.build(snap, egos, R)


def unoccluded_scene():
    """One leader per lane, all in view and none hiding another."""
    return table_for([vehicle("ego", 1, 100.0), vehicle("own", 1, 100.0 + 30.0 + L, 8.0),
                      vehicle("left", 0, 100.0 + 20.0 + L, 9.0),
                      vehicle("right", 2, 100.0 + 25.0 + L, 7.0)],
                     [Ego("ego", "A", 1, 100.0)])


class TestCameraModel:
    def test_a_perfect_camera_sees_exactly_the_oracle_s_candidates(self):
        table = unoccluded_scene()
        det = model().observe(table, np.random.default_rng(0))
        assert det.detected.all()
        assert det.d_hat.tolist() == table.d.tolist()
        assert det.v_hat.tolist() == table.speed.tolist()
        est = estimate(table, det, model().R_eff)
        truth = OracleSensor(80.0).per_lane_counts(table)
        assert est.n_hat.tolist() == truth.tolist() == [[1, 1, 1]]

    def test_detection_share_at_p_one_half(self):
        table = unoccluded_scene()
        cam = model(p=0.5)
        rng = np.random.default_rng(1)
        share = np.mean([cam.observe(table, rng).detected.mean() for _ in range(10000 // 3 + 1)])
        assert share == pytest.approx(0.5, abs=0.02)

    def test_bias_scales_reported_distance_and_lowers_the_count_in_range(self):
        table = table_for([vehicle("ego", 1, 100.0), vehicle("adj", 0, 100.0 + 70.0 + L)],
                          [Ego("ego", "A", 1, 100.0)])
        plain = model().observe(table, np.random.default_rng(0))
        far = model(bias=1.25).observe(table, np.random.default_rng(0))
        assert far.d_hat[0] == pytest.approx(1.25 * plain.d_hat[0])
        assert estimate(table, plain, 80.0).n_hat[0, 0] == 1
        assert estimate(table, far, 80.0).n_hat[0, 0] == 0  # 87.5 m is beyond 80 m

    def test_nothing_is_detected_beyond_the_range(self):
        table = table_for([vehicle("ego", 1, 100.0), vehicle("own", 1, 100.0 + 95.0 + L)],
                          [Ego("ego", "A", 1, 100.0)], R=120.0)
        assert not model(R_max=90.0).observe(table, np.random.default_rng(0)).detected.any()

    def test_hidden_and_out_of_view_candidates_are_never_detected(self):
        table = table_for([vehicle("ego", 1, 100.0), vehicle("own1", 1, 100.0 + 10.0 + L),
                           vehicle("own2", 1, 100.0 + 20.0 + L),
                           vehicle("close_adj", 2, 100.0 + 5.0 + L)],
                          [Ego("ego", "A", 1, 100.0)])
        det = model().observe(table, np.random.default_rng(0))
        assert det.detected.tolist() == table.visible.tolist()
        assert not table.visible.all()

    def test_noise_follows_the_table(self):
        table = unoccluded_scene()
        cam = model(sigma=2.0, sigma_v=0.5)
        rng = np.random.default_rng(3)
        errors = np.concatenate([cam.observe(table, rng).d_hat - table.d for _ in range(4000)])
        speed_errors = np.concatenate([cam.observe(table, rng).v_hat - table.speed
                                       for _ in range(4000)])
        assert np.std(errors) == pytest.approx(2.0, rel=0.05)
        assert np.std(speed_errors) == pytest.approx(0.5, rel=0.05)

    def test_sigma_is_interpolated_between_bin_centres(self):
        cam = CameraModel.from_tables(bin_m=10.0, p=np.ones(3), sigma_m=np.array([1.0, 3.0, 5.0]),
                                      sigma_v_mps=np.zeros(3), confidences=[np.array([0.5])] * 3,
                                      R_max=30.0)
        assert cam.sigma_at(np.array([5.0, 10.0, 15.0, 25.0, 40.0])).tolist() == [
            1.0, 2.0, 3.0, 5.0, 5.0]

    def test_confidence_is_resampled_from_the_candidate_s_own_bin(self):
        cam = CameraModel.from_tables(bin_m=10.0, p=np.ones(4), sigma_m=np.zeros(4),
                                      sigma_v_mps=np.zeros(4),
                                      confidences=[np.array([0.9]), np.array([0.8]),
                                                   np.array([0.7]), np.array([0.6, 0.4])],
                                      R_max=40.0)
        table = table_for([vehicle("ego", 1, 100.0), vehicle("own", 1, 100.0 + 35.0 + L)],
                          [Ego("ego", "A", 1, 100.0)], R=50.0)
        draws = {float(cam.observe(table, np.random.default_rng(s)).conf[0]) for s in range(40)}
        assert draws == {0.6, 0.4}

    def test_a_fixed_seed_gives_identical_draws(self):
        table = unoccluded_scene()
        cam = model(p=0.6, sigma=1.0, sigma_v=0.3)
        a = cam.observe(table, np.random.default_rng(42))
        b = cam.observe(table, np.random.default_rng(42))
        assert a.detected.tolist() == b.detected.tolist()
        np.testing.assert_array_equal(a.d_hat, b.d_hat)
        np.testing.assert_array_equal(a.v_hat, b.v_hat)
        np.testing.assert_array_equal(a.conf, b.conf)
        c = cam.observe(table, np.random.default_rng(43))
        assert not (np.array_equal(a.d_hat, c.d_hat, equal_nan=True)
                    and a.detected.tolist() == c.detected.tolist())

    def test_switches_turn_each_error_off(self):
        table = table_for([vehicle("ego", 1, 100.0), vehicle("own1", 1, 100.0 + 10.0 + L),
                           vehicle("own2", 1, 100.0 + 20.0 + L)],
                          [Ego("ego", "A", 1, 100.0)])
        cam = model(p=0.0, sigma=3.0)
        assert not cam.observe(table, np.random.default_rng(0)).detected.any()
        lossless = cam.with_switches(detection_loss=False, noise=False)
        det = lossless.observe(table, np.random.default_rng(0))
        assert det.detected.tolist() == [True, False]
        unhidden = lossless.with_switches(occlusion=False).observe(table, np.random.default_rng(0))
        assert unhidden.detected.tolist() == [True, True]
        assert unhidden.d_hat.tolist() == table.d.tolist()

    def test_from_artefact_reads_the_sumo_table(self, tmp_path):
        artefact = {
            "schema": "camera_error_model/1", "bin_m": 10.0, "sumo_table": "0.20",
            "tables": {"0.20": {
                "bins_m": [[0.0, 10.0], [10.0, 20.0]],
                "detection": {"p": [0.9, 0.5], "opportunities": [100, 60]},
                "distance_noise": {"sigma_ema_m": [0.5, None], "sigma_raw_m": [0.7, 1.0]},
                "closing_rate_noise": {"sigma_v_mps": [0.2, 0.4]},
                "confidence": {"values": [[0.9, 0.8], [0.5]]},
                "R_max_m": 20.0}}}
        path = tmp_path / "camera.json"
        path.write_text(json.dumps(artefact))
        cam = CameraModel.from_artefact(path, bias=1.25)
        assert cam.R_max == 20.0 and cam.bias == 1.25 and cam.R_eff == 20.0
        assert cam.p_at(np.array([5.0, 15.0, 25.0])).tolist() == [0.9, 0.5, 0.0]
        # A bin without a noise estimate takes its neighbours' interpolation.
        assert cam.sigma_at(np.array([15.0])).tolist() == [0.5]
        assert len(cam.source["sha256"]) == 64


class TestEstimates:
    def test_own_lane_density_is_from_the_gap(self):
        table = table_for([vehicle("ego", 1, 100.0), vehicle("own", 1, 100.0 + 15.5 + L)],
                          [Ego("ego", "A", 1, 100.0)])
        est = estimate(table, model().observe(table, np.random.default_rng(0)), 80.0)
        assert est.g_hat[0] == pytest.approx(15.5)
        # Three visible lanes; the adjacent ones are empty.
        assert est.k_hat[0] == pytest.approx((4.5 / (15.5 + 4.5)) / 3.0)

    def test_adjacent_density_is_the_count_over_the_range(self):
        table = table_for([vehicle("ego", 0, 100.0), vehicle("a1", 1, 100.0 + 20.0 + L),
                           vehicle("a2", 1, 100.0 + 70.0 + L)], [Ego("ego", "A", 0, 100.0)])
        est = estimate(table, model().observe(table, np.random.default_rng(0)), 80.0)
        assert est.n_hat[0].tolist() == [0, 0, 2]
        assert est.lanes_visible[0].tolist() == [False, True, True]
        assert est.k_hat[0] == pytest.approx((0.0 + 2 * 4.5 / 80.0) / 2.0)
        assert np.isnan(est.g_hat[0])

    def test_adjacent_density_uses_the_model_s_own_range(self):
        # R_max 60 m makes R_eff 60 m: one adjacent detection at 30 m is 4.5/60 of its lane
        # (kills dividing by a fixed 80 m, M21).
        cam = model(R_max=60.0)
        assert cam.R_eff == 60.0
        table = table_for([vehicle("ego", 1, 100.0), vehicle("adj", 2, 100.0 + 30.0 + L)],
                          [Ego("ego", "A", 1, 100.0)])
        est = estimate(table, cam.observe(table, np.random.default_rng(0)), cam.R_eff)
        assert est.n_hat[0].tolist() == [0, 0, 1]
        assert est.k_hat[0] == pytest.approx((4.5 / 60.0) / 3.0)

    def test_speeds_confidence_and_count(self):
        table = unoccluded_scene()
        est = estimate(table, model(conf=0.7).observe(table, np.random.default_rng(0)), 80.0)
        assert est.own_speed[0] == pytest.approx(8.0)
        assert est.mean_speed[0] == pytest.approx(8.0)
        assert est.conf_mean[0] == pytest.approx(0.7)
        assert est.n_det[0] == 3

    def test_nothing_detected_gives_none(self):
        table = table_for([vehicle("ego", 1, 100.0)], [Ego("ego", "A", 1, 100.0)])
        est = estimate(table, model().observe(table, np.random.default_rng(0)), 80.0)
        assert est.k_hat[0] == 0.0 and est.n_det[0] == 0
        assert np.isnan(est.own_speed[0]) and np.isnan(est.mean_speed[0])
        assert np.isnan(est.conf_mean[0])

    def test_oracle_counts_every_candidate_at_its_true_distance(self):
        # Two own-lane leaders (one hidden) and one out-of-view adjacent vehicle at 6 m:
        # the oracle counts all three within 80 m over three lanes.
        table = table_for([vehicle("ego", 1, 100.0), vehicle("o1", 1, 100.0 + 10.0 + L),
                           vehicle("o2", 1, 100.0 + 20.0 + L), vehicle("a", 2, 100.0 + 6.0 + L),
                           vehicle("far", 1, 100.0 + 90.0 + L)],
                          [Ego("ego", "A", 1, 100.0)], R=200.0)
        assert OracleSensor(80.0).k_true(table)[0] == pytest.approx(3 * 4.5 / (80.0 * 3))
        assert OracleSensor(200.0).k_true(table)[0] == pytest.approx(4 * 4.5 / (200.0 * 3))


class TestTrend:
    def test_seven_points_rising_0_01_per_5_s_give_0_002_per_s(self):
        trend = TrendTracker()
        slopes = None
        for k in range(7):
            slopes = trend.update(["v"], 5.0 * k, np.array([0.1 + 0.01 * k]))
        assert slopes[0] == pytest.approx(0.002)

    def test_fewer_than_three_values_give_zero(self):
        trend = TrendTracker()
        assert trend.update(["v"], 0.0, np.array([0.1]))[0] == 0.0
        assert trend.update(["v"], 5.0, np.array([0.3]))[0] == 0.0
        assert trend.update(["v"], 10.0, np.array([0.5]))[0] == pytest.approx(0.04)

    def test_only_the_last_30_s_count(self):
        trend = TrendTracker()
        for k in range(10):
            value = 0.5 if k < 3 else 0.1 + 0.01 * k
            slopes = trend.update(["v"], 5.0 * k, np.array([value]))
        assert slopes[0] == pytest.approx(0.002)

    def test_after_a_gap_only_the_values_of_the_last_30_s_count(self):
        # Six values 0..25 s, a gap (a junction crossing), then 0.2 at 50 s: the 30 s window
        # holds 20, 25 and 50 s only (kills a 60 s window, M16, which would take all seven).
        trend = TrendTracker()
        for k in range(6):
            trend.update(["v"], 5.0 * k, np.array([0.1]))
        slope = trend.update(["v"], 50.0, np.array([0.2]))[0]
        expected = np.polyfit([20.0, 25.0, 50.0], [0.1, 0.1, 0.2], 1)[0]
        wider = np.polyfit([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0], [0.1] * 6 + [0.2], 1)[0]
        assert slope == pytest.approx(expected) and abs(expected - wider) > 1e-4

    def test_vehicles_are_tracked_separately_and_nan_is_skipped(self):
        trend = TrendTracker()
        for k in range(5):
            slopes = trend.update(["a", "b"], 5.0 * k,
                                  np.array([0.01 * k, np.nan if k == 2 else -0.02 * k]))
        assert slopes.tolist() == pytest.approx([0.002, -0.004])


# ----------------------------------------------------------------- SUMO sanity


def test_camera_estimates_differ_from_the_truth_and_occlusion_lowers_them():
    import torch

    import evaluate_mainz_checkpoints as emc
    import train_mainz_src as tms
    from src.analysis.gains import load_routes
    from src.sumo.mainz import HERE_FEATURES
    from src.sumo.sensing import successors
    from src.rl.src_q import greedy_actions

    paths = tms.SCENARIOS["mainz"]
    model_q = emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                                  emc.CHECKPOINT_FEATURES["here"], 12)
    env = tms.make_env(paths, seed=1, duration_s=600.0, features=HERE_FEATURES,
                       step_length=0.5, gate_entries=True)
    succ = successors(load_routes(REPO_ROOT / paths["gated_routes"]))
    segment_edges = {e for s in env.segments for e in s}
    cam = CameraModel.from_tables(
        bin_m=10.0, p=np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        sigma_m=np.linspace(0.3, 6.0, 10), sigma_v_mps=np.linspace(0.2, 2.0, 10),
        confidences=[np.array([0.8, 0.6, 0.4])] * 10, R_max=100.0)
    no_occlusion = cam.with_switches(occlusion=False)
    oracle = OracleSensor(80.0)
    rng_a, rng_b = np.random.default_rng(5001), np.random.default_rng(5001)
    differs: dict[str, bool] = {}
    sensed: dict[str, int] = {}
    with_occ, without_occ = [], []
    try:
        state = env.reset()
        builder = LaneSnapshot.builder(env, succ)

        def on_tick(t):
            snap = builder.build(t)
            egos = snap.egos(env.controlled, segment_edges)
            table = CandidateTable.build(snap, egos, 100.0)
            est = estimate(table, cam.observe(table, rng_a), cam.R_eff)
            free = estimate(table, no_occlusion.observe(table, rng_b), cam.R_eff)
            truth = oracle.k_true(table)
            for k, vid in enumerate(table.ego_ids):
                sensed[vid] = sensed.get(vid, 0) + 1
                if truth[k] > 0:
                    differs[vid] = differs.get(vid, False) or bool(est.k_hat[k] != truth[k])
            with_occ.append(float(est.k_hat.mean()) if len(est.k_hat) else 0.0)
            without_occ.append(float(free.k_hat.mean()) if len(free.k_hat) else 0.0)

        while env.time_s() < 600.0:
            actions = greedy_actions(model_q, torch.tensor(state, dtype=torch.float)).tolist()
            env._command(actions)
            env.advance_ticks(60.0, tick_s=5.0, on_tick=on_tick)
            state = env.observe()
    finally:
        env.close()
    regular = [v for v, n in sensed.items() if n >= 6 and v in differs]
    assert len(regular) > 100
    assert all(differs[v] for v in regular)
    assert np.mean(without_occ) > np.mean(with_occ)
