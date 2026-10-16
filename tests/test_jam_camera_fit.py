"""The camera error model's tables (plan section 2.3, step S8).

Unit tests use synthetic tracks with planted error rates. The real smoke at the bottom fits
the 50-frame records of the S7 smoke (`outputs/jam/smoke/frames`) and checks that every
table is present; the values of a 50-frame fit are not interpreted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.analysis import camera_error as ce  # noqa: E402

DT = 0.2


def _records(tracks, frames_per_track=40, miss=None, noise=0.0, seed=0, start=50.0,
             speed=-5.0, conf=0.8):
    """Frame and det records for `tracks` vehicles, one after another in time."""
    rng = np.random.default_rng(seed)
    records, frame_id, t0 = [], 0, 0.0
    for track in range(tracks):
        for k in range(frames_per_track):
            t = t0 + DT * k
            records.append({"type": "frame", "frame_id": frame_id, "t": t})
            seen = miss is None or not miss(k) or k in (0, frames_per_track - 1)
            if seen:
                d = start + speed * DT * k
                # Width prior (primary) and ground plane (alternative), each with its own
                # noise; the ground plane reads 1/1.1 of the width prior.
                records.append({"type": "det", "frame_id": frame_id, "t": t,
                                "track_id": track, "cls": 2, "conf": conf,
                                "width_prior_m": d + noise * rng.standard_normal(),
                                "ema_wp_m": d + noise * rng.standard_normal(),
                                "raw_ground_m": d / 1.1 + noise * rng.standard_normal(),
                                "ema_gp_m": d / 1.1 + noise * rng.standard_normal(),
                                "closing_wp_mps": speed, "closing_wp_valid": True})
            frame_id += 1
        t0 += DT * frames_per_track + 5.0
    return records


class TestDetectionRate:
    def test_a_one_in_four_miss_rate_gives_three_quarters(self):
        # Each frame but a track's first and last is missed with probability 1/4.
        rng = np.random.default_rng(9)
        records = _records(60, miss=lambda k: rng.random() < 0.25)
        table = ce.detection_rate_table(records)
        detected = sum(table["detected"])
        missed = sum(table["missed"])
        assert detected / (detected + missed) == pytest.approx(0.75, abs=0.02)
        bins = [p for p, n in zip(table["p"], table["opportunities"]) if n >= 100]
        assert bins and all(p == pytest.approx(0.75, abs=0.05) for p in bins)

    def test_missed_frames_take_the_interpolated_distance(self):
        records = _records(1, frames_per_track=4, miss=lambda k: k == 1, start=25.0, speed=-25.0)
        # Detections at 25, 15 and 10 m; the miss at t = 0.2 s lies at 20 m, in bin 2. The
        # first and last detections are not opportunities.
        table = ce.detection_rate_table(records)
        assert table["missed"][2] == 1 and table["detected"][1] == 1
        assert sum(table["detected"]) == 1 and sum(table["missed"]) == 1

    def test_tracks_with_fewer_than_three_detections_are_ignored(self):
        records = _records(5, frames_per_track=2)
        assert sum(ce.detection_rate_table(records)["opportunities"]) == 0


class TestNoise:
    def test_linear_tracks_with_1_5_m_noise_give_sigma_1_5(self):
        records = _records(300, frames_per_track=40, noise=1.5, seed=3, start=70.0)
        table = ce.distance_noise_table(records)
        for key, count in (("sigma_raw_m", "n_raw"), ("sigma_ema_m", "n_ema")):
            pooled = [s for s, n in zip(table[key], table[count]) if n >= 1000]
            assert pooled and all(s == pytest.approx(1.5, abs=0.1) for s in pooled), (key, pooled)

    def test_closing_rate_noise_is_the_slope_standard_error(self):
        sigma = 1.0
        records = _records(600, frames_per_track=48, noise=sigma, seed=5, start=70.0)
        table = ce.closing_rate_noise_table(records)
        # Eight points 0.2 s apart: SE of the slope = sigma / sqrt(0.04 * 42).
        expected = sigma / np.sqrt(0.04 * 42.0)
        pooled = [s for s, n in zip(table["sigma_v_mps"], table["n"]) if n >= 100]
        assert pooled and all(s == pytest.approx(expected, rel=0.12) for s in pooled), pooled

    def test_noise_free_tracks_give_zero(self):
        table = ce.distance_noise_table(_records(10, noise=0.0))
        assert all(s in (None, pytest.approx(0.0, abs=1e-9)) for s in table["sigma_ema_m"])


class TestOtherTables:
    def test_confidences_are_stored_whole_by_bin(self):
        table = ce.confidence_table(_records(3, frames_per_track=10, start=15.0, speed=0.0,
                                             conf=0.7))
        assert table["values"][1] == [0.7] * 30 and table["n"][1] == 30
        assert table["values"][0] == []

    def test_bias_ratio_is_ground_over_width_prior(self):
        table = ce.bias_ratio_table(_records(3, frames_per_track=10, start=15.0, speed=0.0))
        assert table["median"][1] == pytest.approx(1 / 1.1)

    def test_the_method_key_picks_the_distance_every_table_reads(self):
        # Three cars standing at a width-prior 21 m (bin 2) read 21/1.1 = 19.1 m on the
        # ground plane (bin 1).
        records = _records(3, frames_per_track=10, start=21.0, speed=0.0)
        wp = ce.error_tables(records, "width_prior")
        gp = ce.error_tables(records, "ground_plane")
        assert wp["method"] == "width_prior" and gp["method"] == "ground_plane"
        assert wp["confidence"]["n"][2] == 30 and wp["confidence"]["n"][1] == 0
        assert gp["confidence"]["n"][1] == 30 and gp["confidence"]["n"][2] == 0
        with pytest.raises(KeyError):
            ce.error_tables(records, "bogus")

    def test_first_detection_of_approaching_tracks(self):
        out = ce.first_detection_table(_records(4, frames_per_track=10, start=60.0))
        assert out["n"] == 4 and out["median_m"] == pytest.approx(60.0)
        receding = ce.first_detection_table(_records(4, frames_per_track=10, start=20.0,
                                                     speed=5.0))
        assert receding["n"] == 0

    def test_r_max_is_the_largest_bin_with_50_observations_and_p_at_least_0_2(self):
        detection = {"opportunities": [100, 80, 60, 49, 200, 10],
                     "p": [0.9, 0.5, 0.25, 0.9, 0.1, 0.9]}
        assert ce.r_max(detection, bin_m=10.0) == 30.0
        assert ce.r_max({"opportunities": [10], "p": [0.9]}, 10.0) is None


class TestHorizon:
    def test_the_fit_recovers_a_planted_horizon_and_camera_height(self):
        rng = np.random.default_rng(0)
        boxes = []
        for z in rng.uniform(8.0, 80.0, 400):
            width = 800.0 * 1.8 / z
            bottom = 825.0 + 1.25 * 800.0 / z + rng.normal(0.0, 0.5)
            boxes.append((np.array([360 - width / 2, bottom - 0.75 * width, 360 + width / 2,
                                    bottom], np.float32), 0.9, 2))
        fit = ce.horizon_fit(boxes)
        assert fit["source"] == "fit"
        assert fit["horizon_y_px"] == pytest.approx(825.0, abs=1.0)
        assert fit["camera_height_m"] == pytest.approx(1.25, abs=0.02)

    def test_too_few_boxes_fall_back_and_say_so(self):
        fit = ce.horizon_fit([])
        assert fit["source"] == "fallback" and fit["horizon_y_px"] == 825.0


class TestGrouping:
    def test_drives_are_grouped_by_their_frame_interval(self):
        assert ce.interval_group([0.0, 0.2, 0.4, 0.6]) == "0.20"
        assert ce.interval_group([0.0, 0.67, 1.34]) == "0.67"


# ------------------------------------------------------------------ real smoke


SMOKE = REPO_ROOT / "outputs" / "jam" / "smoke" / "frames"


@pytest.mark.skipif(not all((SMOKE / "run_20260908_190548" / name).exists()
                            for name in ("detections_640.jsonl", "detections_1280.jsonl",
                                         "raw_640.jsonl")),
                    reason="the S7 frame smoke at 640 and 1,280 px has not run")
def test_the_50_frame_records_give_a_parseable_artefact_with_every_table(tmp_path):
    import fit_camera_model as fcm

    from src.sumo.sensing import CameraModel

    out = tmp_path / "camera_error_model.json"
    assert fcm.main(["--frames-dir", str(SMOKE), "--imgsz", "640", "--sensitivity-imgsz",
                     "--out", str(out)]) == 0
    artefact = json.loads(out.read_text())
    assert artefact["schema"] == "camera_error_model/1" and artefact["sumo_table"] == "0.20"
    # The SUMO model reads the width-prior tables; the ground-plane ones are recorded.
    assert artefact["method"] == "width_prior"
    assert artefact["tables"]["0.20"]["method"] == "width_prior"
    assert artefact["alternative"]["ground_plane"]["tables"]["0.20"]["method"] == "ground_plane"
    table = artefact["tables"]["0.20"]
    for key in ("detection", "distance_noise", "closing_rate_noise", "confidence",
                "first_detection", "bias_ratio", "R_max_m", "bins_m", "drives"):
        assert key in table
    assert table["drives"] == ["run_20260908_190548"]
    # The records carry the width-prior distance, so the tables are not empty.
    assert sum(table["detection"]["opportunities"]) > 0 and sum(table["confidence"]["n"]) > 0
    # The primary detection rate is the cross-resolution share against 1,280 px.
    assert table["detection"]["source"] == "cross_resolution"
    assert {"held_track", "cross_resolution_iou0.1"} <= set(table["detection_alternatives"])
    assert artefact["inputs"][0]["sha256"]
    assert CameraModel.from_artefact(out).source["method"] == "width_prior"


def _two_resolution(n_frames=400):
    """Planted: 1,280 px vehicles at 25 m (bin 2) and 55 m (bin 5) in every frame. A 640 px
    tracked box matches the 25 m vehicle in every second frame (share 0.5). For the 55 m one
    it matches in one frame of four, and in another one of four a shifted box overlaps it
    with IoU about 0.2 (shares 0.25 at IoU >= 0.3, 0.5 at IoU >= 0.1). The 640 px
    distances read 10% short, as the validator measured, and do not set the bin."""
    near, far = [300.0, 800.0, 340.0, 830.0], [400.0, 700.0, 420.0, 715.0]
    shifted = [413.0, 700.0, 433.0, 715.0]  # 7 px of 20 overlap: IoU 7/33 = 0.21
    reference, target, raw = [], [], {}
    for k in range(n_frames):
        for track, (box, d) in enumerate(((near, 25.0), (far, 55.0))):
            reference.append({"type": "det", "run": "r", "frame_id": k, "t": 0.2 * k,
                              "track_id": track, "box": box, "ema_wp_m": d, "ema_gp_m": d})
        boxes = []
        if k % 2 == 0:
            boxes.append(near)
        if k % 4 == 0:
            boxes.append(far)
        elif k % 4 == 1:
            boxes.append(shifted)
        for track, box in enumerate(boxes):
            target.append({"type": "det", "run": "r", "frame_id": k, "t": 0.2 * k,
                           "track_id": 10 + track, "box": box, "ema_wp_m": 0.9 * 40.0,
                           "ema_gp_m": 0.9 * 40.0, "conf": 0.8, "width_prior_m": 36.0,
                           "raw_ground_m": 36.0})
        # The raw boxes hold every tracked box plus the far vehicle in every frame.
        raw[("r", k)] = boxes + [far]
    return reference, target, raw


class TestCrossResolution:
    def test_the_planted_shares_are_measured_back_and_binned_by_the_reference(self):
        reference, target, _ = _two_resolution()
        table = ce.cross_resolution_table(reference, ce.tracked_boxes(target), iou_min=0.3)
        assert table["opportunities"][2] == 400 and table["opportunities"][5] == 400
        assert table["p"][2] == pytest.approx(0.5) and table["p"][5] == pytest.approx(0.25)
        assert table["opportunities"][3] == 0  # the 640 px distance (36 m) sets no bin
        loose = ce.cross_resolution_table(reference, ce.tracked_boxes(target), iou_min=0.1)
        assert loose["p"][5] == pytest.approx(0.5) and loose["p"][2] == pytest.approx(0.5)

    def test_the_primary_detection_table_is_the_cross_resolution_share(self):
        reference, target, raw = _two_resolution()
        records = target + [{"type": "frame", "run": "r", "frame_id": k, "t": 0.2 * k}
                            for k in range(400)]
        tables = ce.error_tables(records, "width_prior", reference=reference, raw_boxes=raw)
        assert tables["detection"]["p"][5] == pytest.approx(0.25)
        assert tables["detection"]["source"] == "cross_resolution"
        alternatives = tables["detection_alternatives"]
        assert alternatives["cross_resolution_iou0.1"]["p"][5] == pytest.approx(0.5)
        assert alternatives["cross_resolution_raw_iou0.3"]["p"][5] == pytest.approx(1.0)
        assert "held_track" in alternatives
        # R_max by section 1.2: the largest bin with 50 observations and p >= 0.2.
        assert tables["R_max_m"] == 60.0
        assert tables["R_max_alternatives"]["held_track"] == ce.r_max(alternatives["held_track"])

    def test_without_a_reference_the_table_is_the_held_track_rate(self):
        records = _records(5, frames_per_track=10)
        tables = ce.error_tables(records, "width_prior")
        assert tables["detection"]["source"] == "held_track"
