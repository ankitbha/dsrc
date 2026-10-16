"""Re-processing the drive frames (plan section 2.2, step S7).

Unit tests build synthetic drive directories and detections. The real smoke at the bottom
runs YOLOv8n on 50 frames of `run_20260908_190548` with the fetched weights; the drive
data is only read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.analysis import camera_error as ce  # noqa: E402

DRIVES = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")


def _drive(tmp_path: Path, index_lines: list[str], ticks: list[tuple[int, float]]) -> Path:
    run = tmp_path / "run_x"
    run.mkdir()
    (run / "video_index.jsonl").write_text("\n".join(index_lines) + "\n")
    meta = [json.dumps({"type": "system", "t_wall": 1.0})]
    for frame_id, t in ticks:
        meta.append(json.dumps({"type": "tick", "frame_id": frame_id,
                                "t_capture_mono_ns": int(round(t * 1e9)), "gps": {}}))
    (run / "metadata.jsonl").write_text("\n".join(meta) + "\n")
    return run


class TestIndexAndTimes:
    def test_duplicate_ids_keep_their_first_position_and_bad_lines_are_counted(self, tmp_path):
        run = _drive(tmp_path, ['{"pos": 0, "frame_id": 5}', '{"pos": 1, "frame_id": 6}',
                                '{"pos": 2, "frame_id": 5}', '{"pos": 3, "frame_id"',
                                '{"pos": 4, "frame_id": 7}'], [])
        index = ce.read_index(run)
        assert index.positions == {5: 0, 6: 1, 7: 4}
        assert index.lines == 5 and index.duplicates == 1 and index.unparseable == 1

    def test_frame_times_are_tick_capture_times(self, tmp_path):
        run = _drive(tmp_path, [json.dumps({"pos": k, "frame_id": 10 + k}) for k in range(5)],
                     [(10 + k, 100.0 + 0.2 * k) for k in range(5)])
        times = ce.frame_times(run)
        gaps = np.diff([times[10 + k] for k in range(5)])
        assert gaps == pytest.approx([0.2] * 4)
        assert not np.allclose(gaps, 1.0 / 30.0)

    def test_frames_without_a_tick_are_skipped_and_counted(self, tmp_path):
        run = _drive(tmp_path, [json.dumps({"pos": k, "frame_id": 10 + k}) for k in range(4)],
                     [(10, 1.0), (11, 1.2), (13, 1.6)])
        plan = ce.frame_plan(run)
        assert [(f.frame_id, f.pos) for f in plan.frames] == [(10, 0), (11, 1), (13, 3)]
        assert plan.no_tick == 1 and plan.indexed == 4
        assert [f.t for f in plan.frames] == pytest.approx([1.0, 1.2, 1.6])

    def test_the_first_tick_of_a_frame_wins(self, tmp_path):
        run = _drive(tmp_path, ['{"pos": 0, "frame_id": 1}'], [(1, 5.0), (1, 6.0)])
        assert ce.frame_times(run) == {1: pytest.approx(5.0)}


class TestRotation:
    def test_a_1280_by_720_frame_becomes_720_by_1280_turned_clockwise(self):
        raw = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw[0, 0] = 255   # top-left of the landscape buffer
        upright = ce.rotate(raw)
        assert upright.shape == (1280, 720, 3)
        # Clockwise: the buffer's top-left lands at the upright frame's top-right.
        assert upright[0, 719].tolist() == [255, 255, 255]
        assert upright.sum() == 255 * 3


def _box(cx, bottom, width):
    return np.array([cx - width / 2, bottom - 0.75 * width, cx + width / 2, bottom],
                    dtype=np.float32)


class TestTrackAndMeasure:
    def test_an_approaching_car_closes_with_its_own_frame_times(self):
        # A car of width 1.8 m straight ahead, from 40 m to 30 m over 2 s, 0.2 s frames;
        # its box follows the pinhole model at fx = 800 with the horizon at 825 px.
        frames = []
        for k in range(11):
            t = 10.0 + 0.2 * k
            z = 40.0 - 5.0 * (t - 10.0)
            width = 800.0 * 1.8 / z
            bottom = 825.0 + 1.25 * 800.0 / z
            frames.append(ce.FrameDetections(frame_id=k, pos=k, t=t,
                                             boxes=[(_box(360.0, bottom, width), 0.9, 2)]))
        records = ce.track_and_measure(frames, horizon_y=825.0, camera_height=1.25)
        rows = [r for r in records if r["type"] == "det"]
        assert len({r["track_id"] for r in rows}) == 1
        assert rows[-1]["raw_ground_m"] == pytest.approx(30.0, rel=1e-3)
        assert rows[-1]["width_prior_m"] == pytest.approx(30.0, rel=1e-3)
        assert rows[-1]["ema_m"] > 30.0  # the EMA lags the raw value
        valid = [r for r in rows if r["closing_valid"]]
        assert valid and all(r["closing_rate_mps"] < 0 for r in valid)
        assert valid[-1]["closing_rate_mps"] == pytest.approx(-5.0, rel=0.2)
        assert [r["t"] for r in rows] == [f.t for f in frames[1:]]

    def test_a_box_below_the_hood_line_has_no_ground_plane_distance(self):
        frames = [ce.FrameDetections(k, k, 0.2 * k, [(_box(360.0, 1100.0, 300.0), 0.9, 2)])
                  for k in range(3)]
        rows = [r for r in ce.track_and_measure(frames, 825.0, 1.25) if r["type"] == "det"]
        assert rows and all(r["raw_ground_m"] is None for r in rows)
        assert all(r["method"] == "width_prior" for r in rows)

    def test_every_frame_is_listed_with_its_tracked_count(self):
        frames = [ce.FrameDetections(k, k, 0.2 * k, []) for k in range(4)]
        out = ce.track_and_measure(frames, 825.0, 1.25)
        assert [(r["frame_id"], r["tracked"]) for r in out if r["type"] == "frame"] == [
            (0, 0), (1, 0), (2, 0), (3, 0)]


class TestHoodFilter:
    def test_boxes_whose_centre_is_below_the_hood_line_are_dropped(self):
        kept = ce.drop_hood_boxes([(np.array([0, 900, 10, 1000], np.float32), 0.9, 2),
                                   (np.array([0, 1000, 10, 1100], np.float32), 0.9, 2)],
                                  hood_line_y=1010.0)
        assert len(kept) == 1 and kept[0][0][1] == 900


# ------------------------------------------------------------------ real smoke


@pytest.mark.skipif(not (REPO_ROOT / "outputs/models/yolov8n.pt").exists(),
                    reason="the official yolov8n.pt has not been fetched")
def test_fifty_real_frames_give_records_timed_by_their_ticks(tmp_path):
    import process_drive_frames as pdf

    out = tmp_path / "frames"
    assert pdf.main(["--runs", "run_20260908_190548", "--imgsz", "640", "--limit", "50",
                     "--start-pos", "4000", "--out-dir", str(out)]) == 0
    run_dir = out / "run_20260908_190548"
    raw = [json.loads(l) for l in (run_dir / "raw_640.jsonl").read_text().splitlines()]
    header, frames = raw[0], [r for r in raw if r["type"] == "frame"]
    weights = json.loads((REPO_ROOT / "results/jam/weights.json").read_text())
    sha = hashlib.sha256((REPO_ROOT / "outputs/models/yolov8n.pt").read_bytes()).hexdigest()
    assert header["weights_sha256"] == weights["sha256"] == sha
    assert header["imgsz"] == 640 and header["upright_shape"] == [1280, 720, 3]
    assert len(frames) == 50
    times = ce.frame_times(DRIVES / "run_20260908_190548")
    assert all(f["t"] == pytest.approx(times[f["frame_id"]]) for f in frames)
    gaps = np.diff([f["t"] for f in frames])
    assert np.median(gaps) == pytest.approx(0.2, abs=0.05)
    detections = [json.loads(l) for l in (run_dir / "detections_640.jsonl").read_text().splitlines()]
    assert detections[0]["type"] == "header"
    assert detections[0]["horizon"]["source"] in ("fit", "fallback")
    assert {r["type"] for r in detections[1:]} <= {"frame", "det"}
    assert sum(1 for r in detections if r["type"] == "frame") == 50
