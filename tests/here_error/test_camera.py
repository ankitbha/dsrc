import json

import cv2
import numpy as np
import pytest

from here_error import camera
from here_error.camera import Box
from here_error.models import Frame


def test_detector_receives_the_clockwise_rotation():
    raw = np.zeros((720, 1280, 3), np.uint8)
    raw[0, 0] = (1, 2, 3)          # top-left of the recorded frame
    raw[0, 1279] = (9, 9, 9)       # top-right
    seen = {}

    def det(img):
        seen["img"] = img
        return []

    camera.analyse_frame(raw, det)
    img = seen["img"]
    assert img.shape == (1280, 720, 3)
    # Clockwise: the top-left pixel moves to the top-right; the top-right moves to the bottom-right.
    assert tuple(img[0, 719]) == (1, 2, 3)
    assert tuple(img[1279, 719]) == (9, 9, 9)
    assert np.array_equal(img, cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE))


def test_class_filter_and_confidence():
    boxes = [Box(2, 0.9, 0, 0, 10, 10), Box(0, 0.99, 0, 0, 10, 10),      # car, person
             Box(7, 0.5, 0, 0, 10, 10), Box(5, 0.2, 0, 0, 10, 10),       # truck, low-confidence bus
             Box(3, 0.4, 0, 0, 10, 10), Box(1, 0.9, 0, 0, 10, 10)]       # motorcycle, bicycle
    assert [b.cls for b in camera.vehicle_boxes(boxes, 720, 1280)] == [2, 7, 3]


def test_leader_region():
    w, h = 720, 1280
    centred_tall = Box(2, 0.9, 300, 600, 420, 700)           # centre 360, height 100 >= 51.2
    assert camera.has_leader([centred_tall], w, h)
    off_centre = Box(2, 0.9, 20, 600, 120, 700)              # centre 70: outside the middle 30%
    assert not camera.has_leader([off_centre], w, h)
    edge_in = Box(2, 0.9, 468 - 50, 600, 468 + 50, 700)      # centre 468 = 360 + 108 (the limit)
    assert camera.has_leader([edge_in], w, h)
    edge_out = Box(2, 0.9, 470 - 50, 600, 470 + 50, 700)     # centre 470 > limit
    assert not camera.has_leader([edge_out], w, h)
    too_small = Box(2, 0.9, 300, 600, 420, 640)              # height 40 < 51.2
    assert not camera.has_leader([too_small], w, h)
    person = Box(0, 0.9, 300, 600, 420, 700)
    assert not camera.has_leader([person], w, h)


def write_video(path, n, size=(1280, 720)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, size)
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), 40 + i, np.uint8))
    vw.release()


FP = camera.make_fingerprint("w" * 64)


def frames(n):
    return [Frame("run_t", i, 100 + i, 1000.0 + i) for i in range(n)]


def test_run_writes_one_line_per_frame_and_honours_limit(tmp_path):
    write_video(tmp_path / "v.avi", 6)
    calls = []

    def det(img):
        calls.append(1)
        return [Box(2, 0.9, 300, 600, 420, 700)]

    st = camera.run_camera("run_t", tmp_path / "v.avi", frames(6), det, tmp_path / "d.jsonl", FP, limit=4)
    assert st.n_new == 4 and len(calls) == 4
    rows = camera.load_detections(tmp_path / "d.jsonl")
    assert [r.frame_id for r in rows] == [100, 101, 102, 103]
    assert all(r.n_vehicles == 1 and r.leader for r in rows)
    assert rows[0].utc_s == 1000.0 and rows[1].brightness > rows[0].brightness


def test_resume_after_a_truncated_last_line(tmp_path):
    write_video(tmp_path / "v.avi", 5)
    out = tmp_path / "d.jsonl"
    camera.run_camera("run_t", tmp_path / "v.avi", frames(5), lambda i: [], out, FP, limit=3)
    good = out.read_text()
    out.write_text(good + '{"run": "run_t", "pos": 3, "fra')          # killed mid-write
    calls = []

    def det(img):
        calls.append(1)
        return []

    st = camera.run_camera("run_t", tmp_path / "v.avi", frames(5), det, out, FP)
    assert st.n_skipped == 3 and st.n_new == 2 and len(calls) == 2
    rows = camera.load_detections(out)
    assert [r.pos for r in rows] == [0, 1, 2, 3, 4]
    for line in out.read_text().splitlines():
        json.loads(line)


def test_frame_count_mismatch_refused(tmp_path):
    write_video(tmp_path / "v.avi", 3)
    with pytest.raises(camera.CameraError, match="holds 3 frames"):
        camera.run_camera("run_t", tmp_path / "v.avi", frames(4), lambda i: [], tmp_path / "d.jsonl", FP)


def test_own_bonnet_is_neither_a_vehicle_nor_a_leader():
    w, h = 720, 1280
    hood = Box(2, 0.81, 1.6, 955.6, 718.0, 1276.9)        # the box the detector gives for the bonnet
    assert camera.is_ego_hood(hood, w, h)
    assert camera.vehicle_boxes([hood], w, h) == []
    assert not camera.has_leader([hood], w, h)
    # A wide vehicle that ends well above the bottom edge, or a narrow one at the bottom, still counts.
    near_car = Box(2, 0.9, 20, 800, 700, 1000)
    narrow_low = Box(2, 0.9, 300, 1100, 420, 1275)
    assert not camera.is_ego_hood(near_car, w, h) and not camera.is_ego_hood(narrow_low, w, h)
    assert camera.has_leader([hood, near_car], w, h)


def test_fingerprint_names_the_weights_and_every_parameter_that_shapes_a_detection():
    fp = camera.make_fingerprint("abc")
    assert fp["weights_sha256"] == "abc" and fp["rotation"] == "ROTATE_90_CLOCKWISE"
    for key in ("yolo_conf", "vehicle_classes", "leader_centre_share", "leader_min_height_share",
                "ego_hood_bottom_share", "ego_hood_min_width_share"):
        assert key in fp
    assert camera.make_fingerprint("abd") != fp


def test_resume_refuses_a_file_made_with_other_weights_or_parameters(tmp_path, monkeypatch):
    write_video(tmp_path / "v.avi", 4)
    out = tmp_path / "d.jsonl"
    camera.run_camera("run_t", tmp_path / "v.avi", frames(4), lambda i: [], out, FP, limit=2)
    other = camera.make_fingerprint("x" * 64)
    with pytest.raises(camera.CameraError, match="different weights or parameters.*delete"):
        camera.run_camera("run_t", tmp_path / "v.avi", frames(4), lambda i: [], out, other)
    monkeypatch.setattr(camera.params, "YOLO_CONF", 0.5)
    with pytest.raises(camera.CameraError, match="delete"):
        camera.run_camera("run_t", tmp_path / "v.avi", frames(4), lambda i: [], out, camera.make_fingerprint("w" * 64))
    monkeypatch.undo()
    st = camera.run_camera("run_t", tmp_path / "v.avi", frames(4), lambda i: [], out, FP)
    assert st.n_skipped == 2 and st.n_new == 2


def test_a_file_without_a_fingerprint_counts_as_a_mismatch(tmp_path):
    write_video(tmp_path / "v.avi", 3)
    out = tmp_path / "d.jsonl"
    out.write_text(json.dumps({"run": "run_t", "pos": 0, "frame_id": 100, "utc_s": 1000.0, "n_vehicles": 0,
                               "leader": False, "brightness": 1.0}) + "\n")
    with pytest.raises(camera.CameraError, match="no fingerprint"):
        camera.run_camera("run_t", tmp_path / "v.avi", frames(3), lambda i: [], out, FP)
    assert len(camera.load_detections(out)) == 1             # the old file is left alone for the user to delete


def test_fingerprint_header_is_not_a_detection_and_is_written_once(tmp_path):
    write_video(tmp_path / "v.avi", 3)
    out = tmp_path / "d.jsonl"
    camera.run_camera("run_t", tmp_path / "v.avi", frames(3), lambda i: [], out, FP, limit=1)
    camera.run_camera("run_t", tmp_path / "v.avi", frames(3), lambda i: [], out, FP)
    lines = out.read_text().splitlines()
    assert sum('"fingerprint"' in l for l in lines) == 1 and '"fingerprint"' in lines[0]
    assert [d.pos for d in camera.load_detections(out)] == [0, 1, 2]
    assert camera.read_fingerprint(out) == FP
