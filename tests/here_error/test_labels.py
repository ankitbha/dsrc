import csv
import datetime as dt

import pytest

from here_error import labels
from here_error.models import Frame, FrameDetections

DUSK = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.timezone.utc).timestamp()   # 19:15 EDT


def pool():
    return ([Frame("run_a", i, i, DUSK - 4000 + i) for i in range(300)]
            + [Frame("run_b", i, i, DUSK + 100 + i) for i in range(300)])


def test_sample_is_stratified_and_reproducible():
    a = labels.sample_frames(pool())
    b = labels.sample_frames(pool())
    assert [f.key for f in a] == [f.key for f in b]
    assert len(a) == 150 and len({f.key for f in a}) == 150
    assert sum(f.capture_utc_s < DUSK for f in a) == 75
    assert sum(f.capture_utc_s >= DUSK for f in a) == 75


def test_sample_refuses_a_thin_half():
    thin = [f for f in pool() if f.capture_utc_s < DUSK] + pool()[-10:]
    with pytest.raises(labels.LabelError):
        labels.sample_frames(thin)


def det(key, n, leader, t):
    run, fid = key.split(":")
    return FrameDetections(run, int(fid), int(fid), t, n, leader, 100.0)


def test_scoring_arithmetic():
    rows = [labels.LabelRow("r:1", 2, 1, True), labels.LabelRow("r:2", 1, 0, False),
            labels.LabelRow("r:3", 0, 0, False)]
    dets = {"r:1": det("r:1", 2, True, DUSK - 10), "r:2": det("r:2", 3, True, DUSK - 5),
            "r:3": det("r:3", 0, False, DUSK + 5)}
    before, after = labels.score(rows, dets)
    # before: detected 2 and 3; labelled (moving+parked) 3 and 1 -> matched 2 + 1 = 3.
    assert before.n_frames == 2
    assert before.precision_all == pytest.approx(3 / 5)
    assert before.recall_all == pytest.approx(3 / 4)
    # moving only: labelled 2 and 1 -> matched 2 + 1 = 3 -> recall 3/3, precision 3/5
    assert before.recall_moving == pytest.approx(1.0)
    assert before.precision_moving == pytest.approx(3 / 5)
    assert before.leader_accuracy == pytest.approx(0.5)
    assert before.parked_share == pytest.approx(1 / 4)
    assert after.n_frames == 1 and after.precision_all is None and after.recall_all is None
    assert after.leader_accuracy == 1.0


def test_unknown_frame_refused():
    with pytest.raises(labels.LabelError, match="no detections"):
        labels.score([labels.LabelRow("x:9", 1, 0, True)], {})


def test_csv_reading_refuses_blanks_and_duplicates(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("frame_key,n_vehicles_moving,n_vehicles_parked,leader_present\nr:1,2,1,yes\nr:2,0,0,0\n")
    rows = labels.read_labels(p)
    assert rows[0] == labels.LabelRow("r:1", 2, 1, True) and rows[1].leader_present is False
    p.write_text("frame_key,n_vehicles_moving,n_vehicles_parked,leader_present\nr:1,2,,yes\n")
    with pytest.raises(labels.LabelError, match="blank"):
        labels.read_labels(p)
    p.write_text("frame_key,n_vehicles_moving,n_vehicles_parked,leader_present\nr:1,2,0,yes\nr:1,1,0,no\n")
    with pytest.raises(labels.LabelError, match="twice"):
        labels.read_labels(p)


def test_export_writes_images_and_blank_template(tmp_path):
    import cv2
    import numpy as np

    vp = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (1280, 720))
    for i in range(6):
        vw.write(np.full((720, 1280, 3), 50, np.uint8))
    vw.release()
    sample = [Frame("run_a", 1, 11, DUSK - 100), Frame("run_a", 4, 14, DUSK + 100)]
    t = labels.export_sample(sample, {"run_a": vp}, tmp_path / "out")
    assert (tmp_path / "out" / "run_a__11.jpg").exists() and (tmp_path / "out" / "run_a__14.jpg").exists()
    img = cv2.imread(str(tmp_path / "out" / "run_a__11.jpg"))
    assert img.shape == (1280, 720, 3)
    rows = list(csv.reader(t.open()))
    assert rows[0] == list(labels.COLUMNS) and rows[1] == ["run_a:11", "", "", ""]


def test_a_byte_order_mark_from_excel_is_accepted(tmp_path):
    p = tmp_path / "l.csv"
    p.write_bytes(b"\xef\xbb\xbf" + b"frame_key,n_vehicles_moving,n_vehicles_parked,leader_present\r\nr:1,2,1,yes\r\n")
    assert labels.read_labels(p) == [labels.LabelRow("r:1", 2, 1, True)]
