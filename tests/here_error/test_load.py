import json

import pytest

from here_fixtures import body, camera_line, gps_line, here_line, result, write_run
from here_error import load
from here_error.clock import ClockOffset

T0 = 1788901800.0
OFFSET = ClockOffset(1.2, 1.2, 1.2, 10)
PTS = [(40.0, -74.0), (40.001, -74.0), (40.002, -74.0)]


def make(tmp_path, *, index_lat=40.0, video_ids=(11, 12), extra_phone=()):
    phone_lines = [
        gps_line(0, T0, 40.0, -74.0),
        gps_line(1, T0 + 1, 40.0001, -74.0),
        gps_line(2, T0 + 1, 40.0001, -74.0),            # duplicate fix time: dropped
        gps_line(3, T0 + 2, None, -74.0, valid=False),  # invalid: dropped
        gps_line(4, T0 + 3, 40.0002, -74.0),
        here_line(0, T0 + 5.2, 40.0, -74.0),
        camera_line(11, T0 + 5.2, mono_gap_s=0.5),
        camera_line(12, T0 + 6.2, mono_gap_s=0.5),
        *extra_phone,
    ]
    index = [{"seq": 0, "file": "here/000000.json", "query_lat": index_lat, "query_lon": -74.0}]
    bodies = {"here/000000.json": body("2026-09-08T21:10:00Z", [result("Main St", 222.0, PTS)])}
    d, phone = write_run(tmp_path, "run_x", phone_lines, index, bodies, list(video_ids))
    return d, phone


def test_phone_log_filters_and_sorts_gps(tmp_path):
    d, phone = make(tmp_path)
    log = load.load_phone_log(phone)
    assert [f.utc_s for f in log.gps] == [T0, T0 + 1, T0 + 3]
    assert log.n_gps_invalid == 1 and log.n_gps_duplicate_time == 1
    assert log.gps[0].wall_s == pytest.approx(T0 + 1.2)


def test_here_body_joined_and_response_time_in_utc(tmp_path):
    d, phone = make(tmp_path)
    log = load.load_phone_log(phone)
    (b,) = load.load_here_bodies("run_x", d, log, OFFSET)
    assert b.response_utc_s == pytest.approx(T0 + 5.2 - 1.2)
    import datetime as dt
    assert b.data_time_s == dt.datetime(2026, 9, 8, 21, 10, tzinfo=dt.timezone.utc).timestamp()
    (s,) = b.segments
    assert s.key == "Main St|222|40.00000,-74.00000|40.00200,-74.00000"
    assert len(s.lat) == 3 and s.functional_class == 2 and s.subsegments == ()


def test_mismatched_query_centre_refused(tmp_path):
    d, phone = make(tmp_path, index_lat=40.0001)
    log = load.load_phone_log(phone)
    with pytest.raises(load.LoadError, match="query centres disagree"):
        load.load_here_bodies("run_x", d, log, OFFSET)


def test_index_sequence_without_phone_record_refused(tmp_path):
    d, phone = make(tmp_path)
    idx = d / "here_index.jsonl"
    idx.write_text(idx.read_text() + json.dumps({"seq": 7, "file": "here/000000.json", "query_lat": 40.0, "query_lon": -74.0}) + "\n")
    with pytest.raises(load.LoadError, match="no phone record"):
        load.load_here_bodies("run_x", d, load.load_phone_log(phone), OFFSET)


def test_frame_time_arithmetic(tmp_path):
    d, phone = make(tmp_path)
    log = load.load_phone_log(phone)
    frames = load.load_frames("run_x", d, log, OFFSET)
    # wall = T0+5.2, capture is 0.5 s earlier on the phone clock, then minus the 1.2 s offset.
    assert frames[0].capture_utc_s == pytest.approx(T0 + 5.2 - 0.5 - 1.2)
    assert frames[1].capture_utc_s == pytest.approx(T0 + 6.2 - 0.5 - 1.2)
    assert frames[0].key == "run_x:11"


def test_repeated_frame_id_refused(tmp_path):
    d, phone = make(tmp_path, video_ids=(11, 11))
    with pytest.raises(load.LoadError, match="repeats"):
        load.load_frames("run_x", d, load.load_phone_log(phone), OFFSET)


def test_repeated_frame_id_in_phone_log_refused(tmp_path):
    d, phone = make(tmp_path, extra_phone=(camera_line(11, T0 + 9),))
    with pytest.raises(load.LoadError, match="appears twice"):
        load.load_frames("run_x", d, load.load_phone_log(phone), OFFSET)


def test_frame_without_camera_record_refused(tmp_path):
    d, phone = make(tmp_path, video_ids=(11, 99))
    with pytest.raises(load.LoadError, match="no camera record"):
        load.load_frames("run_x", d, load.load_phone_log(phone), OFFSET)


def test_reading_without_speed_is_kept_and_marked(tmp_path):
    d, phone = make(tmp_path)
    r = result("No Speed Rd", 100.0, PTS)
    del r["currentFlow"]["speed"]
    (d / "here/000000.json").write_text(json.dumps(body("2026-09-08T21:10:00Z", [r])))
    (b,) = load.load_here_bodies("run_x", d, load.load_phone_log(phone), OFFSET)
    assert b.segments[0].speed_mps is None
