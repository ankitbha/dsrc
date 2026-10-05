import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from here_fixtures import fix
from here_error import pipeline, report
from here_error.clock import ClockOffset
from here_error.labels import HalfScore
from here_error.models import Frame, FrameDetections, InputFile, Pass, PassTimes, Provenance, RouteRequest, RouteResult
from here_error.pipeline import Assembly, RunData
from here_error.here_times import V4Result
from here_error.stretches import StretchResult, build_stretches
from here_error.here_times import AnalysedPass

DUSK = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.timezone.utc).timestamp()
T0 = DUSK - 3000.0


def spec(i, road, t0, dur=100.0, matched=1000.0, here=90.0, ff=70.0, stopped=0.0, jam=1.0):
    return dict(i=i, road=road, t0=t0, dur=dur, matched=matched, here=here, ff=ff, stopped=stopped, jam=jam)


def make_asm(specs, n_frames=200, detections_ready=True):
    passes, times = [], []
    for s in specs:
        pid = f"t-{s['i']:03d}"
        fixes = (fix(s["t0"], 0, 0), fix(s["t0"] + s["dur"], s["matched"], 0))
        passes.append(Pass(pid, "run_t", f"k{s['road']}", s["road"], 2, fixes, 0.0, s["matched"], False,
                           s["matched"], s["matched"], None))
        times.append(PassTimes(
            pid, 0, s["t0"] - 30, 30.0, True, s["dur"], s["here"], s["ff"], (s["here"] - s["dur"]) / s["dur"],
            (s["ff"] - s["dur"]) / s["dur"], s["jam"], 0.99, s["stopped"], s["dur"] - s["stopped"],
            0, (s["here"] - s["dur"]) / s["dur"], 0, (s["here"] - s["dur"]) / s["dur"], 1, 0.5, None))
    frames = tuple(Frame("run_t", k, k, T0 + k * 10.0) for k in range(n_frames))
    rd = RunData("run_t", Path("."), Path("."), None, ClockOffset(1.2, 1.19, 1.21, 100), (), frames)
    asm = Assembly((rd,), tuple(passes), tuple(times), {}, StretchResult((), 0, 0.0), V4Result(0, None, None), ())
    sr = build_stretches(asm.analysed)
    return Assembly(asm.runs, asm.passes, asm.times, asm.catalogues, sr, asm.v4, ())


def detections(asm, n_vehicles=2, leader=True):
    return {"run_t": [FrameDetections("run_t", f.pos, f.frame_id, f.capture_utc_s, n_vehicles, leader, 100.0)
                      for f in asm.runs[0].frames]}


PROV = Provenance("abc123", False, (InputFile("x", "y"),), {})
SPECS = [spec(i, f"Road {i % 6}", T0 + 200 * i, dur=100.0, here=100.0 + 5 * (i - 5), ff=70.0, jam=1.0 + (i % 3)) for i in range(12)]


def scores(dusk_recall=0.9):
    return [HalfScore("before_dusk", 75, 0.9, 0.9, 0.9, 0.9, 0.8, 0.1), HalfScore("after_dusk", 75, 0.9, dusk_recall, 0.9, dusk_recall, 0.8, 0.1)]


def routes(asm):
    return [RouteResult(q.request_id, q.kind, 95.0, 90.0, 1000.0, q.observed_s, -0.05, True, 1.0, 0.0, None)
            for q in pipeline.route_requests(asm)]


def inputs_for(asm, **kw):
    base = dict(asm=asm, detections=detections(asm), label_scores=scores(), routing=routes(asm), requests=pipeline.route_requests(asm))
    base.update(kw)
    return report.ReportInputs(**base)


def test_complete_inputs_give_a_complete_headline_with_directions():
    asm = make_asm(SPECS)
    text, reasons = report.render(inputs_for(asm), PROV)
    assert reasons == []
    assert "**STATUS: COMPLETE.**" in text
    assert "a positive value means HERE's time is longer than the time the car took" in text
    assert "(HERE time - observed time) / observed time" in text
    # median of (here-100)/100 with here = 100 + 5*(i-5), i = 0..11 -> errors -0.25..+0.30; median of 12 values
    errs = [0.05 * (i - 5) for i in range(12)]
    assert f"{100 * np.median(errs):+.1f}%" in text
    assert "Measured standard deviation of the per-pass signed error" in text
    assert "Pass: abs(HERE error) - abs(free-flow error), in percentage points" in text


def test_missing_detections_marks_incomplete_and_skips_camera_sections():
    asm = make_asm(SPECS)
    text, reasons = report.render(inputs_for(asm, detections={"run_t": detections(asm)["run_t"][:50]}), PROV)
    assert any("detections cover 50 of 200 frames" in r for r in reasons)
    assert "**STATUS: INCOMPLETE.**" in text
    assert "camera conditions: not computed" in text and "driver offset: not computed" in text


def test_missing_labels_and_routing_are_reported():
    asm = make_asm(SPECS)
    _, reasons = report.render(inputs_for(asm, label_scores=None, routing=None), PROV)
    assert any("labels" in r for r in reasons) and any("routing results are missing" in r for r in reasons)
    short = routes(asm)[:-1]
    _, reasons = report.render(inputs_for(asm, routing=short), PROV)
    assert any("cover" in r and "requests" in r for r in reasons)


def test_conditions_come_from_frames_inside_the_pass():
    asm = make_asm([spec(1, "A", T0 + 100, dur=50)])
    dets = detections(asm)
    # frames at T0+100 .. T0+150 inclusive are positions 10..15 (every 10 s): give them 4 vehicles, no leader.
    dets["run_t"] = [FrameDetections(d.run, d.pos, d.frame_id, d.utc_s, 4 if 10 <= d.pos <= 15 else 0,
                                     not (10 <= d.pos <= 15), 100.0) for d in dets["run_t"]]
    (row,) = report.pass_rows(inputs_for(asm, detections=dets))
    assert row.n_frames == 6 and row.mean_vehicles == 4.0 and row.leader_share == 0.0


def test_dusk_recall_below_threshold_uses_daylight_frames_only():
    # The pass straddles dusk: frames before dusk have 1 vehicle, frames after it 9.
    asm = make_asm([spec(1, "A", DUSK - 50, dur=100)], n_frames=400)
    dets = {"run_t": [FrameDetections("run_t", f.pos, f.frame_id, f.capture_utc_s, 1 if f.capture_utc_s < DUSK else 9, True, 1.0)
                      for f in asm.runs[0].frames]}
    both = report.pass_rows(inputs_for(asm, detections=dets, label_scores=scores(0.9)))[0]
    day = report.pass_rows(inputs_for(asm, detections=dets, label_scores=scores(0.5)))[0]
    assert report.daylight_only(inputs_for(asm, label_scores=scores(0.5)))
    assert both.mean_vehicles > 1.0 and day.mean_vehicles == 1.0


def test_driver_offset_arithmetic():
    # Moving 1000 m in 50 s = 20 m/s; free-flow time 62.5 s = 16 m/s; offset = +25%. No stops, open road.
    asm = make_asm([spec(i, f"R{i}", T0 + 200 * i, dur=50.0, here=60.0, ff=62.5, jam=1.0) for i in range(4)])
    dets = detections(asm, leader=False)
    rows = report.pass_rows(inputs_for(asm, detections=dets))
    off = report.driver_offset(rows)
    assert off.n_passes == 4 and off.offset == pytest.approx(0.25)
    # A pass behind a leader or in heavy jam is left out.
    jam = make_asm([spec(0, "R", T0, dur=50.0, here=60.0, ff=62.5, jam=3.0)])
    assert report.driver_offset(report.pass_rows(inputs_for(jam, detections=detections(jam, leader=False)))) is None
    lead = make_asm([spec(0, "R", T0, dur=50.0, here=60.0, ff=62.5)])
    assert report.driver_offset(report.pass_rows(inputs_for(lead, detections=detections(lead, leader=True)))) is None


def test_stage5_reports_error_with_and_without_offset():
    asm = make_asm([spec(i, f"R{i}", T0 + 200 * i, dur=50.0, here=60.0, ff=62.5) for i in range(6)])
    text, _ = report.render(inputs_for(asm, detections=detections(asm, leader=False)), PROV)
    assert "Driver offset: median of (moving speed / HERE free-flow speed - 1) over 6 open-road passes: +25.0%" in text
    # observed 50 s -> adjusted 50 * 1.25 = 62.5 s; HERE 60 s -> -4.0%; unadjusted +20.0%
    assert "without removing the offset: +20.0%" in text and "-4.0% (" in text


def test_csv_outputs_are_written(tmp_path):
    asm = make_asm(SPECS)
    inp = inputs_for(asm)
    rows = report.pass_rows(inp)
    report.write_passes_csv(asm, rows, tmp_path / "p.csv")
    report.write_stretches_csv(asm, tmp_path / "s.csv")
    import csv

    got = list(csv.DictReader((tmp_path / "p.csv").open()))
    assert len(got) == 12 and got[0]["pass_id"] == "t-000" and float(got[0]["signed_error"]) == pytest.approx(-0.25)
    assert got[0]["mean_vehicles"] == "2.0"
