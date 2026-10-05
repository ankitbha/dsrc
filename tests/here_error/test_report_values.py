"""Pins each report quantity to a value computed here with numpy from a fixture in which every quantity differs."""
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from here_fixtures import fix
from here_error import params, pipeline, report, stats
from here_error.clock import ClockOffset
from here_error.here_times import AnalysedPass, V4Result
from here_error.models import Frame, FrameDetections, InputFile, Pass, PassTimes, Provenance, RouteResult
from here_error.pipeline import Assembly, RunData
from here_error.stretches import StretchResult, build_stretches

DUSK = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.timezone.utc).timestamp()
T0 = DUSK - 6000.0
PROV = Provenance("abc", False, (InputFile("x", "y"),), {})


def mk(i, rid, seg, t0, dur, here, ff, sens, arr, shuf, stopped=0.0, jam=1.0, fc=2, matched=1000.0, arrived=True):
    pid = f"t-{i:03d}"
    fixes = (fix(t0, 0, 0), fix(t0 + dur, matched, 0))
    p = Pass(pid, "run_t", seg, f"desc {seg}", fc, fixes, 0.0, matched, False, matched, matched, None, rid, f"{rid} label")
    e = lambda h: None if h is None else (h - dur) / dur
    t = PassTimes(pid, 0, t0 - 30, 30.0, arrived, dur, here, ff, e(here), e(ff), jam, 0.99, stopped, dur - stopped,
                  0, e(sens), 0 if arr is not None else None, e(arr), 1, e(shuf), None)
    return AnalysedPass(p, t)


def asm_of(aps, n_frames=300):
    frames = tuple(Frame("run_t", k, k, T0 + k * 20.0) for k in range(n_frames))
    rd = RunData("run_t", Path("."), Path("."), None, ClockOffset(1.2, 1.19, 1.21, 10), (), frames)
    passes = tuple(a.p for a in aps)
    times = tuple(a.t for a in aps)
    a = Assembly((rd,), passes, times, {}, StretchResult((), 0, 0.0), V4Result(0, None, None), ())
    return Assembly(a.runs, a.passes, a.times, a.catalogues, build_stretches(a.analysed), a.v4, ())


# Six roads, one or two passes each. Every quantity has its own value.
APS = [
    mk(0, "R1", "s1", T0 + 0, 100, 110, 70, 120, 105, 160),
    mk(1, "R1", "s2", T0 + 300, 100, 90, 60, 100, None, 140),
    mk(2, "R2", "s3", T0 + 600, 100, 130, 80, 95, 125, None, arrived=False),
    mk(3, "R3", "s4", T0 + 900, 100, 100, 75, 105, None, 170, stopped=10),
    mk(4, "R4", "s5", T0 + 1200, 100, 140, 90, 150, 135, 190),
    mk(5, "R5", "s6", T0 + 1500, 100, 80, 85, 85, None, None, arrived=False),
    mk(6, "R6", "s7", T0 + 1800, 100, 120, 95, 125, 118, 200),
]


def table_cells(text, label_start):
    line = next(l for l in text.splitlines() if l.startswith("| " + label_start))
    return [c.strip() for c in line.strip("|").split("|")]


def render(aps=APS, **kw):
    asm = asm_of(aps)
    base = dict(asm=asm, detections={}, label_scores=None, routing=None, requests=pipeline.route_requests(asm))
    base.update(kw)
    inp = report.ReportInputs(**base)
    return report.render(inp, PROV)[0], inp


def err(h, d=100.0):
    return (h - d) / d


def test_each_headline_row_carries_its_own_median():
    text, _ = render()
    he = [err(a.t.here_s, a.t.observed_s) for a in APS]
    ff = [err(a.t.free_flow_s, a.t.observed_s) for a in APS]
    diff = [abs(h) - abs(f) for h, f in zip(he, ff)]
    sens = [a.t.sensitivity_signed_error for a in APS]
    arr = [a.t.arrived_signed_error for a in APS if a.t.arrived_signed_error is not None]
    cells = table_cells(text, "Pass: (HERE time")
    assert cells[1] == f"{100 * np.median(he):+.1f}%"
    assert table_cells(text, "Pass: (free-flow")[1] == f"{100 * np.median(ff):+.1f}%"
    assert table_cells(text, "Pass: abs(HERE")[1] == f"{100 * np.median(diff):+.1f} pp"
    assert table_cells(text, "Pass, sensitivity: HERE reading")[1] == f"{100 * np.median(sens):+.1f}%"
    assert table_cells(text, "Pass, sensitivity: latest reading")[1] == f"{100 * np.median(arr):+.1f}%"
    assert len({table_cells(text, k)[1] for k in ("Pass: (HERE time", "Pass: (free-flow", "Pass, sensitivity: HERE", "Pass, sensitivity: latest")}) == 4


def test_the_median_is_reported_not_the_mean():
    aps = [mk(i, f"R{i}", f"s{i}", T0 + 300 * i, 100, h, 70, 100, None, None) for i, h in enumerate([100, 100, 100, 100, 400])]
    text, _ = render(aps)
    assert table_cells(text, "Pass: (HERE time")[1] == "+0.0%"


def test_interval_columns_resample_roads_and_segments_separately():
    text, _ = render()
    he = [a.t.signed_error for a in APS]
    cells = table_cells(text, "Pass: (HERE time")
    road = stats.cluster_bootstrap_median(he, [a.p.road_id for a in APS])
    seg = stats.cluster_bootstrap_median(he, [a.p.segment_key for a in APS])
    assert cells[2] == f"{100 * road.lo:+.1f}% to {100 * road.hi:+.1f}%"
    assert cells[3] == f"{100 * seg.lo:+.1f}% to {100 * seg.hi:+.1f}%"
    assert "treats each directed segment as independent" in text and "resampling physical roads" in text
    assert "6 physical roads and 7 directed segments" in cells[4]


def test_fewer_than_five_roads_is_not_resolvable():
    aps = [mk(i, f"R{i % 3}", f"s{i}", T0 + 300 * i, 100, 100 + 7 * i, 70, 100, None, None) for i in range(8)]
    text, _ = render(aps)
    cells = table_cells(text, "Pass: (HERE time")
    assert cells[2] == "not resolvable: 3 physical roads"
    assert "to" in cells[3]
    one = [mk(i, "R1", f"s{i}", T0 + 300 * i, 100, 100 + 7 * i, 70, 100, None, None) for i in range(6)]
    assert "not resolvable: 1 physical road" in table_cells(render(one)[0], "Pass: (HERE time")[2]
    assert "no interval across roads is resolvable" in text


def test_per_road_table_gives_each_roads_median_and_n():
    text, _ = render()
    line = next(l for l in text.splitlines() if l.startswith("| R1 |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[2] == "2" and cells[3] == "2"
    assert cells[4] == f"{100 * np.median([0.10, -0.10]):+.1f}%"
    assert cells[5] == f"{100 * np.median([err(70), err(60)]):+.1f}%"


def test_subset_rows_pair_arrival_and_v4_with_the_primary_on_the_same_passes():
    text, _ = render()
    sub = [a for a in APS if a.t.arrived_signed_error is not None]
    prim = [a.t.signed_error for a in sub]
    arr = [a.t.arrived_signed_error for a in sub]
    line = next(l for l in text.splitlines() if l.startswith("- Arrival-constrained"))
    assert f"on its {len(sub)} passes" in line
    assert f"with the primary reading {100 * np.median(prim):+.1f}%" in line
    assert f"{100 * np.median(np.array(arr) - np.array(prim)):+.1f} pp" in line
    v = [a for a in APS if a.t.shuffled_signed_error is not None]
    real = np.array([abs(a.t.signed_error) for a in v])
    shuf = np.array([abs(a.t.shuffled_signed_error) for a in v])
    vline = next(l for l in text.splitlines() if l.startswith("- Control V4, on its"))
    assert f"on its {len(v)} passes" in vline and f"{100 * np.median(shuf - real):+.1f} pp" in vline


def test_stretch_rows_use_stretch_values():
    aps = [mk(i, "R1", "s1", T0 + 150 * i, 100, 150, 50, 100, None, None, matched=1200.0) for i in range(4)]
    text, _ = render(aps)
    assert table_cells(text, "Stretch: (HERE time")[1] == "+50.0%"
    assert table_cells(text, "Stretch: (free-flow")[1] == "-50.0%"
    assert table_cells(text, "Stretch: abs")[1] == "+0.0 pp"


def test_arrived_late_count_and_exclusions_are_printed():
    text, _ = render()
    assert "primary reading had not yet arrived at the pass's first fix: 2 of 7" in text


def test_stage_conditions_dark_by_midpoint_and_road_class_labels():
    mid_after = mk(0, "R1", "s1", DUSK - 30, 100, 110, 70, 100, None, None, fc=2)     # starts before dusk, midpoint after
    mid_before = mk(1, "R2", "s2", DUSK - 200, 100, 110, 70, 100, None, None, fc=5)   # ends before dusk
    asm = asm_of([mid_after, mid_before])
    rows = report.pass_rows(report.ReportInputs(asm, {}, None, None, []))
    assert [r.dark for r in rows] == [True, False]
    assert [r.road_class for r in rows] == ["class 2", "class 3 and above"]
    assert report._road_class(3) == "class 3 and above" and report._road_class(1) == "class 1"


def test_stopped_share_is_stopped_over_observed():
    asm = asm_of([mk(0, "R1", "s1", T0, 100, 110, 70, 100, None, None, stopped=25)])
    (row,) = report.pass_rows(report.ReportInputs(asm, {}, None, None, []))
    assert row.stopped_share == 0.25


# ---- stage 5 ----------------------------------------------------------------------------------

def detections_for(asm, leader=False, n=1):
    return {"run_t": [FrameDetections("run_t", f.pos, f.frame_id, f.capture_utc_s, n, leader, 100.0) for f in asm.runs[0].frames]}


def test_repeated_visits_by_segment_with_one_pair_is_not_computed():
    aps = [mk(0, "R1", "s1", T0, 100, 110, 70, 100, None, None), mk(1, "R1", "s1", T0 + 377, 100, 120, 70, 100, None, None)]
    lines = report.repeated_visits(report.pass_rows(report.ReportInputs(asm_of(aps), {}, None, None, [])))
    assert len(lines) == 1 and "not computed" in lines[0] and "1 directed segment driven" in lines[0]


def test_two_passes_within_60_s_are_one_visit_and_the_two_directions_are_not_paired():
    split = [mk(0, "R1", "s1", T0, 100, 110, 70, 100, None, None), mk(1, "R1", "s1", T0 + 160, 100, 110, 70, 100, None, None)]
    rows = report.pass_rows(report.ReportInputs(asm_of(split), {}, None, None, []))
    assert {k: len(v) for k, v in report.segment_visits(rows).items()} == {"s1": 1}        # gap 60 s: one visit
    two_dirs = [mk(0, "R1", "north", T0, 100, 110, 70, 100, None, None), mk(1, "R1", "south", T0 + 600, 100, 120, 70, 100, None, None)]
    rows = report.pass_rows(report.ReportInputs(asm_of(two_dirs), {}, None, None, []))
    assert "not computed" in report.repeated_visits(rows)[0] and "0 directed segments" in report.repeated_visits(rows)[0]


def test_repeated_visit_spread_is_pinned():
    aps = [mk(0, "R1", "a", T0, 100, 110, 70, 100, None, None), mk(1, "R1", "a", T0 + 500, 100, 130, 70, 100, None, None),
           mk(2, "R2", "b", T0 + 1000, 100, 90, 70, 100, None, None), mk(3, "R2", "b", T0 + 1500, 100, 100, 70, 100, None, None)]
    (line,) = report.repeated_visits(report.pass_rows(report.ReportInputs(asm_of(aps), {}, None, None, [])))
    errs = {"a": [0.10, 0.30], "b": [-0.10, 0.0]}
    dev = [e - np.mean(v) for v in errs.values() for e in v]
    sd = np.sqrt(np.sum(np.square(dev)) / 2)
    assert f"{100 * sd:.1f} percentage points" in line and "2 directed segments" in line


def test_driver_offset_uses_moving_time_and_filters_with_stopped_share_and_mean_leader_share():
    def p(i, stopped, jam=1.0):
        return mk(i, f"R{i}", f"s{i}", T0 + 300 * i, 100, 60, 125, 100, None, None, stopped=stopped, jam=jam)

    aps = [p(0, 0), p(1, 4), p(2, 5), p(3, 30), p(4, 0, jam=2.0)]
    asm = asm_of(aps)
    # Leader on every second frame: a mean of about 0.5 per pass, a max of 1.
    dets = detections_for(asm, leader=False)
    rows = report.pass_rows(report.ReportInputs(asm, dets, None, None, []))
    off = report.driver_offset(rows)
    # Passes 0 and 1 qualify (stopped share 0 and 0.04; 5 percent and 0.30 do not, jam 2.0 does not).
    assert off.n_passes == 2
    # Pass 0: moving 100 s over 1000 m against 125 s free flow: 1.25 - 1 = +25%. Pass 1: moving 96 s: 1.30208 - 1.
    exp = np.median([1000 / 100 / (1000 / 125) - 1, 1000 / 96 / (1000 / 125) - 1])
    assert off.offset == pytest.approx(exp)
    half = [FrameDetections("run_t", f.pos, f.frame_id, f.capture_utc_s, 1, f.pos % 2 == 0, 1.0) for f in asm.runs[0].frames]
    mixed = report.pass_rows(report.ReportInputs(asm, {"run_t": half}, None, None, []))
    assert 0.3 < mixed[0].leader_share < 0.7                      # a mean, not 1.0
    assert report.driver_offset(mixed) is None                    # every pass is behind a leader about half the time


def test_stage5_prints_error_with_and_without_the_offset():
    aps = [mk(i, f"R{i}", f"s{i}", T0 + 300 * i, 50, 60, 62.5, 50, None, None) for i in range(6)]
    asm = asm_of(aps)
    text, _ = render(aps, detections=detections_for(asm, leader=False))
    assert "over 6 open-road passes: +25.0%" in text
    assert "without removing the offset: +20.0%" in text and "-4.0%" in text


# ---- routing ----------------------------------------------------------------------------------

def rr(request_id, kind, err_, observed, excl=None):
    return RouteResult(request_id, kind, observed * (1 + err_), observed * 0.9, 1000.0, observed, err_, excl is None, 1.0, 0.0, excl)


def test_routing_keeps_only_current_requests_and_lists_the_rest():
    asm = asm_of(APS)
    reqs = pipeline.route_requests(asm)
    good = [rr(q.request_id, q.kind, 0.1 + 0.01 * i, q.observed_s) for i, q in enumerate(reqs)]
    stale = rr("t-000", "pass", 9.0, 12345.0)             # right id, wrong observed time
    foreign = rr("zzz-999", "pass", 9.0, 100.0)
    inp = report.ReportInputs(asm, {}, None, good[:-1] + [stale, foreign], reqs)
    reasons = report.incomplete_reasons(inp)
    assert any("3 routing rows match no current request" in r for r in reasons) or any("2 routing rows match no current request" in r for r in reasons)
    assert any("cover" in r and "current requests" in r for r in reasons)
    text, _ = report.render(inp, PROV)
    assert "9.0" not in text.split("## Stage 6")[1].split("## Gates")[0].replace("90.0", "")
    assert 9.0 not in [r.signed_error for r in report.valid_routes(inp)]


def test_routing_headline_excludes_routes_off_the_path_and_counts_them():
    asm = asm_of(APS)
    reqs = [q for q in pipeline.route_requests(asm) if q.kind == "pass"]
    vals = [0.10, 0.20, 0.30, 0.40, 0.50, 5.0, 6.0]
    res = [rr(q.request_id, "pass", v, q.observed_s, excl="route_does_not_follow_path" if v > 1 else None) for q, v in zip(reqs, vals)]
    text, _ = render(routing=res, requests=reqs)
    line = next(l for l in text.splitlines() if l.startswith("- pass: median of (HERE routing"))
    assert "over 5 routes: +30.0%" in line and "2 routes excluded" in line
    assert "not resolvable" in line or "to" in line
