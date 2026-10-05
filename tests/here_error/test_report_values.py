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
from here_error.load import PhoneLog
from here_error.pipeline import Assembly, RunData
from here_error.stretches import StretchResult, build_stretches

DUSK = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.timezone.utc).timestamp()
T0 = DUSK - 6000.0
PROV = Provenance("abc", False, (InputFile("x", "y"),), {})


def mk(i, rid, seg, t0, dur, here, ff, sens, arr, shuf, stopped=0.0, jam=1.0, fc=2, matched=1000.0, arrived=True, unc=None):
    pid = f"t-{i:03d}"
    fixes = (fix(t0, 0, 0), fix(t0 + dur, matched, 0))
    p = Pass(pid, "run_t", seg, f"desc {seg}", fc, fixes, 0.0, matched, False, matched, matched, None, rid, f"{rid} label")
    e = lambda h: None if h is None else (h - dur) / dur
    t = PassTimes(pid, 0, t0 - 30, 30.0, arrived, dur, here, ff, e(here), e(ff), jam, 0.99, stopped, dur - stopped,
                  0, e(sens), 0 if arr is not None else None, e(arr), 1, e(shuf), None,
                  here_uncapped_s=here if unc is None else unc, uncapped_signed_error=e(here if unc is None else unc))
    return AnalysedPass(p, t)


def asm_of(aps, n_frames=300):
    frames = tuple(Frame("run_t", k, k, T0 + k * 20.0) for k in range(n_frames))
    phone = PhoneLog("x", (fix(T0, 0, 0), fix(T0 + 7200, 0, 0)), (), (), 0, 0)
    rd = RunData("run_t", Path("."), Path("."), phone, ClockOffset(1.2, 1.19, 1.21, 10), (), frames)
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
    assert "treats each directed segment (passes) or stretch (stretches) as independent" in text and "resampling physical roads" in text
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
    # No frame has a leader, so the stopped share and the jam factor decide which passes qualify.
    dets = detections_for(asm, leader=False)
    rows = report.pass_rows(report.ReportInputs(asm, dets, None, None, []))
    off = report.driver_offset(rows)
    # Passes 0 and 1 qualify, with stopped shares 0 and 0.04. Passes 2 and 3 have stopped shares 0.05 and 0.30,
    # at or above the 0.05 limit, and pass 4 has jam factor 2.0, at the 2.0 limit.
    assert off.n_passes == 2
    # Pass 0: moving 100 s over 1000 m against 125 s free flow: 1.25 - 1 = +25%. Pass 1: moving 96 s: 1.30208 - 1.
    exp = np.median([1000 / 100 / (1000 / 125) - 1, 1000 / 96 / (1000 / 125) - 1])
    assert off.offset == pytest.approx(exp)
    # Leader on every second frame: each pass's leader share is the mean over its frames, about 0.5, not the maximum of 1.
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
    assert any("2 routing rows match no current request" in r for r in reasons)
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
    roads = ["R1", "R1", "R2", "R3", "R4"]
    segs = ["s1", "s2", "s3", "s4", "s5"]
    own = stats.cluster_bootstrap_median([0.10, 0.20, 0.30, 0.40, 0.50], segs)
    assert "resampling physical roads: not resolvable: 4 physical roads;" in line
    assert f"treating each directed segment as independent: {100 * own.lo:+.1f}% to {100 * own.hi:+.1f}%;" in line


# Six stretches of two passes each (50 s apart). Each tuple: (road of pass 1, metres, road of pass 2, metres).
STRETCH_SPEC = [("R1", 1200, "R1", 1200), ("R1", 400, "R2", 1800), ("R3", 1200, "R4", 1200),
                ("R4", 1200, "R4", 1200), ("R5", 1200, "R5", 1200), ("R2", 1200, "R2", 1200)]
EXPECTED_STRETCH_ROADS = ["R1", "R2", "R3", "R4", "R5", "R2"]     # majority of metres; tie goes to the first pass


def stretch_aps(spec=STRETCH_SPEC):
    out, t = [], T0
    for k, (r1, m1, r2, m2) in enumerate(spec):
        out.append(mk(2 * k, r1, f"s{2 * k}", t, 100, 150, 50, 100, None, None, matched=float(m1)))
        out.append(mk(2 * k + 1, r2, f"s{2 * k + 1}", t + 150, 100, 130 + 10 * k, 50, 100, None, None, matched=float(m2)))
        t += 600
    return out


def test_stretch_belongs_to_the_road_with_most_metres_and_a_tie_goes_to_the_first_pass():
    asm = asm_of(stretch_aps())
    by_id = {p.pass_id: p for p in asm.passes}
    assert [report.stretch_road(s, by_id) for s in asm.stretch_result.stretches] == EXPECTED_STRETCH_ROADS


def test_stretch_rows_use_the_road_rule_and_the_independent_column():
    text, _ = render(stretch_aps())
    asm = asm_of(stretch_aps())
    st = asm.stretch_result.stretches
    assert len(st) == 6
    vals = [s.signed_error for s in st]
    road = stats.cluster_bootstrap_median(vals, EXPECTED_STRETCH_ROADS)
    own = stats.cluster_bootstrap_median(vals, [s.stretch_id for s in st])
    cells = table_cells(text, "Stretch: (HERE time")
    assert cells[2] == f"{100 * road.lo:+.1f}% to {100 * road.hi:+.1f}%"
    assert cells[3] == f"{100 * own.lo:+.1f}% to {100 * own.hi:+.1f}%"
    assert cells[2] != cells[3]                    # two stretches share a road, so the two resamplings differ
    assert cells[4] == "6 stretches on 5 physical roads"


def test_stretches_on_few_roads_are_not_resolvable_in_the_road_column():
    spec = [("R1", 1200, "R1", 1200), ("R1", 1200, "R2", 1200), ("R2", 1200, "R2", 1200), ("R1", 1200, "R1", 1200),
            ("R2", 1200, "R2", 1200), ("R1", 1200, "R1", 1200)]
    cells = table_cells(render(stretch_aps(spec))[0], "Stretch: (HERE time")
    assert cells[2] == "not resolvable: 2 physical roads" and " to " in cells[3]


def stretch_routes(asm, errs):
    reqs = pipeline.route_requests(asm)
    sreq = [q for q in reqs if q.kind == "stretch"]
    return reqs, [rr(q.request_id, "stretch", e, q.observed_s) for q, e in zip(sreq, errs)]


def test_routing_stretch_line_follows_the_two_column_road_rule():
    asm = asm_of(stretch_aps())
    reqs, res = stretch_routes(asm, [0.10, 0.25, 0.05, 0.40, 0.30, 0.15])
    text, _ = render(stretch_aps(), routing=res, requests=reqs)
    line = next(l for l in text.splitlines() if l.startswith("- stretch: median of (HERE routing"))
    ids = [s.stretch_id for s in asm.stretch_result.stretches]
    road = stats.cluster_bootstrap_median([r.signed_error for r in res], EXPECTED_STRETCH_ROADS)
    own = stats.cluster_bootstrap_median([r.signed_error for r in res], ids)
    assert f"95% interval resampling physical roads: {100 * road.lo:+.1f}% to {100 * road.hi:+.1f}%;" in line
    assert f"treating each stretch as independent: {100 * own.lo:+.1f}% to {100 * own.hi:+.1f}%;" in line
    assert "whole stretches" not in line


def test_routing_stretch_line_on_few_roads_is_not_resolvable():
    spec = [("R1", 1200, "R1", 1200)] * 6
    asm = asm_of(stretch_aps(spec))
    reqs, res = stretch_routes(asm, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    line = next(l for l in render(stretch_aps(spec), routing=res, requests=reqs)[0].splitlines() if l.startswith("- stretch: median"))
    assert "resampling physical roads: not resolvable: 1 physical road;" in line and "treating each stretch as independent: +" in line


def test_no_printed_road_label_is_a_lone_description():
    text, _ = render()
    for ap in APS:
        assert ap.p.road_label != ap.p.description
    assert "(a segment's description names the cross street at its end, not the road)" in text


def test_a_repeated_request_id_counts_once_and_is_listed():
    asm = asm_of(APS)
    reqs = pipeline.route_requests(asm)
    good = [rr(q.request_id, q.kind, 0.1, q.observed_s) for q in reqs]
    first = good[0]
    dup = rr(first.request_id, first.kind, 7.7, first.observed_s)
    inp = report.ReportInputs(asm, {}, None, good + [dup], reqs)
    kept = report.valid_routes(inp)
    assert len(kept) == len(reqs) and kept[0].signed_error == 0.1 and 7.7 not in [r.signed_error for r in kept]
    assert any("1 routing rows repeat a request id" in r for r in report.incomplete_reasons(inp))
    clean = report.ReportInputs(asm, {}, None, good, reqs)
    assert not any("repeat" in r for r in report.incomplete_reasons(clean))


# ---- clustering: roads against segments against passes ------------------------------------------

def three_roads_many_segments(n=9):
    return [mk(i, f"R{i % 3}", f"s{i}", T0 + 300 * i, 100, 100 + 6 * i, 70, 100, None, None) for i in range(n)]


def test_stage4_groups_resample_roads_not_segments():
    text, _ = render(three_roads_many_segments())
    line = next(l for l in text.splitlines() if l.strip().startswith("- lowest"))
    assert "resampling physical roads: not resolvable: 3 physical roads" in line


def test_driver_offset_intervals_resample_roads_not_segments():
    aps = [mk(i, f"R{i % 3}", f"s{i}", T0 + 300 * i, 50, 60 + i, 62.5, 50, None, None) for i in range(9)]
    asm = asm_of(aps)
    text, _ = render(aps, detections=detections_for(asm, leader=False))
    line = next(l for l in text.splitlines() if l.startswith("- Median signed error of HERE time without removing"))
    assert line.count("not resolvable: 3 physical roads") == 2


def test_routing_pass_road_column_resamples_roads_not_segments():
    aps = three_roads_many_segments()
    asm = asm_of(aps)
    reqs = [q for q in pipeline.route_requests(asm) if q.kind == "pass"]
    res = [rr(q.request_id, "pass", 0.05 * i, q.observed_s) for i, q in enumerate(reqs)]
    line = next(l for l in render(aps, routing=res, requests=reqs)[0].splitlines() if l.startswith("- pass: median of (HERE routing"))
    assert "resampling physical roads: not resolvable: 3 physical roads;" in line
    assert "treating each directed segment as independent: -" in line or "treating each directed segment as independent: +" in line


def test_headline_segment_column_resamples_segments_not_passes():
    # Six passes, but only four directed segments (one segment holds three passes).
    aps = [mk(i, f"R{i}", f"s{min(i, 3)}", T0 + 300 * i, 100, 100 + 9 * i, 70, 100, None, None) for i in range(6)]
    cells = table_cells(render(aps)[0], "Pass: (HERE time")
    assert cells[3] == "not resolvable: 4 directed segments"
    assert cells[4].startswith("6 passes on 6 physical roads and 4 directed segments")


def test_per_road_table_counts_distinct_segments_and_uses_the_median():
    aps = [mk(0, "R1", "sA", T0, 100, 100, 70, 100, None, None), mk(1, "R1", "sA", T0 + 300, 100, 100, 70, 100, None, None),
           mk(2, "R1", "sB", T0 + 600, 100, 190, 70, 100, None, None), mk(3, "R2", "sC", T0 + 900, 100, 120, 70, 100, None, None)]
    text, _ = render(aps)
    cells = [c.strip() for c in next(l for l in text.splitlines() if l.startswith("| R1 |")).strip("|").split("|")]
    assert cells[2] == "3" and cells[3] == "2"            # three passes on two distinct segments
    assert cells[4] == "+0.0%"                            # the median of 0, 0, +90%; the mean would be +30%


def test_v4_paragraph_prints_the_shuffled_value_in_its_place():
    asm = asm_of(APS)
    asm = Assembly(asm.runs, asm.passes, asm.times, asm.catalogues, asm.stretch_result, V4Result(5, 0.05, 0.37), ())
    inp = report.ReportInputs(asm, {}, None, None, [])
    text, _ = report.render(inp, PROV)
    assert "5.0% with the real reading, 37.0% with the shuffled reading" in text


def test_repeated_visit_sentence_names_the_segment_by_its_end_and_gives_the_road():
    aps = [mk(0, "R9", "s1", T0, 100, 110, 70, 100, None, None), mk(1, "R9", "s1", T0 + 377, 100, 120, 70, 100, None, None)]
    (line,) = report.repeated_visits(report.pass_rows(report.ReportInputs(asm_of(aps), {}, None, None, [])))
    assert "the directed segment ending at desc s1, road R9, 1000 m matched" in line


def test_passes_csv_carries_the_catalogue_size_of_each_road(tmp_path):
    import csv

    from here_error import corridors
    from here_fixtures import segment

    segs = [segment("A", [(0, 0), (300, 0)]), segment("B", [(300, 0), (600, 0)]), segment("C", [(600, 0), (900, 0)])]
    roads = corridors.build_roads(segs)
    assert len(roads.roads) == 1
    aps = [mk(0, roads.roads[0].road_id, segs[0].key, T0, 100, 110, 70, 100, None, None)]
    a = asm_of(aps)
    a = Assembly(a.runs, a.passes, a.times, a.catalogues, a.stretch_result, a.v4, (), roads)
    report.write_passes_csv(a, report.pass_rows(report.ReportInputs(a, {}, None, None, [])), tmp_path / "p.csv")
    (row,) = list(csv.DictReader((tmp_path / "p.csv").open()))
    assert row["road_catalogue_segments"] == "3"


def test_removing_a_negative_driver_offset_moves_the_median_as_printed():
    # Moving 1000 m in 150 s against 125 s free flow: offset -16.67 percent (the car is slower than free flow).
    aps = [mk(i, f"R{i}", f"s{i}", T0 + 300 * i, 150, 150, 125, 150, None, None) for i in range(6)]
    asm = asm_of(aps)
    text, _ = render(aps, detections=detections_for(asm, leader=False))
    assert "over 6 open-road passes: -16.7%" in text
    # Unadjusted error 0.0%. With the offset removed the observed time is 150 * (1 - 1/6) = 125 s, which HERE's 150 s exceeds by 20.0%.
    assert "removing the driver offset moves the median signed error from +0.0% to +20.0%" in text
    assert "upper bound" not in text


def test_summary_states_limits_and_the_two_column_and_stretch_rules():
    text, _ = render(APS)
    assert "## Limits" in text
    assert "One driver, one car, one evening (2026-09-08, 17:35 to 19:35 EDT)." in text
    assert "The 7 analysed passes lie on 6 physical roads; the largest holds 2 of them." in text
    assert ("HERE Traffic Flow speeds are HERE's input to routing, not its routing output; only the routing section measures "
            "routing travel time, and its past departure times get HERE's typical traffic for that weekday and hour.") in text
    assert text.index("## Limits") < text.index("## Headline table")
    assert ("The second interval column treats each directed segment, or each stretch, as independent. Segments along one road "
            "at one time share HERE's behaviour and the evening's traffic, so that column understates the uncertainty.") in text
    assert ("A stretch belongs to the road holding most of its matched metres, so the stretch rows count 0 physical roads "
            "while the pass rows count 6.") in text


def test_limits_numbers_are_computed_from_the_passes():
    aps = [mk(i, "RA" if i < 5 else "RB", f"s{i}", T0 + 300 * i, 100, 110, 70, 100, None, None) for i in range(8)]
    text, _ = render(aps)
    assert "The 8 analysed passes lie on 2 physical roads; the largest holds 5 of them." in text


def test_limits_time_span_is_derived_from_the_gps_fixes_in_local_time():
    from dataclasses import replace

    asm = asm_of(APS)
    other = replace(asm.runs[0], phone=PhoneLog("y", (fix(DUSK - 12 * 3600, 0, 0), fix(DUSK + 3600 + 86400, 0, 0)), (), (), 0, 0))
    moved = Assembly((asm.runs[0], other), asm.passes, asm.times, asm.catalogues, asm.stretch_result, asm.v4, ())
    # First fix 2026-09-08 07:15 EDT; last fix 2026-09-09 20:15 EDT: two dates.
    assert report.drive_span_text(moved) == "2026-09-08 07:15 to 2026-09-09 20:15 EDT"
    empty = replace(asm.runs[0], phone=PhoneLog("z", (), (), (), 0, 0))
    assert "unavailable" in report.drive_span_text(Assembly((empty,), asm.passes, asm.times, asm.catalogues, asm.stretch_result, asm.v4, ()))


def test_limits_states_the_dusk_split_as_a_fixed_local_time(monkeypatch):
    text, _ = render(APS)
    assert "The daylight/dark split is a fixed local time, 19:15, chosen for this drive." in text
    monkeypatch.setattr(params, "DUSK_SPLIT_LOCAL", (18, 5))
    assert "fixed local time, 18:05, chosen for this drive." in render(APS)[0]


def test_stage4_group_lines_show_both_intervals_when_there_are_enough_passes():
    aps = three_roads_many_segments(9)
    text, _ = render(aps)
    line = next(l for l in text.splitlines() if l.strip().startswith("- lowest"))
    errs = [a.t.signed_error for a in aps]
    seg = stats.cluster_bootstrap_median(errs, [a.p.segment_key for a in aps])
    assert "resampling physical roads: not resolvable: 3 physical roads;" in line
    assert f"95% interval, treats each directed segment as independent: {100 * seg.lo:+.1f}% to {100 * seg.hi:+.1f}%" in line
    few = [mk(i, "R1", f"s{i}", T0 + 300 * i, 100, 100 + 6 * i, 70, 100, None, None) for i in range(4)]
    fline = next(l for l in render(few)[0].splitlines() if l.strip().startswith("- lowest"))
    assert "directed-segment interval needs at least 5 passes" in fline and "treats each directed segment" not in fline


def test_uncapped_rows_are_pinned_in_both_interval_columns():
    # Capped HERE time 110 s, uncapped 140 s, free flow 70 s, observed 100 s on every pass but with distinct per-pass values.
    aps = [mk(i, f"R{i % 6}", f"s{i}", T0 + 300 * i, 100, 100 + 4 * i, 70 + i, 100, None, None, unc=100 + 9 * i) for i in range(12)]
    text, _ = render(aps)
    unc = [(9 * i) / 100 for i in range(12)]
    cells = table_cells(text, "Pass, sensitivity: HERE time from speedUncapped")
    assert cells[1] == f"{100 * np.median(unc):+.1f}%"
    road = stats.cluster_bootstrap_median(unc, [f"R{i % 6}" for i in range(12)])
    seg = stats.cluster_bootstrap_median(unc, [f"s{i}" for i in range(12)])
    assert cells[2] == f"{100 * road.lo:+.1f}% to {100 * road.hi:+.1f}%" and cells[3] == f"{100 * seg.lo:+.1f}% to {100 * seg.hi:+.1f}%"
    diff = [abs(9 * i / 100) - abs((70 + i - 100) / 100) for i in range(12)]
    cells = table_cells(text, "Pass, sensitivity: abs(HERE uncapped error)")
    assert cells[1] == f"{100 * np.median(diff):+.1f} pp"
    assert cells[1] != table_cells(text, "Pass: abs(HERE error)")[1]


def test_uncapped_stretch_rows_use_the_stretch_uncapped_time():
    aps = stretch_aps()
    text, _ = render(aps)
    asm = asm_of(aps)
    st = asm.stretch_result.stretches
    vals = [s.uncapped_error for s in st]
    cells = table_cells(text, "Stretch, sensitivity: HERE time from speedUncapped")
    own = stats.cluster_bootstrap_median(vals, [s.stretch_id for s in st])
    road = stats.cluster_bootstrap_median(vals, EXPECTED_STRETCH_ROADS)
    assert cells[1] == f"{100 * np.median(vals):+.1f}%"
    assert cells[2] == f"{100 * road.lo:+.1f}% to {100 * road.hi:+.1f}%" and cells[3] == f"{100 * own.lo:+.1f}% to {100 * own.hi:+.1f}%"
    d = [abs(s.uncapped_error) - abs(s.free_flow_error) for s in st]
    assert table_cells(text, "Stretch, sensitivity: abs(HERE uncapped")[1] == f"{100 * np.median(d):+.1f} pp"


def test_summary_names_the_speed_fields_and_counts_the_fallbacks():
    from here_error.models import HereBody
    from here_fixtures import segment

    text, _ = render(APS)
    assert ("HERE time uses HERE's `speed` field, which is capped at the speed limit, while free-flow time uses `freeFlow`, which is not; "
            "the speedUncapped rows use HERE's `speedUncapped` field.") in text
    seg_fb = segment("A", [(0, 0), (500, 0)])                      # no speedUncapped: falls back
    seg_ok = segment("B", [(500, 0), (1000, 0)])
    from dataclasses import replace
    seg_ok = replace(seg_ok, speed_uncapped_mps=12.0, uncapped_fallback=False)
    body = HereBody("run_t", 0, "f", 0.0, 0.0, 0.0, 0.0, (seg_fb, seg_ok))
    asm = asm_of(APS)
    asm = Assembly((replace(asm.runs[0], bodies=(body,)),), asm.passes, asm.times, asm.catalogues, asm.stretch_result, asm.v4, ())
    assert report.uncapped_fallbacks(asm) == (1, 2)
    t2, _ = report.render(report.ReportInputs(asm, {}, None, None, []), PROV)
    assert "no speedUncapped field, where the capped speed was used in its place: 1 of 2." in t2


def test_routing_section_says_stretch_comparison_covers_the_gaps():
    text, _ = render(APS)
    assert ("A stretch's routing comparison runs from its first fix to its last, including the unmatched gaps between passes, "
            "so it covers more ground than the stage-2 stretch rows.") in text


def test_no_route_rows_are_counted_among_the_routing_exclusions():
    nan = float("nan")
    asm = asm_of(APS)
    reqs = [q for q in pipeline.route_requests(asm) if q.kind == "pass"]
    res = [rr(q.request_id, "pass", 0.1 * (i + 1), q.observed_s) for i, q in enumerate(reqs)]
    bad = reqs[0]
    res[0] = RouteResult(bad.request_id, "pass", nan, nan, nan, bad.observed_s, nan, False, 0.0, nan, "no_route")
    res[1] = rr(reqs[1].request_id, "pass", 9.0, reqs[1].observed_s, excl="route_does_not_follow_path")
    text, _ = render(routing=res, requests=reqs)
    line = next(l for l in text.splitlines() if l.startswith("- pass: median of (HERE routing"))
    assert f"over {len(reqs) - 2} routes:" in line
    assert "2 routes excluded (no_route 1, route_does_not_follow_path 1)" in line
