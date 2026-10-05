"""Stages 4, 5 and 7: conditions, driver offset, and the summary with its headline table.

A summary never presents a headline from partial inputs without saying so. Every input that is
missing or incomplete (detections for every frame, the scored labels, the routing results) is
listed under STATUS, and the sections that depend on it print "not computed".
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from here_error import params, pipeline, stats
from here_error.clock import is_after_dusk
from here_error.here_times import AnalysedPass
from here_error.labels import HalfScore
from here_error.models import FrameDetections, Provenance, RouteRequest, RouteResult

NOT_COMPUTED = "not computed"
POSITIVE_MEANS = "a positive value means HERE's time is longer than the time the car took"


@dataclass(frozen=True)
class ReportInputs:
    asm: pipeline.Assembly
    detections: dict[str, list[FrameDetections]]
    label_scores: list[HalfScore] | None
    routing: list[RouteResult] | None
    requests: list[RouteRequest]


@dataclass(frozen=True)
class PassRow:
    """One analysed pass with the conditions measured while it was driven."""

    a: AnalysedPass
    road_class: str
    stopped_share: float
    data_age_s: float
    dark: bool
    n_frames: int | None
    mean_vehicles: float | None
    leader_share: float | None

    @property
    def road(self) -> str:
        return self.a.p.description


def read_label_scores(path: Path) -> list[HalfScore]:
    return [HalfScore(**d) for d in json.loads(path.read_text())]


def write_label_scores(scores: list[HalfScore], path: Path) -> None:
    from dataclasses import asdict

    path.write_text(json.dumps([asdict(s) for s in scores], indent=2))


def camera_coverage(inputs: ReportInputs) -> list[str]:
    """Reasons the detections do not cover every frame of every run."""
    out = []
    for r in inputs.asm.runs:
        got = {d.pos for d in inputs.detections.get(r.run, [])}
        if len(got) != len(r.frames) or got != {f.pos for f in r.frames}:
            out.append(f"detections cover {len(got)} of {len(r.frames)} frames of {r.run}")
    return out


def incomplete_reasons(inputs: ReportInputs) -> list[str]:
    reasons = camera_coverage(inputs)
    if inputs.label_scores is None:
        reasons.append("detector labels (gate V6) are not scored")
    if inputs.routing is None:
        reasons.append("routing results are missing")
    else:
        have = {r.request_id for r in inputs.routing}
        need = {q.request_id for q in inputs.requests}
        if need - have:
            reasons.append(f"routing results cover {len(need & have)} of {len(need)} requests")
    return reasons


def daylight_only(inputs: ReportInputs) -> bool:
    """Gate V6 rule: when dusk recall is below 0.7 the camera results use daylight frames only."""
    if inputs.label_scores is None:
        return False
    after = [s for s in inputs.label_scores if s.half == "after_dusk"]
    return bool(after) and after[0].recall_all is not None and after[0].recall_all < 0.7


def _road_class(fc: int | None) -> str:
    return "unknown" if fc is None else ("class 2" if fc <= 2 else "class 3-4")


def pass_rows(inputs: ReportInputs) -> list[PassRow]:
    cam_ok = not camera_coverage(inputs)
    day_only = daylight_only(inputs)
    dets = {}
    for run, lst in inputs.detections.items():
        lst = sorted(lst, key=lambda d: d.utc_s)
        dets[run] = (np.array([d.utc_s for d in lst]), lst)
    rows = []
    for a in inputs.asm.analysed:
        p, t = a.p, a.t
        n = mean_v = leader = None
        if cam_ok and p.run in dets:
            times, lst = dets[p.run]
            lo = np.searchsorted(times, p.first_utc_s, side="left")
            hi = np.searchsorted(times, p.last_utc_s, side="right")
            sel = [d for d in lst[lo:hi] if not (day_only and is_after_dusk(d.utc_s))]
            if sel:
                n = len(sel)
                mean_v = float(np.mean([d.n_vehicles for d in sel]))
                leader = float(np.mean([d.leader for d in sel]))
        mid = (p.first_utc_s + p.last_utc_s) / 2
        rows.append(PassRow(a, _road_class(p.functional_class), t.stopped_s / t.observed_s,
                            t.reading_age_s, is_after_dusk(mid), n, mean_v, leader))
    return rows


def pct(x: float, signed: bool = True) -> str:
    return f"{100 * x:+.1f}%" if signed else f"{100 * x:.1f}%"


def interval(ci: stats.MedianCI) -> str:
    return f"{pct(ci.lo)} to {pct(ci.hi)}"


def _ci(values, clusters):
    return stats.cluster_bootstrap_median(values, clusters) if len(values) else None


def pp(x: float) -> str:
    """A difference of two signed errors, in percentage points of travel time."""
    return f"{100 * x:+.1f} pp"


def _row(label, ci, n_unit, fmt=pct):
    if ci is None:
        return f"| {label} | {NOT_COMPUTED}: no data | | |"
    return f"| {label} | {fmt(ci.median)} | {fmt(ci.lo)} to {fmt(ci.hi)} | {ci.n} {n_unit} on {ci.n_clusters} clusters |"


def headline(rows: list[PassRow], stretches) -> list[str]:
    roads = [r.road for r in rows]
    he = [r.a.t.signed_error for r in rows]
    ff = [r.a.t.free_flow_error for r in rows]
    diff = [abs(r.a.t.signed_error) - abs(r.a.t.free_flow_error) for r in rows]
    sens = [r.a.t.sensitivity_signed_error for r in rows if r.a.t.sensitivity_signed_error is not None]
    sens_roads = [r.road for r in rows if r.a.t.sensitivity_signed_error is not None]
    arr = [r.a.t.arrived_signed_error for r in rows if r.a.t.arrived_signed_error is not None]
    arr_roads = [r.road for r in rows if r.a.t.arrived_signed_error is not None]
    sids = [s.stretch_id for s in stretches]
    out = [
        "| Quantity compared | Median | 95% interval | n |",
        "|---|---|---|---|",
        _row("Pass: (HERE time - observed time) / observed time, HERE reading with the latest data time before the pass", _ci(he, roads), "passes"),
        _row("Pass: (free-flow time - observed time) / observed time", _ci(ff, roads), "passes"),
        _row("Pass: abs(HERE error) - abs(free-flow error), in percentage points; negative means HERE's live speed is closer to the observed time", _ci(diff, roads), "passes", pp),
        _row("Pass, sensitivity: HERE reading with data time nearest the pass midpoint", _ci(sens, sens_roads), "passes"),
        _row("Pass, sensitivity: latest reading that had also arrived before the pass began", _ci(arr, arr_roads), "passes"),
        _row("Stretch: (HERE time - observed time) / observed time", _ci([s.signed_error for s in stretches], sids), "stretches"),
        _row("Stretch: (free-flow time - observed time) / observed time", _ci([s.free_flow_error for s in stretches], sids), "stretches"),
        _row("Stretch: abs(HERE error) - abs(free-flow error), in percentage points", _ci([abs(s.signed_error) - abs(s.free_flow_error) for s in stretches], sids), "stretches", pp),
    ]
    return out


def _groups(rows: list[PassRow], values: list[float | None], name: str) -> list[str]:
    ok = [(r, v) for r, v in zip(rows, values) if v is not None]
    if len(ok) < 3:
        return [f"- {name}: {NOT_COMPUTED}: fewer than 3 passes have a value"]
    g = stats.thirds([v for _, v in ok])
    out = [f"- {name}, thirds of passes by the value (lowest third first):"]
    for k, label in enumerate(("lowest", "middle", "highest")):
        sel = [(r, v) for (r, v), gi in zip(ok, g) if gi == k]
        out.append(_group_line(label, [r for r, _ in sel], f"values {min(v for _, v in sel):.2f} to {max(v for _, v in sel):.2f}" if sel else ""))
    return out


def _group_line(label, sel: list[PassRow], extra="") -> str:
    if not sel:
        return f"    - {label}: 0 passes"
    errs = [r.a.t.signed_error for r in sel]
    text = f"    - {label}{(' (' + extra + ')') if extra else ''}: {len(sel)} passes, median {pct(float(np.median(errs)))}"
    if len(sel) >= params.MIN_N_FOR_INTERVAL:
        text += f", 95% interval {interval(stats.cluster_bootstrap_median(errs, [r.road for r in sel]))}"
    return text


def _classes(rows, key, name) -> list[str]:
    out = [f"- {name}:"]
    for v in sorted({key(r) for r in rows}, key=str):
        out.append(_group_line(str(v), [r for r in rows if key(r) == v]))
    return out


def stage4(rows: list[PassRow], camera_note: str | None) -> list[str]:
    out = []
    out += _groups(rows, [r.stopped_share for r in rows], "share of the pass's time stopped (GPS speed below 1 m/s)")
    out += _groups(rows, [r.data_age_s for r in rows], "age of the HERE data at the pass's first fix, seconds")
    out += _classes(rows, lambda r: r.road_class, "road class (HERE functional class)")
    out += _classes(rows, lambda r: "dark (after 19:15 local)" if r.dark else "daylight", "daylight or dark, by pass midpoint")
    if camera_note:
        out.append(f"- camera conditions: {NOT_COMPUTED}: {camera_note}")
        return out
    out += _groups(rows, [r.mean_vehicles for r in rows], "mean vehicles detected per frame")
    out += _groups(rows, [r.leader_share for r in rows], "share of frames with a vehicle ahead in the ego lane")
    pairs = [(r.a.t.jam_factor, r.mean_vehicles) for r in rows if r.a.t.jam_factor is not None and r.mean_vehicles is not None]
    if len(pairs) >= 3:
        rho = stats.spearman([a for a, _ in pairs], [b for _, b in pairs])
        out.append(f"- Spearman rank correlation between HERE's jam factor (higher means more congested) and the mean "
                   f"vehicles per frame, over {len(pairs)} passes: {rho:+.2f}; a positive value means passes where HERE "
                   f"reports more congestion are passes where the camera sees more vehicles.")
    else:
        out.append(f"- jam factor against camera count: {NOT_COMPUTED}: fewer than 3 passes")
    return out


@dataclass(frozen=True)
class DriverOffset:
    n_passes: int
    offset: float


def driver_offset(rows: list[PassRow]) -> DriverOffset | None:
    """Median of (moving speed / HERE free-flow speed - 1) over passes with open road and no stops."""
    vals = []
    for r in rows:
        t, p = r.a.t, r.a.p
        if (r.leader_share is None or r.leader_share >= params.OFFSET_MAX_LEADER_SHARE
                or r.stopped_share >= params.OFFSET_MAX_STOPPED_SHARE
                or t.jam_factor is None or t.jam_factor >= params.OFFSET_MAX_JAM or t.moving_s <= 0):
            continue
        vals.append((p.matched_m / t.moving_s) / (p.matched_m / t.free_flow_s) - 1.0)
    return DriverOffset(len(vals), float(np.median(vals))) if vals else None


def stage5(rows: list[PassRow], camera_note: str | None) -> list[str]:
    out = []
    by_road: dict[str, list[PassRow]] = {}
    for r in rows:
        by_road.setdefault(r.road, []).append(r)
    rep = {k: v for k, v in by_road.items() if len(v) >= 2}
    if rep:
        dev_err = [r.a.t.signed_error - np.mean([x.a.t.signed_error for x in v]) for v in rep.values() for r in v]
        spd = lambda r: r.a.p.matched_m / r.a.t.here_s
        dev_spd = [(spd(r) - np.mean([spd(x) for x in v])) / np.mean([spd(x) for x in v]) for v in rep.values() for r in v]
        dof = sum(len(v) - 1 for v in rep.values())
        out.append(f"- Repeated passes: {len(rep)} roads have 2 or more analysed passes ({sum(len(v) for v in rep.values())} passes). "
                   f"Within-road standard deviation of the signed error of HERE time: {100 * np.sqrt(np.sum(np.square(dev_err)) / dof):.1f} percentage points; "
                   f"within-road standard deviation of HERE's mean speed over the matched portion: {100 * np.sqrt(np.sum(np.square(dev_spd)) / dof):.1f}% of the road's mean HERE speed.")
    else:
        out.append(f"- Repeated passes: {NOT_COMPUTED}: no road has 2 analysed passes")
    if camera_note:
        out.append(f"- driver offset: {NOT_COMPUTED}: {camera_note}")
        return out
    off = driver_offset(rows)
    if off is None:
        out.append(f"- driver offset: {NOT_COMPUTED}: no pass has leader share below {params.OFFSET_MAX_LEADER_SHARE}, "
                   f"stopped share below {params.OFFSET_MAX_STOPPED_SHARE} and jam factor below {params.OFFSET_MAX_JAM}")
        return out
    out.append(f"- Driver offset: median of (moving speed / HERE free-flow speed - 1) over {off.n_passes} open-road passes: "
               f"{pct(off.offset)}; a positive value means the car moved faster than HERE's free-flow speed.")
    adj = []
    for r in rows:
        t = r.a.t
        obs = t.stopped_s + t.moving_s * (1 + off.offset)
        adj.append((t.here_s - obs) / obs)
    roads = [r.road for r in rows]
    raw = stats.cluster_bootstrap_median([r.a.t.signed_error for r in rows], roads)
    fixed = stats.cluster_bootstrap_median(adj, roads)
    out.append(f"- Median signed error of HERE time without removing the offset: {pct(raw.median)} ({interval(raw)}); "
               f"with the offset removed from the observed time (stopped time + moving time x (1 + offset)): "
               f"{pct(fixed.median)} ({interval(fixed)}). With one driver the offset is an estimate, not a correction; "
               f"without it the first figure is an upper bound on HERE's error.")
    return out


def routing_section(inputs: ReportInputs, rows: list[PassRow]) -> list[str]:
    if inputs.routing is None:
        return [f"- {NOT_COMPUTED}: no routing results"]
    road_of = {r.a.p.pass_id: r.a.p.description for r in rows}
    out = []
    for kind in ("pass", "stretch"):
        rs = [r for r in inputs.routing if r.kind == kind]
        ok = [r for r in rs if r.exclusion is None]
        if not ok:
            out.append(f"- {kind}: {NOT_COMPUTED}: {len(rs)} results, none follow the driven path")
            continue
        ci = stats.cluster_bootstrap_median([r.signed_error for r in ok], [road_of.get(r.request_id, r.request_id) for r in ok])
        out.append(f"- {kind}: median of (HERE routing duration - observed time) / observed time over {len(ok)} routes "
                   f"(clusters: {ci.n_clusters}): {pct(ci.median)}, 95% interval {interval(ci)}; {POSITIVE_MEANS}. "
                   f"{len(rs) - len(ok)} routes excluded because the route did not follow the driven path.")
        nt = [(r.no_traffic_duration_s - r.observed_s) / r.observed_s for r in ok if r.no_traffic_duration_s is not None]
        if nt:
            out.append(f"  - median of (HERE no-traffic duration - observed time) / observed time: {pct(float(np.median(nt)))}")
    return out


def render(inputs: ReportInputs, prov: Provenance) -> tuple[str, list[str]]:
    asm = inputs.asm
    reasons = incomplete_reasons(inputs)
    rows = pass_rows(inputs)
    cam_reasons = camera_coverage(inputs)
    camera_note = ("; ".join(cam_reasons)) if cam_reasons else None
    L: list[str] = []
    L.append("# HERE travel-time error from the 2026-09-08 drives")
    L.append("")
    if reasons:
        L.append("**STATUS: INCOMPLETE.** The numbers below are not final. Missing or partial inputs:")
        L += [f"- {r}" for r in reasons]
    else:
        L.append("**STATUS: COMPLETE.**")
    L.append("")
    L.append(f"Provenance: git commit {prov.git_commit}, working tree {'dirty' if prov.dirty else 'clean' if prov.dirty is False else 'state unknown'}, "
             f"{len(prov.inputs)} input files hashed (listed in the provenance sidecar).")
    L.append("")
    L.append(f"Signed error = (HERE time - observed time) / observed time; {POSITIVE_MEANS}. "
             "Intervals are 95% percentile intervals from a bootstrap that resamples whole roads (passes) or whole stretches.")
    L.append("")
    L.append("## Headline table")
    if reasons:
        L.append("Printed while the report is INCOMPLETE. These rows use only GPS and HERE data, which are present; "
                 "the rows that need the missing inputs say so below.")
        L.append("")
    L += headline(rows, asm.stretch_result.stretches)
    L.append("")
    sd = float(np.std([r.a.t.signed_error for r in rows], ddof=1)) if len(rows) > 1 else float("nan")
    n_roads = len({r.road for r in rows})
    L.append(f"Measured standard deviation of the per-pass signed error: {100 * sd:.1f} percentage points over {len(rows)} passes. "
             f"With {n_roads} roads as independent clusters, the implied half-width of the 95% interval on the median is "
             f"{100 * stats.median_half_width(sd, n_roads):.1f} percentage points; a bias smaller than that is not resolved.")
    L.append("")
    L.append("## Control V4: time-shuffled HERE reading")
    v4 = asm.v4
    if v4.n:
        L.append(f"Over {v4.n} passes with a reading of the same segment at least {params.SHUFFLE_MIN_S:.0f} s away: median |signed error| "
                 f"{pct(v4.median_abs_real, False)} with the real reading, {pct(v4.median_abs_shuffled, False)} with the shuffled reading. "
                 f"If the second does not exceed the first, this sample cannot resolve HERE's real-time signal.")
    else:
        L.append(f"{NOT_COMPUTED}: no pass has a reading of its segment at least {params.SHUFFLE_MIN_S:.0f} s away.")
    L.append("")
    L.append("## Stage 4: signed error by condition")
    if daylight_only(inputs):
        L.append("Gate V6: dusk recall is below 0.7, so camera conditions use daylight frames only.")
    elif inputs.label_scores is None:
        L.append("Gate V6 is not scored, so camera conditions are provisional.")
    L += stage4(rows, camera_note)
    L.append("")
    L.append("## Stage 5: separating HERE's error from this driver")
    L += stage5(rows, camera_note)
    L.append("")
    L.append("## Stage 6: HERE Routing v8")
    L.append("A past departure time makes HERE use typical traffic for that weekday and clock time, not the traffic recorded that day.")
    L += routing_section(inputs, rows)
    L.append("")
    L.append("## Gates and exclusions")
    for r in asm.runs:
        o = r.offset
        L.append(f"- V1 {r.run}: phone clock minus GPS UTC median {o.median_s:.3f} s, 5th to 95th percentile spread {o.spread_s:.3f} s "
                 f"over {o.n} fixes ({'passes' if o.passes_gate else 'FAILS'} the limit of {params.V1_MAX_SPREAD_S} s).")
    flagged, total, on_flagged = pipeline.length_flag_counts(asm)
    L.append(f"- Passes found {len(asm.passes)}; with matched length at least {params.MIN_PASS_M:.0f} m: "
             f"{sum(p.matched_m >= params.MIN_PASS_M for p in asm.passes)}; analysed: {len(rows)}.")
    ex = pipeline.exclusion_counts(asm)
    L.append("- Exclusions by reason: " + (", ".join(f"{k} {v}" for k, v in sorted(ex.items())) if ex else "none") + ".")
    L.append(f"- Segments whose stated length differs from the shape length by more than {100 * params.SEGMENT_LENGTH_TOL:.0f}%: "
             f"{flagged} of {total}; analysed passes on such segments: {on_flagged}.")
    arrived_late = sum(1 for r in rows if r.a.t.reading_arrived_before_start is False)
    L.append(f"- Analysed passes whose primary reading had not yet arrived at the pass's first fix: {arrived_late} of {len(rows)}.")
    sr = asm.stretch_result
    L.append(f"- Stretches {len(sr.stretches)}; passes dropped from stretches {sr.dropped_passes} ({sr.dropped_m:.0f} m).")
    if inputs.label_scores:
        for s in inputs.label_scores:
            f = lambda x: "n/a" if x is None else f"{x:.2f}"
            L.append(f"- V6 {s.half}: {s.n_frames} frames; precision {f(s.precision_all)} and recall {f(s.recall_all)} counting parked vehicles, "
                     f"precision {f(s.precision_moving)} and recall {f(s.recall_moving)} counting moving vehicles only; leader-flag accuracy {f(s.leader_accuracy)}.")
    L.append("")
    return "\n".join(L), reasons


PASS_COLUMNS = (
    "pass_id", "run", "road", "functional_class", "first_utc_s", "last_utc_s", "n_fixes", "chain_start_m", "chain_end_m",
    "matched_m", "observed_s", "path_positions_m", "path_speed_m", "v3_rel_diff", "segment_length_flag", "exclusion",
    "reading_seq", "reading_age_s", "reading_arrived_before_start", "here_s", "free_flow_s", "signed_error",
    "free_flow_error", "sensitivity_signed_error", "arrived_signed_error", "shuffled_signed_error", "jam_factor",
    "stopped_s", "moving_s", "n_frames", "mean_vehicles", "leader_share",
)


def write_passes_csv(asm: pipeline.Assembly, rows: list[PassRow], path: Path) -> None:
    cond = {r.a.p.pass_id: r for r in rows}
    times = {t.pass_id: t for t in asm.times}
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PASS_COLUMNS)
        w.writeheader()
        for p in asm.passes:
            t = times[p.pass_id]
            c = cond.get(p.pass_id)
            d = dict(
                pass_id=p.pass_id, run=p.run, road=p.description, functional_class=p.functional_class,
                first_utc_s=p.first_utc_s, last_utc_s=p.last_utc_s, n_fixes=len(p.fixes),
                chain_start_m=p.chain_start_m, chain_end_m=p.chain_end_m, matched_m=p.matched_m, observed_s=p.observed_s,
                path_positions_m=p.path_positions_m, path_speed_m=p.path_speed_m, v3_rel_diff=p.v3_rel_diff,
                segment_length_flag=int(p.segment_length_flag), exclusion=t.exclusion, reading_seq=t.reading_seq,
                reading_age_s=t.reading_age_s, reading_arrived_before_start=t.reading_arrived_before_start,
                here_s=t.here_s, free_flow_s=t.free_flow_s, signed_error=t.signed_error, free_flow_error=t.free_flow_error,
                sensitivity_signed_error=t.sensitivity_signed_error, arrived_signed_error=t.arrived_signed_error,
                shuffled_signed_error=t.shuffled_signed_error, jam_factor=t.jam_factor, stopped_s=t.stopped_s,
                moving_s=t.moving_s, n_frames=c.n_frames if c else None, mean_vehicles=c.mean_vehicles if c else None,
                leader_share=c.leader_share if c else None,
            )
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})


STRETCH_COLUMNS = ("stretch_id", "run", "pass_ids", "start_utc_s", "end_utc_s", "matched_m", "observed_s", "here_s",
                   "free_flow_s", "stopped_s", "signed_error", "free_flow_error")


def write_stretches_csv(asm: pipeline.Assembly, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STRETCH_COLUMNS)
        for s in asm.stretch_result.stretches:
            w.writerow([s.stretch_id, s.run, " ".join(s.pass_ids), s.start_utc_s, s.end_utc_s, s.matched_m, s.observed_s,
                        s.here_s, s.free_flow_s, s.stopped_s, s.signed_error, s.free_flow_error])
