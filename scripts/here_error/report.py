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
    def road_id(self) -> str:
        return self.a.p.road_id

    @property
    def segment_key(self) -> str:
        return self.a.p.segment_key


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


def audit_routes(inputs: ReportInputs) -> tuple[list[RouteResult], int, int]:
    """Routing rows that belong to a current request (same id and observed time), each request counted once.

    Returns the usable rows, the number of rows matching no current request, and the number of
    repeated rows for a request id that was already counted.
    """
    need = {q.request_id: q.observed_s for q in inputs.requests}
    good, seen, stale, dup = [], set(), 0, 0
    for r in inputs.routing or []:
        if r.request_id not in need or abs(r.observed_s - need[r.request_id]) >= 1e-6:
            stale += 1
        elif r.request_id in seen:
            dup += 1
        else:
            seen.add(r.request_id)
            good.append(r)
    return good, stale, dup


def valid_routes(inputs: ReportInputs) -> list[RouteResult]:
    return audit_routes(inputs)[0]


def incomplete_reasons(inputs: ReportInputs) -> list[str]:
    reasons = camera_coverage(inputs)
    if inputs.label_scores is None:
        reasons.append("detector labels (gate V6) are not scored")
    if inputs.routing is None:
        reasons.append("routing results are missing")
    else:
        good, stale, dup = audit_routes(inputs)
        if stale:
            reasons.append(f"{stale} routing rows match no current request (stale or foreign id or observed time) and are ignored")
        if dup:
            reasons.append(f"{dup} routing rows repeat a request id already counted and are ignored")
        have = {r.request_id for r in good}
        need = {q.request_id for q in inputs.requests}
        if need - have:
            reasons.append(f"routing results cover {len(need & have)} of {len(need)} current requests")
    return reasons


def daylight_only(inputs: ReportInputs) -> bool:
    """Gate V6 rule: when dusk recall is below 0.7 the camera results use daylight frames only."""
    if inputs.label_scores is None:
        return False
    after = [s for s in inputs.label_scores if s.half == "after_dusk"]
    return bool(after) and after[0].recall_all is not None and after[0].recall_all < 0.7


def _road_class(fc: int | None) -> str:
    if fc is None:
        return "unknown"
    return f"class {fc}" if fc <= 2 else "class 3 and above"


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


def pp(x: float) -> str:
    """A difference of two signed errors, in percentage points of travel time."""
    return f"{100 * x:+.1f} pp"


_SINGULAR = {"physical roads": "physical road", "directed segments": "directed segment", "stretches": "stretch"}


def _interval_text(values, clusters, unit: str, fmt=pct) -> str:
    """95% cluster-bootstrap interval, or a statement that too few clusters exist to resample."""
    n = len(set(clusters))
    if n < params.MIN_N_FOR_INTERVAL:
        return f"not resolvable: {n} {_SINGULAR[unit] if n == 1 else unit}"
    ci = stats.cluster_bootstrap_median(values, clusters)
    return f"{fmt(ci.lo)} to {fmt(ci.hi)}"


ROAD_COL = "95% interval resampling physical roads"
SEGMENT_COL = "95% interval, treats each directed segment (passes) or stretch (stretches) as independent"


def _row(label, values, roads, segs, n_unit, fmt=pct):
    if not len(values):
        return f"| {label} | {NOT_COMPUTED}: no data | | | |"
    med = float(np.median(values))
    return (f"| {label} | {fmt(med)} | {_interval_text(values, roads, 'physical roads', fmt)} "
            f"| {_interval_text(values, segs, 'directed segments', fmt)} "
            f"| {len(values)} {n_unit} on {len(set(roads))} physical roads and {len(set(segs))} directed segments |")


def stretch_road(stretch, passes_by_id: dict) -> str:
    """The road holding most of the stretch's matched metres; a tie goes to the road of its first pass."""
    metres: dict[str, float] = {}
    for pid in stretch.pass_ids:
        p = passes_by_id[pid]
        metres[p.road_id] = metres.get(p.road_id, 0.0) + p.matched_m
    best = max(metres.values())
    first = passes_by_id[stretch.pass_ids[0]].road_id
    return first if metres[first] == best else next(r for r in metres if metres[r] == best)


def stretch_columns(values, stretch_ids, roads, fmt=pct) -> tuple[str, str]:
    """The two interval cells for stretch values: resampling physical roads, and each stretch alone."""
    return _interval_text(values, roads, 'physical roads', fmt), _interval_text(values, stretch_ids, 'stretches', fmt)


def _stretch_row(label, values, ids, roads, fmt=pct):
    """Stretch values in the same columns as passes; `roads` holds the road each stretch belongs to."""
    if not len(values):
        return f"| {label} | {NOT_COMPUTED}: no data | | | |"
    road_cell, own_cell = stretch_columns(values, ids, roads, fmt)
    return (f"| {label} | {fmt(float(np.median(values)))} | {road_cell} | {own_cell} "
            f"| {len(values)} stretches on {len(set(roads))} physical roads |")


def headline(rows: list[PassRow], stretches, passes_by_id: dict) -> list[str]:
    def col(f, pick=lambda r: True):
        sel = [r for r in rows if pick(r)]
        return [f(r) for r in sel], [r.road_id for r in sel], [r.segment_key for r in sel]

    sids = [s.stretch_id for s in stretches]
    sroads = [stretch_road(s, passes_by_id) for s in stretches]
    out = [f"| Quantity compared | Median | {ROAD_COL} | {SEGMENT_COL} | n |", "|---|---|---|---|---|"]
    out.append(_row("Pass: (HERE time - observed time) / observed time, HERE reading with the latest data time before the pass",
                    *col(lambda r: r.a.t.signed_error), "passes"))
    out.append(_row("Pass: (free-flow time - observed time) / observed time",
                    *col(lambda r: r.a.t.free_flow_error), "passes"))
    out.append(_row("Pass: abs(HERE error) - abs(free-flow error), in percentage points; negative means HERE's live speed is closer to the observed time",
                    *col(lambda r: abs(r.a.t.signed_error) - abs(r.a.t.free_flow_error)), "passes", pp))
    out.append(_row("Pass, sensitivity: HERE reading with data time nearest the pass midpoint",
                    *col(lambda r: r.a.t.sensitivity_signed_error, lambda r: r.a.t.sensitivity_signed_error is not None), "passes"))
    out.append(_row("Pass, sensitivity: latest reading that had also arrived before the pass began",
                    *col(lambda r: r.a.t.arrived_signed_error, lambda r: r.a.t.arrived_signed_error is not None), "passes"))
    out.append(_stretch_row("Stretch: (HERE time - observed time) / observed time", [s.signed_error for s in stretches], sids, sroads))
    out.append(_stretch_row("Stretch: (free-flow time - observed time) / observed time", [s.free_flow_error for s in stretches], sids, sroads))
    out.append(_stretch_row("Stretch: abs(HERE error) - abs(free-flow error), in percentage points",
                            [abs(s.signed_error) - abs(s.free_flow_error) for s in stretches], sids, sroads, pp))
    return out


def per_road_table(rows: list[PassRow]) -> list[str]:
    """Median errors for each physical road, with the number of passes and directed segments behind them."""
    if not rows:
        return []
    out = ["Per physical road:", "",
           "| Road | Geometry (a segment's description names the cross street at its end, not the road) | Passes | Directed segments | Median HERE error | Median free-flow error |",
           "|---|---|---|---|---|---|"]
    for rid in sorted({r.road_id for r in rows}):
        sel = [r for r in rows if r.road_id == rid]
        out.append(f"| {rid} | {sel[0].a.p.road_label} | {len(sel)} | {len({r.segment_key for r in sel})} "
                   f"| {pct(float(np.median([r.a.t.signed_error for r in sel])))} "
                   f"| {pct(float(np.median([r.a.t.free_flow_error for r in sel])))} |")
    return out


def arrival_rows(rows: list[PassRow]) -> list[str]:
    """A companion row computed on a subset is compared with the primary on that same subset."""
    out = []
    sub = [r for r in rows if r.a.t.arrived_signed_error is not None]
    if sub:
        prim = [r.a.t.signed_error for r in sub]
        arr = [r.a.t.arrived_signed_error for r in sub]
        out.append(f"- Arrival-constrained reading, on its {len(sub)} passes (the others had no reading that had arrived by the first fix): "
                   f"median signed error with the primary reading {pct(float(np.median(prim)))}, with the arrival-constrained reading "
                   f"{pct(float(np.median(arr)))}; median of the paired difference (arrival-constrained - primary) "
                   f"{pp(float(np.median(np.array(arr) - np.array(prim))))}.")
    return out


def v4_rows(rows: list[PassRow]) -> list[str]:
    out = []
    sub = [r for r in rows if r.a.t.shuffled_signed_error is not None]
    if sub:
        real = np.array([abs(r.a.t.signed_error) for r in sub])
        shuf = np.array([abs(r.a.t.shuffled_signed_error) for r in sub])
        out.append(f"- Control V4, on its {len(sub)} passes that have a reading at least {params.SHUFFLE_MIN_S:.0f} s away: median abs(signed error) "
                   f"{pct(float(np.median(real)), False)} with the real reading, {pct(float(np.median(shuf)), False)} with the shuffled reading; "
                   f"median of the paired difference (shuffled - real) {pp(float(np.median(shuf - real)))}.")
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
    text = f"    - {label}{(' (' + extra + ')') if extra else ''}: {len(sel)} passes, median {pct(float(np.median(errs)))}, " \
           f"95% interval resampling physical roads: {_interval_text(errs, [r.road_id for r in sel], 'physical roads')}"
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


@dataclass(frozen=True)
class Visit:
    """Passes on one directed segment that follow each other within STRETCH_MAX_GAP_S: one drive over it."""

    segment_key: str
    rows: tuple[PassRow, ...]

    @property
    def observed_s(self) -> float:
        return sum(r.a.t.observed_s for r in self.rows)

    @property
    def here_s(self) -> float:
        return sum(r.a.t.here_s for r in self.rows)

    @property
    def matched_m(self) -> float:
        return sum(r.a.p.matched_m for r in self.rows)

    @property
    def signed_error(self) -> float:
        return (self.here_s - self.observed_s) / self.observed_s

    @property
    def here_speed(self) -> float:
        return self.matched_m / self.here_s


def segment_visits(rows: list[PassRow]) -> dict[str, list[Visit]]:
    out: dict[str, list[Visit]] = {}
    by_seg: dict[str, list[PassRow]] = {}
    for r in rows:
        by_seg.setdefault(r.segment_key, []).append(r)
    for key, rs in by_seg.items():
        rs = sorted(rs, key=lambda r: r.a.p.first_utc_s)
        groups = [[rs[0]]]
        for r in rs[1:]:
            if r.a.p.first_utc_s - groups[-1][-1].a.p.last_utc_s > params.STRETCH_MAX_GAP_S:
                groups.append([r])
            else:
                groups[-1].append(r)
        out[key] = [Visit(key, tuple(g)) for g in groups]
    return out


def repeated_visits(rows: list[PassRow]) -> list[str]:
    """Spread between separate drives over the same directed segment, against the spread of HERE's reading."""
    rep = {k: v for k, v in segment_visits(rows).items() if len(v) >= 2}
    if len(rep) < 2:
        found = ", ".join(f"the directed segment ending at {v[0].rows[0].a.p.description}, road {v[0].rows[0].road_id}, {v[0].matched_m:.0f} m matched" for v in rep.values())
        return [f"- Repeated passes: {NOT_COMPUTED}: {len(rep)} directed segment{'s' if len(rep) != 1 else ''} driven at least twice "
                f"with the drives more than {params.STRETCH_MAX_GAP_S:.0f} s apart" + (f" ({found})" if found else "")
                + "; at least 2 are needed for a spread."]
    dev_err, dev_spd, dof = [], [], 0
    for visits in rep.values():
        m_e = np.mean([v.signed_error for v in visits])
        m_s = np.mean([v.here_speed for v in visits])
        dev_err += [v.signed_error - m_e for v in visits]
        dev_spd += [(v.here_speed - m_s) / m_s for v in visits]
        dof += len(visits) - 1
    return [f"- Repeated passes: {len(rep)} directed segments were driven at least twice with the drives more than {params.STRETCH_MAX_GAP_S:.0f} s apart "
            f"({sum(len(v) for v in rep.values())} drives). Within-segment standard deviation of the signed error of HERE time: "
            f"{100 * np.sqrt(np.sum(np.square(dev_err)) / dof):.1f} percentage points; within-segment standard deviation of HERE's mean speed over the "
            f"matched portion: {100 * np.sqrt(np.sum(np.square(dev_spd)) / dof):.1f}% of the segment's mean HERE speed."]


def stage5(rows: list[PassRow], camera_note: str | None) -> list[str]:
    out = repeated_visits(rows)
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
    roads = [r.road_id for r in rows]
    raw_v = [r.a.t.signed_error for r in rows]
    out.append(f"- Median signed error of HERE time without removing the offset: {pct(float(np.median(raw_v)))} "
               f"({_interval_text(raw_v, roads, 'physical roads')}); "
               f"with the offset removed from the observed time (stopped time + moving time x (1 + offset)): "
               f"{pct(float(np.median(adj)))} ({_interval_text(adj, roads, 'physical roads')}). With one driver the offset is an estimate, not a correction; "
               f"without it the first figure is an upper bound on HERE's error.")
    return out


def routing_section(inputs: ReportInputs, passes_by_id: dict, stretch_roads: dict) -> list[str]:
    if inputs.routing is None:
        return [f"- {NOT_COMPUTED}: no routing results"]
    good = valid_routes(inputs)
    out = []
    for kind in ("pass", "stretch"):
        rs = [r for r in good if r.kind == kind]
        ok = [r for r in rs if r.exclusion is None]
        if not ok:
            out.append(f"- {kind}: {NOT_COMPUTED}: {len(rs)} results, none follow the driven path")
            continue
        vals = [r.signed_error for r in ok]
        head = (f"- {kind}: median of (HERE routing duration - observed time) / observed time over {len(ok)} routes: "
                f"{pct(float(np.median(vals)))}; ")
        if kind == "pass":
            roads = [passes_by_id[r.request_id].road_id for r in ok]
            segs = [passes_by_id[r.request_id].segment_key for r in ok]
            head += (f"95% interval resampling physical roads: {_interval_text(vals, roads, 'physical roads')}; "
                     f"treating each directed segment as independent: {_interval_text(vals, segs, 'directed segments')}; ")
        else:
            road_cell, own_cell = stretch_columns(vals, [r.request_id for r in ok], [stretch_roads[r.request_id] for r in ok])
            head += (f"95% interval resampling physical roads: {road_cell}; "
                     f"treating each stretch as independent: {own_cell}; ")
        out.append(head + f"{POSITIVE_MEANS}. {len(rs) - len(ok)} routes excluded because the route did not follow the driven path.")
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
             "Intervals are 95% percentile intervals from a bootstrap that resamples physical roads (built from segment geometry, not from "
             "HERE's descriptions, which name the cross street at a segment's end), or directed segments, or whole stretches, as each column says.")
    L.append("")
    L.append("## Headline table")
    if reasons:
        L.append("Printed while the report is INCOMPLETE. These rows use only GPS and HERE data, which are present; "
                 "the rows that need the missing inputs say so below.")
        L.append("")
    L += headline(rows, asm.stretch_result.stretches, {p.pass_id: p for p in asm.passes})
    L.append("")
    L += per_road_table(rows)
    L.append("")
    L += arrival_rows(rows)
    L.append("")
    sd = float(np.std([r.a.t.signed_error for r in rows], ddof=1)) if len(rows) > 1 else float("nan")
    n_roads = len({r.road_id for r in rows})
    L.append(f"Measured standard deviation of the per-pass signed error: {100 * sd:.1f} percentage points over {len(rows)} passes on "
             f"{n_roads} physical road{'s' if n_roads != 1 else ''}. "
             + (f"Treating the {n_roads} roads as independent, the half-width of the 95% interval on the median would be "
                f"{100 * stats.median_half_width(sd, n_roads):.1f} percentage points."
                if n_roads >= params.MIN_N_FOR_INTERVAL else
                f"With fewer than {params.MIN_N_FOR_INTERVAL} physical roads, no interval across roads is resolvable."))
    L.append("")
    L.append("## Control V4: time-shuffled HERE reading")
    v4 = asm.v4
    if v4.n:
        L.append(f"Over {v4.n} passes with a reading of the same segment at least {params.SHUFFLE_MIN_S:.0f} s away: median abs(signed error) "
                 f"{pct(v4.median_abs_real, False)} with the real reading, {pct(v4.median_abs_shuffled, False)} with the shuffled reading. "
                 f"If the second does not exceed the first, this sample cannot resolve HERE's real-time signal.")
    else:
        L.append(f"{NOT_COMPUTED}: no pass has a reading of its segment at least {params.SHUFFLE_MIN_S:.0f} s away.")
    L += v4_rows(rows)
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
    by_id = {p.pass_id: p for p in asm.passes}
    L += routing_section(inputs, by_id, {st.stretch_id: stretch_road(st, by_id) for st in asm.stretch_result.stretches})
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
    "pass_id", "run", "segment_end_description", "segment_key", "road_id", "road_label", "road_catalogue_segments", "functional_class", "first_utc_s", "last_utc_s", "n_fixes", "chain_start_m", "chain_end_m",
    "matched_m", "observed_s", "path_positions_m", "path_speed_m", "v3_rel_diff", "segment_length_flag", "exclusion",
    "reading_seq", "reading_age_s", "reading_arrived_before_start", "here_s", "free_flow_s", "signed_error",
    "free_flow_error", "sensitivity_signed_error", "arrived_signed_error", "shuffled_signed_error", "jam_factor",
    "stopped_s", "moving_s", "n_frames", "mean_vehicles", "leader_share",
)


def write_passes_csv(asm: pipeline.Assembly, rows: list[PassRow], path: Path) -> None:
    cond = {r.a.p.pass_id: r for r in rows}
    # Every segment of the catalogue the road reaches: larger than the segments its passes lie on.
    catalogue_size = {r.road_id: len(r.segment_keys) for r in asm.roads.roads} if asm.roads else {}
    times = {t.pass_id: t for t in asm.times}
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PASS_COLUMNS)
        w.writeheader()
        for p in asm.passes:
            t = times[p.pass_id]
            c = cond.get(p.pass_id)
            d = dict(
                pass_id=p.pass_id, run=p.run, segment_end_description=p.description, segment_key=p.segment_key,
                road_id=p.road_id, road_label=p.road_label,
                road_catalogue_segments=catalogue_size.get(p.road_id), functional_class=p.functional_class,
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
