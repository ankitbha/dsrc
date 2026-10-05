"""Stage 1: map-match one-second GPS fixes to HERE segments and extract passes (gates V2, V3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from here_error import params
from here_error.geometry import (
    LocalFrame,
    angle_between_deg,
    haversine_m,
    polyline_length,
    project_points,
)
from here_error.models import GpsFix, HereBody, HereSegment, Pass


@dataclass(frozen=True, eq=False)
class CatalogueEntry:
    segment: HereSegment
    frame: LocalFrame
    xy: np.ndarray
    shape_length_m: float
    length_flag: bool


def build_catalogue(bodies) -> dict[str, CatalogueEntry]:
    """One entry per distinct segment key over all bodies of a run.

    Geometry is static, so matching does not depend on which body a segment came from.
    """
    out: dict[str, CatalogueEntry] = {}
    for body in bodies:
        for seg in body.segments:
            if seg.key in out:
                continue
            frame = LocalFrame(float(np.mean(seg.lat)), seg.lon[0])
            xy = frame.to_xy(seg.lat, seg.lon)
            length = polyline_length(xy)
            flag = abs(length - seg.stated_length_m) / seg.stated_length_m > params.SEGMENT_LENGTH_TOL
            out[seg.key] = CatalogueEntry(seg, frame, xy, length, bool(flag))
    return out


@dataclass(frozen=True)
class _Candidate:
    key: str
    distance_m: float
    chainage_m: float


def _candidates(fixes: tuple[GpsFix, ...], catalogue: dict[str, CatalogueEntry]) -> list[list[_Candidate]]:
    lat = np.array([f.lat for f in fixes])
    lon = np.array([f.lon for f in fixes])
    cands: list[list[_Candidate]] = [[] for _ in fixes]
    m = params.MATCH_M
    for key, entry in catalogue.items():
        pts = entry.frame.to_xy(lat, lon)
        lo = entry.xy.min(axis=0) - m
        hi = entry.xy.max(axis=0) + m
        near = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
        if near.size == 0:
            continue
        for i, pr in zip(near, project_points(entry.xy, pts[near])):
            if pr.distance_m > m or pr.clamped_start or pr.clamped_end:
                continue
            f = fixes[i]
            if (f.speed_mps is not None and f.speed_mps >= params.HEADING_MIN_SPEED_MPS
                    and f.heading_deg is not None
                    and angle_between_deg(f.heading_deg, pr.bearing_deg) > params.HEADING_TOL_DEG):
                continue
            cands[i].append(_Candidate(key, pr.distance_m, pr.chainage_m))
    return cands


def _choose(fixes, cands) -> list[_Candidate | None]:
    """Keep the previous fix's segment when it is still a candidate; otherwise take the nearest."""
    chosen: list[_Candidate | None] = []
    for i, cs in enumerate(cands):
        if not cs:
            chosen.append(None)
            continue
        prev = chosen[-1] if i else None
        if prev is not None and fixes[i].utc_s - fixes[i - 1].utc_s <= params.PASS_GAP_S:
            same = [c for c in cs if c.key == prev.key]
            if same:
                chosen.append(same[0])
                continue
        chosen.append(min(cs, key=lambda c: c.distance_m))
    return chosen


def _path_from_positions(fixes) -> float:
    if len(fixes) < 2:
        return 0.0
    lat = np.array([f.lat for f in fixes])
    lon = np.array([f.lon for f in fixes])
    return float(np.sum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])))


def _path_from_speed(fixes) -> float | None:
    if len(fixes) < 2:
        return 0.0
    if any(f.speed_mps is None for f in fixes):
        return None
    t = np.array([f.utc_s for f in fixes])
    v = np.array([f.speed_mps for f in fixes])
    return float(np.sum(np.diff(t) * (v[:-1] + v[1:]) / 2.0))


def _make_pass(run, n, entry, fixes, chains) -> Pass:
    chain_start, chain_end = chains[0], chains[-1]
    pos = _path_from_positions(fixes)
    spd = _path_from_speed(fixes)
    reversed_ = float(np.max(np.maximum.accumulate(chains) - chains)) > params.REVERSE_TOL_M
    exclusion = None
    if reversed_:
        exclusion = "chainage_reversed"
    elif chain_end - chain_start < params.MIN_PASS_M:
        exclusion = "shorter_than_min"
    elif spd is None or pos <= 0 or abs(pos - spd) / pos > params.V3_MAX_REL_DIFF:
        exclusion = "v3_failed"
    return Pass(
        pass_id=f"{run.rsplit('_', 1)[-1]}-{n:03d}",
        run=run,
        segment_key=entry.segment.key,
        description=entry.segment.description,
        functional_class=entry.segment.functional_class,
        fixes=tuple(fixes),
        chain_start_m=float(chain_start),
        chain_end_m=float(chain_end),
        segment_length_flag=entry.length_flag,
        path_positions_m=pos,
        path_speed_m=spd,
        exclusion=exclusion,
    )


def extract_passes(run: str, fixes: tuple[GpsFix, ...], bodies: tuple[HereBody, ...]) -> tuple[list[Pass], dict[str, CatalogueEntry]]:
    """Find every pass in `fixes` against the segments of `bodies`.

    A fix lies on a segment when its foot point is inside the segment's shape (not clamped to
    either end), it is within MATCH_M of the shape, and, where the speed allows it, its heading
    is within HEADING_TOL_DEG of the segment direction. A pass is a maximal run of fixes on one
    segment with gaps of at most PASS_GAP_S.
    """
    catalogue = build_catalogue(bodies)
    chosen = _choose(fixes, _candidates(fixes, catalogue))
    passes: list[Pass] = []
    run_fixes: list[GpsFix] = []
    run_chain: list[float] = []
    cur_key: str | None = None

    def flush():
        if cur_key is not None and run_fixes:
            passes.append(_make_pass(run, len(passes) + 1, catalogue[cur_key], run_fixes, np.array(run_chain)))

    for fix, c in zip(fixes, chosen):
        continues = (
            c is not None and cur_key == c.key and run_fixes
            and fix.utc_s - run_fixes[-1].utc_s <= params.PASS_GAP_S
        )
        if continues:
            run_fixes.append(fix)
            run_chain.append(c.chainage_m)
            continue
        flush()
        run_fixes, run_chain = [], []
        cur_key = c.key if c is not None else None
        if c is not None:
            run_fixes.append(fix)
            run_chain.append(c.chainage_m)
    flush()
    return passes, catalogue


def pass_title(p: Pass) -> str:
    """A HERE description names the cross street at the end of its segment, so it is never given as a road's name."""
    return (f"{p.pass_id}, road {p.road_id or '(not assigned)'}, segment ending at {p.description}\n"
            f"chainage {p.chain_start_m:.0f} to {p.chain_end_m:.0f} m, {p.observed_s:.0f} s, {len(p.fixes)} fixes, exclusion: {p.exclusion}")


def neighbour_label(segment: HereSegment) -> str:
    return f"ends at {segment.description}"


def plot_pass(p: Pass, catalogue: dict[str, CatalogueEntry], all_fixes: tuple[GpsFix, ...], path: Path) -> None:
    """Gate V2: the segment, the pass's fixes and the neighbouring segments, for inspection by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entry = catalogue[p.segment_key]
    frame = entry.frame
    own = frame.to_xy([f.lat for f in p.fixes], [f.lon for f in p.fixes])
    lo, hi = own.min(axis=0) - 150.0, own.max(axis=0) + 150.0
    around = [f for f in all_fixes if p.first_utc_s - 90 <= f.utc_s <= p.last_utc_s + 90]
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, e in catalogue.items():
        if key == p.segment_key:
            continue
        xy = frame.to_xy(e.segment.lat, e.segment.lon)
        if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
            ax.plot(xy[:, 0], xy[:, 1], color="0.6", lw=1.0)
            mid = xy[len(xy) // 2]
            if lo[0] <= mid[0] <= hi[0] and lo[1] <= mid[1] <= hi[1]:
                ax.annotate(neighbour_label(e.segment), mid, fontsize=6, color="0.4")
    ax.plot(entry.xy[:, 0], entry.xy[:, 1], color="tab:blue", lw=3, alpha=0.6, label="matched segment")
    ax.plot(entry.xy[0, 0], entry.xy[0, 1], "g^", ms=9, label="segment start")
    if around:
        a = frame.to_xy([f.lat for f in around], [f.lon for f in around])
        ax.plot(a[:, 0], a[:, 1], ".", color="0.5", ms=3, label="other fixes, +-90 s")
    ax.plot(own[:, 0], own[:, 1], ".", color="tab:red", ms=4, label="fixes in pass")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.set_title(pass_title(p), fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
