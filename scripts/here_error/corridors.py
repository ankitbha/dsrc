"""Physical roads built from segment geometry.

HERE's `location.description` names the cross street at a segment's end, not the road being
driven, and HERE reports each direction of a road as its own run of segments. Segments are
therefore grouped by geometry: end-to-end continuation, and opposite carriageways that run
alongside each other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from here_error import params
from here_error.geometry import LocalFrame, angle_between_deg, polyline_length, project_points
from here_error.models import HereSegment


@dataclass(frozen=True)
class Road:
    road_id: str
    segment_keys: tuple[str, ...]


@dataclass(frozen=True)
class Roads:
    roads: tuple[Road, ...]
    by_segment: dict[str, Road]


def _end_bearing(seg: HereSegment, at_end: bool) -> float:
    i = (len(seg.lat) - 2, len(seg.lat) - 1) if at_end else (0, 1)
    f = LocalFrame(seg.lat[i[0]], seg.lon[i[0]])
    (x0, y0), (x1, y1) = f.to_xy([seg.lat[i[0]], seg.lat[i[1]]], [seg.lon[i[0]], seg.lon[i[1]]])
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def _joins(a: HereSegment, b: HereSegment) -> bool:
    """b continues a: a's last point is within JOIN_M of b's first and the bearing turns by less than JOIN_TURN_DEG."""
    f = LocalFrame(a.lat[-1], a.lon[-1])
    gap = float(np.hypot(*(f.to_xy(b.lat[0], b.lon[0]) - f.to_xy(a.lat[-1], a.lon[-1]))))
    if gap > params.ROAD_JOIN_M:
        return False
    return angle_between_deg(_end_bearing(a, True), _end_bearing(b, False)) < params.ROAD_JOIN_TURN_DEG


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    d = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    return np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], axis=1)


def _alongside(a: HereSegment, b: HereSegment) -> bool:
    """Opposite carriageways: most of the shorter segment's length runs within ROAD_PAIR_M of the other, heading the other way."""
    f = LocalFrame(float(np.mean(a.lat)), a.lon[0])
    xa, xb = f.to_xy(a.lat, a.lon), f.to_xy(b.lat, b.lon)
    short, long_ = (xa, xb) if polyline_length(xa) <= polyline_length(xb) else (xb, xa)
    lo, hi = long_.min(axis=0) - params.ROAD_PAIR_M, long_.max(axis=0) + params.ROAD_PAIR_M
    if np.any(short.max(axis=0) < lo) or np.any(short.min(axis=0) > hi):
        return False
    pts = _densify(short, params.ROAD_SAMPLE_M)
    # Bearing of the short segment at each sample, from its own pieces.
    piece_b = np.degrees(np.arctan2(*np.diff(short, axis=0).T)) % 360.0
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(short, axis=0).T))])
    s = np.arange(0.0, cum[-1] + 1e-9, params.ROAD_SAMPLE_M)
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(piece_b) - 1)
    near = 0
    for p, kk, pr in zip(pts, k, project_points(long_, pts)):
        if (pr.distance_m <= params.ROAD_PAIR_M and not pr.clamped_start and not pr.clamped_end
                and angle_between_deg(float(piece_b[kk]), pr.bearing_deg) > params.ROAD_ANTIPARALLEL_DEG):
            near += 1
    return near / len(pts) >= params.ROAD_PAIR_SHARE and near * params.ROAD_SAMPLE_M >= params.ROAD_PAIR_MIN_OVERLAP_M


def build_roads(segments) -> Roads:
    """Group segments (any iterable of HereSegment, duplicates by key ignored) into physical roads."""
    segs: dict[str, HereSegment] = {}
    for s in segments:
        segs.setdefault(s.key, s)
    keys = sorted(segs)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            a, b = segs[ka], segs[kb]
            if _joins(a, b) or _joins(b, a) or _alongside(a, b):
                union(ka, kb)
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    roads, by_segment = [], {}
    for n, root in enumerate(sorted(groups), start=1):
        members = groups[root]
        road = Road(f"R{n:02d}", tuple(members))
        roads.append(road)
        for k in members:
            by_segment[k] = road
    return Roads(tuple(roads), by_segment)


def describe(road_id: str, members: list[HereSegment]) -> str:
    """A label made of geometry, never of one segment's description.

    `members` are the segments the label speaks for; the caller passes those that carry analysed
    passes, because the join rule also chains ramps and side roads onto the main line.

    A HERE description names the cross street at the end of its segment, so it does not name the
    road driven. The label gives the number of directed segments, the length of shape, the axis
    bearing, and the descriptions of the segments at the two ends of the road in chain order,
    introduced as segments *ending at* those streets.
    """
    f = LocalFrame(float(np.mean([s.lat[0] for s in members])), members[0].lon[0])
    # Axis: length-weighted mean of the doubled bearing, so opposite carriageways reinforce each other.
    sx = sy = 0.0
    total = 0.0
    for s in members:
        xy = f.to_xy(s.lat, s.lon)
        d = np.diff(xy, axis=0)
        for (dx, dy) in d:
            length = math.hypot(dx, dy)
            ang = 2 * math.atan2(dx, dy)
            sx += length * math.sin(ang)
            sy += length * math.cos(ang)
        total += polyline_length(xy)
    axis = (math.degrees(math.atan2(sx, sy)) / 2.0) % 180.0
    ux, uy = math.sin(math.radians(axis)), math.cos(math.radians(axis))
    along = lambda s: float(np.dot(np.mean(f.to_xy(s.lat, s.lon), axis=0), (ux, uy)))
    ordered = sorted(members, key=along)
    first, last = ordered[0].description or "(unnamed)", ordered[-1].description or "(unnamed)"
    ends = f"segment ending at {first}" if len(members) == 1 else f"segments ending at {first} ... {last}"
    return (f"{road_id}: {len(members)} directed segment{'s' if len(members) != 1 else ''}, {total / 1000:.1f} km of shape, "
            f"bearing about {axis:.0f}/{(axis + 180) % 360:.0f} degrees, {ends}")
