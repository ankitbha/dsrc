"""Planar geometry on short stretches of road. No I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class LocalFrame:
    """Equirectangular projection about a reference point: x east, y north, in metres.

    Error is below 0.1% over the few kilometres a HERE segment spans.
    """

    lat0: float
    lon0: float

    def to_xy(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = np.radians(lon - self.lon0) * EARTH_RADIUS_M * math.cos(math.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return np.stack([x, y], axis=-1)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def polyline_length(xy: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))


def cumulative_lengths(xy: np.ndarray) -> np.ndarray:
    """Distance along the polyline at each vertex; the first is 0."""
    d = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def angle_between_deg(a: float, b: float) -> float:
    """Smallest angle between two bearings, in [0, 180]."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


@dataclass(frozen=True)
class PolylineProjection:
    distance_m: float
    chainage_m: float
    bearing_deg: float
    clamped_start: bool
    clamped_end: bool


def project_points(xy: np.ndarray, points: np.ndarray) -> list[PolylineProjection]:
    """Project each point onto the nearest place on the polyline.

    `clamped_start` / `clamped_end` are true when the nearest place is the first / last
    vertex because the point lies beyond that end of the polyline. A vertex in the middle
    of the polyline is not an end.
    """
    points = np.atleast_2d(points)
    a = xy[:-1]
    b = xy[1:]
    ab = b - a
    seg_len2 = np.sum(ab * ab, axis=1)
    seg_len = np.sqrt(seg_len2)
    cum = cumulative_lengths(xy)
    n_pieces = len(a)
    ap = points[:, None, :] - a[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_raw = np.where(seg_len2 > 0, np.sum(ap * ab[None], axis=2) / seg_len2, 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    foot = a[None] + t[:, :, None] * ab[None]
    dist = np.hypot(*(points[:, None, :] - foot).transpose(2, 0, 1))
    best = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    out = []
    for i in rows:
        k = int(best[i])
        bearing = math.degrees(math.atan2(ab[k, 0], ab[k, 1])) % 360.0
        out.append(
            PolylineProjection(
                distance_m=float(dist[i, k]),
                chainage_m=float(cum[k] + t[i, k] * seg_len[k]),
                bearing_deg=bearing,
                clamped_start=bool(k == 0 and t_raw[i, k] < 0.0),
                clamped_end=bool(k == n_pieces - 1 and t_raw[i, k] > 1.0),
            )
        )
    return out


def project_point(xy: np.ndarray, point) -> PolylineProjection:
    return project_points(xy, np.asarray(point, dtype=float))[0]
