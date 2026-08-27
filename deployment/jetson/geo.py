"""Geodesy with no other dependencies.

`haversine_m` lived in `v2v/beacon.py`, which imports `PeerState` from
`perception.observation_builder`. So the moment the observation builder needed
anything that needed the distance function -- which it does, now that the traffic
feed owns a field -- the import graph closed a loop:

    observation_builder -> feed_fusion -> here_feed -> v2v.beacon -> observation_builder

A formula with no state and no collaborators does not belong in a module that
depends on the sensing pipeline. Duplicating it would have broken the loop too, and
would have left two copies of the same arithmetic to disagree later.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0

#: Metres per degree of latitude.
DEG_M = math.pi * EARTH_RADIUS_M / 180.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def point_to_segment_m(lat: float, lon: float,
                       alat: float, alon: float,
                       blat: float, blon: float) -> float:
    """Distance from a point to a great-circle segment, not to its endpoints.

    A local equirectangular projection about the point, which is exact enough here
    by a wide margin: link segments run hundreds of metres, and the projection's
    error over that span is millimetres against an association radius of tens of
    metres.

    The degenerate segment -- both endpoints equal -- falls out of the `length_sq`
    guard as the endpoint distance, which is the right answer rather than a special
    case.
    """
    lat0 = math.radians(lat)
    scale = math.cos(lat0)

    def xy(plat: float, plon: float) -> tuple[float, float]:
        return ((plon - lon) * scale * DEG_M, (plat - lat) * DEG_M)

    ax, ay = xy(alat, alon)
    bx, by = xy(blat, blon)
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(ax, ay)
    # Projection of the origin (the point itself) onto the segment, clamped to it.
    t = -(ax * dx + ay * dy) / length_sq
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(ax + t * dx, ay + t * dy)
