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


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
