"""Driver-facing text for the DSRC controller's per-super-segment speed advisory.

Split out of `policy.advisory` when the local-sensing 39-field runtime was
removed. The two decoders never shared anything but a module: this one turns a
`DsrcDecision` into one recommended speed per super-segment, against a fraction
of that segment's own static speed limit, and shows the driver the row their own
position falls on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from geo import haversine_m, point_to_segment_m
from perception.segment_state import (
    DEFAULT_NETWORK_DEFINITION_PATH,
    SEGMENT_MATCH_TOLERANCE_M,
)
from policy.dsrc_contract import SPEED_ACTION_FRACTIONS, network_fingerprint
from policy.dsrc_runtime import DsrcDecision

MPS_TO_MPH = 2.236936
MPS_TO_KMH = 3.6


@dataclass
class SegmentAdvisoryRow:
    """One super-segment's decoded speed advisory."""

    segment_id: str
    action_index: int
    fraction: float
    recommended_speed_mps: float
    recommended_speed_display: float


@dataclass
class SegmentAdvisory:
    """One DSRC decision, decoded: a row per super-segment, or none at all.

    `AdvisoryDecoder`/`Advisory` are untouched -- these are two different
    controllers (plan_task145 section 7's table: one bin per head for the
    ego vehicle against an offset base, versus one action per super-segment
    against a fraction of that segment's own speed limit) and a shared
    decoder branching on which one produced the action would hide exactly
    the difference this task exists to surface.
    """

    units: str
    #: Mirrors `policy.dsrc_runtime.DsrcDecision.outcome`: "ok" when `rows`
    #: is populated, otherwise the named reason there is no advisory at all
    #: (never an empty list standing in silently for "nothing to show").
    outcome: str
    rows: tuple[SegmentAdvisoryRow, ...]
    #: The index into `rows` (and into the network's own segment order) the
    #: vehicle's own fix currently falls on, or `None` when it is not near
    #: any segment or no fix was given. The driver is shown one number, and
    #: this is the one.
    ego_segment: int | None


def _distance_to_polyline(lat: float, lon: float, points: tuple[tuple[float, float], ...]) -> float:
    """Distance from a point to a polyline's shape, not to its nearest vertex.

    Same construction as `sensors.here_feed.FlowLink.distance_m` (kept
    independent rather than imported, since a device-side advisory module
    reasoning about the vehicle's own position has no other reason to
    depend on the traffic-feed module): every vertex first, then every
    segment between consecutive vertices, because a vehicle sitting between
    two sampled points is not "distance to the nearer endpoint".
    """
    best = min(haversine_m(lat, lon, plat, plon) for plat, plon in points)
    for start, end in zip(points, points[1:]):
        best = min(best, point_to_segment_m(lat, lon, *start, *end))
    return best


class SegmentAdvisoryDecoder:
    """Decodes a `DsrcDecision` into driver-facing per-segment advisories.

    The base is each segment's own STATIC speed limit
    (`segment_speed_limits_kmh`), never HERE's `freeFlow`: `_command` in
    `src.sumo.mainz` multiplies the action fraction by the map's own speed
    limit, not by a live reading, so decoding against HERE's `freeFlow`
    would make the advisory depend on the feed twice (once through the
    observation, once through the decode) where the simulator depends on it
    once. On Mainz the two coincide at 60 km/h, so this choice is invisible
    on the demonstration and would only show up on a different network --
    which is why it is fixed now rather than left until it does.

    Task 144's safety filter bounds this output; this decoder returns the
    unfiltered advisory and does not add a floor of its own (plan_task145
    section 7: two floors applied in sequence is the defect
    `feedback_a_fix_replaces_what_it_should_add` describes).
    """

    def __init__(
        self,
        segment_ids: tuple[str, ...],
        segment_speed_limits_kmh: tuple[float, ...],
        segment_points: tuple[tuple[tuple[float, float], ...], ...],
        *,
        units: str = "mph",
        ego_match_tolerance_m: float = SEGMENT_MATCH_TOLERANCE_M,
        network_fingerprint: str | None = None,
        expected_network_fingerprint: str | None = None,
    ) -> None:
        if units not in ("mph", "kmh", "mps"):
            raise ValueError(f"unknown units '{units}'")
        if not (len(segment_ids) == len(segment_speed_limits_kmh) == len(segment_points)):
            raise ValueError("segment_ids, segment_speed_limits_kmh and segment_points "
                             "must be the same length, one entry per super-segment")
        self.units = units
        self.segment_ids = tuple(segment_ids)
        self.segment_speed_limits_mps = tuple(v / MPS_TO_KMH for v in segment_speed_limits_kmh)
        self.segment_points = tuple(segment_points)
        self.ego_match_tolerance_m = ego_match_tolerance_m
        # Unlike SegmentStateBuilder, this constructor is not handed enough
        # of the network definition to compute its own network_fingerprint
        # (segment_ids here are the display ids, not the edge-id lists the
        # hash covers, and neither feature_names nor lanes/length_km are
        # given at all) -- `from_network_definition` below computes it from
        # the full definition and passes both this and the value to check
        # it against through. Guards the runtime/decoder pair the same way
        # `SegmentStateBuilder.__init__` guards the runtime/builder pair
        # (validator round 1, fix 2b).
        self.network_fingerprint = network_fingerprint
        if expected_network_fingerprint is not None and self.network_fingerprint is None:
            raise ValueError(
                "SegmentAdvisoryDecoder was given expected_network_fingerprint "
                f"{expected_network_fingerprint!r} to check against, but no "
                "network_fingerprint of its own -- this constructor is not handed enough "
                "of the network definition to compute one (segment_ids here are display "
                "ids, not the edge-id lists the hash covers, and lanes/length_km are not "
                "given at all), so a mismatch would silently compare against None and "
                "never fire. Use SegmentAdvisoryDecoder.from_network_definition, which "
                "computes network_fingerprint from the full network definition and can "
                "actually check it, or omit expected_network_fingerprint if no check is "
                "wanted."
            )
        if (
            expected_network_fingerprint is not None
            and self.network_fingerprint is not None
            and expected_network_fingerprint != self.network_fingerprint
        ):
            raise RuntimeError(
                f"SegmentAdvisoryDecoder loaded network_fingerprint "
                f"{self.network_fingerprint}, but the paired runtime expects "
                f"{expected_network_fingerprint} -- same segment count, "
                "different segment order, ids, lanes, length or polylines; "
                "decoder and runtime must be constructed from the same "
                "network definition."
            )

    @classmethod
    def from_network_definition(
        cls,
        path: Path | str = DEFAULT_NETWORK_DEFINITION_PATH,
        *,
        units: str = "mph",
        expected_network_fingerprint: str | None = None,
    ) -> "SegmentAdvisoryDecoder":
        definition = json.loads(Path(path).read_text())
        segment_ids = tuple(segment["segment_id"] for segment in definition["segments"])
        edge_ids = tuple(tuple(segment["edge_ids"]) for segment in definition["segments"])
        speed_limits = tuple(
            float(segment["speed_limit_kmh"]) for segment in definition["segments"]
        )
        lanes = tuple(float(segment["lanes"]) for segment in definition["segments"])
        length_kms = tuple(float(segment["length_km"]) for segment in definition["segments"])
        points = tuple(
            tuple(
                (float(lat), float(lon))
                for edge in segment["edges"]
                for lat, lon in edge["polyline"]
            )
            for segment in definition["segments"]
        )
        definition_fingerprint = network_fingerprint(
            network_id=definition["network_id"],
            feature_names=tuple(definition["feature_names"]),
            segment_ids=edge_ids,
            segment_speed_limits_kmh=speed_limits,
            segment_lanes=lanes,
            segment_length_km=length_kms,
            segment_polylines=points,
        )
        return cls(
            segment_ids, speed_limits, points, units=units,
            network_fingerprint=definition_fingerprint,
            expected_network_fingerprint=expected_network_fingerprint,
        )

    def _display(self, mps: float) -> float:
        if self.units == "mph":
            return mps * MPS_TO_MPH
        if self.units == "kmh":
            return mps * MPS_TO_KMH
        return mps

    def _ego_segment(self, lat: float | None, lon: float | None) -> int | None:
        if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
            return None
        best_index, best_distance = None, None
        for index, points in enumerate(self.segment_points):
            if not points:
                continue
            distance = _distance_to_polyline(lat, lon, points)
            if best_distance is None or distance < best_distance:
                best_distance, best_index = distance, index
        if best_distance is None or best_distance > self.ego_match_tolerance_m:
            return None
        return best_index

    def decode(
        self,
        decision: DsrcDecision,
        *,
        ego_lat: float | None = None,
        ego_lon: float | None = None,
    ) -> SegmentAdvisory:
        ego_segment = self._ego_segment(ego_lat, ego_lon)
        if decision.actions is None:
            return SegmentAdvisory(
                units=self.units, outcome=decision.outcome, rows=(), ego_segment=ego_segment,
            )
        rows = []
        for index, segment_id in enumerate(self.segment_ids):
            action_index = int(decision.actions[index])
            fraction = SPEED_ACTION_FRACTIONS[action_index]
            recommended_mps = fraction * self.segment_speed_limits_mps[index]
            rows.append(SegmentAdvisoryRow(
                segment_id=segment_id,
                action_index=action_index,
                fraction=fraction,
                recommended_speed_mps=recommended_mps,
                recommended_speed_display=self._display(recommended_mps),
            ))
        return SegmentAdvisory(
            units=self.units, outcome=decision.outcome, rows=tuple(rows), ego_segment=ego_segment,
        )
