"""Assemble the DSRC controller's whole-network state from the traffic feed.

`SrcQNetwork` reads one row per Mainz super-segment:
`("speed", "free_flow", "jam_factor", "lanes", "length_km")`
(`policy.dsrc_contract.HERE_FEATURES`). Three of those five look dynamic and
are not: `src.sumo.mainz.MainzEnv._per_edge` sets `free_flow_kmh` from the
map's own speed limit on every step, never from a live reading -- the
docstring of `HERE_FEATURES` in that module says so directly ("free_flow,
lanes and length_km are static and come from the map rather than the API").
So only `speed` is ever a live quantity in the trained policy's input;
`free_flow` has to come from the same static network definition as `lanes`
and `length_km`, or the device would feed the network a feature that moved
when the policy was trained on one that never does. (An earlier draft of
this module read `free_flow` from HERE's own `freeFlow` field, following
plan_task145 section 4.2 literally; re-reading `mainz.py` line by line
during implementation showed that section to be wrong about which fields
are dynamic, and this module follows the simulator's actual behaviour
instead of the plan's paraphrase of it.)

`jam_factor` is the one field HERE's schema and the simulator disagree on
outright -- see `policy.dsrc_contract.jam_factor`'s docstring -- and is
always recomputed here from the two features above, never taken from a
matched link's own `jam_factor`.

**A HERE link is matched to its single nearest segment, not to every
segment within tolerance.** 11 of 66 Mainz super-segment pairs have
polylines within `SEGMENT_MATCH_TOLERANCE_M` of each other, so a link near
two segments would otherwise be credited to both. `build` below assigns
each link to its argmin segment (still subject to the tolerance check), an
exclusive partition -- matching the simulator's own super-segments, which
are also an exclusive partition of the road network, rather than an
overlapping set of candidate matches.

**Every segment must be feed-measured, or nothing is emitted.** Substituting
one segment's row with a zero-fill changed at least one OTHER segment's
action in 675 of 4,000 draws (plan_task145 section 1.7): the state is
flattened through one dense layer, so a fabricated row is not confined to
its own output. `SegmentStateResult.state` is therefore `None` unless every
segment's basis is `SEGMENT_BASIS_MEASURED`; the per-segment bases are
still reported when it is not, so a caller can log why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policy.dsrc_contract import jam_factor, network_fingerprint
from sensors.here_feed import FlowLink, FlowReading, Outcome

JETSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = JETSON_DIR.parents[1]
DEFAULT_NETWORK_DEFINITION_PATH = REPO_ROOT / "specs" / "dsrc_network_mainz.json"

#: How far a HERE link's shape may pass from a segment's own polyline and
#: still be treated as describing that segment.
#:
#: This reuses `sensors.here_feed.ASSOCIATION_RADIUS_M` (60.0 m) rather than
#: a value calibrated for this purpose, and that is a real limitation, not a
#: rounding choice: the New Jersey corpus plan_task145 step 3 names (123
#: HERE bodies collected 2026-09-08, `plans/implementation_records.md` item
#: 71) is not present anywhere in this checkout -- it was written to a
#: run directory this repository never committed -- so there is no real
#: corpus here to sweep a tolerance against. Mainz's own polylines are
#: independently synthetic (`scripts/export_dsrc_network.py`'s docstring:
#: `mainz.net.xml` carries no geo-projection at all), so even with the
#: corpus in hand there is no ground truth for which HERE link belongs to
#: which Mainz super-segment (plan_task145 risk 4 names the weaker version
#: of this problem). `ASSOCIATION_RADIUS_M`'s reasoning -- a link further
#: than this describes a different road -- is the same physical argument
#: this match needs, so it is reused rather than a new number invented
#: from a sweep over data this module also had to fabricate. See
#: `scripts/measure_dsrc_match_tolerance.py` for the sweep that was run
#: and why its curve is provisional.
SEGMENT_MATCH_TOLERANCE_M = 60.0

#: At least one link's nearest segment was this one, and within tolerance,
#: in this decision's snapshot -- not "some link was within tolerance of
#: this segment", since a link within tolerance of more than one segment is
#: credited to only the nearest (the exclusive-partition rule above).
SEGMENT_BASIS_MEASURED = "measured"
#: No link matched; only the network definition's static fields are known.
#: Not evidence about this decision -- the same discipline
#: `perception.provenance.SUBSTITUTED` applies to its own classes.
SEGMENT_BASIS_SUBSTITUTED = "substituted"
#: The network definition has no entry for this segment. A load-time
#: refusal (`DsrcRuntime.__init__` checks the definition against the bundle
#: manifest), unreachable from `SegmentStateBuilder.build` since it can only
#: ever iterate the segments its own definition names.
SEGMENT_BASIS_ABSENT = "absent"

#: Every segment matched at least one link; `state` is populated.
OUTCOME_OK = "ok"
#: The feed answered, but not every segment matched a link; `state` is None.
OUTCOME_INCOMPLETE_COVERAGE = "incomplete_coverage"


@dataclass(frozen=True)
class SegmentStateResult:
    """One decision's worth of super-segment state, or the reason there is none."""

    #: `(num_segments, num_features)` float32, or `None` when incomplete.
    state: np.ndarray | None
    #: One of the `SEGMENT_BASIS_*` classes, per segment, in network order.
    segment_basis: tuple[str, ...]
    #: `OUTCOME_OK`, `OUTCOME_INCOMPLETE_COVERAGE`, or a `sensors.here_feed
    #: .Outcome` value when the feed itself had nothing to offer (stale, no
    #: response yet, ...) -- propagated rather than collapsed into one
    #: generic name, since a caller deciding whether to keep waiting needs
    #: to tell "stale" from "no response yet" apart.
    outcome: str
    #: How many links matched each segment, in network order.
    matched_links: tuple[int, ...]
    response_age_s: float | None


class SegmentStateBuilder:
    """Turns a HERE snapshot into the DSRC controller's `(segments, features)` state."""

    def __init__(
        self,
        network_definition_path: Path | str = DEFAULT_NETWORK_DEFINITION_PATH,
        *,
        match_tolerance_m: float = SEGMENT_MATCH_TOLERANCE_M,
        expected_network_fingerprint: str | None = None,
    ) -> None:
        definition = json.loads(Path(network_definition_path).read_text())
        self.network_id: str = definition["network_id"]
        self.feature_names: tuple[str, ...] = tuple(definition["feature_names"])
        self.match_tolerance_m = match_tolerance_m
        self.segment_ids: tuple[str, ...] = tuple(
            segment["segment_id"] for segment in definition["segments"]
        )
        self.segment_edge_ids: tuple[tuple[str, ...], ...] = tuple(
            tuple(segment["edge_ids"]) for segment in definition["segments"]
        )
        self.segment_speed_limits_kmh: tuple[float, ...] = tuple(
            float(segment["speed_limit_kmh"]) for segment in definition["segments"]
        )
        self._static_lanes: tuple[float, ...] = tuple(
            float(segment["lanes"]) for segment in definition["segments"]
        )
        self._static_length_km: tuple[float, ...] = tuple(
            float(segment["length_km"]) for segment in definition["segments"]
        )
        # Every point of every edge's polyline, per segment, computed once:
        # this never changes tick to tick, only the links do.
        self._segment_points: tuple[tuple[tuple[float, float], ...], ...] = tuple(
            tuple(
                (float(lat), float(lon))
                for edge in segment["edges"]
                for lat, lon in edge["polyline"]
            )
            for segment in definition["segments"]
        )

        # This builder loads its network definition independently of
        # whichever DsrcRuntime it is paired with -- nothing enforced the
        # two agreed. A builder built on a definition with two segments
        # swapped and a runtime built on the correct one produced no
        # refusal and changed the emitted action vector in 536 of 2,000
        # random draws (26.8%). `network_fingerprint` covers exactly the
        # fields above (network_id, feature_names, segment edge ids, speed
        # limits, lanes, length_km and the polylines), so a caller that
        # already has the paired runtime's own `DsrcRuntime.network_fingerprint`
        # can hand it in here and have a mismatch refused at construction,
        # the same way `DsrcRuntime.__init__` refuses a bundle whose
        # fingerprint disagrees with its own network definition.
        self.network_fingerprint = network_fingerprint(
            network_id=self.network_id,
            feature_names=self.feature_names,
            segment_ids=self.segment_edge_ids,
            segment_speed_limits_kmh=self.segment_speed_limits_kmh,
            segment_lanes=self._static_lanes,
            segment_length_km=self._static_length_km,
            segment_polylines=self._segment_points,
        )
        if (
            expected_network_fingerprint is not None
            and expected_network_fingerprint != self.network_fingerprint
        ):
            raise RuntimeError(
                f"SegmentStateBuilder loaded network_fingerprint "
                f"{self.network_fingerprint} from {network_definition_path}, but "
                f"the paired runtime expects {expected_network_fingerprint} -- "
                "same segment count and feature names, different segment "
                "order, ids, lanes, length or polylines; builder and runtime "
                "must be constructed from the same network definition."
            )

    @property
    def num_segments(self) -> int:
        return len(self.segment_ids)

    def build(
        self, links: tuple[FlowLink, ...], reading: FlowReading, t_mono: float,
    ) -> SegmentStateResult:
        """One decision's state, from `HereFeed.snapshot_links`'s own return.

        `t_mono` is accepted for symmetry with the rest of this deployment's
        builders and is not currently used: `reading.response_age_s` already
        carries the age this decision needs, computed by `snapshot_links`
        against the same clock.
        """
        n = self.num_segments
        if reading.outcome != Outcome.OK:
            return SegmentStateResult(
                state=None,
                segment_basis=tuple(SEGMENT_BASIS_SUBSTITUTED for _ in range(n)),
                outcome=reading.outcome,
                matched_links=tuple(0 for _ in range(n)),
                response_age_s=reading.response_age_s,
            )

        matched_speeds_kmh: list[list[float]] = [[] for _ in range(n)]
        for link in links:
            if link.speed_mps is None:
                continue
            # The single nearest segment, not every segment within
            # tolerance: with no exclusivity and no argmin, a link near two
            # segments' polylines (11 of 66 segment pairs in the Mainz
            # network are within 60 m of each other) was credited to both,
            # so a segment could read "measured" on a link that actually
            # describes its neighbour. The simulator's own super-segments
            # are an exclusive partition of the road network, so this
            # reproduces that partition rather than approximating it.
            best_index, best_distance = None, None
            for index, points in enumerate(self._segment_points):
                if not points:
                    continue
                distance = min(link.distance_m(lat, lon) for lat, lon in points)
                if best_distance is None or distance < best_distance:
                    best_distance, best_index = distance, index
            if best_index is not None and best_distance <= self.match_tolerance_m:
                matched_speeds_kmh[best_index].append(link.speed_mps * 3.6)

        matched_counts = tuple(len(values) for values in matched_speeds_kmh)
        basis = tuple(
            SEGMENT_BASIS_MEASURED if count > 0 else SEGMENT_BASIS_SUBSTITUTED
            for count in matched_counts
        )
        complete = all(b == SEGMENT_BASIS_MEASURED for b in basis)

        state = None
        if complete:
            rows = []
            for index in range(n):
                speed_kmh = float(np.mean(matched_speeds_kmh[index]))
                free_flow_kmh = self.segment_speed_limits_kmh[index]
                row = {
                    "speed": speed_kmh,
                    "free_flow": free_flow_kmh,
                    "jam_factor": jam_factor(speed_kmh, free_flow_kmh),
                    "lanes": self._static_lanes[index],
                    "length_km": self._static_length_km[index],
                }
                rows.append([row[name] for name in self.feature_names])
            state = np.array(rows, dtype=np.float32)

        return SegmentStateResult(
            state=state,
            segment_basis=basis,
            outcome=OUTCOME_OK if complete else OUTCOME_INCOMPLETE_COVERAGE,
            matched_links=matched_counts,
            response_age_s=reading.response_age_s,
        )
