"""perception/segment_state.py: assembling the DSRC state from a HERE snapshot.

Every case runs on a small synthetic two-segment network definition, written
to `tmp_path` -- this repo may not call HERE (`test_here_feed.py`'s own
docstring) and Mainz's own network definition carries no ground-truth
correspondence to any real traffic feed (see `segment_state.py`'s module
docstring), so there is nothing to gain by pointing these tests at the real
`specs/dsrc_network_mainz.json`. `test_export_dsrc_network.py` already
checks that file's own numbers.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from perception.segment_state import (
    SEGMENT_BASIS_MEASURED,
    SEGMENT_BASIS_SUBSTITUTED,
    OUTCOME_INCOMPLETE_COVERAGE,
    OUTCOME_OK,
    SegmentStateBuilder,
)
from sensors.here_feed import FlowLink, FlowReading, Outcome

SEG0_LAT, SEG0_LON = 40.0, -74.0
SEG1_LAT, SEG1_LON = 40.2, -74.3


def offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lat + dlat, lon + dlon)


def straight_polyline(lat: float, lon: float, east_m: float, points: int = 3) -> list[list[float]]:
    step = east_m / max(points - 1, 1)
    return [list(offset(lat, lon, 0.0, step * i)) for i in range(points)]


def _definition_dict() -> dict:
    return {
        "network_id": "test_net",
        "feature_names": ["speed", "free_flow", "jam_factor", "lanes", "length_km"],
        "action_fractions": [0.5, 0.75, 1.0],
        "segments": [
            {
                "segment_id": "0",
                "edge_ids": ["e0"],
                "lanes": 2.0,
                "speed_limit_kmh": 60.0,
                "length_km": 1.0,
                "edges": [{
                    "edge_id": "e0", "lanes": 2.0, "length_m": 1000.0,
                    "speed_limit_kmh": 60.0,
                    "polyline": straight_polyline(SEG0_LAT, SEG0_LON, 1000.0),
                }],
            },
            {
                "segment_id": "1",
                "edge_ids": ["e1"],
                "lanes": 3.0,
                "speed_limit_kmh": 100.0,
                "length_km": 2.0,
                "edges": [{
                    "edge_id": "e1", "lanes": 3.0, "length_m": 2000.0,
                    "speed_limit_kmh": 100.0,
                    "polyline": straight_polyline(SEG1_LAT, SEG1_LON, 1000.0),
                }],
            },
        ],
    }


def write_definition(tmp_path) -> str:
    path = tmp_path / "network.json"
    path.write_text(json.dumps(_definition_dict()))
    return str(path)


def write_swapped_definition(tmp_path) -> str:
    """The same two segments, in the opposite order.

    Reproduces the validator's round-1 repro for fix 2b: a builder loaded
    from this file and a runtime (or another builder) loaded from
    `write_definition`'s file describe the same two segments but disagree
    on which is "segment 0" -- same shapes throughout, so nothing before
    the fingerprint check would catch it.
    """
    definition = _definition_dict()
    definition["segments"] = list(reversed(definition["segments"]))
    path = tmp_path / "network_swapped.json"
    path.write_text(json.dumps(definition))
    return str(path)


def write_adjacent_definition(tmp_path) -> str:
    """Two segments 40 m apart -- both within the 60 m match tolerance of
    a link that sits close to one of them, at a scale a unit test can
    check exactly. Reproduces the validator's finding on the real Mainz
    network (11 of 66 segment pairs have polylines within 60 m of each
    other, and 3 of 12 links in the tests'/bench's synthetic feed were
    claimed by two segments each)."""
    definition = {
        "network_id": "test_net",
        "feature_names": ["speed", "free_flow", "jam_factor", "lanes", "length_km"],
        "action_fractions": [0.5, 0.75, 1.0],
        "segments": [
            {
                "segment_id": "0", "edge_ids": ["e0"], "lanes": 2.0,
                "speed_limit_kmh": 60.0, "length_km": 1.0,
                "edges": [{
                    "edge_id": "e0", "lanes": 2.0, "length_m": 1000.0,
                    "speed_limit_kmh": 60.0,
                    "polyline": straight_polyline(SEG0_LAT, SEG0_LON, 1000.0),
                }],
            },
            {
                "segment_id": "1", "edge_ids": ["e1"], "lanes": 2.0,
                "speed_limit_kmh": 60.0, "length_km": 1.0,
                "edges": [{
                    "edge_id": "e1", "lanes": 2.0, "length_m": 1000.0,
                    "speed_limit_kmh": 60.0,
                    "polyline": [
                        list(offset(SEG0_LAT, SEG0_LON, north_m=40.0, east_m=e))
                        for e in (0.0, 500.0, 1000.0)
                    ],
                }],
            },
        ],
    }
    path = tmp_path / "network_adjacent.json"
    path.write_text(json.dumps(definition))
    return str(path)


def link_on(lat: float, lon: float, east_m: float, *, speed_mps: float,
           free_flow_mps: float = 30.0, jam_factor: float | None = 2.0) -> FlowLink:
    points = tuple(tuple(p) for p in straight_polyline(lat, lon, east_m))
    return FlowLink(
        points=points, speed_mps=speed_mps, free_flow_mps=free_flow_mps,
        jam_factor=jam_factor, confidence=0.9, traversability="open", length_m=east_m,
    )


@pytest.fixture
def builder(tmp_path):
    return SegmentStateBuilder(write_definition(tmp_path), match_tolerance_m=60.0)


def ok_reading(response_age_s: float = 1.0) -> FlowReading:
    return FlowReading(outcome=Outcome.OK, response_age_s=response_age_s)


class TestFullCoverage:
    def test_both_segments_measured_and_state_populated(self, builder):
        links = (
            link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0),
            link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0),
        )
        result = builder.build(links, ok_reading(), t_mono=0.0)

        assert result.outcome == OUTCOME_OK
        assert result.segment_basis == (SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_MEASURED)
        assert result.matched_links == (1, 1)
        assert result.state is not None
        assert result.state.shape == (2, 5)
        assert result.state.dtype == np.float32

    def test_speed_feature_is_the_matched_links_speed_in_kmh(self, builder):
        links = (
            link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0),
            link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0),
        )
        result = builder.build(links, ok_reading(), t_mono=0.0)
        assert result.state[0][0] == pytest.approx(20.0 * 3.6)
        assert result.state[1][0] == pytest.approx(25.0 * 3.6)

    def test_free_flow_comes_from_the_static_speed_limit_not_the_link(self, builder):
        """The plan-literal reading (HERE's own freeFlow) would put 30*3.6
        here; the simulator's own free_flow is always the map's speed
        limit, never a live reading -- see the module docstring."""
        links = (
            link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0, free_flow_mps=999.0),
            link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0, free_flow_mps=999.0),
        )
        result = builder.build(links, ok_reading(), t_mono=0.0)
        assert result.state[0][1] == pytest.approx(60.0)
        assert result.state[1][1] == pytest.approx(100.0)

    def test_jam_factor_is_recomputed_from_speed_and_static_free_flow(self, builder):
        from policy.dsrc_contract import jam_factor as expected_jam_factor

        links = (
            link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0, jam_factor=7.5),
            link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0, jam_factor=7.5),
        )
        result = builder.build(links, ok_reading(), t_mono=0.0)
        # Never the link's own reported jam_factor (7.5 for both, distinct
        # segments should disagree once free_flow does).
        assert result.state[0][2] == pytest.approx(
            expected_jam_factor(20.0 * 3.6, 60.0)
        )
        assert result.state[1][2] == pytest.approx(
            expected_jam_factor(25.0 * 3.6, 100.0)
        )
        assert result.state[0][2] != pytest.approx(7.5)

    def test_lanes_and_length_km_come_from_the_static_definition(self, builder):
        result = builder.build(
            (link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0),
             link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0)),
            ok_reading(), t_mono=0.0,
        )
        assert result.state[0][3] == pytest.approx(2.0)
        assert result.state[0][4] == pytest.approx(1.0)
        assert result.state[1][3] == pytest.approx(3.0)
        assert result.state[1][4] == pytest.approx(2.0)

    def test_multiple_matching_links_are_averaged(self, builder):
        links = (
            link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=10.0),
            link_on(SEG0_LAT, SEG0_LON, 900.0, speed_mps=30.0),
            link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0),
        )
        result = builder.build(links, ok_reading(), t_mono=0.0)
        assert result.matched_links == (2, 1)
        assert result.state[0][0] == pytest.approx(((10.0 + 30.0) / 2.0) * 3.6)


class TestOneSegmentMissing:
    def test_no_state_and_the_missing_segment_is_substituted(self, builder):
        links = (link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0),)
        result = builder.build(links, ok_reading(), t_mono=0.0)

        assert result.outcome == OUTCOME_INCOMPLETE_COVERAGE
        assert result.state is None
        assert result.segment_basis == (SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_SUBSTITUTED)
        assert result.matched_links == (1, 0)


class TestAllSegmentsMissing:
    def test_a_response_with_no_relevant_links_substitutes_every_segment(self, builder):
        far_away = link_on(0.0, 0.0, 1000.0, speed_mps=20.0)
        result = builder.build((far_away,), ok_reading(), t_mono=0.0)

        assert result.outcome == OUTCOME_INCOMPLETE_COVERAGE
        assert result.state is None
        assert result.segment_basis == (SEGMENT_BASIS_SUBSTITUTED, SEGMENT_BASIS_SUBSTITUTED)
        assert result.matched_links == (0, 0)

    def test_an_empty_link_tuple_substitutes_every_segment(self, builder):
        result = builder.build((), ok_reading(), t_mono=0.0)
        assert result.state is None
        assert result.segment_basis == (SEGMENT_BASIS_SUBSTITUTED, SEGMENT_BASIS_SUBSTITUTED)


class TestStaleResponse:
    def test_a_stale_reading_substitutes_every_segment_and_propagates_the_outcome(self, builder):
        stale = FlowReading(outcome=Outcome.STALE, response_age_s=45.0)
        result = builder.build((link_on(SEG0_LAT, SEG0_LON, 1000.0, speed_mps=20.0),),
                               stale, t_mono=0.0)

        assert result.outcome == Outcome.STALE
        assert result.state is None
        assert result.segment_basis == (SEGMENT_BASIS_SUBSTITUTED, SEGMENT_BASIS_SUBSTITUTED)
        assert result.matched_links == (0, 0)
        assert result.response_age_s == pytest.approx(45.0)

    def test_no_response_yet_also_substitutes_every_segment(self, builder):
        result = builder.build((), FlowReading(outcome=Outcome.NO_RESPONSE_YET), t_mono=0.0)
        assert result.outcome == Outcome.NO_RESPONSE_YET
        assert result.state is None


class TestMatchTolerance:
    def test_a_link_within_tolerance_of_the_segment_matches(self, builder):
        # 40 m north of segment 0's polyline, inside the 60 m tolerance.
        near_lat, near_lon = offset(SEG0_LAT, SEG0_LON, north_m=40.0, east_m=0.0)
        link = link_on(near_lat, near_lon, 1000.0, speed_mps=20.0)
        result = builder.build((link, link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0)),
                               ok_reading(), t_mono=0.0)
        assert result.segment_basis[0] == SEGMENT_BASIS_MEASURED

    def test_a_link_well_beyond_tolerance_does_not_match(self, builder):
        # 500 m north of segment 0's polyline, well outside the 60 m tolerance.
        far_lat, far_lon = offset(SEG0_LAT, SEG0_LON, north_m=500.0, east_m=0.0)
        link = link_on(far_lat, far_lon, 1000.0, speed_mps=20.0)
        result = builder.build((link, link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0)),
                               ok_reading(), t_mono=0.0)
        assert result.segment_basis[0] == SEGMENT_BASIS_SUBSTITUTED


class TestLinkToSegmentPartition:
    """Validator round 1, fix 3: `build` used to test every segment
    independently with no exclusivity and no `break` -- a link within
    tolerance of two segments' polylines was credited to both. Only the
    single nearest segment (argmin over segments of the min vertex
    distance) is credited now, reproducing the simulator's own exclusive
    partition of the road network."""

    def test_a_link_between_two_close_segments_matches_only_the_nearer_one(self, tmp_path):
        builder = SegmentStateBuilder(write_adjacent_definition(tmp_path), match_tolerance_m=60.0)
        # 5 m north of segment 0's polyline and (40 - 5) = 35 m south of
        # segment 1's -- both inside the 60 m tolerance under the old
        # any-within-tolerance rule, so the old code would have credited
        # this one link to both segments.
        near_lat, near_lon = offset(SEG0_LAT, SEG0_LON, north_m=5.0, east_m=500.0)
        link = link_on(near_lat, near_lon, 200.0, speed_mps=20.0)

        result = builder.build((link,), ok_reading(), t_mono=0.0)

        assert result.matched_links == (1, 0)
        assert result.segment_basis == (SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_SUBSTITUTED)


class TestBuilderProperties:
    def test_segment_ids_and_edge_ids_and_speed_limits_are_exposed_in_order(self, builder):
        assert builder.segment_ids == ("0", "1")
        assert builder.segment_edge_ids == (("e0",), ("e1",))
        assert builder.segment_speed_limits_kmh == (60.0, 100.0)
        assert builder.num_segments == 2

    def test_a_link_with_no_speed_is_never_counted_as_a_match(self, builder):
        no_speed = FlowLink(
            points=tuple(tuple(p) for p in straight_polyline(SEG0_LAT, SEG0_LON, 1000.0)),
            speed_mps=None, free_flow_mps=None, jam_factor=5.0,
            confidence=0.9, traversability="open", length_m=1000.0,
        )
        result = builder.build((no_speed, link_on(SEG1_LAT, SEG1_LON, 1000.0, speed_mps=25.0)),
                               ok_reading(), t_mono=0.0)
        assert result.segment_basis[0] == SEGMENT_BASIS_SUBSTITUTED
        assert result.matched_links[0] == 0


class TestNetworkFingerprintGuard:
    """Validator round 1, fix 2b: a builder loaded from one network
    definition and a runtime (or another builder) loaded from a
    differently-ordered one produced no refusal and a silently wrong
    action vector in 26.8% of a 2,000-state sample. `SegmentStateBuilder`
    now takes the paired runtime's own `network_fingerprint` and refuses a
    mismatch at construction."""

    def test_no_expected_fingerprint_does_not_raise(self, tmp_path):
        """Backward compatible: existing callers that pass nothing get no
        check at all, the same as before this guard existed."""
        SegmentStateBuilder(write_definition(tmp_path))

    def test_matching_fingerprint_does_not_raise(self, tmp_path):
        reference = SegmentStateBuilder(write_definition(tmp_path))
        SegmentStateBuilder(
            write_definition(tmp_path),
            expected_network_fingerprint=reference.network_fingerprint,
        )

    def test_two_segments_swapped_is_refused(self, tmp_path):
        reference = SegmentStateBuilder(write_definition(tmp_path))
        with pytest.raises(RuntimeError, match="network_fingerprint"):
            SegmentStateBuilder(
                write_swapped_definition(tmp_path),
                expected_network_fingerprint=reference.network_fingerprint,
            )
