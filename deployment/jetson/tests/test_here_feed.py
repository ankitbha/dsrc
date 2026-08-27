"""Ingesting the traffic feed, and refusing to guess when it cannot be read.

Every test here runs on a synthetic body. Nothing in this repo may call HERE -- the
key is shared with Nash production and absent from every build -- so the parser will
meet its first real response in production, and these cover the ways a body can
disagree with the schema it was written against.
"""

from __future__ import annotations

import json
import time
import math
from dataclasses import replace

import pytest

from sensors.gps_reader import GpsFix
from sensors.here_feed import (
    ASSOCIATION_RADIUS_M,
    DOWNSTREAM_HORIZON_M,
    MAX_RESPONSE_AGE_S,
    HereFeed,
    Outcome,
    angle_between,
    bearing_deg,
    parse_flow,
)

#: A point on the A4 west of London, and one 500 m due east of it.
HOME = (51.4900, -0.2000)


def offset(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """A point displaced by metres, near enough for a test at this latitude."""
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lat + dlat, lon + dlon)


def stretch(lat: float, lon: float, east_m: float, points: int = 5) -> list[dict]:
    """A link running `east_m` metres east from a point, sampled along its length.

    Links are polylines hundreds of metres long, and that matters here rather than
    being decoration: a fixture that gave each link a single point put a vehicle
    driving along a road 200 m from that road's only point, so the association
    radius rejected the carriageway under its wheels. A single-point link is not a
    link and cannot exercise the geometry.
    """
    step = east_m / max(points - 1, 1)
    return [
        {"lat": offset(lat, lon, 0.0, step * i)[0], "lng": offset(lat, lon, 0.0, step * i)[1]}
        for i in range(points)
    ]


def body(*shapes, **flow_overrides) -> bytes:
    """A v7-shaped flow response carrying the given link shapes."""
    results = []
    for shape in shapes:
        flow = {"speed": 13.4, "freeFlow": 27.8, "jamFactor": 4.2, "confidence": 0.95}
        flow.update(flow_overrides)
        results.append({
            "location": {"length": 800.0, "shape": {"links": [{"points": shape}]}},
            "currentFlow": flow,
        })
    return json.dumps({"results": results}).encode("utf-8")


def one_point(lat: float, lon: float) -> list[dict]:
    """A degenerate single-point shape, for the parse tests that only need a shape."""
    return [{"lat": lat, "lng": lon}]


def fix(lat: float, lon: float, heading_deg: float = 90.0, valid: bool = True,
        t_mono: float = 0.0) -> GpsFix:
    """A fix, stamped at the moment the query asks about it unless told otherwise.

    `t_mono` used to default to zero while every caller queried at 100 s or later,
    so every fix in this file was already minutes old and nothing noticed -- which
    is exactly how `at()` came to have no fix-age gate at all. Passing the query's
    own time makes freshness the default and staleness something a test has to ask
    for.
    """
    return GpsFix(valid=valid, lat=lat, lon=lon, heading_deg=heading_deg,
                  speed_mps=20.0, t_mono=t_mono)


class TestParsing:

    def test_a_body_that_is_not_json_is_not_an_empty_road(self):
        # None and [] are different answers and must stay different: the first is
        # "not understood", the second "understood, no links". Folded together, a
        # broken parser reports clear road ahead.
        assert parse_flow(b"<html>gateway timeout</html>") is None
        assert parse_flow(b"") is None
        assert parse_flow(b'{"results": {}}') is None
        assert parse_flow(b'{"results": []}') == []

    def test_a_link_with_no_flow_is_a_shape_and_not_a_measurement(self):
        parsed = parse_flow(json.dumps({"results": [
            {"location": {"shape": {"links": [{"points": [{"lat": 51.49, "lng": -0.2}]}]}}}
        ]}).encode())
        assert len(parsed) == 1
        assert parsed[0].usable is False
        assert parsed[0].jam_factor is None

    def test_out_of_range_values_are_refused_rather_than_clamped(self):
        # A jamFactor of 40 is not a jam factor of 10. Clamping turns a response
        # this parser does not understand into a confident maximum.
        parsed = parse_flow(body(one_point(*HOME), jamFactor=40.0, confidence=3.0, speed=-5.0))
        assert parsed[0].jam_factor is None
        assert parsed[0].confidence is None
        assert parsed[0].speed_mps is None

    def test_non_finite_and_wrong_typed_values_are_refused(self):
        raw = '{"results":[{"location":{"shape":{"links":[{"points":[{"lat":51.49,"lng":-0.2}]}]}},' \
              '"currentFlow":{"speed":"fast","jamFactor":NaN,"confidence":true}}]}'
        parsed = parse_flow(raw.encode())
        assert parsed[0].speed_mps is None
        assert parsed[0].jam_factor is None
        # True is an int in Python and must not read as a confidence of 1.0.
        assert parsed[0].confidence is None

    def test_a_zero_jam_factor_survives_because_zero_is_a_measurement(self):
        # The whole point of None-for-unknown is that a real zero still means
        # something. A parser that folded them would lose "the road is clear".
        parsed = parse_flow(body(one_point(*HOME), jamFactor=0.0))
        assert parsed[0].jam_factor == 0.0
        assert parsed[0].usable is True


class TestGeometry:

    def test_bearing_and_angle_agree_with_the_compass(self):
        north = offset(*HOME, north_m=1000.0, east_m=0.0)
        east = offset(*HOME, north_m=0.0, east_m=1000.0)
        assert bearing_deg(*HOME, *north) == 0.0 or abs(bearing_deg(*HOME, *north)) < 1.0
        assert abs(bearing_deg(*HOME, *east) - 90.0) < 1.0
        assert angle_between(350.0, 10.0) == 20.0
        assert angle_between(10.0, 350.0) == 20.0


class TestOutcomes:

    def test_nothing_received_yet_says_so(self):
        feed = HereFeed()
        assert feed.at(fix(*HOME, t_mono=100.0), t_mono=100.0).outcome == Outcome.NO_RESPONSE_YET

    def test_an_http_error_is_named_and_counted_not_swallowed(self):
        feed = HereFeed()
        assert feed.offer(status=503, body=b"", received_t_mono=100.0) is False
        assert feed.at(fix(*HOME, t_mono=100.0), t_mono=100.0).outcome == Outcome.HTTP_ERROR
        assert feed.refused_by_reason == {"http_error:status 503": 1}

    def test_an_unparseable_body_is_named(self):
        feed = HereFeed()
        assert feed.offer(status=200, body=b"<html>", received_t_mono=100.0) is False
        assert feed.at(fix(*HOME, t_mono=100.0), t_mono=100.0).outcome == Outcome.UNPARSEABLE

    def test_a_link_ahead_is_reported_with_its_flow_and_a_measured_age(self):
        feed = HereFeed()
        road = stretch(*HOME, east_m=800.0)
        assert feed.offer(status=200, body=body(road), received_t_mono=100.0) is True

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=102.5), t_mono=102.5)
        assert reading.ok
        assert reading.link.jam_factor == 4.2
        assert reading.response_age_s == 2.5

    def test_a_link_behind_us_is_not_downstream(self):
        # Heading east, with the only link to the west. Reporting it would let the
        # queue we have just cleared describe the road in front.
        feed = HereFeed()
        # A link that runs west and ends where we are: we are on it, so the radius
        # matches, and nothing of it lies in front.
        behind = stretch(*offset(*HOME, 0.0, -800.0), east_m=800.0)
        feed.offer(status=200, body=body(behind), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0)
        assert reading.outcome == Outcome.NO_LINK_AHEAD
        assert reading.link is None

    def test_a_road_the_response_does_not_cover_is_not_matched(self):
        # The association radius answers "are we on a road this body covers at
        # all". Without it the nearest link a kilometre away would be reported as
        # ours -- a different street, described confidently.
        feed = HereFeed()
        far = stretch(*offset(*HOME, north_m=1200.0, east_m=0.0), east_m=800.0)
        feed.offer(status=200, body=body(far), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=0.0, t_mono=100.0), t_mono=100.0)
        assert reading.outcome == Outcome.NO_LINK_MATCHED

    def test_the_radius_and_the_horizon_are_not_the_same_threshold(self):
        # Pinned because an earlier version tested `d <= radius or d <= horizon`,
        # which is just the horizon since it is fifty times larger -- so the radius
        # decided nothing at all.
        assert ASSOCIATION_RADIUS_M < DOWNSTREAM_HORIZON_M / 10

    def test_a_fix_without_a_heading_cannot_say_what_is_ahead(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=float("nan"), t_mono=100.0), t_mono=100.0)
        assert reading.outcome == Outcome.UNUSABLE_FIX
        assert reading.detail == "no heading"

    def test_an_invalid_fix_is_named_rather_than_guessed_from(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        assert feed.at(fix(*HOME, valid=False, t_mono=100.0), t_mono=100.0).outcome == Outcome.UNUSABLE_FIX

    def test_a_response_past_its_age_is_stale_not_the_last_answer(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME, t_mono=100.0), t_mono=100.0).ok

        reading = feed.at(fix(*HOME, t_mono=100.0 + MAX_RESPONSE_AGE_S + 1.0), t_mono=100.0 + MAX_RESPONSE_AGE_S + 1.0)
        assert reading.outcome == Outcome.STALE
        assert reading.link is None


class TestCaching:

    def test_a_query_between_responses_is_reanswered_from_the_new_position(self):
        # The cache holds links, not the last answer. Re-serving the answer would
        # keep reporting congestion for a link already behind us.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=500.0)), received_t_mono=100.0)

        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0).ok
        # Driven to the end of that link. Still on it, so it still matches -- but
        # none of it is in front any more, answered from the same cached response
        # rather than by repeating the previous answer.
        end = offset(*HOME, north_m=0.0, east_m=500.0)
        assert feed.at(fix(*end, heading_deg=90.0, t_mono=101.0), t_mono=101.0).outcome == Outcome.NO_LINK_AHEAD

    def test_a_newer_response_that_matched_nothing_still_supersedes(self):
        # Superseding is by arrival, not by content: the older body describes a
        # place we have left, so keeping it because it happened to match is worse
        # than having nothing.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0).ok

        feed.offer(status=200, body=b'{"results": []}', received_t_mono=105.0)
        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=105.0), t_mono=105.0).outcome == Outcome.NO_LINK_MATCHED

    def test_a_refused_response_does_not_replace_a_usable_one(self):
        # An HTTP error says nothing about the road, so it must not throw away
        # links that still describe it.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        feed.offer(status=500, body=b"", received_t_mono=101.0)

        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=101.0), t_mono=101.0).ok


class TestRecord:

    def test_the_record_says_feed_lag_is_unmeasurable_rather_than_omitting_it(self):
        # An absent key reads as an oversight; a null with a note reads as the
        # measurement nobody can make. HERE v7 flow carries no per-result
        # observation time, so the larger part of the total staleness is unknown.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        record = feed.to_record()

        assert record["feed_lag_s"] is None
        assert "feed_lag_note" in record
        assert record["responses_received"] == 1
        assert record["responses_parsed"] == 1

    def test_refusals_are_counted_by_reason(self):
        feed = HereFeed()
        feed.offer(status=503, body=b"", received_t_mono=100.0)
        feed.offer(status=200, body=b"<html>", received_t_mono=101.0)

        assert feed.to_record()["responses_received"] == 2
        assert feed.to_record()["responses_parsed"] == 0
        assert sum(feed.refused_by_reason.values()) == 2


class TestNothingInABodyMayRaise:
    """A remote body must always end as a named outcome.

    An OverflowError out of `float()` left the parser, killed the reader thread for
    the rest of the drive, counted nothing, and left the feed serving pre-poison
    links until they aged out. One malformed response and the feed goes quiet.
    """

    def test_an_integer_too_large_for_a_double_is_refused_not_raised(self):
        huge = b'{"results":[{"location":{"shape":{"links":[{"points":[{"lat":51.49,"lng":-0.2}]}]}},' \
               b'"currentFlow":{"jamFactor":' + b"9" * 400 + b'}}]}'
        parsed = parse_flow(huge)
        assert parsed is not None
        assert parsed[0].jam_factor is None

    def test_a_huge_integer_in_the_geometry_is_refused_not_raised(self):
        huge = b'{"results":[{"location":{"shape":{"links":[{"points":[{"lat":' + b"9" * 400 + \
               b',"lng":-0.2}]}]}},"currentFlow":{"jamFactor":4.2}}]}'
        assert parse_flow(huge) == []

    def test_a_deeply_nested_body_is_not_understood_rather_than_fatal(self):
        deep = b"[" * 60_000 + b"]" * 60_000
        assert parse_flow(deep) is None


class TestBearingFloor:
    """A point under the wheels has no bearing.

    `bearing_deg` to a coincident point is atan2(0, 0) == 0.0 -- due north -- so a
    link running due west was "ahead" for any northerly heading, and with realistic
    GPS-vs-map offset the bearing from such a point is noise.
    """

    def test_a_link_running_west_is_never_ahead_whatever_the_heading(self):
        feed = HereFeed()
        # Ends exactly under the fix, which is what the earlier fixture produced.
        west = stretch(*offset(*HOME, 0.0, -800.0), east_m=800.0)
        feed.offer(status=200, body=body(west), received_t_mono=100.0)

        for heading in range(0, 360, 15):
            reading = feed.at(fix(*HOME, heading_deg=float(heading), t_mono=100.0), t_mono=100.0)
            if 200.0 <= heading <= 340.0:
                continue  # driving back down it: legitimately ahead
            assert reading.outcome != Outcome.OK, (
                f"a link entirely west was called ahead at heading {heading}"
            )


class TestStalenessIsSymmetric:
    """`PhoneGpsReader.is_stale` states the rule: the freshness predicates in this
    codebase must not disagree about a stamp from this clock's future. This is the
    third such predicate, and it was the one that disagreed."""

    def test_a_stamp_from_the_future_is_not_fresh(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=1000.0)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=500.0), t_mono=500.0)
        assert reading.outcome == Outcome.STALE
        assert reading.link is None


class TestTheRecordReportsTheLastQuery:

    def test_ok_does_not_survive_a_later_failed_query(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0).ok

        feed.offer(status=200, body=b'{"results": []}', received_t_mono=105.0)
        feed.at(fix(*HOME, heading_deg=90.0, t_mono=105.0), t_mono=105.0)

        assert feed.to_record()["last_outcome"] == Outcome.NO_LINK_MATCHED

    def test_a_recovered_from_http_error_is_not_reported_as_the_outcome(self):
        # An operator reading "the HERE calls were failing" for a drive where they
        # came good again is being told the wrong thing about the drive.
        feed = HereFeed()
        feed.offer(status=503, body=b"", received_t_mono=100.0)
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=101.0)
        feed.at(fix(*HOME, heading_deg=90.0, t_mono=101.0), t_mono=101.0)

        record = feed.to_record()
        assert record["last_outcome"] == Outcome.OK
        assert record["last_refusal"] == Outcome.HTTP_ERROR


class TestStampProvenance:

    def test_a_proxied_arrival_stamp_is_visible_in_the_reading_and_the_record(self):
        # Every response in the opening seconds of a drive is aged on a proxy,
        # before the timebase converges. A run that cannot tell those apart cannot
        # say how much of its feed was aged on a guess.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)),
                   received_t_mono=100.0, bound_s=0.004, proxy=True)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.5), t_mono=100.5)
        assert reading.ok
        assert reading.response_age_is_proxy is True
        assert reading.response_age_bound_s == 0.004
        assert feed.to_record()["proxied_stamps"] == 1


class TestOneStore:
    """A reader sees one whole response or the other, never a mix.

    Every field of a response was published as a separate attribute, and no
    ordering of independent stores is safe. Links-then-stamp over-stated age and
    returned `stale`, which is harmless. Stamp-then-links -- introduced while
    claiming to fix the first -- under-stated it and returned `ok`, handing back a
    live congestion number for a road already behind: one wrongly-fresh reading in
    396,380 queries, reporting an age of 1.0 s for links 46 s old.
    """

    def test_a_reading_never_pairs_one_response_with_another_ones_stamp(self):
        import threading

        feed = HereFeed()
        near = stretch(*HOME, east_m=800.0)
        far = stretch(*offset(*HOME, north_m=0.0, east_m=2000.0), east_m=800.0)
        stop = threading.Event()
        seen: list[tuple[str, float]] = []

        def writer():
            i = 0
            deadline = time.monotonic() + 2.0
            while not stop.is_set() and time.monotonic() < deadline:
                i += 1
                if i % 2:
                    # Under us, but stamped long ago: every correct reading is stale.
                    feed.offer(status=200, body=body(near), received_t_mono=1000.0 - 45.0)
                else:
                    # Fresh, but 2 km away: every correct reading is no_link_matched.
                    feed.offer(status=200, body=body(far), received_t_mono=1000.0)

        worker = threading.Thread(target=writer, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                r = feed.at(fix(*HOME, heading_deg=90.0, t_mono=1000.0), t_mono=1000.0)
                seen.append((r.outcome, r.response_age_s or 0.0))
        finally:
            stop.set()
            worker.join(timeout=3.0)

        wrong = [s for s in seen if s[0] == Outcome.OK]
        assert not wrong, f"{len(wrong)} of {len(seen)} readings mixed two responses: {wrong[:3]}"

    def test_provenance_belongs_to_the_response_it_came_with(self):
        # `bound_s` and `proxy` were stored after the links, so a query in that gap
        # paired a new response with the previous one's provenance -- reporting a
        # 4 ms bound for a stamp that was actually a proxy with none, which defeats
        # the proxy flag exactly where it matters, in the opening seconds.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)),
                   received_t_mono=100.0, bound_s=0.004, proxy=False)
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)),
                   received_t_mono=101.0, bound_s=None, proxy=True)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=101.0), t_mono=101.0)
        assert reading.response_age_is_proxy is True
        assert reading.response_age_bound_s is None


class TestSpaceProvenance:
    """`ok` alone cannot say whether the match is under the wheels or kilometres aside.

    The cone is a fixed half-angle, so its lateral tolerance grows with range: at
    the 3 km horizon it admits 2.6 km of offset. The radius is not the fix -- a
    downstream link is legitimately far -- so the distance travels with the reading
    and the caller weights it.
    """

    def test_a_match_far_off_the_heading_ray_reports_how_far(self):
        feed = HereFeed()
        ours = stretch(*offset(*HOME, 0.0, -825.0), east_m=800.0)   # behind, matches the radius
        # Inside the 3 km horizon and on the cone's edge: bearing ~56 deg against a
        # heading of 0, so geometrically 'ahead' and 2.4 km to the side.
        aside = stretch(*offset(*HOME, north_m=1600.0, east_m=2400.0), east_m=800.0)
        feed.offer(status=200, body=json.dumps({"results": [
            {"location": {"shape": {"links": [{"points": ours}]}},
             "currentFlow": {"jamFactor": 1.0, "speed": 20.0, "freeFlow": 25.0}},
            {"location": {"shape": {"links": [{"points": aside}]}},
             "currentFlow": {"jamFactor": 9.9, "speed": 2.0, "freeFlow": 25.0}},
        ]}).encode(), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=0.0, t_mono=100.0), t_mono=100.0)
        assert reading.ok
        # Reported, not refused -- but the caller can now see it is not our road.
        assert reading.link_cross_track_m > 1000.0
        assert reading.link_distance_m > 2000.0

    def test_a_match_under_the_wheels_reports_a_small_offset(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0)
        assert reading.ok
        assert reading.link_cross_track_m < 50.0


class TestTheReportedPairDescribesOnePlace:
    """`link_distance_m` and `link_cross_track_m` must come from the same point.

    They did not: distance was min-over-all-points and the bearing came from
    `points[0]`, so the product was a magnitude from one place and an angle from
    another and described nowhere. On a curving link it inverted the very decision
    the pair was added to enable.
    """

    def test_a_curving_link_reports_the_offset_of_the_point_it_measured(self):
        # Starts near the heading ray 2.9 km ahead and sweeps 2 km east. The
        # nearest ahead point is the far-east end; reporting `points[0]`'s offset
        # called a 2 km lateral match an 80 m one.
        curve = [
            {"lat": offset(*HOME, 2900.0, 100.0)[0], "lng": offset(*HOME, 2900.0, 100.0)[1]},
            {"lat": offset(*HOME, 2600.0, 900.0)[0], "lng": offset(*HOME, 2600.0, 900.0)[1]},
            {"lat": offset(*HOME, 2000.0, 1600.0)[0], "lng": offset(*HOME, 2000.0, 1600.0)[1]},
            {"lat": offset(*HOME, 1200.0, 2000.0)[0], "lng": offset(*HOME, 1200.0, 2000.0)[1]},
        ]
        ours = [
            {"lat": offset(*HOME, 0.0, -820.0)[0], "lng": offset(*HOME, 0.0, -820.0)[1]},
            {"lat": offset(*HOME, 0.0, -20.0)[0], "lng": offset(*HOME, 0.0, -20.0)[1]},
        ]
        feed = HereFeed()
        feed.offer(status=200, body=json.dumps({"results": [
            {"location": {"shape": {"links": [{"points": ours}]}},
             "currentFlow": {"jamFactor": 1.0, "speed": 20.0, "freeFlow": 25.0}},
            {"location": {"shape": {"links": [{"points": curve}]}},
             "currentFlow": {"jamFactor": 9.9, "speed": 2.0, "freeFlow": 25.0}},
        ]}).encode(), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=0.0, t_mono=100.0), t_mono=100.0)
        assert reading.ok
        # The pair must be consistent: a match this far off the ray cannot report a
        # lateral offset of tens of metres.
        assert reading.link_cross_track_m > 1000.0, (
            f"reported {reading.link_cross_track_m:.0f} m of offset for a match at "
            f"{reading.link_distance_m:.0f} m -- the angle came from another point"
        )

    def test_the_offset_never_exceeds_the_cone_so_sin_cannot_fold(self):
        # `angle_between` returns 0-180 and sin is symmetric about 90, so an angle
        # taken from a point BEHIND folded onto a cross-track of zero -- reading as
        # a perfect on-axis match for a point directly astern. Taking the angle
        # from the matched point bounds it by the half-angle, where sin is
        # monotonic, so the fold is unreachable rather than merely unlikely.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=2000.0)), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0)
        assert reading.ok
        implied = math.degrees(math.asin(
            min(1.0, reading.link_cross_track_m / max(reading.link_distance_m, 1e-9))
        ))
        assert implied <= 60.0 + 1e-6


class TestTheFixHasAnAgeToo:
    """The axis `at()` did not have, and every consumer of the same fix did.

    `at()` answers from geometry against the position it is handed, so an old fix
    does not degrade the answer -- it relocates it. A fresh response against a
    five-minute-old fix produced `ok`, a distance, and a live congestion number about
    road the vehicle had left, while `PhoneGpsReader.is_stale`,
    `ObservationBuilder.build` and `SensingController._usable_position` all refused
    that same fix.

    It does not self-correct either. `HerePipeline.setQuery` is
    `if (next != null) query = next`, so a command carrying no query leaves the phone
    fetching against the last position it was told about: responses stay fresh
    indefinitely while the fix does not.
    """

    def stale_and_fresh(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)),
                   received_t_mono=100.0)
        return feed

    def test_an_old_fix_does_not_get_a_live_congestion_number(self):
        feed = self.stale_and_fresh()
        reading = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0 - 300.0),
                          t_mono=100.0)
        assert reading.outcome == Outcome.STALE_FIX
        assert reading.link is None
        assert "300.0s" in (reading.detail or "")

    def test_a_fresh_fix_against_the_same_response_still_answers(self):
        # The gate must not be the whole feed switched off.
        feed = self.stale_and_fresh()
        assert feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0), t_mono=100.0).ok

    def test_a_stale_fix_is_not_reported_as_an_absent_one(self):
        # A dead GPS and a stalled one need opposite investigations. Folded under
        # `unusable_fix` the operator cannot tell which they have -- the mistake a
        # single code hiding seven causes has already cost this project once.
        feed = self.stale_and_fresh()
        absent = feed.at(fix(*HOME, valid=False, t_mono=100.0), t_mono=100.0)
        stalled = feed.at(fix(*HOME, heading_deg=90.0, t_mono=40.0), t_mono=100.0)
        assert absent.outcome == Outcome.UNUSABLE_FIX
        assert stalled.outcome == Outcome.STALE_FIX

    def test_a_fix_from_this_clocks_future_is_not_fresh_either(self):
        # The rule `PhoneGpsReader.is_stale` states, and the side the timebase
        # actually biases: `OneWayEstimator` makes every converted stamp look newer
        # than it is.
        feed = self.stale_and_fresh()
        ahead = feed.at(fix(*HOME, heading_deg=90.0, t_mono=100.0 + 60.0), t_mono=100.0)
        assert ahead.outcome == Outcome.STALE_FIX

    def test_the_bound_is_charged_the_way_the_observation_builder_charges_it(self):
        # One convention on this axis, not a fourth. A fix inside the window on its
        # own, whose arrival time is known only to within a second, is outside it
        # once the bound is charged.
        from sensors.phone_source import TimebaseStamp

        feed = self.stale_and_fresh()
        borderline = fix(*HOME, heading_deg=90.0, t_mono=100.0 - 1.5)
        assert feed.at(borderline, t_mono=100.0).ok

        with_bound = replace(borderline, timebase=TimebaseStamp(
            t_capture_mono=borderline.t_mono, t_arrival_mono=100.0,
            bound_s=1.0, proxy=False, estimate_id=1))
        assert feed.at(with_bound, t_mono=100.0).outcome == Outcome.STALE_FIX


def northward(lat: float, lon: float, north_m: float, points: int = 5) -> list[dict]:
    """A link running due north from a point, sampled evenly."""
    return [{"lat": offset(lat, lon, north_m * i / (points - 1), 0.0)[0], "lng": lon}
            for i in range(points)]


class TestDistanceIsToTheShapeNotItsVertices:
    """Keeping the shape was necessary and not sufficient."""

    def test_a_vehicle_on_the_road_between_two_vertices_matches_it(self):
        # `_points`' docstring says a vehicle "driving along a link it is certainly
        # on" must not come back unmatched, and keeping every point was its fix. The
        # distance function still measured to the nearest VERTEX, so on this file's
        # own fixture -- 199.8 m spacing against a 60 m association radius -- 40% of
        # the link was refused while the vehicle sat exactly on the carriageway.
        points = stretch(*HOME, east_m=800.0)
        coords = [(p["lat"], p["lng"]) for p in points]
        midway = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)

        feed = HereFeed()
        feed.offer(status=200, body=body(points), received_t_mono=100.0)
        reading = feed.at(fix(*midway, heading_deg=90.0, t_mono=100.0), t_mono=100.0)
        assert reading.outcome == Outcome.OK, (
            "the vehicle is on the road and the feed did not match it"
        )

    def test_the_distance_is_zero_on_the_segment_and_grows_off_it(self):
        from sensors.here_feed import FlowLink

        link = FlowLink(points=tuple((p["lat"], p["lng"])
                                     for p in stretch(*HOME, east_m=800.0)),
                        speed_mps=10.0, free_flow_mps=25.0, jam_factor=1.0,
                        confidence=0.9, traversability="open", length_m=800.0)
        coords = link.points
        midway = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)
        assert link.distance_m(*midway) == pytest.approx(0.0, abs=1.0)
        # 50 m north of that same spot: measured off the segment, not to a vertex.
        aside = offset(*midway, north_m=50.0, east_m=0.0)
        assert link.distance_m(*aside) == pytest.approx(50.0, abs=2.0)


class TestWhichLinkIsChosen:
    """Only one link survives to the caller, so the choice is the whole answer."""

    def test_a_jammed_road_alongside_does_not_evict_our_own(self):
        # The cone is a fixed half-angle and at any real range admits roads that are
        # not ours. Choosing the nearest ahead POINT let a parallel road 100 m aside
        # replace our own carriageway -- and `feed_fusion` only ever sees the one
        # link, so the on-corridor link it would have weighted was already gone.
        # Measured before the fix: congestion 0.92 from the side road's jam instead
        # of 0.04 from our clear one, over `disagreement`'s 0.5 threshold.
        # The side road has to be NEARER than ours or the old distance-only rule
        # picks ours anyway and the test proves nothing.
        #
        # Ours runs UNDER the vehicle and on ahead, sampled sparsely -- which is the
        # ordinary case, and why `distance_m` measures to the shape: the association
        # gate sees 0 m, while the nearest point AHEAD is the next vertex at 400 m.
        # The side road starts 150 m ahead and 100 m east, so it is ~180 m away:
        # nearer on distance, 100 m off the corridor.
        ours = northward(*offset(*HOME, north_m=-400.0, east_m=0.0),
                         north_m=1200.0, points=4)
        aside = northward(*offset(*HOME, north_m=150.0, east_m=100.0), north_m=600.0)

        both = json.loads(body(ours, speed=24.0, freeFlow=25.0).decode("utf-8"))
        both["results"].extend(
            json.loads(body(aside, speed=2.0, freeFlow=25.0).decode("utf-8"))["results"])

        feed = HereFeed()
        feed.offer(status=200, body=json.dumps(both).encode("utf-8"),
                   received_t_mono=100.0)
        reading = feed.at(fix(*HOME, heading_deg=0.0, t_mono=100.0), t_mono=100.0)

        assert reading.ok
        assert reading.link.speed_mps == pytest.approx(24.0), (
            f"chose the road aside at {reading.link.speed_mps} m/s over ours at 24.0"
        )
        assert reading.link_cross_track_m == pytest.approx(0.0, abs=5.0)

    def test_the_nearer_stretch_of_our_own_road_wins_the_tie(self):
        # Cross-track first, distance second. With two stretches of our own road
        # ahead, both on the heading, the nearer one is the answer.
        start = offset(*HOME, north_m=50.0, east_m=0.0)
        near = northward(*start, north_m=200.0)
        far = northward(*offset(*start, north_m=1000.0, east_m=0.0), north_m=200.0)

        both = json.loads(body(near, speed=20.0, freeFlow=25.0).decode("utf-8"))
        both["results"].extend(
            json.loads(body(far, speed=5.0, freeFlow=25.0).decode("utf-8"))["results"])

        feed = HereFeed()
        feed.offer(status=200, body=json.dumps(both).encode("utf-8"),
                   received_t_mono=100.0)
        reading = feed.at(fix(*HOME, heading_deg=0.0, t_mono=100.0), t_mono=100.0)
        assert reading.ok
        assert reading.link.speed_mps == pytest.approx(20.0)
