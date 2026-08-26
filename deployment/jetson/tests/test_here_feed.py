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


def fix(lat: float, lon: float, heading_deg: float = 90.0, valid: bool = True) -> GpsFix:
    return GpsFix(valid=valid, lat=lat, lon=lon, heading_deg=heading_deg, speed_mps=20.0)


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
        assert feed.at(fix(*HOME), t_mono=100.0).outcome == Outcome.NO_RESPONSE_YET

    def test_an_http_error_is_named_and_counted_not_swallowed(self):
        feed = HereFeed()
        assert feed.offer(status=503, body=b"", received_t_mono=100.0) is False
        assert feed.at(fix(*HOME), t_mono=100.0).outcome == Outcome.HTTP_ERROR
        assert feed.refused_by_reason == {"http_error:status 503": 1}

    def test_an_unparseable_body_is_named(self):
        feed = HereFeed()
        assert feed.offer(status=200, body=b"<html>", received_t_mono=100.0) is False
        assert feed.at(fix(*HOME), t_mono=100.0).outcome == Outcome.UNPARSEABLE

    def test_a_link_ahead_is_reported_with_its_flow_and_a_measured_age(self):
        feed = HereFeed()
        road = stretch(*HOME, east_m=800.0)
        assert feed.offer(status=200, body=body(road), received_t_mono=100.0) is True

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=102.5)
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

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0)
        assert reading.outcome == Outcome.NO_LINK_AHEAD
        assert reading.link is None

    def test_a_road_the_response_does_not_cover_is_not_matched(self):
        # The association radius answers "are we on a road this body covers at
        # all". Without it the nearest link a kilometre away would be reported as
        # ours -- a different street, described confidently.
        feed = HereFeed()
        far = stretch(*offset(*HOME, north_m=1200.0, east_m=0.0), east_m=800.0)
        feed.offer(status=200, body=body(far), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=0.0), t_mono=100.0)
        assert reading.outcome == Outcome.NO_LINK_MATCHED

    def test_the_radius_and_the_horizon_are_not_the_same_threshold(self):
        # Pinned because an earlier version tested `d <= radius or d <= horizon`,
        # which is just the horizon since it is fifty times larger -- so the radius
        # decided nothing at all.
        assert ASSOCIATION_RADIUS_M < DOWNSTREAM_HORIZON_M / 10

    def test_a_fix_without_a_heading_cannot_say_what_is_ahead(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=float("nan")), t_mono=100.0)
        assert reading.outcome == Outcome.UNUSABLE_FIX
        assert reading.detail == "no heading"

    def test_an_invalid_fix_is_named_rather_than_guessed_from(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        assert feed.at(fix(*HOME, valid=False), t_mono=100.0).outcome == Outcome.UNUSABLE_FIX

    def test_a_response_past_its_age_is_stale_not_the_last_answer(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME), t_mono=100.0).ok

        reading = feed.at(fix(*HOME), t_mono=100.0 + MAX_RESPONSE_AGE_S + 1.0)
        assert reading.outcome == Outcome.STALE
        assert reading.link is None


class TestCaching:

    def test_a_query_between_responses_is_reanswered_from_the_new_position(self):
        # The cache holds links, not the last answer. Re-serving the answer would
        # keep reporting congestion for a link already behind us.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=500.0)), received_t_mono=100.0)

        assert feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0).ok
        # Driven to the end of that link. Still on it, so it still matches -- but
        # none of it is in front any more, answered from the same cached response
        # rather than by repeating the previous answer.
        end = offset(*HOME, north_m=0.0, east_m=500.0)
        assert feed.at(fix(*end, heading_deg=90.0), t_mono=101.0).outcome == Outcome.NO_LINK_AHEAD

    def test_a_newer_response_that_matched_nothing_still_supersedes(self):
        # Superseding is by arrival, not by content: the older body describes a
        # place we have left, so keeping it because it happened to match is worse
        # than having nothing.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0).ok

        feed.offer(status=200, body=b'{"results": []}', received_t_mono=105.0)
        assert feed.at(fix(*HOME, heading_deg=90.0), t_mono=105.0).outcome == Outcome.NO_LINK_MATCHED

    def test_a_refused_response_does_not_replace_a_usable_one(self):
        # An HTTP error says nothing about the road, so it must not throw away
        # links that still describe it.
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        feed.offer(status=500, body=b"", received_t_mono=101.0)

        assert feed.at(fix(*HOME, heading_deg=90.0), t_mono=101.0).ok


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
            reading = feed.at(fix(*HOME, heading_deg=float(heading)), t_mono=100.0)
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

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=500.0)
        assert reading.outcome == Outcome.STALE
        assert reading.link is None


class TestTheRecordReportsTheLastQuery:

    def test_ok_does_not_survive_a_later_failed_query(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)
        assert feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0).ok

        feed.offer(status=200, body=b'{"results": []}', received_t_mono=105.0)
        feed.at(fix(*HOME, heading_deg=90.0), t_mono=105.0)

        assert feed.to_record()["last_outcome"] == Outcome.NO_LINK_MATCHED

    def test_a_recovered_from_http_error_is_not_reported_as_the_outcome(self):
        # An operator reading "the HERE calls were failing" for a drive where they
        # came good again is being told the wrong thing about the drive.
        feed = HereFeed()
        feed.offer(status=503, body=b"", received_t_mono=100.0)
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=101.0)
        feed.at(fix(*HOME, heading_deg=90.0), t_mono=101.0)

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

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.5)
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
                r = feed.at(fix(*HOME, heading_deg=90.0), t_mono=1000.0)
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

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=101.0)
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

        reading = feed.at(fix(*HOME, heading_deg=0.0), t_mono=100.0)
        assert reading.ok
        # Reported, not refused -- but the caller can now see it is not our road.
        assert reading.link_cross_track_m > 1000.0
        assert reading.link_distance_m > 2000.0

    def test_a_match_under_the_wheels_reports_a_small_offset(self):
        feed = HereFeed()
        feed.offer(status=200, body=body(stretch(*HOME, east_m=800.0)), received_t_mono=100.0)

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0)
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

        reading = feed.at(fix(*HOME, heading_deg=0.0), t_mono=100.0)
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

        reading = feed.at(fix(*HOME, heading_deg=90.0), t_mono=100.0)
        assert reading.ok
        implied = math.degrees(math.asin(
            min(1.0, reading.link_cross_track_m / max(reading.link_distance_m, 1e-9))
        ))
        assert implied <= 60.0 + 1e-6
