"""What the traffic feed says about the road ahead, and how much of that to believe.

The phone fetches HERE and forwards the bytes; nothing on this side has ever opened
them. `downstream_congestion_estimate` -- the observation field this informs -- comes
from V2V peers as a hardcoded 0.0, or from `sim_contract.neutral_cooperation`
otherwise, so a field the advisory partly rests on has never been informed by traffic
data.

**Nothing here returns a congestion number for a question it could not answer.**
Every failure is a named outcome. Zero in this field does not read as "unknown", it
reads as "clear road ahead", and a caller that cannot tell those apart will act on
the second when it has the first. That is the same defect the task-26 experiment
found in another guise -- a value that looks measured and is not survives every test
and every drive.

**This parser meets its first real HERE body in production.** The key is shared with
Nash production and absent from every build here, so nothing in this repo may call
the API and the schema below is written from the v7 flow documentation rather than
discovered by probing. Everything is therefore treated as possibly absent, possibly
the wrong type, and possibly outside its documented range.

**Two ages, one knowable.** The response age -- how long since the phone received the
bytes -- is measured, on the timebase the phone-fed backends already convert through.
The feed's own lag, how long before that the conditions held, is not reported
anywhere in a v7 flow response: it is minutes, it varies, and nothing says by how
much. So this reports the first as a number and the second as absent, and never sums
them into one figure that would look measured when its dominant term was a guess.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from sensors.gps_reader import GpsFix
from geo import haversine_m

#: How far off the road a link's nearest point may be and still be our road. A
#: motorway and the service road beside it are tens of metres apart, and reporting
#: the wrong one's jam factor describes a different street confidently.
ASSOCIATION_RADIUS_M = 60.0

#: Half-angle around the heading within which a link counts as ahead. Wide enough to
#: hold a curve, narrow enough that the carriageway behind us is not "downstream".
DOWNSTREAM_HALF_ANGLE_DEG = 60.0

#: How far ahead is still about where we are going.
DOWNSTREAM_HORIZON_M = 3_000.0

#: Nearer than this, a shape point has no usable bearing from the vehicle: the
#: fix's own error is metres, so the direction to a point a metre away is noise.
#: Sized above typical GPS-vs-map lateral offset.
BEARING_FLOOR_M = 15.0

#: Past this, the feed describes a road we may no longer be on. Chosen against the
#: phone's own `here_hz` default of 0.2 Hz -- five seconds between fetches -- so this
#: tolerates several missed responses without tolerating a stale minute.
MAX_RESPONSE_AGE_S = 30.0


class Outcome:
    """Why there is, or is not, an answer. A caller branches on these."""

    OK = "ok"
    NO_RESPONSE_YET = "no_response_yet"
    HTTP_ERROR = "http_error"
    UNPARSEABLE = "unparseable"
    NO_LINK_MATCHED = "no_link_matched"
    NO_LINK_AHEAD = "no_link_ahead"
    STALE = "stale"
    UNUSABLE_FIX = "unusable_fix"


@dataclass(frozen=True)
class FlowLink:
    """One stretch of road and what the feed says is happening on it."""

    #: The link's shape, in order. Distance to a link is distance to this, not to
    #: any single representative point.
    points: tuple[tuple[float, float], ...]
    #: Metres per second. `None` when the field was absent or unusable, which is
    #: different from zero -- zero is a jam.
    speed_mps: float | None
    free_flow_mps: float | None
    #: HERE's 0-10 scale. 10 is stopped.
    jam_factor: float | None
    confidence: float | None
    traversability: str | None
    length_m: float | None

    def distance_m(self, lat: float, lon: float) -> float:
        """How far this link's shape passes from a point."""
        return min(haversine_m(lat, lon, plat, plon) for plat, plon in self.points)

    def ahead_point(self, gps_lat: float, gps_lon: float, heading_deg: float,
                    half_angle_deg: float, horizon_m: float) -> tuple[float, float, float] | None:
        """The nearest point of this link that lies ahead: (distance, offset, point index).

        Returns the matched point rather than a bool, so distance and lateral
        offset are computed from ONE place. They were not: distance was
        min-over-all-points and the bearing came from `points[0]`, so a curving
        link whose polyline starts near the heading ray and sweeps away reported
        2329 m of range with 80 m of lateral offset while the point that range
        described was 1997 m aside -- the field inverted in exactly the case it
        exists to catch.

        Any part of the link, not the nearest overall: the link we are driving ON
        has its nearest point beside us, where the bearing is arbitrary and often
        backwards, while the stretch that matters runs out in front.
        """
        best: tuple[float, float, int] | None = None
        for index, (plat, plon) in enumerate(self.points):
            distance = haversine_m(gps_lat, gps_lon, plat, plon)
            if distance > horizon_m:
                continue
            if distance < BEARING_FLOOR_M:
                # A point essentially under the vehicle has no meaningful bearing:
                # `bearing_deg` to a coincident point is atan2(0, 0) == 0.0, due
                # north, so a link running due WEST was called "ahead" for any
                # northerly heading. With realistic GPS-vs-map offset the bearing
                # from such a point is decided by noise, and a link passing within
                # that noise was admitted for about a third of the compass.
                continue
            offset = angle_between(bearing_deg(gps_lat, gps_lon, plat, plon), heading_deg)
            if offset > half_angle_deg:
                continue
            if best is None or distance < best[0]:
                best = (distance, offset, index)
        return best

    @property
    def usable(self) -> bool:
        """Whether this link says anything about congestion at all.

        A link with no jam factor and no speed pair is a shape and nothing more.
        """
        return self.jam_factor is not None or (
            self.speed_mps is not None and self.free_flow_mps is not None
        )


@dataclass(frozen=True)
class _Snapshot:
    """One response, published in a single store.

    Every field here was a separate attribute, and no ordering of independent
    stores is safe. Links-then-stamp let a concurrent reader see new links against
    the old arrival time: age over-stated, outcome `stale`, harmless.
    Stamp-then-links -- which an earlier comment introduced while claiming to fix
    the first -- let it see OLD links against the NEW time: age under-stated,
    outcome `ok`, and a live congestion number handed back for a road already
    behind. Measured at one wrongly-fresh reading in 396,380, reporting
    `response_age_s: 1.0` for links 46 s old.

    Rebinding one frozen object is a single attribute store, so a reader sees
    either the whole previous response or the whole new one.
    """

    links: tuple[FlowLink, ...]
    received_t_mono: float
    bound_s: float | None
    proxy: bool


@dataclass(frozen=True)
class FlowReading:
    """The answer, or the reason there is not one."""

    outcome: str
    link: FlowLink | None = None
    #: Seconds since the phone received the bytes. Measured.
    response_age_s: float | None = None
    #: What the cross-device conversion of that stamp is worth, and whether it was
    #: converted at all. A caller charging age against a threshold needs both, the
    #: way the observation builder charges `uncertainty_s` against staleness.
    response_age_bound_s: float | None = None
    response_age_is_proxy: bool = False
    #: How far the reported link is, and how far it sits off the heading ray. The
    #: cone is a fixed half-angle, so its lateral tolerance grows with range --
    #: 2.6 km at the 3 km horizon -- and `ok` alone cannot distinguish the tarmac
    #: under the wheels from a motorway that far to the side. After the age bound
    #: the reading carried provenance for the time axis and none for the space
    #: axis; a caller weighting a match needs both.
    link_distance_m: float | None = None
    link_cross_track_m: float | None = None
    #: Always None, and deliberately so -- see the module docstring. Present as a
    #: field so a consumer reads absence rather than never thinking to ask.
    feed_lag_s: float | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == Outcome.OK


def _number(value: Any, low: float | None = None, high: float | None = None) -> float | None:
    """A finite number in range, or None.

    Out-of-range is None rather than clamped. A `jamFactor` of 40 is not a jam
    factor of 10, it is a response this parser does not understand, and clamping
    would turn a schema surprise into a confident maximum.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        # A JSON integer too large for a double is a number Python will hold and
        # float() will not take. Uncaught it left parse_flow, killed the reader
        # thread for the rest of the drive, and counted nothing -- one malformed
        # body and the feed goes quiet while still serving its last links until
        # they age out.
        return None
    if not math.isfinite(number):
        return None
    if low is not None and number < low:
        return None
    if high is not None and number > high:
        return None
    return number


def _points(location: Any) -> tuple[tuple[float, float], ...]:
    """Every point of a link's shape, in order.

    v7 nests `shape.links[].points[]`. An earlier version kept only the first,
    arguing the association radius was far larger than a link's own sampling. It
    is the other way round: links run for hundreds of metres and the radius is
    tens, so a vehicle driving along a link it is certainly on sat 200 m from that
    link's first point and the road came back unmatched. Distance to a link means
    distance to its shape, so the shape has to be kept.
    """
    if not isinstance(location, dict):
        return ()
    shape = location.get("shape")
    if not isinstance(shape, dict):
        return ()
    links = shape.get("links")
    if not isinstance(links, list):
        return ()
    found: list[tuple[float, float]] = []
    for entry in links:
        if not isinstance(entry, dict):
            continue
        points = entry.get("points")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            lat = _number(point.get("lat"), -90.0, 90.0)
            lon = _number(point.get("lng"), -180.0, 180.0)
            if lat is not None and lon is not None:
                found.append((lat, lon))
    return tuple(found)


def parse_flow(body: bytes) -> list[FlowLink] | None:
    """Links from a v7 flow body, or None if the body is not one.

    None and an empty list are different answers: the first means the response was
    not understood, the second that it was understood and named no links. A caller
    that folded them together would report an empty road for a broken parser.
    """
    try:
        document = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        # RecursionError included: a deeply nested body raises it out of
        # json.loads, and it is not a ValueError. Everything a remote body can do
        # to this parser has to end as "not understood", never as an exception --
        # the one failure that is not a named outcome is a hard kill.
        return None
    if not isinstance(document, dict):
        return None
    results = document.get("results")
    if not isinstance(results, list):
        return None

    links: list[FlowLink] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        points = _points(result.get("location"))
        if not points:
            continue
        flow = result.get("currentFlow")
        flow = flow if isinstance(flow, dict) else {}
        location = result.get("location")
        links.append(
            FlowLink(
                points=points,
                speed_mps=_number(flow.get("speed"), 0.0, 120.0),
                free_flow_mps=_number(flow.get("freeFlow"), 0.0, 120.0),
                jam_factor=_number(flow.get("jamFactor"), 0.0, 10.0),
                confidence=_number(flow.get("confidence"), 0.0, 1.0),
                traversability=(
                    flow.get("traversability")
                    if isinstance(flow.get("traversability"), str) else None
                ),
                length_m=_number(
                    location.get("length") if isinstance(location, dict) else None, 0.0
                ),
            )
        )
    return links


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from one point to another, in degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def angle_between(a_deg: float, b_deg: float) -> float:
    """Smallest angle between two headings, 0-180."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


class HereFeed:
    """The most recent usable response, and what it says about where we are now.

    The cache holds parsed *links*, not the last answer. Between responses the
    vehicle keeps moving, so a query is re-answered by geometry against a fresh
    position; re-serving the previous answer would go on reporting congestion for a
    link already behind us.

    Superseding is by arrival, not by content: a newer response replaces an older
    one even when it matched nothing, because the older one describes a place we
    have left.
    """

    def __init__(
        self,
        *,
        association_radius_m: float = ASSOCIATION_RADIUS_M,
        downstream_half_angle_deg: float = DOWNSTREAM_HALF_ANGLE_DEG,
        downstream_horizon_m: float = DOWNSTREAM_HORIZON_M,
        max_response_age_s: float = MAX_RESPONSE_AGE_S,
    ) -> None:
        self._radius_m = association_radius_m
        self._half_angle_deg = downstream_half_angle_deg
        self._horizon_m = downstream_horizon_m
        self._max_age_s = max_response_age_s
        self._current: _Snapshot | None = None
        #: Why nothing usable has arrived, when nothing has. Distinct from the
        #: outcome of a query: an HTTP error says nothing about the road and must
        #: not overwrite what the last actual question was answered with.
        self._last_refusal: str = Outcome.NO_RESPONSE_YET
        #: The outcome of the most recent `at()`. This is what the record reports.
        #: Writing only `ok` and refusals into one field made the run artifact
        #: claim `ok` for a drive whose every later query failed, and report a
        #: recovered-from `http_error` for one where the calls came good again.
        self._last_query: str = Outcome.NO_RESPONSE_YET
        self.responses_received = 0
        self.responses_parsed = 0
        self.proxied_stamps = 0
        self.refused_by_reason: dict[str, int] = {}

    def offer(self, status: int, body: bytes, received_t_mono: float,
              *, bound_s: float | None = None, proxy: bool = False) -> bool:
        """Take one `here` response. False when it was refused, and why is counted.

        `bound_s` and `proxy` describe the arrival stamp, not the traffic. A stamp
        that took the proxy path -- normal in the opening seconds of a drive, before
        the timebase converges -- ages the response against arrival rather than
        against the phone's own clock, and a run artifact that cannot tell those
        apart cannot say how much of its feed was aged on a guess.
        """
        self.responses_received += 1
        if proxy:
            self.proxied_stamps += 1
        if not 200 <= status < 300:
            self._refuse(Outcome.HTTP_ERROR, f"status {status}")
            return False
        links = parse_flow(body)
        if links is None:
            self._refuse(Outcome.UNPARSEABLE)
            return False
        # Accepted even when empty: the response was understood, and it supersedes
        # an older one that describes a road we have since left.
        # One store. See `_Snapshot`.
        self._current = _Snapshot(
            links=tuple(links), received_t_mono=received_t_mono,
            bound_s=bound_s, proxy=proxy,
        )
        self.responses_parsed += 1
        return True

    def _refuse(self, reason: str, detail: str | None = None) -> None:
        self._last_refusal = reason
        key = reason if detail is None else f"{reason}:{detail}"
        self.refused_by_reason[key] = self.refused_by_reason.get(key, 0) + 1

    def at(self, gps: GpsFix, t_mono: float) -> FlowReading:
        """What the feed says about the road ahead of this fix, or why it cannot."""
        # Read once. Everything below sees one response or the other, never a
        # stamp from one and links from another.
        snapshot = self._current
        reading = self._at(gps, t_mono, snapshot)
        self._last_query = reading.outcome
        return reading

    def _at(self, gps: GpsFix, t_mono: float, snapshot: "_Snapshot | None") -> FlowReading:
        if snapshot is None:
            return FlowReading(outcome=self._last_refusal, detail="nothing usable has arrived")

        age_s = t_mono - snapshot.received_t_mono
        provenance = {
            "response_age_s": age_s,
            "response_age_bound_s": snapshot.bound_s,
            "response_age_is_proxy": snapshot.proxy,
        }
        # Symmetric, because `PhoneGpsReader.is_stale` says the freshness
        # predicates in this codebase must not disagree about it and this is the
        # third one. A one-sided `>` called a stamp from this clock's future fresh
        # and handed back a live congestion number with `response_age_s: -500.0` --
        # and the future side is the biased one here, since `OneWayEstimator`
        # documents that its conversion makes every stamp look newer than it is.
        if abs(age_s) > self._max_age_s:
            return FlowReading(outcome=Outcome.STALE, **provenance,
                               detail=f"{age_s:.1f}s since the response")

        if not gps.valid or not math.isfinite(gps.lat) or not math.isfinite(gps.lon):
            return FlowReading(outcome=Outcome.UNUSABLE_FIX, **provenance,
                               detail="no position")
        if not math.isfinite(gps.heading_deg):
            # Without a course, "ahead" is undefined. Treating every link as ahead
            # would let the queue we just cleared describe the road in front.
            return FlowReading(outcome=Outcome.UNUSABLE_FIX, **provenance,
                               detail="no heading")

        near = [
            (link.distance_m(gps.lat, gps.lon), link)
            for link in snapshot.links
            if link.usable
        ]
        if not near:
            return FlowReading(outcome=Outcome.NO_LINK_MATCHED, **provenance,
                               detail="no usable link in the response")

        # The radius answers "are we on a road this response covers at all". If the
        # NEAREST link of any kind is further than that, we are off the covered
        # network and every link in the body describes somewhere else. It does not
        # bound the ANSWER: a downstream link is legitimately kilometres away, and
        # bounding the answer by a lateral tolerance would be a different check
        # from this one. An earlier version tested `d <= radius or d <= horizon`,
        # which is just the horizon since it is fifty times larger, so the radius
        # decided nothing at all.
        if min(d for d, _ in near) > self._radius_m:
            return FlowReading(outcome=Outcome.NO_LINK_MATCHED, **provenance,
                               detail="nearest link beyond the association radius")

        ahead = []
        for _, link in near:
            matched = link.ahead_point(gps.lat, gps.lon, gps.heading_deg,
                                       self._half_angle_deg, self._horizon_m)
            if matched is not None:
                ahead.append((matched[0], matched[1], link))
        if not ahead:
            return FlowReading(outcome=Outcome.NO_LINK_AHEAD, **provenance)

        distance, offset_deg, nearest = min(ahead, key=lambda triple: triple[0])
        # Both from the one matched point, so the pair describes a single place.
        # `link_distance_m` therefore means "to the nearest point of this link that
        # is ahead", not "to the link" -- the honest reading when the two differ.
        #
        # Cross-track says how far that place sits off the heading ray. The cone is
        # a fixed half-angle, so at the 3 km horizon it admits 2.6 km of offset and
        # `ok` alone cannot tell a road under the wheels from one that far aside.
        # Reported rather than refused, because a downstream link IS legitimately
        # far -- the caller is the one placed to weight it. The offset is at most
        # the half-angle by construction, so `sin` is monotonic over its range and
        # cannot fold a point behind us onto a cross-track of zero.
        return FlowReading(outcome=Outcome.OK, link=nearest, **provenance,
                           link_distance_m=distance,
                           link_cross_track_m=abs(distance * math.sin(math.radians(offset_deg))))

    def to_record(self) -> dict[str, Any]:
        """What the feed did this run.

        `feed_lag_s` is absent on purpose and says so, rather than being omitted --
        an absent key reads as an oversight, a null with a name reads as the
        measurement nobody can make.
        """
        return {
            "responses_received": self.responses_received,
            "responses_parsed": self.responses_parsed,
            "refused_by_reason": dict(self.refused_by_reason),
            "last_outcome": self._last_query,
            "last_refusal": self._last_refusal,
            "links_cached": 0 if self._current is None else len(self._current.links),
            "proxied_stamps": self.proxied_stamps,
            "feed_lag_s": None,
            "feed_lag_note": "not reported by HERE v7 flow; minutes, and unmeasurable here",
        }
