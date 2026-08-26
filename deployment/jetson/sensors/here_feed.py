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
from v2v.beacon import haversine_m

#: How far off the road a link's nearest point may be and still be our road. A
#: motorway and the service road beside it are tens of metres apart, and reporting
#: the wrong one's jam factor describes a different street confidently.
ASSOCIATION_RADIUS_M = 60.0

#: Half-angle around the heading within which a link counts as ahead. Wide enough to
#: hold a curve, narrow enough that the carriageway behind us is not "downstream".
DOWNSTREAM_HALF_ANGLE_DEG = 60.0

#: How far ahead is still about where we are going.
DOWNSTREAM_HORIZON_M = 3_000.0

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

    def extends_ahead(self, gps_lat: float, gps_lon: float, heading_deg: float,
                      half_angle_deg: float, horizon_m: float) -> bool:
        """Whether any part of this link lies ahead, within the horizon.

        Any part, not the nearest: the link we are driving ON has its nearest
        point beside us, where the bearing is arbitrary and often backwards, while
        the stretch that matters runs out in front. Testing only the nearest point
        rejected the road under the wheels.
        """
        for plat, plon in self.points:
            if haversine_m(gps_lat, gps_lon, plat, plon) > horizon_m:
                continue
            if angle_between(bearing_deg(gps_lat, gps_lon, plat, plon), heading_deg) <= half_angle_deg:
                return True
        return False

    @property
    def usable(self) -> bool:
        """Whether this link says anything about congestion at all.

        A link with no jam factor and no speed pair is a shape and nothing more.
        """
        return self.jam_factor is not None or (
            self.speed_mps is not None and self.free_flow_mps is not None
        )


@dataclass(frozen=True)
class FlowReading:
    """The answer, or the reason there is not one."""

    outcome: str
    link: FlowLink | None = None
    #: Seconds since the phone received the bytes. Measured.
    response_age_s: float | None = None
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
    number = float(value)
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
    except (ValueError, UnicodeDecodeError):
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
        self._links: list[FlowLink] = []
        self._received_t_mono: float | None = None
        self._last_outcome: str = Outcome.NO_RESPONSE_YET
        self.responses_received = 0
        self.responses_parsed = 0
        self.refused_by_reason: dict[str, int] = {}

    def offer(self, status: int, body: bytes, received_t_mono: float) -> bool:
        """Take one `here` response. False when it was refused, and why is counted."""
        self.responses_received += 1
        if not 200 <= status < 300:
            self._refuse(Outcome.HTTP_ERROR, f"status {status}")
            return False
        links = parse_flow(body)
        if links is None:
            self._refuse(Outcome.UNPARSEABLE)
            return False
        # Accepted even when empty: the response was understood, and it supersedes
        # an older one that describes a road we have since left.
        self._links = links
        self._received_t_mono = received_t_mono
        self.responses_parsed += 1
        return True

    def _refuse(self, reason: str, detail: str | None = None) -> None:
        self._last_outcome = reason
        key = reason if detail is None else f"{reason}:{detail}"
        self.refused_by_reason[key] = self.refused_by_reason.get(key, 0) + 1

    def at(self, gps: GpsFix, t_mono: float) -> FlowReading:
        """What the feed says about the road ahead of this fix, or why it cannot."""
        if self._received_t_mono is None:
            return FlowReading(outcome=self._last_outcome
                               if self._last_outcome != Outcome.OK else Outcome.NO_RESPONSE_YET)

        age_s = t_mono - self._received_t_mono
        if age_s > self._max_age_s:
            return FlowReading(outcome=Outcome.STALE, response_age_s=age_s,
                               detail=f"{age_s:.1f}s since the response")

        if not gps.valid or not math.isfinite(gps.lat) or not math.isfinite(gps.lon):
            return FlowReading(outcome=Outcome.UNUSABLE_FIX, response_age_s=age_s,
                               detail="no position")
        if not math.isfinite(gps.heading_deg):
            # Without a course, "ahead" is undefined. Treating every link as ahead
            # would let the queue we just cleared describe the road in front.
            return FlowReading(outcome=Outcome.UNUSABLE_FIX, response_age_s=age_s,
                               detail="no heading")

        near = [
            (link.distance_m(gps.lat, gps.lon), link)
            for link in self._links
            if link.usable
        ]
        if not near:
            return FlowReading(outcome=Outcome.NO_LINK_MATCHED, response_age_s=age_s,
                               detail="no usable link in the response")

        # The two thresholds do different jobs and are not interchangeable. The
        # radius answers "are we on a road this response covers at all" -- if the
        # NEAREST link of any kind is further than that, we are off the covered
        # network and every link in the body describes somewhere else. The horizon
        # answers "is this link near enough ahead to be where we are going". An
        # earlier version tested `d <= radius or d <= horizon`, which is just the
        # horizon, because it is fifty times larger -- so the radius decided
        # nothing and a run on an uncovered road reported the motorway a kilometre
        # away as its own.
        if min(d for d, _ in near) > self._radius_m:
            return FlowReading(outcome=Outcome.NO_LINK_MATCHED, response_age_s=age_s,
                               detail="nearest link beyond the association radius")

        ahead = [
            (d, link) for d, link in near
            if link.extends_ahead(gps.lat, gps.lon, gps.heading_deg,
                                  self._half_angle_deg, self._horizon_m)
        ]
        if not ahead:
            return FlowReading(outcome=Outcome.NO_LINK_AHEAD, response_age_s=age_s)

        _, nearest = min(ahead, key=lambda pair: pair[0])
        self._last_outcome = Outcome.OK
        return FlowReading(outcome=Outcome.OK, link=nearest, response_age_s=age_s)

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
            "last_outcome": self._last_outcome,
            "links_cached": len(self._links),
            "feed_lag_s": None,
            "feed_lag_note": "not reported by HERE v7 flow; minutes, and unmeasurable here",
        }
