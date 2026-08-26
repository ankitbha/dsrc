"""Which observation fields the traffic feed may own, and on what terms.

The camera sees a few hundred metres, at 30 Hz, now. The feed sees kilometres, at
0.2 Hz, some unknown number of minutes ago. They are not two measurements of one
quantity, so this is **ownership, not averaging** -- averaging them would produce a
number neither source ever measured. Each field here has one owner and a declared
fallback, and `field_sources` says which fired.

**The mapping is the hard part, and it is a units problem.** The simulator's
`downstream_congestion_estimate` is `jam_fraction`: the proportion of a segment that
is jammed (`src/sensing/local.py:455`). HERE's `jamFactor` is a 0-10 severity scale.
Dividing by ten makes the ranges agree while the quantities stay different, and a
jamFactor of 5 means "notably slow", not "half the segment is stopped". The speed
ratio below is used instead -- the fraction of free-flow speed lost, which is a
proportion of the same kind, computed from two fields the response already carries.

It is still an approximation of something the live system cannot observe: a phone has
no segment-wide view. So it carries its own provenance value rather than `measured`,
and **task 47's parity check for this field has to compare behaviour, not equality.**

Two fields were considered and refused, both for the same reason:

- `distance_to_downstream_bottleneck` looks like exactly what `link_distance_m` is.
  It is not: the simulator sets it to `0.0` when the ego is IN a bottleneck segment
  and `inf` otherwise -- a flag wearing a distance's units, and the policy has only
  ever seen `{0, inf}` there. Putting 1800.0 into it is the identical error to
  `jamFactor / 10`, one field along.
- `merge_pressure` is local geometry the feed does not observe at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sensors.here_feed import FlowReading, Outcome

#: How far off the heading ray a match may sit and still be treated as our road.
#: Task 27's cone is a fixed half-angle, so it admits 2.6 km of lateral offset at
#: the 3 km horizon; a link that far aside is a different road and its jam is not
#: ours. This is what `link_cross_track_m` was added to make checkable.
MAX_CROSS_TRACK_M = 150.0

#: The oldest feed reading that may own a field, bound included. Wider than the
#: GPS window because the quantity itself changes slowly -- a queue two kilometres
#: ahead does not clear in a second -- and narrower than the feed's own cache limit,
#: so the fusion stops trusting a reading before the feed stops serving it.
MAX_FEED_AGE_S = 20.0

#: The smallest free-flow speed that makes a ratio meaningful. Below this the
#: denominator is noise and `1 - speed/freeFlow` swings wildly on nothing.
MIN_FREE_FLOW_MPS = 2.0

#: `field_sources` value for a field the feed owns. Deliberately not `measured`:
#: it is derived from a service's estimate of a quantity this vehicle cannot see,
#: and a reader must be able to tell that from a camera measurement.
SOURCE_FEED = "feed_derived"


class Decline:
    """Why the feed does not own a field on this tick."""

    NO_READING = "no_reading"
    NOT_OK = "feed_outcome"
    STALE = "feed_stale"
    OFF_CORRIDOR = "off_corridor"
    NO_SPEEDS = "no_speed_pair"
    NO_FREE_FLOW = "free_flow_too_low"


@dataclass(frozen=True)
class FeedOwnership:
    """What the feed can contribute this tick, and why not when it cannot."""

    #: `1 - speed/freeFlow`, clamped to [0, 1]. None when the feed does not own it.
    downstream_congestion: float | None = None
    #: The link's own free-flow speed, m/s. Same quantity as the simulator's
    #: `ego.free_flow_speed_mps`, so this one is a true substitution.
    free_flow_mps: float | None = None
    #: Named reason, when nothing is owned.
    declined: str | None = None
    #: Age of the reading these came from, and whether its stamp was converted.
    age_s: float | None = None
    is_proxy: bool = False

    @property
    def owns_congestion(self) -> bool:
        return self.downstream_congestion is not None


def _fresh(reading: FlowReading, max_age_s: float) -> bool:
    """Age charged with its own bound, symmetric about now.

    Symmetric because `PhoneGpsReader.is_stale` says the freshness predicates in
    this codebase must not disagree about a stamp from this clock's future, and
    task 27 found this feed's own predicate breaking that rule. Charging the bound
    follows `ObservationBuilder`, which adds `uncertainty_s` before comparing.
    """
    if reading.response_age_s is None:
        return False
    bound = reading.response_age_bound_s or 0.0
    return abs(reading.response_age_s) + bound <= max_age_s


def own(
    reading: FlowReading | None,
    *,
    max_cross_track_m: float = MAX_CROSS_TRACK_M,
    max_age_s: float = MAX_FEED_AGE_S,
    min_free_flow_mps: float = MIN_FREE_FLOW_MPS,
) -> FeedOwnership:
    """What the feed owns given this reading, or a named reason it owns nothing.

    Declining is the ordinary case and is not a failure: no phone, no response yet,
    a road the body does not cover, a match off to the side. Every one of them hands
    the field to the next owner rather than contributing a number.
    """
    if reading is None:
        return FeedOwnership(declined=Decline.NO_READING)
    if reading.outcome != Outcome.OK or reading.link is None:
        return FeedOwnership(declined=Decline.NOT_OK, age_s=reading.response_age_s)
    if not _fresh(reading, max_age_s):
        return FeedOwnership(declined=Decline.STALE, age_s=reading.response_age_s)
    if reading.link_cross_track_m is None or reading.link_cross_track_m > max_cross_track_m:
        # Geometrically ahead is not the same as on our road. The cone widens with
        # range, so without this a motorway kilometres to the side supplies the
        # congestion the driver is told about.
        return FeedOwnership(declined=Decline.OFF_CORRIDOR, age_s=reading.response_age_s)

    link = reading.link
    if link.speed_mps is None or link.free_flow_mps is None:
        # No ratio without both. `jamFactor` alone is NOT a fallback here: it is a
        # different quantity, and reaching for it because it is present is exactly
        # the substitution this module refuses.
        return FeedOwnership(declined=Decline.NO_SPEEDS, age_s=reading.response_age_s)
    if link.free_flow_mps < min_free_flow_mps:
        return FeedOwnership(declined=Decline.NO_FREE_FLOW, age_s=reading.response_age_s)

    ratio = 1.0 - (link.speed_mps / link.free_flow_mps)
    if not math.isfinite(ratio):
        return FeedOwnership(declined=Decline.NO_SPEEDS, age_s=reading.response_age_s)
    return FeedOwnership(
        downstream_congestion=max(0.0, min(1.0, ratio)),
        free_flow_mps=link.free_flow_mps,
        age_s=reading.response_age_s,
        is_proxy=reading.response_age_is_proxy,
    )


def to_record(ownership: FeedOwnership) -> dict[str, Any]:
    """What the feed contributed, for the run artifact.

    A drive where the feed never owned a field must say so as a reason rather than
    by the congestion column being quietly neutral throughout.
    """
    return {
        "owns_congestion": ownership.owns_congestion,
        "declined": ownership.declined,
        "age_s": None if ownership.age_s is None else round(ownership.age_s, 3),
        "age_is_proxy": ownership.is_proxy,
        "downstream_congestion": ownership.downstream_congestion,
    }
