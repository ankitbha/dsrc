# Task 27 — HERE response ingestion

## The short version

The Jetson has none. `deployment/jetson/` does not parse a HERE body anywhere; the
only occurrences of the word are the English one. The phone fetches and forwards —
`here` channel, `request_url` / `status` / `content_type` / `query_lat` /
`query_lon` / `query_radius_m` / `t_request_mono_ns` / `t_response_mono_ns` in the
header, the response body as payload — and nothing on the receiving side opens it.

`downstream_congestion_estimate` is the observation field this feeds. Today it comes
from V2V peers as a hardcoded `0.0`, or from `sim_contract.neutral_cooperation`
otherwise. So the field a driver's advisory partly rests on has never once been
informed by traffic data.

Task 27 is: parse what arrives, work out which links are ahead of us, keep them
between responses, and be explicit about how old and how trustworthy the answer is.

## Scope boundary

In: ingestion. Parse, associate, cache, age, and expose a queryable view with named
failures.

Out: deciding who wins. Task 28 owns per-field source ownership between the
wide-lagging feed and the narrow-current camera, and **task 27 does not touch
`ObservationBuilder`**. It publishes something 28 can consult, and nothing more.
Also out: asking for a query — `rate_cmd` carries the `here` object and the phone
performs the fetch; the Jetson chooses *what* to ask for in task 29.

## The constraint that shapes everything

**No HERE call, ever, from anywhere in this repo.** The key is shared with Nash
production, is absent from the build, and `HttpHereClient` refuses to construct
without it. So every test here runs on recorded or synthetic bodies, and the parse
contract is written from the v7 flow schema rather than discovered by probing.

That has a consequence worth stating rather than discovering later: **the parser
will meet its first real HERE body in production.** So it must treat every field as
possibly absent, possibly the wrong type, and possibly outside its documented range,
and must fail to a named outcome rather than to a number. A parser that assumes the
schema it was written against is the one that returns 0.0 for "I did not
understand", and 0.0 in this field means "clear road ahead".

## The honest position on staleness

Two ages, and only one of them is knowable.

- **Response age** is exact. `t_response_mono_ns` is the phone's clock, converted by
  the timebase task 26 built, so how long ago the phone received the bytes is a
  measured quantity with a bound.
- **Feed lag** — how long before that the conditions were true — is *not reported*.
  HERE v7 flow gives no per-result observation timestamp. It is minutes, it varies,
  and nothing in the response says by how much.

So the total staleness is unknown by an unknown amount. The design consequence is
that this module reports response age as a number and feed lag as **absent**, and
never sums them into one figure that would look measured. Task 28 needs to know
which part of the delay it can account for; a single blended "age" would hide that
the larger part is a guess.

## Failure semantics, named

Each of these is a distinct outcome a caller can branch on, and none of them is a
congestion value:

| outcome | when |
|---|---|
| `no_response_yet` | nothing has arrived this session |
| `http_error` | `status` outside 2xx — carries the status |
| `unparseable` | body is not JSON, or `results` is missing or not a list |
| `no_link_matched` | nearest link centre is beyond the association radius |
| `no_link_ahead` | links matched, none of them downstream of us |
| `stale` | response age past the limit |
| `unusable_fix` | GPS invalid, or heading unavailable, so "ahead" is undefined |
| `ok` | a downstream link with a usable `currentFlow` |

`0.0` is never returned for any of the first seven. That is the same failure the
task-26 experiment found in another guise: a value that reads as a measurement when
it means "I do not know" survives every test and every drive.

## Link association

Each `results[]` entry carries a `location` with a polyline and a `currentFlow` with
`speed`, `freeFlow`, `jamFactor` (0–10), `confidence` (0–1) and `traversability`.

- **Nearest link** by point-to-polyline distance from the GPS fix, using the
  existing `haversine_m` in `v2v/beacon.py` rather than a second implementation of
  the same formula.
- **Ahead** by heading: a link's representative point is downstream if its bearing
  from us is within a half-angle of `heading_deg`. `GpsFix.heading_deg` is NaN when
  the receiver has no course — stationary, or a fix without it — which is
  `unusable_fix`, not "everything is ahead".
- **Association radius** bounds a match. Beyond it the nearest link is not our road,
  and reporting its jam factor would describe a different street.

Recommended: nearest-link association on the polyline, a heading half-angle, and a
downstream horizon, all named constants with their reasons, because every one of
them is a threshold whose wrong value produces a confident wrong answer rather than
an error.

## Caching

Responses arrive around 0.2 Hz and the vehicle moves the whole time. The cache holds
the parsed links from the most recent usable response, so a query between responses
is answered from geometry against a fresh position rather than by repeating the last
answer. That distinction matters: re-serving the last *answer* would keep reporting
congestion for a link we have already passed.

Superseding is by arrival, not by content. A newer response replaces an older one
even if it matched no link, because the older one describes a place we have left.

## The work

1. A `HereFlow` parse of a v7 body into links, tolerant of every absent or malformed
   field, returning named failures.
2. Association: nearest link, downstream filter, both on `haversine_m`.
3. A `HereFeed` holding the cache, taking `(body, header)` from the `here` channel
   and answering `at(gps, t_mono)` with an outcome plus, when `ok`, the flow and the
   response age.
4. Ingestion from the transport, alongside the camera and GPS backends, so a
   phone-fed run receives them. `PhoneLink` is where the other two are wired.
5. A record: responses received, parsed, refused by reason, cache age, and the last
   outcome — so a drive that never got a usable link says so as a number rather than
   as a field that was quietly neutral.

## Tests

- A body with every field absent parses to `unparseable` rather than to zeros.
- `jamFactor` out of range, `confidence` out of range, `speed` negative or
  non-finite: each refused and counted, not clamped silently.
- A link 3 km off the road is `no_link_matched`; a link behind us is `no_link_ahead`.
- A NaN heading is `unusable_fix`, not "all links are ahead".
- A response older than the limit is `stale`, and the outcome says so rather than
  returning the last flow.
- A newer response that matched nothing supersedes an older one that matched.
- Feed lag is absent in the record, and response age is present — the two never
  collapse into one number.

## Needs sign-off

- Reporting feed lag as unknown rather than assuming a constant. The alternative is
  to pick a number for HERE's internal delay, which would make every downstream age
  computation look measured when its dominant term was invented.
- Returning named failures rather than a neutral congestion value, which means
  task 28 must handle seven outcomes rather than reading one float.
