# Task 28 — Fusion: per-field source ownership

## The short version

Two sources now reach the Jetson and they see different things. The camera sees a
few hundred metres, at 30 Hz, right now. The feed sees kilometres, at 0.2 Hz, some
unknown number of minutes ago. **They are not two measurements of one quantity, so
fusion here is ownership, not averaging.** Each observation field gets exactly one
owner, a declared fallback, and provenance saying which fired.

The task list's own wording is the design: *"The sources observe different parts of
the state and are not substitutable."* Averaging them would produce a number neither
source ever measured.

**The blocking problem is not plumbing, it is units.** The simulator's
`downstream_congestion_estimate` is `jam_fraction` — the *proportion of a segment
that is jammed* (`src/sensing/local.py:455-456`, clamped to [0,1]). HERE's
`jamFactor` is a **0–10 severity scale**. Dividing by ten makes the ranges agree
while the quantities stay different, and a silent unit substitution into a field the
policy consumes is the failure this project keeps finding in other clothes.

## Scope boundary

In: deciding which source owns which field, mapping a `FlowReading` into the
observation, the staleness aging term, and provenance for every fused field.

Out: choosing rates or asking for a query (task 29); shadow/live gating (30); the
tick-loop integration and advisory return (31). Also out: changing the observation's
*shape* — task 47 requires field-for-field parity with the simulator, so no field is
added, removed or renamed here.

## The units problem, and what I recommend

`jam_fraction` is a spatial proportion the live system **cannot observe**: it has no
segment-wide view of how much of a stretch is stopped. Whatever goes in that field
from a phone-fed drive is an approximation, and the only question is which one and
whether it is declared.

Candidates:

1. **`jamFactor / 10`.** Ranges match, meanings do not. A jamFactor of 5 means
   "notably slow", not "half the segment is stopped". Rejected as the primary: it is
   the substitution that looks right in a diff and is wrong in the field.
2. **Speed ratio, `1 - speed/freeFlow`, clamped.** *Recommended.* It is the fraction
   of free-flow speed lost on the link, which is a proportion of the same kind as
   `jam_fraction` even though it is not the same measurement, and it is computable
   from two fields the response already carries. When either speed is absent the
   field is not owned by the feed at all.
3. Something learned from drives. Real, and not available before there are drives.

**Recommended: (2), declared in one place, with the mapping named in the record and
in `field_sources`.** Not `measured` — a new provenance value, so nobody downstream
reads a derived proportion as a segment metric. This is the decision I would most
want overturned if it is wrong, and it is why it leads the plan rather than sitting
in an appendix.

**Consequence for task 47 that must not be discovered later:** the sim-parity check
cannot compare this field for identity. The live system does not observe what the
simulator observes, so parity for `downstream_congestion_estimate` has to be a
statement about range and behaviour, not equality. Better said now than found by a
failing parity test with a drive's data already collected.

## Ownership

| field | owner | fallback | why |
|---|---|---|---|
| `leader_gap`, `leader_relative_speed`, lane gaps, `local_density_bin`, `local_mean_speed_bin`, `local_queue_estimate` | camera | neutral | the feed cannot see the car in front |
| `ego_speed`, `ego_acceleration` | GPS | neutral | neither camera nor feed measures it |
| `downstream_congestion_estimate` | **feed** | V2V peers, then neutral | the camera cannot see 2 km ahead |
| `segment_target_speed` | V2V peers | feed's `freeFlow`, then config | peers measure it; the feed reports the road's own free-flow |
| `merge_pressure` | camera/V2V | neutral | local geometry, not a feed quantity |
| `distance_to_downstream_bottleneck` | ~~feed~~ **nobody** | stays `sim_parity` | see below |

Nothing appears in two rows. Where a field has a chain, the chain is ordered and the
provenance names the link that fired.

**Corrected during implementation, and worth recording because it is the same
mistake twice.** The first draft of that table gave `distance_to_downstream_bottleneck`
to the feed, on the grounds that `link_distance_m` is exactly a distance to a jam.
It is not the same quantity: the simulator sets that field to `0.0` when the ego is
*in* a bottleneck segment and `inf` otherwise (`src/sensing/local.py:242`) — a flag
wearing a distance's units. The policy has only ever seen `{0, inf}` there. Putting
1800.0 into it is the identical error to `jamFactor / 10`, one field along, and I
made it in the plan after writing the paragraph warning against it. The field keeps
its `sim_parity` provenance and the feed does not own it.

`segment_target_speed` survives the same check: the simulator uses
`ego.free_flow_speed_mps` when no AVs are near (`src/sensing/local.py:203-207`), and
HERE's `freeFlow` is a free-flow speed in m/s. Same quantity, same units.

One further parity subtlety, since it bears on task 47: in the simulator
`downstream_congestion_estimate` is a **cooperation** quantity — it is `0.0` unless
local AVs are sensed (`src/sensing/local.py:204-206`). Feeding it from a traffic
service changes where the number comes from, not just how it is computed.

## The staleness aging term

The feed's value ages twice over: the measured response age, and the feed's own lag,
which task 27 established is unreported and unmeasurable.

**Recommended: a declared confidence that travels beside the field, not a silent
blend into it.** The field carries the feed's value while it is fresh enough, and
falls to the next owner when it is not; the age and the confidence are recorded.
The rejected alternative is interpolating toward neutral as the value ages, which
manufactures a middle number that neither source measured and that reads as a
moderate congestion — the same shape as returning `0.0` for "unknown", which task 27
exists to avoid.

Freshness follows the pattern already in `ObservationBuilder`: charge the bound
against the limit, `age + bound <= limit`, with a symmetric guard against a stamp
from this clock's future — the rule `PhoneGpsReader.is_stale` states and which task
27 found the HERE feed breaking.

**The cross-track offset gates ownership.** Task 27's cone admits 2.6 km of lateral
offset at the horizon, so a match far off the heading ray is not our road. The feed
owns the field only when the match is within a corridor; beyond it, the next owner
takes over and the provenance says why. This is what `link_cross_track_m` was added
for.

## The work

1. A `FeedFusion` that turns a `FlowReading` into `(value, provenance)` per field it
   owns, or declines with a named reason. One place holds every mapping.
2. Ownership resolution in `ObservationBuilder`: an ordered chain per field, taking
   the first owner that can answer, recording which.
3. The aging term and the corridor gate, both as named constants with their reasons.
4. Provenance: a new `field_sources` value for feed-derived fields, and the feed's
   outcome plus response age in the diagnostics block.

## Tests

- Camera and feed never both write a field; ownership is exclusive.
- A jammed link ahead sets the congestion field; the identical reading beyond the
  corridor does not, and the provenance says which.
- The value is the speed ratio, not `jamFactor/10` — pinned against a reading where
  the two differ, so a later "simplification" to the tempting mapping fails.
- A stale reading falls to the next owner rather than to a decayed value.
- A stamp from this clock's future does not read as fresh.
- With no feed at all, every field and provenance is exactly what it is today —
  ingestion changed nothing for a run without a phone.

## Needs sign-off

- The speed-ratio mapping (decision above), and with it the admission that
  `downstream_congestion_estimate` from a real drive is an approximation of a
  quantity the live system cannot observe.
- That task 47's parity check for this field compares behaviour, not equality.
