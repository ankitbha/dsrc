"""The closed provenance vocabulary every observation field is tagged with.

`ObservationBuilder.build` (observation_builder.py) tags each of the 39
encoder slots with one of the classes below, in `field_sources`. `Inputs`
(policy/sensing_controller.py) carries the class behind four of its own
fields, filled in by `inputs_from` (policy/sensing_loop.py). Both read the
vocabulary from here rather than typing the strings themselves, so a class
spelled one way in the builder and another in the controller cannot drift
apart silently.

`SUBSTITUTED` is the one partition anything downstream keys on: a class in
it means the value is not evidence about this tick -- no measurement or
derivation of this tick's inputs produced it, so a rule that requires
evidence must treat it as absent rather than decide on it.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Read directly off a sensor this tick.
SOURCE_MEASURED = "measured"
#: Measured, but the stamp establishing its freshness was converted across a
#: timebase estimate rather than read on this device.
SOURCE_MEASURED_CONVERTED = "measured_converted"
#: Measured, with an arrival-time proxy standing in for the reading's own
#: capture stamp.
SOURCE_MEASURED_ARRIVAL_PROXY = "measured_arrival_proxy"
#: Computed from one or more measured quantities of this tick.
SOURCE_DERIVED = "derived"
#: Computed the same way as `SOURCE_DERIVED`, but the only input was an
#: absence -- an empty detection set, not a substituted number. The count is
#: correct; there was nothing to count.
SOURCE_DERIVED_EMPTY = "derived_empty"
#: Computed from a measured quantity that answers a different question than
#: the simulator's own formula does, documented at the call site.
SOURCE_APPROXIMATED = "approximated"
#: Owned by the traffic feed rather than by the vehicle's own sensors.
SOURCE_FEED = "feed_derived"
#: An operator-provided constant, not read or derived this tick.
SOURCE_STATIC_CONFIG = "static_config"
#: A value the simulator itself hardcodes, kept identical for parity.
SOURCE_SIM_PARITY = "sim_parity"
#: The spec-mandated neutral value, standing in for a measurement this tick
#: could not make.
SOURCE_FALLBACK_NEUTRAL = "fallback_neutral"
#: `inputs_from`'s own class for a field `field_sources` carries no entry
#: for at all. The builder never writes it -- the coverage test makes it
#: unreachable in production -- so its only producer is a caller bug.
SOURCE_UNATTRIBUTED = "unattributed"

SOURCES = frozenset({
    SOURCE_MEASURED,
    SOURCE_MEASURED_CONVERTED,
    SOURCE_MEASURED_ARRIVAL_PROXY,
    SOURCE_DERIVED,
    SOURCE_DERIVED_EMPTY,
    SOURCE_APPROXIMATED,
    SOURCE_FEED,
    SOURCE_STATIC_CONFIG,
    SOURCE_SIM_PARITY,
    SOURCE_FALLBACK_NEUTRAL,
    SOURCE_UNATTRIBUTED,
})

#: Classes whose value is not evidence about this tick. `SOURCE_DERIVED_EMPTY`
#: is deliberately absent: a zero count from an empty detection set is a
#: real, if weak, observation, and excluding it would delete the
#: disagreement rule, whose only firing condition under shipped constants is
#: an empty detection set.
SUBSTITUTED = frozenset({
    SOURCE_FALLBACK_NEUTRAL,
    SOURCE_STATIC_CONFIG,
    SOURCE_SIM_PARITY,
    SOURCE_UNATTRIBUTED,
})


def is_substituted(source: str | None) -> bool:
    """Whether a value tagged with this class is evidence about this tick."""
    return source in SUBSTITUTED


def summarise(field_sources: Mapping[str, str]) -> dict[str, Any]:
    """The provenance rollup over one tick's `field_sources`.

    `missingness` and `fallback_fields` are the two names `obs_diagnostics`
    carried before this module existed, computed here so the formula has one
    home instead of being re-typed at every call site. `fields` and
    `by_source` describe the map exactly as given; whether that count is the
    full encoder contract is for the caller to decide, against
    `sim_contract.local_obs_dim()`.
    """
    by_source: dict[str, int] = {}
    for source in field_sources.values():
        by_source[source] = by_source.get(source, 0) + 1
    fallback_fields = [k for k, v in field_sources.items() if v == SOURCE_FALLBACK_NEUTRAL]
    total = len(field_sources)
    return {
        "fields": total,
        "by_source": by_source,
        "missingness": round(len(fallback_fields) / total, 3) if total else 0.0,
        "fallback_fields": fallback_fields,
    }
