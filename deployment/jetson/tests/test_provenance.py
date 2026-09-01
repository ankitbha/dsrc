"""The closed provenance vocabulary, and the one partition anything keys on.

`provenance.py` is the single home for every class `field_sources` and the
four `Inputs` source fields may carry. These tests pin the vocabulary itself
and `summarise`'s arithmetic; the builder and controller tests pin that real
code only ever emits members of it.
"""

from __future__ import annotations

import itertools
import time

import pytest

from perception import provenance
from perception.observation_builder import BuilderConfig, ObservationBuilder, PeerState
from sensors.gps_reader import GpsFix


def test_every_named_constant_is_a_member_of_sources():
    names = [
        provenance.SOURCE_MEASURED,
        provenance.SOURCE_MEASURED_CONVERTED,
        provenance.SOURCE_MEASURED_ARRIVAL_PROXY,
        provenance.SOURCE_DERIVED,
        provenance.SOURCE_DERIVED_EMPTY,
        provenance.SOURCE_APPROXIMATED,
        provenance.SOURCE_FEED,
        provenance.SOURCE_STATIC_CONFIG,
        provenance.SOURCE_SIM_PARITY,
        provenance.SOURCE_FALLBACK_NEUTRAL,
        provenance.SOURCE_UNATTRIBUTED,
    ]
    assert len(names) == len(set(names)) == 11
    assert set(names) == provenance.SOURCES


def test_substituted_is_a_subset_of_sources():
    assert provenance.SUBSTITUTED <= provenance.SOURCES


def test_derived_empty_is_not_substituted():
    # D4: under shipped constants the disagreement rule fires if and only if
    # the camera detected nothing (`local_density_bin <= 0`), which is
    # exactly what `derived_empty` means. Gating on it here would delete the
    # rule's only firing path -- this is the decision most likely to be
    # "fixed" by a later reader, so it is pinned by name.
    assert provenance.SOURCE_DERIVED_EMPTY not in provenance.SUBSTITUTED


@pytest.mark.parametrize("source", sorted(provenance.SOURCES))
def test_is_substituted_matches_membership_for_every_source(source):
    assert provenance.is_substituted(source) == (source in provenance.SUBSTITUTED)


def test_is_substituted_is_false_for_none():
    # None means "no class was read at all" (a caller error further up),
    # which is a different fact from "the field is unattributed" -- the
    # latter is itself a `SOURCES` member (`SOURCE_UNATTRIBUTED`) and is
    # substituted; a bare `None` is not a class this function judges.
    assert provenance.is_substituted(None) is False


def test_summarise_counts_by_class_and_computes_missingness():
    field_sources = {
        "a": provenance.SOURCE_MEASURED,
        "b": provenance.SOURCE_FALLBACK_NEUTRAL,
        "c": provenance.SOURCE_FALLBACK_NEUTRAL,
        "d": provenance.SOURCE_DERIVED_EMPTY,
    }
    summary = provenance.summarise(field_sources)
    assert summary["fields"] == 4
    assert summary["by_source"] == {
        provenance.SOURCE_MEASURED: 1,
        provenance.SOURCE_FALLBACK_NEUTRAL: 2,
        provenance.SOURCE_DERIVED_EMPTY: 1,
    }
    assert summary["missingness"] == 0.5
    assert sorted(summary["fallback_fields"]) == ["b", "c"]


def test_summarise_of_an_empty_map_does_not_divide_by_zero():
    summary = provenance.summarise({})
    assert summary == {"fields": 0, "by_source": {}, "missingness": 0.0, "fallback_fields": []}


def _fresh_fix(speed: float, t_mono: float) -> GpsFix:
    return GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=speed, heading_deg=90.0,
                 fix_quality=1, num_sats=8, hdop=1.0, t_mono=t_mono, t_wall=time.time())


def _vehicle(track_id: int, dist: float, *, rel_valid: bool = True):
    import numpy as np

    from perception.distance import TrackedVehicle

    return TrackedVehicle(
        track_id=track_id, xyxy=np.array([0, 0, 10, 10], dtype=np.float32), cls=2,
        conf=0.9, distance_m=dist, lateral_m=0.0, rel_speed_mps=1.0,
        rel_speed_valid=rel_valid, method="ground_plane",
    )


def test_every_value_across_a_state_sweep_is_a_member_of_sources():
    """No fix / stale fix / fresh fix; no tracks / tracks with and without
    measurable speeds; no peers / peers with and without `lane_id`. Every
    value the builder emits, across all of it, has to be a `SOURCES` member.
    """
    cfg = BuilderConfig()
    t = 1000.0
    no_peers: list[PeerState] = []
    peers_with_lane = [PeerState(peer_id="a", distance_m=50.0, speed_mps=20.0, lane_id=1)]
    peers_without_lane = [PeerState(peer_id="b", distance_m=50.0, speed_mps=20.0, lane_id=None)]

    vehicle_sets = [
        [],
        [_vehicle(1, 30.0, rel_valid=True)],
        [_vehicle(1, 30.0, rel_valid=False)],
    ]
    gps_choices = [
        GpsFix(valid=False),
        GpsFix(valid=True, speed_mps=20.0, t_mono=t - 10.0, t_wall=0.0),  # stale
        _fresh_fix(20.0, t),
    ]
    peer_choices = [no_peers, peers_with_lane, peers_without_lane]

    for vehicles, gps, peers in itertools.product(vehicle_sets, gps_choices, peer_choices):
        builder = ObservationBuilder(cfg)
        # Warm the ego state up over a few fresh ticks first, so the sweep
        # also visits `ego_acceleration`'s `derived` branch and not only its
        # substitutions.
        for i in range(4):
            builder.build([], _fresh_fix(20.0, t + i * 0.1), t + i * 0.1)
        result = builder.build(vehicles, gps, t + 0.5, peers)
        for field, source in result.field_sources.items():
            assert source in provenance.SOURCES, (field, source)
