"""scripts/export_dsrc_network.py: the DSRC network definition it writes.

Imported the same way test_mainz_port.py imports src.sumo.mainz -- this is
the generator for specs/dsrc_network_mainz.json, and its numbers are
required to round-trip to the same lane counts and lengths section 1.3 of
plan_task145 measured directly from sumolib, since the runtime's static
feature half (`lanes`, `length_km`) is read from this file rather than from
the map at inference time.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import export_dsrc_network as gen  # noqa: E402

NET = gen.build_network_definition(gen.NET_FILE, gen.SEGMENT_FILE)

#: Section 1.3's table, measured directly with sumolib against the working
#: tree. Order matches data/mainz/mainz_segments.json.
EXPECTED = [
    (60.0, 7.238, 1.96),
    (60.0, 1.945, 2.60),
    (60.0, 7.876, 1.64),
    (60.0, 4.848, 2.00),
    (60.0, 2.680, 1.82),
    (60.0, 9.122, 2.10),
    (60.0, 3.822, 3.33),
    (60.0, 4.072, 2.47),
    (60.0, 2.600, 2.83),
    (60.0, 2.370, 5.00),
    (60.0, 3.735, 1.73),
    (60.0, 0.790, 3.50),
]


def test_twelve_segments_in_order():
    assert [s["segment_id"] for s in NET["segments"]] == [str(i) for i in range(12)]


@pytest.mark.parametrize("index", range(12))
def test_segment_statics_match_plan_section_1_3(index):
    speed_limit, length_km, lanes = EXPECTED[index]
    segment = NET["segments"][index]
    assert segment["speed_limit_kmh"] == pytest.approx(speed_limit, abs=0.05)
    assert segment["length_km"] == pytest.approx(length_km, abs=1e-3)
    assert segment["lanes"] == pytest.approx(lanes, abs=1e-2)


def test_edge_ids_match_the_segment_file():
    import json

    segments = json.loads(gen.SEGMENT_FILE.read_text())
    assert [s["edge_ids"] for s in NET["segments"]] == segments


def test_every_edge_carries_a_nonempty_polyline():
    for segment in NET["segments"]:
        for edge in segment["edges"]:
            assert len(edge["polyline"]) >= 2
            for lat, lon in edge["polyline"]:
                assert -90.0 <= lat <= 90.0
                assert -180.0 <= lon <= 180.0


def test_geo_reference_documents_the_placement_as_synthetic():
    ref = NET["geo_reference"]
    assert ref["kind"] == "synthetic_local_tangent_plane"
    assert ref["anchor_lat"] == gen.ANCHOR_LAT
    assert ref["anchor_lon"] == gen.ANCHOR_LON
    assert "projParameter" in ref["note"]


def test_the_network_centre_lands_on_the_anchor():
    """The bounding-box centre, run through the same projection, is the anchor."""
    lat, lon = gen._local_to_lonlat(0.0, 0.0, 0.0, 0.0)
    assert lat == pytest.approx(gen.ANCHOR_LAT)
    assert lon == pytest.approx(gen.ANCHOR_LON)


def test_a_1000m_east_offset_is_about_1000m_at_this_latitude():
    """Loose sanity check on the projection's scale, not a precision claim."""
    from geo import haversine_m

    lat0, lon0 = gen._local_to_lonlat(0.0, 0.0, 0.0, 0.0)
    lat1, lon1 = gen._local_to_lonlat(1000.0, 0.0, 0.0, 0.0)
    assert haversine_m(lat0, lon0, lat1, lon1) == pytest.approx(1000.0, rel=0.01)


def test_nanmean_excludes_nan_matching_mainz_pooling():
    assert gen._nanmean([1.0, math.nan, 3.0]) == pytest.approx(2.0)
    assert math.isnan(gen._nanmean([math.nan, math.nan]))
