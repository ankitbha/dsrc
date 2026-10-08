import math

import numpy as np
import pytest

from here_error import geometry as g

# An L-shaped polyline: 100 m east, then 50 m north.
L = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0]])


def test_length_and_cumulative():
    assert g.polyline_length(L) == pytest.approx(150.0)
    assert list(g.cumulative_lengths(L)) == pytest.approx([0.0, 100.0, 150.0])


def test_point_beside_first_piece():
    p = g.project_point(L, [40.0, 7.0])
    assert p.distance_m == pytest.approx(7.0)
    assert p.chainage_m == pytest.approx(40.0)
    assert p.bearing_deg == pytest.approx(90.0)
    assert not p.clamped_start and not p.clamped_end


def test_point_on_second_piece_has_northward_bearing():
    p = g.project_point(L, [103.0, 20.0])
    assert p.distance_m == pytest.approx(3.0)
    assert p.chainage_m == pytest.approx(120.0)
    assert p.bearing_deg == pytest.approx(0.0)


def test_vertex_in_the_middle_is_not_an_end():
    # Outside the bend: nearest place is the middle vertex (100, 0).
    p = g.project_point(L, [105.0, -5.0])
    assert p.chainage_m == pytest.approx(100.0)
    assert not p.clamped_start and not p.clamped_end


def test_point_before_start_is_clamped_to_start():
    p = g.project_point(L, [-6.0, 2.0])
    assert p.chainage_m == pytest.approx(0.0)
    assert p.distance_m == pytest.approx(math.hypot(6, 2))
    assert p.clamped_start and not p.clamped_end


def test_point_past_end_is_clamped_to_end():
    p = g.project_point(L, [100.0, 58.0])
    assert p.chainage_m == pytest.approx(150.0)
    assert p.distance_m == pytest.approx(8.0)
    assert p.clamped_end and not p.clamped_start


def test_point_exactly_at_the_ends_is_inside():
    assert not g.project_point(L, [0.0, 0.0]).clamped_start
    assert not g.project_point(L, [100.0, 50.0]).clamped_end


def test_angle_between_bearings_wraps():
    assert g.angle_between_deg(350, 10) == pytest.approx(20)
    assert g.angle_between_deg(0, 180) == pytest.approx(180)
    assert g.angle_between_deg(90, 90) == 0


def test_haversine_and_local_frame_agree():
    d = g.haversine_m(40.0, -74.0, 40.01, -74.0)
    assert d == pytest.approx(1111.95, rel=1e-3)
    f = g.LocalFrame(40.0, -74.0)
    xy = f.to_xy([40.0, 40.01], [-74.0, -74.0])
    assert g.polyline_length(xy) == pytest.approx(d, rel=1e-3)
    east = f.to_xy([40.0], [-73.99])[0]
    assert east[0] == pytest.approx(g.haversine_m(40.0, -74.0, 40.0, -73.99), rel=1e-3)
