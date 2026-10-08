from here_fixtures import segment
from here_error import corridors


def roads_of(*segs):
    r = corridors.build_roads(segs)
    return r, [r.by_segment[s.key].road_id for s in segs]


def test_end_to_end_segments_form_one_road_whatever_their_descriptions():
    a = segment("Cross St A", [(0, 0), (500, 0)])
    b = segment("Cross St B", [(500, 0), (1000, 40)])       # 4.6 degree turn
    c = segment("Cross St C", [(1000, 40), (1500, 60)])
    r, ids = roads_of(a, b, c)
    assert len(r.roads) == 1 and len(set(ids)) == 1


def test_joint_gap_limit_is_5_m():
    a = segment("A", [(0, 0), (500, 0)])
    near = segment("B", [(504, 0), (1000, 0)])
    far = segment("C", [(1006, 0), (1500, 0)])
    _, ids = roads_of(a, near, far)
    assert ids[0] == ids[1] and ids[2] != ids[0]


def test_sharp_turn_at_the_joint_is_a_different_road():
    a = segment("A", [(0, 0), (500, 0)])
    turn = segment("B", [(500, 0), (500, 500)])              # 90 degrees
    gentle = segment("C", [(500, 0), (900, 200)])            # 26.6 degrees
    _, ids = roads_of(a, turn, gentle)
    assert ids[1] != ids[0] and ids[2] == ids[0]


def test_opposite_carriageways_alongside_are_one_road():
    east = segment("E", [(0, 0), (1000, 0)])
    west = segment("W", [(1000, 25), (0, 25)])
    _, ids = roads_of(east, west)
    assert ids[0] == ids[1]


def test_opposite_carriageways_further_than_40_m_apart_are_not():
    east = segment("E", [(0, 0), (1000, 0)])
    west = segment("W", [(1000, 60), (0, 60)])
    _, ids = roads_of(east, west)
    assert ids[0] != ids[1]


def test_a_crossing_street_is_not_alongside():
    main = segment("Main", [(0, 0), (1000, 0)])
    cross = segment("Cross", [(500, -400), (500, 400)])
    _, ids = roads_of(main, cross)
    assert ids[0] != ids[1]


def test_same_direction_parallel_road_is_not_merged():
    a = segment("A", [(0, 0), (1000, 0)])
    b = segment("B", [(0, 25), (1000, 25)])
    _, ids = roads_of(a, b)
    assert ids[0] != ids[1]


def test_label_is_geometry_never_a_lone_description():
    a = segment("Short St", [(0, 0), (300, 0)])
    b = segment("Long Ave", [(300, 0), (1300, 0)])
    west = segment("Opposite Rd", [(1300, 20), (0, 20)])
    r1, _ = roads_of(a, b, west)
    r2, _ = roads_of(west, b, a)
    (road,) = r1.roads
    assert road.road_id == "R01" and r2.roads[0].segment_keys == road.segment_keys     # independent of input order
    label = corridors.describe(road.road_id, [a, b, west])
    assert label == corridors.describe(road.road_id, [west, b, a])
    assert label.startswith("R01: 3 directed segments, 2.6 km of shape, bearing about 90/270 degrees")
    assert "segments ending at Short St ... " in label                           # first segment in chain order
    assert label not in {"Short St", "Long Ave", "Opposite Rd"} and "corridor" not in label
    one = corridors.describe("R02", [segment("Only St", [(0, 0), (500, 0)])])
    assert one.endswith("segment ending at Only St") and one != "Only St"


def test_ids_are_stable_across_input_order():
    a = segment("A", [(0, 0), (300, 0)])
    b = segment("B", [(5000, 0), (5300, 0)])
    assert [x.road_id for x in roads_of(a, b)[0].roads] == [x.road_id for x in roads_of(b, a)[0].roads] == ["R01", "R02"]


def test_assembly_stamps_every_pass_with_its_road(tmp_path):
    from here_fixtures import drive_east, make_body
    from here_error import matching

    a = segment("Cross St A", [(0, 0), (500, 0)])
    b = segment("Cross St B", [(500, 0), (1000, 0)])
    body = make_body("run_t", 0, [a, b], 0.0)
    fixes = tuple(drive_east(0, 100, 900))
    passes, cat = matching.extract_passes("run_t", fixes, (body,))
    roads = corridors.build_roads(e.segment for e in cat.values())
    assert len(passes) == 2
    assert {roads.by_segment[p.segment_key].road_id for p in passes} == {"R01"}


def test_label_names_an_empty_description():
    assert corridors.describe("R01", [segment("", [(0, 0), (500, 0)])]).endswith("segment ending at (unnamed)")


def test_a_short_crossing_segment_inside_the_pairing_distance_is_not_an_opposite_carriageway():
    main = segment("Main", [(0, 0), (1000, 0)])
    stub = segment("Stub", [(500, -30), (500, 30)])          # every sample is within 40 m, but it runs at 90 degrees
    _, ids = roads_of(main, stub)
    assert ids[0] != ids[1]


def test_a_segment_alongside_for_only_a_small_share_of_its_length_is_not_paired():
    east = segment("East", [(0, 0), (1000, 0)])
    west = segment("Diverging", [(1000, 20), (800, 20), (300, 400)])      # antiparallel for 200 m, then leaves
    _, ids = roads_of(east, west)
    assert ids[0] != ids[1]


def test_a_99_m_stub_is_not_paired_with_a_cross_street_it_crosses():
    # Cross street at bearing 55 degrees; a 99 m stub at bearing 272 degrees crossing it: every stub sample
    # is within 40 m of the cross street and the bearings are 143 degrees apart, but the overlap is under 200 m.
    import math

    d = (math.sin(math.radians(55)), math.cos(math.radians(55)))
    cross = segment("Cross", [(-600 * d[0], -600 * d[1]), (600 * d[0], 600 * d[1])])
    s = (math.sin(math.radians(272)), math.cos(math.radians(272)))
    stub = segment("Stub", [(49.5 * -s[0], 49.5 * -s[1]), (49.5 * s[0], 49.5 * s[1])])
    _, ids = roads_of(cross, stub)
    assert ids[0] != ids[1]
    long_stub = segment("Long", [(300 * -s[0], 300 * -s[1]), (300 * s[0], 300 * s[1])])
    # A 600 m segment on the same line is not paired either: it crosses at 37 degrees, so only 7 of its 31 samples
    # (about 140 m) lie within 40 m of the cross street.
    assert roads_of(cross, long_stub)[1][0] != roads_of(cross, long_stub)[1][1]


def test_opposite_carriageways_overlapping_200_m_or_more_are_paired():
    east = segment("E", [(0, 0), (260, 0)])
    west = segment("W", [(260, 20), (0, 20)])
    assert len(set(roads_of(east, west)[1])) == 1
    east2 = segment("E", [(0, 0), (150, 0)])
    west2 = segment("W", [(150, 20), (0, 20)])
    assert len(set(roads_of(east2, west2)[1])) == 2


def test_join_is_found_whichever_segment_key_sorts_first():
    # The first segment's key ("Z...") sorts after the next segment's ("A..."), so the pair is met as (A, Z) in the loop.
    first = segment("Zebra", [(0, 0), (500, 0)])
    nxt = segment("Aardvark", [(500, 0), (1000, 0)])
    assert first.key > nxt.key
    assert len(set(roads_of(first, nxt)[1])) == 1


def test_a_road_is_described_by_the_segments_that_carry_analysed_passes():
    from dataclasses import replace

    from here_error import pipeline
    from here_error.models import Pass, PassTimes
    from here_fixtures import fix

    main = segment("Main", [(0, 0), (1000, 0)])
    ramp = segment("Ramp", [(1000, 0), (1100, 30)])            # chained on by the join rule, never driven
    roads = corridors.build_roads([main, ramp])
    assert len(roads.roads) == 1
    p = Pass("p1", "r", main.key, "Main", 2, (fix(0, 0, 0), fix(60, 600, 0)), 0.0, 600.0, False, 600.0, 600.0, None)
    t = PassTimes("p1", 0, 0.0, 0.0, True, 60.0, 60.0, 40.0, 0.0, -0.3, 1.0, 0.9, 0.0, 60.0, None, None, None, None, None, None, None)
    (out,) = pipeline.label_roads([p], [t], roads, {main.key: main, ramp.key: ramp})
    assert out.road_id == "R01"
    assert out.road_label.startswith("R01: 1 directed segment, 1.0 km of shape")         # not 2 segments / 1.1 km
    assert out.road_label.endswith("segment ending at Main")
    # With no analysed pass the road is described by the segments its found passes lie on.
    t2 = replace(t, exclusion="no_reading")
    (out2,) = pipeline.label_roads([p], [t2], roads, {main.key: main, ramp.key: ramp})
    assert "1 directed segment" in out2.road_label


def test_join_uses_the_last_piece_of_the_first_segment():
    # The first segment runs east then turns north; the second continues north. Its first piece alone (east) would
    # make the joint a 90 degree turn.
    bent = segment("Bent", [(0, 0), (400, 0), (400, 300)])
    north = segment("North", [(400, 300), (400, 800)])
    assert len(set(roads_of(bent, north)[1])) == 1
    sharp = segment("Sharp", [(400, 300), (900, 300)])        # east again: a 90 degree turn from the last piece
    assert len(set(roads_of(bent, sharp)[1])) == 2


def test_an_excluded_pass_on_another_segment_does_not_enter_the_label():
    from here_error import pipeline
    from here_error.models import Pass, PassTimes
    from here_fixtures import fix

    main = segment("Main", [(0, 0), (1000, 0)])
    side = segment("Side", [(1000, 0), (1500, 0)])
    roads = corridors.build_roads([main, side])
    p1 = Pass("p1", "r", main.key, "Main", 2, (fix(0, 0, 0), fix(60, 600, 0)), 0.0, 600.0, False, 600.0, 600.0, None)
    p2 = Pass("p2", "r", side.key, "Side", 2, (fix(100, 1000, 0), fix(110, 1100, 0)), 0.0, 100.0, False, 100.0, 100.0, "shorter_than_min")
    ok = PassTimes("p1", 0, 0.0, 0.0, True, 60.0, 60.0, 40.0, 0.0, -0.3, 1.0, 0.9, 0.0, 60.0, None, None, None, None, None, None, None)
    bad = PassTimes("p2", None, None, None, None, 10.0, None, None, None, None, None, None, 0.0, 10.0, None, None, None, None, None, None, "shorter_than_min")
    out = pipeline.label_roads([p1, p2], [ok, bad], roads, {main.key: main, side.key: side})
    assert [o.road_label for o in out][0] == [o.road_label for o in out][1]
    assert "1 directed segment, 1.0 km of shape" in out[0].road_label and out[0].road_label.endswith("segment ending at Main")
