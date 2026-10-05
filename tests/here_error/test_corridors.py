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


def test_label_is_the_longest_members_description_and_ids_are_stable():
    a = segment("Short", [(0, 0), (300, 0)])
    b = segment("Long", [(300, 0), (1300, 0)])
    r1, _ = roads_of(a, b)
    r2, _ = roads_of(b, a)
    assert r1.roads[0].label == "Long corridor"
    assert [x.road_id for x in r1.roads] == [x.road_id for x in r2.roads] == ["R01"]


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
