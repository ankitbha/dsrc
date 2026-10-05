"""Every tunable number of the analysis, each with the reason it has that value.

Provenance records `all_params()`, so a result file states the values it was computed with.
"""
from __future__ import annotations

#: Metres from a HERE segment's shape within which a GPS fix counts as lying on it.
MATCH_M = 15.0
#: Largest angle between the GPS heading and the segment's direction. HERE reports each
#: direction of a road as its own segment, so this test picks the direction.
HEADING_TOL_DEG = 45.0
#: Below this speed the GPS heading is unreliable, so it is not tested.
HEADING_MIN_SPEED_MPS = 2.0
#: Largest gap between consecutive fixes inside one pass.
PASS_GAP_S = 3.0
#: Matched-portion length below which a pass is left out of the analysis.
MIN_PASS_M = 200.0
#: Chainage may fall back by at most this much inside a pass (GPS noise), else the car left and re-entered.
REVERSE_TOL_M = 20.0
#: Below this speed the car counts as stopped.
STOP_MPS = 1.0
#: Gate V1: largest 5th-to-95th percentile spread of the phone-clock-minus-GPS-UTC offset.
V1_MAX_SPREAD_S = 1.0
#: Roads: a segment continues another when its first point is within this many metres of the other's last point...
ROAD_JOIN_M = 5.0
#: ...and the bearing changes by less than this many degrees across the joint.
ROAD_JOIN_TURN_DEG = 30.0
#: Roads: opposite carriageways run within this many metres of each other...
ROAD_PAIR_M = 40.0
#: ...for at least this share of the shorter segment's length, sampled every ROAD_SAMPLE_M metres...
ROAD_PAIR_SHARE = 0.8
#: ...and over at least this many metres, so a short stub at a junction is not paired with a cross street...
ROAD_PAIR_MIN_OVERLAP_M = 200.0
ROAD_SAMPLE_M = 20.0
#: ...heading opposite ways: the bearings differ by more than this many degrees.
ROAD_ANTIPARALLEL_DEG = 135.0
#: Gate V3: largest relative difference between the path length from positions and from integrated speed.
V3_MAX_REL_DIFF = 0.03
#: Allowed relative difference between HERE's stated segment length and the shape's length.
SEGMENT_LENGTH_TOL = 0.03
#: Oldest HERE data a pass may use under the primary reading rule.
MAX_READING_AGE_S = 300.0
#: Control V4: the shuffled reading must be at least this far in time from the pass.
SHUFFLE_MIN_S = 1200.0
#: Shortest stretch of consecutive passes.
STRETCH_MIN_M = 2000.0
#: Longest stretch of consecutive passes.
STRETCH_MAX_M = 5000.0
#: Largest gap between one pass's last fix and the next pass's first fix inside a stretch.
STRETCH_MAX_GAP_S = 60.0
#: Bootstrap resamples.
BOOTSTRAP_N = 10000
#: Random seed for the bootstrap and the label sample.
SEED = 757
#: Detector weights file name; stored under the output directory, never in the repository.
YOLO_WEIGHTS = "yolov8m.pt"
#: Detection confidence threshold.
YOLO_CONF = 0.35
#: COCO class ids counted as vehicles: car, motorcycle, bus, truck.
VEHICLE_CLASSES = (2, 3, 5, 7)
#: A vehicle is the leader when its box centre lies inside this central share of the upright image width...
LEADER_CENTRE_SHARE = 0.30
#: ...and its box height is at least this share of the image height. A heuristic, scored by gate V6.
LEADER_MIN_HEIGHT_SHARE = 0.04
#: A box whose bottom edge is within this share of the image height of the bottom, and that is at least
#: EGO_HOOD_MIN_WIDTH_SHARE of the image width, is the recording car's own bonnet, which the detector
#: reports as a car. It is neither a vehicle in the scene nor a leader.
EGO_HOOD_BOTTOM_SHARE = 0.03
EGO_HOOD_MIN_WIDTH_SHARE = 0.8
#: Local time (America/New_York) that splits daylight from dark everywhere: hour and minute.
DUSK_SPLIT_LOCAL = (19, 15)
LOCAL_TZ = "America/New_York"
#: Route-follows-path check: metres from the route polyline.
ROUTE_MATCH_M = 25.0
#: Route-follows-path check: smallest share of the pass's fixes within ROUTE_MATCH_M of the route.
ROUTE_MATCH_SHARE = 0.9
#: Route-follows-path check: largest relative difference between route length and matched length.
ROUTE_LENGTH_TOL = 0.15
#: Largest number of uncached HERE calls one invocation may make.
HERE_CALL_BUDGET = 80
#: Stage-5 driver offset uses passes below these leader share, stopped share and HERE jam factor.
OFFSET_MAX_LEADER_SHARE = 0.2
OFFSET_MAX_STOPPED_SHARE = 0.05
OFFSET_MAX_JAM = 2.0
#: Smallest group size for which the report prints an interval.
MIN_N_FOR_INTERVAL = 5
#: Label sample size per half of the day.
LABEL_SAMPLE_PER_HALF = 75

#: The two runs with stored HERE bodies, and the phone log holding each one's GPS, HERE and camera records.
RUNS = (
    ("run_20260908_170849", "fb3a19d764b9337e-1788901730.jsonl"),
    ("run_20260908_190548", "fb3a19d764b9337e-1788908748.jsonl"),
)
PHONE_SUBDIR = "phone_sessions_20260908_evening"


def all_params() -> dict[str, object]:
    """Every upper-case constant in this module, for provenance records."""
    return {k: v for k, v in sorted(globals().items()) if k.isupper()}
