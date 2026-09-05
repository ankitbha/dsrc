"""Task 47: a slot-by-slot parity ledger between the live and simulated
sensing models.

The brief as worded -- "the observation vector produced live matches the
simulator's sensing model field for field" -- does not hold, and nothing in
the tree proves it either way: `deployment/jetson/tests/test_sim_contract.py`
feeds one hand-written observation dict to both `encode_local_observation`
calls, which proves the two **encoders** agree on arithmetic, not that the
two **sensing models** (`src.sensing.LocalObservationBuilder` here,
`perception.observation_builder.ObservationBuilder` on the Jetson) agree on
what a scene looks like. This module is the first thing in the repository
that imports both producers.

One scene description (`Scene`) is expressed once and instantiated
separately on each side (D20): `sim_observation` builds a real
`TopologySpec`, real `VehicleSnapshot`s and calls
`LocalObservationBuilder.build_one`; `live_observation` builds real
`TrackedVehicle`s and a real `GpsFix` and calls `ObservationBuilder.build`.
Neither reads the other's model of the scene, so a difference is
attributable to the sensing model rather than to a harness that fed one
side's dict to both -- the failure `test_sim_contract.py` already has.

Both resulting dicts are encoded through the SAME `encode_local_observation`
(D20's other half), so a difference downstream of that point is the
sensing model, never the encoder.

The classification per slot (`identical` / `approximated` / `substituted` /
`structurally_absent`) is a fact about the CODE -- which line of
`src/sensing/local.py` and which line of
`deployment/jetson/perception/observation_builder.py` produce it, and
whether the live side reads a real sensor or a constant -- not a fact this
module discovers by running scenes. The scene set exists to prove the
classification is not scene-dependent: `identical` must hold bit-for-bit
on every scene (D19), and a live SUBSTITUTED/STRUCTURALLY_ABSENT value must
equal the named constant on every scene the mechanism applies to.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JETSON_DIR = REPO_ROOT / "deployment" / "jetson"
if str(JETSON_DIR) not in sys.path:
    # Mirrors scripts/generate_transport_golden_frames.py's own insertion:
    # `perception`, `policy`, `sensors` are only importable once the Jetson
    # tree is on the path, and this module needs both that tree AND `src.*`.
    sys.path.insert(0, str(JETSON_DIR))

from perception.distance import TrackedVehicle  # noqa: E402
from perception.observation_builder import BuilderConfig, ObservationBuilder  # noqa: E402
from perception.provenance import SUBSTITUTED  # noqa: E402
from policy import sim_contract  # noqa: E402
from sensors.gps_reader import GpsFix  # noqa: E402

from src.road.topology_factory import build_topology  # noqa: E402
from src.safety import SafetyConstraints, SafetyState  # noqa: E402
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot  # noqa: E402

#: The tolerance D13/47.3 names, matching the 5-decimal rounding
#: `metadata.jsonl` applies to `encoded` (`Tick.to_record()`,
#: `deployment/jetson/pipeline.py:141`) -- a diff finer than that rounding
#: would be invisible in a logged run regardless of what this harness found.
ATOL = 1e-5

CLASS_IDENTICAL = "identical"
CLASS_APPROXIMATED = "approximated"
CLASS_SUBSTITUTED = "substituted"
CLASS_STRUCTURALLY_ABSENT = "structurally_absent"


# -- One scene, expressed once ----------------------------------------------


@dataclass(frozen=True)
class SceneVehicle:
    """One neighbour, relative to the ego. The same four numbers feed both
    producers below; each translates them into its own vehicle model."""

    role: str  # "av" or "human"
    longitudinal_offset_m: float  # + ahead of the ego, - behind it
    lane_offset: int  # 0 = the ego's own lane, -1 = left, +1 = right
    speed_mps: float


@dataclass(frozen=True)
class Scene:
    """One traffic scene. `topology_id`/`segment_id` are real
    `src.road` identifiers -- not placeholders -- so lane adjacency and the
    bottleneck/merge gates are exercised for real on the simulator side."""

    name: str
    topology_id: str
    segment_id: str
    ego_lane: int
    ego_longitudinal_m: float
    ego_speed_mps: float
    free_flow_speed_mps: float = 30.0
    vehicles: tuple[SceneVehicle, ...] = ()
    lane_width_m: float = 3.7
    #: Priming ticks of constant ego speed before the compared tick, so
    #: `ego_acceleration` is genuinely DERIVED on the live side (a fresh
    #: `ObservationBuilder` has no speed window at all) rather than reporting
    #: its cold-start fallback for a reason that has nothing to do with
    #: sensing. A constant speed across the priming window means the true
    #: acceleration is zero on both sides regardless.
    priming_ticks: int = 4
    #: Sim only: which lane `target_lanes` names, relative to the ego's own.
    #: Live has no notion of a target lane distinct from the current one --
    #: `target_lane_front_gap` is always `leader_gap` there -- so this is
    #: the only way to make the sim side recompute target-lane gaps from a
    #: DIFFERENT lane's occupants than the ego's own.
    target_lane_offset: int = 0
    #: Sim only: `SafetyState.lane_changes_last_km` -- live has no lane-
    #: change detection at all, so this field only ever moves the sim side.
    lane_changes_last_km: int = 0
    #: Sim only: seconds before this scene's instant the ego's last lane
    #: change happened, or `None` for "never" (sim reports `inf` then, same
    #: as live's unconditional constant -- which is why at least one scene
    #: needs a real value here, or `time_since_last_lane_change` never shows
    #: its divergence at all).
    last_lane_change_ago_s: float | None = None
    #: Sim only, via `segment_metrics`: `downstream_congestion_estimate` is
    #: this value directly UNLESS there is no local AV neighbour, in which
    #: case it is forced to 0.0 regardless -- live has no segment-level jam
    #: reading at all, cooperating-AV or not.
    jam_fraction: float = 0.0


SCENES: tuple[Scene, ...] = (
    Scene(
        name="empty", topology_id="straight_multilane", segment_id="straight_upstream",
        ego_lane=1, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
    ),
    Scene(
        name="leader_only", topology_id="straight_multilane", segment_id="straight_upstream",
        ego_lane=1, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
        vehicles=(SceneVehicle("human", 40.0, 0, 17.0),),
        # Exercises time_since_last_lane_change / lane_changes_last_km: sim
        # has real SafetyState history to read; live has none, ever.
        lane_changes_last_km=1, last_lane_change_ago_s=12.0,
    ),
    Scene(
        # Vehicles in every direction sim can see and live cannot: one lane
        # change ago (same lane), one behind (same lane), one behind-left,
        # one behind-right -- follower_gap and both *_rear_gap slots are all
        # real, finite numbers on sim and the live constant on every one.
        name="leader_and_follower", topology_id="straight_multilane", segment_id="straight_upstream",
        ego_lane=1, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
        vehicles=(
            SceneVehicle("human", 40.0, 0, 17.0),
            SceneVehicle("human", -25.0, 0, 22.0),   # behind, same lane
            SceneVehicle("human", -20.0, -1, 19.0),  # behind, left lane
            SceneVehicle("human", -18.0, 1, 21.0),   # behind, right lane
        ),
    ),
    Scene(
        # Real adjacent lanes on both sides, and the sim's TARGET lane set
        # to the left one -- so target_lane_front_gap on sim comes from the
        # left lane's occupant, not the ego's own leader, while live still
        # aliases it to leader_gap regardless.
        name="adjacent_lane_traffic", topology_id="straight_multilane", segment_id="straight_upstream",
        ego_lane=1, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
        vehicles=(
            SceneVehicle("human", 30.0, -1, 22.0),
            SceneVehicle("human", 35.0, 1, 15.0),
            SceneVehicle("human", 45.0, 0, 17.0),  # ego's own leader, kept distinct from the target lane's
        ),
        target_lane_offset=-1,
    ),
    Scene(
        # A single-lane road: the adjacent lane does not exist AT ALL. The
        # simulator's own `_adjacent_lane` looks the neighbouring lane index
        # up in the real road network and returns None off the edge of it;
        # the live side has no such concept and computes a lane purely by
        # rounding a lateral offset, which exists whether or not a second
        # lane does. One vehicle on each side, so both left and right show it.
        name="single_lane_road", topology_id="straight_single_lane", segment_id="straight_upstream",
        ego_lane=0, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
        vehicles=(
            SceneVehicle("human", 20.0, 1, 15.0),
            SceneVehicle("human", 22.0, -1, 16.0),
        ),
    ),
    Scene(
        # `tree_bottleneck_d` is the one segment `build_topology
        # ("inverted_tree_bottleneck")` names in `bottleneck_segments`, and
        # `merge_nodes` is non-empty on this topology -- both conditions
        # `_merge_pressure` needs to read anything but a hardcoded 0.0. A
        # slow (queued), locally-sensed AV with no peer, plus a nonzero
        # `jam_fraction`, exercises downstream_congestion_estimate and
        # merge_pressure together with local_queue_estimate's own
        # forward-only-doubled-vs-omnidirectional population difference.
        name="bottleneck", topology_id="inverted_tree_bottleneck", segment_id="tree_bottleneck_d",
        ego_lane=0, ego_longitudinal_m=50.0, ego_speed_mps=10.0,
        vehicles=(
            SceneVehicle("human", 15.0, 0, 8.0),
            SceneVehicle("av", 12.0, 0, 3.0),
        ),
        jam_fraction=0.5,
    ),
    Scene(
        # Locally-sensed AV neighbours, in every lane, with no V2V peer
        # behind any of them. The simulator's cooperation fields read from
        # whatever neighbour its OWN sensing model calls an "av", full stop.
        # The live builder's camera cannot tell an AV from a human at all
        # -- TrackedVehicle carries no such field -- so these fields read
        # from `peers` (V2V beacon reception) alone, and this scene
        # deliberately supplies none: three real AVs a real camera would
        # track, that the cooperation fields on live never learn about.
        # One per lane, so all three `nearby_av_lane_distribution` slots
        # -- not just the ego's own lane -- show the divergence.
        name="local_av_neighbor_no_peer", topology_id="straight_multilane", segment_id="straight_upstream",
        ego_lane=1, ego_longitudinal_m=200.0, ego_speed_mps=20.0,
        vehicles=(
            SceneVehicle("av", 30.0, -1, 24.0),
            SceneVehicle("av", 32.0, 0, 24.0),
            SceneVehicle("av", 34.0, 1, 24.0),
        ),
    ),
)


def _edge_nodes(topology: Any, segment_id: str) -> tuple[str, str]:
    (edge,) = topology.segment_edges[segment_id]
    from_node, to_node = edge.split("->")
    return from_node, to_node


def sim_observation(scene: Scene) -> dict[str, Any]:
    """Instantiate `scene` on the simulator's own sensing model."""
    topology = build_topology(scene.topology_id)
    from_node, to_node = _edge_nodes(topology, scene.segment_id)
    ego_lane_index = (from_node, to_node, scene.ego_lane)

    def snapshot(vehicle_id: str, *, lane_id: int, longitudinal_m: float,
                 speed_mps: float, role: str = "av") -> VehicleSnapshot:
        return VehicleSnapshot(
            vehicle_id=vehicle_id, role=role, segment_id=scene.segment_id,
            lane_index=(from_node, to_node, lane_id), lane_id=lane_id,
            position=(longitudinal_m, lane_id * scene.lane_width_m),
            longitudinal_m=longitudinal_m, speed_mps=speed_mps,
            acceleration_mps2=0.0, free_flow_speed_mps=scene.free_flow_speed_mps,
        )

    ego = snapshot("ego", lane_id=scene.ego_lane, longitudinal_m=scene.ego_longitudinal_m,
                   speed_mps=scene.ego_speed_mps, role="av")
    neighbors = [
        snapshot(f"n{i}", lane_id=scene.ego_lane + v.lane_offset,
                 longitudinal_m=scene.ego_longitudinal_m + v.longitudinal_offset_m,
                 speed_mps=v.speed_mps, role=v.role)
        for i, v in enumerate(scene.vehicles)
    ]
    target_lane_index = (from_node, to_node, scene.ego_lane + scene.target_lane_offset)
    safety_state = SafetyState(
        lane_changes_last_km=scene.lane_changes_last_km,
        last_lane_change_time_s=(
            None if scene.last_lane_change_ago_s is None
            else 0.0 - scene.last_lane_change_ago_s
        ),
    )
    builder = LocalObservationBuilder(SensingConfig(range_m=150.0))
    result = builder.build_all(
        time_s=0.0, topology=topology, snapshots=[ego, *neighbors],
        current_av_ids=["ego"], safety_states={"ego": safety_state},
        target_headways={"ego": 1.6}, target_lanes={"ego": target_lane_index},
        segment_metrics={scene.segment_id: {"density": 0.0, "jam_fraction": scene.jam_fraction}},
        constraints=SafetyConstraints(), rng=np.random.RandomState(7),
    )
    return result["ego"]


def _production_builder_config() -> BuilderConfig:
    """`BuilderConfig` built exactly the way `run_demo.py` builds it (B5,
    validation round 1), not `BuilderConfig()`'s bare defaults.

    `run_demo.py` does not call `BuilderConfig()` either: it reads
    `config["observation"]`, then overrides `gps_stale_after_s` from
    `config["gps"]["stale_after_s"]` and `peer_range_m` from
    `config["v2v"]["range_m"]`, and only THEN calls `BuilderConfig.from_dict`.
    The two constructions agree today only because nobody has edited
    `config.yaml` and this file's bare defaults at the same time; reading
    the real file means a config edit that missed this one cannot leave the
    ledger silently describing a builder nobody runs.
    """
    import yaml

    with open(JETSON_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    obs_cfg = dict(config["observation"])
    obs_cfg["gps_stale_after_s"] = config["gps"]["stale_after_s"]
    obs_cfg["peer_range_m"] = config["v2v"]["range_m"]
    return BuilderConfig.from_dict(obs_cfg)


def live_observation(scene: Scene) -> "ObservationResult":  # noqa: F821
    """Instantiate `scene` on the live sensing model.

    Only FORWARD vehicles (`longitudinal_offset_m > 0`) become a
    `TrackedVehicle` at all: the live sensor is a forward-facing camera, so a
    scene vehicle placed behind the ego has no representation here -- not an
    approximation of one, an absence, which is `follower_gap` and its
    siblings' actual mechanism (`structurally_absent`, not merely unequal).
    """
    cfg = _production_builder_config()
    builder = ObservationBuilder(cfg)
    forward = [v for v in scene.vehicles if v.longitudinal_offset_m > 0]
    vehicles = [
        TrackedVehicle(
            track_id=i, xyxy=np.zeros(4, dtype=np.float32), cls=2, conf=0.9,
            distance_m=v.longitudinal_offset_m, lateral_m=v.lane_offset * scene.lane_width_m,
            rel_speed_mps=v.speed_mps - scene.ego_speed_mps, rel_speed_valid=True,
            method="ground_plane",
        )
        for i, v in enumerate(forward)
    ]
    t_mono = 1000.0
    result = None
    for tick in range(scene.priming_ticks):
        gps = GpsFix(valid=True, lat=0.0, lon=0.0, speed_mps=scene.ego_speed_mps,
                     heading_deg=0.0, fix_quality=1, num_sats=8, hdop=1.0,
                     t_mono=t_mono, t_wall=t_mono)
        result = builder.build(vehicles, gps, t_mono)
        t_mono += 0.1
    assert result is not None
    return result


# -- Encoding, through the SAME encoder on both sides ------------------------


def encode(obs: Mapping[str, Any]) -> np.ndarray:
    return sim_contract.encode_local_observation(obs)


@dataclass(frozen=True)
class SlotDiff:
    slot: str
    sim_value: float
    live_value: float
    equal: bool


def diff_scene(scene: Scene) -> dict[str, SlotDiff]:
    """One scene's per-slot equal/unequal result, both raw obs dicts encoded
    through the identical encoder (D20's second half)."""
    sim_obs = sim_observation(scene)
    live_result = live_observation(scene)
    sim_encoded = encode(sim_obs)
    live_encoded = live_result.encoded
    names = sim_contract.encoded_slot_names()
    return {
        name: SlotDiff(
            slot=name, sim_value=float(sim_encoded[i]), live_value=float(live_encoded[i]),
            equal=bool(np.isclose(sim_encoded[i], live_encoded[i], atol=ATOL, rtol=0.0)),
        )
        for i, name in enumerate(names)
    }


# -- The ledger: mechanism, per slot -----------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    slot: str
    cls: str
    mechanism: str
    #: For SUBSTITUTED/STRUCTURALLY_ABSENT slots only: the live value every
    #: scene must show (47.9's per-tick check on a real drive; here, the
    #: per-scene check that the constant is really constant).
    live_constant: float | None = None


#: One row per slot in `sim_contract.encoded_slot_names()` order (D13),
#: naming the CLASS and the MECHANISM. This is a fact about the two pieces
#: of code (`src/sensing/local.py`, `perception/observation_builder.py`),
#: cited by line in `test_observation_parity.py`'s own docstrings rather
#: than restated here -- the classification is checked against the ACTUAL
#: per-scene diff and against `field_sources`, not asserted on its own say-so.
LEDGER: dict[str, LedgerEntry] = {
    "is_active": LedgerEntry("is_active", CLASS_IDENTICAL, "both always True for an active agent"),
    "ego_speed": LedgerEntry("ego_speed", CLASS_IDENTICAL, "both read the ego's own current speed"),
    "ego_acceleration": LedgerEntry(
        "ego_acceleration", CLASS_IDENTICAL,
        "sim reads VehicleSnapshot.acceleration_mps2 directly; live derives a "
        "slope from a GPS speed window -- different mechanisms, equal on a "
        "primed, constant-speed scene where the true acceleration is zero "
        "either way",
    ),
    "ego_lane": LedgerEntry(
        "ego_lane", CLASS_SUBSTITUTED,
        "sim: int(ego.lane_id), a real lane index. live: int(cfg.assumed_lane), "
        "a config constant (1) -- no lane estimation exists",
        live_constant=1.0 / sim_contract.FIELD_SCALES["ego_lane"],
    ),
    "ego_headway_s": LedgerEntry("ego_headway_s", CLASS_IDENTICAL, "both: leader_gap / ego_speed, same formula"),
    "target_headway_s": LedgerEntry("target_headway_s", CLASS_IDENTICAL, "both hold the last commanded headway"),
    "time_since_last_lane_change": LedgerEntry(
        "time_since_last_lane_change", CLASS_SUBSTITUTED,
        "sim: finite once a lane change has happened. live: INF unconditionally "
        "-- no lane-change detection exists",
        live_constant=200.0 / sim_contract.FIELD_SCALES["time_since_last_lane_change"],
    ),
    "lane_changes_last_km": LedgerEntry(
        "lane_changes_last_km", CLASS_SUBSTITUTED,
        "sim: SafetyState.lane_changes_last_km, a real count. live: 0 "
        "unconditionally -- no lane-change detection exists",
        live_constant=0.0,
    ),
    "distance_to_next_merge": LedgerEntry(
        "distance_to_next_merge", CLASS_IDENTICAL,
        "sim hardcodes 0.0 (map matching not implemented there either); live "
        "matches it for parity, both sim_parity in mechanism and value",
    ),
    "distance_to_downstream_bottleneck": LedgerEntry(
        "distance_to_downstream_bottleneck", CLASS_SUBSTITUTED,
        "sim: 0.0 at a bottleneck segment, else inf, from real topology. "
        "live: INF unconditionally -- no map matching exists",
        live_constant=200.0 / sim_contract.FIELD_SCALES["distance_to_downstream_bottleneck"],
    ),
    "leader_gap": LedgerEntry("leader_gap", CLASS_IDENTICAL, "both: nearest same-lane vehicle ahead, by distance"),
    "leader_relative_speed": LedgerEntry(
        "leader_relative_speed", CLASS_IDENTICAL,
        "both: the nearest same-lane neighbour ahead's speed minus the "
        "ego's; live substitutes a real tracker's own estimate (falling "
        "back to 0.0 only until its window validates), which this harness "
        "does not model -- it feeds a `TrackedVehicle` with `rel_speed_mps` "
        "already computed",
    ),
    "follower_gap": LedgerEntry(
        "follower_gap", CLASS_STRUCTURALLY_ABSENT,
        "no rear sensor: the live vehicle list is forward-camera detections "
        "only, so a rear neighbour has no representation to measure at all",
        live_constant=200.0 / sim_contract.FIELD_SCALES["follower_gap"],
    ),
    "follower_relative_speed": LedgerEntry(
        "follower_relative_speed", CLASS_STRUCTURALLY_ABSENT, "same mechanism as follower_gap",
        live_constant=0.0,
    ),
    "left_lane_front_gap": LedgerEntry(
        "left_lane_front_gap", CLASS_APPROXIMATED,
        "sim: inf when the left lane does not exist in the real road network, "
        "else the nearest neighbour there. live: a measured gap for ANY "
        "detection at a left-of-ego lateral offset, whether or not that lane "
        "structurally exists -- a different domain of validity for the same "
        "measured quantity when it does exist",
    ),
    "left_lane_rear_gap": LedgerEntry(
        "left_lane_rear_gap", CLASS_STRUCTURALLY_ABSENT, "same mechanism as follower_gap",
        live_constant=200.0 / sim_contract.FIELD_SCALES["left_lane_rear_gap"],
    ),
    "right_lane_front_gap": LedgerEntry(
        "right_lane_front_gap", CLASS_APPROXIMATED, "same mechanism as left_lane_front_gap, mirrored",
    ),
    "right_lane_rear_gap": LedgerEntry(
        "right_lane_rear_gap", CLASS_STRUCTURALLY_ABSENT, "same mechanism as follower_gap",
        live_constant=200.0 / sim_contract.FIELD_SCALES["right_lane_rear_gap"],
    ),
    "target_lane_front_gap": LedgerEntry(
        "target_lane_front_gap", CLASS_APPROXIMATED,
        "sim: recomputed from the actual target lane's own occupants. live: "
        "aliased to leader_gap (the target lane always defaults to the "
        "current one) -- an approximation, exact only when the target lane "
        "is the ego's own",
    ),
    "target_lane_rear_gap": LedgerEntry(
        "target_lane_rear_gap", CLASS_STRUCTURALLY_ABSENT, "same mechanism as follower_gap",
        live_constant=200.0 / sim_contract.FIELD_SCALES["target_lane_rear_gap"],
    ),
    "target_lane_rear_required_decel": LedgerEntry(
        "target_lane_rear_required_decel", CLASS_STRUCTURALLY_ABSENT,
        "computed from a rear neighbour's closing speed, which live cannot see",
        live_constant=0.0,
    ),
    "downstream_congestion_estimate": LedgerEntry(
        "downstream_congestion_estimate", CLASS_APPROXIMATED,
        "both: 0.0 with no cooperating AV neighbours, else a function of "
        "them -- but sim's population is every locally-sensed role=='av' "
        "neighbour and live's is V2V peers only, a channel the camera "
        "cannot populate: a locally-tracked AV with no beacon is invisible "
        "to this field on live and not on sim",
    ),
    "merge_pressure": LedgerEntry(
        "merge_pressure", CLASS_APPROXIMATED, "same mechanism as downstream_congestion_estimate",
    ),
    "segment_target_speed": LedgerEntry(
        "segment_target_speed", CLASS_APPROXIMATED, "same mechanism as downstream_congestion_estimate",
    ),
    "uncongested_low_speed_flag": LedgerEntry(
        "uncongested_low_speed_flag", CLASS_APPROXIMATED,
        "same formula and threshold (src/safety/etiquette.py mirrored), "
        "different DENSITY input: sim reads the segment's own density from "
        "segment_metrics, live reads the locally sensed +-range density -- "
        "the same threshold applied to two different quantities",
    ),
    "local_density_bin": LedgerEntry(
        "local_density_bin", CLASS_APPROXIMATED,
        "same bin edges, different population and denominator: sim counts "
        "every neighbour within range_m (both directions) over 2*range_m; "
        "live counts forward-camera detections within effective_range_m, "
        "DOUBLED, over 2*effective_range_m -- range_m and effective_range_m "
        "are different config values (150 m vs 80 m by default)",
    ),
    "local_mean_speed_bin": LedgerEntry(
        "local_mean_speed_bin", CLASS_APPROXIMATED,
        "sim: mean over every measured neighbour. live: mean over only "
        "tracks with a VALID relative speed, falling back to ego speed when "
        "none validate -- a different subset of the same population",
    ),
    "local_queue_estimate": LedgerEntry(
        "local_queue_estimate", CLASS_APPROXIMATED,
        "same threshold (queue_speed_mps), same population difference as "
        "local_density_bin (forward-only, doubled, on live)",
    ),
    "active_vehicle_count_local": LedgerEntry(
        "active_vehicle_count_local", CLASS_APPROXIMATED,
        "same population difference as local_density_bin",
    ),
    "active_av_count_local": LedgerEntry(
        "active_av_count_local", CLASS_APPROXIMATED,
        "sim: count of local role=='av' neighbours, from the same sensing "
        "model as every other local count. live: aliased to nearby_av_count "
        "(V2V peers only) -- the camera cannot tell an AV from a human at "
        "all (TrackedVehicle carries no such field), so a locally-tracked "
        "AV with no beacon is invisible to this field",
    ),
    "nearby_av_count": LedgerEntry(
        "nearby_av_count", CLASS_APPROXIMATED, "same mechanism as active_av_count_local",
    ),
    "nearby_av_density": LedgerEntry(
        "nearby_av_density", CLASS_APPROXIMATED,
        "same population difference as active_av_count_local, and ALSO a "
        "different denominator when peers do exist: sim divides by range_m "
        "(150 m default), live by peer_range_m (also 150 m by default, but "
        "a SEPARATE config value from effective_range_m -- the two coincide "
        "only because nobody has changed either default yet)",
    ),
    "nearby_av_mean_speed": LedgerEntry(
        "nearby_av_mean_speed", CLASS_APPROXIMATED, "same population difference as active_av_count_local",
    ),
    "cooperation.segment_target_speed": LedgerEntry(
        "cooperation.segment_target_speed", CLASS_APPROXIMATED, "same value as the flat field, nested",
    ),
    "cooperation.merge_pressure": LedgerEntry(
        "cooperation.merge_pressure", CLASS_APPROXIMATED, "same value as the flat field, nested",
    ),
    "cooperation.downstream_congestion_estimate": LedgerEntry(
        "cooperation.downstream_congestion_estimate", CLASS_APPROXIMATED, "same value as the flat field, nested",
    ),
    "nearby_av_lane_distribution.0": LedgerEntry(
        "nearby_av_lane_distribution.0", CLASS_SUBSTITUTED,
        "sim: each cooperating neighbour's real absolute lane_id. live: each "
        "V2V peer's self-reported assumed_lane, which is the SAME constant "
        "(1, from BuilderConfig.assumed_lane) every peer sends -- a "
        "substituted constant, not a measurement, whenever any peer exists; "
        "this scene set carries no peers, so both read empty/neutral",
    ),
    "nearby_av_lane_distribution.1": LedgerEntry(
        "nearby_av_lane_distribution.1", CLASS_SUBSTITUTED, "same mechanism as lane 0",
    ),
    "nearby_av_lane_distribution.2": LedgerEntry(
        "nearby_av_lane_distribution.2", CLASS_SUBSTITUTED, "same mechanism as lane 0",
    ),
}

assert set(LEDGER) == set(sim_contract.encoded_slot_names()), (
    "LEDGER must classify every slot sim_contract.encoded_slot_names() lists, "
    "and no others"
)


def build_ledger(scenes: tuple[Scene, ...] = SCENES) -> dict[str, Any]:
    """Run every scene through both producers and assemble the ledger.

    Returns a JSON-able dict: per-slot classification, mechanism, the
    per-scene diffs, and whether the classification actually held on every
    scene it was checked against.
    """
    per_scene: dict[str, dict[str, SlotDiff]] = {}
    per_scene_field_sources: dict[str, dict[str, str | None]] = {}
    for scene in scenes:
        per_scene[scene.name] = diff_scene(scene)
        # field_sources needs the live ObservationResult, which diff_scene
        # does not return (it exists for the encoded comparison alone), so
        # live_observation runs a second time here.
        live_result = live_observation(scene)
        per_scene_field_sources[scene.name] = dict(live_result.field_sources)

    rows = []
    for slot in sim_contract.encoded_slot_names():
        entry = LEDGER[slot]
        scene_results = {name: diffs[slot] for name, diffs in per_scene.items()}
        always_equal = all(d.equal for d in scene_results.values())
        # Every scene this slot's SUBSTITUTED/STRUCTURALLY_ABSENT constant
        # was checked against, and whether the live side actually held it.
        constant_holds = None
        if entry.live_constant is not None:
            constant_holds = all(
                abs(diffs[slot].live_value - entry.live_constant) <= ATOL
                for diffs in per_scene.values()
            )
        provenance_classes = {
            name: per_scene_field_sources[name].get(slot) for name in per_scene_field_sources
        }
        provenance_ok = True
        if entry.cls in (CLASS_SUBSTITUTED, CLASS_STRUCTURALLY_ABSENT):
            provenance_ok = all(
                cls in SUBSTITUTED for cls in provenance_classes.values() if cls is not None
            )
        rows.append({
            "slot": slot,
            "class": entry.cls,
            "mechanism": entry.mechanism,
            "always_equal_across_scenes": always_equal,
            "matches_claimed_class": (
                always_equal if entry.cls == CLASS_IDENTICAL else not always_equal
            ),
            "live_constant": entry.live_constant,
            "constant_holds_across_scenes": constant_holds,
            "field_sources_by_scene": provenance_classes,
            "provenance_in_substituted_partition": provenance_ok,
            "per_scene": {
                name: {"sim": diffs[slot].sim_value, "live": diffs[slot].live_value,
                       "equal": diffs[slot].equal}
                for name, diffs in per_scene.items()
            },
        })
    return {
        "atol": ATOL,
        "scenes": [s.name for s in scenes],
        "slots": rows,
    }


# -- 47.8/47.9: checked against a real drive's own logged ticks -------------


def check_run_against_ledger(
    run_dir: Path, ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Two checks over one run's `metadata.jsonl`, neither of which reads
    `field_sources` alone -- both check the ENCODED vector a real tick
    actually carried.

    47.8: `encoded` is logged once, at capture time
    (`pipeline.Tick.to_record()`); nothing in the tree re-derives it and
    compares. A logged vector that silently drifted from what re-encoding
    the logged `obs` produces would mean the policy saw something this
    record does not describe.

    47.9: every SUBSTITUTED/STRUCTURALLY_ABSENT slot the ledger names a
    `live_constant` for must equal it on every tick, not merely in the
    scenes this module built -- a real drive is the harness's premise
    checked against reality rather than against itself.
    """
    ledger = ledger or build_ledger()
    slot_names = sim_contract.encoded_slot_names()
    constants = {
        row["slot"]: row["live_constant"] for row in ledger["slots"]
        if row["live_constant"] is not None
    }
    reencode_mismatches: list[Any] = []
    constant_mismatches: dict[str, list[Any]] = {}
    ticks_checked = 0
    with open(run_dir / "metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "tick":
                continue
            ticks_checked += 1
            logged = record["encoded"]
            reencoded = [round(float(x), 5) for x in encode(record["obs"])]
            if logged != reencoded:
                reencode_mismatches.append(record.get("tick_id"))
            for slot, constant in constants.items():
                index = slot_names.index(slot)
                if abs(logged[index] - constant) > ATOL:
                    constant_mismatches.setdefault(slot, []).append(record.get("tick_id"))
    return {
        "run": str(run_dir),
        "ticks_checked": ticks_checked,
        "reencode_mismatches": reencode_mismatches,
        "constant_mismatches": constant_mismatches,
        "ok": ticks_checked > 0 and not reencode_mismatches and not constant_mismatches,
    }
