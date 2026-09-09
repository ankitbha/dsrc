"""The simulator presenting the sensing model the rig actually has.

`src/analysis/observation_parity.py` classifies all 39 observation slots. Six the
rig cannot produce at all -- every rear-facing field, because the live vehicle list
is forward-camera derived and there is no rear sensor -- and seven it replaces with
a constant. So an actor trained on the full vector learns from 13 of 39 inputs the
deployed vehicle either cannot sense or fills in, which is a third of its input and
the largest single reason a policy would not transfer.

Values are taken from `deployment/jetson/perception/observation_builder.py` rather
than invented, so a change on the rig shows up here as a failure.

The flag defaults off. Turning it on for every run would silently change every
existing measurement, and the analysis harnesses in section C need the full vector
to ablate against -- the flag IS the ablation knob. `mappo_deploysense.yaml` sets
it, because that config exists to describe the deployed profile.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config.loaders import load_named_config
from src.road.topology_factory import build_topology
from src.safety import SafetyConstraints, SafetyState
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot

INF = float("inf")

#: What the rig puts in each field it cannot sense. Mirrors
#: `observation_builder.py` lines 421-440.
RIG_VALUES = {
    "ego_lane": 1,
    "time_since_last_lane_change": INF,
    "lane_changes_last_km": 0,
    "distance_to_downstream_bottleneck": INF,
    "follower_gap": INF,
    "follower_relative_speed": 0.0,
    "left_lane_rear_gap": INF,
    "right_lane_rear_gap": INF,
    "target_lane_rear_gap": INF,
    "target_lane_rear_required_decel": 0.0,
}


def _tree():
    return build_topology("inverted_tree", load_named_config("topology", "inverted_tree"))


def _snapshot(topology, vehicle_id, lane_index, longitudinal_m, *, speed_mps=20.0, role="human"):
    lane = topology.road_network.get_lane(lane_index)
    x, y = lane.position(longitudinal_m, 0)
    return VehicleSnapshot(
        vehicle_id=vehicle_id, role=role, segment_id=topology.segment_for_lane(lane_index),
        lane_index=lane_index, lane_id=lane_index[2], position=(float(x), float(y)),
        longitudinal_m=float(longitudinal_m), speed_mps=speed_mps, acceleration_mps2=0.0,
        free_flow_speed_mps=30.0,
    )


def _observe(topology, snapshots, *, fidelity):
    builder = LocalObservationBuilder(SensingConfig(deployed_fidelity=fidelity))
    return builder.build_all(
        time_s=0.0, topology=topology, snapshots=snapshots, current_av_ids=["av_0"],
        safety_states={"av_0": SafetyState()}, target_headways={"av_0": 1.6},
        target_lanes={"av_0": None}, segment_metrics={},
        constraints=SafetyConstraints(), rng=np.random.RandomState(7),
    )["av_0"]


@pytest.fixture
def scene():
    topology = _tree()
    ego = _snapshot(topology, "av_0", ("a5_entry", "b2", 0), 300.0, role="av")
    behind = _snapshot(topology, "h_behind", ("a5_entry", "b2", 0), 260.0)
    ahead = _snapshot(topology, "h_ahead", ("a5_entry", "b2", 0), 340.0, speed_mps=10.0)
    return topology, [ego, behind, ahead]


class TestFidelityModeMatchesTheRig:

    @pytest.mark.parametrize("field", sorted(RIG_VALUES))
    def test_each_unsensable_field_reads_as_the_rig_reads_it(self, scene, field):
        topology, snapshots = scene
        observation = _observe(topology, snapshots, fidelity=True)
        assert observation[field] == RIG_VALUES[field], (
            f"{field} is {observation[field]}, the rig produces {RIG_VALUES[field]}"
        )

    def test_the_rear_neighbour_is_genuinely_hidden(self, scene):
        # The control that matters: there IS a vehicle 40 m behind, and the full
        # vector reports it. If fidelity mode reported inf only because the scene
        # were empty, this test would pass for the wrong reason.
        topology, snapshots = scene
        full = _observe(topology, snapshots, fidelity=False)
        assert full["follower_gap"] == pytest.approx(35.0, abs=6.0), (
            f"the scene has no rear neighbour to hide; follower_gap is {full['follower_gap']}"
        )
        assert math.isinf(_observe(topology, snapshots, fidelity=True)["follower_gap"])

    def test_what_the_rig_CAN_sense_is_untouched(self, scene):
        # Forward camera quantities must be identical under both settings, or the
        # flag is removing more than the rig lacks.
        topology, snapshots = scene
        full = _observe(topology, snapshots, fidelity=False)
        rig = _observe(topology, snapshots, fidelity=True)
        for field in ("ego_speed", "ego_acceleration", "leader_gap",
                      "leader_relative_speed", "ego_headway_s"):
            assert rig[field] == full[field], f"{field} changed: {full[field]} -> {rig[field]}"

    def test_the_flag_defaults_off(self):
        assert SensingConfig().deployed_fidelity is False
