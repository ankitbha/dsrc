"""Guard the vendored contract against drift from the simulation source.

The encoder comparison runs anywhere torch is installed (src/rl/encoders.py
has no other dependencies). The action-constant comparison needs the env
stack (highway_env) and skips gracefully where it is absent - run the full
file on a dev machine after any sim contract change.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from policy import sim_contract

sim_encoders = pytest.importorskip("src.rl.encoders", reason="sim repo not importable")


FULL_OBS = {
    "is_active": True,
    "ego_speed": 23.4,
    "ego_acceleration": -1.2,
    "ego_lane": 1,
    "ego_headway_s": 1.9,
    "target_headway_s": 1.6,
    "time_since_last_lane_change": 42.0,
    "lane_changes_last_km": 1,
    "distance_to_next_merge": 0.0,
    "distance_to_downstream_bottleneck": float("inf"),
    "leader_gap": 44.5,
    "leader_relative_speed": -2.1,
    "follower_gap": float("inf"),
    "follower_relative_speed": 0.0,
    "left_lane_front_gap": 25.0,
    "left_lane_rear_gap": float("inf"),
    "right_lane_front_gap": float("inf"),
    "right_lane_rear_gap": float("inf"),
    "target_lane_front_gap": 44.5,
    "target_lane_rear_gap": float("inf"),
    "target_lane_rear_required_decel": 0.0,
    "downstream_congestion_estimate": 0.0,
    "merge_pressure": 0.0,
    "segment_target_speed": 30.0,
    "uncongested_low_speed_flag": False,
    "local_density_bin": 2,
    "local_mean_speed_bin": 1,
    "local_queue_estimate": 0,
    "active_vehicle_count_local": 6,
    "active_av_count_local": 0,
    "nearby_av_count": 0,
    "nearby_av_density": 0.0,
    "nearby_av_mean_speed": 30.0,
    "nearby_av_lane_distribution": {"0": 0.5, "1": 0.25, "2": 0.25},
    "cooperation": {
        "segment_target_speed": 30.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
    },
}

EDGE_CASES = [
    {},  # everything missing
    {field: None for field in sim_contract.LOCAL_OBS_FIELDS},
    FULL_OBS,
    {**FULL_OBS, "leader_gap": float("inf"), "ego_headway_s": float("inf")},
    {**FULL_OBS, "leader_relative_speed": float("-inf")},
    {**FULL_OBS, "ego_speed": float("nan")},
    {**FULL_OBS, "is_active": False, "uncongested_low_speed_flag": True},
    {**FULL_OBS, "ego_speed": "18.5"},          # numeric string parses
    {**FULL_OBS, "ego_speed": "not-a-number"},  # junk -> 0.0
    {**FULL_OBS, "cooperation": "garbage"},      # non-mapping ignored
    {**FULL_OBS, "nearby_av_lane_distribution": 7},
    {**FULL_OBS, "time_since_last_lane_change": float("inf")},
]


def test_field_lists_match_sim() -> None:
    assert sim_contract.LOCAL_OBS_FIELDS == sim_encoders.LOCAL_OBS_FIELDS
    assert sim_contract.COOPERATION_FIELDS == sim_encoders.COOPERATION_FIELDS
    assert sim_contract.LANE_DISTRIBUTION_LANES == sim_encoders.LANE_DISTRIBUTION_LANES
    assert sim_contract.local_obs_dim() == sim_encoders.local_obs_dim()


def test_field_scales_match_sim() -> None:
    for field, scale in sim_contract.FIELD_SCALES.items():
        assert sim_encoders.FIELD_SCALES[field] == scale, field


@pytest.mark.parametrize("case", range(len(EDGE_CASES)))
def test_encoding_matches_sim(case: int) -> None:
    obs = EDGE_CASES[case]
    ours = sim_contract.encode_local_observation(obs)
    sim = sim_encoders.encode_local_observation(obs).numpy()
    assert ours.shape == sim.shape == (sim_contract.local_obs_dim(),)
    np.testing.assert_allclose(ours, sim, rtol=0, atol=1e-6)


def test_action_constants_match_sim() -> None:
    actions_mod = pytest.importorskip(
        "src.rl.actions", reason="sim env stack (highway_env) not installed here"
    )
    assert sim_contract.ACTION_HEADS == actions_mod.ACTION_HEADS
    for head, values in sim_contract.ACTION_VALUES.items():
        assert values == actions_mod.ACTION_VALUES[head], head
    assert sim_contract.FORCED_ACTIONS == actions_mod.FORCED_ACTIONS


def test_decoders_match_sim_wrappers() -> None:
    wrappers = pytest.importorskip(
        "src.envs.wrappers", reason="sim env stack (highway_env) not installed here"
    )
    for headway_bin, seconds in sim_contract.HEADWAY_BIN_S.items():
        assert wrappers.decode_headway_bin(headway_bin) == seconds
    for speed_bin in sim_contract.SPEED_BIN_OFFSETS_MPS:
        for free_flow in (20.0, 30.0, 13.0):
            assert sim_contract.decode_speed_bin(speed_bin, free_flow) == wrappers.decode_speed_bin(
                speed_bin, free_flow
            )


def test_actor_state_dict_layout_matches_sim() -> None:
    models_mod = pytest.importorskip(
        "src.rl.models", reason="sim env stack (highway_env) not installed here"
    )
    from policy.export_policy import VendoredActor

    sim_actor = models_mod.MultiCategoricalActor(sim_contract.local_obs_dim())
    ours = VendoredActor(sim_contract.local_obs_dim())
    sim_keys = {k: tuple(v.shape) for k, v in sim_actor.state_dict().items()}
    our_keys = {k: tuple(v.shape) for k, v in ours.state_dict().items()}
    assert sim_keys == our_keys


def test_inf_encoding_known_values() -> None:
    """Pin the (non-obvious) inf behavior: inf -> 200 clamped to 5*scale, /scale."""
    obs = {"leader_gap": float("inf")}  # scale 150 -> 200/150
    encoded = sim_contract.encode_local_observation(obs)
    idx = sim_contract.LOCAL_OBS_FIELDS.index("leader_gap")
    assert math.isclose(encoded[idx], 200.0 / 150.0, rel_tol=1e-6)
    obs = {"ego_headway_s": float("inf")}  # scale 10 -> clamp at 50 -> 5.0
    encoded = sim_contract.encode_local_observation(obs)
    idx = sim_contract.LOCAL_OBS_FIELDS.index("ego_headway_s")
    assert math.isclose(encoded[idx], 5.0, rel_tol=1e-6)


class TestTheBundleGuardSeesMoreThanTheDimension:
    """A reorder leaves the dimension at 39 and puts every value in the wrong slot."""

    def test_the_fingerprint_moves_when_the_field_order_does(self):
        from policy import sim_contract as sc

        before = sc.contract_fingerprint()
        original = sc.LOCAL_OBS_FIELDS
        try:
            swapped = list(original)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            sc.LOCAL_OBS_FIELDS = tuple(swapped)
            assert len(sc.LOCAL_OBS_FIELDS) == len(original), "the dimension is unchanged"
            assert sc.contract_fingerprint() != before, (
                "two fields swapped and the guard cannot tell"
            )
        finally:
            sc.LOCAL_OBS_FIELDS = original
        assert sc.contract_fingerprint() == before

    def test_the_fingerprint_moves_when_a_scale_does(self):
        from policy import sim_contract as sc

        before = sc.contract_fingerprint()
        key = next(iter(sc.FIELD_SCALES))
        original = sc.FIELD_SCALES[key]
        try:
            sc.FIELD_SCALES[key] = original * 2.0
            assert sc.contract_fingerprint() != before
        finally:
            sc.FIELD_SCALES[key] = original
        assert sc.contract_fingerprint() == before

    def test_the_fingerprint_is_stable_across_calls(self):
        from policy import sim_contract as sc

        assert sc.contract_fingerprint() == sc.contract_fingerprint()
