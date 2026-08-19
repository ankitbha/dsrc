from __future__ import annotations

import numpy as np
import pytest

from src.analysis.observation_audit import DEFAULT_NEAR_CONSTANT_VARIANCE
from src.analysis import (
    SampleSpec,
    audit_fields,
    collect_encoded_samples,
    encoded_fallback_candidates,
    encoded_field_names,
    free_flow_speeds_for_topology,
)
from src.rl.encoders import encode_local_observation, local_obs_dim


# --------------------------------------------------------------------------
# unit tests: audit_fields against synthetic input with known answers
# --------------------------------------------------------------------------


def audit_one(column, **kwargs):
    return audit_fields(np.asarray(column, dtype=float).reshape(-1, 1), ["f"], **kwargs)[0]


def test_field_names_match_encoder_width_and_are_unique():
    names = encoded_field_names()
    assert len(names) == local_obs_dim() == 39
    assert len(set(names)) == len(names)


def test_exactly_constant_column_is_flagged_constant():
    result = audit_one([2.5] * 8)
    assert result.is_strictly_constant is True
    assert result.is_near_constant is True
    assert result.unique_values == 1
    assert result.variance == pytest.approx(0.0)
    assert result.constant_value == pytest.approx(2.5)
    assert result.n_samples == 8


def test_jitter_below_threshold_is_near_constant_but_not_strictly():
    # std 1e-3 -> variance 1e-6, well under the 1e-4 threshold
    column = [1.0 + 1e-3, 1.0 - 1e-3] * 10
    result = audit_one(column)
    assert result.is_strictly_constant is False
    assert result.is_near_constant is True
    assert result.constant_value is None


def test_jitter_above_threshold_is_neither():
    # std 0.5 -> variance 0.25, far above threshold
    result = audit_one([1.5, 0.5] * 10)
    assert result.is_strictly_constant is False
    assert result.is_near_constant is False
    assert result.variance > DEFAULT_NEAR_CONSTANT_VARIANCE


def test_column_pinned_at_fallback_is_flagged():
    fallbacks = encoded_fallback_candidates([25.0])
    pinned = fallbacks["nearby_av_count"][0]
    samples = np.full((6, 1), pinned)
    result = audit_fields(samples, ["nearby_av_count"], fallback_candidates=fallbacks)[0]
    assert result.never_left_fallback is True

    departed = samples.copy()
    departed[3, 0] = pinned + 1.0
    moved = audit_fields(departed, ["nearby_av_count"], fallback_candidates=fallbacks)[0]
    assert moved.never_left_fallback is False


def test_field_without_defined_fallback_reports_none():
    result = audit_one([1.0, 2.0], fallback_candidates=encoded_fallback_candidates([25.0]))
    assert result.never_left_fallback is None


def test_empty_samples_report_zero_coverage_not_constancy():
    result = audit_fields(np.empty((0, 1)), ["f"])[0]
    assert result.n_samples == 0
    assert result.unique_values == 0
    assert result.is_strictly_constant is False
    assert result.is_near_constant is False
    assert result.constant_value is None


def test_infinite_observation_values_encode_finite_and_audit_as_constant():
    observation = {"leader_gap": float("inf"), "ego_speed": 10.0}
    encoded = encode_local_observation(observation).numpy()
    assert np.all(np.isfinite(encoded))
    names = encoded_field_names()
    result = {a.field: a for a in audit_fields(np.stack([encoded, encoded]), names)}
    assert result["leader_gap"].is_strictly_constant is True
    assert np.isfinite(result["leader_gap"].constant_value)


def test_column_count_mismatch_raises():
    with pytest.raises(ValueError):
        audit_fields(np.zeros((4, 3)), ["a", "b"])


# --------------------------------------------------------------------------
# sanity tests: behavioural expectations against real rollouts
# --------------------------------------------------------------------------

SANITY_STEPS = 60


def audit_condition(topology: str, av_count: int, seed: int = 7):
    samples = collect_encoded_samples(
        SampleSpec(
            topology=topology,
            controller="random_av",
            seed=seed,
            duration_steps=SANITY_STEPS,
            controlled_vehicles=av_count,
        )
    )
    audits = audit_fields(
        samples,
        encoded_field_names(),
        fallback_candidates=encoded_fallback_candidates(free_flow_speeds_for_topology(topology)),
    )
    return {a.field: a for a in audits}


@pytest.fixture(scope="module")
def ring_audit():
    return audit_condition("ring", 8)


@pytest.fixture(scope="module")
def multilane_audit():
    return audit_condition("straight_multilane", 8)


@pytest.fixture(scope="module")
def merge_audit():
    return audit_condition("merge", 8)


@pytest.mark.parametrize("topology,av_count", [("ring", 2), ("ring", 8), ("straight_multilane", 8), ("merge", 8)])
def test_distance_to_next_merge_is_constant_everywhere(topology, av_count):
    # hardcoded to 0.0 in src/sensing/local.py, so it cannot carry information
    result = audit_condition(topology, av_count)["distance_to_next_merge"]
    assert result.is_strictly_constant is True
    assert result.constant_value == pytest.approx(0.0)


def test_lane_geometry_field_is_dead_on_ring_but_live_on_multilane(ring_audit, multilane_audit):
    # ring has no adjacent lane, so the gap stays at its empty-road value
    assert ring_audit["left_lane_front_gap"].is_strictly_constant is True
    assert multilane_audit["left_lane_front_gap"].is_strictly_constant is False


def test_nearby_av_lane_distribution_is_distinct_from_lane_geometry(multilane_audit):
    # describes nearby AVs, not lane structure: stays dead on multilane even
    # where the geometry field varies
    assert multilane_audit["nearby_av_lane_distribution.0"].is_strictly_constant is True


def test_downstream_bottleneck_is_dead_on_ring_but_live_on_merge(ring_audit, merge_audit):
    assert ring_audit["distance_to_downstream_bottleneck"].is_strictly_constant is True
    assert merge_audit["distance_to_downstream_bottleneck"].is_strictly_constant is False


@pytest.mark.parametrize("fixture_name", ["ring_audit", "multilane_audit", "merge_audit"])
def test_ego_speed_always_varies(fixture_name, request):
    result = request.getfixturevalue(fixture_name)["ego_speed"]
    assert result.is_strictly_constant is False
    assert result.is_near_constant is False


def test_no_av_condition_yields_no_samples():
    samples = collect_encoded_samples(
        SampleSpec(topology="ring", controller="no_av", seed=7, duration_steps=20)
    )
    assert samples.shape == (0, 39)


def test_wrong_width_is_rejected_not_silently_reshaped():
    # a (4, 3) array reshaped to 2 columns would become (6, 2) and misalign
    # every column against its field name
    with pytest.raises(ValueError):
        audit_fields(np.zeros((4, 3)), ["a", "b"])
    with pytest.raises(ValueError):
        audit_fields(np.zeros(7), ["a", "b"])
    with pytest.raises(ValueError):
        audit_fields(np.zeros((2, 2, 2)), ["a", "b"])
