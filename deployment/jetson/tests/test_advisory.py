from __future__ import annotations

import pytest

from policy.actor_runtime import PolicyOutput
from policy.advisory import MPS_TO_MPH, AdvisoryDecoder


def policy_out(speed_bin: str = "nominal", headway: str = "normal", lane: str = "keep",
               merge: str = "normal", confidence: float = 0.8) -> PolicyOutput:
    return PolicyOutput(
        action={
            "desired_speed_bin": speed_bin,
            "desired_headway_bin": headway,
            "lane_preference": lane,
            "merge_mode": merge,
        },
        head_probs={}, chosen_prob={}, confidence=confidence, latency_ms=0.1,
    )


def obs_with(target_speed: float = 30.0, ego_speed: float = 25.0, density_bin: int = 1) -> dict:
    return {
        "ego_speed": ego_speed,
        "local_density_bin": density_bin,
        "cooperation": {"segment_target_speed": target_speed},
    }


def test_nominal_decode_matches_sim_wrapper() -> None:
    adv = AdvisoryDecoder(units="mph").decode(policy_out("nominal"), obs_with(30.0))
    assert adv.recommended_speed_mps == pytest.approx(27.0)  # 30 - 3
    assert adv.recommended_speed_display == pytest.approx(27.0 * MPS_TO_MPH)


def test_slow_decode_floors_at_contextual_minimum() -> None:
    adv = AdvisoryDecoder().decode(policy_out("slow"), obs_with(target_speed=20.0))
    assert adv.recommended_speed_mps == 12.0  # max(12, 20 - 10)


def test_headway_and_texts() -> None:
    adv = AdvisoryDecoder().decode(
        policy_out(headway="largest", lane="prefer_left_if_safe", merge="create_gap"),
        obs_with(),
    )
    assert adv.headway_target_s == 3.0
    assert "left" in adv.lane_text.lower()
    assert "gap" in adv.merge_text.lower()
    assert adv.traffic_text == "Moderate"


@pytest.mark.parametrize(
    "confidence,label", [(0.30, "low"), (0.55, "medium"), (0.69, "medium"), (0.95, "high")]
)
def test_confidence_labels(confidence: float, label: str) -> None:
    adv = AdvisoryDecoder().decode(policy_out(confidence=confidence), obs_with())
    assert adv.confidence_label == label


def test_units_kmh() -> None:
    adv = AdvisoryDecoder(units="kmh").decode(policy_out(), obs_with(30.0))
    assert adv.recommended_speed_display == pytest.approx(27.0 * 3.6)


# ---------------------------------------------------------------------------
# SegmentAdvisory / SegmentAdvisoryDecoder (task 145, decision 6)
# ---------------------------------------------------------------------------

import numpy as np

from policy.advisory import SegmentAdvisory, SegmentAdvisoryDecoder, SegmentAdvisoryRow
from policy.dsrc_runtime import DsrcDecision


def straight_line(lat: float, lon: float, east_deg: float, points: int = 3) -> tuple:
    step = east_deg / max(points - 1, 1)
    return tuple((lat, lon + step * i) for i in range(points))


def make_decoder(units: str = "mph") -> SegmentAdvisoryDecoder:
    return SegmentAdvisoryDecoder(
        segment_ids=("0", "1"),
        segment_speed_limits_kmh=(60.0, 100.0),
        segment_points=(straight_line(40.0, -74.0, 0.01), straight_line(41.0, -75.0, 0.01)),
        units=units,
    )


def decision(actions) -> DsrcDecision:
    return DsrcDecision(
        outcome="ok",
        actions=np.array(actions, dtype=np.int64),
        q_values=np.zeros((len(actions), 3), dtype=np.float32),
        segment_basis=("measured",) * len(actions),
        matched_links=(1,) * len(actions),
        latency_ms=0.5,
    )


def no_action_decision(outcome: str = "incomplete_coverage") -> DsrcDecision:
    return DsrcDecision(
        outcome=outcome, actions=None, q_values=None,
        segment_basis=("measured", "substituted"), matched_links=(1, 0), latency_ms=None,
    )


class TestSegmentAdvisoryDecode:
    def test_one_row_per_segment_in_order(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([2, 0]))
        assert advisory.outcome == "ok"
        assert [row.segment_id for row in advisory.rows] == ["0", "1"]

    def test_recommended_speed_is_fraction_times_the_static_speed_limit(self):
        decoder = make_decoder(units="kmh")
        advisory = decoder.decode(decision([2, 0]))  # fraction 1.0, fraction 0.5
        assert advisory.rows[0].fraction == pytest.approx(1.0)
        assert advisory.rows[0].recommended_speed_display == pytest.approx(60.0)
        assert advisory.rows[1].fraction == pytest.approx(0.5)
        assert advisory.rows[1].recommended_speed_display == pytest.approx(50.0)

    def test_units_mps_is_the_undisplayed_value(self):
        decoder = make_decoder(units="mps")
        advisory = decoder.decode(decision([2, 2]))
        assert advisory.rows[0].recommended_speed_mps == pytest.approx(60.0 / 3.6)
        assert advisory.rows[0].recommended_speed_display == pytest.approx(
            advisory.rows[0].recommended_speed_mps
        )

    def test_no_action_yields_no_rows_and_propagates_the_outcome(self):
        decoder = make_decoder()
        advisory = decoder.decode(no_action_decision("incomplete_coverage"))
        assert advisory.rows == ()
        assert advisory.outcome == "incomplete_coverage"

    def test_unknown_units_are_refused(self):
        with pytest.raises(ValueError):
            SegmentAdvisoryDecoder(("0",), (60.0,), (straight_line(0, 0, 0.01),), units="furlongs")

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            SegmentAdvisoryDecoder(("0", "1"), (60.0,), (straight_line(0, 0, 0.01),))


class TestEgoSegment:
    def test_no_fix_gives_no_ego_segment(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([0, 0]))
        assert advisory.ego_segment is None

    def test_a_fix_near_segment_zero_matches_it(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([0, 0]), ego_lat=40.0, ego_lon=-74.0002)
        assert advisory.ego_segment == 0

    def test_a_fix_near_segment_one_matches_it(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([0, 0]), ego_lat=41.0, ego_lon=-75.0002)
        assert advisory.ego_segment == 1

    def test_a_fix_far_from_every_segment_matches_none(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([0, 0]), ego_lat=0.0, ego_lon=0.0)
        assert advisory.ego_segment is None

    def test_a_nonfinite_fix_matches_none(self):
        decoder = make_decoder()
        advisory = decoder.decode(decision([0, 0]), ego_lat=float("nan"), ego_lon=-74.0)
        assert advisory.ego_segment is None

    def test_ego_segment_is_reported_even_with_no_action(self):
        """The driver's own position is not evidence about coverage."""
        decoder = make_decoder()
        advisory = decoder.decode(no_action_decision(), ego_lat=40.0, ego_lon=-74.0002)
        assert advisory.ego_segment == 0
        assert advisory.rows == ()


class TestFromNetworkDefinition:
    def test_loads_the_real_mainz_network(self):
        decoder = SegmentAdvisoryDecoder.from_network_definition()
        assert len(decoder.segment_ids) == 12
        assert decoder.segment_ids == tuple(str(i) for i in range(12))
        assert all(limit == pytest.approx(60.0 / 3.6, abs=0.01)
                  for limit in decoder.segment_speed_limits_mps)


class TestNetworkFingerprintGuard:
    """Validator round 1, fix 2b: `SegmentAdvisoryDecoder.__init__` now takes
    the paired runtime's own `network_fingerprint` and refuses a mismatch,
    guarding the runtime/decoder pair the same way
    `SegmentStateBuilder.__init__` guards the runtime/builder pair."""

    def test_raw_construction_with_no_fingerprint_does_not_raise(self):
        """Backward compatible: `make_decoder` above passes neither
        fingerprint argument at all, the same as every existing caller."""
        make_decoder()

    def test_from_network_definition_exposes_its_own_fingerprint(self):
        decoder = SegmentAdvisoryDecoder.from_network_definition()
        assert decoder.network_fingerprint is not None
        assert len(decoder.network_fingerprint) == 16

    def test_matching_expected_fingerprint_does_not_raise(self):
        reference = SegmentAdvisoryDecoder.from_network_definition()
        SegmentAdvisoryDecoder.from_network_definition(
            expected_network_fingerprint=reference.network_fingerprint,
        )

    def test_wrong_expected_fingerprint_is_refused(self):
        with pytest.raises(RuntimeError, match="network_fingerprint"):
            SegmentAdvisoryDecoder.from_network_definition(
                expected_network_fingerprint="0" * 16,
            )
