"""Validator round 2, R2-1 and R2-1b: the two network_fingerprint guards
`test_dsrc_network_identity.py` -- self-contained, so it collides with
nothing else on this branch.

Per `feedback_run_your_guard_against_a_control` and
`feedback_computed_printed_never_asserted`: a guard that has only ever
been checked by hand is not yet a guard. Each class below constructs the
exact broken wiring its guard exists to catch and asserts the refusal,
alongside the matching positive case that must NOT raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from perception.segment_state import SegmentStateBuilder
from pipeline import PerceptionPolicyPipeline
from policy import export_dsrc_policy as export_mod
from policy.advisory import SegmentAdvisoryDecoder
from policy.dsrc_runtime import DsrcRuntime


def tiny_definition(network_id: str = "test_net_identity") -> dict:
    """A 2-segment, 5-feature definition -- small and fast, not Mainz.

    `network_id` is the one field two calls of this function need to
    differ on to produce two definitions with different
    `network_fingerprint` values (R2-1b's "three different networks"
    case); everything else about the shape can stay identical.
    """
    return {
        "network_id": network_id,
        "feature_names": ["speed", "free_flow", "jam_factor", "lanes", "length_km"],
        "action_fractions": [0.5, 0.75, 1.0],
        "segments": [
            {"segment_id": "0", "edge_ids": ["a", "b"], "lanes": 2.0,
             "speed_limit_kmh": 60.0, "length_km": 1.0, "edges": []},
            {"segment_id": "1", "edge_ids": ["c"], "lanes": 1.0,
             "speed_limit_kmh": 45.0, "length_km": 2.0, "edges": []},
        ],
    }


def export_bundle(tmp_path: Path, definition: dict, name: str) -> tuple[str, str]:
    """Write `definition` and a matching random-weight bundle under `name`,
    both inside `tmp_path` -- flat, not nested, so no directory needs
    creating beyond what `export_dsrc_policy.export` already makes."""
    def_path = tmp_path / f"{name}_network.json"
    def_path.write_text(json.dumps(definition))
    model, info = export_mod.build_random(definition, seed=0)
    out_prefix = str(tmp_path / f"{name}_bundle")
    export_mod.export(model, info, definition, out_prefix)
    return out_prefix, str(def_path)


def real_matched_parts(tmp_path: Path, name: str = "matched"):
    """A DsrcRuntime, SegmentStateBuilder and SegmentAdvisoryDecoder built
    from the SAME network definition, each checking its own
    network_fingerprint against the runtime's -- the wiring R2-1b's
    positive case (and every other test file's real usage) exercises."""
    definition = tiny_definition(f"{name}_network")
    prefix, def_path = export_bundle(tmp_path, definition, name)
    runtime = DsrcRuntime(prefix, network_definition_path=def_path)
    builder = SegmentStateBuilder(
        def_path, expected_network_fingerprint=runtime.network_fingerprint,
    )
    decoder = SegmentAdvisoryDecoder.from_network_definition(
        def_path, expected_network_fingerprint=runtime.network_fingerprint,
    )
    return runtime, builder, decoder


def pipeline_stub_args() -> tuple:
    """The six required positional arguments PerceptionPolicyPipeline.__init__
    stores on self but never calls a method on -- only step() would. Mocks
    are enough to reach and exercise __init__'s dsrc-wiring checks."""
    return tuple(Mock() for _ in range(6))


class TestSegmentAdvisoryDecoderFingerprintGuard:
    """R2-1: SegmentAdvisoryDecoder's raw constructor cannot compute its own
    network_fingerprint (unlike from_network_definition, or
    SegmentStateBuilder, which always can), so asking it to check against
    an expectation without one must refuse rather than silently compare
    the expectation against None and pass."""

    def test_raw_constructor_with_expectation_raises(self):
        with pytest.raises(ValueError, match="from_network_definition"):
            SegmentAdvisoryDecoder(
                ("0", "1"), (60.0, 45.0), ((), ()),
                expected_network_fingerprint="0" * 16,
            )

    def test_raw_constructor_with_no_expectation_still_constructs(self):
        """The negative case: no check was asked for, so none is required."""
        decoder = SegmentAdvisoryDecoder(("0", "1"), (60.0, 45.0), ((), ()))
        assert decoder.network_fingerprint is None


class TestPipelineNetworkIdentityGuard:
    """R2-1b: PerceptionPolicyPipeline.__init__ is the one place holding
    dsrc_runtime, dsrc_segment_builder and dsrc_advisory_decoder together,
    and is where a three-parts-disagree wiring mistake must be caught --
    not left to whichever caller remembered to pass
    expected_network_fingerprint into each part individually.
    """

    def test_raw_constructed_decoder_is_refused(self, tmp_path):
        """A decoder built by SegmentAdvisoryDecoder's raw constructor
        carries network_fingerprint=None (R2-1 does not fire here because
        no expected_network_fingerprint was passed to the decoder itself
        -- this is exactly the gap R2-1b closes at the pipeline boundary
        instead)."""
        runtime, builder, _ = real_matched_parts(tmp_path)
        raw_decoder = SegmentAdvisoryDecoder(("0", "1"), (60.0, 45.0), ((), ()))
        assert raw_decoder.network_fingerprint is None

        with pytest.raises(ValueError, match="network_fingerprint values disagree"):
            PerceptionPolicyPipeline(
                *pipeline_stub_args(),
                dsrc_runtime=runtime,
                dsrc_segment_builder=builder,
                dsrc_advisory_decoder=raw_decoder,
            )

    def test_three_different_network_definitions_are_refused(self, tmp_path):
        """Each of the three parts is individually well-formed -- built
        from a complete definition and checking nothing against the
        others -- but the three definitions are not the same network."""
        runtime_a, _, _ = real_matched_parts(tmp_path, name="network_a")
        _, builder_b, _ = real_matched_parts(tmp_path, name="network_b")
        _, _, decoder_c = real_matched_parts(tmp_path, name="network_c")

        assert len({runtime_a.network_fingerprint, builder_b.network_fingerprint,
                    decoder_c.network_fingerprint}) == 3, (
            "test setup bug: the three parts should carry three distinct "
            "network_fingerprint values, not fewer"
        )

        with pytest.raises(ValueError, match="network_fingerprint values disagree"):
            PerceptionPolicyPipeline(
                *pipeline_stub_args(),
                dsrc_runtime=runtime_a,
                dsrc_segment_builder=builder_b,
                dsrc_advisory_decoder=decoder_c,
            )

    def test_three_matching_fingerprints_construct_fine(self, tmp_path):
        """The positive case: real parts, one shared network definition,
        no refusal. `len(set(fingerprints)) > 1` is false when all three
        agree, so construction must succeed exactly as it did before
        R2-1b existed.
        """
        runtime, builder, decoder = real_matched_parts(tmp_path)
        pipeline = PerceptionPolicyPipeline(
            *pipeline_stub_args(),
            dsrc_runtime=runtime,
            dsrc_segment_builder=builder,
            dsrc_advisory_decoder=decoder,
        )
        assert pipeline.dsrc_runtime is runtime

    def test_all_three_none_cannot_arise_through_real_construction(self, tmp_path):
        """`len(set(fingerprints)) > 1` is also false when all three
        fingerprints are None -- a real gap in the comparison, in
        principle. It is unreachable in practice: R2-1b's cross-check
        only runs `if dsrc_runtime is not None`, and DsrcRuntime.__init__
        always computes its own network_fingerprint from the network
        definition it loads (there is no code path that leaves it None on
        a constructed DsrcRuntime) -- confirmed here directly rather than
        argued.
        """
        runtime, _, _ = real_matched_parts(tmp_path)
        assert runtime.network_fingerprint is not None
