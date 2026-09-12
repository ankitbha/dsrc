"""policy/dsrc_runtime.py: loading a DSRC bundle and refusing a wrong one.

Per `feedback_run_your_guard_against_a_control`, every refusal below is
exercised against a bundle built to trigger it -- a guard that has only
ever passed is not a guard. The golden-action demonstration (Decision 4)
lives in `TestGoldenActions` at the bottom of this file, once
`specs/dsrc_golden_actions.json` exists.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from perception.segment_state import SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_SUBSTITUTED, SegmentStateResult
from policy import export_dsrc_policy as export_mod
from policy.dsrc_contract import SPEED_ACTION_FRACTIONS, SrcQNetwork
from policy.dsrc_runtime import DsrcRuntime, OUTCOME_OK


def tiny_definition() -> dict:
    """A 2-segment, 5-feature definition -- small and fast, not Mainz."""
    return {
        "network_id": "test_net",
        "feature_names": ["speed", "free_flow", "jam_factor", "lanes", "length_km"],
        "action_fractions": [0.5, 0.75, 1.0],
        "segments": [
            {"segment_id": "0", "edge_ids": ["a", "b"], "lanes": 2.0,
             "speed_limit_kmh": 60.0, "length_km": 1.0, "edges": []},
            {"segment_id": "1", "edge_ids": ["c"], "lanes": 1.0,
             "speed_limit_kmh": 45.0, "length_km": 2.0, "edges": []},
        ],
    }


def export_tiny_bundle(tmp_path: Path, definition: dict | None = None) -> str:
    definition = definition or tiny_definition()
    def_path = tmp_path / "network.json"
    def_path.write_text(json.dumps(definition))
    model, info = export_mod.build_random(definition, seed=0)
    out_prefix = str(tmp_path / "bundle")
    export_mod.export(model, info, definition, out_prefix)
    return out_prefix, str(def_path)


class TestLoadsAValidBundle:
    def test_dims_and_trained_flag(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        runtime = DsrcRuntime(prefix, network_definition_path=def_path)
        assert (runtime.num_segments, runtime.num_features, runtime.num_actions) == (2, 5, 3)
        assert runtime.is_trained is False

    def test_missing_module_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DsrcRuntime(str(tmp_path / "nowhere"), network_definition_path=(
                export_tiny_bundle(tmp_path)[1]
            ))

    def test_act_runs_and_returns_one_action_per_segment(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        runtime = DsrcRuntime(prefix, network_definition_path=def_path)
        state = np.zeros((2, 5), dtype=np.float32)
        result = runtime.act(state)
        assert result.actions.shape == (2,)
        assert result.q_values.shape == (2, 3)
        assert result.latency_ms >= 0.0


class TestManifestGuardsHaveBeenSeenToFail:
    """Each test below constructs the exact broken bundle the guard exists
    to catch, so a pass here means the guard actually fired -- not that it
    was never exercised."""

    def test_missing_network_fingerprint_is_refused_no_grandfather(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        manifest_path = Path(prefix + ".json")
        manifest = json.loads(manifest_path.read_text())
        del manifest["network_fingerprint"]
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="no network_fingerprint"):
            DsrcRuntime(prefix, network_definition_path=def_path)

    def test_wrong_network_fingerprint_is_refused(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        manifest_path = Path(prefix + ".json")
        manifest = json.loads(manifest_path.read_text())
        manifest["network_fingerprint"] = "0" * 16
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="network_fingerprint"):
            DsrcRuntime(prefix, network_definition_path=def_path)

    def test_two_segments_swapped_is_refused(self, tmp_path):
        """Control 1 of 3 (plan section 3, last paragraph): a reordered
        segment_ids changes nothing about the tensor shapes."""
        definition = tiny_definition()
        prefix, _ = export_tiny_bundle(tmp_path, definition)

        swapped = copy.deepcopy(definition)
        swapped["segments"] = [swapped["segments"][1], swapped["segments"][0]]
        swapped_path = tmp_path / "swapped_network.json"
        swapped_path.write_text(json.dumps(swapped))

        with pytest.raises(RuntimeError, match="network_fingerprint"):
            DsrcRuntime(prefix, network_definition_path=str(swapped_path))

    def test_a_changed_speed_limit_is_refused(self, tmp_path):
        """Control 2 of 3: a network whose limits moved under the same ids."""
        definition = tiny_definition()
        prefix, _ = export_tiny_bundle(tmp_path, definition)

        changed = copy.deepcopy(definition)
        changed["segments"][0]["speed_limit_kmh"] = 30.0
        changed_path = tmp_path / "changed_network.json"
        changed_path.write_text(json.dumps(changed))

        with pytest.raises(RuntimeError, match="network_fingerprint"):
            DsrcRuntime(prefix, network_definition_path=str(changed_path))

    def test_reordered_feature_names_is_refused(self, tmp_path):
        """Control 3 of 3: swapping two HERE_FEATURES entries leaves the
        input width at 5."""
        definition = tiny_definition()
        prefix, _ = export_tiny_bundle(tmp_path, definition)

        reordered = copy.deepcopy(definition)
        names = reordered["feature_names"]
        names[0], names[1] = names[1], names[0]
        reordered_path = tmp_path / "reordered_network.json"
        reordered_path.write_text(json.dumps(reordered))

        with pytest.raises(RuntimeError, match="network_fingerprint"):
            DsrcRuntime(prefix, network_definition_path=str(reordered_path))

    def test_changed_action_fractions_is_refused(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        manifest_path = Path(prefix + ".json")
        manifest = json.loads(manifest_path.read_text())
        manifest["action_fractions"] = [0.3, 0.6, 1.0]
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="action_fractions"):
            DsrcRuntime(prefix, network_definition_path=def_path)

    def test_mismatched_declared_dims_is_refused(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        manifest_path = Path(prefix + ".json")
        manifest = json.loads(manifest_path.read_text())
        manifest["num_segments"] = 99
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="num_segments"):
            DsrcRuntime(prefix, network_definition_path=def_path)


class TestDecideEnforcesFullCoverage:
    """Decision 3: an action only when every segment's basis is measured."""

    def test_full_coverage_runs_the_network(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        runtime = DsrcRuntime(prefix, network_definition_path=def_path)
        segment_state = SegmentStateResult(
            state=np.zeros((2, 5), dtype=np.float32),
            segment_basis=(SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_MEASURED),
            outcome="ok",
            matched_links=(1, 1),
            response_age_s=1.0,
        )
        decision = runtime.decide(segment_state)
        assert decision.outcome == OUTCOME_OK
        assert decision.actions is not None
        assert decision.actions.shape == (2,)
        assert decision.latency_ms is not None

    def test_one_substituted_segment_emits_no_action(self, tmp_path):
        prefix, def_path = export_tiny_bundle(tmp_path)
        runtime = DsrcRuntime(prefix, network_definition_path=def_path)
        segment_state = SegmentStateResult(
            state=None,
            segment_basis=(SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_SUBSTITUTED),
            outcome="incomplete_coverage",
            matched_links=(1, 0),
            response_age_s=1.0,
        )
        decision = runtime.decide(segment_state)
        assert decision.actions is None
        assert decision.q_values is None
        assert decision.latency_ms is None
        assert decision.outcome == "incomplete_coverage"
        assert decision.segment_basis == segment_state.segment_basis
