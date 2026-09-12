"""policy/dsrc_runtime.py: loading a DSRC bundle and refusing a wrong one.

Per `feedback_run_your_guard_against_a_control`, every refusal below is
exercised against a bundle built to trigger it -- a guard that has only
ever passed is not a guard. The golden-action demonstration (Decision 4)
lives in `TestGoldenActions` at the bottom of this file, once
`specs/dsrc_golden_actions.json` exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from perception.segment_state import SEGMENT_BASIS_MEASURED, SEGMENT_BASIS_SUBSTITUTED, SegmentStateResult
from policy import export_dsrc_policy as export_mod
from policy.dsrc_contract import SPEED_ACTION_FRACTIONS, SrcQNetwork
from policy.dsrc_runtime import DsrcRuntime, OUTCOME_OK

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import export_dsrc_golden as export_mod_golden  # noqa: E402


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


# ---------------------------------------------------------------------------
# Decision 4's demonstration: action equality against src.rl.src_q.greedy_actions
# on specs/dsrc_golden_actions.json (scripts/export_dsrc_golden.py).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "specs" / "dsrc_golden_actions.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt"
NETWORK_DEFINITION_PATH = REPO_ROOT / "specs" / "dsrc_network_mainz.json"

requires_golden = pytest.mark.skipif(
    not GOLDEN_PATH.exists(), reason="specs/dsrc_golden_actions.json not generated"
)
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(), reason="results/checkpoints/mainz_here_best.pt not present"
)


def _from_hex_f32(nested) -> np.ndarray:
    """Inverse of scripts/export_dsrc_golden.py's _hex_f32: exact bits back."""
    arr = np.array(nested, dtype=object)
    flat = arr.ravel()
    values = np.array(
        [np.frombuffer(bytes.fromhex(h), dtype=np.float32)[0] for h in flat], dtype=np.float32
    )
    return values.reshape(arr.shape)


def _export_real_bundle(tmp_path, network_definition_path=NETWORK_DEFINITION_PATH) -> str:
    definition = json.loads(Path(network_definition_path).read_text())
    model, info = export_mod.build_from_checkpoint(str(CHECKPOINT_PATH), definition)
    out_prefix = str(Path(tmp_path) / "dsrc_policy")
    export_mod.export(model, info, definition, out_prefix, checkpoint_path=str(CHECKPOINT_PATH))
    return out_prefix


@requires_golden
@requires_checkpoint
class TestGoldenActions:
    """Decision 4, section 5.2's acceptance table, checked directly."""

    @pytest.fixture(scope="class")
    def golden(self):
        golden = json.loads(GOLDEN_PATH.read_text())
        # Neither population size was checked before: a regeneration with
        # e.g. --random-states 100 would pass every test below while the
        # file's own "random" block claims 20,000, and TEST_SEEDS producing
        # a different episode count would do the same for "recorded".
        assert len(golden["recorded"]) == 185, (
            f"{GOLDEN_PATH} claims {len(golden['recorded'])} recorded "
            "decisions, not 185 -- 185 is derived from scripts/export_dsrc_golden.py's "
            "TEST_SEEDS (5 episodes), DURATION_S and WARMUP_S (how many decisions fit in "
            "one episode) and policy.dsrc_contract.DECISION_INTERVAL_S (the decision "
            "cadence), not a constant on its own; regenerated against a different value "
            "of one of those four?"
        )
        assert golden["random"]["count"] == 20_000, (
            f"{GOLDEN_PATH} claims {golden['random']['count']} random "
            "states, not 20,000 -- regenerated with a different "
            "--random-states?"
        )
        return golden

    @pytest.fixture(scope="class")
    def runtime(self, tmp_path_factory, golden):
        bundle = _export_real_bundle(tmp_path_factory.mktemp("dsrc_golden_bundle"))
        rt = DsrcRuntime(bundle, network_definition_path=str(NETWORK_DEFINITION_PATH))
        assert rt.manifest["network_fingerprint"] == golden["network_fingerprint"], (
            "the golden file and the freshly-exported bundle disagree on which "
            "network they describe -- regenerate one against the other"
        )
        # golden["checkpoint_sha256"] was stored and never checked: a
        # divergence caused by a changed checkpoint would otherwise fail
        # test_recorded_actions_and_q_values as "the runtime diverged",
        # which names the wrong cause.
        checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
        assert golden["checkpoint_sha256"] == checkpoint_sha256, (
            f"{GOLDEN_PATH}'s checkpoint_sha256 does not match "
            f"{CHECKPOINT_PATH} -- the checkpoint changed since the golden "
            "file was generated; regenerate it against the current one "
            "before trusting a Q-value or action mismatch below as a "
            "DsrcRuntime defect"
        )
        return rt

    def test_recorded_actions_and_q_values(self, runtime, golden):
        """0 mismatches of ~180 recorded simulator states; Q-values within 1e-5."""
        mismatches = 0
        max_diff = 0.0
        for case in golden["recorded"]:
            state = _from_hex_f32(case["state_hex"])
            result = runtime.act(state)
            expected_action = np.array(case["action"], dtype=np.int64)
            if not np.array_equal(result.actions, expected_action):
                mismatches += 1
            expected_q = _from_hex_f32(case["q_values_hex"])
            max_diff = max(max_diff, float(np.max(np.abs(result.q_values - expected_q))))
        assert mismatches == 0, (
            f"{mismatches} of {len(golden['recorded'])} recorded states mismatched"
        )
        assert max_diff < 1e-5, f"max |Q-value diff| {max_diff} >= 1e-5"

    def test_recorded_q_values_are_bit_exact_on_this_machine(self, runtime, golden):
        """TorchScript vs the eager reference, same machine that exported both."""
        for case in golden["recorded"]:
            state = _from_hex_f32(case["state_hex"])
            result = runtime.act(state)
            expected_q = _from_hex_f32(case["q_values_hex"])
            assert np.array_equal(result.q_values, expected_q), (
                "TorchScript Q-values are not bit-exact with the eager reference "
                "on this machine"
            )

    def test_random_states_hash_to_the_frozen_digest(self, runtime, golden):
        """0 mismatches of 20,000 random states, checked by hash rather than
        by stored arrays: specs/dsrc_golden_actions.json stores only the
        seed, the feature-box bounds and the count for this population (see
        scripts/export_dsrc_golden.py's module docstring for why -- storing
        the arrays put the file at 10.8 MB, a bulk dump of a deterministic
        generator's output). The states are regenerated here from those four
        numbers, run through DsrcRuntime, and hashed the same way the
        generator hashed the reference's own output; a hash mismatch means
        DsrcRuntime's actions or Q-values diverged from the checkpoint's
        reference somewhere in the 20,000, without saying which -- the 185
        recorded states above already give a per-state accounting, and are
        where the "any machine, 1e-5" Q-value claim is actually checked with
        real numbers; this hash is a same-machine, bit-exact comparison,
        like network_fingerprint and checkpoint_sha256 elsewhere in this
        file.
        """
        random_block = golden["random"]
        n = random_block["count"]
        s, f = runtime.num_segments, runtime.num_features
        rng = np.random.default_rng(random_block["seed"])
        states = rng.uniform(
            random_block["feature_low"], random_block["feature_high"], size=(n, s, f)
        ).astype(np.float32)

        actions = np.zeros((n, s), dtype=np.int8)
        q_values = np.zeros((n, s, runtime.num_actions), dtype=np.float32)
        for i in range(n):
            result = runtime.act(states[i])
            actions[i] = result.actions.astype(np.int8)
            q_values[i] = result.q_values

        digest = export_mod_golden.hash_actions_q_values(actions, q_values)
        assert digest == random_block["sha256_actions_q_values"]


@requires_golden
@requires_checkpoint
class TestTheDemonstrationHasBeenSeenToFail:
    """Section 5.3: three deliberately broken runtimes, each seen to fail
    before the passing result above is trusted."""

    def test_control_1_a_transposed_weight_breaks_the_shape(self):
        """A transposed weight load -- must fail on Q-values (here, before
        any: none of this network's weight matrices is square, so a
        transpose is never shape-compatible and the corruption is caught at
        load time rather than surfacing as a silently wrong number)."""
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        corrupted = dict(state_dict)
        corrupted["stack.0.weight"] = corrupted["stack.0.weight"].T.contiguous()
        model = SrcQNetwork(12, 5, 3)
        with pytest.raises(RuntimeError):
            model.load_state_dict(corrupted)

    def test_control_2_two_segments_swapped_fails_at_load_before_any_action(self, tmp_path):
        """Two segments swapped in the network definition -- must fail on
        the network fingerprint at load, before any action is computed."""
        definition = json.loads(NETWORK_DEFINITION_PATH.read_text())
        bundle = _export_real_bundle(tmp_path, network_definition_path=NETWORK_DEFINITION_PATH)

        swapped = json.loads(NETWORK_DEFINITION_PATH.read_text())
        swapped["segments"][0], swapped["segments"][1] = (
            swapped["segments"][1], swapped["segments"][0],
        )
        swapped_path = tmp_path / "swapped_mainz.json"
        swapped_path.write_text(json.dumps(swapped))

        with pytest.raises(RuntimeError, match="network_fingerprint"):
            DsrcRuntime(bundle, network_definition_path=str(swapped_path))

    def test_control_3_jam_factor_from_an_independent_field_changes_actions(
        self, tmp_path_factory,
    ):
        """jam_factor computed from an independent field instead of the
        simulator's formula -- records how many of the ~185 recorded
        actions change. This is the measurement open item 2 needs, not a
        pass/fail gate: there is no real HERE jamFactor reading paired with
        these simulated states (no HERE response was ever collected over
        Mainz -- plan risk 4), so "an independent field" is modelled here as
        a value drawn independently of the recomputed formula, over HERE's
        full 0-10 range.

        This does NOT upper-bound how much a genuinely uncorrelated field
        could move the outcome (an earlier version of this docstring
        claimed that): a constant-0 substitution is equally uncorrelated
        with the recomputed formula and moves 174/185, more than the
        143/185 a uniform draw moves. What is reported below is the effect
        of this one particular substitution -- U(0, 10) -- on this
        checkpoint and these 185 states, nothing more general. Part of the
        143/185 is not decorrelation at all: the substitute also displaces
        the column's own location (the recorded jam_factor spans
        [0.000, 7.540] with mean 2.128; the draw is U(0, 10) with mean
        5.0), so this number conflates decorrelation with a location
        shift.

        The whole-network reading also redraws all 12 segments' jam_factor
        at once, so "at least one of 12 segments' action differed" is a
        much easier question to trigger than open item 2's own "how
        sensitive is one segment's own reading" -- redrawing only segment
        0's jam_factor and leaving the other 11 at their recorded values
        (the single-segment reading below) is what a per-segment answer
        to that item actually needs.
        """
        golden = json.loads(GOLDEN_PATH.read_text())
        bundle = _export_real_bundle(tmp_path_factory.mktemp("dsrc_control_3"))
        runtime = DsrcRuntime(bundle, network_definition_path=str(NETWORK_DEFINITION_PATH))

        rng = np.random.default_rng(2026)
        decisions_changed = 0
        per_segment_changed = 0
        per_segment_total = 0
        for case in golden["recorded"]:
            state = _from_hex_f32(case["state_hex"])
            original_action = runtime.act(state).actions

            altered = state.copy()
            altered[:, 2] = rng.uniform(0.0, 10.0, size=altered.shape[0]).astype(np.float32)
            altered_action = runtime.act(altered).actions

            per_segment_total += original_action.shape[0]
            per_segment_changed += int(np.sum(original_action != altered_action))
            if not np.array_equal(original_action, altered_action):
                decisions_changed += 1

        rng_single = np.random.default_rng(2026)
        single_segment_changed = 0
        for case in golden["recorded"]:
            state = _from_hex_f32(case["state_hex"])
            original_action = runtime.act(state).actions

            altered = state.copy()
            altered[0, 2] = rng_single.uniform(0.0, 10.0)
            altered_action = runtime.act(altered).actions

            if not np.array_equal(original_action, altered_action):
                single_segment_changed += 1

        n = len(golden["recorded"])
        print(
            f"\ncontrol 3: independent jam_factor changed "
            f"{decisions_changed}/{n} recorded decisions when redrawn for the "
            f"whole network at once (>=1 segment's action differed), "
            f"{per_segment_changed}/{per_segment_total} individual per-segment "
            f"actions, {single_segment_changed}/{n} decisions when redrawn for "
            f"segment 0 alone"
        )
        # A measurement, not a threshold, but the mechanism must actually
        # run: the validator replaced the perturbation with a no-op and
        # this test printed 0/185 and still passed against the previous
        # `assert 0 <= decisions_changed <= n`, which no count can fail.
        assert decisions_changed > 0
        # per_segment_changed (printed above) is NOT asserted separately:
        # int(np.sum(a != b)) > 0 and not np.array_equal(a, b) are the same
        # predicate over the same pair, so summed across cases
        # per_segment_changed > 0 holds exactly when decisions_changed > 0
        # does -- an assertion here could not fail without the one above
        # already having failed first. It stays a reported measurement, not
        # a second gate on the same fact.
        #
        # Round 2 found the same gap in the single-segment loop above,
        # fifteen lines from decisions_changed's own assertion: computed
        # and printed, gated by nothing. The validator proved it by
        # replacing the perturbation `altered[0, 2] = rng_single.uniform(0.0,
        # 10.0)` with the no-op `altered[0, 2] = state[0, 2]`: this test
        # printed "0/185 decisions ... redrawn for segment 0 alone" and
        # still passed, because nothing here checked that number.
        assert single_segment_changed > 0
