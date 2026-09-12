"""Guard the vendored contract against drift from the simulation source.

The sim modules this file used to import (`src.rl.encoders`, `src.rl.actions`,
`src.rl.models`) were deleted in `6b538f2`; there is no machine on which they
import any more. What replaces the comparison is
`specs/sim_contract_golden_vectors.json`: 14 encoded observations, the action
and decoder constants, and the actor layout, each derived twice -- once from
the simulation at `d477dba` and once from `policy/sim_contract.py` -- by
`scripts/generate_sim_contract_golden_vectors.py`, which refuses to write the
file if the two disagree. This module reads that file and imports no
simulation module, so it runs on every machine instead of no machine.

Two tests still need more than numpy. The actor-layout test needs `torch`
(already an unconditional import in `policy/actor_runtime.py`), and the
regeneration test spawns the generator, which needs `torch` and a `git`
repository holding the `SIM_COMMIT` object. Both skip with a stated reason if
their dependency is absent; the gate requires both to run and pass on the
development machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from policy import sim_contract

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "specs" / "sim_contract_golden_vectors.json"
RAW_TEXT = GOLDEN_PATH.read_text()
DOCUMENT = json.loads(RAW_TEXT)

CASES = DOCUMENT["cases"]
CASE_IDS = [case["name"] for case in CASES]

# The literal this file is pinned against, independent of the JSON. A
# regeneration cannot rewrite this: two calls to the same pure function agree
# with each other no matter what either one returns, so a test that only
# compares `contract_fingerprint()` against itself can never fail. Comparing
# against a string typed here as well as against the file is what makes a
# silent regeneration visible.
PINNED_FINGERPRINT = "918ec57cf2f2e1db"


def _decode(value):
    """Reverse of the generator's json_safe(): the three exact sentinel
    strings become their non-finite floats; everything else, including
    ordinary strings like "18.5" or "not-a-number", passes through unchanged."""
    if isinstance(value, str) and value in ("Infinity", "-Infinity", "NaN"):
        return {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}[value]
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _obs_for(case: dict) -> dict:
    return _decode(case["obs"])


def _vector_sha256(values: list[float]) -> str:
    payload = json.dumps(values, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# -- the file itself -----------------------------------------------------


def test_the_file_declares_itself_frozen():
    assert DOCUMENT["frozen"] is True


def test_the_file_is_for_this_sim_commit():
    assert DOCUMENT["sim_commit"] == sim_contract.SIM_COMMIT


def test_local_obs_dim_matches():
    assert DOCUMENT["local_obs_dim"] == sim_contract.local_obs_dim() == 39


def test_case_names_are_unique():
    assert len(CASE_IDS) == len(set(CASE_IDS))


def test_the_note_carries_the_freeze_rule():
    assert "ADDING A CASE IS NOT A CONTRACT CHANGE" in DOCUMENT["note"]


def test_the_raw_json_contains_no_bare_nan_or_infinity_token():
    """JSON has no literal for infinity or NaN; Python's parser accepts the
    non-standard bare tokens Infinity/-Infinity/NaN unless told not to.
    Quoted occurrences (the sentinel strings this file uses deliberately)
    parse as ordinary JSON strings and never reach parse_constant; only a
    bare, unquoted token would."""

    def _reject(token):
        raise AssertionError(f"bare {token!r} token in the golden file; it must be quoted")

    json.loads(RAW_TEXT, parse_constant=_reject)


# -- slot order ------------------------------------------------------------


def test_slot_names_match_encoded_slot_names():
    assert tuple(DOCUMENT["slot_names"]) == sim_contract.encoded_slot_names()


def test_the_first_33_slot_names_are_local_obs_fields_in_order():
    assert tuple(DOCUMENT["slot_names"][:33]) == sim_contract.LOCAL_OBS_FIELDS


def test_the_tail_is_the_two_nested_blocks_dotted_in_recorded_order():
    names = DOCUMENT["slot_names"]
    assert names[33:36] == [f"cooperation.{f}" for f in sim_contract.COOPERATION_FIELDS]
    assert names[36:39] == [
        f"nearby_av_lane_distribution.{lane}" for lane in sim_contract.LANE_DISTRIBUTION_LANES
    ]


# -- scales ------------------------------------------------------------------


def test_field_scales_match_in_both_directions():
    recorded = {entry["field"]: entry["scale"] for entry in DOCUMENT["field_scales"]}
    # Both directions (O3): a scale FIELD_SCALES has that the recorded file
    # lacks is invisible to a one-direction check, and vice versa.
    assert set(recorded) == set(sim_contract.FIELD_SCALES)
    for field, scale in recorded.items():
        assert sim_contract.FIELD_SCALES[field] == scale, field


# -- encoding (Account B: the vendored encoder) -------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_encoding_matches_the_recorded_vector(case):
    obs = _obs_for(case)
    encoded = sim_contract.encode_local_observation(obs)
    assert encoded.shape == (sim_contract.local_obs_dim(),)
    recorded_values = [slot["value"] for slot in case["slots"]]
    np.testing.assert_allclose(encoded, recorded_values, rtol=0, atol=1e-6)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_vector_sha256_matches_the_recorded_values(case):
    recorded_values = [slot["value"] for slot in case["slots"]]
    assert _vector_sha256(recorded_values) == case["vector_sha256"]


# -- derivation (Account C: arithmetic restated here, calling neither encoder) --


def _account_c(raw, scale: float, rule: str) -> float:
    """`_number`'s five branches, restated independently of both encoders.

    `sign` must be the comparison `raw > 0`, not `math.copysign`: `_number`
    (sim_contract.py) takes the negative branch for anything that is not
    strictly positive, and `NaN > 0` is False, so NaN encodes negative.
    `math.copysign(5.0, float("nan"))` returns +5.0 and would disagree with
    both other accounts on the nan_ego_speed case.
    """
    if rule == "bool":
        return float(raw)
    if rule == "none":
        return 0.0
    if rule == "parse_fail":
        return 0.0
    if rule == "inf_clamp":
        sign = 1.0 if raw > 0 else -1.0
        return sign * min(200.0, 5.0 * max(scale, 1e-9)) / max(scale, 1e-9)
    if rule == "plain":
        return float(raw) / max(scale, 1e-9)
    raise AssertionError(f"unknown rule {rule!r}")


def _actual_rule(raw) -> str:
    """Which branch `raw` selects -- independent of the recorded `rule`, so a
    `raw` of 2.0 recorded as `inf_clamp` is a failure even when the arithmetic
    happens to agree."""
    if isinstance(raw, bool):
        return "bool"
    if raw is None:
        return "none"
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return "parse_fail"
    if not math.isfinite(result):
        return "inf_clamp"
    return "plain"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_derivation_matches_every_slot(case):
    for slot in case["slots"]:
        raw = _decode(slot["raw"])
        assert _actual_rule(raw) == slot["rule"], (case["name"], slot["slot"])
        recomputed = _account_c(raw, slot["scale"], slot["rule"])
        assert math.isclose(recomputed, slot["value"], rel_tol=0, abs_tol=1e-6), (
            case["name"],
            slot["slot"],
        )


RULES_IN_USE = {slot["rule"] for case in CASES for slot in case["slots"]}


def test_every_number_rule_is_exercised_by_at_least_one_slot():
    assert RULES_IN_USE == {"bool", "none", "parse_fail", "inf_clamp", "plain"}


# -- actions -----------------------------------------------------------------


def test_action_heads_match():
    assert tuple(DOCUMENT["actions"]["heads"]) == sim_contract.ACTION_HEADS


def test_action_values_and_their_order_match():
    for entry in DOCUMENT["actions"]["values"]:
        assert tuple(entry["values"]) == sim_contract.ACTION_VALUES[entry["head"]], entry["head"]


def test_forced_actions_match():
    recorded = {entry["head"]: entry["value"] for entry in DOCUMENT["actions"]["forced"]}
    assert recorded == sim_contract.FORCED_ACTIONS


def test_action_profiles_match():
    for entry in DOCUMENT["actions"]["profiles"]:
        assert tuple(entry["heads"]) == sim_contract.ACTION_PROFILES[entry["profile"]]


def test_default_indices_match():
    recorded = {entry["head"]: entry["index"] for entry in DOCUMENT["actions"]["default_indices"]}
    assert recorded == sim_contract.default_indices()


# -- decoders ------------------------------------------------------------


def test_decode_headway_bin_matches_the_recorded_grid():
    for entry in DOCUMENT["decoders"]["headway_bin_s"]:
        assert sim_contract.decode_headway_bin(entry["bin"]) == entry["seconds"]


def test_decode_speed_bin_matches_the_recorded_grid():
    for entry in DOCUMENT["decoders"]["speed_bin_mps"]:
        got = sim_contract.decode_speed_bin(
            entry["bin"], entry["free_flow_mps"], entry["min_contextual_mps"]
        )
        assert got == entry["value"], entry


def test_bin_index_matches_the_recorded_grid():
    for entry in DOCUMENT["decoders"]["bin_index"]:
        got = sim_contract.bin_index(entry["value"], entry["edges"])
        assert got == entry["index"], entry


# -- neutral fallbacks ---------------------------------------------------


def test_neutral_cooperation_matches_the_recorded_values():
    for entry in DOCUMENT["neutral_cooperation"]:
        got = sim_contract.neutral_cooperation(entry["free_flow_mps"])
        recorded = {v["field"]: v["value"] for v in entry["values"]}
        assert got == recorded


def test_neutral_cooperation_matches_the_observation_schema_doc():
    """Cross-checked against the source sim_contract.py cites
    (specs/observation_schema.md), which is still in the tree."""
    schema_path = REPO_ROOT / "specs" / "observation_schema.md"
    text = schema_path.read_text()
    assert "nearby_av_mean_speed: free_flow_speed" in text
    assert "cooperation.segment_target_speed: free_flow_speed" in text
    assert "cooperation.merge_pressure: 0.0" in text
    assert "cooperation.downstream_congestion_estimate: 0.0" in text
    got = sim_contract.neutral_cooperation(30.0)
    assert got == {
        "segment_target_speed": 30.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
    }


# -- fingerprint --------------------------------------------------------------


def test_contract_fingerprint_matches_the_file_and_the_pinned_literal():
    assert sim_contract.contract_fingerprint() == DOCUMENT["contract_fingerprint"]
    assert sim_contract.contract_fingerprint() == PINNED_FINGERPRINT


def test_the_fingerprint_is_stable_across_calls():
    assert sim_contract.contract_fingerprint() == sim_contract.contract_fingerprint()


class TestTheBundleGuardSeesMoreThanTheDimension:
    """A reorder leaves the dimension at 39 and puts every value in the wrong
    slot -- the fingerprint has to move on that, and does."""

    def test_the_fingerprint_moves_when_the_field_order_does(self):
        before = sim_contract.contract_fingerprint()
        original = sim_contract.LOCAL_OBS_FIELDS
        try:
            swapped = list(original)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            sim_contract.LOCAL_OBS_FIELDS = tuple(swapped)
            assert len(sim_contract.LOCAL_OBS_FIELDS) == len(original), "the dimension is unchanged"
            assert sim_contract.contract_fingerprint() != before, (
                "two fields swapped and the guard cannot tell"
            )
        finally:
            sim_contract.LOCAL_OBS_FIELDS = original
        assert sim_contract.contract_fingerprint() == before

    def test_the_fingerprint_moves_when_a_scale_does(self):
        before = sim_contract.contract_fingerprint()
        key = next(iter(sim_contract.FIELD_SCALES))
        original = sim_contract.FIELD_SCALES[key]
        try:
            sim_contract.FIELD_SCALES[key] = original * 2.0
            assert sim_contract.contract_fingerprint() != before
        finally:
            sim_contract.FIELD_SCALES[key] = original
        assert sim_contract.contract_fingerprint() == before


class TestEncodedSlotNames:
    """The 39 names `encode_local_observation` fills, in encoder order --
    what `field_sources` has to cover for `missingness` to be a statement
    about the whole vector rather than the 33 flat fields alone.
    """

    def test_the_first_33_are_local_obs_fields_in_order(self):
        assert sim_contract.encoded_slot_names()[:33] == sim_contract.LOCAL_OBS_FIELDS

    def test_length_matches_local_obs_dim(self):
        assert len(sim_contract.encoded_slot_names()) == sim_contract.local_obs_dim() == 39

    def test_the_tail_is_the_two_nested_blocks_dotted(self):
        names = sim_contract.encoded_slot_names()
        assert names[33:36] == (
            "cooperation.segment_target_speed",
            "cooperation.merge_pressure",
            "cooperation.downstream_congestion_estimate",
        )
        assert names[36:39] == (
            "nearby_av_lane_distribution.0",
            "nearby_av_lane_distribution.1",
            "nearby_av_lane_distribution.2",
        )

    def test_additive_the_fingerprint_does_not_depend_on_it(self):
        # `contract_fingerprint` hashes only `LOCAL_OBS_FIELDS` and
        # `FIELD_SCALES` (sim_contract.py) -- this helper introduces no new
        # constant, so calling it must not move the fingerprint.
        before = sim_contract.contract_fingerprint()
        assert before == PINNED_FINGERPRINT
        sim_contract.encoded_slot_names()
        assert sim_contract.contract_fingerprint() == before


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


# -- regeneration --------------------------------------------------------


def test_regeneration_leaves_every_pre_existing_case_byte_identical():
    """The guard the note promises. The generator rewrites the whole file, so
    the only thing standing between "added a case" and "silently moved an
    existing one" is this check. Needs torch (to import the reference actor
    module) and a git repository holding SIM_COMMIT; skips with a stated
    reason if either is unavailable."""
    pytest.importorskip("torch", reason="the generator imports torch to build the reference actor")
    script = REPO_ROOT / "scripts" / "generate_sim_contract_golden_vectors.py"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "regenerated.json"
        result = subprocess.run(
            [sys.executable, str(script), "--out", str(out), "--write", "--force"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0 and "git archive" in (result.stdout + result.stderr):
            pytest.skip("SIM_COMMIT not reachable from this git repository")
        assert result.returncode == 0, result.stdout + result.stderr
        fresh = json.loads(out.read_text())
    recorded_cases = {case["name"]: case for case in DOCUMENT["cases"]}
    fresh_cases = {case["name"]: case for case in fresh["cases"]}
    assert set(recorded_cases) <= set(fresh_cases), "a pre-existing case went missing"
    for name, case in recorded_cases.items():
        assert fresh_cases[name]["vector_sha256"] == case["vector_sha256"], name
        assert fresh_cases[name]["slots"] == case["slots"], name
    assert fresh["contract_fingerprint"] == DOCUMENT["contract_fingerprint"]
    assert fresh["slot_names"] == DOCUMENT["slot_names"]
    assert fresh["field_scales"] == DOCUMENT["field_scales"]


# -- actor layout (needs torch) --------------------------------------------


def test_actor_state_dict_layout_matches_the_recorded_layout():
    pytest.importorskip("torch", reason="the actor layout is a torch state_dict")
    from policy.export_policy import VendoredActor

    actor = VendoredActor(sim_contract.local_obs_dim())
    layout = {k: list(v.shape) for k, v in actor.state_dict().items()}
    recorded = {entry["key"]: entry["shape"] for entry in DOCUMENT["actor_state_dict_layout"]}
    assert layout == recorded
