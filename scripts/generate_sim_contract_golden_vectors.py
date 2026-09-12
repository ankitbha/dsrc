#!/usr/bin/env python3
"""Generate specs/sim_contract_golden_vectors.json.

The frozen file this writes is what deployment/jetson/tests/test_sim_contract.py
reads and asserts against; the test never regenerates it and imports no
simulation module. This script is the only place the two still meet.

Every recorded quantity is derived twice: once from the simulation at
SIM_COMMIT (`policy.sim_contract.SIM_COMMIT`), extracted with `git archive`
into a temporary directory that is never added to sys.path[0]-adjacent to the
repository root; and once from `deployment/jetson/policy/sim_contract.py`, the
vendored copy the device actually runs. If any pair disagrees, this prints the
disagreement and exits non-zero, having written nothing -- with `--write
--force` included. The only way to change a recorded value is to point
SIM_COMMIT at a commit whose reference actually produces the new value, which
is a contract change and is visible in the diff.

Default mode is --check: derive everything, compare against the recorded
file, write nothing, exit non-zero on any difference (including "the file
does not exist"). --write requires --force to overwrite an existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
JETSON_DIR = REPO_ROOT / "deployment" / "jetson"
OUT_DEFAULT = REPO_ROOT / "specs" / "sim_contract_golden_vectors.json"

# `policy` has to be importable before anything else touches sys.path, so that
# SIM_COMMIT can be read from the vendored copy rather than duplicated here as
# a second literal that could drift from it.
sys.path.insert(0, str(JETSON_DIR))
from policy import sim_contract  # noqa: E402

SPEED_BIN_FREE_FLOWS = (13.0, 20.0, 22.0, 30.0, 40.0)
BIN_INDEX_EDGES = (0.5, 1.0, 2.0)
BIN_INDEX_VALUES = (0.0, 0.4999, 0.5, 0.9999, 1.0, 1.5, 2.0, 2.5)
NEUTRAL_FREE_FLOWS = (13.0, 30.0)

NOTE = (
    "Frozen contract vectors: every value here was derived twice when this file "
    "was generated -- once from the simulation at sim_commit (extracted with git "
    "archive), once from policy/sim_contract.py -- and the generator refused to "
    "write if the two disagreed. Changing a recorded value is a contract change "
    "and requires SIM_COMMIT to move to a commit whose reference actually "
    "produces the new value. ADDING A CASE IS NOT A CONTRACT CHANGE, since it "
    "constrains nothing already agreed -- but because regeneration rewrites the "
    "whole file, a test checks every pre-existing case is byte-identical."
)

FULL_OBS: dict[str, Any] = {
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

# The 12 cases test_sim_contract.py has used since it last ran against the sim
# (`:64-77` at 975b7a2), unchanged and in the same order, plus the two D13
# additions with three genuinely distinct values in each nested block so that
# *any* reorder inside COOPERATION_FIELDS or LANE_DISTRIBUTION_LANES changes at
# least one slot's number, not only its name.
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("empty", "everything missing", {}),
    (
        "all_none",
        "every LOCAL_OBS_FIELDS key present and explicitly None",
        {field: None for field in sim_contract.LOCAL_OBS_FIELDS},
    ),
    ("full_obs", "the ordinary case: every field present and numeric", dict(FULL_OBS)),
    (
        "inf_leader_gap_and_headway",
        "two independent inf-clamped fields with different scales in one obs",
        {**FULL_OBS, "leader_gap": float("inf"), "ego_headway_s": float("inf")},
    ),
    (
        "neg_inf_relative_speed",
        "the negative branch of the inf clamp",
        {**FULL_OBS, "leader_relative_speed": float("-inf")},
    ),
    (
        "nan_ego_speed",
        "NaN takes the negative clamp branch (NaN > 0 is False), not +inf's",
        {**FULL_OBS, "ego_speed": float("nan")},
    ),
    (
        "bools_flipped",
        "bools bypass scaling entirely, both values",
        {**FULL_OBS, "is_active": False, "uncongested_low_speed_flag": True},
    ),
    (
        "numeric_string",
        "a numeric string parses through float()",
        {**FULL_OBS, "ego_speed": "18.5"},
    ),
    (
        "junk_string",
        "a non-numeric string takes the parse_fail branch -> 0.0",
        {**FULL_OBS, "ego_speed": "not-a-number"},
    ),
    (
        "cooperation_not_a_mapping",
        "a non-mapping cooperation block is ignored; every cooperation.* slot is none",
        {**FULL_OBS, "cooperation": "garbage"},
    ),
    (
        "lane_distribution_not_a_mapping",
        "a non-mapping lane distribution is ignored; every lane slot is none",
        {**FULL_OBS, "nearby_av_lane_distribution": 7},
    ),
    (
        "inf_time_since_lane_change",
        "inf clamp on a field whose neutral fallback is also inf",
        {**FULL_OBS, "time_since_last_lane_change": float("inf")},
    ),
    (
        "cooperation_distinct_values",
        "three distinct, differently-scaled cooperation values (D13): a "
        "COOPERATION_FIELDS reorder changes at least one slot's number, not "
        "only its name",
        {
            **FULL_OBS,
            "cooperation": {
                "segment_target_speed": 24.0,
                "merge_pressure": 0.3,
                "downstream_congestion_estimate": 0.9,
            },
        },
    ),
    (
        "lane_distribution_distinct_values",
        "three distinct lane-distribution values (D13): a LANE_DISTRIBUTION_LANES "
        "reorder changes at least one slot's number, not only its name",
        {**FULL_OBS, "nearby_av_lane_distribution": {"0": 0.6, "1": 0.25, "2": 0.05}},
    ),
]


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with their string sentinels.

    JSON has no literal for infinity or NaN; `json.dumps` would otherwise emit
    the non-standard bare tokens `Infinity` / `-Infinity` / `NaN`. Booleans and
    ints pass through unchanged (`isinstance(True, float)` is False).
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def classify_rule(raw: Any) -> str:
    """Which of `_number`'s five branches a raw value selects. Classification
    only -- the recorded *value* always comes from calling the real encoder,
    never from re-deriving the arithmetic here (that independent restatement
    is Account C, and it lives in the test file, not the generator)."""
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


def raw_and_scale_per_slot(obs: Mapping[str, Any]) -> list[tuple[str, Any, float]]:
    """(slot_name, raw_input, scale) for all 39 slots, in encoder order --
    against the *vendored* field lists and scales, which is safe because
    `derive()` separately checks those against the reference's own (the
    "field lists + scales" section, above the per-case loop)."""
    entries: list[tuple[str, Any, float]] = []
    for field in sim_contract.LOCAL_OBS_FIELDS:
        entries.append((field, obs.get(field), sim_contract.FIELD_SCALES.get(field, 1.0)))
    cooperation = obs.get("cooperation", {})
    if not isinstance(cooperation, Mapping):
        cooperation = {}
    for field in sim_contract.COOPERATION_FIELDS:
        entries.append(
            (f"cooperation.{field}", cooperation.get(field), sim_contract.FIELD_SCALES.get(field, 1.0))
        )
    lane_distribution = obs.get("nearby_av_lane_distribution", {})
    if not isinstance(lane_distribution, Mapping):
        lane_distribution = {}
    for lane in sim_contract.LANE_DISTRIBUTION_LANES:
        entries.append((f"nearby_av_lane_distribution.{lane}", lane_distribution.get(lane), 1.0))
    return entries


def vector_sha256(values: list[float]) -> str:
    payload = json.dumps(values, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Mismatches:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        self.messages.append(message)

    def check_equal(self, label: str, account_a: Any, account_b: Any) -> None:
        if account_a != account_b:
            self.add(f"{label}: reference={account_a!r} vendored={account_b!r}")

    def __bool__(self) -> bool:
        return bool(self.messages)


def extract_reference(sim_commit: str, dest: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", sim_commit, "src"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if archive.returncode != 0:
        raise RuntimeError(
            f"git archive {sim_commit} src failed: {archive.stderr.decode('utf-8', 'replace')}"
        )
    extract = subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True)
    if extract.returncode != 0:
        raise RuntimeError(f"extracting the archive failed: {extract.stderr.decode('utf-8', 'replace')}")


def derive(sim_commit: str) -> tuple[dict[str, Any], Mismatches]:
    """Build the full golden payload, deriving every quantity from the
    reference (Account A) and from sim_contract.py (Account B), and collect
    every disagreement rather than raising on the first one."""
    mismatches = Mismatches()

    with tempfile.TemporaryDirectory(prefix="sim_contract_ref_") as tmp:
        # Resolved once, up front: macOS puts the real path under /private/var
        # while $TMPDIR reports /var (a symlink), and Path.resolve() below
        # would otherwise make the archive look like it lost the race it won.
        ref_dir = Path(tmp).resolve()
        extract_reference(sim_commit, ref_dir)
        # Never REPO_ROOT: the working tree's own `src` package (which now
        # holds only src_q.py under src/rl/) must not win this import. Once
        # these are bound in sys.modules by name, nothing later in this
        # function can make them re-resolve some other way, so ref_dir is left
        # on sys.path rather than removed again.
        sys.path.insert(0, str(ref_dir))
        import src.rl.encoders as ref_encoders  # noqa: E402
        import src.rl.actions as ref_actions  # noqa: E402
        import src.envs.wrappers as ref_wrappers  # noqa: E402
        import src.rl.models as ref_models  # noqa: E402
        import src.sensing.local as ref_local  # noqa: E402

        resolved = Path(ref_encoders.__file__).resolve()
        if ref_dir not in resolved.parents:
            raise RuntimeError(
                f"src.rl.encoders resolved to {resolved}, not under the archive at {ref_dir} "
                "-- the working tree's own src package won a race it must never win"
            )

        payload: dict[str, Any] = {}

        # -- field lists + scales ------------------------------------------------
        ref_slot_names = (
            *ref_encoders.LOCAL_OBS_FIELDS,
            *(f"cooperation.{f}" for f in ref_encoders.COOPERATION_FIELDS),
            *(f"nearby_av_lane_distribution.{lane}" for lane in ref_encoders.LANE_DISTRIBUTION_LANES),
        )
        our_slot_names = sim_contract.encoded_slot_names()
        mismatches.check_equal("slot_names", ref_slot_names, our_slot_names)
        mismatches.check_equal(
            "local_obs_dim", ref_encoders.local_obs_dim(), sim_contract.local_obs_dim()
        )
        payload["slot_names"] = list(our_slot_names)
        payload["local_obs_dim"] = sim_contract.local_obs_dim()

        local_fields = set(ref_encoders.LOCAL_OBS_FIELDS) | set(ref_encoders.COOPERATION_FIELDS)
        ref_restricted_scales = {k: v for k, v in ref_encoders.FIELD_SCALES.items() if k in local_fields}
        mismatches.check_equal(
            "field_scales key set (restricted to fields the local encoder reads)",
            dict(sorted(ref_restricted_scales.items())),
            dict(sorted(sim_contract.FIELD_SCALES.items())),
        )
        field_scales_entries = []
        seen = set()
        for field in sim_contract.LOCAL_OBS_FIELDS:
            if field in sim_contract.FIELD_SCALES:
                field_scales_entries.append({"field": field, "scale": sim_contract.FIELD_SCALES[field]})
                seen.add(field)
        for field in sorted(set(sim_contract.FIELD_SCALES) - seen):
            field_scales_entries.append({"field": field, "scale": sim_contract.FIELD_SCALES[field]})
        payload["field_scales"] = field_scales_entries

        # -- cases -----------------------------------------------------------
        case_payloads = []
        for name, why, obs in CASES:
            entries = raw_and_scale_per_slot(obs)
            account_b_vector = sim_contract.encode_local_observation(obs)
            account_a_vector = ref_encoders.encode_local_observation(obs).tolist()
            if account_b_vector.shape != (sim_contract.local_obs_dim(),):
                mismatches.add(f"case {name!r}: vendored vector shape {account_b_vector.shape}")
            for index, (slot_name, _raw, _scale) in enumerate(entries):
                a_val = float(account_a_vector[index])
                b_val = float(account_b_vector[index])
                if not math.isclose(a_val, b_val, rel_tol=0, abs_tol=1e-6):
                    mismatches.add(
                        f"case {name!r} slot {slot_name!r}: reference={a_val} vendored={b_val}"
                    )
            slots = [
                {
                    "slot": slot_name,
                    "raw": json_safe(raw),
                    "scale": float(scale),
                    "rule": classify_rule(raw),
                    "value": float(account_b_vector[index]),
                }
                for index, (slot_name, raw, scale) in enumerate(entries)
            ]
            values = [s["value"] for s in slots]
            case_payloads.append(
                {
                    "name": name,
                    "why": why,
                    "obs": json_safe(obs),
                    "vector_sha256": vector_sha256(values),
                    "slots": slots,
                }
            )
        payload["cases"] = case_payloads

        # -- actions -----------------------------------------------------------
        mismatches.check_equal("ACTION_HEADS", ref_actions.ACTION_HEADS, sim_contract.ACTION_HEADS)
        for head in sim_contract.ACTION_HEADS:
            mismatches.check_equal(
                f"ACTION_VALUES[{head!r}]",
                tuple(ref_actions.ACTION_VALUES.get(head, ())),
                sim_contract.ACTION_VALUES[head],
            )
        mismatches.check_equal("FORCED_ACTIONS", ref_actions.FORCED_ACTIONS, sim_contract.FORCED_ACTIONS)
        ref_profiles = {
            profile: ref_actions.ActionSpec(profile).active_heads
            for profile in sim_contract.ACTION_PROFILES
        }
        for profile in sim_contract.ACTION_PROFILES:
            mismatches.check_equal(
                f"ACTION_PROFILES[{profile!r}]",
                ref_profiles[profile],
                sim_contract.ACTION_PROFILES[profile],
            )
        ref_defaults = ref_actions.ActionSpec("full").default_indices()
        mismatches.check_equal("default_indices()", ref_defaults, sim_contract.default_indices())

        payload["actions"] = {
            "heads": list(sim_contract.ACTION_HEADS),
            "values": [
                {"head": head, "values": list(sim_contract.ACTION_VALUES[head])}
                for head in sim_contract.ACTION_HEADS
            ],
            "forced": [
                {"head": head, "value": value} for head, value in sim_contract.FORCED_ACTIONS.items()
            ],
            "profiles": [
                {"profile": profile, "heads": list(heads)}
                for profile, heads in sim_contract.ACTION_PROFILES.items()
            ],
            "default_indices": [
                {"head": head, "index": sim_contract.default_indices()[head]}
                for head in sim_contract.ACTION_HEADS
            ],
        }

        # -- decoders ----------------------------------------------------------
        headway_bin_s = []
        for headway_bin, seconds in sim_contract.HEADWAY_BIN_S.items():
            ref_seconds = ref_wrappers.decode_headway_bin(headway_bin)
            mismatches.check_equal(f"decode_headway_bin({headway_bin!r})", ref_seconds, seconds)
            headway_bin_s.append({"bin": headway_bin, "seconds": seconds})

        speed_bin_mps = []
        for speed_bin in sim_contract.SPEED_BIN_OFFSETS_MPS:
            for free_flow in SPEED_BIN_FREE_FLOWS:
                ref_value = ref_wrappers.decode_speed_bin(speed_bin, free_flow)
                our_value = sim_contract.decode_speed_bin(speed_bin, free_flow)
                mismatches.check_equal(
                    f"decode_speed_bin({speed_bin!r}, {free_flow})", ref_value, our_value
                )
                speed_bin_mps.append(
                    {
                        "bin": speed_bin,
                        "free_flow_mps": free_flow,
                        "min_contextual_mps": 12.0,
                        "value": our_value,
                    }
                )

        bin_index_grid = []
        for value in BIN_INDEX_VALUES:
            ref_index = ref_local._bin(value, BIN_INDEX_EDGES)
            our_index = sim_contract.bin_index(value, BIN_INDEX_EDGES)
            mismatches.check_equal(f"bin_index({value}, {BIN_INDEX_EDGES})", ref_index, our_index)
            bin_index_grid.append({"value": value, "edges": list(BIN_INDEX_EDGES), "index": our_index})

        payload["decoders"] = {
            "headway_bin_s": headway_bin_s,
            "speed_bin_mps": speed_bin_mps,
            "bin_index": bin_index_grid,
        }

        # B1 (write side): a grid this loop iterates zero times contributes zero
        # mismatches and an empty list, and the generator would print "the reference
        # and the vendored contract agree on every recorded quantity" and write that
        # empty section. Refuse instead: every grid constant must be non-empty, and
        # each emitted list's length must equal what its own source implies.
        if not sim_contract.HEADWAY_BIN_S:
            mismatches.add("sim_contract.HEADWAY_BIN_S is empty -- headway_bin_s would record nothing")
        elif len(headway_bin_s) != len(sim_contract.HEADWAY_BIN_S):
            mismatches.add(
                f"headway_bin_s: emitted {len(headway_bin_s)} entries, "
                f"HEADWAY_BIN_S has {len(sim_contract.HEADWAY_BIN_S)}"
            )
        if not SPEED_BIN_FREE_FLOWS:
            mismatches.add("SPEED_BIN_FREE_FLOWS is empty -- speed_bin_mps would record nothing")
        if not sim_contract.SPEED_BIN_OFFSETS_MPS:
            mismatches.add(
                "sim_contract.SPEED_BIN_OFFSETS_MPS is empty -- speed_bin_mps would record nothing"
            )
        expected_speed_bin_mps = len(sim_contract.SPEED_BIN_OFFSETS_MPS) * len(SPEED_BIN_FREE_FLOWS)
        if len(speed_bin_mps) != expected_speed_bin_mps:
            mismatches.add(
                f"speed_bin_mps: emitted {len(speed_bin_mps)} entries, "
                f"SPEED_BIN_OFFSETS_MPS x SPEED_BIN_FREE_FLOWS implies {expected_speed_bin_mps}"
            )
        if not BIN_INDEX_VALUES:
            mismatches.add("BIN_INDEX_VALUES is empty -- bin_index would record nothing")
        elif len(bin_index_grid) != len(BIN_INDEX_VALUES):
            mismatches.add(
                f"bin_index: emitted {len(bin_index_grid)} entries, "
                f"BIN_INDEX_VALUES has {len(BIN_INDEX_VALUES)}"
            )

        # -- neutral fallbacks (specs/observation_schema.md) --------------------
        neutral_cooperation = []
        for free_flow in NEUTRAL_FREE_FLOWS:
            expected = {
                "segment_target_speed": float(free_flow),
                "merge_pressure": 0.0,
                "downstream_congestion_estimate": 0.0,
            }
            our_neutral = sim_contract.neutral_cooperation(free_flow)
            mismatches.check_equal(f"neutral_cooperation({free_flow})", expected, our_neutral)
            neutral_cooperation.append(
                {
                    "free_flow_mps": free_flow,
                    "values": [
                        {"field": field, "value": our_neutral[field]}
                        for field in sim_contract.COOPERATION_FIELDS
                    ],
                }
            )
        payload["neutral_cooperation"] = neutral_cooperation

        # B1 (write side), continued: NEUTRAL_FREE_FLOWS is the grid constant behind
        # neutral_cooperation; empty, the loop above contributes nothing and the
        # section is silently recorded as `[]`.
        if not NEUTRAL_FREE_FLOWS:
            mismatches.add("NEUTRAL_FREE_FLOWS is empty -- neutral_cooperation would record nothing")
        elif len(neutral_cooperation) != len(NEUTRAL_FREE_FLOWS):
            mismatches.add(
                f"neutral_cooperation: emitted {len(neutral_cooperation)} entries, "
                f"NEUTRAL_FREE_FLOWS has {len(NEUTRAL_FREE_FLOWS)}"
            )

        # -- actor state-dict layout --------------------------------------------
        import policy.export_policy as export_policy  # noqa: E402  (adds JETSON_DIR at sys.path[0] again; harmless)

        sim_actor = ref_models.MultiCategoricalActor(sim_contract.local_obs_dim())
        our_actor = export_policy.VendoredActor(sim_contract.local_obs_dim())
        ref_layout = {k: tuple(v.shape) for k, v in sim_actor.state_dict().items()}
        our_layout = {k: tuple(v.shape) for k, v in our_actor.state_dict().items()}
        mismatches.check_equal("actor state_dict layout", ref_layout, our_layout)
        payload["actor_state_dict_layout"] = [
            {"key": key, "shape": list(shape)} for key, shape in our_layout.items()
        ]

        # -- fingerprint ---------------------------------------------------------
        payload["contract_fingerprint"] = sim_contract.contract_fingerprint()

    payload = {
        "sim_commit": sim_commit,
        "frozen": True,
        "generated_from": f"git archive {sim_commit} src",
        "contract_fingerprint": payload["contract_fingerprint"],
        "local_obs_dim": payload["local_obs_dim"],
        "note": NOTE,
        "slot_names": payload["slot_names"],
        "field_scales": payload["field_scales"],
        "cases": payload["cases"],
        "actions": payload["actions"],
        "decoders": payload["decoders"],
        "neutral_cooperation": payload["neutral_cooperation"],
        "actor_state_dict_layout": payload["actor_state_dict_layout"],
    }
    return payload, mismatches


def compare_against_recorded(recorded: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    diffs = []
    for key in fresh:
        if recorded.get(key) != fresh[key]:
            diffs.append(f"{key} differs from the recorded file")
    for key in recorded:
        if key not in fresh:
            diffs.append(f"{key} is in the recorded file but not derived")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--write", action="store_true", help="write the file (default is --check)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="the default: derive and compare against the recorded file, write nothing",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing file; refused otherwise"
    )
    args = parser.parse_args()

    sim_commit = sim_contract.SIM_COMMIT
    print(f"deriving every recorded quantity from sim commit {sim_commit} and from sim_contract.py ...")
    payload, mismatches = derive(sim_commit)

    if mismatches:
        print(f"REFUSING: {len(mismatches.messages)} disagreement(s) between the reference and the "
              "vendored contract -- nothing was written:")
        for message in mismatches.messages:
            print(f"  - {message}")
        return 1

    print("the reference and the vendored contract agree on every recorded quantity.")

    if args.write:
        if args.out.exists() and not args.force:
            print(f"{args.out} exists and is frozen; pass --force only for a contract change")
            return 1
        args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"wrote {args.out}")
        return 0

    # --check (default)
    if not args.out.exists():
        print(f"{args.out} does not exist")
        return 1
    recorded = json.loads(args.out.read_text())
    diffs = compare_against_recorded(recorded, payload)
    if diffs:
        print(f"{args.out} disagrees with a fresh derivation:")
        for diff in diffs:
            print(f"  - {diff}")
        return 1
    print(f"{args.out} matches a fresh derivation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
