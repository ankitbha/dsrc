"""Task 47: the observation parity ledger between the live and simulated
sensing models (`src.analysis.observation_parity`).

`test_sim_contract.py` (`deployment/jetson/tests/`) already proves the two
ENCODERS agree; `test_local_sensing.py` (this directory) exercises the
simulator's own sensing model in isolation. Neither compares the two
sensing models to each other, which is this file's subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis import observation_audit
from src.analysis.observation_parity import (
    CLASS_APPROXIMATED,
    CLASS_IDENTICAL,
    CLASS_STRUCTURALLY_ABSENT,
    CLASS_SUBSTITUTED,
    JETSON_DIR,
    _production_builder_config,
    build_ledger,
    check_run_against_ledger,
    encode,
)

# `observation_parity`'s own import inserts deployment/jetson onto
# sys.path (mirroring scripts/generate_transport_golden_frames.py), so
# `policy` -- a deployment/jetson package, not a `src.*` one -- is only
# reachable bare, and only after that import has already run.
from perception.provenance import SUBSTITUTED  # noqa: E402
from policy import sim_contract  # noqa: E402


#: Pinned independently of `observation_parity.LEDGER` (D21/47.6): a slot
#: silently moving from `identical` to `substituted` because someone edited
#: one of the two sensing models is exactly the change a later reader must
#: not miss, and this dict does not import from the module it is checking --
#: it would agree with a drifted LEDGER just as readily as with a correct
#: one if it did.
EXPECTED_CLASSES = {
    "is_active": CLASS_IDENTICAL,
    "ego_speed": CLASS_IDENTICAL,
    "ego_acceleration": CLASS_IDENTICAL,
    "ego_lane": CLASS_SUBSTITUTED,
    "ego_headway_s": CLASS_IDENTICAL,
    "target_headway_s": CLASS_IDENTICAL,
    "time_since_last_lane_change": CLASS_SUBSTITUTED,
    "lane_changes_last_km": CLASS_SUBSTITUTED,
    "distance_to_next_merge": CLASS_IDENTICAL,
    "distance_to_downstream_bottleneck": CLASS_SUBSTITUTED,
    "leader_gap": CLASS_IDENTICAL,
    "leader_relative_speed": CLASS_IDENTICAL,
    "follower_gap": CLASS_STRUCTURALLY_ABSENT,
    "follower_relative_speed": CLASS_STRUCTURALLY_ABSENT,
    "left_lane_front_gap": CLASS_APPROXIMATED,
    "left_lane_rear_gap": CLASS_STRUCTURALLY_ABSENT,
    "right_lane_front_gap": CLASS_APPROXIMATED,
    "right_lane_rear_gap": CLASS_STRUCTURALLY_ABSENT,
    "target_lane_front_gap": CLASS_APPROXIMATED,
    "target_lane_rear_gap": CLASS_STRUCTURALLY_ABSENT,
    "target_lane_rear_required_decel": CLASS_STRUCTURALLY_ABSENT,
    "downstream_congestion_estimate": CLASS_APPROXIMATED,
    "merge_pressure": CLASS_APPROXIMATED,
    "segment_target_speed": CLASS_APPROXIMATED,
    "uncongested_low_speed_flag": CLASS_APPROXIMATED,
    "local_density_bin": CLASS_APPROXIMATED,
    "local_mean_speed_bin": CLASS_APPROXIMATED,
    "local_queue_estimate": CLASS_APPROXIMATED,
    "active_vehicle_count_local": CLASS_APPROXIMATED,
    "active_av_count_local": CLASS_APPROXIMATED,
    "nearby_av_count": CLASS_APPROXIMATED,
    "nearby_av_density": CLASS_APPROXIMATED,
    "nearby_av_mean_speed": CLASS_APPROXIMATED,
    "cooperation.segment_target_speed": CLASS_APPROXIMATED,
    "cooperation.merge_pressure": CLASS_APPROXIMATED,
    "cooperation.downstream_congestion_estimate": CLASS_APPROXIMATED,
    "nearby_av_lane_distribution.0": CLASS_SUBSTITUTED,
    "nearby_av_lane_distribution.1": CLASS_SUBSTITUTED,
    "nearby_av_lane_distribution.2": CLASS_SUBSTITUTED,
}


@pytest.fixture(scope="module")
def ledger():
    return build_ledger()


def test_the_ledger_covers_all_39_slots_and_no_others(ledger):
    assert {row["slot"] for row in ledger["slots"]} == set(sim_contract.encoded_slot_names())
    assert len(ledger["slots"]) == 39


def test_the_classification_is_pinned(ledger):
    """D21/47.6: a regression fence on the parity claim itself, over a
    fixture independent of the module under test."""
    actual = {row["slot"]: row["class"] for row in ledger["slots"]}
    assert actual == EXPECTED_CLASSES


def test_the_ledger_names_ten_or_more_non_identical_slots(ledger):
    """The brief's own falsification (task 47's finding): "at least ten of
    the 39 encoded slots differ by construction"."""
    non_identical = sum(1 for cls in EXPECTED_CLASSES.values() if cls != CLASS_IDENTICAL)
    assert non_identical >= 10


def test_exactly_six_slots_are_structurally_absent():
    """"six of them because the vehicle has no rear sensor" -- the plan's
    own count, pinned directly."""
    absent = [slot for slot, cls in EXPECTED_CLASSES.items() if cls == CLASS_STRUCTURALLY_ABSENT]
    assert len(absent) == 6
    assert set(absent) == {
        "follower_gap", "follower_relative_speed", "left_lane_rear_gap",
        "right_lane_rear_gap", "target_lane_rear_gap", "target_lane_rear_required_decel",
    }


def test_every_claimed_class_matches_the_scenes_actually_run(ledger):
    """D19: `identical` must hold bit-for-bit on every scene, and a
    non-identical claim must show a real disagreement on at least one --
    a check written to pass regardless would prove nothing about the code.

    Recomputed here from `row["per_scene"][*]["equal"]` (B4, validation
    round 1) rather than trusting `row["matches_claimed_class"]`, which the
    module under test also computed: a bug that hardcoded that field to
    `True` would still pass a test reading it back, and did -- 12 tests
    stayed green under exactly that mutation.
    """
    mismatches = []
    for row in ledger["slots"]:
        always_equal = all(v["equal"] for v in row["per_scene"].values())
        should_hold = always_equal if row["class"] == CLASS_IDENTICAL else not always_equal
        if not should_hold:
            mismatches.append(row["slot"])
    assert mismatches == []


def test_every_substituted_or_absent_slot_carries_substituted_provenance(ledger):
    """D19 acceptance item 4: a slot substituted in fact but marked measured
    in provenance is a defect and fails the task.

    Recomputed from `row["field_sources_by_scene"]` against
    `perception.provenance.SUBSTITUTED` directly (B4), not from
    `row["provenance_in_substituted_partition"]`, which the module under
    test also computed.
    """
    failing = []
    for row in ledger["slots"]:
        if row["class"] not in (CLASS_SUBSTITUTED, CLASS_STRUCTURALLY_ABSENT):
            continue
        classes = [c for c in row["field_sources_by_scene"].values() if c is not None]
        if not classes or not all(c in SUBSTITUTED for c in classes):
            failing.append(row["slot"])
    assert failing == []


def test_every_named_constant_holds_across_every_scene(ledger):
    """Recomputed from `row["per_scene"][*]["live"]` against
    `row["live_constant"]` directly (B4), not from
    `row["constant_holds_across_scenes"]`."""
    rows_with_constants = [row for row in ledger["slots"] if row["live_constant"] is not None]
    assert len(rows_with_constants) > 0, "fixture must exercise at least one named constant"
    failing = []
    for row in rows_with_constants:
        constant = row["live_constant"]
        if not all(
            abs(v["live"] - constant) <= 1e-5 for v in row["per_scene"].values()
        ):
            failing.append(row["slot"])
    assert failing == []


# -- 47.7: the two independent constructions of the slot names agree -------


def test_slot_names_agree_with_observation_audit():
    """`sim_contract.encoded_slot_names()` (vendored, deployment/jetson) and
    `observation_audit.encoded_field_names()` (imports src.rl.encoders
    directly) are two independent readings of the same 39 names. They agree
    today and nothing compares them -- closing the soft gap D13 names."""
    assert sim_contract.encoded_slot_names() == observation_audit.encoded_field_names()


# -- 47.8/47.9: checked against a run's own logged ticks --------------------


def _write_run(tmp_path: Path, records: list[dict]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return run_dir


def _real_tick_record(tick_id: int = 0) -> dict:
    """A tick record shaped exactly like `Tick.to_record()`'s `obs`/`encoded`
    pair, built from a REAL encode rather than typed by hand -- so a test
    corrupting one field is corrupting a genuine tick, not a fixture that
    never resembled one."""
    obs = {
        "is_active": True, "ego_speed": 18.5, "ego_acceleration": 0.4,
        "ego_lane": 1, "ego_headway_s": 2.1, "target_headway_s": 1.6,
        "time_since_last_lane_change": float("inf"), "lane_changes_last_km": 0,
        "distance_to_next_merge": 0.0, "distance_to_downstream_bottleneck": float("inf"),
        "leader_gap": 42.0, "leader_relative_speed": -1.5,
        "follower_gap": float("inf"), "follower_relative_speed": 0.0,
        "left_lane_front_gap": float("inf"), "left_lane_rear_gap": float("inf"),
        "right_lane_front_gap": float("inf"), "right_lane_rear_gap": float("inf"),
        "target_lane_front_gap": 42.0, "target_lane_rear_gap": float("inf"),
        "target_lane_rear_required_decel": 0.0,
        "downstream_congestion_estimate": 0.0, "merge_pressure": 0.0,
        "segment_target_speed": 30.0, "uncongested_low_speed_flag": False,
        "local_density_bin": 1, "local_mean_speed_bin": 1, "local_queue_estimate": 0,
        "active_vehicle_count_local": 2, "active_av_count_local": 0,
        "nearby_av_count": 0, "nearby_av_density": 0.0, "nearby_av_mean_speed": 30.0,
        "nearby_av_lane_distribution": {},
        "cooperation": {"segment_target_speed": 30.0, "merge_pressure": 0.0,
                         "downstream_congestion_estimate": 0.0},
    }
    encoded = [round(float(x), 5) for x in encode(obs)]
    return {"type": "tick", "tick_id": tick_id, "obs": obs, "encoded": encoded}


def test_check_run_against_ledger_passes_on_a_genuinely_reencodable_tick(tmp_path, ledger):
    run_dir = _write_run(tmp_path, [_real_tick_record(0), _real_tick_record(1)])
    result = check_run_against_ledger(run_dir, ledger)
    assert result["ticks_checked"] == 2
    assert result["reencode_mismatches"] == []
    assert result["constant_mismatches"] == {}
    assert result["ok"] is True


def test_check_run_against_ledger_catches_a_reencode_drift(tmp_path, ledger):
    """47.8: the logged vector must equal what re-encoding the logged obs
    produces -- corrupted here by changing `encoded` without touching `obs`,
    the shape a stale cache or a hand-edited log would take."""
    record = _real_tick_record(0)
    record["encoded"][1] = record["encoded"][1] + 1.0  # ego_speed slot, drifted
    run_dir = _write_run(tmp_path, [record])
    result = check_run_against_ledger(run_dir, ledger)
    assert result["ok"] is False
    assert result["reencode_mismatches"] == [0]


def test_check_run_against_ledger_catches_a_constant_that_stopped_holding(tmp_path, ledger):
    """47.9: `ego_lane` is claimed SUBSTITUTED at a fixed constant
    (`cfg.assumed_lane`); a tick whose logged vector shows a different lane
    is real evidence the mechanism changed, not a fixture artefact -- built
    by re-encoding an `obs` with `ego_lane` moved to 2."""
    record = _real_tick_record(0)
    record["obs"]["ego_lane"] = 2
    record["encoded"] = [round(float(x), 5) for x in encode(record["obs"])]
    run_dir = _write_run(tmp_path, [record])
    result = check_run_against_ledger(run_dir, ledger)
    assert result["ok"] is False
    assert "ego_lane" in result["constant_mismatches"]
    assert result["constant_mismatches"]["ego_lane"] == [0]
    # The re-encode check must still pass -- this tick is internally
    # consistent, just evidence the SUBSTITUTED claim broke.
    assert result["reencode_mismatches"] == []


def test_check_run_against_ledger_refuses_an_empty_run(tmp_path, ledger):
    run_dir = _write_run(tmp_path, [])
    result = check_run_against_ledger(run_dir, ledger)
    assert result["ticks_checked"] == 0
    assert result["ok"] is False


# -- B5 (validation round 1): the harness builds from the real config.yaml -


def test_production_builder_config_reads_the_real_config_yaml():
    """`_production_builder_config` is now a thin wrapper -- read the real
    file, call `BuilderConfig.from_full_config` (B11, validation round 2:
    the SAME classmethod `run_demo.build_components` calls, extracted so
    the merge exists once rather than as a copy at each call site -- see
    `TestBuilderConfigFromFullConfig` in `deployment/jetson/tests/
    test_observation_builder.py` for that merge's own tests). What is
    worth pinning here is narrower and does not re-derive the merge: that
    this function reads THIS repository's real `config.yaml` rather than a
    stale copy or the wrong path.
    """
    import yaml
    from perception.observation_builder import BuilderConfig

    with open(JETSON_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)
    expected = BuilderConfig.from_full_config(config)

    assert _production_builder_config() == expected
    # And, today, the two constructions coincide with bare defaults too --
    # which is exactly why that was worth pinning rather than leaving as a
    # coincidence.
    assert _production_builder_config() == BuilderConfig()
