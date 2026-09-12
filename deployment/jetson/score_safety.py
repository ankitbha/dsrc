#!/usr/bin/env python3
"""Score the safety/etiquette gate's per-rule evaluability against recorded runs.

Task 30/34's `score_shadow.py` established the idiom this follows: refuse
before misleading. Applied here to plan_task144's own finding -- that a
"clamp rate" computed from a rig whose rules were never evaluable is a
statement about the fallback values, not about traffic -- this tool:

  1. Reports the per-rule EVALUABILITY CENSUS first: how many of the loaded
     ticks each of the twelve rules could actually be checked on. A rule
     evaluable on zero ticks is reported as "not evaluable on any tick" with
     no percentage at all -- printing "0% firing rate" for a rule that was
     never evaluable is exactly the defect this task exists to avoid.

  2. Refuses before it misleads. A log that already carries a `safety` block
     (written by a pipeline with this task's gate wired in) must have that
     block reproduced EXACTLY from its own recorded `obs`/`field_sources`/
     `obs_diagnostics`/`action` before anything is reported -- the same
     incumbent-replay gate `score_shadow.py` runs against `SensingController`.
     A log with no `safety` block at all (every run recorded before this
     task, including every run under outputs/task42_usb/) has every number
     in its output labelled COUNTERFACTUAL: nothing on that rig ever ran the
     gate, so this is "what it would have done", not what happened.

  3. Reports three counterfactual arms over the SAME loaded ticks: the
     recorded action, and the action with `desired_speed_bin` forced to
     `nominal` and to `slow` -- plan_task144 E5's demonstration that the
     first two arms clamp on 0 of N ticks while the third clamps on all of
     them, because `low_speed_uncongested` reduces algebraically to
     `desired_speed_bin == "slow"` once `local_density_veh_per_km` is not
     evidence (which it is not, on any tick this rig has recorded).

Every SafetyContext input is reconstructable from a tick record's `obs`,
`obs_diagnostics`, `field_sources` and `action` alone, so this needs no
video and no device -- it is a pure function of `metadata.jsonl`.

    python3 deployment/jetson/score_safety.py <run_dir> [<run_dir> ...] [--no-json]

Exit code: 0 = scored (with or without a `safety` block to replay against),
2 = refused (no ticks loaded, or a `safety` block failed to reproduce).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import numpy as np  # noqa: E402

from eval_run import load_records  # noqa: E402
from perception.observation_builder import ObservationResult  # noqa: E402
from policy.safety_gate import (  # noqa: E402
    RULE_ALIASED_PROVENANCE_FIELDS,
    RULE_COMPARED_VALUE_KEY,
    RULE_FIRED,
    RULE_NAMES,
    RULE_NOT_EVALUABLE,
    RULE_QUIET,
    SafetyConstraints,
    SafetyState,
    run_safety_gate,
    safety_inputs_from_observation,
)

#: This tool's own defaults for the two inputs `safety_inputs_from_observation`
#: needs beyond what a tick record carries -- config.yaml's current values
#: (`policy.min_contextual_speed_mps`, and `2 * gps.stale_after_s` for the
#: density derived_empty carve-out). Used ONLY for a tick with no `safety`
#: block at all (a run recorded before task 144 shipped the gate); a tick
#: that carries one always replays with ITS OWN recorded `safety.config`
#: instead (validator round 1, F3/F4) -- see `_tick_gate_config`.
DEFAULT_MIN_CONTEXTUAL_SPEED_MPS = 12.0
DEFAULT_DENSITY_MAX_AGE_S = 4.0

#: validator round 1, Fix 4: the five keys `safety.config` must carry for
#: this tool to replay a tick's own recorded configuration rather than
#: assuming its own defaults. F3's bug was `withhold_lane_when_not_evaluable`
#: never being read back at all (this tool always replayed with the
#: default True, so it could not read a run recorded with the flag off);
#: F4's was reconstructing `time_s` as the tick's epoch `t_wall` when the
#: live pipeline passes `time.monotonic()` -- a ~1.79e9 s discrepancy,
#: invisible only because it reaches the record solely through
#: `lane_change_dwell`, not_evaluable on every tick in this corpus.
REQUIRED_SAFETY_CONFIG_KEYS: tuple[str, ...] = (
    "enabled", "withhold_lane_when_not_evaluable", "time_s",
    "min_contextual_speed_mps", "density_max_age_s",
)

#: plan_task144 E5's three arms: the action as recorded, and the same action
#: with desired_speed_bin forced -- everything else (headway, lane
#: preference, merge mode) untouched.
ARMS: tuple[tuple[str, str | None], ...] = (
    ("recorded", None),
    ("forced_nominal", "nominal"),
    ("forced_slow", "slow"),
)


def _observation_result_from_tick(tick: dict[str, Any]) -> ObservationResult:
    """The three fields safety_inputs_from_observation reads, taken straight
    from a logged tick record. `encoded` is unused downstream of this call
    and is never logged in full precision, so it is a placeholder here.
    """
    return ObservationResult(
        obs=tick["obs"],
        encoded=np.zeros(1, dtype=np.float32),
        field_sources=tick["field_sources"],
        diagnostics=tick["obs_diagnostics"],
        feed=None,
    )


def _forced_action(action: dict[str, str], desired_speed_bin: str | None) -> dict[str, str]:
    if desired_speed_bin is None:
        return dict(action)
    forced = dict(action)
    forced["desired_speed_bin"] = desired_speed_bin
    return forced


def _tick_gate_config(tick: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """This tick's own `safety.config`, or the name of the first key missing
    from it, or `(None, None)` for a tick with no `safety` block at all.

    Three cases, not two: a tick with no `safety` block predates task 144
    entirely (the caller falls back to this tool's own module defaults --
    ordinary COUNTERFACTUAL reporting, unaffected by this fix); a `safety`
    block with no `config` key, or missing one of `REQUIRED_SAFETY_CONFIG_
    KEYS`, is a record from after task 144 shipped but before Fix 4 added
    this block -- refused, naming the missing key, rather than silently
    assumed; a complete `config` is replayed verbatim.
    """
    safety = tick.get("safety")
    if safety is None:
        return None, None
    config = safety.get("config")
    if config is None:
        return None, "config"
    for key in REQUIRED_SAFETY_CONFIG_KEYS:
        if key not in config:
            return None, key
    return config, None


def score_ticks(
    ticks: list[dict[str, Any]],
    *,
    min_contextual_speed_mps: float = DEFAULT_MIN_CONTEXTUAL_SPEED_MPS,
    density_max_age_s: float = DEFAULT_DENSITY_MAX_AGE_S,
) -> dict[str, Any]:
    """The whole report: the evaluability census (arm-independent -- rule
    evaluability is a fact about the INPUTS, not about which action bin is
    being tested) plus the three E5 arms, or a refusal.
    """
    if not ticks:
        return {"refused": "no_ticks"}

    constraints = SafetyConstraints()
    has_safety_block = any(t.get("safety") is not None for t in ticks)

    # validator round 1, Fix 4: refuse up front, before anything is reported,
    # when a tick carries a `safety` block that predates this fix -- naming
    # the missing key rather than silently assuming a default for it (a
    # `safety.config` that reads `withhold_lane_when_not_evaluable: false`
    # is exactly the run this tool most needs to be able to read).
    for tick in ticks:
        _, missing_key = _tick_gate_config(tick)
        if missing_key is not None:
            return {
                "refused": "safety_config_missing_key",
                "missing_key": missing_key,
                "tick_id": tick.get("tick_id"),
            }

    census: dict[str, Counter] = {name: Counter() for name in RULE_NAMES}
    # validator round 1, F5: how many of each rule's EVALUABLE ticks carry a
    # non-finite compared value (see RULE_COMPARED_VALUE_KEY) -- arm-
    # independent, like the census itself, computed only on the "recorded"
    # arm below.
    non_finite_compared: dict[str, int] = {name: 0 for name in RULE_COMPARED_VALUE_KEY}
    # validator round 1, F5 (provenance-consistency finding): how many of
    # each rule's evaluable ticks disagree with the field_sources entry
    # they are defined to alias -- see RULE_ALIASED_PROVENANCE_FIELDS.
    stale_provenance: dict[str, int] = {name: 0 for name in RULE_ALIASED_PROVENANCE_FIELDS}
    replay_mismatches: list[dict[str, Any]] = []
    arm_results: dict[str, dict[str, Any]] = {}

    for arm_name, forced_bin in ARMS:
        clamped = 0
        # validator round 1, F6: kept apart by direction -- a pooled mean
        # cannot say whether the gate raised or lowered the recommended
        # speed, and "clamped" alone reads as a reduction regardless.
        raised_deltas: list[float] = []
        lowered_deltas: list[float] = []
        emergency_ticks = 0
        lane_withheld_ticks = 0
        event_counter: Counter = Counter()

        for tick in ticks:
            obs_result = _observation_result_from_tick(tick)
            # A fresh SafetyState every tick: nothing on this rig advances the
            # lane-change counters (no lane-change detector exists), so the
            # running state a live pipeline would carry never differs from
            # the class default -- see safety_gate.py's class (C) list.
            state = SafetyState()
            # This tick's OWN recorded configuration when it has one (Fix
            # 4), never this tool's module defaults -- a tick with no
            # `safety` block at all (pre-task-144) is the one case those
            # defaults are still for.
            tick_config, _ = _tick_gate_config(tick)
            if tick_config is not None:
                tick_time_s = tick_config["time_s"]
                tick_min_contextual_speed_mps = tick_config["min_contextual_speed_mps"]
                tick_density_max_age_s = tick_config["density_max_age_s"]
                tick_withhold = tick_config["withhold_lane_when_not_evaluable"]
                tick_enabled = tick_config["enabled"]
            else:
                tick_time_s = float(tick.get("t_wall", 0.0))
                tick_min_contextual_speed_mps = min_contextual_speed_mps
                tick_density_max_age_s = density_max_age_s
                tick_withhold = True
                tick_enabled = True
            inputs = safety_inputs_from_observation(
                obs_result, state, time_s=tick_time_s,
                min_contextual_speed_mps=tick_min_contextual_speed_mps,
                density_max_age_s=tick_density_max_age_s,
            )
            action = _forced_action(tick["action"], forced_bin)
            result = run_safety_gate(
                action, inputs, state, constraints,
                withhold_lane_when_not_evaluable=tick_withhold, enabled=tick_enabled,
            )

            if arm_name == "recorded":
                for name in RULE_NAMES:
                    status = result.rules[name].status
                    census[name][status] += 1
                    compared_key = RULE_COMPARED_VALUE_KEY.get(name)
                    if compared_key is not None and status != RULE_NOT_EVALUABLE:
                        compared_value = result.rules[name].evidence.get(compared_key)
                        if compared_value is not None and not math.isfinite(compared_value):
                            non_finite_compared[name] += 1
                    alias = RULE_ALIASED_PROVENANCE_FIELDS.get(name)
                    if alias is not None and status != RULE_NOT_EVALUABLE:
                        own_key, aliased_key = alias
                        own_source = tick["field_sources"].get(own_key)
                        aliased_source = tick["field_sources"].get(aliased_key)
                        if own_source != aliased_source:
                            stale_provenance[name] += 1
                recorded_safety = tick.get("safety")
                if has_safety_block and recorded_safety is not None:
                    if result.to_record() != recorded_safety:
                        replay_mismatches.append({"tick_id": tick.get("tick_id")})

            if result.delta_speed_mps != 0.0:
                clamped += 1
                if result.delta_speed_mps > 0.0:
                    raised_deltas.append(result.delta_speed_mps)
                else:
                    lowered_deltas.append(result.delta_speed_mps)
                # Attributed from apply_safety_layer's own (unchanged) elif-chain
                # diagnostics, matching plan_task144 E5's own methodology --
                # NOT from the independent per-rule census, which correctly
                # reports low_speed_uncongested as not_evaluable even on a
                # tick the raw decision clamped using a substituted density.
                for events in result.raw_diagnostics.get("etiquette_blocked_action", []):
                    event_counter[f"etiquette_blocked_action:{events['reason']}"] += 1
                for events in result.raw_diagnostics.get("safety_masked_action", []):
                    event_counter[f"safety_masked_action:{events['reason']}"] += 1
            if result.emergency_override:
                emergency_ticks += 1
            if result.lane_withheld is not None:
                lane_withheld_ticks += 1

        all_deltas = raised_deltas + lowered_deltas
        arm_results[arm_name] = {
            "n_ticks": len(ticks),
            "clamped_ticks": clamped,
            "clamped_fraction": clamped / len(ticks),
            # Kept for existing consumers: the pooled mean across BOTH
            # directions, which is exactly what F6 says cannot state a
            # direction on its own -- prefer raised_ticks/lowered_ticks and
            # their own means below.
            "mean_delta_speed_mps": (sum(all_deltas) / len(all_deltas)) if all_deltas else None,
            "raised_ticks": len(raised_deltas),
            "lowered_ticks": len(lowered_deltas),
            "mean_raise_mps": (sum(raised_deltas) / len(raised_deltas)) if raised_deltas else None,
            "mean_lower_mps": (sum(lowered_deltas) / len(lowered_deltas)) if lowered_deltas else None,
            "events": dict(event_counter),
            "emergency_override_ticks": emergency_ticks,
            "lane_withheld_ticks": lane_withheld_ticks,
        }

    if has_safety_block and replay_mismatches:
        return {
            "refused": "incumbent_replay_mismatch",
            "n_mismatches": len(replay_mismatches),
            "mismatches": replay_mismatches[:10],
        }

    rule_census: dict[str, Any] = {}
    for name in RULE_NAMES:
        counts = census[name]
        evaluable = counts[RULE_FIRED] + counts[RULE_QUIET]
        entry: dict[str, Any] = {
            "total_ticks": len(ticks),
            "evaluable_ticks": evaluable,
            "not_evaluable_ticks": counts[RULE_NOT_EVALUABLE],
        }
        if evaluable == 0:
            entry["rate"] = None
            entry["note"] = "not evaluable on any tick"
        else:
            entry["fired_ticks"] = counts[RULE_FIRED]
            entry["fired_fraction_of_evaluable"] = counts[RULE_FIRED] / evaluable
            # validator round 1, F5: a fired-fraction computed entirely over
            # ticks whose compared value is non-finite is not a firing rate
            # -- the threshold could never physically be crossed.
            if name in RULE_COMPARED_VALUE_KEY:
                entry["non_finite_compared_ticks"] = non_finite_compared[name]
            # validator round 1, F5 (provenance-consistency finding): named
            # rather than silently trusted or silently excluded.
            if name in RULE_ALIASED_PROVENANCE_FIELDS:
                entry["stale_provenance_ticks"] = stale_provenance[name]
        rule_census[name] = entry

    return {
        "n_ticks": len(ticks),
        # True for every run recorded before this task: nothing on that rig
        # ever ran the gate, so every number below describes what the gate
        # WOULD have done, not what the driver saw.
        "counterfactual": not has_safety_block,
        "replayed_incumbent": has_safety_block and not replay_mismatches,
        "rule_census": rule_census,
        "arms": arm_results,
    }


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if "refused" in result:
        lines.append(f"REFUSED: {result['refused']}")
        if result["refused"] == "safety_config_missing_key":
            lines.append(
                f"  missing key: {result['missing_key']!r} "
                f"(tick_id={result.get('tick_id')}) -- this run predates "
                "validator round 1's Fix 4 and cannot be replayed without "
                "assuming a value this tool refuses to guess."
            )
        return "\n".join(lines)

    lines.append(f"n_ticks: {result['n_ticks']}")
    if result["counterfactual"]:
        lines.append(
            "COUNTERFACTUAL: no `safety` block in these logs -- nothing below "
            "describes what the rig showed a driver, only what the gate would "
            "have decided against the same recorded inputs."
        )
    else:
        lines.append("replayed the incumbent `safety` block exactly on every tick.")

    lines.append("")
    lines.append("-- per-rule evaluability census --")
    for name, entry in result["rule_census"].items():
        if entry["rate"] is None if "rate" in entry else "fired_ticks" not in entry:
            lines.append(
                f"  {name}: not evaluable on any tick "
                f"(0 of {entry['total_ticks']})"
            )
        else:
            line = (
                f"  {name}: evaluable on {entry['evaluable_ticks']} of {entry['total_ticks']} ticks; "
                f"fired on {entry['fired_ticks']} of {entry['evaluable_ticks']} evaluable "
                f"({entry['fired_fraction_of_evaluable']:.1%})"
            )
            # validator round 1, F5: name it when the fraction above cannot
            # move -- the compared value was non-finite on some or all of
            # the ticks the rule was otherwise evaluable on.
            non_finite = entry.get("non_finite_compared_ticks")
            if non_finite:
                compared_key = RULE_COMPARED_VALUE_KEY[name]
                if non_finite == entry["evaluable_ticks"]:
                    line += f"; the compared value ({compared_key}) is inf on all {non_finite}"
                else:
                    line += (
                        f"; the compared value ({compared_key}) is inf on "
                        f"{non_finite} of {entry['evaluable_ticks']} evaluable"
                    )
            # validator round 1, F5 (provenance-consistency finding): print
            # these ticks as what they are -- still counted evaluable above
            # (that is what today's evidence rule says of the RECORDED
            # source), but flagged, not silently trusted or dropped.
            stale = entry.get("stale_provenance_ticks")
            if stale:
                own_key, aliased_key = RULE_ALIASED_PROVENANCE_FIELDS[name]
                if stale == entry["evaluable_ticks"]:
                    line += (
                        f"; all {stale} disagree with {aliased_key}'s own source "
                        f"({own_key} is defined to alias it verbatim) -- recorded "
                        "under a superseded contract, not evidence under today's rule"
                    )
                else:
                    line += (
                        f"; {stale} of {entry['evaluable_ticks']} evaluable disagree "
                        f"with {aliased_key}'s own source ({own_key} is defined to "
                        "alias it verbatim) -- recorded under a superseded contract"
                    )
            lines.append(line)

    lines.append("")
    lines.append("-- counterfactual arms --")
    for arm_name, arm in result["arms"].items():
        # validator round 1, F6: named quantity, named direction. "clamped"
        # alone reads as a reduction; it is a speed INCREASE on this rig's
        # only reachable path (low_speed_uncongested), and a pooled mean
        # across both directions cannot say which one moved.
        raise_clause = (
            f"recommended speed raised on {arm['raised_ticks']} of {arm['n_ticks']} ticks "
            f"(mean +{arm['mean_raise_mps']:.2f} m/s)" if arm["raised_ticks"]
            else f"recommended speed raised on 0 of {arm['n_ticks']} ticks"
        )
        lower_clause = (
            f"lowered on {arm['lowered_ticks']} of {arm['n_ticks']} ticks "
            f"(mean {arm['mean_lower_mps']:.2f} m/s)" if arm["lowered_ticks"]
            else f"lowered on 0 of {arm['n_ticks']} ticks"
        )
        lines.append(
            f"  {arm_name}: {raise_clause}; {lower_clause}; "
            f"events={arm['events']}, emergency_override={arm['emergency_override_ticks']}, "
            f"lane_withheld={arm['lane_withheld_ticks']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--no-json", action="store_true", help="skip the JSON dump, report only")
    args = parser.parse_args()

    ticks: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        metadata_path = run_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            print(f"[score_safety] no metadata.jsonl under {run_dir}", file=sys.stderr)
            return 2
        loaded = load_records(metadata_path)
        ticks.extend(loaded.ticks)

    result = score_ticks(ticks)
    if not args.no_json:
        print(json.dumps(result, indent=2, default=str))
    print(render_report(result), file=sys.stderr)
    return 2 if "refused" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
