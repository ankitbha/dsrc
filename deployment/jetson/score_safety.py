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
#: density derived_empty carve-out). Every run under outputs/task42_usb/ was
#: recorded under this config; a run recorded under a different one would
#: need its own override, which no run in this repository needs today.
DEFAULT_MIN_CONTEXTUAL_SPEED_MPS = 12.0
DEFAULT_DENSITY_MAX_AGE_S = 4.0

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

    census: dict[str, Counter] = {name: Counter() for name in RULE_NAMES}
    replay_mismatches: list[dict[str, Any]] = []
    arm_results: dict[str, dict[str, Any]] = {}

    for arm_name, forced_bin in ARMS:
        clamped = 0
        deltas: list[float] = []
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
            inputs = safety_inputs_from_observation(
                obs_result, state, time_s=float(tick.get("t_wall", 0.0)),
                min_contextual_speed_mps=min_contextual_speed_mps,
                density_max_age_s=density_max_age_s,
            )
            action = _forced_action(tick["action"], forced_bin)
            result = run_safety_gate(action, inputs, state, constraints)

            if arm_name == "recorded":
                for name in RULE_NAMES:
                    census[name][result.rules[name].status] += 1
                recorded_safety = tick.get("safety")
                if has_safety_block and recorded_safety is not None:
                    if result.to_record() != recorded_safety:
                        replay_mismatches.append({"tick_id": tick.get("tick_id")})

            if result.delta_speed_mps != 0.0:
                clamped += 1
                deltas.append(result.delta_speed_mps)
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

        arm_results[arm_name] = {
            "n_ticks": len(ticks),
            "clamped_ticks": clamped,
            "clamped_fraction": clamped / len(ticks),
            "mean_delta_speed_mps": (sum(deltas) / len(deltas)) if deltas else None,
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
            lines.append(
                f"  {name}: evaluable on {entry['evaluable_ticks']} of {entry['total_ticks']} ticks; "
                f"fired on {entry['fired_ticks']} of {entry['evaluable_ticks']} evaluable "
                f"({entry['fired_fraction_of_evaluable']:.1%})"
            )

    lines.append("")
    lines.append("-- counterfactual arms --")
    for arm_name, arm in result["arms"].items():
        mean = arm["mean_delta_speed_mps"]
        mean_str = f"{mean:+.2f} m/s" if mean is not None else "n/a"
        lines.append(
            f"  {arm_name}: clamped {arm['clamped_ticks']} of {arm['n_ticks']} "
            f"({arm['clamped_fraction']:.1%}), mean delta {mean_str}, "
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
