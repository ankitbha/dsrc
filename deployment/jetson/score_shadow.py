#!/usr/bin/env python3
"""Score candidate sensing controllers against one drive's logged decisions.

Task 30 recorded the decision, task 31 sent it, task 34 gave every rule a
three-state status instead of a silent absence. What none of them built is
the *scorable* half: a way to replay the logged decisions exactly and then ask
what a different controller would have decided against the same inputs. That
is what this tool does, and it refuses before it misleads.

**"Identical traffic" does not mean identical traffic.** A pure shadow drive
never queries HERE at all -- `shadow_mode.py`'s module docstring states this as
a structural fact, not a degraded input -- so `Trigger.DISAGREEMENT` cannot
fire on such a drive and no candidate can be credited or debited on it. What
this tool CAN promise is that every candidate is scored against the same
recorded per-tick inputs, tick for tick, from one drive. It cannot promise the
traffic feed existed, and it cannot promise a candidate's own trajectory --
only the decision function, against inputs the incumbent actually produced.

**The scorer refuses before it misleads.** It first replays the incumbent
`SensingController` from the log alone and requires its output to match the
log byte-for-byte. A log that fails that identity check scores nobody: no
candidate can be refereed by a record that cannot reproduce its own author's
decisions. A log recorded before this task carries no `decision_inputs` at
all and is refused by name rather than approximated from rounded evidence.

**Reproducing the incumbent is not the same claim as an uncorrupted log.**
The replay gate proves the log can reproduce the decisions it recorded; it
does not prove the log is uncorrupted, because a corruption the incumbent's
own thresholds never cross replays identically. Measured on one drive:
rounding latitude to four decimal places (about 11 m) passed the gate on all
800 ticks, and rounding `policy_margin` passed on 601 of 800 ticks, while
rounding `ego_speed` was caught. A candidate with different thresholds may be
sensitive exactly where the incumbent is not.

    python3 deployment/jetson/score_shadow.py <run_dir> \\
        [--candidate label=module:factory ...] [--no-json]

Exit code: 0 = the log replayed exactly (with or without candidates scored),
2 = the log refused to score at all.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

from eval_run import load_records  # noqa: E402
from policy.sensing_controller import (  # noqa: E402
    MAX_TELEMETRY_AGE_S,
    RULE_FIRED,
    RULE_NOT_EVALUABLE,
    RULE_QUIET,
    RULES,
    Inputs,
    SensingController,
    Trigger,
)
from policy.shadow_mode import ABSENT_IN_PURE_SHADOW  # noqa: E402
from transport.messages import RATE_KEYS  # noqa: E402

#: The raise rules `activity.raises` counts. `Trigger.THERMAL` is excluded: it
#: backs rates off rather than asking for more, so it is not a "raise" in the
#: sense this counts.
RAISE_RULES = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT)

#: Echoed verbatim from `shadow_mode.ModeHolder.to_record` when `summary.json`
#: carries the mode block. When it does not, this is the fallback -- the claim
#: is a fixed fact about what a shadow log can predict, not a per-drive
#: measurement, so deriving it from ticks would mean re-deriving a constant.
SHADOW_PREDICTS = "the decision function, not the trajectory"

#: Named refusals. Each means the log cannot be scored at all -- not "scored
#: with a caveat" -- because the thing that failed is the log's ability to
#: referee anyone, including its own incumbent.
REFUSAL_NO_METADATA = "no_metadata_jsonl"
REFUSAL_NO_TICKS = "no_tick_records"
REFUSAL_PHONELESS = "phoneless_run"
REFUSAL_PRE_TASK_35 = "decision_inputs_absent"

#: Why a candidate's own score is refused rather than emitted on rates alone.
#: A candidate that cannot report the same three-state attribution the
#: incumbent does cannot be held to the discipline this task exists for.
CANDIDATE_WITHOUT_ATTRIBUTION = "candidate_without_attribution"


class ReplayClock:
    """A clock fed by the log instead of by time.

    Set once per tick to that tick's `decided_at_mono`, and held rather than
    popped: a candidate may read its clock more than once inside one `decide`
    call, and every such read within a tick must see the same recorded instant.
    """

    def __init__(self) -> None:
        self._current: float | None = None

    def set(self, instant: float) -> None:
        self._current = instant

    def __call__(self) -> float:
        if self._current is None:
            raise RuntimeError("ReplayClock read before the first tick was set")
        return self._current


def _sensing_ticks(ticks: list[dict]) -> list[dict]:
    """Ticks that carry a `sensing` block, in log order."""
    return [t for t in ticks if "sensing" in t]


def _log_refusal(ticks: list[dict], sensing_ticks: list[dict]) -> str | None:
    """Whether this log can be scored at all, before anything is replayed."""
    if not ticks:
        return REFUSAL_NO_TICKS
    if not sensing_ticks:
        return REFUSAL_PHONELESS
    for t in sensing_ticks:
        sensing = t["sensing"]
        if "decision_inputs" not in sensing or "decided_at_mono" not in sensing:
            return REFUSAL_PRE_TASK_35
    return None


def _replay_incumbent(sensing_ticks: list[dict]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the incumbent from the log alone; report whether it matches.

    Returns the `replay_identity` record and, when it matches, the incumbent's
    own replayed `to_record()` per tick -- reused for `vs_incumbent` rather
    than replayed a second time, so the comparison is against literally the
    run the identity check just certified.
    """
    clock = ReplayClock()
    controller = SensingController(clock=clock)
    replayed: list[dict[str, Any]] = []
    mismatched = 0
    first_mismatch: dict[str, Any] | None = None
    for t in sensing_ticks:
        sensing = t["sensing"]
        inputs = Inputs.from_record(sensing["decision_inputs"])
        clock.set(sensing["decided_at_mono"])
        record = controller.decide(inputs).to_record()
        replayed.append(record)
        differing = [k for k, v in record.items() if sensing.get(k) != v]
        if differing:
            mismatched += 1
            if first_mismatch is None:
                first_mismatch = {"tick_id": t.get("tick_id"), "keys": differing}
    status = "ok" if mismatched == 0 else "failed"
    return (
        {
            "status": status,
            "ticks": len(sensing_ticks),
            "mismatched": mismatched,
            "first_mismatch": first_mismatch,
        },
        replayed,
    )


def _segments(sensing_ticks: list[dict]) -> dict[str, Any]:
    """`reference` = ticks strictly before the first live one; the rest is
    `contaminated` (D9). A drive that never went live is reference throughout.
    """
    first_live_index = None
    first_live_tick_id = None
    for i, t in enumerate(sensing_ticks):
        if t["sensing"]["shadow"] is False:
            first_live_index = i
            first_live_tick_id = t.get("tick_id")
            break
    if first_live_index is None:
        return {
            "reference_ticks": len(sensing_ticks),
            "contaminated_ticks": 0,
            "first_live_tick_id": None,
        }
    return {
        "reference_ticks": first_live_index,
        "contaminated_ticks": len(sensing_ticks) - first_live_index,
        "first_live_tick_id": first_live_tick_id,
    }


def _limits(sensing_ticks: list[dict], summary: dict[str, Any] | None) -> dict[str, Any]:
    """What this log can and cannot say about identical traffic, echoed from
    `summary.json` when it exists, or derived from the per-tick flags when it
    does not -- `summary.json` is written only at `close()`, so a truncated
    run may have ticks and no summary, and that must not be a hard dependency.
    """
    mode = None
    if summary is not None:
        mode = summary.get("sensing", {}).get("mode")
    if mode is not None:
        return {
            "shadow_predicts": mode["shadow_predicts"],
            "structurally_absent": list(mode["structurally_absent"]),
            "reference_rates_hold": mode["reference_rates_hold"],
        }
    born_live = bool(sensing_ticks) and sensing_ticks[0]["sensing"]["shadow"] is False
    ever_live = any(t["sensing"]["shadow"] is False for t in sensing_ticks)
    return {
        "shadow_predicts": SHADOW_PREDICTS,
        "structurally_absent": [] if born_live else list(ABSENT_IN_PURE_SHADOW),
        "reference_rates_hold": not ever_live,
        "mode_derived_from_ticks": True,
    }


def _log_completeness(sensing_ticks: list[dict], summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Whether the tick records this scorer read match the count the run
    itself recorded, or `None` when there is nothing to compare against.

    `summary["sensing"]["ticks"]` is `SensingLoop.ticks`: incremented once
    per call to `on_tick`, independent of whatever later made it into
    `metadata.jsonl`. `len(sensing_ticks)` is what this scorer actually read
    back off that log. A run whose log lost its tail after every tick had
    already been decided states a higher count than it can produce records
    for, and nothing before this compared the two.

    This catches a log truncated after a clean `close()` -- an interrupted
    `adb pull` or `scp` off the device, a full disk, a partial copy -- which
    is the likely shape of a log that arrives over a cable rather than one
    still on the Jetson. It does not catch the more common truncation cause,
    `MetadataLogger`'s unflushed write queue when `close()` never runs,
    because in that case `summary.json` itself was never written and this
    comparison has nothing to read: the check is unavailable exactly when
    the loss is largest. `unparseable_lines` does not stand in for it either
    -- that count is 0 or 1 regardless of how many ticks are missing,
    because a record `close()` never wrote leaves no line behind to fail
    parsing.
    """
    if summary is None:
        return None
    recorded = summary.get("sensing", {}).get("ticks")
    if not isinstance(recorded, int):
        return None
    scored = len(sensing_ticks)
    return {"ticks_recorded": recorded, "ticks_scored": scored, "ticks_missing": recorded - scored}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_stale_report(age_s: float) -> bool:
    """Whether a report this old is one the incumbent controller would have
    discarded, by the same predicate `_thermal_scale` applies to the same
    age (`sensing_controller.py`): not finite, or more than
    `MAX_TELEMETRY_AGE_S` away from now in either direction. A witness using
    a different predicate could call a report fresh that the controller
    itself never used, or the reverse.
    """
    return not math.isfinite(age_s) or abs(age_s) > MAX_TELEMETRY_AGE_S


def _reference_witness(sensing_ticks: list[dict]) -> dict[str, Any]:
    """The full-rate reference, witnessed rather than assumed: how many ticks
    the phone actually reported on, how many distinct reports those ticks
    came from, and whether the reports were fresh enough for the incumbent to
    have used them.

    `PhoneLink.telemetry` holds the latest report and is cleared only on
    rebind, so a phone that reported once and then had its telemetry thread
    die looks identical to one reporting every tick if presence is all that
    is read -- `ticks_with_achieved` alone cannot tell the two drives apart.
    `reports` counts the distinct arrival instants (`reference.at_mono`)
    rather than the ticks that echo one: an age recomputed against a fresh
    `now` on every tick does not by itself reveal that the underlying report
    did not change, and undercounts whenever the tick interval is at least
    as long as the telemetry interval. Ticks whose report is stale
    (`_is_stale_report`) are excluded from `achieved_mean` and
    `dropped_final` -- a mean that quietly included them would be a mean
    over readings the decision log itself never used.
    """
    with_achieved = [t["sensing"]["reference"] for t in sensing_ticks
                      if t["sensing"]["reference"]["absent"] is None]
    no_telemetry = sum(1 for t in sensing_ticks if t["sensing"]["reference"]["absent"] is not None)

    known_age = [r for r in with_achieved if r["age_s"] is not None]
    # Only finite ages go into the max and the mean. `max()` and the mean propagate a
    # NaN to the whole field, and `json.dumps` writes it as a bare `NaN`, which strict
    # parsers refuse -- so one unusable age would cost a reader both numbers and the
    # ability to load the file at all. A non-finite age is still counted in
    # `ticks_stale`, which is where it belongs.
    ages = [r["age_s"] for r in known_age if math.isfinite(r["age_s"])]
    reports = len({r["at_mono"] for r in known_age})
    fresh = [r for r in known_age if not _is_stale_report(r["age_s"])]
    stale = [r for r in known_age if _is_stale_report(r["age_s"])]
    assert len(fresh) + len(stale) == len(known_age)

    achieved_mean = (
        {key: _mean([r["achieved"][key] for r in fresh]) for key in RATE_KEYS}
        if fresh else None
    )
    dropped_final = fresh[-1]["dropped"] if fresh else None
    return {
        "ticks_with_achieved": len(with_achieved),
        "ticks_no_telemetry": no_telemetry,
        "reports": reports,
        "age_s_max": max(ages) if ages else None,
        "age_s_mean": _mean(ages),
        "ticks_stale": len(stale),
        # Both computed over fresh (non-stale) reports only -- see the
        # docstring above.
        "achieved_mean": achieved_mean,
        "dropped_final": dropped_final,
    }


def _rules_never_exercised(records: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    """Rules `not_evaluable` on every tick -- named, not folded into agreement.

    A rule missing its input on 100% of a drive is not the same fact as a
    candidate agreeing with the incumbent on it, and this is what keeps the
    two from being reported as the same thing.
    """
    out: list[dict[str, Any]] = []
    for rule in RULES:
        statuses = [r["attribution"]["rules"][rule]["status"] for r in records]
        if total == 0 or statuses.count(RULE_NOT_EVALUABLE) != total:
            continue
        missing_counts: dict[str, int] = {}
        declined_counts: dict[str, int] = {}
        for r in records:
            check = r["attribution"]["rules"][rule]
            for name in check.get("missing", ()):
                missing_counts[name] = missing_counts.get(name, 0) + 1
            if "feed_declined" in check:
                declined_counts[check["feed_declined"]] = declined_counts.get(check["feed_declined"], 0) + 1
        out.append({
            "rule": rule, "ticks": total,
            "missing": missing_counts,
            **({"feed_declined": declined_counts} if declined_counts else {}),
        })
    return out


def _has_valid_attribution(record: dict[str, Any]) -> bool:
    rules = record.get("attribution", {}).get("rules", {})
    if set(rules) != set(RULES):
        return False
    return all(
        rules[rule].get("status") in (RULE_FIRED, RULE_QUIET, RULE_NOT_EVALUABLE)
        for rule in RULES
    )


def _score_candidate(
    sensing_ticks: list[dict], incumbent_records: list[dict[str, Any]], factory: Callable[[Any], Any],
) -> dict[str, Any]:
    clock = ReplayClock()
    candidate = factory(clock)
    records: list[dict[str, Any]] = []
    for t in sensing_ticks:
        sensing = t["sensing"]
        inputs = Inputs.from_record(sensing["decision_inputs"])
        clock.set(sensing["decided_at_mono"])
        records.append(candidate.decide(inputs).to_record())

    if records and not all(_has_valid_attribution(r) for r in records):
        return {"refused": CANDIDATE_WITHOUT_ATTRIBUTION}

    total = len(records)
    rules: dict[str, dict[str, int]] = {}
    for rule in RULES:
        counts = {RULE_FIRED: 0, RULE_QUIET: 0, RULE_NOT_EVALUABLE: 0}
        for r in records:
            counts[r["attribution"]["rules"][rule]["status"]] += 1
        rules[rule] = counts

    rates_same = rates_differ = 0
    first_differ_tick_id = None
    trigger_same = trigger_differ = 0
    per_rule: dict[str, dict[str, int]] = {
        rule: {"agree": 0, "differ": 0, "not_evaluable": 0} for rule in RULES
    }
    # Evaluability is a property of the inputs and is shared by construction
    # (D6/D7): the incumbent and the candidate see the same `Inputs` each
    # tick, so a candidate that reports `fired`/`quiet` where the incumbent
    # reports `not_evaluable` did not see more than the incumbent did -- it
    # did not apply the same "missing input means not_evaluable" discipline.
    # `per_rule` still counts that tick `not_evaluable`, matching every other
    # candidate's denominator; this counts it again, separately, so the
    # divergence is not silently absorbed into an unqualified `quiet`.
    candidate_evaluated_where_incumbent_could_not: dict[str, int] = {rule: 0 for rule in RULES}
    ticks_active = 0
    raises = 0
    for t, candidate_r, incumbent_r in zip(sensing_ticks, records, incumbent_records):
        if candidate_r["rates"] == incumbent_r["rates"]:
            rates_same += 1
        else:
            rates_differ += 1
            if first_differ_tick_id is None:
                first_differ_tick_id = t.get("tick_id")
        if candidate_r["trigger"] == incumbent_r["trigger"]:
            trigger_same += 1
        else:
            trigger_differ += 1
        for rule in RULES:
            incumbent_status = incumbent_r["attribution"]["rules"][rule]["status"]
            candidate_status = candidate_r["attribution"]["rules"][rule]["status"]
            if incumbent_status == RULE_NOT_EVALUABLE:
                per_rule[rule]["not_evaluable"] += 1
                if candidate_status != RULE_NOT_EVALUABLE:
                    candidate_evaluated_where_incumbent_could_not[rule] += 1
            elif incumbent_status == candidate_status:
                per_rule[rule]["agree"] += 1
            else:
                per_rule[rule]["differ"] += 1
        if candidate_r["attribution"]["gates"]["level"] == "active":
            ticks_active += 1
        if any(candidate_r["attribution"]["rules"][rule]["status"] == RULE_FIRED for rule in RAISE_RULES):
            raises += 1

    mean_commanded = {key: _mean([r["rates"][key] for r in records]) for key in RATE_KEYS}
    incumbent_mean_commanded = {key: _mean([r["rates"][key] for r in incumbent_records]) for key in RATE_KEYS}

    return {
        "rules": rules,
        "rules_never_exercised": _rules_never_exercised(records, total),
        "vs_incumbent": {
            "rates": {
                "same": rates_same, "differ": rates_differ,
                "first_differ_tick_id": first_differ_tick_id,
                "mean_commanded": mean_commanded,
                "incumbent_mean_commanded": incumbent_mean_commanded,
            },
            "trigger": {"same": trigger_same, "differ": trigger_differ},
            "per_rule": per_rule,
        },
        "activity": {"ticks_active": ticks_active, "raises": raises},
        "candidate_evaluated_where_incumbent_could_not": candidate_evaluated_where_incumbent_could_not,
    }


def score(run_dir: Path, candidates: Mapping[str, Callable[[Any], Any]] | None = None) -> dict[str, Any]:
    """Everything this tool reports, as a JSON-able dict. No file I/O beyond
    reading the run directory -- `main` owns writing `shadow_score.json`.
    """
    candidates = candidates or {}
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.exists():
        # A run directory this shape -- a phone-side session.jsonl and no
        # Jetson metadata log at all -- has nothing for `load_records` to
        # open. Named refusal, not `load_records`' own FileNotFoundError.
        return {"run": str(run_dir), "refused": REFUSAL_NO_METADATA}
    ticks, _scenario, _timebase, unparseable = load_records(metadata_path)
    sensing_ticks = _sensing_ticks(ticks)

    refusal = _log_refusal(ticks, sensing_ticks)
    if refusal is not None:
        return {"run": str(run_dir), "refused": refusal}

    summary = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    replay_identity, incumbent_records = _replay_incumbent(sensing_ticks)
    segments = _segments(sensing_ticks)
    reference_ticks = sensing_ticks[:segments["reference_ticks"]]
    contaminated_ticks = sensing_ticks[segments["reference_ticks"]:]
    result: dict[str, Any] = {
        "run": str(run_dir),
        "ticks": len(sensing_ticks),
        "unparseable_lines": unparseable,
        "log_completeness": _log_completeness(sensing_ticks, summary),
        "replay_identity": replay_identity,
        "segments": segments,
        "limits": _limits(sensing_ticks, summary),
        # Computed over the reference segment only (D9's own boundary) -- a
        # mean over ticks the mode record itself calls contaminated would be
        # a mean over a segment `reference_rates_hold` says is not the
        # reference. The contaminated segment gets its own witness rather
        # than being pooled into this one or silently dropped. Both are
        # None on a segment of zero ticks, matching each other: `segments`
        # already states the tick count, so a witness here would describe a
        # segment the drive does not have rather than an empty one it does.
        "reference_witness": _reference_witness(reference_ticks) if reference_ticks else None,
        "reference_witness_contaminated": (
            _reference_witness(contaminated_ticks) if contaminated_ticks else None
        ),
    }
    if replay_identity["status"] != "ok":
        # No candidate scores are emitted (D8): a log that cannot reproduce its
        # own incumbent's decisions cannot referee anyone else's.
        return result

    # A property of the log itself, not of any candidate: whether a rule went
    # unexercised over the whole drive. Computed here so it is present even
    # when zero candidates are supplied -- running the tool with none is
    # exactly the log-validity check this states the result of.
    result["rules_never_exercised"] = _rules_never_exercised(incumbent_records, len(incumbent_records))

    result["candidates"] = {
        label: _score_candidate(sensing_ticks, incumbent_records, factory)
        for label, factory in candidates.items()
    }
    return result


def render_table(result: dict[str, Any]) -> str:
    """The stdout table `main` prints beside `shadow_score.json`."""
    if "refused" in result:
        return f"[score_shadow] REFUSED: {result['refused']} ({result['run']})"

    lines = [f"[score_shadow] {result['run']}", f"  ticks: {result['ticks']}"
             f" (unparseable lines: {result['unparseable_lines']})"]
    lc = result["log_completeness"]
    if lc is not None:
        lines.append(f"  log_completeness: recorded={lc['ticks_recorded']} "
                     f"scored={lc['ticks_scored']} missing={lc['ticks_missing']}")
    ri = result["replay_identity"]
    lines.append(f"  replay_identity: {ri['status']} ({ri['mismatched']}/{ri['ticks']} mismatched)")
    if ri["status"] != "ok":
        lines.append(f"    first mismatch: {ri['first_mismatch']}")
        return "\n".join(lines)
    seg = result["segments"]
    lines.append(f"  segments: reference={seg['reference_ticks']} "
                 f"contaminated={seg['contaminated_ticks']} first_live_tick_id={seg['first_live_tick_id']}")
    lim = result["limits"]
    lines.append(f"  limits: shadow_predicts={lim['shadow_predicts']!r}")
    lines.append(f"    structurally_absent={lim['structurally_absent']}")
    lines.append(f"    reference_rates_hold={lim['reference_rates_hold']}")
    rw = result["reference_witness"]
    if rw is not None:
        lines.append(f"  reference_witness (reference segment): {rw['ticks_with_achieved']} ticks with achieved, "
                     f"{rw['ticks_no_telemetry']} with no telemetry, {rw['reports']} reports, "
                     f"{rw['ticks_stale']} stale")
    rwc = result["reference_witness_contaminated"]
    if rwc is not None:
        lines.append(f"  reference_witness (contaminated segment): {rwc['ticks_with_achieved']} ticks "
                     f"with achieved, {rwc['ticks_no_telemetry']} with no telemetry, "
                     f"{rwc['reports']} reports, {rwc['ticks_stale']} stale")
    # Printed unconditionally -- a property of the log itself, present even
    # when no candidates are supplied (D7): running with none is exactly
    # the log-validity check this states the result of.
    for entry in result["rules_never_exercised"]:
        lines.append(f"  RULE NEVER EXERCISED (log-wide): {entry['rule']} "
                     f"(all {entry['ticks']} ticks, missing={entry['missing']})")
    for label, c in result.get("candidates", {}).items():
        lines.append(f"  candidate {label}:")
        if "refused" in c:
            lines.append(f"    REFUSED: {c['refused']}")
            continue
        rates = c["vs_incumbent"]["rates"]
        lines.append(f"    rates vs incumbent: same={rates['same']} differ={rates['differ']} "
                     f"first_differ_tick_id={rates['first_differ_tick_id']}")
        for rule, counts in c["vs_incumbent"]["per_rule"].items():
            lines.append(f"    {rule}: agree={counts['agree']} differ={counts['differ']} "
                         f"not_evaluable={counts['not_evaluable']}")
        if c["rules_never_exercised"]:
            for entry in c["rules_never_exercised"]:
                lines.append(f"    RULE NEVER EXERCISED: {entry['rule']} "
                             f"(all {entry['ticks']} ticks, missing={entry['missing']})")
    return "\n".join(lines)


def _resolve_candidate(spec: str) -> tuple[str, Callable[[Any], Any]]:
    """`label=module:factory` -> `(label, factory)`, importing `module` fresh."""
    label, path = spec.split("=", 1)
    module_name, factory_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return label, getattr(module, factory_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run directory containing metadata.jsonl")
    parser.add_argument("--candidate", action="append", default=[], metavar="label=module:factory",
                         help="a candidate sensing controller; factory(clock) -> object with "
                              ".decide(Inputs) -> Decision-like. Repeatable.")
    parser.add_argument("--no-json", action="store_true", help="do not write shadow_score.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    candidates = dict(_resolve_candidate(spec) for spec in args.candidate)
    result = score(run_dir, candidates)

    print(render_table(result))
    if not args.no_json and "refused" not in result:
        (run_dir / "shadow_score.json").write_text(json.dumps(result, indent=2))
        print(f"[score_shadow] wrote {run_dir / 'shadow_score.json'}")

    if "refused" in result or result["replay_identity"]["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
