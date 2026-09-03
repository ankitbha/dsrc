#!/usr/bin/env python3
"""Evaluate a logged run: did the pipeline actually perform well?

Post-processes a run directory's metadata.jsonl (live, simulated-drive,
or replay) into report.md / report.json plus timeline plots, and checks
a small set of PASS/FAIL gates:

  latency      jetson p95 < 200 ms (plan_deployment.md headline target)
  throughput   median tick rate >= 25 Hz (file sources are paced at 30)
  gps          >= 95% of ticks with a fresh fix outside scripted dropouts
  gps_speed    ego-speed RMSE vs the scripted profile < 1.0 m/s
               (simulated runs only - catches unit/staleness wiring bugs)
  perception   >= 1 tracked vehicle in >= 50% of ticks (traffic footage)

Advisory content is reported but never gated: with an UNTRAINED bundle
the actions are arbitrary by construction, and even trained actions have
no ground truth here. What this tool certifies is the *plumbing*:
sensors -> observation -> actor -> advisory at real-time rates.

  python3 eval_run.py ~/dsrc_logs/run_20260612_153000
  python3 eval_run.py <run_dir> --no-plots

Exit code: 0 = all applicable gates pass, 2 = at least one failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import numpy as np  # noqa: E402

from perception import provenance  # noqa: E402
from policy import sim_contract  # noqa: E402
from policy.sensing_controller import (  # noqa: E402
    MAX_TELEMETRY_AGE_S,
    RULE_FIRED,
    RULE_NOT_EVALUABLE,
    RULE_QUIET,
    RULES,
)
from policy.shadow_mode import LIVE, SHADOW  # noqa: E402
from sensors.thermal import ABSENT_REASONS, THERMAL_BASIS_STALE  # noqa: E402
from sensors.time_sync import STAGE_BASIS_MEASURED, StageTiming  # noqa: E402
from transport.messages import RATE_KEYS  # noqa: E402
from transport.timebase import (  # noqa: E402
    MAX_ACCEPTABLE_RTT_NS,
    MIN_OFFSET_SAMPLES,
    TimebaseEstimate,
)

#: How far apart two devices' `timebase_estimate` and advisory-receipt wall
#: stamps may sit and still be treated as describing the same moment in the
#: link's history. Both clocks are NTP-locked, so this is generous against
#: NTP's own accuracy rather than a tight tolerance.
ESTIMATE_MATCH_WINDOW_S = 30.0

# The gate is on the ON-JETSON segment, not on end-to-end. The threshold is a
# claim about what this hardware can do, so charging it for a link the Jetson
# does not control would make a run fail for the network's behaviour. It would
# also loosen silently: when the timebase cannot convert a capture stamp the link
# segment drops out of e2e, so the same gate would quietly get easier exactly
# when the timing is least trustworthy. For a local camera the two coincide.
GATE_JETSON_P95_MS = 200.0
GATE_MIN_RATE_HZ = 25.0
GATE_GPS_FRESH_FRACTION = 0.95
GATE_GPS_SPEED_RMSE_MPS = 1.0
GATE_VEHICLE_TICK_FRACTION = 0.50
#: A ratio, not a raw tick count, so the threshold reads the same regardless
#: of a drive's length: `_tick_coverage`'s missing_ticks over its own
#: expected_ticks (actual plus missing) -- the fraction of the drive's
#: expected span that a gap this large or bigger would have to have cost.
GATE_TICK_COVERAGE_MISSING_FRACTION = 0.05
DROPOUT_RECOVERY_MARGIN_S = 2.5  # stale_after_s + one fix interval


@dataclass(frozen=True)
class LoadedRecords:
    """Every record `load_records` sorts a `metadata.jsonl` line into, one
    field per record type.

    A frozen dataclass rather than the seven-wide tuple this replaced (D18):
    the tuple was six wide and unpacked positionally at two call sites, three
    of its members were already `list[dict]` and a fourth would be --
    adjacent, same-typed, and a mis-ordered unpack type-checks and passes most
    tests. Growth from here on is a new field, not a new position.
    """

    ticks: list[dict]
    scenario: dict | None
    timebase_estimates: list[dict]
    unparseable: int
    thermal_samples: list[dict]
    thermal_events: list[dict]
    failure_scans: list[dict] = field(default_factory=list)
    failure_events: list[dict] = field(default_factory=list)


def load_records(metadata_path: Path) -> LoadedRecords:
    """Every record type `metadata.jsonl` carries, sorted by `type`, plus how
    many lines would not parse at all.

    The count is returned rather than swallowed. `MetadataLogger` buffers a mebibyte
    and flushes only in `close()`, and there is a path where `close()` never runs --
    so the last line of a run is routinely a half-written record, and at ~1.5 KB each
    an unflushed buffer is some seven hundred ticks, not one. A reader that skips
    them silently analyses a truncated run as a complete one.
    """
    ticks: list[dict] = []
    scenario: dict | None = None
    timebase_estimates: list[dict] = []
    thermal_samples: list[dict] = []
    thermal_events: list[dict] = []
    failure_scans: list[dict] = []
    failure_events: list[dict] = []
    unparseable = 0
    with open(metadata_path) as f:
        for line in f:
            try:
                record = json.loads(line)  # Python json accepts Infinity literals
            except ValueError:
                unparseable += 1
                continue
            record_type = record.get("type")
            if record_type == "tick":
                ticks.append(record)
            elif record_type == "scenario":
                scenario = record
            elif record_type == "timebase_estimate":
                timebase_estimates.append(record)
            elif record_type == "thermal_sample":
                thermal_samples.append(record)
            elif record_type == "thermal_event":
                thermal_events.append(record)
            elif record_type == "failure_scan":
                failure_scans.append(record)
            elif record_type == "failure_event":
                failure_events.append(record)
    return LoadedRecords(
        ticks=ticks, scenario=scenario, timebase_estimates=timebase_estimates,
        unparseable=unparseable, thermal_samples=thermal_samples, thermal_events=thermal_events,
        failure_scans=failure_scans, failure_events=failure_events,
    )


def load_phone_log(phone_log_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Inbound advisory lines, advisory_shown lines, and failure lines from a
    phone's session log, ignoring everything else the file holds.

    Every outbound line `SessionLog` writes is a bare frame header -- the
    canonical JSON that went on the wire, verbatim, with no wrapper -- so it
    carries no `dir` key at all. The three line shapes this reader wants all
    do, which is what tells them apart from an outbound line and from each
    other without this reader having to know every line shape a later task
    adds.
    """
    inbound_advisories: list[dict] = []
    shown: list[dict] = []
    failures: list[dict] = []
    with open(phone_log_path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            direction = record.get("dir")
            if direction == "in" and record.get("header", {}).get("ch") == "advisory":
                inbound_advisories.append(record)
            elif direction == "shown":
                shown.append(record)
            elif direction == "fail":
                failures.append(record)
    return inbound_advisories, shown, failures


def _reconstruct_estimate(record: dict) -> TimebaseEstimate:
    """A `TimebaseEstimate` from its persisted `to_record()`, the re-derivation
    the estimate_id on every converted stamp promises is possible."""
    fields = {
        k: v for k, v in record.items()
        if k in TimebaseEstimate.__dataclass_fields__
    }
    return TimebaseEstimate(**fields)


def _was_usable(record: dict) -> bool:
    """Whether the live adapter would have converted against this estimate.

    `usable` was added to the persisted line alongside `why_not_usable`, both
    read off the estimator at the moment it was written. A run logged before
    that field existed carries neither, and staleness is the one gate
    condition that is genuinely unrecoverable from an old-format line: it is a
    question about how long ago the newest sample arrived relative to *now*,
    at the moment the estimator was asked, and that moment was never
    captured. The sample count and the RTT ceiling are not like that --
    `offset_samples` and `rtt_min_ns` are properties of the estimate itself,
    written into every persisted line whatever its format, so both gate
    conditions can still be applied.

    The RTT ceiling is a round-trip-estimator concept: on a one-way line
    `rtt_min_ns` is a delay spread, not a round trip, and `OneWayEstimator`
    has no ceiling clause on it at all. Applying the round-trip bound there
    would refuse estimates the live one-way path accepts.
    """
    if "usable" in record:
        return bool(record["usable"])
    if record.get("offset_samples", 0) < MIN_OFFSET_SAMPLES:
        return False
    if record.get("source") != "round_trip":
        return True
    rtt = record.get("rtt_min_ns")
    return rtt is None or rtt <= MAX_ACCEPTABLE_RTT_NS


def _nearest_estimate(
    timebase_estimates: list[dict], *, source: str, near_wall_ns: int, session_id: Any = None,
) -> TimebaseEstimate | None:
    """The persisted estimate of the given source closest in Jetson wall time
    to `near_wall_ns`, or None when there is none within the match window.

    Wall time, not monotonic: the two logs come from two devices whose
    monotonic clocks share no origin, so nothing else lets an offline reader
    line up "the estimate that was current then" across the two files. Both
    devices are NTP-locked -- the same premise `phone_source.py` states for
    using wall stamps as a correlation key, never as the latency measurement
    itself, which is why the actual conversion below still runs on monotonic
    nanoseconds through the estimate's own arithmetic.

    Estimates the live adapter would have refused to convert against are
    excluded outright, and so are estimates from a different session: both
    estimators are rebuilt whole on every redial, so wall time alone cannot
    tell a stale estimate from the previous peer apart from a current one.
    """
    candidates = [
        e for e in timebase_estimates
        if e.get("source") == source
        and e.get("session_id") == session_id
        and _was_usable(e)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda e: abs(int(e["t_wall"] * 1e9) - near_wall_ns))
    if abs(int(best["t_wall"] * 1e9) - near_wall_ns) > int(ESTIMATE_MATCH_WINDOW_S * 1e9):
        return None
    return _reconstruct_estimate(best)


def _return_stage(
    header: dict, receipt: dict, timebase_estimates: list[dict], *, session_id: Any = None,
) -> StageTiming:
    """Advisory wire departure (Jetson clock) to phone receipt (phone clock),
    converted offline against whichever persisted estimate was nearest in
    wall time -- round-trip preferred, one-way as a fallback, exactly the
    adapter's own live preference order.
    """
    wire_ns = header.get("t_wire_mono_ns")
    if wire_ns is None:
        return StageTiming.absent(clock="cross", reason="advisory carried no wire stamp")
    recv_ns = receipt.get("recv_mono_ns")
    recv_wall_ns = receipt.get("recv_wall_ns")
    if recv_ns is None or recv_wall_ns is None:
        return StageTiming.absent(clock="cross", reason="phone log did not record the receipt")

    for source in ("round_trip", "one_way"):
        estimate = _nearest_estimate(
            timebase_estimates, source=source, near_wall_ns=recv_wall_ns, session_id=session_id,
        )
        if estimate is None:
            continue
        converted = estimate.convert_to_local(recv_ns)
        if converted is None:
            continue
        return_ms = (converted.t_remote_mono_ns - wire_ns) / 1e6
        return StageTiming.converted(
            return_ms, bound_ms=converted.bound_ns / 1e6,
            estimate_id=converted.estimate_id, source=source,
        )
    return StageTiming.absent(
        clock="cross", reason="no usable timebase estimate near this receipt's wall time"
    )


def _render_stage(receipt: dict, shown_record: dict | None) -> StageTiming:
    """Phone receipt to the first `current()` call that returned this
    advisory -- both phone-clock, so no conversion is needed at all.

    Absent when nothing in the phone log ever marked this advisory shown.
    That is not necessarily an expiry: `AdvisoryHolder.accept` replaces
    `latest` unconditionally, so the ordinary cause is a newer advisory
    arriving before anything polled this one, which the Jetson does every
    tick against a 250 ms UI poll. A dropped `SessionLog` line or a null
    `liveLog` produce the same absence here and cannot be told apart from
    supersession by this join alone.
    """
    if shown_record is None:
        return StageTiming.absent(
            clock="phone", reason="no advisory_shown line for this capture stamp"
        )
    recv_ns = receipt.get("recv_mono_ns")
    shown_ns = shown_record.get("shown_mono_ns")
    if recv_ns is None or shown_ns is None:
        return StageTiming.absent(clock="phone", reason="phone log is missing a stamp")
    return StageTiming.measured((shown_ns - recv_ns) / 1e6, clock="phone")


#: Used for both a `return` and a `render` stage on a Jetson tick whose
#: advisory the phone never logged as received -- the join has nothing more
#: specific to say than that the far side of the exchange is missing.
NO_ADVISORY_FOR_TICK_REASON = "the phone logged no advisory for this frame"


def join_phone_log(
    ticks: list[dict],
    timebase_estimates: list[dict],
    inbound_advisories: list[dict],
    shown: list[dict],
) -> dict[str, Any]:
    """The ten-stage table: each Jetson tick's own eight stages, plus `return`
    and `render` -- facts only the phone witnesses -- joined on the exact
    nanosecond key `AdvisoryMessage.t_capture_mono_ns` carries on both ends of
    the exchange (`run_phone_drive.py` already joins the same way).

    The join is checked in both directions. An inbound advisory with no
    matching tick is counted under `unmatched` rather than dropped silently.
    A Jetson tick whose advisory never came back is the opposite miss --
    `jetson_ticks_with_no_advisory` counts them, and each one still gets a
    row, with `return` and `render` recorded `absent`, so its eight Jetson
    stages are not discarded along with the two the phone would have
    supplied. Without this, `rows` -- the population every per-stage
    statistic in the report is computed over -- silently excludes every tick
    whose return trip failed, which is not a random sample: a tick goes
    unmatched exactly when the link was down, so the surviving population is
    conditioned on the link having worked.
    """
    ticks_by_capture_ns = {
        t["t_capture_mono_ns"]: t for t in ticks if "t_capture_mono_ns" in t
    }
    shown_by_capture_ns: dict[int, dict] = {}
    for record in shown:
        capture_ns = record.get("t_capture_mono_ns")
        if capture_ns is not None:
            shown_by_capture_ns[capture_ns] = record

    rows: list[dict[str, Any]] = []
    matched_capture_ns: set[int] = set()
    unmatched = 0
    for record in inbound_advisories:
        header = record.get("header", {})
        capture_ns = header.get("t_capture_mono_ns")
        tick = ticks_by_capture_ns.get(capture_ns)
        if tick is None:
            unmatched += 1
            continue
        matched_capture_ns.add(capture_ns)
        stages = dict(tick.get("stages", {}))
        stages["return"] = _return_stage(
            header, record, timebase_estimates, session_id=tick.get("session_id"),
        ).to_record()
        stages["render"] = _render_stage(
            record, shown_by_capture_ns.get(capture_ns)
        ).to_record()
        rows.append({
            "t_capture_mono_ns": capture_ns,
            "tick_id": tick.get("tick_id"),
            "stages": stages,
        })

    # `matched` counts rows produced above -- one per advisory that found a
    # tick, same meaning as before this join was checked in both directions.
    matched = len(rows)

    absent_return = StageTiming.absent(clock="cross", reason=NO_ADVISORY_FOR_TICK_REASON).to_record()
    absent_render = StageTiming.absent(clock="phone", reason=NO_ADVISORY_FOR_TICK_REASON).to_record()
    for capture_ns, tick in ticks_by_capture_ns.items():
        if capture_ns in matched_capture_ns:
            continue
        stages = dict(tick.get("stages", {}))
        stages["return"] = dict(absent_return)
        stages["render"] = dict(absent_render)
        rows.append({
            "t_capture_mono_ns": capture_ns,
            "tick_id": tick.get("tick_id"),
            "stages": stages,
        })

    return {
        "advisories_seen_by_the_phone": len(inbound_advisories),
        "matched": matched,
        "unmatched": unmatched,
        "jetson_ticks_with_no_advisory": len(ticks_by_capture_ns) - len(matched_capture_ns),
        "rows": rows,
    }


def pctl(values: list[float]) -> dict[str, float]:
    """mean/p50/p95/max, plus the count and both extremes.

    `n` and `min` were missing, and for the link segment that mattered: a
    converted capture stamp may legitimately land after the arrival it preceded,
    so the segment can be negative -- and with only mean/p50/p95/max reported, a
    negative merely lowered the p95 and was otherwise invisible. `negative`
    counts them outright, so the case where the bound is being tested is not
    something a reader has to infer.
    """
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0,
                "min": 0.0, "negative": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "negative": int((arr < 0).sum()),
    }


def fmt_row(name: str, s: dict[str, float]) -> str:
    """Includes n, min and the negative count.

    Those were added "so the case where the bound is being tested is not
    something a reader has to infer" -- and then rendered nowhere, so they sat in
    report.json while report.md showed mean/p50/p95/max. A field a reader never
    sees is a field that does not exist.
    """
    negative = s.get("negative", 0)
    flag = f" ({negative} negative)" if negative else ""
    return (
        f"| {name} | {s.get('n', 0)} | {s.get('min', 0.0):.1f} | {s['mean']:.1f} | "
        f"{s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f}{flag} |"
    )


#: The loop's stages in the order the signal travels, so the table reads as a path
#: rather than as an alphabetical list. `return` and `render` come from the phone join
#: and are absent from a run with no phone log.
STAGE_ORDER = (
    "capture", "capture_to_encode_start", "encode", "encode_done_to_enqueue",
    "enqueue_to_wire", "transport", "jpeg_decode", "detect", "track", "fuse",
    "infer", "decode", "return", "render",
)


def stage_timings(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-stage durations, kept apart from how each one was arrived at.

    A stage is `measured` on one clock, `converted` across two with a bound,
    `absent` with a named reason, or an `instant` that is a point rather than a
    duration. The distinction is the whole point of the table: a stage that could
    not be measured must not average in as though it were, and must not show a
    zero. So the statistics are computed over the values that exist, `n` says how
    many that was, and the basis counts say what the rest were.
    """
    out: dict[str, Any] = {}
    for tick in ticks:
        for name, entry in (tick.get("stages") or {}).items():
            slot = out.setdefault(
                name, {"values": [], "basis": {}, "absent_reasons": {}, "bounds": []}
            )
            basis = entry.get("basis")
            slot["basis"][basis] = slot["basis"].get(basis, 0) + 1
            if basis == "absent":
                why = entry.get("reason") or "unstated"
                slot["absent_reasons"][why] = slot["absent_reasons"].get(why, 0) + 1
                continue
            if basis == "instant":
                # A point in time carries `ms: 0.0` as a placeholder, not a duration of
                # zero. Averaging it in reports a stage that took no time, which is the
                # one thing this table exists to make impossible.
                continue
            ms = entry.get("ms")
            if ms is not None:
                slot["values"].append(float(ms))
            bound = entry.get("bound_ms")
            if bound is not None:
                slot["bounds"].append(float(bound))
    for name, slot in out.items():
        slot["stats"] = pctl(slot.pop("values")) if slot["values"] else None
        slot["bound_ms"] = pctl(slot.pop("bounds")) if slot["bounds"] else None
    return out


def thermal_result(
    ticks: list[dict[str, Any]], thermal_samples: list[dict], thermal_events: list[dict],
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    """The thermal section: the summary's own rollup plus what only the raw
    records can say. `None` when the log carries no thermal records or
    summary block at all -- not a failed drive, the same reading
    `latency["jetson_ms_source"]` gives a run that predates the jetson/e2e
    split.

    `ticks_by_basis` counts the Jetson's own per-tick `basis` -- the phone's
    tick block carries no single `basis` field, so it is not pooled in here.
    It is also the real signal for a sampler thread that died partway through
    a run: `sample_gaps_s` cannot show that, because a gap exists only between
    two samples that were actually written, so a sampler that stops writing
    simply produces a shorter list of gaps that are each still close to 1.0 s
    -- the stall shows up as ticks reporting `stale` for the rest of the run,
    which is what `ticks_by_basis` counts.
    """
    has_tick_block = any(t.get("thermal") is not None for t in ticks)
    if not has_tick_block and not thermal_samples and not thermal_events and "thermal" not in summary:
        return None

    ticks_by_basis: dict[str, int] = {}
    for t in ticks:
        block = t.get("thermal")
        if not block:
            continue
        basis = (block.get("jetson") or {}).get("basis")
        if basis is not None:
            ticks_by_basis[basis] = ticks_by_basis.get(basis, 0) + 1

    ordered = sorted(r["t_mono"] for r in thermal_samples if "t_mono" in r)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]

    return {
        "summary": summary.get("thermal"),
        "ticks_by_basis": ticks_by_basis,
        "sample_gaps_s": pctl(gaps) if gaps else None,
        "events": thermal_events,
    }


def _join_failure_episodes(failure_events: list[dict]) -> list[dict]:
    """Pair every `open` record with its `close`, by `episode_id`.

    An `open` with no matching `close` is kept, with `closed=False` -- the
    reading rule is that this means the log was truncated (`stop()` closes
    every open episode before `to_record()` runs), not that the episode is
    still open. A validator checks this against `summary["failures"]`
    directly; this function only reports what pairs and what does not.
    """
    opens = {r["episode_id"]: r for r in failure_events if r.get("phase") == "open"}
    closes = {r["episode_id"]: r for r in failure_events if r.get("phase") == "close"}
    episodes = []
    for episode_id, open_record in sorted(opens.items()):
        close_record = closes.get(episode_id)
        episodes.append({
            "episode_id": episode_id, "source": open_record.get("source"),
            "reason": open_record.get("reason"), "device": open_record.get("device"),
            "opened_t_mono": open_record.get("t_mono"),
            "closed": close_record is not None,
            "outcome": close_record.get("outcome") if close_record else None,
            "duration_s": close_record.get("duration_s") if close_record else None,
            # `n` is occurrences over the WHOLE episode -- only the close
            # record says that, so an episode with none is `None` rather than
            # standing in the movement on its opening pass alone, which
            # `first_pass_n` already names on its own key. The two used to
            # share this one key, so a truncated log's still-open episode
            # read as if it had definitely closed after exactly one pass's
            # worth of occurrences.
            "n": close_record.get("n") if close_record else None,
            "first_pass_n": open_record.get("first_pass_n"),
        })
    return episodes


def _read_summary(run_dir: Path) -> dict[str, Any]:
    """`summary.json`'s contents, or `{}` when it was never written -- a run
    that did not reach `close()` (run_demo.py:708 is the last statement
    before it) has no summary at all, and every reader downstream of this
    treats that the same way it treats a summary missing one key.
    """
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def _read_log_health(run_dir: Path) -> dict[str, Any] | None:
    """`log_health.json`'s contents, or `None` when it was never written or
    did not parse. Written after `close()`, so it can be absent even on a
    run whose `summary.json` exists.
    """
    health_path = run_dir / "log_health.json"
    if health_path.exists():
        try:
            return json.loads(health_path.read_text())
        except ValueError:
            return None
    return None


def failures_result(
    ticks: list[dict[str, Any]], loaded: "LoadedRecords", summary: dict[str, Any],
    run_dir: Path | None = None,
) -> dict[str, Any] | None:
    """The failures section: the sampler's own summary rollup plus what only
    the raw records can say. `None` when the log carries no failure records
    and no summary block at all -- not a failed drive, the same reading
    `thermal_result` gives a run that predates it.
    """
    has_tick_block = any(t.get("failures") is not None for t in ticks)
    log_health = None if run_dir is None else _read_log_health(run_dir)
    if not has_tick_block and not loaded.failure_scans and not loaded.failure_events \
            and "failures" not in summary and log_health is None:
        return None

    ordered = sorted(r["t_mono"] for r in loaded.failure_scans if "t_mono" in r)
    scan_gaps = [b - a for a, b in zip(ordered, ordered[1:])]

    ticks_seen_values = [
        r["ticks_seen"] for r in loaded.failure_scans if r.get("ticks_seen") is not None
    ]

    ticks_by_basis: dict[str, int] = {}
    for t in ticks:
        block = t.get("failures")
        if not block:
            continue
        basis = block.get("basis")
        if basis is not None:
            ticks_by_basis[basis] = ticks_by_basis.get(basis, 0) + 1

    return {
        "summary": summary.get("failures"),
        "scan_gaps_s": pctl(scan_gaps) if scan_gaps else None,
        "ticks_seen_per_pass": (
            {"p50": pctl(ticks_seen_values)["p50"], "min": min(ticks_seen_values)}
            if ticks_seen_values else None
        ),
        "ticks_by_basis": ticks_by_basis,
        "episodes": _join_failure_episodes(loaded.failure_events),
        "log_health": log_health,
    }


def in_dropout_affected(elapsed_s: float, dropouts: list[tuple[float, float]]) -> bool:
    return any(a <= elapsed_s < b + DROPOUT_RECOVERY_MARGIN_S for a, b in dropouts)


def _log_integrity(
    ticks: list[dict[str, Any]], summary: dict[str, Any], unparseable: int,
) -> dict[str, Any]:
    """What the run said it produced against what survived to be read. The
    summary is written by `write_summary` before `close()` flushes the log,
    so a run that lost its buffer states its own tick count beside a file
    that is short of it, and this check compares the two directly rather
    than by subtraction against anything else.

    This compares the log against what the run itself reported producing --
    it has no way to see a tick the run never produced at all. An
    interruption that stops tick production reduces `ticks_read` and
    `summary["ticks"]` together, so `tick_ids_absent_from_log` reads zero on
    a drive that lost a real span of ticks to an outage; `_tick_coverage`,
    built from the gap distribution rather than a reported count, is what
    catches that.

    `log_complete` is `True` when every tick the run reported is present and
    every line parsed; `False` when the log is short of that count or a line
    failed to parse; and `None` when there is nothing to compare against at
    all (`summary["ticks"]` absent, or not a positive int) -- a third state,
    distinct from both, because a drive whose completeness cannot be
    measured is not the same fact as one measured complete and must not
    read as a pass.
    """
    expected_ticks = summary.get("ticks")
    shortfall = None
    if isinstance(expected_ticks, int) and expected_ticks > 0:
        shortfall = expected_ticks - len(ticks)
    if unparseable != 0 or (shortfall is not None and shortfall != 0):
        log_complete = False
    elif shortfall is None:
        log_complete = None
    else:
        log_complete = True
    return {
        "ticks_read": len(ticks),
        "ticks_the_run_reported": expected_ticks,
        "tick_ids_absent_from_log": shortfall,
        "unparseable_lines": unparseable,
        "log_complete": log_complete,
    }


def analyze(
    run_dir: Path, phone_log_path: Path | None = None, *, loaded: "LoadedRecords | None" = None,
) -> dict[str, Any]:
    """All metrics + gates as a JSON-able dict (report rendering is separate).

    `phone_log_path` is optional: a run with no phone behind it, or one whose
    phone log was not pulled off the handset, still analyses fully on the
    eight Jetson-side stages every tick already carries. When it is supplied,
    the two logs are joined into a ten-stage table that adds `return` and
    `render`, the two facts only the phone witnesses.

    `loaded` lets a caller that has already read `metadata.jsonl` (`main()`,
    which needs it before deciding whether this function can even run) pass
    the same `LoadedRecords` in rather than paying for a second parse.
    """
    if loaded is None:
        loaded = load_records(run_dir / "metadata.jsonl")
    ticks = loaded.ticks
    scenario, timebase_estimates = loaded.scenario, loaded.timebase_estimates
    unparseable = loaded.unparseable
    thermal_samples, thermal_events = loaded.thermal_samples, loaded.thermal_events
    if not ticks:
        # A drive that produced no ticks at all -- the camera never delivered a
        # frame -- is exactly the drive with the most failures, and this task
        # does not make analysis tolerate it (open item 1). What it does add:
        # naming what the log holds anyway, so the exit message is not silent
        # about the one thing this task guarantees survives a zero-tick run.
        raise SystemExit(
            f"no tick records in {run_dir / 'metadata.jsonl'} "
            f"({len(loaded.failure_scans)} failure_scan, {len(loaded.failure_events)} "
            "failure_event records present)"
        )
    summary = _read_summary(run_dir)
    integrity = _log_integrity(ticks, summary, unparseable)

    t_wall = np.array([t["t_wall"] for t in ticks])
    duration_s = float(t_wall[-1] - t_wall[0]) if len(ticks) > 1 else 0.0
    periods = np.diff(t_wall)
    rate_hz = 1.0 / float(np.median(periods)) if len(periods) else 0.0

    # --- latency ---------------------------------------------------------
    latency = {"e2e_ms": pctl([t["e2e_ms"] for t in ticks])}
    # jetson_ms is absent from runs recorded before the split; fall back to e2e,
    # which is what it meant on a local camera anyway, and say so in the record
    # rather than reporting a zero.
    jetson_samples = [t["jetson_ms"] for t in ticks if t.get("jetson_ms") is not None]
    latency["jetson_ms"] = pctl(jetson_samples or [t["e2e_ms"] for t in ticks])
    latency["jetson_ms_source"] = (
        "measured" if jetson_samples else "absent from this run; e2e used"
    )
    link_samples = [t["link_ms"] for t in ticks if t.get("link_ms") is not None]
    latency["link_ms"] = pctl(link_samples) if link_samples else None
    for stage in ticks[0]["stage_ms"]:
        latency[stage + "_ms"] = pctl([t["stage_ms"][stage] for t in ticks])

    # --- perception ------------------------------------------------------
    n_vehicles = [len(t.get("vehicles", [])) for t in ticks]
    leader_gaps = [t["obs"]["leader_gap"] for t in ticks]
    leader_present = [math.isfinite(g) for g in leader_gaps]
    finite_gaps = [g for g in leader_gaps if math.isfinite(g)]
    rel_measured = [
        t["field_sources"].get("leader_relative_speed") == "measured"
        for t, present in zip(ticks, leader_present)
        if present
    ]
    method_counts: Counter[str] = Counter()
    track_spans: dict[int, list[int]] = defaultdict(list)
    for t in ticks:
        for v in t.get("vehicles", []):
            method_counts[v["method"]] += 1
            track_spans[v["id"]].append(t["tick_id"])
    lifetimes_s = [
        (max(span) - min(span) + 1) / max(rate_hz, 1e-9) for span in track_spans.values()
    ]
    perception = {
        "ticks_with_vehicle_fraction": float(np.mean([n > 0 for n in n_vehicles])),
        "mean_vehicles_per_tick": float(np.mean(n_vehicles)),
        "leader_present_fraction": float(np.mean(leader_present)),
        # `None`, not a zeroed-out `pctl([])`, when no tick ever had a leader
        # or a track: a gap of 0.0 m states the vehicle ahead is touching the
        # bumper, and `stage_timings` already avoids this exact shape for the
        # same reason.
        "leader_gap_m": pctl(finite_gaps) if finite_gaps else None,
        "leader_rel_speed_measured_fraction": (
            float(np.mean(rel_measured)) if rel_measured else 0.0
        ),
        "distance_method_counts": dict(method_counts),
        "unique_tracks": len(track_spans),
        "track_lifetime_s": pctl(lifetimes_s) if lifetimes_s else None,
    }

    # --- observation quality ----------------------------------------------
    missingness = [t["obs_diagnostics"]["missingness"] for t in ticks]
    fallback_counter: Counter[str] = Counter()
    for t in ticks:
        fallback_counter.update(t["obs_diagnostics"].get("fallback_fields", []))

    # Every tick back to the beginning carries `field_sources`, so this reads
    # cleanly off a run recorded before this task too -- it just reports
    # fewer fields and `covers_encoder: False`, the `jetson_ms_source`
    # precedent for a run whose shape predates the field being measured.
    #
    # `by_source`/`fields_by_source` pool every tick's `field_sources`
    # regardless of its size, so a run whose maps are not all the same size
    # (a 33-field prefix followed by a 39-field tail, say) needs to say so
    # rather than reporting the first tick's size as if it applied to all of
    # them -- `covers_encoder` in particular is not a meaningful yes/no over
    # a mixture. `provenance_fields` is the size of the FIRST tick's map
    # (matching the "first tick has {pf}" wording below), recorded for every
    # tick including one whose map is empty -- a tick that carries no
    # provenance is part of the mixture, not an exception to it.
    provenance_fields: int | None = None
    provenance_field_sizes: set[int] = set()
    provenance_field_names: set[str] = set()
    by_source_counter: Counter[str] = Counter()
    fields_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for t in ticks:
        field_sources = t.get("field_sources") or {}
        provenance_field_sizes.add(len(field_sources))
        if provenance_fields is None:
            provenance_fields = len(field_sources)
        provenance_field_names.update(field_sources)
        for field_name, source in field_sources.items():
            by_source_counter[source] += 1
            fields_by_class[source][field_name] += 1
    total_field_ticks = sum(by_source_counter.values())
    provenance_fields_mixed = len(provenance_field_sizes) > 1
    # Coverage is decided by NAME, mirroring
    # `ObservationBuilder._covers_encoder` -- a map with the right number of
    # keys but the wrong ones (a real slot missing, a name the encoder never
    # reads standing in for it) is not coverage, and a count comparison
    # cannot tell the two apart.
    covers_encoder = (
        None if provenance_fields is None
        else False if provenance_fields_mixed
        else provenance_field_names == set(sim_contract.encoded_slot_names())
    )
    observation = {
        "missingness": pctl(missingness),
        "top_fallback_fields": {
            k: round(c / len(ticks), 3) for k, c in fallback_counter.most_common(8)
        },
        "provenance_fields": provenance_fields,
        "provenance_fields_mixed": provenance_fields_mixed,
        "covers_encoder": covers_encoder,
        "by_source": (
            {k: round(c / total_field_ticks, 3) for k, c in by_source_counter.items()}
            if total_field_ticks else {}
        ),
        "fields_by_source": {
            source: {field: round(c / len(ticks), 3) for field, c in counter.most_common(8)}
            for source, counter in fields_by_class.items()
        },
    }

    # --- gps ---------------------------------------------------------------
    gps_fresh = [bool(t["obs_diagnostics"]["gps_fresh"]) for t in ticks]
    gps_metrics: dict[str, Any] = {"fresh_fraction_overall": float(np.mean(gps_fresh))}
    sim_truth = None
    if scenario is not None and scenario.get("gps_profile"):
        from sensors.gps_sim import GpsSimProfile, GpsSimulator

        profile = GpsSimProfile.from_spec(scenario["gps_profile"])
        sim = GpsSimulator(profile)
        start_wall = float(scenario["gps_start_wall"])
        dropouts = [(float(a), float(b)) for a, b in profile.dropouts_s]
        elapsed = t_wall - start_wall
        measured = np.array([t["obs"]["ego_speed"] for t in ticks])
        truth = np.array([sim.speed_at(float(e)) for e in elapsed])
        clean = np.array(
            [not in_dropout_affected(float(e), dropouts) for e in elapsed]
        ) & np.array(gps_fresh)
        outside = np.array([not in_dropout_affected(float(e), dropouts) for e in elapsed])
        err = measured - truth
        gps_metrics.update(
            {
                "scripted_dropouts_s": dropouts,
                "fresh_fraction_outside_dropouts": (
                    float(np.mean(np.asarray(gps_fresh)[outside])) if outside.any() else 1.0
                ),
                "speed_rmse_mps": float(np.sqrt(np.mean(err[clean] ** 2))) if clean.any() else None,
                "speed_max_abs_err_mps": float(np.max(np.abs(err[clean]))) if clean.any() else None,
                "speed_max_drift_during_dropout_mps": (
                    float(np.max(np.abs(err[~outside]))) if (~outside).any() else None
                ),
            }
        )
        sim_truth = {"elapsed": elapsed, "truth": truth, "measured": measured, "dropouts": dropouts}

    # --- policy / advisory --------------------------------------------------
    head_dists: dict[str, Counter] = defaultdict(Counter)
    switches = 0
    prev_action = None
    for t in ticks:
        action = t["action"]
        for head, value in action.items():
            head_dists[head][value] += 1
        if prev_action is not None:
            switches += sum(1 for h in action if action[h] != prev_action[h])
        prev_action = action
    adv_speeds = [t["advisory"]["recommended_speed_mps"] for t in ticks]
    confidence_labels = Counter(t["advisory"]["confidence_label"] for t in ticks)
    advisory = {
        "trained_policy": bool(summary.get("policy_trained", False)),
        "head_distributions": {
            h: {k: round(c / len(ticks), 3) for k, c in dist.items()}
            for h, dist in head_dists.items()
        },
        "recommended_speed_mps": pctl(adv_speeds),
        "head_switches_per_minute": (
            switches / (duration_s / 60.0) if duration_s > 0 else 0.0
        ),
        "confidence_labels": {k: round(c / len(ticks), 3) for k, c in confidence_labels.items()},
    }

    # --- gates ---------------------------------------------------------------
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, value, threshold: str, ok: bool | None) -> None:
        gates[name] = {"value": value, "threshold": threshold, "pass": ok}

    gate(
        "latency_jetson_p95",
        round(latency["jetson_ms"]["p95"], 1),
        f"< {GATE_JETSON_P95_MS:.0f} ms",
        latency["jetson_ms"]["p95"] < GATE_JETSON_P95_MS,
    )
    gate(
        "throughput_median",
        round(rate_hz, 1),
        f">= {GATE_MIN_RATE_HZ:.0f} Hz",
        rate_hz >= GATE_MIN_RATE_HZ,
    )
    fresh_frac = gps_metrics.get("fresh_fraction_outside_dropouts", gps_metrics["fresh_fraction_overall"])
    gps_used = any(t["gps"]["valid"] for t in ticks)
    gate(
        "gps_fresh",
        round(fresh_frac, 3),
        f">= {GATE_GPS_FRESH_FRACTION}",
        fresh_frac >= GATE_GPS_FRESH_FRACTION if gps_used else None,
    )
    rmse = gps_metrics.get("speed_rmse_mps")
    gate(
        "gps_speed_rmse",
        round(rmse, 3) if rmse is not None else None,
        f"< {GATE_GPS_SPEED_RMSE_MPS} m/s",
        rmse < GATE_GPS_SPEED_RMSE_MPS if rmse is not None else None,
    )
    gate(
        "perception_coverage",
        round(perception["ticks_with_vehicle_fraction"], 3),
        f">= {GATE_VEHICLE_TICK_FRACTION}",
        perception["ticks_with_vehicle_fraction"] >= GATE_VEHICLE_TICK_FRACTION,
    )
    # A check on tick PRODUCTION rather than on any one instrument: every
    # other gate is a ratio over the ticks that exist, which is blind by
    # construction to a stretch where none were produced at all, because
    # that destroys the numerator and denominator together. `None` (fewer
    # than three ticks, or the run's clock never advanced) is not applicable
    # rather than a manufactured pass or fail.
    coverage = _tick_coverage(ticks)
    coverage_fraction = (
        coverage["missing_ticks"] / coverage["expected_ticks"] if coverage else None
    )
    gate(
        "tick_coverage",
        round(coverage_fraction, 4) if coverage_fraction is not None else None,
        f"<= {GATE_TICK_COVERAGE_MISSING_FRACTION} missing/expected",
        coverage_fraction <= GATE_TICK_COVERAGE_MISSING_FRACTION if coverage_fraction is not None else None,
    )
    applicable = [g["pass"] for g in gates.values() if g["pass"] is not None]
    overall = all(applicable) if applicable else False

    phone_join = None
    phone_failures: list[dict] | None = None
    if phone_log_path is not None:
        inbound_advisories, shown, phone_failures = load_phone_log(phone_log_path)
        phone_join = join_phone_log(ticks, timebase_estimates, inbound_advisories, shown)
    failures = failures_result(ticks, loaded, summary, run_dir)
    if phone_log_path is not None:
        if failures is None:
            # The Jetson-side log predates the failure event log entirely --
            # no tick block, no scan or event records, no summary, no
            # log_health.json -- but the phone's own failures were pulled
            # independently of the run directory (D11) and must not be
            # dropped just because the Jetson side has nothing to add.
            failures = {"phone": phone_failures, "jetson_predates_failure_log": True}
        else:
            failures["phone"] = phone_failures
    return {
        "run_dir": str(run_dir),
        "scenario": {
            "path": scenario.get("scenario_path") if scenario else None,
            "description": scenario.get("description") if scenario else None,
            "video_source": scenario.get("video_source") if scenario else None,
        },
        "n_ticks": len(ticks),
        "duration_s": round(duration_s, 1),
        "tick_rate_hz_median": round(rate_hz, 2),
        "camera_dropped_frames": summary.get("camera_dropped_frames"),
        "log_integrity": integrity,
        "latency_ms": latency,
        "perception": perception,
        "observation": observation,
        "gps": gps_metrics,
        "advisory": advisory,
        "gates": gates,
        # A run whose log is short did not pass; it was not fully read. Folded into
        # the verdict rather than reported beside it, because a field nobody looks at
        # is the failure this whole check exists to close. `log_complete` can also be
        # `None` (nothing to compare against) -- coerced to a plain bool here so an
        # unmeasured drive reads as a fail, not as `null` sitting where a verdict goes.
        "overall_pass": bool(overall and integrity["log_complete"]),
        "phone_join": phone_join,
        # Sourced from the joined rows when a phone log was supplied, because those
        # carry `return` and `render` as well -- the two stages only the phone
        # witnesses. Without one, the Jetson-side stages every tick already holds are
        # the whole table, and the two phone stages are simply not in it rather than
        # present and empty.
        "stage_timings": stage_timings(
            phone_join["rows"] if phone_join and phone_join.get("rows") else ticks
        ),
        "thermal": thermal_result(ticks, thermal_samples, thermal_events, summary),
        "failures": failures,
        "_sim_truth": sim_truth,  # stripped before JSON dump
        "_ticks": ticks,
    }


# ---------------------------------------------------------------------------


def render_plots(result: dict[str, Any], run_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ticks = result["_ticks"]
    t0 = ticks[0]["t_wall"]
    ts = np.array([t["t_wall"] - t0 for t in ticks])
    written = []

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    # The gated series belongs on the chart that carries the gate's line. It
    # plotted only e2e while drawing a threshold for jetson -- a target line for
    # a series that was not on the figure.
    jetson = [t.get("jetson_ms") for t in ticks]
    if all(v is not None for v in jetson):
        ax.plot(ts, jetson, lw=0.9, label="jetson (gated)")
    link = [t.get("link_ms") for t in ticks]
    if any(v is not None for v in link):
        ax.plot(ts, [v if v is not None else float("nan") for v in link],
                lw=0.7, ls=":", label="link")
    ax.plot(ts, [t["e2e_ms"] for t in ticks], lw=0.7, label="e2e")
    ax.plot(ts, [t["stage_ms"]["detect"] for t in ticks], lw=0.7, label="detect")
    ax.axhline(GATE_JETSON_P95_MS, color="r", ls="--", lw=0.8, label="200 ms target")
    ax.set_xlabel("run time (s)"); ax.set_ylabel("latency (ms)")
    ax.set_ylim(0, max(60.0, 1.1 * max(t["e2e_ms"] for t in ticks)))
    ax.legend(loc="upper right", fontsize=8); ax.set_title("Latency timeline")
    fig.tight_layout(); fig.savefig(run_dir / "eval_latency.png"); plt.close(fig)
    written.append("eval_latency.png")

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    ax.plot(ts, [t["obs"]["ego_speed"] for t in ticks], lw=1.0, label="ego speed (obs)")
    sim_truth = result["_sim_truth"]
    if sim_truth is not None:
        ax.plot(sim_truth["elapsed"], sim_truth["truth"], lw=1.0, ls="--", label="scripted truth")
        for a, b in sim_truth["dropouts"]:
            ax.axvspan(a, b, color="orange", alpha=0.25)
    ax.plot(
        ts, [t["advisory"]["recommended_speed_mps"] for t in ticks],
        lw=0.8, alpha=0.8, label="advisory speed",
    )
    ax.set_xlabel("run time (s)"); ax.set_ylabel("m/s")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Ego speed vs scripted GPS truth (shaded = scripted dropout)")
    fig.tight_layout(); fig.savefig(run_dir / "eval_speed.png"); plt.close(fig)
    written.append("eval_speed.png")

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    gaps = np.array([t["obs"]["leader_gap"] for t in ticks])
    gaps = np.where(np.isfinite(gaps), gaps, np.nan)
    ax.plot(ts, gaps, lw=0.9, label="leader gap (m)")
    ax2 = ax.twinx()
    rels = []
    for t in ticks:
        lead = t["obs_diagnostics"].get("leader_track_id")
        rel = next(
            (v["rel_mps"] for v in t.get("vehicles", []) if v["id"] == lead and v["rel_mps"] is not None),
            np.nan,
        )
        rels.append(rel)
    ax2.plot(ts, rels, lw=0.7, color="tab:red", alpha=0.7, label="leader rel speed (m/s)")
    ax2.axhline(0.0, color="tab:red", lw=0.4, alpha=0.4)
    ax.set_xlabel("run time (s)"); ax.set_ylabel("gap (m)"); ax2.set_ylabel("rel speed (m/s)")
    ax.set_title("Leader gap / relative speed")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "eval_leader.png"); plt.close(fig)
    written.append("eval_leader.png")
    return written


def _thermal_lines(thermal: dict[str, Any] | None) -> list[str]:
    """The `## Thermal` section: a measurement with nowhere to go is as good
    as unmeasured, so this prints the sampler's own summary rather than
    leaving it in summary.json for nobody to read. Absent entirely when
    `thermal` itself is `None` -- the log has no thermal records at all.

    A log whose sampler wrote records but whose summary was never written
    (the run did not reach its normal teardown) still gets a section, built
    from `ticks_by_basis` and `events` -- the raw records, not the summary
    `None` would otherwise hide entirely.
    """
    if not thermal:
        return []
    lines = ["", "## Thermal", ""]
    s = thermal.get("summary")

    if not s:
        lines.append(
            "- thermal records exist for this drive but no summary was written "
            "(the run likely did not reach its normal teardown)"
        )
    else:
        jetson = s.get("jetson") or {}
        temp = jetson.get("temp_c")
        basis = jetson.get("basis_counts") or {}
        if temp is not None:
            per_zone_max = jetson.get("per_zone_max_c") or {}
            hottest = max(per_zone_max, key=per_zone_max.get) if per_zone_max else None
            zones_seen = jetson.get("zones_seen") or []
            zones_missing = jetson.get("zones_missing") or {}
            # Every zone that ever listed, whether or not it ever gave a
            # plausible reading -- a zone that always refused would
            # otherwise leave no trace in this count at all.
            never_read = sorted(
                set(zones_missing.get("unreadable") or {}) | set(zones_missing.get("implausible") or {})
            )
            total_zones = len(zones_seen) + len(never_read)
            never_read_note = f" ({', '.join(never_read)} never read)" if never_read else ""
            peak = (
                f"; {len(zones_seen)} of {total_zones} zones read{never_read_note}, hottest at peak "
                f"{hottest} {per_zone_max[hottest]:.1f} C" if hottest else ""
            )
            lines.append(
                f"- jetson {jetson.get('selected_zone')}: p50 {temp['p50']:.1f} C, "
                f"p95 {temp['p95']:.1f} C, max {temp['max']:.1f} C over "
                f"{jetson.get('samples', 0)} samples (measured {basis.get('measured', 0)}, "
                f"absent {basis.get('absent', 0)}{peak})"
            )
        else:
            lines.append(f"- jetson: no temperature reached over {jetson.get('samples', 0)} samples")

        phone = s.get("phone") or {}
        n_phone = phone.get("samples", 0)
        status_counts = phone.get("status_counts") or {}
        if status_counts:
            # Every status seen, not only the modal one -- a build that spent
            # 78 of 178 reports `severe` must not render as if it never left
            # `nominal`.
            breakdown = ", ".join(
                f"{status} {n}" for status, n in sorted(status_counts.items(), key=lambda kv: -kv[1])
            )
            skin = phone.get("skin_temp_c")
            skin_str = (
                f"; skin {phone.get('skin_zone')} p50 {skin['p50']:.1f} C, max {skin['max']:.1f} C"
                if skin else ""
            )
            lines.append(f"- phone: {breakdown} of {n_phone} reports{skin_str}")
        headroom_absent = phone.get("headroom_absent_counts") or {}
        if headroom_absent:
            lines.append(
                f"- phone headroom: not reported on {sum(headroom_absent.values())} of "
                f"{n_phone} reports ({', '.join(sorted(headroom_absent))})"
            )

        events = s.get("events") or {}
        for device in ("jetson", "phone"):
            ev = events.get(device) or {}
            status, count = ev.get("status"), ev.get("count", 0)
            if status == RULE_NOT_EVALUABLE:
                missing = ", ".join(ev.get("missing") or [])
                passes = (
                    f" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)"
                    if ev.get("passes_attempted") else ""
                )
                lines.append(
                    f"- throttle events, {device}: NOT EVALUABLE -- missing {missing}{passes}; "
                    f"this drive says nothing about whether the {device} throttled"
                )
            elif status == RULE_FIRED:
                # `missing` can accompany `fired` too (a real transition observed
                # while some cooling device on the same pass, or a later pass,
                # never gave a reading) -- named here so the count is not read as
                # a complete picture when it is not.
                passes = (
                    f" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)"
                    if ev.get("missing") else ""
                )
                lines.append(f"- throttle events, {device}: fired -- {count} transitions{passes}")
            elif device == "jetson":
                # `quiet` is a claim of full observation, so it carries the
                # same pass counters `not_evaluable` does -- a reader can
                # check "readable throughout" instead of taking it on faith.
                passes = (
                    f" ({ev.get('passes_readable')} of {ev.get('passes_attempted')} passes fully readable)"
                    if ev.get("passes_attempted") else ""
                )
                lines.append(f"- throttle events, jetson: quiet -- cooling devices readable throughout, "
                             f"{count} transitions{passes}")
            else:
                lines.append(f"- throttle events, phone: quiet -- {count} status transitions "
                             f"in {n_phone} reports")

    ticks_by_basis = thermal.get("ticks_by_basis") or {}
    if ticks_by_basis:
        # The real signal for a sampler thread that died partway through the
        # run (see `thermal_result`'s docstring): `sample_gaps_s` cannot show
        # it, because a gap exists only between samples that were written.
        parts = ", ".join(f"{basis} {n}" for basis, n in sorted(ticks_by_basis.items()))
        lines.append(f"- jetson thermal seen by ticks: {parts}")
    return lines


#: `SessionLog.MAX_LINES_PER_KIND`, ported: how many `offerFailure` lines of
#: one `kind` the phone will ever write in a session. A kind that reaches it
#: stops writing lines at all -- not even a `suppressed` count on a later
#: line, because there is no later accepted line for the count to ride on --
#: so this reader's own total goes on undercounting past this point, and
#: `report.md` says so rather than presenting the total as complete.
PHONE_FAILURE_LIFETIME_CAP = 64


def _phone_failure_lines(failures: dict[str, Any]) -> list[str]:
    """The one line naming the phone's own failures, read only offline
    (D11) -- appended regardless of how much the Jetson side has to say,
    since the phone's session log is pulled independently of the run
    directory."""
    phone_failures = failures.get("phone")
    if phone_failures is None:
        return ["- phone-side failures: not read (no --phone-log)"]
    if not phone_failures:
        return ["- phone (offline): 0 failure lines in the session log"]
    by_kind: dict[str, int] = {}
    lines_by_kind: dict[str, int] = {}
    for record in phone_failures:
        kind = record.get("kind", "?")
        # D12's own compensation: an occurrence the per-second rate cap held
        # back rides forward on the next accepted line's `suppressed` count.
        # Reading `n` alone silently discards it.
        occurrences = int(record.get("n", 1)) + int(record.get("suppressed", 0))
        by_kind[kind] = by_kind.get(kind, 0) + occurrences
        lines_by_kind[kind] = lines_by_kind.get(kind, 0) + 1
    parts = []
    for kind, n in sorted(by_kind.items()):
        note = ""
        if lines_by_kind[kind] >= PHONE_FAILURE_LIFETIME_CAP:
            # Past this many lines the phone stops writing this kind
            # altogether (D12's lifetime cap), so nothing -- not even a
            # `suppressed` count -- says how many more happened.
            note = " (reached the phone's per-kind lifetime cap; further occurrences, if any, are not recorded)"
        parts.append(f"{kind} {n}{note}")
    breakdown = ", ".join(parts)
    return [f"- phone (offline): {breakdown}"]


def _log_health_lines(failures: dict[str, Any]) -> list[str]:
    """The metadata logger's own final state (D16), read from `log_health.json`
    rather than from `summary["failures"]` -- see that file's own docstring
    for why the two must stay separate."""
    log_health = failures.get("log_health")
    if log_health is None:
        return []
    writer = "writer healthy" if log_health.get("writer_failure") is None else \
        f"writer failed: {log_health['writer_failure']}"
    return [f"- log: {log_health.get('dropped_records', 0)} records dropped, {writer}"]


def _failure_lines(failures: dict[str, Any] | None, thermal_section_present: bool = True) -> list[str]:
    """The `## Failures` section. Absent entirely when `failures` itself is
    `None` -- the log carries no failure records at all, the reading a
    pre-task-38 run gets (task 33 and task 36 both shipped a measurement
    with no surface; this exists so that mistake is not repeated a third
    time).

    `thermal_section_present` (default `True`, matching every pre-existing
    call site) is whether `## Thermal` will actually render in the same
    document -- this section used to point at it unconditionally, which on
    a drive with no thermal records at all pointed at a section that does
    not exist (the hardware-drive round's D3).
    """
    if not failures:
        return []
    lines = ["", "## Failures", ""]

    if failures.get("jetson_predates_failure_log"):
        # Distinct from "no summary was written" below: that line means a
        # run that HAD this feature did not reach its normal teardown. This
        # one means the Jetson-side log never had the feature at all -- a
        # different fact, and the wrong one would blame a wiring bug that
        # does not exist here.
        lines.append(
            "- this run's Jetson-side log predates the failure event log; "
            "only the phone's own failures, if supplied, are shown below"
        )
        return lines + _phone_failure_lines(failures)

    s = failures.get("summary")

    if not s:
        lines.append(
            "- failure records exist for this drive but no summary was written "
            "(the run likely did not reach its normal teardown)"
        )
        return lines + _log_health_lines(failures) + _phone_failure_lines(failures)

    scan = s.get("scan") or {}
    interval = scan.get("interval_s") or {}
    if interval.get("p50") is not None:
        lines.append(
            f"- {scan.get('sources_n', 0)} sources scanned on {scan.get('passes', 0)} passes; "
            f"scan interval p50 {interval['p50']:.3f} s, max {interval.get('max', 0):.3f} s"
        )
    else:
        lines.append(
            f"- failure sampling: NOT EVALUABLE -- sampler_stopped; "
            "this drive says nothing about whether anything failed"
        )
        return lines + _log_health_lines(failures) + _phone_failure_lines(failures)

    ticks_seen = failures.get("ticks_seen_per_pass")
    if ticks_seen is not None:
        stall_note = "" if ticks_seen.get("min", 0) > 0 else " -- the tick loop stalled at least once"
        lines.append(
            f"- ticks seen per pass: p50 {ticks_seen.get('p50')}, min {ticks_seen.get('min')}{stall_note}"
        )

    episodes = failures.get("episodes") or []
    outcomes = s.get("outcomes") or {}
    if episodes or outcomes:
        by_outcome = ", ".join(f"{n} {name.replace('_', ' ')}" for name, n in sorted(outcomes.items()))
        closed_total = sum(outcomes.values())
        if len(episodes) == closed_total:
            # The common case: every closed episode belongs to a source with
            # `event_records=True`, so the open/close records this reader
            # joined and the summary's own closed count agree, and one
            # number describes both.
            lines.append(f"- {len(episodes)} episodes: {by_outcome}" if by_outcome else f"- {len(episodes)} episodes")
        else:
            # `episodes` counts the open/close record pairs this reader
            # joined from the log, which exist only for a source with
            # `event_records=True`. `closed_total` is the summary's own
            # count of every episode that closed, on every source. The two
            # differ by exactly the episodes closed on a source with
            # `event_records=False`, which never wrote a record to join --
            # naming both numbers instead of one avoids implying they count
            # the same population.
            lines.append(
                f"- {len(episodes)} episodes recorded (open and close events); "
                f"{closed_total} episodes closed in total"
                + (f": {by_outcome}" if by_outcome else "")
                + f" -- {closed_total - len(episodes)} closed on a source with no event record"
            )

    durations_by_source: dict[str, list[float]] = {}
    for episode in episodes:
        if episode.get("duration_s") is not None:
            durations_by_source.setdefault(episode["source"], []).append(episode["duration_s"])

    sources = s.get("sources") or {}
    for name, row in sorted(sources.items()):
        status = row.get("status")
        if status == RULE_FIRED:
            longest = max(durations_by_source[name]) if durations_by_source.get(name) else None
            longest_note = f", longest {longest:.1f} s" if longest is not None else ""
            # A cumulative source's `total` is real occurrences: a counter
            # moved that many times. A predicate source's `total` is passes
            # the condition held -- a sticky error active for a whole 180 s
            # drive reports 178, one per second it stayed true, and calling
            # that "occurrences" reads as 178 distinct failures rather than
            # one that lasted. `row` predates this distinction on a run
            # recorded before it existed, so the default keeps that
            # reading (`True`) rather than silently reclassifying it.
            quantity = "occurrences" if row.get("cumulative", True) else "passes with the condition active"
            lines.append(
                f"- {name}: FIRED -- {row.get('episodes', 0)} episode(s), "
                f"{row.get('total', 0)} {quantity}{longest_note}"
            )
        elif status == RULE_NOT_EVALUABLE:
            missing = ", ".join(row.get("missing") or [])
            passes_attempted = row.get("passes_attempted", 0)
            passes_unreadable = passes_attempted - row.get("passes_readable", 0)
            lines.append(
                f"- {name}: NOT EVALUABLE on {passes_unreadable} of "
                f"{passes_attempted} passes -- missing {missing}; "
                f"this drive says nothing about {name}'s failures during that window"
            )
        else:
            lines.append(
                f"- {name}: quiet -- readable on {row.get('passes_readable', 0)} of "
                f"{row.get('passes_attempted', 0)} passes"
            )

    backwards = s.get("counter_went_backwards") or {}
    for name, steps in backwards.items():
        for step in steps:
            lines.append(
                f"- {name}: counter went backwards, {step.get('from')} -> {step.get('to')} "
                "(the counter is right; this record is the one that is wrong)"
            )

    lines.append(
        f"- blind ticks: {s.get('blind_ticks', 0)}; "
        f"pipeline exception: {s.get('pipeline_exception') or 'none'}"
    )

    lines += _log_health_lines(failures)
    lines.append(
        "- thermal failures are recorded separately -- see ## Thermal" if thermal_section_present
        else "- thermal failures are recorded separately, but this drive has no ## Thermal "
             "section (no thermal records at all)"
    )
    lines += _phone_failure_lines(failures)
    return lines


#: The noun each axis's `attempted`/`answered` count over, for the one-line
#: rendering. `rates` is not a pure "reports" count: `attempted` is distinct
#: telemetry reports plus one per tick that observed none at all (a tick, not
#: a report), so it is worded as an observation rather than claimed to be a
#: report count it is not.
_AXIS_NOUN = {
    "latency": "ticks", "rates": "telemetry observations", "api_calls": "ticks",
    "triggers": "ticks", "failures": "ticks", "thermal": "ticks", "provenance": "ticks",
}

#: Which `## Session summary` axis each reconciliation contradicts, for
#: `_reconciliation_line` -- a failed reconciliation used to be keyed only on
#: its own check name, rendering several lines away from the axis headline it
#: was about with nothing in the text connecting the two (plan §5.4).
_RECONCILIATION_AXIS = {
    "blind_ticks_matches_camera_source": "failures",
    "source_status_fired_iff_total_positive": "failures",
    "source_totals_partition": "failures",
    "source_passes_readable_le_attempted": "failures",
    "sources_count_is_30": "failures",
    "events_written_matches_open_and_close_records": "failures",
    "thermal_samples_count": "thermal",
    "triggers_match_summary": "triggers",
    "here_calls_ge_responses_received": "api_calls",
    "reference_absent_iff_fields_null": "api_calls",
}


def _axis_fully_answered(axis: dict[str, Any]) -> bool:
    return (
        axis["unbuildable"] is None
        and axis["attempted"] not in (None, 0)
        and axis["answered"] == axis["attempted"]
    )


def _axis_extra_clause(axis: dict[str, Any], context: dict[str, Any] | None) -> str:
    """The context that does not fit the generic `answered of attempted`
    shape: which rules were legitimately `not_evaluable` (a state that is
    not "unanswered", plan D3) and why `rates` may have nothing to compare
    achieved against (plan D8, a shadow drive) -- both read off `sensing`,
    when it is known.

    Two more, read off the axis record and `context["thermal"]` directly,
    from the hardware-drive validation round rather than the plan:
    `failures`'s own per-source unreadability census (that round's D2), and
    `thermal`'s jetson-only scope when the phone half of the same instrument
    answered nothing at all (that round's D8).
    """
    context = context or {}
    sensing = context.get("sensing")
    name = axis["axis"]
    if sensing is not None:
        if name == "triggers":
            bits = []
            for rule, statuses in sorted(sensing["triggers"]["rules_by_status"].items()):
                n = statuses.get(RULE_NOT_EVALUABLE, 0)
                if not n:
                    continue
                missing = sensing["triggers"]["rules_missing"].get(rule) or {}
                missing_str = ", ".join(sorted(missing)) if missing else "an input"
                bits.append(f"{rule} was not_evaluable on {n} of them, missing {missing_str}")
            return (" " + "; ".join(bits) + ".") if bits else ""
        if name == "rates" and not sensing["ever_live"]:
            return (
                f" Commanded rates were never applied (mode {sensing['mode']} on "
                f"{sensing['ticks']} of {sensing['ticks']} decisions), so achieved is "
                "not a shortfall against them."
            )
    if name == "failures" and axis.get("not_evaluable_by_rule"):
        sources = axis["not_evaluable_by_rule"]
        total_unreadable = sum(sources.values())
        return (
            f" {len(sources)} of the underlying failure sources were not fully readable "
            f"on every pass ({total_unreadable} source-passes unreadable across them) -- "
            "see ## Failures for which."
        )
    if name == "thermal":
        phone = ((context.get("thermal") or {}).get("summary") or {}).get("phone") or {}
        n_phone = phone.get("samples", 0)
        headroom_absent = sum((phone.get("headroom_absent_counts") or {}).values())
        if n_phone and headroom_absent == n_phone:
            return (
                f" This counts the jetson zone only: the phone half of the same instrument "
                f"answered none of its {n_phone} reports."
            )
    return ""


def _axis_headline_line(axis: dict[str, Any], context: dict[str, Any] | None) -> str:
    name, section = axis["axis"], axis["section"]
    rendered_sections = (context or {}).get("rendered_sections")
    pointer = (
        f" See {section}." if rendered_sections is None or section in rendered_sections
        else f" {section} does not appear in this report."
    )
    if axis["unbuildable"] is not None:
        return f"- **{name}**: not built -- {axis['unbuildable']}.{pointer}"
    noun = _AXIS_NOUN[name]
    reason = ", ".join(f"{word} {n}" for word, n in sorted(axis["unanswered_by_reason"].items()))
    if not reason and axis["attempted"] == 0 and axis.get("zero_attempted_context"):
        # The one shape whose census is structurally empty (no per-tick
        # record to read a reason off) -- the hardware-drive round's D4.
        reason = axis["zero_attempted_context"]
    line = f"- **{name}**: {axis['answered']} of {axis['attempted']} {noun} answered"
    if reason:
        line += f" -- {reason}"
    line += "."
    line += _axis_extra_clause(axis, context)
    return line + pointer


def _reconciliation_line(rec: dict[str, Any]) -> str:
    axis = _RECONCILIATION_AXIS.get(rec["name"], rec["name"])
    detail = rec.get("detail")
    prefix = f"- **{axis}** ({rec['name']}): {rec['status']}"
    return f"{prefix} -- {detail}" if detail else prefix


def _tick_coverage_line(coverage: dict[str, Any] | None) -> list[str]:
    """The one line the hardware-drive round's D1 exists for: a check on
    tick PRODUCTION, placed in the first few lines of `## Session summary`
    so an interruption long enough to matter cannot hide behind seven axes
    that are all, individually, ratios of ticks that DO exist. Absent
    (rather than a manufactured zero) on fewer than three ticks -- there is
    no gap distribution to read a median from.
    """
    if coverage is None:
        return []
    largest = coverage["largest_gap_s"]
    next_largest = coverage["next_largest_gap_s"]
    missing = coverage["missing_ticks"]
    expected = coverage["expected_ticks"]
    fraction = f" ({missing / expected:.1%} of {expected} expected)" if missing and expected else ""
    return [
        f"Tick coverage: {coverage['actual_ticks']} ticks read over {coverage['span_s']:.2f} s "
        f"(median inter-tick gap {coverage['median_gap_s'] * 1000:.1f} ms); largest gap "
        f"{largest:.2f} s (next-largest {next_largest:.2f} s), costing an estimated "
        f"{missing} tick(s) at {coverage['gap_multiple']:g}x the median or bigger{fraction}.",
        "",
    ]


def _session_summary_lines(session: dict[str, Any], context: dict[str, Any] | None = None) -> list[str]:
    """`## Session summary`: every axis, always -- rule 22 forbids counting
    how many answered without enumerating the ones that did not, on the same
    lines, so the enumeration is never partial.

    `context` carries `sensing`, `thermal` and `rendered_sections` for the
    axis headlines (`_axis_headline_line`/`_axis_extra_clause`); a caller
    that does not pass one (every pre-existing call site) gets the same
    reading as before -- `sensing` from `session` itself, no thermal detail,
    and every axis's named section assumed present.
    """
    axes = session["axes"]
    if context is None:
        context = {"sensing": session.get("sensing"), "thermal": None, "rendered_sections": None}
    fully_answered = [a for a in axes if _axis_fully_answered(a)]
    unbuildable = [a for a in axes if a["unbuildable"] is not None]
    did_not = [a for a in axes if a not in fully_answered and a not in unbuildable]
    lines = [
        "", "## Session summary", "",
        f"{len(axes)} axes, {len(fully_answered)} answered. "
        f"{len(did_not)} did not, {len(unbuildable)} could not be built.",
        "",
    ]
    lines += _tick_coverage_line(session.get("tick_coverage"))
    lines += [_axis_headline_line(axis, context) for axis in axes]

    recs = session["reconciliations"]
    held = [r for r in recs if r["status"] == "held"]
    failed = [r for r in recs if r["status"] == "failed"]
    unavailable = [r for r in recs if r["status"] == "unavailable"]
    lines += [
        "",
        f"{len(recs)} reconciliations, {len(held)} held, {len(failed)} failed, "
        f"{len(unavailable)} unavailable.",
    ]
    not_held = [_reconciliation_line(rec) for rec in recs if rec["status"] != "held"]
    if not_held:
        lines.append("")
        lines += not_held

    inputs = session["inputs"]
    lines += [
        "",
        f"Inputs: metadata.jsonl {_yes_no(inputs['metadata_jsonl'])}, "
        f"summary.json {_yes_no(inputs['summary_json'])}, "
        f"log_health.json {_yes_no(inputs['log_health_json'])}, "
        f"phone log {_yes_no(inputs['phone_log'])}.",
    ]
    return lines


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _overall_clause(session: dict[str, Any]) -> str:
    """M2: `unbuildable` axes (no phone on this drive, say) are not axes that
    "did not answer" -- folding the two together used to make a healthy
    phone-less drive print "3 of 7 instrument axes did not answer" on this
    line while the section it points to correctly read "0 did not, 3 could
    not be built". Both counts are named here so the two can never disagree.
    """
    axes = session["axes"]
    fully_answered = sum(1 for a in axes if _axis_fully_answered(a))
    if fully_answered == len(axes):
        return f"the five gates; all {len(axes)} instrument axes answered."
    unbuildable = sum(1 for a in axes if a["unbuildable"] is not None)
    did_not = len(axes) - fully_answered - unbuildable
    return (
        f"the five gates. {did_not} of {len(axes)} instrument axes did not answer, "
        f"{unbuildable} could not be built; see ## Session summary."
    )


def _sensing_lines(sensing: dict[str, Any] | None) -> list[str]:
    """`## Sensing`: the rate, trigger and HERE detail behind three axes that
    have no other section of their own.
    """
    if sensing is None:
        return []
    lines = [
        "", "## Sensing", "",
        f"- mode: {sensing['mode']} ({sensing['mode_source']}); ever live: {sensing['ever_live']}",
    ]
    window = sensing["telemetry_window"]
    if window["unbuildable"]:
        lines.append(f"- telemetry window: unbuildable -- {window['unbuildable']}")
    else:
        lines.append(
            f"- telemetry window: observed median {window['observed_median_s']:.3f} s "
            f"over {window['reports']} distinct reports"
        )
    lines += [
        "",
        "| rate | commanded distinct | commanded time mean | achieved time mean | comparable | p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    # Rendered separately from the table (MINOR): computed, written to
    # `report.json`, and otherwise rendered nowhere.
    scaling_notes: list[str] = []
    for key in RATE_KEYS:
        row = sensing["rates"][key]
        comparable = "yes" if row["comparable"] else f"no ({row['not_comparable_because']})"
        p50, p95 = row["percentiles"]["p50"], row["percentiles"]["p95"]
        note = row["percentiles_suppressed"]
        if p50 is None and p95 is None:
            # A real suppression (D9, lambda < 1.0): no numbers to show.
            p50_str = p95_str = f"suppressed ({note})" if note else "n/a"
        else:
            p50_str = f"{p50:.3f}"
            p95_str = f"{p95:.3f}"
            if note:
                # lambda == 1.0 exactly: the numbers ARE reported, and the
                # quantisation is named beside the table rather than hidden
                # behind "suppressed" text that would claim there are none.
                p50_str += " [quantised]"
                p95_str += " [quantised]"
                scaling_notes.append(f"- {key}: {note}")
        achieved = row["achieved_time_mean"]
        achieved_str = "n/a" if achieved is None else f"{achieved:.3f}"
        commanded_str = "n/a" if row["commanded_time_mean"] is None else f"{row['commanded_time_mean']:.3f}"
        lines.append(
            f"| {key} | {row['commanded_distinct']} | {commanded_str} | "
            f"{achieved_str} | {comparable} | {p50_str} | {p95_str} |"
        )
        if row["clamped_ticks"] or row["thermal_scaled_ticks"]:
            scaling_notes.append(
                f"- {key}: clamped on {row['clamped_ticks']} ticks, "
                f"thermal-scaled on {row['thermal_scaled_ticks']} ticks"
            )
    if scaling_notes:
        lines += [""] + scaling_notes
    triggers = sensing["triggers"]
    lines += ["", f"- decisions by trigger: {triggers['decisions_by_trigger']}"]
    for rule, statuses in sorted(triggers["rules_by_status"].items()):
        missing = triggers["rules_missing"].get(rule)
        missing_str = f"; missing {missing}" if missing else ""
        lines.append(f"- {rule}: {statuses}{missing_str}")
    if triggers["summary_agrees"] is False:
        lines.append("- WARNING: computed trigger/rule counts disagree with summary.json")
    here = sensing["here"]
    if here.get("not_measured"):
        lines.append(f"- HERE calls: not measured -- {here['not_measured']}")
    else:
        backwards = (
            f"; counter_went_backwards: {here['counter_went_backwards']}"
            if here.get("counter_went_backwards") else ""
        )
        lines.append(
            f"- HERE calls: {here['calls_total']} total across {len(here['by_session'])} session(s) "
            f"({here['uncounted_prefix']} placed before the first observed report in each session, "
            f"not counted), {here['errors_total']} errors, expected from commanded rate "
            f"{here['expected_from_commanded']:.2f} (credited over "
            f"{here['expected_from_commanded_weighted_over_s']:.1f} of "
            f"{here['expected_from_commanded_span_s']:.1f} s spanned by commanded-rate ticks), "
            f"phone-reported responses received {here['responses_received']}"
            + (f" -- {here['zero_calls_because']}" if here.get("zero_calls_because") else "")
            + backwards
        )
    return lines


def render_session_summary_only(
    run_dir: Path, session: dict[str, Any], loaded: "LoadedRecords",
    log_health: dict[str, Any] | None = None,
) -> str:
    """The whole `report.md` for a drive with no tick records at all (D13):
    `analyze()` cannot run -- every metric block indexes `ticks` -- but the
    session summary is built from `metadata.jsonl`, `summary.json` and
    `log_health.json` alone, none of which needs a single tick.

    `log_health`'s own detail (records dropped, writer status) is rendered
    here directly: its only other renderer, `_log_health_lines`, is called
    from `## Failures`, which this branch never produces -- so a zero-tick
    drive used to say nothing about the one fact (the log dropped records)
    that might explain why there are no ticks.
    """
    lines = [
        f"# Run evaluation - {run_dir.name}", "",
        f"0 ticks read from {run_dir / 'metadata.jsonl'} "
        f"({len(loaded.failure_scans)} failure_scan, {len(loaded.failure_events)} "
        "failure_event records present)",
        "",
        "No tick records exist for this drive, so none of the metric sections below can be "
        "built -- `analyze()` raises on a zero-tick drive by design. This section reads "
        "`metadata.jsonl`, `summary.json` and `log_health.json` alone.",
    ]
    sensing = session.get("sensing")
    # This branch never reaches `analyze()`, so `## Thermal` and `## Failures`
    # never render here regardless of what those axes' `section` fields say
    # (the hardware-drive round's D3) -- `## Sensing` is the only one of the
    # three that can, and only when the drive has a phone.
    rendered_sections = frozenset({"## Sensing"}) if sensing is not None else frozenset()
    context = {"sensing": sensing, "thermal": None, "rendered_sections": rendered_sections}
    lines += _session_summary_lines(session, context)
    if log_health is not None:
        lines += _log_health_lines({"log_health": log_health})
    lines += _sensing_lines(sensing)
    lines.append("")
    return "\n".join(lines)


def render_markdown(
    result: dict[str, Any], plots: list[str], session: dict[str, Any] | None = None,
) -> str:
    r = result
    lines = [f"# Run evaluation - {Path(r['run_dir']).name}", ""]
    if r["scenario"]["description"]:
        lines += [f"Scenario: {r['scenario']['description']}", ""]
    if r["scenario"]["video_source"]:
        lines += [f"Video: `{r['scenario']['video_source']}`", ""]
    if not r["advisory"]["trained_policy"]:
        lines += [
            "**UNTRAINED policy bundle** - advisory values are random-init placeholders;",
            "this report certifies plumbing and latency, not advisory quality.",
            "",
        ]
    lines += [
        f"{r['n_ticks']} ticks over {r['duration_s']} s "
        f"(median {r['tick_rate_hz_median']} Hz, "
        f"{r['camera_dropped_frames']} camera frames dropped)",
    ]
    # `## Thermal`/`## Failures`/`## Sensing` are each conditional further
    # down in this function -- computed here, before the axis headlines are
    # rendered, so an axis whose named `section` will not actually appear in
    # this document says so instead of pointing at nothing (the
    # hardware-drive round's D3). The other four sections below are
    # unconditional.
    rendered_sections = {"## Latency (full run)", "## Observation quality"}
    if r.get("thermal"):
        rendered_sections.add("## Thermal")
    if r.get("failures"):
        rendered_sections.add("## Failures")
    if session is not None and session.get("sensing") is not None:
        rendered_sections.add("## Sensing")
    context = None
    if session is not None:
        context = {
            "sensing": session.get("sensing"), "thermal": r.get("thermal"),
            "rendered_sections": frozenset(rendered_sections),
        }
    # Placed above ## Gates (D2): a reader who reads one section reads the
    # first one, and that used to be the verdict this section exists to
    # qualify.
    if session is not None:
        lines += _session_summary_lines(session, context)
    lines += [
        "",
        "## Gates",
        "",
        "| gate | value | threshold | verdict |",
        "|---|---|---|---|",
    ]
    for name, g in r["gates"].items():
        verdict = "n/a" if g["pass"] is None else ("PASS" if g["pass"] else "**FAIL**")
        lines.append(f"| {name} | {g['value']} | {g['threshold']} | {verdict} |")
    integrity = r.get("log_integrity", {})
    if integrity.get("log_complete") is False:
        lines.append(
            f"| log complete | read {integrity.get('ticks_read')} of "
            f"{integrity.get('ticks_the_run_reported')} ticks, "
            f"{integrity.get('unparseable_lines')} unparseable lines | 0 missing | "
            f"**FAIL** |"
        )
    elif integrity.get("log_complete") is None:
        lines.append(
            f"| log complete | read {integrity.get('ticks_read')} ticks; the run's own "
            "summary carries no tick count to compare against | 0 missing | "
            f"**FAIL (unmeasurable)** |"
        )
    overall_line = f"**Overall: {'PASS' if r['overall_pass'] else 'FAIL'}**"
    if session is not None:
        overall_line += f" -- {_overall_clause(session)}"
    lines += [
        "",
        overall_line,
        "",
        "## Latency (full run)",
        "",
        "| stage | n | min | mean | p50 | p95 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    order = ["jetson_ms", "link_ms", "e2e_ms", "capture_to_start_ms", "detect_ms",
             "track_distance_ms",
             "observe_ms", "policy_advisory_ms"]
    for key in order:
        # None means the series was absent -- a local run has no link segment --
        # and an absent series must not render as zeros.
        if r["latency_ms"].get(key) is not None:
            lines.append(fmt_row(key.removesuffix("_ms"), r["latency_ms"][key]))
    stages = r.get("stage_timings") or {}
    if stages:
        lines += [
            "",
            "## Per-stage timings",
            "",
            "`measured` is one clock. `converted` crossed two and carries a bound. "
            "`absent` names why there is no number; it is never a zero. `instant` is a "
            "point in time, not a duration.",
            "",
            "| stage | basis | n | min | mean | p50 | p95 | max |",
            "|---|---|---|---|---|---|---|---|",
        ]
        named = [k for k in STAGE_ORDER if k in stages]
        for name in named + sorted(k for k in stages if k not in STAGE_ORDER):
            slot = stages[name]
            basis = ", ".join(f"{k} {v}" for k, v in sorted(slot["basis"].items()))
            st = slot.get("stats")
            if st is None:
                lines.append(f"| {name} | {basis} | 0 | | | | | |")
            else:
                lines.append(
                    f"| {name} | {basis} | {st['n']} | {st.get('min', 0.0):.1f} | "
                    f"{st['mean']:.1f} | {st['p50']:.1f} | {st['p95']:.1f} | "
                    f"{st['max']:.1f} |"
                )
        absent = {n: s["absent_reasons"] for n, s in stages.items() if s["absent_reasons"]}
        if absent:
            lines += ["", "Why a stage had no number:"]
            for name, reasons in sorted(absent.items()):
                for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                    lines.append(f"- `{name}` x{n}: {why}")

    p = r["perception"]
    leader_gap_line = (
        f"gap p50 {p['leader_gap_m']['p50']:.1f} m (min-side mean {p['leader_gap_m']['mean']:.1f} m)"
        if p["leader_gap_m"] is not None else "no leader observed on any tick"
    )
    track_lifetime_line = (
        f"{p['unique_tracks']} tracks, lifetime p50 {p['track_lifetime_s']['p50']:.1f} s "
        f"(p95 {p['track_lifetime_s']['p95']:.1f} s)"
        if p["track_lifetime_s"] is not None else "no tracks"
    )
    lines += [
        "",
        "## Perception",
        "",
        f"- ticks with >= 1 tracked vehicle: {p['ticks_with_vehicle_fraction']:.1%} "
        f"(mean {p['mean_vehicles_per_tick']:.2f}/tick)",
        f"- leader present: {p['leader_present_fraction']:.1%} of ticks; {leader_gap_line}",
        f"- leader relative speed measured (vs neutral fallback): "
        f"{p['leader_rel_speed_measured_fraction']:.1%} of leader ticks",
        f"- distance methods: {p['distance_method_counts']}",
        f"- {track_lifetime_line}",
        "",
        "## Observation quality",
        "",
    ]
    obs = r["observation"]
    pf = obs.get("provenance_fields")
    m = obs["missingness"]
    # A mean is one number for a metric that only takes a handful of discrete
    # values (a count of fallback fields over a fixed encoder-field total), so
    # it can land on a percentage no tick ever produced. The range tells a
    # reader that; the distinct-value count tells them more directly still,
    # because with one mode dominating, p50 and p95 both read as that mode and
    # hide a rarer one sitting between it and the other extreme.
    distinct = sorted({t["obs_diagnostics"]["missingness"] for t in r["_ticks"]})
    spread = f"min {m['min']:.1%}, p50 {m['p50']:.1%}, p95 {m['p95']:.1%}, max {m['max']:.1%}"
    if len(distinct) <= 8:  # same "worth naming individually" cutoff as most_common(8) below
        spread += f", {len(distinct)} distinct value{'s' if len(distinct) != 1 else ''}"
    lines.append(
        f"- encoder-field missingness: mean {m['mean']:.1%}"
        + (f" of {pf} provenance-tagged fields" if pf is not None else "")
        + f" ({spread})"
    )
    if pf is not None:
        if obs.get("provenance_fields_mixed"):
            lines.append(
                f"- provenance_fields varies across ticks (first tick has {pf}); "
                "`by_source` below is pooled across sizes and `covers_encoder` "
                "is not meaningful for this run"
            )
        else:
            lines.append(
                f"- provenance covers {pf} of {sim_contract.local_obs_dim()} encoder slots"
            )
    if obs.get("by_source"):
        by_source_str = ", ".join(
            f"{source} {frac:.1%}"
            for source, frac in sorted(obs["by_source"].items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- by source: {by_source_str}")
    derived_empty_fields = obs.get("fields_by_source", {}).get("derived_empty")
    if derived_empty_fields:
        parts = ", ".join(
            f"{field} {frac:.0%} of ticks" for field, frac in derived_empty_fields.items()
        )
        lines.append(
            "- derived from an absence (a blind sensor and an empty road are the same "
            f"number here): {parts}"
        )
    lines += [
        f"- most frequent fallback fields (fraction of ticks): "
        f"{r['observation']['top_fallback_fields']}",
    ]
    lines += _thermal_lines(r.get("thermal"))
    lines += _failure_lines(r.get("failures"), bool(r.get("thermal")))
    lines += [
        "",
        "## GPS",
        "",
        f"- fresh fix on {r['gps']['fresh_fraction_overall']:.1%} of ticks",
    ]
    if "speed_rmse_mps" in r["gps"]:
        rmse = r["gps"]["speed_rmse_mps"]
        drift = r["gps"]["speed_max_drift_during_dropout_mps"]
        lines += [
            f"- scripted dropouts: {r['gps']['scripted_dropouts_s']}; fresh outside them: "
            f"{r['gps']['fresh_fraction_outside_dropouts']:.1%}",
            f"- ego speed vs scripted truth: RMSE {rmse:.3f} m/s, "
            f"max |err| {r['gps']['speed_max_abs_err_mps']:.3f} m/s"
            + (f", max drift during dropout {drift:.2f} m/s (held last fix)" if drift is not None else ""),
        ]
    a = r["advisory"]
    lines += [
        "",
        "## Advisory (not gated"
        + ("" if a["trained_policy"] else "; UNTRAINED bundle")
        + ")",
        "",
        f"- recommended speed: p50 {a['recommended_speed_mps']['p50']:.1f} m/s "
        f"(mean {a['recommended_speed_mps']['mean']:.1f})",
        f"- head switches: {a['head_switches_per_minute']:.1f} / min",
        f"- confidence labels: {a['confidence_labels']}",
        f"- head distributions: {json.dumps(a['head_distributions'], indent=2)}",
    ]
    if session is not None:
        lines += _sensing_lines(session.get("sensing"))
    join = r.get("phone_join")
    if join is not None:
        return_ms = [
            row["stages"]["return"]["ms"] for row in join["rows"]
            if row["stages"]["return"]["ms"] is not None
        ]
        render_ms = [
            row["stages"]["render"]["ms"] for row in join["rows"]
            if row["stages"]["render"]["ms"] is not None
        ]
        lines += [
            "",
            "## Phone join (return / render)",
            "",
            f"- advisories the phone logged as received: {join['advisories_seen_by_the_phone']}",
            f"- matched to a Jetson tick: {join['matched']}; unmatched: {join['unmatched']}",
            f"- Jetson ticks with no returned advisory: {join['jetson_ticks_with_no_advisory']} "
            "(return/render absent on these rows; the tick's own stages still count)",
            f"- return_ms measured on {len(return_ms)} of {join['matched']} matched rows"
            + (f" (p50 {sorted(return_ms)[len(return_ms) // 2]:.1f} ms)" if return_ms else ""),
            f"- render_ms measured on {len(render_ms)} of {join['matched']} matched rows"
            + (f" (p50 {sorted(render_ms)[len(render_ms) // 2]:.1f} ms)" if render_ms else ""),
        ]
    if plots:
        lines += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots]
    lines.append("")
    return "\n".join(lines)


#: The seven instruments a drive runs, in the order the summary lists them.
#: Fixed: an axis that cannot be built appears with `attempted: null` and a
#: named missing input, never by being absent from the list (rule 6).
AXES = ("latency", "rates", "api_calls", "triggers", "failures", "thermal", "provenance")

#: Reused across the two axes whose census comes off `policy.sensing_loop.reference_from`:
#: a report is either fresh, stale by the controller's own predicate, or never
#: arrived at all.
REFERENCE_NO_TELEMETRY = "no_telemetry"
REFERENCE_STALE = "stale"

#: The `latency` axis has no closed vocabulary of its own: `jetson_ms` is either
#: on the tick or it is not, and the one wording used when it is not is the same
#: one `latency["jetson_ms_source"]` already renders. Owned here, not in
#: `sensors.time_sync`, which has no such combined constant (see final report).
LATENCY_ABSENT_REASON = "absent from this run; e2e used"
LATENCY_VOCABULARY = frozenset({LATENCY_ABSENT_REASON})

#: `rates`/`api_calls` census words, both traceable to `reference_from`'s own
#: two-word vocabulary (`"no_telemetry"`) plus the controller's own staleness
#: predicate (`"stale"`, from comparing `age_s` against `MAX_TELEMETRY_AGE_S`).
RATES_VOCABULARY = frozenset({REFERENCE_NO_TELEMETRY, REFERENCE_STALE})
#: `field_not_recorded` is not one of `reference_from`'s own words: it names
#: the shape violation where `absent` reads `None` (telemetry present) but
#: `here_calls` is still null, which `reference_from`'s own branches never
#: produce (task 39's shape rule) but a malformed or hand-built record can.
API_CALLS_FIELD_NOT_RECORDED = "field_not_recorded"
API_CALLS_VOCABULARY = frozenset({REFERENCE_NO_TELEMETRY, API_CALLS_FIELD_NOT_RECORDED})

#: `triggers` answers "did this tick's attribution parse", not "did every rule
#: fire" -- a rule being `not_evaluable` is a legitimate state (D3), not an
#: unanswered tick. The one way a tick fails to answer is a malformed shape:
#: not exactly the four `RULES`, or a status outside the three-word set.
TRIGGERS_UNANSWERED_REASON = "invalid_attribution_shape"
TRIGGERS_VOCABULARY = frozenset({TRIGGERS_UNANSWERED_REASON})

#: `thermal`/`failures` share one vocabulary: `sensors.thermal.ABSENT_REASONS`
#: (six words, reused verbatim by `logio.failure_log` for its own per-tick
#: `basis`) plus the `stale` word both modules also share via
#: `sensors.thermal.THERMAL_BASIS_STALE`.
THERMAL_FAILURES_VOCABULARY = frozenset({THERMAL_BASIS_STALE}) | ABSENT_REASONS

#: `provenance` states the SHAPE of `field_sources` (a map of the wrong size
#: is `short`/`long`, no vocabulary of legal wrong sizes exists) and, on a map
#: of the right size, whether it carries any field that is actually evidence
#: about this tick.
#:
#: D7: `sim_contract`'s always-derived slots (`ego_headway_s`,
#: `target_lane_front_gap`, `uncongested_low_speed_flag`) carry `derived` or
#: `approximated` on every tick of every drive, regardless of whether the
#: quantities they were computed from were themselves measured or
#: substituted -- `field_sources` records the class assigned to a field, not
#: the provenance of that field's own inputs, so a tick built entirely from
#: `fallback_neutral` values still shows a non-excluded class on these three
#: and this axis answered `749 of 749` on a drive that measured nothing.
#: Excluding `derived`/`approximated` from evidence on their own fixes the
#: false positive without needing a dependency graph: a tick "answers" only
#: when some OTHER field in the same map carries a genuinely primary class
#: (`PROVENANCE_PRIMARY_EVIDENCE`, below) -- which a real derivation would
#: have been built from, since there is no input to `derived`/`approximated`
#: outside this same map.
PROVENANCE_MIXED_REASON = "provenance_fields_mixed"
PROVENANCE_EXCLUDED = provenance.SUBSTITUTED | {
    provenance.SOURCE_DERIVED_EMPTY, provenance.SOURCE_DERIVED, provenance.SOURCE_APPROXIMATED,
}
PROVENANCE_VOCABULARY = frozenset({PROVENANCE_MIXED_REASON}) | PROVENANCE_EXCLUDED
#: The classes that are themselves a reading this tick (directly, or from an
#: external feed) rather than something computed or substituted -- the
#: positive test `_axis_provenance` actually applies (D7).
PROVENANCE_PRIMARY_EVIDENCE = frozenset({
    provenance.SOURCE_MEASURED, provenance.SOURCE_MEASURED_CONVERTED,
    provenance.SOURCE_MEASURED_ARRIVAL_PROXY, provenance.SOURCE_FEED,
})

#: The three words `RuleCheck.status` may hold. Duplicated from
#: `policy.sensing_controller` rather than imported as a set, because that
#: module states them as three separate names, not one collection.
VALID_RULE_STATUSES = frozenset({RULE_FIRED, RULE_QUIET, RULE_NOT_EVALUABLE})

#: Why an axis whose instrument depends on a phone cannot be built at all --
#: distinct from `attempted: 0`, which is a real fact about a drive that DID
#: have the instrument and measured nothing (task 37's drive). A drive with no
#: phone has no rates/api_calls/triggers axis to be zero.
NOT_A_PHONE_RUN = "no tick carries a sensing block (not a phone run)"

#: D6: distinct from `NOT_A_PHONE_RUN` -- a phone run, on a log recorded
#: before `here_calls`/`here_errors` existed. Shared verbatim by
#: `_axis_api_calls`'s `unbuildable`, `_here_calls_by_session`'s
#: `not_measured`, and `_reconcile_reference_shape`'s `unavailable` detail,
#: so the three surfaces that used to classify this shape three different
#: ways (unbuildable nowhere on the axis; a measured zero on the HERE line;
#: a shape violation on the reconciliation) now name the same fact in the
#: same words.
HERE_CALLS_PREDATES_TASK_39 = "no tick carries here_calls (log predates task 39)"


@dataclass(frozen=True)
class AxisResult:
    """One session-summary axis: two independently counted integers and,
    only when they differ, the census of the axis's own reason words (§2/§3).

    Never a ratio, a percentage or a single summarising word -- see
    `session_summary`'s own shape test for the enforcement of that.
    """

    axis: str
    attempted: int | None
    answered: int | None
    attempted_is: str
    answered_is: str
    unanswered_by_reason: dict[str, int]
    vocabulary: str
    vocabulary_violations: dict[str, int]
    unbuildable: str | None
    section: str
    #: An auxiliary census that never touches rule 2's identity (kept off
    #: `unanswered_by_reason` on purpose): per-rule count of ticks where that
    #: rule was legitimately `not_evaluable` on `triggers`; per-source count
    #: of unreadable passes (`passes_attempted - passes_readable`, from
    #: `summary["failures"]["sources"]`) on `failures` (D2 -- `failures`
    #: previously had no equivalent, so a tick-level freshness count of
    #: `749 of 749 answered` could stand beside `620 of 930 source-passes
    #: unreadable` in `## Failures` with nothing on the axis record itself
    #: saying so). Empty on the other five axes.
    not_evaluable_by_rule: dict[str, int] = field(default_factory=dict)
    #: Set only when `attempted == 0` (`thermal`/`failures`): the one shape
    #: where the reason census is structurally empty, because there are no
    #: per-tick records to read a reason off (D4). Distinguishes "this
    #: instrument was never configured for this drive" from the more severe
    #: "summary.json shows this instrument ran, but no tick carries its
    #: block at all" -- worse than any tick reading `absent`, and otherwise
    #: indistinguishable from ordinary non-use.
    zero_attempted_context: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "axis": self.axis, "attempted": self.attempted, "answered": self.answered,
            "attempted_is": self.attempted_is, "answered_is": self.answered_is,
            "unanswered_by_reason": dict(self.unanswered_by_reason),
            "vocabulary": self.vocabulary,
            "vocabulary_violations": dict(self.vocabulary_violations),
            "unbuildable": self.unbuildable, "section": self.section,
            "not_evaluable_by_rule": dict(self.not_evaluable_by_rule),
            "zero_attempted_context": self.zero_attempted_context,
        }


@dataclass(frozen=True)
class Reconciliation:
    """One cross-check between two accounts of the same fact (§6 rules 11-19).

    `status` is `"held"`, `"failed"` (both sides computed and disagree -- the
    `detail` names both numbers) or `"unavailable"` (one side could not be
    computed at all -- the `detail` names which input was missing). Never
    resolved in either direction: a disagreement is reported, not decided (D12).
    """

    name: str
    status: str
    detail: str | None = None
    #: The size of the population actually compared -- a row count or a
    #: record count, named by `compared_is` (`"rows_compared"` or
    #: `"records_compared"`). `None` is only for the (rare) reconciliation
    #: that has not been updated to report one; every reconciliation this
    #: task's fix touches sets it, and a check whose population is 0 reports
    #: `"unavailable"`, never `"held"` -- a check that compared nothing is not
    #: a check that passed.
    compared: int | None = None
    compared_is: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.detail is not None:
            record["detail"] = self.detail
        if self.compared is not None and self.compared_is is not None:
            record[self.compared_is] = self.compared
        return record


def _census_and_violations(
    counts: dict[str, int], vocabulary: frozenset[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """`unanswered_by_reason` keeps every key verbatim; `vocabulary_violations`
    is the subset not in `vocabulary`, at the same count (§2, §6 rule 3) -- a
    word outside the declared set is counted, never absorbed into one already
    in it.
    """
    violations = {k: v for k, v in counts.items() if k not in vocabulary}
    return dict(counts), violations


def _sensing_ticks(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in ticks if t.get("sensing") is not None]


def _axis_latency(ticks: list[dict[str, Any]]) -> AxisResult:
    attempted = len(ticks)
    answered = 0
    unanswered = 0
    for t in ticks:
        if t.get("jetson_ms") is not None:
            answered += 1
        else:
            unanswered += 1
    counts = {LATENCY_ABSENT_REASON: unanswered} if unanswered else {}
    census, violations = _census_and_violations(counts, LATENCY_VOCABULARY)
    return AxisResult(
        axis="latency", attempted=attempted, answered=answered,
        attempted_is="ticks", answered_is="ticks with jetson_ms non-null",
        unanswered_by_reason=census, vocabulary="eval_run.LATENCY_VOCABULARY",
        vocabulary_violations=violations, unbuildable=None,
        section="## Latency (full run)",
    )


#: D4's two `zero_attempted_context` messages, shared by `thermal` and
#: `failures` -- both read `attempted == 0` off per-tick records, so the
#: only other fact available to explain it is whether `summary.json` itself
#: carries a block for the instrument.
def _zero_attempted_context(summary: dict[str, Any] | None, name: str) -> str:
    if (summary or {}).get(name):
        return (
            f"summary.json carries a {name!r} block for this drive, but no tick "
            f"carries a {name} block at all -- worse than any tick reading absent"
        )
    return (
        f"no tick carries a {name} block and summary.json carries no {name!r} "
        f"block either ({name} was not enabled for this drive)"
    )


def _axis_thermal(ticks: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> AxisResult:
    attempted = 0
    answered = 0
    counts: dict[str, int] = {}
    for t in ticks:
        block = t.get("thermal")
        if not block:
            continue
        jetson = block.get("jetson") or {}
        basis = jetson.get("basis")
        if basis is None:
            continue
        attempted += 1
        if basis == STAGE_BASIS_MEASURED:
            answered += 1
        elif basis == THERMAL_BASIS_STALE:
            counts[THERMAL_BASIS_STALE] = counts.get(THERMAL_BASIS_STALE, 0) + 1
        else:
            reason = jetson.get("reason") or "unstated"
            counts[reason] = counts.get(reason, 0) + 1
    census, violations = _census_and_violations(counts, THERMAL_FAILURES_VOCABULARY)
    return AxisResult(
        axis="thermal", attempted=attempted, answered=answered,
        attempted_is="ticks carrying a thermal block",
        answered_is="ticks whose thermal.jetson.basis is measured",
        unanswered_by_reason=census,
        vocabulary="sensors.thermal.ABSENT_REASONS (+ sensors.thermal.THERMAL_BASIS_STALE)",
        vocabulary_violations=violations, unbuildable=None, section="## Thermal",
        zero_attempted_context=(
            _zero_attempted_context(summary, "thermal") if attempted == 0 else None
        ),
    )


def _axis_failures(ticks: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> AxisResult:
    attempted = 0
    answered = 0
    counts: dict[str, int] = {}
    for t in ticks:
        block = t.get("failures")
        if not block:
            continue
        basis = block.get("basis")
        if basis is None:
            continue
        attempted += 1
        if basis == STAGE_BASIS_MEASURED:
            answered += 1
        elif basis == THERMAL_BASIS_STALE:
            counts[THERMAL_BASIS_STALE] = counts.get(THERMAL_BASIS_STALE, 0) + 1
        else:
            reason = block.get("reason") or "unstated"
            counts[reason] = counts.get(reason, 0) + 1
    census, violations = _census_and_violations(counts, THERMAL_FAILURES_VOCABULARY)
    # D2: `answered`/`attempted` above are a tick-level FRESHNESS count
    # (`failures.basis`) -- they say nothing about whether the underlying
    # failure SOURCES could be read at all, which is a different population
    # `## Failures` already reports per source. `not_evaluable_by_rule`
    # (reused here keyed by source name, not by rule) carries that second
    # population onto the axis record itself, the way `triggers` already
    # does for its own rules -- without touching rule 2's identity, since it
    # is not part of `unanswered_by_reason`.
    not_evaluable_sources: dict[str, int] = {}
    for source_name, row in (((summary or {}).get("failures") or {}).get("sources") or {}).items():
        unreadable = (row.get("passes_attempted", 0) or 0) - (row.get("passes_readable", 0) or 0)
        if unreadable > 0:
            not_evaluable_sources[source_name] = unreadable
    return AxisResult(
        axis="failures", attempted=attempted, answered=answered,
        attempted_is="ticks carrying a failures block",
        answered_is="ticks whose failures.basis is measured",
        unanswered_by_reason=census,
        vocabulary="sensors.thermal.ABSENT_REASONS (+ sensors.thermal.THERMAL_BASIS_STALE)",
        vocabulary_violations=violations, unbuildable=None, section="## Failures",
        not_evaluable_by_rule=not_evaluable_sources,
        zero_attempted_context=(
            _zero_attempted_context(summary, "failures") if attempted == 0 else None
        ),
    )


def _axis_provenance(ticks: list[dict[str, Any]]) -> AxisResult:
    """Answers "did this tick's provenance map cover the encoder contract
    *and measure something*" (M4) -- a map of the right shape whose every
    field is `perception.provenance.SUBSTITUTED`, `derived_empty`, or
    `derived`/`approximated` with no primary evidence anywhere else in the
    map (D7) is a drive that measured nothing, and a shape check alone reads
    that as a full house.
    """
    attempted = len(ticks)
    answered = 0
    encoder_slots = set(sim_contract.encoded_slot_names())
    counts: dict[str, int] = {}
    for t in ticks:
        field_sources = t.get("field_sources") or {}
        keys = set(field_sources)
        if keys != encoder_slots:
            if len(keys) != len(encoder_slots):
                key = f"short: {len(keys)}" if len(keys) < len(encoder_slots) else f"long: {len(keys)}"
            else:
                key = PROVENANCE_MIXED_REASON
            counts[key] = counts.get(key, 0) + 1
            continue
        classes_present = set(field_sources.values())
        # D7: `derived`/`approximated` are not primary evidence on their own
        # (see PROVENANCE_PRIMARY_EVIDENCE's docstring) -- a tick answers
        # only when some field carries a class that is actually a reading,
        # which is what a genuine derivation would have been built from.
        if classes_present & PROVENANCE_PRIMARY_EVIDENCE:
            answered += 1
        else:
            # Every field is substituted, derived from an absence, or
            # derived/approximated with nothing else in the map to have been
            # derived from: censused under whichever such class dominates
            # this tick's own map, using ## Observation quality's own
            # by_source vocabulary rather than a new word.
            dominant = Counter(field_sources.values()).most_common(1)[0][0]
            counts[dominant] = counts.get(dominant, 0) + 1
    census, violations = _census_and_violations(counts, PROVENANCE_VOCABULARY)
    return AxisResult(
        axis="provenance", attempted=attempted, answered=answered,
        attempted_is="ticks",
        answered_is=(
            "ticks whose field_sources key set equals sim_contract.encoded_slot_names() "
            "and carry at least one field in eval_run.PROVENANCE_PRIMARY_EVIDENCE"
        ),
        unanswered_by_reason=census, vocabulary="eval_run.PROVENANCE_VOCABULARY",
        vocabulary_violations=violations, unbuildable=None, section="## Observation quality",
    )


def _axis_rates(ticks: list[dict[str, Any]]) -> AxisResult:
    sensing_ticks = _sensing_ticks(ticks)
    if not sensing_ticks:
        return AxisResult(
            axis="rates", attempted=None, answered=None,
            attempted_is="distinct telemetry reports observed (sensing.reference.at_mono)",
            answered_is="reports fresh by the controller's own staleness predicate",
            unanswered_by_reason={}, vocabulary="policy.sensing_loop.reference_from",
            vocabulary_violations={}, unbuildable=NOT_A_PHONE_RUN, section="## Sensing",
        )
    no_telemetry_ticks = 0
    ages_by_report: dict[Any, float | None] = {}
    for t in sensing_ticks:
        ref = t["sensing"].get("reference") or {}
        if ref.get("absent") is not None:
            no_telemetry_ticks += 1
            continue
        ages_by_report[ref.get("at_mono")] = ref.get("age_s")
    fresh = 0
    stale = 0
    for age in ages_by_report.values():
        if _is_fresh(age):
            fresh += 1
        else:
            stale += 1
    attempted = len(ages_by_report) + no_telemetry_ticks
    counts: dict[str, int] = {}
    if stale:
        counts[REFERENCE_STALE] = stale
    if no_telemetry_ticks:
        counts[REFERENCE_NO_TELEMETRY] = no_telemetry_ticks
    census, violations = _census_and_violations(counts, RATES_VOCABULARY)
    return AxisResult(
        axis="rates", attempted=attempted, answered=fresh,
        attempted_is=(
            "distinct telemetry reports observed (sensing.reference.at_mono), "
            "plus one per tick observing no telemetry"
        ),
        answered_is="reports fresh by the controller's own staleness predicate",
        unanswered_by_reason=census, vocabulary="policy.sensing_loop.reference_from",
        vocabulary_violations=violations, unbuildable=None, section="## Sensing",
    )


def _axis_api_calls(ticks: list[dict[str, Any]]) -> AxisResult:
    """Answers "did this tick's telemetry report a HERE call count" (M1).
    Branches on `reference.absent`, the field that actually says whether
    telemetry arrived, rather than on `here_calls is None` alone: the two
    agree under the shape rule (§6 rule 8), but a record that violates it --
    `absent` set *and* `here_calls` a real number -- must still census under
    `no_telemetry`, not read as answered.
    """
    sensing_ticks = _sensing_ticks(ticks)
    if not sensing_ticks:
        return AxisResult(
            axis="api_calls", attempted=None, answered=None,
            attempted_is="ticks carrying a sensing.reference block",
            answered_is="ticks whose reference.absent is None and here_calls is non-null",
            unanswered_by_reason={}, vocabulary="policy.sensing_loop.reference_from",
            vocabulary_violations={}, unbuildable=NOT_A_PHONE_RUN, section="## Sensing",
        )
    # D6: a phone run whose log predates `here_calls`/`here_errors` (neither
    # key exists on `reference`, on any tick -- distinct from the modern
    # `absent` branch, which sets the keys present but null) is a different
    # fact from a phone that failed to answer: it is a build the field could
    # not have reached. `_here_calls_by_session`/`_reconcile_reference_shape`
    # already carve this out; the axis previously did not, and counted every
    # such tick under `field_not_recorded` -- a real answered/unanswered
    # count for a shape that could never have answered at all.
    if not any("here_calls" in (t["sensing"].get("reference") or {}) for t in sensing_ticks):
        return AxisResult(
            axis="api_calls", attempted=None, answered=None,
            attempted_is="ticks carrying a sensing.reference block",
            answered_is="ticks whose reference.absent is None and here_calls is non-null",
            unanswered_by_reason={}, vocabulary="policy.sensing_loop.reference_from",
            vocabulary_violations={}, unbuildable=HERE_CALLS_PREDATES_TASK_39, section="## Sensing",
        )
    attempted = len(sensing_ticks)
    answered = 0
    counts: dict[str, int] = {}
    for t in sensing_ticks:
        ref = t["sensing"].get("reference") or {}
        if ref.get("absent") is not None:
            counts[REFERENCE_NO_TELEMETRY] = counts.get(REFERENCE_NO_TELEMETRY, 0) + 1
        elif ref.get("here_calls") is None:
            # Telemetry was present (absent is None) but here_calls is still
            # null: the shape rule forbids this, so it is a distinct word
            # from no_telemetry rather than silently merged into it.
            counts[API_CALLS_FIELD_NOT_RECORDED] = counts.get(API_CALLS_FIELD_NOT_RECORDED, 0) + 1
        else:
            answered += 1
    census, violations = _census_and_violations(counts, API_CALLS_VOCABULARY)
    return AxisResult(
        axis="api_calls", attempted=attempted, answered=answered,
        attempted_is="ticks carrying a sensing.reference block",
        answered_is="ticks whose reference.absent is None and here_calls is non-null",
        unanswered_by_reason=census, vocabulary="policy.sensing_loop.reference_from",
        vocabulary_violations=violations, unbuildable=None, section="## Sensing",
    )


def _axis_triggers(ticks: list[dict[str, Any]]) -> AxisResult:
    sensing_ticks = _sensing_ticks(ticks)
    if not sensing_ticks:
        return AxisResult(
            axis="triggers", attempted=None, answered=None,
            attempted_is="ticks carrying a sensing block",
            answered_is="ticks whose attribution carries all four RULES with a valid status",
            unanswered_by_reason={}, vocabulary="policy.sensing_controller.RULES",
            vocabulary_violations={}, unbuildable=NOT_A_PHONE_RUN, section="## Sensing",
        )
    attempted = len(sensing_ticks)
    answered = 0
    counts: dict[str, int] = {}
    not_evaluable_by_rule: dict[str, int] = {}
    for t in sensing_ticks:
        rules = t["sensing"].get("attribution", {}).get("rules", {})
        valid = set(rules) == set(RULES) and all(
            rules[name].get("status") in VALID_RULE_STATUSES for name in RULES
        )
        if valid:
            answered += 1
            # A rule being not_evaluable is a legitimate state (D3), so it
            # never moves `unanswered_by_reason` -- but it is real information
            # this axis record would otherwise carry no trace of at all (M5).
            for name in RULES:
                if rules[name].get("status") == RULE_NOT_EVALUABLE:
                    not_evaluable_by_rule[name] = not_evaluable_by_rule.get(name, 0) + 1
        else:
            counts[TRIGGERS_UNANSWERED_REASON] = counts.get(TRIGGERS_UNANSWERED_REASON, 0) + 1
    census, violations = _census_and_violations(counts, TRIGGERS_VOCABULARY)
    return AxisResult(
        axis="triggers", attempted=attempted, answered=answered,
        attempted_is="ticks carrying a sensing block",
        answered_is="ticks whose attribution carries all four RULES with a valid status",
        unanswered_by_reason=census, vocabulary="policy.sensing_controller.RULES",
        vocabulary_violations=violations, unbuildable=None, section="## Sensing",
        not_evaluable_by_rule=not_evaluable_by_rule,
    )


#: `thermal`/`failures` take an optional `summary` second argument (D2/D4);
#: the other five take `ticks` alone. `session_summary` special-cases the
#: call, so the type here is deliberately loose rather than a lie.
_AXIS_BUILDERS: dict[str, Callable[..., AxisResult]] = {
    "latency": _axis_latency,
    "rates": _axis_rates,
    "api_calls": _axis_api_calls,
    "triggers": _axis_triggers,
    "failures": _axis_failures,
    "thermal": _axis_thermal,
    "provenance": _axis_provenance,
}


def _is_fresh(age_s: float | None) -> bool:
    """The incumbent's own staleness predicate (`score_shadow._is_stale_report`,
    negated): a report is fresh when its age is finite and within
    `MAX_TELEMETRY_AGE_S`. An age that is `None` or non-finite is not fresh --
    there is nothing to confirm freshness from.
    """
    return age_s is not None and math.isfinite(age_s) and abs(age_s) <= MAX_TELEMETRY_AGE_S


def _distinct_reports(sensing_ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One `reference` per distinct `at_mono`, in arrival order (D7). The
    content behind two ticks sharing an `at_mono` is identical --
    `PhoneLink.telemetry` holds one report at a time -- so keeping the last
    tick's copy loses nothing.
    """
    by_at_mono: dict[Any, dict[str, Any]] = {}
    for t in sensing_ticks:
        ref = t["sensing"].get("reference") or {}
        if ref.get("absent") is None and ref.get("at_mono") is not None:
            by_at_mono[ref["at_mono"]] = ref
    return [by_at_mono[key] for key in sorted(by_at_mono)]


def _telemetry_window(sensing_ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """The median gap between distinct telemetry arrivals (D10) -- measured
    from the arrival instants this side observed, not mirrored from the
    phone's own `PERIOD_MS` constant. Unbuildable on fewer than two reports:
    a single arrival has no gap to measure.
    """
    at_monos = sorted({r["at_mono"] for r in _distinct_reports(sensing_ticks)})
    reports = len(at_monos)
    if reports < 2:
        return {
            "observed_median_s": None, "reports": reports,
            "unbuildable": "fewer than two distinct telemetry reports",
        }
    diffs = sorted(b - a for a, b in zip(at_monos, at_monos[1:]))
    mid = len(diffs) // 2
    median = diffs[mid] if len(diffs) % 2 else (diffs[mid - 1] + diffs[mid]) / 2.0
    return {"observed_median_s": median, "reports": reports, "unbuildable": None}


def _time_weighted_weights(times: list[float]) -> list[float]:
    """One weight per point in `times` (already sorted): the gap to the next
    point, and, for the last point, the gap immediately before it again --
    the best available estimate of how long its value would have held, since
    a drive's own tick spacing is usually close to constant and there is no
    later point to measure it from. A single point gets weight 1.0 (any
    positive constant works, since it cancels in a mean).

    Each gap is capped at `TICK_COVERAGE_GAP_MULTIPLE` times the median gap
    in this same series -- the criterion `_tick_coverage` uses to call a gap
    an interruption rather than jitter. Uncapped, a gap spanning a stretch
    where the system was not running at all (no ticks, so no point exists in
    that stretch) is credited entirely to whatever value the point before it
    held, which states that value was in effect for the whole outage. The
    cap bounds that credit to what an ordinary gap in this series looks
    like.
    """
    if len(times) == 1:
        return [1.0]
    gaps = [b - a for a, b in zip(times, times[1:])]
    sorted_gaps = sorted(gaps)
    mid = len(sorted_gaps) // 2
    median_gap = (
        sorted_gaps[mid] if len(sorted_gaps) % 2 else (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2.0
    )
    cap = TICK_COVERAGE_GAP_MULTIPLE * median_gap if median_gap > 0 else None
    capped = gaps if cap is None else [min(g, cap) for g in gaps]
    return capped + [capped[-1]]


@dataclass(frozen=True)
class TimeWeightedStat:
    """A time-weighted statistic (`_time_weighted_mean`'s mean, or
    `_time_weighted_integral`'s integral) together with the window it was
    computed over. `span_s` is the raw elapsed time between the first and
    last point, uncapped. `weighted_over_s` is the total of the
    (possibly capped) per-point weights `value` was actually computed from.
    The two agree, up to one trailing extrapolated gap, on a series with no
    capped gap; they diverge when a gap got capped, and the difference
    between them is real elapsed time `value` says nothing about.
    """

    value: float | None
    span_s: float
    weighted_over_s: float


def _time_weighted_mean(points: list[tuple[float, float]]) -> TimeWeightedStat:
    """Mean of `value`, weighted by how long (wall/monotonic seconds) it held
    -- not a mean over samples, which would overweight whatever period
    happened to produce more ticks. On an evenly spaced series with no
    capped gap this reduces to the ordinary sample mean.
    """
    if not points:
        return TimeWeightedStat(None, 0.0, 0.0)
    ordered = sorted(points, key=lambda p: p[0])
    times = [t for t, _ in ordered]
    weights = _time_weighted_weights(times)
    span_s = times[-1] - times[0]
    weighted_over_s = sum(weights)
    if weighted_over_s <= 0:
        return TimeWeightedStat(float(ordered[-1][1]), span_s, weighted_over_s)
    mean = sum(v * w for (_, v), w in zip(ordered, weights)) / weighted_over_s
    return TimeWeightedStat(mean, span_s, weighted_over_s)


def _time_weighted_integral(points: list[tuple[float, float]]) -> TimeWeightedStat:
    """`sum(value * weight)` over the same weighting as `_time_weighted_mean`
    -- the expected count of events a rate would produce over the span
    covered.
    """
    if len(points) < 2:
        return TimeWeightedStat(0.0, 0.0, 0.0)
    ordered = sorted(points, key=lambda p: p[0])
    times = [t for t, _ in ordered]
    weights = _time_weighted_weights(times)
    integral = sum(v * w for (_, v), w in zip(ordered, weights))
    return TimeWeightedStat(integral, times[-1] - times[0], sum(weights))


def _comparable(ever_live: bool, distinct_reports: list[dict[str, Any]], ticks: int) -> tuple[bool, str | None]:
    """Whether achieved is a meaningful comparison against commanded (D8).
    False on a pure-shadow drive -- the phone was never told to run the
    commanded rates at all (`ConfigApplier.apply` returns before touching any
    rate on the shadow branch) -- and false when there is no fresh telemetry
    to compare, or too little of it to say anything about a window.
    """
    if not ever_live:
        return False, f"mode shadow on {ticks} of {ticks} decisions"
    if not distinct_reports:
        return False, "no fresh telemetry report observed"
    if len(distinct_reports) < 2:
        return False, "fewer than two distinct telemetry reports; the window is unmeasurable"
    return True, None


def _tally_triggers(ticks: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """`decisions_by_trigger` and `rules_by_status`, computed from tick
    records alone -- the same rollup `SensingLoop` keeps live, recomputed
    here so a log can be cross-checked against its own summary (reconciliation
    18) rather than trusted on its word.
    """
    by_trigger: dict[str, int] = {}
    rules_by_status: dict[str, dict[str, int]] = {}
    for t in ticks:
        sensing = t.get("sensing")
        if sensing is None:
            continue
        trigger = sensing.get("trigger")
        by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
        for rule_name, check in sensing.get("attribution", {}).get("rules", {}).items():
            counts = rules_by_status.setdefault(rule_name, {})
            status = check.get("status")
            counts[status] = counts.get(status, 0) + 1
    return by_trigger, rules_by_status


def _tally_rules_missing(ticks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """`{rule: {missing_field: n}}`, over ticks where that rule was
    `not_evaluable` -- the field name a `not_evaluable` verdict names, not
    just how often it fired (§7's own fixture for this).
    """
    out: dict[str, dict[str, int]] = {}
    for t in ticks:
        sensing = t.get("sensing")
        if sensing is None:
            continue
        for rule_name, check in sensing.get("attribution", {}).get("rules", {}).items():
            if check.get("status") != RULE_NOT_EVALUABLE:
                continue
            for field_name in check.get("missing") or ():
                bucket = out.setdefault(rule_name, {})
                bucket[field_name] = bucket.get(field_name, 0) + 1
    return out


def _here_calls_by_session(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-`session_id`, `last - first` of `here_calls`/`here_errors`, summed
    (D11). `PhoneLink._rebind` clears telemetry on every redial and a
    different handset's `HerePipeline.calls` restarts at zero, so `last -
    first` ACROSS a redial can be negative; splitting by session and clamping
    each session's own delta at zero avoids that, at the cost of not counting
    calls placed before a session's first observed report (`uncounted_prefix`,
    a bound rather than a measurement -- open item 1).

    `calls_total`/`errors_total`/`uncounted_prefix` are `None`, with
    `not_measured` naming why, when no tick carries `here_calls` at all --
    a log that predates this field measured nothing, and a `0` there would
    read as a phone that placed zero calls (C2). A session whose own counter
    decreases (`last < first`) is still clamped to zero for the total, since
    a negative call count cannot be reported, but the session is named under
    `counter_went_backwards` (task 38's own word for the same situation on
    its own counters, M7) rather than the decrease being silently absorbed.
    """
    by_session: dict[Any, dict[str, list[int]]] = {}
    ticks_with_here_calls = 0
    for t in ticks:
        sensing = t.get("sensing")
        if sensing is None:
            continue
        ref = sensing.get("reference") or {}
        calls = ref.get("here_calls")
        if calls is None:
            continue
        ticks_with_here_calls += 1
        errors = ref.get("here_errors") or 0
        bucket = by_session.setdefault(t.get("session_id"), {"calls": [], "errors": []})
        bucket["calls"].append(calls)
        bucket["errors"].append(errors)
    rows = []
    calls_total = 0
    errors_total = 0
    uncounted_prefix = 0
    backwards_sessions = []
    for session_id, values in by_session.items():
        first_calls, last_calls = values["calls"][0], values["calls"][-1]
        first_errors, last_errors = values["errors"][0], values["errors"][-1]
        rows.append({
            "session_id": session_id, "first": first_calls, "last": last_calls,
            "observations": len(values["calls"]),
        })
        if last_calls - first_calls < 0:
            backwards_sessions.append(session_id)
        calls_total += max(0, last_calls - first_calls)
        errors_total += max(0, last_errors - first_errors)
        uncounted_prefix += first_calls
    not_measured = None if ticks_with_here_calls else HERE_CALLS_PREDATES_TASK_39
    return {
        "by_session": rows,
        "calls_total": None if not_measured else calls_total,
        "errors_total": None if not_measured else errors_total,
        "uncounted_prefix": None if not_measured else uncounted_prefix,
        "not_measured": not_measured,
        "counter_went_backwards": backwards_sessions,
        "ticks_with_here_calls": ticks_with_here_calls,
    }


def sensing_result(
    ticks: list[dict[str, Any]], summary: dict[str, Any],
    *, phone_offline_kinds: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """The rate, trigger and HERE detail behind the `rates`/`triggers`/
    `api_calls` axes -- `None` on a run with no phone at all, the same
    reading `thermal_result` gives a run that predates thermal.

    `phone_offline_kinds` (D5) is the phone's own session-log failure kinds
    (`PHONE_OFFLINE_KINDS`), when known -- used only to widen
    `here["zero_calls_because"]`; empty leaves that computation exactly as
    it was before D5.
    """
    sensing_ticks = _sensing_ticks(ticks)
    if not sensing_ticks:
        return None

    sensing_summary = (summary or {}).get("sensing")
    mode_block = (sensing_summary or {}).get("mode") or {}
    # `ever_live` scans every tick; the fallback below is built from the same
    # scan rather than the first tick alone, so the two can never contradict
    # each other -- a drive whose first tick happened to be shadow could
    # otherwise be reported as a shadow drive while `ever_live` said True.
    ever_live = any(t["sensing"].get("shadow") is False for t in sensing_ticks)
    mode = mode_block.get("mode")
    if mode is not None:
        mode_source = "summary[sensing].mode.mode"
    else:
        mode = LIVE if ever_live else SHADOW
        mode_source = "derived from whether any tick's own shadow flag is False (no summary.json)"

    computed_by_trigger, computed_rules_by_status = _tally_triggers(ticks)
    rules_missing = _tally_rules_missing(ticks)
    triggers_summary_agrees = (
        None if sensing_summary is None
        else (
            computed_by_trigger == sensing_summary.get("decisions_by_trigger")
            and computed_rules_by_status == sensing_summary.get("rules_by_status")
        )
    )

    window = _telemetry_window(sensing_ticks)
    distinct_reports = _distinct_reports(sensing_ticks)
    comparable, not_comparable_because = _comparable(ever_live, distinct_reports, len(sensing_ticks))

    rates: dict[str, Any] = {}
    for key in RATE_KEYS:
        points = [
            (t["sensing"].get("decided_at_mono"), (t["sensing"].get("rates") or {}).get(key))
            for t in sensing_ticks
        ]
        # A malformed tick (missing decided_at_mono or this rate key) is
        # dropped rather than raising -- one bad record should not abort the
        # whole report, including the axes that would have survived it.
        points = [(t_mono, v) for t_mono, v in points if t_mono is not None and v is not None]
        census = Counter(str(v) for _, v in points)
        commanded_stat = _time_weighted_mean(points)
        commanded_time_mean = commanded_stat.value
        clamped_ticks = sum(
            1 for t in sensing_ticks
            if ((t["sensing"].get("attribution") or {}).get("per_sensor") or {}).get(key, {}).get("clamped")
        )
        thermal_scaled_ticks = sum(
            1 for t in sensing_ticks
            if (((t["sensing"].get("attribution") or {}).get("per_sensor") or {}).get(key) or {})
            .get("scale", 1.0) != 1.0
        )
        entry: dict[str, Any] = {
            "commanded_by_value": dict(census),
            "commanded_distinct": len(census),
            "commanded_time_mean": commanded_time_mean,
            "commanded_span_s": commanded_stat.span_s,
            "commanded_weighted_over_s": commanded_stat.weighted_over_s,
            "clamped_ticks": clamped_ticks,
            "thermal_scaled_ticks": thermal_scaled_ticks,
            "achieved_time_mean": None,
            "achieved_span_s": None,
            "achieved_weighted_over_s": None,
            "lambda_per_window": None,
            "percentiles": {"p50": None, "p95": None},
            "percentiles_suppressed": None,
            "comparable": comparable,
            "not_comparable_because": not_comparable_because,
        }
        if comparable:
            achieved_points = [
                (r.get("at_mono"), (r.get("achieved") or {}).get(key)) for r in distinct_reports
            ]
            achieved_points = [(t_mono, v) for t_mono, v in achieved_points if t_mono is not None and v is not None]
            achieved_stat = _time_weighted_mean(achieved_points)
            entry["achieved_time_mean"] = achieved_stat.value
            entry["achieved_span_s"] = achieved_stat.span_s
            entry["achieved_weighted_over_s"] = achieved_stat.weighted_over_s
            lambda_per_window = (
                None if window["observed_median_s"] is None or commanded_time_mean is None
                else commanded_time_mean * window["observed_median_s"]
            )
            entry["lambda_per_window"] = lambda_per_window
            if lambda_per_window is None:
                pass
            elif lambda_per_window < 1.0:
                windows_per_delivery = 1.0 / lambda_per_window if lambda_per_window > 0 else float("inf")
                entry["percentiles_suppressed"] = (
                    f"commanded {commanded_time_mean:g} Hz over a "
                    f"{window['observed_median_s']:.3f} s window is one delivery per "
                    f"{windows_per_delivery:.0f} windows; a median over per-window rates "
                    "would be 0.0 and is not a statement about the rate"
                )
            elif achieved_points:
                stats = pctl([v for _, v in achieved_points])
                entry["percentiles"] = {"p50": stats["p50"], "p95": stats["p95"]}
                if lambda_per_window == 1.0:
                    # D9: the percentiles ARE reported here, unlike the
                    # `< 1.0` branch above -- this names the quantisation
                    # rather than hiding the numbers behind it.
                    entry["percentiles_suppressed"] = (
                        f"not suppressed, but quantised to multiples of "
                        f"{1.0 / window['observed_median_s']:g} Hz: commanded "
                        f"{commanded_time_mean:g} Hz over a {window['observed_median_s']:.3f} s window "
                        "delivers exactly one event per window"
                    )
        rates[key] = entry

    here = _here_calls_by_session(ticks)
    here_points = [
        (t["sensing"].get("decided_at_mono"), (t["sensing"].get("rates") or {}).get("here_hz"))
        for t in sensing_ticks
    ]
    here_points = [(t_mono, v) for t_mono, v in here_points if t_mono is not None and v is not None]
    here_stat = _time_weighted_integral(here_points)
    here["expected_from_commanded"] = here_stat.value
    here["expected_from_commanded_span_s"] = here_stat.span_s
    here["expected_from_commanded_weighted_over_s"] = here_stat.weighted_over_s
    here["responses_received"] = (
        ((summary or {}).get("phone") or {}).get("here") or {}
    ).get("responses_received")
    # D5: a shadow mode and a missing HERE API key are both, independently,
    # sufficient to place zero calls, and only reporting the first invites a
    # false counterfactual -- "a live-mode drive would have placed the ~N
    # calls this line computes as expected" -- when the build could not have
    # placed any regardless of mode. Both true causes are named when both
    # apply; neither is preferred over the other, per D12.
    zero_calls_causes: list[str] = []
    if (
        here["calls_total"] == 0
        and any(row["observations"] >= 2 for row in here["by_session"])
    ):
        if mode == SHADOW:
            zero_calls_causes.append(f"mode {mode}; a shadow command never reaches setHereQuery")
        if "here.unconfigured" in phone_offline_kinds:
            zero_calls_causes.append(
                "the phone recorded here.unconfigured (no HERE API key in this build); "
                "a live-mode drive would also place zero calls"
            )
    here["zero_calls_because"] = "; and ".join(zero_calls_causes) if zero_calls_causes else None

    return {
        "present": True,
        "ticks": len(sensing_ticks),
        "mode": mode,
        "mode_source": mode_source,
        "ever_live": ever_live,
        "telemetry_window": window,
        "rates": rates,
        "triggers": {
            "decisions_by_trigger": computed_by_trigger,
            "rules_by_status": computed_rules_by_status,
            "rules_missing": rules_missing,
            "summary_agrees": triggers_summary_agrees,
        },
        "here": here,
    }


def _reconcile_rows(
    name: str, failures_summary: dict[str, Any], predicate: Callable[[str, dict], bool],
    describe: Callable[[list[str]], str],
) -> Reconciliation:
    """C1: `bad` is empty both when every row satisfies `predicate` and when
    `sources` is empty -- the two used to render identically as `held`. The
    row count is carried and checked: zero rows compared is `unavailable`,
    never a check that passed.
    """
    sources = failures_summary.get("sources") or {}
    if not sources:
        return Reconciliation(name, "unavailable", "no source rows to compare",
                               compared=0, compared_is="rows_compared")
    bad = sorted(name for name, row in sources.items() if not predicate(name, row))
    if not bad:
        return Reconciliation(name, "held", compared=len(sources), compared_is="rows_compared")
    return Reconciliation(name, "failed", describe(bad), compared=len(sources), compared_is="rows_compared")


def _reconcile_blind_ticks(failures_summary: dict[str, Any]) -> Reconciliation:
    """C1's narrower case: a `camera.blind_ticks` row present but empty
    (`row.get("total")` absent) used to compare `None == None` against a
    missing `blind_ticks` field and read as `held` -- a record carrying
    neither number. Either side missing is `unavailable`, not a coincidental
    equality.
    """
    sources = failures_summary.get("sources") or {}
    row = sources.get("camera.blind_ticks")
    blind_ticks = failures_summary.get("blind_ticks")
    total = None if row is None else row.get("total")
    if total is None or blind_ticks is None:
        return Reconciliation(
            "blind_ticks_matches_camera_source", "unavailable",
            "camera.blind_ticks total or blind_ticks not present in this record",
            compared=0, compared_is="rows_compared",
        )
    if total == blind_ticks:
        return Reconciliation("blind_ticks_matches_camera_source", "held",
                               compared=1, compared_is="rows_compared")
    return Reconciliation(
        "blind_ticks_matches_camera_source", "failed",
        f'sources["camera.blind_ticks"].total is {total} and blind_ticks is {blind_ticks}, '
        "on the same record; this drive's failure counts are not usable",
        compared=1, compared_is="rows_compared",
    )


def _reconcile_source_status(failures_summary: dict[str, Any]) -> Reconciliation:
    return _reconcile_rows(
        "source_status_fired_iff_total_positive", failures_summary,
        lambda name, row: (row.get("status") == RULE_FIRED) == (row.get("total", 0) > 0),
        lambda bad: f"status disagrees with total on: {bad}",
    )


def _reconcile_source_totals(failures_summary: dict[str, Any]) -> Reconciliation:
    return _reconcile_rows(
        "source_totals_partition", failures_summary,
        lambda name, row: (
            row.get("kept_total", 0) + row.get("suppressed", 0) + row.get("below_episode_threshold", 0)
            == row.get("total", 0)
        ),
        lambda bad: f"kept_total + suppressed + below_episode_threshold != total on: {bad}",
    )


def _reconcile_passes(failures_summary: dict[str, Any]) -> Reconciliation:
    return _reconcile_rows(
        "source_passes_readable_le_attempted", failures_summary,
        lambda name, row: row.get("passes_readable", 0) <= row.get("passes_attempted", 0),
        lambda bad: f"passes_readable > passes_attempted on: {bad}",
    )


def _reconcile_source_count(failures_summary: dict[str, Any]) -> Reconciliation:
    sources = failures_summary.get("sources") or {}
    scan = failures_summary.get("scan") or {}
    n, scan_n = len(sources), scan.get("sources_n")
    if n == scan_n == 30:
        return Reconciliation("sources_count_is_30", "held", compared=n, compared_is="rows_compared")
    return Reconciliation(
        "sources_count_is_30", "failed", f"len(sources)={n}, scan.sources_n={scan_n}, expected 30 both",
        compared=n, compared_is="rows_compared",
    )


def _reconcile_events_written(loaded: "LoadedRecords", failures_summary: dict[str, Any]) -> Reconciliation:
    """`FailureLog._open_episode`/`_close_episode` each increment `events_written`
    once, on every write to the sink (`logio/failure_log.py:1313-1316,1344-1348)`
    -- once per open, once per close -- so the identity is against every
    `failure_event` record, not the `open` half of it alone. Verified against
    a real drive (31 open + 31 close = 62 records, `events_written` summed to
    62): the plan this task implements against states the narrower identity
    (open records only), which a truncated log with an odd number of
    `failure_event` lines would satisfy by coincidence and a normal,
    cleanly-closed drive never would.

    C1: `expected` sums over `sources`, so an empty `sources` map made this
    hold at `0 == 0` even on a drive with no source rows to sum at all.
    """
    sources = failures_summary.get("sources") or {}
    if not sources:
        return Reconciliation(
            "events_written_matches_open_and_close_records", "unavailable",
            "no source rows to sum events_written over", compared=0, compared_is="rows_compared",
        )
    expected = sum(row.get("events_written", 0) for row in sources.values())
    actual = sum(1 for r in loaded.failure_events if r.get("phase") in ("open", "close"))
    if expected == actual:
        return Reconciliation("events_written_matches_open_and_close_records", "held",
                               compared=len(sources), compared_is="rows_compared")
    return Reconciliation(
        "events_written_matches_open_and_close_records", "failed",
        f"sum(events_written)={expected}, open+close failure_event records in metadata.jsonl={actual}",
        compared=len(sources), compared_is="rows_compared",
    )


def _reconcile_thermal_samples(loaded: "LoadedRecords", thermal_summary: dict[str, Any]) -> Reconciliation:
    reported = (thermal_summary.get("jetson") or {}).get("samples")
    if reported is None:
        return Reconciliation(
            "thermal_samples_count", "unavailable", "summary[thermal][jetson] carries no samples count"
        )
    actual = len(loaded.thermal_samples)
    diff = reported - actual
    if diff in (0, 1):
        return Reconciliation("thermal_samples_count", "held", compared=actual, compared_is="records_compared")
    return Reconciliation(
        "thermal_samples_count", "failed",
        f"summary reports {reported} samples, {actual} thermal_sample records read "
        f"(difference {diff}, expected 0 or 1)",
        compared=actual, compared_is="records_compared",
    )


def _reconcile_triggers(loaded: "LoadedRecords", sensing_summary: dict[str, Any]) -> Reconciliation:
    """C1: with no sensing ticks at all, both the computed maps and an empty
    (or absent) summary rollup are `{}`, which used to compare equal and
    read as `held` -- zero decisions, on both sides.
    """
    sensing_ticks_n = len(_sensing_ticks(loaded.ticks))
    if sensing_ticks_n == 0:
        return Reconciliation("triggers_match_summary", "unavailable",
                               "no tick carries a sensing block", compared=0, compared_is="records_compared")
    computed_by_trigger, computed_rules_by_status = _tally_triggers(loaded.ticks)
    summary_by_trigger = sensing_summary.get("decisions_by_trigger")
    summary_rules_by_status = sensing_summary.get("rules_by_status")
    if computed_by_trigger == summary_by_trigger and computed_rules_by_status == summary_rules_by_status:
        return Reconciliation("triggers_match_summary", "held",
                               compared=sensing_ticks_n, compared_is="records_compared")
    return Reconciliation(
        "triggers_match_summary", "failed",
        f"decisions_by_trigger: computed {computed_by_trigger} vs summary {summary_by_trigger}; "
        f"rules_by_status: computed {computed_rules_by_status} vs summary {summary_rules_by_status}",
        compared=sensing_ticks_n, compared_is="records_compared",
    )


def _reconcile_here_vs_responses(loaded: "LoadedRecords", phone_summary: dict[str, Any]) -> Reconciliation:
    """C1: the real task-38 drive held here because `calls_total` was
    computed as `0` from a `here_calls` field present on zero ticks, against
    a `responses_received` of `0` -- two zeros that were never compared.
    `_here_calls_by_session` now reports `calls_total` as `None`, with a
    named reason, when no tick carried the field at all.
    """
    responses = (phone_summary.get("here") or {}).get("responses_received")
    here = _here_calls_by_session(loaded.ticks)
    calls_total = here["calls_total"]
    if responses is None or calls_total is None:
        reason = (
            "summary[phone][here] carries no responses_received" if responses is None
            else here["not_measured"]
        )
        return Reconciliation("here_calls_ge_responses_received", "unavailable", reason,
                               compared=0, compared_is="records_compared")
    if calls_total >= responses:
        return Reconciliation("here_calls_ge_responses_received", "held",
                               compared=here["ticks_with_here_calls"], compared_is="records_compared")
    return Reconciliation(
        "here_calls_ge_responses_received", "failed",
        f"here.calls_total={calls_total} < summary[phone][here].responses_received={responses}",
        compared=here["ticks_with_here_calls"], compared_is="records_compared",
    )


def _reconcile_reference_shape(loaded: "LoadedRecords") -> Reconciliation:
    """§6 rule 8, across every tick: `sensing.reference.absent` is `None` if
    and only if `achieved`, `dropped`, `here_calls` and `here_errors` are all
    non-null. Needs only tick records, never `summary.json` -- M1's fix,
    since `api_calls` branches on `absent` and a record that violates this
    shape would otherwise be silently miscounted rather than flagged.

    A log recorded before task 39 has `achieved`/`dropped` (pre-existing)
    but no `here_calls`/`here_errors` KEY at all -- not a value nulled by
    `reference_from`, a key that did not exist when the record was written.
    That is the same backward-compatibility shape `api_calls`/`here` already
    carve out (M1/C2): checked here the same way, by whether any tick's
    `reference` carries the key at all, rather than reported as a shape
    violation on every telemetry-present tick of every log this task
    predates.
    """
    sensing_ticks = _sensing_ticks(loaded.ticks)
    if not sensing_ticks:
        return Reconciliation(
            "reference_absent_iff_fields_null", "unavailable",
            "no tick carries a sensing block", compared=0, compared_is="records_compared",
        )
    if not any("here_calls" in (t["sensing"].get("reference") or {}) for t in sensing_ticks):
        return Reconciliation(
            "reference_absent_iff_fields_null", "unavailable",
            HERE_CALLS_PREDATES_TASK_39,
            compared=0, compared_is="records_compared",
        )
    bad = 0
    for t in sensing_ticks:
        ref = t["sensing"].get("reference") or {}
        present = ref.get("absent") is None
        fields_present = (
            ref.get("achieved") is not None and ref.get("dropped") is not None
            and ref.get("here_calls") is not None and ref.get("here_errors") is not None
        )
        if present != fields_present:
            bad += 1
    if bad == 0:
        return Reconciliation("reference_absent_iff_fields_null", "held",
                               compared=len(sensing_ticks), compared_is="records_compared")
    return Reconciliation(
        "reference_absent_iff_fields_null", "failed",
        f"{bad} of {len(sensing_ticks)} ticks have reference.absent disagreeing with whether "
        "achieved/dropped/here_calls/here_errors are all non-null",
        compared=len(sensing_ticks), compared_is="records_compared",
    )


def reconciliations(loaded: "LoadedRecords", summary: dict[str, Any]) -> list[Reconciliation]:
    """The nine cross-checks §6 rules 11-19 name, plus `reference_absent_iff_fields_null`
    (M1, §6 rule 8 restated as a reconciliation). The first nine all need
    `summary.json`; the five within `summary["failures"]` need that block
    specifically. A missing input reports `unavailable`, never `held` --
    an absent summary is not a passed check (mutation pin 9). The tenth needs
    only tick records and runs unconditionally.
    """
    out: list[Reconciliation] = []
    failures_summary = (summary or {}).get("failures")
    if not failures_summary:
        reason = "summary.json not written" if not summary else "summary.json carries no failures block"
        out += [
            Reconciliation("blind_ticks_matches_camera_source", "unavailable", reason),
            Reconciliation("source_status_fired_iff_total_positive", "unavailable", reason),
            Reconciliation("source_totals_partition", "unavailable", reason),
            Reconciliation("source_passes_readable_le_attempted", "unavailable", reason),
            Reconciliation("sources_count_is_30", "unavailable", reason),
        ]
    else:
        out += [
            _reconcile_blind_ticks(failures_summary),
            _reconcile_source_status(failures_summary),
            _reconcile_source_totals(failures_summary),
            _reconcile_passes(failures_summary),
            _reconcile_source_count(failures_summary),
        ]

    if not summary:
        reason = "summary.json not written"
        out += [
            Reconciliation("events_written_matches_open_and_close_records", "unavailable", reason),
            Reconciliation("thermal_samples_count", "unavailable", reason),
            Reconciliation("triggers_match_summary", "unavailable", reason),
            Reconciliation("here_calls_ge_responses_received", "unavailable", reason),
        ]
    else:
        out.append(
            _reconcile_events_written(loaded, failures_summary)
            if failures_summary else
            Reconciliation("events_written_matches_open_and_close_records", "unavailable",
                            "summary.json carries no failures block")
        )
        thermal_summary = summary.get("thermal")
        out.append(
            _reconcile_thermal_samples(loaded, thermal_summary) if thermal_summary else
            Reconciliation("thermal_samples_count", "unavailable", "summary.json carries no thermal block")
        )
        sensing_summary = summary.get("sensing")
        out.append(
            _reconcile_triggers(loaded, sensing_summary) if sensing_summary else
            Reconciliation("triggers_match_summary", "unavailable", "summary.json carries no sensing block")
        )
        phone_summary = summary.get("phone")
        out.append(
            _reconcile_here_vs_responses(loaded, phone_summary) if phone_summary else
            Reconciliation("here_calls_ge_responses_received", "unavailable",
                            "summary.json carries no phone block")
        )
    out.append(_reconcile_reference_shape(loaded))
    return out


#: D1's default: how many times the median inter-tick gap a gap has to be
#: before `_tick_coverage` treats it as an interruption rather than
#: ordinary jitter. Verified against three real drives -- see
#: `_tick_coverage`'s own docstring for the numbers.
TICK_COVERAGE_GAP_MULTIPLE = 3.0


def _tick_coverage(ticks: list[dict[str, Any]], *, gap_multiple: float = TICK_COVERAGE_GAP_MULTIPLE) -> dict[str, Any] | None:
    """A check on tick PRODUCTION itself, not on any one instrument (D1).

    Every axis in this module reports `answered of attempted` over the ticks
    that exist -- a ratio of that form is blind by construction to an event
    that stops ticks from being produced at all, because it destroys
    `attempted` and `answered` together. A 54.58 s link outage on a real
    drive cost about 270 ticks, and the only trace of it anywhere in
    `## Session summary` was `no_telemetry` on two axes plus ten held
    reconciliations -- nothing that says a stretch of the drive is simply
    missing.

    **The first shape tried for this was wrong, and measuring it is what
    caught it.** `expected = span_s * (actual_ticks / span_s)` is
    `actual_ticks` by algebra, on every drive, healthy or not -- verified
    against the real outage log: 1229 actual against 1229.0 "expected", a
    zero-width miss. That is the same defect this section has already
    removed twice (a `stale` count nothing could increment; a `basis_counts`
    literal): an aggregate that is constant by construction is not a
    measurement, and a mean tick rate computed from the same span it is then
    multiplied back into can only ever reproduce the tick count it started
    from.

    **What this reads instead is the GAP distribution.** `median_gap_s` is
    the median of every inter-tick wall-clock gap -- robust to one outlier,
    so it stands in for the drive's ordinary cadence even across a long
    stall. `missing_ticks` sums, over every gap more than `gap_multiple`
    times that median, how many ticks the ordinary cadence would have
    produced in that gap beyond the one tick bounding it; a uniformly paced
    drive scores zero regardless of its tick count or span, because nothing
    but an actual pause in production can move it. Verified against three
    real drives' tick records: a drive with an induced 54.58 s outage scores
    269 missing of 1229 read (largest gap 270x the median); a historical
    drive with its own, unrelated 85.22 s gap scores 421 of 1073; a drive
    with no induced fault scores 0 of 749 (largest gap 1.5x the median).
    """
    t_wall = sorted(t["t_wall"] for t in ticks if t.get("t_wall") is not None)
    if len(t_wall) < 3:
        return None
    gaps = [b - a for a, b in zip(t_wall, t_wall[1:])]
    span_s = t_wall[-1] - t_wall[0]
    sorted_gaps = sorted(gaps)
    mid = len(sorted_gaps) // 2
    median_gap_s = (
        sorted_gaps[mid] if len(sorted_gaps) % 2 else (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2.0
    )
    largest_gap_s = sorted_gaps[-1]
    next_largest_gap_s = sorted_gaps[-2]
    missing_ticks = 0
    if median_gap_s > 0:
        threshold = gap_multiple * median_gap_s
        for g in gaps:
            if g > threshold:
                missing_ticks += round(g / median_gap_s) - 1
    actual_ticks = len(t_wall)
    return {
        "actual_ticks": actual_ticks,
        "span_s": span_s,
        "median_gap_s": median_gap_s,
        "largest_gap_s": largest_gap_s,
        "next_largest_gap_s": next_largest_gap_s,
        "gap_multiple": gap_multiple,
        "missing_ticks": missing_ticks,
        "expected_ticks": actual_ticks + missing_ticks,
    }


def session_summary(
    loaded: "LoadedRecords", summary: dict[str, Any], log_health: dict[str, Any] | None,
    *, phone_log_supplied: bool, metadata_jsonl_present: bool = True,
    phone_offline_kinds: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The `## Session summary` section's data: seven axes, a tick-coverage
    check, ten reconciliations, which inputs were available, and the sensing
    detail behind three of the axes. Built before `analyze()` can even be
    attempted (D13) -- every axis indexes `loaded.ticks` directly, never
    `analyze`'s output.

    `metadata_jsonl_present` defaults to `True` for a caller that has already
    read the file successfully (every existing caller); `main()` is the one
    caller that can pass `False`, for the run directory that never wrote the
    file at all -- previously `inputs.metadata_jsonl` was a literal `True`
    that could never say otherwise.

    `phone_offline_kinds` (D5) is the set of `kind` words the phone's own
    session log recorded (`here.unconfigured` among them) -- known only
    after `analyze()` has joined that log, so `main()` passes it in on a
    second call for a drive with ticks, and it stays empty for the zero-tick
    branch, which never reaches that join.
    """
    ticks = loaded.ticks
    axes = [
        _AXIS_BUILDERS[name](ticks, summary) if name in ("thermal", "failures")
        else _AXIS_BUILDERS[name](ticks)
        for name in AXES
    ]
    return {
        "axes": [a.to_record() for a in axes],
        "tick_coverage": _tick_coverage(ticks),
        "reconciliations": [r.to_record() for r in reconciliations(loaded, summary)],
        "inputs": {
            "metadata_jsonl": metadata_jsonl_present,
            "summary_json": bool(summary),
            "log_health_json": log_health is not None,
            "phone_log": phone_log_supplied,
        },
        "sensing": sensing_result(ticks, summary, phone_offline_kinds=phone_offline_kinds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run directory containing metadata.jsonl")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--phone-log", type=Path, default=None,
        help="the phone's own session log (SessionLog output), to join return/render "
             "onto the eight Jetson-side stages every tick already carries",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()
    phone_log = args.phone_log.expanduser() if args.phone_log else None

    # Built before `analyze` can even be attempted (D13): every axis indexes
    # `loaded.ticks` directly, so a drive with no tick records still gets a
    # `## Session summary` rather than nothing but `analyze`'s SystemExit.
    metadata_path = run_dir / "metadata.jsonl"
    metadata_jsonl_present = metadata_path.exists()
    loaded = (
        load_records(metadata_path) if metadata_jsonl_present
        else LoadedRecords(ticks=[], scenario=None, timebase_estimates=[], unparseable=0,
                           thermal_samples=[], thermal_events=[])
    )
    summary = _read_summary(run_dir)
    log_health = _read_log_health(run_dir)
    if not loaded.ticks:
        session = session_summary(
            loaded, summary, log_health, phone_log_supplied=phone_log is not None,
            metadata_jsonl_present=metadata_jsonl_present,
        )
        report_md = render_session_summary_only(run_dir, session, loaded, log_health)
        (run_dir / "report.md").write_text(report_md)
        (run_dir / "report.json").write_text(json.dumps(
            {
                "run_dir": str(run_dir), "n_ticks": 0, "session_summary": session,
                "analysis": None,
                "analysis_absent": "no tick records; every metric block indexes ticks",
            },
            indent=2,
        ))
        print(report_md)
        print(f"[eval] wrote {run_dir / 'report.md'}, report.json (no tick records)")
        return 2

    result = analyze(run_dir, phone_log, loaded=loaded)
    # Built AFTER `analyze()` here (unlike the zero-tick branch above, which
    # cannot reach it): the phone's own session-log failure kinds are known
    # only once `analyze()` has joined that log, and `here.unconfigured`
    # among them is what lets `sensing_result` name a build with no HERE API
    # key as a cause independent of shadow mode (D5).
    phone_offline_kinds = frozenset(
        record.get("kind") for record in (result.get("failures") or {}).get("phone") or []
    )
    session = session_summary(
        loaded, summary, log_health, phone_log_supplied=phone_log is not None,
        metadata_jsonl_present=metadata_jsonl_present, phone_offline_kinds=phone_offline_kinds,
    )
    plots = [] if args.no_plots else render_plots(result, run_dir)
    report_md = render_markdown(result, plots, session)
    (run_dir / "report.md").write_text(report_md)
    json_result = {k: v for k, v in result.items() if not k.startswith("_")}
    json_result["session_summary"] = session
    (run_dir / "report.json").write_text(json.dumps(json_result, indent=2))

    print(report_md)
    print(f"[eval] wrote {run_dir / 'report.md'}, report.json"
          + (f", {len(plots)} plots" if plots else ""))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
