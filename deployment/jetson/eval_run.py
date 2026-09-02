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
from pathlib import Path
from typing import Any

JETSON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JETSON_DIR))

import numpy as np  # noqa: E402

from policy import sim_contract  # noqa: E402
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE  # noqa: E402
from sensors.time_sync import StageTiming  # noqa: E402
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
DROPOUT_RECOVERY_MARGIN_S = 2.5  # stale_after_s + one fix interval


def load_records(
    metadata_path: Path,
) -> tuple[list[dict], dict | None, list[dict], int, list[dict], list[dict]]:
    """Ticks, the scenario, the timebase_estimate lines, how many lines would
    not parse, the thermal_sample lines, and the thermal_event lines.

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
    unparseable = 0
    with open(metadata_path) as f:
        for line in f:
            try:
                record = json.loads(line)  # Python json accepts Infinity literals
            except ValueError:
                unparseable += 1
                continue
            if record.get("type") == "tick":
                ticks.append(record)
            elif record.get("type") == "scenario":
                scenario = record
            elif record.get("type") == "timebase_estimate":
                timebase_estimates.append(record)
            elif record.get("type") == "thermal_sample":
                thermal_samples.append(record)
            elif record.get("type") == "thermal_event":
                thermal_events.append(record)
    return ticks, scenario, timebase_estimates, unparseable, thermal_samples, thermal_events


def load_phone_log(phone_log_path: Path) -> tuple[list[dict], list[dict]]:
    """Inbound advisory lines and advisory_shown lines from a phone's session
    log, ignoring everything else the file holds.

    Every outbound line `SessionLog` writes is a bare frame header -- the
    canonical JSON that went on the wire, verbatim, with no wrapper -- so it
    carries no `dir` key at all. The two line shapes this join wants both do,
    which is what tells them apart from an outbound line and from each other
    without this reader having to know every line shape a later task adds.
    """
    inbound_advisories: list[dict] = []
    shown: list[dict] = []
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
    return inbound_advisories, shown


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

    An inbound advisory with no matching tick is counted rather than dropped
    silently: nothing about a mismatch here says which side is wrong, but a
    reader needs to know it happened.
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
    unmatched = 0
    for record in inbound_advisories:
        header = record.get("header", {})
        capture_ns = header.get("t_capture_mono_ns")
        tick = ticks_by_capture_ns.get(capture_ns)
        if tick is None:
            unmatched += 1
            continue
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
    return {
        "advisories_seen_by_the_phone": len(inbound_advisories),
        "matched": len(rows),
        "unmatched": unmatched,
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


def in_dropout_affected(elapsed_s: float, dropouts: list[tuple[float, float]]) -> bool:
    return any(a <= elapsed_s < b + DROPOUT_RECOVERY_MARGIN_S for a, b in dropouts)


def analyze(run_dir: Path, phone_log_path: Path | None = None) -> dict[str, Any]:
    """All metrics + gates as a JSON-able dict (report rendering is separate).

    `phone_log_path` is optional: a run with no phone behind it, or one whose
    phone log was not pulled off the handset, still analyses fully on the
    eight Jetson-side stages every tick already carries. When it is supplied,
    the two logs are joined into a ten-stage table that adds `return` and
    `render`, the two facts only the phone witnesses.
    """
    ticks, scenario, timebase_estimates, unparseable, thermal_samples, thermal_events = load_records(
        run_dir / "metadata.jsonl"
    )
    if not ticks:
        raise SystemExit(f"no tick records in {run_dir / 'metadata.jsonl'}")
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    # What the run said it produced against what survived to be read. The summary is
    # written by `write_summary` before `close()` flushes the log, so a run that lost
    # its buffer states its own tick count beside a file that is short of it -- and
    # the gap went unread, so the report certified a run whose records were missing
    # with the evidence sitting in the same directory. Counted from the records, never
    # by subtraction, and reported whether or not it is zero.
    expected_ticks = summary.get("ticks")
    shortfall = None
    if isinstance(expected_ticks, int) and expected_ticks > 0:
        shortfall = expected_ticks - len(ticks)
    integrity = {
        "ticks_read": len(ticks),
        "ticks_the_run_reported": expected_ticks,
        "missing_ticks": shortfall,
        "unparseable_lines": unparseable,
        # The one gate. A short or truncated log is not a run that passed.
        "log_complete": bool(unparseable == 0 and (shortfall is None or shortfall == 0)),
    }

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
        "leader_gap_m": pctl(finite_gaps),
        "leader_rel_speed_measured_fraction": (
            float(np.mean(rel_measured)) if rel_measured else 0.0
        ),
        "distance_method_counts": dict(method_counts),
        "unique_tracks": len(track_spans),
        "track_lifetime_s": pctl(lifetimes_s),
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
    applicable = [g["pass"] for g in gates.values() if g["pass"] is not None]
    overall = all(applicable) if applicable else False

    phone_join = (
        None if phone_log_path is None
        else join_phone_log(ticks, timebase_estimates, *load_phone_log(phone_log_path))
    )
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
        # is the failure this whole check exists to close.
        "overall_pass": overall and integrity["log_complete"],
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
            peak = (
                f"; {len(jetson.get('zones_seen') or [])} zones read, hottest at peak "
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
                    f" ({ev['passes_readable']} of {ev['passes_attempted']} passes fully readable)"
                    if ev.get("passes_attempted") else ""
                )
                lines.append(
                    f"- throttle events, {device}: NOT EVALUABLE -- missing {missing}{passes}; "
                    f"this drive says nothing about whether the {device} throttled"
                )
            elif status == RULE_FIRED:
                lines.append(f"- throttle events, {device}: fired -- {count} transitions")
            elif device == "jetson":
                lines.append(f"- throttle events, jetson: quiet -- cooling devices readable throughout, "
                             f"{count} transitions")
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


def render_markdown(result: dict[str, Any], plots: list[str]) -> str:
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
    if not integrity.get("log_complete", True):
        lines.append(
            f"| log complete | read {integrity.get('ticks_read')} of "
            f"{integrity.get('ticks_the_run_reported')} ticks, "
            f"{integrity.get('unparseable_lines')} unparseable lines | 0 missing | "
            f"**FAIL** |"
        )
    lines += [
        "",
        f"**Overall: {'PASS' if r['overall_pass'] else 'FAIL'}**",
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
    lines += [
        "",
        "## Perception",
        "",
        f"- ticks with >= 1 tracked vehicle: {p['ticks_with_vehicle_fraction']:.1%} "
        f"(mean {p['mean_vehicles_per_tick']:.2f}/tick)",
        f"- leader present: {p['leader_present_fraction']:.1%} of ticks; "
        f"gap p50 {p['leader_gap_m']['p50']:.1f} m (min-side mean {p['leader_gap_m']['mean']:.1f} m)",
        f"- leader relative speed measured (vs neutral fallback): "
        f"{p['leader_rel_speed_measured_fraction']:.1%} of leader ticks",
        f"- distance methods: {p['distance_method_counts']}",
        f"- {p['unique_tracks']} tracks, lifetime p50 {p['track_lifetime_s']['p50']:.1f} s "
        f"(p95 {p['track_lifetime_s']['p95']:.1f} s)",
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
            f"- return_ms measured on {len(return_ms)} of {join['matched']} matched rows"
            + (f" (p50 {sorted(return_ms)[len(return_ms) // 2]:.1f} ms)" if return_ms else ""),
            f"- render_ms measured on {len(render_ms)} of {join['matched']} matched rows"
            + (f" (p50 {sorted(render_ms)[len(render_ms) // 2]:.1f} ms)" if render_ms else ""),
        ]
    if plots:
        lines += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots]
    lines.append("")
    return "\n".join(lines)


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

    result = analyze(run_dir, args.phone_log.expanduser() if args.phone_log else None)
    plots = [] if args.no_plots else render_plots(result, run_dir)
    report_md = render_markdown(result, plots)
    (run_dir / "report.md").write_text(report_md)
    json_result = {k: v for k, v in result.items() if not k.startswith("_")}
    (run_dir / "report.json").write_text(json.dumps(json_result, indent=2))

    print(report_md)
    print(f"[eval] wrote {run_dir / 'report.md'}, report.json"
          + (f", {len(plots)} plots" if plots else ""))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
