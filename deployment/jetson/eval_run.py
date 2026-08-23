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


def load_records(metadata_path: Path) -> tuple[list[dict], dict | None]:
    ticks: list[dict] = []
    scenario: dict | None = None
    with open(metadata_path) as f:
        for line in f:
            try:
                record = json.loads(line)  # Python json accepts Infinity literals
            except ValueError:
                continue
            if record.get("type") == "tick":
                ticks.append(record)
            elif record.get("type") == "scenario":
                scenario = record
    return ticks, scenario


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
    return (
        f"| {name} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |"
    )


def in_dropout_affected(elapsed_s: float, dropouts: list[tuple[float, float]]) -> bool:
    return any(a <= elapsed_s < b + DROPOUT_RECOVERY_MARGIN_S for a, b in dropouts)


def analyze(run_dir: Path) -> dict[str, Any]:
    """All metrics + gates as a JSON-able dict (report rendering is separate)."""
    ticks, scenario = load_records(run_dir / "metadata.jsonl")
    if not ticks:
        raise SystemExit(f"no tick records in {run_dir / 'metadata.jsonl'}")
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

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
    observation = {
        "missingness": pctl(missingness),
        "top_fallback_fields": {
            k: round(c / len(ticks), 3) for k, c in fallback_counter.most_common(8)
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
        "latency_ms": latency,
        "perception": perception,
        "observation": observation,
        "gps": gps_metrics,
        "advisory": advisory,
        "gates": gates,
        "overall_pass": overall,
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
    lines += [
        "",
        f"**Overall: {'PASS' if r['overall_pass'] else 'FAIL'}**",
        "",
        "## Latency (full run)",
        "",
        "| stage | mean | p50 | p95 | max |",
        "|---|---|---|---|---|",
    ]
    order = ["jetson_ms", "link_ms", "e2e_ms", "capture_to_start_ms", "detect_ms",
             "track_distance_ms",
             "observe_ms", "policy_advisory_ms"]
    for key in order:
        # None means the series was absent -- a local run has no link segment --
        # and an absent series must not render as zeros.
        if r["latency_ms"].get(key) is not None:
            lines.append(fmt_row(key.removesuffix("_ms"), r["latency_ms"][key]))
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
        f"- encoder-field missingness: mean {r['observation']['missingness']['mean']:.1%}",
        f"- most frequent fallback fields (fraction of ticks): "
        f"{r['observation']['top_fallback_fields']}",
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
    if plots:
        lines += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run directory containing metadata.jsonl")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()

    result = analyze(run_dir)
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
