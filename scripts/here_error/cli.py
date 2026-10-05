#!/usr/bin/env python3
"""Command line for the HERE travel-time error analysis.

    .venv/bin/python scripts/here_error/cli.py assemble
    .venv/bin/python scripts/here_error/cli.py camera --limit 50
    .venv/bin/python scripts/here_error/cli.py label-sample
    .venv/bin/python scripts/here_error/cli.py score-labels --labels labels.csv
    .venv/bin/python scripts/here_error/cli.py routing --dry-run
    .venv/bin/python scripts/here_error/cli.py report

Run from the repository root. `--final` refuses to start on a dirty working tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from here_error import camera, labels, matching, params, pipeline, report, routing  # noqa: E402
from here_error.provenance import make_provenance, write_sidecar  # noqa: E402

DEFAULT_DATA = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc_drive_data_20260908")
DEFAULT_OUT = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/here-error-out")


def _prov(args, inputs):
    return make_provenance(inputs, require_clean=args.final)


def _assemble(args) -> pipeline.Assembly:
    asm = pipeline.assemble(args.data_dir)
    if asm.gate_v1_failed:
        for r in asm.runs:
            print(f"gate V1 {r.run}: spread {r.offset.spread_s:.3f} s -> {'pass' if r.offset.passes_gate else 'FAIL'}")
        raise SystemExit(f"gate V1 failed for {', '.join(asm.gate_v1_failed)}: frames and HERE responses cannot be placed on the GPS trace")
    return asm


def _detections(out: Path, asm) -> dict:
    return {r.run: camera.load_detections(out / f"detections_{r.run}.jsonl") for r in asm.runs}


def cmd_assemble(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asm = _assemble(args)
    prov = _prov(args, asm.input_files())
    rows = report.pass_rows(report.ReportInputs(asm, {}, None, None, []))
    report.write_passes_csv(asm, rows, args.out_dir / "passes.csv")
    report.write_stretches_csv(asm, args.out_dir / "stretches.csv")
    for name in ("passes.csv", "stretches.csv"):
        write_sidecar(args.out_dir / name, prov)
    maps = args.out_dir / "v2_maps"
    n_maps = 0
    for p in asm.passes:
        if p.exclusion != "shorter_than_min":
            run = next(r for r in asm.runs if r.run == p.run)
            matching.plot_pass(p, asm.catalogues[p.run], run.phone.gps, maps / f"{p.pass_id}.png")
            n_maps += 1
    flagged, total, on_flagged = pipeline.length_flag_counts(asm)
    summary = {
        "runs": {r.run: {"fixes": len(r.phone.gps), "invalid_fixes": r.phone.n_gps_invalid,
                         "duplicate_fix_times": r.phone.n_gps_duplicate_time, "here_bodies": len(r.bodies),
                         "frames": len(r.frames), "clock_offset": asdict(r.offset) | {"spread_s": r.offset.spread_s}}
                 for r in asm.runs},
        "passes_found": len(asm.passes),
        "passes_at_least_min_length": sum(p.matched_m >= params.MIN_PASS_M for p in asm.passes),
        "passes_analysed": len(asm.analysed),
        "exclusions": pipeline.exclusion_counts(asm),
        "segment_length_flags": {"flagged_segments": flagged, "segments": total, "analysed_passes_on_flagged": on_flagged},
        "stretches": len(asm.stretch_result.stretches),
        "stretch_passes_dropped": asm.stretch_result.dropped_passes,
        "v4": asdict(asm.v4),
        "v2_maps": n_maps,
        "provenance": json.loads(json.dumps(asdict(prov), default=list)),
    }
    (args.out_dir / "assembly.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    for r in asm.runs:
        o = r.offset
        print(f"V1 {r.run}: offset median {o.median_s:.3f} s, 5th-95th spread {o.spread_s:.3f} s over {o.n} fixes -> pass")
    print(f"passes found {summary['passes_found']}; at least {params.MIN_PASS_M:.0f} m: {summary['passes_at_least_min_length']}; "
          f"analysed {summary['passes_analysed']}; exclusions {summary['exclusions']}")
    ex = summary["exclusions"]
    print(f"V3 failures {ex.get('v3_failed', 0)}; chainage reversals {ex.get('chainage_reversed', 0)}")
    print(f"segment-length flags: {flagged} of {total} segments; analysed passes on a flagged segment: {on_flagged}")
    print(f"stretches {summary['stretches']}; passes dropped from stretches {summary['stretch_passes_dropped']}; "
          f"V4 n={asm.v4.n} real={asm.v4.median_abs_real} shuffled={asm.v4.median_abs_shuffled}; V2 maps {n_maps}")
    return 0


def cmd_camera(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights or args.out_dir / "weights" / params.YOLO_WEIGHTS
    detector = camera.make_yolo_detector(weights, args.device)
    from here_error.provenance import sha256_file

    for run, phone_name in params.RUNS:
        if args.run not in ("all", run):
            continue
        rd = pipeline.load_run(args.data_dir, run, phone_name)
        if not rd.offset.passes_gate:
            raise SystemExit(f"gate V1 failed for {run}")
        out = args.out_dir / f"detections_{run}.jsonl"
        prov = _prov(args, [rd.phone_path, rd.run_dir / "video_index.jsonl", rd.run_dir / "video.avi", weights])
        st = camera.run_camera(run, rd.run_dir / "video.avi", rd.frames, detector, out, args.limit)
        write_sidecar(out, prov)
        stats = {"run": run, "device": args.device, "weights": str(weights), "weights_sha256": sha256_file(weights),
                 "new_frames": st.n_new, "skipped_frames": st.n_skipped, "seconds": st.seconds,
                 "seconds_per_new_frame": st.seconds / st.n_new if st.n_new else None}
        (args.out_dir / f"camera_{run}.json").write_text(json.dumps(stats, indent=2))
        dets = camera.load_detections(out)
        print(f"{run}: {st.n_new} new frames, {st.n_skipped} skipped, {st.seconds:.1f} s "
              f"({stats['seconds_per_new_frame'] or 0:.3f} s per frame); detections on file {len(dets)}, "
              f"vehicles in total {sum(d.n_vehicles for d in dets)}, frames with a leader {sum(d.leader for d in dets)}")
    return 0


def cmd_label_sample(args) -> int:
    frames, videos, inputs = [], {}, []
    for run, phone_name in params.RUNS:
        rd = pipeline.load_run(args.data_dir, run, phone_name)
        if not rd.offset.passes_gate:
            raise SystemExit(f"gate V1 failed for {run}")
        frames += rd.frames
        videos[run] = rd.run_dir / "video.avi"
        inputs += [rd.phone_path, rd.run_dir / "video_index.jsonl", rd.run_dir / "video.avi"]
    prov = _prov(args, inputs)
    sample = labels.sample_frames(frames)
    template = labels.export_sample(sample, videos, args.out_dir / "labels")
    write_sidecar(template, prov)
    print(f"wrote {len(sample)} images and {template}")
    return 0


def cmd_score_labels(args) -> int:
    asm = _assemble(args)
    rows = labels.read_labels(args.labels)
    dets = {f"{d.run}:{d.frame_id}": d for lst in _detections(args.out_dir, asm).values() for d in lst}
    scores = labels.score(rows, dets)
    out = args.out_dir / "label_scores.json"
    report.write_label_scores(scores, out)
    write_sidecar(out, _prov(args, [args.labels] + [args.out_dir / f"detections_{r.run}.jsonl" for r in asm.runs]))
    for s in scores:
        print(s)
    return 0


def cmd_routing(args) -> int:
    asm = _assemble(args)
    requests = pipeline.route_requests(asm)
    cache = args.out_dir / "routing_cache"
    if args.dry_run:
        plan = routing.plan_calls(requests, cache)
        todo = [c for c in plan if not c.cached]
        print(f"dry run: {len(requests)} requests ({sum(r.kind == 'pass' for r in requests)} passes, "
              f"{sum(r.kind == 'stretch' for r in requests)} stretches); {len(todo)} would be HERE calls, "
              f"{len(plan) - len(todo)} are cached; budget {params.HERE_CALL_BUDGET}; calls made: 0")
        for c in todo:
            print(f"  {c.request.request_id}: {c.query['origin']} -> {c.query['destination']} departing {c.query['departureTime']}")
        return 0 if len(todo) <= params.HERE_CALL_BUDGET else 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prov = _prov(args, asm.input_files())
    run = routing.run_routing(requests, cache)
    out = args.out_dir / "routing.csv"
    routing.write_csv(run.results, out)
    write_sidecar(out, prov)
    print(f"{len(run.results)} routes; {run.n_calls} HERE calls made; {run.n_cached} from cache")
    return 0


def cmd_report(args) -> int:
    asm = _assemble(args)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scores_path, routing_path = out / "label_scores.json", out / "routing.csv"
    dets = _detections(out, asm)
    inputs = report.ReportInputs(
        asm, dets,
        report.read_label_scores(scores_path) if scores_path.exists() else None,
        routing.read_csv(routing_path) if routing_path.exists() else None,
        pipeline.route_requests(asm),
    )
    extra = [p for p in [scores_path, routing_path] + [out / f"detections_{r.run}.jsonl" for r in asm.runs] if p.exists()]
    prov = _prov(args, asm.input_files() + extra)
    text, reasons = report.render(inputs, prov)
    provenance_block = "\n## Provenance\n\n```json\n" + json.dumps(asdict(prov) | {"inputs": f"{len(prov.inputs)} files, see sidecar"}, indent=2, default=list) + "\n```\n"
    (out / "summary.md").write_text(text + provenance_block)
    write_sidecar(out / "summary.md", prov)
    rows = report.pass_rows(inputs)
    report.write_passes_csv(asm, rows, out / "passes.csv")
    report.write_stretches_csv(asm, out / "stretches.csv")
    for name in ("passes.csv", "stretches.csv"):
        write_sidecar(out / name, prov)
    if reasons:
        print("STATUS: INCOMPLETE")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("STATUS: COMPLETE")
    print(text)
    return 3 if reasons else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--final", action="store_true", help="refuse to run on a dirty working tree")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("assemble").set_defaults(fn=cmd_assemble)
    c = sub.add_parser("camera")
    c.add_argument("--run", default="all", help="a run directory name, or all")
    c.add_argument("--limit", type=int, default=None, help="process only the first N frames of each run")
    c.add_argument("--weights", type=Path, default=None)
    c.add_argument("--device", default=None)
    c.set_defaults(fn=cmd_camera)
    sub.add_parser("label-sample").set_defaults(fn=cmd_label_sample)
    s = sub.add_parser("score-labels")
    s.add_argument("--labels", type=Path, required=True)
    s.set_defaults(fn=cmd_score_labels)
    r = sub.add_parser("routing")
    r.add_argument("--dry-run", action="store_true", help="list the calls and make none")
    r.set_defaults(fn=cmd_routing)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
