#!/usr/bin/env python3
"""Fit the camera error model from the re-processed drive frames (jam plan section 2.3).

    .venv/bin/python scripts/fit_camera_model.py
    .venv/bin/python scripts/fit_camera_model.py --frames-dir outputs/jam/smoke/frames \\
        --sensitivity-imgsz --out outputs/jam/smoke/camera_error_model.json   # the S8 smoke

Reads `<frames-dir>/<run>/detections_<imgsz>.jsonl` (written by
`process_drive_frames.py`), groups the drives by frame interval (the 0.20 s drives and the
0.67 s drives) and writes one table set per group: detection rate, distance noise (raw
ground-plane and EMA), closing-rate noise, confidences, the ground-plane to width-prior
ratio, first-detection distances and `R_max`, all in 10 m bins of EMA distance. The primary
tables (`method`, the SUMO camera model's) are on the width-prior distance, which uses only
`fx` and the class widths; the same tables on the ground-plane distance are recorded under
`alternative`, because its scale follows each drive's fitted camera height. The SUMO
camera model reads the 0.20 s table (`sumo_table`). The 1,280 px records give the same
tables as a sensitivity. Every input file's sha256 is recorded, and the inputs must share
one weights sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import camera_error as ce  # noqa: E402

SUMO_TABLE = "0.20"


def load_records(path: Path, run: str) -> tuple[dict, list[dict]]:
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    records = []
    for line in lines[1:]:
        record = json.loads(line)
        record["run"] = run
        records.append(record)
    return header, records


def raw_boxes(path: Path, run: str) -> dict[tuple, list]:
    """`(run, frame_id) -> boxes` of a drive's raw detector output."""
    out: dict[tuple, list] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "frame":
            out[(run, row["frame_id"])] = [b[:4] for b in row["boxes"]]
    return out


def tables_for(frames_dir: Path, imgsz: int, reference_imgsz: int | None = None) -> dict:
    """The tables of `imgsz`'s records per frame-interval group. With `reference_imgsz`,
    the detection rate is the cross-resolution share against that pass (Amendment 3)."""
    groups: dict[str, list[dict]] = {}
    references: dict[str, list[dict]] = {}
    raws: dict[str, dict[tuple, list]] = {}
    runs, inputs, weights = {}, [], set()
    for run_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        path = run_dir / f"detections_{imgsz}.jsonl"
        if not path.exists():
            continue
        header, records = load_records(path, run_dir.name)
        weights.add(header.get("weights_sha256"))
        times = [r["t"] for r in records if r["type"] == "frame"]
        group = ce.interval_group(times)
        groups.setdefault(group, []).extend(records)
        runs[run_dir.name] = {"group": group, "frames": len(times), "horizon": header.get("horizon"),
                              "detections": sum(1 for r in records if r["type"] == "det")}
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if reference_imgsz is not None:
            ref_path = run_dir / f"detections_{reference_imgsz}.jsonl"
            raw_path = run_dir / f"raw_{imgsz}.jsonl"
            if not ref_path.exists() or not raw_path.exists():
                raise SystemExit(f"the cross-resolution detection rate needs {ref_path.name} and "
                                 f"{raw_path.name} in {run_dir}")
            ref_header, ref_records = load_records(ref_path, run_dir.name)
            weights.add(ref_header.get("weights_sha256"))
            references.setdefault(group, []).extend(ref_records)
            raws.setdefault(group, {}).update(raw_boxes(raw_path, run_dir.name))
            runs[run_dir.name]["reference_detections"] = sum(
                1 for r in ref_records if r["type"] == "det")
            for extra in (ref_path, raw_path):
                inputs.append({"path": str(extra),
                               "sha256": hashlib.sha256(extra.read_bytes()).hexdigest()})
    if not inputs:
        raise SystemExit(f"no detections_{imgsz}.jsonl under {frames_dir}")
    if len(weights) != 1:
        raise SystemExit(f"the records ran different weights: {sorted(weights, key=str)}")

    def build(method):
        return {group: {**ce.error_tables(records, method, reference=references.get(group),
                                          raw_boxes=raws.get(group)),
                        "drives": sorted(r for r, v in runs.items() if v["group"] == group)}
                for group, records in sorted(groups.items())}

    alternatives = {m: {"tables": build(m)} for m in ce.METHODS if m != ce.PRIMARY_METHOD}
    return {"method": ce.PRIMARY_METHOD, "tables": build(ce.PRIMARY_METHOD),
            "alternative": alternatives, "runs": runs, "inputs": inputs,
            "reference_imgsz": reference_imgsz, "weights_sha256": weights.pop()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", default="outputs/jam/frames")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--reference-imgsz", type=int, default=1280,
                        help="the pass whose tracked vehicles the detection rate is measured "
                             "against (Amendment 3, R2-3)")
    parser.add_argument("--sensitivity-imgsz", type=int, nargs="*", default=[1280])
    parser.add_argument("--out", default="results/jam/camera_error_model.json")
    args = parser.parse_args(argv)

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    frames_dir = resolve(args.frames_dir)
    primary = tables_for(frames_dir, args.imgsz, reference_imgsz=args.reference_imgsz)
    if SUMO_TABLE not in primary["tables"]:
        raise SystemExit(f"no drive in {frames_dir} has 0.20 s frames; the SUMO table needs one")
    artefact = {
        "schema": "camera_error_model/1", "generated_by": "scripts/fit_camera_model.py",
        "generated_wall": time.time(), "bin_m": ce.BIN_M, "imgsz": args.imgsz,
        "sumo_table": SUMO_TABLE, "fx_px": ce.FX_PX,
        "method_note": ("primary tables on the width-prior EMA distance (fx and class widths "
                        "only); the ground-plane tables, whose scale follows each drive's "
                        "fitted camera height (0.40 to 0.89 m against the configured 1.25 m), "
                        "are recorded under 'alternative'"),
        "detection_note": (f"p(d) is the share of the {args.reference_imgsz} px tracker's "
                           f"vehicles with a {args.imgsz} px tracked box at IoU >= "
                           f"{ce.CROSS_IOU:g} in the same frame, binned by the "
                           f"{args.reference_imgsz} px distance (Amendment 3, R2-3); the "
                           "held-track rate and the IoU 0.1 and raw-box shares are under "
                           "detection_alternatives. The sensitivity tables use the held-track "
                           "rate, having no higher-resolution reference"),
        "distance_scale_note": ("fx was never recorded; the absolute scale is swept as a "
                                "multiplicative bias b in {0.8, 1.0, 1.25} (plan section 2.3)"),
        **primary,
        "sensitivity": {str(size): tables_for(frames_dir, size)
                        for size in args.sensitivity_imgsz},
    }
    # Amendment 3 (R2-4): the SUMO camera's persistent pair spread is this group's.
    artefact["pair_sigma_log"] = artefact["tables"][SUMO_TABLE]["persistent_error"]["relative_spread"]
    artefact["pair_sigma_note"] = (
        "sigma_log of the persistent per-(ego, leader) distance factor: the relative 1.4826 MAD "
        "of per-track median ground-plane / width-prior ratios on the 0.20 s drives; it holds "
        "both estimators' per-vehicle errors, so it is an upper bound on the width prior's")
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artefact, indent=1))
    table = artefact["tables"][SUMO_TABLE]
    print(f"wrote {out}: groups {sorted(artefact['tables'])}, SUMO table R_max "
          f"{table['R_max_m']} m from {sum(table['detection']['opportunities'])} opportunities, "
          f"pair sigma_log {artefact['pair_sigma_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
