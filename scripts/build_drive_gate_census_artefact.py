#!/usr/bin/env python3
"""Assemble the committed artefact for the rotation-corrected drive gate census.

Reads the replay directories `replay_drive_safety_census.py` wrote, runs
`score_safety.py` over them, and records the result beside the two things that
bound it: the unrotated control, and the share of leader gaps sitting on the
distance estimator's width-prior saturation value.

    python3 scripts/build_drive_gate_census_artefact.py \
        --rotated <dir-of-run-dirs> --control <dir-of-run-dirs> \
        --replay-log <log> --out results/safety/drive_replay_gate_census
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JETSON = REPO / "deployment" / "jetson"
PY = str(REPO / ".venv" / "bin" / "python")

#: 800 px focal length, a 1.8 m car, a 720 px rotated frame: the value
#: `DistanceEstimator`'s width prior returns when a box spans the whole frame.
#: A leader gap here is a saturated estimate, not a measured two metres.
WIDTH_PRIOR_SATURATION_M = 2.00


def run_census(run_dirs: list[Path]) -> tuple[dict, str]:
    proc = subprocess.run(
        [PY, str(JETSON / "score_safety.py"), *[str(d) for d in run_dirs]],
        capture_output=True, text=True, cwd=str(JETSON),
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return payload, proc.stderr


def gap_stats(run_dirs: list[Path]) -> dict:
    gaps: list[float] = []
    speeds: list[float] = []
    ticks = 0
    for d in run_dirs:
        path = d / "metadata.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ticks += 1
            obs = rec.get("obs") or {}
            g = obs.get("leader_gap")
            if isinstance(g, (int, float)) and math.isfinite(g):
                gaps.append(float(g))
            s = obs.get("ego_speed")
            if isinstance(s, (int, float)) and math.isfinite(s):
                speeds.append(float(s))
    gaps.sort()
    saturated = [g for g in gaps if abs(g - WIDTH_PRIOR_SATURATION_M) < 0.01]
    unsaturated = [g for g in gaps if abs(g - WIDTH_PRIOR_SATURATION_M) >= 0.01]

    def pct(xs, q):
        return None if not xs else xs[min(len(xs) - 1, int(q * len(xs)))]

    return {
        "ticks": ticks,
        "ticks_with_a_finite_leader_gap": len(gaps),
        "leader_gap_median_m": pct(gaps, 0.5),
        "leader_gap_p75_m": pct(gaps, 0.75),
        "leader_gap_max_m": gaps[-1] if gaps else None,
        "ego_speed_median_mps": pct(sorted(speeds), 0.5),
        "saturated_at_width_prior": {
            "value_m": WIDTH_PRIOR_SATURATION_M,
            "ticks": len(saturated),
            "share_of_leader_ticks": (len(saturated) / len(gaps)) if gaps else None,
            "why": "800 px focal length times a 1.8 m car width over a 720 px frame is "
                   "exactly 2.00 m, so a detection box spanning the full rotated frame "
                   "produces this value through the width-prior fallback. These are not "
                   "vehicles two metres ahead.",
        },
        "leader_gap_median_excluding_saturated_m": pct(unsaturated, 0.5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotated", required=True, type=Path)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--replay-log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rot_dirs = sorted(d for d in args.rotated.expanduser().iterdir() if d.is_dir())
    ctl_dirs = sorted(d for d in args.control.expanduser().iterdir() if d.is_dir())

    rot_json, rot_report = run_census(rot_dirs)
    ctl_json, ctl_report = run_census(ctl_dirs)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=str(REPO)).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True,
                                cwd=str(REPO)).stdout.strip())

    artefact = {
        "what": "The safety gate's per-rule evaluability census over the 2026-09-08 road "
                "drives, with the ninety-degree frame rotation corrected offline. The "
                "drives themselves recorded the rotation defect, so every rule read "
                "not_evaluable on every tick; this rebuilds the observation from the same "
                "stored frames with the fix applied.",
        "generator": "scripts/replay_drive_safety_census.py, then "
                     "deployment/jetson/score_safety.py, assembled by "
                     "scripts/build_drive_gate_census_artefact.py",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "tracked_files_dirty": dirty,
        "environment": {"machine": platform.platform(), "python": platform.python_version()},
        "substitutions_that_limit_the_claim": [
            "The detector is ultralytics YOLOv8n in FP32 on the CPU, not the device's "
            "YOLOv8n TensorRT FP16 engine. Neither TensorRT nor CUDA nor any weights file "
            "exists off-device. Same architecture and the same confidence, IoU and "
            "vehicle-class settings read from the deployed config; a different "
            "implementation and a different precision.",
            "The action is the one each drive recorded, not one re-inferred from the "
            "rebuilt observation. The actor bundle is absent off-device, and the "
            "perception half is the half the defect broke.",
            "Timing is the replay's. Frames are paired to ticks by index, while the live "
            "loop is latest-value-wins and drops frames under load.",
            "Distance rests on intrinsics measured over forty frames of one run, which "
            "nothing in the corpus independently validates.",
            "No `safety` block exists in these logs, so score_safety labels every number "
            "COUNTERFACTUAL: this is what the gate could have evaluated, not what it did.",
        ],
        "rotated": {
            "runs": [d.name for d in rot_dirs],
            "census": rot_json,
            "report": rot_report,
            "gaps": gap_stats(rot_dirs),
        },
        "control_no_rotation": {
            "why": "Same frames, same code, same thresholds, rotation removed. It has to "
                   "reproduce the drive's own recorded zero, or the rotated count is an "
                   "assertion rather than a measurement.",
            "runs": [d.name for d in ctl_dirs],
            "census": ctl_json,
            "report": ctl_report,
            "gaps": gap_stats(ctl_dirs),
        },
        "replay_log": args.replay_log.expanduser().read_text(errors="replace")
        if args.replay_log.expanduser().is_file() else None,
    }

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(artefact, indent=1, default=str) + "\n")
    out.with_suffix(".log").write_text(
        "=== ROTATED (the fix applied offline) ===\n" + rot_report +
        "\n=== CONTROL (rotation removed) ===\n" + ctl_report + "\n"
    )
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
