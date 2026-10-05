"""Gate V6: a hand-labelled sample of frames, and the detector's score on it."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from here_error import params
from here_error.camera import upright
from here_error.clock import is_after_dusk
from here_error.models import Frame, FrameDetections

COLUMNS = ("frame_key", "n_vehicles_moving", "n_vehicles_parked", "leader_present")


class LabelError(ValueError):
    pass


def sample_frames(frames: Sequence[Frame]) -> list[Frame]:
    """LABEL_SAMPLE_PER_HALF frames before the dusk split and as many after, drawn with a fixed seed."""
    rng = np.random.default_rng(params.SEED)
    chosen: list[Frame] = []
    for dark in (False, True):
        pool = sorted((f for f in frames if is_after_dusk(f.capture_utc_s) == dark), key=lambda f: (f.run, f.pos))
        if len(pool) < params.LABEL_SAMPLE_PER_HALF:
            raise LabelError(f"only {len(pool)} frames {'after' if dark else 'before'} the dusk split; "
                             f"need {params.LABEL_SAMPLE_PER_HALF}")
        idx = rng.choice(len(pool), size=params.LABEL_SAMPLE_PER_HALF, replace=False)
        chosen.extend(pool[i] for i in sorted(idx))
    return chosen


def image_name(f: Frame) -> str:
    return f"{f.run}__{f.frame_id}.jpg"


def export_sample(sample: Sequence[Frame], videos: dict[str, Path], out_dir: Path) -> Path:
    """Write the sample's upright JPEGs and a blank CSV template; returns the template path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for run, video in videos.items():
        want = {f.pos: f for f in sample if f.run == run}
        if not want:
            continue
        cap = cv2.VideoCapture(str(video))
        try:
            for pos in range(max(want) + 1):
                ok, raw = cap.read()
                if not ok:
                    raise LabelError(f"{video}: could not decode frame {pos}")
                if pos in want:
                    cv2.imwrite(str(out_dir / image_name(want[pos])), upright(raw), [cv2.IMWRITE_JPEG_QUALITY, 90])
        finally:
            cap.release()
    template = out_dir / "labels_template.csv"
    with template.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for f in sample:
            w.writerow([f.key, "", "", ""])
    return template


@dataclass(frozen=True)
class LabelRow:
    frame_key: str
    n_moving: int
    n_parked: int
    leader_present: bool


def _parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in ("1", "true", "yes", "y"):
        return True
    if t in ("0", "false", "no", "n"):
        return False
    raise LabelError(f"cannot read leader_present value {text!r}")


def read_labels(path: Path | str) -> list[LabelRow]:
    """Refuses a blank cell, a repeated frame or a header that is not the template's."""
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise LabelError(f"label file columns must be {COLUMNS}")
        rows, seen = [], set()
        for r in reader:
            if any((r[c] or "").strip() == "" for c in COLUMNS):
                raise LabelError(f"blank label cell for {r['frame_key']!r}")
            if r["frame_key"] in seen:
                raise LabelError(f"frame {r['frame_key']!r} labelled twice")
            seen.add(r["frame_key"])
            rows.append(LabelRow(r["frame_key"], int(r["n_vehicles_moving"]), int(r["n_vehicles_parked"]),
                                 _parse_bool(r["leader_present"])))
    return rows


@dataclass(frozen=True)
class HalfScore:
    half: str
    n_frames: int
    precision_all: float | None
    recall_all: float | None
    precision_moving: float | None
    recall_moving: float | None
    leader_accuracy: float | None
    parked_share: float | None


def _pr(det: list[int], lab: list[int]) -> tuple[float | None, float | None]:
    hit = sum(min(d, l) for d, l in zip(det, lab))
    sd, sl = sum(det), sum(lab)
    return (hit / sd if sd else None), (hit / sl if sl else None)


def score(rows: Sequence[LabelRow], detections: dict[str, FrameDetections]) -> list[HalfScore]:
    """Count precision and recall per half, with parked vehicles counted once as vehicles and once left out.

    Per frame, matched = min(detected, labelled); precision is the sum of matched over the sum
    detected, recall the sum of matched over the sum labelled.
    """
    missing = [r.frame_key for r in rows if r.frame_key not in detections]
    if missing:
        raise LabelError(f"{len(missing)} labelled frames have no detections, for example {missing[0]!r}")
    out = []
    for name, dark in (("before_dusk", False), ("after_dusk", True)):
        sel = [r for r in rows if is_after_dusk(detections[r.frame_key].utc_s) == dark]
        det = [detections[r.frame_key].n_vehicles for r in sel]
        pa, ra = _pr(det, [r.n_moving + r.n_parked for r in sel])
        pm, rm = _pr(det, [r.n_moving for r in sel])
        total = sum(r.n_moving + r.n_parked for r in sel)
        out.append(HalfScore(
            half=name, n_frames=len(sel),
            precision_all=pa, recall_all=ra, precision_moving=pm, recall_moving=rm,
            leader_accuracy=(sum(detections[r.frame_key].leader == r.leader_present for r in sel) / len(sel)) if sel else None,
            parked_share=(sum(r.n_parked for r in sel) / total) if total else None,
        ))
    return out
