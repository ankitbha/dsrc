"""Stages 0 to 2 end to end: load both runs, check the clock, match, time, and build stretches."""
from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

from here_error import params
from here_error.clock import ClockOffset, estimate_offset
from here_error.here_times import (
    AnalysedPass,
    V4Result,
    compute_pass_times,
    index_readings,
    v4_summary,
)
from here_error.load import PhoneLog, load_frames, load_here_bodies, load_phone_log
from here_error.matching import CatalogueEntry, extract_passes
from here_error.models import Frame, HereBody, Pass, PassTimes
from here_error.stretches import StretchResult, build_stretches


@dataclass(frozen=True)
class RunData:
    run: str
    run_dir: Path
    phone_path: Path
    phone: PhoneLog
    offset: ClockOffset
    bodies: tuple[HereBody, ...]
    frames: tuple[Frame, ...]


@dataclass(frozen=True, eq=False)
class Assembly:
    runs: tuple[RunData, ...]
    passes: tuple[Pass, ...]
    times: tuple[PassTimes, ...]
    catalogues: dict[str, dict[str, CatalogueEntry]]
    stretch_result: StretchResult
    v4: V4Result
    gate_v1_failed: tuple[str, ...]

    @property
    def analysed(self) -> list[AnalysedPass]:
        by_id = {t.pass_id: t for t in self.times}
        return [AnalysedPass(p, by_id[p.pass_id]) for p in self.passes if by_id[p.pass_id].exclusion is None]

    def input_files(self) -> list[Path]:
        files = []
        for r in self.runs:
            files += [r.phone_path, r.run_dir / "here_index.jsonl", r.run_dir / "video_index.jsonl"]
            files += [r.run_dir / b.source_file for b in r.bodies]
        return files


def load_run(data_dir: Path, run: str, phone_name: str) -> RunData:
    run_dir = data_dir / run
    phone_path = data_dir / params.PHONE_SUBDIR / phone_name
    phone = load_phone_log(phone_path)
    offset = estimate_offset([f.wall_s for f in phone.gps], [f.utc_s for f in phone.gps])
    if not offset.passes_gate:
        return RunData(run, run_dir, phone_path, phone, offset, (), ())
    return RunData(run, run_dir, phone_path, phone, offset,
                   load_here_bodies(run, run_dir, phone, offset), load_frames(run, run_dir, phone, offset))


def assemble(data_dir: Path) -> Assembly:
    """Gate V1 stops the stages after it: a run that fails it contributes no passes and is named."""
    runs = tuple(load_run(data_dir, run, phone) for run, phone in params.RUNS)
    failed = tuple(r.run for r in runs if not r.offset.passes_gate)
    if failed:
        return Assembly(runs, (), (), {}, StretchResult((), 0, 0.0), V4Result(0, None, None), failed)
    all_readings = index_readings([b for r in runs for b in r.bodies])
    passes: list[Pass] = []
    times: list[PassTimes] = []
    catalogues = {}
    for r in runs:
        run_passes, cat = extract_passes(r.run, r.phone.gps, r.bodies)
        catalogues[r.run] = cat
        readings = index_readings(r.bodies)
        for p in run_passes:
            passes.append(p)
            times.append(compute_pass_times(p, cat[p.segment_key].shape_length_m, readings, all_readings))
    asm = Assembly(runs, tuple(passes), tuple(times), catalogues, StretchResult((), 0, 0.0),
                   v4_summary(times), ())
    stretches = build_stretches(asm.analysed)
    return Assembly(asm.runs, asm.passes, asm.times, asm.catalogues, stretches, asm.v4, ())


def exclusion_counts(asm: Assembly) -> dict[str, int]:
    return dict(collections.Counter(t.exclusion for t in asm.times if t.exclusion is not None))


def length_flag_counts(asm: Assembly) -> tuple[int, int, int]:
    """Segments in the catalogues flagged for a stated length that differs from the shape, and the passes on them.

    Returns (flagged segments, total segments, analysed passes on a flagged segment).
    """
    flagged = sum(e.length_flag for c in asm.catalogues.values() for e in c.values())
    total = sum(len(c) for c in asm.catalogues.values())
    on_flagged = sum(1 for a in asm.analysed if a.p.segment_length_flag)
    return flagged, total, on_flagged
