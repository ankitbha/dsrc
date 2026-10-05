"""Stretches: sequences of consecutive passes covering 2 to 5 km."""
from __future__ import annotations

from dataclasses import dataclass

from here_error import params
from here_error.here_times import AnalysedPass
from here_error.models import Stretch


@dataclass(frozen=True)
class StretchResult:
    stretches: tuple[Stretch, ...]
    dropped_passes: int
    dropped_m: float


def _stretch(run: str, n: int, rows: list[AnalysedPass]) -> Stretch:
    return Stretch(
        stretch_id=f"{run.rsplit('_', 1)[-1]}-S{n:02d}",
        run=run,
        pass_ids=tuple(r.p.pass_id for r in rows),
        start_utc_s=rows[0].p.first_utc_s,
        end_utc_s=rows[-1].p.last_utc_s,
        matched_m=sum(r.p.matched_m for r in rows),
        observed_s=sum(r.t.observed_s for r in rows),
        here_s=sum(r.t.here_s for r in rows),
        free_flow_s=sum(r.t.free_flow_s for r in rows),
        stopped_s=sum(r.t.stopped_s for r in rows),
    )


def build_stretches(analysed: list[AnalysedPass]) -> StretchResult:
    """Walk the analysed passes of each run in time order.

    A pass joins the open stretch when the gap from the previous pass's last fix is at most
    STRETCH_MAX_GAP_S. A stretch closes when its matched length reaches STRETCH_MIN_M, or
    before a pass that would take it over STRETCH_MAX_M. An open stretch that ends shorter than
    STRETCH_MIN_M is dropped and counted. Times are sums over the passes, so the unmatched gaps
    between passes are excluded from both sides.
    """
    stretches: list[Stretch] = []
    dropped_n, dropped_m = 0, 0.0
    runs = sorted({a.p.run for a in analysed})
    for run in runs:
        rows = sorted((a for a in analysed if a.p.run == run), key=lambda a: a.p.first_utc_s)
        cur: list[AnalysedPass] = []

        def drop():
            nonlocal cur, dropped_n, dropped_m
            if cur:
                dropped_n += len(cur)
                dropped_m += sum(r.p.matched_m for r in cur)
            cur = []

        for a in rows:
            if cur and a.p.first_utc_s - cur[-1].p.last_utc_s > params.STRETCH_MAX_GAP_S:
                drop()
            cur_m = sum(r.p.matched_m for r in cur)
            if cur and cur_m + a.p.matched_m > params.STRETCH_MAX_M:
                drop()
            if a.p.matched_m > params.STRETCH_MAX_M:
                dropped_n += 1
                dropped_m += a.p.matched_m
                continue
            cur.append(a)
            if sum(r.p.matched_m for r in cur) >= params.STRETCH_MIN_M:
                stretches.append(_stretch(run, len([s for s in stretches if s.run == run]) + 1, cur))
                cur = []
        drop()
    return StretchResult(tuple(stretches), dropped_n, dropped_m)
