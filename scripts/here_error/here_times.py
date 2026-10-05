"""Stage 2: HERE time, free-flow time and signed error for each pass; control V4."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from here_error import params
from here_error.models import HereBody, HereSegment, Pass, PassTimes


@dataclass(frozen=True)
class Reading:
    """One segment as reported in one HERE body."""

    run: str
    seq: int
    data_time_s: float
    response_utc_s: float
    segment: HereSegment


@dataclass(frozen=True)
class AnalysedPass:
    p: Pass
    t: PassTimes


def index_readings(bodies) -> dict[str, list[Reading]]:
    """Readings by segment key, each list ordered by (data time, response time)."""
    out: dict[str, list[Reading]] = {}
    for b in bodies:
        for s in b.segments:
            out.setdefault(s.key, []).append(Reading(b.run, b.seq, b.data_time_s, b.response_utc_s, s))
    for lst in out.values():
        lst.sort(key=lambda r: (r.data_time_s, r.response_utc_s))
    return out


@dataclass(frozen=True)
class _Piece:
    start_m: float
    end_m: float
    speed_mps: float | None
    free_flow_mps: float | None
    jam_factor: float | None


def pieces(seg: HereSegment, shape_length_m: float) -> list[_Piece]:
    """Sub-segments laid along the shape by cumulative length, scaled to the shape length.

    Where HERE gives no sub-segments the whole segment is one piece at the segment speed.
    A sub-segment missing a field takes the segment's value for it.
    """
    if not seg.subsegments:
        return [_Piece(0.0, shape_length_m, seg.speed_mps, seg.free_flow_mps, seg.jam_factor)]
    total = sum(s.length_m for s in seg.subsegments)
    scale = shape_length_m / total
    out, pos = [], 0.0
    for s in seg.subsegments:
        end = pos + s.length_m * scale
        out.append(_Piece(
            pos, end,
            s.speed_mps if s.speed_mps is not None else seg.speed_mps,
            s.free_flow_mps if s.free_flow_mps is not None else seg.free_flow_mps,
            s.jam_factor if s.jam_factor is not None else seg.jam_factor,
        ))
        pos = end
    return out


def _overlap(piece: _Piece, c0: float, c1: float) -> float:
    return max(0.0, min(piece.end_m, c1) - max(piece.start_m, c0))


def travel_time_s(seg: HereSegment, shape_length_m: float, c0: float, c1: float, *, free_flow: bool) -> float | None:
    """Sum over pieces of (length of [c0, c1] inside the piece) / (the piece's speed).

    None when a piece that the portion crosses has no positive speed.
    """
    total = 0.0
    for pc in pieces(seg, shape_length_m):
        ov = _overlap(pc, c0, c1)
        if ov <= 0:
            continue
        v = pc.free_flow_mps if free_flow else pc.speed_mps
        if v is None or v <= 0:
            return None
        total += ov / v
    return total


def matched_jam_factor(seg: HereSegment, shape_length_m: float, c0: float, c1: float) -> float | None:
    num = den = 0.0
    for pc in pieces(seg, shape_length_m):
        ov = _overlap(pc, c0, c1)
        if ov > 0 and pc.jam_factor is not None:
            num += ov * pc.jam_factor
            den += ov
    return num / den if den > 0 else None


def signed_error(here_s: float, observed_s: float) -> float:
    """(HERE time - observed time) / observed time; positive when HERE's time is the longer."""
    return (here_s - observed_s) / observed_s


def stopped_seconds(p: Pass) -> float:
    """Seconds between consecutive fixes whose earlier fix has GPS speed below STOP_MPS."""
    total = 0.0
    for a, b in zip(p.fixes, p.fixes[1:]):
        if a.speed_mps is not None and a.speed_mps < params.STOP_MPS:
            total += b.utc_s - a.utc_s
    return total


def select_primary(readings: list[Reading], first_utc_s: float) -> tuple[Reading | None, str | None]:
    """Latest reading with data time at or before the pass's first fix, no older than MAX_READING_AGE_S."""
    before = [r for r in readings if r.data_time_s <= first_utc_s]
    if not before:
        return None, "no_reading"
    r = before[-1]
    if first_utc_s - r.data_time_s > params.MAX_READING_AGE_S:
        return None, "reading_too_old"
    return r, None


def select_sensitivity(readings: list[Reading], mid_utc_s: float) -> Reading | None:
    """The reading whose data time is nearest the pass midpoint."""
    if not readings:
        return None
    return min(readings, key=lambda r: abs(r.data_time_s - mid_utc_s))


def select_shuffled(readings: list[Reading], first_utc_s: float, last_utc_s: float) -> Reading | None:
    """Control V4: the reading nearest in time among those at least SHUFFLE_MIN_S from the pass."""
    best, best_d = None, None
    for r in readings:
        if r.data_time_s <= first_utc_s:
            d = first_utc_s - r.data_time_s
        elif r.data_time_s >= last_utc_s:
            d = r.data_time_s - last_utc_s
        else:
            d = 0.0
        if d >= params.SHUFFLE_MIN_S and (best_d is None or d < best_d):
            best, best_d = r, d
    return best


def _error_with(reading: Reading | None, shape_len: float, p: Pass) -> float | None:
    if reading is None:
        return None
    h = travel_time_s(reading.segment, shape_len, p.chain_start_m, p.chain_end_m, free_flow=False)
    return None if h is None else signed_error(h, p.observed_s)


def compute_pass_times(
    p: Pass,
    shape_length_m: float,
    readings: dict[str, list[Reading]],
    all_readings: dict[str, list[Reading]],
) -> PassTimes:
    """`readings` are this run's; `all_readings` pool both runs and are used only for control V4."""
    stopped = stopped_seconds(p)
    moving = p.observed_s - stopped

    def blank(reason, **kw):
        base = dict(
            pass_id=p.pass_id, reading_seq=None, reading_data_time_s=None, reading_age_s=None,
            reading_arrived_before_start=None, observed_s=p.observed_s, here_s=None, free_flow_s=None,
            signed_error=None, free_flow_error=None, jam_factor=None, confidence=None,
            stopped_s=stopped, moving_s=moving, sensitivity_seq=None, sensitivity_signed_error=None,
            shuffled_seq=None, shuffled_signed_error=None, exclusion=reason,
        )
        base.update(kw)
        return PassTimes(**base)

    if p.exclusion is not None:
        return blank(p.exclusion)
    if p.observed_s <= 0:
        return blank("zero_observed_time")
    cands = readings.get(p.segment_key, [])
    primary, why = select_primary(cands, p.first_utc_s)
    if primary is None:
        return blank(why)
    seg = primary.segment
    here_s = travel_time_s(seg, shape_length_m, p.chain_start_m, p.chain_end_m, free_flow=False)
    if here_s is None:
        return blank("reading_without_speed", reading_seq=primary.seq, reading_data_time_s=primary.data_time_s)
    ff_s = travel_time_s(seg, shape_length_m, p.chain_start_m, p.chain_end_m, free_flow=True)
    if ff_s is None:
        return blank("reading_without_free_flow", reading_seq=primary.seq, reading_data_time_s=primary.data_time_s)
    mid = (p.first_utc_s + p.last_utc_s) / 2.0
    sens = select_sensitivity(cands, mid)
    shuf = select_shuffled(all_readings.get(p.segment_key, []), p.first_utc_s, p.last_utc_s)
    return PassTimes(
        pass_id=p.pass_id,
        reading_seq=primary.seq,
        reading_data_time_s=primary.data_time_s,
        reading_age_s=p.first_utc_s - primary.data_time_s,
        reading_arrived_before_start=primary.response_utc_s <= p.first_utc_s,
        observed_s=p.observed_s,
        here_s=here_s,
        free_flow_s=ff_s,
        signed_error=signed_error(here_s, p.observed_s),
        free_flow_error=signed_error(ff_s, p.observed_s),
        jam_factor=matched_jam_factor(seg, shape_length_m, p.chain_start_m, p.chain_end_m),
        confidence=seg.confidence,
        stopped_s=stopped,
        moving_s=moving,
        sensitivity_seq=None if sens is None else sens.seq,
        sensitivity_signed_error=_error_with(sens, shape_length_m, p),
        shuffled_seq=None if shuf is None else shuf.seq,
        shuffled_signed_error=_error_with(shuf, shape_length_m, p),
        exclusion=None,
    )


@dataclass(frozen=True)
class V4Result:
    """Control V4: median absolute signed error with the real reading and with a shuffled one."""

    n: int
    median_abs_real: float | None
    median_abs_shuffled: float | None


def v4_summary(times) -> V4Result:
    rows = [t for t in times if t.exclusion is None and t.shuffled_signed_error is not None]
    if not rows:
        return V4Result(0, None, None)
    return V4Result(
        len(rows),
        float(np.median([abs(t.signed_error) for t in rows])),
        float(np.median([abs(t.shuffled_signed_error) for t in rows])),
    )
