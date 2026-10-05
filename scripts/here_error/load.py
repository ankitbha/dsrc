"""Read the phone log, the HERE bodies and the video index of one run.

Loaders refuse (raise `LoadError`) rather than return something partly wrong.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from here_error.clock import ClockOffset, wall_to_utc
from here_error.models import (
    Frame,
    GpsFix,
    HereBody,
    HereSegment,
    HereSubSegment,
)

QUERY_CENTRE_TOL_DEG = 1e-6


class LoadError(ValueError):
    pass


@dataclass(frozen=True)
class PhoneHereRecord:
    seq: int
    wall_s: float
    response_mono_s: float | None
    query_lat: float
    query_lon: float


@dataclass(frozen=True)
class PhoneCameraRecord:
    frame_id: int
    wall_s: float
    capture_wall_s: float


@dataclass(frozen=True)
class PhoneLog:
    path: str
    gps: tuple[GpsFix, ...]
    here: tuple[PhoneHereRecord, ...]
    camera: tuple[PhoneCameraRecord, ...]
    n_gps_invalid: int
    n_gps_duplicate_time: int


def _ns(x) -> float:
    return x / 1e9


def load_phone_log(path: Path | str) -> PhoneLog:
    """Valid GPS fixes sorted by fix time with duplicate fix times dropped, HERE records, camera records.

    Lines are compact JSON with alphabetically ordered keys, so the channel is matched as a
    substring anywhere in the line.
    """
    path = Path(path)
    gps, here, camera = [], [], []
    n_invalid = 0
    with path.open() as fh:
        for line in fh:
            if '"ch":"gps"' in line:
                r = json.loads(line)
                if not r.get("valid") or r.get("lat") is None or r.get("lon") is None:
                    n_invalid += 1
                    continue
                gps.append((r["utc_epoch_ns"], GpsFix(
                    utc_s=_ns(r["utc_epoch_ns"]),
                    wall_s=_ns(r["t_wall_ns"]),
                    lat=r["lat"],
                    lon=r["lon"],
                    speed_mps=r.get("speed_mps"),
                    heading_deg=r.get("heading_deg"),
                )))
            elif '"ch":"here"' in line:
                r = json.loads(line)
                here.append(PhoneHereRecord(
                    r["seq"], _ns(r["t_wall_ns"]),
                    _ns(r["t_response_mono_ns"]) if r.get("t_response_mono_ns") is not None else None,
                    r["query_lat"], r["query_lon"]))
            elif '"ch":"camera"' in line:
                r = json.loads(line)
                capture = _ns(r["t_wall_ns"]) - (r["t_mono_ns"] - r["t_capture_mono_ns"]) / 1e9
                camera.append(PhoneCameraRecord(r["frame_id"], _ns(r["t_wall_ns"]), capture))
    gps.sort(key=lambda x: x[0])
    kept, n_dup, last = [], 0, None
    for ns, fix in gps:
        if ns == last:
            n_dup += 1
            continue
        kept.append(fix)
        last = ns
    return PhoneLog(str(path), tuple(kept), tuple(here), tuple(camera), n_invalid, n_dup)


def _parse_time(text: str) -> float:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _modal_functional_class(links) -> int | None:
    by_class: dict[int, float] = {}
    for link in links:
        fc = link.get("functionalClass")
        if fc is not None:
            by_class[fc] = by_class.get(fc, 0.0) + float(link.get("length", 0.0))
    if not by_class:
        return None
    return max(by_class, key=lambda k: (by_class[k], -k))


def _uncapped(flow: dict) -> tuple[float | None, bool]:
    """HERE's `speedUncapped`, or `speed` with a flag when the body does not carry it.

    `speed` is capped at the speed limit; `speedUncapped` and `freeFlow` are not.
    """
    if flow.get("speedUncapped") is not None:
        return flow["speedUncapped"], False
    return flow.get("speed"), True


def parse_segment(result: dict) -> HereSegment:
    loc = result["location"]
    links = loc["shape"]["links"]
    lat: list[float] = []
    lon: list[float] = []
    for link in links:
        for p in link["points"]:
            if lat and lat[-1] == p["lat"] and lon[-1] == p["lng"]:
                continue
            lat.append(p["lat"])
            lon.append(p["lng"])
    if len(lat) < 2:
        raise LoadError(f"segment {loc.get('description')!r} has fewer than two shape points")
    desc = loc.get("description") or ""
    length = float(loc["length"])
    key = f"{desc}|{round(length)}|{lat[0]:.5f},{lon[0]:.5f}|{lat[-1]:.5f},{lon[-1]:.5f}"
    flow = result.get("currentFlow", {})
    subs = []
    for s in flow.get("subSegments", []):
        unc, fell_back = _uncapped(s)
        subs.append(HereSubSegment(
            length_m=float(s["length"]),
            speed_mps=s.get("speed"),
            free_flow_mps=s.get("freeFlow"),
            jam_factor=s.get("jamFactor"),
            speed_uncapped_mps=unc,
            uncapped_fallback=fell_back,
        ))
    unc, fell_back = _uncapped(flow)
    return HereSegment(
        key=key,
        description=desc,
        stated_length_m=length,
        lat=tuple(lat),
        lon=tuple(lon),
        speed_mps=flow.get("speed"),
        free_flow_mps=flow.get("freeFlow"),
        jam_factor=flow.get("jamFactor"),
        confidence=flow.get("confidence"),
        functional_class=_modal_functional_class(links),
        subsegments=tuple(subs),
        speed_uncapped_mps=unc,
        uncapped_fallback=fell_back,
    )


MONO_OFFSET_TOL_S = 0.5


def load_here_bodies(run: str, run_dir: Path | str, phone: PhoneLog, offset: ClockOffset) -> tuple[HereBody, ...]:
    """Pair the run's `here_index.jsonl` records with the phone log's `here` records, in order.

    The phone's own sequence number is not a key: in one phone log it restarts from 0 partway
    through, so the i-th index record is paired with the i-th phone record. Three checks guard
    the pairing, and any failure refuses the load: the counts agree; the two query centres
    agree; and the phone's monotonic response time minus the index's receipt time is the same
    for every pair, within MONO_OFFSET_TOL_S, which a mis-paired record would break. The phone
    wall time of the pair is what places a body on the GPS axis.
    """
    run_dir = Path(run_dir)
    index = []
    with (run_dir / "here_index.jsonl").open() as fh:
        index = [json.loads(line) for line in fh if line.strip()]
    if len(index) != len(phone.here):
        raise LoadError(f"{run}: {len(index)} HERE index records but {len(phone.here)} phone records")
    mono_gaps = []
    no_mono = [idx["seq"] for idx, ph in zip(index, phone.here)
               if ph.response_mono_s is None or idx.get("received_t_mono") is None]
    if no_mono:
        raise LoadError(f"{run}: {len(no_mono)} HERE records lack the monotonic times (phone t_response_mono_ns, index "
                        f"received_t_mono) that the receipt-time pairing check needs, first at index seq {no_mono[0]}; "
                        "refusing to pair them unchecked")
    for idx, ph in zip(index, phone.here):
        if (abs(ph.query_lat - idx["query_lat"]) > QUERY_CENTRE_TOL_DEG
                or abs(ph.query_lon - idx["query_lon"]) > QUERY_CENTRE_TOL_DEG):
            raise LoadError(f"{run}: HERE index record {idx['seq']} and its phone record: query centres disagree")
        mono_gaps.append(ph.response_mono_s - idx["received_t_mono"])
    if mono_gaps and max(mono_gaps) - min(mono_gaps) > MONO_OFFSET_TOL_S:
        raise LoadError(f"{run}: phone and Jetson receipt times do not line up across HERE records "
                        f"(spread {max(mono_gaps) - min(mono_gaps):.3f} s)")
    bodies = []
    for idx, ph in zip(index, phone.here):
        body = json.loads((run_dir / idx["file"]).read_text())
        bodies.append(HereBody(
            run=run,
            seq=idx["seq"],
            source_file=idx["file"],
            data_time_s=_parse_time(body["sourceUpdated"]),
            response_utc_s=wall_to_utc(ph.wall_s, offset),
            query_lat=idx["query_lat"],
            query_lon=idx["query_lon"],
            segments=tuple(parse_segment(r) for r in body.get("results", [])),
        ))
    return tuple(bodies)


def load_frames(run: str, run_dir: Path | str, phone: PhoneLog, offset: ClockOffset) -> tuple[Frame, ...]:
    """Map each `video_index.jsonl` position to its frame id, then to the phone's camera record.

    Refuses on a repeated frame id or a frame id with no camera record.
    """
    run_dir = Path(run_dir)
    cam: dict[int, PhoneCameraRecord] = {}
    for rec in phone.camera:
        if rec.frame_id in cam:
            raise LoadError(f"{phone.path}: camera frame_id {rec.frame_id} appears twice")
        cam[rec.frame_id] = rec
    frames, seen = [], set()
    with (run_dir / "video_index.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            v = json.loads(line)
            fid = v["frame_id"]
            if fid in seen:
                raise LoadError(f"{run}: frame_id {fid} repeats in the video index")
            seen.add(fid)
            rec = cam.get(fid)
            if rec is None:
                raise LoadError(f"{run}: frame_id {fid} has no camera record in the phone log")
            frames.append(Frame(run, v["pos"], fid, wall_to_utc(rec.capture_wall_s, offset)))
    return tuple(frames)
