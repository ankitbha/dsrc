"""What a dashboard phone camera on a controlled vehicle measures, in SUMO.

Plan 2026-10-16 sections 1.2 to 1.5. Every 5 s one pass reads every lane of the scanned
edges (`LaneSnapshot`), and for every sensing vehicle the leaders it could see are listed
in one vectorised table (`CandidateTable`): vehicles ahead on its own and adjacent lanes
within range, continuing onto its route successor, with the field of view and occlusion
by closer vehicles applied. Step S4 adds the camera model that detects, mis-measures and
summarises them, and the oracle that counts every candidate at its true distance.

Geometry, as the plan states it. A candidate's gap is to its back, `d = x_leader -
length_leader - x`, in `(0, R]`. Its lateral offset is its lane offset times the ego lane's
width. A vehicle is a box 1.8 m wide, taller than the camera, centred on its lane on a flat
road, so its azimuth interval is `[(y - 0.9)/d, (y + 0.9)/d]` (small angle). It is hidden
when closer candidates cover at least half of that interval. An adjacent-lane vehicle is in
the field of view when `|y| / d < 360/800`, the upright frame's half width over `fx`.
Vehicles crossing a junction are on internal lanes, which are not scanned.

Which lanes count as visible is the ego's own edge's: the offsets -1, 0, +1 that exist
there. A successor lane whose offset is not one of them is not scanned for that ego.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Any

import numpy as np

#: The vehicle box's width (plan section 1.2).
VEHICLE_WIDTH_M: float = 1.8
#: The upright frame is 720 px wide and the camera's assumed focal length is 800 px.
FX_PX: float = 800.0
HALF_WIDTH_PX: float = 360.0
#: `tan` of the horizontal half-angle, `atan(360/800) = 24.2 deg`; the plan rounds it to
#: 0.449 (`tan(24.2 deg)`), the unrounded ratio is 0.45.
FOV_HALF_TAN: float = HALF_WIDTH_PX / FX_PX
#: The deployment's `observation.effective_range_m`.
DEPLOYED_RANGE_M: float = 80.0
#: Camera draws are `default_rng(seed + CAMERA_SEED_OFFSET)` (plan J33).
CAMERA_SEED_OFFSET: int = 5000
_STANDARD_NORMAL = NormalDist()
LANE_OFFSETS: tuple[int, ...] = (-1, 0, 1)

# Rows of the global sort key are `lane_key * _SPAN + _PAD + back`; `_PAD` keeps a query
# window that starts before the lane's origin (a successor's) inside its lane's block.
_SPAN: float = 1.0e5
_PAD: float = 5.0e3


@dataclass(frozen=True)
class EdgeGeometry:
    lanes: int
    length_m: float
    widths: tuple[float, ...]


@dataclass(frozen=True)
class Ego:
    """A sensing vehicle: its id, edge, lane index and front position on the lane."""

    vehicle_id: str
    edge: str
    lane: int
    x: float


def successors(routes: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Each edge's successor along the routes; an edge with two successors is refused."""
    out: dict[str, str] = {}
    for edges in routes.values():
        for edge, following in zip(edges, edges[1:]):
            if out.setdefault(edge, following) != following:
                raise ValueError(f"edge {edge} has two route successors, "
                                 f"{out[edge]} and {following}")
    return out


class LaneSnapshot:
    """Every vehicle on the scanned lanes at one instant, ordered by lane and back position.

    `ids`, `edge`, `lane`, `front`, `back`, `length` and `speed` are parallel, in that
    order; `geometry` holds every scanned edge and `successor` the route successor map.
    """

    def __init__(self, t: float, geometry: Mapping[str, EdgeGeometry],
                 successor: Mapping[str, str], ids: Sequence[str], edges: Sequence[str],
                 lanes: Sequence[int], front: Sequence[float], length: Sequence[float],
                 speed: Sequence[float]) -> None:
        self.t = float(t)
        self.geometry = dict(geometry)
        self.successor = dict(successor)
        too_long = [e for e, g in self.geometry.items() if g.length_m > _SPAN - 2.0 * _PAD]
        if too_long:
            raise ValueError(f"edges {too_long[:3]} exceed the {(_SPAN - 2 * _PAD) / 1000:g} km a "
                             "lane's block of the sort key holds")
        self.lane_keys: dict[tuple[str, int], int] = {}
        for edge in sorted(self.geometry):
            for lane in range(self.geometry[edge].lanes):
                self.lane_keys[(edge, lane)] = len(self.lane_keys)
        keys = np.array([self.lane_keys[(e, int(l))] for e, l in zip(edges, lanes)],
                        dtype=np.int64)
        front = np.asarray(front, dtype=float)
        length = np.asarray(length, dtype=float)
        back = front - length
        order = np.lexsort((back, keys))
        self.ids = [ids[k] for k in order]
        self.edge = [edges[k] for k in order]
        self.lane = np.asarray(lanes, dtype=np.int64)[order]
        self.front = front[order]
        self.back = back[order]
        self.length = length[order]
        self.speed = np.asarray(speed, dtype=float)[order]
        self._key = keys[order]
        self.sort_key = self._key * _SPAN + _PAD + self.back
        self.index = {vid: k for k, vid in enumerate(self.ids)}

    @classmethod
    def from_vehicles(cls, vehicles: Iterable[Mapping[str, Any]],
                      geometry: Mapping[str, EdgeGeometry], successor: Mapping[str, str],
                      t: float = 0.0) -> "LaneSnapshot":
        """From dictionaries with `id`, `edge`, `lane`, `front_m`, `length_m`, `speed_mps`."""
        rows = list(vehicles)
        return cls(t, geometry, successor, [r["id"] for r in rows], [r["edge"] for r in rows],
                   [int(r["lane"]) for r in rows], [float(r["front_m"]) for r in rows],
                   [float(r["length_m"]) for r in rows], [float(r["speed_mps"]) for r in rows])

    @staticmethod
    def builder(env, successor: Mapping[str, str]) -> "SnapshotBuilder":
        return SnapshotBuilder(env, successor)

    def egos(self, controlled: Callable[[str], bool], ego_edges: set[str]) -> list[Ego]:
        """The vehicles on `ego_edges` for which `controlled` holds, as egos."""
        out = []
        for k, vid in enumerate(self.ids):
            if self.edge[k] in ego_edges and controlled(vid):
                out.append(Ego(vid, self.edge[k], int(self.lane[k]), float(self.front[k])))
        return out


class SnapshotBuilder:
    """Reads one `LaneSnapshot` per call from a running env's SUMO.

    Scans every lane of the env's segment edges and of their route successors, which
    include the unsegmented `153` and `exit_road`, so a vehicle near a segment's end sees
    across onto them. On the approach boundary the segment edges include the approach
    edges (`env.all_segments`), whose successors are the next approach edge or the entry
    link. Geometry is read once; a vehicle's length is read the first time it is seen.
    """

    def __init__(self, env, successor: Mapping[str, str]) -> None:
        from src.sumo.binding import _sumo

        self._sumo = _sumo
        self.successor = dict(successor)
        segments = getattr(env, "all_segments", None) or env.segments
        segment_edges = [e for segment in segments for e in segment]
        scan = list(dict.fromkeys(segment_edges + [self.successor[e] for e in segment_edges
                                                   if e in self.successor]))
        self.geometry: dict[str, EdgeGeometry] = {}
        for edge in scan:
            lanes = int(_sumo.edge.getLaneNumber(edge))
            self.geometry[edge] = EdgeGeometry(
                lanes=lanes, length_m=float(_sumo.lane.getLength(f"{edge}_0")),
                widths=tuple(float(_sumo.lane.getWidth(f"{edge}_{i}")) for i in range(lanes)))
        self._lanes = [(edge, i, f"{edge}_{i}") for edge in scan
                       for i in range(self.geometry[edge].lanes)]
        self._length: dict[str, float] = {}

    def build(self, t: float) -> LaneSnapshot:
        sumo = self._sumo
        ids, edges, lanes, front, length, speed = [], [], [], [], [], []
        known = self._length
        for edge, lane, lane_id in self._lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane_id):
                ids.append(vid)
                edges.append(edge)
                lanes.append(lane)
                front.append(sumo.vehicle.getLanePosition(vid))
                speed.append(sumo.vehicle.getSpeed(vid))
                if vid not in known:
                    known[vid] = float(sumo.vehicle.getLength(vid))
                length.append(known[vid])
        return LaneSnapshot(t, self.geometry, self.successor, ids, edges, lanes, front,
                            length, speed)


@dataclass
class CandidateTable:
    """Every (ego, candidate) pair of one pass, ordered by ego id and then gap.

    Per row: `ego` (index into `ego_ids`), `vehicle` (index into the snapshot), gap `d`,
    lane offset `dlane`, lateral offset `y`, true `speed`, `in_fov`, `occluded`, and
    `visible = in_fov & ~occluded`. Per ego: `lanes_visible` (n x 3, offsets -1, 0, +1
    present on the ego's edge), `ego_edges`, `ego_lane`, `ego_x`.
    """

    snapshot: LaneSnapshot
    R: float
    ego_ids: list[str]
    ego_edges: list[str]
    ego_lane: np.ndarray
    ego_x: np.ndarray
    lanes_visible: np.ndarray
    ego: np.ndarray
    vehicle: np.ndarray
    d: np.ndarray
    dlane: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    in_fov: np.ndarray
    occluded: np.ndarray
    visible: np.ndarray

    @property
    def n_egos(self) -> int:
        return len(self.ego_ids)

    def counts(self, R: float | None = None) -> np.ndarray:
        """Candidates per ego on its visible lanes with `d <= R` (default the table's)."""
        keep = self.d <= (self.R if R is None else R)
        return np.bincount(self.ego[keep], minlength=self.n_egos)

    @classmethod
    def build(cls, snapshot: LaneSnapshot, egos: Sequence[Ego], R: float,
              width: float = VEHICLE_WIDTH_M) -> "CandidateTable":
        egos = sorted(egos, key=lambda e: e.vehicle_id)
        n = len(egos)
        geometry = snapshot.geometry
        ego_lane = np.array([e.lane for e in egos], dtype=np.int64)
        ego_x = np.array([e.x for e in egos], dtype=float)
        lanes_visible = np.zeros((n, 3), dtype=bool)
        q_ego, q_dlane, q_key, q_base = [], [], [], []
        ego_width = np.empty(n)
        for k, ego in enumerate(egos):
            g = geometry[ego.edge]
            ego_width[k] = g.widths[ego.lane]
            nxt = snapshot.successor.get(ego.edge)
            sees_next = nxt in geometry and ego.x + R > g.length_m
            j = min(ego.lane, geometry[nxt].lanes - 1) if sees_next else 0
            for column, offset in enumerate(LANE_OFFSETS):
                lane = ego.lane + offset
                if not 0 <= lane < g.lanes:
                    continue
                lanes_visible[k, column] = True
                q_ego.append(k)
                q_dlane.append(offset)
                q_key.append(snapshot.lane_keys[(ego.edge, lane)])
                q_base.append(ego.x)
                if sees_next and 0 <= j + offset < geometry[nxt].lanes:
                    q_ego.append(k)
                    q_dlane.append(offset)
                    q_key.append(snapshot.lane_keys[(nxt, j + offset)])
                    q_base.append(ego.x - g.length_m)
        q_ego = np.asarray(q_ego, dtype=np.int64)
        q_dlane = np.asarray(q_dlane, dtype=np.int64)
        q_base = np.asarray(q_base, dtype=float)
        start = np.asarray(q_key, dtype=float) * _SPAN + _PAD + q_base
        lo = np.searchsorted(snapshot.sort_key, start, side="right")
        hi = np.searchsorted(snapshot.sort_key, start + R, side="right")
        sizes = hi - lo
        total = int(sizes.sum())
        query = np.repeat(np.arange(len(sizes)), sizes)
        within = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        vehicle = lo[query] + within
        ego = q_ego[query]
        d = snapshot.back[vehicle] - q_base[query]
        dlane = q_dlane[query]
        order = np.lexsort((d, ego))
        ego, vehicle, d, dlane = ego[order], vehicle[order], d[order], dlane[order]
        y = dlane * ego_width[ego]
        in_fov = (dlane == 0) | (np.abs(y) / d < FOV_HALF_TAN)
        hidden = _occluded_table(ego, d, dlane, y, n, width)
        return cls(snapshot, float(R), [e.vehicle_id for e in egos], [e.edge for e in egos],
                   ego_lane, ego_x, lanes_visible, ego, vehicle, d, dlane, y,
                   snapshot.speed[vehicle], in_fov, hidden, in_fov & ~hidden)


def _occluded_table(ego: np.ndarray, d: np.ndarray, dlane: np.ndarray, y: np.ndarray,
                    n_egos: int, width: float) -> np.ndarray:
    """The general interval rule, computed exactly for equal-width lanes per ego.

    Own-lane intervals are nested, so the union of the closer own-lane ones is the nearest
    one's; same-side adjacent intervals shift toward the axis with distance, so within a
    farther vehicle's interval the closer same-lane ones are covered by the next closer
    one's; opposite-side intervals do not meet. A candidate's covered share is therefore
    the union of at most two overlaps, which `occluded` computes in general.
    """
    hw = width / 2.0
    hidden = np.zeros(len(d), dtype=bool)
    if not len(d):
        return hidden
    own = dlane == 0
    d0 = np.full(n_egos, np.inf)
    np.minimum.at(d0, ego[own], d[own])
    hidden[own] = d[own] > d0[ego[own]]
    side = ~own
    if not side.any():
        return hidden
    rows = np.flatnonzero(side)
    order = rows[np.lexsort((d[rows], dlane[rows], ego[rows]))]
    same_group = np.zeros(len(order), dtype=bool)
    same_group[1:] = (ego[order][1:] == ego[order][:-1]) & (dlane[order][1:] == dlane[order][:-1])
    prev_d = np.full(len(order), np.nan)
    prev_d[1:] = np.where(same_group[1:], d[order][:-1], np.nan)
    dk, yk = d[order], y[order]
    lo_c, hi_c = (yk - hw) / dk, (yk + hw) / dk
    nearest = d0[ego[order]]
    has_a = nearest < dk
    a_lo = np.where(has_a, np.maximum(lo_c, -hw / nearest), 0.0)
    a_hi = np.where(has_a, np.minimum(hi_c, hw / nearest), 0.0)
    has_b = prev_d < dk  # nan compares False
    safe_prev = np.where(has_b, prev_d, 1.0)
    b_lo = np.where(has_b, np.maximum(lo_c, (yk - hw) / safe_prev), 0.0)
    b_hi = np.where(has_b, np.minimum(hi_c, (yk + hw) / safe_prev), 0.0)
    len_a = np.maximum(a_hi - a_lo, 0.0) * has_a
    len_b = np.maximum(b_hi - b_lo, 0.0) * has_b
    both = np.maximum(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0) * (has_a & has_b)
    covered = len_a + len_b - both
    hidden[order] = covered >= 0.5 * (hi_c - lo_c)
    return hidden


def occluded(d: np.ndarray, y: np.ndarray, width: float = VEHICLE_WIDTH_M) -> np.ndarray:
    """The plan's definition, for one ego's candidates: hidden when the union of the
    closer candidates' azimuth intervals covers at least half of its own interval."""
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    hw = width / 2.0
    lo, hi = (y - hw) / d, (y + hw) / d
    out = np.zeros(len(d), dtype=bool)
    for k in range(len(d)):
        closer = np.flatnonzero(d < d[k])
        pieces = sorted((max(lo[j], lo[k]), min(hi[j], hi[k])) for j in closer
                        if min(hi[j], hi[k]) > max(lo[j], lo[k]))
        covered, reach = 0.0, -math.inf
        for a, b in pieces:
            a = max(a, reach)
            if b > a:
                covered += b - a
                reach = b
        out[k] = covered >= 0.5 * (hi[k] - lo[k])
    return out


def closed_form_occluded(d: np.ndarray, dlane: np.ndarray, lane_width: float,
                         width: float = VEHICLE_WIDTH_M) -> np.ndarray:
    """The plan's three closed-form rules on equal-width lanes, kept as a check.

    Own lane: only the nearest leader is visible. Adjacent lane: hidden by the next closer
    vehicle in its lane at `d1` when `d2 <= y/(y - w/2) d1` (1.32 at 3.7 m), and by the
    own-lane leader at `d0` when `d2 >= y/(w/2) d0` (4.11). They are sufficient but not
    necessary: the own-lane leader and the next closer same-lane vehicle can together
    cover half of an interval that neither covers alone, which `occluded` counts.
    """
    d = np.asarray(d, dtype=float)
    dlane = np.asarray(dlane)
    hw = width / 2.0
    y = lane_width
    out = np.zeros(len(d), dtype=bool)
    own = d[dlane == 0]
    for k in range(len(d)):
        if dlane[k] == 0:
            out[k] = bool(np.any(own < d[k]))
            continue
        same = d[(dlane == dlane[k]) & (d < d[k])]
        if len(same) and d[k] <= y / (y - hw) * same.max():
            out[k] = True
        nearer_own = own[own < d[k]]
        if len(nearer_own) and d[k] >= y / hw * nearer_own.min():
            out[k] = True
    return out


@dataclass
class EgoCandidates:
    """One ego's rows of a `CandidateTable`, in gap order."""

    ids: list[str]
    d: np.ndarray
    dlane: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    in_fov: np.ndarray
    occluded: np.ndarray
    visible: np.ndarray


def visible_candidates(snapshot: LaneSnapshot, ego: Ego, R: float) -> EgoCandidates:
    """The leaders `ego` could see within `R`, with the field of view and occlusion flags."""
    table = CandidateTable.build(snapshot, [ego], R)
    return EgoCandidates([snapshot.ids[v] for v in table.vehicle], table.d, table.dlane,
                         table.y, table.speed, table.in_fov, table.occluded, table.visible)


# ----------------------------------------------------------------- the camera


#: The metres of road one vehicle occupies, the same as `mainz.VEHICLE_LENGTH_M`, so the
#: local density is in the units of the critical occupancy 0.178.
OCCUPANCY_LENGTH_M: float = 4.5


def _fill_by_interpolation(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Bins without an estimate (nan) take the interpolation of the bins with one; a table
    with no estimate at all is zero."""
    values = np.asarray(values, dtype=float)
    known = ~np.isnan(values)
    if not known.any():
        return np.zeros_like(values)
    return np.interp(centres, centres[known], values[known])


@dataclass(frozen=True)
class Detections:
    """Per row of a `CandidateTable`: detected, reported gap, reported speed, confidence.

    Rows not detected carry nan in `d_hat`, `v_hat` and `conf`.
    """

    detected: np.ndarray
    d_hat: np.ndarray
    v_hat: np.ndarray
    conf: np.ndarray


@dataclass(frozen=True)
class CameraModel:
    """Detection probability, distance noise, speed noise and confidence by distance bin.

    `p_at(d)` is the bin's detection rate and zero beyond `R_max`; `sigma_at(d)` and
    `sigma_v_at(d)` interpolate the bin values linearly between bin centres, holding the
    end values beyond them. A detected candidate is reported at `max(b d + eps, 0)` with
    `eps ~ N(0, sigma(d)^2)` and at speed `v + eta`, `eta ~ N(0, sigma_v(d)^2)`, with a
    confidence drawn from the drive's confidences in its bin. The three switches turn
    occlusion, detection loss and noise off one at a time (plan attack point 3).
    """

    bin_m: float
    p: np.ndarray
    sigma_m: np.ndarray
    sigma_v_mps: np.ndarray
    confidences: tuple[np.ndarray, ...]
    R_max: float
    bias: float = 1.0
    occlusion: bool = True
    detection_loss: bool = True
    noise: bool = True
    source: dict[str, Any] | None = None
    #: Amendment 3 (R2-4): the log-spread of a persistent per-(ego, leader) distance
    #: factor; 0 turns it off.
    pair_sigma_log: float = 0.0

    @classmethod
    def from_tables(cls, *, bin_m: float, p, sigma_m, sigma_v_mps, confidences, R_max: float,
                    bias: float = 1.0, source: dict[str, Any] | None = None) -> "CameraModel":
        p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0)
        centres = bin_m * (np.arange(len(p)) + 0.5)
        sigma = _fill_by_interpolation(np.asarray(sigma_m, dtype=float), centres)
        sigma_v = _fill_by_interpolation(np.asarray(sigma_v_mps, dtype=float), centres)
        confs = tuple(np.asarray(c, dtype=float) for c in confidences)
        return cls(float(bin_m), p, sigma, sigma_v, confs, float(R_max), float(bias),
                   source=source)

    @classmethod
    def from_artefact(cls, path, bias: float = 1.0, table: str | None = None,
                      pair_sigma_log: float | None = None) -> "CameraModel":
        """Read `results/jam/camera_error_model.json` (written by `fit_camera_model.py`):
        its SUMO table (`sumo_table`, the 0.20 s drives) unless `table` names another. The
        persistent pair factor's spread is the table's stored `persistent_error` spread
        unless `pair_sigma_log` overrides it."""
        import hashlib
        import json
        from pathlib import Path

        path = Path(path)
        raw = path.read_bytes()
        artefact = json.loads(raw)
        name = table or artefact["sumo_table"]
        t = artefact["tables"][name]

        def column(values):
            return [math.nan if v is None else float(v) for v in values]

        stored = (t.get("persistent_error") or {}).get("relative_spread")
        spread = float(pair_sigma_log if pair_sigma_log is not None else (stored or 0.0))
        model = cls.from_tables(
            bin_m=float(artefact["bin_m"]), p=column(t["detection"]["p"]),
            sigma_m=column(t["distance_noise"]["sigma_ema_m"]),
            sigma_v_mps=column(t["closing_rate_noise"]["sigma_v_mps"]),
            # A bin without a detection draws from every bin's confidences.
            confidences=[v or [c for b in t["confidence"]["values"] for c in b] or [math.nan]
                         for v in t["confidence"]["values"]],
            R_max=float(t["R_max_m"] or 0.0), bias=bias,
            source={"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                    "table": name, "method": artefact.get("method", t.get("method")),
                    "stored_pair_sigma_log": stored})
        return model.with_pair_sigma(spread)

    def with_switches(self, **switches: bool) -> "CameraModel":
        from dataclasses import replace

        return replace(self, **switches)

    def with_bias(self, bias: float) -> "CameraModel":
        from dataclasses import replace

        return replace(self, bias=float(bias))

    def with_pair_sigma(self, sigma_log: float) -> "CameraModel":
        from dataclasses import replace

        if sigma_log < 0.0:
            raise ValueError(f"a spread cannot be negative, got {sigma_log}")
        return replace(self, pair_sigma_log=float(sigma_log))

    @property
    def R_eff(self) -> float:
        """`min(80 m, R_max)`: the range the counts and the density use."""
        return min(DEPLOYED_RANGE_M, self.R_max)

    def _bin(self, d: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(d, dtype=float) // self.bin_m).astype(np.int64), 0,
                       len(self.p) - 1)

    def p_at(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        inside = (d <= self.R_max) & (d < self.bin_m * len(self.p))
        return np.where(inside, self.p[self._bin(d)], 0.0)

    def _interp(self, values: np.ndarray, d: np.ndarray) -> np.ndarray:
        centres = self.bin_m * (np.arange(len(values)) + 0.5)
        return np.interp(np.asarray(d, dtype=float), centres, values)

    def sigma_at(self, d: np.ndarray) -> np.ndarray:
        return self._interp(self.sigma_m, d)

    def sigma_v_at(self, d: np.ndarray) -> np.ndarray:
        return self._interp(self.sigma_v_mps, d)

    def observe(self, table: CandidateTable, rng: np.random.Generator,
                seed: int | None = None) -> Detections:
        """One tick of the camera for every ego of `table`.

        Draws are made for the rows the camera could see (in view, not hidden unless
        occlusion is off, within `R_max`), in the table's order (ego id, then gap), four
        per row: detection, distance noise, speed noise, confidence index. With a positive
        `pair_sigma_log`, each (ego, leader) pair's reported distance also carries the
        persistent factor `pair_factor(seed, ego, leader)`, so `seed` (the episode's) is
        required; the factor multiplies with `b`.
        """
        if self.pair_sigma_log > 0.0 and seed is None:
            raise ValueError("a persistent pair factor needs the episode seed")
        n = len(table.d)
        seeable = (table.in_fov & (~table.occluded if self.occlusion else True)
                   & (table.d <= self.R_max))
        rows = np.flatnonzero(seeable)
        u = rng.random(len(rows))
        z_d = rng.standard_normal(len(rows))
        z_v = rng.standard_normal(len(rows))
        u_c = rng.random(len(rows))
        d = table.d[rows]
        p = self.p_at(d) if self.detection_loss else np.ones(len(rows))
        hit = u < p
        detected = np.zeros(n, dtype=bool)
        detected[rows[hit]] = True
        d_hat = np.full(n, np.nan)
        v_hat = np.full(n, np.nan)
        conf = np.full(n, np.nan)
        scale = 1.0 if self.noise else 0.0
        factor = np.ones(len(rows))
        if self.pair_sigma_log > 0.0 and len(rows):
            egos = table.ego_ids
            ids = table.snapshot.ids
            factor = np.array([pair_factor(int(seed), egos[e], ids[v], self.pair_sigma_log)
                               for e, v in zip(table.ego[rows], table.vehicle[rows])])
        reported = np.maximum(self.bias * factor * d + scale * self.sigma_at(d) * z_d, 0.0)
        speed = table.speed[rows] + scale * self.sigma_v_at(d) * z_v
        bins = self._bin(d)
        drawn = np.array([self.confidences[b][min(int(u * len(self.confidences[b])),
                                                  len(self.confidences[b]) - 1)]
                          for b, u in zip(bins, u_c)], dtype=float)
        d_hat[rows[hit]] = reported[hit]
        v_hat[rows[hit]] = speed[hit]
        conf[rows[hit]] = drawn[hit]
        return Detections(detected, d_hat, v_hat, conf)


@lru_cache(maxsize=1 << 20)
def _pair_z(seed: int, ego: str, leader: str) -> float:
    """A standard normal fixed by `(seed, ego, leader)`, from a stable hash (not Python's
    salted `hash`), so a pair keeps its value for the whole episode and across processes."""
    digest = hashlib.blake2b(f"{seed}|{ego}|{leader}".encode(), digest_size=8).digest()
    u = (int.from_bytes(digest, "big") + 0.5) / 2.0 ** 64
    return _STANDARD_NORMAL.inv_cdf(u)


def pair_factor(seed: int, ego: str, leader: str, sigma_log: float) -> float:
    """Amendment 3 (R2-4): the persistent log-normal distance factor of one (ego, leader)
    pair in one episode, `exp(sigma_log z)`. It stands for the per-vehicle scale error of a
    monocular estimator (a vehicle's true width against the class prior, its angle of
    view), constant while the pair lasts."""
    return math.exp(sigma_log * _pair_z(seed, ego, leader)) if sigma_log > 0.0 else 1.0


@dataclass(frozen=True)
class LocalEstimate:
    """Per ego (in `CandidateTable.ego_ids` order), what the device computes (section 1.3).

    `n_hat` (n x 3) counts detections within `R_eff` on offsets -1, 0, +1; `g_hat` is the
    reported gap of the nearest detected own-lane leader (nan if none); `k_hat` is the mean
    over the visible lanes of `L/(g_hat + L)` (own lane, 0 when no leader within `R_eff`)
    and `n_hat L / R_eff` (adjacent lanes); `own_speed` and `mean_speed` are reported
    leader speeds in m/s (nan when nothing is detected); `conf_mean` and `n_det` are the
    mean confidence and the count of detections within `R_eff`.
    """

    ego_ids: list[str]
    lanes_visible: np.ndarray
    n_hat: np.ndarray
    g_hat: np.ndarray
    k_hat: np.ndarray
    own_speed: np.ndarray
    mean_speed: np.ndarray
    conf_mean: np.ndarray
    n_det: np.ndarray


def estimate(table: CandidateTable, detections: Detections, R_eff: float,
             L: float = OCCUPANCY_LENGTH_M) -> LocalEstimate:
    n = table.n_egos
    det = detections.detected
    in_range = det & (np.nan_to_num(detections.d_hat, nan=np.inf) <= R_eff)
    column = table.dlane + 1
    n_hat = np.bincount(table.ego[in_range] * 3 + column[in_range],
                        minlength=3 * n).reshape(n, 3)
    # The nearest detected own-lane leader: the first such row of each ego in gap order.
    own_rows = np.flatnonzero(det & (table.dlane == 0))
    g_hat = np.full(n, np.nan)
    own_speed = np.full(n, np.nan)
    if len(own_rows):
        first = own_rows[np.unique(table.ego[own_rows], return_index=True)[1]]
        g_hat[table.ego[first]] = detections.d_hat[first]
        own_speed[table.ego[first]] = detections.v_hat[first]
    with np.errstate(invalid="ignore"):
        k_own = np.where(np.nan_to_num(g_hat, nan=np.inf) <= R_eff,
                         L / (np.nan_to_num(g_hat, nan=0.0) + L), 0.0)
    k_side = n_hat[:, [0, 2]] * L / R_eff if R_eff > 0 else np.zeros((n, 2))
    side_visible = table.lanes_visible[:, [0, 2]]
    lanes = table.lanes_visible.sum(axis=1)
    k_hat = (k_own + (k_side * side_visible).sum(axis=1)) / np.maximum(lanes, 1)
    count_all = np.bincount(table.ego[det], minlength=n)
    speed_sum = np.bincount(table.ego[det], weights=detections.v_hat[det], minlength=n)
    count_in = np.bincount(table.ego[in_range], minlength=n)
    conf_sum = np.bincount(table.ego[in_range], weights=detections.conf[in_range], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(count_all > 0, speed_sum / count_all, np.nan)
        conf_mean = np.where(count_in > 0, conf_sum / count_in, np.nan)
    return LocalEstimate(list(table.ego_ids), table.lanes_visible, n_hat, g_hat, k_hat,
                         own_speed, mean_speed, conf_mean, count_in)


@dataclass(frozen=True)
class OracleSensor:
    """Every candidate on the visible lanes within `R`, at its true distance, with no field
    of view, occlusion or detection loss (plan section 1.3; `R = 200` is `oracle_wide`)."""

    R: float = DEPLOYED_RANGE_M

    def per_lane_counts(self, table: CandidateTable) -> np.ndarray:
        keep = table.d <= self.R
        return np.bincount(table.ego[keep] * 3 + table.dlane[keep] + 1,
                           minlength=3 * table.n_egos).reshape(table.n_egos, 3)

    def k_true(self, table: CandidateTable, L: float = OCCUPANCY_LENGTH_M) -> np.ndarray:
        if table.R < self.R:
            raise ValueError(f"the table reaches {table.R} m, short of the oracle's {self.R} m")
        lanes = np.maximum(table.lanes_visible.sum(axis=1), 1)
        return table.counts(self.R) * L / (self.R * lanes)


class TrendTracker:
    """Per vehicle, the least-squares slope of its values over the last `window_s` (at most
    `max_samples` values); zero with fewer than `min_samples`. Nan values are skipped."""

    def __init__(self, window_s: float = 30.0, max_samples: int = 7,
                 min_samples: int = 3) -> None:
        self.window_s = window_s
        self.max_samples = max_samples
        self.min_samples = min_samples
        self._slot: dict[str, int] = {}
        self._t = np.full((64, max_samples), np.nan)
        self._v = np.full((64, max_samples), np.nan)
        self._next = np.zeros(64, dtype=np.int64)

    def _slots(self, ids: Sequence[str]) -> np.ndarray:
        slots = np.empty(len(ids), dtype=np.int64)
        for k, vid in enumerate(ids):
            slot = self._slot.get(vid)
            if slot is None:
                slot = self._slot[vid] = len(self._slot)
                if slot >= len(self._t):
                    grow = len(self._t)
                    self._t = np.vstack([self._t, np.full((grow, self.max_samples), np.nan)])
                    self._v = np.vstack([self._v, np.full((grow, self.max_samples), np.nan)])
                    self._next = np.concatenate([self._next, np.zeros(grow, dtype=np.int64)])
            slots[k] = slot
        return slots

    def update(self, ids: Sequence[str], t: float, values: np.ndarray) -> np.ndarray:
        if not len(ids):
            return np.zeros(0)
        slots = self._slots(ids)
        position = self._next[slots] % self.max_samples
        self._t[slots, position] = t
        self._v[slots, position] = np.asarray(values, dtype=float)
        self._next[slots] += 1
        times, vals = self._t[slots], self._v[slots]
        valid = (times >= t - self.window_s - 1e-6) & ~np.isnan(vals)
        count = valid.sum(axis=1)
        w = valid.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_mean = np.where(count > 0, np.nansum(np.where(valid, times, 0.0), axis=1)
                              / np.maximum(count, 1), 0.0)
            v_mean = np.where(count > 0, np.nansum(np.where(valid, vals, 0.0), axis=1)
                              / np.maximum(count, 1), 0.0)
            dt = np.where(valid, times - t_mean[:, None], 0.0)
            dv = np.where(valid, vals - v_mean[:, None], 0.0)
            sxx = (w * dt * dt).sum(axis=1)
            slope = np.where((count >= self.min_samples) & (sxx > 0),
                             (w * dt * dv).sum(axis=1) / np.where(sxx > 0, sxx, 1.0), 0.0)
        return slope
