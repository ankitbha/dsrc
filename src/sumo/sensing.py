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

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    across onto them. Geometry is read once; a vehicle's length is read the first time it
    is seen.
    """

    def __init__(self, env, successor: Mapping[str, str]) -> None:
        from src.sumo.binding import _sumo

        self._sumo = _sumo
        self.successor = dict(successor)
        segment_edges = [e for segment in env.segments for e in segment]
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
