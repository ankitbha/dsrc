"""Build the simulation actor's observation dict from real sensors.

This is the sim-to-real alignment core. Every field of the sim's local
observation (specs/observation_schema.md) is produced here, each tagged
with a provenance class in ``field_sources``:

  measured          directly from camera/GPS/V2V this tick
  derived           computed from measured quantities (e.g. headway)
  fallback_neutral  spec-mandated neutral value (e.g. no rear sensing)
  static_config     operator-provided constant (e.g. assumed lane)
  sim_parity        value the sim itself hardcodes (kept identical)

The provenance map is logged every tick and is the basis for the paper's
"observation missingness" metric.

Key geometry conventions (right-hand traffic, camera ~lane-centered):
  lateral_m > 0 is right of the camera axis; lane assignment is
  round(lateral / lane_width): 0 = ego lane, -1 = left, +1 = right.

Known v0 gaps (documented in ARCHITECTURE.md with upgrade paths):
  - no rear sensing -> follower_* and *_rear_gap use the sim's "empty
    road" values (inf gap / 0 relative speed); a second rear-facing
    camera fills these via a second detector instance.
  - forward-only counts -> density uses symmetric extrapolation
    (2 x forward count over +-range), toggleable via symmetrize_counts.
  - no map matching yet -> merge/bottleneck distances use sim-parity
    values (the sim currently hardcodes distance_to_next_merge = 0.0).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.distance import TrackedVehicle
from policy import sim_contract
from sensors.gps_reader import GpsFix

INF = float("inf")


@dataclass
class ObservationResult:
    obs: dict[str, Any]                # sim-schema observation dict
    encoded: np.ndarray                # (39,) float32 actor input
    field_sources: dict[str, str]      # provenance per field
    diagnostics: dict[str, Any]        # raw values for logging/eval


@dataclass
class BuilderConfig:
    effective_range_m: float = 80.0
    symmetrize_counts: bool = True
    free_flow_speed_mps: float = 30.0
    assumed_lane: int = 1
    lane_width_m: float = 3.7
    target_headway_default_s: float = 1.6
    queue_speed_mps: float = 5.0
    density_bin_edges_veh_per_km: tuple[float, ...] = (12.0, 30.0)
    mean_speed_bin_edges_mps: tuple[float, ...] = (8.0, 18.0)
    uncongested_density_threshold_veh_per_km: float = 12.0
    low_speed_free_flow_delta_mps: float = 8.0
    gps_stale_after_s: float = 2.0
    # How far into this clock's future a reading may sit and still be believed,
    # when it carries no uncertainty of its own. Covers `now` being sampled just
    # before the reading; anything larger is a clock problem, not sampling order.
    clock_sampling_epsilon_s: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BuilderConfig":
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in raw:
                value = raw[f]
                kwargs[f] = tuple(value) if isinstance(value, list) else value
        return cls(**kwargs)


@dataclass
class PeerState:
    """A cooperating AV heard over the V2V beacon channel."""
    peer_id: str
    distance_m: float
    speed_mps: float
    lane_id: int | None = None


@dataclass
class _EgoState:
    speed_samples: deque = field(default_factory=lambda: deque(maxlen=20))
    last_speed_mps: float = 0.0
    ever_had_fix: bool = False
    target_headway_s: float = 1.6


def _speed_provenance(gps: GpsFix) -> str:
    """`measured` for a local fix; for a remote one, how its stamp was obtained.

    Three outcomes rather than two, because "measured" would hide the difference
    between a reading whose freshness was established exactly and one where it
    rested on a converted stamp or on an arrival-time proxy.
    """
    stamp = getattr(gps, "timebase", None)
    if stamp is None:
        return "measured"
    if stamp.proxy:
        return "measured_arrival_proxy"
    return "measured_converted"


class ObservationBuilder:
    def __init__(self, config: BuilderConfig) -> None:
        self.config = config
        self._ego = _EgoState(target_headway_s=config.target_headway_default_s)

    def set_target_headway(self, headway_s: float) -> None:
        """Feed back the last commanded headway bin (mirrors the sim loop,
        where target_headway_s reflects the previous action)."""
        self._ego.target_headway_s = headway_s

    # ------------------------------------------------------------------

    def build(
        self,
        vehicles: list[TrackedVehicle],
        gps: GpsFix,
        t_mono: float,
        peers: list[PeerState] | None = None,
    ) -> ObservationResult:
        cfg = self.config
        peers = peers or []
        src: dict[str, str] = {}

        # --- ego motion from GPS -------------------------------------
        gps_age = gps.age_s(t_mono)
        # Bounded on both sides, and the two sides are not the same size.
        #
        # The past side is the staleness window: how old a reading may be and
        # still describe now. The future side is much tighter, because a reading
        # from this clock's future is nonsense -- and it is exactly what an
        # unconverted cross-device stamp produces whenever the peer booted later
        # than this device. A one-sided `<=` accepted those, so an arbitrarily
        # old fix read as current: the dangerous direction of the clock-mixing
        # failure, where the positive direction merely falls back to neutral.
        #
        # It is not zero, because two legitimate cases land slightly ahead: `now`
        # is often sampled just before the reading, and a converted stamp may sit
        # inside its own bound after the arrival it preceded. So the allowance is
        # that bound when the stamp carries one, and a small sampling epsilon
        # otherwise -- rather than the full staleness window, which would have
        # granted two seconds of future tolerance on the strength of an eight
        # millisecond uncertainty.
        stamp = getattr(gps, "timebase", None)
        future_allowance = cfg.clock_sampling_epsilon_s
        if stamp is not None and stamp.bound_s is not None:
            future_allowance = max(future_allowance, stamp.bound_s)
        gps_fresh = gps.valid and -future_allowance <= gps_age <= cfg.gps_stale_after_s
        if gps_fresh:
            ego_speed = max(0.0, gps.speed_mps) if math.isfinite(gps.speed_mps) else 0.0
            self._ego.last_speed_mps = ego_speed
            self._ego.ever_had_fix = True
            self._ego.speed_samples.append((t_mono, ego_speed))
            # A fix from another device had its capture stamp converted before
            # `gps_age` above could mean anything, so the freshness this branch
            # turns on is only as good as that conversion. Recorded as a distinct
            # provenance rather than folded into "measured": the value is
            # measured, the decision to trust it is not, and a reader of the
            # field-source table is exactly the person who needs to know which.
            src["ego_speed"] = _speed_provenance(gps)
        else:
            # hold last known speed rather than reporting 0 (= "stopped")
            ego_speed = self._ego.last_speed_mps if self._ego.ever_had_fix else 0.0
            src["ego_speed"] = "fallback_neutral"
        ego_accel = self._speed_slope()
        src["ego_acceleration"] = "derived" if len(self._ego.speed_samples) >= 3 else "fallback_neutral"

        # --- lane assignment from lateral offsets --------------------
        in_range = [v for v in vehicles if v.distance_m <= cfg.effective_range_m]
        lanes: dict[int, list[TrackedVehicle]] = {}
        for v in in_range:
            lanes.setdefault(self._lane_of(v), []).append(v)

        leader = min(lanes.get(0, []), key=lambda v: v.distance_m, default=None)
        left_front = min(lanes.get(-1, []), key=lambda v: v.distance_m, default=None)
        right_front = min(lanes.get(1, []), key=lambda v: v.distance_m, default=None)

        leader_gap = leader.distance_m if leader else INF
        leader_rel = (
            leader.rel_speed_mps if leader is not None and leader.rel_speed_valid else 0.0
        )
        src["leader_gap"] = "measured" if leader else "fallback_neutral"
        src["leader_relative_speed"] = (
            "measured" if leader is not None and leader.rel_speed_valid else "fallback_neutral"
        )
        src["left_lane_front_gap"] = "measured" if left_front else "fallback_neutral"
        src["right_lane_front_gap"] = "measured" if right_front else "fallback_neutral"

        # --- counts, density, speed statistics ------------------------
        n_forward = len(in_range)
        n_local = 2 * n_forward if cfg.symmetrize_counts else n_forward
        # sim formula: count / (2 * range_m / 1000)  over +-range_m
        density = n_local / max((2.0 * cfg.effective_range_m) / 1000.0, 1e-9)
        src["active_vehicle_count_local"] = "derived" if cfg.symmetrize_counts else "measured"

        abs_speeds = [
            max(0.0, ego_speed + v.rel_speed_mps) for v in in_range if v.rel_speed_valid
        ]
        mean_speed = float(np.mean(abs_speeds)) if abs_speeds else ego_speed
        src["local_mean_speed_bin"] = "derived" if abs_speeds else "fallback_neutral"
        queue_count = sum(1 for s in abs_speeds if s < cfg.queue_speed_mps)
        if cfg.symmetrize_counts:
            queue_count *= 2

        # --- cooperation / nearby AVs (V2V beacons, else neutral) ------
        if peers:
            av_count = len(peers)
            av_density = av_count / max((2.0 * 150.0) / 1000.0, 1e-9)  # radio range +-150 m, sim default
            av_mean_speed = float(np.mean([p.speed_mps for p in peers]))
            cooperation = {
                "segment_target_speed": av_mean_speed,
                "merge_pressure": 0.0,
                "downstream_congestion_estimate": 0.0,
            }
            lane_distribution = self._peer_lane_distribution(peers)
            src["nearby_av_count"] = "measured"
        else:
            av_count = 0
            av_density = 0.0
            av_mean_speed = cfg.free_flow_speed_mps
            cooperation = sim_contract.neutral_cooperation(cfg.free_flow_speed_mps)
            lane_distribution: dict[str, float] = {}
            src["nearby_av_count"] = "fallback_neutral"

        # --- etiquette flag (mirrors src/safety/etiquette.py) ----------
        uncongested_low_speed = bool(
            density < cfg.uncongested_density_threshold_veh_per_km
            and ego_speed < cfg.free_flow_speed_mps - cfg.low_speed_free_flow_delta_mps
        )

        ego_headway = (
            INF
            if not math.isfinite(leader_gap) or ego_speed <= 0
            else max(0.0, leader_gap / max(ego_speed, 1e-6))
        )

        obs: dict[str, Any] = {
            "is_active": True,
            "ego_speed": float(ego_speed),
            "ego_acceleration": float(ego_accel),
            "ego_lane": int(cfg.assumed_lane),
            "ego_headway_s": ego_headway,
            "target_headway_s": float(self._ego.target_headway_s),
            "time_since_last_lane_change": INF,   # sim start-state convention
            "lane_changes_last_km": 0,
            "current_segment": None,              # map matching not implemented
            "distance_to_next_merge": 0.0,        # sim parity: sim hardcodes 0.0
            "distance_to_downstream_bottleneck": INF,  # sim value off-bottleneck
            "leader_gap": float(leader_gap),
            "leader_relative_speed": float(leader_rel),
            "follower_gap": INF,                  # no rear sensing (yet)
            "follower_relative_speed": 0.0,
            "left_lane_front_gap": float(left_front.distance_m) if left_front else INF,
            "left_lane_rear_gap": INF,
            "right_lane_front_gap": float(right_front.distance_m) if right_front else INF,
            "right_lane_rear_gap": INF,
            # sim: target lane defaults to the current lane
            "target_lane_front_gap": float(leader_gap),
            "target_lane_rear_gap": INF,
            "target_lane_rear_required_decel": 0.0,
            "downstream_congestion_estimate": cooperation["downstream_congestion_estimate"],
            "merge_pressure": cooperation["merge_pressure"],
            "segment_target_speed": cooperation["segment_target_speed"],
            "uncongested_low_speed_flag": uncongested_low_speed,
            "local_density_bin": sim_contract.bin_index(density, cfg.density_bin_edges_veh_per_km),
            "local_mean_speed_bin": sim_contract.bin_index(mean_speed, cfg.mean_speed_bin_edges_mps),
            "local_queue_estimate": int(queue_count),
            "active_vehicle_count_local": int(n_local),
            "active_av_count_local": int(av_count),
            "nearby_av_count": int(av_count),
            "nearby_av_density": float(av_density if peers else 0.0),
            "nearby_av_mean_speed": float(av_mean_speed),
            "nearby_av_lane_distribution": lane_distribution,
            "sensor": {
                "range_m": float(cfg.effective_range_m),
                "latency_s": 0.0,
                "position_noise_std": 0.0,
                "speed_noise_std": 0.0,
            },
            "cooperation": cooperation,
        }

        defaults = {
            "is_active": "static_config",
            "ego_lane": "static_config",
            "ego_headway_s": "derived",
            "target_headway_s": "static_config",
            "time_since_last_lane_change": "fallback_neutral",
            "lane_changes_last_km": "fallback_neutral",
            "distance_to_next_merge": "sim_parity",
            "distance_to_downstream_bottleneck": "sim_parity",
            "follower_gap": "fallback_neutral",
            "follower_relative_speed": "fallback_neutral",
            "left_lane_rear_gap": "fallback_neutral",
            "right_lane_rear_gap": "fallback_neutral",
            "target_lane_front_gap": "derived",
            "target_lane_rear_gap": "fallback_neutral",
            "target_lane_rear_required_decel": "fallback_neutral",
            "downstream_congestion_estimate": "fallback_neutral" if not peers else "measured",
            "merge_pressure": "fallback_neutral",
            "segment_target_speed": "fallback_neutral" if not peers else "measured",
            "uncongested_low_speed_flag": "derived",
            "local_density_bin": "derived",
            "local_queue_estimate": "derived",
            "active_av_count_local": src["nearby_av_count"],
            "nearby_av_density": src["nearby_av_count"],
            "nearby_av_mean_speed": src["nearby_av_count"],
        }
        for key, value in defaults.items():
            src.setdefault(key, value)

        encoded = sim_contract.encode_local_observation(obs)
        fallback_fields = [k for k, v in src.items() if v == "fallback_neutral"]
        diagnostics = {
            "gps_valid": gps.valid,
            "gps_age_s": round(gps_age, 3) if math.isfinite(gps_age) else None,
            "gps_fresh": gps_fresh,
            # What the freshness decision rested on. None for a local fix.
            "gps_timebase": (
                None if getattr(gps, "timebase", None) is None else gps.timebase.to_record()
            ),
            "n_tracked": len(vehicles),
            "n_forward_in_range": n_forward,
            "leader_track_id": leader.track_id if leader else None,
            "leader_method": leader.method if leader else None,
            "density_veh_per_km": round(density, 2),
            "mean_speed_mps": round(mean_speed, 2),
            "missingness": round(len(fallback_fields) / max(len(src), 1), 3),
            "fallback_fields": fallback_fields,
        }
        return ObservationResult(obs=obs, encoded=encoded, field_sources=src, diagnostics=diagnostics)

    # ------------------------------------------------------------------

    def _lane_of(self, vehicle: TrackedVehicle) -> int:
        offset = vehicle.lateral_m / self.config.lane_width_m
        lane = int(round(offset))
        return max(-2, min(2, lane))

    def _speed_slope(self) -> float:
        samples = list(self._ego.speed_samples)[-10:]
        if len(samples) < 3:
            return 0.0
        t = np.array([s[0] for s in samples])
        v = np.array([s[1] for s in samples])
        if t[-1] - t[0] < 0.3:
            return 0.0
        t = t - t.mean()
        return float((t * (v - v.mean())).sum() / max((t * t).sum(), 1e-9))

    @staticmethod
    def _peer_lane_distribution(peers: list[PeerState]) -> dict[str, float]:
        with_lane = [p for p in peers if p.lane_id is not None]
        if not with_lane:
            return {}
        counts: dict[str, int] = {}
        for p in with_lane:
            key = str(p.lane_id)
            counts[key] = counts.get(key, 0) + 1
        return {k: c / len(with_lane) for k, c in counts.items()}
