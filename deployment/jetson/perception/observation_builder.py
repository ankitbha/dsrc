"""Build the simulation actor's observation dict from real sensors.

This is the sim-to-real alignment core. Every one of the 39 slots the actor
reads (`sim_contract.encoded_slot_names()`) is produced here, each tagged
with a provenance class in ``field_sources``. The vocabulary itself lives in
`perception.provenance`, not here: `ego_speed`, `ego_acceleration` and
`local_density_bin` below are three of eleven classes a field can carry, and
`provenance.SOURCES` is the closed list.

The provenance map is logged every tick and is the basis for the paper's
"observation missingness" metric, and (since this module's own field-level
fixes) for the sensing controller's free-tier event rule -- a substituted
`ego_acceleration` no longer reads as a calm road.

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
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception import feed_fusion, provenance
from perception.distance import TrackedVehicle
from policy import sim_contract
from sensors.gps_reader import GpsFix
from sensors.here_feed import FlowReading

INF = float("inf")


@dataclass
class ObservationResult:
    obs: dict[str, Any]                # sim-schema observation dict
    encoded: np.ndarray                # (39,) float32 actor input
    field_sources: dict[str, str]      # provenance per field
    diagnostics: dict[str, Any]        # raw values for logging/eval
    #: What the traffic feed offered this tick. Deliberately beside the vector
    #: rather than in it -- see the note in `build`. The sensing controller reads
    #: this; the policy does not.
    feed: "feed_fusion.FeedOwnership | None" = None


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
    #: The range peers are admitted at, which `nearby_av_density` divides by. Must
    #: match `v2v.range_m` -- `run_demo` routes that to `BeaconTransceiver` for
    #: admission and this is the other half of the same number.
    peer_range_m: float = 150.0
    low_speed_free_flow_delta_mps: float = 8.0
    gps_stale_after_s: float = 2.0
    # How far into this clock's future a reading may sit and still be believed,
    # when it carries no uncertainty of its own. Covers `now` being sampled just
    # before the reading; anything larger is a clock problem, not sampling order.
    clock_sampling_epsilon_s: float = 0.05
    # A cross-device stamp whose uncertainty exceeds this share of the staleness
    # window cannot answer the freshness question, so it is refused rather than
    # given that much benefit of the doubt.
    max_bound_fraction: float = 0.5

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
    #: `t_mono` of the last tick whose `in_range` was non-empty. None until
    #: the first in-range detection this builder has ever seen.
    last_in_range_at: float | None = None


def _speed_provenance(gps: GpsFix) -> str:
    """`measured` for a local fix; for a remote one, how its stamp was obtained.

    Three outcomes rather than two, because "measured" would hide the difference
    between a reading whose freshness was established exactly and one where it
    rested on a converted stamp or on an arrival-time proxy.
    """
    stamp = getattr(gps, "timebase", None)
    if stamp is None:
        return provenance.SOURCE_MEASURED
    if stamp.proxy:
        return provenance.SOURCE_MEASURED_ARRIVAL_PROXY
    return provenance.SOURCE_MEASURED_CONVERTED


class ObservationBuilder:
    def __init__(self, config: BuilderConfig) -> None:
        self.config = config
        self._ego = _EgoState(target_headway_s=config.target_headway_default_s)
        # Set before the first build, so a reader does not have to guard for an
        # attribute that only exists after a tick has run.
        self.last_feed_ownership = feed_fusion.own(None)
        #: How long the last `build()` spent on each named sub-segment, keyed by
        #: name and in milliseconds. Same precedent as
        #: `TrtYoloDetector.last_timings`: a plain dict the caller reads after
        #: the call, rather than a return value every caller would otherwise
        #: have to thread through. Empty before the first build -- a caller
        #: reading `last_timings["fuse_ms"]` here is reading a builder nothing
        #: has run yet, and that is a missing value, not a zero-length fuse.
        self.last_timings: dict[str, float] = {}

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
        feed: FlowReading | None = None,
    ) -> ObservationResult:
        cfg = self.config
        peers = peers or []
        src: dict[str, str] = {}

        # --- ego motion from GPS -------------------------------------
        gps_age = gps.age_s(t_mono)
        stamp = getattr(gps, "timebase", None)
        bound_s = None if stamp is None else stamp.bound_s
        # Conservative on both sides, and capped.
        #
        # The past side charges the bound: a reading 1.9 s old with a 0.4 s
        # uncertainty may really be 2.3 s old, and calling that fresh inside a
        # 2 s window is answering "possibly" as "certainly". The future side
        # allows the bound, because a converted stamp may legitimately land after
        # the arrival it preceded -- plus a sampling epsilon, because `now` is
        # often read just before the reading.
        #
        # And a bound wider than half the window means the timebase cannot
        # resolve the question at all. Granting it that much *future* tolerance
        # would have been more slack than the whole past window: at a 10 s bound
        # a stamp nine seconds into this clock's future read as measured. So it
        # is refused with its own provenance, which says the timebase could not
        # answer rather than pretending it did.
        uncertainty_s = 0.0 if bound_s is None else bound_s
        timebase_unresolved = uncertainty_s > cfg.gps_stale_after_s * cfg.max_bound_fraction
        # No separate cap on the allowance. A bound large enough for one to bind
        # is already past the unresolved threshold above -- half the window --
        # so a cap would be unreachable code, and this project has enough of
        # those to know they rot. The single guard is the one that fires.
        future_allowance = max(cfg.clock_sampling_epsilon_s, uncertainty_s)
        gps_fresh = (
            gps.valid
            and not timebase_unresolved
            and -future_allowance <= gps_age
            and gps_age + uncertainty_s <= cfg.gps_stale_after_s
        )
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
            src["ego_speed"] = provenance.SOURCE_FALLBACK_NEUTRAL
            # The window's invariant: it holds only samples from an unbroken run
            # of fresh fixes. Without this, a gap in the middle of the window is
            # invisible to the span check below (which only sees first-to-last),
            # so the slope fitted across the gap keeps being reported as
            # `derived` once GPS resumes -- a real hole in the data, reported as
            # a real measurement. Clearing here means acceleration is
            # unavailable for a further ~0.3 s after any dropout ends, while the
            # window refills from the first fresh sample back up to three.
            self._ego.speed_samples.clear()
        # From the branch actually taken, not from the sample count -- the count
        # cannot see the window-span guard below it, and passing this tick's
        # own freshness verdict lets the branch also refuse a window whose fix
        # has gone stale even though a sample was appended for it on every
        # tick it was still (nominally) fresh, which the sample count cannot
        # see either.
        ego_accel, accel_derived = self._speed_slope(gps_fresh)
        src["ego_acceleration"] = (
            provenance.SOURCE_DERIVED if accel_derived else provenance.SOURCE_FALLBACK_NEUTRAL
        )

        # --- lane assignment from lateral offsets --------------------
        in_range = [v for v in vehicles if v.distance_m <= cfg.effective_range_m]
        if in_range:
            # Recorded before anything below can refuse or shortcut, so a
            # tick that has a detection always advances this -- the last
            # instant the perception chain produced a track, independent of
            # what the density/count/queue formulas do with it afterward.
            self._ego.last_in_range_at = t_mono
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
        src["leader_gap"] = (
            provenance.SOURCE_MEASURED if leader else provenance.SOURCE_FALLBACK_NEUTRAL
        )
        src["leader_relative_speed"] = (
            provenance.SOURCE_MEASURED
            if leader is not None and leader.rel_speed_valid
            else provenance.SOURCE_FALLBACK_NEUTRAL
        )
        src["left_lane_front_gap"] = (
            provenance.SOURCE_MEASURED if left_front else provenance.SOURCE_FALLBACK_NEUTRAL
        )
        src["right_lane_front_gap"] = (
            provenance.SOURCE_MEASURED if right_front else provenance.SOURCE_FALLBACK_NEUTRAL
        )

        # --- counts, density, speed statistics ------------------------
        n_forward = len(in_range)
        n_local = 2 * n_forward if cfg.symmetrize_counts else n_forward
        # sim formula: count / (2 * range_m / 1000)  over +-range_m
        density = n_local / max((2.0 * cfg.effective_range_m) / 1000.0, 1e-9)
        # Zero in-range tracks makes both of these `derived_empty`: the count
        # really is zero, nothing was substituted for a measurement that
        # failed, so calling it a plain `derived` would say the same thing
        # about "nothing to count" and "something we could not count".
        if n_forward == 0:
            src["active_vehicle_count_local"] = provenance.SOURCE_DERIVED_EMPTY
            src["local_density_bin"] = provenance.SOURCE_DERIVED_EMPTY
        else:
            src["active_vehicle_count_local"] = (
                provenance.SOURCE_DERIVED if cfg.symmetrize_counts else provenance.SOURCE_MEASURED
            )
            src["local_density_bin"] = provenance.SOURCE_DERIVED

        # Only vehicles whose relative speed the tracker could measure have a usable
        # absolute speed, so this population is a subset of `in_range` -- where the
        # simulator counts the queue and the vehicle count over ONE population
        # (`src/sensing/local.py:181-188`: `measured` feeds both). The edge cannot
        # close that gap by inventing a speed for a track it has not measured; what it
        # can do is stop reporting the shortfall as a measurement.
        abs_speeds = [
            max(0.0, ego_speed + v.rel_speed_mps) for v in in_range if v.rel_speed_valid
        ]
        measured_fraction = len(abs_speeds) / len(in_range) if in_range else 1.0
        mean_speed = float(np.mean(abs_speeds)) if abs_speeds else ego_speed
        src["local_mean_speed_bin"] = (
            provenance.SOURCE_DERIVED if abs_speeds else provenance.SOURCE_FALLBACK_NEUTRAL
        )
        queue_count = sum(1 for s in abs_speeds if s < cfg.queue_speed_mps)
        if cfg.symmetrize_counts:
            queue_count *= 2

        # --- cooperation / nearby AVs, and the traffic feed: "fuse" --------
        #
        # Timed together as one sub-segment because both are the same job seen
        # from two sources: folding a reading this vehicle cannot itself
        # measure -- another AV's beacon, a traffic service's estimate -- into
        # what this tick knows, before the vector is assembled. Precedent:
        # `TrtYoloDetector.last_timings`, a plain dict read after the call
        # rather than a return value every caller would have to thread through.
        fuse_started = time.monotonic()
        if peers:
            av_count = len(peers)
            # The admission range peers were actually accepted at, not a literal.
            # `config.yaml`'s `v2v.range_m` reaches `BeaconTransceiver`, which admits
            # peers out to it, and nothing carried it here -- so raising it to 300 m
            # left the density computed over 150 and reporting twice the vehicles per
            # km the admission range implies.
            av_density = av_count / max((2.0 * cfg.peer_range_m) / 1000.0, 1e-9)
            av_mean_speed = float(np.mean([p.speed_mps for p in peers]))
            cooperation = {
                "segment_target_speed": av_mean_speed,
                "merge_pressure": 0.0,
                "downstream_congestion_estimate": 0.0,
            }
            lane_distribution = self._peer_lane_distribution(peers)
            src["nearby_av_count"] = provenance.SOURCE_MEASURED
        else:
            av_count = 0
            av_density = 0.0
            av_mean_speed = cfg.free_flow_speed_mps
            cooperation = sim_contract.neutral_cooperation(cfg.free_flow_speed_mps)
            lane_distribution: dict[str, float] = {}
            src["nearby_av_count"] = provenance.SOURCE_FALLBACK_NEUTRAL
        # `_peer_lane_distribution` returns {} both when there are no peers and
        # when none of them carry a `lane_id` -- either way the three encoded
        # lane-share slots are neutral rather than computed from a measured
        # peer.
        lane_source = (
            provenance.SOURCE_DERIVED if lane_distribution else provenance.SOURCE_FALLBACK_NEUTRAL
        )
        for lane in sim_contract.LANE_DISTRIBUTION_LANES:
            src[f"nearby_av_lane_distribution.{lane}"] = lane_source

        # --- the traffic feed: derived, recorded, and NOT in the vector -----
        #
        # It owns no observation field, which is the opposite of what this task
        # set out to do and is what the evidence supports.
        #
        # The simulator's `if not local_av:` at `src/sensing/local.py:203` is a
        # BLOCK gate, not a congestion gate: with no AVs near it pins
        # downstream_congestion, merge_pressure and segment_target_speed together,
        # while density goes to zero, lane distribution empties and both AV counts
        # go to zero. So `congestion > 0` implies `nearby_av_count >= 1` in every
        # observation the sim can emit -- measured at 0 of 1,095 rollout samples in
        # the other cell, against 42 with congestion and 1..7 AVs.
        #
        # A lone instrumented car has no equipped neighbours, so `peers` is empty on
        # every tick of a real drive. Writing the feed's congestion here would put
        # the policy in that empty cell not occasionally but always -- one field
        # lifted out of a neutral block whose other five stay pinned.
        #
        # Owning it only when peers exist would make the feed fire essentially
        # never. Making the vector legitimately feed-informed needs the simulator's
        # sensing model to produce congestion without AVs, which is a change to the
        # training side and outside section F. Until then the reading is published
        # on the result for the sensing controller and written to the record, where
        # it informs decisions without becoming an input the policy never saw.
        owned = feed_fusion.own(feed)
        self.last_feed_ownership = owned
        self.last_timings["fuse_ms"] = (time.monotonic() - fuse_started) * 1000.0

        # --- etiquette flag (mirrors src/safety/etiquette.py) ----------
        # `density` here is the locally sensed one; the simulator uses the segment
        # density it gets from `segment_metrics` (`src/sensing/local.py:214-221`).
        # Different quantities, same threshold -- see the provenance below.
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
            "is_active": provenance.SOURCE_STATIC_CONFIG,
            "ego_lane": provenance.SOURCE_STATIC_CONFIG,
            "ego_headway_s": provenance.SOURCE_DERIVED,
            "target_headway_s": provenance.SOURCE_STATIC_CONFIG,
            "time_since_last_lane_change": provenance.SOURCE_FALLBACK_NEUTRAL,
            "lane_changes_last_km": provenance.SOURCE_FALLBACK_NEUTRAL,
            "distance_to_next_merge": provenance.SOURCE_SIM_PARITY,
            "distance_to_downstream_bottleneck": provenance.SOURCE_SIM_PARITY,
            "follower_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "follower_relative_speed": provenance.SOURCE_FALLBACK_NEUTRAL,
            "left_lane_rear_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "right_lane_rear_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "target_lane_front_gap": provenance.SOURCE_DERIVED,
            "target_lane_rear_gap": provenance.SOURCE_FALLBACK_NEUTRAL,
            "target_lane_rear_required_decel": provenance.SOURCE_FALLBACK_NEUTRAL,
            "downstream_congestion_estimate": (
                provenance.SOURCE_FALLBACK_NEUTRAL if not peers else provenance.SOURCE_MEASURED
            ),
            "merge_pressure": provenance.SOURCE_FALLBACK_NEUTRAL,
            "segment_target_speed": (
                provenance.SOURCE_FALLBACK_NEUTRAL if not peers else provenance.SOURCE_MEASURED
            ),
            # `approximated`, not `derived`. The formula mirrors
            # `src/safety/etiquette.py`, but the simulator feeds it a SEGMENT density
            # from `segment_metrics` and this feeds it the locally sensed +-range
            # density -- the same 12.0 threshold applied to a different quantity. A
            # single instrumented car has no segment-level view, so the edge cannot
            # produce the sim's input; substituting one that moves differently is the
            # unit substitution task 28 retracted two fields for. Marked rather than
            # silently equated, so the missingness metric and anyone reading the
            # vector can see which it is.
            "uncongested_low_speed_flag": provenance.SOURCE_APPROXIMATED,
            # `local_density_bin` and `active_vehicle_count_local` are set above,
            # beside `n_forward`, because they need it. `local_queue_estimate`
            # reads the same population (`in_range`, `abs_speeds`) but has a
            # third class no other field here needs: an absence of tracks is
            # `derived_empty`, tracks present but none with a measurable speed
            # is `fallback_neutral` -- the distinction :415-420 exists for --
            # and a measurable one is `derived`.
            "local_queue_estimate": (
                provenance.SOURCE_DERIVED if abs_speeds
                else provenance.SOURCE_DERIVED_EMPTY if not in_range
                else provenance.SOURCE_FALLBACK_NEUTRAL
            ),
            "active_av_count_local": src["nearby_av_count"],
            "nearby_av_density": src["nearby_av_count"],
            "nearby_av_mean_speed": src["nearby_av_count"],
        }
        for key, value in defaults.items():
            src.setdefault(key, value)
        # The nested `cooperation` block is the same three values as the flat
        # fields above it, read through a different key -- so their class is
        # the flat field's class, not a fresh judgement.
        src["cooperation.segment_target_speed"] = src["segment_target_speed"]
        src["cooperation.merge_pressure"] = src["merge_pressure"]
        src["cooperation.downstream_congestion_estimate"] = src["downstream_congestion_estimate"]

        encoded = sim_contract.encode_local_observation(obs)
        prov = provenance.summarise(src)
        if in_range:
            last_detection_age_s = 0.0
        elif self._ego.last_in_range_at is None:
            last_detection_age_s = None
        else:
            last_detection_age_s = round(t_mono - self._ego.last_in_range_at, 3)
        diagnostics = {
            "gps_valid": gps.valid,
            "gps_age_s": round(gps_age, 3) if math.isfinite(gps_age) else None,
            "gps_fresh": gps_fresh,
            # True when the stamp's own uncertainty was too wide to decide
            # freshness at all -- distinct from a stale fix and from a missing one.
            "gps_timebase_unresolved": timebase_unresolved,
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
            "missingness": prov["missingness"],
            # So a drive where the feed never owned a field says why, rather
            # than the congestion column being quietly neutral throughout.
            "feed": feed_fusion.to_record(owned),
            "fallback_fields": prov["fallback_fields"],
            "provenance": {
                "fields": prov["fields"],
                "by_source": prov["by_source"],
                "covers_encoder": self._covers_encoder(src),
            },
            # How long since the perception chain last produced an in-range
            # track -- the only bound available on whether an empty
            # `local_density_bin` is an empty road or a blind camera. None
            # until the first in-range detection this builder has ever seen;
            # a measured 0.0 on a tick that has one, never a substituted zero.
            "last_detection_age_s": last_detection_age_s,
        }
        return ObservationResult(obs=obs, encoded=encoded, field_sources=src,
                                 diagnostics=diagnostics, feed=owned)

    # ------------------------------------------------------------------

    def _lane_of(self, vehicle: TrackedVehicle) -> int:
        offset = vehicle.lateral_m / self.config.lane_width_m
        lane = int(round(offset))
        return max(-2, min(2, lane))

    def _speed_slope(self, gps_fresh: bool) -> tuple[float, bool]:
        """The ego acceleration and whether it was actually derived.

        Two ways to fall back and they used to be reported as one: the caller set
        the provenance from the sample COUNT alone, so a window too short to fit a
        slope returned the neutral 0.0 tagged `derived`. At the shipped 30 fps the
        ten-sample slice spans exactly 9/30 = 0.3 s, landing on the guard, so which
        branch ran was decided by frame-timing noise -- measured at 533 of 888 ticks
        under a constant -3.0 m/s^2 deceleration.

        That matters twice over. `SensingController` raises rates on
        `abs(ego_acceleration) >= EVENT_ACCEL_MPS2`, so the free tier -- the thing
        that says when to spend the expensive modalities -- was silent on more than
        half the ticks of a braking event. And `field_sources` is what this module's
        docstring calls the basis for the observation-missingness metric, so the
        missingness was under-counted by the same margin.

        A third way to fall back: a sample is appended only under a fresh GPS fix,
        but "fresh" describes the fix, not the window -- a receiver that has
        stopped producing new readings and keeps returning the last one it had is
        still fresh by that test for as long as its own age stays inside
        `gps_stale_after_s`, so a sample keeps being appended, stamped with this
        tick's own clock, every tick that receiver is silently dead. Measuring
        staleness from the newest sample's own timestamp let a window built that
        way look freshly appended for a further `gps_stale_after_s` after the fix
        behind it had already gone stale -- double the delay the bound is supposed
        to be. Taking `gps_fresh` directly, the same verdict `ego_speed` was
        already built from, makes the two fields go stale on the same tick,
        whatever caused it.
        """
        samples = list(self._ego.speed_samples)[-10:]
        if len(samples) < 3:
            return 0.0, False
        t = np.array([s[0] for s in samples])
        v = np.array([s[1] for s in samples])
        if t[-1] - t[0] < 0.3:
            return 0.0, False
        if not gps_fresh:
            return 0.0, False
        t = t - t.mean()
        return float((t * (v - v.mean())).sum() / max((t * t).sum(), 1e-9)), True

    @staticmethod
    def _covers_encoder(field_sources: dict[str, str]) -> bool:
        """Whether `field_sources` tags every slot the encoder reads, by NAME
        rather than by count -- a map with the right number of keys but the
        wrong ones (one encoder slot missing, one name the encoder never
        reads standing in for it) is not coverage, and a count comparison
        cannot tell the two apart.
        """
        return set(field_sources) == set(sim_contract.encoded_slot_names())

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
