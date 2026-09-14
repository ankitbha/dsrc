"""What one tick measures about the road, for the readers that read it.

This used to build the 39-field observation the local-sensing actor was
trained on, and encode it into that actor's input vector. The actor is gone.
What is left is a measurement, not a model input, so it carries what is
actually read and nothing else:

  `ego_speed`, `leader_gap`, `leader_relative_speed` and `local_density_bin`
  are the safety gate's four evidence-required inputs;
  `ego_acceleration`, `ego_speed` and `local_density_bin` are the sensing
  scheduler's three; `segment_target_speed` is the gate's one configured
  input; `active_vehicle_count_local` is on the dashboard.

Seven fields, six producers -- `ego_speed` and `local_density_bin` are each
read by three of the four consumers. The other thirty-two were the sim's
"empty road" constants for sensors this rig does not have, and a constant
carried into a decision that never compares it is indistinguishable, in the
record, from a measurement that did not matter.

Each field is tagged with a provenance class in ``field_sources``. The
vocabulary lives in `perception.provenance`, not here, and
`provenance.SOURCES` is the closed list. The provenance map is logged every
tick and is the basis for the paper's "observation missingness" metric and
for the sensing controller's free-tier event rule -- a substituted
`ego_acceleration` does not read as a calm road.

Key geometry conventions (right-hand traffic, camera ~lane-centered):
  lateral_m > 0 is right of the camera axis; lane assignment is
  round(lateral / lane_width): 0 = ego lane, -1 = left, +1 = right.

Known v0 gaps (documented in ARCHITECTURE.md with upgrade paths):
  - forward-only counts -> density uses symmetric extrapolation
    (2 x forward count over +-range), toggleable via symmetrize_counts.
  - no rear sensing and no lane detection, which is why the gate's rear and
    lane rules were removed rather than fed constants.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from perception import feed_fusion, provenance
from perception.distance import TrackedVehicle
from sensors.gps_reader import GpsFix
from sensors.here_feed import FlowReading

INF = float("inf")

#: Every key `obs` carries, and the one place the set is written down. Used by
#: the coverage check below, which compares `field_sources` against it BY NAME:
#: a map with the right number of keys but the wrong ones is not coverage, and
#: a count comparison cannot tell the two apart.
OBS_FIELDS: tuple[str, ...] = (
    "ego_speed",
    "ego_acceleration",
    "leader_gap",
    "leader_relative_speed",
    "local_density_bin",
    "active_vehicle_count_local",
    "segment_target_speed",
)


def bin_index(value: float, edges: tuple[float, ...] | list[float]) -> int:
    """How many of `edges` the value is at or above.

    Was `policy.sim_contract.bin_index`, vendored from the simulator's own
    binning. That module is gone with the actor it served, and this is the one
    place left that bins anything, so it lives here rather than in a module of
    its own.
    """
    index = 0
    for edge in edges:
        if value >= edge:
            index += 1
    return index


@dataclass
class ObservationResult:
    obs: dict[str, Any]                # the seven measured fields, by name
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
    #: What `segment_target_speed` reports when no cooperating peer is heard,
    #: which on a lone instrumented car is every tick.
    free_flow_speed_mps: float = 30.0
    lane_width_m: float = 3.7
    density_bin_edges_veh_per_km: tuple[float, ...] = (12.0, 30.0)
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

    @classmethod
    def from_full_config(cls, config: Mapping[str, Any]) -> "BuilderConfig":
        """The one construction from a whole `config.yaml`-shaped dict:
        `config["observation"]` plus two cross-section overrides this
        builder cannot get from its own section alone.

        `gps_stale_after_s` comes from `config["gps"]["stale_after_s"]`. The
        `v2v.range_m` override went with `nearby_av_density`, the only field
        that divided by it.

        Extracted (B11, validation round 2) so `run_demo.build_components`
        and `src.analysis.observation_parity._production_builder_config`
        call the SAME merge rather than each carrying its own copy: two
        copies agree only until one of them gains an override the other does
        not, and a test built by re-typing the same lines independently would
        not catch that either -- it would only ever agree with whichever copy
        it was transcribed from.

        `config["observation"]` still carries keys this builder no longer
        has (`assumed_lane` is read by `run_demo` for the V2V beacon;
        the bin edges and thresholds the removed fields used are left in the
        file rather than silently dropped). `from_dict` takes only the names
        it declares, so an unknown key is ignored rather than raising.
        """
        obs_cfg = dict(config["observation"])
        obs_cfg["gps_stale_after_s"] = config["gps"]["stale_after_s"]
        return cls.from_dict(obs_cfg)


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
        self._ego = _EgoState()
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
            # what the density and count formulas do with it afterward.
            self._ego.last_in_range_at = t_mono
        lanes: dict[int, list[TrackedVehicle]] = {}
        for v in in_range:
            lanes.setdefault(self._lane_of(v), []).append(v)

        leader = min(lanes.get(0, []), key=lambda v: v.distance_m, default=None)

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
        # --- counts and density ---------------------------------------
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

        # --- the cooperation reading, and the traffic feed: "fuse" --------
        #
        # Timed together as one sub-segment because both are the same job seen
        # from two sources: folding a reading this vehicle cannot itself
        # measure -- another AV's beacon, a traffic service's estimate -- into
        # what this tick knows. Precedent: `TrtYoloDetector.last_timings`, a
        # plain dict read after the call rather than a return value every
        # caller would have to thread through.
        fuse_started = time.monotonic()
        if peers:
            segment_target_speed = float(np.mean([p.speed_mps for p in peers]))
            src["segment_target_speed"] = provenance.SOURCE_DERIVED
        else:
            segment_target_speed = cfg.free_flow_speed_mps
            src["segment_target_speed"] = provenance.SOURCE_FALLBACK_NEUTRAL

        # --- the traffic feed: derived, recorded, and NOT a field ----------
        #
        # It owns no observation field. `feed_fusion.own` is run for the
        # reading's own record and for the sensing controller, which reads
        # `ObservationResult.feed`; the safety gate does not.
        owned = feed_fusion.own(feed)
        self.last_feed_ownership = owned
        self.last_timings["fuse_ms"] = (time.monotonic() - fuse_started) * 1000.0

        obs: dict[str, Any] = {
            "ego_speed": float(ego_speed),
            "ego_acceleration": float(ego_accel),
            "leader_gap": float(leader_gap),
            "leader_relative_speed": float(leader_rel),
            "local_density_bin": bin_index(density, cfg.density_bin_edges_veh_per_km),
            "active_vehicle_count_local": int(n_local),
            "segment_target_speed": float(segment_target_speed),
        }

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
            "missingness": prov["missingness"],
            # So a drive where the feed never owned a field says why, rather
            # than the congestion column being quietly neutral throughout.
            "feed": feed_fusion.to_record(owned),
            "fallback_fields": prov["fallback_fields"],
            "provenance": {
                "fields": prov["fields"],
                "by_source": prov["by_source"],
                "covers_obs": self._covers_obs(src),
            },
            # How long since the perception chain last produced an in-range
            # track -- the only bound available on whether an empty
            # `local_density_bin` is an empty road or a blind camera. None
            # until the first in-range detection this builder has ever seen;
            # a measured 0.0 on a tick that has one, never a substituted zero.
            "last_detection_age_s": last_detection_age_s,
        }
        return ObservationResult(obs=obs, field_sources=src,
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
    def _covers_obs(field_sources: dict[str, str]) -> bool:
        """Whether `field_sources` tags every field `obs` carries, by NAME
        rather than by count -- a map with the right number of keys but the
        wrong ones (one field missing, one name `obs` does not carry standing
        in for it) is not coverage, and a count comparison cannot tell the two
        apart.
        """
        return set(field_sources) == set(OBS_FIELDS)
