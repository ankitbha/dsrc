"""Per-tick orchestration: frame + GPS -> detections -> tracks -> distances
-> observation -> actor -> advisory, with stage-level latency accounting.

This is the single place where the dataflow is wired; run_demo.py,
replay_demo.py and bench_latency.py all drive this same object so live,
replay and bench numbers are directly comparable.

End-to-end latency (e2e_ms) is measured from the camera capture
timestamp to advisory readiness - it includes time the frame spent
waiting for the pipeline, not just compute.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.detector import Detection
from perception.distance import DistanceEstimator, TrackedVehicle
from perception.observation_builder import ObservationBuilder, ObservationResult, PeerState
from perception.tracker import IouTracker
from policy.actor_runtime import ActorRuntime, PolicyOutput
from policy.advisory import Advisory, AdvisoryDecoder
from sensors.camera_stream import Frame
from sensors.gps_reader import GpsFix


class RollingStats:
    def __init__(self, window: int = 300) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    def summary(self) -> dict[str, float]:
        if not self._values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        arr = np.asarray(self._values)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }


@dataclass
class Tick:
    tick_id: int
    frame_id: int
    t_capture_mono: float
    t_capture_wall: float
    stage_ms: dict[str, float]
    # e2e_ms is capture to advisory. When the capture happened on another device
    # it is the sum of a bounded link segment and an exact on-Jetson one, and the
    # two are reported apart because they fail differently: a reader cannot tell a
    # slow link from a slow Jetson in one number, and the deployment claim is
    # about the Jetson. link_ms is None for a local camera, where there is no link.
    e2e_ms: float
    # Required, not defaulted: a default of 0.0 would let a Tick built without it
    # report zero latency as though it had been measured.
    jetson_ms: float
    fps: float
    n_detections: int
    vehicles: list[TrackedVehicle]
    obs_result: ObservationResult
    policy: PolicyOutput
    advisory: Advisory
    gps: GpsFix
    n_peers: int = 0
    # None for a local camera, where there is no link and nothing was converted.
    link_ms: float | None = None
    timebase: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """JSON-able log record (uses Python JSON's Infinity literal for inf)."""
        return {
            "type": "tick",
            "tick_id": self.tick_id,
            "frame_id": self.frame_id,
            "t_wall": self.t_capture_wall,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "e2e_ms": round(self.e2e_ms, 2),
            "jetson_ms": round(self.jetson_ms, 2),
            "link_ms": None if self.link_ms is None else round(self.link_ms, 2),
            "timebase": self.timebase,
            "fps": round(self.fps, 2),
            "n_detections": self.n_detections,
            "vehicles": [
                {
                    "id": v.track_id,
                    "cls": v.cls,
                    "conf": round(v.conf, 3),
                    "dist_m": round(v.distance_m, 2),
                    "lat_m": round(v.lateral_m, 2),
                    "rel_mps": round(v.rel_speed_mps, 2) if v.rel_speed_valid else None,
                    "method": v.method,
                    "bbox": [int(x) for x in v.xyxy],
                }
                for v in self.vehicles
            ],
            "obs": self.obs_result.obs,
            "encoded": [round(float(x), 5) for x in self.obs_result.encoded],
            "field_sources": self.obs_result.field_sources,
            "obs_diagnostics": self.obs_result.diagnostics,
            "action": self.policy.action,
            "head_probs": self.policy.head_probs,
            "confidence": round(self.policy.confidence, 3),
            "advisory": {
                "recommended_speed_mps": round(self.advisory.recommended_speed_mps, 2),
                "recommended_speed_display": round(self.advisory.recommended_speed_display, 1),
                "units": self.advisory.units,
                "headway_target_s": self.advisory.headway_target_s,
                "lane_text": self.advisory.lane_text,
                "merge_text": self.advisory.merge_text,
                "confidence_label": self.advisory.confidence_label,
            },
            "gps": {
                "valid": self.gps.valid,
                "lat": self.gps.lat if math.isfinite(self.gps.lat) else None,
                "lon": self.gps.lon if math.isfinite(self.gps.lon) else None,
                "speed_mps": round(self.gps.speed_mps, 2)
                if math.isfinite(self.gps.speed_mps)
                else None,
                "heading_deg": round(self.gps.heading_deg, 1)
                if math.isfinite(self.gps.heading_deg)
                else None,
                "num_sats": self.gps.num_sats,
                "hdop": self.gps.hdop if math.isfinite(self.gps.hdop) else None,
            },
            "n_peers": self.n_peers,
        }


@dataclass
class PipelineStats:
    e2e: RollingStats = field(default_factory=RollingStats)
    jetson: RollingStats = field(default_factory=RollingStats)
    link: RollingStats = field(default_factory=RollingStats)
    detect: RollingStats = field(default_factory=RollingStats)
    track: RollingStats = field(default_factory=RollingStats)
    observe: RollingStats = field(default_factory=RollingStats)
    policy: RollingStats = field(default_factory=RollingStats)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "e2e_ms": self.e2e.summary(),
            "jetson_ms": self.jetson.summary(),
            "link_ms": self.link.summary(),
            "detect_ms": self.detect.summary(),
            "track_ms": self.track.summary(),
            "observe_ms": self.observe.summary(),
            "policy_ms": self.policy.summary(),
        }


class PerceptionPolicyPipeline:
    def __init__(
        self,
        detector,
        tracker: IouTracker,
        distance: DistanceEstimator,
        builder: ObservationBuilder,
        actor: ActorRuntime,
        advisory_decoder: AdvisoryDecoder,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.distance = distance
        self.builder = builder
        self.actor = actor
        self.advisory_decoder = advisory_decoder
        self.stats = PipelineStats()
        self._tick_counter = 0
        self._last_step_mono: float | None = None
        self._fps_ema = 0.0

    def step(
        self,
        frame: Frame,
        gps: GpsFix,
        peers: list[PeerState] | None = None,
        detections_override: list[Detection] | None = None,
        run_detector_with_override: bool = False,
    ) -> Tick:
        t0 = time.monotonic()
        if detections_override is None:
            detections = self.detector.infer(frame.image)
        else:
            if run_detector_with_override:
                # benchmark mode: keep real detector timing in the stats while
                # the scripted detections drive the downstream stages
                self.detector.infer(frame.image)
            detections = detections_override
        t1 = time.monotonic()

        tracks = self.tracker.update(detections, frame.t_mono)
        vehicles = self.distance.update(tracks, frame.t_mono)
        t2 = time.monotonic()

        obs_result: ObservationResult = self.builder.build(
            vehicles, gps, time.monotonic(), peers
        )
        t3 = time.monotonic()

        policy_out: PolicyOutput = self.actor.act(obs_result.encoded)
        advisory: Advisory = self.advisory_decoder.decode(policy_out, obs_result.obs)
        self.builder.set_target_headway(advisory.headway_target_s)
        t4 = time.monotonic()

        # Arrival is exact and local; capture may be converted from a peer clock.
        # With no timebase stamp the frame was captured here, so the two coincide.
        t_arrival = frame.timebase.t_arrival_mono if frame.timebase else frame.t_mono
        e2e_ms = (t4 - frame.t_mono) * 1000.0
        jetson_ms = (t4 - t_arrival) * 1000.0
        link_ms = None if frame.timebase is None else frame.timebase.link_s * 1000.0
        stage_ms = {
            "detect": (t1 - t0) * 1000.0,
            "track_distance": (t2 - t1) * 1000.0,
            "observe": (t3 - t2) * 1000.0,
            "policy_advisory": (t4 - t3) * 1000.0,
            "capture_to_start": (t0 - frame.t_mono) * 1000.0,
        }
        self.stats.e2e.add(e2e_ms)
        self.stats.jetson.add(jetson_ms)
        if link_ms is not None:
            self.stats.link.add(link_ms)
        self.stats.detect.add(stage_ms["detect"])
        self.stats.track.add(stage_ms["track_distance"])
        self.stats.observe.add(stage_ms["observe"])
        self.stats.policy.add(stage_ms["policy_advisory"])

        if self._last_step_mono is not None:
            dt = t4 - self._last_step_mono
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst
        self._last_step_mono = t4

        tick = Tick(
            tick_id=self._tick_counter,
            frame_id=frame.frame_id,
            t_capture_mono=frame.t_mono,
            t_capture_wall=frame.t_wall,
            stage_ms=stage_ms,
            e2e_ms=e2e_ms,
            jetson_ms=jetson_ms,
            link_ms=link_ms,
            timebase=None if frame.timebase is None else frame.timebase.to_record(),
            fps=self._fps_ema,
            n_detections=len(detections),
            vehicles=vehicles,
            obs_result=obs_result,
            policy=policy_out,
            advisory=advisory,
            gps=gps,
            n_peers=len(peers) if peers else 0,
        )
        self._tick_counter += 1
        return tick
