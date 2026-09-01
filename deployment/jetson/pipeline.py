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
from sensors.time_sync import StageTiming, capture_stamp_ns

#: Why a phone-side stage is absent on a tick with no phone behind it. Named
#: once rather than restated at each of the five stages a local camera has
#: none of, so the reason string cannot drift between them.
NO_PHONE_STAGES_REASON = "no phone behind this frame; captured locally"


class RollingStats:
    def __init__(self, window: int = 300) -> None:
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(value)

    def summary(self) -> dict[str, float] | None:
        """None for an empty series, not zeros.

        Zeros read as a measured latency of zero. That is wrong for any series
        and it became reachable when the link segment arrived: a local-camera run
        has no link at all, so every such run was publishing
        `link_ms: 0/0/0` for a segment that does not exist.

        `n` is included because a distribution without its count cannot be
        weighed against another one.
        """
        if not self._values:
            return None
        arr = np.asarray(self._values)
        return {
            "n": len(self._values),
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
    #: The task-33 per-stage record: capture, encode and its two neighbouring
    #: dwell segments, transport, jpeg_decode, detect, track, fuse, infer and
    #: decode -- ten and sometimes eleven entries (a local camera adds no
    #: encode-dwell segments of its own but still names why they are absent).
    #: `return` and `render` are not here: they are facts only the phone
    #: witnesses, joined in offline by `eval_run.py --phone-log`.
    stages: dict[str, StageTiming] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON-able log record (uses Python JSON's Infinity literal for inf)."""
        return {
            "type": "tick",
            "tick_id": self.tick_id,
            "frame_id": self.frame_id,
            "t_wall": self.t_capture_wall,
            # The exact join key an offline phone-log reader needs: `capture_stamp_ns`
            # is the same conversion `policy.sensing_loop` puts through when it builds
            # `AdvisoryMessage.t_capture_mono_ns` for this tick, so a phone's inbound
            # advisory line and this tick's record carry the identical integer.
            "t_capture_mono_ns": capture_stamp_ns(self.t_capture_mono),
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "e2e_ms": round(self.e2e_ms, 2),
            "jetson_ms": round(self.jetson_ms, 2),
            "link_ms": None if self.link_ms is None else round(self.link_ms, 2),
            "timebase": self.timebase,
            "stages": {name: stage.to_record() for name, stage in self.stages.items()},
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
    #: The network hop alone (wire departure to Jetson arrival), kept in two
    #: series rather than one. A round-trip-converted sample's error is
    #: bounded by half a round trip; a one-way-converted one's is bounded only
    #: by a delay spread with an unobservable floor. Pooling the two would let
    #: one series launder the other's confidence.
    transport_round_trip: RollingStats = field(default_factory=RollingStats)
    transport_one_way: RollingStats = field(default_factory=RollingStats)
    jpeg_decode: RollingStats = field(default_factory=RollingStats)
    fuse: RollingStats = field(default_factory=RollingStats)
    infer: RollingStats = field(default_factory=RollingStats)
    decode: RollingStats = field(default_factory=RollingStats)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "e2e_ms": self.e2e.summary(),
            "jetson_ms": self.jetson.summary(),
            "link_ms": self.link.summary(),
            "detect_ms": self.detect.summary(),
            "track_ms": self.track.summary(),
            "observe_ms": self.observe.summary(),
            "policy_ms": self.policy.summary(),
            "transport_round_trip_ms": self.transport_round_trip.summary(),
            "transport_one_way_ms": self.transport_one_way.summary(),
            "jpeg_decode_ms": self.jpeg_decode.summary(),
            "fuse_ms": self.fuse.summary(),
            "infer_ms": self.infer.summary(),
            "decode_ms": self.decode.summary(),
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
        feed: Any = None,
    ) -> Tick:
        """One tick. `feed` is the traffic reading this tick asked for, or None.

        Threaded through rather than held: `HereFeed.at` answers against the caller's
        position at the moment it asks, so a reading fetched here would be about
        wherever the pipeline happened to be, not about the frame being processed.
        Without it `ObservationBuilder.build` defaulted `feed` to None on every tick,
        `feed_fusion.own(None)` declined with `no_reading`, and the entire HERE
        ingestion path -- parse, associate, age, publish -- terminated in a log
        record. `Trigger.DISAGREEMENT`, one of the controller's three raise rules,
        could not fire on any drive.
        """
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
            vehicles, gps, time.monotonic(), peers, feed
        )
        t3 = time.monotonic()

        policy_out: PolicyOutput = self.actor.act(obs_result.encoded)
        t3a = time.monotonic()
        advisory: Advisory = self.advisory_decoder.decode(policy_out, obs_result.obs)
        t3b = time.monotonic()
        self.builder.set_target_headway(advisory.headway_target_s)
        t4 = time.monotonic()
        infer_ms = (t3a - t3) * 1000.0
        decode_ms = (t3b - t3a) * 1000.0

        # Arrival is exact and local; capture may be converted from a peer clock.
        # With no timebase stamp the frame was captured here, so the two coincide.
        t_arrival = frame.timebase.t_arrival_mono if frame.timebase else frame.t_mono
        e2e_ms = (t4 - frame.t_mono) * 1000.0
        jetson_ms = (t4 - t_arrival) * 1000.0
        # None for a local camera (there is no link) and None under the proxy
        # (capture IS arrival, so the difference is zero for a reason that has
        # nothing to do with the link). A zero here therefore always means a
        # measured zero.
        link_s = None if frame.timebase is None else frame.timebase.link_s
        link_ms = None if link_s is None else link_s * 1000.0
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

        stages = self._stages(
            frame, stage_ms=stage_ms, infer_ms=infer_ms, decode_ms=decode_ms
        )
        transport = stages["transport"]
        if transport.basis == "converted":
            if transport.source == "round_trip":
                self.stats.transport_round_trip.add(transport.ms)
            elif transport.source == "one_way":
                self.stats.transport_one_way.add(transport.ms)
        if stages["jpeg_decode"].ms is not None:
            self.stats.jpeg_decode.add(stages["jpeg_decode"].ms)
        if stages["fuse"].ms is not None:
            self.stats.fuse.add(stages["fuse"].ms)
        self.stats.infer.add(infer_ms)
        self.stats.decode.add(decode_ms)

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
            stages=stages,
        )
        self._tick_counter += 1
        return tick

    def _stages(
        self, frame: Frame, *, stage_ms: dict[str, float], infer_ms: float, decode_ms: float,
    ) -> dict[str, StageTiming]:
        """The task-33 per-tick record: what the phone's header answered
        exactly, what this device measured on its own clock, and what could
        not be answered at all -- never a zero standing in for "not measured".
        """
        if frame.phone_stages is not None:
            stages: dict[str, StageTiming] = dict(frame.phone_stages)
        else:
            # A local camera has none of the phone-side segments and no
            # network hop: capture happened here, on this device's own clock.
            stages = {
                "capture": StageTiming.instant(clock="jetson"),
                "capture_to_encode_start": StageTiming.absent(
                    clock="phone", reason=NO_PHONE_STAGES_REASON
                ),
                "encode": StageTiming.absent(clock="phone", reason=NO_PHONE_STAGES_REASON),
                "encode_done_to_enqueue": StageTiming.absent(
                    clock="phone", reason=NO_PHONE_STAGES_REASON
                ),
                "enqueue_to_wire": StageTiming.absent(
                    clock="phone", reason=NO_PHONE_STAGES_REASON
                ),
                "transport": StageTiming.absent(clock="cross", reason=NO_PHONE_STAGES_REASON),
            }

        if frame.jpeg_decode_s is not None:
            stages["jpeg_decode"] = StageTiming.measured(
                frame.jpeg_decode_s * 1000.0, clock="jetson"
            )
        else:
            stages["jpeg_decode"] = StageTiming.absent(
                clock="jetson", reason="decoded inside the local camera source, not timed here"
            )

        stages["detect"] = StageTiming.measured(stage_ms["detect"], clock="jetson")
        stages["track"] = StageTiming.measured(stage_ms["track_distance"], clock="jetson")
        fuse_ms = self.builder.last_timings.get("fuse_ms")
        stages["fuse"] = (
            StageTiming.absent(clock="jetson", reason="builder recorded no fuse timing this tick")
            if fuse_ms is None else StageTiming.measured(fuse_ms, clock="jetson")
        )
        stages["infer"] = StageTiming.measured(infer_ms, clock="jetson")
        stages["decode"] = StageTiming.measured(decode_ms, clock="jetson")
        return stages
