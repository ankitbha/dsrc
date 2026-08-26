"""Full-pipeline smoke test without camera, GPS hardware, or GPU.

A scripted scene (leader closing at -2 m/s, one vehicle per adjacent
lane) is projected through the pinhole model and fed to the real
tracker -> distance -> observation -> actor -> advisory chain with a
random-init policy bundle. Verifies wiring, sim-schema conformance,
JSON-serializability of log records, and the action->observation
headway feedback loop.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from perception.detector import Detection
from perception.distance import DistanceEstimator
from perception.observation_builder import BuilderConfig, ObservationBuilder
from perception.tracker import IouTracker
from pipeline import PerceptionPolicyPipeline
from policy import sim_contract
from policy.actor_runtime import ActorRuntime
from policy.advisory import AdvisoryDecoder
from policy.export_policy import build_random, export
from sensors.camera_stream import Frame
from sensors.gps_reader import GpsFix

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25


class FakeDetector:
    """Stands in for TrtYoloDetector; pipeline never calls it when
    detections_override is supplied, but keeps the interface complete."""

    last_timings: dict[str, float] = {}

    def infer(self, image) -> list[Detection]:
        return []

    def warmup(self, iterations: int = 1) -> float:
        return 0.0


def project_box(z_m: float, x_m: float) -> np.ndarray:
    w_px = FX * 1.8 / z_m
    h_px = 0.85 * w_px
    u = CX + x_m * FX / z_m
    v_bottom = HORIZON + CAM_H * FX / z_m
    return np.array([u - w_px / 2, v_bottom - h_px, u + w_px / 2, v_bottom], dtype=np.float32)


def scene_detections(t_s: float) -> list[Detection]:
    leader_z = max(10.0, 45.0 - 2.0 * t_s)  # closing at 2 m/s
    boxes = [
        project_box(leader_z, 0.0),
        project_box(28.0, -3.7),
        project_box(60.0, 3.7),
    ]
    return [Detection(xyxy=b, conf=0.9, cls=2) for b in boxes]


@pytest.fixture(scope="module")
def actor_bundle(tmp_path_factory) -> str:
    prefix = tmp_path_factory.mktemp("bundle") / "actor_policy"
    actor, info = build_random(seed=0)
    export(actor, info, str(prefix))
    return str(prefix)


@pytest.fixture
def pipeline(actor_bundle: str) -> PerceptionPolicyPipeline:
    return PerceptionPolicyPipeline(
        detector=FakeDetector(),
        tracker=IouTracker(min_hits=2),
        distance=DistanceEstimator(
            fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H, ema_alpha=0.6
        ),
        builder=ObservationBuilder(BuilderConfig()),
        actor=ActorRuntime(actor_bundle),
        advisory_decoder=AdvisoryDecoder(units="mph"),
    )


def run_ticks(pipeline: PerceptionPolicyPipeline, n: int, dt: float = 1 / 30):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # synthetic capture times must lie in the past (e2e is measured against
    # the real monotonic clock) while still being spaced dt apart
    base_mono = time.monotonic() - n * dt - 0.01
    base_wall = time.time() - n * dt - 0.01
    tick = None
    for i in range(n):
        t = i * dt
        frame = Frame(image=image, frame_id=i, t_mono=base_mono + t, t_wall=base_wall + t)
        fix = GpsFix(
            valid=True, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
            fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
            utc_epoch_s=base_wall + t, t_mono=base_mono + t, t_wall=base_wall + t,
        )
        tick = pipeline.step(frame, fix, detections_override=scene_detections(t))
    return tick


def test_pipeline_produces_aligned_observation_and_advisory(pipeline) -> None:
    tick = run_ticks(pipeline, 45)
    obs = tick.obs_result.obs

    assert tick.obs_result.encoded.shape == (sim_contract.local_obs_dim(),)
    assert obs["ego_speed"] == pytest.approx(27.0)
    # leader should be locked on and closing
    assert math.isfinite(obs["leader_gap"])
    assert 10.0 < obs["leader_gap"] < 45.0
    assert obs["leader_relative_speed"] == pytest.approx(-2.0, abs=0.7)
    assert obs["left_lane_front_gap"] == pytest.approx(28.0, rel=0.1)
    assert obs["right_lane_front_gap"] == pytest.approx(60.0, rel=0.1)
    # 3 forward vehicles -> symmetrized count
    assert obs["active_vehicle_count_local"] == 6

    assert tick.advisory.recommended_speed_mps >= 12.0
    assert tick.advisory.lane_text
    assert tick.policy.action["desired_speed_bin"] in sim_contract.ACTION_VALUES["desired_speed_bin"]


def test_headway_feedback_loop(pipeline) -> None:
    tick = run_ticks(pipeline, 5)
    expected = sim_contract.decode_headway_bin(tick.policy.action["desired_headway_bin"])
    next_tick = run_ticks(pipeline, 1)
    assert next_tick.obs_result.obs["target_headway_s"] == expected


def test_tick_record_is_json_serializable(pipeline) -> None:
    tick = run_ticks(pipeline, 10)
    record = tick.to_record()
    text = json.dumps(record)  # Python JSON: Infinity literals allowed
    parsed = json.loads(text)
    assert parsed["type"] == "tick"
    assert parsed["obs"]["follower_gap"] == math.inf
    assert len(parsed["encoded"]) == sim_contract.local_obs_dim()
    assert parsed["advisory"]["recommended_speed_display"] > 0


def test_stage_timings_recorded(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    for stage in ("detect", "track_distance", "observe", "policy_advisory"):
        assert stage in tick.stage_ms
    assert tick.e2e_ms >= 0.0
    snapshot = pipeline.stats.snapshot()
    assert snapshot["e2e_ms"]["p95"] >= snapshot["e2e_ms"]["p50"] >= 0.0


def test_the_sensing_loop_reads_a_real_tick(pipeline) -> None:
    """Task 31's wiring, against a Tick the pipeline actually built.

    Every other test of `inputs_from` uses a fake tick, and a fake agrees with the
    code that reads it rather than with the object the pipeline produces. If a field
    were renamed -- `obs_result.feed`, `policy.head_probs`, `diagnostics["gps_age_s"]`
    -- those tests would keep passing while the live loop decided from Nones.
    """
    from policy.sensing_loop import SensingLoop, inputs_from

    tick = run_ticks(pipeline, 45)
    inputs = inputs_from(tick, None, now=time.monotonic())

    # The free tier and the camera's own view: present, not defaulted away.
    assert inputs.ego_speed == pytest.approx(27.0)
    assert inputs.ego_acceleration is not None
    assert inputs.camera_density_bin in (0, 1, 2, 3)
    assert inputs.policy_margin is not None and 0.0 <= inputs.policy_margin <= 1.0
    assert inputs.lat == pytest.approx(40.0) and inputs.lon == pytest.approx(-74.0)
    assert inputs.position_valid is True
    assert inputs.position_age_s is not None
    # No phone in this run, so no feed and no telemetry -- and silence is not
    # nominal, so these must be None rather than a comfortable default.
    assert inputs.feed_congestion is None
    assert inputs.thermal_status is None and inputs.telemetry_age_s is None

    loop = SensingLoop()
    outcome = loop.on_tick(tick, None)
    assert outcome.decision.rates
    assert outcome.command.shadow is True
    assert outcome.command.t_capture_mono_ns == int(tick.t_capture_mono * 1e9)
