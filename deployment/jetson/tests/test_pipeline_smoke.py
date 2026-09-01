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


# -- the per-tick stages record -----------------------------------------------


ALL_STAGE_KEYS = {
    "capture", "capture_to_encode_start", "encode", "encode_done_to_enqueue",
    "enqueue_to_wire", "transport", "jpeg_decode", "detect", "track", "fuse",
    "infer", "decode",
}


def test_every_stage_key_is_present_on_a_local_camera_tick(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    assert set(tick.stages) == ALL_STAGE_KEYS


def test_a_local_camera_reports_capture_as_an_instant_on_its_own_clock(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    capture = tick.stages["capture"]
    assert capture.basis == "instant"
    assert capture.clock == "jetson"
    assert capture.ms == 0.0


def test_a_local_camera_has_no_phone_dwell_or_transport_stages(pipeline) -> None:
    """None of these happened -- captured locally, on this device -- and each
    must say so by name rather than reporting a zero for a segment that does
    not exist."""
    tick = run_ticks(pipeline, 3)
    for key in ("capture_to_encode_start", "encode", "encode_done_to_enqueue",
                "enqueue_to_wire", "transport"):
        stage = tick.stages[key]
        assert stage.basis == "absent", key
        assert stage.ms is None, key
        assert stage.reason, key


def test_jpeg_decode_is_absent_for_a_local_source(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    stage = tick.stages["jpeg_decode"]
    assert stage.basis == "absent"
    assert stage.ms is None
    assert "local" in stage.reason


def test_the_jetson_side_stages_are_measured_and_non_negative(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    for key in ("detect", "track", "fuse", "infer", "decode"):
        stage = tick.stages[key]
        assert stage.basis == "measured", key
        assert stage.clock == "jetson", key
        assert stage.ms is not None and stage.ms >= 0.0, key


def test_infer_plus_decode_equals_policy_advisory_within_rounding(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    total = tick.stages["infer"].ms + tick.stages["decode"].ms
    # set_target_headway runs between the decode stamp and t4, so the two are
    # close but not required to be bit-identical.
    assert total == pytest.approx(tick.stage_ms["policy_advisory"], abs=1.0)


def test_fuse_never_exceeds_the_observe_segment_it_is_timed_inside(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    assert tick.stages["fuse"].ms <= tick.stage_ms["observe"] + 1e-6


def test_fuse_reports_the_builders_own_measurement(pipeline) -> None:
    """A peer list and a traffic feed give `fuse` real work to do -- feed
    fusion and the peer merge -- so its duration reads straight off the
    builder rather than off a default that happens to also be a number.
    """
    from perception.observation_builder import PeerState

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    now = time.monotonic()
    frame = Frame(image=image, frame_id=1, t_mono=now - 0.05, t_wall=time.time() - 0.05)
    fix = GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
        utc_epoch_s=time.time(), t_mono=now - 0.05, t_wall=time.time() - 0.05,
    )
    peers = [
        PeerState(peer_id="a", distance_m=80.0, speed_mps=24.0, lane_id=1),
        PeerState(peer_id="b", distance_m=120.0, speed_mps=26.0, lane_id=2),
    ]
    tick = pipeline.step(
        frame, fix, peers, detections_override=scene_detections(0.0), feed=jammed_reading(),
    )

    assert tick.stages["fuse"].ms == pytest.approx(pipeline.builder.last_timings["fuse_ms"])
    assert tick.stages["fuse"].ms > 0.0


def test_a_proxied_transport_segment_feeds_neither_stat_series(pipeline) -> None:
    """A stage read as absent must not sneak into a series through the back
    door: `basis == "absent"` is the only thing pipeline.step checks before
    routing a sample into transport_round_trip or transport_one_way, so a
    proxied phone stage -- absent for a different reason than "no phone
    behind this frame" -- must be excluded on the same terms a local
    camera's is."""
    from sensors.time_sync import StageTiming

    phone_stages = {
        "capture": StageTiming.instant(clock="phone"),
        "capture_to_encode_start": StageTiming.absent(clock="phone", reason="x"),
        "encode": StageTiming.absent(clock="phone", reason="x"),
        "encode_done_to_enqueue": StageTiming.absent(clock="phone", reason="x"),
        "enqueue_to_wire": StageTiming.measured(1.0, clock="phone"),
        "transport": StageTiming.absent(clock="cross", reason="samples too old"),
    }
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    now = time.monotonic()
    frame = Frame(
        image=image, frame_id=1000, t_mono=now - 0.05, t_wall=time.time() - 0.05,
        jpeg_decode_s=0.002, phone_stages=phone_stages,
    )
    fix = GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
        utc_epoch_s=time.time(), t_mono=now - 0.05, t_wall=time.time() - 0.05,
    )
    tick = pipeline.step(frame, fix, detections_override=scene_detections(0.0))

    assert tick.stages["transport"].basis == "absent"
    snapshot = pipeline.stats.snapshot()
    assert snapshot["transport_round_trip_ms"] is None
    assert snapshot["transport_one_way_ms"] is None


def test_the_stages_dict_survives_json_round_trip(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    record = tick.to_record()
    parsed = json.loads(json.dumps(record))
    assert set(parsed["stages"]) == ALL_STAGE_KEYS
    assert parsed["stages"]["capture"]["basis"] == "instant"
    assert parsed["stages"]["transport"]["basis"] == "absent"


def test_the_tick_record_carries_the_capture_stamp_in_nanoseconds(pipeline) -> None:
    tick = run_ticks(pipeline, 3)
    record = tick.to_record()
    assert record["t_capture_mono_ns"] == int(round(tick.t_capture_mono * 1e9))


def test_capture_stamp_ns_rounds_rather_than_truncates() -> None:
    from sensors.time_sync import capture_stamp_ns

    # Chosen so the value's nanosecond-scale fraction sits just past the
    # halfway point: truncating and rounding disagree by exactly one
    # nanosecond here. Real `time.monotonic()` values on this machine's
    # coarser clock essentially never land on such a fraction, which is why
    # the two call sites this helper replaces could disagree for a long time
    # without a local test run ever seeing it.
    t = 50000.1234567895
    assert int(t * 1e9) == 50000123456789
    assert capture_stamp_ns(t) == 50000123456790


def test_the_tick_records_key_matches_what_sensing_loop_puts_on_the_wire(pipeline) -> None:
    """`Tick.to_record()` and `SensingLoop.on_tick` used to convert the same
    `t_capture_mono` float through two different spellings -- truncate here,
    round there -- so a phone's inbound advisory line and this tick's own
    record disagreed on the join key for the tick they are both about.
    """
    from policy.sensing_loop import SensingLoop

    boundary_t_mono = 50000.1234567895
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = Frame(image=image, frame_id=1, t_mono=boundary_t_mono, t_wall=time.time())
    fix = GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
        utc_epoch_s=time.time(), t_mono=boundary_t_mono, t_wall=time.time(),
    )
    tick = pipeline.step(frame, fix, detections_override=scene_detections(0.0))

    outcome = SensingLoop().on_tick(tick, None)
    assert tick.to_record()["t_capture_mono_ns"] == outcome.command.t_capture_mono_ns


def test_new_stat_series_are_in_the_snapshot(pipeline) -> None:
    run_ticks(pipeline, 3)
    snapshot = pipeline.stats.snapshot()
    for key in ("transport_round_trip_ms", "transport_one_way_ms",
                "jpeg_decode_ms", "fuse_ms", "infer_ms", "decode_ms"):
        assert key in snapshot, key
    # No phone behind any of these ticks: nothing was converted, so the two
    # transport series and jpeg_decode stay empty rather than reporting zeros.
    assert snapshot["transport_round_trip_ms"] is None
    assert snapshot["transport_one_way_ms"] is None
    assert snapshot["jpeg_decode_ms"] is None
    # fuse/infer/decode ran on every tick, local or not.
    assert snapshot["fuse_ms"]["n"] == 3
    assert snapshot["infer_ms"]["n"] == 3
    assert snapshot["decode_ms"]["n"] == 3


def test_a_phone_fed_frames_stages_pass_through_unchanged(pipeline) -> None:
    from sensors.time_sync import StageTiming

    phone_stages = {
        "capture": StageTiming.instant(clock="phone"),
        "capture_to_encode_start": StageTiming.measured(3.0, clock="phone"),
        "encode": StageTiming.measured(12.0, clock="phone"),
        "encode_done_to_enqueue": StageTiming.measured(1.5, clock="phone"),
        "enqueue_to_wire": StageTiming.measured(0.5, clock="phone"),
        "transport": StageTiming.converted(
            9.0, bound_ms=25.0, estimate_id=7, source="round_trip"
        ),
    }
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    now = time.monotonic()
    frame = Frame(
        image=image, frame_id=999, t_mono=now - 0.05, t_wall=time.time() - 0.05,
        jpeg_decode_s=0.004, phone_stages=phone_stages,
    )
    fix = GpsFix(
        valid=True, lat=40.0, lon=-74.0, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
        utc_epoch_s=time.time(), t_mono=now - 0.05, t_wall=time.time() - 0.05,
    )
    tick = pipeline.step(frame, fix, detections_override=scene_detections(0.0))

    for key, expected in phone_stages.items():
        assert tick.stages[key] == expected, key
    assert tick.stages["jpeg_decode"].ms == pytest.approx(4.0)
    assert tick.stages["jpeg_decode"].basis == "measured"

    snapshot = pipeline.stats.snapshot()
    assert snapshot["transport_round_trip_ms"]["n"] == 1
    assert snapshot["transport_round_trip_ms"]["mean"] == pytest.approx(9.0)
    assert snapshot["transport_one_way_ms"] is None
    assert snapshot["jpeg_decode_ms"]["n"] == 1


def test_the_sensing_loop_reads_a_real_tick(pipeline) -> None:
    """Task 31's wiring, against a Tick the pipeline actually built.

    Every other test of `inputs_from` uses a fake tick, and a fake agrees with the
    code that reads it rather than with the object the pipeline produces. If a field
    were renamed -- `obs_result.feed`, `policy.head_probs`, `diagnostics["gps_age_s"]`
    -- those tests would keep passing while the live loop decided from Nones.
    """
    from policy.sensing_loop import SensingLoop, inputs_from

    from perception import provenance

    tick = run_ticks(pipeline, 45)
    inputs = inputs_from(tick, None, now=time.monotonic())

    # The free tier and the camera's own view: present, not defaulted away.
    assert inputs.ego_speed == pytest.approx(27.0)
    # Whether the speed window settles to a real slope or is refused as stale
    # depends on how much real wall-clock time this test's own 45 ticks took
    # against the scripted GPS schedule -- `run_ticks` deliberately lags GPS
    # behind real time, because e2e latency is measured against the real
    # clock -- so the invariant checked is that a null value and a
    # substituted source agree with each other, not a specific outcome.
    assert (inputs.ego_acceleration is None) == provenance.is_substituted(
        inputs.ego_acceleration_source
    )
    assert inputs.ego_speed_source is not None
    assert inputs.camera_density_bin_source is not None
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
    assert outcome.command.t_capture_mono_ns == int(round(tick.t_capture_mono * 1e9))


def jammed_reading():
    """A feed reading the fusion layer will actually take ownership of."""
    from sensors.here_feed import FlowLink, FlowReading, Outcome

    return FlowReading(
        outcome=Outcome.OK,
        link=FlowLink(
            points=((40.0, -74.0), (40.01, -74.0)),
            speed_mps=3.0, free_flow_mps=30.0, jam_factor=9.0,
            confidence=0.9, traversability="open", length_m=1000.0,
        ),
        response_age_s=1.0, response_age_bound_s=0.05,
        link_distance_m=400.0, link_cross_track_m=5.0,
    )


def test_the_feed_reaches_the_controller_through_the_real_pipeline(pipeline) -> None:
    """Task 31's other half, and the one a fake tick cannot check.

    `pipeline.step` had no `feed` parameter, so `ObservationBuilder.build` defaulted
    it to None on every tick, `feed_fusion.own(None)` declined with `no_reading`, and
    the whole HERE ingestion path -- parse, associate, age, publish -- terminated in
    a log record. `Trigger.DISAGREEMENT`, one of the controller's three raise rules,
    could not fire on any drive. The test that was supposed to pin this passed over a
    tick shape the pipeline does not build.
    """
    from policy.sensing_loop import inputs_from

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    base_mono = time.monotonic() - 2.0
    tick = None
    for i in range(45):
        t = i / 30
        frame = Frame(image=image, frame_id=i, t_mono=base_mono + t,
                      t_wall=time.time() - 2.0 + t)
        fix = GpsFix(valid=True, lat=40.0, lon=-74.0, speed_mps=27.0,
                     heading_deg=0.0, fix_quality=1, num_sats=9, hdop=0.9,
                     altitude_m=3.0, t_mono=base_mono + t, t_wall=time.time())
        tick = pipeline.step(frame, fix, detections_override=scene_detections(t),
                             feed=jammed_reading())

    assert tick.obs_result.feed is not None
    assert tick.obs_result.feed.declined is None, tick.obs_result.feed
    # 1 - 3/30
    assert tick.obs_result.feed.downstream_congestion == pytest.approx(0.9, abs=1e-6)
    assert inputs_from(tick, None, now=time.monotonic()).feed_congestion == pytest.approx(0.9)


def test_no_feed_still_builds_a_tick(pipeline) -> None:
    # The local-camera path, which passes nothing. Declining is the ordinary case.
    tick = run_ticks(pipeline, 10)
    assert tick.obs_result.feed is not None
    assert tick.obs_result.feed.downstream_congestion is None
