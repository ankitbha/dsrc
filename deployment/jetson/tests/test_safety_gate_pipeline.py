"""Pipeline-level invariants for the task-144 safety gate (step 9).

Deliberately self-contained rather than sharing fixtures with
test_pipeline_smoke.py: two agents were editing that file concurrently while
this one was written, and duplicating a small fixture is cheaper than adding
a cross-file dependency to a file already under contention.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from perception.detector import Detection
from perception.distance import DistanceEstimator
from perception.observation_builder import BuilderConfig, ObservationBuilder
from perception.tracker import IouTracker
from pipeline import PerceptionPolicyPipeline
from policy.actor_runtime import ActorRuntime, PolicyOutput
from policy.advisory import GATED_LANE_TEXT, AdvisoryDecoder
from policy.export_policy import build_random, export
from policy.sensing_controller import RULE_FIRED, RULE_NOT_EVALUABLE
from sensors.camera_stream import Frame
from sensors.gps_reader import GpsFix

#: A fixed action guaranteed to exercise the gate: "slow" reduces
#: algebraically to a clamp under the configured free-flow speed (plan_task144
#: E5), so pinning it -- rather than trusting the random-init actor to pick it
#: on its own -- is what makes the identity test below a real discriminator
#: instead of one that happens to pass because the gate never changed
#: anything on this particular random rollout.
FORCED_SLOW_ACTION: dict[str, str] = {
    "desired_speed_bin": "slow",
    "desired_headway_bin": "normal",
    "lane_preference": "prefer_left_if_safe",
    "merge_mode": "normal",
}

FX, CX, HORIZON, CAM_H = 800.0, 640.0, 360.0, 1.25


class FakeDetector:
    last_timings: dict[str, float] = {}

    def infer(self, image):
        return []

    def warmup(self, iterations: int = 1) -> float:
        return 0.0


def project_box(z_m: float, x_m: float) -> np.ndarray:
    w_px = FX * 1.8 / z_m
    h_px = 0.85 * w_px
    u = CX + x_m * FX / z_m
    v_bottom = HORIZON + CAM_H * FX / z_m
    return np.array([u - w_px / 2, v_bottom - h_px, u + w_px / 2, v_bottom], dtype=np.float32)


@pytest.fixture(scope="module")
def actor_bundle(tmp_path_factory) -> str:
    prefix = tmp_path_factory.mktemp("bundle") / "actor_policy"
    actor, info = build_random(seed=0)
    export(actor, info, str(prefix))
    return str(prefix)


def _make_pipeline(actor_bundle: str, *, withhold: bool = True) -> PerceptionPolicyPipeline:
    return PerceptionPolicyPipeline(
        detector=FakeDetector(),
        tracker=IouTracker(min_hits=1),
        distance=DistanceEstimator(
            fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H, ema_alpha=0.6
        ),
        builder=ObservationBuilder(BuilderConfig()),
        actor=ActorRuntime(actor_bundle),
        advisory_decoder=AdvisoryDecoder(units="mph"),
        withhold_lane_when_not_evaluable=withhold,
    )


def run_ticks(pipeline: PerceptionPolicyPipeline, n: int, *, with_leader: bool = False, dt: float = 1 / 30):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
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
        detections = (
            [Detection(xyxy=project_box(20.0, 0.0), conf=0.9, cls=2)] if with_leader else []
        )
        tick = pipeline.step(frame, fix, detections_override=detections)
    return tick


def _force_evidenced_low_density(pipeline: PerceptionPolicyPipeline, density_veh_per_km: float) -> None:
    """Wraps `pipeline.builder.build` so the returned `ObservationResult`
    reports `local_density_veh_per_km` as genuine evidence (source
    `derived`, not `derived_empty`/substituted) at the given value.

    Needed because validator round 1's Fix 1 means a not_evaluable
    low_speed_uncongested no longer moves the speed at all -- so a test
    that wants to see the gate genuinely clamp can no longer rely on this
    rig's ordinary not-evidence behaviour (F1's own reproduction) and must
    force a REAL low-density reading instead, the same way `with_leader`
    already forces a real `target_lane_front_gap_m` elsewhere in this file.
    """
    from perception import provenance

    original_build = pipeline.builder.build

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        result.field_sources["local_density_bin"] = provenance.SOURCE_DERIVED
        result.diagnostics["density_veh_per_km"] = density_veh_per_km
        return result

    pipeline.builder.build = build


def test_advisory_speed_equals_bounded_speed(actor_bundle: str) -> None:
    """decision 2's own identity: advisory.recommended_speed_mps continues to
    mean the number shown to the driver, and equals safety.bounded.speed_mps
    by construction -- checked on a tick where the gate actually changes the
    speed (forced "slow" against a genuinely-evidenced low density, per
    validator round 1's Fix 1), not one where proposed and bounded happen to
    coincide because nothing fired.
    """
    pipeline = _make_pipeline(actor_bundle)
    pipeline.actor.act = lambda encoded: PolicyOutput(
        action=dict(FORCED_SLOW_ACTION), head_probs={}, chosen_prob={}, confidence=1.0, latency_ms=0.0,
    )
    _force_evidenced_low_density(pipeline, density_veh_per_km=2.0)
    tick = run_ticks(pipeline, 5)
    assert tick.safety_gate.rules["low_speed_uncongested"].status == RULE_FIRED
    assert tick.safety_gate.proposed_speed_mps != tick.safety_gate.bounded_speed_mps  # the gate DID something
    assert tick.advisory.recommended_speed_mps == tick.safety_gate.bounded_speed_mps
    record = tick.to_record()
    assert record["advisory"]["recommended_speed_mps"] == record["safety"]["bounded"]["speed_mps"]


def test_lane_text_reflects_the_bounded_lane_action(actor_bundle: str) -> None:
    pipeline = _make_pipeline(actor_bundle)
    tick = run_ticks(pipeline, 10)
    assert tick.advisory.lane_text == GATED_LANE_TEXT[tick.safety_gate.bounded_lane_action]


def test_lane_action_is_withheld_by_default_on_this_rig(actor_bundle: str) -> None:
    """Open item 1, reproduced through the real loop: with no rear camera,
    at least one lane guard is not_evaluable on every tick, so whenever the
    policy proposes a lane change the gate withholds it and says why."""
    pipeline = _make_pipeline(actor_bundle, withhold=True)
    tick = run_ticks(pipeline, 20)
    if tick.safety_gate.proposed_lane_action is not None:
        assert tick.safety_gate.bounded_lane_action is None
        assert tick.safety_gate.lane_withheld == RULE_NOT_EVALUABLE
        assert tick.advisory.lane_text == "Keep lane"


def test_withhold_flag_off_restores_the_raw_lane_action(actor_bundle: str) -> None:
    pipeline = _make_pipeline(actor_bundle, withhold=False)
    tick = run_ticks(pipeline, 20)
    assert tick.safety_gate.bounded_lane_action == tick.safety_gate.proposed_lane_action
    assert tick.safety_gate.lane_withheld is None


def test_target_lane_front_gap_becomes_evaluable_with_a_tracked_leader(actor_bundle: str) -> None:
    """The one lane guard this rig can ever evaluate: target_lane_front_gap_m
    is fed from the leader gap, so a tracked leader makes it evidence
    (matching plan_task144's E3/E4: this is exactly why the 2026-09-02 run's
    1,229 ticks were evaluable while the other three runs' 2,684 were not)."""
    pipeline = _make_pipeline(actor_bundle)
    tick = run_ticks(pipeline, 10, with_leader=True)
    assert tick.safety_gate.rules["target_lane_front_gap"].status != RULE_NOT_EVALUABLE


def test_safety_block_is_present_and_json_shaped(actor_bundle: str) -> None:
    import json

    pipeline = _make_pipeline(actor_bundle)
    tick = run_ticks(pipeline, 5)
    record = tick.to_record()
    text = json.dumps(record)
    parsed = json.loads(text)
    safety = parsed["safety"]
    assert set(safety) == {
        "proposed", "bounded", "delta_speed_mps", "emergency_override",
        "lane_withheld", "evaluable", "not_evaluable", "rules", "config",
    }
    assert len(safety["rules"]) == 12
    assert safety["evaluable"] + safety["not_evaluable"] == 12
    # validator round 1, Fix 4 (F3/F4): the gate's own configuration this
    # tick ran under, so a replay tool need not assume its own defaults.
    assert set(safety["config"]) == {
        "enabled", "withhold_lane_when_not_evaluable", "time_s",
        "min_contextual_speed_mps", "density_max_age_s",
    }


def test_gate_ms_is_always_measured_never_absent(actor_bundle: str) -> None:
    """The gate runs the full rule evaluation on every tick regardless of
    which rules are not_evaluable -- there is no early-return path here the
    way there is for dsrc_infer, so this stage must never be `absent`."""
    pipeline = _make_pipeline(actor_bundle)
    tick = run_ticks(pipeline, 5)
    record = tick.to_record()
    assert record["stages"]["gate"]["basis"] == "measured"
    assert record["stages"]["gate"]["ms"] is not None
    assert record["stages"]["gate"]["ms"] >= 0.0
