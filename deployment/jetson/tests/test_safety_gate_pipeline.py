"""Pipeline-level invariants for the task-144 safety gate (step 9).

Deliberately self-contained rather than sharing fixtures with
test_pipeline_smoke.py: two agents were editing that file concurrently while
this one was written, and duplicating a small fixture is cheaper than adding
a cross-file dependency to a file already under contention.

The gate now bounds the DSRC controller's per-segment speed recommendation.
It runs only on a tick that has an ego row, so every fixture here drives a
real decision: a fully-covering HERE feed, and a GPS fix sitting on the first
super-segment's own polyline.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from perception.detector import Detection
from perception.distance import DistanceEstimator
from perception.observation_builder import BuilderConfig, ObservationBuilder
from perception.segment_state import DEFAULT_NETWORK_DEFINITION_PATH, SegmentStateBuilder
from perception.tracker import IouTracker
from pipeline import PerceptionPolicyPipeline
from policy import export_dsrc_policy
from policy.dsrc_runtime import DsrcRuntime
from policy.segment_advisory import SegmentAdvisoryDecoder
from policy.sensing_controller import RULE_FIRED
from sensors.camera_stream import Frame
from sensors.gps_reader import GpsFix
from sensors.here_feed import HereFeed

#: Action index 0 is `SPEED_ACTION_FRACTIONS[0]` == 0.5, so on Mainz's 60 km/h
#: super-segments it recommends 8.3 m/s. That is below
#: `min_contextual_speed_mps` (12.0), which is what makes
#: `low_speed_uncongested` able to fire and the gate able to change the
#: number. Pinned rather than trusted to the random-init policy, so the
#: identity test below is a real discriminator instead of one that happens to
#: pass because the gate never changed anything on this particular rollout.
FORCED_SLOW_ACTION_INDEX = 0

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


def _network_definition() -> dict:
    return json.loads(DEFAULT_NETWORK_DEFINITION_PATH.read_text())


#: The first super-segment's own first shape point. A fix here matches segment
#: 0 at zero distance, so `ego_segment` is 0 and the gate has a row to bound.
EGO_LAT, EGO_LON = _network_definition()["segments"][0]["edges"][0]["polyline"][0]


@pytest.fixture(scope="module")
def dsrc_bundle(tmp_path_factory) -> str:
    definition = _network_definition()
    model, info = export_dsrc_policy.build_random(definition, seed=0)
    prefix = str(tmp_path_factory.mktemp("dsrc_bundle") / "dsrc_policy")
    export_dsrc_policy.export(model, info, definition, prefix)
    return prefix


def _make_pipeline(
    dsrc_bundle: str, *, safety_enabled: bool = True,
) -> PerceptionPolicyPipeline:
    dsrc_runtime = DsrcRuntime(dsrc_bundle)
    return PerceptionPolicyPipeline(
        detector=FakeDetector(),
        tracker=IouTracker(min_hits=1),
        distance=DistanceEstimator(
            fx_px=FX, cx_px=CX, horizon_y_px=HORIZON, camera_height_m=CAM_H, ema_alpha=0.6
        ),
        builder=ObservationBuilder(BuilderConfig()),
        dsrc_runtime=dsrc_runtime,
        dsrc_segment_builder=SegmentStateBuilder(
            expected_network_fingerprint=dsrc_runtime.network_fingerprint,
        ),
        dsrc_advisory_decoder=SegmentAdvisoryDecoder.from_network_definition(
            expected_network_fingerprint=dsrc_runtime.network_fingerprint,
        ),
        safety_enabled=safety_enabled,
    )


def _covering_feed(t_mono: float) -> HereFeed:
    """A HERE response carrying one link on every super-segment's own anchor,
    so the coverage gate passes and the network actually runs."""
    feed = HereFeed()
    bodies = []
    for segment in _network_definition()["segments"]:
        lat, lon = segment["edges"][0]["polyline"][0]
        bodies.append({
            "location": {
                "length": 200.0,
                "shape": {"links": [{"points": [
                    {"lat": lat, "lng": lon},
                    {"lat": lat, "lng": lon + 0.0005},
                ]}]},
            },
            "currentFlow": {"speed": 15.0, "freeFlow": 16.67, "jamFactor": 3.0, "confidence": 0.9},
        })
    feed.offer(status=200, body=json.dumps({"results": bodies}).encode("utf-8"),
               received_t_mono=t_mono)
    return feed


def force_action(pipeline: PerceptionPolicyPipeline, action_index: int) -> None:
    """Pin every super-segment's action, leaving the rest of the decision --
    outcome, coverage, latency -- exactly as the real runtime produced it."""
    from dataclasses import replace

    original_decide = pipeline.dsrc_runtime.decide

    def decide(segment_state):
        decision = original_decide(segment_state)
        if decision.actions is None:
            return decision
        return replace(decision, actions=np.full_like(decision.actions, action_index))

    pipeline.dsrc_runtime.decide = decide


def run_ticks(pipeline: PerceptionPolicyPipeline, n: int, *, with_leader: bool = False,
              dt: float = 1 / 30):
    """`with_leader` puts one vehicle in the ego lane, closing at 2 m/s.

    Closing, not held at a fixed distance, and over enough ticks to span the
    0.2 s the relative-speed slope needs: `DistanceEstimator` reports
    `rel_speed_valid` False until then, and an invalid relative speed is what
    `forward_ttc` records as not evaluable.
    """
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    base_mono = time.monotonic() - n * dt - 0.01
    base_wall = time.time() - n * dt - 0.01
    feed = _covering_feed(base_mono)
    tick = None
    for i in range(n):
        t = i * dt
        frame = Frame(image=image, frame_id=i, t_mono=base_mono + t, t_wall=base_wall + t)
        fix = GpsFix(
            valid=True, lat=EGO_LAT, lon=EGO_LON, speed_mps=27.0, heading_deg=90.0,
            fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
            utc_epoch_s=base_wall + t, t_mono=base_mono + t, t_wall=base_wall + t,
        )
        detections = (
            [Detection(xyxy=project_box(max(10.0, 45.0 - 2.0 * t), 0.0), conf=0.9, cls=2)]
            if with_leader else []
        )
        tick = pipeline.step(
            frame, fix, detections_override=detections, here_feed_source=feed,
        )
    return tick


def _force_evidenced_low_density(pipeline: PerceptionPolicyPipeline,
                                 density_veh_per_km: float) -> None:
    """Wraps `pipeline.builder.build` so the returned `ObservationResult`
    reports `local_density_veh_per_km` as genuine evidence (source
    `derived`, not `derived_empty`/substituted) at the given value.

    Needed because validator round 1's Fix 1 means a not_evaluable
    low_speed_uncongested no longer moves the speed at all -- so a test
    that wants to see the gate genuinely clamp can no longer rely on this
    rig's ordinary not-evidence behaviour (F1's own reproduction) and must
    force a REAL low-density reading instead, the same way `with_leader`
    already forces a real leader gap elsewhere in this file.
    """
    from perception import provenance

    original_build = pipeline.builder.build

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        result.field_sources["local_density_bin"] = provenance.SOURCE_DERIVED
        result.diagnostics["density_veh_per_km"] = density_veh_per_km
        return result

    pipeline.builder.build = build


def test_the_ego_row_is_the_bounded_speed(dsrc_bundle: str) -> None:
    """decision 2's own identity: the advisory's recommended speed continues
    to mean the number shown to the driver, and equals safety.bounded.speed_mps
    by construction -- checked on a tick where the gate actually changes the
    speed (the forced slow action against a genuinely-evidenced low density,
    per validator round 1's Fix 1), not one where proposed and bounded happen
    to coincide because nothing fired.
    """
    pipeline = _make_pipeline(dsrc_bundle)
    force_action(pipeline, FORCED_SLOW_ACTION_INDEX)
    _force_evidenced_low_density(pipeline, density_veh_per_km=2.0)
    tick = run_ticks(pipeline, 5)

    assert tick.dsrc is not None and tick.dsrc.ego_segment == 0
    assert tick.safety_gate.rules["low_speed_uncongested"].status == RULE_FIRED
    # the gate DID something
    assert tick.safety_gate.proposed_speed_mps != tick.safety_gate.bounded_speed_mps
    row = tick.dsrc.rows[tick.dsrc.ego_segment]
    assert row.recommended_speed_mps == tick.safety_gate.bounded_speed_mps
    record = tick.to_record()
    assert record["advisory"]["recommended_speed_mps"] == record["safety"]["bounded"]["speed_mps"]


def test_only_the_ego_row_is_bounded(dsrc_bundle: str) -> None:
    """The gate bounds what the driver is shown, which is one segment's
    recommendation. Every other row is the controller's own output for a
    stretch of road this vehicle is not on, and the gate has no evidence
    about those stretches to bound them with."""
    pipeline = _make_pipeline(dsrc_bundle)
    force_action(pipeline, FORCED_SLOW_ACTION_INDEX)
    _force_evidenced_low_density(pipeline, density_veh_per_km=2.0)
    tick = run_ticks(pipeline, 5)

    ego = tick.dsrc.ego_segment
    others = [row for i, row in enumerate(tick.dsrc.rows) if i != ego]
    assert len(others) == 11
    assert all(
        row.recommended_speed_mps == pytest.approx(tick.safety_gate.proposed_speed_mps)
        for row in others
    ), "an unbounded row moved"


def test_no_ego_row_means_no_gate_ran(dsrc_bundle: str) -> None:
    """Off the network, there is no recommendation to bound. The tick records
    `safety: None` rather than a gate that found nothing to do, which are
    different facts."""
    pipeline = _make_pipeline(dsrc_bundle)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    t_mono = time.monotonic() - 0.01
    frame = Frame(image=image, frame_id=0, t_mono=t_mono, t_wall=time.time())
    # Far from every Mainz segment, so no fix matches and ego_segment is None.
    fix = GpsFix(
        valid=True, lat=51.49, lon=-0.20, speed_mps=27.0, heading_deg=90.0,
        fix_quality=1, num_sats=9, hdop=0.9, altitude_m=3.0,
        utc_epoch_s=time.time(), t_mono=t_mono, t_wall=time.time(),
    )
    tick = pipeline.step(frame, fix, detections_override=[],
                         here_feed_source=_covering_feed(t_mono))

    assert tick.dsrc is not None and tick.dsrc.ego_segment is None
    assert tick.safety_gate is None
    record = tick.to_record()
    assert record["safety"] is None
    assert record["advisory"] is None


def test_safety_enabled_false_restores_bounded_equals_proposed(dsrc_bundle: str) -> None:
    """validator round 1, Fix 3: `safety.enabled` is the rollback for the
    whole gate. The forced slow action against a genuinely-evidenced low
    density is exactly the setup `test_the_ego_row_is_the_bounded_speed` uses
    to prove the gate DOES clamp when enabled; with the gate disabled it must
    not.
    """
    pipeline = _make_pipeline(dsrc_bundle, safety_enabled=False)
    force_action(pipeline, FORCED_SLOW_ACTION_INDEX)
    _force_evidenced_low_density(pipeline, density_veh_per_km=2.0)
    tick = run_ticks(pipeline, 5)

    assert tick.safety_gate.rules["low_speed_uncongested"].status == RULE_FIRED
    assert tick.safety_gate.bounded_speed_mps == tick.safety_gate.proposed_speed_mps
    assert tick.safety_gate.bounded_headway_s == tick.safety_gate.proposed_headway_s
    row = tick.dsrc.rows[tick.dsrc.ego_segment]
    assert row.recommended_speed_mps == tick.safety_gate.proposed_speed_mps


def test_the_safety_enabled_test_fails_against_the_unfixed_pipeline(
    dsrc_bundle: str, monkeypatch,
) -> None:
    """Neuters Fix 3 at the pipeline boundary -- makes `PerceptionPolicyPipeline`
    ignore `safety_enabled` the way it did before this fix (always calling
    `run_safety_gate` with the default `enabled=True`) -- and confirms the
    test above would then fail. Patches `pipeline.run_safety_gate` (the
    name `pipeline.py` binds via `from policy.safety_gate import ...
    run_safety_gate`), not `policy.safety_gate.run_safety_gate`, since that
    is the reference `PerceptionPolicyPipeline.step` actually calls.
    """
    import pipeline as pipeline_module
    from policy.safety_gate import run_safety_gate as real_run_safety_gate

    def ignores_enabled(*args, **kwargs):
        kwargs.pop("enabled", None)
        return real_run_safety_gate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_safety_gate", ignores_enabled)

    pipeline = _make_pipeline(dsrc_bundle, safety_enabled=False)
    force_action(pipeline, FORCED_SLOW_ACTION_INDEX)
    _force_evidenced_low_density(pipeline, density_veh_per_km=2.0)
    tick = run_ticks(pipeline, 5)
    with pytest.raises(AssertionError):
        assert tick.safety_gate.bounded_speed_mps == tick.safety_gate.proposed_speed_mps


def test_safety_block_is_present_and_json_shaped(dsrc_bundle: str) -> None:
    pipeline = _make_pipeline(dsrc_bundle)
    tick = run_ticks(pipeline, 5)
    record = tick.to_record()
    parsed = json.loads(json.dumps(record))
    safety = parsed["safety"]
    assert set(safety) == {
        "proposed", "bounded", "delta_speed_mps", "emergency_override",
        "evaluable", "not_evaluable", "rules", "config",
    }
    assert len(safety["rules"]) == 2
    assert safety["evaluable"] + safety["not_evaluable"] == 2
    # validator round 1, Fix 4 (F3/F4): the gate's own configuration this
    # tick ran under, so a replay tool need not assume its own defaults.
    assert set(safety["config"]) == {
        "enabled", "time_s",
        "min_contextual_speed_mps", "density_max_age_s",
    }


def test_gate_ms_is_always_measured_never_absent(dsrc_bundle: str) -> None:
    """The gate runs the full rule evaluation on every tick that has a row to
    bound, regardless of which rules are not_evaluable -- there is no
    early-return path here the way there is for dsrc_infer, so this stage must
    never be `absent`."""
    pipeline = _make_pipeline(dsrc_bundle)
    tick = run_ticks(pipeline, 5)
    record = tick.to_record()
    assert record["stages"]["gate"]["basis"] == "measured"
    assert record["stages"]["gate"]["ms"] is not None
    assert record["stages"]["gate"]["ms"] >= 0.0


def test_a_closing_leader_makes_forward_ttc_evaluable(dsrc_bundle: str) -> None:
    """The gate's two rules are `low_speed_uncongested` and `forward_ttc`.
    `forward_ttc` reads the leader gap and the relative speed, so with no
    vehicle in front it is not evaluable and cannot move the number. A real
    detection in the ego lane is what gives it something to read.
    """
    from policy.sensing_controller import RULE_NOT_EVALUABLE

    empty_road = run_ticks(_make_pipeline(dsrc_bundle), 5)
    assert empty_road.safety_gate.rules["forward_ttc"].status == RULE_NOT_EVALUABLE

    with_leader = run_ticks(_make_pipeline(dsrc_bundle), 20, with_leader=True)
    assert with_leader.safety_gate.rules["forward_ttc"].status != RULE_NOT_EVALUABLE
