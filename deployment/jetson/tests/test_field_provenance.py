"""Task 36's two central claims, proved end to end: a real `ObservationResult`
through `inputs_from` into `SensingController.decide`, not a hand-built
`Inputs`.

1. The accelerometer path provably changes no rate: a substituted
   acceleration and a measured calm one are different records with
   identical rates.
2. The stale-window guard (D6) is the one change in this task that can
   remove a rate raise, and only there: a dropout entered with a frozen
   window whose slope exceeds the event threshold latches the active rates
   before the guard applies, and returns to idle once it does and the hold
   elapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perception.observation_builder import BuilderConfig, ObservationBuilder
from policy.sensing_controller import (
    ACTIVE_RATES,
    HOLD_S,
    IDLE_RATES,
    RULE_FIRED,
    RULE_NOT_EVALUABLE,
    SensingController,
    Trigger,
)
from policy.sensing_loop import inputs_from
from sensors.gps_reader import GpsFix


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeGps:
    lat: float = 51.49
    lon: float = -0.20
    valid: bool = True


@dataclass
class FakePolicy:
    head_probs: dict = field(default_factory=dict)


@dataclass
class FakeTick:
    obs_result: Any
    policy: FakePolicy = field(default_factory=FakePolicy)
    gps: FakeGps = field(default_factory=FakeGps)


def _gps(speed: float, t: float) -> GpsFix:
    return GpsFix(valid=True, lat=51.49, lon=-0.20, speed_mps=speed, heading_deg=90.0,
                 fix_quality=1, num_sats=8, hdop=1.0, t_mono=t, t_wall=0.0)


class FakeTelemetry:
    thermal_status = "nominal"
    skin_temp_c = 25.0


class FakePhone:
    """A phone that reports a cool, nominal status throughout -- so thermal
    backoff (an orthogonal rule) never confounds the rates these tests
    compare against `IDLE_RATES`/`ACTIVE_RATES` directly.
    """

    telemetry = FakeTelemetry()
    telemetry_at_mono = None


class TestTheAccelerometerPathChangesNoRate:
    """`fallback_neutral` acceleration is always exactly 0.0, and 0.0 never
    fires `abs(value) >= EVENT_ACCEL_MPS2` -- so nulling it changes the
    record and cannot change a rate. Asserted here on two real ticks that
    differ only in whether the speed window was fresh.
    """

    def _decide(self, builder: ObservationBuilder, gps: GpsFix, t: float):
        obs_result = builder.build([], gps, t)
        tick = FakeTick(obs_result=obs_result)
        inputs = inputs_from(tick, FakePhone(), now=t)
        controller = SensingController(clock=Clock(t))
        return controller.decide(inputs), obs_result

    def test_derived_zero_and_substituted_zero_produce_equal_decisions(self):
        # Tick A: enough fresh, constant-speed history that the slope is
        # `derived` at exactly 0.0 ("the road was calm").
        calm_builder = ObservationBuilder(BuilderConfig())
        t = 1000.0
        for i in range(5):
            t = 1000.0 + i * 0.1
            calm_builder.build([], _gps(20.0, t), t)
        calm_decision, calm_obs = self._decide(calm_builder, _gps(20.0, t + 0.1), t + 0.1)
        assert calm_obs.field_sources["ego_acceleration"] == "derived"

        # Tick B: no GPS history at all ("the sensor was dead") -- cold
        # start, `fallback_neutral`, substituted 0.0.
        dead_builder = ObservationBuilder(BuilderConfig())
        dead_decision, dead_obs = self._decide(dead_builder, GpsFix(valid=False), 2000.0)
        assert dead_obs.field_sources["ego_acceleration"] == "fallback_neutral"
        assert dead_obs.obs["ego_acceleration"] == 0.0

        calm_record = calm_decision.to_record()
        dead_record = dead_decision.to_record()
        for key in ("rates", "trigger", "rules_fired", "reasons", "thermal_scale", "clamped"):
            assert calm_record[key] == dead_record[key], key
        assert calm_record["here_radius_m"] == dead_record["here_radius_m"]
        # The record differs on exactly the field this task adds.
        assert (calm_record["attribution"]["rules"][Trigger.EVENT]
                != dead_record["attribution"]["rules"][Trigger.EVENT])
        assert calm_record["attribution"]["rules"][Trigger.EVENT]["status"] == "quiet"
        assert dead_record["attribution"]["rules"][Trigger.EVENT]["status"] == "not_evaluable"


class TestTheStaleWindowGuardIsTheOneRateChange:
    """The one behaviour change in this task, asserted directly: a dropout
    entered with a frozen window whose slope exceeds `EVENT_ACCEL_MPS2`
    latches `ACTIVE_RATES` before the guard applies and returns to
    `IDLE_RATES` after `HOLD_S` once it does.
    """

    def test_the_camera_latches_then_releases_across_the_dropout(self):
        cfg = BuilderConfig()
        builder = ObservationBuilder(cfg)
        clock = Clock(1000.0)
        controller = SensingController(clock=clock)

        def step(gps: GpsFix, dt: float):
            clock.advance(dt)
            obs_result = builder.build([], gps, clock.t)
            tick = FakeTick(obs_result=obs_result)
            inputs = inputs_from(tick, FakePhone(), now=clock.t)
            return controller.decide(inputs), obs_result

        # Phase 1: hard braking under fresh GPS, well past the dwell, so the
        # camera is latched active on a real, freshly derived slope.
        speed = 20.0
        decision = obs_result = None
        for _ in range(8):
            decision, obs_result = step(_gps(speed, clock.t + 0.2), 0.2)
            speed -= 3.0 * 0.2
        assert obs_result.field_sources["ego_acceleration"] == "derived"
        assert decision.attribution.rules[Trigger.EVENT].status == RULE_FIRED
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

        # Phase 2: GPS drops out entirely. Within `gps_stale_after_s` of the
        # last fresh fix the window is frozen, not invented -- it is still
        # `derived`, the event rule still correctly fires on it, and the
        # camera stays latched.
        dead_gps = GpsFix(valid=False)
        decision, obs_result = step(dead_gps, 1.0)
        assert obs_result.field_sources["ego_acceleration"] == "derived"
        assert decision.attribution.rules[Trigger.EVENT].status == RULE_FIRED
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

        # Phase 3: past `gps_stale_after_s` (cumulative 2.5 s into the
        # dropout), the guard refuses the frozen window -- this is the one
        # rate-affecting change in the task.
        decision, obs_result = step(dead_gps, 1.5)
        assert obs_result.field_sources["ego_acceleration"] == "fallback_neutral"
        assert decision.attribution.rules[Trigger.EVENT].status == RULE_NOT_EVALUABLE
        assert decision.attribution.rules[Trigger.EVENT].missing == ("ego_acceleration",)

        # Phase 4: once the hold from the last true evidence elapses with no
        # other raise rule firing, the rate returns to idle.
        decision, obs_result = step(dead_gps, HOLD_S + 1.0)
        assert decision.rates["camera_hz"] == IDLE_RATES["camera_hz"]
        assert decision.trigger == Trigger.IDLE
