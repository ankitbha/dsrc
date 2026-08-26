"""The four rates, and the rules that move them.

Every rate change in the system originates here: the phone reports and never lowers
a rate on its own. So a defect in this module is a defect in what the whole drive
sampled.
"""

from __future__ import annotations

import pytest

from policy.sensing_controller import (
    ACTIVE_RATES,
    EVENT_ACCEL_MPS2,
    HOLD_S,
    IDLE_RATES,
    MAX_RATE_HZ,
    MIN_RATE_HZ,
    NARROW_MARGIN,
    RAISE_DWELL_S,
    SKIN_HOT_C,
    SKIN_WARM_C,
    Decision,
    Inputs,
    SensingController,
    Trigger,
    disagreement,
)

RATE_KEYS = ("camera_hz", "gps_hz", "imu_hz", "here_hz")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def settled(controller: SensingController, clock: Clock, inputs: Inputs,
            seconds: float = RAISE_DWELL_S + 0.1) -> Decision:
    """Hold one input steady past the dwell, which is what raising requires."""
    controller.decide(inputs)
    clock.advance(seconds)
    return controller.decide(inputs)


def calm(**over) -> Inputs:
    fields = dict(ego_acceleration=0.0, ego_speed=20.0, policy_margin=0.9,
                  feed_congestion=0.1, camera_density_bin=2,
                  thermal_status="nominal", skin_temp_c=30.0)
    fields.update(over)
    return Inputs(**fields)


class TestEveryRateIsSendable:

    def test_no_input_combination_produces_a_rate_the_wire_refuses(self):
        # Zero is refused as out_of_range, and the obvious implementation of "stop
        # the camera" is a rate of zero. Swept across every rule and the hottest
        # backoff, which is where a multiplied rate would go under.
        clock = Clock()
        controller = SensingController(clock=clock)
        combos = [
            calm(),
            calm(ego_acceleration=9.0),
            calm(policy_margin=0.0),
            calm(feed_congestion=1.0, camera_density_bin=0),
            calm(thermal_status="shutdown"),
            calm(thermal_status="shutdown", skin_temp_c=90.0),
            calm(thermal_status=None, skin_temp_c=None),
            Inputs(),
        ]
        for inputs in combos:
            decision = settled(controller, clock, inputs)
            for key in RATE_KEYS:
                hz = decision.rates[key]
                assert MIN_RATE_HZ <= hz <= MAX_RATE_HZ, f"{key}={hz} for {inputs}"
                assert hz > 0.0

    def test_all_four_rates_are_always_present(self):
        decision = SensingController(clock=Clock()).decide(Inputs())
        assert set(decision.rates) == set(RATE_KEYS)


class TestTheTriggerVocabulary:

    def test_every_trigger_produced_is_a_member(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        for inputs in (calm(), calm(ego_acceleration=5.0), calm(policy_margin=0.01),
                       calm(feed_congestion=0.9, camera_density_bin=0),
                       calm(thermal_status="severe"), Inputs()):
            assert settled(controller, clock, inputs).trigger in Trigger.ALL


class TestTheFreeTierTriggersTheExpensiveOnes:

    def test_a_calm_drive_settles_to_idle_and_stays(self):
        # No thrash without evidence: every change costs a sensor re-registration
        # or a camera rebind, and a drive whose rates are noise is not a decision.
        clock = Clock()
        controller = SensingController(clock=clock)
        seen = set()
        for _ in range(20):
            clock.advance(0.5)
            decision = controller.decide(calm())
            seen.add(tuple(decision.rates[k] for k in RATE_KEYS))
        assert len(seen) == 1
        assert controller.decide(calm()).rates["camera_hz"] == IDLE_RATES["camera_hz"]

    def test_acceleration_from_the_free_tier_raises_the_camera(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(ego_acceleration=EVENT_ACCEL_MPS2 + 0.1))

        assert decision.trigger == Trigger.EVENT
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

    def test_one_tick_of_evidence_is_not_enough(self):
        # The dwell. A single tick is noise, and paying a camera rebind for it costs
        # more than the tick was worth.
        clock = Clock()
        controller = SensingController(clock=clock)
        first = controller.decide(calm(ego_acceleration=9.0))
        assert first.rates["camera_hz"] == IDLE_RATES["camera_hz"]

        clock.advance(RAISE_DWELL_S + 0.1)
        assert controller.decide(calm(ego_acceleration=9.0)).rates["camera_hz"] == \
            ACTIVE_RATES["camera_hz"]

    def test_the_raised_rate_holds_after_the_event_passes(self):
        # Asymmetric on purpose: coming down early re-pays the cost of going up.
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))

        clock.advance(1.0)
        during = controller.decide(calm())
        assert during.trigger == Trigger.HOLD
        assert during.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

        clock.advance(HOLD_S + 1.0)
        assert controller.decide(calm()).rates["camera_hz"] == IDLE_RATES["camera_hz"]

    def test_the_free_tier_itself_never_idles_down(self):
        # It is the detector that says when to spend the others. Backing it off to
        # save a tenth of a percent of the stream blinds the controller exactly
        # when it has decided to look less.
        clock = Clock()
        controller = SensingController(clock=clock)
        for inputs in (calm(), calm(ego_acceleration=9.0), calm(thermal_status="shutdown")):
            decision = settled(controller, clock, inputs)
            assert decision.rates["imu_hz"] == IDLE_RATES["imu_hz"]
            assert decision.rates["gps_hz"] == IDLE_RATES["gps_hz"]


class TestPolicyMargin:

    def test_a_confident_policy_does_not_buy_more_samples(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        assert settled(controller, clock, calm(policy_margin=0.95)).trigger == Trigger.IDLE

    def test_a_near_tie_does(self):
        # The policy emits bins, so there is no continuous value near a boundary.
        # The margin IS the proximity: a small one means one input change flips the
        # advice, which is when better inputs are worth paying for.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(policy_margin=NARROW_MARGIN - 0.01))

        assert decision.trigger == Trigger.NARROW_MARGIN
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

    def test_no_margin_yet_is_not_a_narrow_one(self):
        # Before the first inference there is no margin. Treating None as zero would
        # raise every rate on every drive's first tick.
        clock = Clock()
        controller = SensingController(clock=clock)
        assert settled(controller, clock, calm(policy_margin=None)).trigger == Trigger.IDLE


class TestDisagreement:

    def test_jammed_ahead_but_empty_here_is_a_disagreement(self):
        assert disagreement(0.9, 0) is True

    def test_agreement_is_not(self):
        assert disagreement(0.9, 3) is False
        assert disagreement(0.1, 0) is False

    def test_a_missing_view_is_not_a_disagreement(self):
        # One source silent is not two sources contradicting. Raising rates for it
        # would spend samples on every tick before the feed has answered once.
        assert disagreement(None, 0) is False
        assert disagreement(0.9, None) is False

    def test_it_raises_what_resolves_it(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(feed_congestion=0.9, camera_density_bin=0))

        assert decision.trigger == Trigger.DISAGREEMENT
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]


class TestThermalWins:

    def test_backoff_beats_an_input_that_would_raise(self):
        # The only input arguing for less, so it is applied last and wins by
        # construction rather than by ordering luck.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock,
                           calm(ego_acceleration=9.0, thermal_status="severe"))

        assert decision.trigger == Trigger.THERMAL
        assert decision.rates["camera_hz"] < ACTIVE_RATES["camera_hz"]

    def test_skin_temperature_backs_off_before_the_status_moves(self):
        # Measured on the handset: it warmed 5.4 C under camera load while
        # thermal_status stayed `nominal` for all 81 telemetry frames. Waiting for
        # the status is waiting until the phone is already in trouble.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock,
                           calm(thermal_status="nominal", skin_temp_c=SKIN_WARM_C + 1.0))

        assert decision.thermal_scale < 1.0
        assert decision.rates["camera_hz"] < ACTIVE_RATES["camera_hz"]

    def test_hotter_skin_backs_off_further(self):
        clock = Clock()
        warm = settled(controller := SensingController(clock=clock),
                       clock, calm(skin_temp_c=SKIN_WARM_C + 1.0))
        hot = settled(controller, clock, calm(skin_temp_c=SKIN_HOT_C + 1.0))
        assert hot.thermal_scale < warm.thermal_scale

    def test_silence_is_not_nominal(self):
        # A drive that never heard from the phone must not run hot rates on the
        # assumption that no news is cool news.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status=None, skin_temp_c=None))

        assert decision.thermal_scale < 1.0
        assert any("thermal unknown" in r for r in decision.reasons)

    def test_backoff_still_produces_a_sendable_rate(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="shutdown"))
        assert decision.rates["here_hz"] > 0.0
        assert decision.rates["camera_hz"] > 0.0


class TestTheRecord:

    def test_a_decision_says_which_rule_fired_and_why(self):
        # Task 34 attributes rate changes afterwards, and free text would make that
        # a text-mining problem over a drive's logs.
        clock = Clock()
        controller = SensingController(clock=clock)
        record = settled(controller, clock, calm(ego_acceleration=9.0)).to_record()

        assert record["trigger"] == Trigger.EVENT
        assert record["reasons"]
        assert set(record["rates"]) == set(RATE_KEYS)
