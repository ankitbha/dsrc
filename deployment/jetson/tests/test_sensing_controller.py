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
    JAMMED_CONGESTION,
    EMPTY_DENSITY_BIN,
    MAX_POSITION_AGE_S,
    MAX_TELEMETRY_AGE_S,
    MIN_QUERY_RADIUS_M,
    MAX_QUERY_RADIUS_M,
    MAX_EVIDENCE_GAP_S,
    MAX_RATE_HZ,
    MIN_RATE_HZ,
    NARROW_MARGIN,
    RAISE_DWELL_S,
    RULES,
    RULE_FIRED,
    RULE_NOT_EVALUABLE,
    RULE_QUIET,
    SKIN_HOT_C,
    SKIN_HYSTERESIS_C,
    SKIN_WARM_C,
    THERMAL_CAUSE_NO_TELEMETRY,
    THERMAL_CAUSE_SKIN_HOT,
    THERMAL_CAUSE_SKIN_WARM,
    THERMAL_CAUSE_STALE_TELEMETRY,
    THERMAL_CAUSE_STATUS,
    THERMAL_CAUSE_UNSTAMPED_TELEMETRY,
    THERMAL_CAUSES,
    THERMAL_SCALE,
    THERMAL_SCALED_KEYS,
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
    # `telemetry_age_s` defaults to a known-fresh age: `calm()` means a
    # nominal drive with a real, current telemetry report, not one whose
    # age happens to be unknown -- a caller testing the unstamped-telemetry
    # path passes `telemetry_age_s=None` explicitly.
    fields = dict(ego_acceleration=0.0, ego_speed=20.0, policy_margin=0.9,
                  feed_congestion=0.1, camera_density_bin=2,
                  thermal_status="nominal", skin_temp_c=30.0, telemetry_age_s=0.5)
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

        # The rates are what "beats" means, and they are cut.
        assert decision.rates["camera_hz"] < ACTIVE_RATES["camera_hz"]
        # But the one word does not say `thermal_backoff` while the camera tripled.
        # It names the rule that moved the rates off idle; both are in rules_fired,
        # which is what task 34 attributes on.
        assert decision.trigger == Trigger.EVENT
        assert Trigger.THERMAL in decision.rules_fired
        assert Trigger.EVENT in decision.rules_fired

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

    def test_the_attribution_record_has_the_shape_a_reader_can_rely_on(self):
        # `to_record()` is the only artifact a later reader has -- `run_demo`
        # spreads it into the tick log and nothing else carries `decision.attribution`
        # forward. Round-tripping through it and checking the emitted shape is the
        # only way a key silently dropped from `to_record()` would be caught.
        import json

        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(ego_acceleration=9.0))
        record = decision.to_record()

        attribution = record["attribution"]
        assert set(attribution) == {"first_decision", "rules", "gates", "per_sensor"}
        # The fixture's second `decide()` is not the controller's first, so this is
        # only true because `to_record()` copies the field rather than recomputing
        # it -- a copy that goes stale reports the opposite of what happened.
        assert attribution["first_decision"] is False
        # `rules` is transformed through `RuleCheck.to_record()` on the way out, so
        # this is the only place these hold at the record level rather than on the
        # `Attribution` object a reader of the tick log never sees.
        assert set(attribution["rules"]) == set(RULES)
        assert all(r["status"] in (RULE_FIRED, RULE_QUIET, RULE_NOT_EVALUABLE)
                   for r in attribution["rules"].values())
        assert record["rules_fired"] == [n for n in RULES
                                         if attribution["rules"][n]["status"] == RULE_FIRED]
        # `gates` and `per_sensor` are the same objects here as on the dataclass, so
        # the assertions elsewhere that read them off `decision.attribution` cover the
        # emitted record too. That is only true while they are passed by reference: a
        # defensive copy would leave the record side of both silently unasserted, and
        # this is what fails on the day someone writes one.
        assert record["attribution"]["gates"] is decision.attribution.gates
        assert record["attribution"]["per_sensor"] is decision.attribution.per_sensor
        assert set(attribution["gates"]) == {
            "wants_more", "gapped", "dwell", "hold", "bridged", "level",
        }
        assert set(attribution["per_sensor"]) == set(RATE_KEYS)
        one_sensor = attribution["per_sensor"]["camera_hz"]
        assert set(one_sensor) == {
            "hz", "base_hz", "level_sensitive", "thermal_exempt", "scale",
            "clamped", "previous_hz", "changed",
        }
        json.dumps(record)


class TestEvidenceNeverLowersRates:
    """More evidence must never produce lower rates.

    Re-arming the dwell on fresh evidence cancelled a hold that was already
    running, so a hard-braking event 1.4 s into a valid 5 s hold took the camera
    from 5 Hz to 1 Hz. With a signal straddling the threshold it produced a camera
    rebind per tick -- the exact thrash the dwell and hold were built to prevent,
    produced by them.
    """

    def test_a_fresh_event_during_a_hold_does_not_drop_the_rates(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))
        assert controller.decide(calm()).rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

        clock.advance(1.2)
        during_hold = controller.decide(calm(ego_acceleration=9.0))
        assert during_hold.rates["camera_hz"] == ACTIVE_RATES["camera_hz"], (
            "a new event mid-hold lowered the rates it should have kept up"
        )

    def test_a_straddling_signal_does_not_rebind_the_camera_every_tick(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))

        seen = set()
        for i in range(40):
            clock.advance(0.1)
            accel = 9.0 if i % 2 else 0.0
            seen.add(controller.decide(calm(ego_acceleration=accel)).rates["camera_hz"])
        assert seen == {ACTIVE_RATES["camera_hz"]}, f"camera rate oscillated across {seen}"


class TestTheTriggerDescribesTheDecision:

    def test_a_raise_rule_is_not_reported_while_the_rates_are_idle(self):
        # During the dwell the rates are the idle set. Naming a raise rule there
        # made a drive log show an event on the tick the camera did not move.
        controller = SensingController(clock=Clock())
        first = controller.decide(calm(ego_acceleration=9.0))

        assert first.rates["camera_hz"] == IDLE_RATES["camera_hz"]
        assert first.trigger == Trigger.IDLE
        # The rule still fired, and is not lost.
        assert Trigger.EVENT in first.rules_fired

    def test_every_rule_that_fired_is_recorded_not_just_the_last(self):
        # Several rules fire at once and the wire carries one word. Resolving by
        # source order silently masked two of three, leaving them in free text --
        # which is what the closed vocabulary was introduced to avoid.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(
            ego_acceleration=9.0, policy_margin=0.01,
            feed_congestion=0.9, camera_density_bin=0, thermal_status="moderate",
        ))

        for rule in (Trigger.EVENT, Trigger.NARROW_MARGIN,
                     Trigger.DISAGREEMENT, Trigger.THERMAL):
            assert rule in decision.rules_fired
        assert all(r in Trigger.ALL for r in decision.rules_fired)

    def test_thermal_claims_the_word_only_when_it_is_the_whole_story(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        assert settled(controller, clock, calm(thermal_status="severe")).trigger == Trigger.THERMAL


class TestBackoffReachesTheModalityItIsFor:

    def test_here_backs_off_at_idle_rather_than_being_clamped_up(self):
        # HERE's sample is a cellular HTTP call, so it is the modality backoff most
        # wants to cut -- and the floor sat exactly at its idle rate, so every
        # backoff was clamped straight back up and here_hz was byte-identical
        # across all eight thermal statuses.
        clock = Clock()
        controller = SensingController(clock=clock)
        seen = {}
        for status in ("nominal", "moderate", "severe", "shutdown"):
            clock.advance(HOLD_S + 1.0)
            seen[status] = settled(controller, clock, calm(thermal_status=status)).rates["here_hz"]

        assert len(set(seen.values())) == 4, f"here_hz did not vary with thermal: {seen}"
        assert seen["shutdown"] < seen["severe"] < seen["moderate"] < seen["nominal"]

    def test_the_scale_reproduces_the_emitted_rates(self):
        # Otherwise `thermal_scale` is a false statement about what was sent.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="severe"))

        for key in ("camera_hz", "here_hz"):
            if key not in decision.clamped:
                assert decision.rates[key] == pytest.approx(
                    IDLE_RATES[key] * decision.thermal_scale
                )

    def test_a_bound_floor_is_recorded_rather_than_silent(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="shutdown"))
        assert isinstance(decision.clamped, list)


class TestStaleTelemetry:

    def test_a_report_that_has_aged_out_is_no_report(self):
        # The phone's telemetry thread can die while the session stays healthy -- it
        # has, on Android 10. One `nominal` from minutes ago would otherwise pin
        # full rates for the rest of the drive, which is the "no news is cool news"
        # reading this controller refuses.
        from policy.sensing_controller import MAX_TELEMETRY_AGE_S

        clock = Clock()
        controller = SensingController(clock=clock)
        fresh = settled(controller, clock,
                        calm(thermal_status="nominal", telemetry_age_s=1.0))
        assert fresh.thermal_scale == 1.0

        clock.advance(HOLD_S + 1.0)
        stale = settled(controller, clock,
                        calm(thermal_status="nominal",
                             telemetry_age_s=MAX_TELEMETRY_AGE_S + 1.0))
        assert stale.thermal_scale < 1.0
        assert any("stale" in r for r in stale.reasons)

    def test_a_backoff_always_says_why(self):
        # A 40% rate cut with an empty reasons list is not a decision anyone can
        # audit. Reachable when a skin reading arrives without a status.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock,
                           Inputs(thermal_status=None, skin_temp_c=20.0))
        assert decision.thermal_scale < 1.0
        assert decision.reasons


class TestTheQueryGoesDownWithTheRate:

    def test_a_slower_rate_asks_about_more_road(self):
        # Asking less often means each answer has to cover more of the road ahead,
        # because the vehicle travels further between responses.
        clock = Clock()
        controller = SensingController(clock=clock)
        idle = settled(controller, clock, calm(lat=51.49, lon=-0.20, ego_speed=25.0))
        clock.advance(HOLD_S + 1.0)
        active = settled(controller, clock,
                         calm(lat=51.49, lon=-0.20, ego_speed=25.0, ego_acceleration=9.0))

        assert idle.here_query is not None and active.here_query is not None
        assert idle.rates["here_hz"] < active.rates["here_hz"]
        assert idle.here_query.radius_m > active.here_query.radius_m

    def test_no_fix_means_no_query_rather_than_a_query_about_nowhere(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        assert settled(controller, clock, calm(lat=None, lon=None)).here_query is None

    def test_the_query_is_the_shape_the_phone_turns_into_a_url(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        query = settled(controller, clock, calm(lat=51.49, lon=-0.20)).here_query

        assert query.in_.startswith("circle:")
        assert ";r=" in query.in_
        assert query.location_ref == "shape"


class TestNoPositionIsSpelledNaN:
    """`GpsFix` says "no fix" with NaN, not None.

    `GpsFix.lat`/`.lon` default to NaN and `PhoneGpsReader` writes `_or_nan(...)`
    for every field the phone omits, so a dropout, a cold start or a tunnel arrives
    as NaN. A `is None` guard let it through and built `circle:nan,nan`.

    The cost is not the wasted call: the encoder turns NaN into null, the decoder
    refuses a null lat, and `MessageRouter.send` validates by round-tripping and
    RAISES rather than dropping -- so one NaN fix takes all four rates down with it
    and the controller stops commanding anything for the length of the dropout.
    """

    def controller(self):
        return SensingController(clock=Clock())

    def test_a_nan_fix_builds_no_query(self):
        decision = self.controller().decide(calm(lat=float("nan"), lon=float("nan")))
        assert decision.here_query is None

    def test_an_infinite_or_out_of_range_fix_builds_no_query(self):
        for lat, lon in ((float("inf"), 0.0), (0.0, float("-inf")),
                         (91.0, 0.0), (0.0, 181.0)):
            assert self.controller().decide(calm(lat=lat, lon=lon)).here_query is None

    def test_every_query_this_controller_builds_survives_the_wire(self):
        # The property that matters, asserted against the real codec rather than
        # against my idea of it.
        from transport.channels import Channel
        from transport.messages import RateCommand, decode_message

        clock = Clock()
        controller = SensingController(clock=clock)
        for lat, lon in ((51.49, -0.20), (float("nan"), float("nan")),
                         (0.0, 0.0), (-89.9, 179.9), (float("inf"), 2.0)):
            clock.advance(HOLD_S + 1.0)
            decision = settled(controller, clock, calm(lat=lat, lon=lon))
            command = RateCommand(t_capture_mono_ns=1, rates=decision.rates,
                                  trigger=decision.trigger, shadow=True,
                                  here=decision.here_query)
            extensions, payload = command.to_wire()
            decode_message(Channel.RATE_CMD, extensions, payload)


class TestBackoffDoesNotCancelItself:

    def test_the_query_does_not_grow_as_the_rate_is_cut(self):
        # Cut the call count 6.7x and grow the area 45x and the phone's cellular
        # bytes, decode work and heat are unchanged or worse -- the axis the backoff
        # exists to relieve. Backing off means seeing less of the road, not asking
        # one enormous question instead of several small ones.
        clock = Clock()
        controller = SensingController(clock=clock)
        radii = {}
        for status in ("nominal", "moderate", "severe", "critical"):
            clock.advance(HOLD_S + 1.0)
            decision = settled(controller, clock, calm(lat=51.49, lon=-0.20,
                                                       ego_speed=25.0,
                                                       thermal_status=status))
            radii[status] = decision.here_query.radius_m

        assert len(set(radii.values())) == 1, f"the radius grew with the backoff: {radii}"


class TestTheHoldBoundary:

    def test_evidence_returning_late_in_a_hold_does_not_drop_the_rate(self):
        # Same class as the round-1 defect, narrowed to the last RAISE_DWELL_S of
        # every hold: the rate stayed up via `holding`, then fell to idle the moment
        # the hold lapsed, with the evidence continuously present.
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))

        clock.advance(HOLD_S - 0.4)
        controller.decide(calm())
        seen = set()
        for _ in range(5):
            clock.advance(0.2)
            seen.add(controller.decide(calm(ego_acceleration=9.0)).rates["camera_hz"])
        assert seen == {ACTIVE_RATES["camera_hz"]}, f"the rate moved across {seen}"

    def test_a_blip_that_never_dwells_does_not_latch_the_rate_high(self):
        # The trap in the obvious repair. Refreshing the hold on evidence that has
        # not dwelled holds the camera at 5 Hz forever on one blip every four
        # seconds; the bridge has to decay instead.
        clock = Clock()
        controller = SensingController(clock=clock)
        seen = set()
        for i in range(300):
            clock.advance(0.2)
            seen.add(controller.decide(
                calm(ego_acceleration=9.0 if i % 20 == 0 else 0.0)
            ).rates["camera_hz"])
        assert seen == {IDLE_RATES["camera_hz"]}, f"a blip lifted the rates: {seen}"


def test_a_future_telemetry_stamp_is_not_fresh(monkeypatch):
    # The fourth freshness predicate in this codebase. `PhoneGpsReader.is_stale`
    # states the rule and the other three follow it.
    from policy.sensing_controller import MAX_TELEMETRY_AGE_S

    clock = Clock()
    controller = SensingController(clock=clock)
    decision = settled(controller, clock,
                       calm(thermal_status="nominal",
                            telemetry_age_s=-(MAX_TELEMETRY_AGE_S + 5.0)))
    assert decision.thermal_scale < 1.0


def test_a_rate_pulled_up_by_the_floor_is_recorded(monkeypatch):
    # `clamped` is structurally unreachable with today's constants -- the lowest
    # producible rate is 7.5x the floor -- so the bookkeeping is exercised against a
    # raised floor rather than asserted to be a list and left to rot.
    import policy.sensing_controller as sc

    monkeypatch.setattr(sc, "MIN_RATE_HZ", 1.0)
    clock = Clock()
    controller = sc.SensingController(clock=clock)
    decision = settled(controller, clock, calm(thermal_status="shutdown"))

    assert "here_hz" in decision.clamped
    assert decision.rates["here_hz"] == 1.0


class TestTheBridgeSurvivesThermalScaling:
    """The bridge must not stop working because the phone is hot.

    Deriving "was active" from `camera_hz > IDLE_RATES["camera_hz"]` was exact only
    while the thermal scale stayed above 0.2: at `critical` an ACTIVE camera is
    5.0 x 0.15 = 0.75, BELOW the unscaled idle of 1.0, so the proxy read False and
    the hold-boundary defect returned -- on a phone at critical, which is the one
    moment a spurious rebind is least affordable.
    """

    def drive_across_the_hold_boundary(self, status):
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0, thermal_status=status))
        clock.advance(HOLD_S - 0.4)
        controller.decide(calm(thermal_status=status))
        seen = []
        for _ in range(5):
            clock.advance(0.2)
            seen.append(controller.decide(
                calm(ego_acceleration=9.0, thermal_status=status)).rates["camera_hz"])
        return seen

    @pytest.mark.parametrize("status", ["nominal", "moderate", "severe",
                                        "critical", "emergency", "shutdown"])
    def test_the_rate_holds_across_the_boundary_at_every_thermal_level(self, status):
        seen = self.drive_across_the_hold_boundary(status)
        assert len(set(seen)) == 1, f"{status}: the rate moved across {seen}"


class TestABridgedTickNamesItself:

    def test_a_bridged_tick_does_not_report_idle_at_five_hertz(self):
        # The round-2 statement inverted: then a raise word sat on idle rates, now
        # an idle word would sit on raised ones. Task 34 attributes on this field
        # either way.
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))
        clock.advance(HOLD_S - 0.4)
        controller.decide(calm())

        for _ in range(5):
            clock.advance(0.2)
            decision = controller.decide(calm(ego_acceleration=9.0))
            if decision.rates["camera_hz"] > IDLE_RATES["camera_hz"]:
                assert decision.trigger != Trigger.IDLE, (
                    f"idle word on {decision.rates['camera_hz']} Hz"
                )

    def test_the_word_and_the_rates_agree_across_a_long_mixed_drive(self):
        # Both directions, since checking only one is how the inverse shape got in.
        import random

        rng = random.Random(11)
        clock = Clock()
        controller = SensingController(clock=clock)
        for _ in range(4000):
            clock.advance(rng.choice([0.1, 0.2, 0.5, 1.0]))
            decision = controller.decide(calm(
                ego_acceleration=rng.choice([0.0, 0.5, 9.0]),
                policy_margin=rng.choice([0.9, 0.01]),
                thermal_status=rng.choice(["nominal", "moderate", "critical"]),
            ))
            raised = decision.rates["camera_hz"] > IDLE_RATES["camera_hz"] * decision.thermal_scale
            if raised:
                assert decision.trigger != Trigger.IDLE
            else:
                assert decision.trigger not in {Trigger.EVENT, Trigger.NARROW_MARGIN,
                                                Trigger.DISAGREEMENT, Trigger.HOLD}
            assert decision.trigger in Trigger.ALL


class TestAFixTheProtocolCallsMeaningless:

    def test_an_invalid_fix_buys_no_cellular_call(self):
        # The wire lets an invalid fix carry whatever the receiver had, so
        # coordinates alone do not mean there is a position. `HereFeed.at` already
        # refuses exactly this as UNUSABLE_FIX; this guard had two of its three
        # conditions.
        controller = SensingController(clock=Clock())
        decision = controller.decide(calm(lat=52.5, lon=13.4, position_valid=False))
        assert decision.here_query is None

    def test_a_stale_fix_does_not_centre_a_query_on_the_tunnel_entrance(self):
        from policy.sensing_controller import MAX_POSITION_AGE_S

        controller = SensingController(clock=Clock())
        fresh = controller.decide(calm(lat=52.5, lon=13.4, position_age_s=0.5))
        assert fresh.here_query is not None

        stale = controller.decide(
            calm(lat=52.5, lon=13.4, position_age_s=MAX_POSITION_AGE_S + 1.0))
        assert stale.here_query is None

    def test_a_fix_from_this_clocks_future_is_not_fresh(self):
        from policy.sensing_controller import MAX_POSITION_AGE_S

        controller = SensingController(clock=Clock())
        decision = controller.decide(
            calm(lat=52.5, lon=13.4, position_age_s=-(MAX_POSITION_AGE_S + 5.0)))
        assert decision.here_query is None


def test_the_bridge_does_not_span_a_gap_between_ticks():
    # `_last_active` is the previous DECISION, however old. After a stalled or
    # event-driven caller, one tick of evidence would otherwise raise the camera
    # with no dwell at all.
    clock = Clock()
    controller = SensingController(clock=clock)
    settled(controller, clock, calm(ego_acceleration=9.0))

    # Evidence stops, which clears the dwell but leaves the last decision active.
    clock.advance(0.2)
    assert controller.decide(calm()).rates["camera_hz"] == ACTIVE_RATES["camera_hz"]

    # An hour passes with no ticks at all, then evidence returns for one tick. The
    # dwell restarts from here, so nothing has held for long enough -- and the
    # bridge must not treat an hour-old decision as "the previous tick".
    clock.advance(3600.0)
    assert controller.decide(calm(ego_acceleration=9.0)).rates["camera_hz"] == \
        IDLE_RATES["camera_hz"]


class TestAStallIsNotDwellTime:
    """The dwell measures whether evidence PERSISTED, across a stretch we watched."""

    def evidence(self):
        return Inputs(ego_acceleration=9.0, ego_speed=20.0, policy_margin=0.9,
                      camera_density_bin=2, thermal_status="nominal",
                      skin_temp_c=30.0, lat=51.49, lon=-0.20,
                      position_age_s=0.4, telemetry_age_s=1.0)

    def test_a_redial_length_gap_is_not_credited_as_dwell(self):
        # One tick of hard braking, a 120 s rebind, one more tick. Nothing resets the
        # controller on a redial and `run_demo`'s worker `continue`s without calling
        # it for as long as the rebind takes, so a 120 s absence of evidence was read
        # as 120 s of held evidence and raised the camera outright -- and armed a 5 s
        # hold behind it.
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(self.evidence())
        clock.advance(120.0)

        decision = controller.decide(self.evidence())
        assert decision.rates["camera_hz"] == IDLE_RATES["camera_hz"]
        assert decision.trigger == Trigger.IDLE

    def test_the_guard_does_not_fire_on_a_normal_tick_at_any_rate(self):
        # The bound cannot be `RAISE_DWELL_S`: an idle tick is already 1 s apart, so
        # resetting on any gap wider than the dwell resets on every normal tick and
        # the rates can never rise at all. These are the two slowest cadences this
        # controller can itself command.
        for gap in (1.0 / IDLE_RATES["camera_hz"],
                    1.0 / (IDLE_RATES["camera_hz"] * min(THERMAL_SCALE.values()))):
            clock = Clock()
            controller = SensingController(clock=clock)
            controller.decide(self.evidence())
            clock.advance(gap)
            decision = controller.decide(self.evidence())
            assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"], (
                f"a normal {gap:.1f}s tick was treated as a stall"
            )

    def test_the_bound_covers_the_slowest_rate_this_controller_can_command(self):
        # Derived, not typed. A typed constant stops covering the idle rate the
        # moment either input changes, silently.
        slowest_tick_s = 1.0 / (IDLE_RATES["camera_hz"] * min(THERMAL_SCALE.values()))
        assert MAX_EVIDENCE_GAP_S > slowest_tick_s


class TestBrakingIsAnEvent:
    """Every acceleration in this file was positive, so half the rule was untested."""

    def test_hard_braking_raises_the_rates(self):
        # `abs(ego_acceleration) >= EVENT_ACCEL_MPS2`. Dropping the `abs()` left the
        # whole suite green, because no test ever planted a negative value -- and
        # braking is the event this module's own comments are written about. 9.0 m/s2
        # appears throughout as "an event"; it is ~0.9 g, outside any real vehicle's
        # envelope and on the wrong side of zero.
        clock = Clock()
        controller = SensingController(clock=clock)
        braking = Inputs(ego_acceleration=-3.5, ego_speed=20.0, policy_margin=0.9,
                         camera_density_bin=2, thermal_status="nominal",
                         skin_temp_c=30.0, lat=51.49, lon=-0.20,
                         position_age_s=0.4, telemetry_age_s=1.0)
        controller.decide(braking)
        clock.advance(1.0)
        decision = controller.decide(braking)

        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]
        assert Trigger.EVENT in decision.rules_fired

    def test_braking_and_accelerating_are_treated_alike(self):
        # The magnitude is the signal; the sign is not.
        for accel in (-EVENT_ACCEL_MPS2 - 0.1, EVENT_ACCEL_MPS2 + 0.1):
            clock = Clock()
            controller = SensingController(clock=clock)
            inputs = Inputs(ego_acceleration=accel, ego_speed=20.0, policy_margin=0.9,
                            camera_density_bin=2, thermal_status="nominal",
                            skin_temp_c=30.0, lat=51.49, lon=-0.20,
                            position_age_s=0.4, telemetry_age_s=1.0)
            controller.decide(inputs)
            clock.advance(1.0)
            assert Trigger.EVENT in controller.decide(inputs).rules_fired, (
                f"acceleration {accel} did not fire the event rule"
            )


class TestTheConstantsAreValuesNotSelfReferences:
    """A threshold compared only against itself can be changed by 10x in silence."""

    def test_each_gate_is_pinned_to_its_value(self):
        # Every one of these survived a ~10x change with the suite green, because the
        # tests that mention them compare against `module.CONSTANT +/- 1` rather than
        # against a number. `test_every_timebase_constant_in_the_spec_matches_the_code`
        # is the counter-example that shows the fix is cheap.
        #
        # These are decisions, not arithmetic: how stale a fix may be before a query
        # is refused, how long a hold lasts, how far ahead a query reaches. Changing
        # one is a change of behaviour and should have to be written down twice.
        assert MAX_TELEMETRY_AGE_S == 10.0
        assert MAX_POSITION_AGE_S == 2.0
        assert RAISE_DWELL_S == 0.5
        assert HOLD_S == 5.0
        assert EVENT_ACCEL_MPS2 == 1.5
        assert NARROW_MARGIN == 0.15
        assert SKIN_WARM_C == 40.0
        assert SKIN_HOT_C == 45.0
        # Pinned, not derived: the dead band is sized against the handset's own
        # sensor noise (p95 0.087 C, max step 0.369 C over 900 samples), so a
        # change to it is a claim about the sensor and should be argued, not typed.
        assert SKIN_HYSTERESIS_C == 1.0
        assert MIN_RATE_HZ == 0.001
        assert MAX_RATE_HZ == 1000.0
        assert MIN_QUERY_RADIUS_M == 500.0
        assert MAX_QUERY_RADIUS_M == 10_000.0
        assert IDLE_RATES == {"camera_hz": 1.0, "gps_hz": 1.0,
                              "imu_hz": 50.0, "here_hz": 0.05}
        assert ACTIVE_RATES == {"camera_hz": 5.0, "gps_hz": 1.0,
                                "imu_hz": 50.0, "here_hz": 0.2}
        assert THERMAL_SCALE == {"nominal": 1.0, "light": 1.0, "moderate": 0.6,
                                 "severe": 0.3, "critical": 0.15, "emergency": 0.15,
                                 "shutdown": 0.15, "unknown": 0.6}

    def test_the_position_gate_agrees_with_the_reader_that_produces_it(self):
        # `MAX_POSITION_AGE_S`'s docstring says it and `PhoneGpsReader.is_stale` are
        # "answering the same question about the same reading". Asserted, so the two
        # cannot drift apart silently.
        from sensors.phone_source import PhoneGpsReader

        import inspect
        default = inspect.signature(PhoneGpsReader.__init__).parameters["stale_after_s"]
        assert default.default == MAX_POSITION_AGE_S


class TestThreeStatesAreThreeStates:
    """`rules_fired` alone cannot tell a rule that fired from one that ran and found
    nothing, or from one that never had its input. `attribution.rules` can.
    """

    def test_event_fired_carries_its_signed_value_and_threshold(self):
        decision = SensingController(clock=Clock()).decide(calm(ego_acceleration=9.0))
        check = decision.attribution.rules[Trigger.EVENT]
        assert check.status == RULE_FIRED
        assert check.to_record() == {"status": "fired", "value": 9.0,
                                     "threshold": EVENT_ACCEL_MPS2,
                                     "ego_acceleration_source": None}

    def test_event_quiet_at_zero(self):
        decision = SensingController(clock=Clock()).decide(calm())
        check = decision.attribution.rules[Trigger.EVENT]
        assert check.status == RULE_QUIET
        assert check.to_record() == {"status": "quiet", "value": 0.0,
                                     "threshold": EVENT_ACCEL_MPS2,
                                     "ego_acceleration_source": None}

    def test_margin_not_evaluable_before_the_first_inference(self):
        decision = SensingController(clock=Clock()).decide(calm(policy_margin=None))
        check = decision.attribution.rules[Trigger.NARROW_MARGIN]
        assert check.to_record() == {"status": "not_evaluable", "missing": ["policy_margin"]}

    def test_event_not_evaluable_without_an_acceleration(self):
        # The sibling of the margin case above, and the one rule whose
        # `not_evaluable` branch had no assertion. The randomized closure test
        # reaches it but asserts only that the status is in the closed set, and
        # `quiet` is in that set -- so the branch was exercised and unchecked.
        decision = SensingController(clock=Clock()).decide(calm(ego_acceleration=None))
        check = decision.attribution.rules[Trigger.EVENT]
        assert check.to_record() == {"status": "not_evaluable",
                                     "missing": ["ego_acceleration"],
                                     "ego_acceleration_source": None}

    def test_a_named_missing_input_and_a_quiet_status_cannot_coexist(self):
        # `RuleCheck` can represent `{"status": "quiet", "missing": ["x"]}` -- a status
        # contradicting its own evidence, which downstream reads as "the sensor was
        # read and the road was calm". Nothing produces it; nothing forbade it either.
        for name, inputs in (
            (Trigger.EVENT, calm(ego_acceleration=None)),
            (Trigger.NARROW_MARGIN, calm(policy_margin=None)),
            (Trigger.DISAGREEMENT, calm(feed_congestion=None)),
        ):
            check = SensingController(clock=Clock()).decide(inputs).attribution.rules[name]
            record = check.to_record()
            assert ("missing" in record) == (record["status"] == RULE_NOT_EVALUABLE), record

    def test_disagreement_not_evaluable_when_the_feed_is_silent(self):
        # Restates the scripts/remutate.py:824 pin from the record side: a missing
        # view is not_evaluable, never fired, and `rules_fired` still lacks it. Every
        # other rule is quiet on this fixture, so the whole list must come back
        # empty -- not just missing the one name -- or a status of "quiet" being
        # counted as fired would slip through unnoticed.
        decision = SensingController(clock=Clock()).decide(calm(feed_congestion=None))
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.to_record() == {
            "status": "not_evaluable", "missing": ["feed_congestion"],
            "camera_density_bin_source": None, "camera_last_detection_age_s": None,
        }
        assert decision.rules_fired == []

    def test_disagreement_names_both_missing_views(self):
        decision = SensingController(clock=Clock()).decide(
            calm(feed_congestion=None, camera_density_bin=None)
        )
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.missing == ("feed_congestion", "camera_density_bin")

    def test_feed_declined_rides_disagreements_not_evaluable_entry(self):
        decision = SensingController(clock=Clock()).decide(
            calm(feed_congestion=None, feed_declined="feed_stale")
        )
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.to_record() == {
            "status": "not_evaluable", "missing": ["feed_congestion"],
            "camera_density_bin_source": None, "camera_last_detection_age_s": None,
            "feed_declined": "feed_stale",
        }

    def test_feed_declined_is_absent_rather_than_invented_when_unset(self):
        decision = SensingController(clock=Clock()).decide(calm(feed_congestion=None))
        assert "feed_declined" not in decision.attribution.rules[Trigger.DISAGREEMENT].to_record()


class TestRulesFiredIsAnIdentity:

    def test_a_fully_quiet_decision_has_an_empty_rules_fired(self):
        # Every rule ran its comparison and found nothing -- "quiet" must not be
        # counted as "fired" by the derivation, only "not not_evaluable" would let
        # that happen without a single rule actually crossing its threshold.
        decision = SensingController(clock=Clock()).decide(calm())
        for check in decision.attribution.rules.values():
            assert check.status == RULE_QUIET
        assert decision.rules_fired == []

    def test_rules_fired_equals_the_fired_checks_in_rules_order(self):
        # An identity by construction, not a resemblance checked afterwards: both
        # are derived from the same `checks` dict inside `decide`.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(
            ego_acceleration=9.0, policy_margin=0.01,
            feed_congestion=0.9, camera_density_bin=0, thermal_status="moderate",
        ))
        expected = [name for name in RULES
                    if decision.attribution.rules[name].status == RULE_FIRED]
        assert decision.rules_fired == expected
        assert expected == [Trigger.EVENT, Trigger.NARROW_MARGIN,
                            Trigger.DISAGREEMENT, Trigger.THERMAL]


class TestGatesTellDwellFromIdle:

    def test_one_tick_of_evidence_is_fired_but_blocked_by_the_dwell(self):
        # The record now distinguishes "fired, blocked by dwell" from "idle, quiet",
        # which `trigger == "idle"` alone could not.
        decision = SensingController(clock=Clock()).decide(calm(ego_acceleration=9.0))

        assert decision.attribution.rules[Trigger.EVENT].status == RULE_FIRED
        gates = decision.attribution.gates
        assert gates["wants_more"] is True
        assert gates["dwell"]["satisfied"] is False
        assert gates["dwell"]["elapsed_s"] == 0.0
        assert gates["dwell"]["required_s"] == RAISE_DWELL_S
        assert gates["level"] == "idle"
        assert decision.rates["camera_hz"] == IDLE_RATES["camera_hz"]

    def test_no_dwell_running_is_null_not_zero(self):
        decision = SensingController(clock=Clock()).decide(calm())
        assert decision.attribution.gates["wants_more"] is False
        assert decision.attribution.gates["dwell"]["elapsed_s"] is None

    def test_a_holding_tick_has_all_raises_quiet_and_a_positive_remaining(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))

        clock.advance(1.0)
        decision = controller.decide(calm())
        gates = decision.attribution.gates
        assert decision.trigger == Trigger.HOLD
        for rule in (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT):
            assert decision.attribution.rules[rule].status == RULE_QUIET
        assert gates["hold"]["active"] is True
        assert gates["hold"]["remaining_s"] > 0.0

    def test_hold_remaining_is_null_when_not_holding(self):
        decision = SensingController(clock=Clock()).decide(calm())
        assert decision.attribution.gates["hold"]["active"] is False
        assert decision.attribution.gates["hold"]["remaining_s"] is None

    def test_a_bridged_tick_names_itself_bridged(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        settled(controller, clock, calm(ego_acceleration=9.0))
        clock.advance(HOLD_S - 0.4)
        controller.decide(calm())

        clock.advance(0.2)
        decision = controller.decide(calm(ego_acceleration=9.0))
        assert decision.attribution.gates["bridged"] is True

    def test_a_redial_length_gap_reports_gapped_on_the_resuming_tick(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(ego_acceleration=9.0))
        clock.advance(120.0)

        decision = controller.decide(calm(ego_acceleration=9.0))
        assert decision.attribution.gates["gapped"] is True
        assert decision.trigger == Trigger.IDLE


class TestThermalCause:

    def test_status_alone_is_the_cause(self):
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="moderate", skin_temp_c=None)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.status == RULE_FIRED
        assert check.evidence["cause"] == THERMAL_CAUSE_STATUS

    def test_hot_skin_under_a_nominal_status_is_skin_hot(self):
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", skin_temp_c=46.0)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.evidence["cause"] == THERMAL_CAUSE_SKIN_HOT
        assert check.evidence["scale"] == THERMAL_SCALE["severe"]

    def test_stale_telemetry_is_its_own_cause(self):
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", telemetry_age_s=MAX_TELEMETRY_AGE_S + 1.0)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.evidence["cause"] == THERMAL_CAUSE_STALE_TELEMETRY

    def test_a_status_with_no_age_is_unstamped_not_fresh(self):
        """A null `telemetry_age_s` guarded as `age is not None and ...`
        falls through every staleness check straight to
        `"telemetry": "fresh"`, labelling an age that cannot be checked
        against `MAX_TELEMETRY_AGE_S` the same as one that was checked and
        passed.
        """
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", telemetry_age_s=None)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.evidence["cause"] == THERMAL_CAUSE_UNSTAMPED_TELEMETRY
        assert check.evidence["telemetry"] == "unstamped"
        assert check.evidence["scale"] == THERMAL_SCALE["unknown"]

    def test_unstamped_and_stale_are_reached_from_the_same_nominal_status(self):
        """The same `thermal_status="nominal"` reaches three different
        causes depending only on `telemetry_age_s` -- proof the three are
        genuinely distinguished on age, not on the status carried beside it.
        """
        fresh = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", telemetry_age_s=1.0)
        )
        unstamped = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", telemetry_age_s=None)
        )
        stale = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", telemetry_age_s=MAX_TELEMETRY_AGE_S + 1.0)
        )
        assert fresh.attribution.rules[Trigger.THERMAL].evidence["cause"] is None
        assert (unstamped.attribution.rules[Trigger.THERMAL].evidence["cause"]
                == THERMAL_CAUSE_UNSTAMPED_TELEMETRY)
        assert (stale.attribution.rules[Trigger.THERMAL].evidence["cause"]
                == THERMAL_CAUSE_STALE_TELEMETRY)

    def test_total_silence_is_no_telemetry(self):
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status=None, skin_temp_c=None)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.evidence["cause"] == THERMAL_CAUSE_NO_TELEMETRY

    def test_a_min_tie_leaves_the_earlier_cause_standing(self):
        # `severe` status already reaches the scale hot skin would compute, so skin
        # does not strictly lower it and the cause stays with the status.
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="severe", skin_temp_c=46.0)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.evidence["cause"] == THERMAL_CAUSE_STATUS
        assert check.evidence["scale"] == THERMAL_SCALE["severe"]

    def test_a_nominal_scale_is_quiet_with_a_null_cause(self):
        decision = SensingController(clock=Clock()).decide(
            calm(thermal_status="nominal", skin_temp_c=30.0)
        )
        check = decision.attribution.rules[Trigger.THERMAL]
        assert check.status == RULE_QUIET
        assert check.evidence["cause"] is None

    def test_every_cause_ever_produced_is_a_member(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        for status in ("nominal", "moderate", "severe", "critical", None):
            for skin in (None, 30.0, 41.0, 46.0):
                clock.advance(0.1)
                decision = controller.decide(calm(thermal_status=status, skin_temp_c=skin))
                cause = decision.attribution.rules[Trigger.THERMAL].evidence["cause"]
                assert cause is None or cause in THERMAL_CAUSES


class TestDisagreementEvidenceNamesItsConstants:
    """The two literals `disagreement` compares against, echoed beside the values
    they judged rather than left implicit in the boolean it produced.
    """

    def test_fired_evidence_echoes_the_threshold_beside_the_value(self):
        decision = SensingController(clock=Clock()).decide(
            calm(feed_congestion=0.9, camera_density_bin=0)
        )
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.status == RULE_FIRED
        assert check.evidence == {
            "feed_congestion": 0.9, "jammed_congestion": JAMMED_CONGESTION,
            "camera_density_bin": 0, "empty_density_bin": EMPTY_DENSITY_BIN,
            "camera_density_bin_source": None, "camera_last_detection_age_s": None,
        }

    def test_quiet_evidence_carries_the_same_shape(self):
        decision = SensingController(clock=Clock()).decide(calm())
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.status == RULE_QUIET
        assert set(check.evidence) == {"feed_congestion", "jammed_congestion",
                                       "camera_density_bin", "empty_density_bin",
                                       "camera_density_bin_source",
                                       "camera_last_detection_age_s"}


class TestDisagreementStillFiresOnADerivedEmptyBin:
    """D4: the over-report is named, not removed. Under shipped constants the
    rule fires if and only if the camera detected nothing, and gating on
    `derived_empty` would delete its only firing path -- so it still fires,
    and the record now says what the camera's "empty road" claim rested on.
    """

    def test_fires_and_names_the_basis(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(
            feed_congestion=0.9, camera_density_bin=0,
            camera_density_bin_source="derived_empty",
            camera_last_detection_age_s=241.7,
        ))
        check = decision.attribution.rules[Trigger.DISAGREEMENT]
        assert check.status == RULE_FIRED
        assert decision.trigger == Trigger.DISAGREEMENT
        assert decision.rates["camera_hz"] == ACTIVE_RATES["camera_hz"]
        assert check.evidence["camera_density_bin_source"] == "derived_empty"
        assert check.evidence["camera_last_detection_age_s"] == 241.7


class TestPerSensorChain:

    def _assert_reconstructs(self, decision):
        from policy.sensing_controller import _clamp

        for key, entry in decision.attribution.per_sensor.items():
            assert _clamp(entry["base_hz"] * entry["scale"]) == pytest.approx(entry["hz"])
            assert entry["hz"] == decision.rates[key]

    def test_reconstruction_identity_holds_across_a_mixed_sweep(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        scenarios = [
            calm(),
            calm(ego_acceleration=9.0),
            calm(policy_margin=0.01),
            calm(feed_congestion=0.9, camera_density_bin=0),
            calm(thermal_status="severe"),
            calm(thermal_status="critical"),
            calm(thermal_status="shutdown"),
        ]
        for inputs in scenarios:
            decision = settled(controller, clock, inputs)
            self._assert_reconstructs(decision)
            for key in RATE_KEYS:
                entry = decision.attribution.per_sensor[key]
                assert entry["clamped"] == (key in decision.clamped)

    def test_gps_and_imu_are_never_level_sensitive_thermal_exempt_or_changed(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        for inputs in (calm(), calm(ego_acceleration=9.0), calm(thermal_status="shutdown")):
            decision = settled(controller, clock, inputs)
            for key in ("gps_hz", "imu_hz"):
                entry = decision.attribution.per_sensor[key]
                assert entry["level_sensitive"] is False
                assert entry["thermal_exempt"] is True
                assert entry["changed"] is False

    def test_the_first_decision_has_no_previous_and_has_not_changed(self):
        controller = SensingController(clock=Clock())
        decision = controller.decide(calm())
        assert decision.attribution.first_decision is True
        for entry in decision.attribution.per_sensor.values():
            assert entry["previous_hz"] is None
            assert entry["changed"] is False

        # The second call is not a first decision, even though nothing about the
        # inputs changed -- `first_decision` separates "no history yet" from
        # "history exists and agrees", which `changed` alone cannot do.
        second = controller.decide(calm())
        assert second.attribution.first_decision is False
        for entry in second.attribution.per_sensor.values():
            assert entry["previous_hz"] is not None

    def test_a_raise_tick_shows_the_camera_changed_from_idle(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(ego_acceleration=9.0))
        entry = decision.attribution.per_sensor["camera_hz"]
        assert entry["previous_hz"] == IDLE_RATES["camera_hz"]
        assert entry["changed"] is True

    def test_the_floor_agrees_with_decisions_clamped_list(self, monkeypatch):
        import policy.sensing_controller as sc

        monkeypatch.setattr(sc, "MIN_RATE_HZ", 1.0)
        clock = Clock()
        controller = sc.SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="shutdown"))
        entry = decision.attribution.per_sensor["here_hz"]
        assert entry["clamped"] is True
        assert "here_hz" in decision.clamped

    def test_thermal_exempt_matches_thermal_scaled_keys(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="severe"))
        for key, entry in decision.attribution.per_sensor.items():
            assert entry["thermal_exempt"] == (key not in THERMAL_SCALED_KEYS)

    def test_level_sensitive_matches_whether_idle_and_active_rates_differ(self):
        # gps_hz and imu_hz are the free always-on tier and never move with the
        # level, but camera_hz and here_hz do -- 1.0 -> 5.0 and 0.05 -> 0.2. Checked
        # across all four keys, not just the two that happen to be False.
        clock = Clock()
        controller = SensingController(clock=clock)
        decision = settled(controller, clock, calm(thermal_status="severe"))
        for key, entry in decision.attribution.per_sensor.items():
            assert entry["level_sensitive"] == (IDLE_RATES[key] != ACTIVE_RATES[key])


class TestTriggerAgreesWithGates:

    def test_the_trigger_matches_its_gates_branch_across_a_long_mixed_drive(self):
        # Both directions: the branch a decision's own gates block implies is the one
        # `trigger` actually names, over a drive that visits every branch many times.
        import random

        rng = random.Random(29)
        clock = Clock()
        controller = SensingController(clock=clock)
        raise_names = (Trigger.EVENT, Trigger.NARROW_MARGIN, Trigger.DISAGREEMENT)
        for _ in range(3000):
            clock.advance(rng.choice([0.1, 0.2, 0.5, 1.0]))
            decision = controller.decide(calm(
                ego_acceleration=rng.choice([0.0, 0.5, 9.0]),
                policy_margin=rng.choice([0.9, 0.01]),
                feed_congestion=rng.choice([0.1, 0.9]),
                camera_density_bin=rng.choice([2, 0]),
                thermal_status=rng.choice(["nominal", "moderate", "critical"]),
            ))
            gates = decision.attribution.gates
            any_raise_fired = any(decision.attribution.rules[r].status == RULE_FIRED
                                  for r in raise_names)
            thermal_fired = decision.attribution.rules[Trigger.THERMAL].status == RULE_FIRED

            if (gates["level"] == "active"
                    and (gates["dwell"]["satisfied"] or gates["bridged"])
                    and any_raise_fired):
                assert decision.trigger in raise_names
            elif gates["level"] == "active" and gates["hold"]["active"]:
                assert decision.trigger == Trigger.HOLD
            elif thermal_fired:
                assert decision.trigger == Trigger.THERMAL
            else:
                assert decision.trigger == Trigger.IDLE
            assert decision.trigger in Trigger.ALL


class TestAttributionVocabularyClosure:

    def test_every_status_cause_and_level_produced_is_a_member(self):
        import random

        rng = random.Random(7)
        clock = Clock()
        controller = SensingController(clock=clock)
        for _ in range(500):
            clock.advance(rng.choice([0.1, 0.5, 1.0]))
            decision = controller.decide(calm(
                ego_acceleration=rng.choice([0.0, 9.0, None]),
                policy_margin=rng.choice([0.9, 0.01, None]),
                feed_congestion=rng.choice([0.1, 0.9, None]),
                camera_density_bin=rng.choice([2, 0, None]),
                thermal_status=rng.choice(["nominal", "moderate", "critical", None]),
                skin_temp_c=rng.choice([30.0, 46.0, None]),
            ))
            for check in decision.attribution.rules.values():
                assert check.status in (RULE_FIRED, RULE_QUIET, RULE_NOT_EVALUABLE)
            cause = decision.attribution.rules[Trigger.THERMAL].evidence["cause"]
            assert cause is None or cause in THERMAL_CAUSES
            assert decision.attribution.gates["level"] in ("idle", "active")


class TestInputsRoundTrip:
    """`Inputs.to_record`/`from_record` is the replay substrate task 35 scores
    candidates against. A value that survives the round trip inexactly, or a
    schema drift `from_record` lets through silently, is a different replay.
    """

    ALL_FIELD_NAMES = {
        "ego_acceleration", "ego_speed", "policy_margin", "feed_congestion",
        "camera_density_bin", "feed_declined", "thermal_status", "skin_temp_c",
        "telemetry_age_s", "ego_acceleration_source", "ego_speed_source",
        "camera_density_bin_source", "camera_last_detection_age_s",
        "lat", "lon", "position_valid", "position_age_s",
    }

    def _round_trip(self, inputs: Inputs) -> Inputs:
        import json

        return Inputs.from_record(json.loads(json.dumps(inputs.to_record())))

    def test_to_record_has_exactly_the_seventeen_field_names(self):
        assert set(Inputs().to_record()) == self.ALL_FIELD_NAMES

    def test_an_all_none_instance_round_trips(self):
        assert self._round_trip(Inputs()) == Inputs()

    def test_an_all_set_instance_round_trips(self):
        inputs = Inputs(
            ego_acceleration=0.5, ego_speed=13.4, policy_margin=0.2,
            feed_congestion=0.9, camera_density_bin=0, feed_declined="stale",
            thermal_status="moderate", skin_temp_c=41.0, telemetry_age_s=0.4,
            ego_acceleration_source="derived", ego_speed_source="measured",
            camera_density_bin_source="derived_empty",
            camera_last_detection_age_s=12.4,
            lat=37.42, lon=-122.08, position_valid=True, position_age_s=0.2,
        )
        assert self._round_trip(inputs) == inputs

    def test_a_seventeen_significant_digit_float_survives_unchanged(self):
        # Pins D2 against "simplifying" to the evidence path's four-place
        # rounding: a replay built from a rounded copy can flip a threshold
        # comparison this decision never made.
        value = 0.03199999999999998
        inputs = Inputs(policy_margin=value)
        assert self._round_trip(inputs).policy_margin == value

    def test_camera_last_detection_age_s_survives_at_full_precision(self):
        # The same precision guarantee as `policy_margin` above, on the field
        # this task adds: a liveness bound a candidate might threshold on
        # must not be replayed from a rounded copy either.
        value = 12.399999999999999
        inputs = Inputs(camera_last_detection_age_s=value)
        assert self._round_trip(inputs).camera_last_detection_age_s == value

    def test_from_record_refuses_a_missing_key_by_name(self):
        record = Inputs().to_record()
        del record["skin_temp_c"]
        with pytest.raises(ValueError, match="skin_temp_c"):
            Inputs.from_record(record)

    def test_from_record_refuses_an_unknown_key_by_name(self):
        record = Inputs().to_record()
        record["extra_field"] = 1.0
        with pytest.raises(ValueError, match="extra_field"):
            Inputs.from_record(record)


def _mixed_sequence() -> list[tuple[float, Inputs]]:
    """(advance_seconds, inputs) pairs that visit idle, an armed-but-unsatisfied
    dwell, a satisfied dwell, a hold, a bridge across the end of that hold, a
    gap wider than `MAX_EVIDENCE_GAP_S` that resets the dwell, and every
    thermal scale tier -- the shape the replay-identity property has to hold
    across, not just at a single quiet tick.
    """
    return [
        (0.0, calm()),                                           # idle
        (0.2, calm(ego_acceleration=9.0)),                       # armed, not satisfied
        (RAISE_DWELL_S + 0.05, calm(ego_acceleration=9.0)),      # dwell satisfied -> active
        (HOLD_S - 0.01, calm()),                                 # evidence gone, holding
        (0.05, calm(ego_acceleration=9.0)),                      # hold just lapsed, bridged
        (MAX_EVIDENCE_GAP_S + 1.0, calm(ego_acceleration=9.0)),  # gap resets the dwell
        (RAISE_DWELL_S + 0.1, calm(thermal_status="nominal")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="light")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="moderate")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="severe")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="critical")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="emergency")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="shutdown")),
        (RAISE_DWELL_S + 0.1, calm(thermal_status="unknown")),
    ]


class _ReplayClock:
    """Fed by the log instead of by time: set once per tick, held rather than
    popped, exactly what `score_shadow.ReplayClock` does for a candidate."""

    def __init__(self) -> None:
        self._current: float | None = None

    def set(self, value: float) -> None:
        self._current = value

    def __call__(self) -> float:
        return self._current


class TestReplayIdentity:
    """The property `score_shadow.py` rests on: a controller replayed from
    `Inputs.to_record()`/`from_record()` and the logged `decided_at_mono`
    reproduces `to_record()` exactly, tick for tick, with no access to the
    original `Inputs` objects or the original clock.
    """

    def _drive(self, controller: SensingController, clock: Clock,
               sequence: list[tuple[float, Inputs]]) -> list[dict]:
        records = []
        for advance, inputs in sequence:
            clock.advance(advance)
            records.append(controller.decide(inputs).to_record())
        return records

    def _replay(self, sequence: list[tuple[float, Inputs]], records: list[dict]) -> list[dict]:
        import json

        replay_clock = _ReplayClock()
        replay_controller = SensingController(clock=replay_clock)
        replayed = []
        for (_, inputs), record in zip(sequence, records):
            replay_clock.set(record["decided_at_mono"])
            # Round-tripped through JSON, like a real log line -- not the
            # original `Inputs` object -- so this tests what a logged drive
            # can actually reconstruct, not what the test happens to hold.
            from_log = Inputs.from_record(json.loads(json.dumps(inputs.to_record())))
            replayed.append(replay_controller.decide(from_log).to_record())
        return replayed

    def test_a_scripted_mixed_drive_replays_exactly(self):
        sequence = _mixed_sequence()
        clock = Clock()
        controller = SensingController(clock=clock)
        records = self._drive(controller, clock, sequence)

        gates_seen = {"armed_not_satisfied": False, "dwell_satisfied": False,
                      "holding": False, "bridged": False, "gapped": False}
        for record in records:
            gates = record["attribution"]["gates"]
            if gates["wants_more"] and not gates["dwell"]["satisfied"]:
                gates_seen["armed_not_satisfied"] = True
            if gates["dwell"]["satisfied"]:
                gates_seen["dwell_satisfied"] = True
            if gates["hold"]["active"]:
                gates_seen["holding"] = True
            if gates["bridged"]:
                gates_seen["bridged"] = True
            if gates["gapped"]:
                gates_seen["gapped"] = True
        assert all(gates_seen.values()), (
            f"the script never reached: {[k for k, v in gates_seen.items() if not v]}"
        )

        replayed = self._replay(sequence, records)
        for i, (record, replay) in enumerate(zip(records, replayed)):
            assert replay == record, f"tick {i} diverged on replay"

    def test_a_clamp_replays_exactly_too(self, monkeypatch):
        # `clamped` is structurally unreachable with today's constants (the
        # existing floor test raises it to exercise the bookkeeping), and the
        # replay identity has to hold there too: a candidate's clamp math is
        # exactly the incumbent's, so a log that cannot replay a clamped tick
        # cannot referee a candidate that would have clamped differently.
        import policy.sensing_controller as sc

        monkeypatch.setattr(sc, "MIN_RATE_HZ", 1.0)
        sequence = _mixed_sequence() + [(RAISE_DWELL_S + 0.1, calm(thermal_status="shutdown"))]
        clock = Clock()
        controller = sc.SensingController(clock=clock)
        records = self._drive(controller, clock, sequence)
        assert records[-1]["clamped"]

        replayed = self._replay(sequence, records)
        for i, (record, replay) in enumerate(zip(records, replayed)):
            assert replay == record, f"tick {i} diverged on replay"

    def test_a_shifted_decided_at_mono_diverges_at_a_dwell_boundary(self):
        # D3 is load-bearing, not decorative: replaying with a clock read a few
        # hundred microseconds away from the exact `decided_at_mono` -- the gap
        # between the tick loop's own read and the controller's -- can flip a
        # dwell comparison and produce a different decision.
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(ego_acceleration=9.0))
        clock.advance(RAISE_DWELL_S)
        original = controller.decide(calm(ego_acceleration=9.0))
        assert original.attribution.gates["dwell"]["satisfied"] is True

        replay_clock = _ReplayClock()
        replay_controller = SensingController(clock=replay_clock)
        replay_clock.set(1000.0)
        replay_controller.decide(calm(ego_acceleration=9.0))
        replay_clock.set(1000.0 + RAISE_DWELL_S - 300e-6)
        shifted = replay_controller.decide(calm(ego_acceleration=9.0))

        assert shifted.attribution.gates["dwell"]["satisfied"] is False
        assert shifted.rates != original.rates


class TestSkinHysteresis:
    """The dead band on the skin thresholds -- task 57.

    Every temperature here is a reading the handset actually produced. The sequence
    in the first test is verbatim from `run_20260905_142351`, which is the run that
    found the defect: without a dead band it commanded eight camera rate changes in
    thirty seconds, and in live mode each one is a camera rebind on the phone.
    """

    #: The seven readings, in order, that the handset reported while it sat on
    #: SKIN_WARM_C. Under the bare comparison they alternate over and under.
    CHATTER = (39.991, 40.001, 39.994, 40.027, 39.998, 40.005, 39.991)

    def _scales(self, readings):
        clock = Clock()
        controller = SensingController(clock=clock)
        out = []
        for celsius in readings:
            out.append(controller.decide(calm(skin_temp_c=celsius)).thermal_scale)
            clock.advance(1.0)
        return out

    def test_a_reading_resting_on_the_threshold_changes_the_scale_once(self):
        scales = self._scales(self.CHATTER)
        # First reading is under the threshold and has nothing latched, so full rate.
        assert scales[0] == 1.0
        # Every reading after the first crossing stays backed off, including the four
        # that are under 40.0 by hundredths.
        assert scales[1:] == [THERMAL_SCALE["moderate"]] * 6, scales
        changes = sum(1 for a, b in zip(scales, scales[1:]) if a != b)
        assert changes == 1, f"{changes} scale changes over {self.CHATTER}"

    def test_the_bare_comparison_is_what_chattered(self):
        # The control. Without this the test above cannot show the dead band did
        # anything: a sequence that never chattered would pass it unchanged.
        bare = [1.0 if c < SKIN_WARM_C else THERMAL_SCALE["moderate"] for c in self.CHATTER]
        changes = sum(1 for a, b in zip(bare, bare[1:]) if a != b)
        assert changes == 6, bare

    def test_backing_off_is_not_delayed_by_the_dead_band(self):
        # The asymmetry the dead band exists to preserve: entering takes the bare
        # threshold, on the first decision that sees it, with no dwell.
        controller = SensingController(clock=Clock())
        decision = controller.decide(calm(skin_temp_c=SKIN_WARM_C))
        assert decision.thermal_scale == THERMAL_SCALE["moderate"]

    def test_recovery_waits_for_the_full_dead_band(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(skin_temp_c=SKIN_WARM_C + 0.5))
        held = controller.decide(calm(skin_temp_c=SKIN_WARM_C - 0.5))
        assert held.thermal_scale == THERMAL_SCALE["moderate"]
        released = controller.decide(calm(skin_temp_c=SKIN_WARM_C - 1.5))
        assert released.thermal_scale == 1.0

    def test_the_hot_threshold_has_its_own_dead_band(self):
        # Releasing `severe` must land on `moderate`, not on 1.0: the reading is
        # still far above SKIN_WARM_C, and one latch must not release the other.
        clock = Clock()
        controller = SensingController(clock=clock)
        assert controller.decide(calm(skin_temp_c=SKIN_HOT_C)).thermal_scale \
            == THERMAL_SCALE["severe"]
        held = controller.decide(calm(skin_temp_c=SKIN_HOT_C - 0.5))
        assert held.thermal_scale == THERMAL_SCALE["severe"]
        released = controller.decide(calm(skin_temp_c=SKIN_HOT_C - 1.5))
        assert released.thermal_scale == THERMAL_SCALE["moderate"]

    def test_a_latch_survives_a_telemetry_gap(self):
        # Silence is not cooling. A gap must not hand back full rates to a handset
        # that was hot when it last spoke, which is the same reading of silence the
        # `unknown` tier already takes.
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(skin_temp_c=SKIN_WARM_C + 0.5))
        gap = controller.decide(calm(thermal_status=None, skin_temp_c=None))
        assert gap.thermal_scale == THERMAL_SCALE["unknown"]
        back = controller.decide(calm(skin_temp_c=SKIN_WARM_C - 0.5))
        assert back.thermal_scale == THERMAL_SCALE["moderate"]

    def test_every_thermal_evidence_path_carries_both_latches(self):
        # The key set must not vary by path: a census of `skin_warm_latched` should
        # never have to tell "false" from "this path does not say".
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(skin_temp_c=SKIN_WARM_C + 0.5))
        paths = [
            calm(skin_temp_c=SKIN_WARM_C + 0.5),          # fresh
            calm(thermal_status=None, skin_temp_c=None),  # no_telemetry
            calm(telemetry_age_s=None),                   # unstamped
            calm(telemetry_age_s=MAX_TELEMETRY_AGE_S + 1),  # stale
        ]
        for inputs in paths:
            evidence = controller.decide(inputs).attribution.rules[Trigger.THERMAL].evidence
            assert "skin_warm_latched" in evidence, inputs
            assert "skin_hot_latched" in evidence, inputs

    def test_the_reason_does_not_claim_a_crossing_that_did_not_happen(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        controller.decide(calm(skin_temp_c=SKIN_WARM_C + 0.5))
        held = controller.decide(calm(skin_temp_c=SKIN_WARM_C - 0.5))
        reason = " ".join(held.reasons)
        assert "held" in reason and ">=" not in reason, reason
