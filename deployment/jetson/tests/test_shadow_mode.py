"""Shadow and live, and the property that makes shadow logs worth reading.

Task 43 checks that logged shadow decisions match what live gating produces on the
same input. That check is only meaningful if the decision cannot see the mode -- if
it could, the comparison would be a function against itself.
"""

from __future__ import annotations

import threading
import time

import pytest

from policy.sensing_controller import Inputs, SensingController
from policy.shadow_mode import LIVE, MODES, SHADOW, ModeHolder, command_for
from transport.channels import Channel
from transport.messages import decode_message


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def drive_inputs(**over) -> Inputs:
    fields = dict(ego_acceleration=0.0, ego_speed=20.0, policy_margin=0.9,
                  feed_congestion=0.1, camera_density_bin=2,
                  thermal_status="nominal", skin_temp_c=30.0,
                  lat=51.49, lon=-0.20, position_age_s=0.4, telemetry_age_s=1.0)
    fields.update(over)
    return Inputs(**fields)


class TestTheDecisionCannotSeeTheMode:
    """The property task 43 rests on."""

    def test_the_two_modes_differ_by_exactly_one_boolean(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        cases = [
            drive_inputs(),
            drive_inputs(ego_acceleration=9.0),
            drive_inputs(policy_margin=0.01),
            drive_inputs(thermal_status="severe"),
            drive_inputs(lat=float("nan"), lon=float("nan")),
        ]
        for inputs in cases:
            clock.advance(10.0)
            controller.decide(inputs)
            clock.advance(1.0)
            decision = controller.decide(inputs)

            shadowed = command_for(decision, SHADOW, t_capture_mono_ns=1)
            live = command_for(decision, LIVE, t_capture_mono_ns=1)

            assert shadowed.rates == live.rates
            assert shadowed.trigger == live.trigger
            assert shadowed.here == live.here
            assert shadowed.shadow is True and live.shadow is False

    def test_decide_takes_no_mode_argument(self):
        # Structural, and deliberately so: the moment `decide` can be handed a mode,
        # a shadow log stops predicting live and task 43's comparison passes for
        # free. Asserted rather than trusted to review.
        import inspect

        parameters = set(inspect.signature(SensingController.decide).parameters)
        assert parameters == {"self", "inputs"}

    def test_a_drive_replayed_in_both_modes_makes_identical_decisions(self):
        # The same inputs through two controllers, one in each mode: every rate,
        # trigger and query matches tick for tick.
        script = [drive_inputs(ego_acceleration=a, policy_margin=m, thermal_status=s)
                  for a, m, s in [(0.0, 0.9, "nominal"), (9.0, 0.9, "nominal"),
                                  (9.0, 0.4, "nominal"), (0.0, 0.05, "moderate"),
                                  (0.0, 0.9, "severe"), (0.0, 0.9, "nominal")]]
        results = {}
        for mode in MODES:
            clock = Clock()
            controller = SensingController(clock=clock)
            holder = ModeHolder(mode, clock=clock)
            emitted = []
            for inputs in script:
                clock.advance(0.4)
                command = command_for(controller.decide(inputs), holder.mode,
                                      t_capture_mono_ns=int(clock.now * 1e9))
                emitted.append((tuple(sorted(command.rates.items())), command.trigger))
            results[mode] = emitted

        assert results[SHADOW] == results[LIVE]


class TestFlipping:

    def test_shadow_is_the_default(self):
        # A process that comes up gating for real because nobody said otherwise is
        # the wrong failure to have.
        assert ModeHolder().mode == SHADOW
        assert ModeHolder().is_live is False

    def test_a_flip_changes_the_flag_and_not_the_decision(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        holder = ModeHolder(clock=clock)

        clock.advance(1.0)
        decision = controller.decide(drive_inputs())
        before = command_for(decision, holder.mode, t_capture_mono_ns=1)

        assert holder.flip_to(LIVE) is True
        after = command_for(decision, holder.mode, t_capture_mono_ns=1)

        assert before.shadow is True and after.shadow is False
        assert before.rates == after.rates and before.trigger == after.trigger

    def test_flipping_to_the_current_mode_is_not_a_flip(self):
        holder = ModeHolder(clock=Clock())
        assert holder.flip_to(SHADOW) is False
        assert holder.to_record()["flip_count"] == 0

    def test_an_unknown_mode_is_refused_rather_than_defaulted(self):
        # Defaulting an unknown mode is how a typo becomes a live drive.
        with pytest.raises(ValueError):
            ModeHolder("live-ish")
        with pytest.raises(ValueError):
            ModeHolder().flip_to("on")
        with pytest.raises(ValueError):
            command_for(object(), "on", t_capture_mono_ns=1)

    def test_a_flip_from_another_thread_never_splits_a_command(self):
        # The rates and the flag must come from one decision and one mode. Read
        # once per command, so a flip lands between commands or not at all.
        clock = Clock()
        controller = SensingController(clock=clock)
        holder = ModeHolder(clock=clock)
        stop = threading.Event()

        def flipper():
            target = LIVE
            while not stop.is_set():
                holder.flip_to(target)
                target = SHADOW if target == LIVE else LIVE

        worker = threading.Thread(target=flipper, daemon=True)
        worker.start()
        try:
            seen = set()
            for _ in range(2000):
                clock.advance(0.05)
                decision = controller.decide(drive_inputs())
                command = command_for(decision, holder.mode, t_capture_mono_ns=1)
                seen.add((command.shadow, tuple(sorted(command.rates.items()))))
            # Two shapes at most: one per mode, with identical rates.
            rates = {r for _, r in seen}
            assert len(rates) == 1, f"rates varied with the mode: {rates}"
        finally:
            stop.set()
            worker.join(timeout=3.0)


class TestTheRecord:

    def test_every_flip_is_named_with_when(self):
        # A log that reports only the final mode cannot say which decisions were
        # gated, and every decision before and after gets scored together.
        clock = Clock()
        holder = ModeHolder(clock=clock)

        clock.advance(5.0)
        holder.flip_to(LIVE)
        clock.advance(3.0)
        holder.flip_to(SHADOW)

        record = holder.to_record()
        assert record["mode"] == SHADOW
        assert record["flip_count"] == 2
        assert [f["now"] for f in record["flips"]] == [LIVE, SHADOW]
        assert record["flips"][0]["at_mono"] == 1005.0
        assert record["flips"][1]["at_mono"] == 1008.0

    def test_the_record_states_what_a_shadow_log_predicts(self):
        # In shadow the phone keeps the reference rates, so the controller decides
        # from full-rate inputs. Live feeds its own reduced observations back in, so
        # the trajectories diverge after the first change. A reader a month later
        # must not take a shadow log for a live prediction.
        assert "not the trajectory" in ModeHolder().to_record()["shadow_predicts"]


class TestTheCommandIsSendable:

    def test_a_shadow_command_is_a_legal_rate_cmd(self):
        clock = Clock()
        controller = SensingController(clock=clock)
        for mode in MODES:
            for inputs in (drive_inputs(), drive_inputs(ego_acceleration=9.0),
                           drive_inputs(lat=float("nan"), lon=float("nan")),
                           drive_inputs(thermal_status="shutdown")):
                clock.advance(10.0)
                controller.decide(inputs)
                clock.advance(1.0)
                command = command_for(controller.decide(inputs), mode, t_capture_mono_ns=7)
                decoded = decode_message(Channel.RATE_CMD, *command.to_wire())
                assert decoded.shadow is (mode != LIVE)


class TestTheRecordNamesWhatShadowCannotSee:
    """A shadow drive's gaps are structural, and the record must say so.

    The phone makes no HERE call until the Jetson sends a query, and
    `ConfigApplier` returns on the shadow branch before reaching `setHereQuery`. So
    a drive that has only ever been in shadow has no traffic feed at all:
    `feed_congestion` is None on every tick and `source_disagreement` -- one of the
    controller's three raise rules -- cannot fire. Task 35 scores candidate policies
    from these logs, so a reader who does not know that would credit every policy
    equally for a rule none of them could have used.
    """

    def test_the_record_names_the_inputs_a_shadow_drive_cannot_have(self):
        # Spelled out, not compared against the constant it comes from. `set(record)
        # == set(CONSTANT)` reads both sides from one place, so it holds for whatever
        # the constant says -- adding `camera_density_bin` to it, an input a pure
        # shadow drive plainly HAS since the camera runs at reference rates precisely
        # because nothing is applied, passed such a check unchanged.
        record = ModeHolder(clock=Clock()).to_record()
        assert set(record["structurally_absent"]) == {"feed_congestion", "source_disagreement"}

    def test_every_input_named_absent_really_is_unobtainable(self):
        # The list is a claim about the controller's inputs, so it is checked against
        # them. `feed_congestion` is the field, `source_disagreement` the rule it is
        # the sole evidence for; both are unavailable exactly when no HERE response
        # has arrived, which is every tick of a pure shadow drive.
        from policy.sensing_controller import Trigger, disagreement

        record = ModeHolder(clock=Clock()).to_record()
        assert disagreement(None, 0) is False
        assert Trigger.DISAGREEMENT == "source_disagreement"
        for name in record["structurally_absent"]:
            assert name in {"feed_congestion", "source_disagreement"}, (
                f"{name} is named absent but is not a consequence of the missing feed"
            )

    def test_only_a_drive_that_was_never_live_claims_reference_rates(self):
        # The discriminator is having ENTERED live, not having left it. A flip records
        # the mode it came from, so a predicate over `f.was == LIVE` calls a drive that
        # is still live a full-rate one -- and a drive constructed live has no flips at
        # all. Every shape, because only two of the four are wrong.
        clock = Clock()

        never = ModeHolder(clock=clock)
        assert never.to_record()["reference_rates_hold"] is True

        promoted = ModeHolder(clock=clock)
        promoted.flip_to(LIVE)
        assert promoted.to_record()["reference_rates_hold"] is False

        clock.advance(5.0)
        promoted.flip_to(SHADOW)
        assert promoted.to_record()["reference_rates_hold"] is False

        born_live = ModeHolder(LIVE, clock=clock)
        assert born_live.to_record()["reference_rates_hold"] is False

    def test_a_live_drive_does_not_report_the_feed_as_absent(self):
        # The two fields used to agree on the wrong answer: a wholly live drive, in
        # which ConfigApplier reached setHereQuery on every tick, claimed to hold the
        # reference rates AND named the feed absent. Task 35 would have read it as a
        # pure shadow log.
        clock = Clock()
        holder = ModeHolder(clock=clock)
        holder.flip_to(LIVE)

        record = holder.to_record()
        assert record["structurally_absent"] == []
        assert record["reference_rates_hold"] is False

    def test_the_disagreement_rule_really_is_unreachable_without_a_feed(self):
        # Asserted on `rules_fired`, not on `trigger`. `trigger` is `raises[0]` and
        # DISAGREEMENT is appended last of the three raise rules, so any tick where
        # EVENT or NARROW_MARGIN also fires hides it -- a test on `trigger` would pass
        # with the rule firing on every tick. `rules_fired` is also the field task 34
        # attributes on, which is what the claim is about.
        from policy.sensing_controller import Trigger

        clock = Clock()
        controller = SensingController(clock=clock)
        fired: set[str] = set()
        # Each case pairs the missing feed with a rule that WOULD mask it in `trigger`.
        cases = [dict(), dict(ego_acceleration=9.0), dict(policy_margin=0.01),
                 dict(ego_acceleration=9.0, policy_margin=0.01)]
        for density in (0, 1, 2, 3):
            for extra in cases:
                clock.advance(10.0)
                inputs = drive_inputs(feed_congestion=None,
                                      camera_density_bin=density, **extra)
                controller.decide(inputs)
                clock.advance(1.0)
                fired.update(controller.decide(inputs).rules_fired)
        assert Trigger.DISAGREEMENT not in fired
        # The masking rules did fire, so the assertion above is about a blocked rule
        # and not about a fixture that raised nothing.
        assert Trigger.EVENT in fired and Trigger.NARROW_MARGIN in fired

    def test_every_path_that_touches_the_state_takes_the_lock(self):
        # Structural, because the behavioural version of this cannot be trusted.
        # `flip_to` appends the flip before assigning the mode, so an unlocked
        # `to_record` really can return a flip list ending in `now="live"` beside
        # `mode="shadow"` -- but that window is a few bytecodes wide, and a test that
        # races for it reports SURVIVED on one run and CAUGHT on the next. This repo
        # has already retired one probabilistic pin for exactly that. So the lock is
        # observed directly: removing it from any of the three paths fails here, on
        # every run, instead of on some of them.
        holder = ModeHolder(clock=Clock())
        real = holder._lock
        entered: list[str] = []

        class Watched:
            def __enter__(self):
                entered.append("in")
                return real.__enter__()

            def __exit__(self, *exc):
                return real.__exit__(*exc)

        holder._lock = Watched()

        entered.clear()
        holder.mode
        assert entered, "the mode getter read _mode without the lock"

        entered.clear()
        holder.flip_to(LIVE)
        assert entered, "flip_to did check-append-set without the lock"

        entered.clear()
        holder.to_record()
        assert entered, "to_record read the mode and the flip log without the lock"

    def test_the_record_is_never_read_mid_flip(self):
        # What the lock buys, on the state a reader actually sees. Not the pin for the
        # lock itself -- see above -- but it would catch a gross error, and it is what
        # makes the invariants explicit: the mode and the flip log are one state.
        clock = Clock()
        holder = ModeHolder(clock=clock)
        stop = threading.Event()
        # Bounded, because the flip log grows without bound and `to_record` copies
        # all of it: an unbounded flipper made each read O(flips so far) against a
        # writer that never yields, and the test took minutes instead of a second.
        # The deadline is inside the loop, so it holds however the thread ends.
        flips_to_make = 400
        deadline = time.monotonic() + 10.0

        def flipper() -> None:
            target = LIVE
            for _ in range(flips_to_make):
                if stop.is_set() or time.monotonic() > deadline:
                    return
                clock.advance(0.001)
                holder.flip_to(target)
                target = SHADOW if target == LIVE else LIVE

        worker = threading.Thread(target=flipper, daemon=True)
        worker.start()
        try:
            while worker.is_alive() and time.monotonic() < deadline:
                record = holder.to_record()
                flips = record["flips"]
                assert record["flip_count"] == len(flips)
                # The mode and the flip log are one state, so the last flip's
                # destination is the current mode.
                expected = flips[-1]["now"] if flips else SHADOW
                assert record["mode"] == expected, (
                    f"mode {record['mode']} with a flip log ending in {expected}"
                )
                # A transition logged twice is a flip that never happened.
                for earlier, later in zip(flips, flips[1:]):
                    assert earlier["now"] == later["was"], f"{earlier} then {later}"
                assert record["reference_rates_hold"] is not any(
                    f["now"] == LIVE for f in flips
                )
            assert holder.to_record()["flip_count"] == flips_to_make, (
                "the flipper did not finish, so the reader raced nothing"
            )
        finally:
            stop.set()
            worker.join(timeout=3.0)
