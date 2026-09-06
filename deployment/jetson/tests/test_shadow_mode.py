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
            # Carried, not merely equal. `shadowed.here == live.here` is two
            # references to one object and holds just as well when `command_for`
            # drops the query entirely -- and the query is the sole mechanism by
            # which the feed comes to exist, so dropping it produces a pure-shadow
            # log in live mode. Same for the capture stamp, which is the only thing
            # tying the command to the tick that produced it.
            assert shadowed.here is decision.here_query
            assert shadowed.t_capture_mono_ns == 1

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
        # Both branches. `is_live` returning a constant False passed everything.
        assert ModeHolder(LIVE).is_live is True

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

    def test_only_a_drive_that_was_live_from_the_first_tick_reports_nothing_absent(self):
        # Three shapes, because two of them were being collapsed into one. The version
        # this replaces called `ModeHolder(clock=...)` then `flip_to(LIVE)` a drive "in
        # which ConfigApplier reached setHereQuery on every tick" -- but that fixture
        # BEGINS in shadow, so it did not, and the test asserted the reading meant for
        # a born-live drive against a mixed one.
        clock = Clock()

        never = ModeHolder(clock=clock).to_record()
        assert never["structurally_absent"] == ["feed_congestion", "source_disagreement"]
        assert never["feed_possible_from_mono"] is None

        born_live = ModeHolder(LIVE, clock=clock).to_record()
        assert born_live["structurally_absent"] == []
        assert born_live["feed_possible_from_mono"] == clock.now

        # Promoted mid-drive. Its leading segment had no feed at all, so the inputs are
        # still named -- reporting `[]` here would credit a candidate policy on a rule
        # that could not have fired for that stretch.
        promoted = ModeHolder(clock=clock)
        clock.advance(600.0)
        promoted.flip_to(LIVE)
        record = promoted.to_record()
        assert record["structurally_absent"] == ["feed_congestion", "source_disagreement"]
        assert record["feed_possible_from_mono"] == clock.now
        assert record["reference_rates_hold"] is False

    def test_the_feed_boundary_is_the_first_promotion_not_the_last(self):
        # A drive that flips in and out must date the feed from when it FIRST became
        # possible: the query survives a live -> shadow flip, so the feed does not go
        # away again, and re-dating it would blank the ticks in between.
        clock = Clock()
        holder = ModeHolder(clock=clock)
        clock.advance(10.0)
        holder.flip_to(LIVE)
        first = clock.now
        clock.advance(10.0)
        holder.flip_to(SHADOW)
        clock.advance(10.0)
        holder.flip_to(LIVE)

        assert holder.to_record()["feed_possible_from_mono"] == first

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

    def test_a_reader_cannot_see_the_state_mid_flip(self):
        # The invariant, forced rather than raced for. The version this replaces ran a
        # flipper thread and read the record in a `while worker.is_alive()` loop -- and
        # `Thread.start()` releases the GIL, so 400 flips finished before the main
        # thread was rescheduled and the loop body executed ZERO times in every trial.
        # Four assertions were dead, and the surviving one's failure message read "the
        # flipper did not finish, so the reader raced nothing" while passing on a run
        # where the reader raced nothing.
        #
        # `flip_to` calls the clock inside its critical section, so an injected clock
        # IS the middle of the flip. A reader started from there must not complete
        # while it runs.
        holder = ModeHolder(clock=lambda: 1000.0)
        seen: list[dict] = []
        readers: list[threading.Thread] = []

        def clock_inside_the_flip() -> float:
            reader = threading.Thread(
                target=lambda: seen.append(holder.to_record()), daemon=True
            )
            readers.append(reader)
            reader.start()
            reader.join(timeout=0.25)
            return 1000.0

        holder._now = clock_inside_the_flip
        try:
            holder.flip_to(LIVE)
            # Read while the flip was in progress. Empty because the reader blocked --
            # not because it was never started, which the join above waited out.
            assert seen == [], f"a reader completed mid-flip and saw {seen}"
            assert readers and readers[0].is_alive()
        finally:
            for reader in readers:
                reader.join(timeout=3.0)

        assert len(seen) == 1, "the blocked reader never completed after the flip"
        record = seen[0]
        assert record["mode"] == LIVE
        assert [f["now"] for f in record["flips"]] == [LIVE]

    def test_a_flip_records_the_mode_it_came_FROM(self):
        # `was` is the only thing in the record that says which side a drive started
        # on, and the module hands the mixed drive to the flip log entirely. Nothing
        # pinned it: writing `was` from the mode being flipped TO gives was == now on
        # every entry, and the whole suite stayed green.
        clock = Clock()
        holder = ModeHolder(clock=clock)
        for target in (LIVE, SHADOW, LIVE):
            clock.advance(1.0)
            holder.flip_to(target)

        flips = holder.to_record()["flips"]
        assert [(f["was"], f["now"]) for f in flips] == [
            (SHADOW, LIVE), (LIVE, SHADOW), (SHADOW, LIVE)
        ]
        # The chain is what makes the log replayable: each flip leaves the mode the
        # next one starts from, and the first `was` is the mode the drive began in.
        for earlier, later in zip(flips, flips[1:]):
            assert earlier["now"] == later["was"]
        assert flips[0]["was"] == SHADOW


class TestModeOrigin:
    """Who chose the mode, recorded beside what the mode is.

    Two builds of the phone and one service configuration produce the same `mode`
    for different reasons, and a drive whose mode cannot be attributed is a drive
    whose mode is a guess later.
    """

    def test_the_origin_defaults_to_the_command_line(self):
        assert ModeHolder().to_record()["mode_origin"] == ModeHolder.ORIGIN_COMMAND_LINE

    def test_a_phone_request_is_recorded_as_one(self):
        holder = ModeHolder(LIVE, origin=ModeHolder.ORIGIN_PHONE_REQUEST)
        record = holder.to_record()
        assert record["mode"] == LIVE
        assert record["mode_origin"] == ModeHolder.ORIGIN_PHONE_REQUEST
        # Born live, so nothing is structurally absent and no flip happened: this is
        # the property the whole start-time design exists to preserve.
        assert record["flips"] == []
        assert record["structurally_absent"] == []

    def test_an_unknown_origin_is_refused(self):
        # Closed set, like every other vocabulary in this file. An origin nobody
        # recognises is worse than none, because a census of the field would
        # silently grow a bucket.
        with pytest.raises(ValueError, match="origin"):
            ModeHolder(SHADOW, origin="somebody")
