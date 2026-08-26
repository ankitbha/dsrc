"""Shadow and live, and the property that makes shadow logs worth reading.

Task 43 checks that logged shadow decisions match what live gating produces on the
same input. That check is only meaningful if the decision cannot see the mode -- if
it could, the comparison would be a function against itself.
"""

from __future__ import annotations

import threading

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
