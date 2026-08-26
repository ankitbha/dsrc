"""The tick loop's half of sensing control.

The property worth pinning is the send cadence. `rate_cmd` is `reliable` at depth
16 and the tick loop runs at camera rate, so a command per tick would make the
channel designed never to lose a record lose records continuously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from policy.advisory import Advisory
from policy.sensing_loop import QUERY_REFRESH_FRACTION, SensingLoop, inputs_from
from policy.shadow_mode import LIVE, SHADOW, ModeHolder
from transport.channels import Channel
from transport.messages import ACTION_HEADS, decode_message


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeGps:
    lat: float = 51.49
    lon: float = -0.20
    valid: bool = True


@dataclass
class FakeObs:
    obs: dict
    feed: Any = None
    diagnostics: dict = field(default_factory=lambda: {"gps_age_s": 0.4})


@dataclass
class FakePolicy:
    head_probs: dict


@dataclass
class FakeTick:
    obs_result: FakeObs
    policy: FakePolicy
    gps: FakeGps
    advisory: Advisory
    t_capture_mono: float = 1000.0


def advisory() -> Advisory:
    return Advisory(
        recommended_speed_mps=13.4, recommended_speed_display=30.0,
        current_speed_display=28.0, units="mph", headway_target_s=2.0,
        lane_text="keep lane", merge_text="no merge", traffic_text="moderate",
        confidence=0.8, confidence_label="high",
        action={"desired_speed_bin": "nominal", "desired_headway_bin": "normal",
                "lane_preference": "keep", "merge_mode": "normal"})


def tick(*, accel=0.0, speed=20.0, density=2, margins=(0.9, 0.4), feed=None,
         capture=1000.0, lat=51.49, lon=-0.20) -> FakeTick:
    return FakeTick(
        obs_result=FakeObs(obs={"ego_acceleration": accel, "ego_speed": speed,
                                "local_density_bin": float(density)}, feed=feed),
        # One entry per head, keyed by the real head names. Keying them all
        # "desired_speed_bin" left a dict of one, so a test named for the head
        # closest to a boundary could only ever see the last margin in the tuple.
        policy=FakePolicy(head_probs={
            head: [0.5 + m / 2, 0.5 - m / 2]
            for head, m in zip(ACTION_HEADS, margins)}),
        gps=FakeGps(lat=lat, lon=lon),
        advisory=advisory(),
        t_capture_mono=capture,
    )


class Phone:
    """Counts what went down the link, and can refuse like a dropped one."""

    def __init__(self, up: bool = True) -> None:
        self.up = up
        self.advisories: list[tuple] = []
        self.commands: list[Any] = []
        self.telemetry = None
        self.telemetry_at_mono = None

    def send_advisory(self, advisory_obj, *, t_capture_mono_ns):
        if not self.up:
            return False
        self.advisories.append((advisory_obj, t_capture_mono_ns))
        return True

    def send_rate_command(self, command):
        if not self.up:
            return False
        self.commands.append(command)
        return True


class TestTheAdvisoryGoesEveryTick:

    def test_the_advisory_carries_the_tick_it_is_about_not_now(self):
        # `AdvisoryMessage` has no frame id, so the capture stamp is the only thing
        # tying a recommendation to the frame that produced it.
        clock = Clock()
        loop = SensingLoop(clock=clock)
        phone = Phone()
        loop.on_tick(tick(capture=1234.5), phone)
        assert phone.advisories[0][1] == int(1234.5 * 1e9)

    def test_every_tick_sends_one(self):
        # `advisory` is latest_wins at depth one: the newest is the only one worth
        # having, so there is nothing to gain by holding it back.
        clock = Clock()
        loop = SensingLoop(clock=clock)
        phone = Phone()
        for _ in range(20):
            clock.advance(0.05)
            loop.on_tick(tick(), phone)
        assert len(phone.advisories) == 20


class TestTheCommandCadence:
    """The reason this class exists: reliable at depth 16, ticking at camera rate."""

    def test_an_unchanged_decision_is_not_resent_every_tick(self):
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone()
        for _ in range(50):
            clock.advance(0.02)
            loop.on_tick(tick(), phone)
        assert len(phone.commands) == 1
        assert loop.sends_by_reason == {"first": 1}

    def test_the_first_command_does_not_wait_out_the_heartbeat(self):
        # Otherwise the phone runs its startup rates for the opening seconds of
        # every drive, which is exactly when the drive is least settled.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=5.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        assert len(phone.commands) == 1
        assert loop.sends_by_reason == {"first": 1}

    def test_a_changed_decision_is_sent(self):
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        clock.advance(10.0)
        loop.on_tick(tick(accel=9.0), phone)
        clock.advance(1.0)
        loop.on_tick(tick(accel=9.0), phone)
        assert len(phone.commands) >= 2
        assert loop.sends_by_reason.get("changed", 0) >= 1

    def test_a_mode_flip_is_sent_even_when_the_rates_are_identical(self):
        # The flag is the whole difference between a recorded decision and a gated
        # one. Comparing only the rates would leave the phone in shadow after the
        # operator went live, with the log saying live.
        clock = Clock()
        modes = ModeHolder(SHADOW, clock=clock)
        loop = SensingLoop(clock=clock, modes=modes, heartbeat_s=1000.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        before = phone.commands[-1]

        modes.flip_to(LIVE)
        clock.advance(0.02)
        loop.on_tick(tick(), phone)

        assert len(phone.commands) == 2
        after = phone.commands[-1]
        assert before.shadow is True and after.shadow is False
        assert before.rates == after.rates
        assert loop.sends_by_reason.get("changed") == 1

    def test_the_heartbeat_resends_an_unchanged_command(self):
        # What puts a phone that reconnected mid-drive back onto the current
        # command: a rebind leaves it running whatever it had.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=5.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        clock.advance(4.9)
        loop.on_tick(tick(), phone)
        assert len(phone.commands) == 1
        clock.advance(0.2)
        loop.on_tick(tick(), phone)
        assert len(phone.commands) == 2
        assert loop.sends_by_reason.get("heartbeat") == 1

    def test_the_capture_stamp_alone_never_counts_as_a_change(self):
        # It differs every tick. Including it would make the whole cadence rule a
        # no-op while still looking like it worked.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone()
        for i in range(30):
            clock.advance(0.02)
            loop.on_tick(tick(capture=1000.0 + i), phone)
        assert len(phone.commands) == 1


class TestTheQueryGoesStaleInSpace:

    def test_the_query_is_resent_once_the_vehicle_has_left_the_stretch(self):
        # The query's `in_` is formatted to five decimals, about a metre, so it
        # differs on nearly every tick and cannot be part of the changed test. It is
        # about the road ahead, so it ages by distance travelled.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        radius = phone.commands[0].here.radius_m

        # A tenth of the refresh distance: not yet.
        moved = QUERY_REFRESH_FRACTION * radius
        clock.advance(0.1)
        loop.on_tick(tick(lat=51.49 + _degrees_north(moved * 0.1)), phone)
        assert len(phone.commands) == 1

        clock.advance(0.1)
        loop.on_tick(tick(lat=51.49 + _degrees_north(moved * 1.1)), phone)
        assert len(phone.commands) == 2
        assert loop.sends_by_reason.get("query_moved") == 1

    def test_losing_the_fix_is_a_change_not_a_move(self):
        # Reporting a distance from a position that does not exist is how the two
        # link-distance figures came from different points once already.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone()
        loop.on_tick(tick(), phone)
        assert phone.commands[0].here is not None

        clock.advance(0.1)
        blind = tick()
        blind.gps.valid = False
        outcome = loop.on_tick(blind, phone)
        assert outcome.command.here is None
        assert outcome.send_reason == "changed"


class TestNoSession:

    def test_a_dropped_phone_does_not_stop_the_loop_or_the_decisions(self):
        # The supervisor is reattaching it. A raise per tick, or a loop that stopped
        # deciding, would take the run down for the condition it exists to survive.
        clock = Clock()
        loop = SensingLoop(clock=clock)
        phone = Phone(up=False)
        for _ in range(10):
            clock.advance(0.05)
            outcome = loop.on_tick(tick(), phone)
            assert outcome.decision is not None
            assert outcome.advisory_sent is False
            assert outcome.command_sent is False
        assert loop.ticks == 10

    def test_a_refused_command_is_not_recorded_as_the_phones_state(self):
        # Otherwise the loop believes the phone has a command it never received, and
        # withholds the resend that would have fixed it -- for the whole drive, since
        # nothing changed afterwards.
        clock = Clock()
        loop = SensingLoop(clock=clock, heartbeat_s=1000.0)
        phone = Phone(up=False)
        loop.on_tick(tick(), phone)
        phone.up = True
        clock.advance(0.02)
        loop.on_tick(tick(), phone)
        assert len(phone.commands) == 1

    def test_no_phone_at_all_still_decides(self):
        clock = Clock()
        loop = SensingLoop(clock=clock)
        outcome = loop.on_tick(tick(), None)
        assert outcome.decision is not None
        assert outcome.command_sent is False


class TestTheWireIsTheArbiter:

    def test_every_command_the_loop_emits_is_a_legal_rate_cmd(self):
        clock = Clock()
        for mode in (SHADOW, LIVE):
            loop = SensingLoop(clock=clock, modes=ModeHolder(mode, clock=clock),
                               heartbeat_s=0.0)
            phone = Phone()
            for accel, density, speed in [(0.0, 2, 20.0), (9.0, 0, 30.0),
                                          (-4.0, 3, 5.0), (0.0, 1, 0.0)]:
                clock.advance(1.0)
                loop.on_tick(tick(accel=accel, density=density, speed=speed), phone)
            assert phone.commands
            for command in phone.commands:
                decoded = decode_message(Channel.RATE_CMD, *command.to_wire())
                assert decoded.shadow is (mode != LIVE)


class TestInputsFromATick:

    def test_the_margin_is_the_head_closest_to_a_boundary(self):
        # The minimum over heads, not the mean: the policy is at a boundary if ANY
        # head is, and averaging lets three confident heads hide the one about to
        # change its mind.
        inputs = inputs_from(tick(margins=(0.9, 0.02, 0.5)), None, now=1000.0)
        assert inputs.policy_margin == pytest.approx(0.02, abs=1e-6)

    def test_telemetry_age_is_measured_against_this_instant(self):
        clock = Clock()
        phone = Phone()
        phone.telemetry = type("T", (), {"thermal_status": "moderate",
                                          "skin_temp_c": 41.0})()
        phone.telemetry_at_mono = clock.now - 3.0
        inputs = inputs_from(tick(), phone, now=clock.now)
        assert inputs.telemetry_age_s == pytest.approx(3.0)
        assert inputs.thermal_status == "moderate"

    def test_a_phone_that_never_reported_is_not_read_as_cool(self):
        inputs = inputs_from(tick(), Phone(), now=1000.0)
        assert inputs.thermal_status is None
        assert inputs.telemetry_age_s is None


def _degrees_north(metres: float) -> float:
    return metres / 111_320.0
