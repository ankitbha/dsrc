"""`MainzEnv`'s run-time penetration, its gate at partial penetration and the ticked path.

The unit tests read the committed schedule and never start SUMO. The sanity tests at
the bottom run short SUMO episodes on seed 16; run this file in its own process.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.sumo import mainz  # noqa: E402
from src.sumo.mainz import MainzEnv, controlled_index  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
SCHEDULE = json.loads((DATA / "mainz_schedule.json").read_text())


def _controlled_count(p: float) -> int:
    env = MainzEnv(penetration=p)
    return sum(1 for row in SCHEDULE if env.controlled(row["id"]))


class TestTheFloorRule:
    def test_full_penetration_controls_all_3124(self):
        assert len(SCHEDULE) == 3124
        assert _controlled_count(1.0) == 3124

    def test_zero_penetration_controls_none(self):
        assert _controlled_count(0.0) == 0

    def test_half_controls_exactly_1562(self):
        assert _controlled_count(0.5) == 1562

    def test_three_quarters_controls_exactly_2343(self):
        # The builder's `--av-fraction 0.75` controls every vehicle; the floor rule
        # controls floor(0.75 * 3124) = 2343.
        assert _controlled_count(0.75) == 2343

    def test_a_tenth_controls_312(self):
        assert _controlled_count(0.1) == 312

    @pytest.mark.parametrize("p", [0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0])
    def test_consecutive_controlled_indices_are_never_more_than_ceil_1_over_p_apart(self, p):
        indices = [i for i in range(3124) if controlled_index(i, p)]
        gaps = np.diff(indices)
        assert gaps.max() <= math.ceil(1.0 / p)
        assert len(indices) == math.floor(3124 * p)

    def test_the_rule_follows_schedule_order_not_the_vehicle_id(self):
        # Schedule index 1 is v182_521; at p=0.5 index 1 is controlled and index 0 not.
        env = MainzEnv(penetration=0.5)
        assert env.controlled(SCHEDULE[1]["id"]) is True
        assert env.controlled(SCHEDULE[0]["id"]) is False

    def test_an_unknown_vehicle_is_an_error_not_uncontrolled(self):
        with pytest.raises(KeyError):
            MainzEnv(penetration=0.5).controlled("nobody")


class TestSegmentState:
    def test_speed_is_weighted_by_count_and_density_is_occupancy(self):
        env = MainzEnv()
        env.segments = [["a", "b"], ["c"]]
        stub = {
            "a": {"count": 3.0, "speed_kmh": 10.0, "lanes": 2.0, "length_m": 100.0},
            "b": {"count": 1.0, "speed_kmh": 50.0, "lanes": 1.0, "length_m": 100.0},
            "c": {"count": 0.0, "speed_kmh": math.nan, "lanes": 1.0, "length_m": 50.0},
        }
        env._per_edge = lambda: stub

        state = env.segment_state()

        # (3 * 10 + 1 * 50) / 4 = 20, not the unweighted (10 + 50) / 2 = 30.
        assert state["speed_kmh"][0] == pytest.approx(20.0)
        assert state["count"][0] == 4
        # 4 vehicles * 4.5 m over 2 * 100 + 1 * 100 lane-metres.
        assert state["density"][0] == pytest.approx(4 * 4.5 / 300.0)
        assert math.isnan(state["speed_kmh"][1])
        assert state["density"][1] == 0.0
        assert state["count"][1] == 0


# ------------------------------------------------------------------------ SUMO


def _here_model():
    import evaluate_mainz_checkpoints as emc
    from train_mainz_src import SCENARIOS

    return emc.load_arm_model("here", REPO_ROOT / "results" / "checkpoints",
                              emc._num_segments(SCENARIOS["mainz"])).model


def _env(**kwargs) -> MainzEnv:
    return MainzEnv(seed=16, step_length_s=0.5, gate_entries=True, **kwargs)


def test_at_half_penetration_the_cap_reaches_exactly_the_controlled_ids(monkeypatch):
    written: list[str] = []
    added: set[str] = set()
    set_speed: list[str] = []
    real_cap, real_add = mainz._sumo.vehicle.setMaxSpeed, mainz._sumo.vehicle.add

    def record_cap(vehicle, speed):
        written.append(vehicle)
        real_cap(vehicle, speed)

    def record_add(vehicle, *args, **kwargs):
        added.add(vehicle)
        real_add(vehicle, *args, **kwargs)

    monkeypatch.setattr(mainz._sumo.vehicle, "setMaxSpeed", record_cap)
    monkeypatch.setattr(mainz._sumo.vehicle, "setSpeed",
                        lambda vehicle, speed: set_speed.append(vehicle))
    monkeypatch.setattr(mainz._sumo.vehicle, "add", record_add)
    env = _env(duration_s=400.0, penetration=0.5)
    checks = []
    try:
        env.reset()
        for _ in range(2):  # decisions at 300 and 360
            env._command([0] * 12)

            def check(t):
                if int(t) % 10:
                    return
                due = [r for r in SCHEDULE if r["depart"] <= t]
                # `_admit` runs before each 0.5 s step, so at the callback for t the last
                # admission ran at t - 0.5; a vehicle departing in (t - 0.5, t] is next.
                uncontrolled_waiting = [r["id"] for r in due
                                        if r["depart"] <= t - 0.5
                                        and not env.controlled(r["id"]) and r["id"] not in added]
                controlled_waiting = sum(1 for r in due
                                         if env.controlled(r["id"]) and r["id"] not in added)
                checks.append((t, uncontrolled_waiting, controlled_waiting,
                               env.held_at_entry()))

            env.advance_ticks(60.0, on_tick=check)
    finally:
        env.close()
    assert written, "no cap was written"
    assert set_speed == [], "advice must not go through setSpeed under the cap"
    assert all(env.controlled(v) for v in written)
    # Uncontrolled vehicles are in the network too, and none was ever written.
    uncontrolled_in = [v for v in added if not env.controlled(v)]
    assert len(uncontrolled_in) > 100
    assert not set(written) & set(uncontrolled_in)
    assert len(checks) == 12
    for t, uncontrolled_waiting, controlled_waiting, held in checks:
        assert uncontrolled_waiting == [], t
        assert held == controlled_waiting, t


def test_full_penetration_commands_the_same_ids_as_the_type_test(monkeypatch):
    """At p = 1 `controlled` agrees with `getTypeID == "av"` on every vehicle commanded."""
    env = _env(duration_s=360.0, penetration=1.0)
    try:
        env.reset()
        ids = [v for seg in env.segments for e in seg
               for v in mainz._sumo.edge.getLastStepVehicleIDs(e)]
        assert ids
        assert all(env.controlled(v) == (mainz._sumo.vehicle.getTypeID(v) == "av")
                   for v in ids)
    finally:
        env.close()


def _run_600(model, ticked: bool) -> dict:
    env = _env(duration_s=600.0)
    try:
        state = torch.tensor(env.reset(), dtype=torch.float)
        from src.rl.src_q import greedy_actions
        while True:
            action = greedy_actions(model, state).tolist()
            if ticked:
                env._command(action)
                env.advance_ticks(60.0, on_tick=lambda t: None)
                raw = env.observe()
                done = mainz._sumo.simulation.getTime() >= env.duration_s
            else:
                raw, _, done = env.step(action)
            state = torch.tensor(raw, dtype=torch.float)
            if done:
                break
        return env.metrics()
    finally:
        env.close()


def test_advance_ticks_reproduces_step_on_seed_16_at_600_s():
    model = _here_model()
    stepped = _run_600(model, ticked=False)
    ticked = _run_600(model, ticked=True)
    assert stepped["arrived"] > 0
    assert ticked["arrived"] == stepped["arrived"]
    assert ticked["flow_veh_per_h"] == stepped["flow_veh_per_h"]
    assert ticked["mean_vehicles"] == stepped["mean_vehicles"]
    assert ticked["space_mean_speed_kmh"] == stepped["space_mean_speed_kmh"]
    assert ticked["held_at_entry"] == stepped["held_at_entry"]


def test_advance_ticks_calls_back_once_per_second_with_the_simulation_time():
    env = _env(duration_s=320.0, warmup_s=10.0)
    seen = []
    try:
        env.reset()
        env.advance_ticks(5.0, on_tick=seen.append)
    finally:
        env.close()
    assert seen == [11.0, 12.0, 13.0, 14.0, 15.0]


def test_a_ticked_warm_up_equals_the_plain_one():
    seen = []
    plain, ticked = _env(duration_s=400.0), _env(duration_s=400.0)
    try:
        plain.reset()
        first = (plain.metrics(), plain.observe())
    finally:
        plain.close()
    try:
        ticked.reset(on_warmup_tick=seen.append)
        second = (ticked.metrics(), ticked.observe())
    finally:
        ticked.close()
    assert seen[0] == 1.0 and seen[-1] == 300.0 and len(seen) == 300
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])


class TestTheCap:
    def test_an_unknown_actuation_is_refused(self):
        with pytest.raises(ValueError):
            MainzEnv(actuation="speed_mode")

    def test_release_restores_the_type_maximum_and_touches_nothing_else(self, monkeypatch):
        env = _env(duration_s=420.0)
        try:
            env.reset()
            env._command([0] * 12)  # half of each segment's limit
            on = [v for v, _, _ in env.controlled_on_segments()]
            assert on
            capped = [mainz._sumo.vehicle.getMaxSpeed(v) for v in on]
            assert max(capped) == pytest.approx(0.5 * 60.012 / 3.6, abs=0.01)
            calls: list[str] = []
            real = mainz._sumo.vehicle.setMaxSpeed
            monkeypatch.setattr(mainz._sumo.vehicle, "setMaxSpeed",
                                lambda v, s: (calls.append(v), real(v, s)))
            env.release()
            assert sorted(calls) == sorted(on)
            assert all(mainz._sumo.vehicle.getMaxSpeed(v) == pytest.approx(16.67)
                       for v in on)
            calls.clear()
            env.release()  # nothing is capped any more, so nothing is written
            assert calls == []
        finally:
            env.close()

    def test_collisions_and_teleports_are_counted_per_step(self):
        """The legacy actuation collides on seed 16; the counts must see it."""
        model = _here_model()
        from src.rl.src_q import greedy_actions

        counts = {}
        for actuation in ("set_speed", "max_speed"):
            env = _env(duration_s=1500.0, actuation=actuation)
            try:
                state = torch.tensor(env.reset(), dtype=torch.float)
                while True:
                    raw, _, done = env.step(greedy_actions(model, state).tolist())
                    state = torch.tensor(raw, dtype=torch.float)
                    if done:
                        break
                counts[actuation] = env.metrics()
            finally:
                env.close()
        legacy, cap = counts["set_speed"], counts["max_speed"]
        assert legacy["collision_teleports"] > 0
        assert legacy["colliding_vehicles"] >= legacy["collision_teleports"]
        assert cap["colliding_vehicles"] == 0 and cap["collision_teleports"] == 0
        assert cap["actuation"] == "max_speed"
