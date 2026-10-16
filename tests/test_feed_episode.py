"""The feed-aware episode: per-tick bookkeeping on a fake env, then short SUMO runs.

The unit tests drive `FeedEngine.on_tick` by hand against a fake env whose vehicles,
observations and commands are set by the test. The sanity tests at the bottom run SUMO
on seed 16; run this file in its own process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.sumo.feed import FeedProcess, draw_phases  # noqa: E402
from src.sumo.feed_episode import FeedEngine, Mode, run_feed_episode  # noqa: E402
from src.sumo.mainz import SPEED_ACTION_FRACTIONS  # noqa: E402


class FakeEnv:
    """Two segments; `placement(t)` says which vehicles are on which segment at `t`."""

    def __init__(self, placement):
        self.placement = placement
        self.segments = [["e0"], ["e1"]]
        self.t = 0.0
        self.writes: list[tuple[float, str, float]] = []
        self.command: dict[str, float] = {}

    def controlled_on_segments(self):
        return [(vid, f"e{seg}", seg) for vid, seg in self.placement(self.t)]

    def observe(self):
        # The snapshot carries the instant it was taken, so a test can tell which one a
        # vector was computed from.
        return np.full((2, 5), self.t, dtype=np.float32)

    def command_vehicles(self, mapping, edge_of=None):
        for vid, fraction in mapping.items():
            self.writes.append((self.t, vid, fraction))
            self.command[vid] = fraction
        return len(mapping)

    def segment_state(self):
        return {"speed_kmh": [50.0, 50.0], "density": [0.1, 0.1], "count": [1, 1]}

    def held_at_entry(self):
        return 0

    def pending(self):
        return 0

    def running(self):
        return len(self.placement(self.t))

    arrived_total = 0

    def position(self, vid):
        return int(vid[1:])

    def time_s(self):
        return self.t


def by_minute(x) -> list[int]:
    """A policy whose action changes every minute of snapshot time."""
    return [int(x[0][0] // 60) % 3] * 2


def by_ten(x) -> list[int]:
    return [int(x[0][0] // 10) % 3] * 2


def _drive(engine: FeedEngine, env: FakeEnv, end: int) -> None:
    for t in range(1, end + 1):
        env.t = float(t)
        engine.on_tick(float(t))


def _engine(env, mode, feed, *, control=0.0, window=0.0, end=600.0, decisions=None,
            policy=by_minute, seed=1):
    return FeedEngine(env, policy, mode, feed, seed=seed, control_start_s=control,
                      window_start_s=window, trace_every_s=10.0, end_s=end,
                      decision_instants=decisions or [])


class TestScheduleModes:
    feed = FeedProcess.synthetic(60.0 * np.arange(1, 12), np.full(11, 5.0))

    def test_a_vehicle_switching_at_its_switch_instant_is_commanded_once_per_version(self):
        env = FakeEnv(lambda t: [("v0", 0)])
        engine = _engine(env, Mode.version_keyed(P_q=10.0, D=20.0), self.feed)
        _drive(engine, env, 600)
        # Every fetch lands by T_k + 15 < T_k + 20, so each switch is at T_k + 20.
        assert [t for t, *_ in env.writes] == [80.0 + 60.0 * k for k in range(9)]
        expected = [SPEED_ACTION_FRACTIONS[(k + 1) % 3] for k in range(9)]
        assert [f for *_, f in env.writes] == expected

    def test_nothing_is_written_before_control_starts_and_everyone_is_at_its_start(self):
        env = FakeEnv(lambda t: [("v0", 0), ("v1", 1)])
        engine = _engine(env, Mode.version_keyed(P_q=10.0, D=20.0), self.feed, control=150.0)
        _drive(engine, env, 210)
        # At 150 version 2 (switched at 140) is written to both, then version 3 at 200.
        assert [(t, v) for t, v, _ in env.writes] == [
            (150.0, "v0"), (150.0, "v1"), (200.0, "v0"), (200.0, "v1")]
        assert env.writes[0][2] == SPEED_ACTION_FRACTIONS[2]

    def test_a_vehicle_off_its_segments_at_its_switch_keeps_its_command(self):
        # v0 is on a junction (absent) at the switch at 140 and back at 141: not written.
        env = FakeEnv(lambda t: [] if t == 140.0 else [("v0", 0)])
        engine = _engine(env, Mode.version_keyed(P_q=10.0, D=20.0), self.feed)
        _drive(engine, env, 210)
        assert [t for t, *_ in env.writes] == [80.0, 200.0]

    def test_published_switches_everyone_at_the_decision_instants(self):
        env = FakeEnv(lambda t: [(f"v{i}", i % 2) for i in range(5)])
        decisions = [300.0 + 60.0 * m for m in range(5)]
        engine = _engine(env, Mode.published(), None, control=300.0, window=300.0,
                         decisions=decisions)
        _drive(engine, env, 599)
        assert sorted({t for t, *_ in env.writes}) == decisions
        assert len(env.writes) == 25
        summary = engine.summary()
        assert summary["delta_mean"] == 0.0
        # Age runs 0..59 s after each decision; the time average is 29.5 s.
        assert summary["age_mean_s"] == pytest.approx(29.5)

    def test_the_tick_averaged_disagreement_is_the_rule_s_own(self):
        """`delta_mean` equals the per-second disagreement of the rule's own schedules."""
        from src.sumo.feed import NoRule, VehicleFeedState, disagreement

        fleet = [f"v{i}" for i in range(50)]
        env = FakeEnv(lambda t: [(v, 0) for v in fleet])
        engine = _engine(env, Mode.no_rule(60.0), self.feed, window=100.0)
        _drive(engine, env, 600)
        u = draw_phases(1, 50)
        rule = NoRule(60.0)
        expected = np.mean([
            disagreement([rule.applied_version(VehicleFeedState(v, float(u[i])),
                                               self.feed, float(t))
                          for i, v in enumerate(fleet)])
            for t in range(100, 601)])
        assert engine.summary()["delta_mean"] == pytest.approx(expected)
        assert 0.15 < expected < 0.35


class TestInjectDelta:
    def test_thirty_percent_late_at_each_decision_apply_the_previous_vector(self):
        fleet = [f"v{i}" for i in range(100)]
        env = FakeEnv(lambda t: [(v, i % 2) for i, v in enumerate(fleet)])
        decisions = [300.0 + 60.0 * m for m in range(6)]
        engine = _engine(env, Mode.inject_delta(0.3), None, control=300.0, window=300.0,
                         decisions=decisions)
        late_sets = []
        for t in range(1, 661):
            env.t = float(t)
            before = len(env.writes)
            engine.on_tick(float(t))
            if t in decisions and t > 300:
                k = decisions.index(t) + 1
                versions = engine.applied_versions()
                late = {v for v in fleet if versions[v] == k - 1}
                assert len(late) == 30
                assert all(versions[v] == k for v in fleet if v not in late)
                previous = SPEED_ACTION_FRACTIONS[(t // 60 - 1) % 3]
                current = SPEED_ACTION_FRACTIONS[(t // 60) % 3]
                assert previous != current
                assert all(env.command[v] == previous for v in late)
                assert all(env.command[v] == current for v in fleet if v not in late)
                # A vehicle late now and on time before keeps its command: no write.
                twice = late & late_sets[-1]
                assert len(env.writes) - before == 70 + len(twice)
                late_sets.append(late)
            elif t == 300:
                late_sets.append({v for v, k in engine.applied_versions().items() if k == 0})
        assert len({frozenset(s) for s in late_sets}) == len(late_sets)
        summary = engine.summary()
        assert summary["writes_late"] > 0 and summary["writes_on_time"] > 0
        assert summary["delta_mean"] == pytest.approx(0.3)


class TestInjectStale:
    def test_the_vector_comes_from_the_snapshot_s_seconds_old(self):
        env = FakeEnv(lambda t: [("v0", 0)])
        decisions = [300.0 + 60.0 * m for m in range(4)]
        engine = _engine(env, Mode.inject_stale(60.0), None, control=300.0, window=300.0,
                         decisions=decisions, policy=by_ten)
        _drive(engine, env, 539)
        # by_ten maps a snapshot taken at t to (t // 10) % 3; the stale one is t - 60.
        assert [(t, f) for t, _, f in env.writes] == [
            (d, SPEED_ACTION_FRACTIONS[int((d - 60.0) // 10) % 3]) for d in decisions]
        assert [s["t"] for s in engine.summary()["snapshots"]] == [d - 60.0 for d in decisions]
        assert engine.summary()["age_mean_s"] == pytest.approx(60.0 + 29.5, abs=0.6)

    def test_a_decision_with_no_snapshot_that_old_writes_nothing(self):
        env = FakeEnv(lambda t: [("v0", 0)])
        engine = _engine(env, Mode.inject_stale(300.0), None, control=300.0,
                         decisions=[300.0, 360.0], policy=by_ten)
        _drive(engine, env, 400)
        assert [t for t, *_ in env.writes] == [360.0]


# ------------------------------------------------------------------------ SUMO


def _here():
    import evaluate_mainz_checkpoints as emc
    from train_mainz_src import SCENARIOS

    return emc.load_arm_model("here", REPO_ROOT / "results" / "checkpoints",
                              emc._num_segments(SCENARIOS["mainz"])).model


@pytest.fixture(scope="module")
def model():
    return _here()


def test_published_reproduces_the_committed_here_row_on_seed_16(model):
    result = run_feed_episode(model, 16, "mainz", Mode.published(), duration_s=2500.0)
    assert result["flow"] == pytest.approx(3751.1111111111113, abs=1e-9)
    assert result["arrived"] == 1989
    assert result["held_at_entry"] == 56
    assert result["return"] == 177.89599609375
    assert result["mean_vehicles"] == 896.2592592592592
    assert result["delta_mean"] == 0.0
    assert all(sample["delta"] in (0.0, None) for sample in result["trace"])
    assert result["trace"][-1]["t"] == 2520.0


def test_no_control_reproduces_the_committed_row_on_seed_16(model):
    result = run_feed_episode(model, 16, "mainz", Mode.no_control(), duration_s=2500.0)
    assert result["flow"] == pytest.approx(3722.222222222222, abs=1e-9)
    assert result["arrived"] == 2022
    assert result["return"] == 124.62904697656631
    assert result["set_speed_writes"] == 0
    assert all(sample["held"] == 0 for sample in result["trace"])


def test_inject_delta_one_differs_from_published_at_1200_s(model):
    published = run_feed_episode(model, 16, "mainz", Mode.published(), duration_s=1200.0)
    late = run_feed_episode(model, 16, "mainz", Mode.inject_delta(1.0), duration_s=1200.0)
    assert late["flow"] != published["flow"]
    assert late["writes_late"] > 0


def test_inject_stale_60_equals_applying_the_previous_decision_vector(model):
    """An independent implementation: snapshots at the decision grid shifted back 60 s."""
    import torch

    from src.rl.src_q import greedy_actions
    from src.sumo.mainz import MainzEnv

    stale = run_feed_episode(model, 16, "mainz", Mode.inject_stale(60.0), duration_s=1200.0)
    env = MainzEnv(seed=16, duration_s=1200.0, step_length_s=0.5, warmup_s=240.0)
    try:
        previous = env.reset()           # x(240)
        env._advance(60.0)               # to 300, uncommanded, as the warm-up would be
        env.mark_window()
        marked = False
        while True:
            now = env.observe()
            env._command(greedy_actions(model, torch.tensor(previous, dtype=torch.float)).tolist())
            env._advance(60.0)
            previous = now
            if not marked and env.time_s() >= 900.0:
                env.mark_window()
                marked = True
            if env.time_s() >= 1200.0:
                break
        direct = env.metrics()
    finally:
        env.close()
    assert stale["arrived"] == direct["arrived"]
    assert stale["flow"] == direct["flow_veh_per_h"]
    assert stale["held_at_entry"] == direct["held_at_entry"]
