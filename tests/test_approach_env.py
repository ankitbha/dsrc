"""`MainzEnv` on the approach boundary (approach plan sections 3, 4.1 and 5.4, step S4).

Unit tests swap the env's SUMO binding for a small fake and need no SUMO. The sanity tests
at the bottom run SUMO on the built approach net; run this file in its own process.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.sumo.mainz as mainz  # noqa: E402
from src.sumo.mainz import MainzEnv, contiguous_queue, jam_factor  # noqa: E402

DATA = REPO_ROOT / "data" / "mainz"
APPROACH = DATA / "mainz_approach.json"
ENTRIES = ("277", "289", "231", "255", "140", "182", "293", "295")


class TestContiguousQueue:
    def test_the_queue_ends_at_the_first_gap_over_100_m(self):
        assert contiguous_queue([0.0, 50.0, 140.0, 300.0, 2500.0], gap=100.0) == 140.0

    def test_order_does_not_matter(self):
        assert contiguous_queue([300.0, 140.0, 2500.0, 0.0, 50.0], gap=100.0) == 140.0

    def test_no_slow_vehicle_is_no_queue(self):
        assert contiguous_queue([], gap=100.0) == 0.0

    def test_a_vehicle_more_than_the_gap_from_the_entry_is_not_a_queue(self):
        assert contiguous_queue([120.0], gap=100.0) == 0.0
        assert contiguous_queue([100.0], gap=100.0) == 100.0

    def test_a_long_chain_is_followed_to_its_end(self):
        chain = [99.0 * k for k in range(30)]
        assert contiguous_queue(chain, gap=100.0) == pytest.approx(99.0 * 29)


# ------------------------------------------------------------------- the fake


class FakeSumo:
    """Edges hold vehicles at lane positions and speeds; every lane is 1,000 m at 60 km/h."""

    LIMIT_MPS = 60.0 / 3.6

    def __init__(self):
        self.on: dict[str, list[str]] = {}
        self.position: dict[str, float] = {}
        self.speed: dict[str, float] = {}
        self.pending: list[str] = []
        fake = self
        self.edge = SimpleNamespace(
            getLastStepVehicleIDs=lambda e: list(fake.on.get(e, [])),
            getLastStepVehicleNumber=lambda e: len(fake.on.get(e, [])),
            getLastStepMeanSpeed=lambda e: (sum(fake.speed[v] for v in fake.on[e])
                                            / len(fake.on[e]) if fake.on.get(e)
                                            else fake.LIMIT_MPS),
            getLaneNumber=lambda e: 2)
        self.lane = SimpleNamespace(getLength=lambda lane: 1000.0,
                                    getMaxSpeed=lambda lane: fake.LIMIT_MPS)
        self.vehicle = SimpleNamespace(getSpeed=lambda v: fake.speed[v],
                                       getLanePosition=lambda v: fake.position[v],
                                       getIDCount=lambda: sum(map(len, fake.on.values())),
                                       getTypeID=lambda v: "av")
        self.simulation = SimpleNamespace(getTime=lambda: 600.0,
                                          getPendingVehicles=lambda: list(fake.pending))

    def put(self, edge, vehicle, position_m, speed_kmh):
        self.on.setdefault(edge, []).append(vehicle)
        self.position[vehicle] = position_m
        self.speed[vehicle] = speed_kmh / 3.6


@pytest.fixture()
def fake(monkeypatch):
    sumo = FakeSumo()
    monkeypatch.setattr(mainz, "_sumo", sumo)
    return sumo


def _env(fake, approach=True):
    env = MainzEnv(seed=1, step_length_s=0.5, gate_entries=False,
                   approach_file=APPROACH if approach else None,
                   net_file=DATA / "mainz_approach.net.xml",
                   route_file=DATA / "mainz_bg_A.rou.xml",
                   schedule_file=DATA / ("mainz_bg_A_schedule.json" if approach
                                         else "mainz_bg_schedule.json"))
    env._load_static()
    env._running = True
    return env


class TestWithoutApproachFile:
    def test_the_tables_are_the_identity_on_the_twelve(self, fake):
        env = _env(fake, approach=False)
        assert env.approach_segments == [] and env.all_segments == env.segments
        assert env.plan_row == list(range(12))
        assert env.downstream == [list(range(s + 1, 12)) for s in range(12)]
        assert env.observe_approaches().shape == (0, 5)
        assert env.approach_report() == {"on_approach": 0, "queue_m": [],
                                         "slow_near_insertion": []}

    def test_segment_state_and_the_scan_are_unchanged(self, fake):
        env = _env(fake, approach=False)
        fake.put(env.segments[3][0], "v140_0", 200.0, 30.0)
        fake.put(env.segments[3][0], "v140_1", 220.0, 50.0)
        fake.put("a277_0", "v277_3572", 100.0, 5.0)  # an approach edge this env does not know
        state = env.segment_state()
        assert len(state["count"]) == 12 and state["count"][3] == 2
        assert state["speed_kmh"][3] == pytest.approx(40.0)
        assert env.controlled_on_segments() == [("v140_0", env.segments[3][0], 3),
                                                ("v140_1", env.segments[3][0], 3)]

    def test_the_gate_cannot_be_on_with_approach_roads(self, fake):
        with pytest.raises(ValueError):
            MainzEnv(gate_entries=True, approach_file=APPROACH)


class TestApproachTables:
    def test_36_super_segments_and_the_section_4_1_examples(self, fake):
        env = _env(fake)
        assert len(env.segments) == 12 and len(env.approach_segments) == 24
        assert len(env.all_segments) == len(env.plan_row) == len(env.downstream) == 36
        assert env.approach_segments[0] == ["a277_0"] and env.approach_segments[20] == ["a293_2"]
        assert env.downstream[12] == [13, 14] + list(range(12))
        assert env.downstream[32] == [8, 9, 10, 11]
        assert env.plan_row[27] == 5 and env.plan_row[26] == 5 and env.plan_row[21] == 2

    def test_approach_vehicles_are_scanned_with_their_indices(self, fake):
        env = _env(fake)
        fake.put("a140_2", "v140_0", 500.0, 50.0)
        fake.put(env.segments[0][0], "v277_3572", 10.0, 50.0)
        scan = env.controlled_on_segments()
        assert ("v140_0", "a140_2", 26) in scan
        assert ("v277_3572", env.segments[0][0], 0) in scan
        assert {s for _, _, s in scan} <= set(range(36))


class TestApproachState:
    def test_segment_state_has_36_columns(self, fake):
        env = _env(fake)
        fake.put("a289_1", "v289_4564", 300.0, 12.0)
        fake.put("a289_1", "v289_4565", 320.0, 18.0)
        state = env.segment_state()
        assert len(state["speed_kmh"]) == len(state["density"]) == len(state["count"]) == 36
        assert state["count"][16] == 2 and state["speed_kmh"][16] == pytest.approx(15.0)
        assert state["density"][16] == pytest.approx(2 * 4.5 / 2000.0)
        assert sum(state["count"][:12]) == 0

    def test_observe_stays_12_by_5_and_the_approach_rows_follow_its_formula(self, fake):
        env = _env(fake)
        fake.put("a255_0", "v255_2977", 100.0, 24.0)
        assert env.observe().shape == (12, 5)
        rows = env.observe_approaches()
        assert rows.shape == (24, 5)
        row = rows[3 * 3 + 0]  # entry 255 is j = 3, its far-end edge i = 0
        assert row[0] == pytest.approx(24.0) and row[1] == pytest.approx(60.0)
        assert row[2] == pytest.approx(jam_factor(24.0, 60.0))
        assert row[3] == 2.0 and row[4] == pytest.approx(1.0)
        empty = rows[0]
        assert empty[0] == 0.0 and empty[2] == 0.0 and empty[1] == pytest.approx(60.0)


class TestApproachReport:
    def test_queue_from_the_entry_with_a_100_m_gap(self, fake):
        env = _env(fake)
        # Approach 182 (j = 5): distances upstream of the entry's start are 1,000 m per
        # edge after the vehicle's own plus (1,000 - position).
        fake.put("a182_2", "q1", 990.0, 2.0)   # 10 m
        fake.put("a182_2", "q2", 920.0, 5.0)   # 80 m
        fake.put("a182_2", "q3", 840.0, 10.0)  # 160 m
        # 210 m at 25 km/h: above the 19 km/h queue speed, so not a member, although it is
        # within 100 m of q3 (validation R1-5, mutant MZ3: a 30 km/h threshold took it in).
        fake.put("a182_2", "c1", 790.0, 25.0)
        fake.put("a182_2", "f1", 700.0, 50.0)  # 300 m, moving: not in the queue
        fake.put("a182_2", "q4", 600.0, 3.0)   # 400 m, 240 m beyond q3: ends the queue
        fake.put("a182_0", "n1", 150.0, 15.0)  # slow near the far end, 2,850 m
        report = env.approach_report()
        assert report["queue_m"][5] == pytest.approx(160.0)
        assert report["slow_near_insertion"][5] == 1
        assert report["on_approach"] == 7
        assert report["queue_m"][:5] == [0.0] * 5 and len(report["queue_m"]) == 8

    def test_a_queue_reaching_the_far_edge_is_measured_across_edges(self, fake):
        env = _env(fake)
        for k in range(30):  # 0 to 2,900 m upstream of entry 295, every 100 m
            d = 100.0 * k
            edge = f"a295_{2 - int(d // 1000)}"
            fake.put(edge, f"s{k}", 1000.0 - d % 1000.0, 1.0)
        report = env.approach_report()
        assert report["queue_m"][7] == pytest.approx(2900.0)
        assert report["slow_near_insertion"][7] == 2  # 2,800 and 2,900 m: within 200 m


# ----------------------------------------------------------------- SUMO sanity


@pytest.fixture(scope="module")
def meta():
    return json.loads(APPROACH.read_text())


def _model():
    import evaluate_mainz_checkpoints as emc

    return emc._build_and_load(REPO_ROOT / "results/checkpoints/mainz_here_best.pt",
                               emc.CHECKPOINT_FEATURES["here"], 12)


def test_an_840_s_plan_episode_on_the_approach_boundary(meta):
    import torch

    from src.rl.src_q import greedy_actions

    model = _model()
    W = meta["W"]
    env = MainzEnv(seed=1, duration_s=600.0 + W, warmup_s=300.0 + W, step_length_s=0.5,
                   gate_entries=False, approach_file=APPROACH,
                   net_file=DATA / "mainz_approach.net.xml",
                   route_file=DATA / "mainz_b9000_A.rou.xml",
                   schedule_file=DATA / "mainz_b9000_A_schedule.json")
    seen_approach, held, pending = set(), [], []
    try:
        state = env.reset()
        while True:
            assert state.shape == (12, 5)
            assert len(env.segment_state()["count"]) == 36
            assert env.observe_approaches().shape == (24, 5)
            scan = env.controlled_on_segments()
            seen_approach |= {s for _, _, s in scan if s >= 12}
            held.append(env.held_at_entry())
            pending.append(env.pending())
            action = greedy_actions(model, torch.tensor(state, dtype=torch.float))
            state, _, done = env.step(action.tolist())
            if done:
                break
        metrics = env.metrics()
        report = env.approach_report()
    finally:
        env.close()
    assert seen_approach and seen_approach <= set(range(12, 36))
    assert set(held) == {0} and metrics["admitted"] == -1
    assert set(pending) == {0}
    assert metrics["colliding_vehicles"] == 0 and metrics["collision_teleports"] == 0
    assert report["on_approach"] > 0 and len(report["queue_m"]) == 8


def _follow(cap: bool) -> list[tuple[float, float, float, float, float]]:
    """Section 6's follower test on the one-lane approach 255: an `av` leader at 0 s,
    capped at half of 16.67 m/s at 5 s; a `human` follower at 6 s. Rows of `(time, leader
    speed, follower speed, gap, leader position - follower position)`."""
    from sumolib import checkBinary

    from src.sumo.binding import _sumo

    _sumo.start([checkBinary("sumo"), "-n", str(DATA / "mainz_approach.net.xml"),
                 "-r", str(DATA / "mainz_bg_A_routes.rou.xml"), "--step-length", "0.5",
                 "--seed", "1", "--no-step-log", "true", "--no-warnings", "true",
                 "--max-depart-delay", "-1", "--time-to-teleport", "-1"])
    rows = []
    try:
        _sumo.vehicle.add("lead", "r255", typeID="av", depart="now", departLane="best",
                          departSpeed="max")
        while _sumo.simulation.getTime() < 120.0:
            t = _sumo.simulation.getTime()
            if cap and abs(t - 5.0) < 1e-9:
                _sumo.vehicle.setMaxSpeed("lead", 0.5 * 16.67)
            if abs(t - 6.0) < 1e-9:
                _sumo.vehicle.add("follow", "r255", typeID="human", depart="now",
                                  departLane="best", departSpeed="max")
            _sumo.simulationStep()
            ids = _sumo.vehicle.getIDList()
            if "lead" in ids and "follow" in ids:
                distance = _sumo.vehicle.getDrivingDistance(
                    "follow", _sumo.vehicle.getRoadID("lead"),
                    _sumo.vehicle.getLanePosition("lead"))
                leader = _sumo.vehicle.getLeader("follow", 500.0)
                rows.append((_sumo.simulation.getTime(), _sumo.vehicle.getSpeed("lead"),
                             _sumo.vehicle.getSpeed("follow"),
                             leader[1] if leader else math.inf, distance))
    finally:
        _sumo.close()
    return rows


def _held_behind(rows) -> bool:
    window = [r for r in rows if 36.0 <= r[0] <= 96.0]
    return bool(window) and all(
        abs(follower - leader) <= 0.5 and abs(follower - 0.5 * 16.67) <= 0.5 and gap >= 2.5
        and ahead > 0.0 for _, leader, follower, gap, ahead in window)


def test_a_slowed_controlled_vehicle_holds_an_uncontrolled_one_behind_it_on_approach_255():
    rows = _follow(cap=True)
    assert _held_behind(rows)
    assert all(ahead > 0.0 for *_, ahead in rows)  # the follower never passes


def test_the_follower_test_fails_when_the_leader_is_not_capped():
    assert not _held_behind(_follow(cap=False))
