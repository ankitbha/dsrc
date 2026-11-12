"""Incidents in `MainzEnv` (plan section 4.5, step S2).

An incident is one stopped vehicle per blocked lane. It is on the road, so cameras see it,
but it is not traffic: it is excluded from edge counts and speeds, from the segment state
the feed snapshots, from arrivals, and from the pending and running counts.

The unit tests swap the env's SUMO binding for a fake whose edges, vehicles and step
results the test sets. The sanity tests at the bottom run SUMO; run this file in its own
process.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import src.sumo.mainz as mainz  # noqa: E402
from src.sumo.mainz import MainzEnv  # noqa: E402


class FakeSumo:
    """The TraCI calls `MainzEnv` makes, answered the way SUMO 1.27 was measured to answer.

    Vehicles sit on lanes (`place`). A lane's mean speed leaves out stopped vehicles on an
    edge with more than one lane and keeps them on a one-lane edge (SUMO skips stopped
    vehicles only where the edge has a lane changer), and an empty lane reads its speed
    limit. An edge's mean speed is SUMO's `sum_l max(n_l, 1) v_l / sum_l max(n_l, 1)` over
    its lanes, `n_l` counting every vehicle, stopped ones included (measured equal to
    `edge.getLastStepMeanSpeed` to 5e-15 m/s on every segment edge).
    """

    LIMIT_MPS = 60.0 / 3.6

    def __init__(self):
        self.time = 0.0
        self.lanes: dict[tuple[str, int], list[str]] = {}
        self.lane_number: dict[str, int] = {}
        self.speed_mps: dict[str, float] = {}
        self.stopped: set[str] = set()
        self.arrived: list[str] = []
        self.departed: list[str] = []
        self.pendingv: list[str] = []
        self.calls: list[tuple] = []
        fake = self
        self.edge = SimpleNamespace(
            getLastStepVehicleNumber=lambda e: len(fake._edge_ids(e)),
            getLastStepMeanSpeed=fake._edge_mean,
            getLastStepVehicleIDs=fake._edge_ids,
            getLaneNumber=lambda e: fake.lane_number.get(e, 3),
        )
        self.lane = SimpleNamespace(
            getLastStepVehicleIDs=lambda lane: list(fake.lanes.get(fake._key(lane), [])),
            getLastStepVehicleNumber=lambda lane: len(fake.lanes.get(fake._key(lane), [])),
            getLastStepMeanSpeed=lambda lane: fake._lane_mean(*fake._key(lane)),
            getMaxSpeed=lambda lane: fake.LIMIT_MPS,
        )
        self.vehicle = SimpleNamespace(
            getSpeed=lambda v: fake.speed_mps[v],
            add=lambda *a, **k: fake.calls.append(("add", fake.time, a, k)),
            setStop=lambda *a, **k: fake.calls.append(("setStop", fake.time, a, k)),
            remove=lambda *a, **k: fake.calls.append(("remove", fake.time, a, k)),
            # SUMO stamps a departure with the step it was inserted in, one step before
            # the clock `simulationStep` leaves behind.
            getDeparture=lambda v: fake.time - 0.5,
            getIDCount=lambda: sum(len(ids) for ids in fake.lanes.values()),
            getTypeID=lambda v: "human",
        )
        self.route = SimpleNamespace(
            add=lambda *a, **k: fake.calls.append(("route", fake.time, a, k)))
        self.simulation = SimpleNamespace(
            getTime=lambda: fake.time,
            getArrivedNumber=lambda: len(fake.arrived),
            getArrivedIDList=lambda: list(fake.arrived),
            getDepartedIDList=lambda: list(fake.departed),
            getCollidingVehiclesNumber=lambda: 0,
            getStartingTeleportIDList=lambda: [],
            getEndingTeleportIDList=lambda: [],
            getPendingVehicles=lambda: list(fake.pendingv),
        )

    @staticmethod
    def _key(lane_id: str) -> tuple[str, int]:
        edge, index = lane_id.rsplit("_", 1)
        return edge, int(index)

    def place(self, edge, lane, vehicles, speed_mps, stopped=False):
        self.lanes.setdefault((edge, lane), []).extend(vehicles)
        for v in vehicles:
            self.speed_mps[v] = speed_mps
            if stopped:
                self.stopped.add(v)

    def _edge_ids(self, edge):
        return [v for i in range(self.lane_number.get(edge, 3))
                for v in self.lanes.get((edge, i), [])]

    def _lane_mean(self, edge, lane):
        multi = self.lane_number.get(edge, 3) > 1
        counted = [v for v in self.lanes.get((edge, lane), [])
                   if not (multi and v in self.stopped)]
        if not counted:
            return self.LIMIT_MPS
        return sum(self.speed_mps[v] for v in counted) / len(counted)

    def _edge_mean(self, edge):
        weight = total = 0.0
        for i in range(self.lane_number.get(edge, 3)):
            n = max(len(self.lanes.get((edge, i), [])), 1)
            weight += n
            total += n * self._lane_mean(edge, i)
        return total / weight

    def simulationStep(self):
        self.time += 0.5


@pytest.fixture()
def fake(monkeypatch):
    sumo = FakeSumo()
    monkeypatch.setattr(mainz, "_sumo", sumo)
    return sumo


def _env(fake) -> MainzEnv:
    env = MainzEnv(seed=1, step_length_s=0.5, gate_entries=False)
    env._static = {edge: {"lanes": 3.0, "length_m": 100.0, "free_flow_kmh": 60.0}
                   for segment in env.segments for edge in segment}
    env._running = True
    return env


def _put_incident_on_road(env, fake, edge="14"):
    ids = env.schedule_incident(edge, [0], 600.0, 900.0, 300.0)
    fake.time = 900.0
    env._admit()
    fake.departed = ids
    env._step()
    fake.departed = []
    return ids


class TestScheduling:
    def test_one_vehicle_per_lane_registered_as_uncontrolled(self, fake):
        env = _env(fake)
        before = env.schedule_size
        ids = env.schedule_incident("14", [0, 1], 600.0, 900.0, 300.0)
        assert ids == ["incident_0", "incident_1"]
        assert env.schedule_size == before + 2
        assert [env.position(v) for v in ids] == [before, before + 1]
        assert not any(env.controlled(v) for v in ids)

    def test_added_at_the_first_step_at_or_after_the_start(self, fake):
        env = _env(fake)
        env.schedule_incident("14", [1], 600.0, 900.0, 300.0)
        fake.time = 899.5
        env._admit()
        assert not [c for c in fake.calls if c[0] == "add"]
        fake.time = 900.0
        env._admit()
        adds = [c for c in fake.calls if c[0] == "add"]
        assert len(adds) == 1
        _, t, args, kwargs = adds[0]
        assert t == 900.0 and args[0] == "incident_0"
        assert kwargs["departLane"] == "1" and kwargs["departPos"] == "600.0"
        assert kwargs["departSpeed"] == "0" and kwargs["typeID"] == "human"
        stops = [c for c in fake.calls if c[0] == "setStop"]
        assert stops[0][2][:2] == ("incident_0", "14")
        assert stops[0][3] == {"pos": 600.0, "laneIndex": 1, "duration": 300.0}

    def test_a_late_insertion_is_removed_at_its_own_insertion_plus_the_duration(self, fake):
        # SUMO found room at 911.5 s: the block stands 300 s from then, to 1,211.5 s, not
        # to 1,200 s (validation R1-2: a fixed end gave each arm its own block length).
        env = _env(fake)
        ids = env.schedule_incident("14", [1], 600.0, 900.0, 300.0)
        fake.time = 900.0
        env._admit()
        fake.time = 911.5
        env._admit()
        fake.departed = ids
        env._step()  # inserted in the step at 911.5 s
        fake.departed = []
        for t in (1200.0, 1211.0):
            fake.time = t
            env._admit()
        assert not [c for c in fake.calls if c[0] == "remove"]
        fake.time = 1211.5
        env._admit()
        removes = [c for c in fake.calls if c[0] == "remove"]
        assert [(c[1], c[2][0]) for c in removes] == [(1211.5, "incident_0")]
        report = env.incident_report()[0]
        assert report["inserted_s"] == 911.5 and report["removed_s"] == 1211.5
        assert report["standing_s"] == 300.0
        env._admit()
        assert len([c for c in fake.calls if c[0] == "remove"]) == 1

    def test_a_request_never_inserted_is_withdrawn_at_the_end_of_its_window(self, fake):
        env = _env(fake)
        env.schedule_incident("14", [1], 600.0, 900.0, 300.0)
        fake.time = 900.0
        env._admit()
        fake.time = 1199.5
        env._admit()
        assert not [c for c in fake.calls if c[0] == "remove"]
        fake.time = 1200.0
        env._admit()
        assert [c[2][0] for c in fake.calls if c[0] == "remove"] == ["incident_0"]
        report = env.incident_report()[0]
        assert report["never_inserted"] is True and report["standing_s"] == 0.0

    def test_the_report_records_insertion_and_removal(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        fake.time = 1200.0
        env._admit()
        report = env.incident_report()
        assert [r["id"] for r in report] == ids
        assert report[0]["inserted_s"] == 900.0
        assert report[0]["removed_s"] == 1200.0 and report[0]["standing_s"] == 300.0
        assert report[0]["never_inserted"] is False
        assert report[0]["edge"] == "14" and report[0]["lane"] == 0

    def test_a_bad_spec_is_refused(self, fake):
        env = _env(fake)
        with pytest.raises(ValueError):
            env.schedule_incident("14", [], 600.0, 900.0, 300.0)
        with pytest.raises(ValueError):
            env.schedule_incident("14", [0], 600.0, 900.0, 0.0)


class TestExclusion:
    def _incident(self, env, fake, edge="14", lane=0):
        ids = env.schedule_incident(edge, [lane], 600.0, 900.0, 300.0)
        fake.time = 900.0
        env._admit()
        fake.place(edge, lane, ids, 0.0, stopped=True)
        fake.departed = ids
        env._step()
        fake.departed = []
        return ids

    def test_an_edge_with_ten_vehicles_and_one_incident_reports_ten_at_their_speed(self, fake):
        env = _env(fake)
        self._incident(env, fake)
        fake.place("14", 0, [f"a{k}" for k in range(3)], 50.0 / 3.6)
        fake.place("14", 1, [f"b{k}" for k in range(4)], 50.0 / 3.6)
        fake.place("14", 2, [f"c{k}" for k in range(3)], 50.0 / 3.6)
        # SUMO's lane mean already leaves the stopped vehicle out, so its raw edge mean is
        # 50 km/h with the incident counted in lane 0's weight; taking a zero out of that
        # mean a second time read 55 km/h.
        assert fake.edge.getLastStepMeanSpeed("14") == pytest.approx(50.0 / 3.6)
        edge = env._per_edge()["14"]
        assert edge["count"] == 10.0
        assert edge["speed_kmh"] == pytest.approx(50.0)
        assert edge["density"] == pytest.approx(10 * 4.5 / 300.0)

    def test_a_lane_left_empty_reads_its_limit_with_weight_one_as_sumo_does(self, fake):
        env = _env(fake)
        self._incident(env, fake)
        fake.place("14", 1, [f"b{k}" for k in range(5)], 10.0)
        fake.place("14", 2, [f"c{k}" for k in range(5)], 10.0)
        expected = (1 * FakeSumo.LIMIT_MPS + 5 * 10.0 + 5 * 10.0) / 11.0
        count, speed = env._edge_reading("14", env._blocked_edges())
        assert count == 10 and speed == pytest.approx(expected)

    def test_on_a_one_lane_edge_the_stopped_vehicle_s_zero_is_taken_out(self, fake):
        env = _env(fake)
        fake.lane_number["146"] = 1
        env._static["146"] = {"lanes": 1.0, "length_m": 100.0, "free_flow_kmh": 60.0}
        self._incident(env, fake, edge="146")
        fake.place("146", 0, [f"a{k}" for k in range(4)], 10.0)
        assert fake.lane.getLastStepMeanSpeed("146_0") == pytest.approx(8.0)
        assert env._edge_reading("146", env._blocked_edges()) == (4, pytest.approx(10.0))

    def test_an_edge_holding_only_the_incident_is_empty(self, fake):
        env = _env(fake)
        self._incident(env, fake)
        edge = env._per_edge()["14"]
        assert edge["count"] == 0.0
        assert math.isnan(edge["speed_kmh"])

    def test_segment_state_and_segment_vehicles_exclude_the_incident(self, fake):
        env = _env(fake)
        ids = self._incident(env, fake)
        segment = next(i for i, s in enumerate(env.segments) if "14" in s)
        fake.place("14", 0, ["v0"], 10.0)
        state = env.segment_state()
        assert state["count"][segment] == 1
        # Lane 0 holds v0 at 10 m/s; lanes 1 and 2 are empty and read the limit, weight 1.
        assert state["speed_kmh"][segment] == pytest.approx(
            3.6 * (10.0 + 2 * FakeSumo.LIMIT_MPS) / 3.0)
        assert env._segment_vehicles()[segment] == {"v0"}
        assert ids[0] not in env._segment_vehicles()[segment]

    def test_an_edge_without_an_incident_is_read_exactly_as_before(self, fake):
        env = _env(fake)
        self._incident(env, fake)
        fake.place("135", 0, ["a"], 7.0)
        fake.place("135", 2, ["b", "c"], 4.0)
        edge = env._per_edge()["135"]
        assert edge["count"] == 3.0
        assert edge["speed_kmh"] == 3.6 * fake.edge.getLastStepMeanSpeed("135")

    def test_arrivals_pending_and_running_exclude_incident_ids(self, fake):
        env = _env(fake)
        ids = self._incident(env, fake)
        before = env.arrived_total
        fake.arrived = ["v1", "v2", ids[0]]
        env._step()
        assert env.arrived_total - before == 2
        fake.arrived = []
        fake.pendingv = ["p1", "incident_9"]
        env._incident_ids.add("incident_9")
        assert env.pending() == 1
        fake.place("14", 1, ["v0"], 10.0)
        assert env.running() == 1

    def test_without_incidents_arrivals_use_the_step_count(self, fake):
        env = _env(fake)
        fake.arrived = ["v1", "v2"]
        env._step()
        assert env.arrived_total == 2


# ----------------------------------------------------------------- SUMO sanity

I_EXIT = {"edge": "exit_road", "lanes": [0], "pos_m": 100.0, "start_s": 900.0,
          "duration_s": 300.0}


def _run_bg(seed: int, incident: dict | None, until: float = 1300.0):
    """No control on mainz_bg, recording the incident vehicle every step after 890 s."""
    import train_mainz_src as tms
    from src.sumo.binding import _sumo
    from src.sumo.mainz import HERE_FEATURES

    env = tms.make_env(tms.SCENARIOS["mainz_bg"], seed=seed, duration_s=until,
                       features=HERE_FEATURES, step_length=0.5, gate_entries=False)
    seen, arrived_at, speeds = [], {}, {}
    try:
        env.reset()
        ids = env.schedule_incident(**incident) if incident else []
        while env.time_s() < until:
            env.release()
            env._step()
            t = env.time_s()
            if t % 10.0 == 0.0:
                arrived_at[t] = env.arrived_total
                speeds[t] = env.segment_state()["speed_kmh"][11]
            if t > 890.0 and ids:
                present = set(_sumo.vehicle.getIDList())
                for v in ids:
                    if v in present:
                        seen.append((t, v, _sumo.vehicle.getSpeed(v),
                                     _sumo.vehicle.getLaneID(v),
                                     _sumo.vehicle.getLanePosition(v)))
        report = env.incident_report()
        metrics = env.metrics()
    finally:
        env.close()
    return seen, arrived_at, speeds, report, metrics


def test_the_exit_block_stands_stopped_for_its_duration_and_jams_segment_11():
    seen, arrived, speeds, report, metrics = _run_bg(1, I_EXIT)
    base_seen, base_arrived, _, _, _ = _run_bg(1, None)
    assert report[0]["inserted_s"] is not None
    inserted = report[0]["inserted_s"]
    # SUMO inserts a stopped vehicle only once the spot is clear (measured 0 to 15.5 s
    # after the request on three seeds); the request is at 900 s.
    assert 900.0 <= inserted <= 930.0
    times = [t for t, *_ in seen]
    # The block stands its full 300 s from its own insertion (validation R1-2): on the
    # road from the step after insertion to the step it is removed in, 600 steps.
    removed = report[0]["removed_s"]
    assert removed == inserted + 300.0 and report[0]["standing_s"] == 300.0
    assert min(times) == inserted + 0.5
    assert max(times) == removed
    assert len(times) == int(round(300.0 / 0.5))
    assert all(speed == 0.0 for _, _, speed, _, _ in seen)
    assert {lane for *_, lane, _ in seen} == {"exit_road_0"}
    assert {pos for *_, pos in seen} == {100.0}
    # Nothing before the block differs.
    assert {t: n for t, n in arrived.items() if t <= 900.0} == {
        t: n for t, n in base_arrived.items() if t <= 900.0}
    # The queue reaches segment 11 within 300 s of the block.
    jammed = [t for t, v in speeds.items()
              if 900.0 < t <= 1200.0 and v is not None and not math.isnan(v) and v < 19.0]
    assert jammed, "segment 11 never fell below 19 km/h during the block"
    assert metrics["colliding_vehicles"] == 0 and metrics["collision_teleports"] == 0


def test_a_mid_network_block_is_absent_from_its_segment_counts_and_speed():
    """I_mid on seed 1: while the block stands, segment 6 (edges 500, 14, 135) counts only
    the other vehicles, and its speed is SUMO's estimator without the incident vehicles.

    Two per-vehicle computations over the non-incident vehicles read from SUMO: SUMO's
    lane-weighted estimator rebuilt vehicle by vehicle, which must match to 1e-9 m/s, and
    the plain vehicle-weighted mean, which differs from SUMO's estimator only through its
    empty lanes (each read at the limit with weight 1). Measured: at most 0.016 m/s apart
    over 600 steps; the tolerance is 0.05 m/s. Removing the incident's zero a second time
    (validation R1-1) put edge 14 up to 0.48 m/s above the plain mean.
    """
    import numpy as np

    import train_mainz_src as tms
    from src.sumo.binding import _sumo
    from src.sumo.mainz import HERE_FEATURES

    env = tms.make_env(tms.SCENARIOS["mainz_bg"], seed=1, duration_s=1250.0,
                       features=HERE_FEATURES, step_length=0.5, gate_entries=False)
    checked = 0
    worst_plain = 0.0
    try:
        env.reset()
        ids = env.schedule_incident("14", [0, 1], 600.0, 900.0, 300.0)
        segment = next(i for i, s in enumerate(env.segments) if "14" in s)
        while env.time_s() < 1250.0:
            env.release()
            env._step()
            on_road = [v for v in ids if v in set(_sumo.edge.getLastStepVehicleIDs("14"))]
            if not on_road:
                continue
            raw = sum(len(_sumo.edge.getLastStepVehicleIDs(e)) for e in env.segments[segment])
            state = env.segment_state()
            assert state["count"][segment] == raw - len(on_road)
            speeds, weighted, vehicles = [], 0.0, 0
            for edge in env.segments[segment]:
                weight = total = 0.0
                n_edge = 0
                for lane in range(_sumo.edge.getLaneNumber(edge)):
                    lane_id = f"{edge}_{lane}"
                    lane_speeds = [_sumo.vehicle.getSpeed(v)
                                   for v in _sumo.lane.getLastStepVehicleIDs(lane_id)
                                   if v not in ids]
                    speeds += lane_speeds
                    w = max(len(lane_speeds), 1)
                    weight += w
                    total += w * (np.mean(lane_speeds) if lane_speeds
                                  else _sumo.lane.getMaxSpeed(lane_id))
                    n_edge += len(lane_speeds)
                if n_edge:
                    weighted += n_edge * total / weight
                    vehicles += n_edge
            if not speeds:
                continue
            reading = state["speed_kmh"][segment] / 3.6
            assert reading == pytest.approx(weighted / vehicles, abs=1e-9)
            assert abs(reading - float(np.mean(speeds))) <= 0.05
            worst_plain = max(worst_plain, abs(reading - float(np.mean(speeds))))
            assert env.observe() is not None
            checked += 1
    finally:
        env.close()
    assert checked > 500
