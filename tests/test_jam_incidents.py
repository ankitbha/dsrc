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
    """The handful of TraCI calls `MainzEnv` makes, answered from plain dictionaries."""

    def __init__(self):
        self.time = 0.0
        self.edge_ids: dict[str, list[str]] = {}
        self.edge_mean_mps: dict[str, float] = {}
        self.speed_mps: dict[str, float] = {}
        self.arrived: list[str] = []
        self.departed: list[str] = []
        self.pendingv: list[str] = []
        self.calls: list[tuple] = []
        fake = self
        self.edge = SimpleNamespace(
            getLastStepVehicleNumber=lambda e: len(fake.edge_ids.get(e, [])),
            getLastStepMeanSpeed=lambda e: fake.edge_mean_mps.get(e, 0.0),
            getLastStepVehicleIDs=lambda e: list(fake.edge_ids.get(e, [])),
        )
        self.vehicle = SimpleNamespace(
            getSpeed=lambda v: fake.speed_mps[v],
            add=lambda *a, **k: fake.calls.append(("add", fake.time, a, k)),
            setStop=lambda *a, **k: fake.calls.append(("setStop", fake.time, a, k)),
            remove=lambda *a, **k: fake.calls.append(("remove", fake.time, a, k)),
            # SUMO stamps a departure with the step it was inserted in, one step before
            # the clock `simulationStep` leaves behind.
            getDeparture=lambda v: fake.time - 0.5,
            getIDCount=lambda: sum(len(ids) for ids in fake.edge_ids.values()),
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

    def test_added_at_the_first_step_at_or_after_the_start_and_removed_at_the_end(self, fake):
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
        fake.time = 1199.5
        env._admit()
        assert not [c for c in fake.calls if c[0] == "remove"]
        fake.time = 1200.0
        env._admit()
        removes = [c for c in fake.calls if c[0] == "remove"]
        assert [(c[1], c[2][0]) for c in removes] == [(1200.0, "incident_0")]
        env._admit()
        assert len([c for c in fake.calls if c[0] == "remove"]) == 1

    def test_the_report_records_insertion_and_removal(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        fake.time = 1200.0
        env._admit()
        report = env.incident_report()
        assert [r["id"] for r in report] == ids
        assert report[0]["inserted_s"] == 900.0
        assert report[0]["removed_s"] == 1200.0
        assert report[0]["edge"] == "14" and report[0]["lane"] == 0

    def test_a_bad_spec_is_refused(self, fake):
        env = _env(fake)
        with pytest.raises(ValueError):
            env.schedule_incident("14", [], 600.0, 900.0, 300.0)
        with pytest.raises(ValueError):
            env.schedule_incident("14", [0], 600.0, 900.0, 0.0)


class TestExclusion:
    def test_an_edge_with_ten_vehicles_and_one_incident_reports_ten_at_their_speed(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        traffic = [f"v{k}" for k in range(10)]
        fake.edge_ids["14"] = traffic + ids
        for v in traffic:
            fake.speed_mps[v] = 50.0 / 3.6
        fake.speed_mps[ids[0]] = 0.0
        # SUMO's mean over all eleven, the incident's zero included.
        fake.edge_mean_mps["14"] = 10 * (50.0 / 3.6) / 11
        edge = env._per_edge()["14"]
        assert edge["count"] == 10.0
        assert edge["speed_kmh"] == pytest.approx(50.0)
        assert edge["density"] == pytest.approx(10 * 4.5 / 300.0)

    def test_an_edge_holding_only_the_incident_is_empty(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        fake.edge_ids["14"] = list(ids)
        fake.speed_mps[ids[0]] = 0.0
        fake.edge_mean_mps["14"] = 0.0
        edge = env._per_edge()["14"]
        assert edge["count"] == 0.0
        assert math.isnan(edge["speed_kmh"])

    def test_segment_state_and_segment_vehicles_exclude_the_incident(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        segment = next(i for i, s in enumerate(env.segments) if "14" in s)
        fake.edge_ids["14"] = ["v0", ids[0]]
        fake.speed_mps.update({"v0": 10.0, ids[0]: 0.0})
        fake.edge_mean_mps["14"] = 5.0
        state = env.segment_state()
        assert state["count"][segment] == 1
        assert state["speed_kmh"][segment] == pytest.approx(36.0)
        assert env._segment_vehicles()[segment] == {"v0"}

    def test_an_edge_without_an_incident_is_read_exactly_as_before(self, fake):
        env = _env(fake)
        _put_incident_on_road(env, fake)
        fake.edge_ids["135"] = ["a", "b"]
        fake.edge_mean_mps["135"] = 7.0
        edge = env._per_edge()["135"]
        assert edge["count"] == 2.0 and edge["speed_kmh"] == pytest.approx(7.0 * 3.6)

    def test_arrivals_pending_and_running_exclude_incident_ids(self, fake):
        env = _env(fake)
        ids = _put_incident_on_road(env, fake)
        before = env.arrived_total
        fake.arrived = ["v1", "v2", ids[0]]
        env._step()
        assert env.arrived_total - before == 2
        fake.arrived = []
        fake.pendingv = ["p1", "incident_9"]
        env._incident_ids.add("incident_9")
        assert env.pending() == 1
        fake.edge_ids["14"] = ["v0", ids[0]]
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


def test_the_exit_block_stands_stopped_until_its_end_and_jams_segment_11():
    seen, arrived, speeds, report, metrics = _run_bg(1, I_EXIT)
    base_seen, base_arrived, _, _, _ = _run_bg(1, None)
    assert report[0]["inserted_s"] is not None
    inserted = report[0]["inserted_s"]
    # SUMO inserts a stopped vehicle only once the spot is clear (measured 0 to 15.5 s
    # after the request on three seeds); the request is at 900 s.
    assert 900.0 <= inserted <= 930.0
    times = [t for t, *_ in seen]
    assert min(times) == inserted + 0.5
    assert max(times) == 1200.0
    assert len(times) == int(round((1200.0 - inserted) / 0.5))
    assert all(speed == 0.0 for _, _, speed, _, _ in seen)
    assert {lane for *_, lane, _ in seen} == {"exit_road_0"}
    assert {pos for *_, pos in seen} == {100.0}
    assert report[0]["removed_s"] == 1200.0
    # Nothing before the block differs.
    assert {t: n for t, n in arrived.items() if t <= 900.0} == {
        t: n for t, n in base_arrived.items() if t <= 900.0}
    # The queue reaches segment 11 within 300 s of the block.
    jammed = [t for t, v in speeds.items()
              if 900.0 < t <= 1200.0 and v is not None and not math.isnan(v) and v < 19.0]
    assert jammed, "segment 11 never fell below 19 km/h during the block"
    assert metrics["colliding_vehicles"] == 0 and metrics["collision_teleports"] == 0


def test_a_mid_network_block_is_absent_from_its_segment_counts():
    import train_mainz_src as tms
    from src.sumo.binding import _sumo
    from src.sumo.mainz import HERE_FEATURES

    env = tms.make_env(tms.SCENARIOS["mainz_bg"], seed=1, duration_s=1000.0,
                       features=HERE_FEATURES, step_length=0.5, gate_entries=False)
    checked = 0
    try:
        env.reset()
        ids = env.schedule_incident("14", [0, 1], 600.0, 900.0, 300.0)
        segment = next(i for i, s in enumerate(env.segments) if "14" in s)
        while env.time_s() < 1000.0:
            env.release()
            env._step()
            on_road = [v for v in ids if v in set(_sumo.edge.getLastStepVehicleIDs("14"))]
            if on_road:
                raw = sum(len(_sumo.edge.getLastStepVehicleIDs(e)) for e in env.segments[segment])
                assert env.segment_state()["count"][segment] == raw - len(on_road)
                assert env.observe() is not None
                checked += 1
    finally:
        env.close()
    assert checked > 100
