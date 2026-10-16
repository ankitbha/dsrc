"""The SRC Mainz network in SUMO, with a HERE-shaped observation.

SRC trains a centralized controller on six per-super-segment features computed from
full vehicle-level ground truth: density, lane count, mean speed, mean following gap,
and the counts of vehicles entering and leaving in the last minute. Four of those six
are not quantities a traffic API returns. A vehicle running the policy on the road can
obtain speed, free-flow speed and a jam factor, and nothing else, so a policy trained
on the other four cannot be executed by one.

This environment therefore exposes what HERE returns and keeps the reward on ground
truth. Training is centralized in the simulator and may use privileged information;
execution reads only what a deployed vehicle can read. The reward is SRC's unchanged.

The action is SRC's too: one speed from {30, 45, 60} km/h per super-segment per 60 s,
written to every controlled vehicle on that segment as its maximum speed (`ACTUATIONS`).
The car-following model still chooses the speed under that cap, as Vissim does under
`DesSpeed`, so the advisory is gated by local conditions rather than commanded through
them.
"""
from __future__ import annotations

import bisect
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo.binding import _sumo

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "mainz"

#: SRC's action set, `speeds = [30, 45, 60]` km/h in its train.py.
SPEED_ACTIONS_KMH: tuple[float, ...] = (30.0, 45.0, 60.0)
#: The same three actions as fractions of a segment's own speed limit, which is what
#: they are: Mainz is limited to 60 km/h, so SRC's 30, 45 and 60 are a half, three
#: quarters and all of it. Stated this way the set carries to a network with a different
#: limit -- `inverted_tree` is 108 km/h, where a literal 30/45/60 would be a standing
#: order to crawl rather than a speed advisory.
SPEED_ACTION_FRACTIONS: tuple[float, ...] = (0.5, 0.75, 1.0)
#: How an advisory reaches a controlled vehicle.
#:
#: `max_speed` (the default, plan Amendment 1) caps the vehicle's maximum speed with
#: `setMaxSpeed`, which is what Vissim's `DesSpeed` is: the car-following model keeps
#: choosing the speed, under the cap, so it still brakes for a stopped leader.
#: `set_speed` is the actuation every result before the amendment was produced with:
#: `setSpeed` under EIDM with a 1.0 s action step lets a controlled vehicle run into a
#: stopped leader, and SUMO resolves the collision by teleporting the follower (76
#: collision teleports on seed 16 in 2,500 s; none without control). It is kept only so
#: the committed numbers can be reproduced.
ACTUATIONS: tuple[str, ...] = ("max_speed", "set_speed")
#: SRC's `FEEDBACK_STEP`.
DECISION_INTERVAL_S: float = 60.0
#: The density past which flow falls on THIS network, measured rather than adopted.
#:
#: Re-measured for the EIDM fleet, which replaced the paper's W99 calibration because
#: W99 has no capacity drop on SUMO at all. On a straight two-lane road at Mainz's own
#: 60 km/h, EIDM reaches 1,877 veh/h/lane at 39.6 veh/km/lane and stops at a jam density
#: of 128.5 veh/km/lane. Density here is occupancy, `count * VEHICLE_LENGTH_M /
#: lane_metres`, so 39.6 veh/km/lane is 0.178 of the roadway covered.
#:
#: Against the measured jam density the critical point is 39.6/128.5 = 0.308, which is
#: within 3% of the 0.3 of jam SRC publishes. Under the W99 fleet the same comparison
#: gave 0.134, so SRC's threshold was more than twice the critical density; under EIDM
#: it is the critical density. The published value was not wrong, the fleet was.
DENSITY_CRITICAL: float = 0.178
#: SRC's own value, kept so the two can be compared.
DENSITY_CRITICAL_SRC: float = 0.3
#: The metres of road one vehicle occupies in SRC's density, which fixes the jam
#: density at 1000/4.5 = 222 veh/km/lane.
VEHICLE_LENGTH_M: float = 4.5
#: SRC's reward weights: -100 per super-segment over critical, +0.2 per km/h of mean
#: speed. The penalty is 500 times the speed coefficient, so the objective is
#: overwhelmingly "do not cross critical density".
CONGESTION_PENALTY: float = 100.0
SPEED_WEIGHT: float = 0.2

#: What the deployed observation carries. `speed` and `jam_factor` move; `free_flow`,
#: `lanes` and `length_km` are static and come from the map rather than the API.
HERE_FEATURES: tuple[str, ...] = ("speed", "free_flow", "jam_factor", "lanes", "length_km")
#: SRC's original six, kept so the cost of restricting to the deployable set can be
#: measured by training the same model on each.
SRC_FEATURES: tuple[str, ...] = (
    "density", "lanes", "speed", "gap", "input_rate", "exit_rate",
)


def jam_factor(speed_kmh: float, free_flow_kmh: float) -> float:
    """HERE's 0-10 congestion index, as a stated model rather than a measurement.

    HERE does not publish the formula and folds in incident data we do not have, so
    this is the speed ratio it is monotone in, scaled and clipped. Provenance tier 3:
    a stated theoretical model, not our data and not prior work that measured it.
    """
    if free_flow_kmh <= 0.0:
        return 0.0
    return float(min(10.0, max(0.0, 10.0 * (1.0 - speed_kmh / free_flow_kmh))))


def controlled_index(index: int, penetration: float) -> bool:
    """Whether the vehicle at position `index` of the departure schedule runs the policy.

    `floor((i + 1) p) - floor(i p) == 1`. Over the first `n` vehicles this controls
    exactly `floor(n p)` of them, spread so that two consecutive controlled vehicles are
    never more than `ceil(1 / p)` positions apart. The builder's `--av-fraction` rule,
    `i % round(1 / p) == 0`, cannot express 0.75: `round(1 / 0.75)` is 1, so it controls
    every vehicle.
    """
    return math.floor((index + 1) * penetration) - math.floor(index * penetration) == 1


class MainzEnv:
    """One SUMO episode on the SRC Mainz network.

    A step is one SRC decision: the action is held for `DECISION_INTERVAL_S` of
    simulation while the per-segment aggregates accumulate over that window, which is
    both SRC's feedback interval and roughly HERE's update period.
    """

    def __init__(
        self,
        *,
        seed: int = 1,
        duration_s: float = 2500.0,
        warmup_s: float = 300.0,
        step_length_s: float = 1.0,
        features: tuple[str, ...] = HERE_FEATURES,
        #: Hold a vehicle back while its entry link is at or above the critical
        #: density. See `_admit`: this stands in for the policy acting on a link the
        #: scenario does not simulate, so it is on for a controlled run and off for an
        #: uncontrolled one.
        gate_entries: bool = True,
        net_file: Path | None = None,
        route_file: Path | None = None,
        segment_file: Path | None = None,
        schedule_file: Path | None = None,
        #: The share of vehicles that run the policy, chosen at run time by
        #: `controlled_index` over the schedule's departure order. Only controlled
        #: vehicles are commanded, released or held by the gate.
        penetration: float = 1.0,
        #: One of `ACTUATIONS`.
        actuation: str = "max_speed",
    ) -> None:
        if not 0.0 <= penetration <= 1.0:
            raise ValueError(f"penetration must lie in [0, 1], got {penetration}")
        if actuation not in ACTUATIONS:
            raise ValueError(f"actuation must be one of {ACTUATIONS}, got {actuation!r}")
        self.actuation = actuation
        self.seed = seed
        self.penetration = penetration
        self.duration_s = duration_s
        self.warmup_s = warmup_s
        self.step_length_s = step_length_s
        self.features = features
        self.gate_entries = gate_entries
        self.net_file = net_file or DATA / "mainz.net.xml"
        default_routes = "mainz_routes.rou.xml" if gate_entries else "mainz.rou.xml"
        self.route_file = route_file or DATA / default_routes
        self.segment_file = segment_file or DATA / "mainz_segments.json"
        self.schedule_file = schedule_file or DATA / "mainz_schedule.json"
        self.segments: list[list[str]] = json.loads(self.segment_file.read_text())
        self._running = False
        self._static: dict[str, dict[str, float]] = {}
        self._previous: list[set[str]] = [set() for _ in self.segments]
        self._rates: list[tuple[float, float]] = [(0.0, 0.0) for _ in self.segments]
        self.arrived_total = 0
        self._window_start_s = 0.0
        self._window_start_arrived = 0
        # Vehicle-weighted speed and vehicle count, accumulated over the window so the
        # reported speed is an average of the episode rather than its last instant.
        self._speed_weight = 0.0
        self._vehicle_weight = 0.0
        self._samples = 0
        # Per entry link, the CONTROLLED vehicles scheduled to start there and how many
        # of them have been let in, and the same for the uncontrolled ones, which the
        # gate never holds. Empty when SUMO is doing the inserting.
        self._schedule: dict[str, list[dict[str, Any]]] = {}
        self._admitted: dict[str, int] = {}
        self._departs: dict[str, list[float]] = {}
        self._free: dict[str, list[dict[str, Any]]] = {}
        self._freed: dict[str, int] = {}
        # Schedule position of every vehicle and the set the policy controls. A vehicle
        # added later (a pulse) is appended with the next free position.
        self._index: dict[str, int] | None = None
        self._controlled_ids: set[str] = set()
        self._load_controlled()
        # Under `max_speed`, each capped vehicle's own maximum speed before its first
        # cap, which `release` restores.
        self._capped: dict[str, float] = {}
        # Pulse vehicles not yet handed to SUMO, in departure order.
        self._pulse_queue: list[dict[str, Any]] = []
        # Summed over every step since `reset`: vehicles SUMO reports colliding, and
        # teleports started (all of them collision teleports, since jam teleports are
        # off). `_in_teleport` holds the vehicles whose teleport has not ended.
        self.colliding_vehicles = 0
        self.collision_teleports = 0
        self._in_teleport: set[str] = set()
        # Incidents (`schedule_incident`): their specs, their ids, and the ones standing on
        # the road now, by edge.
        self._incidents: list[dict[str, Any]] = []
        self._incident_ids: set[str] = set()
        self._incidents_on_road: dict[str, str] = {}
        self._incident_lane: dict[str, int] = {}

    # ----------------------------------------------------------------- lifecycle

    def reset(self, on_warmup_tick: Callable[[float], None] | None = None) -> np.ndarray:
        """Start SUMO, run the warm-up, open the window and return the first state.

        With `on_warmup_tick` the warm-up runs through `advance_ticks`, which steps the
        simulation identically and calls back once per second.
        """
        self.close()
        _sumo.start([
            self._binary(), "-n", str(self.net_file), "-r", str(self.route_file),
            "--step-length", str(self.step_length_s),
            "--seed", str(self.seed),
            "--no-step-log", "true", "--no-warnings", "true",
            # A vehicle that cannot be inserted waits rather than vanishing, so
            # demand held back by congestion stays in the scenario.
            "--max-depart-delay", "-1",
            "--time-to-teleport", "-1",
        ])
        self._running = True
        self.arrived_total = 0
        self.colliding_vehicles = 0
        self.collision_teleports = 0
        self._in_teleport = set()
        self._capped = {}
        self._pulse_queue = []
        self._incidents, self._incident_ids, self._incidents_on_road = [], set(), {}
        self._incident_lane = {}
        self._load_controlled()  # drops any pulse or incident a previous episode appended
        self._load_schedule()
        self._load_static()
        self._previous = [set() for _ in self.segments]
        self._rates = [(0.0, 0.0) for _ in self.segments]
        if on_warmup_tick is None:
            self._advance(self.warmup_s)
        else:
            self.advance_ticks(self.warmup_s, on_tick=on_warmup_tick)
        self.mark_window()
        return self.observe()

    def close(self) -> None:
        if self._running:
            try:
                _sumo.close()
            finally:
                self._running = False

    @staticmethod
    def _binary() -> str:
        from sumolib import checkBinary

        return checkBinary("sumo")

    def _load_controlled(self) -> None:
        """Index every scheduled vehicle by departure order and pick the controlled set.

        The schedule rows are written in departure order, so a row's position is its
        departure index. Without a schedule file every vehicle typed `av` is controlled,
        which is what the env did before penetration was a run-time setting.
        """
        if not self.schedule_file.exists():
            self._index, self._controlled_ids = None, set()
            return
        rows = json.loads(self.schedule_file.read_text())
        self._index = {row["id"]: position for position, row in enumerate(rows)}
        self._controlled_ids = {row["id"] for position, row in enumerate(rows)
                                if controlled_index(position, self.penetration)}

    def controlled(self, vehicle_id: str) -> bool:
        """Whether this vehicle runs the policy; an id the env never scheduled is an error."""
        if vehicle_id in self._controlled_ids:
            return True
        if self._index is None:
            return _sumo.vehicle.getTypeID(vehicle_id) == "av"
        if vehicle_id in self._index:
            return False
        raise KeyError(f"{vehicle_id} is not a vehicle this env scheduled")

    def _load_schedule(self) -> None:
        """Read the vehicles the route file omits, grouped by the entry they start at.

        Controlled and uncontrolled vehicles are queued separately per entry: the gate
        holds only the first kind (plan decision D8), so an uncontrolled vehicle is
        inserted at its departure time and never waits behind a held controlled one.
        """
        self._schedule, self._admitted, self._departs = {}, {}, {}
        self._free, self._freed = {}, {}
        if not self.gate_entries:
            return
        rows = json.loads(self.schedule_file.read_text())
        for row in rows:
            queue = self._schedule if self.controlled(row["id"]) else self._free
            self._schedule.setdefault(row["entry"], [])
            self._free.setdefault(row["entry"], [])
            queue[row["entry"]].append(row)
        for entry, queued in self._schedule.items():
            queued.sort(key=lambda row: row["depart"])
            self._free[entry].sort(key=lambda row: row["depart"])
            self._departs[entry] = [row["depart"] for row in queued]
            self._admitted[entry] = 0
            self._freed[entry] = 0

    def _entry_open(self, entry: str) -> bool:
        """Whether the entry link is under the density past which its flow falls."""
        static = self._static[entry]
        lane_metres = static["length_m"] * static["lanes"]
        if not lane_metres:
            return True
        count = int(_sumo.edge.getLastStepVehicleNumber(entry))
        # AN EMPTY ENTRY LINK IS OPEN, and its density is zero rather than undefined.
        # `_per_edge` reports nan for an empty edge so that it does not drag a
        # super-segment's mean down; nan fails every comparison, so reusing it here
        # would shut the gate permanently on exactly the links that are clearest.
        return count * VEHICLE_LENGTH_M / lane_metres < DENSITY_CRITICAL

    def _admit(self) -> None:
        """Grant entry only while the entry link is under the critical density.

        SUMO refuses an insertion when no safe gap remains on the lane, which is a
        harder and much later condition than the one the network is controlled
        against: by the time there is no gap, the entry link is far past the density at
        which its flow peaks, and the capacity drop the controller exists to prevent
        has already happened at the boundary. This holds a vehicle back as soon as its
        entry link reaches the critical density.

        THE GATE BELONGS TO THE POLICY, NOT TO THE NETWORK, and callers must not apply
        it to an uncontrolled run. Inside the network a link is protected by slowing the
        link above it. An entry link's upstream neighbour is outside the map, so there
        is no link to slow, and the gate supplies what the policy would have done there:
        it is a boundary condition on the controlled system, not a piece of road. A run
        with no policy has nothing acting on that upstream link either, so gating its
        entries would credit it with a control action it is not taking.

        The exit meter is the opposite case. A signal standing on a road is part of the
        network whoever is driving, so it applies to every arm.

        A vehicle held here has not entered the simulation at all. It is latent demand,
        counted by `metrics` and never discarded. At most one vehicle per entry per
        step is released, so a backlog leaves as a stream rather than as a block.

        At partial penetration the gate holds only controlled vehicles. An uncontrolled
        vehicle does not run the policy on the unsimulated link above the entry either,
        so it is handed to SUMO at its departure time without the density check.
        """
        now = float(_sumo.simulation.getTime())
        if self._incidents:
            self._place_incidents(now)
        while self._pulse_queue and self._pulse_queue[0]["depart"] <= now:
            vehicle = self._pulse_queue.pop(0)
            _sumo.vehicle.add(vehicle["id"], vehicle["route"], typeID=vehicle["type"],
                              depart="now", departLane="best", departSpeed="max")
        if not self._schedule:
            return
        for entry, queued in self._schedule.items():
            free, freed = self._free[entry], self._freed[entry]
            while freed < len(free) and free[freed]["depart"] <= now:
                vehicle = free[freed]
                _sumo.vehicle.add(vehicle["id"], vehicle["route"], typeID=vehicle["type"],
                                  depart="now", departLane="best", departSpeed="max")
                freed += 1
            self._freed[entry] = freed
            index = self._admitted[entry]
            if index >= len(queued) or queued[index]["depart"] > now:
                continue
            if not self._entry_open(entry):
                continue
            vehicle = queued[index]
            _sumo.vehicle.add(vehicle["id"], vehicle["route"], typeID=vehicle["type"],
                              depart="now", departLane="best", departSpeed="max")
            self._admitted[entry] = index + 1

    def held_at_entry(self) -> int:
        """Controlled vehicles whose departure time has passed but which the gate has
        not let in. Uncontrolled vehicles are never held, so they never count here."""
        if not self._schedule:
            return 0
        now = float(_sumo.simulation.getTime())
        return sum(
            max(0, bisect.bisect_right(self._departs[entry], now) - self._admitted[entry])
            for entry in self._schedule
        )

    def pending(self) -> int:
        """Vehicles handed to SUMO that it has not yet found room to insert.

        With `--max-depart-delay -1` such a vehicle waits rather than being dropped, so
        it is latent demand outside the simulated road, like a vehicle held at the gate.
        An incident vehicle waiting for its spot is not demand and is not counted.
        """
        pending = _sumo.simulation.getPendingVehicles()
        if not self._incident_ids:
            return len(pending)
        return sum(1 for vehicle in pending if vehicle not in self._incident_ids)

    # ---------------------------------------------------------------- incidents

    def schedule_incident(self, edge: str, lanes: list[int], pos_m: float, start_s: float,
                          duration_s: float) -> list[str]:
        """Block `lanes` of `edge` at `pos_m` on `[start_s, start_s + duration_s)` (plan
        section 4.5), with one stopped vehicle per lane.

        Call after `reset`, which clears every incident. Vehicle `k` is `incident_<k>`,
        type `human`, on a one-edge route; it is requested at the first step at or after
        `start_s` with `departLane`, `departPos` and `departSpeed=0` and given a stop at
        `pos_m` for `duration_s`, and removed at the first step at or after `start_s +
        duration_s`. SUMO inserts it only once the spot is clear, which was 0 to 15.5 s
        after the request on the plan's two blocks (three seeds each); `incident_report`
        records the instant it actually entered. Its ids take the next departure
        positions as uncontrolled vehicles. It is excluded from edge counts and speeds,
        the segment state, the feed snapshot, arrivals, pending and running, and stays
        visible to anything reading the lanes, as a stopped vehicle on the road.
        """
        if not lanes or duration_s <= 0.0:
            raise ValueError("an incident needs at least one lane and a positive duration")
        ids = []
        for lane in lanes:
            vehicle = f"incident_{len(self._incidents)}"
            if self._index is not None:
                self._index[vehicle] = len(self._index)
            self._incidents.append({
                "id": vehicle, "edge": edge, "lane": int(lane), "pos_m": float(pos_m),
                "start_s": float(start_s), "end_s": float(start_s) + float(duration_s),
                "duration_s": float(duration_s), "requested_s": None, "inserted_s": None,
                "removed_s": None})
            self._incident_ids.add(vehicle)
            self._incident_lane[vehicle] = int(lane)
            ids.append(vehicle)
        return ids

    def _place_incidents(self, now: float) -> None:
        for incident in self._incidents:
            vehicle = incident["id"]
            if incident["requested_s"] is None and now >= incident["start_s"]:
                route = f"incident_route_{vehicle}"
                _sumo.route.add(route, [incident["edge"]])
                _sumo.vehicle.add(vehicle, route, typeID="human", depart="now",
                                  departLane=str(incident["lane"]),
                                  departPos=str(incident["pos_m"]), departSpeed="0")
                _sumo.vehicle.setStop(vehicle, incident["edge"], pos=incident["pos_m"],
                                      laneIndex=incident["lane"],
                                      duration=incident["duration_s"])
                incident["requested_s"] = now
            elif (incident["requested_s"] is not None and incident["removed_s"] is None
                  and now >= incident["end_s"]):
                _sumo.vehicle.remove(vehicle)
                self._incidents_on_road.pop(vehicle, None)
                incident["removed_s"] = now

    def _note_inserted_incidents(self) -> None:
        departed = _sumo.simulation.getDepartedIDList()
        if not departed:
            return
        for incident in self._incidents:
            vehicle = incident["id"]
            if incident["inserted_s"] is None and vehicle in departed:
                incident["inserted_s"] = float(_sumo.vehicle.getDeparture(vehicle))
                self._incidents_on_road[vehicle] = incident["edge"]

    def incident_report(self) -> list[dict[str, Any]]:
        """Every scheduled incident with its requested, inserted and removed instants."""
        return [dict(incident) for incident in self._incidents]

    def _load_static(self) -> None:
        """Lane count, length and free-flow speed per edge, read once from the map."""
        self._static = {}
        for segment in list(self.segments) + [list(self._schedule)]:
            for edge in segment:
                lanes = int(_sumo.edge.getLaneNumber(edge))
                length = float(_sumo.lane.getLength(f"{edge}_0"))
                limit = float(_sumo.lane.getMaxSpeed(f"{edge}_0"))
                self._static[edge] = {
                    "lanes": float(lanes),
                    "length_m": length,
                    "free_flow_kmh": limit * 3.6,
                }

    # --------------------------------------------------------------------- step

    def step(self, actions: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply one speed per super-segment, hold it, and report the new state."""
        self._command(actions)
        self._advance(DECISION_INTERVAL_S)
        state = self.observe()
        reward = self.reward()
        done = _sumo.simulation.getTime() >= self.duration_s
        return state, reward, done

    def _advance(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.step_length_s))):
            self._step()
        self._update_rates()
        self._accumulate()

    def _step(self) -> None:
        """One simulation step: admit, step, and count arrivals, collisions and teleports."""
        self._admit()
        _sumo.simulationStep()
        if self._incidents:
            self._note_inserted_incidents()
            self.arrived_total += sum(1 for vehicle in _sumo.simulation.getArrivedIDList()
                                      if vehicle not in self._incident_ids)
        else:
            self.arrived_total += int(_sumo.simulation.getArrivedNumber())
        self.colliding_vehicles += int(_sumo.simulation.getCollidingVehiclesNumber())
        starting = _sumo.simulation.getStartingTeleportIDList()
        self.collision_teleports += len(starting)
        if starting or self._in_teleport:
            self._in_teleport.update(starting)
            self._in_teleport.difference_update(_sumo.simulation.getEndingTeleportIDList())
            self._in_teleport.difference_update(_sumo.simulation.getArrivedIDList())

    def in_teleport(self) -> int:
        """Vehicles whose teleport started and has not ended: in no other count."""
        return len(self._in_teleport)

    def advance_ticks(self, seconds: float, tick_s: float = 1.0,
                      on_tick: Callable[[float], None] | None = None) -> None:
        """`_advance`, calling `on_tick(t)` after every `tick_s` of simulation.

        The simulation steps are the same as `_advance`'s, one `_admit` before each,
        and the interval ends with the same rate update and window accumulation, so a
        callback that only reads leaves the episode exactly as `step()` would. The
        callback is where per-vehicle feed versions take snapshots, switch vehicles at
        their own instants and record traces.
        """
        steps_per_tick = int(round(tick_s / self.step_length_s))
        if steps_per_tick < 1 or not math.isclose(steps_per_tick * self.step_length_s,
                                                  tick_s, abs_tol=1e-9):
            raise ValueError(f"a tick of {tick_s} s is not a whole number of "
                             f"{self.step_length_s} s steps")
        ticks = int(round(seconds / tick_s))
        if not math.isclose(ticks * tick_s, seconds, abs_tol=1e-9):
            raise ValueError(f"{seconds} s is not a whole number of {tick_s} s ticks")
        for _ in range(ticks):
            for _ in range(steps_per_tick):
                self._step()
            if on_tick is not None:
                on_tick(float(_sumo.simulation.getTime()))
        self._update_rates()
        self._accumulate()

    def _accumulate(self) -> None:
        """Add this interval's vehicle-weighted speed and occupancy to the window.

        `speeds_kmh` averages segments and then averages those, which weights a segment
        holding three vehicles the same as one holding four hundred. That statistic is
        what SRC's reward is written against and stays as it is, but it is not a speed
        the network can be described by: measured on the throttled network it read
        21.0 km/h where the vehicle-weighted speed was 11.9.
        """
        edges = self._per_edge()
        present = sum(values["count"] for values in edges.values())
        self._vehicle_weight += present
        self._speed_weight += sum(values["count"] * values["speed_kmh"]
                                  for values in edges.values() if values["count"])
        self._samples += 1

    def _update_rates(self) -> None:
        """Vehicles that entered and left each super-segment over the last interval.

        Computed here rather than in `observe`, which must be callable twice without
        changing its own answer.
        """
        present = self._segment_vehicles()
        self._rates = [
            (float(len(now - was)), float(len(was - now)))
            for now, was in zip(present, self._previous, strict=True)
        ]
        self._previous = present

    def _command(self, actions: list[int] | np.ndarray) -> None:
        """Write the advisory to the controlled vehicles on each segment.

        Under `max_speed` it is the vehicle's maximum speed, so a vehicle whose leader is
        slower than the advisory follows its leader under the car-following model; that
        is the local safety gate. See `ACTUATIONS` for the legacy `set_speed`.
        """
        for index, segment in enumerate(self.segments):
            fraction = SPEED_ACTION_FRACTIONS[int(actions[index])]
            for edge in segment:
                target_mps = fraction * self._static[edge]["free_flow_kmh"] / 3.6
                for vehicle in _sumo.edge.getLastStepVehicleIDs(edge):
                    if self.controlled(vehicle):
                        self._apply(vehicle, target_mps)

    def _apply(self, vehicle: str, target_mps: float) -> None:
        """Apply one advisory to one controlled vehicle under this env's actuation."""
        if self.actuation == "max_speed":
            if vehicle not in self._capped:
                self._capped[vehicle] = float(_sumo.vehicle.getMaxSpeed(vehicle))
            _sumo.vehicle.setMaxSpeed(vehicle, target_mps)
        else:
            _sumo.vehicle.setSpeed(vehicle, target_mps)

    def controlled_on_segments(self) -> list[tuple[str, str, int]]:
        """`(vehicle, edge, segment index)` for every controlled vehicle on a segment edge.

        In the order `_command` visits them, so a caller that writes in this order
        writes in the same order `step()` does. Vehicles crossing a junction are on an
        internal edge and are not listed, exactly as `_command` does not reach them.
        """
        found = []
        for index, segment in enumerate(self.segments):
            for edge in segment:
                for vehicle in _sumo.edge.getLastStepVehicleIDs(edge):
                    if self.controlled(vehicle):
                        found.append((vehicle, edge, index))
        return found

    def time_s(self) -> float:
        return float(_sumo.simulation.getTime())

    def running(self) -> int:
        """Vehicles in the network, incident vehicles standing on the road excluded."""
        return int(_sumo.vehicle.getIDCount()) - len(self._incidents_on_road)

    def position(self, vehicle_id: str) -> int:
        """The vehicle's position in the departure order (pulses continue it)."""
        if self._index is None:
            raise KeyError("this env has no schedule, so no departure order")
        return self._index[vehicle_id]

    @property
    def schedule_size(self) -> int:
        """Positions taken in the departure order, pulses included."""
        return len(self._index) if self._index is not None else 0

    def pulse_route(self, site: int) -> tuple[str, list[str]]:
        """The route a pulse at `site` rides: `(source route id, edges to the exit)`."""
        from src.analysis.gains import load_routes, pulse_sites

        chosen = pulse_sites(self.segments, load_routes(self.route_file))[site]
        return chosen["route"], chosen["suffix"]

    def inject_pulse(self, site: int, n: int, start_s: float, duration_s: float,
                     seed: int) -> dict[str, Any]:
        """Add `n` vehicles at super-segment `site`, evenly over `[start_s, start_s +
        duration_s)`, for the gain certificate.

        They ride the suffix of the longest route through the site from the site's first
        edge to the exit (`pulse_route`), added as its own route, and depart at
        `start_s + h/2 + j h` with `h = duration_s / n`, the builder's spacing. Each is
        handed to SUMO at the first step at or after its own departure (plan Amendment
        1, A8): handing all of them over at once, with future departure times, changed
        the simulation from that instant on, 900 s before the first one left. The entry
        gate does not apply, since they are a perturbation inside the network rather
        than demand at its boundary. Vehicle `j` is `pulse_{site}_{j}` and takes the
        departure position after the schedule's last, so the floor rule continues over
        it. `seed` is recorded only: the pulse is deterministic.
        """
        if n <= 0 or duration_s <= 0.0:
            raise ValueError("a pulse needs at least one vehicle and a positive duration")
        if self._index is None:
            raise ValueError("a pulse needs the schedule's departure order")
        source, edges = self.pulse_route(site)
        route_id = f"pulse_route_{site}"
        _sumo.route.add(route_id, edges)
        headway = duration_s / n
        first_position = len(self._index)
        ids, departs, controlled = [], [], 0
        for j in range(n):
            vehicle = f"pulse_{site}_{j}"
            position = first_position + j
            self._index[vehicle] = position
            is_controlled = controlled_index(position, self.penetration)
            if is_controlled:
                self._controlled_ids.add(vehicle)
                controlled += 1
            depart = round(start_s + headway / 2.0 + j * headway, 2)
            self._pulse_queue.append({"id": vehicle, "route": route_id, "depart": depart,
                                      "type": "av" if is_controlled else "human"})
            ids.append(vehicle)
            departs.append(depart)
        self._pulse_queue.sort(key=lambda row: row["depart"])
        return {"site": site, "n": n, "start_s": start_s, "duration_s": duration_s,
                "seed": seed, "source_route": source, "route_edges": edges, "ids": ids,
                "departs": departs, "controlled": controlled,
                "first_position": first_position}

    def command_vehicles(self, mapping: dict[str, float],
                         edge_of: dict[str, str] | None = None) -> int:
        """Write `fraction * free-flow speed of the vehicle's edge` to the listed vehicles.

        Vehicles not listed keep their last command, exactly as a vehicle that changes
        segment between two `step()` decisions does. `edge_of` gives each vehicle's edge
        when the caller already has it from an edge scan; otherwise it is read per
        vehicle. Returns the number of writes.
        """
        for vehicle, fraction in mapping.items():
            edge = edge_of[vehicle] if edge_of is not None else _sumo.vehicle.getRoadID(vehicle)
            self._apply(vehicle, fraction * self._static[edge]["free_flow_kmh"] / 3.6)
        return len(mapping)

    def release(self) -> None:
        """Hand every controlled vehicle on a segment back to the car-following model.

        Under `max_speed` a capped vehicle gets back the maximum speed it had before its
        first cap, its type's; a vehicle never capped is not touched, so a run that never
        advises, the no-control arm, makes no call at all.
        """
        for segment in self.segments:
            for edge in segment:
                for vehicle in _sumo.edge.getLastStepVehicleIDs(edge):
                    if not self.controlled(vehicle):
                        continue
                    if self.actuation == "max_speed":
                        if vehicle in self._capped:
                            _sumo.vehicle.setMaxSpeed(vehicle, self._capped.pop(vehicle))
                    else:
                        _sumo.vehicle.setSpeed(vehicle, -1.0)

    # ------------------------------------------------------------------- state

    def _edge_reading(self, edge: str, blocked: dict[str, list[str]]) -> tuple[int, float]:
        """`(count, mean speed in m/s)` of the edge's traffic, incident vehicles removed.

        An edge with no incident is read exactly as SUMO reports it. On an edge holding an
        incident vehicle, SUMO's own estimator is rebuilt from its lanes without it:
        `sum_l max(n_l, 1) v_l / sum_l max(n_l, 1)`, where `n_l` counts every vehicle on
        lane `l` and `v_l` is the lane's mean speed, an empty lane reading its speed limit
        (measured equal to SUMO's edge mean to 5e-15 m/s). A lane without an incident keeps
        SUMO's own count and mean. A lane with one takes the count and the plain mean
        speed of its other vehicles, which is what SUMO's lane mean would be without it:
        on a multi-lane edge SUMO's lane mean already leaves a stopped vehicle out, so
        removing the incident's zero from the edge mean, as an earlier version did,
        removed it a second time (validation R1-1); on a one-lane edge it keeps it.
        """
        count = int(_sumo.edge.getLastStepVehicleNumber(edge))
        speed_mps = float(_sumo.edge.getLastStepMeanSpeed(edge))
        incidents = blocked.get(edge)
        if not incidents:
            return count, speed_mps
        blocked_lanes = {self._incident_lane[v] for v in incidents}
        weight = total = 0.0
        vehicles = 0
        for lane in range(int(self._static[edge]["lanes"])):
            lane_id = f"{edge}_{lane}"
            if lane in blocked_lanes:
                others = [v for v in _sumo.lane.getLastStepVehicleIDs(lane_id)
                          if v not in self._incident_ids]
                n = len(others)
                v_lane = (sum(float(_sumo.vehicle.getSpeed(v)) for v in others) / n if n
                          else float(_sumo.lane.getMaxSpeed(lane_id)))
            else:
                n = int(_sumo.lane.getLastStepVehicleNumber(lane_id))
                v_lane = float(_sumo.lane.getLastStepMeanSpeed(lane_id))
            w = max(n, 1)
            weight += w
            total += w * v_lane
            vehicles += n
        if vehicles == 0:
            return 0, 0.0
        return vehicles, total / weight

    def _blocked_edges(self) -> dict[str, list[str]]:
        blocked: dict[str, list[str]] = {}
        for vehicle, edge in self._incidents_on_road.items():
            blocked.setdefault(edge, []).append(vehicle)
        return blocked

    def _per_edge(self) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        blocked = self._blocked_edges()
        for segment in self.segments:
            for edge in segment:
                static = self._static[edge]
                count, speed_mps = self._edge_reading(edge, blocked)
                lane_metres = static["length_m"] * static["lanes"]
                values[edge] = {
                    "count": float(count),
                    # AN EMPTY EDGE CONTRIBUTES NOTHING, not a zero. Two thirds of the
                    # controlled edges are opposite-direction or unserved roads that
                    # this demand never traverses, and averaging their zeros into a
                    # super-segment put the mean density at 0.23 against a threshold of
                    # 0.3 that then never fired, leaving SRC's reward as 0.2 * speed
                    # with its congestion term dead. A segment's density is the density
                    # of the road that has traffic on it.
                    "speed_kmh": speed_mps * 3.6 if count else math.nan,
                    "density": ((count * VEHICLE_LENGTH_M / lane_metres)
                                if (lane_metres and count) else math.nan),
                    "lanes": static["lanes"],
                    "length_m": static["length_m"],
                    "free_flow_kmh": static["free_flow_kmh"],
                }
        return values

    def _segment_vehicles(self) -> list[set[str]]:
        present = [
            {v for edge in segment for v in _sumo.edge.getLastStepVehicleIDs(edge)}
            for segment in self.segments
        ]
        if self._incidents_on_road:
            present = [vehicles - self._incident_ids for vehicles in present]
        return present

    def _mean_gap(self, vehicles: set[str]) -> float:
        """Mean gap to the leader, SRC's `FollowDistGr` averaged over the segment.

        A vehicle with no leader contributes nothing rather than a zero, for the same
        reason an empty edge contributes no density: an open road is not a zero gap.
        """
        gaps = []
        for vehicle in vehicles:
            leader = _sumo.vehicle.getLeader(vehicle, 200.0)
            if leader is not None and leader[0]:
                gaps.append(float(leader[1]))
        return float(sum(gaps) / len(gaps)) if gaps else math.nan

    def observe(self) -> np.ndarray:
        """The state the controller sees, one row per super-segment.

        Aggregated across a segment's edges with `nanmean`, unweighted by length,
        which is how SRC pools its links. An empty edge contributes no speed rather
        than a zero, so a segment's speed is the speed of its occupied road.
        """
        edges = self._per_edge()
        present = self._segment_vehicles()
        rows = []
        for index, segment in enumerate(self.segments):
            members = [edges[e] for e in segment]
            here = present[index]
            speed = _nanmean([m["speed_kmh"] for m in members])
            free_flow = _nanmean([m["free_flow_kmh"] for m in members])
            row = {
                "speed": speed,
                "free_flow": free_flow,
                "jam_factor": jam_factor(speed, free_flow),
                "lanes": _nanmean([m["lanes"] for m in members]),
                "length_km": sum(m["length_m"] for m in members) / 1000.0,
                "density": _nanmean([m["density"] for m in members]),
                # SRC counts a segment's arrivals and departures as the set
                # difference of the vehicle ids present now and one decision ago.
                "gap": self._mean_gap(here),
                "input_rate": self._rates[index][0],
                "exit_rate": self._rates[index][1],
            }
            rows.append([row[name] for name in self.features])
        return np.nan_to_num(np.array(rows, dtype=np.float32))

    def densities(self) -> np.ndarray:
        """Per-super-segment density, for the reward. Ground truth, not observable."""
        edges = self._per_edge()
        return np.array(
            [_nanmean([edges[e]["density"] for e in segment]) for segment in self.segments],
            dtype=np.float32,
        )

    def segment_state(self) -> dict[str, list[float]]:
        """Per super-segment: vehicle-weighted space-mean speed, density and count.

        From one `_per_edge` read. The speed is `sum(count * speed) / sum(count)` over the
        segment's occupied edges, nan for an empty segment; the density is occupancy,
        `count * VEHICLE_LENGTH_M` over every lane-metre of the segment, zero when
        empty. This is the state the drain metric and the gain estimate read; it is not
        the reward's `densities()`, which averages per-edge densities over occupied
        edges only.
        """
        edges = self._per_edge()
        speed, density, count = [], [], []
        for segment in self.segments:
            members = [edges[e] for e in segment]
            vehicles = sum(m["count"] for m in members)
            lane_metres = sum(m["length_m"] * m["lanes"] for m in members)
            speed.append(sum(m["count"] * m["speed_kmh"] for m in members if m["count"])
                         / vehicles if vehicles else math.nan)
            density.append(vehicles * VEHICLE_LENGTH_M / lane_metres if lane_metres else 0.0)
            count.append(int(vehicles))
        return {"speed_kmh": speed, "density": density, "count": count}

    def speeds_kmh(self) -> np.ndarray:
        edges = self._per_edge()
        return np.array(
            [_nanmean([edges[e]["speed_kmh"] for e in segment]) for segment in self.segments],
            dtype=np.float32,
        )

    def reward(self) -> np.ndarray:
        """SRC's reward, per super-segment, from ground truth.

        `-100 * 1[density > 0.3] + 0.2 * speed`, unchanged from its train.py. Density
        is privileged: it is used to train and never to act.
        """
        density = np.nan_to_num(self.densities())
        speed = np.nan_to_num(self.speeds_kmh())
        over = (density > DENSITY_CRITICAL).astype(np.float32)
        return -over * CONGESTION_PENALTY + SPEED_WEIGHT * speed

    # ----------------------------------------------------------------- reporting

    def mark_window(self) -> None:
        """Open the measurement window here, discarding everything before it.

        Cumulative arrivals over a whole episode are dominated by the ramp-up, during
        which the network is filling and throughput says more about how far the fill
        has got than about the network. Flow over a window after the fill is a rate
        the controller can actually be judged on.
        """
        self._window_start_s = float(_sumo.simulation.getTime())
        self._window_start_arrived = self.arrived_total
        self._speed_weight = 0.0
        self._vehicle_weight = 0.0
        self._samples = 0

    def window_flow_veh_per_h(self) -> float:
        """Vehicles discharged per hour since `mark_window`."""
        elapsed = float(_sumo.simulation.getTime()) - self._window_start_s
        if elapsed <= 0.0:
            return 0.0
        return (self.arrived_total - self._window_start_arrived) * 3600.0 / elapsed

    def space_mean_speed_kmh(self) -> float:
        """Vehicle-weighted mean speed over the window: the v in `q = k v`."""
        if not self._vehicle_weight:
            return 0.0
        return self._speed_weight / self._vehicle_weight

    def mean_vehicles(self) -> float:
        """Vehicles in the network, averaged over the window."""
        return self._vehicle_weight / self._samples if self._samples else 0.0

    def metrics(self) -> dict[str, Any]:
        running = self.running()
        waiting = self.pending()
        speed = float(np.nan_to_num(_nanmean(list(self.speeds_kmh()))))
        density = np.nan_to_num(self.densities())
        return {
            "time_s": float(_sumo.simulation.getTime()),
            "arrived": self.arrived_total,
            "flow_veh_per_h": self.window_flow_veh_per_h(),
            "running": running,
            "waiting_to_enter": waiting,
            "held_at_entry": self.held_at_entry(),
            "admitted": (sum(self._admitted.values()) + sum(self._freed.values())
                         if self._schedule else -1),
            # An unweighted mean of segment means, read at this instant. Kept because
            # it is the quantity SRC's reward is built from; it is not a description of
            # how fast the traffic is moving, for which use `space_mean_speed_kmh`.
            "mean_speed_kmh": speed,
            "space_mean_speed_kmh": self.space_mean_speed_kmh(),
            "mean_vehicles": self.mean_vehicles(),
            "segments_over_critical": int((density > DENSITY_CRITICAL).sum()),
            "max_density": float(density.max()),
            "colliding_vehicles": self.colliding_vehicles,
            "collision_teleports": self.collision_teleports,
            "in_teleport": self.in_teleport(),
            "actuation": self.actuation,
        }


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan
