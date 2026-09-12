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
written to every controlled vehicle on that segment as a DESIRED speed. SUMO bounds it
by the car-following safe speed exactly as Vissim bounds `DesSpeed`, so the advisory is
gated by local conditions rather than commanded through them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo.env import _sumo

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
    ) -> None:
        self.seed = seed
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
        # Per entry link, the vehicles scheduled to start there and how many of them
        # have been let in. Empty when SUMO is doing the inserting.
        self._schedule: dict[str, list[dict[str, Any]]] = {}
        self._admitted: dict[str, int] = {}

    # ----------------------------------------------------------------- lifecycle

    def reset(self) -> np.ndarray:
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
        self._load_schedule()
        self._load_static()
        self._previous = [set() for _ in self.segments]
        self._rates = [(0.0, 0.0) for _ in self.segments]
        self._advance(self.warmup_s)
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

    def _load_schedule(self) -> None:
        """Read the vehicles the route file omits, grouped by the entry they start at."""
        self._schedule, self._admitted = {}, {}
        if not self.gate_entries:
            return
        rows = json.loads(self.schedule_file.read_text())
        for row in rows:
            self._schedule.setdefault(row["entry"], []).append(row)
        for entry, queued in self._schedule.items():
            queued.sort(key=lambda row: row["depart"])
            self._admitted[entry] = 0

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
        """
        if not self._schedule:
            return
        now = float(_sumo.simulation.getTime())
        for entry, queued in self._schedule.items():
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
        """Vehicles whose departure time has passed but which the gate has not let in."""
        if not self._schedule:
            return 0
        now = float(_sumo.simulation.getTime())
        return sum(
            sum(1 for row in queued[self._admitted[entry]:] if row["depart"] <= now)
            for entry, queued in self._schedule.items()
        )

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
            self._admit()
            _sumo.simulationStep()
            self.arrived_total += int(_sumo.simulation.getArrivedNumber())
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
        """Write the advisory to controlled vehicles as a desired speed.

        `setSpeed` is bounded by SUMO's car-following safe speed, so a vehicle whose
        leader is slower than the advisory follows its leader. That is the local
        safety gate, and it is the same semantics as Vissim's `DesSpeed`.
        """
        for index, segment in enumerate(self.segments):
            fraction = SPEED_ACTION_FRACTIONS[int(actions[index])]
            for edge in segment:
                target_mps = fraction * self._static[edge]["free_flow_kmh"] / 3.6
                for vehicle in _sumo.edge.getLastStepVehicleIDs(edge):
                    if _sumo.vehicle.getTypeID(vehicle) == "av":
                        _sumo.vehicle.setSpeed(vehicle, target_mps)

    def release(self) -> None:
        """Hand every controlled vehicle back to the car-following model."""
        for segment in self.segments:
            for edge in segment:
                for vehicle in _sumo.edge.getLastStepVehicleIDs(edge):
                    if _sumo.vehicle.getTypeID(vehicle) == "av":
                        _sumo.vehicle.setSpeed(vehicle, -1.0)

    # ------------------------------------------------------------------- state

    def _per_edge(self) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for segment in self.segments:
            for edge in segment:
                static = self._static[edge]
                count = int(_sumo.edge.getLastStepVehicleNumber(edge))
                speed_mps = float(_sumo.edge.getLastStepMeanSpeed(edge))
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
        return [
            {v for edge in segment for v in _sumo.edge.getLastStepVehicleIDs(edge)}
            for segment in self.segments
        ]

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
        running = int(_sumo.vehicle.getIDCount())
        waiting = len(_sumo.simulation.getPendingVehicles())
        speed = float(np.nan_to_num(_nanmean(list(self.speeds_kmh()))))
        density = np.nan_to_num(self.densities())
        return {
            "time_s": float(_sumo.simulation.getTime()),
            "arrived": self.arrived_total,
            "flow_veh_per_h": self.window_flow_veh_per_h(),
            "running": running,
            "waiting_to_enter": waiting,
            "held_at_entry": self.held_at_entry(),
            "admitted": sum(self._admitted.values()) if self._schedule else -1,
            # An unweighted mean of segment means, read at this instant. Kept because
            # it is the quantity SRC's reward is built from; it is not a description of
            # how fast the traffic is moving, for which use `space_mean_speed_kmh`.
            "mean_speed_kmh": speed,
            "space_mean_speed_kmh": self.space_mean_speed_kmh(),
            "mean_vehicles": self.mean_vehicles(),
            "segments_over_critical": int((density > DENSITY_CRITICAL).sum()),
            "max_density": float(density.max()),
        }


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan
