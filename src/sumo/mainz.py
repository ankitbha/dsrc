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
#: SRC's `FEEDBACK_STEP`.
DECISION_INTERVAL_S: float = 60.0
#: SRC's `density_critical`, against a density of vehicles x 4.5 m over lane-metres.
DENSITY_CRITICAL: float = 0.3
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
        net_file: Path | None = None,
        route_file: Path | None = None,
        segment_file: Path | None = None,
    ) -> None:
        self.seed = seed
        self.duration_s = duration_s
        self.warmup_s = warmup_s
        self.step_length_s = step_length_s
        self.features = features
        self.net_file = net_file or DATA / "mainz.net.xml"
        self.route_file = route_file or DATA / "mainz.rou.xml"
        self.segment_file = segment_file or DATA / "mainz_segments.json"
        self.segments: list[list[str]] = json.loads(self.segment_file.read_text())
        self._running = False
        self._static: dict[str, dict[str, float]] = {}
        self._previous: list[set[str]] = [set() for _ in self.segments]
        self._rates: list[tuple[float, float]] = [(0.0, 0.0) for _ in self.segments]
        self.arrived_total = 0
        self._window_start_s = 0.0
        self._window_start_arrived = 0

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

    def _load_static(self) -> None:
        """Lane count, length and free-flow speed per edge, read once from the map."""
        self._static = {}
        for segment in self.segments:
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
            _sumo.simulationStep()
            self.arrived_total += int(_sumo.simulation.getArrivedNumber())
        self._update_rates()

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
            target_mps = SPEED_ACTIONS_KMH[int(actions[index])] / 3.6
            for edge in segment:
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

    def window_flow_veh_per_h(self) -> float:
        """Vehicles discharged per hour since `mark_window`."""
        elapsed = float(_sumo.simulation.getTime()) - self._window_start_s
        if elapsed <= 0.0:
            return 0.0
        return (self.arrived_total - self._window_start_arrived) * 3600.0 / elapsed

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
            "mean_speed_kmh": speed,
            "segments_over_critical": int((density > DENSITY_CRITICAL).sum()),
            "max_density": float(density.max()),
        }


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else math.nan
