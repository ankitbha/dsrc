"""A SUMO-backed environment producing the snapshots the rest of this project reads.

`VehicleSnapshot` is the seam. Above it sit the sensing model, the safety layer, the
etiquette filters, the observation encoders, the policy and the trainers, none of
which know what produces the snapshots. So switching simulators means producing them
from TraCI instead of from `highway_env`, and nothing above the seam changes.

**Why SUMO.** Its car-following model cannot produce a collision, because the safe
speed is the model rather than a cap applied over it. Measured on the 3-into-1 merge
shape that broke `highway_env`, at 6000 veh/h into a two-lane exit -- three times
past capacity -- for 1800 steps with 134 vehicles present: zero collisions. Three
attempts at constraining `highway_env` from outside took human-only collisions from
98 to 6 at short durations and the residual from 23 to 14 at longer ones, each
trading one collision mode for another.

**`lane_index` keeps its old meaning.** A SUMO edge runs between two junctions, so
`(from_junction, to_junction, lane_ordinal)` is the same shape `highway_env` used
and the same shape the sensing model compares. Nothing downstream has to learn a new
identifier.

**The collision count is read every step and exposed.** A premise that SUMO cannot
collide, held without measurement, is the kind of assumption this project has been
caught by three times. `collision_checks` counts the reads so a test can tell an
un-incremented counter from a genuine zero.
"""
from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.wrappers import decode_headway_bin, decode_speed_bin
from src.metrics.global_metrics import metric_thresholds_from_config
from src.metrics.segment_metrics import compute_segment_metrics
from src.safety import SafetyConstraints, SafetyState
from src.sensing import LocalObservationBuilder, SensingConfig, VehicleSnapshot
from src.sumo.network import SumoNetwork
from src.sumo.road_view import SumoTopologyView

#: libsumo runs in-process and is roughly an order of magnitude faster than TraCI's
#: socket, which would otherwise dominate a 3600-step episode. It is process-global,
#: so only one environment may be live at a time; `close` is what releases it.
try:  # pragma: no cover - exercised by whichever binding is installed
    import libsumo as _sumo

    _BINDING = "libsumo"
except ImportError:  # pragma: no cover
    import traci as _sumo

    _BINDING = "traci"


#: The environment currently holding the process-global SUMO connection, or None.
#: libsumo runs in-process and there is exactly one simulation per process, so two
#: live environments would drive the same one. Before this existed, a second
#: `reset` silently rewound the clock to 0.0 while the first environment stayed
#: `_running`, and its next `step` advanced the SECOND simulation and returned the
#: second's vehicles, with its own step count, arrivals and collision total
#: continuing across both. Nothing raised.
_LIVE: "SumoTopologyEnv | None" = None


class SumoTopologyEnv:
    """Runs one SUMO simulation and reports it as `VehicleSnapshot`s."""

    #: vType ids. Two types rather than one, so a vehicle's role is a property of
    #: the demand that created it rather than something inferred from its id.
    AV_TYPE = "av"
    HUMAN_TYPE = "human"
    #: The distribution the flows draw from, so penetration changes who is
    #: controllable without changing the arrival process.
    MIX_TYPE = "mix"
    #: How SUMO handles a collision. `warn` reports and leaves the vehicles in
    #: place; anything that removes them would make the collision count read zero
    #: whatever happened.
    collision_action = "warn"

    def __init__(self, topology_id: str, config: Mapping[str, Any]) -> None:
        self.topology_id = topology_id
        self.config = dict(config)
        # A temporary directory by default, NOT the working directory. Defaulting to
        # "." wrote generated network and demand files into whatever directory the
        # caller happened to be in, and a `sumo_inverted_tree/` appeared in the
        # repository root. Generated artefacts do not belong in the repo, and a
        # caller that wants to keep them passes `work_dir`.
        configured = self.config.get("work_dir")
        base = Path(configured) if configured else Path(tempfile.gettempdir()) / "dsrc_sumo"
        self.work_dir = base / f"sumo_{topology_id}"
        self.network: SumoNetwork | None = None
        self.step_count = 0
        self.collision_count = 0
        #: How many times the collision count has been read. Exposed so a test can
        #: distinguish "no collisions happened" from "nobody looked".
        self.collision_checks = 0
        self._running = False
        self.view: SumoTopologyView | None = None
        self.agent_ids: list[str] = []
        self.arrived_total = 0
        self._sensing = LocalObservationBuilder(
            SensingConfig.from_config(self.config.get("sensing", {}) or {})
        )
        self._safety_states: dict[str, SafetyState] = {}
        self._target_headways: dict[str, float] = {}
        self._rng = np.random.RandomState(0)
        self._last_metrics: dict[str, Any] = {}
        #: Completion times, for the 60 s rolling throughput window the reward reads.
        self._arrivals: list[float] = []
        #: Which entry edge each live vehicle came from, so an arrival can be
        #: attributed to a branch. Needed because a vehicle is gone by the time it
        #: arrives, so its route cannot be read then.
        self._origin_of: dict[str, str] = {}
        #: Completions per entry branch, which is what fairness is measured over.
        self._branch_completed: dict[str, int] = {}
        self._branch_spawned: dict[str, int] = {}
        #: The segment each vehicle occupied last step, for per-segment in and out
        #: flow.
        self._segment_of: dict[str, str] = {}
        #: New collisions this step, exposed so a test can tell a measured zero from
        #: a hardcoded one.
        self.new_collisions_last_step = 0

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        global _LIVE
        if _LIVE is not None and _LIVE is not self:
            raise RuntimeError(
                "another SumoTopologyEnv already holds the SUMO connection; "
                "libsumo is process-global, so close it before resetting this one"
            )
        self.close()
        topology_cfg = self.config.get("topology", {})
        self.network = SumoNetwork.build(self.topology_id, topology_cfg, self.work_dir)
        self.view = SumoTopologyView(self.network)
        route_file = self._write_routes()
        self.agent_ids = []
        self.arrived_total = 0
        self._safety_states = {}
        self._target_headways = {}
        self._arrivals = []
        self._origin_of = {}
        # Every entry branch starts at zero, so a branch that completes nothing is
        # in the fairness denominator. Seeding these on first completion instead
        # measured Jain's index over the branches that had completed a vehicle: one
        # branch at 1 and five at 0 read 1.0000 against a true 0.1667, and 95 of 120
        # steps of the default evaluation config reported exactly 1.0. A controller
        # starving five branches to feed one scored the maximum.
        self._branch_completed = {edge: 0 for edge in self.network.entry_edges()}
        self._branch_spawned = {edge: 0 for edge in self.network.entry_edges()}
        self._segment_of = {}
        self.new_collisions_last_step = 0
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        self.step_count = 0
        self.collision_count = 0
        self.collision_checks = 0
        args = [
            "-n", str(self.network.net_file),
            "-r", str(route_file),
            "--step-length", str(float(self.config.get("dt", 1.0))),
            "--no-step-log", "true",
            "--no-warnings", "true",
            # Report a collision rather than removing the vehicles, so the count is
            # readable. `remove` would delete them and the count would read 0,
            # which would make every "zero collisions" claim in this migration
            # vacuous -- so the value is an attribute a test can pin.
            "--collision.action", self.collision_action,
        ]
        if seed is not None:
            args += ["--seed", str(int(seed))]
        _sumo.start([self._binary()] + args)
        self._running = True
        _LIVE = self
        self._warm_up()
        return self.get_local_observations(), {"binding": _BINDING}

    def _warm_up(self) -> None:
        """Fill the network before the episode is observed.

        A run starts empty and takes time to reach a steady state: a vehicle must
        traverse 500 + 600 + 900 m before it can arrive, so at 20 m/s nothing
        completes for the first hundred seconds and the early network is not the
        traffic under study. Worse for training, a rollout beginning at t = 0 can
        see no AV at all and collect zero transitions, which is how this was found.

        Warm-up steps advance the simulation but are not counted, observed or
        rewarded, so `duration_steps` still means what it says.
        """
        warmup = int(self.config.get("warmup_steps", 0))
        dt = float(self.config.get("dt", 1.0))
        for index in range(max(0, warmup)):
            _sumo.simulationStep()
            # Collisions during warm-up would still be collisions.
            self.collision_count += int(_sumo.simulation.getCollidingVehiclesNumber())
            self.collision_checks += 1
            # Warm-up arrivals count toward the rolling throughput window, at a
            # NEGATIVE time so the window sees them as already elapsed. Without
            # this the window refilled from empty after a warm-up that had already
            # reached steady state, and throughput_recent read 5.4 over the first 60
            # steps against 11.0 afterwards -- on the term the config gives the
            # largest positive weight.
            # `arrived_total` counts the EPISODE's completions and is deliberately
            # not advanced here: warm-up arrivals are not the traffic under study,
            # and adding them made `arrived_total` read 43 before the episode began
            # and 153 against the same run's 110.
            arrived = int(_sumo.simulation.getArrivedNumber())
            self._arrivals.extend([(index - warmup + 1) * dt] * arrived)
            self._track_branches()
        if warmup > 0:
            # Branch counts are episode-scoped for the same reason `arrived_total`
            # is: with a 300-step warm-up they otherwise carried 43 completions that
            # the episode's controller had no part in, and fairness over 43 evenly
            # spread warm-up completions is close to 1.0 whatever the episode does.
            # `_origin_of` is deliberately NOT cleared: a vehicle that departed
            # during warm-up and arrives during the episode is only attributable to
            # a branch through the origin recorded at its departure.
            self._branch_completed = {edge: 0 for edge in self.network.entry_edges()}
            self._branch_spawned = {edge: 0 for edge in self.network.entry_edges()}
            snapshots = self.vehicle_snapshots()
            self.agent_ids = [s.vehicle_id for s in snapshots if s.role == "av"]

    def close(self) -> None:
        global _LIVE
        if not self._running:
            return
        try:
            _sumo.close()
        except Exception:  # noqa: BLE001 - closing an already-dead connection
            pass
        self._running = False
        if _LIVE is self:
            _LIVE = None

    def __enter__(self) -> "SumoTopologyEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _binary() -> str:
        from sumolib import checkBinary

        return checkBinary("sumo")

    # -------------------------------------------------------------------- step

    def step(self, actions: Mapping[str, Any]) -> tuple[dict, dict, bool, bool, dict]:
        if not self._running:
            raise RuntimeError("step called before reset")
        self._apply_actions(actions)
        _sumo.simulationStep()
        self.step_count += 1

        before = self.collision_count
        self.collision_count += int(_sumo.simulation.getCollidingVehiclesNumber())
        self.collision_checks += 1
        self.new_collisions_last_step = self.collision_count - before
        arrived = int(_sumo.simulation.getArrivedNumber())
        self.arrived_total += arrived
        now = self.step_count * float(self.config.get("dt", 1.0))
        self._arrivals.extend([now] * arrived)
        self._track_branches()

        snapshots = self.vehicle_snapshots()
        self.agent_ids = [s.vehicle_id for s in snapshots if s.role == "av"]
        for agent_id in self.agent_ids:
            self._safety_states.setdefault(agent_id, SafetyState())
            self._target_headways.setdefault(agent_id, 1.6)

        duration = int(self.config.get("duration_steps", 120))
        truncated = self.step_count >= duration
        # `terminated` stays False by construction: SUMO does not crash vehicles, so
        # there is no early-termination condition. A True here would mean the
        # premise of this migration is wrong, which is why a test asserts it.
        terminated = False
        observations = self.get_local_observations(snapshots)
        self._last_metrics = self._build_metrics(snapshots, now)
        # `penalties` is empty because the safety layer does not run on this
        # simulator: SUMO's car-following is the safety guarantee, and
        # `apply_safety_layer` was never invoked here. `safety_penalty_for_agent`
        # therefore returns 0 for every agent and the per-agent reward equals the
        # team reward. Recorded rather than hidden -- an empty dict that looks like
        # "no penalties were incurred" is the shape of a metric that cannot charge
        # you, and this one means "nothing was measured".
        info = {"collisions": self.collision_count, "step": self.step_count,
                "metrics": self._last_metrics,
                "safety": {"penalties": {}, "layer_ran": False}}
        return observations, {}, terminated, truncated, info

    # ---------------------------------------------------------------- actuation

    def _apply_actions(self, actions: Mapping[str, Any]) -> None:
        """Turn each AV's action into a commanded speed.

        `setSpeed` rather than `setAcceleration`: SUMO applies its own safe-speed
        bound to whatever is asked either way, so speed is the more direct
        expression of what the controller decided and the bound stays SUMO's rather
        than being reimplemented here. Its safety guarantee is the reason for the
        migration, so nothing here may bypass it.
        """
        for agent_id, action in (actions or {}).items():
            if agent_id not in self.agent_ids:
                continue
            bin_name = (action or {}).get("desired_speed_bin")
            if bin_name is None:
                continue
            allowed = float(_sumo.vehicle.getAllowedSpeed(agent_id))
            try:
                # Returns a contextual speed in m/s, not a factor.
                target = float(decode_speed_bin(str(bin_name), free_flow_speed_mps=allowed))
            except Exception:  # noqa: BLE001 - an unknown bin is a caller error
                continue
            self.command_speed(agent_id, target)
            headway = (action or {}).get("desired_headway_bin")
            if headway is not None:
                try:
                    self._target_headways[agent_id] = float(decode_headway_bin(str(headway)))
                except Exception:  # noqa: BLE001
                    pass

    def command_speed(self, agent_id: str, speed_mps: float) -> float:
        """Ask SUMO to drive this AV at `speed_mps`, and report what was asked.

        SUMO applies its own safe-speed bound afterwards, so the vehicle may travel
        slower than this; it will never travel faster than is safe. That bound is
        the reason for the migration and nothing here bypasses it.
        """
        allowed = float(_sumo.vehicle.getAllowedSpeed(agent_id))
        asked = max(0.0, min(allowed, float(speed_mps)))
        _sumo.vehicle.setSpeed(agent_id, asked)
        return asked

    def av_speed(self, agent_id: str) -> float:
        """One AV's current speed, for tests and diagnostics."""
        return float(_sumo.vehicle.getSpeed(agent_id))

    # ----------------------------------------------------------------- metrics

    def _build_metrics(self, snapshots: list[VehicleSnapshot], now: float) -> dict[str, Any]:
        """The metric shape the reward and the health check already read.

        Fed from SUMO rather than recomputed differently, so `build_team_reward` and
        `simulator_health` need no change. `collision_count` comes from the counter
        read every step, not from a per-vehicle flag, because SUMO removes nothing
        and a flag would always read clean.
        """
        thresholds = metric_thresholds_from_config(self.config)
        window = float(self.config.get("throughput_window_s", 60.0))
        self._arrivals = [t for t in self._arrivals if now - t <= window]
        speeds = [s.speed_mps for s in snapshots]
        segment_metrics = self.get_segment_metrics(snapshots)
        jam = [m["jam_fraction"] for m in segment_metrics.values()]
        queue = sum(int(m["queue_length"]) for m in segment_metrics.values())
        return {
            "mean_speed": float(sum(speeds) / len(speeds)) if speeds else 0.0,
            "speed_std": float(np.std(speeds)) if speeds else 0.0,
            "throughput_recent": len(self._arrivals),
            "jam_fraction": float(sum(jam) / len(jam)) if jam else 0.0,
            "queue_length_total": queue,
            "collision_count": self.collision_count,
            # The per-step change, not a constant. build_team_reward PREFERS
            # new_collision_count over collision_count whenever the key is present,
            # so a hardcoded 0 made the -5.0 collision weight read nothing.
            "new_collision_count": self.new_collisions_last_step,
            "hard_braking_count": sum(
                1 for s in snapshots
                if s.acceleration_mps2 <= thresholds.hard_braking_mps2
            ),
            # Averaged over segments from the shared implementation. Hardcoding this
            # to 0.0 removed the -2.0 term that penalises AVs holding every lane of a
            # segment below free flow, which -- with crash_penalty inert on SUMO --
            # left every anti-degenerate term in the reward absent at once.
            "rolling_roadblock_score": (
                sum(float(m["rolling_roadblock_score"]) for m in segment_metrics.values())
                / max(len(segment_metrics), 1)
            ),
            "all_lane_av_low_speed_occupancy": (
                sum(float(m["all_lane_av_low_speed_occupancy"]) for m in segment_metrics.values())
                / max(len(segment_metrics), 1)
            ),
            # Over completions per entry branch, which is what highway_env measures
            # and what inverted_tree exists to study, not over instantaneous speeds.
            "fairness_jain": self._branch_fairness(),
            "active_vehicle_count": len(snapshots),
            "active_av_count": len(self.agent_ids),
        }

    def _track_branches(self) -> None:
        """Attribute each departure and arrival to the entry branch it used.

        A vehicle is gone by the time it arrives, so its route cannot be read then;
        the origin is recorded on departure and looked up on arrival. Fairness is
        Jain's index over completions per branch, which is the branch-fairness
        objective `inverted_tree` exists to study -- not, as an earlier version
        computed, Jain's index over instantaneous speeds.
        """
        for vehicle_id in _sumo.simulation.getDepartedIDList():
            try:
                route = _sumo.vehicle.getRoute(vehicle_id)
            except Exception:  # noqa: BLE001 - vanished between the two calls
                continue
            if route:
                self._origin_of[vehicle_id] = route[0]
                self._branch_spawned[route[0]] = self._branch_spawned.get(route[0], 0) + 1
        for vehicle_id in _sumo.simulation.getArrivedIDList():
            origin = self._origin_of.pop(vehicle_id, None)
            if origin is not None:
                self._branch_completed[origin] = self._branch_completed.get(origin, 0) + 1

    def _vehicle_records(self, snapshots: list[VehicleSnapshot]) -> list[dict[str, Any]]:
        """Snapshots in the record shape `compute_segment_metrics` reads."""
        return [{
            "vehicle_id": s.vehicle_id,
            "role": s.role,
            "branch_id": self._origin_of.get(s.vehicle_id),
            "segment_id": s.segment_id,
            "lane_id": s.lane_id,
            "speed": s.speed_mps,
            "free_flow_speed_mps": s.free_flow_speed_mps,
        } for s in snapshots]

    def get_segment_metrics(self, snapshots: list[VehicleSnapshot] | None = None) -> dict:
        """Per-segment metrics, from the SHARED implementation.

        `compute_segment_metrics` is what highway_env uses, so both simulators agree
        on what a queue, a roadblock and lane occupancy are. An earlier version
        computed four fields by hand and left the other seven absent, which starved
        the MAPPO critic: `encode_physical_global_state` reads eleven fields per
        segment and received none of them.
        """
        snapshots = snapshots if snapshots is not None else self.vehicle_snapshots()
        road = (self.config.get("topology") or {}).get("road", {})
        lengths = road.get("segment_lengths", {}) or {}
        segment_ids = sorted({s.segment_id for s in snapshots if s.segment_id} | set(lengths))
        inflow: dict[str, int] = {}
        outflow: dict[str, int] = {}
        for snapshot in snapshots:
            previous = self._segment_of.get(snapshot.vehicle_id)
            if previous != snapshot.segment_id:
                if snapshot.segment_id:
                    inflow[snapshot.segment_id] = inflow.get(snapshot.segment_id, 0) + 1
                if previous:
                    outflow[previous] = outflow.get(previous, 0) + 1
            if snapshot.segment_id:
                self._segment_of[snapshot.vehicle_id] = snapshot.segment_id
        lane_counts = dict(self.view.lane_counts) if self.view is not None else {}
        return compute_segment_metrics(
            segment_ids=segment_ids,
            segment_lengths_m=lengths,
            lane_counts=lane_counts,
            active_vehicle_records=self._vehicle_records(snapshots),
            step_inflow=inflow,
            step_outflow=outflow,
            thresholds=metric_thresholds_from_config(self.config),
        )

    def _branch_fairness(self) -> float:
        """Jain's index over completions per entry branch."""
        from src.metrics.global_metrics import jain_fairness

        return float(jain_fairness(list(self._branch_completed.values())))

    def get_global_state(self) -> dict[str, Any]:
        """What the MAPPO critic reads, in the shape the encoder expects.

        `encode_physical_global_state` reads `time`, `active_vehicle_count`,
        `active_av_count`, then ten segments of eleven fields, then two demand
        fields. An earlier version returned the flat metrics dict, which contains
        none of `time`, `segment_state` or `demand_state`, so the centralized critic
        -- the entire point of MAPPO over IPPO -- was fed two non-zero values out of
        115 and both SUMO training runs are invalid.
        """
        snapshots = self.vehicle_snapshots()
        demand = self.config.get("demand", {})
        return {
            "time": self.step_count * float(self.config.get("dt", 1.0)),
            "topology_id": self.topology_id,
            "active_vehicle_count": len(snapshots),
            "active_av_count": sum(1 for s in snapshots if s.role == "av"),
            "completed_vehicle_count": self.arrived_total,
            "segment_state": self.get_segment_metrics(snapshots),
            "branch_state": {
                "per_branch_spawned": dict(self._branch_spawned),
                "per_branch_completed": dict(self._branch_completed),
                "fairness_jain": self._branch_fairness(),
            },
            "demand_state": {
                "current_vehicles_per_hour": float(demand.get("total_vehicles_per_hour", 0.0)),
                "av_penetration": float(demand.get("av_penetration", 0.0)),
            },
            "step_metrics": dict(self._last_metrics),
        }

    def crashed_agent_ids(self) -> list[str]:
        """Always empty: SUMO's car-following cannot produce a collision.

        So `crash_penalty` is inert on this simulator. That is the intended
        consequence of the migration rather than an oversight -- the penalty existed
        because the previous simulator's dense speed reward drowned a one-step
        collision term, and there are no collisions to drown it now. The per-step
        collision counter is what verifies the claim; this does not assert it.
        """
        return []

    def get_episode_summary(self) -> dict[str, Any]:
        return {"steps": self.step_count, "collisions": self.collision_count,
                "arrived": self.arrived_total, "metrics": dict(self._last_metrics)}

    # ----------------------------------------------------------------- reading

    def vehicle_snapshots(self) -> list[VehicleSnapshot]:
        """Every vehicle on the network, as the sensing model expects them."""
        if not self._running or self.network is None:
            return []
        snapshots: list[VehicleSnapshot] = []
        for vehicle_id in _sumo.vehicle.getIDList():
            lane_id = _sumo.vehicle.getLaneID(vehicle_id)
            if lane_id.startswith(":"):
                continue  # inside a junction: no edge, so no segment to report
            edge_id = _sumo.vehicle.getRoadID(vehicle_id)
            segment_id = self.network.segment_for_edge(edge_id)
            if segment_id is None:
                continue
            ordinal = int(_sumo.vehicle.getLaneIndex(vehicle_id))
            position = _sumo.vehicle.getPosition(vehicle_id)
            snapshots.append(VehicleSnapshot(
                vehicle_id=vehicle_id,
                role="av" if _sumo.vehicle.getTypeID(vehicle_id).startswith(self.AV_TYPE) else "human",
                segment_id=segment_id,
                lane_index=self._lane_index(edge_id, ordinal),
                lane_id=ordinal,
                position=(float(position[0]), float(position[1])),
                longitudinal_m=float(_sumo.vehicle.getLanePosition(vehicle_id)),
                speed_mps=float(_sumo.vehicle.getSpeed(vehicle_id)),
                acceleration_mps2=float(_sumo.vehicle.getAcceleration(vehicle_id)),
                free_flow_speed_mps=float(_sumo.vehicle.getAllowedSpeed(vehicle_id)),
                # SUMO does not produce collisions; the counter above is what
                # verifies that rather than this field asserting it.
                crashed=False,
            ))
        return snapshots

    def _lane_index(self, edge_id: str, ordinal: int) -> tuple[str, str, int]:
        """`(from_junction, to_junction, ordinal)`, the shape `highway_env` used.

        A SUMO edge runs between two junctions, so the old identifier translates
        exactly and nothing downstream has to learn a new one.
        """
        from_node, to_node = self._edge_nodes(edge_id)
        return (from_node, to_node, ordinal)

    def _edge_nodes(self, edge_id: str) -> tuple[str, str]:
        if not hasattr(self, "_edge_node_cache"):
            import sumolib

            net = sumolib.net.readNet(str(self.network.net_file))
            self._edge_node_cache = {
                edge.getID(): (edge.getFromNode().getID(), edge.getToNode().getID())
                for edge in net.getEdges()
                if not edge.getID().startswith(":")
            }
        return self._edge_node_cache.get(edge_id, (edge_id, edge_id))

    def get_local_observations(self, snapshots: list[VehicleSnapshot] | None = None) -> dict[str, Any]:
        """Per-AV observations, from the unchanged sensing model.

        The builder, its route-aware gap search and its merge projection are exactly
        as validated; only the road view and the snapshots come from SUMO.
        """
        if not self._running or self.view is None:
            return {}
        snapshots = snapshots if snapshots is not None else self.vehicle_snapshots()
        agent_ids = [s.vehicle_id for s in snapshots if s.role == "av"]
        if not agent_ids:
            return {}
        for agent_id in agent_ids:
            self._safety_states.setdefault(agent_id, SafetyState())
            self._target_headways.setdefault(agent_id, 1.6)
        return self._sensing.build_all(
            time_s=self.step_count * float(self.config.get("dt", 1.0)),
            topology=self.view,
            snapshots=snapshots,
            current_av_ids=agent_ids,
            safety_states={a: self._safety_states[a] for a in agent_ids},
            target_headways={a: self._target_headways[a] for a in agent_ids},
            target_lanes={a: None for a in agent_ids},
            segment_metrics=self.get_segment_metrics(snapshots),
            # From the topology's own `safety:` block, not library defaults. The
            # previous form passed a bare SafetyConstraints(), so lane-change dwell,
            # the per-km change limit and the follower-braking limit were whatever
            # the library shipped rather than what inverted_tree declares.
            constraints=self._safety_constraints(),
            rng=self._rng,
        )

    def _safety_constraints(self) -> SafetyConstraints:
        """The topology's declared safety limits, falling back to the defaults."""
        declared = (self.config.get("topology") or {}).get("safety", {}) or {}
        fields = {f.name for f in dataclasses.fields(SafetyConstraints)}
        return SafetyConstraints(**{k: v for k, v in declared.items() if k in fields})

    # ------------------------------------------------------------------ demand

    def _write_routes(self) -> Path:
        """Demand as SUMO flows, one AV flow and one human flow per entry edge.

        Flows rather than our own spawner: SUMO handles insertion, refuses to insert
        into an occupied gap, and delays rather than overlapping vehicles. Writing
        our own would reintroduce the spawn-side problems SUMO solves natively.
        """
        assert self.network is not None
        demand = self.config.get("demand", {})
        total_per_hour = float(demand.get("total_vehicles_per_hour", 1800.0))
        penetration = float(demand.get("av_penetration", 0.0))
        speed = demand.get("speed_distribution", {}) or {}
        entries = sorted(self.network.entry_edges())
        per_entry = total_per_hour / max(len(entries), 1)
        # The flow must cover the warm-up as well as the episode, or demand stops
        # before the episode does and the last stretch is a draining network. With
        # a 300-step warm-up and a 600-step episode that was a third of the run,
        # and it read as mean_speed 0.000 at the final step.
        total_steps = (int(self.config.get("warmup_steps", 0))
                       + int(self.config.get("duration_steps", 120)))
        duration_s = total_steps * float(self.config.get("dt", 1.0))

        # The two vTypes are IDENTICAL except for their id and colour. Anything
        # else would confound penetration with a change in the fleet: an earlier
        # version gave humans a speedFactor spread and the AV type none, so raising
        # penetration made the fleet more homogeneous and mean speed rose from 6.15
        # to 11.29 m/s with no controller acting at all. That would have been
        # reported as a control effect.
        max_speed = float(speed.get("max_mps", 32.0))
        speed_factor = "normc(1,0.1,0.8,1.2)"
        lines = [
            "<routes>",
            f'  <vType id="{self.HUMAN_TYPE}" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}"/>',
            # The AV keeps SUMO's safe car-following as a floor; the project's own
            # controller commands speed on top of it through setSpeed.
            f'  <vType id="{self.AV_TYPE}" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" color="1,0,0"/>',
            # ONE distribution, drawn per vehicle, rather than one flow per type.
            # SUMO spaces each flow evenly on its own, so two flows at rates r1 and
            # r2 do not produce the same arrival process as one flow at r1+r2 --
            # which made the arrival pattern depend on penetration and moved mean
            # speed with no controller acting.
            f'  <vTypeDistribution id="{self.MIX_TYPE}">',
            f'    <vType id="{self.HUMAN_TYPE}_d" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" probability="{1.0 - penetration:.4f}"/>',
            f'    <vType id="{self.AV_TYPE}_d" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" color="1,0,0" probability="{penetration:.4f}"/>',
            "  </vTypeDistribution>",
        ]
        for index, edge in enumerate(entries):
            route = self.network.route_to_exit(edge)
            if not route:
                continue
            edges = " ".join(route)
            lines.append(f'  <route id="r{index}" edges="{edges}"/>')
            # One flow at the full rate, its type drawn from the distribution, so
            # the arrival process is identical whatever the penetration is and the
            # only thing penetration changes is which vehicles are controllable.
            lines.append(
                f'  <flow id="f{index}" type="{self.MIX_TYPE}" route="r{index}"'
                f' begin="0" end="{duration_s}" vehsPerHour="{per_entry:.3f}"/>'
            )
        lines.append("</routes>")
        route_file = self.work_dir / "demand.rou.xml"
        route_file.write_text("\n".join(lines) + "\n")
        return route_file
