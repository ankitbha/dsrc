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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.wrappers import decode_headway_bin, decode_speed_bin
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


class SumoTopologyEnv:
    """Runs one SUMO simulation and reports it as `VehicleSnapshot`s."""

    #: vType ids. Two types rather than one, so a vehicle's role is a property of
    #: the demand that created it rather than something inferred from its id.
    AV_TYPE = "av"
    HUMAN_TYPE = "human"
    #: The distribution the flows draw from, so penetration changes who is
    #: controllable without changing the arrival process.
    MIX_TYPE = "mix"

    def __init__(self, topology_id: str, config: Mapping[str, Any]) -> None:
        self.topology_id = topology_id
        self.config = dict(config)
        self.work_dir = Path(self.config.get("work_dir", ".")) / f"sumo_{topology_id}"
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

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
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
            # readable. Nothing is expected to arrive here.
            "--collision.action", "warn",
        ]
        if seed is not None:
            args += ["--seed", str(int(seed))]
        _sumo.start([self._binary()] + args)
        self._running = True
        return self.get_local_observations(), {"binding": _BINDING}

    def close(self) -> None:
        if not self._running:
            return
        try:
            _sumo.close()
        except Exception:  # noqa: BLE001 - closing an already-dead connection
            pass
        self._running = False

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

        self.collision_count += int(_sumo.simulation.getCollidingVehiclesNumber())
        self.collision_checks += 1

        arrived = int(_sumo.simulation.getArrivedNumber())
        self.arrived_total += arrived
        now = self.step_count * float(self.config.get("dt", 1.0))
        self._arrivals.extend([now] * arrived)

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
        info = {"collisions": self.collision_count, "step": self.step_count,
                "metrics": self._last_metrics, "safety": {"penalties": {}}}
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
            "new_collision_count": 0,
            "hard_braking_count": sum(
                1 for s in snapshots if s.acceleration_mps2 < -3.0
            ),
            "rolling_roadblock_score": 0.0,
            "fairness_jain": self._fairness(speeds),
            "active_vehicle_count": len(snapshots),
            "active_av_count": len(self.agent_ids),
        }

    @staticmethod
    def _fairness(speeds: list[float]) -> float:
        """Jain's index over speeds. 1.0 when every vehicle moves alike."""
        if not speeds:
            return 1.0
        total = sum(speeds)
        squares = sum(v * v for v in speeds)
        if squares <= 0.0:
            return 1.0
        return float(total * total / (len(speeds) * squares))

    def get_segment_metrics(self, snapshots: list[VehicleSnapshot] | None = None) -> dict:
        """Per-segment density, jam fraction and queue length.

        A vehicle is queued when it is below `queue_speed_mps`, which is the same
        threshold the sensing model uses, so the two agree on what a queue is.
        """
        snapshots = snapshots if snapshots is not None else self.vehicle_snapshots()
        queue_speed = float(self._sensing.config.queue_speed_mps)
        by_segment: dict[str, list[VehicleSnapshot]] = {}
        for snapshot in snapshots:
            if snapshot.segment_id:
                by_segment.setdefault(snapshot.segment_id, []).append(snapshot)
        out: dict[str, dict[str, float]] = {}
        for segment, group in by_segment.items():
            slow = [s for s in group if s.speed_mps < queue_speed]
            out[segment] = {
                "density": float(len(group)),
                "jam_fraction": float(len(slow)) / float(len(group)),
                "queue_length": float(len(slow)),
                "mean_speed": float(sum(s.speed_mps for s in group) / len(group)),
            }
        return out

    def crashed_agent_ids(self) -> list[str]:
        """Always empty: SUMO's car-following cannot produce a collision.

        So `crash_penalty` is inert on this simulator. That is the intended
        consequence of the migration rather than an oversight -- the penalty existed
        because the previous simulator's dense speed reward otherwise drowned a
        one-step collision term, and there are no collisions to drown it now. The
        per-step collision counter is what verifies the claim; this does not assert
        it.
        """
        return []

    def get_global_state(self) -> dict[str, Any]:
        return dict(self._last_metrics)

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
            constraints=SafetyConstraints(),
            rng=self._rng,
        )

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
        duration_s = int(self.config.get("duration_steps", 120)) * float(self.config.get("dt", 1.0))

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
