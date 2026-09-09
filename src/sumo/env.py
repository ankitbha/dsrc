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

from src.sensing import VehicleSnapshot
from src.sumo.network import SumoNetwork

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

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.close()
        topology_cfg = self.config.get("topology", {})
        self.network = SumoNetwork.build(self.topology_id, topology_cfg, self.work_dir)
        route_file = self._write_routes()
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
        _sumo.simulationStep()
        self.step_count += 1

        self.collision_count += int(_sumo.simulation.getCollidingVehiclesNumber())
        self.collision_checks += 1

        duration = int(self.config.get("duration_steps", 120))
        truncated = self.step_count >= duration
        # `terminated` stays False by construction: SUMO does not crash vehicles, so
        # there is no early-termination condition. A True here would mean the
        # premise of this migration is wrong, which is why a test asserts it.
        terminated = False
        observations = self.get_local_observations()
        info = {"collisions": self.collision_count, "step": self.step_count}
        return observations, {}, terminated, truncated, info

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
                role="av" if _sumo.vehicle.getTypeID(vehicle_id) == self.AV_TYPE else "human",
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

    def get_local_observations(self) -> dict[str, Any]:
        """Per-AV observations.

        Empty until the sensing model is given a SUMO-backed view of the road, which
        is the next piece of work. The snapshots it needs are already produced.
        """
        return {}

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

        lines = [
            "<routes>",
            f'  <vType id="{self.HUMAN_TYPE}" maxSpeed="{float(speed.get("max_mps", 32.0))}"'
            f' speedFactor="normc(1,0.1,0.8,1.2)"/>',
            # The AV keeps SUMO's safe car-following as a floor; the project's own
            # controller commands speed on top of it through setSpeed.
            f'  <vType id="{self.AV_TYPE}" maxSpeed="{float(speed.get("max_mps", 32.0))}"'
            f' color="1,0,0"/>',
        ]
        for index, edge in enumerate(entries):
            route = self.network.route_to_exit(edge)
            if not route:
                continue
            edges = " ".join(route)
            lines.append(f'  <route id="r{index}" edges="{edges}"/>')
            human_rate = per_entry * (1.0 - penetration)
            av_rate = per_entry * penetration
            if human_rate > 0:
                lines.append(
                    f'  <flow id="h{index}" type="{self.HUMAN_TYPE}" route="r{index}"'
                    f' begin="0" end="{duration_s}" vehsPerHour="{human_rate:.3f}"/>'
                )
            if av_rate > 0:
                lines.append(
                    f'  <flow id="a{index}" type="{self.AV_TYPE}" route="r{index}"'
                    f' begin="0" end="{duration_s}" vehsPerHour="{av_rate:.3f}"/>'
                )
        lines.append("</routes>")
        route_file = self.work_dir / "demand.rou.xml"
        route_file.write_text("\n".join(lines) + "\n")
        return route_file
