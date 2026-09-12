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
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.wrappers import (
    decode_headway_bin,
    decode_speed_bin,
    lane_preference_to_action,
)
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
    #: SUMO's default lane-change mode: the vehicle obeys its own safety checks and
    #: may refuse a requested change. Named rather than left implicit because a
    #: controller that could force an unsafe change would defeat the reason this
    #: simulator was chosen.
    LANE_CHANGE_MODE = 1621
    #: `hold_lane`: the vehicle's own lane-change motivations are off (the low bits
    #: are cleared) while collision avoidance stays on. NOT 0. Bits 8-9 of a SUMO
    #: lane-change mode are the collision-avoidance bits, so mode 0 does not mean
    #: "no lane changes", it means "no lane changes and no safety". Measured: with
    #: mode 0, actions varying lane preference and merge mode together produced 646
    #: collisions over 3000 steps, and neither head alone produced any. 1536 is
    #: 512 (respect other drivers) + 1024 (sublane), the two bits the default sets
    #: that are not a motivation to change lane.
    HOLD_LANE_MODE = 1536
    #: Below this a vehicle counts as stopped. SUMO's own definition of a waiting
    #: vehicle uses the same value, so `stopped_fraction` summed over the episode
    #: and multiplied by dt is total waiting time in vehicle-seconds.
    STOPPED_SPEED_MPS = 0.1

    def __init__(self, topology_id: str, config: Mapping[str, Any]) -> None:
        self.topology_id = topology_id
        self.config = dict(config)
        # A temporary directory by default, NOT the working directory. Defaulting to
        # "." wrote generated network and demand files into whatever directory the
        # caller happened to be in, and a `sumo_inverted_tree/` appeared in the
        # repository root. Generated artefacts do not belong in the repo, and a
        # caller that wants to keep them passes `work_dir`.
        # The default is per process. A fixed shared path let two processes running
        # the same topology -- an evaluation sweep, or a pytest-xdist worker --
        # overwrite each other's demand file between the write and the read, so a
        # run could silently use another run's penetration or duration.
        configured = self.config.get("work_dir")
        base = (Path(configured) if configured
                else Path(tempfile.gettempdir()) / "dsrc_sumo" / f"pid_{os.getpid()}")
        self.work_dir = base / f"sumo_{topology_id}"
        self.network: SumoNetwork | None = None
        self.step_count = 0
        self.collision_count = 0
        #: How many times the collision count has been read. Exposed so a test can
        #: distinguish "no collisions happened" from "nobody looked".
        self.collision_checks = 0
        #: Vehicles currently registered as colliding, so a collision that persists
        #: across steps is counted once. `--collision.action warn` leaves the
        #: vehicles on the network, so summing `getCollidingVehiclesNumber` counted
        #: vehicle-steps: two vehicles in contact for three steps read as 6.
        self._colliding_ids: set[str] = set()
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
        self._step_inflow: dict[str, int] = {}
        self._step_outflow: dict[str, int] = {}
        #: Segment metrics are a pure function of one step, and the getter is called
        #: two or three times per step, so they are computed once and reused.
        self._cached_segment_metrics: dict | None = None
        #: New collisions this step, exposed so a test can tell a measured zero from
        #: a hardcoded one.
        self.new_collisions_last_step = 0
        #: Lane changes the controller asked for. SUMO may refuse any of them, so
        #: this counts requests and not changes; it is what tells an inert lane head
        #: from one the policy simply does not use.
        self.lane_change_requests = 0
        #: When each live vehicle departed, so a completed trip's duration can be
        #: measured. Kept for the same reason `_origin_of` is: the vehicle is gone
        #: by the time it arrives, so nothing about it can be read then.
        self._departure_time: dict[str, float] = {}
        #: (arrival time, travel time, free-flow travel time) per completed trip,
        #: pruned to the same rolling window `_arrivals` uses.
        self._completed_trips: list[tuple[float, float, float]] = []
        #: Free-flow travel time of each route, by the entry edge that identifies
        #: it: the route's length divided by the speed limit. The subtrahend in
        #: delay, and a property of the road rather than of any run.
        self._route_free_flow_s: dict[str, float] = {}
        #: Each vehicle's acceleration last step, for jerk. Rebound rather than
        #: updated every step, so a vehicle that was inside a junction last step --
        #: and so absent from the snapshots -- contributes nothing rather than a
        #: difference taken across the gap.
        self._last_accel: dict[str, float] = {}
        #: The acceleration map as it stood BEFORE `_mean_abs_jerk` rebound it.
        #: `_build_metrics` runs before `neighbourhood_metrics` in a step, so by the
        #: time the per-agent reward is priced `_last_accel` already holds this
        #: step's values and a difference against it would be identically zero.
        self._previous_accel: dict[str, float] = {}
        #: When each segment last passed a vehicle downstream, pruned to the
        #: rolling window. The per-segment analogue of `_arrivals`, which is what
        #: lets a local reward price throughput near the agent.
        self._segment_outflow_times: dict[str, list[float]] = {}

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        global _LIVE
        if _LIVE is not None and _LIVE is not self:
            if _LIVE._running:
                raise RuntimeError(
                    "another SumoTopologyEnv already holds the SUMO connection; "
                    "libsumo is process-global, so close it before resetting this one"
                )
            # The marker outlived the env that set it, which holds no connection and
            # must not block this one. Without this an env dropped without `close`
            # locked the process: every later reset raised, so one failing test
            # cascaded into every SUMO test that followed it.
            _LIVE = None
        self.close()
        topology_cfg = self.config.get("topology", {})
        self.network = SumoNetwork.build(self.topology_id, topology_cfg, self.work_dir)
        self.view = SumoTopologyView(self.network)
        route_file = self._write_routes()
        self.agent_ids = []
        self.arrived_total = 0
        self._safety_states = {}
        self._target_headways = {}
        # Cleared like every other episode-scoped value. Left over, it made
        # `get_episode_summary()` and `get_global_state()["step_metrics"]` report
        # the previous episode's numbers on a freshly reset environment.
        self._last_metrics = {}
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
        self._step_inflow = {}
        self._step_outflow = {}
        self._cached_segment_metrics = None
        self.new_collisions_last_step = 0
        self.lane_change_requests = 0
        self._departure_time = {}
        self._completed_trips = []
        self._last_accel = {}
        self._previous_accel = {}
        self._segment_outflow_times = {}
        # Rebuilt every reset, because `reset` rebuilds the network and a cache
        # carried across a change of lane count or segment length would describe
        # the previous road.
        self._edge_node_cache = None
        self._route_free_flow_s = self._route_free_flow_times()
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        self.step_count = 0
        self.collision_count = 0
        self.collision_checks = 0
        self._colliding_ids = set()
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
        try:
            self._warm_up()
            return self.get_local_observations(), {"binding": _BINDING}
        except BaseException:
            # Releasing the connection on the way out, so a failure here does not
            # leave the process holding a marker no one can clear.
            try:
                self.close()
            except BaseException:  # noqa: BLE001 - see below
                # A failure to close must not replace the failure being reported.
                # `close` has already released the marker in its own `finally`, so
                # the only thing lost here is the close's own error, and losing the
                # original would leave no way to diagnose what actually failed.
                pass
            raise

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
        # Over the last window-worth of warm-up steps the per-segment flows are
        # recorded too, at the same negative times `_arrivals` uses. Without it the
        # per-segment rolling outflow that the local reward prices starts empty and
        # refills over the first 60 s of the episode, so the same traffic scores
        # lower at the start than in the middle. Only the tail, because building the
        # snapshots costs about six TraCI calls per vehicle and the warm-up is
        # twelve thousand steps at four hundred vehicles.
        flow_from = max(0, warmup - int(round(
            float(metric_thresholds_from_config(self.config).throughput_window_s) / (dt or 1.0))))
        for index in range(max(0, warmup)):
            _sumo.simulationStep()
            # Collisions during warm-up would still be collisions.
            self.collision_count += self._new_collisions()
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
            now = (index - warmup + 1) * dt
            self._arrivals.extend([now] * arrived)
            self._track_trips(now)
            if index >= flow_from:
                self._update_flows(self.vehicle_snapshots(), now)
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
            # Seed the segment map without emitting flows: the first observed step
            # would otherwise report every vehicle already on the network as an
            # arrival into its segment. Redundant when the warm-up was longer than
            # the flow window, since the tail above has already advanced it, and
            # kept because a warm-up shorter than the window does not reach that
            # branch.
            self._segment_of = {s.vehicle_id: s.segment_id for s in snapshots if s.segment_id}
            # Seeded for the same reason as `_segment_of`: without it every vehicle
            # already on the network reads its whole acceleration as a one-step
            # change on the first observed step.
            self._last_accel = {s.vehicle_id: s.acceleration_mps2 for s in snapshots}
            self.agent_ids = [s.vehicle_id for s in snapshots if s.role == "av"]

    def _new_collisions(self) -> int:
        """Vehicles that entered a collision since the previous step.

        `getCollidingVehiclesNumber` reports how many vehicles are in a collision
        right now, and `--collision.action warn` leaves them on the network, so
        adding it up each step counted vehicle-steps rather than events: two
        vehicles in contact for three steps would read as 6. The reward's -5.0
        weight is applied per event, so the difference is not cosmetic -- it is
        inert only while the count is zero, which is the claim the counter exists
        to check.
        """
        current = set(_sumo.simulation.getCollidingVehiclesIDList())
        new = len(current - self._colliding_ids)
        self._colliding_ids = current
        return new

    def close(self) -> None:
        global _LIVE
        if not self._running:
            return
        try:
            _sumo.close()
        except Exception:  # noqa: BLE001 - closing an already-dead connection
            pass
        finally:
            # In a `finally`, because a BaseException from the close -- a
            # KeyboardInterrupt or a SystemExit -- propagates past the `except
            # Exception` above. It left `_running` True and `_LIVE` set, and every
            # later reset in the process then raised: the same cascade the reset
            # path was hardened against, one step to the side.
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

        self.new_collisions_last_step = self._new_collisions()
        self.collision_count += self.new_collisions_last_step
        self.collision_checks += 1
        arrived = int(_sumo.simulation.getArrivedNumber())
        self.arrived_total += arrived
        now = self.step_count * float(self.config.get("dt", 1.0))
        self._arrivals.extend([now] * arrived)
        self._track_trips(now)

        snapshots = self.vehicle_snapshots()
        self._cached_segment_metrics = None
        self._update_flows(snapshots, now)
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
                # What a per-agent reward is priced from; empty when no agent is on
                # a segment. See `neighbourhood_metrics`.
                "neighbourhood": self.neighbourhood_metrics(snapshots),
                # Per-segment density as a fraction of jam density, which is what
                # the threshold reward reads. Reported alongside the metrics rather
                # than folded into them, because it is a per-segment mapping and
                # they are network aggregates.
                "density_ratios": self.density_ratios(snapshots),
                "segment_metrics": self.get_segment_metrics(snapshots),
                "safety": {"penalties": {}, "layer_ran": False}}
        return observations, {}, terminated, truncated, info

    # ---------------------------------------------------------------- actuation

    def _apply_actions(self, actions: Mapping[str, Any]) -> None:
        """Actuate every head of the action contract.

        `setSpeed` rather than `setAcceleration`: SUMO applies its own safe-speed
        bound to whatever is asked either way, so speed is the more direct
        expression of what the controller decided and the bound stays SUMO's rather
        than being reimplemented here. Its safety guarantee is the reason for the
        migration, so nothing here may bypass it.

        The other three heads previously reached the simulator nowhere: the
        headway bin changed an observation field and nothing else, and lane
        preference and merge mode were discarded. A policy trained on the `full`
        profile was therefore choosing three of its four outputs against no
        consequence, which is not a control experiment.
        """
        for agent_id, action in (actions or {}).items():
            if agent_id not in self.agent_ids:
                continue
            action = action or {}
            self._apply_speed(agent_id, action.get("desired_speed_bin"))
            self._apply_headway(agent_id, action.get("desired_headway_bin"),
                                action.get("merge_mode"))
            self._apply_lane(agent_id, action.get("lane_preference"),
                             action.get("merge_mode"))

    def _apply_speed(self, agent_id: str, bin_name: Any) -> None:
        """Command the speed the chosen bin names.

        `speed_bins_mps` in the config, when present, gives the three bins ABSOLUTE
        values in m/s and bypasses `decode_speed_bin`. Without it the bins are
        `free_flow + offset` with a 12 m/s floor, which at a 30 m/s limit is 20, 27
        and 30 m/s; measured over 114,889 AV-steps at a mean AV speed of 7.33 m/s,
        `slow` binds on 15.6% of them, `nominal` on 0.9% and `fast` on 0.07%, so all
        three values do the same thing on 84% of steps. The predecessor paper
        commands 30, 45 and 60 km/h -- 8.3, 12.5 and 16.7 m/s -- which is the band
        congested traffic is actually in.

        The default is unchanged, so every configuration written before this reads
        the contract's own bins.
        """
        if bin_name is None:
            return
        absolute = self.config.get("speed_bins_mps") or {}
        if absolute:
            target = absolute.get(str(bin_name))
            if target is None:
                raise ValueError(
                    f"speed_bins_mps declares {sorted(absolute)} and the policy chose "
                    f"{bin_name!r}; every bin the action space can emit needs a value"
                )
            self.command_speed(agent_id, float(target))
            return
        allowed = float(_sumo.vehicle.getAllowedSpeed(agent_id))
        try:
            # Returns a contextual speed in m/s, not a factor.
            target = float(decode_speed_bin(str(bin_name), free_flow_speed_mps=allowed))
        except Exception:  # noqa: BLE001 - an unknown bin is a caller error
            return
        self.command_speed(agent_id, target)

    def _apply_headway(self, agent_id: str, headway_bin: Any, merge_mode: Any) -> None:
        """The desired time headway, as SUMO's own car-following parameter.

        `setTau` is the direct expression of a desired headway: it is the tau of
        the Krauss model the vehicle is already following, so asking for a larger
        headway makes the vehicle keep one rather than merely reporting that it
        wants to. `create_gap` adds the project's declared
        `merge_gap_headway_bonus_s` on top, which is what that constant is for.
        """
        if headway_bin is None:
            return
        try:
            target = float(decode_headway_bin(str(headway_bin)))
        except Exception:  # noqa: BLE001 - an unknown bin is a caller error
            return
        if str(merge_mode) == "create_gap":
            target += float(self._safety_constraints().merge_gap_headway_bonus_s)
        self._target_headways[agent_id] = target
        _sumo.vehicle.setTau(agent_id, target)

    def _apply_lane(self, agent_id: str, lane_preference: Any, merge_mode: Any) -> None:
        """A lane-change request, left for SUMO's lane-change model to accept.

        `changeLane` asks; the model still refuses a change it considers unsafe,
        which keeps the safety guarantee where the migration put it. `hold_lane`
        suppresses changes outright by setting the lane-change mode to zero, and is
        restored on the next step by any other action.

        On a single-lane segment there is no lane to move to and the request is a
        no-op. Six of this topology's nine segments have one lane, so this head is
        actionable on a minority of steps; `lane_change_requests` counts how often.
        """
        if lane_preference is None and merge_mode is None:
            return
        if str(merge_mode) == "hold_lane":
            _sumo.vehicle.setLaneChangeMode(agent_id, self.HOLD_LANE_MODE)
            # Cancel a change this agent asked for on an earlier step. `changeLane`
            # holds its choice for a duration, so without this a vehicle told to
            # hold its lane carries on into a change requested before it.
            edge_id = _sumo.vehicle.getRoadID(agent_id)
            if not edge_id.startswith(":"):
                _sumo.vehicle.changeLane(
                    agent_id, int(_sumo.vehicle.getLaneIndex(agent_id)),
                    float(self.config.get("dt", 1.0)))
            return
        _sumo.vehicle.setLaneChangeMode(agent_id, self.LANE_CHANGE_MODE)
        try:
            candidate = lane_preference_to_action(str(lane_preference))
        except KeyError:
            return
        if candidate is None:
            return
        edge_id = _sumo.vehicle.getRoadID(agent_id)
        if edge_id.startswith(":"):
            return  # inside a junction: no lane index to move within
        lane_count = int(_sumo.edge.getLaneNumber(edge_id))
        current = int(_sumo.vehicle.getLaneIndex(agent_id))
        # SUMO numbers lanes from the right, so a higher index is further left.
        target = current + 1 if candidate == "LANE_LEFT" else current - 1
        if not 0 <= target < lane_count:
            return
        self.lane_change_requests += 1
        _sumo.vehicle.changeLane(agent_id, target, float(self.config.get("dt", 1.0)))

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

    def _latent_demand(self) -> int:
        """How many vehicles are waiting to enter and cannot.

        SUMO holds a scheduled vehicle until its entry lane has room, and the route
        file sets no departure deadline, so nothing is discarded: this is the queue
        outside the network. At 3000 veh/h offered it stays at zero for the first
        900 s while the 500 m leaves absorb the excess, then grows linearly once
        they saturate -- 198 vehicles at t = 900 s and 398 at t = 1350 s.
        """
        if not self._running:
            return 0
        return len(_sumo.simulation.getPendingVehicles())

    def _network_census(self) -> tuple[list[float], int, int]:
        """Speeds, vehicle count and AV count over every vehicle on the network.

        `vehicle_snapshots` skips vehicles inside a junction, which have no edge and
        so no segment. That is right for the per-segment metrics and wrong for a
        network-wide count: `HighwayTopologyEnv` counts every vehicle on the road,
        junction-crossing vehicles are moving, and excluding them biased `mean_speed`
        downward -- 6.157 m/s against SUMO's own 6.358 over all vehicles, 3.2% low,
        on 1.44% of vehicle-steps.

        `agent_ids` stays snapshot-derived, because an agent needs an observation and
        a vehicle with no segment cannot be given one. `active_av_count` can
        therefore exceed the number of agents acting in a step, which is the honest
        reading: the vehicle is on the network and controllable next step.
        """
        if not self._running:
            return [], 0, 0
        speeds: list[float] = []
        av_count = 0
        for vehicle_id in _sumo.vehicle.getIDList():
            speeds.append(float(_sumo.vehicle.getSpeed(vehicle_id)))
            if _sumo.vehicle.getTypeID(vehicle_id).startswith(self.AV_TYPE):
                av_count += 1
        return speeds, len(speeds), av_count

    def _build_metrics(self, snapshots: list[VehicleSnapshot], now: float) -> dict[str, Any]:
        """The metric shape the reward and the health check already read.

        Fed from SUMO rather than recomputed differently, so `build_team_reward` and
        `simulator_health` need no change. `collision_count` comes from the counter
        read every step, not from a per-vehicle flag, because SUMO removes nothing
        and a flag would always read clean.
        """
        thresholds = metric_thresholds_from_config(self.config)
        # From `metrics.thresholds`, the same place HighwayTopologyEnv reads it. This
        # read was against the top level of the config, where nothing writes it, so a
        # config setting the window under `metrics.thresholds` -- which is where the
        # experiment configs set it -- was honoured on one simulator and ignored on
        # the other, and the rolling throughput window silently stayed at 60 s.
        window = float(thresholds.throughput_window_s)
        self._arrivals = [t for t in self._arrivals if now - t <= window]
        # Every vehicle on the network, not only those the snapshots cover.
        # `vehicle_snapshots` skips vehicles inside a junction, which have no edge
        # and so no segment; that is right for the per-segment metrics and wrong for
        # a network-wide mean. Junction-crossing vehicles are moving, so excluding
        # them biased `mean_speed` downward -- measured at 6.157 m/s against SUMO's
        # own 6.358 over all vehicles, 3.2% low, on 1.44% of vehicle-steps.
        speeds, active_count, active_av_count = self._network_census()
        segment_metrics = self.get_segment_metrics(snapshots)
        jam = [m["jam_fraction"] for m in segment_metrics.values()]
        queue = sum(int(m["queue_length"]) for m in segment_metrics.values())
        # Same window as `throughput_recent`, for the same reason: a completion is a
        # rare event per step and the reward needs a quantity that is not zero on
        # most of them.
        self._completed_trips = [t for t in self._completed_trips if now - t[0] <= window]
        for times in self._segment_outflow_times.values():
            times[:] = [t for t in times if now - t <= window]
        travel = [t[1] for t in self._completed_trips]
        delay = [max(0.0, t[1] - t[2]) for t in self._completed_trips if t[2] > 0.0]
        jerk = self._mean_abs_jerk(snapshots)
        return {
            "mean_speed": float(sum(speeds) / len(speeds)) if speeds else 0.0,
            "speed_std": float(np.std(speeds)) if speeds else 0.0,
            "throughput_recent": len(self._arrivals),
            # Vehicles that cannot get onto the road at all, because the entry lanes
            # are full. Under a demand above capacity this is where the unserved
            # demand goes, and counting it is what makes the accounting complete:
            # served plus on-road plus latent equals offered. Reported rather than
            # weighted, so it cannot silently enter the objective.
            "latent_demand": self._latent_demand(),
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
            # Reported and logged, but NOT weighted: there is no entry for it in
            # DEFAULT_REWARD_WEIGHTS and no config overrides one, so this aggregate
            # reaches `build_team_reward` at weight zero. The per-segment version is
            # one of the eleven fields the critic reads, so the metric is not dead;
            # only this aggregate is unpriced. `rolling_roadblock_score` above is the
            # term that carries -2.0, and it is built from this one.
            "all_lane_av_low_speed_occupancy": (
                sum(float(m["all_lane_av_low_speed_occupancy"]) for m in segment_metrics.values())
                / max(len(segment_metrics), 1)
            ),
            # Over completions per entry branch, which is what highway_env measures
            # and what inverted_tree exists to study, not over instantaneous speeds.
            "fairness_jain": self._branch_fairness(),
            "active_vehicle_count": active_count,
            "active_av_count": active_av_count,
            # How long a completed trip took, over the rolling window. 0.0 when
            # nothing completed in it, which is a gap in the measurement rather
            # than a fast trip -- `throughput_recent` is what says which.
            "mean_travel_time_recent": float(sum(travel) / len(travel)) if travel else 0.0,
            # Travel time above the route's free-flow time, floored at zero. The
            # axis the predecessor paper's largest gain was on, and one this
            # environment did not measure at all before task 103.
            #
            # It is a mean over COMPLETED trips, so a controller that stops a
            # vehicle from completing improves it. `latent_demand` and
            # `throughput_recent` are what make that visible, and both are reported
            # beside it.
            "mean_delay_recent": float(sum(delay) / len(delay)) if delay else 0.0,
            # The share of vehicles on the network below `STOPPED_SPEED_MPS`.
            # Summed over the episode and multiplied by dt this is total waiting
            # time in vehicle-seconds, which is SUMO's own definition.
            "stopped_fraction": (
                float(sum(1 for v in speeds if v < self.STOPPED_SPEED_MPS) / len(speeds))
                if speeds else 0.0
            ),
            # Mean |change in acceleration| per second, over vehicles present in
            # both this step's snapshots and the previous step's. The smoothness
            # measure: `speed_std` is a spread across vehicles at one instant,
            # which a uniformly stop-and-go fleet leaves small.
            "mean_abs_jerk": jerk,
        }

    def density_ratios(self, snapshots: list[VehicleSnapshot] | None = None) -> dict[str, float]:
        """Each segment's density as a fraction of its jam density.

        The quantity the critical threshold of 0.3 is a fraction OF, and the one the
        threshold reward is priced on. `segment_metrics["density"]` counts vehicles
        per kilometre over the whole segment rather than per lane, so the
        denominator scales with the lane count: a two-lane segment at 130 veh/km is
        at half of jam density, not all of it.
        """
        from src.sensing.local import JAM_DENSITY_VEH_PER_KM_PER_LANE

        segments = self.get_segment_metrics(snapshots)
        lane_counts = dict(self.view.lane_counts) if self.view is not None else {}
        return {
            segment_id: float(metric.get("density", 0.0))
            / (JAM_DENSITY_VEH_PER_KM_PER_LANE * max(int(lane_counts.get(segment_id, 1)), 1))
            for segment_id, metric in segments.items()
        }

    def _mean_abs_jerk(self, snapshots: list[VehicleSnapshot]) -> float:
        """Mean |da/dt| over vehicles seen in both this step and the previous one.

        Rebinds the acceleration map rather than updating it, so a vehicle that was
        inside a junction last step -- and therefore absent from the snapshots --
        contributes nothing, instead of a difference taken across however many
        steps it was away.
        """
        dt = float(self.config.get("dt", 1.0)) or 1.0
        previous = self._last_accel
        changes = [
            abs(s.acceleration_mps2 - previous[s.vehicle_id]) / dt
            for s in snapshots if s.vehicle_id in previous
        ]
        self._previous_accel = previous
        self._last_accel = {s.vehicle_id: s.acceleration_mps2 for s in snapshots}
        return float(sum(changes) / len(changes)) if changes else 0.0

    def neighbourhood_metrics(
        self, snapshots: list[VehicleSnapshot] | None = None
    ) -> dict[str, dict[str, float]]:
        """Per agent, the state of its own segment and of the segments downstream.

        What a per-agent reward is priced from. The environment measures and the
        reward weights it, so the weights stay in the training config rather than
        being spread across the simulator.

        **The neighbourhood includes the segments downstream, not only the agent's
        own.** Speed metering means holding the own segment below free flow to
        relieve the next one, so an own-segment-only term would penalise exactly the
        behaviour this project exists to study. Immediate successors only, from the
        same relation the rolling-roadblock metric uses.

        Each field is the MEAN over the neighbourhood, so an agent with two
        successors is not rewarded more than one with a single successor for the
        same conditions.
        """
        snapshots = snapshots if snapshots is not None else self.vehicle_snapshots()
        segments = self.get_segment_metrics(snapshots)
        downstream = self.view.downstream_segments() if self.view is not None else {}
        segment_of = {s.vehicle_id: s.segment_id for s in snapshots}
        by_id = {s.vehicle_id: s for s in snapshots}
        dt = float(self.config.get("dt", 1.0)) or 1.0
        result: dict[str, dict[str, float]] = {}
        for agent_id in self.agent_ids:
            own = segment_of.get(agent_id)
            if own is None:
                continue
            names = [own, *downstream.get(own, ())]
            present = [segments[name] for name in names if name in segments]
            if not present:
                continue
            vehicle = by_id.get(agent_id)
            result[agent_id] = {
                # THE AGENT'S OWN VEHICLE, which responds to its own action within
                # one step and to nobody else's. The neighbourhood fields below are
                # what the agent's actions are FOR; these are what they immediately
                # are. Measured over 5,765 decisions, the advantage built from the
                # neighbourhood fields alone sits at its own shuffled noise floor
                # (ratio 0.784), so a term the action provably moves is what the
                # gradient was missing.
                #
                # True values, no sensing noise and no fidelity mask: these price a
                # reward during training and never reach the actor.
                "own_speed": float(vehicle.speed_mps) if vehicle else 0.0,
                "own_abs_jerk": (
                    abs(float(vehicle.acceleration_mps2)
                        - self._previous_accel.get(agent_id, float(vehicle.acceleration_mps2))) / dt
                    if vehicle else 0.0
                ),
                "own_stopped": (
                    1.0 if vehicle and vehicle.speed_mps < self.STOPPED_SPEED_MPS else 0.0
                ),
                "mean_speed": float(sum(float(m["mean_speed"]) for m in present) / len(present)),
                "jam_fraction": float(sum(float(m["jam_fraction"]) for m in present) / len(present)),
                "queue_length": float(sum(float(m["queue_length"]) for m in present) / len(present)),
                "outflow_recent": float(
                    sum(len(self._segment_outflow_times.get(name, ())) for name in names) / len(names)
                ),
            }
        return result

    def _track_trips(self, now: float) -> None:
        """Attribute each departure and arrival to its branch, and time the trip.

        A vehicle is gone by the time it arrives, so neither its route nor its
        departure time can be read then; both are recorded on departure and looked
        up on arrival. Fairness is Jain's index over completions per branch, which
        is the branch-fairness objective `inverted_tree` exists to study -- not, as
        an earlier version computed, Jain's index over instantaneous speeds.

        `now` is negative during warm-up, counting back to the first observed step,
        which is the convention `_arrivals` already uses: the rolling windows then
        start already filled rather than refilling from empty on a network that has
        reached a steady state.
        """
        for vehicle_id in _sumo.simulation.getDepartedIDList():
            try:
                route = _sumo.vehicle.getRoute(vehicle_id)
            except Exception:  # noqa: BLE001 - vanished between the two calls
                continue
            if route:
                self._origin_of[vehicle_id] = route[0]
                self._branch_spawned[route[0]] = self._branch_spawned.get(route[0], 0) + 1
            self._departure_time[vehicle_id] = now
        for vehicle_id in _sumo.simulation.getArrivedIDList():
            origin = self._origin_of.pop(vehicle_id, None)
            if origin is not None:
                self._branch_completed[origin] = self._branch_completed.get(origin, 0) + 1
            departed = self._departure_time.pop(vehicle_id, None)
            if departed is not None:
                self._completed_trips.append(
                    (now, now - departed, self._route_free_flow_s.get(origin or "", 0.0))
                )

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
        if self._cached_segment_metrics is not None:
            return self._cached_segment_metrics
        snapshots = snapshots if snapshots is not None else self.vehicle_snapshots()
        road = (self.config.get("topology") or {}).get("road", {})
        lengths = road.get("segment_lengths", {}) or {}
        segment_ids = sorted({s.segment_id for s in snapshots if s.segment_id} | set(lengths))
        lane_counts = dict(self.view.lane_counts) if self.view is not None else {}
        self._cached_segment_metrics = compute_segment_metrics(
            segment_ids=segment_ids,
            segment_lengths_m=lengths,
            lane_counts=lane_counts,
            active_vehicle_records=self._vehicle_records(snapshots),
            step_inflow=self._step_inflow,
            step_outflow=self._step_outflow,
            thresholds=metric_thresholds_from_config(self.config),
            # Lets the rolling-roadblock term tell metering from obstruction: AVs
            # holding a clear segment slow are excused when the next segment is
            # congested. Without it the term punishes the one mechanism the project
            # calls legitimate speed metering.
            downstream_segments=(self.view.downstream_segments() if self.view else None),
        )
        return self._cached_segment_metrics

    def _update_flows(self, snapshots: list[VehicleSnapshot], now: float) -> None:
        """Vehicles crossing each segment's boundaries during the step just taken.

        Called once per step, before the metric getters, because the flows are a
        difference between two steps and cannot be recovered from a single one. An
        earlier version computed them inside `get_segment_metrics` and advanced
        `_segment_of` there as a side effect; that getter is called two or three
        times per step and `get_global_state` is always last, so the critic and the
        logs read the flows of a comparison of the step against itself -- zero on
        every segment, on 2 of the 11 fields the critic receives per segment.

        A vehicle entering the network counts as inflow to its first segment, one
        crossing a boundary as outflow from the old segment and inflow to the new,
        and one that is no longer on any segment -- it arrived, or it is inside a
        junction -- as outflow from its last. `HighwayTopologyEnv` counts only the
        first and last of those, so its interior segments always report zero flow;
        this is a deliberate divergence, recorded in the migration plan, because the
        global state is read by the critic alone and is not part of the deployed
        contract.
        """
        current = {s.vehicle_id: s.segment_id for s in snapshots if s.segment_id}
        inflow: dict[str, int] = {}
        outflow: dict[str, int] = {}
        for vehicle_id, segment_id in current.items():
            previous = self._segment_of.get(vehicle_id)
            if previous == segment_id:
                continue
            inflow[segment_id] = inflow.get(segment_id, 0) + 1
            if previous is not None:
                outflow[previous] = outflow.get(previous, 0) + 1
        for vehicle_id, previous in self._segment_of.items():
            if vehicle_id not in current:
                outflow[previous] = outflow.get(previous, 0) + 1
        # Rebinding rather than updating also prunes vehicles that have left, which
        # otherwise accumulated: 120 entries against 58 live vehicles.
        self._segment_of = current
        self._step_inflow = inflow
        self._step_outflow = outflow
        # The per-segment analogue of `_arrivals`: when each segment last passed a
        # vehicle on. A local reward needs a throughput term, and a single step's
        # outflow is 0 on almost every step at any realistic flow -- 1300 veh/h is
        # 0.036 vehicles per step at dt 0.1 -- so the count is accumulated over the
        # same rolling window `throughput_recent` uses and pruned in `_build_metrics`.
        for segment_id, count in outflow.items():
            self._segment_outflow_times.setdefault(segment_id, []).extend([now] * count)

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
        _, census_count, census_av_count = self._network_census()
        demand = self.config.get("demand", {})
        return {
            "time": self.step_count * float(self.config.get("dt", 1.0)),
            "topology_id": self.topology_id,
            "active_vehicle_count": census_count,
            "active_av_count": census_av_count,
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
        facts = self._edge_facts().get(edge_id)
        return (facts[0], facts[1]) if facts else (edge_id, edge_id)

    def _edge_facts(self) -> dict[str, tuple[str, str, float, float]]:
        """`(from node, to node, length in m, speed limit in m/s)` per edge.

        Read once per episode from the built network rather than from the topology
        config, so it describes the road the vehicles actually drive on. Cleared by
        `reset`, which rebuilds that road.
        """
        if getattr(self, "_edge_node_cache", None) is None:
            import sumolib

            net = sumolib.net.readNet(str(self.network.net_file))
            self._edge_node_cache = {
                edge.getID(): (
                    edge.getFromNode().getID(),
                    edge.getToNode().getID(),
                    float(edge.getLength()),
                    float(edge.getSpeed()),
                )
                for edge in net.getEdges()
                if not edge.getID().startswith(":")
            }
        return self._edge_node_cache

    def _route_free_flow_times(self) -> dict[str, float]:
        """How long each route takes at the speed limit, keyed by its entry edge.

        The subtrahend in delay: a trip that took 180 s on a route whose free-flow
        time is 95 s was delayed 85 s. Computed from edge lengths and limits rather
        than from a measured uncongested run, so it is a property of the road and
        does not move with the demand.

        It uses the LIMIT and not the vehicle's own desired speed, which is drawn
        from the demand's speed distribution and is lower on average (a 24 m/s mean
        against a 30 m/s limit). Delay measured this way therefore includes the
        part of the difference that comes from a driver choosing to go slower than
        the limit, which is constant across arms and does not affect a comparison
        between them.
        """
        if self.network is None:
            return {}
        facts = self._edge_facts()
        times: dict[str, float] = {}
        for entry in self.network.entry_edges():
            route = self.network.route_to_exit(entry)
            total = 0.0
            for edge_id in route or ():
                edge = facts.get(edge_id)
                if edge is None or edge[3] <= 0.0:
                    continue
                total += edge[2] / edge[3]
            times[entry] = total
        return times

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

    #: The vType attribute each `human_model.sumo` key becomes. Named rather than
    #: derived so an unknown key is a caller error instead of silently ignored.
    #: Config key to vType attribute. An allowlist, so a key a config sets and this
    #: map lacks is refused rather than silently dropped -- but it also means a new
    #: model's parameters have to be added here before its config can be used at all.
    _FOLLOWING_ATTRIBUTES = {
        "car_following_model": "carFollowModel",
        "min_gap_m": "minGap",
        "cc1": "cc1", "cc2": "cc2", "cc3": "cc3", "cc4": "cc4", "cc5": "cc5",
        "cc6": "cc6", "cc7": "cc7", "cc8": "cc8", "cc9": "cc9",
        # EIDM's, which `eidm_reaction` sets. actionStepLength is the driver's
        # reaction time and is what gives that fleet a capacity drop.
        "tau": "tau", "action_step_length": "actionStepLength",
        "actionStepLength": "actionStepLength", "sigma": "sigma",
        "startup_delay_s": "startupDelay",
    }

    def _car_following_attributes(self) -> str:
        """The human model's car-following parameters, as vType attributes.

        Absent, every vehicle uses SUMO's default Krauss model, which computes a
        collision-free safe speed exactly and recovers from a disturbance
        immediately. That produces no capacity drop -- served flow rose
        monotonically from 890 to 1110 veh/h as demand went 900 to 2400 -- and with
        no capacity drop there is nothing for a controller to recover, which a
        perfect-information metering oracle confirmed by failing to beat inaction.

        `configs/human_models/w99_calibrated.yaml` carries the Wiedemann-99
        parameters the predecessor paper calibrated for exactly this reason. That
        calibration turned out not to have a capacity drop either on SUMO: its queues
        discharge FASTER than free-flowing traffic at capacity, 1,980 veh/h against
        1,899. `eidm_reaction` is the fleet that has one, and its header carries the
        measurements.
        """
        model = (self.config.get("human_model") or {}).get("sumo") or {}
        unknown = set(model) - set(self._FOLLOWING_ATTRIBUTES)
        if unknown:
            raise ValueError(
                f"unsupported human_model.sumo keys {sorted(unknown)}; "
                f"known keys are {sorted(self._FOLLOWING_ATTRIBUTES)}"
            )
        parts = [f' {self._FOLLOWING_ATTRIBUTES[key]}="{model[key]}"'
                 for key in self._FOLLOWING_ATTRIBUTES if key in model]
        return "".join(parts)

    def _demand_periods(self, duration_s: float, per_entry: float
                        ) -> list[tuple[float, float, float]]:
        """The demand rate per entry as (begin, end, vehicles per hour) periods.

        A steady demand is one period. A burst is three: the base rate, the raised
        rate, and the base rate again -- which is the shape
        `DemandProfile.vehicles_per_hour_at` already describes on the other
        simulator, where the multiplier applies inside [start_s, end_s] and 1.0
        outside it. The config was accepted here and ignored, so a demand declaring
        a burst produced a flat one.

        The config's times are EPISODE times, because that is what they mean on the
        other simulator, which has no warm-up. They are offset by the warm-up here
        so a burst declared at t = 200 s lands 200 s into the observed episode
        rather than during the fill.
        """
        burst = (self.config.get("demand", {}) or {}).get("burst", {}) or {}
        if not bool(burst.get("enabled", False)):
            return [(0.0, duration_s, per_entry)]
        multiplier = float(burst.get("multiplier", 1.0))
        offset = int(self.config.get("warmup_steps", 0)) * float(self.config.get("dt", 1.0))
        start = max(0.0, offset + float(burst.get("start_s", 0.0)))
        end = min(duration_s, offset + float(burst.get("end_s", 0.0)))
        if end <= start:
            return [(0.0, duration_s, per_entry)]
        periods = []
        if start > 0.0:
            periods.append((0.0, start, per_entry))
        periods.append((start, end, per_entry * multiplier))
        if end < duration_s:
            periods.append((end, duration_s, per_entry))
        return periods

    def _entry_weights(self, entries: list[str]) -> dict[str, float]:
        """Each entry edge's share of the total demand, normalised to sum to 1.

        Uniform unless `demand.branch_split` names entries with positive weights.
        The keys are entry edge ids; a name that matches no entry is ignored rather
        than raising, because `branch_split` also carries the other simulator's
        branch ids and a shared demand config has to remain loadable on both.
        """
        split = (self.config.get("demand", {}) or {}).get("branch_split", {}) or {}
        weights = {edge: float(split.get(edge, 0.0)) for edge in entries}
        total = sum(value for value in weights.values() if value > 0.0)
        if total <= 0.0:
            return {edge: 1.0 / max(len(entries), 1) for edge in entries}
        return {edge: max(0.0, weights[edge]) / total for edge in entries}

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
        # `branch_split` reached this writer NOWHERE: the rate was split evenly over
        # the entries and the field, declared in every demand config, was consumed
        # only by `src/demand/spawner.py` on the other simulator. So every SUMO run
        # ever made loaded all six branches identically, and any control law whose
        # mechanism is correcting an imbalance had nothing to correct.
        #
        # Weights are per entry edge, by name; an entry the split does not mention
        # keeps an equal share. An empty or all-zero split is uniform, which is what
        # every existing config resolves to, so no recorded run changes.
        weights = self._entry_weights(entries)
        per_entry_rates = {edge: total_per_hour * weight for edge, weight in weights.items()}
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
        following = self._car_following_attributes()
        # The demand config states its speed distribution in m/s; SUMO states a
        # vehicle's desired speed as a factor on the lane's limit. Every edge this
        # builder writes carries the topology's single `speed_limit_mps`, so the
        # configured distribution maps onto a factor exactly, and a config declaring
        # a 24 m/s mean gets a 24 m/s mean rather than the 30 m/s the limit implies.
        # The previous form hardcoded normc(1,0.1,0.8,1.2), so `mean_mps`, `std_mps`
        # and `min_mps` reached SUMO nowhere and the measured operating point
        # belonged to an undeclared fleet.
        limit = float((self.config.get("topology") or {}).get("road", {})
                      .get("speed_limit_mps", 30.0))
        speed_factor = "normc({:.4f},{:.4f},{:.4f},{:.4f})".format(
            float(speed.get("mean_mps", 24.0)) / limit,
            float(speed.get("std_mps", 2.5)) / limit,
            float(speed.get("min_mps", 12.0)) / limit,
            max_speed / limit,
        )
        # `departSpeed="desired"` enters at the vehicle's own desired speed, which is
        # what the other simulator's spawner does: it draws one speed and uses it as
        # both the entry speed and the cruise target.
        #
        # `spawn_min_gap_m` is deliberately NOT mapped. On `highway_env` it gates
        # insertion -- a lane is eligible only if no vehicle sits within that
        # distance -- and SUMO enforces insertion feasibility itself through the
        # car-following model, which is the stronger criterion. Mapping it to a vType
        # `minGap` would instead change the standstill gap, and so jam density, from
        # a field that on the other simulator changes no physics at all.
        lines = [
            "<routes>",
            f'  <vType id="{self.HUMAN_TYPE}" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}"{following}/>',
            # The AV keeps SUMO's safe car-following as a floor; the project's own
            # controller commands speed on top of it through setSpeed.
            f'  <vType id="{self.AV_TYPE}" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" color="1,0,0"{following}/>',
            # ONE distribution, drawn per vehicle, rather than one flow per type.
            # SUMO spaces each flow evenly on its own, so two flows at rates r1 and
            # r2 do not produce the same arrival process as one flow at r1+r2 --
            # which made the arrival pattern depend on penetration and moved mean
            # speed with no controller acting.
            f'  <vTypeDistribution id="{self.MIX_TYPE}">',
            f'    <vType id="{self.HUMAN_TYPE}_d" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" probability="{1.0 - penetration:.4f}"'
            f'{following}/>',
            f'    <vType id="{self.AV_TYPE}_d" maxSpeed="{max_speed}"'
            f' speedFactor="{speed_factor}" color="1,0,0" probability="{penetration:.4f}"'
            f'{following}/>',
            "  </vTypeDistribution>",
        ]
        # An explicit departure schedule rather than `<flow>`, because a flow cannot
        # express a demand that changes over time without being split, and splitting
        # it loses vehicles: three sequential flows carrying the same total rate as
        # one delivered 71 departures against 150 over the same episode. Every
        # vehicle is written with its own departure time, so the schedule is exactly
        # what the demand profile says.
        #
        # This is a departure SCHEDULE, not a spawner. SUMO still decides whether a
        # vehicle can enter at its time and holds it until it can, which is the
        # property that made flows worth using in the first place.
        departures: list[tuple[float, int, str]] = []
        for index, edge in enumerate(entries):
            route = self.network.route_to_exit(edge)
            if not route:
                continue
            edges = " ".join(route)
            lines.append(f'  <route id="r{index}" edges="{edges}"/>')
            for begin, end, rate in self._demand_periods(
                    duration_s, per_entry_rates.get(edge, per_entry)):
                if rate <= 0.0:
                    continue
                headway = 3600.0 / rate
                # Offset by half a headway so the first vehicle of a period does not
                # depart at the same instant as the last of the previous one.
                time = begin + headway / 2.0
                while time < end:
                    departures.append((time, index, f"v{index}_{len(departures)}"))
                    time += headway
        # SUMO requires a route file sorted by departure time.
        for time, index, vehicle_id in sorted(departures):
            lines.append(
                f'  <vehicle id="{vehicle_id}" type="{self.MIX_TYPE}"'
                f' route="r{index}" depart="{time:.3f}" departSpeed="desired"/>'
            )
        lines.append("</routes>")
        route_file = self.work_dir / "demand.rou.xml"
        route_file.write_text("\n".join(lines) + "\n")
        return route_file
