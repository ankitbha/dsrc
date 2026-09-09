"""A human driver that can see traffic converging from another lane.

`highway_env`'s `IDMVehicle` follows the vehicle ahead **in its own lane**. Every
node in the inverted-tree topology reduces lane count -- `b1` 3 into 2, `b2` 3 into
2, `c` 4 into 2 -- so two vehicles arriving at a node on different incoming lanes
have no mutual awareness whatsoever and drive through each other. Measured before
this class existed: a `no_av` run, which contains nothing but IDM humans, carried a
median of 17 collisions per 120-step run, and 30 of 36 such runs had at least one.

The replication target (Flow, and Vinitsky et al.) runs on SUMO, whose car-following
is collision-free by construction. Human traffic that drives through itself is not a
property of the road being modelled, it is an artefact of the vehicle model, and it
made `episodes_complete` unusable as a criterion.

**The mechanism is projection onto the shared node.** Two vehicles heading for the
same node are ordered by their distance to it: the nearer one arrives first, so from
the other's point of view it is a leader at a following distance of
`d_ego - d_other`. Expressing it that way lets ordinary IDM car-following do the
work, rather than inventing a second control law that would then have to agree with
the first.

The resulting acceleration is combined with the in-lane one by taking the **more
restrictive**, which is the usual way to compose independent constraints and cannot
make the vehicle drive more aggressively than it would have.
"""
from __future__ import annotations

from typing import Any

from src.road.highway_imports import ensure_highway_env_importable

ensure_highway_env_importable()

from highway_env.vehicle.behavior import IDMVehicle  # noqa: E402
from highway_env.vehicle.kinematics import Vehicle  # noqa: E402


class MergeAwareIDMVehicle(IDMVehicle):
    """An `IDMVehicle` that also yields to traffic converging on the same node."""

    #: Ignore a conflict further ahead than this. Beyond it the projected leader is
    #: not yet relevant and including it makes vehicles crawl along empty arcs.
    MERGE_HORIZON_M = 150.0

    #: The horizon over which a yield may bring the vehicle to rest. One second
    #: matches the simulator's decision step, so the strongest permitted brake
    #: reaches exactly zero rather than passing through it.
    MERGE_STOP_TAU_S = 1.0

    def act(self, action: Any = None) -> None:
        super().act(action)
        if not isinstance(self.action, dict):
            return
        merge_acceleration = self._merge_acceleration()
        current = self.action.get("acceleration")
        if merge_acceleration is None:
            combined = current
        elif current is None:
            combined = merge_acceleration
        else:
            # The more restrictive of the two constraints wins.
            combined = min(float(current), merge_acceleration)
        if combined is None:
            return
        # The floor is applied on every call, not only when a merge constrains the
        # vehicle. Applying it only on the merge branch left vehicles reversing at
        # -0.87 m/s: yielding puts them into near-contact states plain IDM never
        # reaches, and IDM's own gap term then brakes hard at walking pace on a
        # later step when no conflict is in view. Whichever constraint wins, and
        # whether or not one exists, a vehicle must not be told to drive backwards.
        self.action["acceleration"] = max(float(combined), self._stopping_floor())

    def _stopping_floor(self) -> float:
        """The steepest deceleration that cannot carry the speed below zero."""
        return -max(float(self.speed), 0.0) / self.MERGE_STOP_TAU_S

    def _successor(self, lane_index):
        """The lane a vehicle on `lane_index` drives onto next, or None."""
        try:
            lane = self.road.network.get_lane(lane_index)
            return self.road.network.next_lane(
                lane_index, route=None, position=lane.position(lane.length, 0)
            )
        except Exception:  # noqa: BLE001 - an exit lane has no successor
            return None

    def _distance_to_node(self, vehicle: Vehicle) -> float | None:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None or self.road is None:
            return None
        try:
            lane = self.road.network.get_lane(lane_index)
            longitudinal, _ = lane.local_coordinates(vehicle.position)
        except Exception:  # noqa: BLE001 - a vehicle mid-transition has no usable lane
            return None
        return float(lane.length) - float(longitudinal)

    def _merge_acceleration(self) -> float | None:
        """IDM acceleration with respect to the nearest converging vehicle, if any."""
        if self.road is None or self.lane_index is None:
            return None
        node = self.lane_index[1]
        own_distance = self._distance_to_node(self)
        if own_distance is None:
            return None

        own_successor = self._successor(self.lane_index)
        nearest_gap = float("inf")
        nearest: Vehicle | None = None
        for other in self.road.vehicles:
            if other is self:
                continue
            # A wreck is decelerated to a stop and then sits at the node for the
            # rest of the run. 21% of phantom leaders in one measured run were
            # already-crashed vehicles, and everything upstream yielded to them
            # indefinitely.
            if getattr(other, "crashed", False):
                continue
            other_lane = getattr(other, "lane_index", None)
            if other_lane is None or other_lane[1] != node:
                continue
            if other_lane[0] == self.lane_index[0]:
                continue  # same arc: ordinary in-lane following already covers it
            # Sharing a node is not the same as converging. At node c, b1->c lane 0
            # and b2->c lane 0 both feed c->exit lane 0 and really do meet; the
            # cross pairs feed different exit lanes and never can. 48% of the
            # conflicts computed at that node were of the second kind.
            if own_successor is not None and self._successor(other_lane) != own_successor:
                continue
            other_distance = self._distance_to_node(other)
            if other_distance is None:
                continue
            gap = own_distance - other_distance
            # A vehicle that arrives behind is not a leader; braking for it would
            # invent an obstacle and deadlock the merge, since it is symmetric.
            if gap <= 0 or gap > self.MERGE_HORIZON_M:
                continue
            if gap < nearest_gap:
                nearest_gap = gap
                nearest = other
        if nearest is None:
            return None

        phantom = self._phantom_leader(nearest, nearest_gap)
        if phantom is None:
            return None
        acceleration = float(self.acceleration(ego_vehicle=self, front_vehicle=phantom))
        # Two separate floors, for two separate failures.
        #
        # IDM's gap term is `(desired_gap / d)^2`, which diverges as d approaches
        # zero. Real vehicles never get that close because a collision is detected
        # first, but a PROJECTED leader can sit a centimetre ahead, and without a
        # bound a no_av run reached a mean_speed of -2.6e11 m/s.
        #
        # A magnitude bound alone is not enough, because it bounds acceleration and
        # the defect is in speed. IDM's own braking fades out as a vehicle stops --
        # its desired gap scales with the ego's speed -- but a projected gap is a
        # difference of distances-to-node and does not, so the full ACC_MAX could be
        # commanded at a standstill. Vehicles then reversed at up to -7.5 m/s, which
        # the mean over forty vehicles hid completely. Stopping within
        # MERGE_STOP_TAU_S is the strongest brake that cannot reverse the vehicle.
        return max(acceleration, -abs(self.ACC_MAX), self._stopping_floor())

    def _phantom_leader(self, other: Vehicle, gap: float) -> Vehicle | None:
        """The converging vehicle, restated as a leader on the ego's own lane.

        `IDMVehicle.acceleration` measures the gap with `lane_distance_to`, which
        projects the leader's position onto the ego's lane. Placing a stand-in at
        `own longitudinal + gap` therefore makes IDM see exactly the projected
        following distance, with no change to IDM itself.
        """
        try:
            lane = self.road.network.get_lane(self.lane_index)
            longitudinal, _ = lane.local_coordinates(self.position)
            position = lane.position(longitudinal + gap, 0)
        except Exception:  # noqa: BLE001
            return None
        phantom = Vehicle(self.road, position, heading=self.heading, speed=float(other.speed))
        phantom.lane_index = self.lane_index
        return phantom
