"""Collision-free car following, enforced by each vehicle on itself.

PTV Vissim and SUMO both make collisions structurally impossible inside the
car-following model rather than penalising them afterwards. This simulator did not:
at 120 steps the AV arms completed 30 of 54, so per-step survival was 0.995114, the
expected time to a first AV crash was 205 steps -- about 3.4 minutes -- and the
chance of an hour-long run reaching its duration was 2.2e-8. An hour of simulated
driving was therefore unmeasurable.

**Each vehicle enforces the bound on itself, from what it can perceive.** That is a
deliberate choice over a single check in the environment's step loop, which would
have been easier to write and easier to test. A referee that reaches into every
vehicle and caps it is not a model of driver behaviour; it is a constraint imposed
on the outside. Keeping it in the vehicle means the guarantee is exactly as good as
the vehicle's perception, which is the property under study in this project: an AV
that senses badly should be less safe, and with a global referee it would not be.

The bound is Krauss's: from `safe_speed(gap, v_lead)`, a vehicle that reacts within
`tau` and then decelerates at `b` stops within the gap plus the leader's own
stopping distance, whatever the leader does.
"""
from __future__ import annotations

import math
from typing import Any

#: Deceleration a vehicle is assumed able to produce. Matches `IDMVehicle.ACC_MAX`
#: and the safety layer's emergency rate, so the bound never assumes braking no
#: vehicle can deliver.
SAFE_DECEL_MPS2 = 6.0

#: Driver reaction time. A property of the driver, not of the integrator, which is
#: why the bound is expressed with it rather than with the simulation step.
SAFE_REACTION_S = 0.5

#: Lateral separation within which two vehicles are treated as sharing space. A
#: vehicle is about 2 m wide and a lane about 3.7 m, so this catches the same-lane
#: case, an adjacent-lane overlap, and two arcs converging to within a metre -- the
#: three classes measured on inverted_tree, 18, 6 and 27 of 51 collisions.
SAFE_LATERAL_M = 2.6

#: Bumper-to-bumper allowance subtracted from a centre-to-centre distance.
SAFE_VEHICLE_LENGTH_M = 5.0

#: How far ahead a vehicle looks. Finite because perception is finite; a vehicle
#: that could see arbitrarily far would make the guarantee a property of the
#: simulator rather than of the driver.
SAFE_PERCEPTION_M = 200.0


def safe_speed(gap_m: float, lead_speed_mps: float, *,
               decel_mps2: float = SAFE_DECEL_MPS2,
               reaction_s: float = SAFE_REACTION_S) -> float:
    """The fastest a vehicle may travel and still avoid the one ahead."""
    b = float(decel_mps2)
    tau = float(reaction_s)
    gap = max(float(gap_m), 0.0)
    lead = max(float(lead_speed_mps), 0.0)
    return float(-b * tau + math.sqrt((b * tau) ** 2 + lead * lead + 2.0 * b * gap))


class CollisionFreeMixin:
    """Refuses to exceed the speed at which this vehicle could still stop.

    Mixed into both the human and the AV vehicle so the two obey the same physical
    limit while perceiving independently. The clamp is applied in `step`, because
    that is where the vehicle learns `dt` and so the only place it can convert a
    speed limit into an acceleration limit without being told the integrator's
    settings.
    """

    SAFE_DECEL_MPS2 = SAFE_DECEL_MPS2
    SAFE_REACTION_S = SAFE_REACTION_S
    SAFE_LATERAL_M = SAFE_LATERAL_M
    SAFE_PERCEPTION_M = SAFE_PERCEPTION_M

    def perceived_safe_speed(self) -> float:
        """The tightest limit any vehicle this one can see imposes on it.

        Geometric rather than lane-based, which is what makes it cover all three
        measured collision classes. A lane-based leader search sees only the
        same-lane case, 18 of 51.
        """
        road = getattr(self, "road", None)
        if road is None:
            return float("inf")
        hx = math.cos(float(self.heading))
        hy = math.sin(float(self.heading))
        x, y = float(self.position[0]), float(self.position[1])
        limit = float("inf")
        for other in road.vehicles:
            if other is self:
                continue
            dx = float(other.position[0]) - x
            dy = float(other.position[1]) - y
            longitudinal = dx * hx + dy * hy
            if longitudinal <= 0.0 or longitudinal > self.SAFE_PERCEPTION_M:
                continue
            if abs(-dx * hy + dy * hx) >= self.SAFE_LATERAL_M:
                continue
            gap = longitudinal - SAFE_VEHICLE_LENGTH_M
            candidate = safe_speed(gap, float(other.speed),
                                   decel_mps2=self.SAFE_DECEL_MPS2,
                                   reaction_s=self.SAFE_REACTION_S)
            if candidate < limit:
                limit = candidate
        # Where a vehicle is going, not only where it is. The loop above sees only
        # what is ahead and within a lane's width; this catches a vehicle alongside
        # whose path crosses the ego's, which is what the measured collisions were.
        for other in road.vehicles:
            if other is self or getattr(other, "crashed", False):
                continue
            candidate = self._converging_limit(other)
            if candidate is not None and candidate < limit:
                limit = candidate
        return limit

    #: How far ahead a predicted conflict is acted on. Beyond it the prediction is
    #: not worth braking for, and acting on it would slow traffic for encounters
    #: that never happen.
    CPA_HORIZON_S = 4.0

    #: Predicted closest approach below which two vehicles are treated as
    #: colliding. A vehicle is about 2 m wide, so this leaves a small margin.
    CPA_MISS_M = 2.6

    def _converging_limit(self, other: Any) -> float | None:
        """Speed limit from a predicted closest approach, or None if the paths miss.

        The forward window in `perceived_safe_speed` asks where a vehicle IS. This
        asks where it is GOING, which is the question that distinguishes a real
        conflict from parallel traffic. Two vehicles 3 m apart in adjacent lanes at
        the same speed never meet and must not constrain each other; two vehicles
        3 m apart whose paths converge will collide, and the first test cannot see
        the difference. Widening the window instead would brake for both.

        Both velocities are projected as straight lines. That is wrong on a curve,
        but only over the horizon, which is short.
        """
        rx = float(other.position[0]) - float(self.position[0])
        ry = float(other.position[1]) - float(self.position[1])
        ego_heading = float(self.heading)
        other_heading = float(getattr(other, "heading", ego_heading))
        vx = float(other.speed) * math.cos(other_heading) - float(self.speed) * math.cos(ego_heading)
        vy = float(other.speed) * math.sin(other_heading) - float(self.speed) * math.sin(ego_heading)
        closing_rate = vx * vx + vy * vy
        if closing_rate < 1e-6:
            return None  # identical velocities: the separation never changes
        approach = rx * vx + ry * vy
        if approach >= 0.0:
            return None  # already separating
        time_to_closest = -approach / closing_rate
        if time_to_closest > self.CPA_HORIZON_S:
            return None
        miss = math.hypot(rx + vx * time_to_closest, ry + vy * time_to_closest)
        if miss >= self.CPA_MISS_M:
            return None
        # The paths meet, so the ego must be able to stop within the separation it
        # has now. The conflict point is treated as stationary because the ego
        # cannot rely on the other vehicle yielding.
        gap = max(0.0, math.hypot(rx, ry) - SAFE_VEHICLE_LENGTH_M)
        return safe_speed(gap, 0.0, decel_mps2=self.SAFE_DECEL_MPS2,
                          reaction_s=self.SAFE_REACTION_S)

    def step(self, dt: float) -> None:  # type: ignore[override]
        limit = self.perceived_safe_speed()
        if math.isfinite(limit) and isinstance(getattr(self, "action", None), dict):
            current = self.action.get("acceleration")
            if current is not None:
                permitted = (limit - float(self.speed)) / max(float(dt), 1e-9)
                if float(current) > permitted:
                    self.action["acceleration"] = float(permitted)
        super().step(dt)  # type: ignore[misc]
