"""The AV vehicle, held to the same physical limit as the human one.

Both vehicle kinds mix in `CollisionFreeMixin`, so both obey the same bound while
perceiving independently. That separation is the point: the AV's safety is a
consequence of what the AV can see, so a badly-sensing AV is measurably less safe.
Enforcing the limit centrally would have made every vehicle equally safe regardless
of its sensing, which would erase the property this project exists to measure.

The safety layer still runs and still produces the AV's command. This is the floor
underneath it, not a replacement: a controller that asks for more speed than the
vehicle could stop from does not get it.
"""
from __future__ import annotations

from src.road.highway_imports import ensure_highway_env_importable

ensure_highway_env_importable()

from highway_env.vehicle.controller import ControlledVehicle  # noqa: E402

from src.vehicles.safe_following import CollisionFreeMixin  # noqa: E402


class SafeControlledVehicle(CollisionFreeMixin, ControlledVehicle):
    """A `ControlledVehicle` that will not exceed its own safe speed."""
