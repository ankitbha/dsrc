"""How a driver turns the shown advisory into the vehicle's speed cap (plan 2026-10-25 §4).

Every other result applies the shown speed at once and exactly: `setMaxSpeed` to `shown x
v_limit` on the tick it is written. `DriverResponse` sits between the write path and SUMO
instead. Per controlled vehicle it draws a reaction time, after which a change of the shown
speed reaches the cap, and a compliance offset, by which the driver aims above (or below) the
advice. A lower cap is approached at a bounded deceleration, one step per 1 s tick; a higher
one applies at once, since car-following already bounds acceleration. Car-following still
brakes for the leader whatever the cap says.

A driver's traits come from `default_rng((seed + DRIVER_SEED_OFFSET, crc32(vehicle id)))`,
so they depend on neither the arm nor the order in which vehicles appear, and two arms run on
the same seed meet the same drivers.

The parameters are assumed (tier 3): the deployment recorded no driver acting on an advisory.
"""
from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DRIVER_SEED_OFFSET: int = 7000
COMPLIANCE_BOUNDS: tuple[float, float] = (-0.1, 0.5)


@dataclass(frozen=True)
class DriverSetting:
    name: str
    reaction_median_s: float
    reaction_sigma_log: float
    comply_mean: float
    comply_sd: float
    decel_mps2: float

    def describe(self) -> dict[str, Any]:
        return asdict(self)


DRIVERS: dict[str, DriverSetting] = {
    "nominal": DriverSetting("nominal", 2.0, 0.4, 0.05, 0.05, 1.5),
    "lax": DriverSetting("lax", 4.0, 0.5, 0.15, 0.10, 1.0),
}


def traits(setting: DriverSetting, seed: int, vehicle: str) -> tuple[float, float]:
    """`(reaction time in s, compliance offset)` of one driver."""
    rng = np.random.default_rng((seed + DRIVER_SEED_OFFSET, zlib.crc32(vehicle.encode())))
    reaction = float(setting.reaction_median_s * np.exp(setting.reaction_sigma_log
                                                        * rng.standard_normal()))
    offset = float(np.clip(setting.comply_mean + setting.comply_sd * rng.standard_normal(),
                           *COMPLIANCE_BOUNDS))
    return reaction, offset


class DriverResponse:
    """Pending changes and caps per vehicle; `apply(vehicle, mps)` writes a cap to SUMO and
    `speed(vehicle)` reads a vehicle's current speed (only when a first cap starts a ramp)."""

    def __init__(self, setting: DriverSetting, seed: int,
                 apply: Callable[[str, float], None], speed: Callable[[str], float]) -> None:
        self.setting = setting
        self.seed = seed
        self._apply = apply
        self._speed = speed
        self._traits: dict[str, tuple[float, float]] = {}
        self._advised: dict[str, float] = {}
        self._pending: dict[str, tuple[float, float]] = {}
        self._goal: dict[str, float] = {}
        self.cap: dict[str, float] = {}
        self.changes = 0
        self.applied = 0

    def traits_of(self, vehicle: str) -> tuple[float, float]:
        if vehicle not in self._traits:
            self._traits[vehicle] = traits(self.setting, self.seed, vehicle)
        return self._traits[vehicle]

    def submit(self, t: float, vehicle: str, advised_mps: float) -> None:
        """The display shows `advised_mps` from `t`; an unchanged advice is not a change."""
        previous = self._advised.get(vehicle)
        if previous is not None and abs(previous - advised_mps) < 1e-9:
            return
        self._advised[vehicle] = advised_mps
        reaction, offset = self.traits_of(vehicle)
        self._pending[vehicle] = (t + reaction, advised_mps * (1.0 + offset))
        self.changes += 1

    def advance(self, t: float, present: set[str], dt: float = 1.0) -> dict[str, float]:
        """Start the changes that are due and move every ramp one tick; returns the caps
        written this tick. Vehicles not in `present` have left and are forgotten."""
        for store in (self._pending, self._goal, self.cap, self._advised):
            for vehicle in [v for v in store if v not in present]:
                del store[vehicle]
        for vehicle, (due, goal) in list(self._pending.items()):
            if due <= t + 1e-9:
                del self._pending[vehicle]
                self._goal[vehicle] = goal
                if vehicle not in self.cap:
                    self.cap[vehicle] = max(goal, float(self._speed(vehicle)))
        written: dict[str, float] = {}
        step = self.setting.decel_mps2 * dt
        for vehicle, goal in list(self._goal.items()):
            current = self.cap[vehicle]
            new = goal if goal >= current else max(goal, current - step)
            self._apply(vehicle, new)
            written[vehicle] = new
            self.cap[vehicle] = new
            if abs(new - goal) < 1e-9:
                del self._goal[vehicle]
        self.applied += len(written)
        return written

    def describe(self) -> dict[str, Any]:
        return {**self.setting.describe(), "seed_offset": DRIVER_SEED_OFFSET,
                "compliance_bounds": list(COMPLIANCE_BOUNDS)}
