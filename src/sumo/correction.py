"""The local corrections on top of the shared feed plan, and the feed lookahead rule.

Plan 2026-10-16 section 5. Every correction chooses, per vehicle and 5 s tick, either the
plan's fraction or something lower; what reaches the vehicle is
`shown_fraction(plan, correction) = min(plan, correction)`, so no correction can raise a
vehicle above the plan. The rule (arm 5), the learned predictor (arm 6) and the oracle
(arm 7, the rule fed true occupancy) step one action down from the plan; take-the-slower
(arm 4) re-runs the plan's Q-network with the camera's leader speed in the vehicle's own
row and may fall two steps. All four share `Hysteresis`: once active, at least 30 s on,
and released only after the release condition holds on 2 consecutive ticks.

The lookahead (arm 2) is not a correction: it reads the same feed versions as the plan,
replaces the plan, and never runs a hysteresis.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.sumo.mainz import DENSITY_CRITICAL, SPEED_ACTION_FRACTIONS, jam_factor

FRACTIONS: tuple[float, ...] = SPEED_ACTION_FRACTIONS
_FRACTION_ARRAY = np.asarray(FRACTIONS, dtype=float)
#: F4's horizon for the rule: the 5 s tick, about 1 s of perception, about 4 s to respond
#: and slow by 15 km/h at 1.5 m/s^2 (plan section 5.4). Fixed, not tuned.
TAU_L_S: float = 10.0
RELEASE_OFFSET: float = 0.05
LEARNED_RELEASE_OFFSET: float = 0.1
MIN_ON_S: float = 30.0
RELEASE_TICKS: int = 2


def step_down(fraction):
    """The next lower action fraction; 0.5 stays 0.5."""
    values = np.asarray(fraction, dtype=float)
    index = np.abs(values[..., None] - _FRACTION_ARRAY).argmin(axis=-1)
    lowered = _FRACTION_ARRAY[np.maximum(index - 1, 0)]
    return float(lowered) if np.ndim(fraction) == 0 else lowered


def shown_fraction(plan, correction):
    """What a corrected vehicle is shown: never above the plan (plan J6)."""
    if np.ndim(plan) == 0 and np.ndim(correction) == 0:
        return min(float(plan), float(correction))
    return np.minimum(np.asarray(plan, dtype=float), np.asarray(correction, dtype=float))


class Hysteresis:
    """Per vehicle: on at a trigger, on for at least `min_on_s`, off only once the release
    condition has held on `release_ticks` consecutive ticks (the current one included)."""

    def __init__(self, min_on_s: float = MIN_ON_S, release_ticks: int = RELEASE_TICKS) -> None:
        self.min_on_s = min_on_s
        self.release_ticks = release_ticks
        self._since: dict[str, float] = {}
        self._streak: dict[str, int] = {}

    def is_active(self, vehicle: str) -> bool:
        return vehicle in self._since

    def update(self, ids: Sequence[str], t: float, trigger: np.ndarray,
               release: np.ndarray) -> np.ndarray:
        active = np.zeros(len(ids), dtype=bool)
        for k, vehicle in enumerate(ids):
            since = self._since.get(vehicle)
            if since is None:
                if trigger[k]:
                    self._since[vehicle] = t
                    self._streak[vehicle] = 0
                    active[k] = True
                continue
            streak = self._streak[vehicle] + 1 if release[k] else 0
            if streak >= self.release_ticks and t - since >= self.min_on_s - 1e-9:
                del self._since[vehicle]
                del self._streak[vehicle]
                continue
            self._streak[vehicle] = streak
            active[k] = True
        return active

    def describe(self) -> dict[str, Any]:
        return {"min_on_s": self.min_on_s, "release_ticks": self.release_ticks}


def rule_margin(k_hat: np.ndarray, s_hat: np.ndarray, k_c: float = DENSITY_CRITICAL,
                tau_l: float = TAU_L_S) -> np.ndarray:
    """F4's margin `M = (k_c - k - tau_l [s]_+) / k_c`."""
    return (k_c - np.asarray(k_hat, float) - tau_l * np.maximum(np.asarray(s_hat, float), 0.0)) / k_c


class RuleCorrection:
    """Arm 5 (and arm 7 when fed true occupancy): on at `M <= m_on`, off at
    `M >= m_on + 0.05` under the hysteresis."""

    def __init__(self, m_on: float, tau_l: float = TAU_L_S, k_c: float = DENSITY_CRITICAL,
                 name: str = "rule") -> None:
        self.m_on = float(m_on)
        self.tau_l = float(tau_l)
        self.k_c = float(k_c)
        self.name = name
        self.hysteresis = Hysteresis()

    def update(self, ids: Sequence[str], t: float, k: np.ndarray, s: np.ndarray) -> np.ndarray:
        margin = rule_margin(k, s, self.k_c, self.tau_l)
        return self.hysteresis.update(ids, t, margin <= self.m_on + 1e-12,
                                      margin >= self.m_on + RELEASE_OFFSET - 1e-12)

    def describe(self) -> dict[str, Any]:
        return {"kind": self.name, "m_on": self.m_on, "tau_l": self.tau_l, "k_c": self.k_c,
                "release_offset": RELEASE_OFFSET, **self.hysteresis.describe()}


class LearnedCorrection:
    """Arm 6: step down when the onset predictor's `p >= theta`, release at
    `p < theta - 0.1`, under the hysteresis."""

    def __init__(self, predictor, theta: float) -> None:
        self.predictor = predictor
        self.theta = float(theta)
        self.hysteresis = Hysteresis()

    def update_from_probability(self, ids: Sequence[str], t: float,
                                probability: np.ndarray) -> np.ndarray:
        p = np.asarray(probability, dtype=float)
        return self.hysteresis.update(ids, t, p >= self.theta - 1e-12,
                                      p < self.theta - LEARNED_RELEASE_OFFSET - 1e-12)

    def update(self, ids: Sequence[str], t: float, features: np.ndarray) -> np.ndarray:
        probability = (self.predictor.predict(features) if len(ids)
                       else np.zeros(0))
        self.last_probability = probability
        return self.update_from_probability(ids, t, probability)

    def describe(self) -> dict[str, Any]:
        card = getattr(self.predictor, "card", None) or {}
        return {"kind": "learned", "theta": self.theta,
                "release_offset": LEARNED_RELEASE_OFFSET,
                "predictor_sha256": card.get("weights_sha256"), **self.hysteresis.describe()}


def batch_greedy(model, states: np.ndarray) -> np.ndarray:
    """`greedy_actions` for a stack of states `(n, segments, features)` in one pass.

    `SrcQNetwork.forward` flattens its whole input, so a batch goes through its layers
    directly; the softmax is kept so the function is SRC's, as in `greedy_actions`.
    """
    states = np.asarray(states, dtype=np.float32)
    if not len(states):
        return np.zeros((0, model.num_segments), dtype=np.int64)
    with torch.no_grad():
        flat = torch.from_numpy(states.reshape(len(states), -1))
        q = model.stack(flat).view(len(states), model.num_segments, model.num_actions)
        return torch.argmax(F.softmax(q, dim=2), dim=2).numpy()


class TakeTheSlower:
    """Arm 4: re-run the plan's network with the vehicle's own row set to what the camera
    sees, and show the slower of the plan and the re-run (plan section 5.3).

    The own row's speed becomes the mean reported leader speed (km/h) and its jam factor
    `10 (1 - v / v_free)` clipped to [0, 10]. A vehicle that detects nothing keeps the
    feed state, so its re-run equals the plan. While the hysteresis holds a vehicle on
    after the re-run has come back up to the plan, it keeps the last stepped-down fraction.
    """

    SPEED, FREE_FLOW, JAM = 0, 1, 2

    def __init__(self, model) -> None:
        self.model = model
        self.hysteresis = Hysteresis()
        self._held: dict[str, float] = {}
        self.active = np.zeros(0, dtype=bool)

    @classmethod
    def replace_row(cls, state: np.ndarray, segment: int, speed_kmh: float) -> np.ndarray:
        changed = np.array(state, dtype=np.float32, copy=True)
        changed[segment, cls.SPEED] = speed_kmh
        changed[segment, cls.JAM] = jam_factor(speed_kmh, float(state[segment, cls.FREE_FLOW]))
        return changed

    def correct(self, ids: Sequence[str], t: float, states: np.ndarray,
                segments: np.ndarray, local_speed_mps: np.ndarray,
                plan: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """`(shown fractions, re-run fractions)` per vehicle, in `ids` order."""
        states = np.asarray(states, dtype=np.float32)
        segments = np.asarray(segments, dtype=np.int64)
        n = len(ids)
        if plan is None:
            plan = _FRACTION_ARRAY[batch_greedy(self.model, states)[np.arange(n), segments]]
        rerun_states = states.copy()
        for k in range(n):
            if not np.isnan(local_speed_mps[k]):
                rerun_states[k] = self.replace_row(states[k], int(segments[k]),
                                                   float(local_speed_mps[k]) * 3.6)
        rerun = _FRACTION_ARRAY[batch_greedy(self.model, rerun_states)[np.arange(n), segments]]
        active = self.hysteresis.update(ids, t, rerun < plan - 1e-9, rerun >= plan - 1e-9)
        correction = np.ones(n)
        for k, vehicle in enumerate(ids):
            if not active[k]:
                self._held.pop(vehicle, None)
                continue
            if rerun[k] < plan[k] - 1e-9:
                self._held[vehicle] = float(rerun[k])
            correction[k] = self._held.get(vehicle, float(rerun[k]))
        self.active = active
        return shown_fraction(plan, correction), rerun

    def describe(self) -> dict[str, Any]:
        return {"kind": "take_the_slower", **self.hysteresis.describe()}


class LookaheadRule:
    """Arm 2: on the vehicle's applied feed version, 0.5 when the next downstream segment's
    jam factor is at least `J_th`, 0.75 when the one after it is, 1.0 otherwise (the
    slowing segment itself included). Downstream is the higher index on every route."""

    def __init__(self, J_th: float) -> None:
        self.J_th = float(J_th)

    def fractions(self, jam: np.ndarray, segments: np.ndarray) -> np.ndarray:
        jam = np.asarray(jam, dtype=float)
        segments = np.asarray(segments, dtype=np.int64)
        hot = np.concatenate([jam >= self.J_th, [False, False]])
        out = np.ones(len(segments))
        out[hot[segments + 2]] = 0.75
        out[hot[segments + 1]] = 0.5
        return out

    def describe(self) -> dict[str, Any]:
        return {"kind": "lookahead", "J_th": self.J_th}
