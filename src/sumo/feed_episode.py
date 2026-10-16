"""One Mainz episode in which every controlled vehicle applies its own feed version.

`run_feed_episode` runs the episode through `MainzEnv.advance_ticks`, so a callback
(`FeedEngine.on_tick`) sees the simulation once per second. The engine takes the feed's
snapshots, works out which version each controlled vehicle applies, writes a command to
a vehicle when its applied version changes, and records disagreement, snapshot age and a
10 s trace.

What a write is, in every mode. A controlled vehicle on a segment edge is written
`a_k[e] * free-flow speed` at the tick its applied version changes to `k`, through the
env's actuation (a maximum-speed cap since plan Amendment 1); a vehicle whose version
did not change is not written and keeps its last command, and a vehicle on a junction
or an unsegmented edge at that tick is not written either. That is exactly what
`MainzEnv.step()` does at a decision instant, so the `published` mode, where every
vehicle's version changes at the decision instants and nowhere else, writes the same
vehicles the same speeds at the same instants and equals the `step()` path.
Writes start at the warm-up's end (`control_start_s`), when every vehicle on a segment
is written its current version once; before that the feed is read but nothing is
commanded, as in the published arm.

Modes (plan section 3.3):

* `published`: the env's decision instants with zero lag, every vehicle switching at
  them. Only this mode, `inject_*` and `no_control` take no feed artefact.
* `rule`: a `Rule` from `feed.py` (synchronous, no rule, fixed phase, version-keyed)
  over a replayed feed process; each vehicle's switches are the rule's schedule for its
  own query phase.
* `inject_delta(delta)`: at each decision instant a fresh, seeded subset of
  `round(delta * n)` of the `n` controlled vehicles on segments applies `a_{k-1}` for
  the whole interval and the rest apply `a_k`. A late vehicle that applied `a_{k-1}`
  over the previous interval is therefore not written; one late twice running is.
* `inject_stale(s)`: at each decision instant every vehicle applies the policy's action
  on the snapshot taken `s` seconds earlier, from a ring of snapshots every 10 s. A
  decision with no snapshot that old (before the first one, at 10 s) writes nothing.
* `no_control`: no gate, `release()` before every interval, exactly as
  `evaluate_no_control`; traced like every other mode.

Disagreement, command disagreement and age are time-averaged at every second of the
scored window (from `window_start_s`). Where every vehicle shares one version
(`published`, `inject_stale`) disagreement is zero by construction and is also measured
from the 10 s scans in the trace.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo.feed import (
    FeedProcess, FixedPhase, NoRule, Rule, Synchronous, VehicleFeedState, VersionKeyed,
    command_disagreement, disagreement, draw_phases,
)
from src.sumo.mainz import (
    DECISION_INTERVAL_S, HERE_FEATURES, REPO_ROOT, SPEED_ACTION_FRACTIONS, MainzEnv,
)

#: The `inject_delta` draws are `default_rng(seed + INJECTION_SEED_OFFSET)`.
INJECTION_SEED_OFFSET: int = 3000
#: Snapshot ring spacing for `inject_stale`, and the trace spacing by default.
RING_S: float = 10.0
#: Phases are drawn for the schedule plus this many later positions (pulses).
EXTRA_POSITIONS: int = 10_000


@dataclass(frozen=True)
class Mode:
    kind: str
    rule: Rule | None = None
    delta: float = 0.0
    stale_s: float = 0.0
    name: str = ""

    @classmethod
    def published(cls) -> "Mode":
        return cls("published", Synchronous(), name="published")

    @classmethod
    def synchronous(cls) -> "Mode":
        return cls("rule", Synchronous(), name="synchronous")

    @classmethod
    def no_rule(cls, P_q: float) -> "Mode":
        return cls("rule", NoRule(P_q), name=f"no_rule_{P_q:g}")

    @classmethod
    def fixed_phase(cls, psi: float, r_bar: float) -> "Mode":
        return cls("rule", FixedPhase(psi, r_bar), name=f"fixed_phase_{psi:g}")

    @classmethod
    def version_keyed(cls, P_q: float, D: float) -> "Mode":
        return cls("rule", VersionKeyed(P_q, D), name=f"version_keyed_{D:g}")

    @classmethod
    def inject_delta(cls, delta: float) -> "Mode":
        if not 0.0 <= delta <= 1.0:
            raise ValueError(f"delta must lie in [0, 1], got {delta}")
        return cls("inject_delta", delta=delta, name=f"inject_delta_{delta:g}")

    @classmethod
    def inject_stale(cls, stale_s: float) -> "Mode":
        if stale_s < 0.0 or stale_s % RING_S:
            raise ValueError(f"staleness must be a non-negative multiple of {RING_S} s")
        return cls("inject_stale", stale_s=stale_s, name=f"inject_stale_{stale_s:g}")

    @classmethod
    def no_control(cls) -> "Mode":
        return cls("no_control", name="no_control")

    @property
    def per_tick(self) -> bool:
        """Vehicles can disagree, so every second is scanned."""
        return self.kind in ("rule", "inject_delta")

    @property
    def schedule_based(self) -> bool:
        return self.kind in ("published", "rule")

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.rule is not None:
            out.update(self.rule.describe())
        if self.kind == "inject_delta":
            out["delta"] = self.delta
        if self.kind == "inject_stale":
            out["stale_s"] = self.stale_s
        return out


@dataclass
class _Vehicle:
    ticks: list[int]
    versions: list[int]
    pointer: int = 0
    version: int = 0
    last_tick: int = -1


@dataclass
class _Accumulator:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if value is not None and not math.isnan(value):
            self.total += value
            self.count += 1

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class FeedEngine:
    """Per-tick feed bookkeeping against an env with `MainzEnv`'s read/write surface."""

    env: Any
    policy: Callable[[np.ndarray], Sequence[int]]
    mode: Mode
    feed: FeedProcess | None
    seed: int
    control_start_s: float
    window_start_s: float
    trace_every_s: float
    end_s: float
    decision_instants: Sequence[float]
    record_trace: bool = True
    trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode.kind in ("published", "inject_delta", "inject_stale"):
            if not len(self.decision_instants):
                raise ValueError(f"{self.mode.kind} needs the decision instants")
            self.feed = FeedProcess.aligned(self.decision_instants)
        elif self.mode.kind == "rule" and self.feed is None:
            raise ValueError("a rule mode needs a feed process")
        self._decision_ticks = {int(round(t)): k + 1
                                for k, t in enumerate(self.decision_instants)}
        self._u = draw_phases(self.seed, getattr(self.env, "schedule_size", 0)
                              + EXTRA_POSITIONS)
        self._rng = np.random.default_rng(self.seed + INJECTION_SEED_OFFSET)
        self._vehicles: dict[str, _Vehicle] = {}
        self._shared: tuple[list[int], list[int]] | None = None
        self.vectors: dict[int, list[int]] = {}
        self.snapshot_t: dict[int, float] = {}
        self.snapshots: list[dict[str, Any]] = []
        self._ring: dict[int, np.ndarray] = {}
        self._next_version = 1
        self._global_version = 0
        self._late: set[str] = set()
        self._late_before: set[str] = set()
        self._interval = 0
        self._last_scan: list[tuple[str, str, int]] = []
        self._last_versions: dict[str, int] = {}
        self.delta = _Accumulator()
        self.command_delta = _Accumulator()
        self.age = _Accumulator()
        self.writes = 0
        self.writes_late = 0
        self.writes_on_time = 0
        self.writes_by_version: Counter = Counter()
        self.switches_by_version: Counter = Counter()
        self.commanded: set[str] = set()
        self.late_draws: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- snapshots

    def _snapshot(self, t: float, version: int, x: np.ndarray, taken_at: float) -> None:
        vector = [int(a) for a in self.policy(x)]
        self.vectors[version] = vector
        self.snapshot_t[version] = taken_at
        self.snapshots.append({"version": version, "t": taken_at, "applied_from": t,
                               "x": np.asarray(x).tolist(), "actions": vector})

    def _take_snapshots(self, t: float) -> None:
        tick = int(round(t))
        if self.mode.kind == "inject_stale":
            if tick % int(RING_S) == 0:
                self._ring[tick] = self.env.observe()
            return
        if not self.mode.schedule_based:
            return  # `inject_delta` snapshots at its decisions, `no_control` never
        # Every version published by this tick is snapshotted at the first tick at or
        # after its stamp; the version's vector is computed from it once.
        while (self._next_version <= self.feed.count
               and math.ceil(self.feed.publish_of(self._next_version) - 1e-9) <= tick):
            version = self._next_version
            self._snapshot(t, version, self.env.observe(), self.feed.publish_of(version))
            self._next_version += 1

    # ----------------------------------------------------------------- vehicles

    def _schedule(self, vehicle: str) -> _Vehicle:
        if isinstance(self.mode.rule, Synchronous):
            # Vehicle-independent: computed once and shared.
            if self._shared is None:
                instants, versions = self.mode.rule.switch_schedule(
                    VehicleFeedState("*", 0.0), self.feed, self.end_s)
                self._shared = (_ticks(instants), [int(v) for v in versions])
            ticks, versions = self._shared
        else:
            state = VehicleFeedState(vehicle, float(self._u[self.env.position(vehicle)]))
            instants, versions = self.mode.rule.switch_schedule(state, self.feed, self.end_s)
            ticks, versions = _ticks(instants), [int(v) for v in versions]
        record = _Vehicle(ticks, versions)
        self._vehicles[vehicle] = record
        return record

    def applied_versions(self) -> dict[str, int]:
        """The applied version of every vehicle in the most recent scan."""
        return dict(self._last_versions)

    def _write(self, t: float, mapping: dict[str, float], edges: dict[str, str],
               versions: dict[str, int], late: set[str] | None = None) -> None:
        if not mapping:
            return
        self.env.command_vehicles(mapping, edges)
        self.writes += len(mapping)
        self.commanded.update(mapping)
        for vehicle in mapping:
            self.writes_by_version[versions[vehicle]] += 1
            if late is not None:
                if vehicle in late:
                    self.writes_late += 1
                else:
                    self.writes_on_time += 1

    def _fraction(self, version: int, segment: int) -> float:
        return SPEED_ACTION_FRACTIONS[self.vectors[version][segment]]

    def _scan_schedule_mode(self, t: float, tick: int, scan) -> dict[str, int]:
        control = t >= self.control_start_s
        starting = tick == int(round(self.control_start_s))
        mapping, edges, versions = {}, {}, {}
        for vehicle, edge, segment in scan:
            record = self._vehicles.get(vehicle) or self._schedule(vehicle)
            while (record.pointer < len(record.ticks)
                   and record.ticks[record.pointer] <= tick):
                record.version = record.versions[record.pointer]
                record.last_tick = record.ticks[record.pointer]
                record.pointer += 1
                self.switches_by_version[record.version] += 1
            versions[vehicle] = record.version
            if (control and record.version >= 1
                    and (record.last_tick == tick or starting)):
                mapping[vehicle] = self._fraction(record.version, segment)
                edges[vehicle] = edge
        self._write(t, mapping, edges, versions)
        return versions

    def _decide_injected(self, t: float, tick: int, scan) -> None:
        """At a decision instant of `inject_delta` or `inject_stale`."""
        k = self._decision_ticks[tick]
        self._interval = k
        if self.mode.kind == "inject_stale":
            old = self._ring.get(tick - int(self.mode.stale_s))
            if old is None:
                return
            self._snapshot(t, k, old, t - self.mode.stale_s)
            self._global_version = k
            if t >= self.control_start_s:
                mapping = {v: self._fraction(k, s) for v, _, s in scan}
                self._write(t, mapping, {v: e for v, e, _ in scan}, {v: k for v in mapping})
            return
        self._snapshot(t, k, self.env.observe(), t)
        size = int(round(self.mode.delta * len(scan)))
        chosen = (self._rng.choice(len(scan), size=size, replace=False)
                  if size else np.array([], dtype=int))
        self._late_before, self._late = self._late, {scan[i][0] for i in chosen}
        self.late_draws.append({"t": t, "on_segments": len(scan), "late": size})
        if t < self.control_start_s:
            return
        mapping, edges, versions = {}, {}, {}
        for vehicle, edge, segment in scan:
            target = k - 1 if vehicle in self._late else k
            before = k - 2 if vehicle in self._late_before else k - 1
            versions[vehicle] = target
            if target != before and target >= 1:
                mapping[vehicle] = self._fraction(target, segment)
                edges[vehicle] = edge
        self._write(t, mapping, edges, versions, late=self._late)

    def _injected_versions(self, scan) -> dict[str, int]:
        if self.mode.kind == "inject_stale":
            return {v: self._global_version for v, _, _ in scan}
        k = self._interval
        return {v: (k - 1 if v in self._late else k) for v, _, _ in scan}

    # --------------------------------------------------------------------- tick

    def on_tick(self, t: float) -> None:
        tick = int(round(t))
        self._take_snapshots(t)
        if self.mode.kind == "no_control":
            if self.record_trace and tick % int(self.trace_every_s) == 0:
                self._record(t, None, {})
            return
        decision = tick in self._decision_ticks
        trace_now = self.record_trace and tick % int(self.trace_every_s) == 0
        control = t >= self.control_start_s
        scan_now = (decision or trace_now or tick == int(round(self.control_start_s))
                    or (self.mode.per_tick and control))
        if not scan_now:
            if self.mode.kind in ("published", "inject_stale") and t >= self.window_start_s:
                self._shared_stats(t)
            return
        scan = self.env.controlled_on_segments()
        if self.mode.schedule_based:
            versions = self._scan_schedule_mode(t, tick, scan)
        else:
            if decision:
                self._decide_injected(t, tick, scan)
            versions = self._injected_versions(scan)
        self._last_scan, self._last_versions = scan, versions
        measured = None
        if t >= self.window_start_s:
            if self.mode.per_tick:
                measured = self._measure(t, scan, versions)
                self.delta.add(measured[0])
                self.command_delta.add(measured[1])
                self.age.add(measured[2])
            else:
                self._shared_stats(t)
        if trace_now:
            self._record(t, measured if measured else self._measure(t, scan, versions),
                         versions)

    def _shared_stats(self, t: float) -> None:
        """Every vehicle applies one version: disagreement is zero and age is common."""
        if self.mode.kind == "published":
            version = self.feed.version_at(t)
        else:
            version = self._global_version
        self.delta.add(0.0)
        self.command_delta.add(0.0)
        self.age.add(t - self.snapshot_t[version] if version >= 1 else math.nan)

    def _measure(self, t: float, scan, versions: dict[str, int]):
        applied = [versions[v] for v, _, _ in scan]
        segments = [s for _, _, s in scan]
        ages = [t - self.snapshot_t[k] for k in applied if k >= 1]
        return (disagreement(applied),
                command_disagreement(applied, segments, self.vectors),
                float(np.mean(ages)) if ages else math.nan)

    def _record(self, t: float, measured, versions: dict[str, int]) -> None:
        state = self.env.segment_state()
        histogram = Counter(versions.values())
        delta, command, age = measured if measured else (None, None, None)
        self.trace.append({
            "t": t,
            "speed_kmh": [None if math.isnan(v) else round(v, 4) for v in state["speed_kmh"]],
            "density": [round(v, 6) for v in state["density"]],
            "count": state["count"],
            "held": self.env.held_at_entry(),
            "pending": self.env.pending(),
            "running": self.env.running(),
            "arrived": self.env.arrived_total,
            # Cumulative since reset: vehicles reported colliding, and collision
            # teleports started (plan Amendment 1).
            "colliding_vehicles": getattr(self.env, "colliding_vehicles", 0),
            "collision_teleports": getattr(self.env, "collision_teleports", 0),
            "versions": {str(k): n for k, n in sorted(histogram.items())},
            "delta": _clean(delta),
            "command_delta": _clean(command),
            "age_s": _clean(age),
        })

    def summary(self) -> dict[str, Any]:
        return {
            "delta_mean": self.delta.mean,
            "command_delta_mean": self.command_delta.mean,
            "age_mean_s": self.age.mean,
            "delta_samples": self.delta.count,
            "delta_sampling": "every second" if self.mode.per_tick else
                              "every second, zero by construction; trace scans measured",
            "set_speed_writes": self.writes,
            "writes_late": self.writes_late,
            "writes_on_time": self.writes_on_time,
            "writes_by_version": {str(k): n for k, n in sorted(self.writes_by_version.items())},
            "switches_by_version": {str(k): n
                                    for k, n in sorted(self.switches_by_version.items())},
            "commanded_ids": len(self.commanded),
            "versions_published": len(self.vectors),
            "late_draws": self.late_draws,
            "snapshots": self.snapshots,
        }


def _ticks(instants: np.ndarray) -> list[int]:
    """A switch at real time `s` takes effect at the first whole second at or after it."""
    return [int(math.ceil(s - 1e-9)) for s in np.asarray(instants, dtype=float)]


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


# ------------------------------------------------------------------ the episode


def _scenario_paths(scenario: str | dict) -> dict:
    if isinstance(scenario, dict):
        return scenario
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from train_mainz_src import SCENARIOS

    return SCENARIOS[scenario]


def decision_instants(warmup_s: float, duration_s: float) -> list[float]:
    """The instants at which `run_episode` decides: warm-up end, every 60 s, while t <
    duration. The episode therefore ends at the first multiple past the duration."""
    instants, t = [], warmup_s
    while t < duration_s:
        instants.append(t)
        t += DECISION_INTERVAL_S
    return instants


def run_feed_episode(model, seed: int, scenario: str | dict = "mainz",
                     mode: Mode | None = None, feed_process: FeedProcess | None = None,
                     penetration: float = 1.0, duration_s: float = 2500.0,
                     trace_every_s: float = 10.0, pulse: dict | None = None,
                     step_length_s: float = 0.5, window_start_s: float = 900.0,
                     features: tuple[str, ...] = HERE_FEATURES,
                     record_trace: bool = True,
                     actuation: str = "max_speed") -> dict[str, Any]:
    """One episode of `mode`; returns `run_episode`'s metrics plus the feed's.

    `pulse`, when given, is `inject_pulse`'s keyword arguments (site, n, start_s,
    duration_s) and is injected after the warm-up. The model must be the `here`
    checkpoint's: the snapshot is `observe()` at the publication instant, which equals
    the published arm's state only for the HERE features.
    """
    import torch

    from src.rl.src_q import greedy_actions

    if features != HERE_FEATURES:
        raise ValueError("the feed episode evaluates the here checkpoint only")
    mode = mode or Mode.published()
    paths = _scenario_paths(scenario)
    gate = mode.kind != "no_control"
    env = MainzEnv(seed=seed, duration_s=duration_s, features=features,
                   step_length_s=step_length_s, gate_entries=gate,
                   net_file=REPO_ROOT / paths["net"],
                   route_file=REPO_ROOT / paths["gated_routes" if gate else "routes"],
                   segment_file=REPO_ROOT / paths["segments"],
                   schedule_file=REPO_ROOT / paths["schedule"],
                   penetration=penetration, actuation=actuation)
    decisions = decision_instants(env.warmup_s, duration_s)
    end_s = decisions[-1] + DECISION_INTERVAL_S if decisions else duration_s
    if mode.kind == "rule" and feed_process is None:
        raise ValueError(f"{mode.name} needs a feed process")

    def policy(x: np.ndarray) -> list[int]:
        return greedy_actions(model, torch.tensor(x, dtype=torch.float)).tolist()

    engine = FeedEngine(env, policy, mode, feed_process, seed=seed,
                        control_start_s=env.warmup_s, window_start_s=window_start_s,
                        trace_every_s=trace_every_s, end_s=end_s,
                        decision_instants=decisions, record_trace=record_trace)
    rewards, total_no_control = [], 0.0
    inserted_pulse = None
    try:
        env.reset(on_warmup_tick=engine.on_tick)
        if pulse is not None:
            inserted_pulse = env.inject_pulse(**pulse, seed=seed)
        marked = window_start_s <= 0.0
        while True:
            if mode.kind == "no_control":
                env.release()
            env.advance_ticks(DECISION_INTERVAL_S, on_tick=engine.on_tick)
            reward = env.reward()
            if mode.kind == "no_control":
                total_no_control += float(reward.mean())
            else:
                rewards.append(torch.tensor(reward, dtype=torch.float))
            if not marked and env.time_s() >= window_start_s:
                env.mark_window()
                marked = True
            if env.time_s() >= duration_s:
                break
        metrics = env.metrics()
    finally:
        env.close()
    episode_return = (total_no_control if mode.kind == "no_control"
                      else float(torch.stack(rewards).mean(dim=1).sum()))
    return {
        "mode": mode.describe(),
        "seed": seed,
        "penetration": penetration,
        "duration_s": duration_s,
        "flow": metrics["flow_veh_per_h"],
        "arrived": metrics["arrived"],
        "return": episode_return,
        "mean_speed_kmh": metrics["mean_speed_kmh"],
        "space_mean_speed_kmh": metrics["space_mean_speed_kmh"],
        "mean_vehicles": metrics["mean_vehicles"],
        "held_at_entry": metrics["held_at_entry"],
        "pending": metrics["waiting_to_enter"],
        "running": metrics["running"],
        "admitted": metrics["admitted"],
        "actuation": actuation,
        "colliding_vehicles": metrics["colliding_vehicles"],
        "collision_teleports": metrics["collision_teleports"],
        "in_teleport": metrics["in_teleport"],
        "feed": engine.feed.describe() if engine.feed is not None else None,
        "pulse": inserted_pulse,
        **engine.summary(),
        "trace": engine.trace,
    }
