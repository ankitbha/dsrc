"""One jam-program episode: the shared feed plan, a local correction, and the camera.

Plan 2026-10-16 section 5.1. `JamEngine` extends `FeedEngine`, which keeps doing what it
did for the feed program: it snapshots each feed version at its stamp, computes the plan's
action vector from it once, and works out which version each controlled vehicle applies
under the version-keyed rule (`P_q = 10 s`, `D = m L_bar + P_q`; the synchronous rule at
zero lag for `m = 0`). What changes is the write path.

**Every write goes through one place** (`_writes`). At every 5 s tick, and at every tick a
vehicle's applied version changes, each controlled vehicle on a segment edge is written
`shown x free-flow speed of its edge` through `MainzEnv.command_vehicles`, which applies it
with `setMaxSpeed`. `shown` is the plan's fraction for the vehicle's current segment (arm
3), the lookahead's fraction (arm 2), or `min(plan, correction)` (arms 4 to 7). Each write
is compared with its base (the plan, or the lookahead's own value for arm 2) and a write
above it is counted; it must never happen. With `refresh` off only the switch writes
remain, which is exactly what `FeedEngine` did, so the published arm is reproduced.

**Sensing** runs every 5 s from the warm-up's end for every controlled vehicle on a segment
edge (every vehicle in the no-control arm, whose env is built at penetration 1.0 and
commands nothing): one `LaneSnapshot`, one `CandidateTable`, then the camera model (when
one is given) and the oracle's true occupancy. Corrected arms decide on it; every arm traces
its per-segment means.

Arms: `no_control`, `lookahead` (J_th), `plan`, `take_the_slower`, `rule` (m_on, tau_l),
`learned` (theta), `oracle` (m_on, tau_l, R). The camera draws come from
`default_rng(seed + 5000)` in ego-id order; feature sampling for block T from
`default_rng(seed + 6000)`.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo.correction import (
    LearnedCorrection, LookaheadRule, RuleCorrection, TakeTheSlower, shown_fraction,
    step_down,
)
from src.sumo.feed import FeedProcess, Synchronous, VersionKeyed, _sha256
from src.sumo.feed_episode import FeedEngine, Mode, _scenario_paths, decision_instants
from src.sumo.mainz import (
    DECISION_INTERVAL_S, HERE_FEATURES, REPO_ROOT, SPEED_ACTION_FRACTIONS, MainzEnv,
)
from src.sumo.sensing import (
    CAMERA_SEED_OFFSET, DEPLOYED_RANGE_M, CameraModel, CandidateTable, OracleSensor,
    SnapshotBuilder, TrendTracker, estimate, successors,
)

ARMS: tuple[str, ...] = ("no_control", "lookahead", "plan", "take_the_slower", "rule",
                         "learned", "oracle")
CORRECTED: tuple[str, ...] = ("take_the_slower", "rule", "learned", "oracle")
CAMERA_ARMS: tuple[str, ...] = ("take_the_slower", "rule", "learned")
P_Q: float = 10.0
SENSE_EVERY_S: int = 5
TRACE_EVERY_S: int = 10
SAMPLE_SEED_OFFSET: int = 6000
#: The replayed process's first stamp, moved to its recorded second of the minute.
FEED_START_S: float = 60.0
#: Publications are replayed this far past the episode's end (plan open item 10).
FEED_TAIL_S: float = 180.0


@dataclass(frozen=True)
class JamArm:
    arm: str
    J_th: float | None = None
    m_on: float | None = None
    tau_l: float = 10.0
    theta: float | None = None
    oracle_R: float = DEPLOYED_RANGE_M
    refresh: bool = True

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; arms are {ARMS}")
        needed = {"lookahead": "J_th", "rule": "m_on", "oracle": "m_on", "learned": "theta"}
        name = needed.get(self.arm)
        if name and getattr(self, name) is None:
            raise ValueError(f"arm {self.arm} needs {name}")

    @property
    def sensor(self) -> str:
        """What the arm's decisions read (plan attack point 4)."""
        if self.arm in CAMERA_ARMS:
            return "camera"
        if self.arm == "oracle":
            return "oracle"
        return "none"

    def describe(self) -> dict[str, Any]:
        return {**asdict(self), "sensor": self.sensor}


@dataclass(frozen=True)
class JamScenario:
    """A demand file set and its perturbation; `start_s` is where every metric clock starts
    and `end_s` where the drain clock starts (plan attack point 15)."""

    name: str
    files: str
    start_s: float
    end_s: float
    incidents: tuple[tuple[str, tuple[int, ...], float, float, float], ...] = ()

    def describe(self) -> dict[str, Any]:
        return asdict(self)


#: Plan section 4.2. BG and S45 have no event; their clocks start at 900 s like the rest.
SCENARIOS: dict[str, JamScenario] = {
    "B5": JamScenario("B5", "mainz_burst", 900.0, 1500.0),
    "B7.5": JamScenario("B7.5", "mainz_b7500", 900.0, 1200.0),
    "B9": JamScenario("B9", "mainz_b9000", 900.0, 1200.0),
    "I_exit": JamScenario("I_exit", "mainz_bg", 900.0, 1200.0,
                          (("exit_road", (0,), 100.0, 900.0, 300.0),)),
    "I_mid": JamScenario("I_mid", "mainz_bg", 900.0, 1200.0,
                         (("14", (0, 1), 600.0, 900.0, 300.0),)),
    "BG": JamScenario("BG", "mainz_bg", 900.0, 900.0),
    "S45": JamScenario("S45", "mainz", 900.0, 900.0),
}


@dataclass(frozen=True)
class FeedSetting:
    """The feed-process artefact a run replays and its lag multiplier (plan J1, J2)."""

    artefact: Path
    session: str
    m: float = 1.0
    P_q: float = P_Q
    allow_placeholder: bool = False

    def parameters(self) -> dict[str, Any]:
        data = json.loads(Path(self.artefact).read_text())
        if self.session not in data["sessions"]:
            raise SystemExit(f"{self.artefact} has no session {self.session!r}; it has "
                             f"{sorted(data['sessions'])}")
        placeholder = bool(data.get("placeholder"))
        if placeholder and not self.allow_placeholder:
            raise SystemExit(f"{self.artefact} is a placeholder; pass --allow-placeholder "
                             "to run on it")
        L_bar = float(data["sessions"][self.session]["L_bar"])
        return {"artefact": str(self.artefact), "artefact_sha256": _sha256(Path(self.artefact)),
                "session": self.session, "placeholder": placeholder, "L_bar": L_bar,
                "m": float(self.m), "P_q": self.P_q,
                "D": (float(self.m) * L_bar + self.P_q) if self.m > 0 else None,
                "rule": "version_keyed" if self.m > 0 else "synchronous"}

    def process(self, seed: int, end_s: float):
        """`(FeedProcess, Rule, parameters)` for one episode: the replay wraps when the
        session is shorter than the episode (plan open item 10), each wrapped stamp keeping
        its recorded second of the minute."""
        params = self.parameters()
        feed = FeedProcess.from_artefact(Path(self.artefact), self.session, seed,
                                         FEED_START_S, end_s, allow_wrap=True)
        feed = feed.with_lag_multiplier(self.m)
        rule = VersionKeyed(self.P_q, params["D"]) if self.m > 0 else Synchronous()
        return feed, rule, params


@dataclass
class _Counter:
    writes: int = 0
    above_base: int = 0
    stepped_down: int = 0
    vehicle_ticks: int = 0
    active_ticks: int = 0
    #: Active ticks on which the plan's fraction is above 0.5, so the step-down changes
    #: the shown speed (`step_down(0.5) = 0.5`; validation R1-7).
    active_changing_ticks: int = 0
    max_speed_checked: int = 0
    max_speed_mismatched: int = 0
    #: Candidate rows whose leader is an incident vehicle (attack point 12).
    incident_rows: int = 0


@dataclass
class JamEngine(FeedEngine):
    """`FeedEngine` with the jam program's write path, sensing and corrections."""

    arm: JamArm | None = None
    camera: CameraModel | None = None
    successor: dict[str, str] | None = None
    q_model: Any = None
    predictor: Any = None
    sense: bool = True
    sample_fraction: float = 0.0
    write_log: bool = False
    check_max_speed: bool = False
    #: The 10 s jam trace. `FeedEngine`'s own trace (`record_trace`) stays off.
    record_jam: bool = True
    #: For tests: a callable `t -> LaneSnapshot` in place of reading SUMO.
    snapshot_source: Callable[[float], Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        arm = self.arm
        if arm is None:
            raise ValueError("a jam engine needs an arm")
        if arm.arm in CAMERA_ARMS and self.camera is None:
            raise ValueError(f"arm {arm.arm} reads the camera and needs a camera model")
        if arm.arm == "learned" and self.predictor is None:
            raise ValueError("the learned arm needs a predictor")
        if arm.arm == "take_the_slower" and self.q_model is None:
            raise ValueError("take-the-slower needs the plan's Q-network")
        self.jam_trace: list[dict[str, Any]] = []
        self.states: dict[int, np.ndarray] = {}
        self.counter = _Counter()
        self.writes_log: list[list[Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.switch_ticks: dict[int, list[int]] = {}
        self.sense_wall_s: list[float] = []
        self._switched: dict[str, float] = {}
        self._shown: dict[str, float] = {}
        self._active: dict[str, bool] = {}
        self._rng_camera = np.random.default_rng(self.seed + CAMERA_SEED_OFFSET)
        self._rng_sample = np.random.default_rng(self.seed + SAMPLE_SEED_OFFSET)
        self._oracle80 = OracleSensor(DEPLOYED_RANGE_M)
        self._oracle = OracleSensor(arm.oracle_R)
        self._R_build = max(DEPLOYED_RANGE_M, arm.oracle_R,
                            self.camera.R_max if self.camera is not None else 0.0)
        self._trend_k = TrendTracker()
        self._trend_true = TrendTracker()
        self._trend_speed = TrendTracker()
        self._builder = None
        self._segment_of = {e: i for i, segment in enumerate(self.env.segments)
                            for e in segment}
        self._sensed: dict[str, Any] | None = None
        self.correction = None
        if arm.arm in ("rule", "oracle"):
            self.correction = RuleCorrection(arm.m_on, arm.tau_l, name=arm.arm)
        elif arm.arm == "learned":
            self.correction = LearnedCorrection(self.predictor, arm.theta)
        elif arm.arm == "take_the_slower":
            self.correction = TakeTheSlower(self.q_model)
        self.lookahead = LookaheadRule(arm.J_th) if arm.arm == "lookahead" else None

    # ------------------------------------------------------------- stage 1 hooks

    def _snapshot(self, t, version, x, taken_at):
        super()._snapshot(t, version, x, taken_at)
        self.states[version] = np.asarray(x, dtype=np.float32)

    def _write(self, t, mapping, edges, versions, late=None):
        """Stage 1 asks for the vehicles whose applied version changed this tick (and for
        every vehicle at control start); they are written in `_writes`."""
        starting = int(round(t)) == int(round(self.control_start_s))
        for vehicle in mapping:
            self._switched[vehicle] = versions[vehicle]
            if not starting:
                self.switch_ticks.setdefault(versions[vehicle], []).append(int(round(t)))

    # --------------------------------------------------------------------- tick

    def on_tick(self, t: float) -> None:
        tick = int(round(t))
        self._switched = {}
        super().on_tick(t)
        sense_now = (tick % SENSE_EVERY_S == 0 and t >= self.control_start_s
                     and (self.sense or self.arm.arm in CORRECTED))
        if sense_now:
            self._sense(t)
        if self.arm.arm != "no_control" and t >= self.control_start_s:
            self._writes(t, tick)
        if self.record_jam and tick % TRACE_EVERY_S == 0:
            self._record_jam(t)

    # ------------------------------------------------------------------ sensing

    def _snapshot_now(self, t: float):
        if self.snapshot_source is not None:
            return self.snapshot_source(t)
        if self._builder is None:
            self._builder = SnapshotBuilder(self.env, self.successor or {})
        return self._builder.build(t)

    def _sense(self, t: float) -> None:
        started = time.perf_counter()
        snap = self._snapshot_now(t)
        egos = snap.egos(self.env.controlled, set(self._segment_of))
        table = CandidateTable.build(snap, egos, self._R_build)
        ids = table.ego_ids
        segments = np.array([self._segment_of[e] for e in table.ego_edges], dtype=np.int64)
        k_true = self._oracle80.k_true(table)
        k_true_arm = k_true if self._oracle.R == DEPLOYED_RANGE_M else self._oracle.k_true(table)
        s_true = self._trend_true.update(ids, t, k_true_arm)
        est = s_hat = v_slope = None
        if self.camera is not None:
            est = estimate(table, self.camera.observe(table, self._rng_camera), self.camera.R_eff)
            s_hat = self._trend_k.update(ids, t, est.k_hat)
            v_slope = self._trend_speed.update(ids, t, est.mean_speed)
        if len(table.vehicle):
            self.counter.incident_rows += sum(1 for v in table.vehicle
                                              if snap.ids[v].startswith("incident_"))
        self._sensed = {"t": t, "ids": ids, "segments": segments, "k_true": k_true,
                        "k_hat": None if est is None else est.k_hat, "table": table}
        if self.arm.arm in CORRECTED:
            self._decide(t, ids, segments, table, est, s_hat, v_slope, k_true_arm, s_true)
        if self.sample_fraction > 0.0 and est is not None:
            self._sample(t, ids, segments, table, est, s_hat, v_slope)
        self.sense_wall_s.append(time.perf_counter() - started)

    def _versions(self, ids: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(ids), dtype=np.int64)
        for k, vehicle in enumerate(ids):
            record = self._vehicles.get(vehicle)
            out[k] = record.version if record is not None else 0
        return out

    def _decide(self, t, ids, segments, table, est, s_hat, v_slope, k_true, s_true) -> None:
        versions = self._versions(ids)
        valid = versions >= 1
        if not valid.any():
            return
        index = np.flatnonzero(valid)
        vids = [ids[k] for k in index]
        plan = np.array([self._fraction(int(versions[k]), int(segments[k])) for k in index])
        arm = self.arm.arm
        if arm == "oracle":
            active = self.correction.update(vids, t, k_true[index], s_true[index])
        elif arm == "rule":
            active = self.correction.update(vids, t, est.k_hat[index], s_hat[index])
        elif arm == "learned":
            from src.rl.onset_predictor import engine_features

            features = engine_features(self, t, index, table, est, s_hat, v_slope, versions,
                                       segments)
            active = self.correction.update(vids, t, features)
        else:  # take_the_slower
            states = np.stack([self.states[int(versions[k])] for k in index])
            self.correction.correct(vids, t, states, segments[index], est.mean_speed[index],
                                    plan=plan)
            active = self.correction.active
        for vehicle, on in zip(vids, active):
            self._active[vehicle] = bool(on)
        self.counter.vehicle_ticks += len(vids)
        self.counter.active_ticks += int(np.sum(active))
        self.counter.active_changing_ticks += int(np.sum(np.asarray(active) & (plan > 0.5 + 1e-9)))

    def _sample(self, t, ids, segments, table, est, s_hat, v_slope) -> None:
        """Block T: features of a 5% sample of the sensing vehicles (plan section 5.5)."""
        from src.rl.onset_predictor import engine_features

        versions = self._versions(ids)
        chosen = np.flatnonzero(self._rng_sample.random(len(ids)) < self.sample_fraction)
        chosen = chosen[versions[chosen] >= 1]
        if not len(chosen):
            return
        features = engine_features(self, t, chosen, table, est, s_hat, v_slope, versions,
                                   segments)
        for row, k in zip(features, chosen):
            self.samples.append({"vehicle": ids[k], "t": t, "segment": int(segments[k]),
                                 "features": [float(v) for v in row]})

    def _correction(self, vehicle: str, plan: float) -> float:
        if not self._active.get(vehicle):
            return 1.0
        if self.arm.arm == "take_the_slower":
            return self.correction._held.get(vehicle, 1.0)
        return step_down(plan)

    # ------------------------------------------------------------------- writes

    def _writes(self, t: float, tick: int) -> None:
        refresh_now = self.arm.refresh and tick % SENSE_EVERY_S == 0
        scan = self._last_scan
        targets = scan if refresh_now else [row for row in scan if row[0] in self._switched]
        mapping, edges = {}, {}
        corrected = self.arm.arm in CORRECTED
        for vehicle, edge, segment in targets:
            record = self._vehicles.get(vehicle)
            if record is None or record.version < 1:
                continue
            version = record.version
            plan = self._fraction(version, segment)
            if self.lookahead is not None:
                base = float(self.lookahead.fractions(self.states[version][:, 2],
                                                      np.array([segment]))[0])
                shown = base
            else:
                base = plan
                shown = (float(shown_fraction(plan, self._correction(vehicle, plan)))
                         if corrected else plan)
            mapping[vehicle] = shown
            edges[vehicle] = edge
            self._shown[vehicle] = shown
            self.counter.writes += 1
            if shown > base + 1e-9:
                self.counter.above_base += 1
            if corrected and shown < plan - 1e-9:
                self.counter.stepped_down += 1
            if self.write_log:
                self.writes_log.append([vehicle, tick, edge, version, plan, shown])
        if mapping:
            self.env.command_vehicles(mapping, edges)
            if self.check_max_speed:
                self._check(mapping, edges)

    def _check(self, mapping: dict[str, float], edges: dict[str, str]) -> None:
        """Read the cap back from SUMO for every stepped-down write (attack point 2)."""
        from src.sumo.binding import _sumo

        for vehicle, shown in mapping.items():
            version = self._vehicles[vehicle].version
            plan = self._fraction(version, self._segment_of[edges[vehicle]])
            if shown >= plan - 1e-9 and self.lookahead is None:
                continue
            free = self.env._static[edges[vehicle]]["free_flow_kmh"] / 3.6
            self.counter.max_speed_checked += 1
            if not math.isclose(_sumo.vehicle.getMaxSpeed(vehicle), shown * free,
                                rel_tol=1e-9, abs_tol=1e-9):
                self.counter.max_speed_mismatched += 1

    # -------------------------------------------------------------------- trace

    def _record_jam(self, t: float) -> None:
        env = self.env
        state = env.segment_state()
        n = len(env.segments)
        plan_sum, shown_sum, on_sum, counted = (np.zeros(n) for _ in range(4))
        if self.arm.arm != "no_control" and t >= self.control_start_s:
            for vehicle, _edge, segment in self._last_scan:
                record = self._vehicles.get(vehicle)
                if record is None or record.version < 1:
                    continue
                counted[segment] += 1
                plan_sum[segment] += self._fraction(record.version, segment)
                shown_sum[segment] += self._shown.get(vehicle, math.nan)
                on_sum[segment] += bool(self._active.get(vehicle))

        def per_segment(total, count):
            return [None if not c or math.isnan(v) else round(float(v / c), 6)
                    for v, c in zip(total, count)]

        k_hat_mean = k_true_mean = [None] * n
        sensed = self._sensed
        if sensed is not None and abs(sensed["t"] - t) < 1e-6 and len(sensed["ids"]):
            count = np.bincount(sensed["segments"], minlength=n).astype(float)
            k_true_mean = per_segment(np.bincount(sensed["segments"], weights=sensed["k_true"],
                                                  minlength=n), count)
            if sensed["k_hat"] is not None:
                k_hat_mean = per_segment(np.bincount(sensed["segments"],
                                                     weights=sensed["k_hat"], minlength=n),
                                         count)
        self.jam_trace.append({
            "t": t,
            "speed_kmh": [None if math.isnan(v) else round(v, 4) for v in state["speed_kmh"]],
            "density": [round(v, 6) for v in state["density"]],
            "count": state["count"],
            "held": env.held_at_entry(), "pending": env.pending(), "running": env.running(),
            "arrived": env.arrived_total,
            "colliding_vehicles": getattr(env, "colliding_vehicles", 0),
            "collision_teleports": getattr(env, "collision_teleports", 0),
            "k_hat_mean": k_hat_mean, "k_true_mean": k_true_mean,
            "plan_mean": per_segment(plan_sum, counted),
            "shown_mean": per_segment(shown_sum, counted),
            "active_share": per_segment(on_sum, counted),
        })

    def jam_summary(self) -> dict[str, Any]:
        c = self.counter
        switches = {str(k): {"T_k": self.feed.publish_of(k) if self.feed is not None else None,
                             "snapshot_t": self.snapshot_t.get(k),
                             "first_tick": min(v), "last_tick": max(v), "count": len(v)}
                    for k, v in sorted(self.switch_ticks.items())}
        walls = np.asarray(self.sense_wall_s)
        return {
            "arm_spec": self.arm.describe(),
            "sensor": self.arm.sensor,
            "camera_model": (self.camera.source if self.camera is not None else None),
            "camera_bias": (self.camera.bias if self.camera is not None else None),
            "correction": (self.correction.describe() if self.correction is not None
                           else self.lookahead.describe() if self.lookahead is not None
                           else None),
            "writes": c.writes, "writes_above_plan": c.above_base,
            "writes_stepped_down": c.stepped_down,
            "corrected_vehicle_ticks": c.vehicle_ticks,
            "active_share": (c.active_ticks / c.vehicle_ticks) if c.vehicle_ticks else None,
            "active_changing_share": ((c.active_changing_ticks / c.vehicle_ticks)
                                      if c.vehicle_ticks else None),
            "max_speed_checked": c.max_speed_checked,
            "max_speed_mismatched": c.max_speed_mismatched,
            "incident_candidate_rows": c.incident_rows,
            "sense_passes": len(walls),
            "sense_ms_mean": float(walls.mean() * 1000.0) if len(walls) else None,
            "sense_ms_p95": float(np.percentile(walls, 95) * 1000.0) if len(walls) else None,
            "switches_by_version": switches,
            "versions_published": len(self.vectors),
        }


# ------------------------------------------------------------------ the episode


def _env_for(scenario: JamScenario, *, seed: int, duration_s: float, arm: JamArm,
             penetration: float, step_length_s: float) -> MainzEnv:
    paths = _scenario_paths(scenario.files)
    gate = arm.arm != "no_control"
    return MainzEnv(seed=seed, duration_s=duration_s, features=HERE_FEATURES,
                    step_length_s=step_length_s, gate_entries=gate,
                    net_file=REPO_ROOT / paths["net"],
                    route_file=REPO_ROOT / paths["gated_routes" if gate else "routes"],
                    segment_file=REPO_ROOT / paths["segments"],
                    schedule_file=REPO_ROOT / paths["schedule"],
                    # No control senses every vehicle and commands none (plan 1.5).
                    penetration=1.0 if not gate else penetration, actuation="max_speed")


def route_successors(scenario: JamScenario) -> dict[str, str]:
    from src.analysis.gains import load_routes

    paths = _scenario_paths(scenario.files)
    return successors(load_routes(REPO_ROOT / paths["gated_routes"]))


def run_jam_episode(model, seed: int, scenario: str | JamScenario, arm: JamArm, *,
                    feed: FeedSetting | None, duration_s: float, penetration: float = 1.0,
                    camera: CameraModel | None = None, predictor: Any = None,
                    step_length_s: float = 0.5, window_start_s: float = 900.0,
                    sense: bool = True, write_log: bool = False,
                    check_max_speed: bool = False, sample_fraction: float = 0.0,
                    aligned_feed: bool = False, record_trace: bool = True) -> dict[str, Any]:
    """One episode; returns its outcome, the engine's counters and the 10 s trace.

    `aligned_feed` publishes at the env's decision instants with zero lag (the published
    arm, used by the equivalence test) instead of replaying `feed`.
    """
    import torch

    from src.rl.src_q import greedy_actions

    scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    env = _env_for(scenario, seed=seed, duration_s=duration_s, arm=arm,
                   penetration=penetration, step_length_s=step_length_s)
    decisions = decision_instants(env.warmup_s, duration_s)
    end_s = decisions[-1] + DECISION_INTERVAL_S if decisions else duration_s
    feed_params = None
    if arm.arm == "no_control":
        mode, process = Mode.no_control(), None
    elif aligned_feed:
        process = FeedProcess.aligned(decisions)
        mode = Mode("rule", Synchronous(), name="aligned_synchronous")
        feed_params = {"rule": "synchronous", "aligned": True, "m": 0.0, "D": None}
    else:
        if feed is None:
            raise ValueError(f"arm {arm.arm} needs a feed setting")
        process, rule, feed_params = feed.process(seed, duration_s + FEED_TAIL_S)
        mode = Mode("rule", rule, name=feed_params["rule"])

    def policy(x: np.ndarray) -> list[int]:
        return greedy_actions(model, torch.tensor(x, dtype=torch.float)).tolist()

    engine = JamEngine(env, policy, mode, process, seed=seed, control_start_s=env.warmup_s,
                       window_start_s=window_start_s, trace_every_s=float(TRACE_EVERY_S),
                       end_s=end_s, decision_instants=decisions, record_trace=False,
                       record_jam=record_trace, arm=arm, camera=camera, successor=route_successors(scenario),
                       q_model=model, predictor=predictor, sense=sense,
                       sample_fraction=sample_fraction, write_log=write_log,
                       check_max_speed=check_max_speed)
    rewards, total_no_control = [], 0.0
    started = time.perf_counter()
    try:
        env.reset(on_warmup_tick=engine.on_tick)
        for edge, lanes, pos, start, length in scenario.incidents:
            env.schedule_incident(edge, list(lanes), pos, start, length)
        marked = window_start_s <= 0.0
        while True:
            if arm.arm == "no_control":
                env.release()
            env.advance_ticks(DECISION_INTERVAL_S, on_tick=engine.on_tick)
            reward = env.reward()
            if arm.arm == "no_control":
                total_no_control += float(reward.mean())
            else:
                rewards.append(torch.tensor(reward, dtype=torch.float))
            if not marked and env.time_s() >= window_start_s:
                env.mark_window()
                marked = True
            if env.time_s() >= duration_s:
                break
        metrics = env.metrics()
        incidents = env.incident_report()
    finally:
        env.close()
    episode_return = (total_no_control if arm.arm == "no_control"
                      else float(torch.stack(rewards).mean(dim=1).sum()))
    return {
        "scenario": scenario.name, "scenario_spec": scenario.describe(), "seed": seed,
        "penetration": penetration, "duration_s": duration_s,
        "flow": metrics["flow_veh_per_h"], "arrived": metrics["arrived"],
        "return": episode_return, "space_mean_speed_kmh": metrics["space_mean_speed_kmh"],
        "held_at_entry": metrics["held_at_entry"], "pending": metrics["waiting_to_enter"],
        "running": metrics["running"], "admitted": metrics["admitted"],
        "actuation": metrics["actuation"],
        "colliding_vehicles": metrics["colliding_vehicles"],
        "collision_teleports": metrics["collision_teleports"],
        "in_teleport": metrics["in_teleport"],
        # The session's `L_bar` (which `D` is built from) stays at the top; the replayed
        # window's own description, whose `L_bar` is its largest replayed lag, is nested.
        "feed": ({**feed_params, "wraps": (process.meta.get("wraps", 0)
                                           if process is not None else 0),
                  "replay": process.describe() if process is not None else None}
                 if feed_params is not None else None),
        "incidents": incidents,
        "episode_wall_s": time.perf_counter() - started,
        **engine.jam_summary(),
        "trace": engine.jam_trace,
        "write_log": engine.writes_log if write_log else None,
        "samples": engine.samples if sample_fraction > 0.0 else None,
    }

