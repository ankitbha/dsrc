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

**The boundary** (approach plan 2026-10-17). `GATED` is the boundary every result above
was produced on. `approach_boundary()` runs the same scenario on `mainz_approach.net.xml`:
every vehicle is inserted by SUMO at the far end of a 3 km road in front of its entry link,
no arm is gated, and every scenario time is `W` later (`run_jam_episode` adds it in one
place). The Q-network still reads the original 12 x 5 state; a vehicle on an approach
super-segment is shown its entry segment's plan (`plan_row`), the lookahead reads the
approach feed rows along the vehicle's downstream chain, and the camera, the oracle and
the corrections act on approach vehicles as on any other.

**The feed gate** (2026-10-20). With `JamArm.gate_J` set, the rule (or the oracle) may switch
a vehicle on only while the feed version that vehicle applies has `gate_signal >= gate_J`:
the largest jam factor over the network's 12 super-segments, measured against one step
below that version's plan (`correction.advised_jam_factor`). Every vehicle that applies the
same version sees the same gate. A vehicle already on when its gate closes is released
through the rule's hysteresis. `gate_J = None` is the rule every earlier result ran.

**Queue release, camera gate, camera alone, driver response** (plan 2026-10-25). The rule
and the oracle take `release_discharge` (off while the road ahead is a queue emptying) and
`camera_gate` (the vehicle's own occupancy also opens its feed gate). `no_plan` runs the rule
on a plan of 1.0 everywhere, the camera layer alone. `run_jam_episode(driver=...)` puts a
`DriverResponse` between the write path and SUMO: the shown speed reaches the cap after a
reaction time, offset by the driver's compliance, and a lower cap is approached at a bounded
deceleration. Every default is the arm every earlier result ran.
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
    LearnedCorrection, LookaheadRule, RuleCorrection, TakeTheSlower, gate_signal,
    shown_fraction, step_down,
)
from src.sumo.driver import DriverResponse, DriverSetting
from src.sumo.feed import FeedProcess, Synchronous, VersionKeyed, _sha256
from src.sumo.feed_episode import FeedEngine, Mode, _scenario_paths, decision_instants
from src.sumo.mainz import (
    DATA, DECISION_INTERVAL_S, HERE_FEATURES, REPO_ROOT, SPEED_ACTION_FRACTIONS, MainzEnv,
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
#: The warm-up's end on the gated boundary: control starts and the first decision is here.
CONTROL_START_S: float = 300.0


@dataclass(frozen=True)
class Boundary:
    """Where the scenario's demand enters the simulated road (approach plan section 2.4).

    `W` moves every scenario time later; `approach_file` is the approach boundary's
    metadata, None on the gated boundary.
    """

    name: str
    W: float = 0.0
    approach_file: Path | None = None

    @property
    def is_approach(self) -> bool:
        return self.approach_file is not None

    def files(self, name: str) -> str:
        """The `SCENARIOS` file set a demand runs from on this boundary."""
        return f"{name}_A" if self.is_approach else name

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "W": self.W,
                               "approach_file": (str(self.approach_file)
                                                 if self.approach_file else None)}
        if self.is_approach:
            meta = json.loads(Path(self.approach_file).read_text())
            out.update(approach_sha256=_sha256(Path(self.approach_file)),
                       length_m=meta["length_m"], segment_m=meta["segment_m"],
                       T_A=meta["T_A"], net_sha256=meta["net"]["sha256"])
        return out


GATED = Boundary("gated")
APPROACH_FILE = DATA / "mainz_approach.json"


def approach_boundary(path: Path = APPROACH_FILE) -> Boundary:
    """The approach boundary of `path`, refused unless its `T_A` has been calibrated."""
    meta = json.loads(Path(path).read_text())
    if not meta.get("calibrated"):
        raise SystemExit(f"{path} carries the provisional T_A of the first build pass; run "
                         "calibrate_approach_travel.py and rebuild with --approach-travel")
    W = float(meta["W"])
    if W <= 0.0 or W % 60.0:
        raise SystemExit(f"{path}: W = {W} s is not a positive multiple of 60 s")
    return Boundary("approach", W, Path(path))


@dataclass(frozen=True)
class JamArm:
    arm: str
    J_th: float | None = None
    m_on: float | None = None
    tau_l: float = 10.0
    theta: float | None = None
    oracle_R: float = DEPLOYED_RANGE_M
    refresh: bool = True
    #: The feed gate (2026-10-20): the rule may switch on only while the applied version's
    #: `gate_signal` is at least this. None is no gate, as every earlier result ran.
    gate_J: float | None = None
    #: Plan 2026-10-25: the queue release, the camera's share in opening the feed gate
    #: (kappa, a multiple of k_c), and the rule on a plan of 1.0 (the camera alone).
    release_discharge: bool = False
    camera_gate: float | None = None
    no_plan: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; arms are {ARMS}")
        needed = {"lookahead": "J_th", "rule": "m_on", "oracle": "m_on", "learned": "theta"}
        name = needed.get(self.arm)
        if name and getattr(self, name) is None:
            raise ValueError(f"arm {self.arm} needs {name}")
        if self.gate_J is not None and self.arm not in ("rule", "oracle"):
            raise ValueError(f"the feed gate acts on the rule's switch; arm {self.arm} has none")
        if (self.release_discharge or self.camera_gate is not None
                or self.no_plan) and self.arm not in ("rule", "oracle"):
            raise ValueError(f"the queue release, the camera gate and no_plan act on the rule; "
                             f"arm {self.arm} has none")
        if self.camera_gate is not None and self.gate_J is None:
            raise ValueError("the camera gate opens the feed gate; give gate_J")
        if self.no_plan and self.gate_J is not None:
            raise ValueError("no_plan reads no feed, so it has no feed gate")

    @property
    def sensor(self) -> str:
        """What the arm's decisions read (plan attack point 4)."""
        if self.arm in CAMERA_ARMS:
            return "camera"
        if self.arm == "oracle":
            return "oracle"
        return "none"

    def describe(self) -> dict[str, Any]:
        out = {**asdict(self), "sensor": self.sensor}
        # Left out at their defaults, so every earlier arm describes itself as it did.
        for name, default in (("release_discharge", False), ("camera_gate", None),
                              ("no_plan", False)):
            if out[name] == default:
                del out[name]
        return out


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

    def process(self, seed: int, end_s: float, start_s: float = FEED_START_S):
        """`(FeedProcess, Rule, parameters)` for one episode: the replay wraps when the
        session is shorter than the episode (plan open item 10), each wrapped stamp keeping
        its recorded second of the minute. `start_s` is the replay's base stamp, moved by
        `W` on the approach boundary."""
        params = self.parameters()
        feed = FeedProcess.from_artefact(Path(self.artefact), self.session, seed,
                                         start_s, end_s, allow_wrap=True)
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
    #: The approach boundary has no gate: every trace sample asserts nothing is held.
    assert_no_gate: bool = False
    #: Plan 2026-10-25 section 4: a driver between the shown speed and the cap; None applies
    #: the shown speed at once, as every earlier result did.
    driver: DriverSetting | None = None

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
        # Every super-segment the env scans, the approach ones (12 to 35) included, and the
        # chain of super-segments a vehicle on each passes next (approach plan 4.1).
        self._all_segments = list(getattr(self.env, "all_segments", None) or self.env.segments)
        n = len(self.env.segments)
        self.downstream: list[list[int]] = [list(c) for c in (
            getattr(self.env, "downstream", None)
            or [list(range(s + 1, n)) for s in range(n)])]
        self.has_approach = bool(getattr(self.env, "approach_segments", None))
        #: Per feed version, the approach super-segments' feed rows at its snapshot.
        self.approach_states: dict[int, np.ndarray] = {}
        #: Per feed version, `gate_signal` over the network's super-segments.
        self.gate_signals: dict[int, float] = {}
        self._gate_open_ticks = 0
        self._segment_of = {e: i for i, segment in enumerate(self._all_segments)
                            for e in segment}
        self._sensed: dict[str, Any] | None = None
        self.correction = None
        if arm.arm in ("rule", "oracle"):
            self.correction = RuleCorrection(arm.m_on, arm.tau_l, name=arm.arm,
                                             release_discharge=arm.release_discharge,
                                             camera_gate=arm.camera_gate)
        elif arm.arm == "learned":
            self.correction = LearnedCorrection(self.predictor, arm.theta)
        elif arm.arm == "take_the_slower":
            self.correction = TakeTheSlower(self.q_model)
        self.lookahead = LookaheadRule(arm.J_th) if arm.arm == "lookahead" else None
        self.driver_response = None
        if self.driver is not None:
            self.driver_response = DriverResponse(self.driver, self.seed, self.env._apply,
                                                  self._vehicle_speed)
            self._driver_checked = self._driver_mismatched = 0

    @staticmethod
    def _vehicle_speed(vehicle: str) -> float:
        from src.sumo.binding import _sumo

        return float(_sumo.vehicle.getSpeed(vehicle))

    def _fraction(self, version: int, segment: int) -> float:
        """The plan's fraction, or 1.0 everywhere for the camera layer alone."""
        if self.arm.no_plan:
            return 1.0
        return super()._fraction(version, segment)

    # ------------------------------------------------------------- stage 1 hooks

    def _snapshot(self, t, version, x, taken_at):
        shape = np.shape(x)
        if shape != (len(self.env.segments), len(HERE_FEATURES)):
            raise ValueError(f"the plan reads {len(self.env.segments)} x {len(HERE_FEATURES)} "
                             f"states; the snapshot is {shape}")
        super()._snapshot(t, version, x, taken_at)
        self.states[version] = np.asarray(x, dtype=np.float32)
        # The network's rows only: the approach roads are a boundary, not the network.
        plan = np.asarray(SPEED_ACTION_FRACTIONS, dtype=float)[self.vectors[version]]
        self.gate_signals[version] = gate_signal(np.asarray(x, dtype=float)[:, 2], plan)
        if self.has_approach:
            # Taken at the same tick as `x`, so both describe one instant (approach 4.3).
            self.approach_states[version] = np.asarray(self.env.observe_approaches(),
                                                       dtype=np.float32)

    def _jam_factors(self, version: int) -> np.ndarray:
        """Version `version`'s jam factor on every super-segment the env scans."""
        jam = self.states[version][:, 2]
        if self.has_approach:
            jam = np.concatenate([jam, self.approach_states[version][:, 2]])
        return jam

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
            if self.driver_response is not None:
                self._drive(t)
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
            est = estimate(table, self.camera.observe(table, self._rng_camera, seed=self.seed),
                           self.camera.R_eff)
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
        allow = None
        if self.arm.gate_J is not None:
            allow = np.array([self.gate_signals[int(versions[k])] >= self.arm.gate_J - 1e-12
                              for k in index], dtype=bool)
            self._gate_open_ticks += int(allow.sum())
        if arm == "oracle":
            active = self.correction.update(vids, t, k_true[index], s_true[index], allow=allow)
        elif arm == "rule":
            active = self.correction.update(vids, t, est.k_hat[index], s_hat[index],
                                            allow=allow)
        elif arm == "learned":
            from src.rl.onset_predictor import engine_features

            features = engine_features(self, t, index, table, est, s_hat, v_slope, versions,
                                       segments)
            active = self.correction.update(vids, t, features)
        else:  # take_the_slower
            states = np.stack([self.states[int(versions[k])] for k in index])
            # An approach vehicle's re-run replaces its entry segment's row (approach 4.4).
            rows = np.array([self._row(int(segments[k])) for k in index], dtype=np.int64)
            self.correction.correct(vids, t, states, rows, est.mean_speed[index], plan=plan)
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
                base = self.lookahead.fraction_chain(self._jam_factors(version),
                                                     self.downstream[segment])
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
        if mapping and self.driver_response is not None:
            for vehicle, shown in mapping.items():
                free = self.env._static[edges[vehicle]]["free_flow_kmh"] / 3.6
                self.driver_response.submit(t, vehicle, shown * free)
        elif mapping:
            self.env.command_vehicles(mapping, edges)
            if self.check_max_speed:
                self._check(mapping, edges)

    def _drive(self, t: float) -> None:
        """Move every driver one tick; with `check_max_speed`, read each cap back."""
        present = {vehicle for vehicle, _edge, _segment in self._last_scan}
        written = self.driver_response.advance(t, present)
        if self.check_max_speed and written:
            from src.sumo.binding import _sumo

            for vehicle, cap in written.items():
                self._driver_checked += 1
                if not math.isclose(_sumo.vehicle.getMaxSpeed(vehicle), cap,
                                    rel_tol=1e-9, abs_tol=1e-9):
                    self._driver_mismatched += 1

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
        n = len(self._all_segments)
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
        held = env.held_at_entry()
        if self.assert_no_gate and held != 0:
            raise AssertionError(f"{held} vehicles held at an entry at {t} s on a boundary "
                                 "with no gate")
        sample = {
            "t": t,
            "speed_kmh": [None if math.isnan(v) else round(v, 4) for v in state["speed_kmh"]],
            "density": [round(v, 6) for v in state["density"]],
            "count": state["count"],
            "held": held, "pending": env.pending(), "running": env.running(),
            "arrived": env.arrived_total,
            "colliding_vehicles": getattr(env, "colliding_vehicles", 0),
            "collision_teleports": getattr(env, "collision_teleports", 0),
            "k_hat_mean": k_hat_mean, "k_true_mean": k_true_mean,
            "plan_mean": per_segment(plan_sum, counted),
            "shown_mean": per_segment(shown_sum, counted),
            "active_share": per_segment(on_sum, counted),
        }
        if self.has_approach:
            # Approach plan sections 5.1 and 5.4: vehicles on the approaches, each
            # approach's queue from its entry and its slow vehicles near the far end.
            sample.update(env.approach_report())
        self.jam_trace.append(sample)

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
            "camera_pair_sigma_log": (self.camera.pair_sigma_log if self.camera is not None
                                      else None),
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
            "gate_J": self.arm.gate_J,
            "driver": ({**self.driver_response.describe(),
                        "changes": self.driver_response.changes,
                        "caps_written": self.driver_response.applied,
                        "caps_checked": self._driver_checked,
                        "caps_mismatched": self._driver_mismatched}
                       if self.driver_response is not None else None),
            "gate_open_share": ((self._gate_open_ticks / c.vehicle_ticks)
                                if self.arm.gate_J is not None and c.vehicle_ticks else None),
            "gate_signal_by_version": {str(k): round(v, 4)
                                       for k, v in sorted(self.gate_signals.items())},
        }


# ------------------------------------------------------------------ the episode


def _env_for(scenario: JamScenario, *, seed: int, duration_s: float, arm: JamArm,
             penetration: float, step_length_s: float,
             boundary: Boundary = GATED) -> MainzEnv:
    """The env of one episode. On the gated boundary every arm but no control is gated; on
    the approach boundary no arm is (approach plan section 3): SUMO inserts every vehicle
    from the full route file at the far end of its approach."""
    paths = _scenario_paths(boundary.files(scenario.files))
    gate = arm.arm != "no_control" and not boundary.is_approach
    approach = None
    if boundary.is_approach:
        approach = REPO_ROOT / paths["approach"]
        if approach.resolve() != Path(boundary.approach_file).resolve():
            raise ValueError(f"{scenario.name}'s files name {approach}, the boundary "
                             f"{boundary.approach_file}")
    return MainzEnv(seed=seed, duration_s=duration_s, warmup_s=CONTROL_START_S + boundary.W,
                    features=HERE_FEATURES,
                    step_length_s=step_length_s, gate_entries=gate,
                    net_file=REPO_ROOT / paths["net"],
                    route_file=REPO_ROOT / paths["gated_routes" if gate else "routes"],
                    segment_file=REPO_ROOT / paths["segments"],
                    schedule_file=REPO_ROOT / paths["schedule"],
                    # No control senses every vehicle and commands none (plan 1.5).
                    penetration=1.0 if arm.arm == "no_control" else penetration,
                    actuation="max_speed", approach_file=approach)


def route_successors(scenario: JamScenario, boundary: Boundary = GATED) -> dict[str, str]:
    from src.analysis.gains import load_routes

    paths = _scenario_paths(boundary.files(scenario.files))
    return successors(load_routes(REPO_ROOT / paths["gated_routes"]))


def run_jam_episode(model, seed: int, scenario: str | JamScenario, arm: JamArm, *,
                    feed: FeedSetting | None, duration_s: float, penetration: float = 1.0,
                    camera: CameraModel | None = None, predictor: Any = None,
                    step_length_s: float = 0.5, window_start_s: float = 900.0,
                    sense: bool = True, write_log: bool = False,
                    check_max_speed: bool = False, sample_fraction: float = 0.0,
                    aligned_feed: bool = False, record_trace: bool = True,
                    boundary: Boundary = GATED,
                    driver: DriverSetting | None = None,
                    on_engine: Callable[[JamEngine], None] | None = None) -> dict[str, Any]:
    """One episode; returns its outcome, the engine's counters and the 10 s trace.

    `aligned_feed` publishes at the env's decision instants with zero lag (the published
    arm, used by the equivalence test) instead of replaying `feed`.

    `duration_s` and `window_start_s` are on the gated clock. `boundary.W` is added here
    and nowhere else (approach plan section 2.4): to the cap, the window, the warm-up
    (control start and first decision), the feed replay's base stamp and every incident's
    request time. The trace stays on SUMO's clock. `on_engine`, when given, is handed the
    engine once the episode ends (tests and validators read its write path from it).
    """
    import torch

    from src.rl.src_q import greedy_actions

    scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    W = boundary.W
    cap_s = duration_s + W
    window_s = window_start_s + W
    env = _env_for(scenario, seed=seed, duration_s=cap_s, arm=arm,
                   penetration=penetration, step_length_s=step_length_s, boundary=boundary)
    decisions = decision_instants(env.warmup_s, cap_s)
    end_s = decisions[-1] + DECISION_INTERVAL_S if decisions else cap_s
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
        process, rule, feed_params = feed.process(seed, cap_s + FEED_TAIL_S,
                                                  start_s=FEED_START_S + W)
        mode = Mode("rule", rule, name=feed_params["rule"])

    def policy(x: np.ndarray) -> list[int]:
        return greedy_actions(model, torch.tensor(x, dtype=torch.float)).tolist()

    engine = JamEngine(env, policy, mode, process, seed=seed, control_start_s=env.warmup_s,
                       window_start_s=window_s, trace_every_s=float(TRACE_EVERY_S),
                       end_s=end_s, decision_instants=decisions, record_trace=False,
                       record_jam=record_trace, arm=arm, camera=camera,
                       successor=route_successors(scenario, boundary),
                       q_model=model, predictor=predictor, sense=sense,
                       sample_fraction=sample_fraction, write_log=write_log,
                       check_max_speed=check_max_speed, assert_no_gate=boundary.is_approach,
                       driver=driver)
    rewards, total_no_control = [], 0.0
    started = time.perf_counter()
    try:
        env.reset(on_warmup_tick=engine.on_tick)
        for edge, lanes, pos, start, length in scenario.incidents:
            env.schedule_incident(edge, list(lanes), pos, start + W, length)
        marked = window_s <= 0.0
        while True:
            if arm.arm == "no_control":
                env.release()
            env.advance_ticks(DECISION_INTERVAL_S, on_tick=engine.on_tick)
            reward = env.reward()
            if arm.arm == "no_control":
                total_no_control += float(reward.mean())
            else:
                rewards.append(torch.tensor(reward, dtype=torch.float))
            if not marked and env.time_s() >= window_s:
                env.mark_window()
                marked = True
            if env.time_s() >= cap_s:
                break
        metrics = env.metrics()
        incidents = env.incident_report()
    finally:
        env.close()
    if on_engine is not None:
        on_engine(engine)
    episode_return = (total_no_control if arm.arm == "no_control"
                      else float(torch.stack(rewards).mean(dim=1).sum()))
    boundary_fields = ({"boundary": boundary.describe(), "W": W,
                        "gate_entries": env.gate_entries, "sim_end_s": cap_s}
                       if boundary.is_approach else {})
    return {
        "scenario": scenario.name, "scenario_spec": scenario.describe(), "seed": seed,
        "penetration": penetration, "duration_s": duration_s, **boundary_fields,
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

