#!/usr/bin/env python3
"""Run the jam-avoidance experiment blocks G, T, S, M, L, P and K, with resume and pause.

    .venv/bin/python scripts/run_jam_experiments.py --block G
    .venv/bin/python scripts/run_jam_experiments.py --block G --g-grid --resume
    .venv/bin/python scripts/run_jam_experiments.py --block T
    .venv/bin/python scripts/run_jam_experiments.py --block T --dagger --resume
    .venv/bin/python scripts/run_jam_experiments.py --block S
    .venv/bin/python scripts/run_jam_experiments.py --block M L P K
    .venv/bin/python scripts/run_jam_experiments.py --block G --summarize-only
    .venv/bin/python scripts/run_jam_experiments.py --block G T S M L P K --smoke

Plan 2026-10-16 sections 8.1 to 8.3. Every episode is keyed `(block, arm, variant,
scenario, seed)`. One JSONL per block under `results/jam/{block}.jsonl`, appended and
flushed after every episode, with a header before the episodes and a footer after them;
traces under `outputs/jam/traces/{block}/`, block T's feature samples under
`outputs/jam/training/`. The summary `results/jam/{block}_summary.json` is rebuilt from
the JSONL at the end of a run, or alone with `--summarize-only`.

* The feed process is the measured afternoon artefact,
  `results/jam/feed_process_afternoon.json`, session `afternoon_20261016_mainz` (the
  orchestrator's decision at commit); the drive-log placeholder runs only with
  `--allow-placeholder`, and the flag travels into every header and summary.
* The plan is the frozen `results/checkpoints/mainz_here_best.pt` (J1).
* Each block runs on its own seeds (G 1-5, T 1-10, S 11-15, M to K 16-30) and refuses any
  other; every arm of a block runs the same seed list.
* `--resume` skips every key already present and refuses to continue a block whose earlier
  headers ran another checkpoint, feed artefact, camera model or predictor.
* Between episodes the runner checks the pause file (default `results/jam/PAUSE`). When it
  exists it writes `results/jam/{block}_state.json` naming the next key and exits 0 with
  `paused before <key>`.
* Blocks M to K read every threshold and the best model from `results/jam/S_summary.json`
  and nowhere else, record its sha256, and refuse one built on another feed artefact,
  camera model or predictor.
* `--smoke` runs one 600 s episode per arm of each named block on seed 1 into
  `outputs/jam/smoke/`, to prove the entry point runs end to end. It is not an experiment
  and never writes into `results/jam/`.

Block F, the frame re-processing, is `scripts/process_drive_frames.py`, not this runner.
Sequential, one process, one SUMO instance at a time: libsumo is process-global.

**The approach boundary** (approach plan 2026-10-17 section 7), `--boundary approach`:

    .venv/bin/python scripts/run_jam_experiments.py --boundary approach --block Q
    .venv/bin/python scripts/run_jam_experiments.py --boundary approach --block M L P K
    .venv/bin/python scripts/run_jam_experiments.py --boundary approach --block Q M L P K --smoke

Every scenario runs from its `_A` files on `mainz_approach.net.xml` with no gate in any arm
and every time `W` later. Block Q (training seeds 1-2) qualifies the boundary before block
M; blocks G, T and S are refused, since nothing is re-selected: the thresholds and the best
model are the committed `results/jam/S_summary.json`, whose sha256 must be
`APPROACH_SELECTION_SHA256`. Results go to `outputs/jam_approach/run/` (pause file
`outputs/jam_approach/run/PAUSE`), traces to `outputs/jam_approach/traces/`, a smoke to
`outputs/jam_approach/smoke/`; nothing is written under `results/jam/`. Each record carries
the jam metrics over all 36 super-segments (`jam`), the original 12 (`jam_network`) and the
24 approach ones (`jam_approach`), the two storage measures and the approach-short flag.

**The network is what is measured** (2026-10-20). The approach roads are a boundary: they
pace the demand into the network and are not part of it. On the approach boundary a
record's `jam` and `jam_30kmh` are over the network's 12 super-segments only, and
`storage.network_min` is the vehicle-minutes spent on the network; `jam_all` (36) and
`jam_approach` (24) stay in the record as diagnostics and enter no result. Block N
(selection seeds 11-15) selects the feed gate's threshold for the rule:

    .venv/bin/python scripts/run_jam_experiments.py --boundary approach --block N
    .venv/bin/python scripts/run_jam_experiments.py --boundary approach --block M L P K \
        --gate-selection <results-dir>/N_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluate_mainz_checkpoints import _provenance  # noqa: E402
from train_mainz_src import SCENARIOS as FILE_SETS  # noqa: E402

BLOCKS = ("G", "T", "S", "M", "L", "P", "K", "Q", "N", "N2", "D")
BLOCK_SEEDS: dict[str, tuple[int, ...]] = {
    "G": tuple(range(1, 6)), "T": tuple(range(1, 11)), "S": tuple(range(11, 16)),
    "M": tuple(range(16, 31)), "L": tuple(range(16, 31)), "P": tuple(range(16, 31)),
    "K": tuple(range(16, 31)), "Q": (1, 2), "N": tuple(range(11, 16)),
    "N2": tuple(range(11, 16)), "D": tuple(range(16, 31)),
}
#: Blocks that select or train (G, T, S) run on the gated boundary only, and block Q, the
#: approach boundary's qualification, on the approach boundary only.
GATED_ONLY_BLOCKS: tuple[str, ...] = ("G", "T", "S")
APPROACH_ONLY_BLOCKS: tuple[str, ...] = ("Q", "N", "N2", "D")
#: Blocks that read the thresholds and the best model from a selection.
SELECTED_BLOCKS: tuple[str, ...] = ("M", "L", "P", "K", "Q", "N", "N2", "D")
#: The committed selection the approach boundary runs (approach plan section 4.5).
APPROACH_SELECTION = REPO_ROOT / "results" / "jam" / "S_summary.json"
APPROACH_SELECTION_SHA256 = "5bce02db2330c63ac873c306988c6c961ef59dadb1a9703b65da9f07c016dc05"
APPROACH_DIRS = {"results": "outputs/jam_approach/run", "traces": "outputs/jam_approach/traces",
                 "training": "outputs/jam_approach/training"}
APPROACH_SMOKE_DIRS = {"results": "outputs/jam_approach/smoke/results",
                       "traces": "outputs/jam_approach/smoke/traces",
                       "training": "outputs/jam_approach/smoke/training"}
THRESHOLDS_NOTE = ("The thresholds and the best model were selected on the gated boundary "
                   "(results/jam/S_summary.json, sha256 5bce02db...); nothing was re-selected "
                   "on the approach boundary except the rule's feed gate, which block N "
                   "selects on seeds 11-15 there.")
SELECTION_SEEDS = BLOCK_SEEDS["S"]
DEFAULT_FEED_ARTEFACT = REPO_ROOT / "results" / "jam" / "feed_process_afternoon.json"
DEFAULT_FEED_SESSION = "afternoon_20261016_mainz"
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt"
DEFAULT_CAMERA = REPO_ROOT / "results" / "jam" / "camera_error_model.json"
DEFAULT_PREDICTOR = REPO_ROOT / "results" / "jam" / "onset_predictor.json"
#: Block T's DAgger round runs the round-1 model (section 5.5).
ROUND1_PREDICTOR = REPO_ROOT / "results" / "jam" / "onset_predictor_round1.json"
DEFAULT_WEIGHTS_RECORD = REPO_ROOT / "results" / "jam" / "weights.json"
SMOKE_SEED = 1
SMOKE_DURATION_S = 600.0
SMOKE_GUARD = {"results": "results/jam", "traces": "outputs/jam/traces"}
#: Section 4.3's precursor baseline.
BASELINE_LEADS: tuple[float, float] = (600.0, 900.0)
#: Section 4.2 and J21.
PERTURBATIONS = ("B5", "B7.5", "B9", "I_exit", "I_mid")
TRAINING_SCENARIOS = ("B5", "B7.5", "B9", "I_exit", "I_mid", "BG")
SHORT_S = 3600.0
LONG_S = 7200.0
S45_S = 2500.0
#: Grids (sections 5.2 to 5.5, 8.1).
J_TH = (2.0, 4.0, 6.0)
M_ON = (0.0, 0.1, 0.2)
THETA = (0.3, 0.45, 0.6, 0.75)
G_TAU = (10.0, 30.0)
G_BASE = {"tau_l": 10.0, "m_on": 0.1}
LAGS_L = (0.0, 2.0, 4.0)  # Amendment 1
PENETRATIONS_P = (0.75, 0.5)
BIASES_K = (0.8, 1.25)
#: Amendment 3 (R2-4): block K also runs the best model at half the stored pair spread.
PAIR_SIGMA_SCALES_K = (0.5,)
SAMPLE_FRACTION = 0.05
DAGGER_THETA = 0.5
WIDE_R = 200.0
#: Section 5.7: candidates within this C of each other are ordered by BG served flow.
C_TIE = 0.02
#: Block N (2026-10-20): the feed gate's thresholds on `gate_signal`, the largest jam
#: factor over the network against one step below the plan. The rule without a gate is the
#: fifth candidate.
GATE_GRID = (0.5, 1.0, 1.5, 2.0)
#: Block N2 (plan 2026-10-25 section 2): the queue release and the camera gate, at the feed
#: gate block N ranked first among its gated candidates. Each entry is the rule's extra
#: parameters on top of its m_on.
N2_GATE_J = 0.5
MECHANISM_GRID: tuple[dict[str, Any], ...] = (
    {"release_discharge": True},
    {"release_discharge": True, "gate_J": N2_GATE_J},
    {"release_discharge": True, "gate_J": N2_GATE_J, "camera_gate": 1.0},
    {"release_discharge": True, "gate_J": N2_GATE_J, "camera_gate": 0.8},
    {"gate_J": N2_GATE_J, "camera_gate": 1.0},
    {"gate_J": N2_GATE_J, "camera_gate": 0.8},
)
#: Block N2's eligibility: with no trigger a candidate serves at least plan alone's flow
#: minus this share of no control's.
N2_BG_ALLOWANCE = 0.01
#: Block D's driver settings (`src.sumo.driver.DRIVERS`).
DRIVER_SETTINGS = ("nominal", "lax")


# --------------------------------------------------------------------- specs


@dataclass(frozen=True)
class Spec:
    block: str
    arm: str
    variant: str
    scenario: str
    seed: int
    duration_s: float
    m: float = 1.0
    penetration: float = 1.0
    bias: float = 1.0
    params: tuple = ()           # sorted (name, value) pairs of the arm's parameters
    sample_fraction: float = 0.0
    #: Block K (Amendment 3, R2-4): the camera's persistent pair spread as a share of the
    #: camera model's stored one.
    pair_sigma_scale: float = 1.0
    #: Block D (plan 2026-10-25 section 4): a `DRIVERS` setting, or None for the shown
    #: speed applied at once.
    driver: str | None = None

    @property
    def key(self) -> str:
        return f"{self.block}/{self.arm}/{self.variant}/{self.scenario}/{self.seed}"

    def trace_name(self) -> str:
        return f"{self.arm}_{self.variant}_{self.scenario}_{self.seed}.json"

    def jam_arm(self):
        from src.sumo.jam_episode import JamArm

        return JamArm(self.arm, **dict(self.params))


def _mechanism(params: dict[str, Any]) -> str:
    """The queue release, feed gate, camera gate and camera-alone suffixes, in that order;
    empty for the rule every earlier result ran."""
    out = "_r" if params.get("release_discharge") else ""
    if params.get("gate_J") is not None:
        out += f"_g{params['gate_J']:g}"
    if params.get("camera_gate") is not None:
        out += f"_c{params['camera_gate']:g}"
    if params.get("no_plan"):
        out += "_local"
    return out


def _variant(arm: str, params: dict[str, Any]) -> str:
    if arm == "lookahead":
        return f"J{params['J_th']:g}"
    if arm == "rule":
        return f"m{params['m_on']:g}" + _mechanism(params)
    if arm == "learned":
        return f"th{params['theta']:g}"
    if arm == "oracle":
        base = f"tau{params.get('tau_l', 10.0):g}_m{params['m_on']:g}"
        return (base + (f"_R{params['oracle_R']:g}" if params.get("oracle_R", 80.0) != 80.0 else "")
                + _mechanism(params))
    return "-"


def _duration(block: str, scenario: str) -> float:
    if block in ("G", "T", "S", "N", "N2"):
        return SHORT_S
    if scenario == "BG":
        return SHORT_S
    if scenario == "S45":
        return S45_S
    return LONG_S


@dataclass
class Selection:
    """What blocks M to K read from `S_summary.json` (section 5.7)."""

    J_th: float
    m_on: float
    theta: float
    best_arm: str
    best_params: dict[str, float]
    path: str | None = None
    sha256: str | None = None
    smoke_defaults: bool = False
    #: Block N's feed gate threshold for the rule; None runs the rule without a gate.
    gate_J: float | None = None
    gate_path: str | None = None
    gate_sha256: str | None = None
    #: False when `gate_J` is one of block N's candidates run on Ankit's decision rather than
    #: the one block N selected (2026-10-20, `--gate-candidate`).
    gate_selected: bool = True
    #: Block N2's selection (plan 2026-10-25): the rule's queue release and camera gate.
    release_discharge: bool = False
    camera_gate: float | None = None
    mechanism_path: str | None = None
    mechanism_sha256: str | None = None
    #: False when the queue release was added to block N2's selection by
    #: `--release-candidate` (block P only), so every header says it was not selected.
    release_selected: bool = True

    def mechanism_params(self) -> dict[str, Any]:
        """The selected rule's parameters beyond m_on: feed gate, release, camera gate."""
        out: dict[str, Any] = {}
        if self.gate_J is not None:
            out["gate_J"] = self.gate_J
        if self.release_discharge:
            out["release_discharge"] = True
        if self.camera_gate is not None:
            out["camera_gate"] = self.camera_gate
        return out

    @property
    def best_reads_camera(self) -> bool:
        return self.best_arm in ("take_the_slower", "rule", "learned")


def arm_list(block: str, selection: Selection | None, grid: bool = False,
             dagger: bool = False, oracle_window: float | None = None,
             penetrations: Sequence[float] = PENETRATIONS_P
             ) -> list[tuple[str, dict[str, float], dict[str, Any]]]:
    """`(arm, params, extra)` for a block; `extra` holds m, penetration, bias, sampling.

    From block S on, an oracle arm's window is the camera's effective window `min(80 m,
    R_max)` (Amendment 3, R2-5), so the gap between the camera arms and the oracle is
    perception error only; block G's oracle keeps the 80 m it ran with.
    """
    if block == "G":
        settings = ([{"tau_l": t, "m_on": m} for t in G_TAU for m in M_ON] if grid
                    else [G_BASE])
        return [("oracle", s, {}) for s in settings] + [("plan", {}, {})]
    if block == "T":
        if dagger:
            return [("learned", {"theta": DAGGER_THETA}, {"sample_fraction": SAMPLE_FRACTION})]
        return [("plan", {}, {"sample_fraction": SAMPLE_FRACTION})]
    if block == "S":
        return ([("no_control", {}, {}), ("plan", {}, {})]
                + [("lookahead", {"J_th": j}, {}) for j in J_TH]
                + [("take_the_slower", {}, {})]
                + [("rule", {"m_on": m}, {}) for m in M_ON]
                + [("learned", {"theta": th}, {}) for th in THETA])
    if selection is None:
        raise SystemExit(f"block {block} reads its thresholds from S_summary.json; none given")
    if block == "N":
        return ([("no_control", {}, {}), ("plan", {}, {}), ("rule", {"m_on": selection.m_on}, {})]
                + [("rule", {"m_on": selection.m_on, "gate_J": g}, {}) for g in GATE_GRID])
    if block == "N2":
        return ([("no_control", {}, {}), ("plan", {}, {})]
                + [("rule", {"m_on": selection.m_on, **extra}, {}) for extra in MECHANISM_GRID])
    best_params = dict(selection.best_params)
    if selection.best_arm == "rule":
        best_params.update(selection.mechanism_params())
    best = (selection.best_arm, best_params, {})
    if block == "M":
        gated = ([("rule", {"m_on": selection.m_on, "gate_J": selection.gate_J}, {})]
                 if selection.gate_J is not None and selection.mechanism_path is None else [])
        oracle = {"m_on": selection.m_on, "tau_l": 10.0, "oracle_R": _window(oracle_window)}
        arms = [("no_control", {}, {}), ("lookahead", {"J_th": selection.J_th}, {}),
                ("plan", {}, {}), ("take_the_slower", {}, {}),
                ("rule", {"m_on": selection.m_on}, {}), *gated,
                ("learned", {"theta": selection.theta}, {}), ("oracle", oracle, {})]
        if selection.mechanism_path is not None:
            # Plan 2026-10-25: the selected system, the same mechanism on true occupancy,
            # and the camera alone (the system's release, no plan and no gate).
            local = {"m_on": selection.m_on, "no_plan": True}
            if selection.release_discharge:
                local["release_discharge"] = True
            arms += [best, ("oracle", {**oracle, **selection.mechanism_params()}, {}),
                     ("rule", local, {})]
        return _unique(arms)
    if block == "D":
        return [(a, p, {"driver": d}) for d in DRIVER_SETTINGS
                for a, p, _ in _unique([("plan", {}, {}), best])]
    if block == "L":
        arms = [("lookahead", {"J_th": selection.J_th}, {}), ("plan", {}, {}), best]
        return [(a, p, {"m": m}) for m in LAGS_L for a, p, _ in _unique(arms)]
    if block == "P":
        arms = _unique([("plan", {}, {}), best])
        return [(a, p, {"penetration": pen}) for pen in penetrations for a, p, _ in arms]
    if block == "Q":
        # Approach plan section 7.1: every block M arm at full penetration (the oracle at
        # the camera's window only), the plan and the rule at 0.5, and the lookahead at
        # zero lag, which held the most vehicles in the network in block L.
        return (arm_list("M", selection, oracle_window=oracle_window)
                + [("plan", {}, {"penetration": 0.5}),
                   ("rule", {"m_on": selection.m_on}, {"penetration": 0.5}),
                   ("lookahead", {"J_th": selection.J_th}, {"m": 0.0})])
    if block == "K":
        if not selection.best_reads_camera:
            return []
        return ([(best[0], best[1], {"bias": b}) for b in BIASES_K]
                + [(best[0], best[1], {"bias": 1.0, "pair_sigma_scale": scale})
                   for scale in PAIR_SIGMA_SCALES_K])
    raise ValueError(f"unknown block {block}")


def _window(oracle_window: float | None) -> float:
    if oracle_window is None:
        raise SystemExit("from block S on the oracle reads the camera's effective window "
                         "min(80 m, R_max); none was given (oracle_window)")
    return float(oracle_window)


def oracle_window(camera_model: Path | None) -> float:
    """The camera model's `R_eff = min(80 m, R_max)`, the oracle's window from block S on."""
    from src.sumo.sensing import CameraModel

    if camera_model is None or not Path(camera_model).exists():
        raise SystemExit(f"the oracle's window is the camera's; {camera_model} does not exist")
    return float(CameraModel.from_artefact(camera_model).R_eff)


def _unique(arms):
    seen, out = set(), []
    for arm, params, extra in arms:
        key = (arm, tuple(sorted(params.items())))
        if key not in seen:
            seen.add(key)
            out.append((arm, params, extra))
    return out


def _scenarios(block: str) -> tuple[str, ...]:
    if block in ("G",):
        return ("B9", "I_exit")
    if block in ("T", "S", "N", "N2"):
        return TRAINING_SCENARIOS
    if block == "M":
        return PERTURBATIONS + ("BG", "S45")
    if block == "Q":
        return ("B9", "I_exit", "S45")
    return ("B9", "I_exit")


def block_specs(block: str, seeds: Sequence[int] | None = None,
                selection: Selection | None = None, grid: bool = False,
                dagger: bool = False, smoke: bool = False,
                oracle_window: float | None = None,
                penetrations: Sequence[float] | None = None) -> list[Spec]:
    """Every episode of a block, arm-major, every arm over the same seed list.

    Seeds outside the block's own set are refused, except in a smoke run, which is not
    an experiment and runs seed 1 for 600 s. `penetrations` replaces block P's
    PENETRATIONS_P (the 2026-10-21 sweep) and is refused for any other block.
    """
    if penetrations is not None:
        if block != "P":
            raise SystemExit(f"--penetrations sets block P's shares; block {block} has none")
        bad = [p for p in penetrations if not 0.0 < p < 1.0]
        if bad or len(set(penetrations)) != len(penetrations):
            raise SystemExit(f"penetrations must be distinct and in (0, 1); got {penetrations}")
    allowed = BLOCK_SEEDS[block]
    seeds = tuple(seeds) if seeds is not None else allowed
    if smoke:
        seeds = (SMOKE_SEED,)
    else:
        outside = sorted(set(seeds) - set(allowed))
        if outside:
            raise SystemExit(f"block {block} runs on seeds {allowed[0]}-{allowed[-1]}; "
                             f"refusing {outside}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("duplicate seeds")
    specs: list[Spec] = []
    for arm, params, extra in arm_list(block, selection, grid, dagger, oracle_window,
                                       penetrations=penetrations or PENETRATIONS_P):
        variant = _variant(arm, params)
        if "m" in extra:
            variant += f"_lag{extra['m']:g}"
        if "penetration" in extra:
            variant += f"_p{extra['penetration']:g}"
        if "bias" in extra:
            variant += f"_b{extra['bias']:g}"
        if "pair_sigma_scale" in extra:
            variant += f"_ps{extra['pair_sigma_scale']:g}"
        if "driver" in extra:
            variant += f"_drv{extra['driver']}"
        if dagger:
            variant += "_dagger"
        for scenario in _scenarios(block):
            for seed in seeds:
                specs.append(Spec(block, arm, variant, scenario, seed,
                                  SMOKE_DURATION_S if smoke else _duration(block, scenario),
                                  m=extra.get("m", 1.0),
                                  penetration=extra.get("penetration", 1.0),
                                  bias=extra.get("bias", 1.0),
                                  params=tuple(sorted(params.items())),
                                  sample_fraction=extra.get("sample_fraction", 0.0),
                                  pair_sigma_scale=extra.get("pair_sigma_scale", 1.0),
                                  driver=extra.get("driver")))
        if block == "M" and arm == "oracle" and not _mechanism(params):
            wide = {**params, "oracle_R": WIDE_R}
            for scenario in ("B9", "I_exit"):
                for seed in seeds:
                    specs.append(Spec(block, arm, _variant(arm, wide), scenario, seed,
                                      SMOKE_DURATION_S if smoke else LONG_S,
                                      params=tuple(sorted(wide.items()))))
    keys = [s.key for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode keys")
    return specs


def smoke_subset(specs: Sequence[Spec]) -> list[Spec]:
    """One episode per arm: each arm's first variant on the block's first scenario."""
    chosen, seen = [], set()
    for spec in specs:
        if spec.arm in seen:
            continue
        seen.add(spec.arm)
        chosen.append(spec)
    return chosen


# ---------------------------------------------------------------- selection


def read_selection(path: Path, *, feed_sha: str | None, camera_sha: str | None,
                   predictor_sha: str | None, required_sha: str | None = None) -> Selection:
    """Blocks M to K's thresholds and best model; refused when the selection ran on
    another feed artefact, camera model or predictor, or on any seed outside 11-15, and,
    with `required_sha` (the approach boundary), when the file is not that one."""
    raw = Path(path).read_bytes()
    if required_sha is not None and hashlib.sha256(raw).hexdigest() != required_sha:
        raise SystemExit(f"{path} has sha256 {hashlib.sha256(raw).hexdigest()}; the approach "
                         f"boundary runs the committed selection {required_sha} and is never "
                         "re-selected")
    data = json.loads(raw)
    seeds = sorted(set(data.get("seeds") or []))
    if seeds != list(SELECTION_SEEDS):
        raise SystemExit(f"{path} lists seeds {seeds}; a selection is made on seeds 11-15, "
                         "all of them (validation R2-1)")
    if not data.get("best"):
        raise SystemExit(f"{path} selected no best model ({data.get('decision')}); blocks M "
                         "to K need a complete block S")
    for name, expected, recorded in (("feed artefact", feed_sha, data.get("feed_artefact_sha256")),
                                     ("camera model", camera_sha, data.get("camera_model_sha256")),
                                     ("predictor", predictor_sha, data.get("predictor_sha256"))):
        if expected is not None and recorded != expected:
            raise SystemExit(f"{path} was built on another {name} ({recorded}, this run "
                             f"{expected}); blocks M to K must share it")
    selected = data["selected"]
    return Selection(J_th=float(selected["lookahead"]["J_th"]),
                     m_on=float(selected["rule"]["m_on"]),
                     theta=float(selected["learned"]["theta"]),
                     best_arm=data["best"]["arm"], best_params=dict(data["best"]["params"]),
                     path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def read_gate_selection(path: Path, selection: Selection,
                        candidate: float | None = None) -> Selection:
    """`selection` with block N's feed gate threshold (2026-10-20). Refused unless block N
    ran on seeds 11-15, all of them, with the rule's `m_on` of `selection`.

    With `candidate`, the gate is that threshold instead of the selected one. It must be one
    of block N's evaluated candidates, and the selection records `gate_selected` False, so
    every header says the gate run was not the one block N chose."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    seeds = sorted(set(data.get("seeds") or []))
    if data.get("block") != "N" or seeds != list(SELECTION_SEEDS):
        raise SystemExit(f"{path} is not a block N summary on seeds 11-15 (block "
                         f"{data.get('block')}, seeds {seeds})")
    chosen = data.get("selected")
    if not chosen:
        raise SystemExit(f"{path} selected no gate ({data.get('decision')})")
    if float(chosen["m_on"]) != float(selection.m_on):
        raise SystemExit(f"{path} selected a gate for the rule at m_on {chosen['m_on']}; this "
                         f"selection's rule runs m_on {selection.m_on}")
    gate = chosen.get("gate_J")
    selected = True
    if candidate is not None:
        evaluated = sorted(float(v["gate_J"]) for v in (data.get("candidates") or {}).values()
                           if v.get("gate_J") is not None)
        if float(candidate) not in evaluated:
            raise SystemExit(f"--gate-candidate {candidate} is not one of block N's evaluated "
                             f"gates {evaluated}")
        selected = gate is not None and float(gate) == float(candidate)
        gate = float(candidate)
    return replace(selection, gate_J=None if gate is None else float(gate), gate_path=str(path),
                   gate_sha256=hashlib.sha256(raw).hexdigest(), gate_selected=selected)


def read_mechanism_selection(path: Path, selection: Selection) -> Selection:
    """`selection` with block N2's choice (plan 2026-10-25): its feed gate, queue release and
    camera gate replace any block N gate. Refused unless block N2 ran on seeds 11-15 with the
    rule's `m_on` of `selection` and selected a candidate."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    seeds = sorted(set(data.get("seeds") or []))
    if data.get("block") != "N2" or seeds != list(SELECTION_SEEDS):
        raise SystemExit(f"{path} is not a block N2 summary on seeds 11-15 (block "
                         f"{data.get('block')}, seeds {seeds})")
    chosen = data.get("selected")
    if not chosen:
        raise SystemExit(f"{path} selected nothing ({data.get('decision')})")
    if float(chosen["m_on"]) != float(selection.m_on):
        raise SystemExit(f"{path} selected the rule at m_on {chosen['m_on']}; this "
                         f"selection's rule runs m_on {selection.m_on}")
    gate = chosen.get("gate_J")
    camera = chosen.get("camera_gate")
    return replace(selection, gate_J=None if gate is None else float(gate),
                   gate_path=None, gate_sha256=None, gate_selected=True,
                   release_discharge=bool(chosen.get("release_discharge")),
                   camera_gate=None if camera is None else float(camera),
                   mechanism_path=str(path), mechanism_sha256=hashlib.sha256(raw).hexdigest())


def smoke_selection() -> Selection:
    """A smoke run's stand-in: each grid's first value and the rule as best. Chooses
    nothing; it exists so blocks M to K can be run end to end before block S has."""
    return Selection(J_th=J_TH[0], m_on=M_ON[0], theta=THETA[0], best_arm="rule",
                     best_params={"m_on": M_ON[0]}, smoke_defaults=True)


# -------------------------------------------------------------------- header


def _sha256(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sumo_version() -> str | None:
    binary = REPO_ROOT / ".venv" / "bin" / "sumo"
    try:
        result = subprocess.run([str(binary if binary.exists() else "sumo"), "--version"],
                                capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for module in ("libsumo", "torch", "numpy"):
        try:
            out[module] = getattr(__import__(module), "__version__", None)
        except ImportError:
            out[module] = None
    return out


@dataclass
class Context:
    """What a run's specs and episodes depend on; recorded in the header."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    feed_artefact: Path = DEFAULT_FEED_ARTEFACT
    feed_session: str = DEFAULT_FEED_SESSION
    allow_placeholder: bool = False
    camera_model: Path | None = DEFAULT_CAMERA
    predictor: Path | None = DEFAULT_PREDICTOR
    weights_record: Path = DEFAULT_WEIGHTS_RECORD
    selection_path: Path | None = None
    #: Block N's summary, read by blocks M to K and Q on the approach boundary.
    gate_selection_path: Path | None = None
    #: Block N2's summary (plan 2026-10-25), read by blocks M to K and D.
    mechanism_selection_path: Path | None = None
    step_length_s: float = 0.5
    window_start_s: float = 900.0
    smoke: bool = False
    write_log: bool = False
    check_max_speed: bool = False
    training_dir: Path = REPO_ROOT / "outputs" / "jam" / "training"
    selection: Selection | None = None
    feed: dict[str, Any] = field(default_factory=dict)
    #: "gated" or "approach" (approach plan section 7).
    boundary: str = "gated"
    approach_file: Path = REPO_ROOT / "data" / "mainz" / "mainz_approach.json"

    def boundary_object(self):
        from src.sumo.jam_episode import GATED, approach_boundary

        return approach_boundary(self.approach_file) if self.boundary == "approach" else GATED

    def as_dict(self) -> dict[str, Any]:
        out = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()
               if k not in ("selection", "feed")}
        out["selection"] = asdict(self.selection) if self.selection else None
        return out


#: Plan section 1.5: in these blocks the no-control arm senses passively on every vehicle
#: (block M's no-control episodes give the camera's view of uncontrolled jams, `W_cam`).
PASSIVE_SENSING_BLOCKS: tuple[str, ...] = ("M", "T", "Q")
CAMERA_ARMS: tuple[str, ...] = ("take_the_slower", "rule", "learned")


def uses_camera(spec: Spec) -> bool:
    """Whether an episode gets the camera: a camera arm, a sampling episode, or a no-control
    episode where section 1.5 requires passive sensing. Decided per spec, so it does not
    depend on which episode a run reaches first (validation R2-6)."""
    return (spec.arm in CAMERA_ARMS or spec.sample_fraction > 0
            or (spec.arm == "no_control" and spec.block in PASSIVE_SENSING_BLOCKS))


def needs_camera(specs: Sequence[Spec]) -> bool:
    return any(uses_camera(s) for s in specs)


def needs_predictor(specs: Sequence[Spec]) -> bool:
    return any(s.arm == "learned" for s in specs)


def header(block: str, context: Context, specs: Sequence[Spec],
           argv: Sequence[str] | None = None) -> dict[str, Any]:
    from src.analysis.jam_metrics import THRESHOLDS
    from src.sumo.jam_episode import SCENARIOS as JAM_SCENARIOS

    boundary = context.boundary_object()
    files: dict[str, Any] = {}
    for scenario in sorted({s.scenario for s in specs}):
        name = boundary.files(JAM_SCENARIOS[scenario].files)
        for role, relative in FILE_SETS[name].items():
            files[f"{name}:{role}"] = {"path": relative, "sha256": _sha256(REPO_ROOT / relative)}
    camera = context.camera_model if needs_camera(specs) else None
    predictor = context.predictor if needs_predictor(specs) else None
    weights = None
    if camera is not None and Path(context.weights_record).exists():
        weights = json.loads(Path(context.weights_record).read_text()).get("sha256")
    return {
        "type": "header", "block": block, "started_wall": time.time(), **_provenance(),
        "sumo_version": _sumo_version(), **{f"{k}_version": v for k, v in _versions().items()},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "command": list(argv) if argv is not None else sys.argv,
        "config": context.as_dict(),
        "checkpoint": {"path": str(context.checkpoint), "sha256": _sha256(context.checkpoint)},
        "feed_artefact": context.feed,
        "camera_model": ({"path": str(camera), "sha256": _sha256(camera)} if camera else None),
        "predictor": ({"path": str(predictor), "sha256": _sha256(predictor)}
                      if predictor else None),
        "weights_sha256": weights,
        "selection": ({"path": context.selection.path, "sha256": context.selection.sha256,
                       "smoke_defaults": context.selection.smoke_defaults,
                       "gate_J": context.selection.gate_J,
                       "gate_path": context.selection.gate_path,
                       "gate_sha256": context.selection.gate_sha256,
                       "gate_selected": context.selection.gate_selected,
                       "release_discharge": context.selection.release_discharge,
                       "camera_gate": context.selection.camera_gate,
                       "mechanism_path": context.selection.mechanism_path,
                       "release_selected": context.selection.release_selected,
                       "mechanism_sha256": context.selection.mechanism_sha256}
                      if context.selection else None),
        "drain_thresholds": {"path": str(THRESHOLDS), "sha256": _sha256(THRESHOLDS)},
        "scenario_files": files, "episodes_planned": len(specs),
        **(approach_header(boundary) if boundary.is_approach else {}),
    }


def approach_header(boundary) -> dict[str, Any]:
    """What an approach-boundary header adds (approach plan section 7.3)."""
    meta = json.loads(Path(boundary.approach_file).read_text())
    data = REPO_ROOT / "data" / "mainz"
    return {"boundary": boundary.name, "W": boundary.W,
            "approach": {"file": str(boundary.approach_file),
                         "approach_sha256": _sha256(boundary.approach_file),
                         "net": meta["net"]["path"], "net_sha256": _sha256(REPO_ROOT / meta["net"]["path"]),
                         "segments_sha256": _sha256(REPO_ROOT / meta["segments_file"]),
                         "travel_sha256": _sha256(data / "mainz_approach_travel.json"),
                         "length_m": meta["length_m"], "segment_m": meta["segment_m"],
                         "T_A": meta["T_A"], "W": meta["W"], "calibrated": meta["calibrated"],
                         "demand": meta["demand"]},
            "thresholds_note": THRESHOLDS_NOTE}


def header_identity(record: dict) -> tuple:
    """`(checkpoint, feed artefact, camera model, predictor)` sha256 a header ran, and on
    the approach boundary the approach net's sha256 as a fifth element. A gated header
    keeps its four, so the committed gated files keep their identity, and a resume never
    mixes the two boundaries in one JSONL."""
    identity = ((record.get("checkpoint") or {}).get("sha256"),
                (record.get("feed_artefact") or {}).get("artefact_sha256"),
                (record.get("camera_model") or {}).get("sha256"),
                (record.get("predictor") or {}).get("sha256"))
    approach = (record.get("approach") or {}).get("net_sha256")
    return identity + (approach,) if approach else identity


# ------------------------------------------------------------------- running


def resume_keys(jsonl: Path) -> set[str]:
    if not jsonl.exists():
        return set()
    keys = set()
    for line in jsonl.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a line cut short by a stop mid-write is not a finished episode
        if record.get("type") == "episode":
            keys.add(record["key"])
    return keys


def read_block(results_dir: Path, block: str) -> tuple[list[dict], list[dict]]:
    """`(headers, episodes)`; a later record of a key replaces an earlier one."""
    jsonl = results_dir / f"{block}.jsonl"
    headers, episodes = [], {}
    if not jsonl.exists():
        return headers, []
    for line in jsonl.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") == "header":
            headers.append(record)
        elif record.get("type") == "episode":
            episodes[record["key"]] = record
    return headers, list(episodes.values())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, default=float))
    temporary.rename(path)


EpisodeFn = Callable[[Spec], dict[str, Any]]
#: Fields of an episode result kept in the trace file rather than the JSONL.
TRACE_FIELDS = ("trace", "write_log")


def run_block(block: str, specs: Sequence[Spec], run_episode: EpisodeFn, *,
              results_dir: Path, traces_dir: Path, header_record: dict[str, Any],
              resume: bool = False, pause_file: Path | None = None,
              training_dir: Path | None = None) -> dict[str, Any]:
    """Run the specs not yet in the block's JSONL, one at a time, pausing on request."""
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl = results_dir / f"{block}.jsonl"
    if jsonl.exists() and not resume:
        raise SystemExit(f"{jsonl} exists; pass --resume to continue it, or move it away")
    identity = header_identity(header_record)
    if jsonl.exists():
        earlier = {header_identity(h) for h in read_block(results_dir, block)[0]}
        if earlier - {identity}:
            raise SystemExit(f"refusing to resume {jsonl}: its headers ran (checkpoint, feed, "
                             f"camera, predictor) {sorted(earlier, key=str)}, this run "
                             f"{identity}")
    done = resume_keys(jsonl)
    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode keys")
    ran, skipped = [], [k for k in keys if k in done]
    with jsonl.open("a") as handle:
        handle.write(json.dumps({**header_record, "resumed": bool(done)}) + "\n")
        handle.flush()
        for spec in specs:
            if spec.key in done:
                continue
            if pause_file is not None and Path(pause_file).exists():
                state = {"block": block, "next_key": spec.key, "completed": len(done) + len(ran),
                         "total": len(specs), "paused_wall": time.time()}
                _write_json(results_dir / f"{block}_state.json", state)
                print(f"paused before {spec.key}")
                return {"status": "paused", "next_key": spec.key, "ran": ran,
                        "skipped": skipped}
            started = time.time()
            result = run_episode(spec)
            wall = time.time() - started
            trace_path = traces_dir / block / spec.trace_name()
            _write_json(trace_path, {"key": spec.key, **{k: result.get(k) for k in TRACE_FIELDS}})
            samples_path = None
            if result.get("samples") is not None and training_dir is not None:
                samples_path = training_dir / block / spec.trace_name()
                _write_json(samples_path, {"key": spec.key, "samples": result["samples"]})
            # The spec's own fields come last, so nothing in the result can replace them.
            record = {"type": "episode",
                      **{k: v for k, v in result.items()
                         if k not in TRACE_FIELDS and k != "samples"},
                      "key": spec.key, "block": block, "arm": spec.arm,
                      "variant": spec.variant, "scenario": spec.scenario, "seed": spec.seed,
                      "duration_s": spec.duration_s, "m": spec.m,
                      "penetration": spec.penetration, "bias": spec.bias,
                      "pair_sigma_scale": spec.pair_sigma_scale,
                      **({"driver_setting": spec.driver} if spec.driver is not None else {}),
                      "params": dict(spec.params), "wall_s": wall,
                      "trace_file": str(trace_path),
                      "samples_file": str(samples_path) if samples_path else None,
                      "identity": list(identity)}
            handle.write(json.dumps(record, default=float) + "\n")
            handle.flush()
            ran.append(spec.key)
        provenance_end = _provenance()
        handle.write(json.dumps({
            "type": "footer", "ended_wall": time.time(), "provenance_end": provenance_end,
            "tree_unchanged_during_run": (
                provenance_end.get("commit") == header_record.get("commit")
                and provenance_end.get("dirty") == header_record.get("dirty"))}) + "\n")
    state_path = results_dir / f"{block}_state.json"
    if state_path.exists():
        state_path.unlink()
    return {"status": "complete", "ran": ran, "skipped": skipped}


def episode_runner(model, context: Context) -> EpisodeFn:
    from src.analysis.jam_metrics import episode_metrics, load_v_jam
    from src.sumo.jam_episode import SCENARIOS as JAM_SCENARIOS
    from src.sumo.jam_episode import FeedSetting

    v_jam, _ = load_v_jam()
    boundary = context.boundary_object()
    camera = predictor = None

    def run(spec: Spec) -> dict[str, Any]:
        nonlocal camera, predictor
        arm = spec.jam_arm()
        if uses_camera(spec):
            if camera is None:
                from src.sumo.sensing import CameraModel

                if context.camera_model is None or not Path(context.camera_model).exists():
                    raise SystemExit(f"{spec.key} reads the camera; {context.camera_model} does "
                                     "not exist (run block F and fit_camera_model.py first)")
                camera = CameraModel.from_artefact(context.camera_model)
        if spec.arm == "learned" and predictor is None:
            from src.rl.onset_predictor import load

            if context.predictor is None or not Path(context.predictor).exists():
                raise SystemExit(f"{spec.key} runs the learned arm; {context.predictor} does not "
                                 "exist (run block T and train_onset_predictor.py first)")
            predictor = load(context.predictor)
        if spec.arm == "learned" and uses_camera(spec):
            # Validation R3-1: the predictor's features are camera features, so it runs only
            # with the camera model it was trained on (block T's DAgger round included).
            trained = predictor.card.get("camera_model_sha256")
            running = (camera.source or {}).get("sha256")
            if trained != running:
                raise SystemExit(f"{spec.key}: the predictor {context.predictor} was trained on "
                                 f"camera model {trained}, but this run's camera "
                                 f"{context.camera_model} is {running}")
        feed = FeedSetting(context.feed_artefact, context.feed_session, m=spec.m,
                           allow_placeholder=context.allow_placeholder)
        scenario = JAM_SCENARIOS[spec.scenario]
        spec_camera = None
        if uses_camera(spec):
            spec_camera = camera.with_bias(spec.bias).with_pair_sigma(
                camera.pair_sigma_log * spec.pair_sigma_scale)
        # Looked up at call time, so a test can stand in for the episode.
        from src.sumo import jam_episode

        kwargs = {"boundary": boundary} if boundary.is_approach else {}
        if spec.driver is not None:
            from src.sumo.driver import DRIVERS

            kwargs["driver"] = DRIVERS[spec.driver]
        result = jam_episode.run_jam_episode(
            model, spec.seed, scenario, arm, feed=feed, duration_s=spec.duration_s,
            penetration=spec.penetration, camera=spec_camera,
            predictor=predictor, step_length_s=context.step_length_s,
            window_start_s=context.window_start_s, write_log=context.write_log,
            check_max_speed=context.check_max_speed, sample_fraction=spec.sample_fraction,
            **kwargs)
        if boundary.is_approach:
            result.update(approach_metrics(result["trace"], scenario, spec, v_jam, boundary))
            return result
        result["jam"] = episode_metrics(result["trace"], v_jam=v_jam, start_s=scenario.start_s,
                                        end_s=scenario.end_s, cap=spec.duration_s)
        result["jam_30kmh"] = episode_metrics(result["trace"], v_jam=30.0,
                                              start_s=scenario.start_s, end_s=scenario.end_s,
                                              cap=spec.duration_s)
        return result

    return run


_SCHEDULES: dict[str, list[dict]] = {}


def approach_metrics(trace: list[dict], scenario, spec: Spec, v_jam: float,
                     boundary) -> dict[str, Any]:
    """An approach episode's jam metrics at 19 km/h and 30 km/h, its storage measures and
    its flag (approach plan sections 5.2 to 5.4). Every clock is the scenario's plus `W`.

    The network is what is measured (2026-10-20): `jam` and `jam_30kmh` are over the
    network's 12 super-segments, and `storage.network_min` is the vehicle-minutes spent on
    the network. The approach roads only pace the demand into it; `jam_all` (all 36) and
    `jam_approach` (the 24 approach super-segments) are kept as diagnostics."""
    from src.analysis.jam_metrics import (
        approach_excess_minutes, episode_metrics, network_minutes, queue_flags,
        system_excess_minutes,
    )

    meta = json.loads(Path(boundary.approach_file).read_text())
    n = int(meta["num_original_segments"])
    total = n + len(meta["approach_segments"])
    parts = {"": list(range(n)), "_all": None, "_approach": list(range(n, total))}
    out: dict[str, Any] = {}
    for threshold, prefix in ((v_jam, "jam"), (30.0, "jam_30kmh")):
        for suffix, columns in parts.items():
            out[prefix + suffix] = episode_metrics(
                trace, v_jam=threshold, start_s=scenario.start_s, end_s=scenario.end_s,
                cap=spec.duration_s, columns=columns, offset_s=boundary.W)
    schedule_path = REPO_ROOT / FILE_SETS[boundary.files(scenario.files)]["schedule"]
    if str(schedule_path) not in _SCHEDULES:
        _SCHEDULES[str(schedule_path)] = json.loads(schedule_path.read_text())
    rows = _SCHEDULES[str(schedule_path)]
    out["storage"] = {
        "network_min": network_minutes(trace, start_s=scenario.start_s, cap=spec.duration_s,
                                       offset_s=boundary.W),
        "approach_excess_min": approach_excess_minutes(
            trace, rows, meta["T_A"], start_s=scenario.start_s, cap=spec.duration_s,
            offset_s=boundary.W),
        "system_excess_min": system_excess_minutes(
            trace, start_s=scenario.start_s, cap=spec.duration_s, offset_s=boundary.W,
            rows=rows, travel_s=meta["T_A"]),
    }
    out["flags"] = queue_flags(trace, limit_m=float(meta["length_m"]) - float(meta["segment_m"]))
    out["approach_short"] = out["flags"]["approach_short"]
    return out


# ---------------------------------------------------------------- summaries


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def collision_counts(episodes: list[dict]) -> dict[str, Any]:
    """Every controlled arm must show zero colliding vehicles and collision teleports."""
    controlled = [r for r in episodes if r["arm"] != "no_control"]
    return {
        # Approach plan section 6: every episode of every arm, no control included.
        "episodes_with_collisions": sum(
            1 for r in episodes
            if r.get("colliding_vehicles", 0) or r.get("collision_teleports", 0)),
        "controlled_episodes": len(controlled),
        "controlled_episodes_with_collisions": sum(
            1 for r in controlled
            if r.get("colliding_vehicles", 0) or r.get("collision_teleports", 0)),
        "colliding_vehicles_total": sum(r.get("colliding_vehicles", 0) for r in episodes),
        "collision_teleports_total": sum(r.get("collision_teleports", 0) for r in episodes),
        "writes_above_plan_total": sum(r.get("writes_above_plan", 0) or 0 for r in episodes),
        "max_speed_mismatched_total": sum(r.get("max_speed_mismatched", 0) or 0
                                          for r in episodes),
    }


def _rows(episodes: list[dict]) -> dict[str, dict[str, Any]]:
    """Per `arm/variant/scenario`: the paper's per-arm quantities over the seeds."""
    from src.analysis.jam_metrics import km_onset
    from src.sumo.drain import kaplan_meier

    groups: dict[str, list[dict]] = {}
    for r in episodes:
        groups.setdefault(f"{r['arm']}/{r['variant']}/{r['scenario']}", []).append(r)
    rows = {}
    for name, records in sorted(groups.items()):
        records = sorted(records, key=lambda r: r["seed"])
        jam = [r["jam"] for r in records]
        start, cap = jam[0]["start_s"], jam[0]["cap_s"]
        km = km_onset([j["onset_s"] for j in jam], start, cap)
        drains = [j for j in jam if j["drained"] is not None]
        km_drain = (kaplan_meier([j["drain_s"] for j in drains], [j["drained"] for j in drains])
                    if drains else None)
        rows[name] = {
            "arm": records[0]["arm"], "variant": records[0]["variant"],
            "scenario": records[0]["scenario"], "seeds": [r["seed"] for r in records],
            "n": len(records), "prevented": sum(1 for j in jam if j["prevented"]),
            "km_onset": km.as_dict(), "km_onset_median_s": km.median,
            "jam_time_min_mean": _mean(j["jam_time_min"] for j in jam),
            "jam_time_min_900_3600_mean": _mean(j["jam_time_min_900_3600"] for j in jam),
            "flow_mean": _mean(r["flow"] for r in records),
            "held_min_mean": _mean(j["held_min"] for j in jam),
            "pending_min_mean": _mean(j["pending_min"] for j in jam),
            "fraction_drained": (sum(1 for j in drains if j["drained"]) / len(drains)
                                 if drains else None),
            "km_drain": km_drain.as_dict() if km_drain else None,
            "km_drain_median_s": km_drain.median if km_drain else None,
            "active_share_mean": _mean(r.get("active_share") for r in records),
            "active_changing_share_mean": _mean(r.get("active_changing_share") for r in records),
            "gate_open_share_mean": _mean(r.get("gate_open_share") for r in records),
            "return_mean": _mean(r.get("return") for r in records),
            "wall_s_mean": _mean(r.get("wall_s") for r in records),
            "by_seed": {str(r["seed"]): {"jam_time_min": r["jam"]["jam_time_min"],
                                         "jam_time_min_900_3600": r["jam"]["jam_time_min_900_3600"],
                                         "onset_s": r["jam"]["onset_s"], "flow": r["flow"],
                                         "prevented": r["jam"]["prevented"]}
                        for r in records},
        }
        if all("jam_approach" in r for r in records):
            extra = _approach_row(records)
            for seed, values in extra.pop("by_seed").items():
                rows[name]["by_seed"][seed].update(values)
            rows[name].update(extra)
    return rows


def _approach_row(records: list[dict]) -> dict[str, Any]:
    """An approach row's diagnostic parts (all 36 and the approach 24), storage and flags
    (section 5). The row's own jam fields are the network's (2026-10-20)."""
    out: dict[str, Any] = {}
    for part in ("all", "approach"):
        jam = [r[f"jam_{part}"] for r in records]
        out[f"jam_{part}_time_min_mean"] = _mean(j["jam_time_min"] for j in jam)
        out[f"jam_{part}_time_min_900_3600_mean"] = _mean(j["jam_time_min_900_3600"]
                                                          for j in jam)
        out[f"prevented_{part}"] = sum(1 for j in jam if j["prevented"])
    out["network_min_mean"] = _mean(r["storage"]["network_min"] for r in records)
    out["approach_excess_min_mean"] = _mean(r["storage"]["approach_excess_min"] for r in records)
    out["system_excess_min_mean"] = _mean(r["storage"]["system_excess_min"] for r in records)
    out["approach_short_episodes"] = sum(1 for r in records if r["approach_short"])
    widths = {len(r["flags"]["queue_max_m"]) for r in records}
    out["queue_max_m"] = ([max(r["flags"]["queue_max_m"][j] for r in records)
                           for j in range(widths.pop())] if len(widths) == 1 else None)
    out["pending_max"] = max(r["flags"]["pending_max"] for r in records)
    out["by_seed"] = {}
    for r in records:
        out["by_seed"][str(r["seed"])] = {
            "jam_all_time_min": r["jam_all"]["jam_time_min"],
            "jam_approach_time_min": r["jam_approach"]["jam_time_min"],
            "prevented_all": r["jam_all"]["prevented"],
            "network_min": r["storage"]["network_min"],
            "approach_excess_min": r["storage"]["approach_excess_min"],
            "system_excess_min": r["storage"]["system_excess_min"],
            "approach_short": r["approach_short"]}
    return out


def g_pass(oracle: dict[str, dict[int, dict]], plan: dict[str, dict[int, dict]]) -> dict[str, Any]:
    """Section 8.1's pass rule for one oracle setting.

    `oracle[scenario][seed]` and `plan[scenario][seed]` are episode jam metrics. Pass when,
    for at least one scenario, the oracle's paired mean jam time is below the plan's by
    more than two standard errors of the paired differences, or on B9 the oracle has no
    120 s spell on at least 2 more seeds than the plan; and in neither scenario is the
    oracle's paired mean above the plan's by more than two standard errors.
    """
    from src.analysis.jam_metrics import Paired

    per, better, worse = {}, False, False
    for scenario in sorted(set(oracle) | set(plan)):
        a = {s: v["jam_time_min"] for s, v in oracle.get(scenario, {}).items()}
        b = {s: v["jam_time_min"] for s, v in plan.get(scenario, {}).items()}
        if set(a) != set(b):
            raise SystemExit(f"block G {scenario}: oracle seeds {sorted(a)} and plan seeds "
                             f"{sorted(b)} are not paired")
        paired = Paired.of(a, b)
        below = (paired.mean is not None and paired.two_se is not None
                 and paired.mean < -paired.two_se)
        above = (paired.mean is not None and paired.two_se is not None
                 and paired.mean > paired.two_se)
        prevented_oracle = sum(1 for v in oracle.get(scenario, {}).values() if v["prevented"])
        prevented_plan = sum(1 for v in plan.get(scenario, {}).values() if v["prevented"])
        prevention = scenario == "B9" and prevented_oracle - prevented_plan >= 2
        per[scenario] = {"paired_jam_time": paired.as_dict(), "below_by_2se": below,
                         "above_by_2se": above, "prevented_oracle": prevented_oracle,
                         "prevented_plan": prevented_plan, "prevention_rule": prevention}
        better = better or below or prevention
        worse = worse or above
    return {"scenarios": per, "passes": bool(better and not worse)}


G_SCENARIOS: tuple[str, ...] = ("B9", "I_exit")


def _missing(results: dict[str, dict[int, dict]]) -> dict[str, list[int]]:
    """Per block G scenario, the seeds 1-5 an arm has not run (empty when complete)."""
    out = {}
    for scenario in G_SCENARIOS:
        absent = sorted(set(BLOCK_SEEDS["G"]) - set(results.get(scenario, {})))
        if absent:
            out[scenario] = absent
    return out


def summarize_g(episodes: list[dict]) -> dict[str, Any]:
    """Section 8.1's decision. A setting is decided only when it and the plan alone have
    all five seeds on both scenarios; otherwise the decision says what is missing."""
    plan: dict[str, dict[int, dict]] = {}
    oracles: dict[str, dict[str, dict[int, dict]]] = {}
    for r in episodes:
        if r["arm"] == "plan":
            plan.setdefault(r["scenario"], {})[r["seed"]] = r["jam"]
        elif r["arm"] == "oracle":
            oracles.setdefault(r["variant"], {}).setdefault(r["scenario"], {})[r["seed"]] = r["jam"]
    base = _variant("oracle", G_BASE)
    grid = {_variant("oracle", {"tau_l": t, "m_on": m}) for t in G_TAU for m in M_ON}
    plan_missing = _missing(plan)
    settings: dict[str, Any] = {}
    for variant, results in sorted(oracles.items()):
        missing = _missing(results)
        if missing or plan_missing:
            settings[variant] = {"complete": False, "passes": None, "missing_seeds": missing}
        else:
            settings[variant] = {"complete": True, **g_pass(results, plan)}
    passing = sorted(v for v, entry in settings.items() if entry["passes"])
    base_result = settings.get(base)
    grid_run = sorted(set(settings) & grid - {base})
    if plan_missing:
        decision = f"incomplete: the plan alone lacks seeds {plan_missing}"
    elif base_result is None:
        decision = "incomplete: the base oracle setting has not run"
    elif not base_result["complete"]:
        decision = f"incomplete: the base oracle setting lacks seeds {base_result['missing_seeds']}"
    elif base_result["passes"]:
        decision = "go"
    elif not grid_run:
        decision = "base setting fails: run the grid (--g-grid --resume)"
    elif passing:
        decision = f"go on the grid ({', '.join(passing)})"
    elif any(not settings[v]["complete"] for v in grid_run) or set(grid_run) != grid - {base}:
        incomplete = sorted((grid - {base}) - {v for v in grid_run if settings[v]["complete"]})
        decision = f"incomplete: grid settings {incomplete} have not run on every seed"
    else:
        decision = "no-go: no oracle setting passes; no camera arm runs (section 8.1)"
    return {"block": "G", "base_setting": base, "settings": settings,
            "passing_settings": passing, "plan_missing_seeds": plan_missing,
            "decision": decision}


def _s_missing(episodes: list[dict]) -> dict[str, list[int]]:
    """Per `arm/variant/scenario` of block S, the seeds 11-15 without a record."""
    have: dict[str, set[int]] = {}
    for r in episodes:
        have.setdefault(f"{r['arm']}/{r['variant']}/{r['scenario']}", set()).add(r["seed"])
    missing = {}
    for spec in block_specs("S"):
        cell = f"{spec.arm}/{spec.variant}/{spec.scenario}"
        if spec.seed not in have.get(cell, set()):
            missing.setdefault(cell, []).append(spec.seed)
    return missing


def summarize_s(episodes: list[dict], context: Context,
                identity: tuple | None = None) -> dict[str, Any]:
    """Criterion C over the selection seeds only (section 5.7).

    Nothing is selected unless no control, the plan and every candidate have all of seeds
    11-15 on all six scenarios (validation R2-1); the decision then says which cells are
    missing. The provenance hashes are the ones block S's episodes ran under, its headers'
    `identity` (validation R2-2).
    """
    from src.analysis.jam_metrics import SELECTION_SCENARIOS, criterion_c

    outside = sorted({r["seed"] for r in episodes} - set(SELECTION_SEEDS))
    if outside and not context.smoke:
        raise SystemExit(f"the selection reads seeds 11-15 only; block S has records on "
                         f"{outside}")
    provenance = {"feed_artefact_sha256": identity[1] if identity else None,
                  "camera_model_sha256": identity[2] if identity else None,
                  "predictor_sha256": identity[3] if identity else None}
    missing = _s_missing(episodes)
    if missing:
        return {"block": "S", "criterion": "C (plan section 5.7)",
                "decision": (f"incomplete: {len(missing)} arm-scenario cells lack seeds of "
                             "11-15; nothing is selected"),
                "missing": missing, "seeds": sorted({r["seed"] for r in episodes}),
                "candidates": {}, "excluded": {},
                "selected": {"lookahead": None, "rule": None, "learned": None}, "best": None,
                "smoke_note": ("smoke run: nothing is selected" if context.smoke else None),
                **provenance}
    J: dict[str, dict[str, dict[int, float]]] = {}
    for r in episodes:
        name = r["arm"] if r["variant"] == "-" else f"{r['arm']}/{r['variant']}"
        J.setdefault(r["scenario"], {}).setdefault(name, {})[r["seed"]] = (
            r["jam"]["jam_time_min_900_3600"])
    scenarios = [s for s in SELECTION_SCENARIOS if s in J]
    candidates: dict[str, Any] = {}
    names = sorted({n for rows in J.values() for n in rows} - {"no_control"})
    excluded = {}
    for name in names:
        if not all(name in J[s] for s in scenarios):
            continue
        records = [r for r in episodes if (r["arm"] if r["variant"] == "-"
                                           else f"{r['arm']}/{r['variant']}") == name]
        bad = sum(1 for r in records if r.get("colliding_vehicles") or r.get("collision_teleports")
                  or (r.get("writes_above_plan") or 0))
        result = criterion_c(J, name, scenarios)
        bg = [r for r in records if r["scenario"] == "BG"]
        candidates[name] = {**result,
                            "bg_flow_mean": _mean(r["flow"] for r in bg),
                            "bg_active_share_mean": _mean(r.get("active_share") for r in bg),
                            "bg_active_changing_share_mean": _mean(
                                r.get("active_changing_share") for r in bg),
                            "return_mean": _mean(r.get("return") for r in records),
                            "prevented": {s: sum(1 for r in records if r["scenario"] == s
                                                 and r["jam"]["prevented"]) for s in scenarios},
                            "km_onset_median_s": {
                                s: _rows([r for r in records if r["scenario"] == s])
                                .get(f"{records[0]['arm']}/{records[0]['variant']}/{s}", {})
                                .get("km_onset_median_s") for s in scenarios},
                            "episodes_with_collision_or_write_above_plan": bad}
        if bad:
            excluded[name] = bad

    def pick(prefix: str) -> tuple[str | None, float | None]:
        grid = {n: c for n, c in candidates.items() if n.startswith(prefix + "/")
                and n not in excluded and c["C"] is not None}
        if not grid:
            return None, None
        name = min(grid, key=lambda n: grid[n]["C"])
        return name, grid[name]["C"]

    look, _ = pick("lookahead")
    rule, _ = pick("rule")
    learned, _ = pick("learned")
    pool = {n: candidates[n] for n in ("plan", "take_the_slower", rule, learned)
            if n and n in candidates and n not in excluded and candidates[n]["C"] is not None}
    best = None
    if pool:
        lowest = min(c["C"] for c in pool.values())
        tied = [n for n, c in pool.items() if c["C"] <= lowest + C_TIE]
        best = max(tied, key=lambda n: (pool[n]["bg_flow_mean"] or -math.inf, -pool[n]["C"]))

    def params(name: str | None) -> dict[str, float]:
        if not name or "/" not in name:
            return {}
        arm, variant = name.split("/", 1)
        if arm == "lookahead":
            return {"J_th": float(variant[1:])}
        if arm == "rule":
            return {"m_on": float(variant[1:])}
        if arm == "learned":
            return {"theta": float(variant[2:])}
        return {}

    return {
        "block": "S", "criterion": "C (plan section 5.7)", "seeds": sorted({r["seed"] for r in episodes}),

        "scenarios": scenarios, "candidates": candidates, "excluded": excluded,
        "selected": {"lookahead": params(look) or None, "rule": params(rule) or None,
                     "learned": params(learned) or None},
        "best": ({"arm": best.split("/")[0], "params": params(best), "name": best,
                  "C": pool[best]["C"]} if best else None),
        "tie_rule": f"within {C_TIE} of the lowest C, the highest BG served flow",
        "decision": "selected" if best else "selected nothing: no candidate qualifies",
        "missing": {}, **provenance,
    }


def summarize_eval(block: str, episodes: list[dict]) -> dict[str, Any]:
    """Blocks M, L, P and K: per-arm rows, and paired jam time and flow against the plan."""
    from src.analysis.jam_metrics import Paired

    rows = _rows(episodes)
    approach = any("jam_approach" in r for r in episodes)
    for name, row in rows.items():
        base = f"plan/-/{row['scenario']}"
        if block == "L":
            base = f"plan/-_lag{name.split('_lag')[-1].split('/')[0]}/{row['scenario']}"
        elif block == "P" or (block == "Q" and "_p" in row["variant"]):
            base = f"plan/-_p{name.split('_p')[-1].split('/')[0]}/{row['scenario']}"
        elif block == "D":
            base = f"plan/-_drv{name.split('_drv')[-1].split('/')[0]}/{row['scenario']}"
        if base in rows and base != name:
            a = {int(s): v["jam_time_min"] for s, v in row["by_seed"].items()}
            b = {int(s): v["jam_time_min"] for s, v in rows[base]["by_seed"].items()}
            fa = {int(s): v["flow"] for s, v in row["by_seed"].items()}
            fb = {int(s): v["flow"] for s, v in rows[base]["by_seed"].items()}
            row["paired_jam_time_vs_plan"] = Paired.of(a, b).as_dict()
            row["paired_flow_vs_plan"] = Paired.of(fa, fb).as_dict()
            if approach and "jam_all_time_min_mean" in row:
                for key, field_name in (("network_min", "network_min"),
                                        ("system_excess", "system_excess_min")):
                    pa = {int(s): v[field_name] for s, v in row["by_seed"].items()}
                    pb = {int(s): v[field_name] for s, v in rows[base]["by_seed"].items()}
                    row[f"paired_{key}_vs_plan"] = Paired.of(pa, pb).as_dict()
    out = {"block": block, "rows": rows}
    if approach:
        out["approach_short_total"] = sum(1 for r in episodes if r.get("approach_short"))
        out["thresholds_note"] = THRESHOLDS_NOTE
    return out


def qualification(episodes: list[dict]) -> dict[str, Any]:
    """Block Q's pass rule (approach plan section 7.1): no episode flagged approach-short
    and no episode with a collision. It chooses no threshold and no arm."""
    def key(r):
        return r.get("key") or f"Q/{r['arm']}/{r['variant']}/{r['scenario']}/{r['seed']}"

    flagged = sorted(key(r) for r in episodes if r.get("approach_short"))
    collided = sorted(key(r) for r in episodes
                      if r.get("colliding_vehicles", 0) or r.get("collision_teleports", 0))
    return {"passes": bool(episodes) and not flagged and not collided, "flagged": flagged,
            "collisions": collided, "episodes": len(episodes),
            "on_failure": "rebuild at 5,000 m (--approach-length-m 5000), recalibrate, rerun Q"}


def write_leads(episodes: list[dict], results_dir: Path,
                baseline: tuple[float, float] | None = None,
                start_s: float | None = None, offset_s: float = 0.0,
                network_columns: list[int] | None = None) -> dict[str, Any]:
    """`leads.json` from block M's no-control episodes (section 4.3 and 6.2).

    `baseline` and `start_s` default to section 4.3's [600, 900) s and each scenario's clock
    start; a smoke run, which ends at 600 s, passes its own. On the approach boundary
    `offset_s` (`W`) moves the baseline, the clock start and the cap, and `network_columns`
    names the network: the leads are taken on the network's own onset (2026-10-20; the
    approach roads are not measured), and the all-segment onset is kept under `all`."""
    from src.analysis.jam_metrics import (
        leads, load_v_jam, max_ramp, select_columns, speed_density_break,
    )
    from src.sumo.jam_episode import SCENARIOS as JAM_SCENARIOS

    v_jam, sha = load_v_jam()
    base = baseline or BASELINE_LEADS
    base = (base[0] + offset_s, base[1] + offset_s)
    per, s45 = [], []
    for r in episodes:
        if r["arm"] != "no_control":
            continue
        trace = json.loads(Path(r["trace_file"]).read_text())["trace"]
        if r["scenario"] == "S45":
            s45.append(trace)
            continue
        scenario = JAM_SCENARIOS[r["scenario"]]
        start = (scenario.start_s if start_s is None else start_s) + offset_s
        cap = r["duration_s"] + offset_s
        measured = (select_columns(trace, network_columns) if network_columns is not None
                    else trace)
        lead = leads(measured, v_jam, start, cap, baseline=base)
        ramp = (max_ramp(measured, lead["segment"], lead["onset"])
                if lead["onset"] is not None else None)
        entry = {"scenario": r["scenario"], "seed": r["seed"], **lead, "ramp": ramp}
        if network_columns is not None:
            lead_a = leads(trace, v_jam, start, cap, baseline=base)
            entry["all"] = {**lead_a, "ramp": (max_ramp(trace, lead_a["segment"],
                                                        lead_a["onset"])
                                               if lead_a["onset"] is not None else None)}
        per.append(entry)
    network_s45 = ([select_columns(t, network_columns) for t in s45]
                   if network_columns is not None else s45)
    out = {"v_jam": v_jam, "v_jam_sha256": sha, "episodes": per,
           "speed_density_break": speed_density_break(network_s45) if s45 else None}
    if network_columns is not None:
        out["offset_s"] = offset_s
        out["measured_columns"] = list(network_columns)
        out["speed_density_break_all"] = speed_density_break(s45) if s45 else None
    _write_json(results_dir / "leads.json", out)
    return out


def _n_cells(episodes: list[dict]) -> tuple[dict[str, list[int]], list[str]]:
    """Block N's `(missing, unexpected)`: per `arm/variant/scenario` of its specs the seeds
    11-15 without a record, and the cells with records that are not in its specs (validation
    2026-10-20, finding 2). The specs are built from the rule's `m_on` in the records."""
    m_on = sorted({float((r.get("params") or {}).get("m_on")) for r in episodes
                   if r["arm"] == "rule"})
    if len(m_on) != 1:
        return {"rule": list(SELECTION_SEEDS)}, [f"rule m_on {m_on}"]
    stand_in = Selection(J_th=0.0, m_on=m_on[0], theta=0.0, best_arm="rule",
                         best_params={"m_on": m_on[0]})
    have: dict[str, set[int]] = {}
    for r in episodes:
        have.setdefault(f"{r['arm']}/{r['variant']}/{r['scenario']}", set()).add(r["seed"])
    missing: dict[str, list[int]] = {}
    expected = set()
    for spec in block_specs("N", selection=stand_in):
        cell = f"{spec.arm}/{spec.variant}/{spec.scenario}"
        expected.add(cell)
        if spec.seed not in have.get(cell, set()):
            missing.setdefault(cell, []).append(spec.seed)
    return missing, sorted(set(have) - expected)


def summarize_n(episodes: list[dict]) -> dict[str, Any]:
    """Block N (2026-10-20): the feed gate's threshold for the rule, from seeds 11-15 only.

    Each candidate, the rule without a gate and the rule at each `GATE_GRID` threshold, gets
    criterion C (section 5.7) on the network's jam time over [900, 3,600] s. The lowest C
    wins, and candidates within `C_TIE` of it are ordered by the flow served with no
    trigger (BG), highest first, as section 5.7 orders the best model; the gate exists to
    lower the cost when no jam would form. A candidate with any collision, collision
    teleport or write above the plan is excluded.
    """
    from src.analysis.jam_metrics import SELECTION_SCENARIOS, criterion_c

    seeds = sorted({r["seed"] for r in episodes})
    outside = sorted(set(seeds) - set(SELECTION_SEEDS))
    if outside:
        raise SystemExit(f"block N selects on seeds 11-15; its records hold {outside}")
    missing, unexpected = _n_cells(episodes)
    if missing or unexpected:
        # A run of some of the seeds (one process per seed), or a block with a cell short,
        # selects nothing: a missing BG cell alone can change the tie-break.
        return {"block": "N", "seeds": seeds, "selected": None, "missing": missing,
                "unexpected": unexpected,
                "decision": f"incomplete: {sum(len(v) for v in missing.values())} episodes "
                            f"missing, {len(unexpected)} unexpected cells"}

    def name(r: dict) -> str:
        return r["arm"] if r["arm"] in ("no_control", "plan") else f"{r['arm']}/{r['variant']}"

    J: dict[str, dict[str, dict[int, float]]] = {}
    for r in episodes:
        J.setdefault(r["scenario"], {}).setdefault(name(r), {})[r["seed"]] = \
            r["jam"]["jam_time_min_900_3600"]
    by_name: dict[str, list[dict]] = {}
    for r in episodes:
        by_name.setdefault(name(r), []).append(r)

    def bg(records: list[dict], key):
        return _mean(key(r) for r in records if r["scenario"] == "BG")

    candidates: dict[str, dict[str, Any]] = {}
    for label in sorted(n for n in by_name if n.startswith("rule/")):
        records = by_name[label]
        params = dict(records[0].get("params") or {})
        excluded = any(r.get("colliding_vehicles") or r.get("collision_teleports")
                       or r.get("writes_above_plan") for r in records)
        c = criterion_c(J, label, SELECTION_SCENARIOS)
        candidates[label] = {
            "variant": records[0]["variant"], "m_on": params.get("m_on"),
            "gate_J": params.get("gate_J"), "C": c["C"], "terms": c["terms"],
            "excluded_scenarios": c["excluded_scenarios"], "excluded": excluded,
            "bg_flow_mean": bg(records, lambda r: r["flow"]),
            "bg_network_min_mean": bg(records, lambda r: r["storage"]["network_min"]),
            "bg_active_changing_share_mean": bg(records, lambda r: r.get("active_changing_share")),
            "bg_gate_open_share_mean": bg(records, lambda r: r.get("gate_open_share")),
            "prevented": {s: sum(1 for r in records if r["scenario"] == s and r["jam"]["prevented"])
                          for s in SELECTION_SCENARIOS},
            "jam_time_min_900_3600_mean": {s: _mean(J[s][label].values())
                                           for s in SELECTION_SCENARIOS},
        }
    reference = {n: {"bg_flow_mean": bg(by_name[n], lambda r: r["flow"]),
                     "bg_network_min_mean": bg(by_name[n], lambda r: r["storage"]["network_min"]),
                     "jam_time_min_900_3600_mean": {s: _mean(J[s][n].values())
                                                    for s in SELECTION_SCENARIOS}}
                 for n in ("no_control", "plan") if n in by_name}
    eligible = [n for n, v in candidates.items() if not v["excluded"] and v["C"] is not None]
    selected, decision = None, "no eligible candidate"
    if eligible:
        lowest = min(candidates[n]["C"] for n in eligible)
        tied = [n for n in eligible if candidates[n]["C"] <= lowest + C_TIE + 1e-12]
        chosen = max(tied, key=lambda n: (candidates[n]["bg_flow_mean"], -candidates[n]["C"]))
        selected = {"label": chosen, "variant": candidates[chosen]["variant"],
                    "m_on": candidates[chosen]["m_on"], "gate_J": candidates[chosen]["gate_J"]}
        decision = (f"{chosen}: lowest C {lowest:.4f}; within {C_TIE} of it "
                    f"{sorted(tied)}, ordered by BG served flow")
    return {"block": "N", "criterion": "C on the network's jam time, ties by BG flow",
            "seeds": seeds, "gate_grid": list(GATE_GRID), "c_tie": C_TIE,
            "candidates": candidates, "reference": reference, "selected": selected,
            "decision": decision}


def _n2_unchanged(n2: list[dict], n: list[dict]) -> list[str]:
    """Block N2 re-runs no control and plan alone on seeds 11-15; each must equal block N's
    record of the same cell (jam time, flow, arrivals), or the code change moved the arms
    that block N's candidates were measured against."""
    earlier = {(r["arm"], r["scenario"], r["seed"]): r for r in n
               if r["arm"] in ("no_control", "plan")}
    differ = []
    for r in n2:
        if r["arm"] not in ("no_control", "plan"):
            continue
        old = earlier.get((r["arm"], r["scenario"], r["seed"]))
        if old is None or (old["jam"]["jam_time_min_900_3600"], old["flow"], old["arrived"]) != \
                (r["jam"]["jam_time_min_900_3600"], r["flow"], r["arrived"]):
            differ.append(f"{r['arm']}/{r['scenario']}/{r['seed']}")
    return sorted(differ)


def summarize_n2(episodes: list[dict], n_episodes: list[dict]) -> dict[str, Any]:
    """Block N2 (plan 2026-10-25 section 2): the queue release and camera gate, selected on
    seeds 11-15 together with block N's candidates.

    A candidate is eligible when it has no collision, collision teleport or write above the
    plan, and with no trigger serves at least plan alone's flow minus `N2_BG_ALLOWANCE` of
    no control's. Among eligible candidates the lowest criterion C wins, and those within
    `C_TIE` of it are ordered by BG flow, highest first. No control and plan alone come from
    N2's own runs, which must equal block N's.
    """
    from src.analysis.jam_metrics import SELECTION_SCENARIOS, criterion_c

    seeds = sorted({r["seed"] for r in episodes})
    if sorted(set(seeds) - set(SELECTION_SEEDS)):
        raise SystemExit(f"block N2 selects on seeds 11-15; its records hold {seeds}")
    m_on = sorted({float((r.get("params") or {}).get("m_on")) for r in episodes + n_episodes
                   if r["arm"] == "rule"})
    if len(m_on) != 1:
        raise SystemExit(f"blocks N and N2 ran the rule at m_on {m_on}; one m_on is selected")
    stand_in = Selection(J_th=0.0, m_on=m_on[0], theta=0.0, best_arm="rule",
                         best_params={"m_on": m_on[0]})
    have = {(r["arm"], r["variant"], r["scenario"], r["seed"]) for r in episodes}
    missing = sorted(s.key for s in block_specs("N2", selection=stand_in)
                     if (s.arm, s.variant, s.scenario, s.seed) not in have)
    n_missing, _ = _n_cells(n_episodes)
    differ = _n2_unchanged(episodes, n_episodes)
    if missing or n_missing or differ:
        return {"block": "N2", "seeds": seeds, "selected": None, "missing": missing,
                "block_n_missing": n_missing, "differ_from_block_n": differ,
                "decision": f"incomplete or changed: {len(missing)} N2 episodes missing, "
                            f"{sum(len(v) for v in n_missing.values())} block N episodes "
                            f"missing, {len(differ)} reference cells differ from block N"}

    def name(r: dict) -> str:
        return r["arm"] if r["arm"] in ("no_control", "plan") else f"{r['arm']}/{r['variant']}"

    pool = episodes + [r for r in n_episodes if r["arm"] == "rule"]
    J: dict[str, dict[str, dict[int, float]]] = {}
    by_name: dict[str, list[dict]] = {}
    for r in pool:
        J.setdefault(r["scenario"], {}).setdefault(name(r), {})[r["seed"]] = \
            r["jam"]["jam_time_min_900_3600"]
        by_name.setdefault(name(r), []).append(r)

    def bg(records: list[dict], key):
        return _mean(key(r) for r in records if r["scenario"] == "BG")

    plan_bg = bg(by_name["plan"], lambda r: r["flow"])
    nc_bg = bg(by_name["no_control"], lambda r: r["flow"])
    floor = plan_bg - N2_BG_ALLOWANCE * nc_bg
    candidates: dict[str, dict[str, Any]] = {}
    for label in sorted(n for n in by_name if n.startswith("rule/")):
        records = by_name[label]
        params = dict(records[0].get("params") or {})
        excluded = any(r.get("colliding_vehicles") or r.get("collision_teleports")
                       or r.get("writes_above_plan") for r in records)
        c = criterion_c(J, label, SELECTION_SCENARIOS)
        flow = bg(records, lambda r: r["flow"])
        candidates[label] = {
            "variant": records[0]["variant"], "block": records[0]["block"],
            "m_on": params.get("m_on"), "gate_J": params.get("gate_J"),
            "release_discharge": bool(params.get("release_discharge")),
            "camera_gate": params.get("camera_gate"),
            "C": c["C"], "terms": c["terms"], "excluded": excluded,
            "bg_flow_mean": flow, "meets_bg_floor": flow >= floor - 1e-9,
            "bg_network_min_mean": bg(records, lambda r: r["storage"]["network_min"]),
            "bg_active_changing_share_mean": bg(records, lambda r: r.get("active_changing_share")),
            "prevented": {s: sum(1 for r in records if r["scenario"] == s and r["jam"]["prevented"])
                          for s in SELECTION_SCENARIOS},
            "jam_time_min_900_3600_mean": {s: _mean(J[s][label].values())
                                           for s in SELECTION_SCENARIOS},
        }
    reference = {n: {"bg_flow_mean": bg(by_name[n], lambda r: r["flow"]),
                     "bg_network_min_mean": bg(by_name[n], lambda r: r["storage"]["network_min"]),
                     "jam_time_min_900_3600_mean": {s: _mean(J[s][n].values())
                                                    for s in SELECTION_SCENARIOS}}
                 for n in ("no_control", "plan")}
    eligible = [n for n, v in candidates.items()
                if not v["excluded"] and v["C"] is not None and v["meets_bg_floor"]]
    selected, decision = None, (f"no candidate serves at least {floor:.1f} veh/h with no "
                                "trigger; nothing selected")
    if eligible:
        lowest = min(candidates[n]["C"] for n in eligible)
        tied = [n for n in eligible if candidates[n]["C"] <= lowest + C_TIE + 1e-12]
        chosen = max(tied, key=lambda n: (candidates[n]["bg_flow_mean"], -candidates[n]["C"]))
        v = candidates[chosen]
        selected = {"label": chosen, "variant": v["variant"], "m_on": v["m_on"],
                    "gate_J": v["gate_J"], "release_discharge": v["release_discharge"],
                    "camera_gate": v["camera_gate"]}
        decision = (f"{chosen}: lowest C {lowest:.4f} among {len(eligible)} candidates at or "
                    f"above the BG floor {floor:.1f} veh/h; within {C_TIE} of it "
                    f"{sorted(tied)}, ordered by BG served flow")
    return {"block": "N2", "criterion": "C on the network's jam time among candidates whose "
                                        "BG flow is at least plan alone's minus 1% of no "
                                        "control's; ties by BG flow",
            "seeds": seeds, "mechanism_grid": [dict(m) for m in MECHANISM_GRID],
            "bg_floor": floor, "bg_allowance": N2_BG_ALLOWANCE, "c_tie": C_TIE,
            "candidates": candidates, "reference": reference, "eligible": sorted(eligible),
            "selected": selected, "decision": decision}


def summarize(block: str, results_dir: Path, context: Context) -> dict[str, Any]:
    headers, episodes = read_block(results_dir, block)
    identities = {header_identity(h) for h in headers}
    if len(identities) > 1:
        raise SystemExit(f"block {block} mixes (checkpoint, feed, camera, predictor) "
                         f"{sorted(identities, key=str)}; it is not summarised")
    if block == "G":
        summary = summarize_g(episodes)
    elif block == "S":
        identity = header_identity(headers[-1]) if headers else None
        if identity is not None:
            # Validation R2-2: the summary is the selection block S's episodes made, so the
            # camera model and predictor given now must be the ones they ran.
            for name, recorded, present in (("camera model", identity[2], context.camera_model),
                                            ("predictor", identity[3], context.predictor)):
                if recorded is not None and recorded != _sha256(present):
                    raise SystemExit(f"block S ran {name} {recorded}; {present} has "
                                     f"{_sha256(present)}. Summarise with the files it ran.")
        summary = summarize_s(episodes, context, identity)
    elif block == "N":
        summary = summarize_n(episodes)
    elif block == "N2":
        n_headers, n_episodes = read_block(results_dir, "N")
        if {header_identity(h) for h in n_headers} - identities:
            raise SystemExit("block N2 and block N ran different (checkpoint, feed, camera, "
                             "predictor); block N's candidates cannot join N2's pool")
        summary = summarize_n2(episodes, n_episodes)
    elif block in ("T", "T_dagger"):
        summary = {"block": block, "rows": _rows(episodes),
                   "samples_files": sorted(r["samples_file"] for r in episodes
                                           if r.get("samples_file"))}
    else:
        summary = summarize_eval(block, episodes)
        approach = bool(headers) and headers[-1].get("boundary") == "approach"
        offset = {"offset_s": float(headers[-1]["W"]), "network_columns": list(range(12))} \
            if approach else {}
        if block == "M":
            summary["leads"] = (write_leads(episodes, results_dir, baseline=(300.0, 450.0),
                                            start_s=450.0, **offset)
                                if context.smoke else write_leads(episodes, results_dir,
                                                                  **offset))
        if block == "Q":
            summary["qualification"] = qualification(episodes)
    seeds_by_arm: dict[str, set] = {}
    for r in episodes:
        seeds_by_arm.setdefault(f"{r['arm']}/{r['variant']}/{r['scenario']}", set()).add(r["seed"])
    last = headers[-1] if headers else {}
    summary.update({
        "episodes": len(episodes), **collision_counts(episodes),
        "seeds_by_arm": {k: sorted(v) for k, v in sorted(seeds_by_arm.items())},
        "identical_seed_lists": len({tuple(sorted(v)) for v in seeds_by_arm.values()}) <= 1,
        "sensors": sorted({f"{r['arm']}:{r.get('sensor')}" for r in episodes}),
        "feed_artefact": last.get("feed_artefact"),
        "placeholder": (last.get("feed_artefact") or {}).get("placeholder"),
        "camera_model": last.get("camera_model"), "predictor": last.get("predictor"),
        "selection": last.get("selection"),
        "headers": [{k: h.get(k) for k in ("started_wall", "commit", "dirty", "sumo_version",
                                            "resumed")} for h in headers],
        "wall_s_total": sum(r.get("wall_s", 0.0) for r in episodes),
        "smoke": bool(last.get("config", {}).get("smoke")),
    })
    if last.get("boundary") == "approach":
        summary.update({"boundary": "approach", "W": last.get("W"),
                        "approach": last.get("approach"),
                        "thresholds_note": THRESHOLDS_NOTE})
    _write_json(results_dir / f"{block}_summary.json", summary)
    return summary


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", nargs="+", choices=BLOCKS, required=True)
    parser.add_argument("--boundary", choices=("gated", "approach"), default="gated",
                        help="approach: the approach roads of plan 2026-10-17, no gate in any "
                             "arm, every time W later; blocks Q, M, L, P and K only")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="a subset of the block's own seeds (default all of them)")
    parser.add_argument("--penetrations", type=float, nargs="+", default=None,
                        help=f"block P: the shares of vehicles that follow (default "
                             f"{' '.join(f'{p:g}' for p in PENETRATIONS_P)})")
    parser.add_argument("--feed-artefact", default=str(DEFAULT_FEED_ARTEFACT))
    parser.add_argument("--feed-session", default=DEFAULT_FEED_SESSION)
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="let the blocks replay a placeholder feed process")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--camera-model", default=str(DEFAULT_CAMERA))
    parser.add_argument("--predictor", default=str(DEFAULT_PREDICTOR))
    parser.add_argument("--selection", default=None,
                        help="S_summary.json for blocks M to K (default <results-dir>)")
    parser.add_argument("--gate-selection", default=None,
                        help="approach boundary: block N's N_summary.json, whose feed gate "
                             "the rule runs in blocks M to K and Q (default: no gate)")
    parser.add_argument("--mechanism-selection", default=None,
                        help="approach boundary: block N2's N2_summary.json, whose queue "
                             "release, feed gate and camera gate the rule runs in blocks M "
                             "to K and D")
    parser.add_argument("--release-candidate", action="store_true",
                        help="block P with --mechanism-selection: add the queue release to the "
                             "selected system; recorded as not selected")
    parser.add_argument("--gate-candidate", type=float, default=None,
                        help="with --gate-selection: run this one of block N's evaluated "
                             "gates instead of the selected one; recorded as not selected")
    parser.add_argument("--results-dir", default=None,
                        help=f"default {SMOKE_GUARD['results']}; outputs/jam/smoke/results "
                             f"with --smoke; {APPROACH_DIRS['results']} on the approach "
                             "boundary")
    parser.add_argument("--traces-dir", default=None,
                        help=f"default {SMOKE_GUARD['traces']}; {APPROACH_DIRS['traces']} on "
                             "the approach boundary")
    parser.add_argument("--training-dir", default=None,
                        help="default outputs/jam/training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-file", default=None, help="default <results-dir>/PAUSE")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--g-grid", action="store_true",
                        help="block G: the 6-setting oracle grid run on failure (section 8.1)")
    parser.add_argument("--dagger", action="store_true",
                        help="block T: the DAgger round with the round-1 predictor active")
    parser.add_argument("--write-log", action="store_true",
                        help="keep every write (vehicle, tick, edge, version, plan, shown)")
    parser.add_argument("--check-max-speed", action="store_true",
                        help="read setMaxSpeed back after every stepped-down write")
    parser.add_argument("--smoke", action="store_true",
                        help=f"one {SMOKE_DURATION_S:g} s episode per arm on seed {SMOKE_SEED}")
    return parser


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def check_block_boundary(block: str, boundary: str) -> None:
    """Blocks G, T and S select or train and run on the gated boundary only; block Q
    qualifies the approach boundary and runs on it only (approach plan section 7.1)."""
    if boundary == "approach" and block in GATED_ONLY_BLOCKS:
        raise SystemExit(f"block {block} selects or trains on the gated boundary; the approach "
                         "boundary re-selects nothing (approach plan section 4.5)")
    if boundary != "approach" and block in APPROACH_ONLY_BLOCKS:
        raise SystemExit(f"block {block} qualifies the approach boundary; pass "
                         "--boundary approach")


def resolve_directories(args) -> None:
    """Fill the directory defaults for the boundary and smoke flag, and refuse to write an
    approach run under `results/jam/`."""
    if args.boundary == "approach":
        defaults = APPROACH_SMOKE_DIRS if args.smoke else APPROACH_DIRS
        args.results_dir = args.results_dir or defaults["results"]
        args.traces_dir = args.traces_dir or defaults["traces"]
        args.training_dir = args.training_dir or defaults["training"]
        guarded = (REPO_ROOT / "results" / "jam").resolve()
        for path in (args.results_dir, args.traces_dir, args.training_dir):
            resolved = _resolve(path).resolve()
            if resolved == guarded or resolved.is_relative_to(guarded):
                raise SystemExit(f"{path} is under results/jam/; the approach boundary never "
                                 "writes there (approach plan section 7.3)")
        return
    args.results_dir = args.results_dir or SMOKE_GUARD["results"]
    args.traces_dir = args.traces_dir or SMOKE_GUARD["traces"]
    args.training_dir = args.training_dir or "outputs/jam/training"
    if args.smoke:
        # A smoke episode must never land in a block's real JSONL, where --resume would
        # take its 600 s result for the experiment's own episode of the same key.
        if args.results_dir == SMOKE_GUARD["results"]:
            args.results_dir = "outputs/jam/smoke/results"
        if args.traces_dir == SMOKE_GUARD["traces"]:
            args.traces_dir = "outputs/jam/smoke/traces"
        if args.training_dir == "outputs/jam/training":
            args.training_dir = "outputs/jam/smoke/training"


def main(argv: list[str] | None = None, run_episode_factory=None) -> int:
    from src.sumo.jam_episode import FeedSetting

    args = build_parser().parse_args(argv)
    for block in args.block:
        check_block_boundary(block, args.boundary)
    if args.gate_candidate is not None and not args.gate_selection:
        raise SystemExit("--gate-candidate names one of block N's gates; pass --gate-selection")
    if args.release_candidate and (not args.mechanism_selection or args.block != ["P"]):
        raise SystemExit("--release-candidate adds the queue release to block N2's selection "
                         "in block P only; pass --mechanism-selection and --block P")
    if args.gate_selection and args.mechanism_selection:
        raise SystemExit("--mechanism-selection replaces block N's gate; pass one of them")
    if args.gate_selection and args.boundary != "approach":
        raise SystemExit("--gate-selection is block N's feed gate, selected on the approach "
                         "boundary; the gated boundary runs the rule without it")
    resolve_directories(args)
    approach = args.boundary == "approach"
    results_dir, traces_dir = _resolve(args.results_dir), _resolve(args.traces_dir)
    pause_file = _resolve(args.pause_file) if args.pause_file else results_dir / "PAUSE"
    context = Context(checkpoint=_resolve(args.checkpoint),
                      feed_artefact=_resolve(args.feed_artefact),
                      feed_session=args.feed_session, allow_placeholder=args.allow_placeholder,
                      camera_model=_resolve(args.camera_model),
                      predictor=_resolve(args.predictor),
                      selection_path=_resolve(args.selection) if args.selection else None,
                      gate_selection_path=(_resolve(args.gate_selection)
                                           if args.gate_selection else None),
                      mechanism_selection_path=(_resolve(args.mechanism_selection)
                                                if args.mechanism_selection else None),
                      window_start_s=300.0 if args.smoke else 900.0, smoke=args.smoke,
                      write_log=args.write_log or args.smoke,
                      check_max_speed=args.check_max_speed or args.smoke,
                      training_dir=_resolve(args.training_dir), boundary=args.boundary)
    context.feed = FeedSetting(context.feed_artefact, context.feed_session,
                               allow_placeholder=context.allow_placeholder).parameters()
    if approach:
        context.boundary_object()  # refuses an uncalibrated approach build up front

    model = None
    for block in args.block:
        if block in SELECTED_BLOCKS and approach and not args.summarize_only:
            # The approach boundary re-selects nothing: the committed selection, checked by
            # its sha256, even in a smoke run.
            context.selection = read_selection(
                context.selection_path or APPROACH_SELECTION,
                feed_sha=context.feed["artefact_sha256"],
                camera_sha=_sha256(context.camera_model),
                predictor_sha=_sha256(context.predictor), required_sha=APPROACH_SELECTION_SHA256)
            if context.gate_selection_path is not None and block not in ("N", "N2"):
                context.selection = read_gate_selection(context.gate_selection_path,
                                                        context.selection,
                                                        candidate=args.gate_candidate)
            if context.mechanism_selection_path is not None and block not in ("N", "N2"):
                context.selection = read_mechanism_selection(context.mechanism_selection_path,
                                                             context.selection)
                if args.release_candidate and not context.selection.release_discharge:
                    context.selection = replace(context.selection, release_discharge=True,
                                                release_selected=False)
            elif block == "D":
                raise SystemExit("block D runs the selected system; pass --mechanism-selection")
        elif block in ("M", "L", "P", "K"):
            selection_path = context.selection_path or results_dir / "S_summary.json"
            if args.smoke:
                # Never a real selection: a smoke run of block S selects nothing.
                context.selection = smoke_selection()
            elif not args.summarize_only:
                context.selection = read_selection(
                    selection_path, feed_sha=context.feed["artefact_sha256"],
                    camera_sha=_sha256(context.camera_model),
                    predictor_sha=_sha256(context.predictor))
        dagger = args.dagger and block == "T"
        # The DAgger round runs a predictor that round 1 did not, so it is its own JSONL:
        # one identity per file.
        name = "T_dagger" if dagger else block
        block_context = context
        if dagger and args.predictor == str(DEFAULT_PREDICTOR):
            block_context = replace(context, predictor=ROUND1_PREDICTOR)
        if args.summarize_only:
            summarize(name, results_dir, block_context)
            print(f"wrote {results_dir / f'{name}_summary.json'}")
            continue
        window = oracle_window(context.camera_model) if block in SELECTED_BLOCKS else None
        specs = block_specs(block, args.seeds, context.selection, grid=args.g_grid,
                            dagger=dagger, smoke=args.smoke, oracle_window=window,
                            penetrations=args.penetrations)
        if args.smoke:
            specs = smoke_subset(specs)
        if not specs:
            print(f"block {block}: no episodes (the best model reads no camera)")
            continue
        if model is None and run_episode_factory is None:
            import evaluate_mainz_checkpoints as emc

            if not context.checkpoint.exists():
                raise SystemExit(f"{context.checkpoint} does not exist")
            model = emc._build_and_load(context.checkpoint, emc.CHECKPOINT_FEATURES["here"],
                                        emc._num_segments(FILE_SETS["mainz"]))
        factory = run_episode_factory or episode_runner
        record = header(block, block_context, specs, argv if argv is not None else sys.argv)
        result = run_block(name, specs, factory(model, block_context), results_dir=results_dir,
                           traces_dir=traces_dir, header_record=record,
                           resume=args.resume, pause_file=pause_file,
                           training_dir=block_context.training_dir)
        if result["status"] == "paused":
            return 0
        summary = summarize(name, results_dir, block_context)
        print(f"block {name}: ran {len(result['ran'])}, skipped {len(result['skipped'])}; "
              f"wrote {results_dir / f'{name}_summary.json'}"
              + (f"; decision: {summary['decision']}" if block == "G" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
