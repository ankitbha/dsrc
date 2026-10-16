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

BLOCKS = ("G", "T", "S", "M", "L", "P", "K")
BLOCK_SEEDS: dict[str, tuple[int, ...]] = {
    "G": tuple(range(1, 6)), "T": tuple(range(1, 11)), "S": tuple(range(11, 16)),
    "M": tuple(range(16, 31)), "L": tuple(range(16, 31)), "P": tuple(range(16, 31)),
    "K": tuple(range(16, 31)),
}
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

    @property
    def key(self) -> str:
        return f"{self.block}/{self.arm}/{self.variant}/{self.scenario}/{self.seed}"

    def trace_name(self) -> str:
        return f"{self.arm}_{self.variant}_{self.scenario}_{self.seed}.json"

    def jam_arm(self):
        from src.sumo.jam_episode import JamArm

        return JamArm(self.arm, **dict(self.params))


def _variant(arm: str, params: dict[str, float]) -> str:
    if arm == "lookahead":
        return f"J{params['J_th']:g}"
    if arm == "rule":
        return f"m{params['m_on']:g}"
    if arm == "learned":
        return f"th{params['theta']:g}"
    if arm == "oracle":
        base = f"tau{params.get('tau_l', 10.0):g}_m{params['m_on']:g}"
        return base + (f"_R{params['oracle_R']:g}" if params.get("oracle_R", 80.0) != 80.0 else "")
    return "-"


def _duration(block: str, scenario: str) -> float:
    if block in ("G", "T", "S"):
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

    @property
    def best_reads_camera(self) -> bool:
        return self.best_arm in ("take_the_slower", "rule", "learned")


def arm_list(block: str, selection: Selection | None, grid: bool = False,
             dagger: bool = False) -> list[tuple[str, dict[str, float], dict[str, Any]]]:
    """`(arm, params, extra)` for a block; `extra` holds m, penetration, bias, sampling."""
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
    best = (selection.best_arm, dict(selection.best_params), {})
    if block == "M":
        return [("no_control", {}, {}), ("lookahead", {"J_th": selection.J_th}, {}),
                ("plan", {}, {}), ("take_the_slower", {}, {}),
                ("rule", {"m_on": selection.m_on}, {}),
                ("learned", {"theta": selection.theta}, {}),
                ("oracle", {"m_on": selection.m_on, "tau_l": 10.0}, {})]
    if block == "L":
        arms = [("lookahead", {"J_th": selection.J_th}, {}), ("plan", {}, {}), best]
        return [(a, p, {"m": m}) for m in LAGS_L for a, p, _ in _unique(arms)]
    if block == "P":
        arms = _unique([("plan", {}, {}), best])
        return [(a, p, {"penetration": pen}) for pen in PENETRATIONS_P for a, p, _ in arms]
    if block == "K":
        if not selection.best_reads_camera:
            return []
        return ([(best[0], best[1], {"bias": b}) for b in BIASES_K]
                + [(best[0], best[1], {"bias": 1.0, "pair_sigma_scale": scale})
                   for scale in PAIR_SIGMA_SCALES_K])
    raise ValueError(f"unknown block {block}")


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
    if block in ("T", "S"):
        return TRAINING_SCENARIOS
    if block == "M":
        return PERTURBATIONS + ("BG", "S45")
    return ("B9", "I_exit")


def block_specs(block: str, seeds: Sequence[int] | None = None,
                selection: Selection | None = None, grid: bool = False,
                dagger: bool = False, smoke: bool = False) -> list[Spec]:
    """Every episode of a block, arm-major, every arm over the same seed list.

    Seeds outside the block's own set are refused, except in a smoke run, which is not
    an experiment and runs seed 1 for 600 s.
    """
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
    for arm, params, extra in arm_list(block, selection, grid, dagger):
        variant = _variant(arm, params)
        if "m" in extra:
            variant += f"_lag{extra['m']:g}"
        if "penetration" in extra:
            variant += f"_p{extra['penetration']:g}"
        if "bias" in extra:
            variant += f"_b{extra['bias']:g}"
        if "pair_sigma_scale" in extra:
            variant += f"_ps{extra['pair_sigma_scale']:g}"
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
                                  pair_sigma_scale=extra.get("pair_sigma_scale", 1.0)))
        if block == "M" and arm == "oracle":
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
                   predictor_sha: str | None) -> Selection:
    """Blocks M to K's thresholds and best model; refused when the selection ran on
    another feed artefact, camera model or predictor, or on any seed outside 11-15."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    seeds = sorted(set(data.get("seeds") or []))
    if not seeds or set(seeds) - set(SELECTION_SEEDS):
        raise SystemExit(f"{path} lists seeds {seeds}; a selection reads seeds 11-15 only")
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
    step_length_s: float = 0.5
    window_start_s: float = 900.0
    smoke: bool = False
    write_log: bool = False
    check_max_speed: bool = False
    training_dir: Path = REPO_ROOT / "outputs" / "jam" / "training"
    selection: Selection | None = None
    feed: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()
               if k not in ("selection", "feed")}
        out["selection"] = asdict(self.selection) if self.selection else None
        return out


def needs_camera(specs: Sequence[Spec]) -> bool:
    return any(s.arm in ("take_the_slower", "rule", "learned") or s.sample_fraction > 0
               for s in specs)


def needs_predictor(specs: Sequence[Spec]) -> bool:
    return any(s.arm == "learned" for s in specs)


def header(block: str, context: Context, specs: Sequence[Spec],
           argv: Sequence[str] | None = None) -> dict[str, Any]:
    from src.analysis.jam_metrics import THRESHOLDS
    from src.sumo.jam_episode import SCENARIOS as JAM_SCENARIOS

    files: dict[str, Any] = {}
    for scenario in sorted({s.scenario for s in specs}):
        name = JAM_SCENARIOS[scenario].files
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
                       "smoke_defaults": context.selection.smoke_defaults}
                      if context.selection else None),
        "drain_thresholds": {"path": str(THRESHOLDS), "sha256": _sha256(THRESHOLDS)},
        "scenario_files": files, "episodes_planned": len(specs),
    }


def header_identity(record: dict) -> tuple:
    """`(checkpoint, feed artefact, camera model, predictor)` sha256 a header ran."""
    return ((record.get("checkpoint") or {}).get("sha256"),
            (record.get("feed_artefact") or {}).get("artefact_sha256"),
            (record.get("camera_model") or {}).get("sha256"),
            (record.get("predictor") or {}).get("sha256"))


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
    from src.sumo.jam_episode import FeedSetting, run_jam_episode

    v_jam, _ = load_v_jam()
    camera = predictor = None

    def run(spec: Spec) -> dict[str, Any]:
        nonlocal camera, predictor
        arm = spec.jam_arm()
        if (spec.arm in ("take_the_slower", "rule", "learned") or spec.sample_fraction > 0):
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
        feed = FeedSetting(context.feed_artefact, context.feed_session, m=spec.m,
                           allow_placeholder=context.allow_placeholder)
        scenario = JAM_SCENARIOS[spec.scenario]
        spec_camera = None
        if camera is not None:
            spec_camera = camera.with_bias(spec.bias).with_pair_sigma(
                camera.pair_sigma_log * spec.pair_sigma_scale)
        result = run_jam_episode(
            model, spec.seed, scenario, arm, feed=feed, duration_s=spec.duration_s,
            penetration=spec.penetration, camera=spec_camera,
            predictor=predictor, step_length_s=context.step_length_s,
            window_start_s=context.window_start_s, write_log=context.write_log,
            check_max_speed=context.check_max_speed, sample_fraction=spec.sample_fraction)
        result["jam"] = episode_metrics(result["trace"], v_jam=v_jam, start_s=scenario.start_s,
                                        end_s=scenario.end_s, cap=spec.duration_s)
        result["jam_30kmh"] = episode_metrics(result["trace"], v_jam=30.0,
                                              start_s=scenario.start_s, end_s=scenario.end_s,
                                              cap=spec.duration_s)
        return result

    return run


# ---------------------------------------------------------------- summaries


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def collision_counts(episodes: list[dict]) -> dict[str, Any]:
    """Every controlled arm must show zero colliding vehicles and collision teleports."""
    controlled = [r for r in episodes if r["arm"] != "no_control"]
    return {
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
            "return_mean": _mean(r.get("return") for r in records),
            "wall_s_mean": _mean(r.get("wall_s") for r in records),
            "by_seed": {str(r["seed"]): {"jam_time_min": r["jam"]["jam_time_min"],
                                         "jam_time_min_900_3600": r["jam"]["jam_time_min_900_3600"],
                                         "onset_s": r["jam"]["onset_s"], "flow": r["flow"],
                                         "prevented": r["jam"]["prevented"]}
                        for r in records},
        }
    return rows


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


def summarize_s(episodes: list[dict], context: Context) -> dict[str, Any]:
    """Criterion C over the selection seeds only (section 5.7)."""
    from src.analysis.jam_metrics import SELECTION_SCENARIOS, criterion_c

    outside = sorted({r["seed"] for r in episodes} - set(SELECTION_SEEDS))
    if outside and not context.smoke:
        raise SystemExit(f"the selection reads seeds 11-15 only; block S has records on "
                         f"{outside}")
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

    if context.smoke:
        # A smoke run's seeds are not selection seeds: nothing is selected from them.
        look = rule = learned = best = None
    return {
        "block": "S", "criterion": "C (plan section 5.7)", "seeds": sorted({r["seed"] for r in episodes}),
        "smoke_note": ("smoke run: candidates computed, nothing selected" if context.smoke
                       else None),
        "scenarios": scenarios, "candidates": candidates, "excluded": excluded,
        "selected": {"lookahead": params(look) or None, "rule": params(rule) or None,
                     "learned": params(learned) or None},
        "best": ({"arm": best.split("/")[0], "params": params(best), "name": best,
                  "C": pool[best]["C"]} if best else None),
        "tie_rule": f"within {C_TIE} of the lowest C, the highest BG served flow",
        "feed_artefact_sha256": context.feed.get("artefact_sha256"),
        "camera_model_sha256": _sha256(context.camera_model),
        "predictor_sha256": _sha256(context.predictor),
    }


def summarize_eval(block: str, episodes: list[dict]) -> dict[str, Any]:
    """Blocks M, L, P and K: per-arm rows, and paired jam time and flow against the plan."""
    from src.analysis.jam_metrics import Paired

    rows = _rows(episodes)
    for name, row in rows.items():
        base = f"plan/-/{row['scenario']}"
        if block == "L":
            base = f"plan/-_lag{name.split('_lag')[-1].split('/')[0]}/{row['scenario']}"
        elif block == "P":
            base = f"plan/-_p{name.split('_p')[-1].split('/')[0]}/{row['scenario']}"
        if base in rows and base != name:
            a = {int(s): v["jam_time_min"] for s, v in row["by_seed"].items()}
            b = {int(s): v["jam_time_min"] for s, v in rows[base]["by_seed"].items()}
            fa = {int(s): v["flow"] for s, v in row["by_seed"].items()}
            fb = {int(s): v["flow"] for s, v in rows[base]["by_seed"].items()}
            row["paired_jam_time_vs_plan"] = Paired.of(a, b).as_dict()
            row["paired_flow_vs_plan"] = Paired.of(fa, fb).as_dict()
    return {"block": block, "rows": rows}


def write_leads(episodes: list[dict], results_dir: Path) -> dict[str, Any]:
    """`leads.json` from block M's no-control episodes (section 4.3 and 6.2)."""
    from src.analysis.jam_metrics import leads, load_v_jam, max_ramp, speed_density_break
    from src.sumo.jam_episode import SCENARIOS as JAM_SCENARIOS

    v_jam, sha = load_v_jam()
    per, s45 = [], []
    for r in episodes:
        if r["arm"] != "no_control":
            continue
        trace = json.loads(Path(r["trace_file"]).read_text())["trace"]
        if r["scenario"] == "S45":
            s45.append(trace)
            continue
        scenario = JAM_SCENARIOS[r["scenario"]]
        lead = leads(trace, v_jam, scenario.start_s, r["duration_s"])
        ramp = (max_ramp(trace, lead["segment"], lead["onset"]) if lead["onset"] is not None
                else None)
        per.append({"scenario": r["scenario"], "seed": r["seed"], **lead, "ramp": ramp})
    out = {"v_jam": v_jam, "v_jam_sha256": sha, "episodes": per,
           "speed_density_break": speed_density_break(s45) if s45 else None}
    _write_json(results_dir / "leads.json", out)
    return out


def summarize(block: str, results_dir: Path, context: Context) -> dict[str, Any]:
    headers, episodes = read_block(results_dir, block)
    identities = {header_identity(h) for h in headers}
    if len(identities) > 1:
        raise SystemExit(f"block {block} mixes (checkpoint, feed, camera, predictor) "
                         f"{sorted(identities, key=str)}; it is not summarised")
    if block == "G":
        summary = summarize_g(episodes)
    elif block == "S":
        summary = summarize_s(episodes, context)
    elif block in ("T", "T_dagger"):
        summary = {"block": block, "rows": _rows(episodes),
                   "samples_files": sorted(r["samples_file"] for r in episodes
                                           if r.get("samples_file"))}
    else:
        summary = summarize_eval(block, episodes)
        if block == "M" and not context.smoke:
            summary["leads"] = write_leads(episodes, results_dir)
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
    _write_json(results_dir / f"{block}_summary.json", summary)
    return summary


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", nargs="+", choices=BLOCKS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="a subset of the block's own seeds (default all of them)")
    parser.add_argument("--feed-artefact", default=str(DEFAULT_FEED_ARTEFACT))
    parser.add_argument("--feed-session", default=DEFAULT_FEED_SESSION)
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="let the blocks replay a placeholder feed process")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--camera-model", default=str(DEFAULT_CAMERA))
    parser.add_argument("--predictor", default=str(DEFAULT_PREDICTOR))
    parser.add_argument("--selection", default=None,
                        help="S_summary.json for blocks M to K (default <results-dir>)")
    parser.add_argument("--results-dir", default=SMOKE_GUARD["results"])
    parser.add_argument("--traces-dir", default=SMOKE_GUARD["traces"])
    parser.add_argument("--training-dir", default="outputs/jam/training")
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


def main(argv: list[str] | None = None, run_episode_factory=None) -> int:
    from src.sumo.jam_episode import FeedSetting

    args = build_parser().parse_args(argv)
    if args.smoke:
        # A smoke episode must never land in a block's real JSONL, where --resume would
        # take its 600 s result for the experiment's own episode of the same key.
        if args.results_dir == SMOKE_GUARD["results"]:
            args.results_dir = "outputs/jam/smoke/results"
        if args.traces_dir == SMOKE_GUARD["traces"]:
            args.traces_dir = "outputs/jam/smoke/traces"
        if args.training_dir == "outputs/jam/training":
            args.training_dir = "outputs/jam/smoke/training"
    results_dir, traces_dir = _resolve(args.results_dir), _resolve(args.traces_dir)
    pause_file = _resolve(args.pause_file) if args.pause_file else results_dir / "PAUSE"
    context = Context(checkpoint=_resolve(args.checkpoint),
                      feed_artefact=_resolve(args.feed_artefact),
                      feed_session=args.feed_session, allow_placeholder=args.allow_placeholder,
                      camera_model=_resolve(args.camera_model),
                      predictor=_resolve(args.predictor),
                      selection_path=_resolve(args.selection) if args.selection else None,
                      window_start_s=300.0 if args.smoke else 900.0, smoke=args.smoke,
                      write_log=args.write_log or args.smoke,
                      check_max_speed=args.check_max_speed or args.smoke,
                      training_dir=_resolve(args.training_dir))
    context.feed = FeedSetting(context.feed_artefact, context.feed_session,
                               allow_placeholder=context.allow_placeholder).parameters()

    model = None
    for block in args.block:
        if block in ("M", "L", "P", "K"):
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
        specs = block_specs(block, args.seeds, context.selection, grid=args.g_grid,
                            dagger=dagger, smoke=args.smoke)
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
