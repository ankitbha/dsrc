#!/usr/bin/env python3
"""Run the feed-agreement experiment blocks A to E, with resume and pause.

    .venv/bin/python scripts/run_feed_experiments.py --block B C D E
    .venv/bin/python scripts/run_feed_experiments.py --block A            # needs polling
    .venv/bin/python scripts/run_feed_experiments.py --block A --allow-placeholder
    .venv/bin/python scripts/run_feed_experiments.py --block D --pilot
    .venv/bin/python scripts/run_feed_experiments.py --block B --summarize-only
    .venv/bin/python scripts/run_feed_experiments.py --block A B C D E --smoke

Plan sections 3.7 and 5. Every episode is keyed `(block, arm, variant, seed)`. One
JSONL per block under `results/feed/{block}.jsonl`, appended and flushed after every
episode; its traces under `outputs/feed/traces/{block}/`. The summary
`results/feed/{block}_summary.json` is rebuilt from the JSONL at the end of a run, or
alone with `--summarize-only`, written whole-then-rename.

* `--resume` skips every key already in the block's JSONL.
* Between episodes the runner checks the pause file (default `results/feed/PAUSE`).
  When it exists the runner writes `results/feed/{block}_state.json` naming the next
  key and exits 0 with `paused before <key>`. Remove the file and re-run with
  `--resume` to continue.
* Block A replays the feed-process artefact and refuses a placeholder (the drive logs,
  before polling exists) unless `--allow-placeholder` is passed; the flag is carried
  into the header and the summary.
* `--smoke` runs one episode of each named block on seed 16 for 600 s, into
  `outputs/feed/smoke/` unless `--results-dir`/`--traces-dir` say otherwise, to prove
  the entry point runs end to end. It is not an experiment, and it never writes into
  `results/feed/`, where `--resume` would take its 600 s episode for the real one.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_mainz_checkpoints import _provenance, paired_statistics  # noqa: E402
from train_mainz_src import SCENARIOS  # noqa: E402

BLOCKS = ("A", "B", "C", "D", "E")
DEFAULT_SEEDS = tuple(range(16, 31))
DEFAULT_ARTEFACT = REPO_ROOT / "results" / "feed" / "feed_process.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "checkpoints" / "mainz_here_best.pt"
COMMITTED_EVALUATION = REPO_ROOT / "results" / "evaluation" / "mainz_paired_seeds.jsonl"
P_Q = 10.0
D_OFFSETS = (-20.0, -5.0, 0.0, 5.0, 10.0)
DELTAS = (0.1, 0.25, 0.5, 0.75)
STALENESS = (30.0, 60.0, 120.0, 180.0, 300.0)
PENETRATIONS = (0.75, 0.5, 0.25, 0.1)
CERTIFICATE_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.1)
LINEARITY_SEEDS = (16, 17, 18)
#: Plan D12 and D13: drain scenarios, their caps and the instant the clock starts.
DRAIN = {"mainz_surge": {"cap_s": 9000.0, "t_0": 1200.0, "onset_from": 0.0},
         "mainz_burst": {"cap_s": 12000.0, "t_0": 1500.0, "onset_from": 900.0}}
HOLD_S = 120.0
HOLD_SENSITIVITIES = (60.0, 300.0)
PULSE_START_S = 1200.0
PULSE_DURATION_S = 60.0
DEVIATION_WINDOW = (1200.0, 1800.0)
SMOKE_SEED = 16
SMOKE_DURATION_S = 600.0


# --------------------------------------------------------------------- specs


@dataclass(frozen=True)
class Spec:
    block: str
    arm: str
    variant: str
    seed: int
    scenario: str = "mainz"
    penetration: float = 1.0
    duration_s: float = 2500.0
    pulse: tuple | None = None          # (site, n, start_s, duration_s)
    mode_args: tuple = ()

    @property
    def key(self) -> str:
        return f"{self.block}/{self.arm}/{self.variant}/{self.seed}"

    def trace_name(self) -> str:
        return f"{self.arm}_{self.variant}_{self.seed}.json"


@dataclass
class Context:
    """What the specs and modes of a run depend on, recorded in the header."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    duration_s: float = 2500.0
    artefact: Path | None = None
    session: str | None = None
    allow_placeholder: bool = False
    start_s: float = 60.0
    P_q: float = P_Q
    pulse_start_s: float = PULSE_START_S
    #: Flow and disagreement are scored from here; a 600 s smoke scores from the
    #: warm-up's end so its disagreement and age are measured at all.
    window_start_s: float = 900.0
    #: How advice reaches a vehicle (`src.sumo.mainz.ACTUATIONS`); Amendment 1's cap.
    actuation: str = "max_speed"
    smoke: bool = False
    feed: dict[str, Any] = field(default_factory=dict)  # block A's parameters

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["artefact"] = str(self.artefact) if self.artefact else None
        return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_feed_parameters(artefact: Path, session: str | None, allow_placeholder: bool,
                         P_q: float = P_Q) -> dict[str, Any]:
    """Block A's lemma and rule parameters, read from the replayed session.

    `L_bar` and `psi*` are the session's own, because the rule arms replay that session's
    lags exactly: the simulated process's largest lag is its `L_bar`. The measured
    bracket's `J` and the replay's own lag spread are both recorded, with the lemma's
    part (a) condition evaluated for each.
    """
    data = json.loads(Path(artefact).read_text())
    if data.get("placeholder") and not allow_placeholder:
        raise SystemExit(f"{artefact} is a placeholder (source {data.get('primary_source')}); "
                         "block A replays the polling artefact. Pass --allow-placeholder to "
                         "run it on the drive-log process anyway.")
    session = session or data["primary"]["default_session"]
    s = data["sessions"][session]
    g_min = s["g_min"]
    J_measured = s["J"]
    J_simulated = s["replayed_lag_spread"]
    return {
        "artefact": str(artefact), "artefact_sha256": _sha256(artefact),
        "placeholder": bool(data.get("placeholder")), "session": session,
        "source": s.get("source"), "L_bar": s["L_bar"], "psi_star": s["psi_star"],
        "r_bar": s["r_bar"], "r_bar_source": s["r_bar_source"], "g_min": g_min,
        "J_measured": J_measured, "J_simulated": J_simulated,
        "min_availability_gap": s.get("min_availability_gap"),
        "P_q": P_q,
        "D_offsets": list(D_OFFSETS),
        "D_values": [s["L_bar"] + offset for offset in D_OFFSETS],
        "condition_measured": (J_measured is not None and P_q <= g_min - J_measured),
        "condition_simulated": P_q <= g_min - J_simulated,
        "no_skip_in_replay": (s.get("min_availability_gap") or 0.0) >= P_q,
    }


def block_specs(block: str, context: Context) -> list[Spec]:
    """Every episode of a block, arm-major, each arm over the same seed list."""
    seeds, duration = context.seeds, context.duration_s
    arms: list[tuple[str, str, dict]] = []
    if block == "A":
        feed = context.feed
        arms = [("published", "-", {}), ("synchronous", "-", {}),
                ("no_rule", "P60", {"mode_args": (60.0,)}),
                ("no_rule", "P10", {"mode_args": (10.0,)}),
                ("fixed_phase", f"psi{feed['psi_star']}",
                 {"mode_args": (float(feed["psi_star"]), float(feed["r_bar"]))})]
        arms += [("version_keyed", f"D{offset:+g}", {"mode_args": (context.P_q, D)})
                 for offset, D in zip(feed["D_offsets"], feed["D_values"])]
    elif block == "B":
        arms = [("inject_delta", f"delta{d:g}", {"mode_args": (d,)}) for d in DELTAS]
        arms += [("inject_stale", f"s{s:g}", {"mode_args": (s,)}) for s in STALENESS]
    elif block == "C":
        arms = [("penetration", f"p{p:g}", {"penetration": p}) for p in PENETRATIONS]
    elif block == "D":
        for scenario, setting in DRAIN.items():
            cap = SMOKE_DURATION_S if context.smoke else setting["cap_s"]
            for arm in ("no_control", "here"):
                arms.append((arm, scenario, {"scenario": scenario, "duration_s": cap}))
    elif block == "E":
        specs = []
        for size, seed_list in ((60, seeds), (30, tuple(s for s in seeds
                                                        if s in LINEARITY_SEEDS))):
            for p in CERTIFICATE_LEVELS:
                for site in range(12):
                    for seed in seed_list:
                        specs.append(Spec("E", f"pulse{size}", f"p{p:g}_site{site}", seed,
                                          penetration=p, duration_s=duration,
                                          pulse=(site, size, context.pulse_start_s,
                                                 PULSE_DURATION_S)))
        return specs
    else:
        raise ValueError(f"unknown block {block}")
    specs = []
    for arm, variant, extra in arms:
        for seed in seeds:
            specs.append(Spec(block, arm, variant, seed,
                              duration_s=extra.get("duration_s", duration),
                              scenario=extra.get("scenario", "mainz"),
                              penetration=extra.get("penetration", 1.0),
                              mode_args=extra.get("mode_args", ())))
    return specs


def baseline_spec(p: float, seed: int, context: Context) -> Spec:
    """Block E's unperturbed episode at penetration `p`, when no earlier block has it."""
    return Spec("E", "baseline", f"p{p:g}", seed, penetration=p,
                duration_s=context.duration_s)


def make_mode(spec: Spec):
    from src.sumo.feed_episode import Mode

    arm, args = spec.arm, spec.mode_args
    if arm in ("published", "here", "penetration", "pulse60", "pulse30", "baseline"):
        return Mode.published()
    if arm == "no_control":
        return Mode.no_control()
    if arm == "synchronous":
        return Mode.synchronous()
    if arm == "no_rule":
        return Mode.no_rule(*args)
    if arm == "fixed_phase":
        return Mode.fixed_phase(*args)
    if arm == "version_keyed":
        return Mode.version_keyed(*args)
    if arm == "inject_delta":
        return Mode.inject_delta(*args)
    if arm == "inject_stale":
        return Mode.inject_stale(*args)
    raise ValueError(f"no mode for arm {arm}")


# -------------------------------------------------------------------- header


def _sumo_version() -> str | None:
    binary = REPO_ROOT / ".venv" / "bin" / "sumo"
    try:
        result = subprocess.run([str(binary if binary.exists() else "sumo"), "--version"],
                                capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _libsumo_version() -> str | None:
    try:
        import libsumo

        return getattr(libsumo, "__version__", None)
    except ImportError:
        return None


def header(block: str, context: Context, checkpoint: Path,
           argv: Sequence[str] | None = None) -> dict[str, Any]:
    scenarios = sorted(DRAIN) if block == "D" else ["mainz"]
    scenario_files = {}
    for name in scenarios:
        for role, relative in SCENARIOS[name].items():
            path = REPO_ROOT / relative
            scenario_files[f"{name}:{role}"] = {"path": relative,
                                                "sha256": _sha256(path) if path.exists() else None}
    return {
        "type": "header", "block": block, "started_wall": time.time(),
        **_provenance(),
        "sumo_version": _sumo_version(), "libsumo_version": _libsumo_version(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "command": list(argv) if argv is not None else sys.argv,
        "config": context.as_dict(),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "feed_artefact": ({"path": context.feed["artefact"],
                           "sha256": context.feed["artefact_sha256"],
                           "session": context.feed["session"],
                           "placeholder": context.feed["placeholder"],
                           "allow_placeholder": context.allow_placeholder}
                          if block == "A" else None),
        "scenario_files": scenario_files,
    }


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


def check_pause(pause_file: Path | None) -> bool:
    return pause_file is not None and Path(pause_file).exists()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, default=float))
    temporary.rename(path)


EpisodeFn = Callable[[Spec], dict[str, Any]]
#: Fields of an episode result kept in the trace file rather than the JSONL.
TRACE_FIELDS = ("trace", "snapshots", "late_draws")


def run_block(block: str, specs: Sequence[Spec], run_episode: EpisodeFn, *,
              results_dir: Path, traces_dir: Path, header_record: dict[str, Any],
              resume: bool = False, pause_file: Path | None = None) -> dict[str, Any]:
    """Run the specs not yet in the block's JSONL, one at a time, pausing on request."""
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl = results_dir / f"{block}.jsonl"
    if jsonl.exists() and not resume:
        raise SystemExit(f"{jsonl} exists; pass --resume to continue it, or move it away")
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
            if check_pause(pause_file):
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
            _write_json(trace_path, {"key": spec.key,
                                     **{k: result.get(k) for k in TRACE_FIELDS}})
            record = {"type": "episode", "key": spec.key, "block": block, "arm": spec.arm,
                      "variant": spec.variant, "seed": spec.seed, "scenario": spec.scenario,
                      "penetration": spec.penetration, "duration_s": spec.duration_s,
                      "pulse": list(spec.pulse) if spec.pulse else None, "wall_s": wall,
                      "trace_file": str(trace_path),
                      **{("pulse_inserted" if k == "pulse" else k): v
                         for k, v in result.items() if k not in TRACE_FIELDS}}
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
    from src.sumo.feed import FeedProcess
    from src.sumo.feed_episode import run_feed_episode

    def run(spec: Spec) -> dict[str, Any]:
        mode = make_mode(spec)
        feed = None
        if mode.kind == "rule":
            feed = FeedProcess.from_artefact(Path(context.feed["artefact"]),
                                             context.feed["session"], spec.seed,
                                             context.start_s, spec.duration_s + 180.0)
        pulse = None
        if spec.pulse:
            site, n, start, duration = spec.pulse
            pulse = {"site": site, "n": n, "start_s": start, "duration_s": duration}
        return run_feed_episode(model, spec.seed, spec.scenario, mode, feed_process=feed,
                                penetration=spec.penetration, duration_s=spec.duration_s,
                                pulse=pulse, window_start_s=context.window_start_s,
                                actuation=context.actuation)

    return run


# ---------------------------------------------------------------- summaries


def read_block(results_dir: Path, block: str) -> tuple[list[dict], list[dict]]:
    """`(headers, episodes)`; a later record for the same key replaces an earlier one."""
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


def committed_references(path: Path = COMMITTED_EVALUATION) -> dict[str, dict[int, float]]:
    """Per-seed flow of the committed `here` (equal to `published`) and `no_control`."""
    out: dict[str, dict[int, float]] = {"here": {}, "no_control": {}}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("arm") in out:
            out[record["arm"]][int(record["seed"])] = float(record["flow"])
    return out


def _paired(arm: dict[int, float], reference: dict[int, float]) -> dict[str, Any] | None:
    seeds = sorted(set(arm) & set(reference))
    if len(seeds) < 2:
        return {"seeds": seeds, "note": "fewer than two paired seeds"} if seeds else None
    stats = paired_statistics([arm[s] for s in seeds], [reference[s] for s in seeds])
    return {"seeds": seeds, **stats}


def _group(episodes: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in episodes:
        groups.setdefault((record["arm"], record["variant"]), []).append(record)
    return groups


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _arm_rows(episodes: list[dict]) -> dict[str, Any]:
    rows = {}
    for (arm, variant), records in sorted(_group(episodes).items()):
        rows[f"{arm}/{variant}"] = {
            "arm": arm, "variant": variant, "seeds": sorted(r["seed"] for r in records),
            "flow_mean": _mean(r["flow"] for r in records),
            "delta_mean": _mean(r.get("delta_mean") for r in records),
            "command_delta_mean": _mean(r.get("command_delta_mean") for r in records),
            "age_mean_s": _mean(r.get("age_mean_s") for r in records),
            "set_speed_writes_mean": _mean(r.get("set_speed_writes") for r in records),
            "commanded_ids_mean": _mean(r.get("commanded_ids") for r in records),
            "wall_s_mean": _mean(r.get("wall_s") for r in records),
            "flow_by_seed": {str(r["seed"]): r["flow"] for r in records},
        }
    return rows


def origin_fit(points: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
    """Least-squares `y = b1 x` and `y = b1 x + b2 x^2`, both through the origin (D25).

    The quadratic term is resolved when `|b2|` exceeds twice its standard error.
    """
    import numpy as np

    if len(points) < 3:
        return None
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    slope = float((x * y).sum() / (x * x).sum())
    design = np.stack([x, x * x], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    dof = len(y) - 2
    out = {"slope": slope, "n": len(points), "quadratic": coefficients.tolist()}
    if dof > 0:
        sigma2 = float(residual @ residual) / dof
        covariance = sigma2 * np.linalg.inv(design.T @ design)
        se = float(math.sqrt(max(covariance[1, 1], 0.0)))
        out.update(quadratic_se=se, quadratic_resolved=bool(abs(coefficients[1]) > 2 * se))
    return out


def summarize_a(episodes, headers, context: Context) -> dict[str, Any]:
    from src.sumo.feed import lemma_cycle_average, lemma_skip_bound

    rows = _arm_rows(episodes)
    published = {r["seed"]: r["flow"] for r in episodes if r["arm"] == "published"}
    for row in rows.values():
        row["paired_vs_published"] = _paired(
            {int(s): f for s, f in row["flow_by_seed"].items()}, published)
    feed = (headers[-1]["config"]["feed"] if headers else context.feed) or {}
    lemma = []
    if feed:
        for offset, D in zip(feed.get("D_offsets", []), feed.get("D_values", [])):
            measured = rows.get(f"version_keyed/D{offset:+g}", {}).get("delta_mean")
            lemma.append({
                "D_minus_L_bar": offset, "D": D,
                "bound_cycle_average_g_min": lemma_cycle_average(D, feed["L_bar"], feed["P_q"],
                                                                 feed["g_min"]),
                "bound_cycle_average_g60": lemma_cycle_average(D, feed["L_bar"], feed["P_q"], 60.0),
                "measured_delta_mean": measured})
    condition = feed.get("condition_simulated")
    return {
        "block": "A", "rows": rows, "lemma": lemma,
        "feed": feed,
        "placeholder": feed.get("placeholder"),
        "lemma_condition": {
            "g_min": feed.get("g_min"), "P_q": feed.get("P_q"),
            "J_measured": feed.get("J_measured"), "J_simulated": feed.get("J_simulated"),
            "P_q_le_g_min_minus_J_measured": feed.get("condition_measured"),
            "P_q_le_g_min_minus_J_simulated": condition,
            "skip_bound_measured": (lemma_skip_bound(feed["P_q"], feed["g_min"], feed["J_measured"])
                                    if feed.get("J_measured") is not None else None),
            "skip_bound_simulated": (lemma_skip_bound(feed["P_q"], feed["g_min"], feed["J_simulated"])
                                     if feed.get("J_simulated") is not None else None),
            "zero_disagreement_claimed": bool(feed.get("condition_measured")),
        },
    }


def summarize_b(episodes, references) -> dict[str, Any]:
    rows = _arm_rows(episodes)
    published = references["here"]
    delta_points, stale_points = [], []
    for row in rows.values():
        by_seed = {int(s): f for s, f in row["flow_by_seed"].items()}
        row["paired_vs_published"] = _paired(by_seed, published)
        row["reference"] = "committed here rows (results/evaluation/mainz_paired_seeds.jsonl)"
        for seed, flow in by_seed.items():
            if seed not in published:
                continue
            loss = published[seed] - flow
            if row["arm"] == "inject_delta":
                delta_points.append((float(row["variant"][len("delta"):]), loss))
            elif row["arm"] == "inject_stale":
                stale_points.append((float(row["variant"][1:]), loss))
    return {"block": "B", "rows": rows,
            "fit_loss_vs_delta": origin_fit(delta_points),
            "fit_loss_vs_staleness": origin_fit(stale_points),
            "loss_points_delta": delta_points, "loss_points_staleness": stale_points}


def summarize_c(episodes, references) -> dict[str, Any]:
    rows = _arm_rows(episodes)
    for row in rows.values():
        by_seed = {int(s): f for s, f in row["flow_by_seed"].items()}
        row["paired_vs_published"] = _paired(by_seed, references["here"])
        row["paired_vs_no_control"] = _paired(by_seed, references["no_control"])
        row["penetration"] = float(row["variant"][1:])
    loss_points = []
    for row in rows.values():
        for seed, flow in row["flow_by_seed"].items():
            if int(seed) in references["here"]:
                loss_points.append((1.0 - row["penetration"],
                                    references["here"][int(seed)] - flow))
    return {"block": "C", "rows": rows, "loss_points_one_minus_p": loss_points,
            "references": {"p=1": "committed here rows", "p=0": "committed no_control rows"}}


def _load_trace(record: dict) -> list[dict] | None:
    path = Path(record["trace_file"])
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("trace")


def pilot_decision(no_control_trace: list[dict], here_trace: list[dict], cap: float,
                   t_0: float = 1500.0, until: float = 3000.0) -> dict[str, Any]:
    """Plan section 5's pilot rule for scenario (b).

    The burst breaks no control down if its served flow over [t_0, until] is below the
    3,650 veh/h background and its pending count at `until` exceeds that at `t_0`. The
    controlled arm must reach held + pending = 0 before the cap.
    """
    def at(trace, t):
        return next((s for s in trace if s["t"] >= t), trace[-1])

    start, end = at(no_control_trace, t_0), at(no_control_trace, until)
    served = (end["arrived"] - start["arrived"]) * 3600.0 / (end["t"] - start["t"])
    breaks = served < 3650.0 and end["pending"] > start["pending"]
    cleared = next((s["t"] for s in here_trace if s["t"] > t_0
                    and s["held"] + s["pending"] == 0), None)
    controlled_drains = cleared is not None and cleared <= cap
    if not breaks:
        action = ("rebuild mainz_burst with a 6,000 veh/h burst and a 16,000 s cap: "
                  "--demand-burst 900 1500 0.333333 --duration-s 16000 "
                  "--demand-duration-s 16000, and repeat the pilot once")
    elif not controlled_drains:
        action = ("rebuild mainz_burst with a 3,600 veh/h background: "
                  "--demand-scale 0.2, and repeat the pilot once")
    else:
        action = "keep the scenario"
    if action != "keep the scenario":
        action += ("; move results/feed/D.jsonl and outputs/feed/traces/D away first, "
                   "since the pilot's episodes are of the old scenario")
    return {"served_flow_no_control": served, "pending_at_t0": start["pending"],
            "pending_at_until": end["pending"], "breaks_no_control_down": breaks,
            "controlled_clears_at": cleared, "controlled_drains_before_cap": controlled_drains,
            "action": action}


def summarize_d(episodes, smoke: bool = False,
                results_dir: Path | None = None) -> dict[str, Any]:
    from src.sumo.drain import (
        drain_summary, drain_time, jam_segment_minutes, onset, thresholds_from_traces,
    )

    out: dict[str, Any] = {"block": "D", "scenarios": {}}
    for scenario, setting in DRAIN.items():
        records = [r for r in episodes if r["scenario"] == scenario]
        if not records:
            continue
        cap = records[0]["duration_s"]
        t_0 = setting["t_0"]
        traces = {(r["arm"], r["seed"]): _load_trace(r) for r in records}
        entry: dict[str, Any] = {"t_0": t_0, "cap_s": cap, "hold_s": HOLD_S,
                                 "hold_sensitivities_s": list(HOLD_SENSITIVITIES)}
        if cap <= t_0:
            entry["note"] = "the episode ends before the drain clock starts (smoke run)"
            entry["episodes"] = [{"arm": r["arm"], "seed": r["seed"], "flow": r["flow"]}
                                 for r in records]
            out["scenarios"][scenario] = entry
            continue
        no_control = [traces[k] for k in traces if k[0] == "no_control" and traces[k]]
        if not no_control:
            entry["note"] = "no no_control trace to fit the jam threshold on"
            out["scenarios"][scenario] = entry
            continue
        thresholds = thresholds_from_traces(no_control, t_0, cap)
        entry["thresholds"] = thresholds
        rules = {"pooled": thresholds["pooled"]["v_jam"],
                 "per_segment": [s["v_jam"] for s in thresholds["per_segment"]],
                 "fixed_30": thresholds["fixed_kmh"]}
        import numpy as np

        results: dict[str, Any] = {}
        for rule, v_jam in rules.items():
            v = np.asarray(v_jam, float)
            for hold in (HOLD_S, *HOLD_SENSITIVITIES):
                per_seed: dict[str, dict[int, tuple[float, bool]]] = {}
                extra = {}
                for (arm, seed), trace in traces.items():
                    if trace is None:
                        continue
                    per_seed.setdefault(arm, {})[seed] = drain_time(trace, t_0, hold, cap, v)
                    extra[f"{arm}/{seed}"] = {
                        "onset": onset(trace, setting["onset_from"], v),
                        "jam_segment_minutes": jam_segment_minutes(trace, v,
                                                                   setting["onset_from"], cap)}
                results[f"{rule}/H{hold:g}"] = {**drain_summary(per_seed, cap - t_0),
                                                "per_episode": extra}
        entry["drain"] = results
        entry["primary"] = f"pooled/H{HOLD_S:g}"
        out["scenarios"][scenario] = entry
    if results_dir is not None:
        _write_json(results_dir / "drain_thresholds.json",
                    {name: entry["thresholds"] for name, entry in out["scenarios"].items()
                     if "thresholds" in entry})
    return out


def summarize_e(episodes, traces_dir: Path, results_dir: Path) -> dict[str, Any]:
    from src.analysis.gains import deviations, estimate_level, linearity_ratio, min_penetration

    def baseline_trace(p: float, seed: int):
        candidates = []
        if p == 1.0:
            candidates.append(traces_dir / "A" / f"published_-_{seed}.json")
        else:
            candidates.append(traces_dir / "C" / f"penetration_p{p:g}_{seed}.json")
        candidates.append(traces_dir / "E" / f"baseline_p{p:g}_{seed}.json")
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text())["trace"], str(path)
        return None, None

    by_level: dict[float, dict[int, dict[int, dict]]] = {}
    small: dict[float, dict[int, dict[int, dict]]] = {}
    for record in episodes:
        if not record["arm"].startswith("pulse"):
            continue
        site = int(record["pulse"][0])
        target = by_level if record["arm"] == "pulse60" else small
        target.setdefault(record["penetration"], {}).setdefault(record["seed"], {})[site] = record
    levels: dict[str, Any] = {}
    rho_by_level: dict[float, float | None] = {}
    missing = []
    for p in sorted(by_level, reverse=True):
        seed_runs, size_checks = {}, []
        for seed, sites in sorted(by_level[p].items()):
            base, base_path = baseline_trace(p, seed)
            if base is None:
                missing.append(f"p{p:g}/seed{seed}")
                continue
            if len(sites) < 12:
                continue
            runs = []
            for site in range(12):
                trace = _load_trace(sites[site])
                runs.append(deviations(trace, base, *DEVIATION_WINDOW))
            seed_runs[seed] = runs
        entry: dict[str, Any] = {"seeds_used": sorted(seed_runs)}
        if len(seed_runs) >= 2:
            entry.update(estimate_level(seed_runs))
            rho_by_level[p] = entry["rho_UCB"]
        else:
            entry["note"] = "fewer than two complete seeds: no bound"
            rho_by_level[p] = None
        ratios = []
        for seed, sites in sorted(small.get(p, {}).items()):
            base, _ = baseline_trace(p, seed)
            if base is None:
                continue
            for site, record in sorted(sites.items()):
                large = by_level.get(p, {}).get(seed, {}).get(site)
                if large is None:
                    continue
                y60 = deviations(_load_trace(large), base, *DEVIATION_WINDOW)
                y30 = deviations(_load_trace(record), base, *DEVIATION_WINDOW)
                ratios.append({"seed": seed, "site": site,
                               "ratio": linearity_ratio(y60, y30, site)})
        entry["linearity"] = ratios
        entry["linearity_outside_1p6_2p4"] = sum(
            1 for r in ratios if not (1.6 <= r["ratio"] <= 2.4))
        levels[f"{p:g}"] = entry
    inserted = _inserted_differences(episodes, results_dir)
    summary = {"block": "E", "levels": levels, "missing_baselines": missing,
               "inserted_difference": inserted,
               "p_min": min_penetration(rho_by_level) if rho_by_level else None,
               "window_s": list(DEVIATION_WINDOW)}
    _write_json(results_dir / "certificate.json", summary)
    return summary


def _accounted(record: dict) -> int:
    return (record["arrived"] + record["running"] + record["pending"]
            + record["held_at_entry"])


def _inserted_differences(episodes: list[dict], results_dir: Path) -> dict[str, Any]:
    """Pulsed minus unperturbed `arrived + running + pending + held`, which must be the
    pulse size exactly (plan attack point 11)."""
    baselines: dict[tuple[float, int], dict] = {}
    for block, arm in (("A", "published"), ("C", "penetration"), ("E", "baseline")):
        for record in read_block(results_dir, block)[1]:
            if record["arm"] == arm:
                baselines.setdefault((record["penetration"], record["seed"]), record)
    checked, mismatched, unmatched = 0, [], 0
    for record in episodes:
        if not record["arm"].startswith("pulse"):
            continue
        base = baselines.get((record["penetration"], record["seed"]))
        if base is None or base["duration_s"] != record["duration_s"]:
            unmatched += 1
            continue
        checked += 1
        difference = _accounted(record) - _accounted(base)
        if difference != record["pulse"][1]:
            mismatched.append({"key": record["key"], "difference": difference})
    return {"checked": checked, "mismatched": mismatched, "without_baseline": unmatched}


def collision_counts(episodes: list[dict]) -> dict[str, Any]:
    """Plan Amendment 1: no reported number may silently contain collisions."""
    controlled = [r for r in episodes if r["arm"] != "no_control"]
    return {
        "controlled_episodes_with_collisions": sum(
            1 for r in controlled
            if r.get("colliding_vehicles", 0) or r.get("collision_teleports", 0)),
        "controlled_episodes": len(controlled),
        "colliding_vehicles_total": sum(r.get("colliding_vehicles", 0) for r in episodes),
        "collision_teleports_total": sum(r.get("collision_teleports", 0) for r in episodes),
        "actuations": sorted({r.get("actuation", "unrecorded") for r in episodes}),
    }


def summarize(block: str, results_dir: Path, traces_dir: Path,
              context: Context) -> dict[str, Any]:
    headers, episodes = read_block(results_dir, block)
    references = committed_references()
    if block == "A":
        summary = summarize_a(episodes, headers, context)
    elif block == "B":
        summary = summarize_b(episodes, references)
    elif block == "C":
        summary = summarize_c(episodes, references)
    elif block == "D":
        summary = summarize_d(episodes, context.smoke, results_dir)
        pilot_path = results_dir / "D_pilot.json"
        if pilot_path.exists():
            summary["pilot"] = json.loads(pilot_path.read_text())
    elif block == "E":
        summary = summarize_e(episodes, traces_dir, results_dir)
    else:
        raise ValueError(block)
    seeds_by_arm = {}
    for record in episodes:
        seeds_by_arm.setdefault(f"{record['arm']}/{record['variant']}", set()).add(record["seed"])
    summary.update({
        "episodes": len(episodes),
        **collision_counts(episodes),
        "seeds_by_arm": {k: sorted(v) for k, v in sorted(seeds_by_arm.items())},
        "identical_seed_lists": len({tuple(sorted(v)) for v in seeds_by_arm.values()}) <= 1,
        "headers": [{k: h.get(k) for k in ("started_wall", "commit", "dirty", "sumo_version",
                                            "libsumo_version", "resumed")} for h in headers],
        "wall_s_total": sum(r.get("wall_s", 0.0) for r in episodes),
        "smoke": bool(headers and headers[-1].get("config", {}).get("smoke")),
    })
    _write_json(results_dir / f"{block}_summary.json", summary)
    return summary


# ----------------------------------------------------------------------- CLI


SMOKE_GUARD = {"results": "results/feed", "traces": "outputs/feed/traces"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", nargs="+", choices=BLOCKS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--duration-s", type=float, default=2500.0)
    parser.add_argument("--feed-artefact", default=str(DEFAULT_ARTEFACT))
    parser.add_argument("--feed-session", default=None,
                        help="the artefact session block A replays; default its own default")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="let block A replay the drive-log placeholder process")
    parser.add_argument("--start-s", type=float, default=60.0,
                        help="simulation time of the replayed process's first stamp, moved "
                             "forward to that stamp's second of the minute")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--actuation", choices=("max_speed", "set_speed"),
                        default="max_speed",
                        help="max_speed caps the vehicle's maximum speed (Amendment 1); "
                             "set_speed is the legacy actuation that collides")
    parser.add_argument("--results-dir", default="results/feed")
    parser.add_argument("--traces-dir", default="outputs/feed/traces")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-file", default=None,
                        help="default <results-dir>/PAUSE")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="block D only: run seed 16 on mainz_burst in both arms and apply "
                             "the plan's pilot rule")
    parser.add_argument("--smoke", action="store_true",
                        help=f"one episode per block on seed {SMOKE_SEED}, "
                             f"{SMOKE_DURATION_S:g} s")
    return parser


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def main(argv: list[str] | None = None, run_episode_factory=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        # A smoke episode must never land in a block's real JSONL, where --resume would
        # take its 600 s result for the experiment's own episode of the same key.
        if args.results_dir == SMOKE_GUARD["results"]:
            args.results_dir = "outputs/feed/smoke/results"
        if args.traces_dir == SMOKE_GUARD["traces"]:
            args.traces_dir = "outputs/feed/smoke/traces"
    results_dir, traces_dir = _resolve(args.results_dir), _resolve(args.traces_dir)
    pause_file = _resolve(args.pause_file) if args.pause_file else results_dir / "PAUSE"
    checkpoint = _resolve(args.checkpoint)
    context = Context(seeds=tuple(args.seeds), duration_s=args.duration_s,
                      artefact=_resolve(args.feed_artefact), session=args.feed_session,
                      allow_placeholder=args.allow_placeholder, start_s=args.start_s,
                      smoke=args.smoke, actuation=args.actuation)
    if args.smoke:
        context.seeds = (SMOKE_SEED,)
        context.duration_s = SMOKE_DURATION_S
        context.pulse_start_s = 400.0
        context.window_start_s = 300.0
    if len(set(context.seeds)) != len(context.seeds):
        raise SystemExit("duplicate seeds")
    if "A" in args.block:
        context.feed = load_feed_parameters(context.artefact, context.session,
                                            context.allow_placeholder, context.P_q)
        context.session = context.feed["session"]

    model = None
    if not args.summarize_only:
        if run_episode_factory is None:
            import evaluate_mainz_checkpoints as emc

            model = emc.load_arm_model("here", checkpoint.parent,
                                       emc._num_segments(SCENARIOS["mainz"])).model
            run_episode_factory = episode_runner
    for block in args.block:
        if args.summarize_only:
            summarize(block, results_dir, traces_dir, context)
            print(f"wrote {results_dir / f'{block}_summary.json'}")
            continue
        specs = block_specs(block, context)
        if block == "D" and args.pilot:
            specs = [s for s in specs if s.scenario == "mainz_burst" and s.seed == SMOKE_SEED]
        if block == "E":
            specs = _with_baselines(specs, context, traces_dir)
        if args.smoke:
            # One episode: block A's version-keyed rule at D = L_bar (it replays the
            # artefact), and every other block's first arm that is not a reused baseline.
            wanted = ((lambda s: s.arm == "version_keyed" and s.variant == "D+0")
                      if block == "A" else (lambda s: s.arm != "baseline"))
            specs = [s for s in specs if wanted(s)][:1]
        record = header(block, context, checkpoint, argv if argv is not None else sys.argv)
        result = run_block(block, specs, run_episode_factory(model, context),
                           results_dir=results_dir, traces_dir=traces_dir,
                           header_record=record, resume=args.resume or args.pilot,
                           pause_file=pause_file)
        if result["status"] == "paused":
            return 0
        if block == "D" and args.pilot:
            _, episodes = read_block(results_dir, "D")
            pilot = [r for r in episodes if r["scenario"] == "mainz_burst"
                     and r["seed"] == SMOKE_SEED]
            traces = {r["arm"]: _load_trace(r) for r in pilot}
            decision = pilot_decision(traces["no_control"], traces["here"],
                                      DRAIN["mainz_burst"]["cap_s"])
            _write_json(results_dir / "D_pilot.json", decision)
            print(f"pilot: {decision['action']}")
        summarize(block, results_dir, traces_dir, context)
        print(f"block {block}: ran {len(result['ran'])}, skipped {len(result['skipped'])}; "
              f"wrote {results_dir / f'{block}_summary.json'}")
    return 0


def _with_baselines(specs: list[Spec], context: Context, traces_dir: Path) -> list[Spec]:
    """Add block E's unperturbed episodes for every (level, seed) no earlier block traced."""
    needed = sorted({(s.penetration, s.seed) for s in specs}, reverse=True)
    extra = []
    for p, seed in needed:
        reused = (traces_dir / "A" / f"published_-_{seed}.json" if p == 1.0 else
                  traces_dir / "C" / f"penetration_p{p:g}_{seed}.json")
        if not reused.exists():
            extra.append(baseline_spec(p, seed, context))
    return extra + specs


if __name__ == "__main__":
    raise SystemExit(main())
