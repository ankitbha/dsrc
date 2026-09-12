#!/usr/bin/env python3
"""Evaluate the committed Mainz checkpoints on seeds 16-30, paired against no control.

    .venv/bin/python scripts/evaluate_mainz_checkpoints.py \\
        > results/evaluation/mainz_paired_seeds.log

Reads `results/checkpoints/mainz_{here,src}_best.pt` and never writes them. For each
arm and each seed it runs one episode, then reports the paired gain over no control:
each seed is one traffic realisation that every arm ran, so the seed's contribution
to the level of flow is common to every arm and cancels when the comparison is a
per-seed difference rather than a difference of separately-computed means. The bar is
`2 * stdev(differences) / sqrt(n)` with the sample standard deviation, `n - 1`; see
`paired_statistics`.

`train_mainz_src.py`'s own `TEST_SEEDS` is `range(16, 21)`, five seeds, read once at
the end of training. This script reads seeds 16-30 against a checkpoint already on
disk, any number of times, which is a different job: re-running the driver to widen
the read would retrain and would keep a different checkpoint, because selection is by
validation return over a run that is not seeded identically end to end. `run_episode`,
`evaluate_no_control` and `SCENARIOS` are imported from `train_mainz_src` rather than
re-implemented, so the two never drift apart on what they measure.

The no-control baseline is recomputed here, per seed, in the same process: it does
not exist for seeds 21-30 anywhere, and pairing is only defined against the same
seed. `gate_for_arm` derives the entry-gate setting from the arm name in one place --
on for `here` and `src`, never for `no_control` -- and `_check_gate_asymmetry` checks
the run actually came out that way rather than assuming it.

The step length, episode duration and throughput-window start below are not this
script's defaults so much as a recovered fact: no command line was ever written down
for the checkpoints this reads, and `train_mainz_src.py`'s own argparse defaults
(`--step-length 1.0 --window-start-s 0.0`) reproduce neither the committed
`no_control` block nor a plausible reading of the training logs. `0.5s / 2500s / 900s`
does reproduce the committed `no_control` block on all seven of its metrics, and
`mean_vehicles = 858.0370370370371 = 115835/(5*27)` pins the window start
independently: 27 accumulation samples after ten 60s decisions past a 300s warmup is
t=900s. Both are recorded in every run's own `config` block so this does not have to
be recovered a second time.

This runs the action mapping that is actually in the tree, `SPEED_ACTION_FRACTIONS`
(a fraction of each edge's own speed limit), not the `SPEED_ACTIONS_KMH` literal
30/45/60 km/h mapping the committed checkpoints were originally read under. Every
controlled edge on Mainz has a free-flow speed of 60.012 km/h rather than 60.000, so
the two mappings differ by 0.02% in commanded speed -- enough to move a five-seed
mean flow by 8.4 veh/h. The stored `test` blocks in `results/checkpoints/*_result.json`
are therefore artifacts of a mapping this repository no longer contains; this script
does not carry a flag to reproduce it, because the only reason to would be to match a
number the code in the tree cannot produce.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_mainz_src import SCENARIOS, evaluate_no_control, run_episode  # noqa: E402
from record_deployed_commit import _git_commit, _git_is_dirty  # noqa: E402
from src.rl.src_q import SrcQNetwork  # noqa: E402
from src.sumo.mainz import HERE_FEATURES, MainzEnv, SRC_FEATURES  # noqa: E402

#: A trained arm's checkpoint file and its feature tuple, bound together under one
#: name so a checkpoint can never be pointed at the wrong feature tuple by passing
#: the two as independent flags.
CHECKPOINT_FEATURES: dict[str, tuple[str, ...]] = {"here": HERE_FEATURES, "src": SRC_FEATURES}

#: `zeroed_head` (plan section 8's falsification arm) is not a checkpoint of its
#: own -- it is this arm's checkpoint with `_zero_output_head` applied afterwards --
#: so it is bound to a real trained arm's checkpoint file rather than given one of
#: its own to keep in sync.
ZEROED_HEAD_BASE_ARM: str = "here"

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(16, 31))
DEFAULT_ARMS: tuple[str, ...] = ("no_control", "here", "src")
DEFAULT_STEP_LENGTH_S: float = 0.5
DEFAULT_DURATION_S: float = 2500.0
DEFAULT_WINDOW_START_S: float = 900.0

#: `MainzEnv`'s own default, read from the class rather than copied as a literal so
#: this cannot go stale the way the run's command line itself once did.
WARMUP_S: float = inspect.signature(MainzEnv).parameters["warmup_s"].default

#: The per-episode fields carried into the JSONL record and the per-seed table.
EPISODE_FIELDS: tuple[str, ...] = (
    "flow", "arrived", "return", "mean_speed_kmh", "space_mean_speed_kmh",
    "mean_vehicles", "held_at_entry",
)


# --------------------------------------------------------------------- checkpoints


def _num_segments(paths: dict) -> int:
    return len(json.loads((REPO_ROOT / paths["segments"]).read_text()))


def _build_and_load(checkpoint_path: Path, features: tuple[str, ...],
                    num_segments: int) -> SrcQNetwork:
    """Construct the network for `features` and load `checkpoint_path` into it.

    `load_state_dict` fails on a shape mismatch by itself, but the message it raises
    names tensor shapes rather than segments and features, so the width is checked
    directly first and the error names what actually disagrees.
    """
    model = SrcQNetwork(num_segments, len(features), 3)
    state_dict = torch.load(checkpoint_path, weights_only=True)
    expected_width = num_segments * len(features)
    checkpoint_width = state_dict["stack.0.weight"].shape[1]
    if checkpoint_width != expected_width:
        raise ValueError(
            f"{checkpoint_path} was built for an input width of {checkpoint_width}, "
            f"but {len(features)} features over {num_segments} segments need "
            f"{expected_width}; this checkpoint and this feature tuple do not match")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _checkpoint_name_for_arm(name: str) -> str:
    """The checkpoint file's own name for `name`: `zeroed_head` has no checkpoint of
    its own and reads `ZEROED_HEAD_BASE_ARM`'s file instead, so it can never drift
    from the checkpoint the arm it falsifies actually uses.
    """
    return ZEROED_HEAD_BASE_ARM if name == "zeroed_head" else name


def checkpoint_path_for_arm(name: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"mainz_{_checkpoint_name_for_arm(name)}_best.pt"


def _file_sha256(path: Path) -> str:
    """A sha256 over `path`'s bytes -- a path names where a checkpoint was read
    from, not what is in it, and a retrained checkpoint written to the same path is
    indistinguishable from the one it replaced unless its bytes are recorded too.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class LoadedArm:
    """One arm's loaded model, bound to the arm name, feature tuple, checkpoint path,
    digest and any post-load modification it was loaded under.

    Everything `run()` needs about a loaded arm comes from this one object instead of
    four separate dicts keyed on the arm name (`models`, `features_by_arm`, a
    checkpoint-path lookup and a sha256 lookup). Four independent dicts is four
    places a wrong key can be read from -- a bug that reads `models["here"]` for an
    arm named `"src"` leaves `features_by_arm["src"]`, the recorded checkpoint path
    and its sha256 all still correct, so nothing downstream disagrees with anything
    else and the mismatch has nowhere to surface. Binding all of it to `name` in one
    object at load time means `run()` has only one lookup (`loaded_arms[name]`) and
    one place to assert that what came back really was loaded for `name`.
    """

    name: str
    model: SrcQNetwork
    features: tuple[str, ...]
    checkpoint_path: Path
    sha256: str
    modification: str | None = None


def load_arm_model(name: str, checkpoint_dir: Path, num_segments: int) -> LoadedArm:
    """Bind a trained arm's checkpoint, feature tuple, path and digest, and load it.

    `zeroed_head` loads `ZEROED_HEAD_BASE_ARM`'s checkpoint and then zeros its
    output layer (`_zero_output_head`): plan section 8's falsification arm must run
    the same checkpoint file the arm it falsifies runs, or it is not testing that
    that arm's own result is distinguishable from an obviously wrong policy. The
    returned `LoadedArm.name` is `name` itself (`"zeroed_head"`, not `"here"`) so a
    caller can confirm what it asked for is what it got back; `modification` records
    that this arm's weights were changed after loading, which the checkpoint path and
    its sha256 alone do not say -- both still name `mainz_here_best.pt` and that
    file's own true digest, the same as the unmodified `here` arm reads.
    """
    checkpoint_name = _checkpoint_name_for_arm(name)
    if checkpoint_name not in CHECKPOINT_FEATURES:
        raise ValueError(f"{name!r} has no checkpoint; only here/src are trained arms")
    features = CHECKPOINT_FEATURES[checkpoint_name]
    checkpoint_path = checkpoint_path_for_arm(name, checkpoint_dir)
    model = _build_and_load(checkpoint_path, features, num_segments)
    modification = None
    if name == "zeroed_head":
        _zero_output_head(model)
        modification = ("stack[2] weight and bias zeroed after loading (plan section "
                        "8 falsification arm)")
    return LoadedArm(name=name, model=model, features=features,
                     checkpoint_path=checkpoint_path,
                     sha256=_file_sha256(checkpoint_path), modification=modification)


def gate_for_arm(name: str) -> bool:
    """Whether the entry gate is on for this arm: every trained arm, and only those.

    The gate stands in for the policy acting on a link above each entry link, which
    this scenario does not simulate (`src/sumo/mainz.py`'s `_admit`). A run with no
    policy has nothing acting on that link either, so gating its entries would credit
    it with a control action it never takes. This is derived from the arm's name in
    this one place rather than accepted as a caller-supplied flag.
    """
    return name != "no_control"


def _zero_output_head(model: SrcQNetwork) -> None:
    """Zero the output layer, so every Q-value is 0 and argmax picks index 0 always.

    With every controlled segment always choosing action 0, the policy becomes a
    standing order to drive every controlled vehicle at half its road's free-flow
    speed. Used to prove this evaluator reports an obviously wrong policy as wrong
    before its numbers on the real checkpoints are believed.
    """
    with torch.no_grad():
        model.stack[2].weight.zero_()
        model.stack[2].bias.zero_()


# ---------------------------------------------------------------------- statistics


def paired_statistics(arm_flows: list[float], baseline_flows: list[float]) -> dict:
    """The paired gain and its bar, seed by seed.

    `d_s = arm_flows[s] - baseline_flows[s]` for each seed, then the gain is the mean
    of `d_s` and the bar is `2 * stdev(d_s) / sqrt(n)` with the SAMPLE standard
    deviation (`statistics.stdev`, `n - 1`), not the population one
    (`statistics.pstdev`, `n`), and not `2 * sqrt(se_arm**2 + se_baseline**2)`, which
    is the same two numbers read as two independent means instead of one paired
    sample. Each seed is one traffic realisation that both the arm and the baseline
    ran, so the seed's own contribution to the level of flow is common to both and
    cancels in `d_s`; computing the bar from each side's separate spread does not
    cancel it and reports a wider interval that describes the wrong quantity.
    """
    if len(arm_flows) != len(baseline_flows):
        raise ValueError(
            f"paired statistics need one baseline reading per arm reading, got "
            f"{len(arm_flows)} and {len(baseline_flows)}")
    n = len(arm_flows)
    if n < 2:
        raise ValueError(f"a sample standard deviation needs at least two seeds, got {n}")
    differences = [a - b for a, b in zip(arm_flows, baseline_flows)]
    gain = statistics.fmean(differences)
    two_se = 2.0 * statistics.stdev(differences) / math.sqrt(n)
    baseline_mean = statistics.fmean(baseline_flows)
    return {
        "n": n,
        "differences": differences,
        "gain": gain,
        "two_se": two_se,
        "resolved": abs(gain) > two_se,
        "percent": 100.0 * gain / baseline_mean if baseline_mean else float("nan"),
    }


def per_arm_two_se(values: list[float]) -> float:
    """`2 * stdev(values) / sqrt(n)` for one arm's own readings, not the paired bar.

    Reported alongside the paired gain because the source documents quote it, but it
    describes one arm's spread across seeds and is never a substitute for
    `paired_statistics`'s bar.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"a sample standard deviation needs at least two seeds, got {n}")
    return 2.0 * statistics.stdev(values) / math.sqrt(n)


# ------------------------------------------------------------------------- the run


def _schedule_rows(paths: dict) -> list[dict]:
    return json.loads((REPO_ROOT / paths["schedule"]).read_text())


def _entry_headways(paths: dict) -> dict[str, float | None]:
    """Each entry link's own constant inter-departure headway, in seconds -- or
    `None` for an entry link whose departures are not spaced at one constant gap.

    The committed schedule departs each entry link at one fixed headway throughout
    (140/182 every 4.8s, 277/289 every 5.76s, 293/295 every 7.2s, 231/255 every
    9.6s), so reading two consecutive departures off one entry link recovers the
    generator's own parameter losslessly, rather than backing a rate out of an
    aggregate (`count / span`, `count / duration_s`) that carries the aggregation's
    own assumption. This is not assumed to hold generally: `build_mainz_scenario.py
    --demand-rush-s` instead ramps each entry through several fixed rates over the
    run (piecewise-constant, not constant), so an entry link built that way has more
    than one distinct gap and reports `None` here instead of a rate silently averaged
    across regimes that were never meant to be pooled.
    """
    by_entry: dict[str, list[float]] = {}
    for row in _schedule_rows(paths):
        by_entry.setdefault(row["entry"], []).append(row["depart"])
    headways: dict[str, float | None] = {}
    for entry, departures in by_entry.items():
        departures.sort()
        gaps = {round(later - earlier, 6)
               for earlier, later in zip(departures, departures[1:])}
        headways[entry] = gaps.pop() if len(gaps) == 1 else None
    return headways


def _demand_veh_per_h(paths: dict) -> float | None:
    """The schedule's own arrival rate, summed one entry link at a time.

    `3600 / headway` is one entry link's own rate; the schedule's total rate is the
    SUM of all eight, not their average or a pooled `count / span` -- eight entry
    links each departing every `headway` seconds put `sum(3600 / headway_e)`
    vehicles onto the road per hour, which is what a solver reading this schedule as
    demand actually sees, and it does not move with how long an episode is
    configured to run (`duration_s` is not a parameter here at all). `None` when any
    entry link's own headway is not the single constant `_entry_headways` needs to
    trust it -- see `_demand_unavailable_reason` for which link and why.
    """
    headways = _entry_headways(paths)
    if any(headway is None for headway in headways.values()):
        return None
    return sum(3600.0 / headway for headway in headways.values())


def _demand_unavailable_reason(paths: dict) -> str | None:
    """Why `_demand_veh_per_h` returned `None`, naming the entry link(s) at fault --
    `None` itself when every entry link's headway was recovered cleanly.
    """
    uneven = sorted(entry for entry, headway in _entry_headways(paths).items()
                    if headway is None)
    if not uneven:
        return None
    return (f"entry link(s) {', '.join(uneven)} do not depart at one constant "
            f"headway -- the schedule's own rate is not recoverable losslessly from "
            f"them (a --demand-rush-s schedule departs at a piecewise-constant rate "
            f"and would trip this)")


def _scheduled_departures(paths: dict) -> int:
    """The schedule's own total row count -- fixed by the file, independent of how
    long an episode is configured to run.
    """
    return len(_schedule_rows(paths))


def _departures_within_episode(paths: dict, duration_s: float) -> int:
    """How many of the schedule's own departures fall inside `[0, duration_s]`.

    This is the quantity that actually varies with `duration_s` -- the schedule's
    own rate does not (`_demand_veh_per_h` no longer takes `duration_s` at all). A
    shorter episode simply sees fewer of the schedule's departures, which is the
    original defect's real content: `duration_s` was wired into a RATE calculation
    it should never have touched, when the COUNT was the thing that was ever going
    to move with it.
    """
    return sum(1 for row in _schedule_rows(paths) if row["depart"] <= duration_s)


def _validate_no_duplicates(*, arms: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    """Reject a duplicate arm or seed before any episode runs.

    A duplicate seed silently shrinks the sample: `seeds=(16, 17, 17, 18)` would run
    four episodes but only three distinct traffic realisations, `config["seeds"]`
    would still list four entries, and every `paired` statistic would be computed
    over `n=3` with nothing in the artifact saying the sample was smaller than it
    claimed. A duplicate arm re-runs and re-records the same (arm, seed) pair twice.
    """
    if len(set(arms)) != len(arms):
        raise ValueError(f"duplicate arms in {arms!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {seeds!r}")


def _validate_seed_count(seeds: tuple[int, ...]) -> None:
    """At least two distinct seeds: `paired_statistics` needs `n >= 2` for a sample
    standard deviation and otherwise raises only after every episode has already
    run, discarding the simulation time it cost to get there.
    """
    if len(set(seeds)) < 2:
        raise ValueError(f"need at least two distinct seeds, got {seeds!r}")


def _provenance() -> dict:
    """Git commit and dirty state, at whatever moment this is called.

    Reuses `record_deployed_commit.py`'s `_git_commit`/`_git_is_dirty` (return-code
    checked, a bounded subprocess timeout, `OSError`/`SubprocessError` both caught)
    rather than a second implementation. The implementation this module wrote for
    itself called `git rev-parse HEAD` with `check=True` from inside the summary
    build, after the episode loop -- so a tree with no `.git` (an untracked rsync
    copy, a `git archive` extraction) raised `CalledProcessError` only once every
    episode had already run, and the run was unrecoverable because `--resume` does
    not exist (plan decision 10). Absence is now a recorded fact instead of a crash.

    `run()` calls this twice: once before the first episode (so a tree with no
    `.git` cannot discard a completed run) and once after the last one. A run of 45
    episodes takes minutes, long enough to span a commit made by someone else working
    in the same tree, and `MainzEnv.reset()`/`_sumo.start()` both re-read the schedule
    and network files from disk every episode -- so the two samples are compared
    (`tree_unchanged_during_run`) rather than either one being trusted alone, and
    disagreement is recorded, never raised: the whole reason F1 stopped raising on an
    unreadable commit was so an unrelated provenance question could not discard a
    completed run, and a changed-mid-run tree is exactly that kind of question, not a
    reason to throw the run away.
    """
    commit = _git_commit(REPO_ROOT)
    if commit is None:
        return {
            "commit": None,
            "dirty": None,
            "commit_unavailable_reason":
                "`git rev-parse HEAD` did not succeed in this tree -- not a git "
                "checkout, git is unreachable, or there is no commit yet",
        }
    return {"commit": commit, "dirty": _git_is_dirty(REPO_ROOT), "commit_unavailable_reason": None}


def _header_record(run_id: str, config: dict, provenance: dict) -> dict:
    """The JSONL's own first line: this run's id, config and starting provenance, so
    a reader holding only the `.jsonl` file -- the summary not yet written, or lost
    separately -- can still tell which run produced it, under what configuration,
    and at what commit. `run_id` is what lets a reader holding BOTH files confirm
    they actually belong to the same run (`run()`'s post-rename check).
    """
    return {"type": "header", "run_id": run_id, "config": config, **provenance}


def _compact_metrics(metrics: dict) -> dict:
    """The serialisable subset of a `run_episode` / `evaluate_no_control` result.

    `run_episode` also returns `states`, `actions` and `rewards` -- the trajectory
    tensors `td_loss` trains on -- which this evaluator never trains with and must
    not carry into a JSON file.
    """
    return {field: metrics[field] for field in EPISODE_FIELDS}


def _episode_record(arm: str, seed: int, metrics: dict) -> dict:
    return {"arm": arm, "seed": seed, **_compact_metrics(metrics)}


def _check_gate_asymmetry(per_seed: dict[str, dict[int, dict]]) -> None:
    """Confirm the entry gate came out asymmetric rather than assuming it did.

    Every `no_control` episode must show `held_at_entry == 0`; if any trained arm ran,
    at least one of its episodes must show `held_at_entry > 0`. Either direction of
    failure means the gate was applied to the wrong arm and the comparison this script
    exists to make is not the comparison that actually ran.
    """
    if "no_control" in per_seed:
        for seed, metrics in per_seed["no_control"].items():
            if metrics["held_at_entry"] != 0:
                raise AssertionError(
                    f"no_control seed {seed} held {metrics['held_at_entry']} vehicles "
                    "at entry; the gate must never apply to an uncontrolled run")
    trained_metrics = [
        metrics for name, seeds in per_seed.items() if name != "no_control"
        for metrics in seeds.values()
    ]
    if trained_metrics and not any(m["held_at_entry"] > 0 for m in trained_metrics):
        raise AssertionError(
            "no trained-arm episode held any vehicle at entry; the entry gate does "
            "not appear to be active")


def _build_summary(*, run_id: str, commit: str | None, dirty: bool | None,
                   commit_unavailable_reason: str | None, provenance_end: dict,
                   tree_unchanged_during_run: bool, config: dict,
                   per_seed: dict[str, dict[int, dict]]) -> dict:
    arms_summary = {}
    for name, seed_metrics in per_seed.items():
        rows = list(seed_metrics.values())
        flows = [row["flow"] for row in rows]
        arms_summary[name] = {
            "flow_mean": statistics.fmean(flows),
            "flow_2se": per_arm_two_se(flows) if len(flows) >= 2 else None,
            "return_mean": statistics.fmean(row["return"] for row in rows),
            "arrived_mean": statistics.fmean(row["arrived"] for row in rows),
            "mean_speed_kmh_mean": statistics.fmean(row["mean_speed_kmh"] for row in rows),
            "space_mean_speed_kmh_mean":
                statistics.fmean(row["space_mean_speed_kmh"] for row in rows),
            "mean_vehicles_mean": statistics.fmean(row["mean_vehicles"] for row in rows),
            "held_at_entry_mean": statistics.fmean(row["held_at_entry"] for row in rows),
        }

    paired = {}
    baseline_seed_metrics = per_seed.get("no_control")
    if baseline_seed_metrics:
        for name, seed_metrics in per_seed.items():
            if name == "no_control":
                continue
            common_seeds = sorted(set(seed_metrics) & set(baseline_seed_metrics))
            if not common_seeds:
                continue
            arm_flows = [seed_metrics[seed]["flow"] for seed in common_seeds]
            baseline_flows = [baseline_seed_metrics[seed]["flow"] for seed in common_seeds]
            paired[name] = paired_statistics(arm_flows, baseline_flows)

    return {"run_id": run_id, "commit": commit, "dirty": dirty,
            "commit_unavailable_reason": commit_unavailable_reason,
            "provenance_end": provenance_end,
            "tree_unchanged_during_run": tree_unchanged_during_run,
            "config": config, "per_seed": per_seed,
            "arms": arms_summary, "paired": paired}


def run(*, arms: tuple[str, ...] = DEFAULT_ARMS, seeds: tuple[int, ...] = DEFAULT_SEEDS,
        step_length: float = DEFAULT_STEP_LENGTH_S,
        duration_s: float = DEFAULT_DURATION_S,
        window_start_s: float = DEFAULT_WINDOW_START_S,
        checkpoint_dir: Path, out_dir: Path, scenario: str = "mainz",
        provenance_start: dict | None = None) -> dict:
    """Run every (arm, seed) episode once, sequentially, and write both artifacts.

    Sequential and strictly one environment at a time: `libsumo` is process-global
    (`src/sumo/binding.py`), so a second episode may only start after the first has
    closed its own, which `run_episode` and `evaluate_no_control` each guarantee with
    a `finally`.

    `provenance_start`, if given, is used as-is instead of calling `_provenance()`
    again -- `main()` already reads it once to print the commit banner before this is
    called, and a second, independent read moments later is not guaranteed to agree
    (a commit landing in that gap would make the banner and the header lie to each
    other about the same run). Left `None` when `run()` is called directly, as every
    test in this module does.
    """
    _validate_no_duplicates(arms=arms, seeds=seeds)
    _validate_seed_count(seeds)

    paths = SCENARIOS[scenario]
    num_segments = _num_segments(paths)

    loaded_arms: dict[str, LoadedArm] = {}
    checkpoints_used: dict[str, dict] = {}
    for name in arms:
        if name == "no_control":
            continue
        loaded = load_arm_model(name, checkpoint_dir, num_segments)
        assert loaded.name == name, (
            f"load_arm_model({name!r}, ...) returned a LoadedArm bound to "
            f"{loaded.name!r}; its model, features, checkpoint path and sha256 all "
            f"travel together precisely so this cannot happen without this failing")
        loaded_arms[name] = loaded
        checkpoints_used[name] = {"path": str(loaded.checkpoint_path),
                                  "sha256": loaded.sha256,
                                  "modification": loaded.modification}

    # Read before the first episode runs, not after the loop: a tree with no `.git`
    # must not be able to discard a completed run just because its provenance is
    # unreadable (F1).
    provenance_start = provenance_start if provenance_start is not None else _provenance()
    run_id = uuid.uuid4().hex
    config = {
        "seeds": list(seeds),
        "duration_s": duration_s,
        "step_length_s": step_length,
        "window_start_s": window_start_s,
        "warmup_s": WARMUP_S,
        "demand_veh_per_h": _demand_veh_per_h(paths),
        "demand_unavailable_reason": _demand_unavailable_reason(paths),
        "scheduled_departures": _scheduled_departures(paths),
        "departures_within_episode": _departures_within_episode(paths, duration_s),
        "scenario": scenario,
        "action_set": "SPEED_ACTION_FRACTIONS",
        "checkpoints": checkpoints_used,
        "gate_entries": {name: gate_for_arm(name) for name in arms},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "mainz_paired_seeds.jsonl"
    partial_jsonl_path = out_dir / "mainz_paired_seeds.jsonl.partial"
    per_seed: dict[str, dict[int, dict]] = {name: {} for name in arms}

    with partial_jsonl_path.open("w") as handle:
        handle.write(json.dumps(_header_record(run_id, config, provenance_start)) + "\n")
        handle.flush()
        for name in arms:
            for seed in seeds:
                if name == "no_control":
                    # `features` only reaches `observe()`, and the no-control loop
                    # inside `evaluate_no_control` never calls it -- every vehicle is
                    # left to the car-following model, so no observation is ever
                    # built. HERE_FEATURES is passed because the parameter is
                    # positional, not because it does anything here.
                    metrics = evaluate_no_control((seed,), HERE_FEATURES, duration_s,
                                                 step_length, window_start_s, paths)
                else:
                    loaded = loaded_arms[name]
                    metrics = run_episode(loaded.model, seed, loaded.features,
                                          duration_s, 0.0, None, step_length,
                                          window_start_s, gate_for_arm(name), paths)
                record = _episode_record(name, seed, metrics)
                per_seed[name][seed] = _compact_metrics(metrics)
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    _check_gate_asymmetry(per_seed)

    # Sampled again now that every episode has actually run: `MainzEnv.reset()` and
    # `_sumo.start()` both re-read the schedule and network files from disk every
    # episode, so a run of 45 episodes can genuinely span a commit that changes that
    # data mid-run -- the header's own provenance, read before the loop, would not
    # say so. Compared, never raised on: see `_provenance`'s own docstring for why.
    provenance_end = _provenance()
    tree_unchanged_during_run = (
        provenance_start["commit"] == provenance_end["commit"]
        and provenance_start["dirty"] == provenance_end["dirty"])

    summary = _build_summary(
        run_id=run_id, commit=provenance_start["commit"], dirty=provenance_start["dirty"],
        commit_unavailable_reason=provenance_start["commit_unavailable_reason"],
        provenance_end=provenance_end, tree_unchanged_during_run=tree_unchanged_during_run,
        config=config, per_seed=per_seed)

    # Whole-then-rename: a reader never sees a partially written summary. The summary
    # is renamed into place BEFORE the JSONL: if this rename fails (disk full, a
    # permissions change, a concurrent writer to the same `out_dir`), the JSONL's own
    # rename below must never run, so a previous run's `mainz_paired_seeds.jsonl` is
    # left completely untouched rather than paired with this run's summary, or with
    # no summary at all -- a narrower re-run must not be able to make a wider
    # previous run's JSONL disappear until its own replacement is actually done (F4).
    summary_path = out_dir / "mainz_paired_seeds.json"
    tmp_path = out_dir / "mainz_paired_seeds.json.tmp"
    tmp_path.write_text(json.dumps(summary, indent=1))
    tmp_path.rename(summary_path)
    partial_jsonl_path.rename(jsonl_path)

    # Neither rename can be made atomic with the other on a POSIX filesystem, so a
    # second process writing into this same `out_dir` at exactly the wrong moment
    # could still race between them. `run_id` exists so that race is at least
    # detectable rather than silent: re-read both files' own `run_id` back off disk
    # -- not the in-memory `summary` and `run_id` above, which would not see a
    # concurrent writer's overwrite -- and refuse to return a summary that does not
    # match the JSONL now sitting beside it. This is detection, not prevention; a
    # per-run directory would close the residual race, which is not worth the
    # restructuring here.
    written_jsonl_run_id = json.loads(jsonl_path.read_text().splitlines()[0])["run_id"]
    written_summary_run_id = json.loads(summary_path.read_text())["run_id"]
    if written_jsonl_run_id != written_summary_run_id:
        raise RuntimeError(
            f"{jsonl_path} now reads run_id {written_jsonl_run_id!r} but "
            f"{summary_path} reads {written_summary_run_id!r}; another writer "
            f"touched {out_dir} between this run's own two renames")

    return summary


# --------------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="mainz")
    parser.add_argument("--arms", nargs="+",
                        choices=("no_control", "here", "src", "zeroed_head"),
                        default=list(DEFAULT_ARMS),
                        help="zeroed_head is the falsification arm (plan section 8): "
                             "the here checkpoint with its output layer zeroed")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="paired against no_control seed for seed; the committed "
                             "checkpoints were selected on seeds 11-15 and never see "
                             "these during training")
    parser.add_argument("--step-length", type=float, default=DEFAULT_STEP_LENGTH_S)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--window-start-s", type=float, default=DEFAULT_WINDOW_START_S,
                        help="flow is counted only from here; the ramp-up before it "
                             "is excluded from every reported rate")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--out", default="results/evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = REPO_ROOT / args.out
    checkpoint_dir = REPO_ROOT / args.checkpoint_dir

    # Before the banner, and before any episode: a duplicate or single-seed request
    # should fail here rather than after 45 episodes have already run (F5, F6).
    _validate_no_duplicates(arms=tuple(args.arms), seeds=tuple(args.seeds))
    _validate_seed_count(tuple(args.seeds))

    # Sampled once and passed into `run()` below, rather than each calling
    # `_provenance()` independently: two reads moments apart are not guaranteed to
    # agree, and this banner and the run's own header should never disagree about
    # what they are both describing as "this run".
    provenance_start = _provenance()
    print(f"scenario {args.scenario}  arms {args.arms}  "
          f"seeds {args.seeds[0]}-{args.seeds[-1]} ({len(args.seeds)} total)")
    print(f"step-length {args.step_length}s  duration {args.duration_s}s  "
          f"window-start {args.window_start_s}s  warmup {WARMUP_S}s")
    if provenance_start["commit"] is None:
        print(f"commit unavailable: {provenance_start['commit_unavailable_reason']}")
    else:
        dirty = provenance_start["dirty"]
        dirty_label = "dirty" if dirty else ("clean" if dirty is False else "dirty unknown")
        print(f"commit {provenance_start['commit'][:12]} ({dirty_label})")

    started = time.time()
    summary = run(arms=tuple(args.arms), seeds=tuple(args.seeds),
                  step_length=args.step_length, duration_s=args.duration_s,
                  window_start_s=args.window_start_s, checkpoint_dir=checkpoint_dir,
                  provenance_start=provenance_start,
                  out_dir=out_dir, scenario=args.scenario)
    elapsed = time.time() - started

    print()
    for name, stats in summary["arms"].items():
        bar = f" +/- {stats['flow_2se']:.1f}" if stats["flow_2se"] is not None else ""
        print(f"  {name:<10} flow {stats['flow_mean']:>8.2f} veh/h{bar}")
    for name, stats in summary["paired"].items():
        status = "resolved" if stats["resolved"] else "NOT resolved"
        print(f"  {name} vs no_control: paired gain {stats['gain']:+.2f} "
              f"+/- {stats['two_se']:.1f} veh/h ({stats['percent']:+.1f}%) -- {status}")

    commit_display = summary["commit"][:12] if summary["commit"] else "unavailable"
    print(f"\nwrote {out_dir / 'mainz_paired_seeds.json'} and "
          f"{out_dir / 'mainz_paired_seeds.jsonl'} in {elapsed:.0f}s, at commit "
          f"{commit_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
