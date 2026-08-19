"""Whether this simulator can produce a traffic-control result, and where.

The simulator is the only source of a flow-level claim, so if it cannot show a
control effect the work above it has no outcome. This classifies each
(topology, demand, penetration) cell against four criteria and reports which
cells, if any, are usable.

A cell is healthy only if all four hold: congestion is reachable, so there is
something to control; baselines separate, so an effect is detectable at all;
episodes complete, so metrics are not truncated by a crash; and throughput does
not collapse, so an apparent improvement is control rather than obstruction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.analysis.observation_audit import SampleSpec, build_env_config
from src.baselines import make_baseline
from src.envs.topology_env import HighwayTopologyEnv

REFERENCE_CONTROLLER = "no_av"
DEFAULT_MIN_JAM_FRACTION = 0.05
DEFAULT_MIN_SPEED_SEPARATION_MPS = 1.0
DEFAULT_MIN_COMPLETED_SEEDS = 2
DEFAULT_MIN_THROUGHPUT_RATIO = 0.8


@dataclass(frozen=True)
class CellSpec:
    topology: str
    demand: str
    av_penetration: float

    @property
    def cell_id(self) -> str:
        return f"{self.topology}/{self.demand}/pen{self.av_penetration:g}"


@dataclass(frozen=True)
class Run:
    """One (cell, controller, seed) execution.

    `completed` distinguishes ending by configured duration from ending on a
    crash; a crashed run's terminal metrics are not comparable with a completed
    run's, so the two are never pooled without the split staying visible.
    """

    cell: CellSpec
    controller: str
    seed: int
    steps: int
    completed: bool
    mean_speed: float
    jam_fraction: float
    throughput: float
    collisions: float


@dataclass(frozen=True)
class ControllerSummary:
    controller: str
    seeds: int
    completed_seeds: int
    mean_speed: float
    jam_fraction: float
    throughput: float
    collisions: float


@dataclass(frozen=True)
class HealthVerdict:
    """The four criteria plus the measured value behind each.

    The values matter as much as the booleans: a cell that fails by a hair and
    one that fails by an order of magnitude call for different responses.
    """

    cell: CellSpec
    by_controller: Mapping[str, ControllerSummary]
    congestion_reachable: bool
    reference_jam_fraction: float
    baselines_separate: bool
    best_speed_separation: float
    best_controller: str | None
    episodes_complete: bool
    min_completed_seeds: int
    throughput_holds: bool
    worst_throughput_ratio: float

    @property
    def healthy(self) -> bool:
        return (
            self.congestion_reachable
            and self.baselines_separate
            and self.episodes_complete
            and self.throughput_holds
        )

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        checks = (
            ("congestion_reachable", self.congestion_reachable),
            ("baselines_separate", self.baselines_separate),
            ("episodes_complete", self.episodes_complete),
            ("throughput_holds", self.throughput_holds),
        )
        return tuple(name for name, ok in checks if not ok)


def run_condition(
    cell: CellSpec,
    controller: str,
    seed: int,
    *,
    duration_steps: int = 120,
    human_model: str = "normal",
) -> Run:
    spec = SampleSpec(
        topology=cell.topology,
        controller=controller,
        seed=seed,
        duration_steps=duration_steps,
        demand=cell.demand,
        human_model=human_model,
        av_penetration=cell.av_penetration,
    )
    env = HighwayTopologyEnv(cell.topology, build_env_config(spec))
    policy = make_baseline(controller)
    policy.reset(env_metadata={"topology_id": cell.topology}, seed=seed)
    observations, _ = env.reset(seed=seed)

    steps = 0
    terminated = False
    truncated = False
    metrics: Mapping[str, Any] = {}
    while not (terminated or truncated):
        observations, _, terminated, truncated, info = env.step(policy.act(observations, global_state=None))
        metrics = info.get("metrics", {})
        steps += 1
    return Run(
        cell=cell,
        controller=controller,
        seed=seed,
        steps=steps,
        completed=not terminated,
        mean_speed=_number(metrics.get("mean_speed")),
        jam_fraction=_number(metrics.get("jam_fraction")),
        throughput=_number(metrics.get("throughput_recent")),
        collisions=_number(metrics.get("collision_count")),
    )


def summarise(runs: Sequence[Run]) -> dict[str, ControllerSummary]:
    by_controller: dict[str, list[Run]] = {}
    for run in runs:
        by_controller.setdefault(run.controller, []).append(run)
    return {
        controller: ControllerSummary(
            controller=controller,
            seeds=len(group),
            completed_seeds=sum(1 for run in group if run.completed),
            mean_speed=mean(run.mean_speed for run in group),
            jam_fraction=mean(run.jam_fraction for run in group),
            throughput=mean(run.throughput for run in group),
            collisions=mean(run.collisions for run in group),
        )
        for controller, group in by_controller.items()
    }


def assess_cell(
    cell: CellSpec,
    runs: Sequence[Run],
    *,
    min_jam_fraction: float = DEFAULT_MIN_JAM_FRACTION,
    min_speed_separation: float = DEFAULT_MIN_SPEED_SEPARATION_MPS,
    min_completed_seeds: int = DEFAULT_MIN_COMPLETED_SEEDS,
    min_throughput_ratio: float = DEFAULT_MIN_THROUGHPUT_RATIO,
) -> HealthVerdict:
    """Apply the four criteria to one cell's runs. Pure: no environment access."""
    summaries = summarise(runs)
    reference = summaries.get(REFERENCE_CONTROLLER)
    treatments = {name: s for name, s in summaries.items() if name != REFERENCE_CONTROLLER}

    reference_jam = reference.jam_fraction if reference else 0.0
    congestion = bool(reference) and reference_jam >= min_jam_fraction

    separation = 0.0
    best_controller: str | None = None
    for name, summary in treatments.items():
        if reference is None:
            continue
        delta = abs(summary.mean_speed - reference.mean_speed)
        if delta > separation:
            separation, best_controller = delta, name
    separates = bool(treatments) and separation >= min_speed_separation

    completed = min((s.completed_seeds for s in summaries.values()), default=0)
    complete = bool(summaries) and completed >= min_completed_seeds

    # A reference throughput of zero cannot be improved on or collapse below;
    # treat the ratio as undefined-but-passing rather than dividing by zero.
    ratio = 1.0
    if reference and reference.throughput > 0 and treatments:
        ratio = min(s.throughput / reference.throughput for s in treatments.values())
    holds = ratio >= min_throughput_ratio

    return HealthVerdict(
        cell=cell,
        by_controller=summaries,
        congestion_reachable=congestion,
        reference_jam_fraction=reference_jam,
        baselines_separate=separates,
        best_speed_separation=separation,
        best_controller=best_controller,
        episodes_complete=complete,
        min_completed_seeds=completed,
        throughput_holds=holds,
        worst_throughput_ratio=ratio,
    )


def select_operating_points(verdicts: Sequence[HealthVerdict]) -> tuple[HealthVerdict, ...]:
    """Healthy cells, strongest control effect first.

    Returns empty when no cell passes all four criteria, which is a legitimate
    outcome and means the simulator cannot currently support a flow-level claim.
    """
    healthy = [v for v in verdicts if v.healthy]
    return tuple(sorted(healthy, key=lambda v: v.best_speed_separation, reverse=True))


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result == result and abs(result) != float("inf") else 0.0
