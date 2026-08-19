from __future__ import annotations

import pytest

from src.analysis.simulator_health import (
    CellSpec,
    HealthVerdict,
    Run,
    assess_cell,
    run_condition,
    select_operating_points,
    summarise,
)

CELL = CellSpec(topology="merge", demand="high", av_penetration=0.10)


def make_run(controller, seed=7, *, completed=True, speed=20.0, jam=0.0, throughput=10.0, steps=120):
    return Run(
        cell=CELL, controller=controller, seed=seed, steps=steps, completed=completed,
        mean_speed=speed, jam_fraction=jam, throughput=throughput, collisions=0.0,
    )


def healthy_runs():
    """A cell that passes all four: congested reference, clear separation, all
    seeds complete, throughput preserved."""
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=12.0, jam=0.20, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=16.0, jam=0.10, throughput=19.0))
    return runs


# --------------------------------------------------------------------------
# unit tests: the aggregator against synthetic runs with known answers
# --------------------------------------------------------------------------


def test_cell_passing_all_four_is_healthy():
    v = assess_cell(CELL, healthy_runs())
    assert v.healthy is True
    assert v.failed_criteria == ()
    assert v.best_controller == "cooperative_smoothing"
    assert v.best_speed_separation == pytest.approx(4.0)
    assert v.reference_jam_fraction == pytest.approx(0.20)


def test_free_flow_cell_fails_congestion_only():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=22.53, jam=0.0, throughput=23.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=16.0, jam=0.0, throughput=22.0))
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("congestion_reachable",)


def test_indistinguishable_controllers_fail_separation_only():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=22.53, jam=0.20, throughput=23.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=22.51, jam=0.20, throughput=23.0))
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("baselines_separate",)
    assert v.best_speed_separation == pytest.approx(0.02, abs=1e-6)


def test_crashing_cell_fails_completion_only():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=12.0, jam=0.20, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, completed=(seed == 7), speed=16.0, throughput=19.0))
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("episodes_complete",)
    assert v.min_completed_seeds == 1


def test_throughput_collapse_is_caught_even_when_speed_improves():
    """The merge/high failure mode: speed up, jam down, throughput destroyed.
    Reporting that as a success would be claiming obstruction as control."""
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=12.40, jam=0.21, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=16.14, jam=0.10, throughput=3.0))
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("throughput_holds",)
    assert v.worst_throughput_ratio == pytest.approx(0.15)


def test_all_seeds_crashed_fails_completion():
    runs = [make_run("no_av", s, completed=False, speed=0.0, jam=1.0, throughput=0.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs)
    assert v.min_completed_seeds == 0
    assert v.episodes_complete is False


def test_cell_with_only_the_reference_controller_cannot_separate():
    runs = [make_run("no_av", s, speed=12.0, jam=0.20) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs)
    assert v.baselines_separate is False
    assert v.best_controller is None
    assert v.congestion_reachable is True


def test_zero_reference_throughput_does_not_divide_by_zero():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=0.0, jam=1.0, throughput=0.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=5.0, throughput=0.0))
    v = assess_cell(CELL, runs)
    assert v.worst_throughput_ratio == pytest.approx(1.0)
    assert v.throughput_holds is True


def test_summarise_averages_across_seeds_and_counts_completions():
    runs = [
        make_run("no_av", 7, speed=10.0, completed=True),
        make_run("no_av", 17, speed=20.0, completed=False),
    ]
    s = summarise(runs)["no_av"]
    assert s.seeds == 2
    assert s.completed_seeds == 1
    assert s.mean_speed == pytest.approx(15.0)


# --------------------------------------------------------------------------
# unit tests: the operating-point selector
# --------------------------------------------------------------------------


def test_selector_returns_empty_when_no_cell_qualifies():
    unhealthy = assess_cell(CELL, [make_run("no_av", s, jam=0.0) for s in (7, 17, 27)])
    assert select_operating_points([unhealthy]) == ()


def test_selector_ranks_by_control_effect_size():
    weak = assess_cell(CellSpec("merge", "high", 0.05), healthy_runs())
    strong_runs = []
    for seed in (7, 17, 27):
        strong_runs.append(make_run("no_av", seed, speed=10.0, jam=0.30, throughput=20.0))
        strong_runs.append(make_run("cooperative_smoothing", seed, speed=18.0, jam=0.10, throughput=19.0))
    strong = assess_cell(CellSpec("merge", "high", 0.20), strong_runs)
    ranked = select_operating_points([weak, strong])
    assert [v.cell.av_penetration for v in ranked] == [0.20, 0.05]


# --------------------------------------------------------------------------
# sanity tests: behavioural, against real runs
# --------------------------------------------------------------------------

SANITY_STEPS = 60


def cell_runs(topology, demand, controllers=("no_av", "cooperative_smoothing"), penetration=0.10, steps=SANITY_STEPS):
    cell = CellSpec(topology=topology, demand=demand, av_penetration=penetration)
    runs = [
        run_condition(cell, controller, seed, duration_steps=steps)
        for controller in controllers
        for seed in (7,)
    ]
    return cell, runs


def test_ring_is_not_a_usable_operating_point():
    """Ring under no_av is gridlock, not congestion: everything crashes and mean
    speed goes to zero. It must not be classified as healthy."""
    cell, runs = cell_runs("ring", "medium")
    verdict = assess_cell(cell, runs, min_completed_seeds=1)
    assert verdict.healthy is False


def test_free_flow_topology_fails_congestion():
    cell, runs = cell_runs("straight_multilane", "medium")
    verdict = assess_cell(cell, runs, min_completed_seeds=1)
    assert verdict.congestion_reachable is False, "multilane at medium should not congest"
    assert "congestion_reachable" in verdict.failed_criteria


def test_congestion_needs_time_to_develop():
    """Duration is itself a health parameter. On merge/high, jam_fraction is
    0.000 at 60 steps and 0.208 at 120: a short sweep reports no congestion
    anywhere, which is a false negative rather than a finding."""
    cell = CellSpec(topology="merge", demand="high", av_penetration=0.10)
    short = run_condition(cell, "no_av", 7, duration_steps=60)
    full = run_condition(cell, "no_av", 7, duration_steps=120)
    assert short.jam_fraction == pytest.approx(0.0)
    assert full.jam_fraction > short.jam_fraction


def test_merge_at_high_demand_congests():
    """The one cell known to congest; it is the reason the sweep is worth running."""
    cell, runs = cell_runs("merge", "high", controllers=("no_av",), steps=120)
    verdict = assess_cell(cell, runs, min_completed_seeds=1)
    assert verdict.reference_jam_fraction > 0.0
