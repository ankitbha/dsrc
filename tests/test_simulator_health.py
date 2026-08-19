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
    assert v.healthy is False


def test_indistinguishable_controllers_fail_separation_only():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=22.53, jam=0.20, throughput=23.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=22.51, jam=0.20, throughput=23.0))
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("baselines_separate",)
    assert v.healthy is False
    assert v.best_speed_separation == pytest.approx(0.02, abs=1e-6)


def test_crashing_cell_fails_completion_only():
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=12.0, jam=0.20, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, completed=(seed == 7), speed=16.0, throughput=19.0))
    # crashed seeds carry wildly different metrics; if they were pooled the
    # summary means would move
    v = assess_cell(CELL, runs)
    assert v.failed_criteria == ("episodes_complete",)
    assert v.healthy is False
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
    assert v.healthy is False, "a single failed criterion must sink the cell"
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
    """No division, and no vacuous pass either: see
    test_zero_reference_throughput_fails_rather_than_passes_vacuously."""
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=0.0, jam=1.0, throughput=0.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=5.0, throughput=0.0))
    v = assess_cell(CELL, runs)
    assert v.worst_throughput_ratio is None
    assert v.throughput_holds is False


def test_summarise_counts_all_seeds_but_averages_only_completed_ones():
    runs = [
        make_run("no_av", 7, speed=10.0, completed=True),
        make_run("no_av", 17, speed=20.0, completed=False),
    ]
    s = summarise(runs)["no_av"]
    assert s.seeds == 2
    assert s.completed_seeds == 1
    assert s.mean_speed == pytest.approx(10.0), "the crashed seed must not enter the mean"


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
    """Ring is gridlock, not congestion: mean speed goes to zero with everything
    crashed. At the sweep's own settings it fails completion and, since ring
    throughput is structurally zero, the throughput criterion too."""
    cell, runs = cell_runs("ring", "medium", steps=120)
    verdict = assess_cell(cell, runs, min_completed_seeds=1)
    assert verdict.healthy is False
    assert "throughput_holds" in verdict.failed_criteria
    assert verdict.worst_throughput_ratio is None


def test_free_flow_topology_fails_congestion():
    """Asserted at the sweep's duration, not the 60-step shortcut: congestion
    needs ~90 steps to develop, so a short run would pass this for the wrong
    reason (see test_congestion_needs_time_to_develop)."""
    cell, runs = cell_runs("straight_multilane", "medium", steps=120)
    verdict = assess_cell(cell, runs, min_completed_seeds=1, min_congested_seeds=1)
    assert verdict.congestion_reachable is False, "multilane at medium should not congest on seed 7"
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


def test_crashed_runs_are_excluded_from_metric_means():
    """throughput_recent is a 60 s rolling window, so a run that dies early cannot
    report a comparable count. Pooling it with completed runs manufactures a
    throughput collapse out of nothing — this was a real misclassification of
    merge/high, where pooling gave ratio 0.721 and completed-only gives 1.000."""
    runs = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    runs.append(make_run("cooperative_smoothing", 7, completed=False, speed=16.0, throughput=3.0, steps=80))
    runs.append(make_run("cooperative_smoothing", 17, speed=12.5, throughput=21.0))
    runs.append(make_run("cooperative_smoothing", 27, speed=12.5, throughput=20.0))
    summary = summarise(runs)["cooperative_smoothing"]
    assert summary.seeds == 3
    assert summary.completed_seeds == 2
    assert summary.throughput == pytest.approx(20.5), "crashed seed must not drag the mean"
    assert summary.mean_speed == pytest.approx(12.5)
    v = assess_cell(CELL, runs, min_completed_seeds=2)
    assert v.throughput_holds is True
    assert v.worst_throughput_ratio == pytest.approx(20.5 / 20.0)


def test_controller_with_no_completed_runs_has_undefined_means():
    runs = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    runs += [make_run("cooperative_smoothing", s, completed=False, speed=16.0) for s in (7, 17, 27)]
    summary = summarise(runs)["cooperative_smoothing"]
    assert summary.completed_seeds == 0
    assert summary.mean_speed is None and summary.throughput is None
    v = assess_cell(CELL, runs)
    assert v.baselines_separate is False, "cannot separate against a controller with no valid runs"
    assert v.healthy is False


def test_zero_reference_throughput_fails_rather_than_passes_vacuously():
    """Ring throughput is structurally zero (_clear_exited_vehicles returns early
    there), so a pass-by-default would let total gridlock through the one
    criterion built to catch obstruction."""
    runs = []
    for seed in (7, 17, 27):
        runs.append(make_run("no_av", seed, speed=0.0, jam=1.0, throughput=0.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=3.0, throughput=0.0))
    v = assess_cell(CELL, runs)
    assert v.worst_throughput_ratio is None
    assert v.throughput_holds is False
    assert v.healthy is False


def test_congestion_is_counted_per_seed_not_on_the_mean():
    """A seed mean straddles the threshold: [0.096, 0.0, 0.051] means 0.049 and
    reads as no congestion, though 2 of 3 seeds individually clear it."""
    runs = []
    for seed, jam in zip((7, 17, 27), (0.0963, 0.0, 0.0505)):
        runs.append(make_run("no_av", seed, speed=12.0, jam=jam, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=16.0, throughput=19.0))
    v = assess_cell(CELL, runs)
    assert v.congested_seeds == 2
    assert v.congestion_reachable is True, "2 of 3 seeds congest; the mean is 0.049 and would say no"


def test_run_that_reaches_duration_while_crashing_counts_as_completed():
    """terminated (a crash) and truncated (duration reached) both fire when an AV
    crashes on the final step; `not terminated` would call that run crashed."""
    cell = CellSpec(topology="merge", demand="high", av_penetration=0.05)
    run = run_condition(cell, "backpressure", 7, duration_steps=120)
    assert run.steps == 120
    assert run.completed is True, "a run that reached its configured duration is not a crash"


def test_real_run_metrics_come_from_the_expected_env_fields():
    """Pins the extraction layer: reading speed_std or completed_vehicle_count
    instead would change every ratio in the report without any test objecting."""
    cell = CellSpec(topology="merge", demand="high", av_penetration=0.10)
    run = run_condition(cell, "no_av", 7, duration_steps=120)
    assert run.mean_speed == pytest.approx(12.40, abs=0.5)
    assert run.throughput == pytest.approx(20.0, abs=2.0)
    assert run.jam_fraction == pytest.approx(0.208, abs=0.02)
