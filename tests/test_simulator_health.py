from __future__ import annotations

import pytest

from src.analysis.simulator_health import (
    DEFAULT_MIN_COMPLETED_SEEDS,
    DEFAULT_MIN_CONGESTED_SEEDS,
    DEFAULT_MIN_JAM_FRACTION,
    DEFAULT_MIN_SPEED_SEPARATION_MPS,
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
    # paired on seeds 17 and 27, where both arms completed
    assert v.worst_throughput_ratio == pytest.approx(20.5 / 20.0)
    assert v.comparisons["cooperative_smoothing"].shared_seeds == 2


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


# --------------------------------------------------------------------------
# paired (seed-matched) comparison
# --------------------------------------------------------------------------


def test_separation_is_paired_on_seeds_where_both_arms_completed():
    """The reference has zero AVs so it never crashes and always contributes
    every seed, while a treatment contributes only the seeds it survived — and
    those it crashes on are the hard ones where the reference is slowest.
    Comparing arm means then credits the controller for the absent seed.

    Real case: straight_single_lane/medium/pen0.05 scored 1.90 m/s arm-wise and
    0.14 m/s paired, which was the difference between the study reporting an
    operating point and reporting none.
    """
    runs = [
        make_run("no_av", 7, speed=0.17, jam=0.20, throughput=20.0),
        make_run("no_av", 17, speed=9.97, jam=0.20, throughput=20.0),
        make_run("no_av", 27, speed=0.96, jam=0.20, throughput=20.0),
        make_run("backpressure", 7, completed=False, speed=17.11, throughput=5.0, steps=65),
        make_run("backpressure", 17, speed=10.98, throughput=20.0),
        make_run("backpressure", 27, speed=0.23, throughput=20.0),
    ]
    v = assess_cell(CELL, runs, min_completed_seeds=2)
    comparison = v.comparisons["backpressure"]
    assert comparison.shared_seeds == 2, "seed 7 crashed for the treatment and must be excluded from both arms"
    assert comparison.speed_separation == pytest.approx(0.14, abs=0.01)
    assert v.baselines_separate is False, "0.14 m/s is far below the 1.0 threshold"


def test_zero_real_effect_is_not_reported_as_separation():
    """inverted_tree/medium/pen0.05: the treatment's one surviving seed was
    identical to the reference on that seed, i.e. no effect at all, and arm-wise
    means reported 7.38 m/s of separation and a 57% throughput improvement."""
    runs = [
        make_run("no_av", 7, speed=10.76, jam=0.20, throughput=12.0),
        make_run("no_av", 17, speed=21.70, jam=0.20, throughput=12.0),
        make_run("no_av", 27, speed=10.52, jam=0.20, throughput=12.0),
        make_run("cooperative_smoothing", 7, completed=False, speed=11.57, throughput=4.0, steps=98),
        make_run("cooperative_smoothing", 17, speed=21.70, throughput=12.0),
        make_run("cooperative_smoothing", 27, completed=False, speed=20.78, throughput=4.0, steps=62),
    ]
    v = assess_cell(CELL, runs, min_completed_seeds=1)
    comparison = v.comparisons["cooperative_smoothing"]
    assert comparison.shared_seeds == 1
    assert comparison.speed_separation == pytest.approx(0.0)
    assert comparison.throughput_ratio == pytest.approx(1.0)
    assert v.baselines_separate is False


def test_signed_delta_distinguishes_improvement_from_harm():
    """The criterion is absolute, so a controller that makes traffic worse can
    satisfy it. The signed delta is what makes that visible in the report."""
    runs = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    runs += [make_run("worse", s, speed=8.0, throughput=20.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs)
    assert v.baselines_separate is True, "abs() means harm satisfies the criterion"
    assert v.best_speed_delta == pytest.approx(-4.0), "the sign is what reveals it as harm"
    assert v.best_speed_separation == pytest.approx(4.0)


def test_treatment_with_no_shared_seeds_cannot_separate():
    runs = [make_run("no_av", 7, speed=12.0, jam=0.20, throughput=20.0)]
    runs += [make_run("cooperative_smoothing", 17, speed=20.0, throughput=20.0)]
    v = assess_cell(CELL, runs, min_completed_seeds=1)
    comparison = v.comparisons["cooperative_smoothing"]
    assert comparison.shared_seeds == 0
    assert comparison.speed_separation is None
    assert v.baselines_separate is False


def test_throughput_ratio_takes_the_worst_treatment():
    """Two treatments, so the min() aggregation is actually exercised."""
    runs = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    runs += [make_run("good", s, speed=16.0, throughput=19.0) for s in (7, 17, 27)]
    runs += [make_run("bad", s, speed=16.0, throughput=4.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs)
    assert v.worst_throughput_ratio == pytest.approx(0.20)
    assert v.throughput_holds is False


# --------------------------------------------------------------------------
# threshold boundaries and defaults
# --------------------------------------------------------------------------


def test_single_congesting_seed_fails_at_the_default():
    """merge/high congests in 1 of 3 seeds. Lowering the default to 1 would flip
    it, so the default itself is load-bearing."""
    runs = []
    for seed, jam in zip((7, 17, 27), (0.208, 0.0, 0.0)):
        runs.append(make_run("no_av", seed, speed=12.0, jam=jam, throughput=20.0))
        runs.append(make_run("cooperative_smoothing", seed, speed=16.0, throughput=19.0))
    v = assess_cell(CELL, runs)
    assert DEFAULT_MIN_CONGESTED_SEEDS == 2
    assert v.congested_seeds == 1
    assert v.congestion_reachable is False
    assert assess_cell(CELL, runs, min_congested_seeds=1).congestion_reachable is True


def test_thresholds_are_inclusive_at_the_boundary():
    """Every criterion is specified as >=, and round 1's cases missed by 0.0011,
    so > versus >= is not cosmetic."""
    exact_jam = [make_run("no_av", s, speed=12.0, jam=DEFAULT_MIN_JAM_FRACTION, throughput=20.0) for s in (7, 17, 27)]
    exact_jam += [make_run("t", s, speed=12.0, throughput=20.0) for s in (7, 17, 27)]
    assert assess_cell(CELL, exact_jam).congestion_reachable is True

    exact_sep = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    exact_sep += [make_run("t", s, speed=12.0 + DEFAULT_MIN_SPEED_SEPARATION_MPS, throughput=20.0) for s in (7, 17, 27)]
    assert assess_cell(CELL, exact_sep).baselines_separate is True

    exact_thr = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    exact_thr += [make_run("t", s, speed=16.0, throughput=16.0) for s in (7, 17, 27)]
    assert assess_cell(CELL, exact_thr).throughput_holds is True, "ratio is exactly 0.8"

    two_of_three = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    two_of_three += [make_run("t", s, completed=(s != 27), speed=16.0, throughput=19.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, two_of_three)
    assert DEFAULT_MIN_COMPLETED_SEEDS == 2
    assert v.min_completed_seeds == 2
    assert v.episodes_complete is True, "at least 2 of 3, not all 3"


def test_paired_comparison_covers_throughput_not_just_speed():
    """The round-2 bias appeared on both metrics: inverted_tree/medium/pen0.05
    reported ratio 1.57 arm-wise and 1.00 paired. Pinning only the speed half
    would leave the throughput half free to regress."""
    runs = [
        make_run("no_av", 7, speed=10.76, jam=0.20, throughput=4.0),
        make_run("no_av", 17, speed=21.70, jam=0.20, throughput=12.0),
        make_run("no_av", 27, speed=10.52, jam=0.20, throughput=4.0),
        make_run("t", 7, completed=False, speed=11.57, throughput=4.0, steps=98),
        make_run("t", 17, speed=21.70, throughput=12.0),
        make_run("t", 27, completed=False, speed=20.78, throughput=4.0, steps=62),
    ]
    v = assess_cell(CELL, runs, min_completed_seeds=1)
    comparison = v.comparisons["t"]
    assert comparison.shared_seeds == 1
    # paired on seed 17 only: 12/12. Over all reference completions it would be
    # 12 / mean(4,12,4) = 1.8
    assert comparison.throughput_ratio == pytest.approx(1.0)
    assert v.worst_throughput_ratio == pytest.approx(1.0)


def test_best_controller_is_the_largest_separation_not_the_smallest():
    runs = [make_run("no_av", s, speed=12.0, jam=0.20, throughput=20.0) for s in (7, 17, 27)]
    runs += [make_run("weak", s, speed=12.5, throughput=20.0) for s in (7, 17, 27)]
    runs += [make_run("strong", s, speed=18.0, throughput=20.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs)
    assert v.best_controller == "strong"
    assert v.best_speed_separation == pytest.approx(6.0)


def test_congested_shared_seeds_distinguishes_no_effect_from_died_under_load():
    """merge/high: the reference congests on seed 7 only, and a controller that
    crashes there is then measured on two free-flow seeds. Without this counter
    the report reads 'cannot affect congestion' when the truth is 'never faced
    it'."""
    runs = []
    for seed, jam in zip((7, 17, 27), (0.208, 0.0, 0.0)):
        runs.append(make_run("no_av", seed, speed=12.0, jam=jam, throughput=20.0))
    runs.append(make_run("dies_under_load", 7, completed=False, speed=16.0, throughput=3.0, steps=80))
    runs.append(make_run("dies_under_load", 17, speed=12.5, throughput=20.0))
    runs.append(make_run("dies_under_load", 27, speed=12.5, throughput=20.0))
    runs += [make_run("survives", s, speed=12.4, throughput=20.0) for s in (7, 17, 27)]
    v = assess_cell(CELL, runs, min_congested_seeds=1, min_completed_seeds=2)
    assert v.comparisons["dies_under_load"].shared_seeds == 2
    assert v.comparisons["dies_under_load"].congested_shared_seeds == 0, "never evaluated under load"
    assert v.comparisons["survives"].congested_shared_seeds == 1


def test_duplicate_seeds_cannot_satisfy_a_per_seed_threshold():
    """One seed run twice is one sample. jam_by_seed must collapse it, or D5's
    'two seeds congest' is satisfiable by a single run counted twice."""
    runs = [
        make_run("no_av", 17, speed=12.0, jam=0.333, throughput=20.0),
        make_run("no_av", 17, speed=12.0, jam=0.333, throughput=20.0),
        make_run("t", 17, speed=13.0, throughput=20.0),
        make_run("t", 17, speed=13.0, throughput=20.0),
    ]
    v = assess_cell(CELL, runs, min_completed_seeds=1)
    assert v.congested_seeds == 1, "a duplicated seed is one congesting seed, not two"
    assert v.congestion_reachable is False, "the default requires two distinct congesting seeds"
