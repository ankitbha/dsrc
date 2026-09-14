"""Tests for `scripts/evaluate_mainz_checkpoints.py`.

No SUMO simulation is started anywhere in this file -- nothing here calls
`MainzEnv.reset()`, so nothing opens `libsumo`. The statistics fixtures are chosen so
the paired formula and the naive independent-means formula disagree: a fixture where
they agree would pass under either implementation and would not be a test of which
one is used. Two are drawn from the five-seed table that the published bars were
measured against (`plans/mainz_src_port.md:930-934`); the third is synthetic.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_mainz_checkpoints as emc  # noqa: E402
from src.rl.src_q import SrcQNetwork, greedy_actions  # noqa: E402


class TestPairedStatisticsDiscriminatesFromIndependentMeans:
    """Each fixture is one the independent-means formula gets visibly wrong."""

    def test_perfectly_correlated_seeds_give_a_near_zero_paired_bar(self):
        # here vs no_control, seeds 16-20 (plans/mainz_src_port.md:930-934), rescaled
        # to be exactly correlated: the arm is the baseline plus a constant.
        baseline = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        arm = [v + 100.0 for v in baseline]

        result = emc.paired_statistics(arm, baseline)

        assert result["gain"] == pytest.approx(100.0)
        assert result["two_se"] == pytest.approx(0.0, abs=1e-9)
        assert result["resolved"] is True

        # The independent-means formula does not cancel the level the two arms
        # share: adding a constant to every seed leaves each side's own spread
        # unchanged, so it reports a bar of about 2000 on the same ten numbers
        # instead of the 0.0 the shared level actually earns.
        n = len(arm)
        independent_two_se = 2.0 * math.sqrt(
            (statistics.stdev(arm) / math.sqrt(n)) ** 2
            + (statistics.stdev(baseline) / math.sqrt(n)) ** 2)
        assert independent_two_se == pytest.approx(2000.0, abs=0.1)
        assert result["two_se"] < independent_two_se / 100

    def test_two_seeds_catches_pstdev_in_place_of_stdev(self):
        baseline = [0.0, 0.0]
        arm = [0.0, 100.0]

        result = emc.paired_statistics(arm, baseline)

        assert result["gain"] == pytest.approx(50.0)
        assert result["two_se"] == pytest.approx(100.0)

        # statistics.pstdev on the same two differences gives a materially smaller
        # bar: the population/sample choice is not a rounding difference here.
        differences = [0.0, 100.0]
        wrong_bar = 2.0 * statistics.pstdev(differences) / math.sqrt(2)
        assert wrong_bar == pytest.approx(70.7106781, abs=1e-3)
        assert result["two_se"] != pytest.approx(wrong_bar, abs=1.0)

    def test_arm_below_baseline_keeps_a_negative_gain(self):
        baseline = [3000.0, 3000.0]
        arm = [2900.0, 2900.0]

        result = emc.paired_statistics(arm, baseline)

        assert result["gain"] == pytest.approx(-100.0)
        assert result["two_se"] == pytest.approx(0.0, abs=1e-9)
        # A sign convention reversed (baseline - arm instead of arm - baseline) would
        # report +100 here and call the same numbers a gain.
        assert result["gain"] < 0
        assert result["resolved"] is True


class TestResolvedCanReportFailure:
    """F2: a mutation that hardcodes `"resolved": True` passed the whole suite before
    this class existed, because nothing here asserted `resolved is False`.
    """

    def test_a_gain_inside_its_own_bar_is_not_resolved(self):
        baseline = [3000.0] * 5
        arm = [3010.0, 2990.0, 3050.0, 2960.0, 3040.0]

        result = emc.paired_statistics(arm, baseline)

        assert result["gain"] == pytest.approx(10.0)
        assert result["two_se"] == pytest.approx(32.8634, abs=0.001)
        assert result["resolved"] is False

    def test_a_gain_exactly_at_the_bar_is_not_resolved(self):
        # |gain| == two_se exactly, pinning `>` against `>=`: a `>=` implementation
        # would mark this resolved, and the correct strict `>` must not.
        baseline = [0.0, 0.0]
        arm = [3.0, 1.0]

        result = emc.paired_statistics(arm, baseline)

        assert result["gain"] == pytest.approx(2.0)
        assert result["two_se"] == pytest.approx(2.0)
        assert result["resolved"] is False


class TestPairedStatisticsAgainstThePublishedFiveSeedTable:
    """`plans/mainz_src_port.md:930-934`'s per-seed table, re-measured for the plan
    and reproduced exactly by this repository's `no_control`, `here` and `src` reads
    at the pre-eb86e72 action mapping (plan section 1.5). The paired bars computed
    from it are what section 1.6 shows equal the published +/-92 and +/-133.
    """

    NO_CONTROL = [3722.22, 3553.33, 3564.44, 3555.56, 3444.44]
    HERE = [3760.00, 3755.56, 3793.33, 3811.11, 3753.33]
    SRC = [3848.89, 3706.67, 3564.44, 3826.67, 3835.56]

    def test_here_gain_and_bar(self):
        result = emc.paired_statistics(self.HERE, self.NO_CONTROL)
        assert result["gain"] == pytest.approx(206.668, abs=0.01)
        assert result["two_se"] == pytest.approx(91.5155, abs=0.01)
        assert result["resolved"] is True

    def test_src_gain_and_bar(self):
        result = emc.paired_statistics(self.SRC, self.NO_CONTROL)
        assert result["gain"] == pytest.approx(188.448, abs=0.01)
        assert result["two_se"] == pytest.approx(133.0193, abs=0.01)
        assert result["resolved"] is True

    def test_here_flow_2se_matches_the_document(self):
        # plans/mainz_src_port.md:932 gives 23 for the here/DSRC arm.
        assert emc.per_arm_two_se(self.HERE) == pytest.approx(23.29, abs=0.01)

    def test_src_flow_2se_matches_the_document(self):
        # plans/mainz_src_port.md:934 gives 109 for the src arm.
        assert emc.per_arm_two_se(self.SRC) == pytest.approx(108.70, abs=0.01)


class TestPairedStatisticsInputValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            emc.paired_statistics([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_a_single_seed_cannot_carry_a_sample_standard_deviation(self):
        with pytest.raises(ValueError):
            emc.paired_statistics([1.0], [1.0])
        with pytest.raises(ValueError):
            emc.per_arm_two_se([1.0])


class TestDuplicateAndSingleSeedRejection:
    """F5/F6: a duplicate silently shrinks the sample it claims to have, and a
    single seed completes both episodes before `paired_statistics` raises on it --
    both are checked before any episode runs, not discovered after.
    """

    def test_duplicate_seeds_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate seeds"):
            emc._validate_no_duplicates(arms=("here",), seeds=(16, 17, 17, 18))

    def test_duplicate_arms_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate arms"):
            emc._validate_no_duplicates(arms=("here", "here"), seeds=(16, 17))

    def test_unique_arms_and_seeds_pass(self):
        emc._validate_no_duplicates(arms=("here", "src"), seeds=(16, 17))  # no raise

    def test_a_single_seed_is_rejected(self):
        with pytest.raises(ValueError, match="at least two distinct seeds"):
            emc._validate_seed_count((16,))

    def test_two_distinct_seeds_pass(self):
        emc._validate_seed_count((16, 17))  # does not raise

    def test_run_rejects_duplicate_seeds_before_any_episode_executes(
            self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("an episode ran despite a duplicate seed")

        monkeypatch.setattr(emc, "evaluate_no_control", boom)
        monkeypatch.setattr(emc, "run_episode", boom)

        with pytest.raises(ValueError, match="duplicate seeds"):
            emc.run(arms=("no_control",), seeds=(16, 17, 17),
                    checkpoint_dir=tmp_path / "checkpoints",
                    out_dir=tmp_path / "evaluation", scenario="mainz")

    def test_run_rejects_a_single_seed_before_any_episode_executes(
            self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("an episode ran despite a single seed")

        monkeypatch.setattr(emc, "evaluate_no_control", boom)
        monkeypatch.setattr(emc, "run_episode", boom)

        with pytest.raises(ValueError, match="at least two distinct seeds"):
            emc.run(arms=("no_control",), seeds=(16,),
                    checkpoint_dir=tmp_path / "checkpoints",
                    out_dir=tmp_path / "evaluation", scenario="mainz")


class TestGateAsymmetryIsCheckedRatherThanAssumed:
    def test_a_correctly_gated_run_passes(self):
        per_seed = {
            "no_control": {16: {"held_at_entry": 0}, 17: {"held_at_entry": 0}},
            "here": {16: {"held_at_entry": 3}, 17: {"held_at_entry": 0}},
        }
        emc._check_gate_asymmetry(per_seed)  # does not raise

    def test_no_control_holding_anyone_at_entry_raises(self):
        per_seed = {
            "no_control": {16: {"held_at_entry": 1}},
            "here": {16: {"held_at_entry": 3}},
        }
        with pytest.raises(AssertionError):
            emc._check_gate_asymmetry(per_seed)

    def test_a_trained_arm_never_holding_anyone_raises(self):
        per_seed = {
            "no_control": {16: {"held_at_entry": 0}},
            "here": {16: {"held_at_entry": 0}, 17: {"held_at_entry": 0}},
        }
        with pytest.raises(AssertionError):
            emc._check_gate_asymmetry(per_seed)

    def test_no_control_alone_needs_no_trained_arm(self):
        per_seed = {"no_control": {16: {"held_at_entry": 0}}}
        emc._check_gate_asymmetry(per_seed)  # does not raise


class TestArmBinding:
    def test_gate_for_arm_is_on_only_for_trained_arms(self):
        assert emc.gate_for_arm("here") is True
        assert emc.gate_for_arm("src") is True
        assert emc.gate_for_arm("no_control") is False

    def test_a_crossed_checkpoint_and_feature_tuple_raises(self):
        checkpoint_dir = Path(__file__).resolve().parents[1] / "results" / "checkpoints"
        num_segments = emc._num_segments(emc.SCENARIOS["mainz"])
        with pytest.raises(ValueError, match="do not match"):
            emc._build_and_load(checkpoint_dir / "mainz_here_best.pt",
                                emc.SRC_FEATURES, num_segments)
        with pytest.raises(ValueError, match="do not match"):
            emc._build_and_load(checkpoint_dir / "mainz_src_best.pt",
                                emc.HERE_FEATURES, num_segments)

    def test_the_committed_checkpoints_load_under_their_own_features(self):
        checkpoint_dir = Path(__file__).resolve().parents[1] / "results" / "checkpoints"
        num_segments = emc._num_segments(emc.SCENARIOS["mainz"])
        here = emc.load_arm_model("here", checkpoint_dir, num_segments)
        assert here.name == "here"
        assert here.features == emc.HERE_FEATURES
        assert isinstance(here.model, SrcQNetwork)
        assert here.checkpoint_path == checkpoint_dir / "mainz_here_best.pt"
        assert here.sha256 == hashlib.sha256(here.checkpoint_path.read_bytes()).hexdigest()
        assert here.modification is None
        src = emc.load_arm_model("src", checkpoint_dir, num_segments)
        assert src.name == "src"
        assert src.features == emc.SRC_FEATURES
        assert isinstance(src.model, SrcQNetwork)
        assert src.modification is None

    def test_no_control_has_no_checkpoint(self):
        checkpoint_dir = Path(__file__).resolve().parents[1] / "results" / "checkpoints"
        with pytest.raises(ValueError):
            emc.load_arm_model("no_control", checkpoint_dir, 12)


class TestZeroedHeadIsARunnableArm:
    """F9: `_zero_output_head` had no caller, so `--arms` could not reach plan
    section 8's falsification arm at all. `zeroed_head` must be reachable the same
    way every other arm is, bound to the checkpoint file it falsifies.
    """

    def test_arms_choices_include_zeroed_head(self):
        args = emc.build_arg_parser().parse_args(["--arms", "zeroed_head"])
        assert args.arms == ["zeroed_head"]

    def test_zeroed_head_shares_the_here_checkpoint_file(self):
        assert emc._checkpoint_name_for_arm("zeroed_head") == "here"
        assert emc._checkpoint_name_for_arm("here") == "here"
        assert emc._checkpoint_name_for_arm("src") == "src"

    def test_zeroed_head_loads_the_here_checkpoint_with_its_output_layer_zeroed(self):
        checkpoint_dir = Path(__file__).resolve().parents[1] / "results" / "checkpoints"
        num_segments = emc._num_segments(emc.SCENARIOS["mainz"])
        loaded = emc.load_arm_model("zeroed_head", checkpoint_dir, num_segments)
        assert loaded.name == "zeroed_head"
        assert loaded.features == emc.HERE_FEATURES
        assert torch.equal(loaded.model.stack[2].weight,
                           torch.zeros_like(loaded.model.stack[2].weight))
        assert torch.equal(loaded.model.stack[2].bias,
                           torch.zeros_like(loaded.model.stack[2].bias))
        # R2-5: the checkpoint path and its sha256 alone say nothing about the
        # weights having been changed after loading -- both still name
        # `mainz_here_best.pt` and that file's own true digest, same as the
        # unmodified `here` arm. `modification` is the only place this is recorded.
        here = emc.load_arm_model("here", checkpoint_dir, num_segments)
        assert loaded.checkpoint_path == here.checkpoint_path
        assert loaded.sha256 == here.sha256
        assert loaded.modification is not None
        assert "zeroed" in loaded.modification
        assert here.modification is None

    def test_zeroed_head_gate_is_on_like_any_trained_arm(self):
        assert emc.gate_for_arm("zeroed_head") is True


class TestZeroedOutputHeadIsAStandingOrderForIndexZero:
    """The falsification arm (plan section 8): every Q-value zero, every argmax 0."""

    def test_every_segment_picks_action_zero_on_a_random_state(self):
        model = SrcQNetwork(num_segments=3, num_features=5, num_actions=3)
        emc._zero_output_head(model)
        state = torch.randn(3, 5)
        actions = greedy_actions(model, state)
        assert torch.equal(actions, torch.zeros(3, dtype=actions.dtype))

    def test_it_is_the_same_regardless_of_the_input(self):
        model = SrcQNetwork(num_segments=2, num_features=6, num_actions=3)
        emc._zero_output_head(model)
        first = greedy_actions(model, torch.randn(2, 6))
        second = greedy_actions(model, torch.randn(2, 6) * 100.0)
        assert torch.equal(first, second)
        assert torch.equal(first, torch.zeros(2, dtype=first.dtype))


class TestNumSegmentsAndDemand:
    def test_mainz_has_twelve_super_segments(self):
        assert emc._num_segments(emc.SCENARIOS["mainz"]) == 12

    def test_demand_veh_per_h_is_exactly_4500_and_does_not_move_with_duration(self):
        # R2-6: the schedule's eight entry links each depart at one constant headway
        # -- 140/182 every 4.8s, 277/289 every 5.76s, 293/295 every 7.2s, 231/255
        # every 9.6s -- and `sum(3600 / headway_e)` over all eight is exactly
        # 4500.0 veh/h, recovered losslessly from the file rather than backed out of
        # an aggregate. `_demand_veh_per_h` no longer takes `duration_s` at all, so
        # there is nothing left for a second call to disagree with: it is called
        # twice here (rather than checked for a fixed default) precisely to pin that
        # it cannot be made to move by varying the one argument the old, buggy
        # version divided by.
        assert emc._demand_veh_per_h(emc.SCENARIOS["mainz"]) == pytest.approx(4500.0)
        assert emc._demand_veh_per_h(emc.SCENARIOS["mainz"]) == pytest.approx(4500.0)

    def test_demand_unavailable_reason_is_none_for_the_committed_schedule(self):
        assert emc._demand_unavailable_reason(emc.SCENARIOS["mainz"]) is None

    def test_scheduled_departures_is_the_files_own_fixed_total(self):
        assert emc._scheduled_departures(emc.SCENARIOS["mainz"]) == 3124

    def test_departures_within_episode_halves_when_duration_halves(self):
        # F7's original defect, restated as what actually does vary with
        # `duration_s`: the schedule's departure COUNT, not its rate.
        at_full_duration = emc._departures_within_episode(emc.SCENARIOS["mainz"], 2500.0)
        at_half_duration = emc._departures_within_episode(emc.SCENARIOS["mainz"], 1250.0)
        assert at_full_duration == 3124
        assert at_half_duration == 1562
        assert at_half_duration == pytest.approx(at_full_duration / 2)

    def test_an_unevenly_spaced_entry_reports_none_with_a_named_reason(
            self, tmp_path, monkeypatch):
        # `--demand-rush-s` (build_mainz_scenario.py) ramps an entry link through
        # several fixed rates over a run instead of departing it at one constant
        # rate throughout -- piecewise-constant, not constant. A schedule built that
        # way must not have its rate silently averaged across regimes that were
        # never meant to be pooled.
        monkeypatch.setattr(emc, "REPO_ROOT", tmp_path)
        (tmp_path / "schedule.json").write_text(json.dumps([
            {"id": "a0", "route": "r", "type": "av", "depart": 0.0, "entry": "e1"},
            {"id": "a1", "route": "r", "type": "av", "depart": 5.0, "entry": "e1"},
            {"id": "a2", "route": "r", "type": "av", "depart": 8.0, "entry": "e1"},
        ]))
        paths = {"schedule": "schedule.json"}

        assert emc._demand_veh_per_h(paths) is None
        reason = emc._demand_unavailable_reason(paths)
        assert reason is not None and "e1" in reason


class TestEpisodeRecord:
    def test_only_the_declared_fields_are_carried_into_the_record(self):
        metrics = {
            "flow": 3568.0, "arrived": 1950.8, "return": 55.1,
            "mean_speed_kmh": 45.4, "space_mean_speed_kmh": 42.7,
            "mean_vehicles": 858.0, "held_at_entry": 0.0,
            # Extra keys `run_episode` returns (states/actions/rewards tensors) must
            # not leak into a JSON-serialisable record.
            "states": object(), "actions": object(), "rewards": object(),
        }
        record = emc._episode_record("no_control", 16, metrics)
        assert record == {
            "arm": "no_control", "seed": 16, "flow": 3568.0, "arrived": 1950.8,
            "return": 55.1, "mean_speed_kmh": 45.4, "space_mean_speed_kmh": 42.7,
            "mean_vehicles": 858.0, "held_at_entry": 0.0,
        }


class TestBuildSummary:
    """Sanity checks on the artifact assembly, with synthetic per-seed data standing
    in for a real run so no SUMO episode has to execute to test it.
    """

    def _per_seed(self):
        no_control = {16: 3722.22, 17: 3553.33, 18: 3564.44, 19: 3555.56, 20: 3444.44}
        here = {16: 3760.00, 17: 3755.56, 18: 3793.33, 19: 3811.11, 20: 3753.33}

        def rows(flows, held_at_entry):
            return {
                seed: {
                    "flow": flow, "arrived": 1900.0, "return": 100.0,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 800.0, "held_at_entry": held_at_entry,
                }
                for seed, flow in flows.items()
            }

        return {
            "no_control": rows(no_control, 0.0),
            "here": rows(here, 5.0),
        }

    #: R2-2/R2-4: fixed provenance-end and run_id arguments for the tests below,
    #: which are exercising the arm/paired tables and are not themselves about
    #: provenance or run identity -- `TestProvenanceEndAndRunId` covers those.
    def _prov_kwargs(self):
        return {
            "run_id": "test-run-id",
            "provenance_end": {"commit": "deadbeef", "dirty": False,
                               "commit_unavailable_reason": None},
            "tree_unchanged_during_run": True,
        }

    def test_summary_carries_config_commit_and_both_tables(self):
        summary = emc._build_summary(
            commit="deadbeef", dirty=False, commit_unavailable_reason=None,
            config={"seeds": [16, 17, 18, 19, 20]}, per_seed=self._per_seed(),
            **self._prov_kwargs())
        assert summary["commit"] == "deadbeef"
        assert summary["dirty"] is False
        assert summary["commit_unavailable_reason"] is None
        assert summary["config"] == {"seeds": [16, 17, 18, 19, 20]}
        assert set(summary["per_seed"]) == {"no_control", "here"}
        assert set(summary["arms"]) == {"no_control", "here"}
        assert set(summary["paired"]) == {"here"}

    def test_summary_carries_an_unavailable_commit_as_a_named_reason(self):
        summary = emc._build_summary(
            commit=None, dirty=None, commit_unavailable_reason="not a git checkout",
            config={}, per_seed=self._per_seed(), **self._prov_kwargs())
        assert summary["commit"] is None
        assert summary["dirty"] is None
        assert summary["commit_unavailable_reason"] == "not a git checkout"

    def test_arms_table_reports_flow_mean_and_bar(self):
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=self._per_seed(), **self._prov_kwargs())
        assert summary["arms"]["no_control"]["flow_mean"] == pytest.approx(3568.0, abs=0.01)
        assert summary["arms"]["here"]["flow_mean"] == pytest.approx(3774.666667, abs=0.01)
        assert summary["arms"]["here"]["flow_2se"] == pytest.approx(23.29, abs=0.01)

    def test_paired_table_matches_the_direct_computation(self):
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=self._per_seed(), **self._prov_kwargs())
        direct = emc.paired_statistics(
            [3760.00, 3755.56, 3793.33, 3811.11, 3753.33],
            [3722.22, 3553.33, 3564.44, 3555.56, 3444.44])
        assert summary["paired"]["here"]["gain"] == pytest.approx(direct["gain"])
        assert summary["paired"]["here"]["two_se"] == pytest.approx(direct["two_se"])
        assert summary["paired"]["here"]["n"] == 5

    def test_no_baseline_means_no_paired_table(self):
        per_seed = {"here": self._per_seed()["here"]}
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=per_seed, **self._prov_kwargs())
        assert summary["paired"] == {}

    def test_summary_carries_run_id_and_provenance_end_verbatim(self):
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=self._per_seed(), **self._prov_kwargs())
        assert summary["run_id"] == "test-run-id"
        assert summary["provenance_end"] == {"commit": "deadbeef", "dirty": False,
                                             "commit_unavailable_reason": None}
        assert summary["tree_unchanged_during_run"] is True


class TestRunOrchestrationWithoutSumo:
    """`run`'s own loop, JSONL writing and whole-then-rename summary, with
    `run_episode` and `evaluate_no_control` stubbed out so this test needs no SUMO
    binding and completes in milliseconds. What it checks is this module's own
    bookkeeping, not the simulator's.
    """

    def test_a_two_seed_three_arm_run_writes_six_episodes_and_a_parseable_summary(
            self, tmp_path, monkeypatch):
        episode_calls: list[tuple[tuple[str, str], int]] = []

        def fake_evaluate_no_control(seeds, features, duration_s, step_length,
                                     window_start_s, paths):
            (seed,) = seeds
            return {"return": 1.0, "arrived": 100.0, "flow": 3000.0 + seed,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 500.0, "held_at_entry": 0.0}

        def fake_run_episode(model, seed, features, duration_s, epsilon, generator,
                             step_length, window_start_s, gate_entries, paths):
            # R2-1: recorded here, not inferred from `features` -- `model` is the
            # sentinel `("model", name)` `fake_load_arm_model` below hands back for
            # exactly one arm, so this pins which model object `run()` actually
            # passed to `run_episode` for this call, and the assertion after `run()`
            # returns compares that directly against what the JSONL claims for the
            # same (arm, seed). Checking `model` against `features` instead (the
            # round-1 test) could not catch a mutation that swapped both together.
            episode_calls.append((model, seed))
            arm = model[1]
            # F8: a mutation that passed a literal `True` for every trained arm
            # regardless of `gate_for_arm` also passed before this compared
            # against the derived value instead of a hardcoded one.
            assert gate_entries == emc.gate_for_arm(arm)
            # The real `run_episode` also returns `states`/`actions`/`rewards`
            # tensors (the trajectory it trains `td_loss` on). Carrying one of
            # those into `per_seed` broke JSON serialisation the first time this
            # was run against real checkpoints; a fixture without them would not
            # have caught it, so they are included here on purpose.
            return {"return": 2.0, "arrived": 120.0, "flow": 3100.0 + seed,
                    "mean_speed_kmh": 41.0, "space_mean_speed_kmh": 41.0,
                    "mean_vehicles": 520.0, "held_at_entry": 4.0,
                    "states": torch.zeros(2, 2), "actions": torch.zeros(2),
                    "rewards": torch.zeros(2, 2)}

        def fake_load_arm_model(name, checkpoint_dir, num_segments):
            # `run`'s own checkpoint-recording code (the sha256) reads this file
            # from disk independently of this stub, so it has to actually exist.
            checkpoint_path = checkpoint_dir / f"mainz_{name}_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_bytes = f"fake checkpoint bytes for {name}".encode()
            checkpoint_path.write_bytes(checkpoint_bytes)
            return emc.LoadedArm(
                name=name, model=("model", name), features=emc.CHECKPOINT_FEATURES[name],
                checkpoint_path=checkpoint_path,
                sha256=hashlib.sha256(checkpoint_bytes).hexdigest(), modification=None)

        monkeypatch.setattr(emc, "evaluate_no_control", fake_evaluate_no_control)
        monkeypatch.setattr(emc, "run_episode", fake_run_episode)
        monkeypatch.setattr(emc, "load_arm_model", fake_load_arm_model)

        out_dir = tmp_path / "evaluation"
        summary = emc.run(arms=("no_control", "here", "src"), seeds=(16, 17),
                          checkpoint_dir=tmp_path / "checkpoints", out_dir=out_dir,
                          scenario="mainz")

        jsonl_lines = (out_dir / "mainz_paired_seeds.jsonl").read_text().splitlines()
        # F4: one header line first, then one line per (arm, seed) episode.
        assert len(jsonl_lines) == 1 + 6
        parsed_lines = [json.loads(line) for line in jsonl_lines]  # each line parses alone
        header, episodes = parsed_lines[0], parsed_lines[1:]
        assert header["type"] == "header"
        assert header["config"] == summary["config"]
        assert header["commit"] == summary["commit"]
        assert header["dirty"] == summary["dirty"]
        assert header["commit_unavailable_reason"] == summary["commit_unavailable_reason"]
        # R2-4: the header and the summary carry the same run_id.
        assert header["run_id"] == summary["run_id"]
        assert isinstance(summary["run_id"], str) and summary["run_id"]

        # The JSONL's own (arm, seed) pairs must be the same set the summary's
        # per_seed table claims -- a swapped arm label would desynchronise them.
        jsonl_pairs = {(row["arm"], row["seed"]) for row in episodes}
        summary_pairs = {(arm, seed) for arm, seeds in summary["per_seed"].items()
                         for seed in seeds}
        assert jsonl_pairs == summary_pairs

        # R2-1: the model object each `run_episode` call actually received is bound
        # to its own arm's name (via the sentinel `fake_load_arm_model` returns), so
        # this is checked against the (arm, seed) pairs the JSONL claims for the
        # trained arms -- a mutation that ran every trained arm against one arm's
        # checkpoint would still write correctly-labelled JSONL rows, but every
        # `episode_calls` entry for the OTHER arm would carry the wrong sentinel.
        sentinel_pairs = {(model[1], seed) for model, seed in episode_calls}
        trained_jsonl_pairs = {(row["arm"], row["seed"]) for row in episodes
                               if row["arm"] != "no_control"}
        assert sentinel_pairs == trained_jsonl_pairs

        assert not (out_dir / "mainz_paired_seeds.jsonl.partial").exists()
        assert not (out_dir / "mainz_paired_seeds.json.tmp").exists()
        written = json.loads((out_dir / "mainz_paired_seeds.json").read_text())
        # JSON round-trips a seed key as a string; everything else compares directly.
        assert written["commit"] == summary["commit"]
        assert written["dirty"] == summary["dirty"]
        assert written["commit_unavailable_reason"] == summary["commit_unavailable_reason"]
        assert written["config"] == summary["config"]
        assert written["arms"] == summary["arms"]
        assert written["paired"] == summary["paired"]
        assert set(written["per_seed"]) == set(summary["per_seed"])
        for arm in summary["per_seed"]:
            restored = {int(seed): metrics
                       for seed, metrics in written["per_seed"][arm].items()}
            assert restored == summary["per_seed"][arm]
        assert summary["config"]["gate_entries"] == {
            "no_control": False, "here": True, "src": True}
        assert set(summary["paired"]) == {"here", "src"}

        # A path cannot distinguish a retrained checkpoint at the same path -- the
        # sha256 must be the fake bytes' own digest, and the unmodified arms carry
        # no `modification` (R2-5's field, exercised for `zeroed_head` elsewhere).
        for arm in ("here", "src"):
            checkpoint_entry = summary["config"]["checkpoints"][arm]
            assert checkpoint_entry["path"].endswith(f"mainz_{arm}_best.pt")
            assert checkpoint_entry["sha256"] == hashlib.sha256(
                f"fake checkpoint bytes for {arm}".encode()).hexdigest()
            assert checkpoint_entry["modification"] is None

    def test_an_inverted_gate_is_caught_before_the_summary_is_written(
            self, tmp_path, monkeypatch):
        def fake_evaluate_no_control(seeds, features, duration_s, step_length,
                                     window_start_s, paths):
            (seed,) = seeds
            # Deliberately wrong: no_control holds vehicles at entry.
            return {"return": 1.0, "arrived": 100.0, "flow": 3000.0,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 500.0, "held_at_entry": 9.0}

        def fake_load_arm_model(name, checkpoint_dir, num_segments):
            # Never actually called: `arms=("no_control",)` below never asks for a
            # trained arm's model, so this stub's shape does not need to match
            # `LoadedArm` -- it exists only so a bug that DID call this would fail
            # loudly on the mismatch rather than on some unrelated attribute error.
            return object(), emc.CHECKPOINT_FEATURES.get(name, ())

        monkeypatch.setattr(emc, "evaluate_no_control", fake_evaluate_no_control)
        monkeypatch.setattr(emc, "load_arm_model", fake_load_arm_model)

        out_dir = tmp_path / "evaluation"
        with pytest.raises(AssertionError):
            emc.run(arms=("no_control",), seeds=(16, 17),
                    checkpoint_dir=tmp_path / "checkpoints", out_dir=out_dir,
                    scenario="mainz")
        # F4: the JSONL is still on disk under its provisional `.partial` name
        # (append-as-you-go, not transactional), but it is never renamed to its
        # final name, and the summary is never written, because the gate check
        # raises before either happens -- a failed run must not look like a
        # complete one.
        assert (out_dir / "mainz_paired_seeds.jsonl.partial").exists()
        assert not (out_dir / "mainz_paired_seeds.jsonl").exists()
        assert not (out_dir / "mainz_paired_seeds.json").exists()


class TestProvenance:
    """R2-3: `_provenance()` exercised directly against a real repo at `tmp_path`,
    rather than only ever through whatever tree happens to be running the test suite
    -- mutating it to hardcode `"dirty": None` left every other test in this file
    passing.
    """

    def _init_repo(self, repo_root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo_root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
        (repo_root / "tracked.txt").write_text("hello\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_root, check=True)

    def test_a_clean_repo_reports_a_commit_and_dirty_false(self, tmp_path, monkeypatch):
        self._init_repo(tmp_path)
        monkeypatch.setattr(emc, "REPO_ROOT", tmp_path)

        result = emc._provenance()

        assert result["commit"] is not None and len(result["commit"]) == 40
        assert result["dirty"] is False
        assert result["commit_unavailable_reason"] is None

    def test_a_touched_tracked_file_reports_dirty_true(self, tmp_path, monkeypatch):
        self._init_repo(tmp_path)
        (tmp_path / "tracked.txt").write_text("changed\n")
        monkeypatch.setattr(emc, "REPO_ROOT", tmp_path)

        result = emc._provenance()

        assert result["commit"] is not None
        assert result["dirty"] is True
        assert result["commit_unavailable_reason"] is None

    def test_a_non_repo_reports_no_commit_and_a_named_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(emc, "REPO_ROOT", tmp_path)  # empty dir, no `git init`

        result = emc._provenance()

        assert result["commit"] is None
        assert result["dirty"] is None
        assert isinstance(result["commit_unavailable_reason"], str)
        assert result["commit_unavailable_reason"]


class TestProvenanceEndAndTreeUnchanged:
    """R2-2: `run()` samples `_provenance()` again after the episode loop and
    compares it against the sample taken before -- and never raises on a
    disagreement, because raising here would throw away a run that already
    completed over a provenance question, the same thing F1's own fix stopped doing
    for an unreadable commit.
    """

    def _stub_no_control(self, monkeypatch):
        def fake_evaluate_no_control(seeds, features, duration_s, step_length,
                                     window_start_s, paths):
            (seed,) = seeds
            return {"return": 1.0, "arrived": 100.0, "flow": 3000.0 + seed,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 500.0, "held_at_entry": 0.0}
        monkeypatch.setattr(emc, "evaluate_no_control", fake_evaluate_no_control)

    def test_an_unchanged_tree_reports_true(self, tmp_path, monkeypatch):
        self._stub_no_control(monkeypatch)
        fixed = {"commit": "abc123", "dirty": False, "commit_unavailable_reason": None}
        monkeypatch.setattr(emc, "_provenance", lambda: dict(fixed))

        summary = emc.run(arms=("no_control",), seeds=(16, 17),
                          checkpoint_dir=tmp_path / "checkpoints",
                          out_dir=tmp_path / "evaluation", scenario="mainz")

        assert summary["tree_unchanged_during_run"] is True
        assert summary["provenance_end"] == fixed
        assert summary["commit"] == "abc123"

    def test_a_commit_mid_run_reports_false_and_does_not_raise(self, tmp_path, monkeypatch):
        # Reproduces the scenario a mid-run commit actually creates: the header (and
        # `summary["commit"]`) name the commit at the start, `provenance_end` names
        # the commit HEAD had moved to by the time the last episode finished, and
        # `run()` returns normally rather than raising over the disagreement.
        self._stub_no_control(monkeypatch)
        samples = iter([
            {"commit": "start111", "dirty": False, "commit_unavailable_reason": None},
            {"commit": "end222", "dirty": False, "commit_unavailable_reason": None},
        ])
        monkeypatch.setattr(emc, "_provenance", lambda: next(samples))

        summary = emc.run(arms=("no_control",), seeds=(16, 17),
                          checkpoint_dir=tmp_path / "checkpoints",
                          out_dir=tmp_path / "evaluation", scenario="mainz")

        assert summary["commit"] == "start111"
        assert summary["provenance_end"]["commit"] == "end222"
        assert summary["tree_unchanged_during_run"] is False

    def test_a_passed_in_provenance_start_is_used_verbatim_and_not_re_fetched(
            self, tmp_path, monkeypatch):
        # `main()` reads `_provenance()` once for its own banner and passes that
        # sample into `run()`; `run()` must use it rather than calling
        # `_provenance()` again for the start value -- a second, independent call
        # moments later is not guaranteed to agree with the first.
        calls = {"n": 0}

        def at_most_one_call():
            calls["n"] += 1
            if calls["n"] > 1:
                raise AssertionError(
                    "run() called _provenance() a second time instead of using the "
                    "provenance_start it was given")
            return {"commit": "endsample", "dirty": False, "commit_unavailable_reason": None}

        self._stub_no_control(monkeypatch)
        monkeypatch.setattr(emc, "_provenance", at_most_one_call)
        provenance_start = {"commit": "passedin", "dirty": True,
                           "commit_unavailable_reason": None}

        summary = emc.run(arms=("no_control",), seeds=(16, 17),
                          checkpoint_dir=tmp_path / "checkpoints",
                          out_dir=tmp_path / "evaluation", scenario="mainz",
                          provenance_start=provenance_start)

        assert summary["commit"] == "passedin"
        assert summary["dirty"] is True
        # The one call `_provenance()` does receive is the after-the-loop sample.
        assert summary["provenance_end"]["commit"] == "endsample"
        assert calls["n"] == 1


class TestRunIdDetectsAPartialRename:
    """R2-4: neither rename (`.tmp` -> summary, `.partial` -> jsonl) can be made
    atomic with the other, so a failure between them is a real possibility, not a
    hypothetical -- the summary is renamed into place FIRST for exactly this reason:
    if that rename fails, the jsonl rename below it must never run, so a previous
    run's `.jsonl` is left completely untouched rather than paired with a summary
    that does not describe it.
    """

    def _stub_no_control(self, monkeypatch):
        def fake_evaluate_no_control(seeds, features, duration_s, step_length,
                                     window_start_s, paths):
            (seed,) = seeds
            return {"return": 1.0, "arrived": 100.0, "flow": 3000.0 + seed,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 500.0, "held_at_entry": 0.0}
        monkeypatch.setattr(emc, "evaluate_no_control", fake_evaluate_no_control)

    def test_a_failed_summary_rename_leaves_the_previous_runs_jsonl_intact(
            self, tmp_path, monkeypatch):
        self._stub_no_control(monkeypatch)
        out_dir = tmp_path / "evaluation"

        first_summary = emc.run(arms=("no_control",), seeds=(16, 17),
                                checkpoint_dir=tmp_path / "checkpoints",
                                out_dir=out_dir, scenario="mainz")
        original_jsonl_text = (out_dir / "mainz_paired_seeds.jsonl").read_text()
        original_summary_text = (out_dir / "mainz_paired_seeds.json").read_text()

        real_rename = Path.rename

        def failing_rename(self, target):
            if self.name == "mainz_paired_seeds.json.tmp":
                raise OSError("simulated failure renaming the summary into place")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", failing_rename)

        with pytest.raises(OSError):
            emc.run(arms=("no_control",), seeds=(18, 19),
                    checkpoint_dir=tmp_path / "checkpoints",
                    out_dir=out_dir, scenario="mainz")

        # The jsonl rename never ran: the previous run's file, and its summary, are
        # untouched byte for byte, and the second run's own episodes never
        # displaced either of them.
        assert (out_dir / "mainz_paired_seeds.jsonl").read_text() == original_jsonl_text
        assert (out_dir / "mainz_paired_seeds.json").read_text() == original_summary_text
        assert json.loads(original_summary_text)["run_id"] == first_summary["run_id"]
        # The second run's own jsonl and tmp-summary are left behind under their
        # provisional names -- the rename that would have replaced the FIRST run's
        # files never ran, so nothing of the second run's ever displaced them.
        assert (out_dir / "mainz_paired_seeds.jsonl.partial").exists()
        assert (out_dir / "mainz_paired_seeds.json.tmp").exists()

    def test_mismatched_run_ids_on_disk_are_raised_rather_than_returned(
            self, tmp_path, monkeypatch):
        # A concurrent writer racing between the two renames is the residual risk
        # this module explicitly does not close (no per-run directory) -- but it
        # must not go unnoticed. Simulated here by rewriting the final jsonl's own
        # run_id immediately after `run()`'s own renames, from inside the rename
        # patch, so the post-rename disk check sees a real disagreement.
        self._stub_no_control(monkeypatch)
        out_dir = tmp_path / "evaluation"
        real_rename = Path.rename

        def tampering_rename(self, target):
            result = real_rename(self, target)
            if self.name == "mainz_paired_seeds.jsonl.partial":
                final = out_dir / "mainz_paired_seeds.jsonl"
                lines = final.read_text().splitlines()
                header = json.loads(lines[0])
                header["run_id"] = "a-different-run-entirely"
                final.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
            return result

        monkeypatch.setattr(Path, "rename", tampering_rename)

        with pytest.raises(RuntimeError, match="run_id"):
            emc.run(arms=("no_control",), seeds=(16, 17),
                    checkpoint_dir=tmp_path / "checkpoints",
                    out_dir=out_dir, scenario="mainz")
