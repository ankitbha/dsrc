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
        here_model, here_features = emc.load_arm_model("here", checkpoint_dir, num_segments)
        assert here_features == emc.HERE_FEATURES
        assert isinstance(here_model, SrcQNetwork)
        src_model, src_features = emc.load_arm_model("src", checkpoint_dir, num_segments)
        assert src_features == emc.SRC_FEATURES
        assert isinstance(src_model, SrcQNetwork)

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
        model, features = emc.load_arm_model("zeroed_head", checkpoint_dir, num_segments)
        assert features == emc.HERE_FEATURES
        assert torch.equal(model.stack[2].weight, torch.zeros_like(model.stack[2].weight))
        assert torch.equal(model.stack[2].bias, torch.zeros_like(model.stack[2].bias))

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

    def test_demand_is_the_schedules_own_span_not_the_configured_duration(self):
        # F7: the schedule's 3,124 departures run 2.4s to 2498.4s regardless of how
        # long an episode is configured to run, so the rate this reports must not
        # move when `duration_s` does -- dividing by `duration_s` instead reported
        # 8997.12 veh/h at duration_s=1250.0, double the true, duration-independent
        # rate of about 4506.
        demand_at_default = emc._demand_veh_per_h(emc.SCENARIOS["mainz"], 2500.0)
        demand_at_half = emc._demand_veh_per_h(emc.SCENARIOS["mainz"], 1250.0)
        assert round(demand_at_default) == 4506
        assert demand_at_half == pytest.approx(demand_at_default)


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

    def test_summary_carries_config_commit_and_both_tables(self):
        summary = emc._build_summary(
            commit="deadbeef", dirty=False, commit_unavailable_reason=None,
            config={"seeds": [16, 17, 18, 19, 20]}, per_seed=self._per_seed())
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
            config={}, per_seed=self._per_seed())
        assert summary["commit"] is None
        assert summary["dirty"] is None
        assert summary["commit_unavailable_reason"] == "not a git checkout"

    def test_arms_table_reports_flow_mean_and_bar(self):
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=self._per_seed())
        assert summary["arms"]["no_control"]["flow_mean"] == pytest.approx(3568.0, abs=0.01)
        assert summary["arms"]["here"]["flow_mean"] == pytest.approx(3774.666667, abs=0.01)
        assert summary["arms"]["here"]["flow_2se"] == pytest.approx(23.29, abs=0.01)

    def test_paired_table_matches_the_direct_computation(self):
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=self._per_seed())
        direct = emc.paired_statistics(
            [3760.00, 3755.56, 3793.33, 3811.11, 3753.33],
            [3722.22, 3553.33, 3564.44, 3555.56, 3444.44])
        assert summary["paired"]["here"]["gain"] == pytest.approx(direct["gain"])
        assert summary["paired"]["here"]["two_se"] == pytest.approx(direct["two_se"])
        assert summary["paired"]["here"]["n"] == 5

    def test_no_baseline_means_no_paired_table(self):
        per_seed = {"here": self._per_seed()["here"]}
        summary = emc._build_summary(commit="c", dirty=False, commit_unavailable_reason=None,
                                     config={}, per_seed=per_seed)
        assert summary["paired"] == {}


class TestRunOrchestrationWithoutSumo:
    """`run`'s own loop, JSONL writing and whole-then-rename summary, with
    `run_episode` and `evaluate_no_control` stubbed out so this test needs no SUMO
    binding and completes in milliseconds. What it checks is this module's own
    bookkeeping, not the simulator's.
    """

    #: Reversed once per test rather than at import time, so a stray `Ellipsis`
    #: added to `CHECKPOINT_FEATURES` later would fail loudly here instead of
    #: silently overwriting an entry.
    @staticmethod
    def _features_to_arm():
        return {features: name for name, features in emc.CHECKPOINT_FEATURES.items()}

    def test_a_two_seed_three_arm_run_writes_six_episodes_and_a_parseable_summary(
            self, tmp_path, monkeypatch):
        features_to_arm = self._features_to_arm()

        def fake_evaluate_no_control(seeds, features, duration_s, step_length,
                                     window_start_s, paths):
            (seed,) = seeds
            return {"return": 1.0, "arrived": 100.0, "flow": 3000.0 + seed,
                    "mean_speed_kmh": 40.0, "space_mean_speed_kmh": 40.0,
                    "mean_vehicles": 500.0, "held_at_entry": 0.0}

        def fake_run_episode(model, seed, features, duration_s, epsilon, generator,
                             step_length, window_start_s, gate_entries, paths):
            # F3: a mutation that ran every trained arm against the wrong
            # checkpoint (same features, wrong model) passed the whole suite
            # before this asserted the two are bound together.
            arm = features_to_arm[features]
            assert model == ("model", arm)
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
            # `run`'s own checkpoint-recording code (F3's sha256) reads this file
            # from disk independently of this stub, so it has to actually exist.
            checkpoint_path = checkpoint_dir / f"mainz_{name}_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"fake checkpoint bytes for {name}".encode())
            return ("model", name), emc.CHECKPOINT_FEATURES[name]

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

        # F3: the JSONL's own (arm, seed) pairs must be the same set the summary's
        # per_seed table claims -- a swapped arm label would desynchronise them.
        jsonl_pairs = {(row["arm"], row["seed"]) for row in episodes}
        summary_pairs = {(arm, seed) for arm, seeds in summary["per_seed"].items()
                         for seed in seeds}
        assert jsonl_pairs == summary_pairs

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

        # F3: a path cannot distinguish a retrained checkpoint at the same path.
        for arm in ("here", "src"):
            checkpoint_entry = summary["config"]["checkpoints"][arm]
            assert checkpoint_entry["path"].endswith(f"mainz_{arm}_best.pt")
            assert checkpoint_entry["sha256"] == hashlib.sha256(
                f"fake checkpoint bytes for {arm}".encode()).hexdigest()

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
