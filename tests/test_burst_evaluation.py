"""The arithmetic that decides whether the pre-registered run found anything.

`paired_verdict` is what turns five paired numbers into "real" or "no effect", so
if it is wrong the conclusion is wrong whatever the simulation did. It is tested
here against cases with known answers, including the one that matters most: a
difference that looks large but is not distinguishable from noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_burst_scenario import paired_verdict  # noqa: E402


def _rows(values, key="arrivals"):
    return [{"seed": seed, key: value} for seed, value in enumerate(values)]


class TestThePreRegisteredBar:

    def test_a_consistent_difference_counts(self):
        # Every seed improves by about 20, with little spread.
        reference = _rows([100, 102, 98, 101, 99])
        arm = _rows([120, 122, 118, 121, 119])
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(20.0)
        assert se == pytest.approx(0.0, abs=1e-9)
        assert verdict == "real"

    def test_a_large_but_inconsistent_difference_does_not(self):
        # The same mean difference, but the sign flips across seeds. This is the
        # case the bar exists for: a headline of "+20 arrivals" that is noise.
        reference = _rows([100, 100, 100, 100, 100])
        arm = _rows([180, 40, 190, 30, 160])
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(20.0)
        assert se > 10.0
        assert verdict == "no effect"

    def test_a_loss_is_also_reported_as_real(self):
        # The bar is on the magnitude, so a consistent regression must be caught
        # rather than reported as no effect.
        reference = _rows([100, 102, 98, 101, 99])
        arm = _rows([80, 82, 78, 81, 79])
        mean, _, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(-20.0)
        assert verdict == "real"

    def test_pairing_is_by_seed_and_not_by_order(self):
        # The rows may arrive in any order. Pairing cannot change the MEAN -- the
        # two sums are the same however they are matched -- so the mean proves
        # nothing here and the spread is what separates the two. Matched by seed
        # the differences are +10 each; matched by order they are +110 and -90,
        # which turns a unanimous result into noise.
        reference = [{"seed": 1, "arrivals": 100}, {"seed": 2, "arrivals": 200}]
        arm = [{"seed": 2, "arrivals": 210}, {"seed": 1, "arrivals": 110}]
        mean, standard_error, verdict = paired_verdict(arm, reference, "arrivals")
        assert mean == pytest.approx(10.0)
        assert standard_error == pytest.approx(0.0, abs=1e-9), (
            f"the paired differences have a spread of {standard_error:.1f}, which "
            "means the rows were matched by position rather than by seed"
        )
        assert verdict == "real"

    def test_an_unpaired_seed_is_dropped_rather_than_compared(self):
        reference = [{"seed": 1, "arrivals": 100}, {"seed": 2, "arrivals": 100}]
        arm = [{"seed": 1, "arrivals": 110}, {"seed": 99, "arrivals": 500}]
        mean, se, verdict = paired_verdict(arm, reference, "arrivals")
        # One pair only, which is not enough to estimate a spread.
        assert verdict == "too few pairs"
        assert mean is None and se is None

    def test_a_missing_measurement_does_not_count_as_zero(self):
        # `recovery_s` is None when the queue never cleared, and treating that as
        # zero would report the fastest possible recovery for a run that never
        # recovered at all.
        reference = [{"seed": s, "recovery_s": 100.0} for s in range(5)]
        arm = [{"seed": s, "recovery_s": None} for s in range(5)]
        mean, se, verdict = paired_verdict(arm, reference, "recovery_s")
        assert verdict == "too few pairs"


class TestTheEvaluationUsesTheTrainingEnvironment:
    """A policy has to be evaluated on the road it learned on. The evaluation built
    its environment from the training config but omitted `human_model`, so it ran
    SUMO's default Krauss -- a road with no capacity drop -- while the policy had
    trained under the calibrated Wiedemann-99. Every field that defines the
    environment is checked here, because the next one omitted will fail the same way
    and just as quietly.
    """

    def test_every_environment_field_reaches_the_run(self, monkeypatch):
        from src.config.loaders import load_named_config
        from scripts import evaluate_burst_scenario as module

        captured = {}

        class Recorder:
            def __init__(self, topology, config):
                captured["topology"] = topology
                captured["config"] = config

            def reset(self, seed=None):
                return {}, {}

            def step(self, actions):
                return {}, 0.0, False, True, {"metrics": {}}

            def close(self):
                pass

            view = None
            arrived_total = 0
            collision_count = 0

        monkeypatch.setattr(module, "SumoTopologyEnv", Recorder)
        training = load_named_config("training", "mappo_sumo")
        module.run_one(controller=None, seed=7, training=training, work_dir=None)

        config = captured["config"]
        assert captured["topology"] == training["topology"]
        assert config["dt"] == training["dt"]
        assert config["duration_steps"] == training["duration_steps"]
        assert config["warmup_steps"] == training["warmup_steps"]
        assert config["sensing"] == training["sensing"]
        assert config["human_model"]["id"] == training["human_model"], (
            "the evaluation is not using the driving model the policy trained under"
        )
        assert config["demand"]["id"] == training["demand"]


class TestTheEvaluationDecidesAtTheRateItTrainedAt:
    """A policy trained at one decision per simulated second must be evaluated at
    one decision per simulated second. Asked ten times as often it is asked for
    actions in states it never saw itself in, and the actuation rate -- which is
    what its reward was earned at -- differs by a factor of ten. The interval is a
    field that has to be passed, and a field built and never passed reads as a
    measured default.
    """

    def test_the_interval_is_read_from_the_training_config(self):
        from scripts.evaluate_burst_scenario import decision_interval_steps

        assert decision_interval_steps({"dt": 0.1, "decision_interval_s": 1.0}) == 10
        assert decision_interval_steps({"dt": 0.5, "decision_interval_s": 1.0}) == 2
        # A config that declares none decides every step, which is what every
        # config written before task 103 does.
        assert decision_interval_steps({"dt": 0.1}) == 1

    def test_the_shipped_config_is_not_evaluated_every_step(self):
        from src.config.loaders import load_named_config
        from scripts.evaluate_burst_scenario import decision_interval_steps

        assert decision_interval_steps(load_named_config("training", "mappo_sumo")) == 10

    def test_the_controller_is_asked_once_per_interval(self, monkeypatch):
        from scripts import evaluate_burst_scenario as module

        class Recorder:
            def __init__(self, topology, config):
                pass

            def reset(self, seed=None):
                return {}, {}

            def step(self, actions):
                return {}, 0.0, False, True, {"metrics": {}}

            def close(self):
                pass

            view = None
            arrived_total = 0
            collision_count = 0

        class CountingController:
            def __init__(self):
                self.calls = 0

            def act(self, observations, global_state=None):
                self.calls += 1
                return {}

        monkeypatch.setattr(module, "SumoTopologyEnv", Recorder)
        controller = CountingController()
        training = {
            "topology": "inverted_tree", "demand": "sumo_capacity_drop",
            "dt": 0.1, "duration_steps": 100, "warmup_steps": 0,
            "sensing": {}, "human_model": "w99_calibrated",
            "decision_interval_s": 1.0,
        }
        module.run_one(controller=controller, seed=7, training=training, work_dir=None)
        assert controller.calls == 10, "the policy was not asked once per second"

        # The control: with no interval declared it is asked every step, so the
        # assertion above is about the interval and not about some other cap.
        every_step = CountingController()
        module.run_one(controller=every_step, seed=7,
                       training={**training, "decision_interval_s": None},
                       work_dir=None)
        assert every_step.calls == 100


class TestTheBurstOnlyMetricsAreAbsentWithoutABurst:

    def test_recovery_is_none_under_a_steady_demand(self, monkeypatch):
        # `recovery_s` is the time from the end of the burst until the queue clears.
        # Under a demand that declares no burst, `burst.end_s` defaults to 0.0, so
        # every step is "after the burst" and the metric would report the first
        # moment the queue happened to fall below ten vehicles -- a number with no
        # referent, printed in the same column as one that had one.
        from src.config.loaders import load_named_config
        from scripts import evaluate_burst_scenario as module

        class Recorder:
            def __init__(self, topology, config):
                pass

            def reset(self, seed=None):
                return {}, {}

            def step(self, actions):
                return {}, 0.0, False, True, {"metrics": {"queue_length_total": 0}}

            def close(self):
                pass

            view = None
            arrived_total = 0
            collision_count = 0

        monkeypatch.setattr(module, "SumoTopologyEnv", Recorder)
        training = dict(load_named_config("training", "mappo_sumo"))
        assert not (load_named_config("demand", str(training["demand"]))
                    .get("burst", {}).get("enabled", False))
        steady = module.run_one(controller=None, seed=7, training=training, work_dir=None)
        assert steady["recovery_s"] is None

        # The control: under a demand that DOES declare a burst, an empty queue
        # after it produces a number.
        training["demand"] = "sumo_burst"
        bursty = module.run_one(controller=None, seed=7, training=training, work_dir=None)
        assert bursty["recovery_s"] is not None
