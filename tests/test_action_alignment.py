"""The instrument every conclusion in tasks 105 to 119 rests on, run against controls.

It answers one question: does the advantage carry information about the action that
earned it? That question is only answerable against a null, and the first null used
-- permuting advantages across the batch -- is invalid when advantages are temporally
correlated, because it destroys each agent's temporal profile as well as the pairing.
The segment arms read a measured value systematically BELOW that null, which is the
bias showing.

These tests do to the instrument what should have been done before it was trusted:
give it a batch with a signal it must see, and a batch with none it must not see.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_action_alignment import alignment_z  # noqa: E402
from src.rl.encoders import local_obs_dim  # noqa: E402
from src.rl.ppo import PPOConfig  # noqa: E402
from src.rl.trainers import SharedPPOTrainer, TrainingConfig, seed_everything  # noqa: E402


def _trainer():
    seed_everything(0)
    return SharedPPOTrainer(
        TrainingConfig(algorithm="shared_ppo", action_profile="speed_headway",
                       hidden_sizes=(16,)),
        PPOConfig(update_epochs=1, minibatch_size=32),
        device="cpu",
    )


def _batch(trainer, n=600, seed=0):
    torch.manual_seed(seed)
    observations = torch.randn(n, local_obs_dim())
    with torch.no_grad():
        _, actions, log_probs, _ = trainer.actor.sample(observations)
    return observations, actions, log_probs


class TestTheInstrumentReadsWhatItShould:

    def test_an_advantage_aligned_with_the_action_reads_far_above_the_null(self):
        # The ceiling case. Advantage is +1 wherever the policy chose `slow` and -1
        # otherwise, so it is perfectly informative about the action. If the
        # instrument cannot see this, it cannot see anything.
        trainer = _trainer()
        observations, actions, log_probs = _batch(trainer)
        advantages = (actions[:, 0] == 0).float() * 2.0 - 1.0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        z, measured, mean, sd = alignment_z(
            trainer, observations, actions, log_probs, advantages,
            clip=0.2, resamples=40, seed=0)
        assert z > 5.0, (z, measured, mean, sd)

    def test_an_advantage_independent_of_the_action_reads_at_the_null(self):
        # The floor case, and the one that matters: an advantage drawn without
        # reference to the action must NOT read as signal. A null that fails here
        # manufactures findings.
        trainer = _trainer()
        observations, actions, log_probs = _batch(trainer)
        torch.manual_seed(99)
        advantages = torch.randn(observations.shape[0])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        z, *_ = alignment_z(trainer, observations, actions, log_probs, advantages,
                            clip=0.2, resamples=40, seed=0)
        assert abs(z) < 3.0, z

    def test_a_temporally_smooth_but_uninformative_advantage_still_reads_at_the_null(self):
        # THE CASE THE PERMUTATION FLOOR GOT WRONG. The advantage is a slow random
        # walk -- strongly autocorrelated, exactly like a real agent's trajectory --
        # and carries no information about the action. Permuting it across the batch
        # would make the sequence rough, raise the null, and push the measured value
        # below it; that is how the segment arms came to read -1.18 and -1.75.
        trainer = _trainer()
        observations, actions, log_probs = _batch(trainer)
        torch.manual_seed(7)
        steps = torch.randn(observations.shape[0]) * 0.1
        advantages = torch.cumsum(steps, dim=0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # The premise: this really is autocorrelated, or the test proves nothing.
        lag = torch.corrcoef(torch.stack([advantages[:-1], advantages[1:]]))[0, 1]
        assert lag > 0.9, f"the fixture is not autocorrelated: lag-1 {lag:.3f}"
        z, *_ = alignment_z(trainer, observations, actions, log_probs, advantages,
                            clip=0.2, resamples=40, seed=0)
        assert abs(z) < 3.0, z

    def test_a_partial_signal_reads_between_the_two(self):
        # Monotonicity: the statistic should grow with how much of the advantage is
        # about the action, not merely fire or not.
        trainer = _trainer()
        observations, actions, log_probs = _batch(trainer)
        torch.manual_seed(5)
        noise = torch.randn(observations.shape[0])
        aligned = (actions[:, 0] == 0).float() * 2.0 - 1.0
        readings = []
        for weight in (0.0, 0.3, 1.0):
            advantages = weight * aligned + (1.0 - weight) * noise
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            z, *_ = alignment_z(trainer, observations, actions, log_probs, advantages,
                                clip=0.2, resamples=40, seed=0)
            readings.append(z)
        assert readings[0] < readings[1] < readings[2], readings


class TestTheSignedDiagnostic:
    """The gradient norm is a magnitude and cannot say which way the advantage points.

    An advantage that is silent about the action and one that says "this action is
    bad" both reduce the norm relative to a perfectly aligned advantage, and the two
    call for opposite conclusions: nothing to learn, against something to learn that
    contradicts the mechanism under study.
    """

    def test_it_separates_aligned_anti_aligned_and_silent(self):
        from scripts.measure_action_alignment import action_advantage_correlation

        actions = torch.tensor([[0, 0], [0, 0], [1, 0], [2, 0]])
        assert action_advantage_correlation(
            actions, torch.tensor([1.0, 1.0, -1.0, -1.0])) == pytest.approx(1.0)
        assert action_advantage_correlation(
            actions, torch.tensor([-1.0, -1.0, 1.0, 1.0])) == pytest.approx(-1.0)
        assert action_advantage_correlation(
            actions, torch.tensor([1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.0)

    def test_a_constant_advantage_reads_zero_rather_than_dividing_by_zero(self):
        from scripts.measure_action_alignment import action_advantage_correlation

        actions = torch.tensor([[0, 0], [1, 0]])
        assert action_advantage_correlation(actions, torch.tensor([1.0, 1.0])) == 0.0

    def test_a_constant_action_reads_zero_rather_than_dividing_by_zero(self):
        # Every agent choosing the same value is what a collapsed policy looks like,
        # and it must not read as a correlation of any sign.
        from scripts.measure_action_alignment import action_advantage_correlation

        actions = torch.tensor([[1, 0], [1, 0]])
        assert action_advantage_correlation(actions, torch.tensor([1.0, -1.0])) == 0.0


class TestTheStatisticIsUnbiasedUnderAKnownNull:
    """z read negative on every real arm; this establishes that is the DATA.

    Under a null that resamples the action from the policy at the same state, the
    resampled and actual actions are exchangeable, so z should centre on zero. Every
    real arm read negative -- +0.00, -0.47, -1.32, -0.44, -0.81 -- which is either a
    property of real rollouts or a fault in the statistic. This decides which.
    """

    def test_it_centres_on_zero_when_the_advantage_is_independent(self):
        import statistics

        zs = []
        for trial in range(12):
            seed_everything(trial)
            trainer = SharedPPOTrainer(
                TrainingConfig(algorithm="shared_ppo", action_profile="speed_headway",
                               hidden_sizes=(16,)),
                PPOConfig(), device="cpu")
            torch.manual_seed(1000 + trial)
            observations = torch.randn(1200, local_obs_dim())
            with torch.no_grad():
                _, actions, log_probs, _ = trainer.actor.sample(observations)
            advantages = torch.randn(1200)
            advantages = (advantages - advantages.mean()) / advantages.std()
            z, *_ = alignment_z(trainer, observations, actions, log_probs, advantages,
                                clip=0.2, resamples=20, seed=trial)
            zs.append(z)

        mean = statistics.fmean(zs)
        se = statistics.stdev(zs) / len(zs) ** 0.5
        # Measured over 30 trials at n=2000: mean +0.148 +/- 0.188, sd 1.030. The
        # bound here is loose enough for 12 trials at n=1200 and tight enough to
        # catch the -0.8 the real arms show.
        assert abs(mean) < 3 * se + 0.3, (mean, se, zs)
        # And the spread is about 1, which is what a z-score should give: a statistic
        # whose null spread were much smaller would report everything as significant.
        assert 0.4 < statistics.stdev(zs) < 2.5, statistics.stdev(zs)
