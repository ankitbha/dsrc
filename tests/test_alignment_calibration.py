"""The calibration's advantage construction, tested without SUMO.

`scripts/measure_alignment_calibration.py` converts a gradient-norm z into a bound on
the action-advantage correlation. Two earlier conversions were wrong -- one applied a
correlation's sampling error to a statistic that is not a correlation, the other
assumed the relationship is linear -- so the construction this one rests on is tested
rather than trusted: the synthesised advantage must carry the correlation it claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_alignment_calibration import build_advantage, correlation_with_action  # noqa: E402


def _actions(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 3, (n, 2), generator=generator)


@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.2, 0.5, 0.75, 1.0])
def test_synthesised_advantage_carries_the_requested_correlation(target: float) -> None:
    """The whole conversion rests on c being what it says, so assert it directly."""
    actions = _actions(4000)
    noise = torch.randn(4000, generator=torch.Generator().manual_seed(1))
    advantage = build_advantage(actions, noise, target)
    assert correlation_with_action(actions, advantage) == pytest.approx(target, abs=5e-3)


def test_a_correlation_of_one_is_an_affine_function_of_the_indicator() -> None:
    """At c = 1 the advantage must be the standardised indicator, not merely close."""
    actions = _actions(500)
    noise = torch.randn(500, generator=torch.Generator().manual_seed(2))
    advantage = build_advantage(actions, noise, 1.0)
    indicator = (actions[:, 0] == 0).float()
    indicator = (indicator - indicator.mean()) / indicator.std()
    assert torch.allclose(advantage, indicator, atol=1e-4)


def test_the_noise_component_does_not_leak_correlation() -> None:
    """At c = 0 the advantage is the noise alone and must carry no correlation.

    This is what fails if the orthogonalisation is dropped: a noise vector that happens
    to correlate with the action would be read as instrument sensitivity.
    """
    actions = _actions(4000, seed=3)
    # Noise deliberately built to correlate with the action before orthogonalisation.
    leaky = (actions[:, 0] == 0).float() * 3.0 + torch.randn(
        4000, generator=torch.Generator().manual_seed(4))
    advantage = build_advantage(actions, leaky, 0.0)
    assert abs(correlation_with_action(actions, advantage)) < 5e-3


def test_the_real_advantage_survives_as_the_noise_component() -> None:
    """At c = 0 the advantage must keep the supplied vector's shape, up to scale.

    The first calibration used i.i.d. noise and read z = -0.59 at c = 0 because the
    null's advantages are autocorrelated and i.i.d. ones are not. Passing the real
    advantage is the fix, so the construction must preserve it rather than replace it.
    """
    actions = _actions(300, seed=5)
    real = torch.cumsum(torch.randn(300, generator=torch.Generator().manual_seed(6)), 0)
    advantage = build_advantage(actions, real, 0.0)
    centred = (real - real.mean()) / real.std()
    indicator = (actions[:, 0] == 0).float()
    indicator = (indicator - indicator.mean()) / indicator.std()
    residual = centred - (centred @ indicator) / (indicator @ indicator) * indicator
    residual = (residual - residual.mean()) / residual.std()
    assert torch.allclose(advantage, residual, atol=1e-4)


def test_an_all_one_action_column_is_refused_rather_than_silently_zero() -> None:
    """If no agent ever chose `slow` the indicator has zero variance and c is undefined.

    Returning a zero-correlation advantage there would look like a measured null.
    """
    actions = torch.ones((100, 2), dtype=torch.long)
    noise = torch.randn(100, generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="never chose"):
        build_advantage(actions, noise, 0.5)
