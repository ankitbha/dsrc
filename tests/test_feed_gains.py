"""The minimum-penetration certificate's estimation and bounds. No SUMO."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.gains import (
    confidence_bounds, deviations, estimate_level, fit_gains, linearity_ratio, load_routes,
    min_penetration, pulse_sites, rho_point, rho_upper_bound, spectral_radius, t_quantile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "mainz"


def _simulate(a: np.ndarray, B: np.ndarray, site: int, samples: int = 60, dt: float = 10.0,
              noise: float = 0.0, rng=None) -> np.ndarray:
    n = len(a)
    y = np.zeros((samples, n))
    y[0, site] = 1.0
    for t in range(samples - 1):
        step = dt * (-a * y[t] + B @ y[t])
        if noise:
            step = step + rng.normal(0.0, noise, n)
        y[t + 1] = y[t] + step
    return y


def _system(rng, n=12):
    a = rng.uniform(0.005, 0.02, n)
    B = rng.uniform(0.0, 0.002, (n, n)) * (rng.random((n, n)) < 0.4)
    np.fill_diagonal(B, 0.0)
    return a, B


class TestStudentT:
    @pytest.mark.parametrize("p,df,value", [(0.95, 14, 1.761310), (0.975, 10, 2.228139),
                                            (0.95, 2, 2.919986), (0.99, 30, 2.457262)])
    def test_known_quantiles(self, p, df, value):
        assert t_quantile(p, df) == pytest.approx(value, abs=1e-5)


class TestFitGains:
    def test_a_known_linear_system_is_recovered_within_ten_percent(self):
        rng = np.random.default_rng(0)
        a, B = _system(rng)
        runs = [_simulate(a, B, site, noise=1e-4, rng=rng) for site in range(12)]
        a_hat, B_hat = fit_gains(runs, dt=10.0)
        assert np.all(np.abs(a_hat - a) <= 0.1 * a)
        big = B > 0.0005
        assert np.all(np.abs(B_hat[big] - B[big]) <= 0.1 * B[big])
        assert np.all(np.diag(B_hat) == 0.0)

    def test_a_coefficient_is_its_magnitude(self):
        # A system whose cross term is negative still yields a nonnegative b.
        a = np.array([0.05, 0.05])
        B = np.array([[0.0, -0.01], [0.0, 0.0]])
        runs = [_simulate(a, B, s) for s in range(2)]
        _, B_hat = fit_gains(runs, dt=10.0)
        assert B_hat[0, 1] == pytest.approx(0.01, rel=1e-6)


class TestBounds:
    def test_lowering_any_a_never_lowers_rho(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            a, B = _system(rng)
            base = rho_upper_bound(a, B)
            e = rng.integers(12)
            lowered = a.copy()
            lowered[e] *= 0.5
            assert rho_upper_bound(lowered, B) >= base - 1e-12

    def test_raising_any_b_never_lowers_rho(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            a, B = _system(rng)
            base = rho_upper_bound(a, B)
            e, j = rng.integers(12, size=2)
            if e == j:
                continue
            raised = B.copy()
            raised[e, j] += 0.003
            assert rho_upper_bound(a, raised) >= base - 1e-12

    def test_the_upper_bound_is_at_least_the_point_estimate_on_100_instances(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            a, B = _system(rng)
            a_seeds = a + rng.normal(0.0, 0.003, (15, 12))
            B_seeds = np.abs(B + rng.normal(0.0, 0.0005, (15, 12, 12)))
            for s in range(15):
                np.fill_diagonal(B_seeds[s], 0.0)
            bounds = confidence_bounds(a_seeds, B_seeds)
            assert np.all(bounds["a_lcb"] <= bounds["a_hat"])
            assert np.all(bounds["b_ucb"] >= bounds["b_hat"])
            upper = rho_upper_bound(bounds["a_lcb"], bounds["b_ucb"])
            assert upper >= rho_point(bounds["a_hat"], bounds["b_hat"]) - 1e-12

    def test_a_nonpositive_lower_bound_gives_no_certificate(self):
        a = np.array([0.05, -0.001])
        assert rho_upper_bound(a, np.zeros((2, 2))) is None
        assert rho_upper_bound(np.array([0.05, 0.0]), np.zeros((2, 2))) is None

    def test_the_bounds_use_the_one_sided_t_quantile(self):
        a_seeds = np.array([[1.0], [2.0], [3.0]])
        B_seeds = np.zeros((3, 1, 1))
        bounds = confidence_bounds(a_seeds, B_seeds)
        se = 1.0 / np.sqrt(3)
        assert bounds["a_lcb"][0] == pytest.approx(2.0 - t_quantile(0.95, 2) * se)
        assert bounds["b_ucb"][0, 0] == 0.0  # max(0, ...) and the diagonal is zero

    def test_spectral_radius_of_a_known_matrix(self):
        assert spectral_radius(np.array([[0.0, 2.0], [0.5, 0.0]])) == pytest.approx(1.0)


class TestMinPenetration:
    def test_the_smallest_certified_level_and_the_first_failing_one(self):
        out = min_penetration({1.0: 0.4, 0.75: 0.6, 0.5: 0.9, 0.25: 1.2, 0.1: None})
        assert out["p_min"] == 0.5
        assert out["first_not_met"] == 0.25
        assert out["non_monotone"] is False

    def test_a_certified_level_below_a_failing_one_is_flagged(self):
        out = min_penetration({1.0: 0.4, 0.75: 1.1, 0.5: 0.9, 0.25: 1.2, 0.1: None})
        assert out["p_min"] == 0.5  # the plan's rule: smallest level with rho_UCB < 1
        assert out["p_min_contiguous"] == 1.0
        assert out["non_monotone"] is True

    def test_no_level_certified(self):
        out = min_penetration({1.0: 1.5, 0.5: None})
        assert out["p_min"] is None and out["first_not_met"] == 1.0


class TestDeviations:
    def test_absolute_density_difference_over_the_window(self):
        base = [{"t": float(t), "density": [0.1, 0.2]} for t in range(1100, 1900, 10)]
        pulse = [{"t": float(t), "density": [0.1 + (0.05 if t >= 1200 else 0.0), 0.15]}
                 for t in range(1100, 1900, 10)]
        y = deviations(pulse, base, 1200.0, 1800.0)
        assert y.shape == (61, 2)
        assert np.allclose(y[:, 0], 0.05) and np.allclose(y[:, 1], 0.05)

    def test_misaligned_traces_are_refused(self):
        base = [{"t": 10.0, "density": [0.1]}]
        with pytest.raises(ValueError):
            deviations([{"t": 20.0, "density": [0.1]}], base, 0.0, 100.0)

    def test_the_linearity_ratio_is_peak_over_peak_at_the_site(self):
        y60 = np.array([[0.0, 0.1], [0.4, 0.2]])
        y30 = np.array([[0.0, 0.1], [0.2, 0.1]])
        assert linearity_ratio(y60, y30, 0) == pytest.approx(2.0)


class TestPulseSites:
    def test_twelve_sites_on_the_committed_network(self):
        segments = json.loads((DATA / "mainz_segments.json").read_text())
        routes = load_routes(DATA / "mainz_routes.rou.xml")
        sites = pulse_sites(segments, routes)
        assert [s["site"] for s in sites] == list(range(12))
        for site in sites:
            assert site["edge"] in segments[site["site"]]
            assert site["suffix"][0] == site["edge"]
            assert site["suffix"][-1] == "exit_tail"
            route = routes[site["route"]]
            assert route[-len(site["suffix"]):] == site["suffix"]

    def test_site_7_is_on_r277_from_edge_505(self):
        segments = json.loads((DATA / "mainz_segments.json").read_text())
        sites = pulse_sites(segments, load_routes(DATA / "mainz_routes.rou.xml"))
        assert sites[7]["route"] == "r277"
        assert sites[7]["edge"] == "505"


class TestR114:
    """Amendment 1, A10: tests that kill the round-1 survivors in this module."""

    def test_a_growing_self_response_gives_a_negative_a_and_no_certificate(self):
        """Mutant G4: a = |coef_ee| / dt would issue a certificate here."""
        rng = np.random.default_rng(11)
        a, B = _system(rng)
        a[3] = -0.004  # segment 3's own deviation grows
        seed_runs = {seed: [_simulate(a, B, site, noise=1e-5, rng=rng) for site in range(12)]
                     for seed in range(5)}
        level = estimate_level(seed_runs)
        assert level["a_hat"][3] < 0
        assert level["a_LCB"][3] < 0
        assert level["rho_UCB"] is None and level["certificate"] is False
        assert level["no_certificate_reason"]

    def test_the_bonferroni_bound_uses_the_bonferroni_quantile(self):
        """Mutant G7: the Bonferroni bound computed at the 95% level."""
        rng = np.random.default_rng(12)
        a, B = _system(rng)
        seed_runs = {seed: [_simulate(a, B, site, noise=1e-4, rng=rng) for site in range(12)]
                     for seed in range(15)}
        level = estimate_level(seed_runs)
        assert level["t_bonferroni"] == pytest.approx(t_quantile(1 - 0.05 / 144, 14))
        assert level["t_bonferroni"] == pytest.approx(4.3283, abs=1e-4)
        assert level["t"] == pytest.approx(t_quantile(0.95, 14))
        assert level["rho_UCB_bonferroni"] > level["rho_UCB"]
