import numpy as np
import pytest

from here_error import stats


def test_seeded_bootstrap_is_reproducible():
    v = np.array([0.1, 0.2, -0.1, 0.4, 0.0, 0.3, 0.7, -0.6, 0.25, 0.05])
    c = ["a", "a", "b", "b", "c", "c", "d", "e", "f", "g"]
    r1 = stats.cluster_bootstrap_median(v, c, n_boot=500, seed=3)
    r2 = stats.cluster_bootstrap_median(v, c, n_boot=500, seed=3)
    r3 = stats.cluster_bootstrap_median(v, c, n_boot=500, seed=4)
    assert (r1.lo, r1.hi) == (r2.lo, r2.hi)
    assert (r1.lo, r1.hi) != (r3.lo, r3.hi)
    assert r1.median == pytest.approx(np.median(v)) and r1.n == 10 and r1.n_clusters == 7


def test_cluster_interval_is_wider_when_one_road_holds_most_passes():
    rng = np.random.default_rng(0)
    # One road with 40 passes near 0.0 and four roads with one pass each at large offsets.
    big = rng.normal(0.0, 0.01, 40)
    small = [0.5, -0.5, 0.6, -0.4]
    values = np.concatenate([big, small])
    clusters = ["big"] * 40 + ["r1", "r2", "r3", "r4"]
    cl = stats.cluster_bootstrap_median(values, clusters, n_boot=2000, seed=1)
    row = stats.row_bootstrap_median(values, n_boot=2000, seed=1)
    assert (cl.hi - cl.lo) > 5 * (row.hi - row.lo)


def test_spearman_with_ties_matches_hand_value():
    x = [1, 2, 2, 3]
    y = [1, 3, 2, 4]
    # ranks x: 1, 2.5, 2.5, 4 ; ranks y: 1, 3, 2, 4 ; Pearson of those ranks.
    rx, ry = np.array([1, 2.5, 2.5, 4]), np.array([1, 3, 2, 4])
    assert stats.spearman(x, y) == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    assert list(stats.average_ranks([5, 1, 1, 3])) == [4.0, 1.5, 1.5, 3.0]
    assert np.isnan(stats.spearman([1, 1, 1], [1, 2, 3]))
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_thirds_split_by_value():
    g = stats.thirds([1, 2, 3, 4, 5, 6])
    assert list(g) == [0, 0, 1, 1, 2, 2]


def test_median_half_width_matches_the_plan_example():
    assert stats.median_half_width(0.25, 18) == pytest.approx(0.145, abs=0.002)


def test_interval_is_the_2_5_to_97_5_percentile_of_resampled_medians():
    v = np.array([0.1, 0.5, -0.3, 0.2, 0.9, -0.7, 0.4])
    c = list("abcdefg")
    got = stats.cluster_bootstrap_median(v, c, n_boot=300, seed=11)
    rng = np.random.default_rng(11)
    draws = [np.median(v[rng.integers(0, 7, size=7)]) for _ in range(300)]
    assert got.lo == pytest.approx(np.percentile(draws, 2.5)) and got.hi == pytest.approx(np.percentile(draws, 97.5))
    mean_draws = [np.mean(v[np.random.default_rng(11).integers(0, 7, size=7)])]
    assert got.median == np.median(v) != np.mean(v)


def test_thirds_use_less_than_or_equal_at_the_quantile():
    # Seven values: the 1/3 and 2/3 quantiles are exactly 3 and 5, and those values belong to the lower group.
    assert list(stats.thirds([1, 2, 3, 4, 5, 6, 7])) == [0, 0, 0, 1, 1, 2, 2]


def test_spearman_ties_use_average_ranks_in_the_coefficient():
    x = [1, 2, 2, 4, 5]
    y = [5, 6, 7, 8, 7]
    rx, ry = stats.average_ranks(x), stats.average_ranks(y)
    assert list(rx) == [1.0, 2.5, 2.5, 4.0, 5.0] and list(ry) == [1.0, 2.0, 3.5, 5.0, 3.5]
    assert stats.spearman(x, y) == pytest.approx(np.corrcoef(rx, ry)[0, 1])
