"""Tests validate abkit against statsmodels/scipy reference implementations and known maths."""
import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

import abkit


def test_ztest_matches_statsmodels():
    r = abkit.two_proportion_ztest(500, 5000, 560, 5000)
    _, p_ref = proportions_ztest([560, 500], [5000, 5000])
    assert r.p_value == pytest.approx(p_ref, rel=1e-9)
    assert r.ci_low < r.abs_diff < r.ci_high


def test_welch_matches_scipy_and_ci_covers_diff():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 400), rng.normal(10.5, 3, 500)
    r = abkit.welch_ttest(a, b)
    assert r.ci_low < r.abs_diff < r.ci_high
    assert r.p_value < 0.01


def test_srm_detects_broken_split():
    assert abkit.srm_check(50_000, 50_100)[1] is False
    assert abkit.srm_check(50_000, 52_000)[1] is True


def test_sample_size_close_to_statsmodels():
    n = abkit.sample_size_proportions(0.10, 0.10)
    es = proportion_effectsize(0.11, 0.10)
    n_ref = NormalIndPower().solve_power(es, alpha=0.05, power=0.8)
    assert abs(n - n_ref) / n_ref < 0.02


def test_power_and_mde_are_inverse():
    n = abkit.sample_size_proportions(0.2, 0.05)
    assert abkit.power_proportions(0.2, 0.05, n) == pytest.approx(0.8, abs=0.01)
    assert abkit.minimum_detectable_effect(0.2, n) == pytest.approx(0.05, rel=0.02)


def test_bayesian_probability_is_sensible():
    clear = abkit.beta_binomial(100, 1000, 200, 1000)
    assert clear["p_treatment_better"] > 0.999
    null = abkit.beta_binomial(100, 1000, 100, 1000)
    assert 0.4 < null["p_treatment_better"] < 0.6
    assert clear["expected_loss_ship_treatment"] < clear["expected_loss_ship_control"]


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a, b = rng.exponential(10, 3000), rng.exponential(11, 3000)
    r = abkit.bootstrap_diff(a, b, n_boot=1000)
    assert r["ci_low"] < 1.0 < r["ci_high"]


def test_cuped_reduces_variance_by_corr_squared():
    rng = np.random.default_rng(2)
    x = rng.normal(size=20_000)
    y = 0.8 * x + rng.normal(scale=0.6, size=20_000)
    y_adj, reduction = abkit.cuped(y, x)
    rho2 = np.corrcoef(x, y)[0, 1] ** 2
    assert reduction == pytest.approx(rho2, abs=0.01)
    assert y_adj.mean() == pytest.approx(y.mean(), abs=1e-9)  # unbiased


def test_peeking_inflates_false_positives():
    r = abkit.simulate_peeking(n_per_group=2000, n_looks=10, n_sims=1000)
    assert r["fpr_fixed_horizon"] < 0.08
    assert r["fpr_with_peeking"] > 2 * r["fpr_fixed_horizon"]


def test_holm_bonferroni():
    # thresholds: 0.05/3, 0.05/2, 0.05/1 — step-down stops at the first failure
    assert abkit.holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert abkit.holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
