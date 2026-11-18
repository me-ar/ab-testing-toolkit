"""Frequentist tests for A/B experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control_mean: float
    treatment_mean: float
    abs_diff: float
    rel_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict:
        return {**asdict(self), "significant": self.significant}


def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> tuple[float, bool]:
    """Sample Ratio Mismatch: chi-square goodness-of-fit on group sizes.

    A tiny p-value (< 0.001 by convention) means randomisation or logging is broken and the
    experiment results should not be trusted.
    """
    total = n_control + n_treatment
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    _, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return float(p), bool(p < 0.001)


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05,
                         metric: str = "conversion") -> TestResult:
    """Pooled z-test for H0: p_t == p_c, with an unpooled Wald CI for the difference."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return TestResult(metric, p_c, p_t, diff, diff / p_c, diff - zc * se, diff + zc * se,
                      float(p_value), "two-proportion z-test")


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05,
                metric: str = "mean") -> TestResult:
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    res = stats.ttest_ind(treatment, control, equal_var=False)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    m_c, m_t = control.mean(), treatment.mean()
    return TestResult(metric, m_c, m_t, m_t - m_c, (m_t - m_c) / m_c, float(ci.low), float(ci.high),
                      float(res.pvalue), "Welch t-test")


def mann_whitney(control: np.ndarray, treatment: np.ndarray, metric: str = "distribution") -> TestResult:
    """Rank-based test — robust to the heavy tails typical of engagement/revenue metrics."""
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    res = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    m_c, m_t = np.median(control), np.median(treatment)
    return TestResult(metric, float(m_c), float(m_t), float(m_t - m_c),
                      float((m_t - m_c) / m_c) if m_c else np.nan, np.nan, np.nan,
                      float(res.pvalue), "Mann-Whitney U (medians shown)")


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Control family-wise error when testing several metrics on the same experiment."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m, reject, still = len(items), {}, True
    for i, (k, p) in enumerate(items):
        still = still and p <= alpha / (m - i)
        reject[k] = still
    return reject
