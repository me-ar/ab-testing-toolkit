"""Experiment design: power analysis, variance reduction (CUPED) and the cost of peeking."""
from __future__ import annotations

import numpy as np
from scipy import stats


def sample_size_proportions(baseline: float, mde_rel: float, alpha: float = 0.05,
                            power: float = 0.8) -> int:
    """Users needed *per group* to detect a relative lift `mde_rel` on a conversion rate."""
    p1, p2 = baseline, baseline * (1 + mde_rel)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def power_proportions(baseline: float, mde_rel: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Achieved power for a two-sided two-proportion test."""
    p1, p2 = baseline, baseline * (1 + mde_rel)
    se = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    d = abs(p2 - p1) / se
    return float(stats.norm.sf(z_a - d) + stats.norm.cdf(-z_a - d))


def minimum_detectable_effect(baseline: float, n_per_group: int, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Smallest relative lift detectable with the given sample size (solved by bisection)."""
    lo, hi = 1e-5, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_proportions(baseline, mid, n_per_group, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def cuped(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED (Deng et al., 2013): y_adj = y - θ(x - mean(x)), θ = cov(x, y)/var(x).

    Using a pre-experiment covariate removes variance that has nothing to do with the
    treatment, so the same effect is detected with fewer users. Returns the adjusted
    metric and the variance-reduction fraction (≈ corr(x, y)^2).
    """
    y, x_pre = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(x_pre, y)[0, 1] / np.var(x_pre, ddof=1)
    y_adj = y - theta * (x_pre - x_pre.mean())
    return y_adj, float(1 - y_adj.var() / y.var())


def simulate_peeking(n_per_group: int = 10_000, n_looks: int = 20, p: float = 0.2, n_sims: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """A/A simulation: how often does 'stop as soon as p < 0.05' declare a false winner?"""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_group / n_looks, n_per_group, n_looks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    a = rng.random((n_sims, n_per_group)) < p
    b = rng.random((n_sims, n_per_group)) < p
    ca, cb = a.cumsum(1)[:, looks - 1], b.cumsum(1)[:, looks - 1]
    pa, pb = ca / looks, cb / looks
    pool = (ca + cb) / (2 * looks)
    z = (pb - pa) / np.sqrt(pool * (1 - pool) * 2 / looks)
    sig = np.abs(z) > z_crit
    return {"fpr_fixed_horizon": float(sig[:, -1].mean()),
            "fpr_with_peeking": float(sig.any(axis=1).mean()),
            "fpr_by_look": sig.cumsum(1).clip(0, 1).mean(0).tolist(),
            "looks": looks.tolist()}
