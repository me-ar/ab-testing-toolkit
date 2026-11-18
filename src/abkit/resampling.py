"""Bootstrap confidence intervals — no distributional assumptions."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_diff(control: np.ndarray, treatment: np.ndarray, stat: Callable = np.mean,
                   n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for stat(treatment) - stat(control).

    Vectorised in chunks so 5,000 resamples of ~45k users run in seconds.
    """
    rng = np.random.default_rng(seed)
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    diffs = np.empty(n_boot)
    chunk = 250
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        ic = rng.integers(0, len(control), (k, len(control)))
        it = rng.integers(0, len(treatment), (k, len(treatment)))
        diffs[start:start + k] = (stat(treatment[it], axis=1) - stat(control[ic], axis=1))
    observed = stat(treatment) - stat(control)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value via the shifted-null approximation
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"observed_diff": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_value_approx": float(min(1.0, p)), "samples": diffs}
