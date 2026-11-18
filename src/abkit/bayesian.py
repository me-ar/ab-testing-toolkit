"""Bayesian A/B analysis with conjugate Beta-Binomial posteriors."""
from __future__ import annotations

import numpy as np


def beta_binomial(x_c: int, n_c: int, x_t: int, n_t: int, prior_a: float = 1.0, prior_b: float = 1.0,
                  draws: int = 200_000, seed: int = 0) -> dict:
    """Posterior summaries for a conversion-rate experiment.

    Returns the probability that treatment beats control, the posterior of the relative lift,
    and the *expected loss* of shipping each variant — the quantity a decision-maker actually
    wants: "if I pick B and I'm wrong, how much do I lose on average?"
    """
    rng = np.random.default_rng(seed)
    pc = rng.beta(prior_a + x_c, prior_b + n_c - x_c, draws)
    pt = rng.beta(prior_a + x_t, prior_b + n_t - x_t, draws)
    lift = (pt - pc) / pc
    return {
        "p_treatment_better": float((pt > pc).mean()),
        "lift_mean": float(lift.mean()),
        "lift_95_hdi": (float(np.percentile(lift, 2.5)), float(np.percentile(lift, 97.5))),
        "expected_loss_ship_treatment": float(np.maximum(pc - pt, 0).mean()),
        "expected_loss_ship_control": float(np.maximum(pt - pc, 0).mean()),
        "_samples": (pc, pt),
    }
