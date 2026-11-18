"""abkit — a small, tested toolkit for analysing A/B experiments."""
from .bayesian import beta_binomial
from .design import (
    cuped,
    minimum_detectable_effect,
    power_proportions,
    sample_size_proportions,
    simulate_peeking,
)
from .frequentist import holm_bonferroni, mann_whitney, srm_check, two_proportion_ztest, welch_ttest
from .resampling import bootstrap_diff

__all__ = ["beta_binomial", "bootstrap_diff", "cuped", "holm_bonferroni", "mann_whitney",
           "minimum_detectable_effect", "power_proportions", "sample_size_proportions",
           "simulate_peeking", "srm_check", "two_proportion_ztest", "welch_ttest"]
__version__ = "1.0.0"
