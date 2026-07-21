"""evalgate — dependency-free statistical checks for AI eval claims.

Before you publish "we lead on subset X", a judge "prefers" your model, or a
"super-linear" scaling law, run the cheap check that most directly asks whether
the number survives. The open companion to independent eval-integrity audits.
"""
from .checks import (
    Bias,
    Correction,
    Fragility,
    bias_rate,
    binomial_test,
    bonferroni,
    correct_best_of,
    leave_one_out,
    ols_slope,
    power_law_exponent,
    sidak,
)

__version__ = "0.1.0"
__all__ = [
    "correct_best_of", "sidak", "bonferroni", "Correction",
    "bias_rate", "binomial_test", "Bias",
    "leave_one_out", "ols_slope", "power_law_exponent", "Fragility",
]
