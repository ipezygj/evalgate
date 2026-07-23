"""evalgate — dependency-free statistical checks for AI eval claims.

Before you publish "we lead on subset X", a judge "prefers" your model, or a
"super-linear" scaling law, run the cheap check that most directly asks whether
the number survives. The open companion to independent eval-integrity audits.
"""
from .checks import (
    Bias,
    Correction,
    Fragility,
    Power,
    bias_rate,
    binomial_test,
    bonferroni,
    correct_best_of,
    leave_one_out,
    min_detectable_effect,
    ols_slope,
    power_check,
    power_law_exponent,
    sidak,
)
from .leaderboard import (
    MatrixAudit,
    PairwiseAudit,
    Dimensionality,
    audit_matrix,
    audit_pairwise,
    latent_dimensions,
    mcnemar_p,
)

__version__ = "0.4.0"
__all__ = [
    "correct_best_of", "sidak", "bonferroni", "Correction",
    "bias_rate", "binomial_test", "Bias",
    "leave_one_out", "ols_slope", "power_law_exponent", "Fragility",
    "power_check", "min_detectable_effect", "Power",
    # whole-leaderboard audits (raw per-item / pairwise data)
    "audit_matrix", "audit_pairwise", "latent_dimensions", "mcnemar_p",
    "MatrixAudit", "PairwiseAudit", "Dimensionality",
]
