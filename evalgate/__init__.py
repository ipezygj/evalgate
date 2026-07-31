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
    base_rate_precision,
    BaseRate,
)
from .leaderboard import (
    ConstantBaseline,
    Dimensionality,
    MatrixAudit,
    PairwiseAudit,
    SelectionAudit,
    StderrAudit,
    audit,
    audit_matrix,
    audit_pairwise,
    constant_baseline,
    latent_dimensions,
    mcnemar_p,
    selection_audit,
    stderr_audit,
)
from .format import format_matrix, format_pairwise, format_dimensions

# Describes THIS source, not whatever wheel happens to be installed — importlib.metadata
# would report the installed distribution, which in a checkout is a different codebase.
# Kept honest by test_package_api.py, which pins it to pyproject.
__version__ = "0.7.0"
__all__ = [
    "correct_best_of", "sidak", "bonferroni", "Correction",
    "bias_rate", "binomial_test", "Bias",
    "leave_one_out", "ols_slope", "power_law_exponent", "Fragility",
    "power_check", "min_detectable_effect", "Power",
    "base_rate_precision", "BaseRate",
    # whole-leaderboard audits (raw per-item / pairwise data)
    "audit", "audit_matrix", "audit_pairwise", "latent_dimensions", "mcnemar_p",
    "MatrixAudit", "PairwiseAudit", "Dimensionality",
    # checks that need only what a leaderboard PRINTS
    "selection_audit", "SelectionAudit",
    "constant_baseline", "ConstantBaseline",
    "stderr_audit", "StderrAudit",
    "format_matrix", "format_pairwise", "format_dimensions",
]
