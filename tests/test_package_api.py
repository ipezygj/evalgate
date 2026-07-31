"""The package must advertise the version it is, and export the checks it ships.

Two failures that look like nothing and read like a lie. `evalgate.__version__` said
0.4.0 while pyproject shipped 0.7.0 — a user asking the package what it is got an
answer three releases stale. And three checks written this week (`selection_audit`,
`constant_baseline`, `stderr_audit`) existed only under `evalgate.leaderboard`, so
the obvious `from evalgate import selection_audit` failed for everyone who tried it.

Neither is caught by testing behaviour: every function worked perfectly.
"""
import pathlib

import pytest

import evalgate

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli")

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"

# every public check a user is told about, in the README, the site, or a write-up
PUBLIC = [
    "audit", "audit_matrix", "audit_pairwise", "latent_dimensions", "mcnemar_p",
    "selection_audit", "constant_baseline", "stderr_audit",
    "correct_best_of", "sidak", "bonferroni", "binomial_test", "bias_rate",
    "power_check", "min_detectable_effect", "base_rate_precision",
    "leave_one_out", "power_law_exponent",
]
RESULTS = ["MatrixAudit", "PairwiseAudit", "Dimensionality",
           "SelectionAudit", "ConstantBaseline", "StderrAudit"]


def _declared_version():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_matches_pyproject():
    assert evalgate.__version__ == _declared_version(), (
        f"evalgate.__version__ is {evalgate.__version__} but pyproject ships "
        f"{_declared_version()} — the package is telling users the wrong thing about itself"
    )


@pytest.mark.parametrize("name", PUBLIC + RESULTS)
def test_public_name_is_importable_from_the_package(name):
    assert hasattr(evalgate, name), f"from evalgate import {name} fails"


@pytest.mark.parametrize("name", PUBLIC + RESULTS)
def test_public_name_is_in_all(name):
    assert name in evalgate.__all__, f"{name} is importable but missing from __all__"


def test_all_names_actually_exist():
    missing = [n for n in evalgate.__all__ if not hasattr(evalgate, n)]
    assert not missing, f"__all__ promises names the package does not have: {missing}"
