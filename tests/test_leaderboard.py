"""Tests for evalgate.leaderboard — whole-leaderboard audits from raw per-item / pairwise data."""
import math
import random

import pytest

from evalgate.leaderboard import (
    audit_matrix,
    audit_pairwise,
    latent_dimensions,
    mcnemar_p,
)


def test_matrix_resolved_leader():
    a = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=300)
    assert a.leader == "A"
    assert a.top_resolved
    assert a.tie_group == ["A"]
    assert a.p_top_is_1 >= 0.85
    assert a.effective_tiers >= 2


def test_matrix_coinflip_tie():
    rng = random.Random(0)
    items = range(200)
    tie = {n: {i for i in items if rng.random() < 0.5} for n in ("X", "Y", "Z")}
    a = audit_matrix(tie, n_boot=300)
    assert not a.top_resolved
    assert len(a.tie_group) >= 2


def test_matrix_accepts_score_maps():
    res = {"P": {i: 1.0 for i in range(50)}, "Q": {i: (1.0 if i < 20 else 0.0) for i in range(50)}}
    a = audit_matrix(res, n_boot=200)
    assert a.leader == "P"


def test_pairwise_transitive_resolved():
    battles = ([("A", "B")] * 40 + [("A", "C")] * 45 + [("B", "C")] * 40
               + [("B", "A")] * 8 + [("C", "A")] * 5 + [("C", "B")] * 10)
    a = audit_pairwise(battles, n_boot=100, min_pair=10)
    assert a.leader == "A"
    assert a.transitive


def test_pairwise_rock_paper_scissors_intransitive():
    rng = random.Random(0)
    mods = [f"C{k}" for k in range(5)]
    battles = []
    for _ in range(400):
        for k in range(5):
            for step in (1, 2):
                x, y = mods[k], mods[(k + step) % 5]
                battles.append((x, y) if rng.random() < 0.85 else (y, x))
    a = audit_pairwise(battles, n_boot=60, min_pair=10)
    assert not a.transitive


def test_dimensionality_one_factor():
    rng = random.Random(0)
    items = list(range(200))
    truth = {f"m{k}": (k - 5) * 0.6 for k in range(11)}
    b_true = [rng.gauss(0, 1.5) for _ in items]
    subs = {n: {i for i in items if rng.random() < 1 / (1 + math.exp(-(truth[n] - b_true[i])))}
            for n in truth}
    d = latent_dimensions(subs, n_perm=20)
    assert d.n_significant == 1


def test_mcnemar_symmetry():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(10, 0) < 0.01
    assert mcnemar_p(5, 5) > 0.5


def test_psychometrics_populated_and_discriminating():
    # clear leader: high reliability, well-separated top two, low winner's-curse
    a = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=200)
    assert a.reliability is not None and a.reliability > 0.9
    assert a.z_top2 is not None and abs(a.z_top2) > 2          # distinguishable
    assert a.winners_curse is not None
    # coin-flip tie: top two indistinguishable in ability
    import random
    rng = random.Random(0)
    tie = {n: {i for i in range(200) if rng.random() < 0.5} for n in ("X", "Y", "Z")}
    t = audit_matrix(tie, n_boot=200)
    assert t.z_top2 is not None and abs(t.z_top2) < 2          # indistinguishable


def test_datasets_split_validation():
    from evalgate.datasets import load_swebench, SWEBENCH_SPLITS
    assert "lite" in SWEBENCH_SPLITS and "test" in SWEBENCH_SPLITS
    with pytest.raises(ValueError):
        load_swebench("not-a-split")


def test_format_renders_ascii_only():
    from evalgate.format import format_matrix
    a = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=150)
    s = format_matrix(a, "Demo")
    assert "REAL #1" in s and "rank-CI" in s
    s.encode("cp1252")   # must not raise — ASCII-safe on Windows consoles


def test_single_model_graceful():
    a = audit_matrix({"solo": set(range(30))}, n_boot=50)
    assert a.n_models == 1 and a.tie_group == ["solo"]
    assert "one submission" in a.verdict.lower()


def test_identical_models_all_tie():
    same = {n: set(range(50)) for n in ("A", "B", "C")}
    a = audit_matrix(same, n_boot=150)
    assert not a.top_resolved
    assert len(a.tie_group) == 3          # perfectly tied -> all in the group


def test_minimum_two_items():
    a = audit_matrix({"A": {0, 1}, "B": {0}}, n_boot=100)
    assert a.leader == "A" and a.n_items == 2


def test_errors_are_friendly():
    import pytest as _p
    with _p.raises(ValueError):
        audit_matrix({"A": set()}, n_boot=10)        # <2 items
    with _p.raises(ValueError):
        audit_pairwise([("A", "A")][:0])              # no battles / <2 players
