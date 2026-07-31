"""Tests for evalgate.leaderboard — whole-leaderboard audits from raw per-item / pairwise data."""
import math
import random

import pytest

from evalgate import leaderboard as lb
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


def test_determinism_same_seed_same_result():
    # the package's core promise: fixed seed -> identical output
    data = {"A": set(range(120)), "B": set(range(90)), "C": set(range(70)), "D": set(range(40))}
    a1 = audit_matrix(data, n_boot=300, seed=42)
    a2 = audit_matrix(data, n_boot=300, seed=42)
    assert a1.verdict == a2.verdict
    assert a1.p_top_is_1 == a2.p_top_is_1 and a1.stay_frac == a2.stay_frac
    assert [(r.model, r.rank_lo, r.rank_hi, r.p_is_1) for r in a1.rows] == \
           [(r.model, r.rank_lo, r.rank_hi, r.p_is_1) for r in a2.rows]


def test_public_api_exports_format():
    import evalgate
    assert hasattr(evalgate, "format_matrix") and hasattr(evalgate, "audit_matrix")


def test_load_results_json_and_battles_csv(tmp_path):
    import json as _j, csv as _c
    from evalgate.datasets import load_results_json, load_battles_csv
    rp = tmp_path / "r.json"
    rp.write_text(_j.dumps({"A": list(range(20)), "B": list(range(10))}))
    r = load_results_json(str(rp))
    assert r["A"] == set(range(20))
    cp = tmp_path / "b.csv"
    with open(cp, "w", newline="") as f:
        w = _c.writer(f); w.writerow(["model_a", "model_b", "winner"])
        w.writerow(["A", "B", "model_a"]); w.writerow(["A", "B", "B"]); w.writerow(["A", "B", "tie"])
    b = load_battles_csv(str(cp))
    assert ("A", "B") in b and ("B", "A") in b and len(b) == 2   # tie dropped


def test_audit_autodispatch():
    from evalgate import audit
    from evalgate.leaderboard import MatrixAudit, PairwiseAudit
    m = audit({"A": set(range(160)), "B": set(range(90))}, n_boot=100)
    assert isinstance(m, MatrixAudit)
    p = audit([("A", "B")] * 30 + [("B", "A")] * 5, n_boot=50)
    assert isinstance(p, PairwiseAudit)
    import pytest as _p
    with _p.raises(ValueError):
        audit(12345)


def test_score_confidence_intervals():
    a = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=300)
    for r in a.rows:
        assert r.score_lo <= r.score <= r.score_hi          # CI brackets the point estimate
    # a clear leader's score CI should not overlap the runner-up's by much
    assert a.rows[0].score_lo > a.rows[1].score_lo


def test_format_ascii_strict():
    """Every formatter's output must be pure ASCII (safe on any console encoding)."""
    from evalgate.format import format_matrix, format_pairwise, format_dimensions
    from evalgate.leaderboard import audit_pairwise, latent_dimensions
    import random
    m = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=150)
    b = audit_pairwise([("A", "B")] * 30 + [("B", "A")] * 6, n_boot=60, min_pair=5)
    rng = random.Random(0)
    subs = {f"m{k}": {i for i in range(150) if rng.random() < 0.3 + 0.03 * k} for k in range(8)}
    d = latent_dimensions(subs, n_perm=12)
    for text in (format_matrix(m), format_pairwise(b), format_dimensions(d)):
        text.encode("ascii")   # raises if any non-ASCII slipped in


# --- selection_audit: the winner's curse from published numbers alone -------------

def test_selection_audit_zero_noise_selects_nothing():
    """With no measurement error there is nothing to get lucky about."""
    a = lb.selection_audit([90, 80, 70], se=0, trials=200)
    assert a.p_wrong_winner == 0.0
    assert a.score_inflation == 0.0
    assert a.gap_in_se == float("inf")


def test_selection_audit_matches_closed_form_for_two_identical_models():
    """Two models of equal ability: E[max] - truth is exactly sigma/sqrt(pi)."""
    a = lb.selection_audit([50.0, 50.0], se=5, trials=8000, seed=1)
    assert abs(a.score_inflation - 5 / math.sqrt(math.pi)) < 0.15
    assert abs(a.p_wrong_winner - 0.5) < 0.05     # a coin flip, and it should say so
    assert a.verdict.startswith("COIN-FLIP")


def test_selection_audit_matches_blom_for_twenty_identical_models():
    """Twenty tied models at sigma=5: inflation approaches sigma * E[max of 20 normals]."""
    a = lb.selection_audit([50.0] * 20, se=5, trials=8000, seed=1)
    assert abs(a.score_inflation - 9.35) < 0.4


def test_selection_audit_calls_a_clear_leader_real():
    """A leader many standard errors clear is not a selection artifact."""
    a = lb.selection_audit([80.0, 60.0, 55.0], se=1.0, trials=500, seed=0)
    assert a.p_wrong_winner == 0.0
    assert a.verdict.startswith("REAL-#1")


def test_selection_audit_inflation_grows_with_the_field():
    """The curse is about how many errors competed, not how big any one of them is."""
    small = lb.selection_audit([50.0] * 3, se=4, trials=4000, seed=2).score_inflation
    large = lb.selection_audit([50.0] * 60, se=4, trials=4000, seed=2).score_inflation
    assert large > small * 1.5


def test_selection_audit_accepts_per_model_errors_and_rejects_bad_input():
    a = lb.selection_audit([50.0, 49.0], se=[3.0, 1.0], trials=500, seed=0)
    assert 0.0 < a.p_wrong_winner < 1.0
    with pytest.raises(ValueError):
        lb.selection_audit([50.0, 49.0], se=[3.0])
    with pytest.raises(ValueError):
        lb.selection_audit([50.0], se=1.0)
    with pytest.raises(ValueError):
        lb.selection_audit([50.0, 49.0], se=-1.0)


def test_selection_audit_tie_verdict_does_not_hinge_on_sampling_noise():
    """Two identical models sit at exactly p=0.5, so a >=0.5 rule flips on the seed alone.

    The margin has to decide it: a leader inside one standard error is unresolved
    whichever side of the coin the simulation landed on.
    """
    verdicts = {lb.selection_audit([50.0, 50.0], se=5, trials=600, seed=s).verdict.split(":")[0]
                for s in range(6)}
    assert verdicts == {"COIN-FLIP-#1"}, f"verdict flipped with the seed: {verdicts}"


# --- constant_baseline: the floor a board is really read against ------------------

def test_constant_baseline_finds_the_skewed_label():
    """A key where D is right on 41 of 100 items gives a 0.41 floor, not 0.25."""
    key = ["D"] * 41 + ["A"] * 22 + ["B"] * 22 + ["C"] * 15
    a = lb.constant_baseline(key)
    assert a.answer == "D"
    assert a.score == 0.41
    assert a.chance == 0.25
    assert a.n_items == 100 and a.n_labels == 4


def test_constant_baseline_on_a_uniform_key_equals_chance():
    a = lb.constant_baseline(["A", "B", "C", "D"] * 25)
    assert a.score == a.chance == 0.25


def test_constant_baseline_counts_what_it_beats():
    key = ["D"] * 41 + ["A"] * 22 + ["B"] * 22 + ["C"] * 15
    a = lb.constant_baseline(key, scores=[0.4733, 0.30, 0.28, 0.24, 0.22, 0.21])
    assert a.beats == 5 and a.n_entries == 6
    assert a.below_chance == 3          # 0.24, 0.22, 0.21 are under 0.25
    assert a.verdict.startswith("FLOOR-DOMINATES")


def test_constant_baseline_says_clear_when_every_entry_beats_the_floor():
    key = ["D"] * 41 + ["A"] * 59
    a = lb.constant_baseline(key, scores=[0.9, 0.8, 0.7])
    assert a.beats == 0
    assert a.verdict.startswith("CLEAR")


def test_constant_baseline_accepts_non_string_labels_and_rejects_empty():
    a = lb.constant_baseline([1, 1, 1, 2], scores=[0.5])
    assert a.answer == 1 and a.score == 0.75
    with pytest.raises(ValueError):
        lb.constant_baseline([])
    with pytest.raises(ValueError):
        lb.constant_baseline(["A", "B"], scores=[])


def test_constant_baseline_tie_between_labels_is_deterministic():
    """Two labels tied at the top must not depend on dict ordering."""
    picks = {lb.constant_baseline(["B", "A", "B", "A"]).answer for _ in range(5)}
    assert picks == {"B"}, f"tie-break drifted: {picks}"
