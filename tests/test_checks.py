"""Tests — also serve as executable documentation of the three case studies."""
import math

from evalgate import (
    bias_rate,
    binomial_test,
    bonferroni,
    correct_best_of,
    leave_one_out,
    ols_slope,
    power_law_exponent,
    sidak,
)


def test_look_elsewhere_rewardbench():
    c = correct_best_of(0.009, 23, "sidak")
    assert 0.18 < c.corrected_p < 0.20
    assert not c.significant
    # a robust pair survives family-wise correction
    assert correct_best_of(0.011 / 23, 23, "sidak").significant


def test_sidak_bonferroni_edges():
    assert sidak(0.0, 10) == 0.0
    assert sidak(1.0, 10) == 1.0
    assert bonferroni(0.02, 100) == 1.0  # clamped


def test_judge_bias_mtbench():
    b = bias_rate(68, 100, label="longer answer wins")
    assert b.biased and b.p_value < 0.001
    assert not bias_rate(50, 100).biased


def test_binomial_matches_known():
    # symmetric case is ~1.0; all-heads is 2 * 0.5**n
    assert abs(binomial_test(5, 10) - 1.0) < 1e-9
    assert abs(binomial_test(10, 10) - 2 * 0.5 ** 10) < 1e-9


def test_binomial_large_n_normal_fallback():
    p = binomial_test(5300, 10000)  # 53% over 10k -> clearly significant
    assert 0.0 <= p < 1e-6


def test_fragility_grokking():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.1, 2.0, 3.2, 3.9, 5.1, 9.0]
    f = leave_one_out(xs, ys, threshold=1.0)
    assert f.crosses_threshold
    assert f.worst_index == 5


def test_fits():
    s, b, r2 = ols_slope([0, 1, 2, 3], [1, 3, 5, 7])
    assert abs(s - 2.0) < 1e-9 and abs(b - 1.0) < 1e-9 and abs(r2 - 1.0) < 1e-9
    a, r2 = power_law_exponent([1, 2, 4, 8], [1, 4, 16, 64])
    assert abs(a - 2.0) < 1e-9 and abs(r2 - 1.0) < 1e-9
