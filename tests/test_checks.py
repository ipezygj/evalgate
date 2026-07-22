"""Tests — also serve as executable documentation of the three case studies."""
import math

from evalgate import (
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


def test_power_underpowered_vs_resolved():
    # a 2pp gap over 200 items is below what the sample can resolve
    u = power_check(200, 0.85, 0.83)
    assert not u.significant and not u.resolvable
    assert 0.09 < u.mde < 0.11
    # a 5pp gap over 2000 items resolves and is significant
    r = power_check(2000, 0.85, 0.80)
    assert r.significant and r.resolvable
    assert r.mde < u.mde  # more data -> smaller detectable effect


def test_min_detectable_effect_shrinks_with_n():
    assert min_detectable_effect(100) > min_detectable_effect(1000)
    # MDE = (z_alpha/2 + z_power) * sqrt(2 p(1-p)/n); at n=1, p=.5 that is z_sum*sqrt(.5)
    expected = (1.959964 + 0.841621) * math.sqrt(0.5)
    assert abs(min_detectable_effect(1, 0.5) - expected) < 1e-3


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
