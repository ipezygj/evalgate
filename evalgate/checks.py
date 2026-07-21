"""evalgate.checks — small, dependency-free statistical checks for eval claims.

Three public checks, one per common way a benchmark headline overstates itself:

  1. correct_best_of  — a "we lead on subset X" win, corrected for how many
     subsets it could have been picked from (look-elsewhere / family-wise error).
  2. bias_rate        — a judge/metric that "prefers" one side: is the winner
     winning, or just longer / first / same-family? (exact binomial test).
  3. leave_one_out    — does one data point flip the slope of your fit? (fragility).

Plus small OLS / power-law helpers the checks build on. Pure Python, no numpy/scipy,
so it installs and runs anywhere. Textbook statistics — nothing proprietary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


# --------------------------------------------------------------------------- #
# 1. Multiple comparisons / look-elsewhere
# --------------------------------------------------------------------------- #
def sidak(p: float, n: int) -> float:
    """Sidak-corrected p-value for the best of `n` independent tests."""
    if not 0.0 <= p <= 1.0 or n < 1:
        raise ValueError("need 0<=p<=1 and n>=1")
    return 1.0 - (1.0 - p) ** n


def bonferroni(p: float, n: int) -> float:
    """Bonferroni-corrected p-value for the best of `n` tests (conservative)."""
    if not 0.0 <= p <= 1.0 or n < 1:
        raise ValueError("need 0<=p<=1 and n>=1")
    return min(1.0, p * n)


@dataclass
class Correction:
    raw_p: float
    n_tested: int
    corrected_p: float
    method: str
    significant: bool          # corrected_p < alpha
    alpha: float

    def __str__(self) -> str:
        verdict = "SURVIVES" if self.significant else "does NOT survive"
        return (f"raw p={self.raw_p:.4g} over {self.n_tested} tests -> "
                f"{self.method} p={self.corrected_p:.4g} "
                f"({verdict} correction at alpha={self.alpha:g})")


def correct_best_of(p: float, n_tested: int, method: str = "sidak",
                    alpha: float = 0.05) -> Correction:
    """Correct a best-of-`n_tested` p-value for the family it was chosen from.

    Report the subset/metric/checkpoint where a model looks best and you are
    reporting the maximum of many noisy tests; this asks whether that win
    survives once you account for how many you could have picked.
    """
    corrected = {"sidak": sidak, "bonferroni": bonferroni}[method](p, n_tested)
    return Correction(p, n_tested, corrected, method, corrected < alpha, alpha)


# --------------------------------------------------------------------------- #
# 2. Judge / metric bias — exact binomial
# --------------------------------------------------------------------------- #
def binomial_test(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial p-value for `k` successes in `n` trials vs p0.

    Exact for modest n; falls back to a normal approximation for large n so it
    stays fast. Two-sided = summing outcomes no more likely than the observed.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError("need n>=1 and 0<=k<=n")
    if n <= 2000:
        pk = math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
        tol = pk * (1 + 1e-9)
        total = 0.0
        for i in range(n + 1):
            pi = math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
            if pi <= tol:
                total += pi
        return min(1.0, total)
    # normal approximation with continuity correction
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    z = (abs(k - mu) - 0.5) / sd
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


@dataclass
class Bias:
    wins: int
    total: int
    rate: float
    p_value: float
    biased: bool
    label: str

    def __str__(self) -> str:
        verdict = "BIAS" if self.biased else "no significant bias"
        return (f"{self.label}: {self.wins}/{self.total} = {self.rate:.1%} "
                f"(p={self.p_value:.4g}) -> {verdict}")


def bias_rate(wins: int, total: int, p0: float = 0.5, alpha: float = 0.05,
              label: str = "preferred side wins") -> Bias:
    """Test whether a side wins at a rate different from chance (`p0`).

    Feed it the count of pairwise verdicts where the longer answer won (verbosity
    bias), the first-listed answer won (position bias), or the judge's own family
    won (self-preference). A rate far from 50% at tiny p is bias wearing the label
    of a result.
    """
    rate = wins / total
    p = binomial_test(wins, total, p0)
    return Bias(wins, total, rate, p, p < alpha, label)


# --------------------------------------------------------------------------- #
# 3. Fits + leave-one-out fragility
# --------------------------------------------------------------------------- #
def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Ordinary least squares: returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise ValueError("need >=2 paired points")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("x has zero variance")
    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 0.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, intercept, r2


def power_law_exponent(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Fit y = c * x**a on log-log axes; returns (exponent a, r_squared)."""
    if any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        raise ValueError("power-law fit needs strictly positive x and y")
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    a, _, r2 = ols_slope(lx, ly)
    return a, r2


@dataclass
class Fragility:
    full_slope: float
    loo_min: float
    loo_max: float
    flips_sign: bool
    crosses_threshold: bool
    threshold: float | None
    worst_index: int

    def __str__(self) -> str:
        msg = (f"slope={self.full_slope:.4g}, "
               f"leave-one-out range [{self.loo_min:.4g}, {self.loo_max:.4g}]")
        flags = []
        if self.flips_sign:
            flags.append("SIGN FLIPS")
        if self.crosses_threshold:
            flags.append(f"CROSSES {self.threshold:g}")
        msg += " -> " + (", ".join(flags) if flags else "robust to leave-one-out")
        msg += f" (most influential point: index {self.worst_index})"
        return msg


def leave_one_out(xs: Sequence[float], ys: Sequence[float],
                  fit: Callable[[Sequence[float], Sequence[float]], tuple] = ols_slope,
                  threshold: float | None = None) -> Fragility:
    """Refit after dropping each point; flag if the conclusion hangs on one row.

    `fit` returns a tuple whose first element is the statistic of interest
    (default: OLS slope; pass `power_law_exponent` for a log-log exponent).
    `threshold` (e.g. 1.0 for "super-linear") flags when dropping a point moves
    the statistic across it.
    """
    n = len(xs)
    if n < 3:
        raise ValueError("need >=3 points to leave one out")
    full = fit(xs, ys)[0]
    loo = []
    for i in range(n):
        sx = xs[:i] + xs[i + 1:]
        sy = ys[:i] + ys[i + 1:]
        loo.append(fit(list(sx), list(sy))[0])
    lo, hi = min(loo), max(loo)
    flips_sign = any((v > 0) != (full > 0) for v in loo)
    crosses = (threshold is not None
               and any((v >= threshold) != (full >= threshold) for v in loo))
    worst = max(range(n), key=lambda i: abs(loo[i] - full))
    return Fragility(full, lo, hi, flips_sign, crosses, threshold, worst)


# --------------------------------------------------------------------------- #
# self-test: reproduces the three public case studies
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # 1. RewardBench: best-of-23 subset win does not survive Sidak
    c = correct_best_of(0.009, 23, "sidak")
    assert 0.18 < c.corrected_p < 0.20 and not c.significant, c
    # a genuinely robust pair DOES survive
    assert correct_best_of(0.011 / 23, 23, "sidak").significant is True

    # 2. MT-Bench: longer answer wins 68/100 -> significant verbosity bias
    b = bias_rate(68, 100, label="longer answer wins")
    assert b.biased and b.p_value < 0.001, b
    # a coin-flip judge shows no bias
    assert not bias_rate(50, 100).biased

    # 3. Grokking: a slope reported >1 that a single point pushes below 1
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [1.1, 2.0, 3.2, 3.9, 5.1, 9.0]   # last point is high-leverage
    f = leave_one_out(xs, ys, threshold=1.0)
    assert f.crosses_threshold and f.worst_index == 5, f

    # OLS / power-law sanity
    s, _, r2 = ols_slope([0, 1, 2, 3], [1, 3, 5, 7])
    assert abs(s - 2.0) < 1e-9 and abs(r2 - 1.0) < 1e-9
    a, _ = power_law_exponent([1, 2, 4, 8], [1, 4, 16, 64])
    assert abs(a - 2.0) < 1e-9
    print("evalgate selftest: OK (reproduced all 3 case studies)")


if __name__ == "__main__":
    _selftest()
