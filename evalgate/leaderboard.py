"""evalgate.leaderboard — audit a WHOLE leaderboard from its raw per-item results.

`checks.py` works on summary numbers (you hand it scores, p-values, win counts). This module works
on the RAW data a leaderboard already publishes — which items each model solved, or the pairwise
battles — and does the real audit instead of an approximation:

  * audit_matrix(results)   — per-item pass/fail per model. Bootstraps the RANK of every model over
    the items to get a 95% rank confidence interval and P(truly #1); finds the significance group
    tied for first with a PAIRED McNemar test (the correct test — same items, look at disagreements);
    counts how many tiers the board can actually resolve; and re-tests stability by splitting the
    items in half many times. This is what `check_top_rank` approximates when all you have is scores.

  * audit_pairwise(battles) — head-to-head win/loss (arenas, A/B preference). Fits a Bradley-Terry
    ranking, bootstraps each player's rank CI + P(#1), and checks whether the preferences are
    transitive or run in rock-paper-scissors cycles (a linear ranking is only honest if transitive).

  * latent_dimensions(results) — does the board measure ONE skill or several? Compares the result
    matrix's eigenspectrum to a shuffled null; >1 significant factor means the single rank is a lossy
    projection of a multi-skilled thing.

Pure Python, no numpy/scipy. Deterministic (fixed seeds). Verdicts are computed from the numbers.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _vectors(results: Mapping) -> tuple[list, dict, list]:
    """Normalize {model: set/list of solved item-ids}  OR  {model: {item: score}} into
    (model_names, {model: [score per item]}, items)."""
    names = list(results)
    if not names:
        raise ValueError("need >=1 model")
    sample = results[names[0]]
    if isinstance(sample, Mapping):                       # {item: score}
        items = sorted({it for v in results.values() for it in v})
        vecs = {m: [float(results[m].get(it, 0.0)) for it in items] for m in names}
    else:                                                 # set/list of solved ids
        items = sorted({it for v in results.values() for it in v})
        vecs = {m: [1.0 if it in set(results[m]) else 0.0 for it in items] for m in names}
    if len(items) < 2:
        raise ValueError("need >=2 items")
    return names, vecs, items


def _ranks(score: dict, names: list) -> dict:
    return {s: 1 + sum(1 for o in names if score[o] > score[s]) for s in names}


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts (b: only-A-right, c: only-B-right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def kendall_tau(ra: dict, rb: dict, keys: Sequence) -> float:
    conc = disc = 0
    ks = list(keys)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a = ra[ks[i]] - ra[ks[j]]
            b = rb[ks[i]] - rb[ks[j]]
            if a * b > 0:
                conc += 1
            elif a * b < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


# --------------------------------------------------------------------------- #
# 1. per-item matrix audit
# --------------------------------------------------------------------------- #
@dataclass
class RankRow:
    model: str
    score: float
    rank: int
    rank_lo: int
    rank_hi: int
    p_is_1: float
    score_lo: float = 0.0     # 95% bootstrap CI on the score (matrix audits; 0 for pairwise)
    score_hi: float = 0.0


@dataclass
class MatrixAudit:
    n_models: int
    n_items: int
    leader: str
    top_score: float
    tie_group: list           # models statistically tied for #1 (McNemar, and rank-CI touches 1)
    top_resolved: bool
    p_top_is_1: float
    stay_frac: float          # fraction of split-halves where #1 stays #1
    kendall_tau: float        # whole-ordering stability across split-halves
    effective_tiers: int
    rows: list = field(default_factory=list)
    verdict: str = ""
    recommendation: str = ""
    # psychometric "why" (Rasch IRT) — None when skipped for very large boards
    reliability: float | None = None      # marginal reliability of the whole ordering (>~0.9 = trustworthy)
    frontier_info: float | None = None    # test information at the top ability (how well the board resolves the frontier)
    z_top2: float | None = None           # #1 vs #2 ability separation in sigma (|z|<2 = indistinguishable)
    winners_curse: float | None = None    # EVT: expected inflation of the displayed #1 from being the max of many


def _sig(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _irt_rasch(names, vecs, items, iters=25):
    """Compact Rasch (1-PL) fit by joint ML. Returns (reliability, info_top, info_median, z_top2).
    The mechanistic 'why': a saturated board has little test information at the frontier, so the top
    ability is unresolvable by construction even when the overall ordering is reliable."""
    S, I = len(names), len(items)

    def logit(p):
        p = min(0.98, max(0.02, p))
        return math.log(p / (1 - p))

    theta = {s: logit(sum(vecs[s]) / I) for s in names}
    b = [-logit(sum(vecs[s][i] for s in names) / S) for i in range(I)]
    for _ in range(iters):
        for s in names:
            num = den = 0.0
            th = theta[s]
            for i in range(I):
                p = _sig(th - b[i]); num += vecs[s][i] - p; den += p * (1 - p)
            if den > 1e-9:
                theta[s] = max(-6.0, min(6.0, th + num / den))
        for i in range(I):
            num = den = 0.0
            bi = b[i]
            for s in names:
                p = _sig(theta[s] - bi); num += vecs[s][i] - p; den += p * (1 - p)
            if den > 1e-9:
                b[i] = max(-6.0, min(6.0, bi - num / den))
        mb = sum(b) / I
        b = [x - mb for x in b]
    se = {}
    for s in names:
        info = sum(_sig(theta[s] - b[i]) * (1 - _sig(theta[s] - b[i])) for i in range(I))
        se[s] = (1 / math.sqrt(info)) if info > 1e-9 else 10.0
    var_t = statistics.pvariance(list(theta.values())) or 1e-9
    mean_se2 = statistics.fmean([min(se[s], 10) ** 2 for s in names])
    reliability = max(0.0, min(1.0, 1 - mean_se2 / var_t))
    ranked = sorted(names, key=lambda s: -theta[s])

    def testinfo(t):
        return sum(_sig(t - b[i]) * (1 - _sig(t - b[i])) for i in range(I))

    info_top = testinfo(theta[ranked[0]])
    info_med = testinfo(statistics.median(list(theta.values())))
    z12 = 0.0
    if len(ranked) > 1 and se[ranked[0]] < 9:
        z12 = (theta[ranked[0]] - theta[ranked[1]]) / math.sqrt(se[ranked[0]] ** 2 + se[ranked[1]] ** 2)
    return reliability, info_top, info_med, z12


def _expected_max_normal(k: int) -> float:
    """E[max of k iid standard normals], Blom approximation (deterministic, no scipy)."""
    from .checks import _probit
    k = max(2, k)
    return _probit((k - 0.375) / (k + 0.25))


def _winners_curse(scores, m):
    """EVT: how much the displayed max is inflated over its tied group's true level (the selection edge)."""
    sc = sorted(scores, reverse=True)
    top = sc[0]
    se_top = (top * (1 - top) / m) ** 0.5 if 0 < top < 1 else (statistics.pstdev(sc) / (len(sc) ** 0.5) if len(sc) > 1 else 0.0)
    group = [x for x in sc if x >= top - 2 * se_top] or [top]
    K = max(2, len(group))
    p = statistics.fmean(group)
    se = (p * (1 - p) / m) ** 0.5 if 0 < p < 1 else se_top
    return se * _expected_max_normal(K)


def _significance_group(names, vecs, items, leader):
    """Models NOT separable from the leader by a paired McNemar test (p>0.05), leader first."""
    idx = {m: i for i, m in enumerate(names)}
    lv = vecs[leader]
    group = [leader]
    for m in names:
        if m == leader:
            continue
        mv = vecs[m]
        b = sum(1 for i in range(len(items)) if lv[i] > mv[i])   # leader solved, m didn't
        c = sum(1 for i in range(len(items)) if mv[i] > lv[i])   # m solved, leader didn't
        if mcnemar_p(b, c) > 0.05:
            group.append(m)
    return group


def _effective_tiers(rows: list) -> int:
    if not rows:
        return 0
    levels = 1
    ref = rows[0]
    for r in rows[1:]:
        if r.rank_lo > ref.rank_hi:
            levels += 1
            ref = r
    return levels


def audit_matrix(results: Mapping, n_boot: int = 1000, seed: int = 0) -> MatrixAudit:
    """Audit a leaderboard from per-item results. `results` maps each model to either the set/list of
    item-ids it solved, or a {item: score} dict. Returns rank confidence intervals, the tie group at
    the top, resolvable tiers, and split-half stability — the real version of check_top_rank."""
    names, vecs, items = _vectors(results)
    m = len(items)
    obs = {s: sum(vecs[s]) / m for s in names}
    ranked = sorted(names, key=lambda s: -obs[s])
    leader = ranked[0]

    if len(names) == 1:                       # nothing to rank against
        row = RankRow(leader, round(obs[leader], 4), 1, 1, 1, 1.0)
        return MatrixAudit(1, m, leader, round(obs[leader], 4), [leader], False, 1.0, 1.0, 1.0, 1,
                           [row], "Only one submission — there is no ranking to audit.",
                           "Add at least one competitor to compare against.")

    obs_rank = _ranks(obs, names)

    rng = random.Random(seed)
    rank_samples = {s: [] for s in names}
    score_samples = {s: [] for s in names}
    is_one = {s: 0 for s in names}
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        bscore = {s: sum(vecs[s][i] for i in idx) for s in names}
        r = _ranks(bscore, names)
        is_one[min(r, key=lambda s: r[s])] += 1
        for s in names:
            rank_samples[s].append(r[s])
            score_samples[s].append(bscore[s] / m)

    def ci(v):
        v = sorted(v)
        return v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]

    rows = []
    for s in ranked:
        lo, hi = ci(rank_samples[s])
        slo, shi = ci(score_samples[s])
        rows.append(RankRow(s, round(obs[s], 4), obs_rank[s], lo, hi, round(is_one[s] / n_boot, 3),
                            round(slo, 4), round(shi, 4)))

    tie = _significance_group(names, vecs, items, leader)
    # a model is in the displayed tie group if McNemar-tied AND its rank CI touches 1
    ci_touch = {r.model for r in rows if r.rank_lo == 1}
    tie = [leader] + [m for m in tie if m != leader and m in ci_touch]

    # split-half stability
    K = 40
    taus, stay = [], 0
    for k in range(K):
        o = items[:]
        random.Random(seed + 100 + k).shuffle(o)
        h = m // 2
        A = set(o[:h])
        sa = {s: sum(v for it, v in zip(items, vecs[s]) if it in A) for s in names}
        sb = {s: sum(v for it, v in zip(items, vecs[s]) if it not in A) for s in names}
        taus.append(kendall_tau(_ranks(sa, names), _ranks(sb, names), names))
        if max(names, key=lambda s: sa[s]) == max(names, key=lambda s: sb[s]):
            stay += 1

    tiers = _effective_tiers(rows)
    p1 = rows[0].p_is_1

    # psychometric "why" — Rasch IRT (skip on very large boards to keep it fast) + winner's-curse
    reliability = frontier_info = z_top2 = winners_curse = None
    if all(v in (0.0, 1.0) for s in names for v in vecs[s]):   # IRT defined on binary pass/fail
        if S := len(names):
            if S * m <= 120_000:
                reliability, frontier_info, _info_med, z_top2 = _irt_rasch(names, vecs, items)
                reliability = round(reliability, 3); frontier_info = round(frontier_info, 2)
                z_top2 = round(z_top2, 2)
        winners_curse = round(_winners_curse([obs[s] for s in names], m), 4)

    resolved = len(tie) == 1 and p1 >= 0.85 and stay >= K - 2
    if resolved:
        verdict = (f"#1 ({leader}) is a REAL, resolved champion — significantly ahead of #2, "
                   f"and it stays #1 on {round(stay / K * 100)}% of random item splits.")
        rec = "Report the #1 as-is; publishing the rank confidence interval keeps it honest."
    elif len(tie) >= 2:
        verdict = (f"#1 is NOT resolved: {len(tie)} models are a statistical tie for first "
                   f"({', '.join(tie)}). The printed #1 is truly first in only {round(p1 * 100)}% of "
                   f"resamples and the title changes hands on {round((1 - stay / K) * 100)}% of item splits.")
        if z_top2 is not None and abs(z_top2) < 2:
            verdict += (f" IRT confirms it: #1 and #2 are {abs(z_top2):.2f} sigma apart in ability"
                        + (f", and the frontier carries little test information — the board has run out of items "
                           f"hard enough to separate the top." if reliability and reliability > 0.9 else "") + ".")
        rec = "Report the significance group tied for #1, or a rank confidence interval — not a lone #1."
    else:
        verdict = (f"#1 ({leader}) is only partly resolved (P(#1)={p1:.2f}, stays #1 on "
                   f"{round(stay / K * 100)}% of splits).")
        rec = "Show the rank confidence interval; treat the exact top order with care."

    return MatrixAudit(len(names), m, leader, round(obs[leader], 4), tie, resolved, p1,
                       round(stay / K, 3), round(statistics.fmean(taus), 3), tiers, rows, verdict, rec,
                       reliability, frontier_info, z_top2, winners_curse)


# --------------------------------------------------------------------------- #
# 2. pairwise (Bradley-Terry) + Condorcet
# --------------------------------------------------------------------------- #
def _bt_strengths(battles, players, idx, iters=60):
    w = [1.0] * len(players)
    wins = [0.0] * len(players)
    pairs: dict = {}
    for a, b in battles:
        wins[idx[a]] += 1
        pairs[(idx[a], idx[b])] = pairs.get((idx[a], idx[b]), 0) + 1
    for _ in range(iters):
        den = [0.0] * len(players)
        for (i, j), c in pairs.items():
            den[i] += c / (w[i] + w[j])
            den[j] += c / (w[i] + w[j])
        nw = [(wins[i] / den[i]) if den[i] > 0 else w[i] for i in range(len(players))]
        g = statistics.fmean([x for x in nw if x > 0]) or 1.0
        w = [x / g for x in nw]
    return w


@dataclass
class PairwiseAudit:
    n_battles: int
    leader: str
    tie_group: list
    top_resolved: bool
    p_top_is_1: float
    intransitivity_pct: float
    null_intransitivity_pct: float
    transitive: bool
    rows: list = field(default_factory=list)
    verdict: str = ""
    recommendation: str = ""


def audit_pairwise(battles: Sequence, n_boot: int = 200, seed: int = 0,
                   min_pair: int = 20) -> PairwiseAudit:
    """Audit a head-to-head board (arena / A-B preference). `battles` is a sequence of
    (winner, loser) pairs. Fits Bradley-Terry, bootstraps each player's rank CI + P(#1), and tests
    whether preferences are transitive or cyclic (a linear ranking is only honest if transitive)."""
    battles = [tuple(x) for x in battles]
    players = sorted({p for w, l in battles for p in (w, l)})
    if len(players) < 2:
        raise ValueError("need >=2 players")
    idx = {p: i for i, p in enumerate(players)}
    obs_w = _bt_strengths(battles, players, idx)
    obs_rank = {players[i]: 1 + sum(1 for j in range(len(players)) if obs_w[j] > obs_w[i])
                for i in range(len(players))}
    rng = random.Random(seed)
    rank_s = {p: [] for p in players}
    is_one = {p: 0 for p in players}
    n = len(battles)
    for _ in range(n_boot):
        samp = [battles[rng.randrange(n)] for _ in range(n)]
        w = _bt_strengths(samp, players, idx)
        r = {players[i]: 1 + sum(1 for j in range(len(players)) if w[j] > w[i]) for i in range(len(players))}
        is_one[min(r, key=lambda p: r[p])] += 1
        for p in players:
            rank_s[p].append(r[p])
    ranked = sorted(players, key=lambda p: obs_rank[p])
    rows = []
    for p in ranked:
        v = sorted(rank_s[p])
        lo, hi = v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]
        rows.append(RankRow(p, 0.0, obs_rank[p], lo, hi, round(is_one[p] / n_boot, 3)))
    tie = [r.model for r in rows if r.rank_lo == 1]
    leader = rows[0].model
    p1 = rows[0].p_is_1

    # Condorcet intransitivity vs a Bradley-Terry (transitive) null
    import collections
    import itertools
    wins = collections.Counter()
    games = collections.Counter()
    for a, b in battles:
        wins[(a, b)] += 1
        games[frozenset((a, b))] += 1

    def beats(a, b, W):
        g = games[frozenset((a, b))]
        if g < min_pair:
            return None
        return W[(a, b)] > g / 2

    def cyc_frac(W):
        tot = cyc = 0
        for a, b, c in itertools.combinations(players, 3):
            ab, bc, ca = beats(a, b, W), beats(b, c, W), beats(c, a, W)
            if ab is None or bc is None or ca is None:
                continue
            tot += 1
            if (ab and bc and ca) or (not ab and not bc and not ca):
                cyc += 1
        return (cyc / tot) if tot else 0.0

    obs_frac = cyc_frac(wins)
    strg = {players[i]: obs_w[i] for i in range(len(players))}
    rng2 = random.Random(seed + 7)
    null = []
    for _ in range(20):
        Wn = collections.Counter()
        for a, b in battles:
            pa = strg[a] / (strg[a] + strg[b])
            Wn[(a, b) if rng2.random() < pa else (b, a)] += 1
        null.append(cyc_frac(Wn))
    null_hi = sorted(null)[int(0.95 * len(null))] if null else 0.0
    transitive = obs_frac <= null_hi

    resolved = len(tie) == 1 and p1 >= 0.85
    if resolved:
        verdict = (f"#1 ({leader}) is a REAL champion — first in {round(p1 * 100)}% of bootstrap "
                   f"resamples over {n:,} comparisons.")
        rec = "Report the #1; it is backed by enough comparisons to resolve."
    else:
        verdict = (f"#1 is NOT resolved: {len(tie)} players tie for first; the printed #1 holds rank 1 "
                   f"in only {round(p1 * 100)}% of resamples over {n:,} comparisons.")
        rec = "Report a tie-group, or gather more comparisons to resolve the top."
    if not transitive:
        verdict += f" Preferences are INTRANSITIVE ({obs_frac*100:.1f}% cyclic triples > null {null_hi*100:.1f}%) — a single line loses real structure."

    return PairwiseAudit(n, leader, tie, resolved, p1, round(obs_frac * 100, 2),
                         round(null_hi * 100, 2), transitive, rows, verdict, rec)


# --------------------------------------------------------------------------- #
# 3. latent dimensionality (does the board measure >1 skill?)
# --------------------------------------------------------------------------- #
def _top_eig(G, iters=90, seed=1):
    n = len(G)
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(n)]
    for _ in range(iters):
        w = [sum(G[i][j] * v[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(x * x for x in w)) or 1.0
        v = [x / nrm for x in w]
    Gv = [sum(G[i][j] * v[j] for j in range(n)) for i in range(n)]
    lam = sum(v[i] * Gv[i] for i in range(n))
    return lam, v


@dataclass
class Dimensionality:
    n_significant: int
    eigenvalues: list
    null_edge: float
    top1_fraction: float
    verdict: str
    recommendation: str


def latent_dimensions(results: Mapping, n_perm: int = 30, topk: int = 5, seed: int = 0) -> Dimensionality:
    """How many distinct skills does the board measure? Compares the result matrix's eigenspectrum to
    a column-shuffled null; >1 factor above the null means a single ranking is a lossy projection."""
    names, vecs, items = _vectors(results)
    S, I = len(names), len(items)
    if S < 4:
        raise ValueError("need >=4 models for a dimensionality estimate")
    col_mean = [sum(vecs[s][i] for s in names) / S for i in range(I)]
    R = [[vecs[s][i] - col_mean[i] for i in range(I)] for s in names]

    def gram(Rm):
        return [[sum(Rm[a][k] * Rm[b][k] for k in range(I)) / I for b in range(S)] for a in range(S)]

    G = gram(R)
    eigs = []
    for t in range(min(topk, S)):
        lam, v = _top_eig(G, seed=seed + t)
        eigs.append(lam)
        for i in range(S):
            for j in range(S):
                G[i][j] -= lam * v[i] * v[j]
    rng = random.Random(seed + 777)
    null_top = []
    for _ in range(n_perm):
        Rp = [row[:] for row in R]
        for k in range(I):
            col = [Rp[a][k] for a in range(S)]
            rng.shuffle(col)
            for a in range(S):
                Rp[a][k] = col[a]
        lam, _ = _top_eig(gram(Rp), iters=60, seed=rng.randrange(10 ** 6))
        null_top.append(lam)
    null_top.sort()
    edge = null_top[int(0.95 * len(null_top))]
    n_sig = sum(1 for e in eigs if e > edge)
    frac = eigs[0] / sum(eigs) if sum(eigs) else 0.0
    if n_sig <= 1:
        verdict = f"ONE dominant skill — a single ranking is appropriate (top factor explains {frac*100:.0f}%)."
        rec = "A scalar leaderboard is a fair summary here."
    else:
        verdict = (f"{n_sig} distinct skills — the single ranking is a lossy projection; two models with "
                   f"the same score can be strong on different parts of the benchmark.")
        rec = "Publish sub-scores or a 2-D skill plot alongside the single rank."
    return Dimensionality(n_sig, [round(e, 3) for e in eigs], round(edge, 3), round(frac, 3), verdict, rec)


# --------------------------------------------------------------------------- #
# convenience: one entry point, auto-detecting the data shape
# --------------------------------------------------------------------------- #
def audit(data, n_boot: int = 1000, seed: int = 0):
    """Auto-detect the data shape and run the right audit:
      * a mapping {model: solved-items | {item: score}}  -> audit_matrix (per-item leaderboard)
      * a sequence of (winner, loser) pairs               -> audit_pairwise (head-to-head board)
    Returns a MatrixAudit or a PairwiseAudit."""
    if isinstance(data, Mapping):
        return audit_matrix(data, n_boot=n_boot, seed=seed)
    try:
        seq = list(data)
    except TypeError:
        raise ValueError("data must be {model: items} or a sequence of (winner, loser) pairs")
    if seq and isinstance(seq[0], (list, tuple)) and len(seq[0]) == 2:
        return audit_pairwise(seq, n_boot=n_boot, seed=seed)
    raise ValueError("data must be {model: items} or a sequence of (winner, loser) pairs")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = random.Random(0)
    items = list(range(200))
    # clear leader -> resolved
    strong = {"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}
    a = audit_matrix(strong, n_boot=300)
    assert a.top_resolved and a.leader == "A" and len(a.tie_group) == 1, a.verdict
    # three coin-flip models -> tie
    tie = {n: set(i for i in items if rng.random() < 0.5) for n in ("X", "Y", "Z")}
    t = audit_matrix(tie, n_boot=300)
    assert len(t.tie_group) >= 2 and not t.top_resolved, t.verdict
    # pairwise: A>B>C transitive, A resolved
    battles = ([("A", "B")] * 40 + [("A", "C")] * 45 + [("B", "C")] * 40
               + [("B", "A")] * 8 + [("C", "A")] * 5 + [("C", "B")] * 10)
    p = audit_pairwise(battles, n_boot=100, min_pair=10)
    assert p.leader == "A" and p.transitive, p.verdict
    # dimensionality: 1-D data -> 1 factor
    truth = {f"m{k}": (k - 5) * 0.6 for k in range(11)}
    b_true = [rng.gauss(0, 1.5) for _ in items]
    subs = {n: set(i for i in items if rng.random() < 1 / (1 + math.exp(-(truth[n] - b_true[i])))) for n in truth}
    d = latent_dimensions(subs, n_perm=20)
    assert d.n_significant == 1, d.verdict
    print("evalgate.leaderboard selftest: OK (resolved + tie + pairwise-transitive + 1-D)")


if __name__ == "__main__":
    _selftest()
