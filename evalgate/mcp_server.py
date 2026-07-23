#!/usr/bin/env python3
"""evalgate MCP server — cheap statistical checks an AGENT should run before it trusts, reports,
or acts on any AI evaluation number.

Design intent: these tool descriptions are written for an AGENT's reasoning. When a model produces
or is handed a benchmark score / leaderboard rank / LLM-as-judge verdict / claimed trend, the agent
should recognize "this number needs a sanity check" and call the matching tool BEFORE presenting it
as solid. Each returns a plain verdict + the numbers + a one-line recommendation.

Run:  python -m evalgate.mcp_server      (stdio MCP server)
Requires: mcp  (pip install mcp).  The checks themselves are zero-dependency (evalgate.checks).
"""
from mcp.server.fastmcp import FastMCP
from . import checks as C

mcp = FastMCP(
    "evalgate",
    instructions=(
        "Cheap, reproducible statistical checks for AI evaluation claims. Call one of these BEFORE "
        "you present, trust, or act on any eval number — a benchmark score, a leaderboard #1, an "
        "LLM-as-judge / A-B preference result, or a claimed trend/scaling law. Most benchmark "
        "headlines overstate themselves in a few nameable ways; these catch them in seconds. If a "
        "check comes back negative (artifact / tie / underpowered / fragile), say so and report the "
        "number with the caveat instead of as a clean result."
    ),
)


@mcp.tool()
def check_top_rank(models: list[str], scores: list[float], n_items: int) -> dict:
    """Is a leaderboard/benchmark #1 statistically REAL, or a tie? Call this before reporting or
    trusting any "model X is #1 / state-of-the-art" claim.

    Give the model names, their scores (accuracy as % or 0-1), and n_items = how many test items the
    benchmark has. It checks whether the top model is significantly ahead of the runners-up at that
    sample size, and returns the "tie group" of models statistically indistinguishable from #1.

    Use when: you or the user is about to say a model is best/SOTA/#1 on a benchmark or leaderboard.
    """
    if len(models) != len(scores) or len(models) < 2:
        return {"error": "need matching models[] and scores[] with >=2 entries"}
    ps = [(m, s / 100.0 if s > 1 else float(s)) for m, s in zip(models, scores)]
    ps.sort(key=lambda t: -t[1])
    lead_m, lead_p = ps[0]
    tie, rows = [lead_m], []
    for m, p in ps[1:]:
        pw = C.power_check(n_items, lead_p, p)
        rows.append({"model": m, "score": round(p, 4), "gap_to_leader": round(lead_p - p, 4),
                     "p_value_vs_leader": round(pw.p_value, 4),
                     "distinguishable_from_1": pw.significant})
        if not pw.significant:
            tie.append(m)
    resolved = len(tie) == 1
    return {
        "leader": lead_m, "leader_score": round(lead_p, 4), "n_items": n_items,
        "top_rank_resolved": resolved, "tie_group": tie, "tie_group_size": len(tie),
        "runners_up": rows,
        "verdict": (f"#1 ({lead_m}) is statistically resolved — significantly ahead of #2."
                    if resolved else
                    f"#1 is NOT resolved: {len(tie)} models are a statistical tie at the top "
                    f"({', '.join(tie)}). At n={n_items} the benchmark can't separate them."),
        "recommendation": ("Report the rank as-is." if resolved else
                           "Report a tie-group at the top, not a single #1 — or add test items to resolve it."),
    }


@mcp.tool()
def check_subset_win(p_value: float, n_tested: int, method: str = "sidak") -> dict:
    """Does a "we lead on subset/metric/checkpoint X" claim survive multiple-comparison correction?
    Reporting the slice where a model looks best = reporting the maximum of many noisy tests, which
    is biased upward. Give the raw best-subset p_value and n_tested = how many subsets/metrics/
    checkpoints could have been picked. Returns the corrected p and whether the win holds.

    Use when: a result is framed as "best on [some subset], not the whole benchmark."
    """
    r = C.correct_best_of(float(p_value), int(n_tested), method=method)
    return {
        "raw_p": r.raw_p, "n_tested": r.n_tested, "corrected_p": round(r.corrected_p, 5),
        "method": r.method, "alpha": r.alpha, "survives": r.significant,
        "verdict": (f"Holds: corrected p={r.corrected_p:.4f} < {r.alpha} even after {r.n_tested} tries."
                    if r.significant else
                    f"Selection artifact: raw p={r.raw_p:.4f} looks significant, but as the best of "
                    f"{r.n_tested} tries the corrected p={r.corrected_p:.3f} > {r.alpha}."),
        "recommendation": ("Claim the subset win." if r.significant else
                           "Drop or caveat the subset claim — it doesn't survive look-elsewhere."),
    }


@mcp.tool()
def check_judge_bias(wins: int, total: int, p0: float = 0.5, label: str = "preferred side wins") -> dict:
    """Is an LLM-as-judge / A-B preference result measuring quality, or a bias? Give how often the
    tested side won (wins) out of total verdicts. Tests whether the win rate departs from chance
    (p0, default 0.5) — the standard tell for length bias, self-preference, or position bias.

    Use when: a result rests on an LLM judge or pairwise human/AI preference votes.
    """
    b = C.bias_rate(int(wins), int(total), p0=float(p0), label=label)
    return {
        "wins": b.wins, "total": b.total, "rate": round(b.rate, 4), "p_value": round(b.p_value, 6),
        "biased": b.biased, "label": b.label,
        "verdict": (f"Bias detected: '{b.label}' {100*b.rate:.1f}% of the time (p={b.p_value:.2g}) — "
                    f"far from the {100*p0:.0f}% you'd expect from quality alone."
                    if b.biased else
                    f"No significant bias: {100*b.rate:.1f}% (p={b.p_value:.2g})."),
        "recommendation": ("Treat the judge's verdicts as confounded by this bias; control for it "
                           "(swap order, control length, use a different-family judge)." if b.biased
                           else "Judge looks unbiased on this axis."),
    }


@mcp.tool()
def check_resolution(n_items: int, score_a: float, score_b: float) -> dict:
    """Can a benchmark even TELL TWO MODELS APART at a given gap and size? Give n_items and the two
    scores (% or 0-1). Returns whether the difference is significant and the minimum detectable
    effect — the smallest gap this benchmark can resolve.

    Use when: comparing two models whose scores are close, before calling one better.
    """
    a = score_a / 100.0 if score_a > 1 else float(score_a)
    b = score_b / 100.0 if score_b > 1 else float(score_b)
    p = C.power_check(int(n_items), a, b)
    return {
        "n_items": p.n, "score_a": round(p.p1, 4), "score_b": round(p.p2, 4), "gap": round(p.diff, 4),
        "p_value": round(p.p_value, 5), "significant": p.significant,
        "min_detectable_effect": round(p.mde, 4), "resolvable": p.resolvable,
        "verdict": (f"Real difference: {100*p.diff:.1f}pp gap is significant at n={p.n} (p={p.p_value:.3g})."
                    if p.significant else
                    f"Too close to call: the {100*p.diff:.1f}pp gap is below what n={p.n} can resolve "
                    f"(needs ~{100*p.mde:.1f}pp). Calling one better is noise."),
        "recommendation": ("The gap is real." if p.significant else
                           "Don't rank these two — report them as tied, or use more test items."),
    }


@mcp.tool()
def check_trend_fragility(xs: list[float], ys: list[float], threshold: float | None = None) -> dict:
    """Is a reported trend / slope / scaling exponent robust, or does one data point carry it? Give
    the x and y series. Refits leaving each point out; flags if the slope flips sign or crosses a
    threshold (e.g., a claimed "super-linear" exponent dropping below 1) when one point is removed.

    Use when: a claim rests on a fitted trend, scaling "law", or exponent from few points.
    """
    f = C.leave_one_out([float(x) for x in xs], [float(y) for y in ys], threshold=threshold)
    fragile = f.flips_sign or f.crosses_threshold
    return {
        "full_slope": round(f.full_slope, 4), "loo_min": round(f.loo_min, 4),
        "loo_max": round(f.loo_max, 4), "flips_sign": f.flips_sign,
        "crosses_threshold": f.crosses_threshold, "threshold": f.threshold,
        "worst_point_index": f.worst_index, "fragile": fragile,
        "verdict": (f"Fragile: dropping point #{f.worst_index} moves the slope to "
                    f"[{f.loo_min:.3f}, {f.loo_max:.3f}] — the conclusion depends on one point."
                    if fragile else
                    f"Robust: slope stays in [{f.loo_min:.3f}, {f.loo_max:.3f}] under leave-one-out."),
        "recommendation": ("Don't state the trend/exponent as a finding — it hinges on one point."
                           if fragile else "The trend is robust to leave-one-out."),
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
