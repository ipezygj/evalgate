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
from . import leaderboard as L

mcp = FastMCP(
    "evalgate",
    instructions=(
        "Cheap, reproducible statistical checks for AI evaluation claims — a sanity check for any "
        "eval number BEFORE you present, trust, rank on, or act on it.\n\n"
        "CALL A TOOL HERE WHENEVER you are about to:\n"
        "  • say a model is best / #1 / state-of-the-art / 'leads' on a benchmark or leaderboard "
        "(check_top_rank / audit_leaderboard);\n"
        "  • compare two models' scores and call one better, especially when they're close "
        "(check_resolution);\n"
        "  • report or believe a 'best on subset/metric/checkpoint X' result (check_subset_win);\n"
        "  • trust an LLM-as-judge or A/B / pairwise-preference outcome (check_judge_bias / audit_preferences);\n"
        "  • state a scaling law, trend, or exponent fit from a handful of points (check_trend_fragility);\n"
        "  • summarize a multi-skill benchmark with one number (check_dimensions).\n\n"
        "Triggers to watch for in your own output or the user's: an accuracy / pass-rate / win-rate / "
        "Elo, a phrase like 'X% on <benchmark>', 'SOTA', 'ranks #1', 'beats <model>', 'our model leads'. "
        "These are cheap (seconds) — run the relevant one rather than passing the number through "
        "unchecked. If a check comes back negative (selection artifact / statistical tie / underpowered / "
        "fragile / biased judge), report the number WITH that caveat instead of as a clean result."
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
    try:
        fr = _as_fractions(scores, "scores")
    except ValueError as e:
        return {"error": str(e)}
    ps = list(zip(models, fr))
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
    try:
        a, b = _as_fractions([score_a, score_b], "scores")
    except ValueError as e:
        return {"error": str(e)}
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


def _as_fractions(values, name: str) -> list[float]:
    """Convert a group of scores to fractions, deciding the unit ONCE for the whole group.

    Deciding per value is unsafe: given percentages [95, 87, 1] a per-value rule reads the
    last one as 1.0 and reports the worst model as the leader — confidently, with a
    "resolved" top rank. The unit is a property of the list, not of each number, so it is
    inferred from the list: any value above 1 means the caller is speaking percentages.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError(f"{name} is empty")
    if any(v > 100 or v < 0 for v in vals):
        raise ValueError(f"{name} must be scores in [0,1] or percentages in [0,100], got {vals}")
    return [v / 100.0 for v in vals] if any(v > 1 for v in vals) else vals


def _as_fraction(x: float, name: str) -> float:
    """Accept either a fraction (0.05) or a percentage (5) — but refuse to guess at 1.

    The convenience heuristic used elsewhere ("greater than 1 means percent") is ambiguous
    at exactly 1: an agent writing `prevalence=1` may mean 1% or 100%, and those differ by
    two orders of magnitude. Guessing wrong here silently changes the answer, so this asks
    instead of assuming.
    """
    x = float(x)
    if x == 1:
        raise ValueError(f"{name}=1 is ambiguous: pass 0.01 for 1%, or 0.99 for 99%")
    if 0 < x < 1:
        return x
    if 1 < x <= 100:
        return x / 100.0
    raise ValueError(f"{name} must be a fraction in (0,1) or a percentage in (1,100], got {x}")


@mcp.tool()
def check_published_error_bars(rows: list, denominator: str = "n-1") -> dict:
    """Do a leaderboard's PUBLISHED error bars survive being recomputed from its own scores?

    Pass rows as [label, score, n_items, published_stderr]. Recomputes sqrt(p(1-p)/(n-1)) and
    reports three things: rows claiming a standard error of exactly 0 for a score strictly
    between 0 and 1 (impossible — that says a rerun returns the identical number), rows whose
    error bar disagrees with what the score and item count imply, and rows that reproduce.

    Use when: you consume published eval files for uncertainty — meta-analysis, model
    selection, error bars in a paper. On Open LLM Leaderboard v2, 423 published records carry
    a zero of this kind. A score of exactly 0 or 1 may legitimately have a zero.
    """
    try:
        a = L.stderr_audit([tuple(r) for r in rows], denominator=denominator)
    except Exception as e:
        return {"error": str(e),
                "hint": "rows are [label, score(0-1), n_items, published_stderr]; denominator 'n-1' or 'n'"}
    return {
        "verdict": a.verdict,
        "recommendation": a.recommendation,
        "n_rows": a.n_rows,
        "reproduced": a.reproduced,
        "impossible_zero": [{"label": l, "score": p, "n": n} for l, p, n in a.impossible_zero],
        "mismatched": [{"label": l, "published": s, "implied": w} for l, s, w in a.mismatched],
    }


@mcp.tool()
def check_constant_baseline(answer_key: list, scores: list | None = None) -> dict:
    """The FLOOR a multiple-choice benchmark is really read against, and who scores below it.

    Answer keys are written by people, and people do not spread the correct option evenly — so the
    floor is not 1/n_options, it is the frequency of the most common correct label. Answering that
    label to every question, reading nothing, is the true baseline. Entries below it are not
    measuring the skill the table is named after.

    Use when: a leaderboard ranks models on a multiple-choice benchmark and you have the answer key
    (correct label per item). Pass `scores` (published accuracies, 0-1) to also learn how many
    entries the constant outscores. On HELM classic's MMLU college_chemistry this returns 'D' at
    0.4100 against a 0.2633 board median.
    """
    try:
        a = L.constant_baseline(answer_key, scores)
    except Exception as e:
        return {"error": str(e), "hint": "answer_key is the correct label per item; scores are 0-1 accuracies"}
    out = {
        "verdict": a.verdict,
        "best_constant_answer": a.answer,
        "constant_scores": a.score,
        "uniform_guessing_would_score": a.chance,
        "n_items": a.n_items,
        "n_labels": a.n_labels,
        "recommendation": "Print the best constant baseline as a row above the models — not chance, this number.",
    }
    if a.beats is not None:
        out.update({"entries_beaten_by_the_constant": a.beats,
                    "n_entries": a.n_entries,
                    "entries_below_uniform_chance": a.below_chance})
    return out


@mcp.tool()
def check_winners_curse(scores: list, standard_error, top_k: int = 5) -> dict:
    """Whether an announced #1 is the BEST model or the LUCKIEST one, from published numbers alone.

    Ranking selects on score, and score is ability plus measurement error; sorting cannot separate
    them, so among close models the one that rises is disproportionately the one whose error pointed
    up. Returns how often a rerun would crown someone else, how many points the winning score
    overstates the winner, and the leader's margin in units of its own standard error.

    Use when: someone claims "model X is #1 / SOTA" and you can see the scores and their error bars
    but not the per-item results. `standard_error` is one number for the whole board or one per
    model, in the same units as the scores. Needs no raw data, so it runs on someone else's board.
    """
    try:
        a = L.selection_audit(scores, standard_error, top_k=top_k)
    except Exception as e:
        return {"error": str(e),
                "hint": "pass at least two scores and a non-negative standard error in score units"}
    return {
        "verdict": a.verdict,
        "n_models": a.n_models,
        "gap": a.gap,
        "gap_in_standard_errors": a.gap_in_se,
        "p_announced_number_1_is_wrong": a.p_wrong_winner,
        "winning_score_inflation": a.score_inflation,
        f"p_true_best_within_top_{a.top_k}": a.p_true_best_in_top,
        "trials": a.trials,
        "recommendation": (
            "Report a tie group, not a lone #1, and quote the score minus the inflation."
            if a.p_wrong_winner >= 0.5 else
            "Print the leader's gap in units of its own standard error next to the ranking."
        ),
    }


@mcp.tool()
def check_deployment_precision(tpr: float, fpr: float, prevalence: float) -> dict:
    """What a detector's PRECISION BECOMES where the target is rare. Give the operating point you
    will actually ship (tpr, fpr — % or 0-1) and the prevalence you expect in the field. Returns the
    precision an operator will really see, and how many false alarms that means per genuine catch.

    Use when: a model is described by AUC or accuracy on a balanced benchmark but will run somewhere
    the interesting class is rare — fraud, security triage, moderation, screening, defect finding.
    Do NOT pass an AUC as tpr: AUC summarises the whole curve and does not pin down a threshold.
    """
    try:
        t = _as_fraction(tpr, "tpr")
        f = _as_fraction(fpr, "fpr")
        pv = _as_fraction(prevalence, "prevalence")
    except ValueError as e:
        return {"error": str(e),
                "hint": "tpr/fpr/prevalence each take a fraction (0.95) or a percentage (95)."}
    r = C.base_rate_precision(t, f, pv)
    collapses = r.precision < 0.5 <= r.benchmark_precision
    return {
        "tpr": round(r.tpr, 4), "fpr": round(r.fpr, 4), "prevalence": r.prevalence,
        "deployment_precision": round(r.precision, 4),
        "benchmark_precision": round(r.benchmark_precision, 4),
        "false_alarms_per_true_catch": round(r.false_per_true, 3),
        "npv": round(r.npv, 6),
        "collapses_in_deployment": collapses,
        "verdict": (f"At {r.prevalence:.3g} prevalence this detector is right {r.precision:.1%} of the "
                    f"times it fires ({r.false_per_true:.3g} false alarms per true catch), while the same "
                    f"operating point reads {r.benchmark_precision:.1%} on a balanced set."),
        "recommendation": ("Report precision at the deployment base rate next to the benchmark score; "
                           "a threshold chosen on balanced data is usually the wrong threshold here."
                           if collapses else
                           "Precision holds up at this base rate — state it explicitly anyway, since "
                           "readers cannot derive it from the benchmark number."),
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


# --------------------------------------------------------------------------- #
# Deeper audits — when you have the RAW per-item results, not just the scores.
# check_top_rank approximates the tie group from summary scores; these do it properly.
# --------------------------------------------------------------------------- #
@mcp.tool()
def audit_leaderboard(results: dict, n_boot: int = 1000) -> dict:
    """Audit a leaderboard PROPERLY from its raw per-item results — the real version of
    check_top_rank. Use this instead of check_top_rank whenever you have, for each model, the list of
    test items it solved (e.g. SWE-bench 'resolved' instance-ids, or a {item: 0/1} map). It bootstraps
    each model's RANK to get a 95% rank confidence interval and P(truly #1), finds the tie group with
    a paired McNemar test (the correct test on shared items), counts resolvable tiers, and re-tests by
    splitting the items in half many times.

    results: {model_name: [solved_item_ids]} OR {model_name: {item_id: score}}.
    Use when: you have per-item / per-instance results and are about to report a #1 or an ordering.
    """
    try:
        a = L.audit_matrix(results, n_boot=int(n_boot))
    except Exception as e:
        return {"error": str(e)}
    return {
        "n_models": a.n_models, "n_items": a.n_items, "leader": a.leader, "top_score": a.top_score,
        "top_rank_resolved": a.top_resolved, "tie_group": a.tie_group, "tie_group_size": len(a.tie_group),
        "p_top_is_1": a.p_top_is_1, "stays_1_across_splits": a.stay_frac,
        "kendall_tau_stability": a.kendall_tau, "effective_tiers": a.effective_tiers,
        # psychometric "why" (Rasch IRT) + winner's-curse; null when skipped for very large boards
        "reliability": a.reliability, "frontier_test_information": a.frontier_info,
        "top2_ability_separation_sigma": a.z_top2, "winners_curse_inflation": a.winners_curse,
        "top_rows": [{"model": r.model, "score": r.score, "score_ci": [r.score_lo, r.score_hi],
                      "rank": r.rank, "rank_ci": [r.rank_lo, r.rank_hi], "p_is_1": r.p_is_1}
                     for r in a.rows[:10]],
        "verdict": a.verdict, "recommendation": a.recommendation,
    }


@mcp.tool()
def audit_preferences(battles: list, n_boot: int = 200) -> dict:
    """Audit a head-to-head / A-B preference leaderboard (arena, human or LLM votes) from the raw
    battles. Fits a Bradley-Terry ranking, bootstraps the top model's rank CI + P(#1), AND checks
    whether the preferences are transitive or run in rock-paper-scissors cycles — a single linear
    ranking is only honest if preferences are transitive.

    battles: a list of [winner, loser] pairs (one per decisive vote).
    Use when: a ranking comes from pairwise votes/comparisons rather than per-item scores.
    """
    try:
        pairs = [(x[0], x[1]) for x in battles]
        a = L.audit_pairwise(pairs, n_boot=int(n_boot))
    except Exception as e:
        return {"error": str(e)}
    return {
        "n_battles": a.n_battles, "leader": a.leader, "top_rank_resolved": a.top_resolved,
        "tie_group": a.tie_group, "p_top_is_1": a.p_top_is_1,
        "intransitivity_pct": a.intransitivity_pct, "null_intransitivity_pct": a.null_intransitivity_pct,
        "preferences_transitive": a.transitive,
        "top_rows": [{"model": r.model, "rank": r.rank, "rank_ci": [r.rank_lo, r.rank_hi],
                      "p_is_1": r.p_is_1} for r in a.rows[:10]],
        "verdict": a.verdict, "recommendation": a.recommendation,
    }


@mcp.tool()
def check_dimensions(results: dict) -> dict:
    """Does a leaderboard measure ONE skill or several? A single rank assumes a total order along one
    axis. This compares the result matrix's eigenspectrum to a shuffled null; more than one factor
    means two models with the same headline score can be strong on different parts of the benchmark,
    and the scalar rank hides that.

    results: {model_name: [solved_item_ids]} OR {model_name: {item_id: score}} (>=4 models).
    Use when: deciding whether a single leaderboard number fairly summarizes a multi-skill benchmark.
    """
    try:
        d = L.latent_dimensions(results)
    except Exception as e:
        return {"error": str(e)}
    return {
        "significant_skills": d.n_significant, "eigenvalues": d.eigenvalues, "null_edge": d.null_edge,
        "top_factor_fraction": d.top1_fraction, "verdict": d.verdict, "recommendation": d.recommendation,
    }


@mcp.tool()
def audit_swebench(split: str = "lite", n_boot: int = 1000) -> dict:
    """Audit a live SWE-bench leaderboard by name — fetches the public per-instance results and runs
    the full audit_leaderboard on them. No data to paste: just the split.

    split: "test" (large, ~1400 tasks), "verified" (500), "lite" (300), or "multimodal".
    Use when: the user references SWE-bench and you want the real confidence verdict on its current #1.
    Note: makes a network request to the public swe-bench/experiments repo.
    """
    try:
        from . import datasets as D
        subs = D.load_swebench(split)
    except Exception as e:
        return {"error": f"could not load SWE-bench {split}: {e}"}
    out = audit_leaderboard({k: list(v) for k, v in subs.items()}, n_boot=int(n_boot))
    out["benchmark"] = f"SWE-bench {split}"
    return out


def main():
    mcp.run()


if __name__ == "__main__":
    main()
