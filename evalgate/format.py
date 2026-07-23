"""evalgate.format — turn a leaderboard audit into a clean, human-readable report.

The audits (leaderboard.audit_matrix / audit_pairwise / latent_dimensions) return dataclasses full of
numbers. These helpers render them as an aligned text block you can print, paste into a PR comment, or
drop in a report. Pure formatting — no new statistics.
"""
from __future__ import annotations

from .leaderboard import MatrixAudit, PairwiseAudit, Dimensionality


def _bar(frac: float, width: int = 20) -> str:
    # ASCII only — renders on any console/encoding (Windows cp1252 included)
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    return "#" * n + "-" * (width - n)


def format_matrix(a: MatrixAudit, title: str = "Leaderboard") -> str:
    mark = "REAL #1" if a.top_resolved else (f"TIE ({len(a.tie_group)})" if len(a.tie_group) > 1 else "UNRESOLVED")
    lines = [
        f"{title} — {a.n_models} submissions x {a.n_items} items",
        f"  verdict:      [{mark}]  P(printed #1 is true #1) {_bar(a.p_top_is_1)} {a.p_top_is_1:.0%}",
        f"  #1:           {a.leader}  (score {a.top_score:.3f})",
        f"  tie group:    {', '.join(a.tie_group)}",
        f"  stability:    stays #1 on {a.stay_frac:.0%} of item splits; Kendall tau {a.kendall_tau:.2f}",
        f"  resolution:   {a.effective_tiers} distinguishable tiers of {a.n_models}",
    ]
    if a.reliability is not None:
        z = f"{a.z_top2:+.2f} sigma" if a.z_top2 is not None else "n/a"
        lines.append(f"  psychometric: reliability {a.reliability:.2f}; #1-vs-#2 ability {z}; "
                     f"frontier info {a.frontier_info:.1f}")
    if a.winners_curse is not None:
        lines.append(f"  winner's-curse inflation of the #1: {a.winners_curse:.3f}")
    lines.append("")
    lines.append("  rank  score  95% rank-CI  P(#1)  submission")
    for r in a.rows[:12]:
        star = " *" if r.model in a.tie_group else "  "
        lines.append(f"  {r.rank:>4}  {r.score:.3f}   [{r.rank_lo:>2},{r.rank_hi:>3}]  {r.p_is_1:.2f}{star} {r.model[:44]}")
    lines.append("")
    lines.append(f"  -> {a.verdict}")
    lines.append(f"  -> {a.recommendation}")
    return "\n".join(lines)


def format_pairwise(a: PairwiseAudit, title: str = "Pairwise leaderboard") -> str:
    mark = "REAL #1" if a.top_resolved else f"TIE ({len(a.tie_group)})"
    trans = "transitive" if a.transitive else "INTRANSITIVE (rock-paper-scissors)"
    lines = [
        f"{title} — {a.n_battles:,} comparisons",
        f"  verdict:      [{mark}]  P(#1) {_bar(a.p_top_is_1)} {a.p_top_is_1:.0%}",
        f"  #1:           {a.leader}",
        f"  tie group:    {', '.join(a.tie_group)}",
        f"  preferences:  {a.intransitivity_pct:.1f}% cyclic triples vs {a.null_intransitivity_pct:.1f}% null -> {trans}",
        "",
        "  rank  95% rank-CI  P(#1)  player",
    ]
    for r in a.rows[:12]:
        star = " *" if r.model in a.tie_group else "  "
        lines.append(f"  {r.rank:>4}   [{r.rank_lo:>2},{r.rank_hi:>3}]  {r.p_is_1:.2f}{star} {r.model[:44]}")
    lines.append("")
    lines.append(f"  -> {a.verdict}")
    lines.append(f"  -> {a.recommendation}")
    return "\n".join(lines)


def format_dimensions(d: Dimensionality, title: str = "Latent dimensionality") -> str:
    return "\n".join([
        f"{title}",
        f"  significant skills: {d.n_significant}  (eigenvalues {', '.join(f'{e:.2f}' for e in d.eigenvalues[:4])}; "
        f"null edge {d.null_edge:.2f}; top factor {d.top1_fraction:.0%})",
        f"  -> {d.verdict}",
        f"  -> {d.recommendation}",
    ])


def _selftest() -> None:
    from .leaderboard import audit_matrix, audit_pairwise, latent_dimensions
    import random
    m = audit_matrix({"A": set(range(160)), "B": set(range(95)), "C": set(range(60))}, n_boot=200)
    s = format_matrix(m, "SWE-bench demo")
    assert "REAL #1" in s and "rank-CI" in s and "->" in s
    b = ([("A", "B")] * 40 + [("A", "C")] * 45 + [("B", "C")] * 40
         + [("B", "A")] * 8 + [("C", "A")] * 5 + [("C", "B")] * 10)
    assert "comparisons" in format_pairwise(audit_pairwise(b, n_boot=80, min_pair=10))
    rng = random.Random(0)
    subs = {f"m{k}": {i for i in range(150) if rng.random() < 0.3 + 0.03 * k} for k in range(8)}
    assert "significant skills" in format_dimensions(latent_dimensions(subs, n_perm=15))
    print("evalgate.format selftest: OK")


if __name__ == "__main__":
    _selftest()
