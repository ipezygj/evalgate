"""A 30-second tour of the leaderboard audits. Run:  python -m evalgate.demo

Builds three tiny synthetic leaderboards - a real champion, a saturated coin-flip tie, and a
head-to-head board - and prints the audit for each. Then runs the three checks that need only
what a leaderboard PRINTS, which is the situation most readers are actually in: you can see the
scores and the error bars, not the per-item results. No network, deterministic, ASCII-only.
"""
import random

from .leaderboard import (audit_matrix, audit_pairwise, constant_baseline,
                          selection_audit, stderr_audit)
from .format import format_matrix, format_pairwise


def _real_board():
    # one clearly-best model, then a spread - the rank is resolvable
    return {
        "clear-leader": set(range(170)),
        "second":       set(range(120)),
        "third":        set(range(95)),
        "fourth":       set(range(70)),
    }


def _saturated_board(seed=0):
    # four frontier models within a whisker of each other on 260 items - a coin-flip top.
    # INDEPENDENT draws per model (not nested thresholds) so no one strictly dominates.
    rng = random.Random(seed)
    items = range(260)
    board = {}
    for name, skill in (("frontier-A", 0.70), ("frontier-B", 0.70),
                        ("frontier-C", 0.69), ("frontier-D", 0.69), ("midpack", 0.55)):
        board[name] = {i for i in items if rng.random() < skill}
    return board


def _pairwise_board(seed=1):
    rng = random.Random(seed)
    strength = {"gpt-ish": 4.0, "claude-ish": 3.7, "gemini-ish": 2.0, "llama-ish": 1.0}
    battles = []
    names = list(strength)
    for _ in range(4000):
        a, b = rng.sample(names, 2)
        pa = strength[a] / (strength[a] + strength[b])
        battles.append((a, b) if rng.random() < pa else (b, a))
    return battles


def main():
    print("=" * 74)
    print("evalgate - leaderboard audit demo (synthetic data, deterministic)")
    print("=" * 74)
    print("\n1) A REAL champion - the top rank is resolvable:\n")
    print(format_matrix(audit_matrix(_real_board(), n_boot=600, seed=0), "Real-champion board"))
    print("\n" + "-" * 74)
    print("\n2) A SATURATED board - four models tied at the top, the #1 is a coin flip:\n")
    print(format_matrix(audit_matrix(_saturated_board(), n_boot=600, seed=0), "Saturated board"))
    print("\n" + "-" * 74)
    print("\n3) A HEAD-TO-HEAD board - from pairwise votes:\n")
    print(format_pairwise(audit_pairwise(_pairwise_board(), n_boot=150, seed=0), "Arena board"))
    print("\n" + "=" * 74)
    print("\nThose three need RAW per-item results. The next three need only what a board PRINTS:\n")

    # shaped like SWE-bench Verified: an exact tie at the top, SE about 2.1
    scores = [70.0, 70.0] + [70.0 - 0.4 * i for i in range(1, 12)]
    curse = selection_audit(scores, se=2.1, trials=4000, seed=0)
    print("4) Is the announced #1 the best model, or the luckiest?")
    print(f"   gap {curse.gap:.2f} = {curse.gap_in_se:.2f} of the leader's own SE")
    print(f"   a rerun crowns someone else {curse.p_wrong_winner:.1%} of the time; "
          f"the winning score is inflated +{curse.score_inflation:.2f}")
    print(f"   -> {curse.verdict}")

    # shaped like HELM classic MMLU college_chemistry: D correct on 41 of 100 items
    key = ["D"] * 41 + ["A"] * 22 + ["B"] * 22 + ["C"] * 15
    floor = constant_baseline(key, scores=[0.4733, 0.30, 0.28, 0.2633, 0.24, 0.22])
    print("\n5) What is the FLOOR this benchmark is read against?")
    print(f"   best constant answer '{floor.answer}' scores {floor.score:.4f}, "
          f"against {floor.chance:.4f} for uniform guessing")
    print(f"   -> {floor.verdict}")

    # shaped like the Open LLM v2 records: an interior score with a zero error bar
    rows = [("bbh", 0.5, 100, (0.5 * 0.5 / 99) ** 0.5),
            ("gpqa", 0.25, 50, (0.25 * 0.75 / 49) ** 0.5),
            ("math", 0.3125, 64, 0.0)]
    bars = stderr_audit(rows)
    print("\n6) Do the published error bars survive being recomputed?")
    print(f"   {bars.reproduced} of {bars.n_rows} rows reproduce; "
          f"{len(bars.impossible_zero)} claim a standard error of exactly 0 for an interior score")
    print(f"   -> {bars.verdict}")

    print("\n" + "=" * 74)
    print("All six are MCP tools too, so an agent can run them before repeating a benchmark claim.")


if __name__ == "__main__":
    main()
