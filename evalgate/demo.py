"""A 30-second tour of the leaderboard audits. Run:  python -m evalgate.demo

Builds three tiny synthetic leaderboards - a real champion, a saturated coin-flip tie, and a
head-to-head board - and prints the audit for each. No network, deterministic, ASCII-only output.
"""
import random

from .leaderboard import audit_matrix, audit_pairwise
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
    print("Same three audits an agent gets from the MCP tools audit_leaderboard / audit_preferences.")


if __name__ == "__main__":
    main()
