"""Command-line interface: `evalgate correct|bias|loo`."""
from __future__ import annotations

import argparse
import sys

from .checks import bias_rate, correct_best_of, leave_one_out, power_law_exponent


def _read_xy(path: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    return xs, ys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="evalgate",
        description="Cheap statistical checks for AI eval claims before you publish them.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correct", help="correct a best-of-N p-value (look-elsewhere)")
    c.add_argument("--p", type=float, required=True, help="the raw best-subset p-value")
    c.add_argument("--n", type=int, required=True, help="how many subsets/metrics were tested")
    c.add_argument("--method", choices=["sidak", "bonferroni"], default="sidak")
    c.add_argument("--alpha", type=float, default=0.05)

    b = sub.add_parser("bias", help="test a judge/metric preference vs chance")
    b.add_argument("--wins", type=int, required=True, help="verdicts won by the tested side")
    b.add_argument("--n", type=int, required=True, help="total verdicts")
    b.add_argument("--p0", type=float, default=0.5, help="chance rate (default 0.5)")
    b.add_argument("--label", default="preferred side wins")

    l = sub.add_parser("loo", help="leave-one-out fragility of a slope/exponent")
    l.add_argument("file", help="whitespace/comma-separated 'x y' per line")
    l.add_argument("--power-law", action="store_true", help="fit exponent on log-log axes")
    l.add_argument("--threshold", type=float, default=None,
                   help="flag if dropping a point crosses this value (e.g. 1.0)")

    args = p.parse_args(argv)

    if args.cmd == "correct":
        print(correct_best_of(args.p, args.n, args.method, args.alpha))
    elif args.cmd == "bias":
        print(bias_rate(args.wins, args.n, args.p0, label=args.label))
    elif args.cmd == "loo":
        xs, ys = _read_xy(args.file)
        fit = power_law_exponent if args.power_law else None
        kw = {"threshold": args.threshold}
        if fit:
            kw["fit"] = fit
        print(leave_one_out(xs, ys, **kw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
