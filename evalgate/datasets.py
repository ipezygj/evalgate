"""evalgate.datasets — fetch a public leaderboard's raw per-item results so an agent can audit a
named benchmark directly, without the user pasting data.

Currently: SWE-bench (the agentic-coding benchmark). Its submissions publish per-instance results in
the public swe-bench/experiments repo, so we can pull "which tasks each system solved" and hand it to
leaderboard.audit_matrix.

Network-only helper (stdlib urllib) — the core checks stay dependency-free; this just saves a copy-
paste. Results are cached in-process per split.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request

SWEBENCH_SPLITS = ("test", "verified", "lite", "multimodal")


def load_results_json(path: str) -> dict:
    """Load per-item results from a local JSON file for audit_matrix. Accepted shapes:
      {"model": ["item1", "item2", ...]}         # list of solved item-ids
      {"model": {"item1": 1, "item2": 0, ...}}   # per-item score/pass map
    Returns the dict unchanged (sets for the list form) — ready for leaderboard.audit_matrix."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or not d:
        raise ValueError("expected a non-empty {model: [items] | {item: score}} object")
    out = {}
    for model, v in d.items():
        out[model] = set(v) if isinstance(v, (list, tuple)) else dict(v)
    return out


def load_battles_csv(path: str, a_col: str = "model_a", b_col: str = "model_b",
                     winner_col: str = "winner") -> list:
    """Load pairwise battles from a CSV for audit_pairwise. Needs columns for the two models and the
    winner. The winner cell may be the winning model's NAME, or 'model_a'/'model_b' (arena style),
    or 'a'/'b'. Rows that are ties/invalid are skipped. Returns a list of (winner, loser) tuples."""
    battles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b, w = row.get(a_col), row.get(b_col), (row.get(winner_col) or "").strip()
            if not a or not b or not w:
                continue
            wl = w.lower()
            if wl in ("model_a", "a", a.lower()):
                battles.append((a, b))
            elif wl in ("model_b", "b", b.lower()):
                battles.append((b, a))
            # anything else (tie / both-bad) is dropped
    if not battles:
        raise ValueError("no decisive battles parsed — check the column names and winner values")
    return battles
_TREE_URL = "https://api.github.com/repos/swe-bench/experiments/git/trees/main?recursive=1"
_RAW = "https://raw.githubusercontent.com/swe-bench/experiments/main/"
_cache: dict = {}


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "evalgate"})
    return json.load(urllib.request.urlopen(req, timeout=45))


def load_swebench(split: str) -> dict:
    """Return {submission_name: set(resolved_instance_ids)} for a SWE-bench split
    ('test' | 'verified' | 'lite' | 'multimodal'). Cached per split."""
    split = split.lower().strip()
    if split not in SWEBENCH_SPLITS:
        raise ValueError(f"split must be one of {SWEBENCH_SPLITS}")
    if split in _cache:
        return _cache[split]
    tree = _get(_TREE_URL)["tree"]
    pat = re.compile(rf"evaluation/{split}/([^/]+)/results/results.json$")
    subs: dict = {}
    for t in tree:
        m = pat.search(t["path"])
        if m:
            try:
                subs[m.group(1)] = set(_get(_RAW + t["path"]).get("resolved") or [])
            except Exception:
                pass
    if not subs:
        raise RuntimeError(f"no submissions found for SWE-bench {split}")
    _cache[split] = subs
    return subs
