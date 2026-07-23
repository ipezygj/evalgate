"""evalgate.datasets — fetch a public leaderboard's raw per-item results so an agent can audit a
named benchmark directly, without the user pasting data.

Currently: SWE-bench (the agentic-coding benchmark). Its submissions publish per-instance results in
the public swe-bench/experiments repo, so we can pull "which tasks each system solved" and hand it to
leaderboard.audit_matrix.

Network-only helper (stdlib urllib) — the core checks stay dependency-free; this just saves a copy-
paste. Results are cached in-process per split.
"""
from __future__ import annotations

import json
import re
import urllib.request

SWEBENCH_SPLITS = ("test", "verified", "lite", "multimodal")
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
