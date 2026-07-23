"""Network golden tests — pin that evalgate reproduces the real SWE-bench verdicts end to end.

Skipped by default (they hit the public swe-bench/experiments repo). Enable with:
    EVALGATE_NETWORK_TESTS=1 pytest tests/test_network.py
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EVALGATE_NETWORK_TESTS") != "1",
    reason="set EVALGATE_NETWORK_TESTS=1 to run network golden tests",
)


def test_swebench_test_is_resolved():
    from evalgate.datasets import load_swebench
    from evalgate.leaderboard import audit_matrix
    a = audit_matrix(load_swebench("test"), n_boot=500)
    # the large split has a genuine, well-separated champion
    assert a.top_resolved
    assert len(a.tie_group) == 1
    assert a.p_top_is_1 >= 0.9
    assert a.z_top2 is not None and a.z_top2 > 2


def test_swebench_lite_is_a_tie():
    from evalgate.datasets import load_swebench
    from evalgate.leaderboard import audit_matrix
    a = audit_matrix(load_swebench("lite"), n_boot=500)
    # the saturated split cannot resolve its top rank
    assert not a.top_resolved
    assert len(a.tie_group) >= 2
    assert a.z_top2 is not None and abs(a.z_top2) < 2
