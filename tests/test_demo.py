"""The demo is the first thing anyone runs. It must run, and say the same thing twice.

`python -m evalgate.demo` is what a new reader executes after installing, and it is
the one code path with no caller to notice when it breaks: no other module imports
it, so a rename anywhere it touches fails only for a stranger. It also quotes the
same figures as the published write-ups, which is a promise worth pinning.
"""
import subprocess
import sys

import pytest


def _run():
    p = subprocess.run([sys.executable, "-m", "evalgate.demo"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, f"demo exited {p.returncode}\n{p.stderr[-800:]}"
    return p.stdout


@pytest.fixture(scope="module")
def output():
    return _run()


def test_demo_runs_and_prints_every_section(output):
    for marker in ("1)", "2)", "3)", "4)", "5)", "6)"):
        assert marker in output, f"section {marker} missing from the demo"


def test_demo_is_deterministic():
    """Section 4 resamples. A tour that prints different numbers each run teaches
    the opposite of what this library is for."""
    assert _run() == _run(), "demo output varies between runs"


def test_demo_reproduces_the_published_figures(output):
    # the HELM answer-key shape: a fixed 'D' at 0.4100 against 0.2500 for guessing
    assert "'D' scores 0.4100" in output
    assert "0.2500 for uniform guessing" in output
    # an interior score published with a zero error bar
    assert "FALSE PRECISION" in output
    # a top-two tie is a coin flip, not a thin lead
    assert "COIN-FLIP-#1" in output


def test_demo_does_not_overclaim_its_tool_count(output):
    """The closing line used to say 'three' after the sixth tool shipped."""
    assert "All six are MCP tools" in output
