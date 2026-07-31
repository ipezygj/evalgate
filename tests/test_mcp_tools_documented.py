"""Every MCP tool must be documented where a user or an agent will look for it.

A tool nobody can find is a tool nobody calls. Two shipped undocumented — a
`check_deployment_precision` written the same week as the write-up that explains
it, and `audit_swebench`, which needs no data at all and is therefore the easiest
one to try. Neither absence breaks a test that exercises behaviour, because both
tools work.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = ROOT / "evalgate" / "mcp_server.py"
README = ROOT / "README.md"


def _tools():
    """Take the first `def` after each decorator.

    A pattern like `@mcp.tool\\([^)]*\\)` stops at the first paren inside
    `annotations=_ann(...)` and matches nothing. That exact mistake made a sweep of
    the sibling repos print "0 tools, none undocumented", which reads like an
    all-clear. This shape survives arguments in the decorator.
    """
    src = SERVER.read_text(encoding="utf-8")
    found = []
    for chunk in src.split("@mcp.tool")[1:]:
        m = re.search(r"\ndef (\w+)\(", chunk)
        if m:
            found.append(m.group(1))
    return sorted(set(found))


def test_the_parser_finds_tools_at_all():
    """Guards the guard: a parser that matches nothing passes every test below."""
    tools = _tools()
    assert len(tools) >= 10, f"only {len(tools)} tools found — the decorator shape probably changed"


@pytest.mark.parametrize("tool", _tools())
def test_tool_is_named_in_the_readme(tool):
    assert f"`{tool}`" in README.read_text(encoding="utf-8"), (
        f"MCP tool {tool} is not mentioned in README.md — an agent choosing a tool "
        "reads that table, so an undocumented tool is an uncallable one"
    )


@pytest.mark.parametrize("tool", _tools())
def test_tool_has_a_docstring_an_agent_can_act_on(tool):
    src = SERVER.read_text(encoding="utf-8")
    body = src.split(f"def {tool}(", 1)[1]
    doc = re.search(r'"""(.*?)"""', body, re.S)
    assert doc, f"{tool} has no docstring; the MCP client shows that text as the tool description"
    text = doc.group(1).strip()
    assert len(text) > 120, f"{tool}'s docstring is too thin to choose it by: {text[:60]!r}"
    assert "Use when" in text or "Use it when" in text, (
        f"{tool}'s docstring never says WHEN to use it, which is what an agent selects on"
    )
