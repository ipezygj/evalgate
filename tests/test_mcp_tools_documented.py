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
    src = SERVER.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"@mcp\.tool\(\)\s*\ndef (\w+)", src)))


def test_there_are_tools_to_document():
    assert _tools(), "no @mcp.tool() functions found — did the decorator change?"


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
