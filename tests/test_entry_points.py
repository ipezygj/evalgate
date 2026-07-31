"""Every declared console script must actually resolve to something callable.

`[project.scripts]` is a string, and nothing checks it until a user installs the
package and runs the command. The test suite imports modules directly, so it
never travels the path the generated wrapper takes — a renamed function fails
only in the packaged install, and the traceback surfaces in someone else's build
log. That is not hypothetical; it is how three MCP listings broke this week.

This walks each declared entry point the way the wrapper does: import the
module, follow the dotted attribute path, require the result to be callable.
"""
import importlib
import pathlib

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli")

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _scripts():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f).get("project", {}).get("scripts", {})


def test_scripts_are_declared():
    assert _scripts(), "pyproject declares no console scripts — did the table get renamed?"


@pytest.mark.parametrize("name,target", sorted(_scripts().items()))
def test_console_script_resolves(name, target):
    module_path, _, attr_path = target.partition(":")
    assert attr_path, f"{name} = {target!r} has no ':attr' part"

    module = importlib.import_module(module_path)

    obj = module
    walked = module_path
    for part in attr_path.split("."):
        assert hasattr(obj, part), f"{name}: {walked} has no attribute {part!r}"
        obj = getattr(obj, part)
        walked = f"{walked}.{part}"

    assert callable(obj), f"{name} points at {walked}, which is not callable"
