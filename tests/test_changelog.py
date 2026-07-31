"""The changelog must describe the version the package actually ships.

A changelog drifts silently: the version gets bumped for a release, the entry is
written next time, and "next time" is the release nobody documented. This one had
stopped at 0.4.0 while the package shipped 0.4.1, 0.4.2 and 0.4.3 — including the
bug fix that read a 1% score as 100%, and the dependency change that decided
whether the MCP server started at all. The releases worth documenting are exactly
the ones you are most tempted to skip.

Nothing here judges the prose. It checks that a heading exists for what is being
shipped, and that the headings still descend.
"""
import pathlib
import re

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # py3.10
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"
HEADING = re.compile(r"^##\s*\[(\d+(?:\.\d+)*)\]", re.M)


def _version():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _headings():
    return HEADING.findall(CHANGELOG.read_text(encoding="utf-8"))


def _key(v):
    return tuple(int(part) for part in v.split("."))


def test_changelog_exists_and_has_entries():
    assert CHANGELOG.exists(), "CHANGELOG.md is gone"
    assert _headings(), "CHANGELOG.md has no '## [x.y.z]' headings"


def test_shipped_version_is_documented():
    version, heads = _version(), _headings()
    assert version in heads, (
        f"pyproject ships {version} and the changelog never mentions it — "
        f"newest entry is {heads[0]}"
    )


def test_shipped_version_is_the_newest_entry():
    version, heads = _version(), _headings()
    assert heads[0] == version, (
        f"changelog's newest entry is {heads[0]} but the package ships {version}: "
        "either the entry was never written, or the bump was forgotten"
    )


def test_headings_descend():
    heads = _headings()
    keys = [_key(h) for h in heads]
    assert keys == sorted(keys, reverse=True), f"changelog entries are out of order: {heads}"


def test_no_duplicate_version_headings():
    heads = _headings()
    dupes = {h for h in heads if heads.count(h) > 1}
    assert not dupes, f"a version is documented twice: {sorted(dupes)}"
