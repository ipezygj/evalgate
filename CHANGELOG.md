# Changelog

All notable changes to `evalgate`. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
this project uses [Semantic Versioning](https://semver.org/).

## [0.4.0]

### Added — whole-leaderboard audits from raw data
The original checks work on *summary numbers*. This release adds an audit layer that works on the
**raw per-item results** a leaderboard already publishes, doing the real thing instead of an
approximation.

- `evalgate.leaderboard`
  - `audit_matrix(results)` — bootstrap **rank confidence intervals** + P(truly #1), the
    paired-McNemar **tie group** at the top, **effective resolvable tiers**, and **split-half
    stability**. Also reports the psychometric "why" (Rasch **reliability**, **frontier test
    information**, **#1-vs-#2 ability separation in sigma**) and an EVT **winner's-curse** inflation.
    Now also returns a 95% **score confidence interval** per row.
  - `audit_pairwise(battles)` — Bradley-Terry ranking with bootstrap rank CIs + a **Condorcet**
    check that preferences are transitive (not rock-paper-scissors cycles).
  - `latent_dimensions(results)` — eigenspectrum vs a shuffled null: does the board measure one
    skill or several?
  - `audit(data)` — auto-dispatches to matrix or pairwise by data shape.
- `evalgate.datasets`
  - `load_swebench(split)` — fetch a live SWE-bench split's per-item results by name.
  - `load_results_json(path)` / `load_battles_csv(path)` — audit your own local files.
- `evalgate.format` — `format_matrix` / `format_pairwise` / `format_dimensions` render an audit as an
  aligned, **pure-ASCII** text block (safe on any console encoding).
- `evalgate.demo` — `python -m evalgate.demo` prints a three-board tour.
- MCP tools: `audit_leaderboard`, `audit_preferences`, `check_dimensions`, `audit_swebench`.
- Tests: whole-leaderboard suite + opt-in network golden tests (`EVALGATE_NETWORK_TESTS=1`) that pin
  the real SWE-bench reproduction (Test resolves; Lite is a statistical tie). Determinism is pinned
  (fixed seed → byte-identical audit).

### Changed
- Distribution renamed to **`eval-gate`** on PyPI (`evalgate` was taken); the import name and CLI
  stay `evalgate`.
- Sharpened the MCP server instructions with explicit agent triggers; added `server.json` for the
  official MCP registry.

## [0.3.0]
- MCP server for agents (`evalgate-mcp`) with the five summary checks; Smithery manifest; PyPI
  trusted-publishing workflow.

## [0.2.0]
- Added the power / minimum-detectable-effect check (is the gap bigger than the sample can resolve?).

## [0.1.0]
- Initial release: dependency-free statistical checks for eval claims — multiple-comparisons
  correction, judge/metric bias, and leave-one-out fragility.
