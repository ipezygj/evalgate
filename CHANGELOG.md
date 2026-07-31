# Changelog

All notable changes to `evalgate`. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
this project uses [Semantic Versioning](https://semver.org/).

## [0.5.0]

### Added — the winner's curse, from published numbers alone
- `evalgate.leaderboard.selection_audit(scores, se)` — is the announced #1 the best model or the
  luckiest? Ranking selects on score, and a score is ability plus measurement error; sorting cannot
  separate them, so among close models the one that rises is disproportionately the one whose error
  pointed up. Returns **P(a rerun crowns someone else)**, the **inflation** of the winning score, the
  leader's **gap in units of its own standard error**, and P(the true best is in the printed top-k).
  Unlike the EVT figure inside `audit_matrix`, this needs only what a leaderboard prints — so it runs
  on someone else's board.
  Validated against closed form, not against itself: zero noise returns exactly zero; two tied models
  at sigma=5 return 2.78 against sigma/sqrt(pi) = 2.82; twenty tied return 9.33 against Blom's 9.35.
- MCP tool `check_winners_curse(scores, standard_error)` exposing the same check to agents.

### Fixed
- A tie no longer changes verdict with the random seed. Two identical models sit at exactly
  p_wrong = 0.5, so a "coin flip if >= 0.5" rule flipped between COIN-FLIP and THIN on sampling
  alone — at precisely the case where the answer matters most. The **margin** decides now: a leader
  inside one standard error is unresolved whatever the simulation returned.

## [0.4.3]

### Changed
- `mcp` is a hard dependency instead of an optional extra. mcp 2.0 removed `mcp.server.fastmcp`,
  which `mcp_server.py` imports, so a bare install left the server unable to start; the `[mcp]`
  extra is kept, empty, so existing install instructions keep working.

### Added
- Tests that the declared console scripts resolve to something callable, and that the README's
  install line names the package PyPI publishes. Both failure modes are invisible to a normal test
  run — they only surface for a stranger installing the package.

## [0.4.2]

### Added
- `base_rate_precision(tpr, fpr, prevalence)` and the MCP tool `check_deployment_precision`: what a
  detector's precision becomes where the target is rare. A benchmark measures on a roughly balanced
  set; deployment usually is not, and the difference appears nowhere in the published score.

### Fixed
- Score units are inferred **once per list**, not per value. Given percentages `[95, 87, 1]` the old
  rule read `1` as the fraction 1.0 — 100% — and reported the weakest model as the leader with the
  top rank marked *resolved*. A unit is a property of the list, not of any single entry.
- CI installs and imports the package on 3.10/3.12/3.13, with a weekly run, so an upstream change
  cannot break the install silently again.

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
