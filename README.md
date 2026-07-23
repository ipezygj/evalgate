# evalgate

**Cheap statistical checks for AI eval claims — run them before you publish.**

Most benchmark headlines overstate themselves in one of a few nameable ways. `evalgate` is four tiny, dependency-free checks, one per failure mode — the same checks behind a set of [independent eval-integrity audits](https://ipezygj.github.io/eval-audit-site/) that caught these mistakes in published work.

Pure Python, zero dependencies, runs anywhere.

```bash
pip install git+https://github.com/ipezygj/evalgate
```

---

## Use it as an MCP tool (for agents)

If you're an AI agent — or you run one (Claude, Cursor, Claude Code, Windsurf…) — evalgate ships an
**MCP server** so the model can *call these checks itself* before it trusts, reports, or acts on any
eval number: a benchmark score, a leaderboard #1, an LLM-as-judge verdict, or a claimed trend.

```bash
pip install "evalgate[mcp] @ git+https://github.com/ipezygj/evalgate"
```

Add it to your MCP client (e.g. `claude_desktop_config.json` / Cursor / Claude Code):

```json
{
  "mcpServers": {
    "evalgate": { "command": "evalgate-mcp" }
  }
}
```

Tools the agent gets, each with a "call this when…" description it can reason about:

| tool | the agent calls it before… |
|---|---|
| `check_top_rank` | claiming a model is #1 / SOTA on a benchmark (is the top rank real or a tie?) |
| `check_subset_win` | trusting a "best on subset/metric X" claim (look-elsewhere correction) |
| `check_judge_bias` | trusting an LLM-as-judge / A-B preference result (length / self-preference / position bias) |
| `check_resolution` | calling one of two close models better (can the benchmark even tell them apart?) |
| `check_trend_fragility` | reporting a fitted trend / scaling exponent (does one point carry it?) |

The point: an agent that produces an eval number should sanity-check it, and now it can — in one
call, with a plain verdict and a recommendation. Reproducible, zero-dependency checks (the `mcp`
extra is only for the server transport).

---

## The four checks

### 1. "We lead on subset X" — corrected for look-elsewhere
Report the subset/metric/checkpoint where a model looks best and you are reporting the **maximum of many noisy tests**. Correct for how many you could have picked.

```bash
evalgate correct --p 0.009 --n 23
# raw p=0.009 over 23 tests -> sidak p=0.19 (does NOT survive correction at alpha=0.05)
```
```python
from evalgate import correct_best_of
correct_best_of(0.009, n_tested=23).significant   # False
```
*(A real RewardBench "best subset" win: raw p=0.009 → p=0.19 after correcting for the 23 subsets. Not a finding.)*

### 2. Is the judge winning, or just longer / first / same-family?
An LLM-as-judge that "prefers" your model may be preferring the longer answer, the first-listed one, or its own family. Feed it the count and test against chance.

```bash
evalgate bias --wins 68 --n 100 --label "longer answer wins"
# longer answer wins: 68/100 = 68.0% (p=0.0004) -> BIAS
```
```python
from evalgate import bias_rate
bias_rate(68, 100).biased    # True
```
*(A widely-used GPT-4 judge preferred the longer answer 68% of the time and its own model family 71.5% — both at p≈0.)*

### 3. Does one data point flip your slope?
A scaling exponent or trend that hangs on a single high-leverage point isn't one. Leave each point out and refit.

```bash
evalgate loo examples/points.txt --power-law --threshold 1.0
# slope=1.08, leave-one-out range [0.87, 1.26] -> CROSSES 1 (most influential point: index 5)
```
```python
from evalgate import leave_one_out, power_law_exponent
leave_one_out(xs, ys, fit=power_law_exponent, threshold=1.0).crosses_threshold  # True
```
*(A reported "super-linear" grokking exponent, α=1.13, fell to 0.97 — with a better fit — when one point was dropped.)*

### 4. Is the gap bigger than the sample can resolve?
A leaderboard orders two models by a two-point accuracy gap on a finite test set. Ask whether that gap is even detectable at this sample size — or smaller than the minimum detectable effect, i.e. a coin flip.

```bash
evalgate power --n 200 --p1 0.85 --p2 0.83
# gap=+0.02 on n=200 (NOT significant, p=0.585); MDE at 80% power=0.103 -> UNDERPOWERED (gap < MDE)
```
```python
from evalgate import power_check
power_check(200, 0.85, 0.83).resolvable   # False — 2pp over 200 items can't be resolved
power_check(2000, 0.85, 0.80).resolvable  # True  — 5pp over 2000 items can
```
*(Frontier models on a fixed benchmark routinely sit a task or two apart — inside the MDE — so the #1 rank is noise. More votes, not a better model, resolves the tie.)*

---

## Library API

```python
from evalgate import (
    correct_best_of, sidak, bonferroni,     # look-elsewhere
    bias_rate, binomial_test,               # judge / metric bias
    leave_one_out, ols_slope, power_law_exponent,   # fragility + fits
    power_check, min_detectable_effect,     # power / minimum detectable effect
)
```
Every function returns a small dataclass that prints a one-line verdict and exposes the numbers (`.corrected_p`, `.p_value`, `.loo_min` …) so you can gate CI on them.

Reproduce the case studies:
```bash
python -m evalgate.checks     # -> evalgate selftest: OK (reproduced all 3 case studies + power check)
```

---

## Why this exists

These are textbook checks — the value isn't the math, it's running **all** of them, adversarially, on a number you're too close to. `evalgate` is the open, do-it-yourself version. When a launch, a paper, or a fundraise rides on a figure and you want it audited independently first, that's [the paid practice](https://ipezygj.github.io/eval-audit-site/).

Want the full checklist and the client-grade report template that wrap these checks? **[The Eval Integrity Kit](https://ipezystudio.gumroad.com/l/csod)** — the 9-check audit checklist, the report template I ship to clients, an `evalgate` quickstart, and three worked case studies.

The fuller story — why AI benchmark scores and trading backtests overpromise, and how to catch them — is in the book **[Measured, Not Believed](https://leanpub.com/measurednotbelieved)** (pay what you want).

## License
MIT — see [LICENSE](LICENSE).
