# geo-probe findings - 2026-07-28T10-43-37Z

Category: **project management software for small teams**
Design: 20 prompts x k=5 reps per (brand, provider) cell, 8 cells.
Excluded: 1/100 anthropic runs (4 run-brand pairs), 1/100 perplexity runs (4 run-brand pairs).

![mention rate by brand and provider](chart.png)

## Findings

Treating the runs in a cell as independent trials overstates precision: across the 8 (brand, provider) cells the design effect averages 2.89, peaking at 3.49 on ClickUp/perplexity, where the cluster interval on ClickUp/perplexity is [0.68, 0.95] against a naive Wilson interval of [0.75, 0.90]. Between-prompt variance accounts for 49% of total variance on average (30-65% across cells). At a fixed run budget N the variance of the estimate is (k*var_between + var_within)/N, so any non-zero between-prompt component means the same spend on more prompts and fewer reps buys a tighter interval. The headline number is the minimum detectable effect: at this sample size, alpha=0.05, power=0.8, a week-over-week change has to exceed 21.3pp on Asana/perplexity, the smallest among cells whose interval clears both bounds, and 35.4pp on Basecamp/perplexity at the other end, before this design can distinguish it from sampling noise. Cells on a bound are excluded from both, since the bootstrap is degenerate there and understates the MDE. Applying the stricter criterion -- an MDE smaller than the headroom on both sides, so the threshold reads as an increase or a decrease -- the floor is 32.6pp on Basecamp/anthropic. A change smaller than that cannot be distinguished from sampling noise at this sample size; the design has no power to detect a move that small, which is a statement about the design and not evidence that no move occurred. Compared like for like, on ClickUp/perplexity the naive MDE is 14.7pp while the cluster MDE is 27.1pp -- a dashboard using the naive figure would treat a change in that range as detectable when this design cannot detect it.

## Per-cell numbers

| brand | provider | n used | k_eff | mention rate | cluster 95% | naive 95% (wrong) | deff | deff pred | resid | between-prompt var share | flip rate | mean rank (n) | MDE | naive MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Asana | anthropic | 99 | 4.95 | 0.96 | [0.88, 1.00] | [0.90, 0.98] | 1.90 | 2.50 | -0.60 | 0.38 | 0.10 | 1.96 (n=95) | 12.4pp† | 7.8pp |
| Trello | anthropic | 99 | 4.95 | 0.91 | [0.79, 1.00] | [0.84, 0.95] | 3.32 | 3.55 | -0.23 | 0.65 | 0.15 | 1.41 (n=90) | 21.4pp† | 11.4pp |
| ClickUp | anthropic | 99 | 4.95 | 0.89 | [0.77, 0.99] | [0.82, 0.94] | 3.35 | 3.06 | +0.29 | 0.52 | 0.20 | 2.53 (n=89) | 21.4pp† | 12.0pp |
| Basecamp | anthropic | 99 | 4.95 | 0.38 | [0.22, 0.54] | [0.28, 0.47] | 2.92 | 2.88 | +0.04 | 0.47 | 0.50 | 3.38 (n=37) | 32.6pp | 19.3pp |
| Trello | perplexity | 99 | 4.95 | 0.94 | [0.86, 1.00] | [0.87, 0.97] | 2.05 | 2.20 | -0.16 | 0.30 | 0.15 | 1.71 (n=93) | 14.0pp† | 9.5pp |
| Asana | perplexity | 99 | 4.95 | 0.89 | [0.78, 0.99] | [0.81, 0.94] | 2.83 | 2.99 | -0.15 | 0.50 | 0.20 | 2.08 (n=88) | 21.3pp† | 12.5pp |
| ClickUp | perplexity | 99 | 4.95 | 0.83 | [0.68, 0.95] | [0.75, 0.90] | 3.49 | 3.24 | +0.25 | 0.57 | 0.30 | 2.33 (n=83) | 27.1pp† | 14.7pp |
| Basecamp | perplexity | 99 | 4.95 | 0.53 | [0.35, 0.70] | [0.43, 0.62] | 3.29 | 3.10 | +0.19 | 0.53 | 0.50 | 3.02 (n=52) | 35.4pp | 19.9pp |

† MDE exceeds the available headroom in this direction; interpret one-sided. A mention rate is bounded in [0, 1], and a symmetric absolute-scale MDE cannot describe a move that would take the rate past the bound.

## Reading notes

- `mention rate` is the mean of the per-prompt rates, not the pooled run rate. The prompt is the unit of analysis.
- `naive 95% (wrong)` is the Wilson interval on pooled runs, and `naive MDE` is the matching detection threshold. Both are printed to be argued with, not used.
- `k_eff` is `n used / n prompts`, the realised cluster size after exclusions. `deff pred` is `1 + (k_eff - 1) * between-prompt var share`; `resid` is observed minus predicted.
- Rank statistics are conditional on the brand being mentioned, so `mean rank` describes only the runs where the brand appeared.
- Providers are never pooled: they are different measurement surfaces.
- Superlatives in the paragraph above are selected among cells whose interval does not touch 0 or 1. A cell on a bound can win 'smallest MDE' by being the most degenerate one in the table rather than the best measured.

Design-effect check: mean |observed - predicted| = 0.24 across 8 cells.

Models returned by the APIs: claude-sonnet-5, sonar-pro.
