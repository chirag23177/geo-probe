# Results

Batch `2026-07-28T10-43-37Z`, run 2026-07-28.

- Category: project management software for small teams. Four brands, one focal
  and three competitors; the config does not record which is which.
- 20 buyer-intent prompts, k=5 repetitions, 2 providers, 200 runs, 8
  (brand, provider) cells.
- Models as returned by the APIs: `claude-sonnet-5`, `sonar-pro`.
- Generated output: [`findings.md`](../reports/2026-07-28T10-43-37Z/findings.md),
  [`chart.png`](../reports/2026-07-28T10-43-37Z/chart.png). Source of truth for
  every number below is `data/agg/2026-07-28T10-43-37Z.json`.

## Headline

| | |
|---|---|
| design effect | **2.89** average, range 1.90–3.49 |
| interval width vs pooled | **1.70×** wider (√2.89) |
| effective sample size | ~35 of every 100 runs |
| MDE, cells clearing both bounds | **21.3pp to 35.4pp** (α=0.05, power=0.80) |
| between-prompt variance share | 49% average, 30–65% across cells |
| runs excluded | 2 of 200 (1.0%) |

The per-cell table is in the generated
[`findings.md`](../reports/2026-07-28T10-43-37Z/findings.md) rather than retyped
here, so it cannot drift from the aggregate.

### Which floor, and why

"The smallest MDE" depends on which cells you allow to compete, and the answer
moves by a factor of two and a half across three defensible criteria. Rather than
defend one number, here is the whole ladder — the same instinct as printing the
naive interval next to the correct one.

| criterion | floor | cell | why it is disqualified |
|---|---|---|---|
| smallest in the table | 12.4pp | Asana/anthropic | interval touches 1.00; the percentile bootstrap is degenerate there and the compression understates the threshold. Not usable. |
| smallest among cells clearing both bounds | **21.3pp** | Asana/perplexity | usable, but still one-sided: at a rate of 0.890 there is only 0.110 of headroom upward, so a 21.3pp *increase* is not a thing this cell can describe. Carries `†`. |
| smallest with an MDE below the headroom on both sides | 32.6pp | Basecamp/anthropic | none — this is the strictest reading. Only the two Basecamp cells qualify. |

**The README quotes 21.3pp**, the middle rung: it is the tightest figure that is
not an artifact of a bound, and it is the one the generated `findings.md`
reports. If you want a threshold that reads in both directions — the natural one
for week-over-week tracking, where a brand can move either way — the honest
number is **32.6pp**, and only two of the eight cells support it at all.

That the strictest rung is available only for the two lowest-rate cells is itself
the finding: near the ceiling there is not enough room left for a symmetric
absolute-scale threshold to mean anything.

## What the design-effect check showed

For a balanced design, `deff ≈ 1 + (k − 1)·ICC`. With a measured ICC of 0.49 and
k_eff of 4.95 that predicts 2.94, against 2.89 observed. Per cell, observed and
predicted differ by 0.24 on average, in both directions and with no provider
pattern.

The check paid for itself on the first pass, when 29 Perplexity runs were being
excluded:

| | mean \|residual\| vs prediction at k=5 | at k_eff |
|---|---|---|
| Perplexity cells, pre-correction (k_eff = 3.55) | 0.53 | 0.14 |
| all cells, post-correction (k_eff = 4.95) | 0.24 | 0.24 |

Dropped runs lower the realised repetitions per prompt, and a smaller realised
cluster size mechanically depresses the measured design effect. The Perplexity
cells fitted their prediction only at the true k_eff of 3.55. Recovering the runs
moved the average design effect from 2.42 to 2.89 — so the pre-correction figure
was about 16% too low, understating the quantity this tool exists to measure, in
the direction that flatters the pooled method.

## The extraction correction

The extractor is required to quote a verbatim span from the response for every
brand it marks as mentioned. On the first pass this rejected 29 of 100 Perplexity
runs and 0 of 100 Anthropic runs, after three retries each.

Diagnosis ran before any fix, and is committed:
[`failure_diagnosis.md`](../reports/2026-07-28T10-43-37Z/failure_diagnosis.md)
(pre-correction) and
[`failure_diagnosis_postfix.md`](../reports/2026-07-28T10-43-37Z/failure_diagnosis_postfix.md).
Failing responses were longer (3901 vs 3412 characters) with more line breaks
(63 vs 52), while naming statistically indistinguishable numbers of tracked
brands (3.3 vs 3.1). Failures were spread across 18 of 20 prompts, not
concentrated. That pattern points at rendering, not at content.

The fix made the comparison invariant to rendering: Unicode normalisation, curly
quotes and dashes folded to ASCII, markdown emphasis and `[n]` citation markers
stripped, whitespace collapsed. There is no casefolding, no fuzzy matching, no
edit distance, and no partial-overlap threshold, and the retry count is unchanged.
A span containing a word absent from the source still fails. Exclusions fell from
14.5% to 1.0% — 2 runs, one per provider.

Recovering the 28 Perplexity runs moved the four Perplexity mention rates as
follows — and only these four are evidence, since the Anthropic cells never had a
run recovered and any movement in them comes from their own single exclusion:

| cell | pre | post | delta |
|---|---|---|---|
| Asana/perplexity | 0.899 | 0.890 | −0.009 |
| Basecamp/perplexity | 0.522 | 0.528 | +0.005 |
| Trello/perplexity | 0.938 | 0.940 | +0.002 |
| ClickUp/perplexity | 0.830 | 0.830 | 0.000 |

**At most 0.009, in mixed directions** — two up, one down, one unchanged. That
argues against the exclusions having been selective: had the dropped runs
differed systematically in whether they mentioned brands, adding 28 of them back
would have shifted the rates consistently in one direction rather than jittering
around zero.

It does **not** establish that the extractor is correct. A false-negative rate
shared by both the excluded and the retained halves would move nothing here and
would stay invisible.

## Extractor validation

**Original sample** — 30 (run, brand) pairs, labelled by hand.

| | |
|---|---|
| raw agreement on `mentioned` | 1.000 (30/30) |
| Cohen's kappa | 1.000 |
| label balance | 27 true, 3 false |
| 95% upper bound on error rate | 11.4% |

Both classes are present, so chance agreement is 0.82 and the kappa is
meaningful. Zero errors in 30 pairs bounds the error rate at 11.4%, not at zero.

**Recovered sample** — 10 pairs drawn only from spans that matched after
normalisation.

| | |
|---|---|
| raw agreement on `mentioned` | 1.000 (10/10) |
| Cohen's kappa | **undefined** |
| label balance | 10 true, 0 false |
| 95% upper bound on error rate | 27.8% |

Kappa is undefined rather than perfect. Every label falls in one class, so chance
agreement is 1.0, the denominator vanishes, and the sample carries no information
about agreement *above* chance. `cohens_kappa` returns `None` here and the CLI
prints "undefined"; reporting 1.0 would be the same species of overclaim this
tool was built to object to.

The one-class balance is structural rather than unlucky. A record only carries an
evidence span when the brand was marked as mentioned, so the pool of
normalisation-recovered pairs contains nothing but positives. **This sample can
detect false positives, and found none in 10. It cannot detect false negatives**,
because a brand the extractor missed in a recovered run was never eligible to be
drawn. Recall on the recovered runs is unmeasured. Sampling whole recovered runs
and labelling all four brands in each would close that gap; it is not implemented.

The two samples are reported separately and never merged. Averaging them would
hide whichever is worse, which is the reason for drawing the second one.

## Limitations

**Six of eight cells sit near the ceiling.** The three incumbents run 0.83–0.96,
and for six cells the MDE exceeds the headroom between the observed rate and 1.0.
Those MDEs are interpretable downward only and are flagged `†` in the generated
table. A symmetric absolute-scale threshold is the wrong summary that close to a
bound.

**Three intervals terminate at exactly 1.00**, where the percentile bootstrap is
degenerate: the interval end is an artifact of the bound, not an estimate. The
largest design-effect residual in the batch (−0.60, Asana/anthropic) is such a
cell, and it is over-determined: two mechanisms push it down, not one. The
degenerate cluster interval compresses the numerator, and in the denominator the
Wilson interval runs wider than the asymptotic one near the bound — for that cell
by enough to scale the design effect by 0.863 on its own. Neither mechanism has
to be doing all the work, and this batch cannot apportion them. Logit-scale
intervals or a BCa bootstrap would be the principled fix; neither is implemented.

**The design effect is itself an estimate.** It is computed from 20 prompts and
carries its own sampling error. The worked example in the design note — ten
prompts with no true between-prompt variance returning 1.49 rather than 1.00 —
shows the scale of that noise. No interval is reported on the design effect, and
the per-cell figures should not be read to two decimal places. The agreement with
the `1 + (k − 1)·ICC` prediction, a mean absolute residual of 0.24 across eight
cells, is evidence that the estimates are not badly off; it is not a substitute
for an interval. Bootstrapping the design effect is the obvious next step.

**One category, two providers, one date.** Nothing here transfers to another
category, and the figures are specific to the model versions recorded in
`model_returned` on 2026-07-28.

**Direct APIs, not consumer products.** Consumer chat products add system prompts,
retrieval, memory, personalisation and routing that the raw API does not. These
measurements describe the API surface.

**Mention rate is not a business outcome**, and nothing here is causal. The tool
measures whether a change is larger than noise; it says nothing about what caused
one.

**The permutation tests in the failure diagnosis are descriptive.** They were
computed on a sample not designed to test them, over correlated features, with no
multiplicity correction and no significance threshold applied.
