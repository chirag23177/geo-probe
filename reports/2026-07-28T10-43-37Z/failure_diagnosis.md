# Extraction failure diagnosis - 2026-07-28T10-43-37Z

What distinguishes the runs whose grading failed from those that succeeded, within each provider. p-values come from a two-sided permutation test on the difference of means (10,000 relabellings). **No significance threshold is applied and none should be read in** -- these are descriptive, computed on a sample that was not designed to test them, and the features are correlated with each other.

## anthropic

0 failed, 100 passed, of 100 usable runs.

Only one group is present, so there is nothing to compare. No failures on this provider.

## perplexity

29 failed, 71 passed, of 100 usable runs.

| feature | mean (failed) | mean (passed) | median (failed) | median (passed) | p |
|---|---|---|---|---|---|
| response length (chars) | 3901.0 | 3412.1 | 4168.0 | 3956.0 | 0.0303 |
| `**` occurrences | 61.8 | 55.5 | 60.0 | 54.0 | 0.1697 |
| `[n]` citation markers | 47.7 | 45.9 | 47.0 | 44.0 | 0.6848 |
| newline count | 63.3 | 52.0 | 68.0 | 58.0 | 0.0410 |
| tracked brands string-matched | 3.3 | 3.1 | 3.0 | 3.0 | 0.3948 |

Failures by prompt (prompts with at least one failure):

| prompt | failed | of |
|---|---|---|
| p18 | 4 | 5 |
| p03 | 3 | 5 |
| p02 | 2 | 5 |
| p08 | 2 | 5 |
| p09 | 2 | 5 |
| p13 | 2 | 5 |
| p14 | 2 | 5 |
| p16 | 2 | 5 |
| p20 | 2 | 5 |
| p01 | 1 | 5 |
| p04 | 1 | 5 |
| p05 | 1 | 5 |
| p06 | 1 | 5 |
| p07 | 1 | 5 |
| p10 | 1 | 5 |
| p11 | 1 | 5 |
| p15 | 1 | 5 |

## How to read this

- A large gap in a rendering feature (bold markers, citation markers, newlines) supports the formatting-mismatch explanation: the extractor copied rendered text where the source carried markup.
- A large gap in `tracked brands string-matched` would instead mean the failures are concentrated in responses that name more brands, which is a content difference and a real bias risk for the mention-rate estimates.
- Failures concentrated in a few prompts would mean the loss is not spread evenly across the cluster structure, which matters because the prompt is the unit of analysis.
