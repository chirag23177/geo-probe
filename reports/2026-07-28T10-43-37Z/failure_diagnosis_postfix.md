# Extraction failure diagnosis - 2026-07-28T10-43-37Z

What distinguishes the runs whose grading failed from those that succeeded, within each provider. p-values come from a two-sided permutation test on the difference of means (10,000 relabellings). **No significance threshold is applied and none should be read in** -- these are descriptive, computed on a sample that was not designed to test them, and the features are correlated with each other.

## anthropic

1 failed, 99 passed, of 100 usable runs.

| feature | mean (failed) | mean (passed) | median (failed) | median (passed) | p |
|---|---|---|---|---|---|
| response length (chars) | 1889.0 | 1972.7 | 1889.0 | 1883.0 | 0.8154 |
| `**` occurrences | 16.0 | 26.0 | 16.0 | 24.0 | 0.3811 |
| `[n]` citation markers | 0.0 | 0.0 | 0.0 | 0.0 | 1.0000 |
| newline count | 18.0 | 37.2 | 18.0 | 37.0 | 0.0407 |
| tracked brands string-matched | 2.0 | 3.1 | 2.0 | 3.0 | 0.1137 |

Failures by prompt (prompts with at least one failure):

| prompt | failed | of |
|---|---|---|
| p16 | 1 | 5 |

## perplexity

1 failed, 99 passed, of 100 usable runs.

| feature | mean (failed) | mean (passed) | median (failed) | median (passed) | p |
|---|---|---|---|---|---|
| response length (chars) | 4374.0 | 3545.6 | 4374.0 | 4019.0 | 0.3218 |
| `**` occurrences | 52.0 | 57.4 | 52.0 | 56.0 | 0.8455 |
| `[n]` citation markers | 58.0 | 46.3 | 58.0 | 44.0 | 0.5906 |
| newline count | 45.0 | 55.4 | 45.0 | 64.0 | 0.7793 |
| tracked brands string-matched | 3.0 | 3.2 | 3.0 | 3.0 | 1.0000 |

Failures by prompt (prompts with at least one failure):

| prompt | failed | of |
|---|---|---|
| p10 | 1 | 5 |

## How to read this

- A large gap in a rendering feature (bold markers, citation markers, newlines) supports the formatting-mismatch explanation: the extractor copied rendered text where the source carried markup.
- A large gap in `tracked brands string-matched` would instead mean the failures are concentrated in responses that name more brands, which is a content difference and a real bias risk for the mention-rate estimates.
- Failures concentrated in a few prompts would mean the loss is not spread evenly across the cluster structure, which matters because the prompt is the unit of analysis.
