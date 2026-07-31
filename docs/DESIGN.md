# Design note

A dashboard reports that your brand's visibility went from 19% to 24% this week.
Someone asks whether that is real. On most such dashboards there is no way to
answer, because the number that would answer it — how large a change has to be
before the measurement can see it — is not computed.

## The problem

Buyers increasingly ask a language model for a shortlist instead of running a
search. Being absent from that shortlist is a commercial problem, so a market for
measuring it appeared: run some prompts, count how often each brand appears,
report a percentage, track it weekly.

But a language model does not return the same answer twice. The percentage is not
a fact that was looked up; it is a measurement, and it carries noise. It is a
bathroom scale that wobbles by four pounds, used to detect a two-pound change —
except the wobble is not a defect, since the variation is the model's real
behaviour.

That LLM answers vary is well documented, and this repository does not claim it
as a discovery. The sensible response, which serious tools already make, is to
run a panel of prompts several times each and average. That is the right
instinct. The gap is what happens next: turning those repeats into an interval.

## Two datasets that are not the same

Ten prompts, each asked five times, brand appears in 30 of the 50 runs.

**Dataset A.** Every prompt mentions the brand in exactly 3 of its 5 repetitions.

**Dataset B.** Six prompts mention it in all 5. Four never mention it at all.

Both are 60%. Pool the 50 runs as independent trials and both get the same
interval: `[0.46, 0.72]`.

They are not the same measurement. In A the prompts agree, so which ten you
picked barely matters — swap them and you would expect 60% again. In B the answer
is entirely determined by the draw: six of these ten happen to be prompts this
brand owns, and a different ten could as easily give 30% or 90%.

Resampling whole prompts makes the difference visible: cluster interval widths of
0.00 for A and 0.60 for B, against 0.26 pooled for both. (Reproduce by running
`stats.cluster_bootstrap` on `[0.6]*10` and `[1.0]*6 + [0.0]*4`.)

The pooled interval cannot see this because it treats the prompt panel as the
whole population. It is a sample from the population of things buyers ask, and
uncertainty about that population is what matters.

So the unit of analysis is the prompt, not the run.

## What that costs, measured

For each brand and provider this tool computes a per-prompt mention rate, takes
their mean, and builds the interval by resampling prompts with replacement 2,000
times. The design effect — cluster variance over pooled variance — says how much
precision the pooled method was imagining.

Here it averages **2.89**, from 1.90 to 3.49. Three readings of that one number:
the interval is √2.89 ≈ **1.7 times wider** (widths go as the square root, so not
2.89 times); matching the precision the pooled interval claims would take about
**289 runs per cell** rather than 100; those 100 runs carry about as much
information as **35 independent ones**.

The pooled interval stays in the output as `ci95_naive_wrong`. It is the thing
being argued with, and an argument is easier to check with both side by side.

## Why this is not just what the bootstrap happened to say

For a balanced design the design effect should be near `1 + (k − 1)·ICC`, where k
is repetitions per prompt and the ICC is the share of variance sitting between
prompts rather than within them.

Both ends make sense in words. At ICC 0, repeating a prompt is as informative as
asking a new one, nothing is lost, and the design effect is 1. At ICC 1, every
repetition returns the identical answer, the four extra runs add nothing, and the
effective sample size collapses from the number of runs to the number of prompts.

Measured ICC averages 0.49, predicting 2.94 against 2.89 observed; per cell the
two differ by 0.24 on average, in both directions.

That check earned its place. On the first pass, 29 runs on one provider were
being dropped by the extractor, lowering realised repetitions per prompt from 5
to 3.55 — and a smaller realised cluster size mechanically depresses the design
effect. Those cells sat below their k = 5 prediction and fitted once the true
figure was used. The discrepancy is what surfaced the extraction bug. Without it
the batch would have reported a design effect about 16% too small, in the
direction that flatters the pooled method.

## The number that answers the question

The minimum detectable effect is the smallest change this design can distinguish
from noise. It is a two-sample quantity: comparing this week against last week
involves two noisy measurements, so the standard error of the difference is √2
times that of one.

It needs two terms. A significance term (1.96) sets how much evidence is required
before calling a change real; a power term (0.84) sets how often a change of that
size is actually caught. An effect sitting exactly on the 1.96 threshold is caught
about half the time, because the estimate lands above the line about half the
time — a threshold with no power term describes a coin flip, not a detection.

Two consequences. Half a confidence interval is roughly half the real detection
threshold, so checking whether error bars overlap is a much weaker test than the
one being claimed. And on this batch, like for like, one cell's pooled MDE is
14.7pp against a cluster MDE of 27.1pp: every change in between is one the pooled
method calls detectable and this design cannot detect.

## Decisions, and what each cost

**Prompt as the unit of analysis.** Intervals roughly double in width and the MDE
goes from sounding usable to sounding discouraging. That is the finding, not a
side effect.

**Cluster bootstrap over a closed-form correction or a mixed model.** The closed
form assumes balance, which breaks the moment runs are excluded. A mixed model
would be more efficient and extend to covariates, at the cost of distributional
assumptions and a dependency. The bootstrap is a dozen readable lines — and is
worse near the boundary, which shows up in this data.

**Wilson rather than the normal approximation for the pooled baseline.** The
normal approximation misbehaves near 0 and 1, and most cells sit above 0.83. The
weaker baseline would have inflated the design effect and flattered the argument.

**A separate extraction pass at temperature 0.** Letting the probe model
self-report structured output would be cheaper, and would couple the measurement
to the thing measured. Grading must be reproducible even though the thing graded
must not be; separating them also means a grading bug costs a cheap re-run rather
than the probe budget.

**Rank and citation matching computed in code.** The model decides only whether a
brand was mentioned and supplies evidence; ordering is then mechanical, so the
rank convention is enforced rather than trusted. A citation counts only when it
resolves to the brand's own registrable domain.

**The extractor is blinded** — never sees the prompt, provider, repetition index,
or which brand is focal, and brand order is randomised per call. It cannot favour
a brand it cannot identify as the one being studied.

**Verbatim evidence spans as a hallucination guard.** Cheap, and it catches
invention. Applied strictly it also rejects a model that copies rendered text
where the source had markdown. Fixing that meant making the comparison invariant
to rendering — a genuine weakening of the guard, stated rather than absorbed.

**More prompts rather than more repetitions.** At a fixed budget of N runs the
variance is `(k·var_between + var_within)/N`, which falls as k falls. Taken
literally that says k = 1. But with one run per prompt you cannot see a prompt
that answers differently on different attempts, and that instability is worth
measuring — here it reaches half the prompts for some brands. So k = 5 buys an
instability metric and pays in precision. The least comfortable decision here.

**Providers are never pooled.** A direct call measures what the model carries in
its weights; a search-grounded call measures what retrieval surfaced today.
Averaging them describes neither, so the code raises instead.

**A neutral product category** — project management software — so the method is
the subject rather than any particular scoreboard.

**Thinking disabled on the direct provider**, for cost and to keep the output a
plain answer. It moves the measurement further from a consumer chat product, a
gap this design already does not close.

## Scope

In: how much a brand's mention rate varies across prompts and repetitions, for two
model versions in one category on one date; how wide the interval honestly is; and
how large a week-over-week change must be for this design to detect it.

Out: other categories; how any particular commercial tool is implemented, since
the standard approach is inferred from how results are publicly reported and not
from reading anyone's code; sentiment accuracy, unvalidated; rank *accuracy* as
opposed to rank convention; consumer chat products as opposed to direct APIs; and
any causal account of why a rate moved.

## Next

Logit-scale intervals, or a BCa bootstrap, for cells whose intervals hit the
ceiling — six of eight sit above 0.83 and the percentile method is degenerate
there. A brand set nearer 50%, where binomial variance is largest. Recall
validation for the extractor, which the current gold sets cannot provide. And more
prompts rather than more repetitions.
