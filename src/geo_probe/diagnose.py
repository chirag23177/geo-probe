"""Characterise what distinguishes the runs that failed grading.

Offline: the raw responses are already on disk, so this makes no API calls. It
exists to turn "bias of unknown direction" into a measured statement, and its
output stays in the repo whether or not the failures are later recovered.

Comparisons are always within a provider. Failure rates differ sharply between
measurement surfaces, so pooling would confound "this feature predicts failure"
with "this provider fails more".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .schemas import Brand, ExtractRecord, RunRecord

_CITATION_MARKER = re.compile(r"\[\d+\]")

PERMUTATION_ITERS = 10_000


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def response_features(run: RunRecord, brands: Sequence[Brand]) -> dict[str, float]:
    """Surface properties of a response that might predict a grading failure.

    All are rendering-related except the brand count, which is included because
    a response naming more brands gives the extractor more spans to get wrong.
    """
    text = run.raw_response or ""
    lowered = text.lower()
    return {
        "length_chars": float(len(text)),
        "bold_markers": float(text.count("**")),
        "citation_markers": float(len(_CITATION_MARKER.findall(text))),
        "newlines": float(text.count("\n")),
        "brands_present": float(sum(1 for b in brands if b.name.lower() in lowered)),
    }


FEATURE_LABELS = {
    "length_chars": "response length (chars)",
    "bold_markers": "`**` occurrences",
    "citation_markers": "`[n]` citation markers",
    "newlines": "newline count",
    "brands_present": "tracked brands string-matched",
}


# --------------------------------------------------------------------------
# Permutation test, written out
# --------------------------------------------------------------------------


def permutation_test_diff_means(
    a: Sequence[float],
    b: Sequence[float],
    iters: int = PERMUTATION_ITERS,
    seed: int = 0,
) -> float:
    """Two-sided p-value for a difference in means, by relabelling.

    The null is that the group labels carry no information, so the reference
    distribution is built by shuffling the labels and recomputing the difference.
    No distributional assumption, no scipy.

    Uses the (count + 1) / (iters + 1) form, which cannot return exactly zero --
    a p-value of 0 from a finite number of permutations is not a thing.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("both groups need at least one observation")
    observed = abs(float(x.mean()) - float(y.mean()))

    pooled = np.concatenate([x, y])
    n_a = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        diff = abs(float(pooled[:n_a].mean()) - float(pooled[n_a:].mean()))
        if diff >= observed:
            count += 1
    return (count + 1) / (iters + 1)


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    mean_failed: float
    mean_passed: float
    median_failed: float
    median_passed: float
    p_value: float


@dataclass(frozen=True)
class ProviderDiagnosis:
    provider: str
    n_failed: int
    n_passed: int
    comparisons: list[FeatureComparison]
    prompt_failures: list[tuple[str, int, int]]
    """(prompt_id, failures, total) sorted by failure count descending."""


def diagnose_provider(
    provider: str,
    runs: Sequence[RunRecord],
    failed_run_ids: set[str],
    brands: Sequence[Brand],
    iters: int = PERMUTATION_ITERS,
    seed: int = 0,
) -> ProviderDiagnosis:
    rows = [r for r in runs if r.provider == provider and r.ok]
    failed = [r for r in rows if r.run_id in failed_run_ids]
    passed = [r for r in rows if r.run_id not in failed_run_ids]

    comparisons: list[FeatureComparison] = []
    if failed and passed:
        for i, feature in enumerate(FEATURE_LABELS):
            a = [response_features(r, brands)[feature] for r in failed]
            b = [response_features(r, brands)[feature] for r in passed]
            comparisons.append(
                FeatureComparison(
                    feature=feature,
                    mean_failed=float(np.mean(a)),
                    mean_passed=float(np.mean(b)),
                    median_failed=float(np.median(a)),
                    median_passed=float(np.median(b)),
                    # Distinct seed per feature so the reference distributions
                    # are not identical across tests.
                    p_value=permutation_test_diff_means(a, b, iters=iters, seed=seed + i),
                )
            )

    totals: dict[str, int] = {}
    fails: dict[str, int] = {}
    for r in rows:
        totals[r.prompt_id] = totals.get(r.prompt_id, 0) + 1
        if r.run_id in failed_run_ids:
            fails[r.prompt_id] = fails.get(r.prompt_id, 0) + 1
    prompt_failures = sorted(
        ((pid, fails.get(pid, 0), totals[pid]) for pid in totals),
        key=lambda t: (-t[1], t[0]),
    )

    return ProviderDiagnosis(
        provider=provider,
        n_failed=len(failed),
        n_passed=len(passed),
        comparisons=comparisons,
        prompt_failures=prompt_failures,
    )


def failed_run_ids(extracts: Sequence[ExtractRecord]) -> set[str]:
    return {e.run_id for e in extracts if not e.ok}


def diagnose(
    runs: Sequence[RunRecord],
    extracts: Sequence[ExtractRecord],
    brands: Sequence[Brand],
    iters: int = PERMUTATION_ITERS,
    seed: int = 0,
) -> list[ProviderDiagnosis]:
    bad = failed_run_ids(extracts)
    return [
        diagnose_provider(p, runs, bad, brands, iters=iters, seed=seed)
        for p in sorted({r.provider for r in runs})
    ]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_diagnosis(diagnoses: Sequence[ProviderDiagnosis], batch_id: str, iters: int) -> str:
    lines: list[str] = []
    lines.append(f"# Extraction failure diagnosis - {batch_id}")
    lines.append("")
    lines.append(
        "What distinguishes the runs whose grading failed from those that succeeded, within each "
        "provider. p-values come from a two-sided permutation test on the difference of means "
        f"({iters:,} relabellings). **No significance threshold is applied and none should be "
        "read in** -- these are descriptive, computed on a sample that was not designed to test "
        "them, and the features are correlated with each other."
    )
    lines.append("")

    for d in diagnoses:
        lines.append(f"## {d.provider}")
        lines.append("")
        lines.append(f"{d.n_failed} failed, {d.n_passed} passed, of {d.n_failed + d.n_passed} usable runs.")
        lines.append("")
        if not d.comparisons:
            lines.append(
                "Only one group is present, so there is nothing to compare. "
                + ("No failures on this provider." if d.n_failed == 0 else "No successes on this provider.")
            )
            lines.append("")
            continue

        lines.append("| feature | mean (failed) | mean (passed) | median (failed) | median (passed) | p |")
        lines.append("|---|---|---|---|---|---|")
        for c in d.comparisons:
            lines.append(
                f"| {FEATURE_LABELS[c.feature]} | {c.mean_failed:.1f} | {c.mean_passed:.1f} | "
                f"{c.median_failed:.1f} | {c.median_passed:.1f} | {c.p_value:.4f} |"
            )
        lines.append("")

        worst = [t for t in d.prompt_failures if t[1] > 0]
        if worst:
            lines.append("Failures by prompt (prompts with at least one failure):")
            lines.append("")
            lines.append("| prompt | failed | of |")
            lines.append("|---|---|---|")
            for pid, f, total in worst:
                lines.append(f"| {pid} | {f} | {total} |")
            lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- A large gap in a rendering feature (bold markers, citation markers, newlines) supports "
        "the formatting-mismatch explanation: the extractor copied rendered text where the source "
        "carried markup."
    )
    lines.append(
        "- A large gap in `tracked brands string-matched` would instead mean the failures are "
        "concentrated in responses that name more brands, which is a content difference and a "
        "real bias risk for the mention-rate estimates."
    )
    lines.append(
        "- Failures concentrated in a few prompts would mean the loss is not spread evenly across "
        "the cluster structure, which matters because the prompt is the unit of analysis."
    )
    lines.append("")
    return "\n".join(lines)
