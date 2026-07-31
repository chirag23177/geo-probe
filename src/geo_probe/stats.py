"""Statistics. This module is the point of the tool; everything else is plumbing.

The central claim: the unit of analysis is the *prompt*, not the run. The k reps
of one prompt are correlated -- a prompt that never surfaces a brand will likely
never surface it in any rep -- so treating (num_prompts * k) runs as that many
independent Bernoulli trials understates the standard error. We therefore compute

  * mention_rate  = mean over prompts of the per-prompt rate p_j
  * ci95_cluster  = percentile interval from a bootstrap that resamples *prompts*
  * ci95_naive_wrong = Wilson interval on the pooled runs, reported so the reader
                       can see the gap
  * design_effect = (cluster width / naive width)^2

No scipy. The normal quantiles below are the only distributional constants used,
and they are written out so the arithmetic is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Normal quantiles, hard-coded so no distribution library is needed and so the
# MDE arithmetic can be checked by hand.
Z_975 = 1.959963984540054  # two-sided alpha = 0.05
Z_POWER_80 = 0.8416212335729143  # power = 0.80
MDE_Z_SUM = Z_975 + Z_POWER_80  # 2.8015852... -- the 2.802 in the spec

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


# --------------------------------------------------------------------------
# Wilson score interval (the naive, wrong-for-this-design interval)
# --------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z_975) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Six lines, by hand.

    This is the interval a naive GEO dashboard reports: it assumes all n runs are
    independent trials. For clustered data it is too narrow, which is exactly why
    we report it alongside the cluster interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------
# Cluster bootstrap over prompts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    """mean(p_j) on the observed data -- not the mean of the bootstrap draws."""
    ci95: tuple[float, float]
    se: float
    """Standard deviation of the bootstrap distribution of mean(p_j)."""
    draws: np.ndarray


def cluster_bootstrap(
    p_by_prompt: Sequence[float],
    iters: int = 2000,
    seed: int = 0,
) -> BootstrapResult:
    """Resample prompts with replacement; recompute mean(p_j) each time.

    Resampling whole prompts (not individual runs) is what carries the
    within-prompt correlation into the interval.
    """
    p = np.asarray(p_by_prompt, dtype=float)
    m = p.size
    if m == 0:
        raise ValueError("need at least one prompt")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(iters, m))
    draws = p[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return BootstrapResult(
        point=float(p.mean()),
        ci95=(float(lo), float(hi)),
        se=float(draws.std(ddof=1)),
        draws=draws,
    )


def se_pooled_binomial(successes: int, n: int) -> float:
    """The standard error a naive dashboard would use: pooled runs, treated as
    independent Bernoulli trials.

    Deliberately the *wrong* estimator. It exists so the naive MDE can be
    computed and compared like-for-like against the cluster MDE -- comparing a
    cluster MDE against half a naive CI width compares two different quantities.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    return (p * (1 - p) / n) ** 0.5


def design_effect(cluster_ci: tuple[float, float], naive_ci: tuple[float, float]) -> float:
    """(cluster CI width / naive CI width)^2.

    ~1 means the clustering costs nothing; >1 means the naive interval is
    overstating precision by that factor in variance terms.
    """
    naive_width = naive_ci[1] - naive_ci[0]
    if naive_width <= 0:
        raise ValueError("naive interval has non-positive width")
    return ((cluster_ci[1] - cluster_ci[0]) / naive_width) ** 2


# --------------------------------------------------------------------------
# Variance decomposition (one-way random effects, unbalanced-safe)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceDecomposition:
    sigma2_between: float
    sigma2_within: float
    between_share: float | None
    within_share: float | None
    """None when total variance is exactly zero (every run identical) -- there is
    no share to report and inventing 0.5/0.5 would be a lie."""


def variance_decomposition(
    counts: Sequence[int],
    reps: Sequence[int],
) -> VarianceDecomposition:
    """Split total variance into between-prompt and within-prompt components.

    Standard one-way ANOVA estimator on the Bernoulli run outcomes, grouped by
    prompt. `counts[j]` is the number of reps of prompt j that mentioned the
    brand; `reps[j]` is the number of usable reps of prompt j. Handles unequal
    reps (some runs may have failed) via the usual k0 correction, which reduces
    to k when the design is balanced.

    Interpretation: if between-prompt dominates, buy more prompts. If
    within-prompt dominates, buy more reps.
    """
    c = np.asarray(counts, dtype=float)
    n = np.asarray(reps, dtype=float)
    if c.shape != n.shape:
        raise ValueError("counts and reps must have the same length")
    m = c.size
    if m < 2:
        raise ValueError("need at least two prompts to decompose variance")
    if np.any(n < 1):
        raise ValueError("every prompt needs at least one usable rep")
    N = float(n.sum())
    if N <= m:
        raise ValueError("need more usable runs than prompts to decompose variance")

    p_j = c / n
    p_bar = float(c.sum() / N)  # run-weighted grand mean

    # For binary values, the within-group sum of squares is exactly n_j*p_j*(1-p_j).
    ssw = float((n * p_j * (1 - p_j)).sum())
    ssb = float((n * (p_j - p_bar) ** 2).sum())

    msw = ssw / (N - m)
    msb = ssb / (m - 1)

    # k0 == k when balanced.
    k0 = (N - float((n**2).sum()) / N) / (m - 1)
    sigma2_between = max(0.0, (msb - msw) / k0)
    sigma2_within = msw

    total = sigma2_between + sigma2_within
    if total <= 0:
        return VarianceDecomposition(sigma2_between, sigma2_within, None, None)
    return VarianceDecomposition(
        sigma2_between,
        sigma2_within,
        sigma2_between / total,
        sigma2_within / total,
    )


# --------------------------------------------------------------------------
# Minimum detectable effect (two-sample, week A vs week B)
# --------------------------------------------------------------------------


def mde_abs(se_cluster: float, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER) -> float:
    """Smallest absolute change in mention rate detectable at this sample size.

    Two-sample: comparing week A to week B, each measured with the same design,
    so se_diff = sqrt(2) * se_cluster. An MDE without its alpha and power is
    meaningless, which is why the caller must carry both into the output record.
    """
    if se_cluster < 0:
        raise ValueError("se_cluster must be non-negative")
    if (alpha, power) != (DEFAULT_ALPHA, DEFAULT_POWER):
        raise NotImplementedError(
            "only alpha=0.05, power=0.80 are supported; add the z quantile explicitly "
            "rather than pulling in a distribution library"
        )
    se_diff = (2.0**0.5) * se_cluster
    return MDE_Z_SUM * se_diff


# --------------------------------------------------------------------------
# Boundary handling: an absolute-scale MDE on a bounded parameter
# --------------------------------------------------------------------------

BOUNDARY_TOL = 0.005


def mde_interpretability(mention_rate: float, mde: float) -> str:
    """Whether a symmetric absolute MDE is interpretable in both directions.

    A mention rate is bounded in [0, 1]. A cell at 0.95 with an MDE of 0.14 has
    only 0.05 of headroom upward, so "detectable change of 14pp" is not a
    statement that can be made about an increase. Returns one of
    "both" / "down_only" / "up_only" / "neither".
    """
    if not 0.0 <= mention_rate <= 1.0:
        raise ValueError("mention_rate must be in [0, 1]")
    if mde < 0:
        raise ValueError("mde must be non-negative")
    fits_up = mde <= 1.0 - mention_rate
    fits_down = mde <= mention_rate
    if fits_up and fits_down:
        return "both"
    if fits_down:
        return "down_only"
    if fits_up:
        return "up_only"
    return "neither"


def boundary_flag(ci: tuple[float, float], tol: float = BOUNDARY_TOL) -> str:
    """Which ends of an interval sit on the parameter boundary.

    A percentile bootstrap that piles up at exactly 0 or 1 is degenerate there:
    the interval end is an artifact of the bound, not an estimate.
    Returns "none" / "lower" / "upper" / "both".
    """
    at_lower = ci[0] <= tol
    at_upper = ci[1] >= 1.0 - tol
    if at_lower and at_upper:
        return "both"
    if at_lower:
        return "lower"
    if at_upper:
        return "upper"
    return "none"


# --------------------------------------------------------------------------
# Design-effect consistency check
# --------------------------------------------------------------------------


def deff_prediction(k_effective: float, icc: float) -> float:
    """deff ~ 1 + (k_eff - 1) * ICC, the standard balanced-design relation.

    Comparing this against the measured design effect is a check on the whole
    clustering story: if the two disagree badly, either the ICC estimate or the
    bootstrap is wrong. Using k_effective rather than the nominal k matters when
    runs have been excluded -- dropping runs lowers the realised cluster size and
    mechanically depresses the design effect.
    """
    if k_effective < 1:
        raise ValueError("k_effective must be at least 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be in [0, 1]")
    return 1.0 + (k_effective - 1.0) * icc
