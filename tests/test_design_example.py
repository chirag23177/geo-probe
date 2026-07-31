"""The worked example in docs/DESIGN.md, pinned.

Every figure quoted in the "Two datasets that are not the same" section is
regenerated here to the precision it is printed at. If the prose and the code
ever disagree, this fails.

The earlier version of that example hand-constructed dataset A as ten prompts
each landing on exactly 3 of 5 -- which is not a sample anyone could draw (its
probability is under one in a million) and which produced a zero-width cluster
interval, contradicting test_cluster_bootstrap_matches_naive_with_no_clustering
in test_stats.py. Dataset A is now an actual binomial draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo_probe.stats import cluster_bootstrap, design_effect, wilson_interval

# Documented in docs/DESIGN.md: the smallest seed whose ten draws sum to exactly
# 30, so both datasets share a pooled estimate of 30/50 and therefore an
# identical pooled interval. Conditioning on the margin makes the comparison
# fair; it is not a search over outcomes.
EXAMPLE_SEED = 0
EXAMPLE_COUNTS = [3, 4, 5, 5, 2, 2, 3, 2, 3, 1]
BOOTSTRAP_SEED = 0
ITERS = 2000

DATASET_B = [1.0] * 6 + [0.0] * 4


def _dataset_a() -> list[float]:
    counts = np.random.default_rng(EXAMPLE_SEED).binomial(5, 0.6, size=10)
    return (counts / 5).tolist()


def test_seed_zero_is_the_smallest_qualifying_seed():
    """Smallest seed summing to 30, and not degenerate in the other direction
    (all ten prompts identical would be just as unrepresentative)."""
    for seed in range(EXAMPLE_SEED):
        counts = np.random.default_rng(seed).binomial(5, 0.6, size=10)
        assert counts.sum() != 30 or len(set(counts.tolist())) == 1, (
            f"seed {seed} also qualifies and is smaller than {EXAMPLE_SEED}"
        )
    counts = np.random.default_rng(EXAMPLE_SEED).binomial(5, 0.6, size=10)
    assert counts.sum() == 30
    assert len(set(counts.tolist())) > 1


def test_dataset_a_counts_match_the_doc():
    counts = np.random.default_rng(EXAMPLE_SEED).binomial(5, 0.6, size=10)
    assert counts.tolist() == EXAMPLE_COUNTS


def test_both_datasets_have_the_same_pooled_estimate():
    """The premise of the comparison: identical margin, identical pooled interval."""
    assert sum(EXAMPLE_COUNTS) == 30
    assert sum(DATASET_B) * 5 == 30
    assert sum(_dataset_a()) / 10 == pytest.approx(0.6)
    assert sum(DATASET_B) / 10 == pytest.approx(0.6)


def test_pooled_interval_quoted_in_the_doc():
    lo, hi = wilson_interval(30, 50)
    assert (round(lo, 2), round(hi, 2)) == (0.46, 0.72)
    assert round(hi - lo, 2) == 0.26


def test_dataset_a_figures_quoted_in_the_doc():
    """Prompts agree apart from ordinary sampling scatter, so the cluster
    interval lands near the pooled one."""
    boot = cluster_bootstrap(_dataset_a(), iters=ITERS, seed=BOOTSTRAP_SEED)
    naive = wilson_interval(30, 50)
    assert (round(boot.ci95[0], 2), round(boot.ci95[1], 2)) == (0.44, 0.76)
    assert round(boot.ci95[1] - boot.ci95[0], 2) == 0.32
    assert round(design_effect(boot.ci95, naive), 2) == 1.49


def test_dataset_b_figures_quoted_in_the_doc():
    """Prompts disagree completely, so the interval is dominated by which
    prompts were drawn."""
    boot = cluster_bootstrap(DATASET_B, iters=ITERS, seed=BOOTSTRAP_SEED)
    naive = wilson_interval(30, 50)
    assert (round(boot.ci95[0], 2), round(boot.ci95[1], 2)) == (0.30, 0.90)
    assert round(boot.ci95[1] - boot.ci95[0], 2) == 0.60
    assert round(design_effect(boot.ci95, naive), 2) == 5.24


def test_the_example_carries_the_argument_it_claims_to():
    """The point of the section: when prompts agree the two methods roughly
    agree; the gap opens only when prompts disagree. B's interval must be
    substantially wider than A's, and A's must be within reach of the pooled one."""
    naive = wilson_interval(30, 50)
    naive_w = naive[1] - naive[0]
    a = cluster_bootstrap(_dataset_a(), iters=ITERS, seed=BOOTSTRAP_SEED)
    b = cluster_bootstrap(DATASET_B, iters=ITERS, seed=BOOTSTRAP_SEED)
    a_w = a.ci95[1] - a.ci95[0]
    b_w = b.ci95[1] - b.ci95[0]

    assert b_w > 1.5 * a_w, "B must be clearly wider than A or the example fails"
    assert a_w < 1.5 * naive_w, "A must stay near the pooled width"
    # And A must not be the impossible zero-width result the old example produced.
    assert a_w > 0.2, "a real ten-prompt draw has non-trivial scatter"


def test_dataset_a_deff_is_above_one_because_ten_prompts_is_a_small_sample():
    """Honest framing for the doc: a single ten-prompt draw does not land at
    deff = 1 exactly. The estimator converges there -- see
    test_cluster_bootstrap_matches_naive_with_no_clustering, which uses 400
    prompts and gets 0.94-1.06 -- but ten prompts carry real scatter in the
    variance estimate itself."""
    naive = wilson_interval(30, 50)
    deff = design_effect(cluster_bootstrap(_dataset_a(), ITERS, BOOTSTRAP_SEED).ci95, naive)
    assert 1.0 < deff < 2.0
