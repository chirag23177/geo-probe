"""Tests 1-4: the statistics.

Tests 2 and 3 are the ones that matter. Test 2 shows the cluster method widens
the interval when the reps are correlated; test 3 shows it does NOT widen it when
they aren't -- which is what proves the method isn't just inflating everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from geo_probe.stats import (
    MDE_Z_SUM,
    Z_975,
    Z_POWER_80,
    boundary_flag,
    cluster_bootstrap,
    deff_prediction,
    design_effect,
    mde_abs,
    mde_interpretability,
    se_pooled_binomial,
    variance_decomposition,
    wilson_interval,
)


# --------------------------------------------------------------------------
# Test 1: Wilson interval matches known reference values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "successes,n,expected",
    [
        (12, 20, (0.3866, 0.7812)),
        (0, 10, (0.0000, 0.2775)),
        (10, 10, (0.7225, 1.0000)),
        (5, 10, (0.2366, 0.7634)),
    ],
)
def test_wilson_matches_reference_values(successes, n, expected):
    lo, hi = wilson_interval(successes, n)
    assert lo == pytest.approx(expected[0], abs=5e-4)
    assert hi == pytest.approx(expected[1], abs=5e-4)


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


# --------------------------------------------------------------------------
# Test 2: high ICC -> cluster interval wider than naive, design effect > 1
# --------------------------------------------------------------------------


def test_cluster_bootstrap_widens_under_high_icc():
    """Maximum clustering: each prompt either always mentions the brand or never
    does. The 5 reps of a prompt carry no information beyond the first, so the
    effective sample size is 40 prompts, not 200 runs."""
    m, k = 40, 5
    counts = [k if j % 2 == 0 else 0 for j in range(m)]  # p_j in {0, 1}
    reps = [k] * m
    p_by_prompt = [c / k for c in counts]

    boot = cluster_bootstrap(p_by_prompt, iters=2000, seed=11)
    naive = wilson_interval(sum(counts), m * k)
    deff = design_effect(boot.ci95, naive)

    cluster_width = boot.ci95[1] - boot.ci95[0]
    naive_width = naive[1] - naive[0]

    assert boot.point == pytest.approx(0.5, abs=1e-12)
    assert cluster_width > naive_width
    assert deff > 1.0
    # Every rep beyond the first is redundant, so deff should land near k.
    assert 3.5 < deff < 7.0

    vd = variance_decomposition(counts, reps)
    assert vd.within_share == pytest.approx(0.0, abs=1e-12)
    assert vd.between_share == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Test 3: zero between-prompt variance -> design effect ~ 1
# --------------------------------------------------------------------------


def test_cluster_bootstrap_matches_naive_with_no_clustering():
    """No prompt effect: every run is an iid Bernoulli(0.5) draw. The cluster
    bootstrap must reproduce the naive interval here, or it is not estimating
    clustering -- it is just inflating everything."""
    m, k, p = 400, 5, 0.5
    rng = np.random.default_rng(2024)
    draws = rng.binomial(1, p, size=(m, k))
    counts = draws.sum(axis=1).tolist()
    reps = [k] * m
    p_by_prompt = [c / k for c in counts]

    boot = cluster_bootstrap(p_by_prompt, iters=4000, seed=7)
    naive = wilson_interval(int(sum(counts)), m * k)
    deff = design_effect(boot.ci95, naive)

    assert 0.75 < deff < 1.35, f"design effect {deff} should be ~1 with no clustering"

    vd = variance_decomposition(counts, reps)
    assert vd.between_share is not None
    assert vd.between_share < 0.10, "there is no prompt effect in this data to find"


def test_variance_decomposition_needs_enough_data():
    with pytest.raises(ValueError):
        variance_decomposition([1], [5])
    with pytest.raises(ValueError):
        variance_decomposition([1, 1], [1, 1])  # N == m, no within-prompt df


def test_variance_decomposition_handles_unequal_reps():
    """k0 reduces to k when balanced, so an unbalanced design must not blow up."""
    vd = variance_decomposition([5, 0, 3], [5, 5, 4])
    assert vd.sigma2_between >= 0.0
    assert vd.sigma2_within >= 0.0
    assert vd.between_share is not None
    assert vd.between_share + vd.within_share == pytest.approx(1.0)


def test_variance_decomposition_reports_none_when_degenerate():
    """Every run identical: no variance to apportion, and 0.5/0.5 would be a lie."""
    vd = variance_decomposition([5, 5, 5], [5, 5, 5])
    assert vd.between_share is None
    assert vd.within_share is None


# --------------------------------------------------------------------------
# Test 4: MDE formula, hand-computed
# --------------------------------------------------------------------------


def test_mde_matches_hand_computation():
    se_cluster = 0.05
    # By hand: (z_0.975 + z_0.80) * sqrt(2) * se_cluster
    #        = (1.959963984540054 + 0.8416212335729143) * 1.4142135623730951 * 0.05
    #        = 2.8015852181129683 * 1.4142135623730951 * 0.05
    #        = 0.1981019905799673
    assert mde_abs(se_cluster) == pytest.approx(0.19810199058, rel=1e-10)


def test_mde_constants_are_what_the_spec_says():
    assert Z_975 == pytest.approx(1.96, abs=5e-4)
    assert Z_POWER_80 == pytest.approx(0.8416, abs=5e-5)
    assert MDE_Z_SUM == pytest.approx(2.802, abs=5e-4)


def test_mde_scales_linearly_with_se():
    assert mde_abs(0.10) == pytest.approx(2 * mde_abs(0.05), rel=1e-12)


def test_mde_refuses_unsupported_alpha_or_power():
    """An MDE without its alpha and power is decorative; silently substituting
    the wrong quantile would be worse."""
    with pytest.raises(NotImplementedError):
        mde_abs(0.05, alpha=0.10)
    with pytest.raises(NotImplementedError):
        mde_abs(0.05, power=0.90)


# --------------------------------------------------------------------------
# Fix 2: the naive MDE, so cluster and naive can be compared like with like
# --------------------------------------------------------------------------


def test_naive_mde_matches_hand_computation():
    # p_pooled = 160/200 = 0.8, so se_naive = sqrt(0.8*0.2/200) = sqrt(0.0008)
    #                                       = 0.0282842712474619
    # mde_naive = (z_0.975 + z_0.80) * sqrt(2) * se_naive
    #           = 3.962039811599346 * 0.0282842712474619
    #           = 0.11206340872451874
    se = se_pooled_binomial(160, 200)
    assert se == pytest.approx(0.0282842712474619, rel=1e-12)
    assert mde_abs(se) == pytest.approx(0.11206340872451874, rel=1e-12)


def test_pooled_se_shrinks_with_n_and_peaks_at_one_half():
    assert se_pooled_binomial(50, 100) > se_pooled_binomial(200, 400)
    assert se_pooled_binomial(50, 100) > se_pooled_binomial(10, 100)


def test_pooled_se_rejects_bad_input():
    with pytest.raises(ValueError):
        se_pooled_binomial(1, 0)
    with pytest.raises(ValueError):
        se_pooled_binomial(11, 10)


def test_naive_mde_is_smaller_than_cluster_mde_under_clustering():
    """The whole point: the naive design overstates its own resolving power."""
    m, k = 40, 5
    counts = [k if j % 2 == 0 else 0 for j in range(m)]
    boot = cluster_bootstrap([c / k for c in counts], iters=2000, seed=11)
    naive = mde_abs(se_pooled_binomial(sum(counts), m * k))
    cluster = mde_abs(boot.se)
    assert cluster > naive
    assert cluster / naive > 1.5


# --------------------------------------------------------------------------
# Fix 3: a symmetric absolute MDE on a bounded parameter
# --------------------------------------------------------------------------


def test_mde_interpretable_down_only_near_the_upper_bound():
    """p=0.98 leaves 2pp of headroom up; a 20pp MDE cannot describe an increase."""
    assert mde_interpretability(0.98, 0.20) == "down_only"


def test_mde_interpretable_both_in_the_middle():
    assert mde_interpretability(0.50, 0.10) == "both"


def test_mde_interpretable_up_only_near_the_lower_bound():
    assert mde_interpretability(0.02, 0.20) == "up_only"


def test_mde_interpretable_neither_when_it_exceeds_both_directions():
    assert mde_interpretability(0.50, 0.90) == "neither"


def test_mde_interpretability_rejects_out_of_range_input():
    with pytest.raises(ValueError):
        mde_interpretability(1.5, 0.1)
    with pytest.raises(ValueError):
        mde_interpretability(0.5, -0.1)


@pytest.mark.parametrize(
    "ci,expected",
    [
        ((0.30, 0.70), "none"),
        ((0.30, 1.00), "upper"),
        ((0.00, 0.70), "lower"),
        ((0.00, 1.00), "both"),
        ((0.30, 0.996), "upper"),  # within tolerance of the bound
        ((0.30, 0.99), "none"),  # outside it
    ],
)
def test_boundary_flag(ci, expected):
    assert boundary_flag(ci) == expected


# --------------------------------------------------------------------------
# Fix 4: deff ~ 1 + (k_eff - 1) * ICC
# --------------------------------------------------------------------------


def test_deff_prediction_endpoints():
    assert deff_prediction(5.0, 0.0) == pytest.approx(1.0)  # no clustering
    assert deff_prediction(5.0, 1.0) == pytest.approx(5.0)  # total clustering
    assert deff_prediction(1.0, 0.9) == pytest.approx(1.0)  # one rep per prompt
    assert deff_prediction(3.55, 0.44) == pytest.approx(1.0 + 2.55 * 0.44)


def test_deff_prediction_rejects_bad_input():
    with pytest.raises(ValueError):
        deff_prediction(0.5, 0.3)
    with pytest.raises(ValueError):
        deff_prediction(5.0, 1.3)


def test_observed_deff_matches_prediction_at_known_icc_and_balanced_k():
    """Prompts graded evenly across p_j = 0, 0.2, ... 1.0 at k=5: a moderate,
    realistic ICC where the prediction and the bootstrap must agree.

    An all-or-nothing design (p_j in {0, 1}) is deliberately not used here. It
    pins the ICC at exactly 1, but it also makes mean(p_j) a lattice with spacing
    1/m, and the percentile interval then snaps to grid points -- the residual
    swings by half a unit on bootstrap seed alone, which would make this a test
    of the RNG rather than of the relation.
    """
    m, k = 120, 5
    counts = [j % 6 for j in range(m)]
    reps = [k] * m
    boot = cluster_bootstrap([c / k for c in counts], iters=6000, seed=3)
    deff = design_effect(boot.ci95, wilson_interval(sum(counts), m * k))
    vd = variance_decomposition(counts, reps)

    predicted = deff_prediction(k, vd.between_share)
    assert vd.between_share == pytest.approx(0.336, abs=0.005)
    assert predicted == pytest.approx(2.344, abs=0.02)
    # Max |residual| across ten bootstrap seeds is 0.143; 0.25 leaves headroom
    # without letting a genuinely broken estimator through.
    assert abs(deff - predicted) < 0.25, f"residual {deff - predicted} too large"


def test_observed_deff_matches_prediction_at_zero_icc():
    """The other end: no prompt effect, so prediction and observation are both ~1."""
    m, k = 400, 5
    rng = np.random.default_rng(99)
    counts = rng.binomial(1, 0.5, size=(m, k)).sum(axis=1).tolist()
    reps = [k] * m
    boot = cluster_bootstrap([c / k for c in counts], iters=4000, seed=4)
    deff = design_effect(boot.ci95, wilson_interval(int(sum(counts)), m * k))
    vd = variance_decomposition(counts, reps)

    predicted = deff_prediction(k, vd.between_share)
    assert predicted == pytest.approx(1.0, abs=0.15)
    assert abs(deff - predicted) < 0.3, f"residual {deff - predicted} too large"
