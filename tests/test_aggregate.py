"""Test 8: aggregation refuses to pool across providers, plus the plumbing
around it.

A plain Anthropic call measures parametric memory; a Perplexity Sonar call
measures retrieval-grounded output. An "all providers" row would be an average
of two different instruments, so the code raises instead of computing it.
"""

from __future__ import annotations

import pytest

from geo_probe.aggregate import (
    Observation,
    ProviderPoolingError,
    aggregate_batch,
    compute_cell,
)
from geo_probe.schemas import Brand, ExperimentConfig, ExtractRecord, RunRecord


def _obs(prompt_id: str, provider: str, mentioned: bool, rank: int | None = None) -> Observation:
    return Observation(
        prompt_id=prompt_id,
        provider=provider,
        mentioned=mentioned,
        rank=rank,
        model_returned="model-x",
    )


# --------------------------------------------------------------------------
# Test 8: the guard
# --------------------------------------------------------------------------


def test_compute_cell_raises_when_observations_span_providers():
    mixed = [
        _obs("p01", "anthropic", True, 1),
        _obs("p01", "perplexity", True, 1),
        _obs("p02", "anthropic", False),
        _obs("p02", "perplexity", False),
    ]
    with pytest.raises(ProviderPoolingError, match="refusing to pool across providers"):
        compute_cell("Asana", "anthropic", mixed, k=2, n_runs_excluded=0, bootstrap_iters=50)


def test_compute_cell_raises_when_no_provider_is_named():
    single = [_obs("p01", "anthropic", True, 1), _obs("p02", "anthropic", False)]
    with pytest.raises(ProviderPoolingError, match="no 'all providers' row"):
        compute_cell("Asana", None, single, k=1, n_runs_excluded=0, bootstrap_iters=50)


def test_compute_cell_accepts_a_single_provider():
    single = [_obs("p01", "anthropic", True, 1), _obs("p02", "anthropic", False)]
    rec = compute_cell("Asana", "anthropic", single, k=1, n_runs_excluded=0, bootstrap_iters=200)
    assert rec.provider == "anthropic"
    assert rec.mention_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The cell arithmetic
# --------------------------------------------------------------------------


def _cell_observations() -> list[Observation]:
    """4 prompts x k=5. Prompt 1 always mentions, prompt 2 flips 3/5, prompt 3
    flips 1/5, prompt 4 never mentions. mean(p_j) = (1.0+0.6+0.2+0.0)/4 = 0.45."""
    layout = {"p01": 5, "p02": 3, "p03": 1, "p04": 0}
    obs = []
    for pid, hits in layout.items():
        for rep in range(5):
            obs.append(_obs(pid, "anthropic", rep < hits, rank=2 if rep < hits else None))
    return obs


def test_mention_rate_is_the_mean_over_prompts_not_over_runs():
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=500, seed=3)
    assert rec.mention_rate == pytest.approx(0.45)
    assert rec.n_prompts == 4
    assert rec.n_runs_used == 20


def test_flip_rate_is_prompt_level():
    """p02 (3/5) and p03 (1/5) flip; p01 (5/5) and p04 (0/5) do not."""
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=200)
    assert rec.flip_rate == pytest.approx(0.5)


def test_rank_stats_are_conditional_on_mentioned():
    obs = [
        _obs("p01", "anthropic", True, 1),
        _obs("p01", "anthropic", True, 3),
        _obs("p02", "anthropic", False, None),
        _obs("p02", "anthropic", False, None),
    ]
    rec = compute_cell("Asana", "anthropic", obs, k=2, n_runs_excluded=0, bootstrap_iters=200)
    assert rec.rank_n == 2  # not 4
    assert rec.mean_rank == pytest.approx(2.0)
    assert rec.rank_stdev == pytest.approx(1.4142135, rel=1e-5)


def test_naive_mde_is_reported_alongside_the_cluster_mde():
    """Fix 2: comparing a cluster MDE against half a naive CI width compares two
    different quantities. Both MDEs must be present so the comparison is honest."""
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=2000, seed=5)
    assert rec.mde_abs_pp_naive > 0
    assert rec.mde_abs_pp > rec.mde_abs_pp_naive  # this cell is clustered
    assert rec.mde_inflation == pytest.approx(rec.mde_abs_pp / rec.mde_abs_pp_naive)
    # se_naive is the pooled-run binomial SE: 9 of 20 runs mentioned.
    assert rec.se_naive == pytest.approx(((9 / 20) * (11 / 20) / 20) ** 0.5, rel=1e-12)


def test_boundary_fields_flag_a_cell_with_no_headroom():
    """Fix 3: a cell near 1.00 has almost no headroom up, so an MDE larger than
    that headroom cannot describe an increase."""
    obs = []
    for j in range(20):
        hits = 5 if j > 0 else 1  # one dissenting prompt keeps the variance non-zero
        for rep in range(5):
            obs.append(_obs(f"p{j:02d}", "anthropic", rep < hits, rank=1 if rep < hits else None))
    rec = compute_cell("Asana", "anthropic", obs, k=5, n_runs_excluded=0,
                       bootstrap_iters=2000, seed=5)
    assert rec.mention_rate == pytest.approx(0.96)
    assert rec.headroom_up == pytest.approx(0.04)
    assert rec.mde_abs_pp > rec.headroom_up
    assert rec.mde_interpretable == "down_only"
    assert rec.ci_at_boundary in ("upper", "both")


def test_degenerate_cell_with_zero_variance_is_flagged_by_the_boundary_field():
    """Every run mentions the brand: the bootstrap has no variance, so the MDE
    collapses to 0 and `mde_interpretable` is vacuously "both". That is
    arithmetically right and substantively useless, so `ci_at_boundary` is what
    marks the cell as untrustworthy."""
    obs = [_obs(f"p{j:02d}", "anthropic", True, 1) for j in range(6) for _ in range(5)]
    rec = compute_cell("Asana", "anthropic", obs, k=5, n_runs_excluded=0, bootstrap_iters=500)
    assert rec.mention_rate == pytest.approx(1.0)
    assert rec.mde_abs_pp == pytest.approx(0.0)
    assert rec.ci_at_boundary == "upper"


def test_boundary_fields_are_clean_for_a_mid_range_cell():
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=2000, seed=5)
    assert rec.headroom_up == pytest.approx(1.0 - rec.mention_rate)
    assert rec.headroom_down == pytest.approx(rec.mention_rate)
    assert rec.ci_at_boundary == "none"


def test_k_effective_reflects_exclusions_not_the_nominal_k():
    """Fix 4: dropping runs lowers the realised cluster size, which mechanically
    depresses the design effect. k_effective is what the prediction must use."""
    obs = _cell_observations()[:18]  # drop 2 of 20 runs
    rec = compute_cell("Asana", "anthropic", obs, k=5, n_runs_excluded=2,
                       bootstrap_iters=500, seed=5)
    assert rec.k == 5  # nominal, unchanged
    assert rec.k_effective == pytest.approx(rec.n_runs_used / rec.n_prompts)
    assert rec.k_effective < 5.0


def test_deff_prediction_and_residual_are_reported():
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=2000, seed=5)
    assert rec.deff_predicted is not None and rec.deff_residual is not None
    assert rec.deff_predicted == pytest.approx(
        1.0 + (rec.k_effective - 1.0) * rec.var_between_prompt_share
    )
    assert rec.deff_residual == pytest.approx(rec.design_effect - rec.deff_predicted)


def test_deff_prediction_is_none_when_variance_is_undecomposable():
    """Every run identical: no ICC to predict from, so no prediction is invented."""
    obs = [_obs(f"p{j:02d}", "anthropic", True, 1) for j in range(4) for _ in range(5)]
    rec = compute_cell("Asana", "anthropic", obs, k=5, n_runs_excluded=0, bootstrap_iters=200)
    assert rec.var_between_prompt_share is None
    assert rec.deff_predicted is None
    assert rec.deff_residual is None


def test_mde_carries_its_alpha_and_power():
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=500)
    assert rec.mde_alpha == 0.05
    assert rec.mde_power == 0.80
    assert rec.mde_abs_pp > 0


def test_design_effect_is_reported_and_naive_interval_is_narrower():
    rec = compute_cell("Asana", "anthropic", _cell_observations(), k=5, n_runs_excluded=0,
                       bootstrap_iters=2000, seed=5)
    cluster_width = rec.ci95_cluster[1] - rec.ci95_cluster[0]
    naive_width = rec.ci95_naive_wrong[1] - rec.ci95_naive_wrong[0]
    assert cluster_width > naive_width
    assert rec.design_effect > 1.0


# --------------------------------------------------------------------------
# End-to-end grouping: one record per (brand, provider), never per brand
# --------------------------------------------------------------------------


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "category": "test category",
            "brands": [
                {"name": "Asana", "domains": ["asana.com"]},
                {"name": "Trello", "domains": ["trello.com"]},
            ],
            "providers": [
                {"id": "anthropic", "model": "claude-sonnet-5", "temperature": 1.0},
                {"id": "perplexity", "model": "sonar-pro"},
            ],
            "extractor": {"model": "claude-haiku-4-5"},
            "k": 2,
            "bootstrap_iters": 200,
        }
    )


def _run(run_id: str, prompt_id: str, provider: str, rep: int, failed: bool = False) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        batch_id="b",
        prompt_id=prompt_id,
        prompt_text="best pm tool",
        intent="listicle",
        provider=provider,  # type: ignore[arg-type]
        model_requested="m",
        model_returned=None if failed else "m-dated",
        rep=rep,
        temperature=1.0,
        raw_response=None if failed else "Asana is good.",
        citation_urls=[],
        latency_ms=1,
        error="boom" if failed else None,
        ts="2026-07-28T10:00:00Z",
    )


def test_aggregate_produces_one_record_per_brand_provider_and_no_pooled_row():
    cfg = _config()
    runs, extracts = [], []
    for provider in ("anthropic", "perplexity"):
        for pid in ("p01", "p02"):
            for rep in (1, 2):
                rid = f"{provider}-{pid}-{rep}"
                runs.append(_run(rid, pid, provider, rep))
                extracts.append(ExtractRecord(run_id=rid, brand="Asana", mentioned=True, rank=1))
                extracts.append(ExtractRecord(run_id=rid, brand="Trello", mentioned=False))

    records = aggregate_batch(cfg, runs, extracts)
    keys = {(r.brand, r.provider) for r in records}
    assert keys == {
        ("Asana", "anthropic"),
        ("Asana", "perplexity"),
        ("Trello", "anthropic"),
        ("Trello", "perplexity"),
    }
    assert len(records) == 4
    assert all(r.provider in ("anthropic", "perplexity") for r in records)


def test_failed_runs_and_extracts_are_excluded_and_counted_not_dropped():
    cfg = _config()
    runs, extracts = [], []
    for pid in ("p01", "p02"):
        for rep in (1, 2):
            rid = f"a-{pid}-{rep}"
            failed = rid == "a-p01-1"
            runs.append(_run(rid, pid, "anthropic", rep, failed=failed))
            if failed:
                continue
            bad_extract = rid == "a-p02-1"
            extracts.append(
                ExtractRecord(
                    run_id=rid,
                    brand="Asana",
                    mentioned=True,
                    rank=1,
                    extract_error="parse failure" if bad_extract else None,
                )
            )
            extracts.append(ExtractRecord(run_id=rid, brand="Trello", mentioned=False))

    records = {(r.brand, r.provider): r for r in aggregate_batch(cfg, runs, extracts)}
    asana = records[("Asana", "anthropic")]
    assert asana.n_runs_used == 2  # 4 planned, 1 run failed, 1 extract failed
    assert asana.n_runs_excluded == 2
    # The Trello cell only lost the failed run; its extract was fine.
    assert records[("Trello", "anthropic")].n_runs_excluded == 1
