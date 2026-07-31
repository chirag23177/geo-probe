"""Stage 3: one record per (brand, provider).

Never pool across providers. A plain Anthropic call measures parametric memory;
a Perplexity Sonar call measures retrieval-grounded output. They are different
measurement surfaces, and averaging them is a category error -- so the code
raises rather than letting it happen quietly.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schemas import AggRecord, ExperimentConfig, ExtractRecord, RunRecord
from .stats import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
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

AGG_DIR = Path("data/agg")

CSV_FIELDS = [
    "brand",
    "provider",
    "model_returned",
    "n_prompts",
    "k",
    "n_runs_used",
    "n_runs_excluded",
    "mention_rate",
    "ci95_cluster_lo",
    "ci95_cluster_hi",
    "ci95_naive_wrong_lo",
    "ci95_naive_wrong_hi",
    "design_effect",
    "var_between_prompt_share",
    "var_within_prompt_share",
    "flip_rate",
    "mean_rank",
    "rank_stdev",
    "rank_n",
    "mde_abs_pp",
    "mde_alpha",
    "mde_power",
    "se_naive",
    "mde_abs_pp_naive",
    "mde_inflation",
    "headroom_up",
    "headroom_down",
    "mde_interpretable",
    "ci_at_boundary",
    "k_effective",
    "deff_predicted",
    "deff_residual",
]


def agg_json_path(batch_id: str) -> Path:
    return AGG_DIR / f"{batch_id}.json"


def agg_csv_path(batch_id: str) -> Path:
    return AGG_DIR / f"{batch_id}.csv"


class ProviderPoolingError(ValueError):
    """Raised on any attempt to aggregate across measurement surfaces."""


@dataclass(frozen=True)
class Observation:
    """One usable (run, brand) pair, carrying the fields aggregation needs."""

    prompt_id: str
    provider: str
    mentioned: bool
    rank: int | None
    model_returned: str


def _require_single_provider(observations: Sequence[Observation], provider: str | None) -> str:
    seen = {o.provider for o in observations}
    if provider is None:
        raise ProviderPoolingError(
            "aggregation requires an explicit provider; there is no 'all providers' row "
            "because a parametric-memory surface and a retrieval-grounded surface are "
            "not the same measurement"
        )
    if seen - {provider}:
        raise ProviderPoolingError(
            f"refusing to pool across providers: expected only {provider!r}, got {sorted(seen)}"
        )
    return provider


def compute_cell(
    brand: str,
    provider: str | None,
    observations: Sequence[Observation],
    k: int,
    n_runs_excluded: int,
    bootstrap_iters: int = 2000,
    seed: int = 0,
) -> AggRecord:
    """Aggregate one (brand, provider) cell. Raises if the observations span
    more than one provider, or if no provider was named."""
    if not observations:
        raise ValueError(f"no usable observations for brand={brand!r} provider={provider!r}")
    provider = _require_single_provider(observations, provider)

    by_prompt: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_prompt[o.prompt_id].append(o)

    prompt_ids = sorted(by_prompt)
    counts = [sum(1 for o in by_prompt[pid] if o.mentioned) for pid in prompt_ids]
    reps = [len(by_prompt[pid]) for pid in prompt_ids]
    p_by_prompt = [c / n for c, n in zip(counts, reps)]

    n_runs_used = sum(reps)
    total_mentions = sum(counts)

    # Unit of analysis is the prompt.
    boot = cluster_bootstrap(p_by_prompt, iters=bootstrap_iters, seed=seed)
    # ...and the interval a naive dashboard would report, for contrast.
    naive = wilson_interval(total_mentions, n_runs_used)
    deff = design_effect(boot.ci95, naive)

    if len(prompt_ids) >= 2 and n_runs_used > len(prompt_ids):
        vd = variance_decomposition(counts, reps)
        between_share, within_share = vd.between_share, vd.within_share
    else:
        between_share = within_share = None

    # flip_rate is prompt-level: a prompt where the brand appears in some but not
    # all reps is unstable. 0 < count < reps.
    flips = sum(1 for c, n in zip(counts, reps) if 0 < c < n)
    flip_rate = flips / len(prompt_ids)

    # Rank statistics are conditional on mentioned == true. Reporting an
    # unconditional mean_rank would be survivorship bias.
    ranks = [o.rank for o in observations if o.mentioned and o.rank is not None]
    mean_rank = statistics.fmean(ranks) if ranks else None
    rank_stdev = statistics.stdev(ranks) if len(ranks) >= 2 else None

    models = sorted({o.model_returned for o in observations})

    # The naive MDE, computed the way a dashboard would: pooled run rate, pooled
    # binomial SE. Comparing this against the cluster MDE compares like with like.
    se_naive = se_pooled_binomial(total_mentions, n_runs_used)
    mde_naive = mde_abs(se_naive)
    mde_cluster = mde_abs(boot.se)

    # Realised cluster size after exclusions -- not the nominal k.
    k_effective = n_runs_used / len(prompt_ids)
    if between_share is None:
        deff_predicted = deff_residual = None
    else:
        deff_predicted = deff_prediction(k_effective, between_share)
        deff_residual = deff - deff_predicted

    return AggRecord(
        brand=brand,
        provider=provider,  # type: ignore[arg-type]
        model_returned=",".join(models),
        n_prompts=len(prompt_ids),
        k=k,
        n_runs_used=n_runs_used,
        n_runs_excluded=n_runs_excluded,
        mention_rate=boot.point,
        ci95_cluster=boot.ci95,
        ci95_naive_wrong=naive,
        design_effect=deff,
        var_between_prompt_share=between_share,
        var_within_prompt_share=within_share,
        flip_rate=flip_rate,
        mean_rank=mean_rank,
        rank_stdev=rank_stdev,
        rank_n=len(ranks),
        mde_abs_pp=mde_cluster,
        mde_alpha=DEFAULT_ALPHA,
        mde_power=DEFAULT_POWER,
        se_naive=se_naive,
        mde_abs_pp_naive=mde_naive,
        mde_inflation=(mde_cluster / mde_naive) if mde_naive > 0 else float("inf"),
        headroom_up=1.0 - boot.point,
        headroom_down=boot.point,
        mde_interpretable=mde_interpretability(boot.point, mde_cluster),  # type: ignore[arg-type]
        ci_at_boundary=boundary_flag(boot.ci95),  # type: ignore[arg-type]
        k_effective=k_effective,
        deff_predicted=deff_predicted,
        deff_residual=deff_residual,
    )


def aggregate_batch(
    cfg: ExperimentConfig,
    runs: Sequence[RunRecord],
    extracts: Sequence[ExtractRecord],
    seed: int = 0,
) -> list[AggRecord]:
    """Group strictly by (brand, provider). There is no all-providers row."""
    extracts_by_run: dict[str, list[ExtractRecord]] = defaultdict(list)
    for e in extracts:
        extracts_by_run[e.run_id].append(e)

    brand_names = [b.name for b in cfg.brands]
    provider_ids = [p.id for p in cfg.providers]

    observations: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    excluded: dict[tuple[str, str], int] = defaultdict(int)

    for run in runs:
        for brand in brand_names:
            key = (brand, run.provider)
            if not run.ok:
                excluded[key] += 1
                continue
            match = next((e for e in extracts_by_run.get(run.run_id, []) if e.brand == brand), None)
            if match is None or not match.ok:
                excluded[key] += 1
                continue
            observations[key].append(
                Observation(
                    prompt_id=run.prompt_id,
                    provider=run.provider,
                    mentioned=match.mentioned,
                    rank=match.rank,
                    model_returned=run.model_returned or run.model_requested,
                )
            )

    records = []
    for provider in provider_ids:
        for brand in brand_names:
            key = (brand, provider)
            obs = observations.get(key, [])
            if not obs:
                continue
            records.append(
                compute_cell(
                    brand=brand,
                    provider=provider,
                    observations=obs,
                    k=cfg.k,
                    n_runs_excluded=excluded.get(key, 0),
                    bootstrap_iters=cfg.bootstrap_iters,
                    seed=seed,
                )
            )
    return records


def write_outputs(batch_id: str, records: Sequence[AggRecord]) -> tuple[Path, Path]:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    jpath, cpath = agg_json_path(batch_id), agg_csv_path(batch_id)

    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump([r.model_dump(mode="json") for r in records], fh, indent=2)
        fh.write("\n")

    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = r.model_dump(mode="json")
            row["ci95_cluster_lo"], row["ci95_cluster_hi"] = row.pop("ci95_cluster")
            row["ci95_naive_wrong_lo"], row["ci95_naive_wrong_hi"] = row.pop("ci95_naive_wrong")
            w.writerow(row)

    return jpath, cpath


def read_agg(batch_id: str) -> list[AggRecord]:
    with open(agg_json_path(batch_id), "r", encoding="utf-8") as fh:
        return [AggRecord.model_validate(d) for d in json.load(fh)]
