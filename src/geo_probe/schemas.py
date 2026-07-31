"""Pydantic models for config and for all three data stages.

Every record is validated on write and on read. The JSONL files are the contract
between stages, so a schema violation should surface at the boundary, not three
stages later in a chart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator

ProviderId = Literal["anthropic", "perplexity"]
Intent = Literal["listicle", "comparison", "validation"]
Sentiment = Literal["positive", "neutral", "negative"]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class Brand(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    domains: list[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    model_config = {"extra": "forbid"}

    id: ProviderId
    model: str
    temperature: float | None = None


class ExtractorConfig(BaseModel):
    model_config = {"extra": "forbid"}

    model: str
    max_parse_retries: int = 3


class ExperimentConfig(BaseModel):
    model_config = {"extra": "forbid"}

    category: str
    brands: list[Brand]
    providers: list[ProviderConfig]
    extractor: ExtractorConfig
    k: int = Field(ge=1)
    bootstrap_iters: int = Field(ge=1)

    @field_validator("brands")
    @classmethod
    def _unique_brand_names(cls, v: list[Brand]) -> list[Brand]:
        names = [b.name for b in v]
        if len(set(names)) != len(names):
            raise ValueError("brand names must be unique")
        return v

    @field_validator("providers")
    @classmethod
    def _unique_provider_ids(cls, v: list[ProviderConfig]) -> list[ProviderConfig]:
        ids = [p.id for p in v]
        if len(set(ids)) != len(ids):
            raise ValueError("provider ids must be unique")
        return v


class PromptSpec(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    text: str
    intent: Intent


class PromptsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    prompts: list[PromptSpec]

    @field_validator("prompts")
    @classmethod
    def _unique_prompt_ids(cls, v: list[PromptSpec]) -> list[PromptSpec]:
        ids = [p.id for p in v]
        if len(set(ids)) != len(ids):
            raise ValueError("prompt ids must be unique")
        return v


def load_experiment(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


def load_prompts(path: str | Path) -> list[PromptSpec]:
    with open(path, "r", encoding="utf-8") as fh:
        return PromptsConfig.model_validate(yaml.safe_load(fh)).prompts


# --------------------------------------------------------------------------
# Stage 1: runs
# --------------------------------------------------------------------------


class RunRecord(BaseModel):
    model_config = {"extra": "forbid"}

    run_id: str
    batch_id: str
    prompt_id: str
    prompt_text: str
    intent: Intent
    provider: ProviderId
    model_requested: str
    model_returned: str | None
    rep: int
    temperature: float | None
    raw_response: str | None
    citation_urls: list[str] = Field(default_factory=list)
    latency_ms: int
    error: str | None = None
    ts: str

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_response is not None


# --------------------------------------------------------------------------
# Stage 2: extracts
# --------------------------------------------------------------------------


class ExtractRecord(BaseModel):
    model_config = {"extra": "forbid"}

    run_id: str
    brand: str
    mentioned: bool
    rank: int | None = None
    sentiment: Sentiment | None = None
    cited: bool = False
    citation_urls: list[str] = Field(default_factory=list)
    evidence_span: str | None = None
    evidence_span_normalized_match: bool = False
    """True when the span matched only after formatting normalization -- i.e. the
    model copied rendered text. Lets the recovered runs be audited separately."""
    extract_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.extract_error is None


# --------------------------------------------------------------------------
# Stage 3: aggregates
# --------------------------------------------------------------------------


class AggRecord(BaseModel):
    """One record per (brand, provider). Never per brand alone -- see aggregate.py."""

    model_config = {"extra": "forbid"}

    brand: str
    provider: ProviderId
    model_returned: str
    n_prompts: int
    k: int
    n_runs_used: int
    n_runs_excluded: int
    mention_rate: float
    ci95_cluster: tuple[float, float]
    ci95_naive_wrong: tuple[float, float]
    design_effect: float
    var_between_prompt_share: float | None
    var_within_prompt_share: float | None
    flip_rate: float
    mean_rank: float | None
    rank_stdev: float | None
    rank_n: int
    mde_abs_pp: float
    mde_alpha: float
    mde_power: float

    # The naive counterpart of the MDE, so the cluster figure can be compared
    # like-for-like instead of against half a CI width.
    se_naive: float
    mde_abs_pp_naive: float
    mde_inflation: float

    # A mention rate is bounded in [0, 1]; a symmetric absolute MDE is not
    # always interpretable in both directions.
    headroom_up: float
    headroom_down: float
    mde_interpretable: Literal["both", "down_only", "up_only", "neither"]
    ci_at_boundary: Literal["none", "upper", "lower", "both"]

    # Consistency check on the clustering story: deff ~ 1 + (k_eff - 1) * ICC.
    k_effective: float
    deff_predicted: float | None
    deff_residual: float | None


# --------------------------------------------------------------------------
# JSONL helpers (append-only)
# --------------------------------------------------------------------------


def append_jsonl(path: str | Path, records: Sequence[BaseModel]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, model: type[BaseModel]) -> Iterator[BaseModel]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - want the line number in the message
                raise ValueError(f"{p}:{lineno} failed {model.__name__} validation: {exc}") from exc


def read_runs(path: str | Path) -> list[RunRecord]:
    return list(read_jsonl(path, RunRecord))  # type: ignore[arg-type]


def read_extracts(path: str | Path) -> list[ExtractRecord]:
    return list(read_jsonl(path, ExtractRecord))  # type: ignore[arg-type]
