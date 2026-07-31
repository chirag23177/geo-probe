"""Stage 1: send prompts to providers, k times each, and write raw runs.

Reads config, writes data/runs/{batch_id}.jsonl. Never calls stage 2.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .providers import Provider, ProviderError, build_provider, with_retries
from .schemas import (
    ExperimentConfig,
    PromptSpec,
    RunRecord,
    append_jsonl,
    read_runs,
)

RUNS_DIR = Path("data/runs")


def new_batch_id(now: datetime | None = None) -> str:
    """Filesystem-safe ISO-8601: colons become dashes so this works as a filename
    on Windows. The batch_id in the record is this same safe string."""
    ts = now or datetime.now(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H-%M-%SZ")


def runs_path(batch_id: str) -> Path:
    return RUNS_DIR / f"{batch_id}.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_runs(prompts: Iterable[PromptSpec], cfg: ExperimentConfig) -> list[tuple[PromptSpec, str, int]]:
    """Every (prompt, provider, rep) triple this batch should contain."""
    plan = []
    for prompt in prompts:
        for provider_cfg in cfg.providers:
            for rep in range(1, cfg.k + 1):
                plan.append((prompt, provider_cfg.id, rep))
    return plan


def completed_triples(path: Path) -> set[tuple[str, str, int]]:
    """Successful (prompt_id, provider, rep) triples already on disk.

    Failed runs are deliberately *not* counted as complete: re-running the batch
    retries them and appends a fresh record. dedupe_runs() below resolves the
    duplicate at read time, so the file stays append-only.
    """
    return {
        (r.prompt_id, r.provider, r.rep)
        for r in read_runs(path)
        if r.ok
    }


def dedupe_runs(runs: list[RunRecord]) -> list[RunRecord]:
    """Collapse retried triples: a successful record wins over a failed one, and
    among equals the later record wins."""
    best: dict[tuple[str, str, int], RunRecord] = {}
    for r in runs:
        key = (r.prompt_id, r.provider, r.rep)
        prev = best.get(key)
        if prev is None or r.ok or not prev.ok:
            best[key] = r
    return list(best.values())


def run_batch(
    cfg: ExperimentConfig,
    prompts: list[PromptSpec],
    batch_id: str,
    providers: dict[str, Provider] | None = None,
    on_progress: Callable[[str], None] = lambda _: None,
) -> Path:
    """Execute the batch, skipping triples already completed. Resumable."""
    path = runs_path(batch_id)
    done = completed_triples(path)
    providers = providers or {p.id: build_provider(p) for p in cfg.providers}
    temp_by_provider = {p.id: p.temperature for p in cfg.providers}
    model_by_provider = {p.id: p.model for p in cfg.providers}

    plan = plan_runs(prompts, cfg)
    todo = [t for t in plan if (t[0].id, t[1], t[2]) not in done]
    on_progress(f"{len(plan)} runs planned, {len(plan) - len(todo)} already complete, {len(todo)} to go")

    for i, (prompt, provider_id, rep) in enumerate(todo, start=1):
        provider = providers[provider_id]
        temperature = temp_by_provider[provider_id]
        record = _execute_one(prompt, provider, provider_id, rep, batch_id, temperature, model_by_provider[provider_id])
        append_jsonl(path, [record])
        status = "ok" if record.ok else f"ERROR {record.error[:60]}"  # type: ignore[index]
        on_progress(f"[{i}/{len(todo)}] {prompt.id} {provider_id} rep{rep} {status}")

    return path


def _execute_one(
    prompt: PromptSpec,
    provider: Provider,
    provider_id: str,
    rep: int,
    batch_id: str,
    temperature: float | None,
    model_requested: str,
) -> RunRecord:
    try:
        resp = with_retries(lambda: provider.complete(prompt.text, temperature))
    except ProviderError as exc:
        # A run that ultimately fails is written with `error` populated and
        # raw_response null. It is excluded from aggregation with the exclusion
        # count reported. Nothing is silently dropped.
        return RunRecord(
            run_id=str(uuid.uuid4()),
            batch_id=batch_id,
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            intent=prompt.intent,
            provider=provider_id,  # type: ignore[arg-type]
            model_requested=model_requested,
            model_returned=None,
            rep=rep,
            temperature=temperature,
            raw_response=None,
            citation_urls=[],
            latency_ms=0,
            error=f"{type(exc).__name__}: {exc}",
            ts=_utc_now_iso(),
        )

    return RunRecord(
        run_id=str(uuid.uuid4()),
        batch_id=batch_id,
        prompt_id=prompt.id,
        prompt_text=prompt.text,
        intent=prompt.intent,
        provider=provider_id,  # type: ignore[arg-type]
        model_requested=model_requested,
        model_returned=resp.model_returned,
        rep=rep,
        temperature=temperature,
        raw_response=resp.text,
        citation_urls=resp.citation_urls,
        latency_ms=resp.latency_ms,
        error=None,
        ts=_utc_now_iso(),
    )
