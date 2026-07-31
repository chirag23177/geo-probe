"""CLI. Four stages, hard boundaries: each reads a file and writes a file.

No stage calls another in-process. That is deliberate -- it keeps the expensive,
non-deterministic step (probe) separate from the cheap, deterministic ones, and
it means a broken extractor never forces a re-probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import aggregate as agg
from . import diagnose as dg
from . import extract as ex
from . import probe as pr
from . import report as rep
from .providers.base import ProviderError, load_dotenv
from .schemas import load_experiment, load_prompts, read_extracts, read_runs
from .stats import wilson_interval

app = typer.Typer(add_completion=False, help="Measure the MDE of LLM brand recommendations.")

# USD per 1M tokens. Sticker rates as of 2026-07; Perplexity also bills a
# per-request search fee, so its estimate is a floor.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "sonar-pro": (3.00, 15.00),
    "sonar": (1.00, 1.00),
}

# Rough per-call token shapes, used only for the pre-flight estimate.
PROBE_IN, PROBE_OUT = 40, 600
EXTRACT_IN, EXTRACT_OUT = 900, 250

SPEND_GATE_USD = 2.00


def _price(model: str) -> tuple[float, float]:
    if model not in PRICES:
        typer.echo(f"  (no price on file for {model!r}; assuming $3/$15 per Mtok)")
        return (3.00, 15.00)
    return PRICES[model]


def _cost(model: str, calls: int, tok_in: int, tok_out: int) -> float:
    p_in, p_out = _price(model)
    return calls * (tok_in * p_in + tok_out * p_out) / 1_000_000


def _confirm(estimate: float, yes: bool) -> None:
    typer.echo(f"Estimated spend: ${estimate:.2f}")
    if estimate > SPEND_GATE_USD and not yes:
        typer.echo(f"Estimate exceeds ${SPEND_GATE_USD:.2f}. Re-run with --yes to proceed.")
        raise typer.Exit(code=1)


def _log(msg: str) -> None:
    typer.echo(msg)


@app.callback()
def _bootstrap() -> None:
    """Runs before every subcommand, on every entry path.

    Loading .env is a CLI convenience, not library behaviour: importing
    geo_probe from your own code still reads the environment and nothing else.
    """
    loaded = load_dotenv()
    if loaded:
        typer.echo(f"Loaded from .env: {', '.join(sorted(loaded))}")


@app.command()
def probe(
    config: Path = typer.Option(Path("config/experiment.yaml"), "--config"),
    prompts: Path = typer.Option(Path("config/prompts.yaml"), "--prompts"),
    batch: str = typer.Option("", "--batch", help="Resume an existing batch id."),
    yes: bool = typer.Option(False, "--yes", help="Proceed past the spend gate."),
) -> None:
    """Stage 1 -> data/runs/{batch_id}.jsonl"""
    cfg = load_experiment(config)
    prompt_list = load_prompts(prompts)
    batch_id = batch or pr.new_batch_id()

    plan = pr.plan_runs(prompt_list, cfg)
    done = pr.completed_triples(pr.runs_path(batch_id))
    remaining = [t for t in plan if (t[0].id, t[1], t[2]) not in done]

    estimate = 0.0
    for provider_cfg in cfg.providers:
        calls = sum(1 for t in remaining if t[1] == provider_cfg.id)
        estimate += _cost(provider_cfg.model, calls, PROBE_IN, PROBE_OUT)
    typer.echo(f"Batch {batch_id}: {len(remaining)} runs to send ({len(plan) - len(remaining)} already done).")
    _confirm(estimate, yes)

    path = pr.run_batch(cfg, prompt_list, batch_id, on_progress=_log)
    typer.echo(f"Wrote {path}")
    typer.echo(f"batch_id: {batch_id}")


@app.command()
def extract(
    batch: str = typer.Option(..., "--batch"),
    config: Path = typer.Option(Path("config/experiment.yaml"), "--config"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Stage 2 -> data/extracts/{batch_id}.jsonl"""
    cfg = load_experiment(config)
    runs = pr.dedupe_runs(read_runs(pr.runs_path(batch)))
    already = {e.run_id for e in read_extracts(ex.extracts_path(batch))}
    todo = [r for r in runs if r.ok and r.run_id not in already]

    typer.echo(f"Batch {batch}: {len(todo)} runs to grade.")
    _confirm(_cost(cfg.extractor.model, len(todo), EXTRACT_IN, EXTRACT_OUT), yes)

    path = ex.extract_batch(cfg, runs, batch, on_progress=_log)
    typer.echo(f"Wrote {path}")


@app.command("sample-gold")
def sample_gold(
    batch: str = typer.Option(..., "--batch"),
    n: int = typer.Option(30, "-n", "--n"),
    seed: int = typer.Option(0, "--seed"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing template."),
    from_recovered: bool = typer.Option(
        False,
        "--from-recovered",
        help="Draw only from pairs whose span matched after normalization.",
    ),
) -> None:
    """Stage 2b -> a gold CSV to fill in by hand (mentioned_human column)."""
    typer.echo("Estimated spend: $0.00 (no API calls)")
    path = ex.sample_gold(batch, n=n, seed=seed, force=force, from_recovered=from_recovered)
    typer.echo(f"Wrote {path}. Fill the mentioned_human column with true/false, then run score-gold.")


def _report_gold(title: str, path: Path) -> None:
    score = ex.score_gold(path)
    n_err = len(score.disagreements)
    typer.echo(f"\n=== {title} ({path}) ===")
    typer.echo(f"labelled pairs: {score.n}  (human: {score.n_human_true} true, "
               f"{score.n_human_false} false)")
    typer.echo(f"raw agreement on `mentioned`: {score.raw_agreement:.3f}")
    if score.kappa is None:
        typer.echo(
            "Cohen's kappa on `mentioned`: undefined -- every human label is the same "
            "class, so chance agreement is 1.0 and there is no agreement-above-chance "
            "to measure. Raw agreement is the only number this sample supports."
        )
    else:
        typer.echo(f"Cohen's kappa on `mentioned`: {score.kappa:.3f}")
    # Perfect agreement on a small sample is a loose bound, not proof.
    upper = wilson_interval(n_err, score.n)[1]
    typer.echo(f"95% upper bound on the extractor's error rate: {upper * 100:.1f}% "
               f"({n_err} error(s) in {score.n})")
    if not score.disagreements:
        typer.echo("no disagreements")
        return
    typer.echo(f"{len(score.disagreements)} disagreement(s):")
    for run_id, brand, machine, human, excerpt in score.disagreements:
        typer.echo(f"\n- {run_id} / {brand}: extractor={machine} human={human}")
        typer.echo(f"  {excerpt!r}")


@app.command("score-gold")
def score_gold(
    batch: str = typer.Option(..., "--batch"),
    path: Path = typer.Option(None, "--path", help="Score one specific file instead."),
) -> None:
    """Stage 2b -> agreement between the extractor and a human on `mentioned`.

    The original and normalization-recovered gold sets are reported separately
    and never merged: the recovered runs are the ones most likely to be
    mis-extracted, so averaging them into the original would hide the difference.
    """
    typer.echo("Estimated spend: $0.00 (no API calls)")
    typer.echo(f"batch: {batch}")
    if path is not None:
        _report_gold("gold set", path)
        return

    found = False
    for title, candidate in (
        ("original sample", ex.GOLD_TEMPLATE),
        ("normalization-recovered sample", ex.GOLD_RECOVERED),
    ):
        if candidate.exists():
            _report_gold(title, candidate)
            found = True
    if not found:
        typer.echo(f"No gold file found at {ex.GOLD_TEMPLATE} or {ex.GOLD_RECOVERED}.")
        raise typer.Exit(code=1)


@app.command("diagnose-failures")
def diagnose_failures(
    batch: str = typer.Option(..., "--batch"),
    config: Path = typer.Option(Path("config/experiment.yaml"), "--config"),
    iters: int = typer.Option(dg.PERMUTATION_ITERS, "--iters", help="Permutation count."),
    seed: int = typer.Option(0, "--seed"),
    extracts_file: Path = typer.Option(
        None, "--extracts", help="Extracts file to diagnose. Defaults to the batch's own."
    ),
    out_name: str = typer.Option("failure_diagnosis.md", "--out-name"),
) -> None:
    """Characterise the runs whose extraction failed -> reports/{batch_id}/{out_name}

    `--extracts` lets an archived extracts file be diagnosed, so a pre-fix and a
    post-fix diagnosis can both be regenerated from scratch instead of existing
    only as whatever happened to be on disk at the time.
    """
    typer.echo("Estimated spend: $0.00 (no API calls)")
    cfg = load_experiment(config)
    runs = pr.dedupe_runs(read_runs(pr.runs_path(batch)))
    source = extracts_file or ex.extracts_path(batch)
    extracts = read_extracts(source)
    if not extracts:
        typer.echo(f"No extracts found at {source}")
        raise typer.Exit(code=1)
    typer.echo(f"Diagnosing {source}")

    diagnoses = dg.diagnose(runs, extracts, cfg.brands, iters=iters, seed=seed)
    out = rep.report_dir(batch) / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dg.render_diagnosis(diagnoses, batch, iters), encoding="utf-8")

    for d in diagnoses:
        typer.echo(f"\n{d.provider}: {d.n_failed} failed / {d.n_failed + d.n_passed} usable runs")
        for c in d.comparisons:
            typer.echo(
                f"  {dg.FEATURE_LABELS[c.feature]:34} "
                f"failed {c.mean_failed:9.1f}  passed {c.mean_passed:9.1f}  p={c.p_value:.4f}"
            )
    typer.echo(f"\nWrote {out}")


@app.command()
def aggregate(
    batch: str = typer.Option(..., "--batch"),
    config: Path = typer.Option(Path("config/experiment.yaml"), "--config"),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed."),
) -> None:
    """Stage 3 -> data/agg/{batch_id}.json and .csv"""
    typer.echo("Estimated spend: $0.00 (no API calls)")
    cfg = load_experiment(config)
    runs = pr.dedupe_runs(read_runs(pr.runs_path(batch)))
    extracts = read_extracts(ex.extracts_path(batch))
    if not runs:
        typer.echo(f"No runs found at {pr.runs_path(batch)}")
        raise typer.Exit(code=1)

    records = agg.aggregate_batch(cfg, runs, extracts, seed=seed)
    jpath, cpath = agg.write_outputs(batch, records)
    typer.echo(f"Wrote {jpath} and {cpath} ({len(records)} brand x provider cells)")


@app.command()
def report(
    batch: str = typer.Option(..., "--batch"),
    config: Path = typer.Option(Path("config/experiment.yaml"), "--config"),
) -> None:
    """Stage 4 -> reports/{batch_id}/chart.png and findings.md"""
    typer.echo("Estimated spend: $0.00 (no API calls)")
    cfg = load_experiment(config)
    records = agg.read_agg(batch)
    chart, findings = rep.build_report(records, cfg.category, batch)
    typer.echo(f"Wrote {chart} and {findings}")


def main() -> None:
    """Entry point. Turns the handful of expected operational failures into a
    one-line message instead of a traceback."""
    try:
        app()
    except (ProviderError, FileNotFoundError, FileExistsError) as exc:
        typer.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
