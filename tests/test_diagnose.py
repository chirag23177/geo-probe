"""Fix 6: the failure diagnosis, including the hand-rolled permutation test."""

from __future__ import annotations

import pytest

from geo_probe.diagnose import (
    diagnose,
    failed_run_ids,
    permutation_test_diff_means,
    render_diagnosis,
    response_features,
)
from geo_probe.schemas import Brand, ExtractRecord, RunRecord

BRANDS = [Brand(name="Asana", domains=["asana.com"]), Brand(name="Trello", domains=["trello.com"])]


def _run(run_id: str, provider: str, text: str, prompt_id: str = "p01") -> RunRecord:
    return RunRecord(
        run_id=run_id, batch_id="b", prompt_id=prompt_id, prompt_text="q", intent="listicle",
        provider=provider, model_requested="m", model_returned="m", rep=1, temperature=1.0,
        raw_response=text, citation_urls=[], latency_ms=1, error=None, ts="2026-07-28T10:00:00Z",
    )


# --------------------------------------------------------------------------
# Permutation test
# --------------------------------------------------------------------------


def test_permutation_test_finds_a_large_separation():
    a = [100.0] * 20
    b = [1.0] * 20
    assert permutation_test_diff_means(a, b, iters=2000, seed=0) < 0.01


def test_permutation_test_finds_nothing_when_groups_are_identical():
    a = [5.0, 6.0, 7.0, 8.0] * 5
    b = [5.0, 6.0, 7.0, 8.0] * 5
    p = permutation_test_diff_means(a, b, iters=2000, seed=0)
    assert p > 0.5, "identical groups must not look separated"


def test_permutation_p_value_is_never_exactly_zero():
    """(count + 1) / (iters + 1): a finite permutation count cannot prove p = 0."""
    p = permutation_test_diff_means([100.0] * 10, [0.0] * 10, iters=200, seed=0)
    assert p > 0.0
    assert p == pytest.approx(1 / 201)


def test_permutation_test_is_bounded_and_reproducible():
    a, b = [1.0, 2.0, 9.0, 4.0], [3.0, 3.5, 2.0, 8.0]
    p1 = permutation_test_diff_means(a, b, iters=1000, seed=7)
    p2 = permutation_test_diff_means(a, b, iters=1000, seed=7)
    assert p1 == p2
    assert 0.0 < p1 <= 1.0


def test_permutation_test_rejects_an_empty_group():
    with pytest.raises(ValueError):
        permutation_test_diff_means([], [1.0, 2.0])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_response_features_count_what_they_claim():
    run = _run("r1", "perplexity", "**Asana**[1] is good.\nTrello[2][3] too.\n")
    f = response_features(run, BRANDS)
    assert f["bold_markers"] == 2  # opening and closing
    assert f["citation_markers"] == 3
    assert f["newlines"] == 2
    assert f["brands_present"] == 2
    assert f["length_chars"] == float(len(run.raw_response))


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def _failed(run_id: str) -> list[ExtractRecord]:
    return [
        ExtractRecord(run_id=run_id, brand=b.name, mentioned=False, extract_error="boom")
        for b in BRANDS
    ]


def _passed(run_id: str) -> list[ExtractRecord]:
    return [ExtractRecord(run_id=run_id, brand=b.name, mentioned=True, rank=1) for b in BRANDS]


def test_failed_run_ids_picks_up_any_errored_record():
    extracts = _failed("r1") + _passed("r2")
    assert failed_run_ids(extracts) == {"r1"}


def test_diagnosis_is_computed_within_provider_never_pooled():
    """Failure rates differ by provider; pooling would confound 'this feature
    predicts failure' with 'this provider fails more'."""
    runs, extracts = [], []
    for i in range(8):
        runs.append(_run(f"a{i}", "anthropic", "short answer", prompt_id=f"p{i:02d}"))
        extracts += _passed(f"a{i}")
    for i in range(8):
        failed = i < 4
        text = ("**bold** " * 40) if failed else "plain answer"
        runs.append(_run(f"x{i}", "perplexity", text, prompt_id=f"p{i:02d}"))
        extracts += _failed(f"x{i}") if failed else _passed(f"x{i}")

    diagnoses = {d.provider: d for d in diagnose(runs, extracts, BRANDS, iters=500)}
    assert set(diagnoses) == {"anthropic", "perplexity"}
    assert diagnoses["anthropic"].n_failed == 0
    assert diagnoses["anthropic"].comparisons == []  # nothing to compare
    assert diagnoses["perplexity"].n_failed == 4
    assert diagnoses["perplexity"].n_passed == 4

    bold = next(c for c in diagnoses["perplexity"].comparisons if c.feature == "bold_markers")
    assert bold.mean_failed > bold.mean_passed
    assert bold.p_value < 0.10


def test_prompt_failure_counts_are_reported():
    runs, extracts = [], []
    for rep in range(3):
        runs.append(_run(f"f{rep}", "perplexity", "x", prompt_id="p01"))
        extracts += _failed(f"f{rep}")
        runs.append(_run(f"g{rep}", "perplexity", "x", prompt_id="p02"))
        extracts += _passed(f"g{rep}")

    d = diagnose(runs, extracts, BRANDS, iters=200)[0]
    assert d.prompt_failures[0] == ("p01", 3, 3)
    assert ("p02", 0, 3) in d.prompt_failures


def test_rendered_report_states_no_significance_verdict():
    runs = [_run("a0", "anthropic", "x")]
    extracts = _passed("a0")
    text = render_diagnosis(diagnose(runs, extracts, BRANDS, iters=200), "B1", 200)
    assert "No significance threshold is applied" in text
    # A p-value without a verdict, per the brief.
    for banned in ("significant", "not significant", "p < 0.05"):
        assert banned.lower() not in text.lower().replace("no significance threshold", "")
